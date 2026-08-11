/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <chrono>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

// Materializing a wide concat retains all input allocations until cuDF has
// allocated and populated the replacement table. Above this bound the small
// reduction in vector count is not worth doubling hundreds of MiB of live
// device data. Keep this above the 256 MiB UCX concat target so that normal
// exchange coalescing remains enabled.
constexpr uint64_t kMaxMaterializedConcatEstimatedBytes = 384ULL << 20;

uint64_t concatThresholdFromEnv(const char* name, uint64_t fallback) {
  if (const char* value = std::getenv(name)) {
    try {
      return std::stoull(value);
    } catch (...) {
    }
  }
  return fallback;
}

uint64_t concatThreshold(
    exec::DriverCtx* driverCtx,
    const char* configKey,
    const char* envName,
    uint64_t fallback) {
  // Prefer the per-query value.  The process-wide CudfConfig is initialized
  // before registration, but a long-lived executor can subsequently execute
  // queries with different batch targets.
  const auto configured =
      driverCtx->queryConfig().get<uint64_t>(configKey, fallback);
  return concatThresholdFromEnv(envName, configured);
}

RowTypePtr getConcatOutputType(
    const std::shared_ptr<const core::PlanNode>& planNode) {
  const auto numSources = planNode->sources().size();
  if (planNode->is<core::AbstractJoinNode>()) {
    VELOX_CHECK_EQ(
        numSources,
        2,
        "CudfBatchConcat expects a join plan node to have exactly 2 sources");
  } else {
    VELOX_CHECK_EQ(
        numSources, 1, "CudfBatchConcat expects a single-source plan node");
  }
  return planNode->sources()[0]->outputType();
}

size_t checkedConcatTargetRows(int32_t targetRows) {
  VELOX_CHECK_GT(
      targetRows, 0, "CudfBatchConcat target row count must be positive");
  return static_cast<size_t>(targetRows);
}

int32_t queryConcatTargetRows(exec::DriverCtx* driverCtx) {
  const auto targetRows = concatThreshold(
      driverCtx,
      CudfConfig::kCudfBatchSizeMinThreshold,
      "GLUTEN_CUDF_BATCH_SIZE_MIN_THRESHOLD_ROWS",
      CudfConfig::getInstance().batchSizeMinThreshold);
  VELOX_CHECK_LE(
      targetRows,
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
      "CudfBatchConcat target row count exceeds INT32_MAX");
  return static_cast<int32_t>(targetRows);
}

std::string getAggregationStep(
    const std::shared_ptr<const core::PlanNode>& planNode) {
  const auto aggregation =
      std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
  return aggregation ? fmt::format("{}", aggregation->step()) : "UNKNOWN";
}

bool logConcatConfig() {
  const auto* value = std::getenv("GLUTEN_CUDF_BATCH_CONCAT_LOG_CONFIG");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool hasVariableWidthColumn(const TypePtr& type) {
  if (type->kind() == TypeKind::VARCHAR ||
      type->kind() == TypeKind::VARBINARY || type->kind() == TypeKind::ARRAY ||
      type->kind() == TypeKind::MAP) {
    return true;
  }
  for (size_t i = 0; i < type->size(); ++i) {
    if (hasVariableWidthColumn(type->childAt(i))) {
      return true;
    }
  }
  return false;
}

bool rebaseVariableWidthConcatEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_BATCH_CONCAT_REBASE_VARIABLE_WIDTH");
  if (value == nullptr) {
    // cudf::table(table_view) cannot normalize every exchange-produced
    // nested slice: a real Job 144 input failed inside libcudf slice before
    // the copy could be made. Keep this experimental path opt-in until the
    // producer supplies the original child base and a validated row range.
    return false;
  }
  return std::string_view(value) != "0" && std::string_view(value) != "false" &&
      std::string_view(value) != "FALSE";
}

bool validateVariableWidthConcatEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_BATCH_CONCAT_VALIDATE_VARIABLE_WIDTH");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false" && std::string_view(value) != "FALSE";
}

void validateVariableWidthColumn(
    const cudf::column_view& column,
    std::string_view path,
    rmm::cuda_stream_view stream,
    std::ostringstream& layout) {
  const auto type = column.type().id();
  layout << " " << path << "{type=" << static_cast<int>(type)
         << ",size=" << column.size() << ",offset=" << column.offset()
         << ",children=" << column.num_children();

  if ((type == cudf::type_id::STRING || type == cudf::type_id::LIST) &&
      column.size() > 0) {
    VELOX_CHECK_GE(
        column.num_children(),
        1,
        "Malformed variable-width column at {}",
        path);
    const auto offsets = column.child(0);
    VELOX_CHECK_EQ(
        static_cast<int>(offsets.type().id()),
        static_cast<int>(cudf::type_id::INT32),
        "Malformed offsets child at {}",
        path);
    const auto beginIndex = column.offset();
    const auto endIndex = column.offset() + column.size();
    VELOX_CHECK_GE(beginIndex, 0, "Negative parent offset at {}", path);
    VELOX_CHECK_LT(
        endIndex,
        offsets.size() + offsets.offset(),
        "Offsets child is too short at {}: endIndex={}, childSize={}, childOffset={}",
        path,
        endIndex,
        offsets.size(),
        offsets.offset());

    cudf::size_type edgeOffsets[2]{};
    const auto* offsetsBase = offsets.head<cudf::size_type>();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &edgeOffsets[0],
        offsetsBase + beginIndex,
        sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &edgeOffsets[1],
        offsetsBase + endIndex,
        sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();

    // STRING chars live in the parent data buffer in current cuDF. LIST
    // payload lives in child(1).
    const bool isString = type == cudf::type_id::STRING;
    VELOX_CHECK(
        isString || column.num_children() >= 2,
        "Malformed LIST column without an element child at {}",
        path);
    const auto payloadBegin =
        isString ? edgeOffsets[0] : column.child(1).offset();
    const auto payloadEnd = isString
        ? edgeOffsets[1]
        : column.child(1).offset() + column.child(1).size();
    layout << ",edgeOffsets=[" << edgeOffsets[0] << "," << edgeOffsets[1]
           << "],payloadRange=[" << payloadBegin << "," << payloadEnd << "]"
           << ",data=" << static_cast<const void*>(column.head<char>());
    VELOX_CHECK_GE(
        edgeOffsets[0], 0, "Negative first child offset at {}", path);
    VELOX_CHECK_GE(
        edgeOffsets[1],
        edgeOffsets[0],
        "Non-monotonic edge offsets at {}: first={}, last={}",
        path,
        edgeOffsets[0],
        edgeOffsets[1]);
    VELOX_CHECK_GE(
        edgeOffsets[0],
        payloadBegin,
        "Child offset precedes payload at {}: first={}, payloadBegin={}",
        path,
        edgeOffsets[0],
        payloadBegin);
    VELOX_CHECK_LE(
        edgeOffsets[1],
        payloadEnd,
        "Child offset exceeds payload at {}: last={}, payloadEnd={}",
        path,
        edgeOffsets[1],
        payloadEnd);
    VELOX_CHECK(
        !isString || edgeOffsets[1] == edgeOffsets[0] ||
            column.head<char>() != nullptr,
        "STRING has non-empty offset range but no chars data at {}",
        path);
  }
  layout << "}";

  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    validateVariableWidthColumn(
        column.child(i), fmt::format("{}.{}", path, i), stream, layout);
  }
}

std::string validateVariableWidthTable(
    const cudf::table_view& table,
    rmm::cuda_stream_view stream) {
  std::ostringstream layout;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    validateVariableWidthColumn(
        table.column(i), fmt::format("c{}", i), stream, layout);
  }
  return layout.str();
}

} // namespace

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode)
    : CudfBatchConcat(
          operatorId,
          driverCtx,
          planNode,
          getConcatOutputType(planNode)) {}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode,
    RowTypePtr outputType)
    : CudfBatchConcat(
          operatorId,
          driverCtx,
          planNode,
          std::move(outputType),
          queryConcatTargetRows(driverCtx)) {}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode,
    RowTypePtr outputType,
    int32_t targetRows)
    : CudfBatchConcat(
          operatorId,
          driverCtx,
          std::move(planNode),
          std::move(outputType),
          targetRows,
          concatThreshold(
              driverCtx,
              CudfConfig::kCudfBatchSizeMinThresholdBytes,
              "GLUTEN_CUDF_BATCH_SIZE_MIN_THRESHOLD_BYTES",
              CudfConfig::getInstance().batchSizeMinThresholdBytes)) {}

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode,
    RowTypePtr outputType,
    int32_t targetRows,
    uint64_t targetBytes)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          std::move(outputType),
          planNode->id(),
          "CudfBatchConcat",
          nvtx3::rgb{211, 211, 211}, /* LightGrey */
          NvtxMethodFlag::kAll,
          std::nullopt,
          planNode),
      driverCtx_(driverCtx),
      aggregationStep_(getAggregationStep(planNode)),
      targetRows_(checkedConcatTargetRows(targetRows)),
      targetBytes_(targetBytes),
      variableWidthInputs_(hasVariableWidthColumn(outputType_)),
      rebaseVariableWidthInputs_(
          variableWidthInputs_ && rebaseVariableWidthConcatEnabled()) {
  if (logConcatConfig()) {
    LOG(WARNING) << "CudfBatchConcat configured targetRows=" << targetRows_
                 << ", targetBytes=" << targetBytes_;
  }
}

bool CudfBatchConcat::needsInput() const {
  return !noMoreInput_ && pendingInput_ == nullptr && outputQueue_.empty() &&
      currentNumRows_ < targetRows_ && !reachedFlushThreshold();
}

bool CudfBatchConcat::requiresConcatWorkspace() const {
  if (pendingInput_ != nullptr) {
    const auto cudfVector =
        std::dynamic_pointer_cast<CudfVector>(pendingInput_);
    VELOX_CHECK_NOT_NULL(
        cudfVector, "CudfBatchConcat expects CudfVector input");
    if (cudfVector->size() == 0) {
      return false;
    }

    // Rebasing allocates a new owning table even without a resident tail.
    if (rebaseVariableWidthInputs_) {
      return true;
    }

    // Either pre-admission flush can concatenate the resident tail before
    // handing off the pending vector.
    const auto inputBytes =
        static_cast<uint64_t>(cudfVector->estimateFlatSize());
    const bool flushResidentTail = !buffer_.empty() &&
        (!hasSameEmptyStringCharsPattern(
             buffer_.front()->getTableView(), cudfVector->getTableView()) ||
         inputBytes > kMaxSafeConcatEstimatedBytes -
                 std::min(currentNumBytes_, kMaxSafeConcatEstimatedBytes));
    if (flushResidentTail && buffer_.size() > 1) {
      return currentNumBytes_ <= kMaxMaterializedConcatEstimatedBytes;
    }

    // Otherwise processPendingInput concatenates only if accepting this
    // vector reaches a row or byte bound and there is already a resident
    // vector. A one-vector flush is an ownership handoff without GPU work.
    if (!flushResidentTail && !buffer_.empty()) {
      const auto combinedRows = currentNumRows_ + cudfVector->size();
      const auto combinedBytes =
          currentNumBytes_ > std::numeric_limits<uint64_t>::max() - inputBytes
          ? std::numeric_limits<uint64_t>::max()
          : currentNumBytes_ + inputBytes;
      const bool flushAfterAdmission = combinedRows >= targetRows_ ||
          combinedBytes >= kMaxSafeConcatEstimatedBytes ||
          (targetBytes_ != 0 && combinedBytes >= targetBytes_);
      return flushAfterAdmission &&
          combinedBytes <= kMaxMaterializedConcatEstimatedBytes;
    }
    return false;
  }

  // noMoreInput_ can trigger the final concat without a pending vector.
  return buffer_.size() > 1 &&
      currentNumBytes_ <= kMaxMaterializedConcatEstimatedBytes &&
      (currentNumRows_ >= targetRows_ || reachedFlushThreshold() ||
       noMoreInput_);
}

exec::BlockingReason CudfBatchConcat::isBlocked(ContinueFuture* future) {
  if (!requiresConcatWorkspace() || workspaceAdmission_.has_value()) {
    return exec::BlockingReason::kNotBlocked;
  }

  // Job 144's failing concat requested 281 MiB. Keep enough margin for cuDF
  // metadata and subsidiary buffers without reproducing the progress failure
  // caused by a 1 GiB minimum request near the device headroom watermark.
  constexpr std::size_t kConcatWorkspaceBytes = 512ULL << 20;
  VELOX_CHECK_NOT_NULL(future);
  auto attempt = workspace_.tryAcquire(
      customPool(kCudfDeviceMemoryResourceTag),
      this,
      kConcatWorkspaceBytes,
      CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
      // Concat owns Exchange receive pages while it waits. Treat the bounded
      // replacement as drain work so it runs ahead of FilterProject: leaving
      // both in the transform FIFO can let filters consume the remaining
      // headroom and form a producer/consumer admission cycle.
      DeviceMemoryWorkspacePriority::kDrain);
  if (!attempt.reservation.has_value()) {
    VELOX_CHECK(workspace_.takeFuture(future));
    return exec::BlockingReason::kWaitForArbitration;
  }
  workspaceAdmission_.emplace(std::move(attempt.reservation.value()));
  addRuntimeStat(
      "batchConcatWorkspaceBytes",
      RuntimeCounter(kConcatWorkspaceBytes, RuntimeCounter::Unit::kBytes));
  return exec::BlockingReason::kNotBlocked;
}

bool CudfBatchConcat::reachedFlushThreshold() const {
  return currentNumBytes_ >= kMaxSafeConcatEstimatedBytes ||
      (targetBytes_ != 0 && currentNumBytes_ >= targetBytes_);
}

void CudfBatchConcat::flushBufferedInputs() {
  if (buffer_.empty()) {
    return;
  }
  if (buffer_.size() == 1) {
    outputQueue_.push(std::move(buffer_.front()));
    buffer_.clear();
  } else if (currentNumBytes_ > kMaxMaterializedConcatEstimatedBytes) {
    largeConcatBypassBatches_ += buffer_.size();
    largeConcatBypassBytes_ += currentNumBytes_;
    addRuntimeStat(
        "batchConcatLargeBypassBatches",
        RuntimeCounter(buffer_.size(), RuntimeCounter::Unit::kNone));
    addRuntimeStat(
        "batchConcatLargeBypassBytes",
        RuntimeCounter(currentNumBytes_, RuntimeCounter::Unit::kBytes));
    for (auto& vector : buffer_) {
      outputQueue_.push(std::move(vector));
    }
    buffer_.clear();
  } else {
    const auto outputStream = buffer_.front()->stream();
    auto outputVectors = getConcatenatedCudfVectorsBatched(
        pool(),
        std::exchange(buffer_, {}),
        outputType_,
        outputStream,
        get_output_mr());
    for (auto& output : outputVectors) {
      outputQueue_.push(std::move(output));
    }
  }
  currentNumRows_ = 0;
  currentNumBytes_ = 0;
}

void CudfBatchConcat::doAddInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(
      pendingInput_, "CudfBatchConcat received a second pending input");
  pendingInput_ = std::move(input);
}

void CudfBatchConcat::processPendingInput(RowVectorPtr input) {
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "CudfBatchConcat expects CudfVector input");

  if (cudfVector->size() == 0) {
    return;
  }

  // STRING, MAP and ARRAY columns are represented by signed 32-bit child
  // offsets. Exchange-delivered slices can retain offsets into the producer's
  // larger child buffer; cuDF 26.06 concatenate can then subtract offsets in
  // the wrong domain and request SIZE_MAX-N bytes. A bounded device-to-device
  // table copy rebuilds every child in a zero-based offset domain before the
  // batch is admitted to concat. Synchronize at this ownership boundary so
  // the exchange may safely recycle the sliced source buffer.
  if (rebaseVariableWidthInputs_) {
    const auto rebaseStart = std::chrono::steady_clock::now();
    const auto stream = cudfVector->stream();
    auto rebased = std::make_unique<cudf::table>(
        cudfVector->getTableView(), stream, get_output_mr());
    stream.synchronize();
    cudfVector = std::make_shared<CudfVector>(
        pool(), outputType_, rebased->num_rows(), std::move(rebased), stream);
    ++rebasedBatches_;
    rebaseMicros_ += std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - rebaseStart)
                         .count();
  }

  if (variableWidthInputs_ && validateVariableWidthConcatEnabled()) {
    LOG(WARNING) << "CudfBatchConcat variable-width layout planNode="
                 << planNodeId() << ", batch=" << (inputBatches_ + 1)
                 << ", rows=" << cudfVector->size() << ", layout="
                 << validateVariableWidthTable(
                        cudfVector->getTableView(), cudfVector->stream());
  }

  // UCX producer publication is the ownership boundary for variable-width
  // data: every wire page is emitted by contiguous_split with independent,
  // zero-based STRING/LIST/MAP children. Other CudfVector producers own full
  // cudf tables or standalone packed pages. It is therefore safe to buffer
  // variable-width inputs on the normal concatenate path. The optional D2D
  // rebase above remains diagnostic-only and must not be needed in production.

  const auto inputBytes = static_cast<uint64_t>(cudfVector->estimateFlatSize());
  // cuDF 26.06's strings memcpy concatenate path rejects a zero-byte source
  // when another input supplies chars for the same logical column. Keep
  // unlike physical layouts in separate groups. A uniform all-empty group is
  // safe because libcudf observes total_bytes == 0 and skips the copy loop.
  if (!buffer_.empty() &&
      !hasSameEmptyStringCharsPattern(
          buffer_.front()->getTableView(), cudfVector->getTableView())) {
    flushBufferedInputs();
  }
  // Flush before accepting a wide variable-length batch if combining it with
  // the resident tail could cross cuDF's signed 32-bit child-offset limit.
  // Retain the incoming vector separately while the queued output drains.
  if (!buffer_.empty() &&
      inputBytes > kMaxSafeConcatEstimatedBytes -
              std::min(currentNumBytes_, kMaxSafeConcatEstimatedBytes)) {
    flushBufferedInputs();
  }

  // Push input cudf table to buffer
  ++inputBatches_;
  totalInputRows_ += cudfVector->size();
  totalInputBytes_ += inputBytes;
  currentNumRows_ += cudfVector->size();
  currentNumBytes_ += inputBytes;
  buffer_.push_back(std::move(cudfVector));

  // Enforce the bound here as well as in needsInput(). Source pipelines may
  // have already scheduled another input before the driver observes the
  // updated needsInput() result. Keeping a ready output in outputQueue_ makes
  // the backpressure explicit and prevents scan batches from accumulating up
  // to device capacity. Avoid a redundant D2D concatenate for one large input.
  if (currentNumRows_ >= targetRows_ || reachedFlushThreshold()) {
    flushBufferedInputs();
  }
}

RowVectorPtr CudfBatchConcat::doGetOutput() {
  SCOPE_EXIT {
    workspaceAdmission_.reset();
  };
  VELOX_CHECK(
      !requiresConcatWorkspace() || workspaceAdmission_.has_value(),
      "CudfBatchConcat GPU work requires a live device workspace admission");

  if (pendingInput_ != nullptr) {
    processPendingInput(std::exchange(pendingInput_, nullptr));
  }

  // Drain the queue if there is any output to be flushed
  if (!outputQueue_.empty()) {
    auto output = std::move(outputQueue_.front());
    outputQueue_.pop();
    ++outputBatches_;
    return output;
  }

  // Merge tables if there are enough rows
  if (!buffer_.empty() &&
      (currentNumRows_ >= targetRows_ || reachedFlushThreshold() ||
       noMoreInput_)) {
    // Preserve the zero-copy Exchange fast path when a single received batch
    // already satisfies the target (or is the final tail batch). CudfVector
    // carries its producing stream, so downstream operators can consume it
    // directly without rebinding allocation ownership to another stream.
    if (buffer_.size() == 1) {
      auto output = std::move(buffer_.front());
      buffer_.clear();
      currentNumRows_ = 0;
      currentNumBytes_ = 0;
      ++outputBatches_;
      return output;
    }
    if (currentNumBytes_ > kMaxMaterializedConcatEstimatedBytes) {
      flushBufferedInputs();
      VELOX_CHECK(!outputQueue_.empty());
      auto output = std::move(outputQueue_.front());
      outputQueue_.pop();
      ++outputBatches_;
      return output;
    }
    // Use stream from existing buffer vectors
    const auto outputStream = buffer_[0]->stream();
    auto outputVectors = getConcatenatedCudfVectorsBatched(
        pool(),
        std::exchange(buffer_, {}),
        outputType_,
        outputStream,
        get_output_mr());

    currentNumRows_ = 0;
    currentNumBytes_ = 0;
    VELOX_CHECK_GT(outputVectors.size(), 0);

    for (auto it = outputVectors.begin(); it + 1 != outputVectors.end(); ++it) {
      outputQueue_.push(std::move(*it));
    }

    // If last table is a smaller batch and we still expect more input and keep
    // it in buffer.
    auto& last = outputVectors.back();
    auto rowCount = last->size();

    const auto lastBytes = last->estimateFlatSize();
    if (!noMoreInput_ && rowCount < targetRows_ &&
        (targetBytes_ == 0 || lastBytes < targetBytes_)) {
      currentNumRows_ = rowCount;
      currentNumBytes_ = lastBytes;
      buffer_.push_back(std::move(last));
    } else {
      outputQueue_.push(std::move(last));
    }

    // Return the first batch from the new queue
    if (!outputQueue_.empty()) {
      auto output = std::move(outputQueue_.front());
      outputQueue_.pop();
      ++outputBatches_;
      return output;
    }
  }

  return nullptr;
}

bool CudfBatchConcat::isFinished() {
  const bool finished = noMoreInput_ && pendingInput_ == nullptr &&
      buffer_.empty() && outputQueue_.empty();
  if (finished && !summaryLogged_ && logConcatConfig()) {
    summaryLogged_ = true;
    LOG(WARNING) << "CudfBatchConcat summary planNode=" << planNodeId()
                 << ", step=" << aggregationStep_
                 << ", inputBatches=" << inputBatches_
                 << ", outputBatches=" << outputBatches_
                 << ", inputRows=" << totalInputRows_
                 << ", inputBytes=" << totalInputBytes_
                 << ", rebasedBatches=" << rebasedBatches_
                 << ", rebaseMicros=" << rebaseMicros_
                 << ", largeConcatBypassBatches=" << largeConcatBypassBatches_
                 << ", largeConcatBypassBytes=" << largeConcatBypassBytes_;
  }
  return finished;
}

} // namespace facebook::velox::cudf_velox
