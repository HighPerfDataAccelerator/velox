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
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include <fmt/format.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/CudfPackedSpill.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/RangePartitionFunction.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/search.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::ucx_exchange {

namespace {
// libcudf's hash-partition exclusive scan can exceed the CUDA kernel launch
// resource limit on very large tables before memory pressure is visible. Keep
// an engine-level call-size ceiling even when the caller does not provide one;
// explicit per-query limits remain authoritative and may be smaller.
constexpr int64_t kDefaultMaxRowsPerHashPartitionCall = 128'000'000;

// Admission is cooperative and can lose a race to another output driver.
// When no reservation can be obtained, cap this driver's estimated transient
// hash-partition peak so concurrent unadmitted fallbacks remain bounded.
constexpr uint64_t kMaxUnadmittedHashPartitionPeakBytes = uint64_t{1} << 30;
constexpr uint64_t kMinPartitionedOutputFlushWorkspaceBytes = 384ULL << 20;
constexpr uint64_t kPartitionedOutputFlushFixedWorkspaceBytes = 256ULL << 20;
constexpr uint64_t kPartitionedOutputFlushSourceCopies = 3;

// Cooperative admission reserves the estimated transient peak, but independent
// cudaMallocAsync streams can still race while converting reusable pool pages
// into concat/hash/pack outputs. Keep the short synchronous allocation window
// exclusive; admission is acquired first so a waiter never holds this mutex
// while waiting for device headroom.
std::mutex& partitionedOutputFlushMutex() {
  static std::mutex mutex;
  return mutex;
}

bool containsStructColumn(const cudf::column_view& column) {
  if (column.type().id() == cudf::type_id::STRUCT) {
    return true;
  }
  for (cudf::size_type child = 0; child < column.num_children(); ++child) {
    if (containsStructColumn(column.child(child))) {
      return true;
    }
  }
  return false;
}

bool containsStructColumn(const cudf::table_view& table) {
  for (cudf::size_type column = 0; column < table.num_columns(); ++column) {
    if (containsStructColumn(table.column(column))) {
      return true;
    }
  }
  return false;
}

const std::string kCacheRootOutputBatchRows =
    "spark.gluten.sql.columnar.backend.velox.cudf.cache_root_output_batch_rows";
const std::string kCacheRootOutputBatchBytes =
    "spark.gluten.sql.columnar.backend.velox.cudf.cache_root_output_batch_bytes";

bool isMppRootOutput(const core::PlanNodeId& planNodeId) {
  return planNodeId == "mpp_output_0";
}

int64_t targetRowsPerUcxChunk(
    const core::QueryConfig& queryConfig,
    const core::PlanNodeId& planNodeId) {
  if (isMppRootOutput(planNodeId)) {
    const auto cacheRows =
        queryConfig.get<int64_t>(kCacheRootOutputBatchRows, 0);
    if (cacheRows > 0) {
      return cacheRows;
    }
  }
  if (const char* value =
          std::getenv("GLUTEN_UCX_PARTITIONED_OUTPUT_BATCH_ROWS")) {
    try {
      const auto parsed = static_cast<int64_t>(std::stoll(value));
      if (parsed > 0) {
        return parsed;
      }
    } catch (...) {
    }
  }
  return queryConfig.ucxPartitionedOutputBatchRows();
}

void normalizePartitionOffsets(
    std::vector<cudf::size_type>& offsets,
    size_t numPartitions) {
  VELOX_CHECK(
      offsets.size() == numPartitions || offsets.size() == numPartitions + 1,
      "Unexpected libcudf partition offset count {} for {} partitions",
      offsets.size(),
      numPartitions);
  VELOX_CHECK_EQ(offsets.front(), 0);
  offsets.erase(offsets.begin());
  if (offsets.size() == numPartitions) {
    offsets.pop_back();
  }
}

int64_t positiveEnvironmentOverride(const char* name) {
  if (const char* value = std::getenv(name)) {
    try {
      const auto parsed = std::stoll(value);
      if (parsed > 0) {
        return static_cast<int64_t>(parsed);
      }
    } catch (...) {
    }
  }
  return 0;
}

int64_t maxRowsPerHashPartitionCall(const core::QueryConfig& queryConfig) {
  const auto environmentRows =
      positiveEnvironmentOverride("GLUTEN_UCX_HASH_PARTITION_INPUT_BATCH_ROWS");
  if (environmentRows > 0) {
    return environmentRows;
  }
  const auto configuredRows = queryConfig.ucxHashPartitionInputBatchRows();
  return configuredRows > 0 ? configuredRows
                            : kDefaultMaxRowsPerHashPartitionCall;
}

int64_t maxRowsPerHashPartitionWindow(const core::QueryConfig& queryConfig) {
  const auto environmentRows =
      positiveEnvironmentOverride("GLUTEN_UCX_HASH_PARTITION_WINDOW_ROWS");
  return environmentRows > 0 ? environmentRows
                             : queryConfig.ucxHashPartitionWindowRows();
}

bool sourceCanExternalizeOnBackpressure(
    std::shared_ptr<const core::PlanNode> source) {
  while (source) {
    if (const auto aggregation =
            std::dynamic_pointer_cast<const core::AggregationNode>(source)) {
      return aggregation->step() == core::AggregationNode::Step::kPartial ||
          aggregation->step() == core::AggregationNode::Step::kIntermediate;
    }
    if (const auto topN =
            std::dynamic_pointer_cast<const core::TopNRowNumberNode>(source)) {
      // A low-reduction partial TopN may return almost the complete input
      // batch.  UcxPartitionedOutput partitions that source in bounded
      // windows.  If it observes queue backpressure after the first window,
      // the complete multi-GiB TopN result otherwise remains resident on the
      // device until a consumer drains the queue.  Multiple independent MPP
      // fragments can all reach this ownership boundary at once and exhaust
      // the device even though each exchange queue is individually bounded.
      // Externalize the unconsumed tail of this one already-produced owner,
      // then let normal queue backpressure stop the upstream pipeline before
      // it creates another.
      return topN->partialOutput();
    }
    if (!std::dynamic_pointer_cast<const core::ProjectNode>(source) ||
        source->sources().size() != 1) {
      return false;
    }
    source = source->sources().front();
  }
  return false;
}

uint64_t packedHostBytesLimit() {
  const auto sharedOverride =
      positiveEnvironmentOverride("CUDF_PACKED_HOST_BYTES");
  if (sharedOverride > 0) {
    return static_cast<uint64_t>(sharedOverride);
  }
  const auto graceCompatibility =
      positiveEnvironmentOverride("CUDF_HASH_JOIN_GRACE_HOST_BYTES");
  if (graceCompatibility > 0) {
    return static_cast<uint64_t>(graceCompatibility);
  }
  return uint64_t{8} << 30;
}

uint64_t targetBytesPerUcxChunk(
    const core::QueryConfig& queryConfig,
    const core::PlanNodeId& planNodeId) {
  if (isMppRootOutput(planNodeId)) {
    const auto cacheBytes =
        queryConfig.get<uint64_t>(kCacheRootOutputBatchBytes, 0);
    if (cacheBytes > 0) {
      return cacheBytes;
    }
  }
  if (const char* value =
          std::getenv("GLUTEN_UCX_PARTITIONED_OUTPUT_BATCH_BYTES")) {
    try {
      return std::stoull(value);
    } catch (...) {
    }
  }
  return queryConfig.ucxPartitionedOutputBatchBytes();
}

cudf::size_type rowsPerUcxChunk(
    cudf::size_type rows,
    uint64_t bytes,
    int64_t targetRows,
    uint64_t targetBytes) {
  auto rowsPerChunk = rows;
  if (targetRows > 0) {
    rowsPerChunk = std::min<cudf::size_type>(rowsPerChunk, targetRows);
  }
  if (targetBytes > 0 && bytes > targetBytes && rows > 0) {
    const auto byteLimitedRows = std::max<uint64_t>(
        1, static_cast<uint64_t>(rows) * targetBytes / bytes);
    rowsPerChunk = std::min<cudf::size_type>(
        rowsPerChunk,
        static_cast<cudf::size_type>(std::min<uint64_t>(
            byteLimitedRows, std::numeric_limits<cudf::size_type>::max())));
  }
  return std::max<cudf::size_type>(1, rowsPerChunk);
}

struct OwningPackedChunk {
  cudf::size_type rows;
  std::unique_ptr<cudf::packed_columns> data;
};

/// Transfers a completed contiguous_split page into the exchange queue.
///
/// packed_columns owns both metadata and device storage, so moving these
/// owners is sufficient.  Copying the whole device page here used to mask a
/// receiver stream-ordering bug and added one full D2D transfer per UCX page.
std::unique_ptr<cudf::packed_columns> takePackedColumns(
    cudf::packed_columns&& packed,
    rmm::cuda_stream_view stream) {
  auto result = std::make_unique<cudf::packed_columns>(
      std::move(packed.metadata), std::move(packed.gpu_data));
  if (exchangeVariableWidthValidationEnabled()) {
    const auto view = cudf::unpack(*result);
    LOG(WARNING) << "UCX producer owning page rows=" << view.num_rows()
                 << " bytes=" << result->gpu_data->size() << " layout="
                 << validateVariableWidthTableLayout(view, stream);
  }
  return result;
}

/// Builds independently-owned wire pages whose nested children all live in
/// the page's own zero-based offset domain.
///
/// Do not implement this as cudf::slice() followed by cudf::pack(). A slice of
/// STRING/LIST/MAP can retain the producer parent's child offset, and packing
/// that view preserves metadata which is only valid in the parent's domain.
/// The receiver then exposes a logically sliced packed table to
/// CudfBatchConcat; libcudf concatenate may interpret the child range against
/// the wrong base and request a SIZE_MAX-N allocation. contiguous_split is the
/// ownership boundary: it copies each logical row range into an independent
/// packed table and rewrites every nested offset relative to that page.
std::vector<OwningPackedChunk> makeOwningPackedChunks(
    cudf::table_view table,
    cudf::size_type rowsPerChunk,
    rmm::cuda_stream_view stream) {
  CudaCallDiagnosticScope callDiagnostic(fmt::format(
      "operator=UcxPartitionedOutput phase=contiguousSplit rows={} "
      "rowsPerChunk={} stream={}",
      table.num_rows(),
      rowsPerChunk,
      static_cast<const void*>(stream.value())));
  VELOX_CHECK_GT(rowsPerChunk, 0);
  if (table.num_rows() == 0) {
    return {};
  }

  std::vector<cudf::size_type> splitOffsets;
  for (cudf::size_type offset = rowsPerChunk; offset < table.num_rows();
       offset += rowsPerChunk) {
    splitOffsets.push_back(offset);
  }
  auto packedTables = cudf::contiguous_split(
      table,
      splitOffsets,
      stream,
      cudf::get_current_device_resource_ref());
  stream.synchronize();

  std::vector<OwningPackedChunk> result;
  result.reserve(packedTables.size());
  for (auto& packedTable : packedTables) {
    const auto rows = packedTable.table.num_rows();
    auto data = takePackedColumns(std::move(packedTable.data), stream);
    result.push_back({rows, std::move(data)});
  }
  return result;
}
} // namespace

// Computes a mapping from names in n2 to names in n1
// and returns that mapping in remap.
// Names in n2 must occurs in n1.
static void getRemapping(
    const RowTypePtr& inputType,
    const RowTypePtr& outputType,
    std::vector<uint32_t>& remap) {
  remap.clear();
  remap.reserve(outputType->size());
  std::unordered_map<std::string, size_t> nextOccurrence;
  for (uint32_t out = 0; out < outputType->size(); ++out) {
    const auto& name = outputType->nameOf(out);
    std::vector<uint32_t> matches;
    for (uint32_t in = 0; in < inputType->size(); ++in) {
      if (inputType->nameOf(in) == name &&
          inputType->childAt(in)->equivalent(*outputType->childAt(out))) {
        matches.push_back(in);
      }
    }
    VELOX_CHECK(
        !matches.empty(),
        "UCX output field {}:{} has no name-and-type match in input {}",
        name,
        outputType->childAt(out)->toString(),
        inputType->toString());
    auto& occurrence = nextOccurrence[name];
    const auto selected = matches[std::min(occurrence, matches.size() - 1)];
    ++occurrence;
    remap.push_back(selected);
  }
}

UcxPartitionedOutput::UcxPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "cudfPartitionedOutput"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queueManager_(UcxOutputQueueManager::getInstanceRef()),
      numPartitions_(planNode->numPartitions()),
      localDeviceRootOutput_(
          planNode->numPartitions() == 1 &&
          ctx->queryConfig().get<bool>(
              LocalDeviceOutputQueueManager::kEnabledConfig, false) &&
          LocalDeviceOutputQueueManager::getInstanceRef()
              ->isDirectOutputTask(ctx->task->taskId())),
      pipelineId_(ctx->pipelineId),
      driverId_(ctx->driverId),
      sourceCanExternalizeOnBackpressure_(
          sourceCanExternalizeOnBackpressure(
              planNode->sources().front())),
      packedHostBytesLimit_(packedHostBytesLimit()),
      maxOutputBufferSize_(ctx->queryConfig().maxOutputBufferSize()),
      targetRowsPerChunk_(
          targetRowsPerUcxChunk(ctx->queryConfig(), planNode->id())),
      targetBytesPerChunk_(
          targetBytesPerUcxChunk(ctx->queryConfig(), planNode->id())),
      hashPartitionInputBatchRows_(
          maxRowsPerHashPartitionCall(ctx->queryConfig())),
      hashPartitionWindowRows_(
          maxRowsPerHashPartitionWindow(ctx->queryConfig())) {
  this->initPartitionKeys(planNode);
  auto sources = planNode->sources();
  std::vector<std::string> inNames, outNames;
  inNames.reserve(planNode->inputType()->size());
  for (int i = 0; i < planNode->inputType()->size(); ++i) {
    inNames.push_back(planNode->inputType()->nameOf(i));
  }
  outNames.reserve(planNode->outputType()->size());
  for (int i = 0; i < planNode->outputType()->size(); ++i) {
    outNames.push_back(planNode->outputType()->nameOf(i));
  }
  if (inNames != outNames) {
    getRemapping(planNode->inputType(), planNode->outputType(), remap_);
  }
}

void UcxPartitionedOutput::addInput(RowVectorPtr input) {
  CudaCallDiagnosticScope callDiagnostic(fmt::format(
      "operator=UcxPartitionedOutput task={} method=addInput rows={}",
      taskId(),
      input == nullptr ? 0 : input->size()));
  CudaAllocationTraceScope allocationTrace(
      fmt::format("UcxPartitionedOutput task={} method=addInput", taskId()));
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " addInput";
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "Input must be a CudfVector");
  VELOX_CHECK(
      !future_.valid() || future_.hasValue(),
      "addInput with outstanding future!");
  VELOX_CHECK(!hasActiveFlush(), "addInput while a flush is still active");
  VELOX_CHECK(
      !pendingFlushReady_, "addInput while a pending flush awaits admission");

  const auto inputFlatBytes = input->estimateFlatSize();
  // Record stats per-input (before buffering).
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(inputFlatBytes, input->size());
  }

  if (localDeviceRootOutput_) {
    VELOX_CHECK(
        remap_.empty(),
        "Direct local device output requires identical input/output order");
    LocalDeviceOutputQueueManager::getInstanceRef()->enqueue(
        this->taskId(), 0, std::move(cudfVector));
    updateBackpressure();
    return;
  }

  pendingRows_ += cudfVector->getTableView().num_rows();
  pendingFlatBytes_ += inputFlatBytes;
  pendingInputs_.push_back(std::move(cudfVector));

  if ((targetRowsPerChunk_ <= 0 && targetBytesPerChunk_ == 0) ||
      (targetRowsPerChunk_ > 0 && pendingRows_ >= targetRowsPerChunk_) ||
      (targetBytesPerChunk_ > 0 &&
       pendingFlatBytes_ >= targetBytesPerChunk_)) {
    pendingFlushReady_ = true;
  }
}

void UcxPartitionedOutput::flushPending() {
  CudaAllocationTraceScope allocationTrace(
      fmt::format(
          "UcxPartitionedOutput task={} method=flushPending", taskId()));
  if (!hasActiveFlush() && pendingInputs_.empty()) {
    return;
  }

  try {
    if (!hasActiveFlush()) {
      preparePendingFlush();
    }
    advanceActiveFlush();

  } catch (const rmm::bad_alloc& e) {
    VLOG(1)
        << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
        << " caught memory alloc error, removing all memory in output queues";
    pendingInputs_.clear();
    pendingRows_ = 0;
    pendingFlatBytes_ = 0;
    pendingFlushReady_ = false;
    clearActiveFlush();
    for (int i = 0; i < numPartitions_; i++) {
      sharedQueueManager()->deleteResults(this->taskId(), i);
    }
    throw;
  }
}

uint64_t UcxPartitionedOutput::flushWorkspaceBytes() const {
  // The source owner is already reflected in cudaMemGetInfo.  A threshold
  // flush can additionally materialize a concatenate result, hash-partition
  // result, and packed destination buffers before their asynchronous releases
  // become reusable.  Account from the actual owner instead of a fixed batch
  // assumption: variable-width TopN output can overshoot the configured byte
  // threshold by one complete input batch.
  const auto sourceBytes = hasActiveFlush() ? activeSourceFlatBytes_
                                            : pendingFlatBytes_;
  if (sourceBytes >
      (std::numeric_limits<uint64_t>::max() -
       kPartitionedOutputFlushFixedWorkspaceBytes) /
          kPartitionedOutputFlushSourceCopies) {
    return std::numeric_limits<uint64_t>::max();
  }
  return std::max<uint64_t>(
      kMinPartitionedOutputFlushWorkspaceBytes,
      sourceBytes * kPartitionedOutputFlushSourceCopies +
          kPartitionedOutputFlushFixedWorkspaceBytes);
}

void UcxPartitionedOutput::preparePendingFlush() {
  VELOX_CHECK(!hasActiveFlush());
  VELOX_CHECK(!pendingInputs_.empty());

  activeInputs_ = std::move(pendingInputs_);
  activeSourceFlatBytes_ = pendingFlatBytes_;
  pendingInputs_.clear();
  pendingRows_ = 0;
  pendingFlatBytes_ = 0;
  pendingFlushReady_ = false;

  auto stream = activeInputs_.back()->stream();
  if (activeInputs_.size() > 1) {
    std::vector<cudf::table_view> views;
    std::vector<rmm::cuda_stream_view> inputStreams;
    views.reserve(activeInputs_.size());
    inputStreams.reserve(activeInputs_.size());
    for (auto& input : activeInputs_) {
      inputStreams.push_back(input->stream());
      views.push_back(
          remap_.empty()
              ? input->getTableView()
              : input->getTableView().select(remap_.begin(), remap_.end()));
    }

    cudf::detail::join_streams(inputStreams, stream);
    activeMergedTable_ = cudf::concatenate(
        views, stream, cudf::get_current_device_resource_ref());
    orderCudfVectorDeallocationsAfterStream(
        activeInputs_, inputStreams, stream);
    // The concatenated table is now the source owner. Releasing the input
    // vectors here preserves the old 2x -> 1x peak-memory behavior.
    activeInputs_.clear();
  }

  activeStream_ = stream;
  activeNextRow_ = 0;
  const auto tableRows = activeTableView().num_rows();
  if (numPartitions_ > 1 && rangeBoundsJson_.empty() &&
      (partitionKeyIndices_.size() > 0 || spec_ == "gather") &&
      hashPartitionInputBatchRows_ > 0) {
    const auto configuredWindowRows = hashPartitionWindowRows_ > 0
        ? hashPartitionWindowRows_
        : (targetRowsPerChunk_ > 0 ? targetRowsPerChunk_
                                   : hashPartitionInputBatchRows_);
    const auto configuredRows = static_cast<uint64_t>(
        std::max<int64_t>(hashPartitionInputBatchRows_, configuredWindowRows));
    // Source residency and destination message size are different bounds.
    // A hash window is distributed across all destinations, so callers with
    // sufficient receive credit may explicitly use a larger source window
    // without increasing the per-destination chunk limit below.
    const auto configuredWindowBytes =
        positiveEnvironmentOverride("GLUTEN_UCX_HASH_PARTITION_WINDOW_BYTES");
    const auto sourceWindowBytes = configuredWindowBytes > 0
        ? static_cast<uint64_t>(configuredWindowBytes)
        : targetBytesPerChunk_;
    activeRowsPerWindow_ = rowsPerUcxChunk(
        tableRows,
        activeSourceFlatBytes_,
        static_cast<int64_t>(std::min<uint64_t>(
            configuredRows,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))),
        sourceWindowBytes);
  } else if (numPartitions_ == 1) {
    // SINGLE/gather exchanges yield after each output-sized chunk as well.
    activeRowsPerWindow_ = rowsPerUcxChunk(
        tableRows,
        activeSourceFlatBytes_,
        targetRowsPerChunk_,
        targetBytesPerChunk_);
  } else {
    // Bound HASH/RANGE residency by the output byte target even when no
    // explicit hash-call row limit is configured.
    activeRowsPerWindow_ = rowsPerUcxChunk(
        tableRows,
        activeSourceFlatBytes_,
        targetRowsPerChunk_,
        targetBytesPerChunk_);
  }

}

bool UcxPartitionedOutput::hasActiveFlush() const {
  return hasActiveDeviceSource() || !activeHostSources_.empty();
}

bool UcxPartitionedOutput::hasActiveDeviceSource() const {
  return activeMergedTable_ != nullptr || !activeInputs_.empty() ||
      activeRestoredSource_.has_value();
}

cudf::table_view UcxPartitionedOutput::activeTableView() {
  VELOX_CHECK(hasActiveDeviceSource());
  if (activeRestoredSource_) {
    VELOX_CHECK_EQ(activeRestoredSource_->tables().size(), 1);
    return activeRestoredSource_->tables().front();
  }
  if (activeMergedTable_) {
    return activeMergedTable_->view();
  }
  VELOX_CHECK_EQ(activeInputs_.size(), 1);
  auto tableView = activeInputs_.front()->getTableView();
  return remap_.empty() ? tableView
                        : tableView.select(remap_.begin(), remap_.end());
}

bool UcxPartitionedOutput::externalizeActiveSourceTail() {
  CudaCallDiagnosticScope callDiagnostic(fmt::format(
      "operator=UcxPartitionedOutput task={} phase=externalizeSourceTail",
      taskId()));
  if (!sourceCanExternalizeOnBackpressure_ || !hasActiveDeviceSource() ||
      !activeStream_) {
    return false;
  }

  const auto tableView = activeTableView();
  const auto tableRows = tableView.num_rows();
  if (activeNextRow_ >= tableRows) {
    return false;
  }
  VELOX_CHECK_GT(activeRowsPerWindow_, 0);

  // Reserve conservatively for the complete source. The reservation is one
  // shared token referenced by every bounded page and is released only after
  // the final page has completed H2D restore. This makes admission atomic: we
  // never copy half a tail to host and then discover that its remainder has
  // no tier capacity.
  auto hostReservation = tryReserveCudfPackedHostMemory(
      activeSourceFlatBytes_, packedHostBytesLimit_);
  if (!hostReservation) {
    LOG(WARNING) << "UcxPartitionedOutput could not externalize oversized "
                    "source tail: packed host tier is full task="
                 << taskId() << " sourceFlatBytes=" << activeSourceFlatBytes_
                 << " hostReservedBytes="
                 << currentCudfPackedHostMemoryReservedBytes()
                 << " hostLimitBytes=" << packedHostBytesLimit_;
    return false;
  }

  auto stream = *activeStream_;
  std::deque<HostSourceChunk> hostChunks;
  uint64_t copiedBytes = 0;
  uint64_t copiedRows = 0;
  for (auto offset = activeNextRow_; offset < tableRows;) {
    const auto end = std::min<cudf::size_type>(
        tableRows, offset + activeRowsPerWindow_);
    auto slices = cudf::slice(tableView, {offset, end}, stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    auto packedChunks =
        makeOwningPackedChunks(slices.front(), end - offset, stream);
    VELOX_CHECK_EQ(packedChunks.size(), 1);
    auto& packed = packedChunks.front();
    const auto dataBytes =
        static_cast<uint64_t>(packed.data->gpu_data->size());
    VELOX_CHECK_LE(
        dataBytes,
        static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
        "UCX packed host page exceeds addressable host allocation size");
    auto hostData = std::shared_ptr<uint8_t>(
        dataBytes == 0 ? nullptr : new uint8_t[static_cast<size_t>(dataBytes)],
        std::default_delete<uint8_t[]>());
    if (dataBytes > 0) {
      CudaCallDiagnosticScope copyDiagnostic(fmt::format(
          "operator=UcxPartitionedOutput task={} phase=externalizeD2H "
          "rows={} bytes={} src={} dst={} stream={}",
          taskId(),
          packed.rows,
          dataBytes,
          packed.data->gpu_data->data(),
          static_cast<const void*>(hostData.get()),
          static_cast<const void*>(stream.value())));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostData.get(),
          packed.data->gpu_data->data(),
          dataBytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    // Synchronize once per bounded page so at most one additional packed
    // device allocation exists beside the original source owner.
    stream.synchronize();
    copiedBytes += dataBytes;
    copiedRows += packed.rows;
    hostChunks.push_back(HostSourceChunk{
        std::move(packed.data->metadata),
        std::move(hostData),
        hostReservation,
        dataBytes,
        packed.rows});
    offset = end;
  }

  diagnosticHostExternalizedBytes_ += copiedBytes;
  diagnosticHostExternalizedChunks_ += hostChunks.size();
  activeHostSources_.insert(
      activeHostSources_.end(),
      std::make_move_iterator(hostChunks.begin()),
      std::make_move_iterator(hostChunks.end()));
  LOG(WARNING) << "UcxPartitionedOutput externalized oversized source tail "
                  "after backpressure task="
               << taskId() << " rows=" << copiedRows
               << " packedBytes=" << copiedBytes
               << " chunks=" << activeHostSources_.size()
               << " cumulativeExternalizedBytes="
               << diagnosticHostExternalizedBytes_
               << " cumulativeExternalizedChunks="
               << diagnosticHostExternalizedChunks_;

  // Every D2H copy is complete. The original multi-GiB partial result can be
  // destroyed even though its exchange queue remains blocked.
  releaseActiveDeviceSource();
  return true;
}

void UcxPartitionedOutput::restoreNextHostSource() {
  VELOX_CHECK(!hasActiveDeviceSource());
  VELOX_CHECK(!activeHostSources_.empty());
  VELOX_CHECK(activeStream_.has_value());

  auto chunk = std::move(activeHostSources_.front());
  activeHostSources_.pop_front();
  const auto dataBytes = chunk.dataBytes;
  const auto rows = chunk.rows;
  CudaCallDiagnosticScope callDiagnostic(fmt::format(
      "operator=UcxPartitionedOutput task={} phase=restoreHostSource "
      "rows={} bytes={} src={} stream={}",
      taskId(),
      rows,
      dataBytes,
      static_cast<const void*>(chunk.data.get()),
      static_cast<const void*>(activeStream_->value())));
  std::vector<CudfPackedHostRestoreChunk> restoreChunks;
  restoreChunks.push_back(CudfPackedHostRestoreChunk{
      std::move(chunk.metadata),
      std::move(chunk.data),
      chunk.dataBytes,
      std::move(chunk.hostReservation)});
  activeRestoredSource_ = bulkRestoreCudfPackedHostChunks(
      std::move(restoreChunks),
      *activeStream_,
      cudf::get_current_device_resource_ref());
  VELOX_CHECK_EQ(activeRestoredSource_->tables().size(), 1);
  VELOX_CHECK_EQ(activeRestoredSource_->tables().front().num_rows(), rows);
  activeSourceFlatBytes_ = dataBytes;
  activeNextRow_ = 0;
  activeRowsPerWindow_ = std::max<cudf::size_type>(1, rows);
  diagnosticHostRestoredBytes_ += dataBytes;
  VLOG(2) << "UcxPartitionedOutput restored one bounded host source task="
          << taskId() << " rows=" << rows << " packedBytes=" << dataBytes
          << " remainingHostChunks=" << activeHostSources_.size()
          << " cumulativeRestoredBytes=" << diagnosticHostRestoredBytes_;
}

void UcxPartitionedOutput::releaseActiveDeviceSource() {
  activeInputs_.clear();
  activeMergedTable_.reset();
  activeRestoredSource_.reset();
  activeSourceFlatBytes_ = 0;
  activeNextRow_ = 0;
  activeRowsPerWindow_ = 0;
  if (activeHostSources_.empty()) {
    activeStream_.reset();
  }
}

void UcxPartitionedOutput::clearActiveFlush() {
  releaseActiveDeviceSource();
  activeHostSources_.clear();
  activeStream_.reset();
}

void UcxPartitionedOutput::updateBackpressure() {
  // The queue is shared by every output driver in this task. Checking after
  // each residency window bounds overshoot to at most one window per driver,
  // instead of allowing one addInput() to enqueue every remaining window.
  // P0 deliberately checks after enqueue: exact packed bytes are only known
  // then, and pre-reserving a window larger than maxSize would deadlock unless
  // the credit protocol also grew a special oversized-window grant.
  auto blocked = localDeviceRootOutput_
      ? LocalDeviceOutputQueueManager::getInstanceRef()->checkBlocked(
            this->taskId(), &future_)
      : sharedQueueManager()->checkBlocked(this->taskId(), &future_);
  if (blocked) {
    VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
            << " is blocked after output window";
  }
  blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                            : exec::BlockingReason::kNotBlocked;
}

void UcxPartitionedOutput::advanceActiveFlush() {
  CudaCallDiagnosticScope callDiagnostic(fmt::format(
      "operator=UcxPartitionedOutput task={} method=advanceActiveFlush",
      taskId()));
  VELOX_CHECK(hasActiveFlush());
  VELOX_CHECK(activeStream_.has_value());
  VELOX_CHECK_EQ(blockingReason_, exec::BlockingReason::kNotBlocked);

  if (!hasActiveDeviceSource()) {
    restoreNextHostSource();
  }

  auto tableView = activeTableView();
  const auto tableRows = tableView.num_rows();
  if (activeNextRow_ >= tableRows) {
    releaseActiveDeviceSource();
    return;
  }

  auto stream = *activeStream_;
  auto rowsThisWindow = std::min<cudf::size_type>(
      activeRowsPerWindow_, tableRows - activeNextRow_);
  std::optional<cudf_velox::DeviceMemoryAdmissionReservation> memoryAdmission;

  if (numPartitions_ > 1 && rangeBoundsJson_.empty() &&
      (partitionKeyIndices_.size() > 0 || spec_ == "gather") &&
      hashPartitionInputBatchRows_ > 0 && rowsThisWindow > 0 &&
      activeSourceFlatBytes_ > 0 && tableRows > 0) {
    const auto sourceRows = static_cast<uint64_t>(tableRows);
    const auto averageRowBytes = activeSourceFlatBytes_ / sourceRows +
        static_cast<uint64_t>(activeSourceFlatBytes_ % sourceRows != 0);
    const auto candidateRows = static_cast<uint64_t>(rowsThisWindow);
    const auto estimatePeakBytes = [&](uint64_t rows) {
      const auto sourceBytes = averageRowBytes * rows;
      const auto scratchBytes = rows * uint64_t{12} +
          static_cast<uint64_t>(numPartitions_) * uint64_t{4096};
      // activeTableView() already owns the source and headroom excludes that
      // live allocation. Admission reserves only the new partitioned output,
      // hash scratch, and allocator headroom.
      const auto workingBytes = sourceBytes + scratchBytes;
      return workingBytes + workingBytes / uint64_t{4};
    };
    const auto candidatePeakBytes = estimatePeakBytes(candidateRows);
    // The unadmitted fallback below already proves that at most 1 GiB of new
    // hash workspace is safe. Avoid a global CUDA/RMM headroom snapshot for
    // windows inside that bound; sampling every normal batch serializes the
    // producer pipeline and caused a measurable exchange regression.
    if (candidatePeakBytes <= kMaxUnadmittedHashPartitionPeakBytes) {
      VLOG(2) << "UcxPartitionedOutput bounded hash fast path task=" << taskId()
              << " sourceRows=" << tableRows
              << " sourceFlatBytes=" << activeSourceFlatBytes_
              << " selectedRows=" << candidateRows
              << " estimatedPeakBytes=" << candidatePeakBytes;
    } else {
      const auto fallbackRows = maxOutputBufferSize_ == 0
          ? candidateRows
          : std::max<uint64_t>(
                1,
                std::min(
                    candidateRows, maxOutputBufferSize_ / averageRowBytes));
      const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
      const auto allocatableBytes = headroom.allocatableBytes();
      const auto reserveBytes = headroom.totalBytes == 0
          ? uint64_t{1} << 30
          : std::max<uint64_t>(
                uint64_t{1} << 30,
                static_cast<uint64_t>(headroom.totalBytes) / uint64_t{50});
      const auto admissionCapacity = allocatableBytes > reserveBytes
          ? allocatableBytes - reserveBytes
          : allocatableBytes / uint64_t{2};
      const auto alreadyReserved =
          cudf_velox::deviceMemoryAdmissionReservedBytes(headroom.device);
      const auto availableCapacity = admissionCapacity > alreadyReserved
          ? admissionCapacity - alreadyReserved
          : uint64_t{0};
      uint64_t selectedRows = candidateRows;
      if (!headroom.cudaValid || candidatePeakBytes > availableCapacity) {
        const auto pressureRows = candidatePeakBytes == 0
            ? candidateRows
            : std::max<uint64_t>(
                  1,
                  static_cast<uint64_t>(
                      static_cast<long double>(candidateRows) *
                      static_cast<long double>(availableCapacity) /
                      static_cast<long double>(candidatePeakBytes)));
        // The 1GiB queue-derived bound is no longer the normal workspace cap.
        // It is retained as the proven emergency floor when a snapshot is
        // unavailable or severely pressured; Q10 passed 10/10 at this size.
        selectedRows =
            std::min(candidateRows, std::max(fallbackRows, pressureRows));
      }

      auto selectedPeakBytes = estimatePeakBytes(selectedRows);
      memoryAdmission = cudf_velox::tryAcquireDeviceMemoryAdmission(
          headroom.device, selectedPeakBytes, admissionCapacity);
      bool admissionRetried = false;
      if (!memoryAdmission && fallbackRows < selectedRows) {
        // The initial size used a non-atomic reservation snapshot. Another
        // driver may have acquired capacity before this atomic attempt, so
        // shrink to the proven queue-sized fallback and retry the reservation.
        selectedRows = fallbackRows;
        selectedPeakBytes = estimatePeakBytes(selectedRows);
        memoryAdmission = cudf_velox::tryAcquireDeviceMemoryAdmission(
            headroom.device, selectedPeakBytes, admissionCapacity);
        admissionRetried = true;
      }
      if (!memoryAdmission &&
          selectedPeakBytes > kMaxUnadmittedHashPartitionPeakBytes) {
        // Admission can still be unavailable when every byte is temporarily
        // reserved, or when CUDA headroom could not be sampled. A driver must
        // not then execute the original large window unaccounted. Find the
        // largest row count whose estimated peak is at most the explicit 1 GiB
        // fallback bound, and give that smaller window one final admission try.
        uint64_t lower = 1;
        uint64_t upper = selectedRows;
        if (estimatePeakBytes(lower) <= kMaxUnadmittedHashPartitionPeakBytes) {
          while (lower < upper) {
            const auto middle = lower + (upper - lower + 1) / 2;
            if (estimatePeakBytes(middle) <=
                kMaxUnadmittedHashPartitionPeakBytes) {
              lower = middle;
            } else {
              upper = middle - 1;
            }
          }
        }
        selectedRows = lower;
        selectedPeakBytes = estimatePeakBytes(selectedRows);
        memoryAdmission = cudf_velox::tryAcquireDeviceMemoryAdmission(
            headroom.device, selectedPeakBytes, admissionCapacity);
        admissionRetried = true;
      }
      const bool unadmittedFallback = !memoryAdmission;
      rowsThisWindow = static_cast<cudf::size_type>(selectedRows);
      VLOG(2) << "UcxPartitionedOutput pressure-aware hash window task="
              << taskId() << " sourceRows=" << tableRows
              << " sourceFlatBytes=" << activeSourceFlatBytes_
              << " averageRowBytes=" << averageRowBytes
              << " candidateRows=" << candidateRows
              << " selectedRows=" << selectedRows
              << " estimatedPeakBytes=" << selectedPeakBytes
              << " cudaFreeBytes=" << headroom.freeBytes
              << " poolReusableBytes=" << headroom.reusablePoolBytes()
              << " admissionCapacityBytes=" << admissionCapacity
              << " alreadyReservedBytes=" << alreadyReserved
              << " admissionRetried=" << admissionRetried
              << " admitted=" << memoryAdmission.has_value()
              << " unadmittedFallback=" << unadmittedFallback;
    }
  }

  const auto end = activeNextRow_ + rowsThisWindow;
  auto slices = cudf::slice(tableView, {activeNextRow_, end}, stream);
  VELOX_CHECK_EQ(slices.size(), 1);

  auto partitionInput = slices[0];
  std::unique_ptr<cudf::table> materializedPartitionInput;
  if (numPartitions_ > 1 && containsStructColumn(partitionInput)) {
    // libcudf partition requires STRUCT children to align with their sliced
    // parent. Materialize this bounded window to normalize nested offsets.
    materializedPartitionInput = std::make_unique<cudf::table>(
        partitionInput, stream, cudf::get_current_device_resource_ref());
    partitionInput = materializedPartitionInput->view();
  }

  if (numPartitions_ > 1) {
    if (!rangeBoundsJson_.empty()) {
      rangePartition(partitionInput, stream);
    } else if (partitionKeyIndices_.size() > 0 || spec_ == "gather") {
      // hashPartition() may internally split this residency window into safe
      // libcudf call-size chunks, but it cannot cross into the next window.
      hashPartition(partitionInput, stream);
    } else {
      equalPartition(partitionInput, stream);
    }
  } else {
    auto packedChunks = makeOwningPackedChunks(
        slices[0], slices[0].num_rows(), stream);
    VELOX_CHECK_EQ(packedChunks.size(), 1);
    sharedQueueManager()->enqueue(
        this->taskId(),
        0,
        std::move(packedChunks.front().data),
        packedChunks.front().rows);
  }

  activeNextRow_ = end;
  if (activeNextRow_ == tableRows) {
    // Every enqueue above synchronizes its stream before publication, so the
    // source owner can be released even if the queue check below blocks.
    releaseActiveDeviceSource();
  }
  updateBackpressure();
  if (blockingReason_ != exec::BlockingReason::kNotBlocked &&
      hasActiveDeviceSource()) {
    // The just-published bounded window filled the task queue. A partial TopN
    // or aggregation can own several GiB behind that 128-MiB queue future.
    // Move only its unconsumed tail to the shared host tier, then let the
    // ordinary Velox Driver/queue future provide backpressure.
    externalizeActiveSourceTail();
  }
}

exec::BlockingReason UcxPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }

  const bool needsFlushWork = hasActiveFlush() || pendingFlushReady_ ||
      (noMoreInput_ && !pendingInputs_.empty());
  if (needsFlushWork && !flushWorkspaceAdmission_.has_value()) {
    ContinueFuture workspaceFuture;
    const auto workspaceBytes = flushWorkspaceBytes();
    auto workspace = cudf_velox::tryAcquireDeviceMemoryWorkspace(
        pool(),
        this,
        workspaceBytes,
        cudf_velox::CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
        cudf_velox::DeviceMemoryWorkspacePriority::kOutput,
        &flushWorkspaceRequest_,
        &workspaceFuture);
    if (!workspace.has_value()) {
      *future = std::move(workspaceFuture);
      return exec::BlockingReason::kWaitForArbitration;
    }
    flushWorkspaceAdmission_.emplace(std::move(workspace.value()));
  }

  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr UcxPartitionedOutput::getOutput() {
  CudaCallDiagnosticScope callDiagnostic(fmt::format(
      "operator=UcxPartitionedOutput task={} method=getOutput", taskId()));
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_) {
    return nullptr;
  }
  // Driver calls isBlocked() before getOutput(). Keep this guard for direct
  // test drivers and to ensure an outstanding future is never overwritten.
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    return nullptr;
  }
  if (hasActiveFlush() || pendingFlushReady_ ||
      (noMoreInput_ && !pendingInputs_.empty())) {
    VELOX_CHECK(
        flushWorkspaceAdmission_.has_value(),
        "Partitioned output flush work requires workspace admission");
    SCOPE_EXIT {
      flushWorkspaceAdmission_.reset();
    };
    std::lock_guard<std::mutex> flushLock(partitionedOutputFlushMutex());
    flushPending();
  }
  // A final work unit may have completed but filled the queue. Defer EOS until
  // its future has been moved by isBlocked() and resumed by the Driver.
  if (noMoreInput_ && !hasActiveFlush() && pendingInputs_.empty() &&
      blockingReason_ == exec::BlockingReason::kNotBlocked) {
    if (localDeviceRootOutput_) {
      LocalDeviceOutputQueueManager::getInstanceRef()->noMoreData(
          this->taskId());
    }
    // The task still owns an ordinary UCX output-buffer manager. Close its
    // empty queue as well so task cleanup and output stats retain their normal
    // lifecycle.
    sharedQueueManager()->noMoreData(this->taskId());
    finished_ = true;
  }
  return nullptr;
}

bool UcxPartitionedOutput::isFinished() {
  return finished_;
}

std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
UcxPartitionedOutput::sharedQueueManager() {
  auto shared_queueManager = queueManager_.lock();
  VELOX_CHECK_NOT_NULL(
      shared_queueManager, "OutputQueueManager was already destructed");
  return shared_queueManager;
}

void UcxPartitionedOutput::initPartitionKeys(
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode) {
  // Following Logic copied direcly from CudLocalPartition (!)

  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  spec_ = planNode->partitionFunctionSpec().toString();

  if (auto* rangeFunctionSpec = dynamic_cast<const RangePartitionFunctionSpec*>(
          &planNode->partitionFunctionSpec())) {
    partitionKeyIndices_ = rangeFunctionSpec->keyChannels();
    rangeBoundsJson_ = rangeFunctionSpec->boundsJson();
    VELOX_CHECK(
        !partitionKeyIndices_.empty() && !rangeBoundsJson_.empty(),
        "RANGE_PID requires both keys and Spark boundaries");
    return;
  }

  // Only parse keys if it's a hash function
  if (spec_.find("HASH(") != std::string::npos) {
    // Extract keys between HASH( and )
    size_t start = spec_.find("HASH(") + 5;
    size_t end = spec_.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec_.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
  }
}

void UcxPartitionedOutput::hashPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  const auto maxRows = hashPartitionInputBatchRows_;
  if (maxRows > 0 && tableView.num_rows() > maxRows) {
    VLOG(2) << "UcxPartitionedOutput pre-slicing hash input task=" << taskId()
            << " rows=" << tableView.num_rows()
            << " maxRowsPerCall=" << maxRows;

    struct PartitionedChunk {
      std::unique_ptr<cudf::table> table;
      std::vector<cudf::size_type> offsets;
    };
    auto queueManager = sharedQueueManager();

    // Bound the temporary residency of the safe hash path.  Keeping every
    // 500K-row partitioned chunk for the full input, followed by all 32
    // recombined destinations and all packed buffers, amplified a wide Q10
    // customer batch by several times before output-buffer backpressure could
    // run.  Work on one output-sized row window and one destination at a time
    // instead.  The source vector remains alive, but all hash, concatenate,
    // and pack temporaries for a window are released before the next window.
    const auto configuredWindowRows = hashPartitionWindowRows_ > 0
        ? hashPartitionWindowRows_
        : (targetRowsPerChunk_ > 0 ? targetRowsPerChunk_ : maxRows);
    const auto rowsPerWindow = static_cast<cudf::size_type>(
        std::max<int64_t>(maxRows, configuredWindowRows));

    // When the safety call size and residency window are identical, every
    // window contains exactly one partitioned table.  Use contiguous_split's
    // single bulk operation instead of packing 32 destinations one at a time.
    // Besides avoiding needless concatenate bookkeeping, this changes Q10
    // from one stream synchronization per destination to one per 8M-row
    // window while preserving the same hard peak-memory bound.  Q21 uses a
    // larger recombination window than its 500K call size and therefore keeps
    // the multi-chunk path below.
    if (rowsPerWindow == maxRows) {
      std::vector<cudf::size_type> partitionKeyIndices;
      partitionKeyIndices.reserve(partitionKeyIndices_.size());
      for (const auto& idx : partitionKeyIndices_) {
        partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
      }
      for (cudf::size_type start = 0; start < tableView.num_rows();
           start += rowsPerWindow) {
        const auto end = std::min<cudf::size_type>(
            tableView.num_rows(), start + rowsPerWindow);
        auto slices = cudf::slice(tableView, {start, end}, stream);
        VELOX_CHECK_EQ(slices.size(), 1);
        auto [partitionedTable, partitionOffsets] = [&]() {
          std::lock_guard<std::mutex> lock(
              cudf_velox::cudfHashPartitionMutex());
          auto result = cudf::hash_partition(
              slices[0],
              partitionKeyIndices,
              numPartitions_,
              cudf::hash_id::HASH_MURMUR3,
              cudf::DEFAULT_HASH_SEED,
              stream);
          stream.synchronize();
          return result;
        }();
        VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
        VELOX_CHECK_EQ(partitionOffsets.front(), 0);
        partitionOffsets.erase(partitionOffsets.begin());
        partitionOffsets.pop_back();
        splitAndEnqueue(
            partitionedTable->view(), std::move(partitionOffsets), stream);
      }
      return;
    }

    for (cudf::size_type windowStart = 0; windowStart < tableView.num_rows();
         windowStart += rowsPerWindow) {
      const auto windowEnd = std::min<cudf::size_type>(
          tableView.num_rows(), windowStart + rowsPerWindow);
      std::vector<PartitionedChunk> chunks;
      chunks.reserve((windowEnd - windowStart + maxRows - 1) / maxRows);

      for (cudf::size_type start = windowStart; start < windowEnd;) {
        const auto end = std::min<cudf::size_type>(
            windowEnd, start + static_cast<cudf::size_type>(maxRows));
        auto slices = cudf::slice(tableView, {start, end}, stream);
        VELOX_CHECK_EQ(slices.size(), 1);

        std::vector<cudf::size_type> partitionKeyIndices;
        partitionKeyIndices.reserve(partitionKeyIndices_.size());
        for (const auto& idx : partitionKeyIndices_) {
          partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
        }
        auto [partitionedTable, partitionOffsets] = [&]() {
          std::lock_guard<std::mutex> lock(
              cudf_velox::cudfHashPartitionMutex());
          auto result = cudf::hash_partition(
              slices[0],
              partitionKeyIndices,
              numPartitions_,
              cudf::hash_id::HASH_MURMUR3,
              cudf::DEFAULT_HASH_SEED,
              stream);
          stream.synchronize();
          return result;
        }();
        VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
        VELOX_CHECK_EQ(partitionOffsets.front(), 0);
        chunks.push_back(
            {std::move(partitionedTable), std::move(partitionOffsets)});
        start = end;
      }

      for (int destination = 0; destination < numPartitions_; ++destination) {
        std::vector<cudf::table_view> destinationViews;
        destinationViews.reserve(chunks.size());
        for (const auto& chunk : chunks) {
          const auto begin = chunk.offsets[destination];
          const auto end = chunk.offsets[destination + 1];
          if (begin == end) {
            continue;
          }
          auto slices = cudf::slice(chunk.table->view(), {begin, end}, stream);
          VELOX_CHECK_EQ(slices.size(), 1);
          destinationViews.push_back(slices[0]);
        }
        if (destinationViews.empty()) {
          continue;
        }

        std::unique_ptr<cudf::table> combinedOwner;
        cudf::table_view destinationView;
        if (destinationViews.size() == 1) {
          destinationView = destinationViews.front();
        } else {
          combinedOwner = cudf::concatenate(
              destinationViews,
              stream,
              cudf::get_current_device_resource_ref());
          destinationView = combinedOwner->view();
        }

        const auto rowsPerMessage = std::max<cudf::size_type>(
            1,
            targetRowsPerChunk_ > 0
                ? std::min<cudf::size_type>(
                      destinationView.num_rows(),
                      static_cast<cudf::size_type>(targetRowsPerChunk_))
                : destinationView.num_rows());
        auto packedPartitions = makeOwningPackedChunks(
            destinationView, rowsPerMessage, stream);
        for (auto& packed : packedPartitions) {
          queueManager->enqueue(
              this->taskId(), destination, std::move(packed.data), packed.rows);
        }
      }
    }
    return;
  }

  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Hashing and partitioning into " << numPartitions_ << " chunks";

  // Use cudf hash partitioning
  std::vector<cudf::size_type> partitionKeyIndices;
  for (const auto& idx : partitionKeyIndices_) {
    partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
  }

  auto [partitionedTable, partitionOffsets] = [&]() {
    std::lock_guard<std::mutex> lock(cudf_velox::cudfHashPartitionMutex());
    auto result = cudf::hash_partition(
        tableView,
        partitionKeyIndices,
        numPartitions_,
        cudf::hash_id::HASH_MURMUR3,
        cudf::DEFAULT_HASH_SEED,
        stream);
    stream.synchronize();
    return result;
  }();

  VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
  VELOX_CHECK_EQ(partitionOffsets[0], 0);

  // Erase first element since it's always 0 and we don't need it.
  partitionOffsets.erase(partitionOffsets.begin());
  partitionOffsets.pop_back();

  splitAndEnqueue(partitionedTable->view(), partitionOffsets, stream);
}

void UcxPartitionedOutput::rangePartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(!rangeBoundsJson_.empty(), "RANGE_PID descriptor is missing");
  VELOX_CHECK(!partitionKeyIndices_.empty(), "RANGE_PID keys are missing");

  if (!rangeBoundaries_) {
    auto boundaryVector = buildRangeBoundaryVector(
        rangeBoundsJson_,
        outputType_,
        partitionKeyIndices_,
        pool(),
        rangeOrders_,
        rangeNullOrders_,
        &rangeSplitEqualKeys_);
    rangeBoundaries_ = cudf_velox::with_arrow::toCudfTable(
        boundaryVector,
        pool(),
        stream,
        cudf::get_current_device_resource_ref());
    VELOX_CHECK_LT(
        rangeBoundaries_->num_rows(),
        numPartitions_,
        "RANGE_PID boundary count must be smaller than requested partitions");
  }

  std::vector<cudf::size_type> rangeKeyIndices;
  rangeKeyIndices.reserve(partitionKeyIndices_.size());
  for (const auto index : partitionKeyIndices_) {
    rangeKeyIndices.push_back(static_cast<cudf::size_type>(index));
  }
  const auto keyTable = tableView.select(rangeKeyIndices);
  auto partitionIds = makeRangePartitionIds(
      rangeBoundaries_->view(),
      keyTable,
      rangeOrders_,
      rangeNullOrders_,
      rangeSplitEqualKeys_,
      static_cast<int32_t>(rangeSplitCounter_),
      stream,
      cudf::get_current_device_resource_ref());
  if (rangeSplitEqualKeys_) {
    rangeSplitCounter_ =
        (rangeSplitCounter_ % numPartitions_ +
         static_cast<size_t>(tableView.num_rows()) % numPartitions_) %
        numPartitions_;
  }
  VELOX_CHECK(
      partitionIds->size() == tableView.num_rows(),
      "RANGE_PID must produce exactly one id per input row");
  VELOX_CHECK(
      partitionIds->type().id() == cudf::type_id::INT32,
      "RANGE_PID must produce an INT32 partition map");

  // libcudf::partition groups by the explicit INT32 map. No hash function is
  // involved; the returned table is routed directly to destination queues.
  auto [partitionedTable, partitionOffsets] = cudf::partition(
      tableView,
      partitionIds->view(),
      numPartitions_,
      stream,
      cudf::get_current_device_resource_ref());
  normalizePartitionOffsets(partitionOffsets, numPartitions_);
  splitAndEnqueue(partitionedTable->view(), partitionOffsets, stream);
}

void UcxPartitionedOutput::equalPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Splitting into " << numPartitions_ << " chunks";
  std::vector<cudf::size_type> offsets;
  cudf::size_type size = tableView.num_rows();
  for (int i = 1; i < numPartitions_; ++i) {
    cudf::size_type idx = size * i / numPartitions_;
    offsets.push_back(idx);
  }
  splitAndEnqueue(tableView, offsets, stream);
}

void UcxPartitionedOutput::splitAndEnqueue(
    cudf::table_view tableView,
    std::vector<cudf::size_type> offsets,
    rmm::cuda_stream_view stream) {
  auto contiguousTables = cudf::contiguous_split(
      tableView, offsets, stream, cudf::get_current_device_resource_ref());

  // Synchronize the stream to ensure CUDA operations complete before enqueuing.
  // UCXX/UCX is not stream-aware, so without syncing, data could be sent before
  // the GPU kernels have finished writing to the buffers.
  stream.synchronize();

  VELOX_CHECK_EQ(
      offsets.size() + 1, numPartitions_, "mismatch in numPartitions_");
  auto queueManager = sharedQueueManager();
  for (int i = 0; i < numPartitions_; ++i) {
    auto const& partitionTable = contiguousTables[i];
    const auto partitionRows = partitionTable.table.num_rows();
    if (partitionRows == 0) {
      // Skip empty partitions.
      continue;
    }

    const auto partitionBytes = partitionTable.data.gpu_data->size();
    const auto rowsPerChunk = rowsPerUcxChunk(
        partitionRows,
        partitionBytes,
        targetRowsPerChunk_,
        targetBytesPerChunk_);
    if (rowsPerChunk < partitionRows) {
      VLOG(2) << "UcxPartitionedOutput chunking task=" << taskId()
              << " destination=" << i << " rows=" << partitionRows
              << " bytes=" << partitionBytes << " rowsPerChunk=" << rowsPerChunk
              << " targetRowsPerChunk=" << targetRowsPerChunk_
              << " targetBytesPerChunk=" << targetBytesPerChunk_;
      // Do not feed a table_view backed by the first contiguous_split's packed
      // allocation directly into a second contiguous_split. Under sustained
      // HASH exchange this produced pages whose metadata initially unpacked
      // correctly but whose STRING offsets were later read from recycled
      // storage (Job 144 source_2/source_22). Normalize the oversized
      // destination into an ordinary owning table before page chunking. Both
      // copies are ordered on `stream`, and makeOwningPackedChunks synchronizes
      // before this owner is released.
      auto owningPartition = std::make_unique<cudf::table>(
          partitionTable.table,
          stream,
          cudf::get_current_device_resource_ref());
      auto packedChunks = makeOwningPackedChunks(
          owningPartition->view(), rowsPerChunk, stream);
      for (auto& packedChunk : packedChunks) {
        queueManager->enqueue(
            this->taskId(),
            i,
            std::move(packedChunk.data),
            packedChunk.rows);
      }
      continue;
    }

    auto packedColsPtr =
        takePackedColumns(std::move(contiguousTables[i].data), stream);

    // enqueue partition data on Ucx Output Buffer
    queueManager->enqueue(
        this->taskId(),
        i,
        std::move(packedColsPtr),
        partitionTable.table.num_rows());
  }
}

} // namespace facebook::velox::ucx_exchange
