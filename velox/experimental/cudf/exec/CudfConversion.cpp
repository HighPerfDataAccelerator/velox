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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

namespace facebook::velox::cudf_velox {

namespace {
// Concatenate multiple RowVectors into a single RowVector.
// Copied from AggregationFuzzer.cpp.
RowVectorPtr mergeRowVectors(
    const std::vector<RowVectorPtr>& results,
    velox::memory::MemoryPool* pool) {
  VELOX_NVTX_FUNC_RANGE();
  if (results.size() == 1) {
    return results[0];
  }
  vector_size_t totalCount = 0;
  for (const auto& result : results) {
    totalCount += result->size();
  }
  auto copy =
      BaseVector::create<RowVector>(results[0]->type(), totalCount, pool);
  auto copyCount = 0;
  for (const auto& result : results) {
    copy->copy(result.get(), copyCount, 0, result->size());
    copyCount += result->size();
  }
  return copy;
}

cudf::size_type preferredGpuBatchSizeRows(
    const facebook::velox::core::QueryConfig& queryConfig) {
  constexpr cudf::size_type kDefaultGpuBatchSizeRows = 100000;
  const auto batchSize = queryConfig.get<int32_t>(
      CudfFromVelox::kGpuBatchSizeRows, kDefaultGpuBatchSizeRows);
  VELOX_CHECK_GT(batchSize, 0, "velox.cudf.gpu_batch_size_rows must be > 0");
  VELOX_CHECK_LE(
      batchSize,
      std::numeric_limits<vector_size_t>::max(),
      "velox.cudf.gpu_batch_size_rows must be <= max(vector_size_t)");
  return batchSize;
}
} // namespace

CudfFromVelox::CudfFromVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfFromVelox",
          nvtx3::rgb{255, 140, 0}, // Orange
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt),
      timestampTimeZone_(driverCtx->queryConfig().get<std::string>(
          facebook::velox::core::QueryConfig::kSessionTimezone)) {
  auto parentId = planNodeId.substr(0, planNodeId.find("-from-velox"));
  stats_.withWLock([&](auto& stats) {
    stats.setStatSplitter(
        [parentId = std::move(parentId)](const auto& combinedStats) {
          auto result = combinedStats;
          result.planNodeId = parentId;
          return std::vector<exec::OperatorStats>{std::move(result)};
        });
  });
}

vector_size_t CudfFromVelox::preferredBatchRows() const {
  return preferredGpuBatchSizeRows(operatorCtx_->driverCtx()->queryConfig());
}

uint64_t CudfFromVelox::maxBatchBytes() const {
  constexpr uint64_t kDefaultMaxBatchBytes = 256ULL << 20;
  const auto configured =
      operatorCtx_->driverCtx()->queryConfig().get<uint64_t>(
          kMaxBatchBytes, kDefaultMaxBatchBytes);
  VELOX_CHECK_GT(
      configured, 0, "velox.cudf.from_velox.max_batch_bytes must be > 0");
  return configured;
}

void CudfFromVelox::doAddInput(RowVectorPtr input) {
  if (input->size() > 0) {
    // Materialize lazy vectors
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    input->loadedVector();

    // Accumulate inputs
    inputs_.push_back(input);
    currentOutputSize_ += input->size();
    currentOutputBytes_ += input->estimateFlatSize();
  }
}

RowVectorPtr CudfFromVelox::doGetOutput() {
  const auto targetOutputSize = preferredBatchRows();
  const auto targetOutputBytes = maxBatchBytes();

  finished_ = noMoreInput_ && inputs_.empty();

  if (finished_ or
      (currentOutputSize_ < targetOutputSize &&
       currentOutputBytes_ < targetOutputBytes && !noMoreInput_) or
      inputs_.empty()) {
    return nullptr;
  }

  // Some iterator-backed inputs (for example Gluten's GPU columnar table
  // cache) already produce a CudfVector.  The source is conservatively
  // classified as CPU, so ToCudf may still insert this conversion boundary.
  // Do not feed that vector through mergeRowVectors: CudfVector deliberately
  // exposes no host RowVector children, and copying it as a regular RowVector
  // produces an invalid row whose type has columns but whose children list is
  // empty.  Preserve ordering and ownership by forwarding one device batch
  // at a time.
  if (std::dynamic_pointer_cast<CudfVector>(inputs_.front())) {
    auto input = std::move(inputs_.front());
    inputs_.erase(inputs_.begin());
    currentOutputSize_ -= input->size();
    currentOutputBytes_ -= input->estimateFlatSize();
    return input;
  }

  // Select a bounded prefix. The old implementation waited for a row target
  // and then merged every queued RowVector, which can turn byte-bounded
  // CudfToVelox slices back into a multi-GiB host allocation.
  std::vector<RowVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;
  uint64_t totalBytes = 0;
  auto const maxVectorSize = std::numeric_limits<vector_size_t>::max();

  for (const auto& input : inputs_) {
    const auto inputSize = static_cast<vector_size_t>(input->size());
    const auto inputBytes = input->estimateFlatSize();
    const bool wouldExceedRows = totalSize > 0 &&
        (totalSize >= targetOutputSize ||
         inputSize > targetOutputSize - totalSize ||
         inputSize > maxVectorSize - totalSize);
    const bool wouldExceedBytes = totalBytes > 0 &&
        (inputBytes > targetOutputBytes ||
         totalBytes > targetOutputBytes - inputBytes);
    if (wouldExceedRows || wouldExceedBytes) {
      break;
    }
    selectedInputs.push_back(input);
    totalSize += inputSize;
    totalBytes += inputBytes;
  }
  VELOX_CHECK(!selectedInputs.empty());

  // Combine selected RowVectors into a single RowVector
  auto input = mergeRowVectors(selectedInputs, inputs_[0]->pool());

  // Remove processed inputs
  inputs_.erase(inputs_.begin(), inputs_.begin() + selectedInputs.size());
  currentOutputSize_ -= totalSize;
  currentOutputBytes_ -= totalBytes;

  // Early return if no input
  if (input->size() == 0) {
    return nullptr;
  }

  // Get a stream from the global stream pool
  auto stream = cudfGlobalStreamPool().get_stream();

  // cuDF tables with zero columns cannot represent a row count, so we
  // create a CudfVector directly with an empty table, preserving the
  // logical row count. This mirrors the zero-column handling in
  // CudfToVelox::doGetOutput().
  if (input->childrenSize() == 0) {
    auto emptyTable = std::make_unique<cudf::table>();
    return std::make_shared<CudfVector>(
        input->pool(),
        outputType_,
        input->size(),
        std::move(emptyTable),
        stream);
  }

  // Convert RowVector to cudf table.  toCudfTable synchronizes the stream
  // internally before releasing Arrow host buffers, so no additional sync
  // is needed here.
  auto tbl = with_arrow::toCudfTable(
      input, input->pool(), stream, get_output_mr(), timestampTimeZone_);

  VELOX_CHECK_NOT_NULL(tbl);

  // cuDF zero-column tables do not have a row count, so preserve the
  // RowVector logical size for count/materialization paths.
  const auto size = tbl->num_columns() == 0 ? input->size() : tbl->num_rows();

  return std::make_shared<CudfVector>(
      input->pool(), outputType_, size, std::move(tbl), stream);
}

void CudfFromVelox::doClose() {
  // TODO(kn): Remove default stream after redesign of CudfFromVelox
  cudf::get_default_stream(cudf::allow_default_stream).synchronize();
  Operator::close();
  inputs_.clear();
  currentOutputSize_ = 0;
  currentOutputBytes_ = 0;
}

CudfToVelox::CudfToVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfToVelox",
          nvtx3::rgb{148, 0, 211}, // Purple
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt) {
  auto parentId = planNodeId.substr(0, planNodeId.find("-to-velox"));
  stats_.withWLock([&](auto& stats) {
    stats.setStatSplitter(
        [parentId = std::move(parentId)](const auto& combinedStats) {
          auto result = combinedStats;
          result.planNodeId = parentId;
          return std::vector<exec::OperatorStats>{std::move(result)};
        });
  });
}

bool CudfToVelox::isPassthroughMode() const {
  return operatorCtx_->driverCtx()->queryConfig().get<bool>(
      kPassthroughMode, true);
}

void CudfToVelox::doAddInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    inputs_.push_back(std::move(cudfInput));
  }
}

uint64_t CudfToVelox::maxBatchBytes() const {
  constexpr uint64_t kDefaultMaxBatchBytes = 256ULL << 20;
  // Leave substantial headroom below Arrow's signed 32-bit string-offset
  // ceiling. estimateFlatSize() is an aggregate estimate and nested columns
  // may not distribute bytes uniformly across rows.
  constexpr uint64_t kHardMaxBatchBytes = 1ULL << 30;
  const auto configured =
      operatorCtx_->driverCtx()->queryConfig().get<uint64_t>(
          kMaxBatchBytes, kDefaultMaxBatchBytes);
  VELOX_CHECK_GT(
      configured, 0, "velox.cudf.to_velox.max_batch_bytes must be > 0");
  return std::min(configured, kHardMaxBatchBytes);
}

vector_size_t CudfToVelox::nextBatchRows() const {
  VELOX_CHECK_NOT_NULL(cudfBuffer_);
  const auto totalRows = static_cast<vector_size_t>(cudfBuffer_->size());
  VELOX_CHECK_LT(cudfOffset_, totalRows);
  const auto remaining = totalRows - cudfOffset_;
  const auto flatBytes =
      std::max<uint64_t>(cudfBuffer_->estimateFlatSize(), totalRows);
  // Use a ceiling average so the byte bound is conservative.
  const auto averageRowBytes =
      std::max<uint64_t>(1, (flatBytes + totalRows - 1) / totalRows);
  const auto byteBoundRows = std::max<vector_size_t>(
      1, std::min<uint64_t>(remaining, maxBatchBytes() / averageRowBytes));
  if (isPassthroughMode()) {
    return byteBoundRows;
  }
  return std::min(
      byteBoundRows, outputBatchRows(std::optional<uint64_t>{averageRowBytes}));
}

// Slice on GPU first, then convert only the bounded view. Converting the
// complete input and slicing the RowVector afterwards can allocate tens of
// GiB on host and fails when any Arrow string data buffer exceeds INT32_MAX.
RowVectorPtr CudfToVelox::convertNextSliceToVelox() {
  VELOX_CHECK_NOT_NULL(cudfBuffer_);
  auto stream = cudfBuffer_->stream();
  const auto rows = nextBatchRows();
  const auto end = cudfOffset_ + rows;
  auto slices =
      cudf::slice(cudfBuffer_->getTableView(), {cudfOffset_, end}, stream);
  VELOX_CHECK_EQ(slices.size(), 1);
  auto output = with_arrow::toVeloxColumn(
      slices.front(), pool(), outputType_, "", stream, get_temp_mr());
  stream.synchronize();
  output->setType(outputType_);
  cudfOffset_ = end;
  if (cudfOffset_ >= cudfBuffer_->size()) {
    cudfBuffer_.reset();
    cudfOffset_ = 0;
  }
  return output;
}

RowVectorPtr CudfToVelox::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  if (outputType_->size() == 0) {
    // cuDF zero-column tables do not have a row count, so we sum the sizes
    // of all CudfVectors in the inputs_, to maintain the logical count.
    // This is necessary to ensure correct behavior for e.g. `count` operators.
    vector_size_t totalSize = 0;
    while (!inputs_.empty()) {
      totalSize += inputs_.front()->size();
      inputs_.pop_front();
    }
    finished_ = noMoreInput_ && inputs_.empty();
    if (totalSize == 0) {
      return nullptr;
    }
    return BaseVector::create<RowVector>(outputType_, totalSize, pool());
  }

  if (!cudfBuffer_) {
    if (inputs_.empty()) {
      finished_ = noMoreInput_;
      return nullptr;
    }
    cudfBuffer_ = std::move(inputs_.front());
    inputs_.pop_front();
    cudfOffset_ = 0;
  }

  auto output = convertNextSliceToVelox();
  finished_ = noMoreInput_ && inputs_.empty() && !cudfBuffer_;
  return output;
}

void CudfToVelox::doClose() {
  Operator::close();
  inputs_.clear();
  cudfBuffer_.reset();
}

} // namespace facebook::velox::cudf_velox
