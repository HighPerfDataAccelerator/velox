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

#include "velox/experimental/cudf/exchange/GpuPartitionedOutput.h"

#include "velox/exec/Task.h"
// TODO(mpp): GpuGuard disabled for MPP prototype.
// #include "velox/experimental/cudf/exec/GpuGuard.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/utilities/error.hpp>

namespace facebook::velox::cudf_velox {

GpuPartitionedOutput::GpuPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const GpuPartitionedOutputNode>& planNode)
    : exec::Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "GpuPartitionedOutput"),
      kind_(planNode->kind()),
      numDestinations_(planNode->numPartitions()),
      keyChannels_(
          planNode->isPartitioned()
              ? exec::toChannels(planNode->inputType(), planNode->keys())
              : std::vector<column_index_t>{}),
      partitionFunction_(
          planNode->isPartitioned() && planNode->partitionFunctionSpec()
              ? planNode->partitionFunctionSpec()->create(
                    numDestinations_,
                    /*localExchange=*/false)
              : nullptr),
      bufferManager_(exec::OutputBufferManager::getInstanceRef()),
      // Hold a reference to the task to prevent it from being deleted while
      // output buffers are being accessed externally.
      bufferReleaseFn_([task = operatorCtx_->task()]() {}) {
  VELOX_USER_CHECK_GT(numDestinations_, 0, "numDestinations must be positive");
}

void GpuPartitionedOutput::addInput(RowVectorPtr input) {
  auto cudfVec = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVec, "GpuPartitionedOutput input must be CudfVector");

  if (cudfVec->size() == 0) {
    return;
  }

  switch (kind_) {
    case GpuPartitionedOutputNode::Kind::kPartitioned:
      partitionAndEnqueue(std::move(cudfVec));
      break;
    case GpuPartitionedOutputNode::Kind::kBroadcast:
      broadcastInput(std::move(cudfVec));
      break;
    case GpuPartitionedOutputNode::Kind::kArbitrary:
      arbitraryInput(std::move(cudfVec));
      break;
  }
}

RowVectorPtr GpuPartitionedOutput::getOutput() {
  if (finished_) {
    return nullptr;
  }

  if (noMoreInput_) {
    auto bufferManager = bufferManager_.lock();
    VELOX_CHECK_NOT_NULL(
        bufferManager, "OutputBufferManager was already destructed");
    bufferManager->noMoreData(operatorCtx_->task()->taskId());
    finished_ = true;
  }

  return nullptr;
}

void GpuPartitionedOutput::partitionAndEnqueue(
    std::shared_ptr<CudfVector> cudfVec) {
  VELOX_CHECK_NOT_NULL(
      partitionFunction_,
      "Partition function required for kPartitioned output");

  auto tableView = cudfVec->getTableView();
  auto stream = cudfVec->stream();
  auto numRows = cudfVec->size();

  // Use the Velox partition function to compute partition IDs on CPU.
  // The partition function works on RowVector (CudfVector is a subclass).
  std::vector<uint32_t> partitions(numRows);
  auto singlePartition = partitionFunction_->partition(*cudfVec, partitions);

  if (singlePartition.has_value()) {
    // All rows go to the same destination -- no GPU partitioning needed.
    enqueuePartition(singlePartition.value(), std::move(cudfVec));
    return;
  }

  // Build a cudf column from the partition IDs to drive cudf::partition().
  // We need to get partition IDs to GPU for cudf::partition to reorder rows.
  //
  // Allocate device column for partition IDs.
  auto pidColDev = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      numRows,
      cudf::mask_state::UNALLOCATED,
      stream);

  // Copy partition IDs host->device.
  static_assert(sizeof(uint32_t) == sizeof(int32_t));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      pidColDev->mutable_view().head<int32_t>(),
      reinterpret_cast<const int32_t*>(partitions.data()),
      numRows * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));

  // cudf::partition reorders rows so that rows with the same partition are
  // contiguous. offsets[i] is the start index of partition i.
  auto [partitioned, offsets] = cudf::partition(
      tableView,
      pidColDev->view(),
      static_cast<cudf::size_type>(numDestinations_),
      stream);
  VELOX_CHECK_EQ(
      offsets.size(),
      static_cast<size_t>(numDestinations_) + 1,
      "cudf::partition must return numPartitions+1 offsets");

  // Sync: cudf::partition reads from pidColDev and input tableView
  // asynchronously. We must synchronize before pidColDev goes out of scope
  // and before slicing the partitioned table.
  stream.synchronize();

  // TODO(mpp): GPU lock is disabled for MPP prototype. Re-enable with proper
  // region management once the architecture is validated.
  // endGpuRegion();

  // Slice the partitioned table into per-destination CudfVectors and enqueue.
  for (int dest = 0; dest < numDestinations_; ++dest) {
    auto startRow = static_cast<cudf::size_type>(offsets[dest]);
    auto endRow = static_cast<cudf::size_type>(offsets[dest + 1]);
    auto sliceRows = endRow - startRow;

    if (sliceRows == 0) {
      continue;
    }

    // cudf::slice returns table_views referencing the partitioned table's
    // device memory. We materialize each slice into an owned table so that
    // each GpuSerializedPage is independently valid.
    auto slicedViews = cudf::slice(
        partitioned->view(), {startRow, endRow}, stream);
    VELOX_CHECK_EQ(slicedViews.size(), 1);

    auto sliceTable =
        std::make_unique<cudf::table>(slicedViews[0], stream);
    // Sync: cudf::table ctor does async deep copy from slice view.
    stream.synchronize();

    auto sliceVec = std::make_shared<CudfVector>(
        pool(),
        outputType_,
        sliceRows,
        std::move(sliceTable),
        stream);

    if (enqueuePartition(dest, std::move(sliceVec))) {
      // Backpressure: buffer is full. Stop processing further partitions.
      // The framework will re-call us after the consumer drains.
      return;
    }
  }
}

bool GpuPartitionedOutput::enqueuePartition(
    int destination,
    std::shared_ptr<CudfVector> partitionData) {
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "OutputBufferManager was already destructed");

  auto page = std::make_unique<GpuSerializedPage>(std::move(partitionData));

  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(page->size(), page->numRows().value_or(0));
  }

  bool blocked = bufferManager->enqueue(
      operatorCtx_->task()->taskId(),
      destination,
      std::move(page),
      &future_);

  if (blocked) {
    blockingReason_ = exec::BlockingReason::kWaitForConsumer;
    return true;
  }
  return false;
}

void GpuPartitionedOutput::broadcastInput(
    std::shared_ptr<CudfVector> cudfVec) {
  // Broadcast: send the same data to all destinations. The shared_ptr
  // inside GpuSerializedPage ensures the GPU memory stays alive until
  // all consumers have consumed their pages.
  //
  // TODO(mpp): GPU lock disabled for prototype.
  // endGpuRegion();

  for (int dest = 0; dest < numDestinations_; ++dest) {
    if (enqueuePartition(dest, cudfVec)) {
      // Backpressure from one destination.
      return;
    }
  }
}

void GpuPartitionedOutput::arbitraryInput(
    std::shared_ptr<CudfVector> cudfVec) {
  // TODO(mpp): GPU lock disabled for prototype.
  // endGpuRegion();

  // Round-robin to a single destination.
  int dest = nextArbitraryDestination_;
  nextArbitraryDestination_ = (nextArbitraryDestination_ + 1) % numDestinations_;
  enqueuePartition(dest, std::move(cudfVec));
}

} // namespace facebook::velox::cudf_velox
