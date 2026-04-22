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

namespace {
GpuPartitionedOutputNode::Kind coreKindToGpu(core::PartitionedOutputNode::Kind k) {
  switch (k) {
    case core::PartitionedOutputNode::Kind::kPartitioned:
      return GpuPartitionedOutputNode::Kind::kPartitioned;
    case core::PartitionedOutputNode::Kind::kBroadcast:
      return GpuPartitionedOutputNode::Kind::kBroadcast;
    case core::PartitionedOutputNode::Kind::kArbitrary:
      return GpuPartitionedOutputNode::Kind::kArbitrary;
  }
  VELOX_UNREACHABLE();
}
} // namespace

GpuPartitionedOutput::GpuPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode)
    : exec::Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "GpuPartitionedOutput"),
      // core::PartitionedOutputNode::single() yields kPartitioned + N=1 with
      // GatherPartitionFunctionSpec. From the data-flow perspective that is
      // identical to kArbitrary: one destination, no real partitioning. Route
      // via arbitraryInput() so we don't need a partitionFunction_.
      kind_(
          (planNode->isPartitioned() && planNode->numPartitions() == 1)
              ? GpuPartitionedOutputNode::Kind::kArbitrary
              : coreKindToGpu(planNode->kind())),
      numDestinations_(planNode->numPartitions()),
      keyChannels_(
          (planNode->isPartitioned() && planNode->numPartitions() > 1)
              ? exec::toChannels(planNode->inputType(), planNode->keys())
              : std::vector<column_index_t>{}),
      partitionFunction_(
          (planNode->isPartitioned() && planNode->numPartitions() > 1)
              ? planNode->partitionFunctionSpec().create(
                    numDestinations_,
                    /*localExchange=*/false)
              : nullptr),
      bufferManager_(exec::OutputBufferManager::getInstanceRef()),
      bufferReleaseFn_([task = operatorCtx_->task()]() {}) {
  VELOX_USER_CHECK_GT(numDestinations_, 0, "numDestinations must be positive");
}

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
  auto tableView = cudfVec->getTableView();
  auto stream = cudfVec->stream();

  // GPU-native partitioning. Invoking the stock CPU partition function here
  // (as we used to) called childAt() on CudfVector, which FATALs because
  // CudfVectors have childrenSize_=0 (children live on GPU). This mirrors
  // CudfLocalPartition::addInput.
  auto [partitioned, offsets] = [&]() {
    if (!keyChannels_.empty()) {
      // HASH: partition on keyChannels using cudf::hash_partition.
      std::vector<cudf::size_type> partitionKeyIndices;
      partitionKeyIndices.reserve(keyChannels_.size());
      for (const auto& ch : keyChannels_) {
        partitionKeyIndices.push_back(static_cast<cudf::size_type>(ch));
      }
      return cudf::hash_partition(
          tableView,
          partitionKeyIndices,
          numDestinations_,
          cudf::hash_id::HASH_MURMUR3,
          cudf::DEFAULT_HASH_SEED,
          stream);
    }
    // No keys -> RoundRobin. start_partition=0 yields stable assignment.
    return cudf::round_robin_partition(
        tableView, numDestinations_, /*start_partition=*/0, stream);
  }();
  VELOX_CHECK_EQ(
      offsets.size(),
      static_cast<size_t>(numDestinations_) + 1,
      "cudf partition must return numPartitions+1 offsets");

  // Sync before slicing (cudf partition APIs are async on `stream`).
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
