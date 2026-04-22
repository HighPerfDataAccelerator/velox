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
#pragma once

#include "velox/exec/Operator.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/experimental/cudf/exchange/GpuPartitionedOutputNode.h"
#include "velox/experimental/cudf/exchange/GpuSerializedPage.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

/// GPU-accelerated partitioned output operator. Receives CudfVector input,
/// partitions on GPU using cudf::partition(), wraps each partition slice as
/// a GpuSerializedPage (no D2H copy), and publishes to OutputBufferManager.
///
/// Supports kPartitioned (hash), kBroadcast, and kArbitrary output kinds.
class GpuPartitionedOutput : public exec::Operator {
 public:
  /// Overload that accepts the stock core::PartitionedOutputNode so that a
  /// cuDF OperatorAdapter can replace the default CPU PartitionedOutput
  /// operator without a separate GPU-specific PlanNode.
  GpuPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  GpuPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const GpuPartitionedOutputNode>& planNode);

  void addInput(RowVectorPtr input) override;

  /// Always returns nullptr. Processing happens in addInput/getOutput cycle;
  /// output is published directly to OutputBufferManager.
  RowVectorPtr getOutput() override;

  bool needsInput() const override {
    return !finished_ && blockingReason_ == exec::BlockingReason::kNotBlocked;
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override {
    if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
      *future = std::move(future_);
      auto reason = blockingReason_;
      blockingReason_ = exec::BlockingReason::kNotBlocked;
      return reason;
    }
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

  void close() override {
    exec::Operator::close();
  }

 private:
  /// Partitions the input CudfVector on GPU and enqueues each partition slice
  /// as a GpuSerializedPage to OutputBufferManager.
  void partitionAndEnqueue(std::shared_ptr<CudfVector> cudfVec);

  /// Enqueues a GpuSerializedPage for a single destination. Returns true if
  /// the enqueue caused backpressure (buffer full).
  bool enqueuePartition(
      int destination,
      std::shared_ptr<CudfVector> partitionData);

  /// Broadcasts the input to all destinations.
  void broadcastInput(std::shared_ptr<CudfVector> cudfVec);

  /// Sends the input to an arbitrary single destination.
  void arbitraryInput(std::shared_ptr<CudfVector> cudfVec);

  const GpuPartitionedOutputNode::Kind kind_;
  const int numDestinations_;
  const std::vector<column_index_t> keyChannels_;
  std::unique_ptr<core::PartitionFunction> partitionFunction_;

  const std::weak_ptr<exec::OutputBufferManager> bufferManager_;
  const std::function<void()> bufferReleaseFn_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};

  /// Round-robin counter for kArbitrary distribution.
  int nextArbitraryDestination_{0};
};

} // namespace facebook::velox::cudf_velox
