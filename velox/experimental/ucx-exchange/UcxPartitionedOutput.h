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

#include <optional>

#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

namespace facebook::velox::ucx_exchange {

/// This is the cudf equivalent of the PartitionedOutput operator for cudf.
/// Instead of serializing and segmenting the partitioned data into an
/// OutputBuffer, the UcxPartitionedOutput operator transfers entire
/// cudf::packed_columns corresponding to CudfVectors to other workers.
class UcxPartitionedOutput : public exec::Operator,
                             public cudf_velox::NvtxHelper {
 public:
  // Default minimum rows to accumulate before flushing. Matches HTTP
  // PartitionedOutput's ~10,000 row target. Overridable via
  // QueryConfig::kUcxPartitionedOutputBatchRows.
  static constexpr int64_t kDefaultTargetRowsPerChunk = 10'000;

  UcxPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush,
      std::optional<uint32_t> numDriversForTest = std::nullopt);

  void addInput(RowVectorPtr input) override;

  /// Always returns nullptr. The action is to further process
  /// unprocessed input. If all input has been processed, 'this' is in
  /// a non-blocked state, otherwise blocked.
  RowVectorPtr getOutput() override;

  /// Refuses a new input while an ordered single-destination batch is being
  /// emitted incrementally after backpressure.
  bool needsInput() const override {
    return !hasPendingSingleChunks();
  }

  /// Returns a consumer future when the shared output queue is over capacity.
  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  // The operaor is finished when the queue manager say the queues have all been
  // drained ?
  bool isFinished() override;

 private:
  std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
  sharedQueueManager();

  // Heuristic method to derive the partition keys from the PartitionNode
  // specification.
  void initPartitionKeys(
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  // Partitions the cudf table view using the partition keys and a hash
  // function using the given stream.
  void hashPartition(cudf::table_view tableView, rmm::cuda_stream_view stream);

  // Splits the cudf table view into equal sizes. This is used when
  // RoundRobin partitioning is requested but round robin on a
  // row-by-row basis is not meaningful for UCX exchange.
  void equalPartition(cudf::table_view tableView, rmm::cuda_stream_view stream);

  // Splits the table along the given offsets and enqueues each offset
  // to the corresponding partition, i.e. first split to the partition 0,
  // second split to partition 1 etc.
  void splitAndEnqueue(
      cudf::table_view tableView,
      std::vector<cudf::size_type> offsets,
      rmm::cuda_stream_view stream);

  /// Starts resumable emission of a one-destination ordered table. The table
  /// is owned either by pendingInputs_ or by 'mergedTable'.
  void startSingleChunkEmission(
      std::unique_ptr<cudf::table> mergedTable,
      cudf::size_type tableRows,
      rmm::cuda_stream_view stream);

  /// Emits ordered chunks until the table is exhausted or the output queue
  /// reports backpressure.
  void emitPendingSingleChunks();

  bool hasPendingSingleChunks() const {
    return pendingSingleNextRow_ < pendingSingleNumRows_;
  }

  void resetPendingSingleEmission();

  const std::weak_ptr<UcxOutputQueueManager> queueManager_;
  std::vector<column_index_t> partitionKeyIndices_;
  const size_t numPartitions_;

  const int pipelineId_;
  const int driverId_;

  exec::BlockingReason blockingReason_{exec::BlockingReason::kNotBlocked};
  ContinueFuture future_;

  bool finished_{false};
  std::string spec_;

  // Used for switching columns when column order differs between input and
  // output.
  std::vector<uint32_t> remap_;

  /// Concatenates pending inputs and partitions/enqueues the merged result.
  void flushPending();

  /// Accumulated CudfVectors awaiting flush.
  std::vector<cudf_velox::CudfVectorPtr> pendingInputs_;
  /// Total rows across pendingInputs_.
  int64_t pendingRows_{0};
  /// Configured row threshold for flushing (from QueryConfig).
  const int64_t targetRowsPerChunk_;

  /// True only for the explicit MERGE_SINGLE contract. Ordinary SINGLE
  /// exchanges preserve their existing whole-batch behavior.
  const bool mergeSingleEnabled_;

  /// Resumable MERGE_SINGLE sender state. A single input remains in
  /// pendingInputs_; a concatenated input is owned here instead.
  std::unique_ptr<cudf::table> pendingSingleTable_;
  std::optional<rmm::cuda_stream_view> pendingSingleStream_;
  cudf::size_type pendingSingleNextRow_{0};
  cudf::size_type pendingSingleNumRows_{0};
};

} // namespace facebook::velox::ucx_exchange
