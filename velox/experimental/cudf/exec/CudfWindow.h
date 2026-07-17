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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"

#include <cudf/io/parquet.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node);

/// Narrow cuDF Window implementation for:
/// - row_number() / rank() with the default UNBOUNDED PRECEDING to CURRENT ROW
///   frame.
/// - sum(field) over a full partition ROWS frame.
/// - sum(field) over a partitioned ordered ROWS UNBOUNDED PRECEDING to
///   CURRENT ROW frame.
/// - first(field) / first_value(field) over a partitioned ordered UNBOUNDED
///   PRECEDING to CURRENT ROW frame.
/// - count(non-null constant) over a full partition ROWS frame.
/// - sum(field) over a partitioned ordered RANGE UNBOUNDED PRECEDING to
///   CURRENT ROW frame.
class CudfWindow : public CudfOperatorBase {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  static void testingSetStreamingMemoryLimits(
      uint64_t activeRowsBytes,
      uint64_t replayChunkBytes);
  static void testingResetStreamingMemoryLimits();
  static uint64_t testingStreamingSpillWrites();
  static uint64_t testingStreamingSpillCleanups();

  bool needsInput() const override {
    return !noMoreInput_ && pendingOutput_ == nullptr &&
        (!boundedStreaming_ ||
         (deferredInput_ == nullptr && streamingReplay_ == nullptr));
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_ && pendingOutput_ == nullptr;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  std::unique_ptr<cudf::column> computeRowNumberColumn(
      cudf::table_view const& sortedInput,
      const TypePtr& expectedType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> computeRankColumn(
      cudf::table_view const& sortedInput,
      const TypePtr& expectedType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> computeFullPartitionSumColumn(
      cudf::table_view const& input,
      const core::WindowNode::Function& function,
      const TypePtr& expectedType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> computeRunningPartitionSumColumn(
      cudf::table_view const& sortedInput,
      const core::WindowNode::Function& function,
      const TypePtr& expectedType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::column> computePartitionFirstColumn(
      cudf::table_view const& sortedInput,
      const core::WindowNode::Function& function,
      const TypePtr& expectedType,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::table> computeOutputTable(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool inputAlreadySorted = false);

  std::unique_ptr<cudf::table> computeFullPartitionCountOutput(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  std::unique_ptr<cudf::table> computeRangeRunningSumOutput(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  std::vector<cudf::size_type> streamingGroupChannels() const;
  std::vector<cudf::order> streamingGroupOrders() const;
  std::vector<cudf::null_order> streamingGroupNullOrders() const;
  cudf::size_type trailingGroupStart(
      cudf::table_view input,
      const std::vector<cudf::size_type>& channels,
      const std::vector<cudf::order>& orders,
      const std::vector<cudf::null_order>& nullOrders,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;
  void advanceBoundedStreaming();
  void processDeferredStreamingInput();
  void startActiveGroup(std::unique_ptr<cudf::table> rows);
  void appendActiveGroup(std::unique_ptr<cudf::table> rows);
  void updateActiveRangeSums(cudf::table_view rows);
  void finalizeActiveGroup();
  void prepareNextStreamingReplayOutput();
  void spillActiveRows();
  uint64_t measureTableBytes(std::unique_ptr<cudf::table>& table);
  void setPendingOutput(std::unique_ptr<cudf::table> output);
  std::unique_ptr<cudf::table> appendConstantResults(
      std::unique_ptr<cudf::table> rows,
      const std::vector<std::unique_ptr<cudf::scalar>>& results);

  std::unique_ptr<cudf::column> fixRankLikeColumn(
      std::unique_ptr<cudf::column> localResult,
      std::string_view functionName,
      cudf::table_view sortedInput,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  void updateRankLikeState(
      cudf::table_view sortedInput,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  cudf::size_type continuingPrefixSize(
      cudf::table_view sortedInput,
      const std::vector<cudf::size_type>& channels,
      const std::unique_ptr<cudf::table>& previousKey,
      const std::vector<cudf::order>& orders,
      const std::vector<cudf::null_order>& nullOrders,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  void spillSortedRun();
  void initializeSortedRunReaders();
  std::unique_ptr<cudf::table> mergeNextSortedBatch(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool& finalBatch);
  std::unique_ptr<cudf::table> takeCompletePartitions(
      std::unique_ptr<cudf::table> sorted,
      bool finalBatch,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr computeNextSortedOutput();
  void cleanupSpillFiles();

  struct StreamingReplay {
    std::vector<std::string> paths;
    size_t nextPath{0};
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
    std::unique_ptr<cudf::table> memoryRows;
    std::vector<std::unique_ptr<cudf::scalar>> results;
  };

  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
  };

  const std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputType_;
  // Only Window shapes whose child guarantees the complete ordering contract
  // use cross-batch streaming state. Other shapes retain legacy behavior.
  const bool rankLikeStreaming_;
  const bool fullPartitionCountStreaming_;
  const bool rangeSumStreaming_;
  const bool boundedStreaming_;
  const rmm::cuda_stream_view stateStream_;
  const uint64_t activeRowsLimit_;
  const uint64_t replayChunkLimit_;
  std::vector<CudfVectorPtr> inputs_;
  CudfVectorPtr pendingOutput_;
  uint64_t bufferedBytes_{0};
  uint64_t nextDiagnosticBufferedBytes_{512ULL << 20};
  std::vector<SortedRun> sortedRuns_;
  std::string spillDirectory_;
  uint64_t spillFileSequence_{0};
  std::unique_ptr<cudf::table> mergeCarry_;
  std::unique_ptr<cudf::table> partitionCarry_;
  bool readersInitialized_{false};
  bool mergeFinished_{false};
  bool spilled_{false};
  bool finished_{false};
  bool streamingSpilled_{false};

  std::unique_ptr<cudf::table> deferredInput_;
  std::unique_ptr<cudf::table> activeRows_;
  uint64_t activeRowsBytes_{0};
  std::vector<std::string> activeSpillPaths_;
  std::unique_ptr<cudf::table> activeKey_;
  int64_t activeRowCount_{0};
  std::vector<std::unique_ptr<cudf::scalar>> activeSums_;
  std::vector<int64_t> activeValidCounts_;

  std::unique_ptr<cudf::table> cumulativePartitionKey_;
  std::vector<std::unique_ptr<cudf::scalar>> cumulativeSums_;
  std::vector<int64_t> cumulativeValidCounts_;

  std::unique_ptr<StreamingReplay> streamingReplay_;

  std::vector<cudf::size_type> partitionKeyChannels_;
  std::vector<cudf::size_type> sortKeyChannels_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> sortNullOrders_;

  // Minimal running-window state. No rows from the trailing partition are
  // retained between batches.
  bool hasRankLikeState_{false};
  std::unique_ptr<cudf::table> previousPartitionKey_;
  std::unique_ptr<cudf::table> previousOrderKey_;
  int64_t previousPartitionRows_{0};
  int64_t previousRank_{0};
};

} // namespace facebook::velox::cudf_velox
