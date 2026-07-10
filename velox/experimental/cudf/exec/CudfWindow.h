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
#include <cudf/types.hpp>

#include <string>
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
class CudfWindow : public CudfOperatorBase {
 public:
  CudfWindow(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowNode>& windowNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
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
      bool inputAlreadySorted = false) const;

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

  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
  };

  const std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputType_;
  std::vector<CudfVectorPtr> inputs_;
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

  std::vector<cudf::size_type> partitionKeyChannels_;
  std::vector<cudf::size_type> sortKeyChannels_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> sortNullOrders_;
};

} // namespace facebook::velox::cudf_velox
