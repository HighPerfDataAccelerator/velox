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

#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node);

/// Narrow cuDF Window implementation for:
/// - row_number() / rank() with the default UNBOUNDED PRECEDING to CURRENT ROW
///   frame.
/// - sum(field) over a full partition ROWS frame.
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

  std::unique_ptr<cudf::table> computeOutputTable(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const;

  const std::shared_ptr<const core::WindowNode> windowNode_;
  const RowTypePtr inputType_;
  std::vector<CudfVectorPtr> inputs_;
  bool finished_{false};

  std::vector<cudf::size_type> partitionKeyChannels_;
  std::vector<cudf::size_type> sortKeyChannels_;
  std::vector<cudf::order> sortOrders_;
  std::vector<cudf::null_order> sortNullOrders_;
};

} // namespace facebook::velox::cudf_velox
