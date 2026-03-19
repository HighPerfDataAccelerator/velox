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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

class CudfExpression;

class CudfNestedLoopJoinBridge : public exec::JoinBridge {
 public:
  using data_type = std::vector<std::shared_ptr<cudf::table>>;

  void setData(std::optional<data_type> data);
  std::optional<data_type> dataOrFuture(ContinueFuture* future);

  void setBuildStream(rmm::cuda_stream_view buildStream);
  std::optional<rmm::cuda_stream_view> getBuildStream();

 private:
  std::optional<data_type> data_;
  std::optional<rmm::cuda_stream_view> buildStream_;
};

class CudfNestedLoopJoinBuild
    : public exec::Operator,
      public NvtxHelper {
 public:
  CudfNestedLoopJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  void addInput(RowVectorPtr input) override;
  bool needsInput() const override;
  RowVectorPtr getOutput() override;
  void noMoreInput() override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

 private:
  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

class CudfNestedLoopJoinProbe
    : public exec::Operator,
      public NvtxHelper {
 public:
  using data_type = CudfNestedLoopJoinBridge::data_type;

  CudfNestedLoopJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::NestedLoopJoinNode> joinNode);

  void initialize() override;
  bool needsInput() const override;
  void addInput(RowVectorPtr input) override;
  void noMoreInput() override;
  RowVectorPtr getOutput() override;
  void close() override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

  static bool isSupportedJoinType(core::JoinType joinType) {
    return joinType == core::JoinType::kInner ||
        joinType == core::JoinType::kLeft ||
        joinType == core::JoinType::kRight ||
        joinType == core::JoinType::kFull ||
        joinType == core::JoinType::kLeftSemiProject;
  }

 private:
  std::shared_ptr<const core::NestedLoopJoinNode> joinNode_;
  core::JoinType joinType_;

  std::optional<data_type> buildData_;
  RowTypePtr probeType_;
  RowTypePtr buildType_;

  // Filter / AST
  cudf::ast::tree tree_;
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  std::shared_ptr<CudfExpression> filterEvaluator_;

  // Column mapping
  std::vector<cudf::size_type> leftColumnIndicesToGather_;
  std::vector<cudf::size_type> rightColumnIndicesToGather_;
  std::vector<size_t> leftColumnOutputIndices_;
  std::vector<size_t> rightColumnOutputIndices_;
  int32_t matchColumnOutputIndex_{-1};

  bool finished_{false};
  bool isLastDriver_{false};
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  std::optional<rmm::cuda_stream_view> buildStream_;

  // For right/full: track which build rows have been matched.
  std::vector<std::unique_ptr<cudf::column>> rightMatchedFlags_;

  std::unique_ptr<cudf::table> crossJoinAndFilter(
      cudf::table_view leftView,
      cudf::table_view rightView,
      rmm::cuda_stream_view stream);

  std::vector<std::unique_ptr<cudf::table>> innerJoin(
      cudf::table_view leftView,
      rmm::cuda_stream_view stream);
  std::vector<std::unique_ptr<cudf::table>> leftJoin(
      cudf::table_view leftView,
      rmm::cuda_stream_view stream);
  std::vector<std::unique_ptr<cudf::table>> rightJoin(
      cudf::table_view leftView,
      rmm::cuda_stream_view stream);
  std::vector<std::unique_ptr<cudf::table>> fullJoin(
      cudf::table_view leftView,
      rmm::cuda_stream_view stream);
  std::vector<std::unique_ptr<cudf::table>> leftSemiProjectJoin(
      cudf::table_view leftView,
      rmm::cuda_stream_view stream);
};

class CudfNestedLoopJoinBridgeTranslator
    : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node);

  std::unique_ptr<exec::JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node);

  exec::OperatorSupplier toOperatorSupplier(
      const core::PlanNodePtr& node);
};

} // namespace facebook::velox::cudf_velox
