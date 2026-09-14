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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/Operator.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/scalar/scalar.hpp>

namespace facebook::velox::cudf_velox {

// TODO: Does not support Filter yet.
class CudfFilterProject : public CudfOperatorBase {
 public:
  CudfFilterProject(
      int32_t operatorId,
      velox::exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::FilterNode>& filter,
      const std::shared_ptr<const core::ProjectNode>& project);

  void initialize() override;

  bool needsInput() const override {
    return !input_;
  }

  void filter(
      std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
      rmm::cuda_stream_view stream);

  std::vector<std::unique_ptr<cudf::column>> project(
      std::vector<std::unique_ptr<cudf::column>>& inputTableColumns,
      vector_size_t outputSize,
      rmm::cuda_stream_view stream);

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

  void doClose() override;

 private:
  struct NullComplexLiteralProjection {
    TypePtr type;
    column_index_t outputChannel;
  };

  struct ComplexLiteralProjection {
    VectorPtr value;
    column_index_t outputChannel;
  };

  bool allInputProcessed();

  bool requiresTransformWorkspace() const;

  // If true exprs_[0] is a filter and the other expressions are projections
  const bool hasFilter_{false};

  // Cached filter and project node for lazy initialization. After
  // initialization, they will be reset, and initialized_ will be set to true.
  std::shared_ptr<const core::ProjectNode> project_;
  std::shared_ptr<const core::FilterNode> filter_;

  std::vector<CudfExpressionPtr> projectEvaluators_;
  CudfExpressionPtr filterEvaluator_;
  CudfExpressionBatchCachePtr projectExpressionCache_;
  CudfExpressionBatchCachePtr filterExpressionCache_;
  std::string filterExpressionLabel_;
  std::vector<std::string> projectExpressionLabels_;
  uint64_t expressionCacheHits_{0};
  uint64_t expressionCacheRetained_{0};

  // Stateless projections can still allocate an output column as large as
  // the complete incoming GPU batch.  Keep the input owned while physical
  // headroom is arbitrated, then release this short-lived lease immediately
  // after getOutput.  Without this admission a FilterProject can race a
  // draining TopN/Join and fail a several-hundred-MiB RMM allocation even
  // though those operators have reclaimable device state.
  bool hasMaterializingExpression_{false};
  std::optional<DeviceMemoryWorkspaceReservation> workspaceAdmission_;
  DeviceMemoryWorkspaceRequest workspaceRequest_;

  std::vector<velox::exec::IdentityProjection> resultProjections_;
  std::vector<velox::exec::IdentityProjection> literalProjections_;
  std::vector<std::unique_ptr<cudf::scalar>> literalScalars_;
  std::vector<NullComplexLiteralProjection> nullComplexLiteralProjections_;
  std::vector<ComplexLiteralProjection> complexLiteralProjections_;
  std::vector<velox::exec::IdentityProjection> identityProjections_;
};

} // namespace facebook::velox::cudf_velox
