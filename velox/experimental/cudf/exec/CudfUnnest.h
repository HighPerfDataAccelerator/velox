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

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

class CudfUnnest : public CudfOperatorBase {
 public:
  CudfUnnest(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::UnnestNode>& unnestNode);

  static bool canRunOnGPU(const std::shared_ptr<const core::UnnestNode>& node);

  bool needsInput() const override {
    return !input_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return noMoreInput_ && !input_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

 private:
  std::vector<column_index_t> replicateChannels_;
  column_index_t unnestChannel_;
  bool hasOrdinality_;
};

} // namespace facebook::velox::cudf_velox
