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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfValues.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/testutil/TestValue.h"
#include "velox/exec/OperatorType.h"

#include <cudf/table/table.hpp>

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cudf_velox {

CudfValues::CudfValues(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::ValuesNode> values)
    : SourceOperator(
          driverCtx,
          values->outputType(),
          operatorId,
          values->id(),
          "CudfValues"),
      CudfOperator(operatorId, values->id(), nvtx3::rgb{70, 130, 180}),
      valueNodes_(std::move(values)),
      roundsLeft_(valueNodes_->repeatTimes()) {}

void CudfValues::initialize() {
  Operator::initialize();
  VELOX_CHECK_NOT_NULL(valueNodes_);
  VELOX_CHECK(values_.empty());

  values_.reserve(valueNodes_->values().size());
  for (auto& vector : valueNodes_->values()) {
    if (vector->size() > 0) {
      if (valueNodes_->testingIsParallelizable()) {
        values_.emplace_back(
            std::static_pointer_cast<RowVector>(
                vector->testingCopyPreserveEncodings()));
      } else {
        values_.emplace_back(vector);
      }
    }
  }
  valueNodes_ = nullptr;
}

RowVectorPtr CudfValues::getOutput() {
  TestValue::adjust("facebook::velox::cudf_velox::CudfValues::getOutput", this);
  if (current_ >= values_.size()) {
    if (roundsLeft_ > 0) {
      --roundsLeft_;
    }
    current_ = 0;
  }

  if (isFinished()) {
    return nullptr;
  }

  auto input = values_[current_++];
  auto stream = cudfGlobalStreamPool().get_stream();
  std::unique_ptr<cudf::table> table;
  if (input->childrenSize() == 0) {
    table = std::make_unique<cudf::table>(
        std::vector<std::unique_ptr<cudf::column>>{});
  } else {
    table = with_arrow::toCudfTable(
        input, input->pool(), stream, get_output_mr(), std::nullopt);
  }

  return std::make_shared<CudfVector>(
      input->pool(), outputType_, input->size(), std::move(table), stream);
}

void CudfValues::close() {
  current_ = values_.size();
  roundsLeft_ = 0;
}

bool CudfValues::isFinished() {
  return roundsLeft_ == 0;
}

} // namespace facebook::velox::cudf_velox
