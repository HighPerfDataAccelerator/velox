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
#include "velox/experimental/cudf/exec/CudfUnnest.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox {

namespace {

column_index_t fieldChannel(
    const RowTypePtr& inputType,
    const core::FieldAccessTypedExprPtr& field) {
  return inputType->getChildIdx(field->name());
}

std::unique_ptr<cudf::column> makeVeloxOrdinality(
    const cudf::column_view& zeroBasedPosition,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto ordinalityType = cudf::data_type{cudf::type_id::INT64};
  auto castPosition = cudf::cast(zeroBasedPosition, ordinalityType, stream, mr);
  cudf::numeric_scalar<int64_t> one(1, true, stream, mr);
  return cudf::binary_operation(
      castPosition->view(),
      one,
      cudf::binary_operator::ADD,
      ordinalityType,
      stream,
      mr);
}

} // namespace

bool CudfUnnest::canRunOnGPU(
    const std::shared_ptr<const core::UnnestNode>& node) {
  if (!node) {
    return false;
  }

  if (node->hasMarker()) {
    return false;
  }

  if (node->unnestVariables().size() != 1) {
    return false;
  }

  if (!node->unnestVariables()[0]->type()->isArray()) {
    return false;
  }

  if (node->hasOrdinality() &&
      node->outputType()->childAt(node->outputType()->size() - 1)->kind() !=
          TypeKind::BIGINT) {
    return false;
  }

  return true;
}

CudfUnnest::CudfUnnest(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::UnnestNode>& unnestNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          unnestNode->outputType(),
          unnestNode->id(),
          "CudfUnnest",
          nvtx3::rgb{46, 139, 87}, // Sea Green
          NvtxMethodFlag::kAll,
          std::nullopt,
          unnestNode),
      hasOrdinality_{unnestNode->hasOrdinality()} {
  VELOX_CHECK(
      canRunOnGPU(unnestNode),
      "cuDF Unnest currently supports only single ARRAY unnest without outer marker");

  const auto& inputType = unnestNode->sources()[0]->outputType();
  replicateChannels_.reserve(unnestNode->replicateVariables().size());
  for (const auto& field : unnestNode->replicateVariables()) {
    replicateChannels_.push_back(fieldChannel(inputType, field));
  }
  unnestChannel_ = fieldChannel(inputType, unnestNode->unnestVariables()[0]);
}

void CudfUnnest::doAddInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(input_);
  input_ = std::move(input);
}

RowVectorPtr CudfUnnest::doGetOutput() {
  if (!input_) {
    return nullptr;
  }

  if (input_->size() == 0) {
    input_.reset();
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  auto inputColumns = cudfInput->release()->release();

  std::vector<std::unique_ptr<cudf::column>> explodeInputColumns;
  explodeInputColumns.reserve(replicateChannels_.size() + 1);
  for (const auto channel : replicateChannels_) {
    VELOX_CHECK_NOT_NULL(inputColumns[channel]);
    explodeInputColumns.push_back(std::move(inputColumns[channel]));
  }
  VELOX_CHECK_NOT_NULL(inputColumns[unnestChannel_]);
  explodeInputColumns.push_back(std::move(inputColumns[unnestChannel_]));

  auto explodeInput =
      std::make_unique<cudf::table>(std::move(explodeInputColumns));
  const auto explodeColumnIdx =
      static_cast<cudf::size_type>(replicateChannels_.size());
  auto exploded = hasOrdinality_
      ? cudf::explode_position(
            explodeInput->view(), explodeColumnIdx, stream, get_output_mr())
      : cudf::explode(
            explodeInput->view(), explodeColumnIdx, stream, get_output_mr());

  const auto outputSize = exploded->num_rows();
  if (outputSize == 0) {
    input_.reset();
    return nullptr;
  }

  auto explodedColumns = exploded->release();
  if (!hasOrdinality_) {
    auto output = std::make_shared<CudfVector>(
        input_->pool(),
        outputType_,
        outputSize,
        std::make_unique<cudf::table>(std::move(explodedColumns)),
        stream);
    input_.reset();
    return output;
  }

  const auto replicatedColumnCount = replicateChannels_.size();
  VELOX_CHECK_EQ(explodedColumns.size(), replicatedColumnCount + 2);

  std::vector<std::unique_ptr<cudf::column>> outputColumns;
  outputColumns.reserve(outputType_->size());
  for (column_index_t i = 0; i < replicatedColumnCount; ++i) {
    outputColumns.push_back(std::move(explodedColumns[i]));
  }

  const auto positionColumn = replicatedColumnCount;
  const auto valueColumn = replicatedColumnCount + 1;
  outputColumns.push_back(std::move(explodedColumns[valueColumn]));
  outputColumns.push_back(makeVeloxOrdinality(
      explodedColumns[positionColumn]->view(), stream, get_output_mr()));

  auto output = std::make_shared<CudfVector>(
      input_->pool(),
      outputType_,
      outputSize,
      std::make_unique<cudf::table>(std::move(outputColumns)),
      stream);
  input_.reset();
  return output;
}

} // namespace facebook::velox::cudf_velox
