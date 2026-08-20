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
#include "velox/experimental/cudf/expression/sparksql/MightContainFunction.h"

#include "velox/common/base/BloomFilter.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>

#include <folly/hash/Hash.h>

namespace facebook::velox::cudf_velox::sparksql {
namespace {

template <typename T>
void copyKeysAsInt64(
    const cudf::column_view& inputView,
    std::vector<int64_t>& hostKeys,
    rmm::cuda_stream_view stream) {
  std::vector<T> tmp(static_cast<size_t>(inputView.size()));
  if (inputView.size() > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        tmp.data(),
        inputView.data<T>(),
        static_cast<size_t>(inputView.size()) * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
  }
  for (cudf::size_type i = 0; i < inputView.size(); ++i) {
    hostKeys[static_cast<size_t>(i)] = static_cast<int64_t>(tmp[i]);
  }
}

void copyKeysToHost(
    const cudf::column_view& inputView,
    std::vector<int64_t>& hostKeys,
    rmm::cuda_stream_view stream) {
  switch (inputView.type().id()) {
    case cudf::type_id::INT64:
      if (inputView.size() > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            hostKeys.data(),
            inputView.data<int64_t>(),
            static_cast<size_t>(inputView.size()) * sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream.value()));
      }
      break;
    case cudf::type_id::INT32:
      copyKeysAsInt64<int32_t>(inputView, hostKeys, stream);
      break;
    case cudf::type_id::INT16:
      copyKeysAsInt64<int16_t>(inputView, hostKeys, stream);
      break;
    case cudf::type_id::INT8:
      copyKeysAsInt64<int8_t>(inputView, hostKeys, stream);
      break;
    case cudf::type_id::UINT64:
      copyKeysAsInt64<uint64_t>(inputView, hostKeys, stream);
      break;
    case cudf::type_id::UINT32:
      copyKeysAsInt64<uint32_t>(inputView, hostKeys, stream);
      break;
    case cudf::type_id::UINT16:
      copyKeysAsInt64<uint16_t>(inputView, hostKeys, stream);
      break;
    case cudf::type_id::UINT8:
      copyKeysAsInt64<uint8_t>(inputView, hostKeys, stream);
      break;
    default:
      VELOX_FAIL(
          "might_contain hash input must be an integer type; saw cudf type_id={}",
          static_cast<int>(inputView.type().id()));
  }
}

} // namespace

bool MightContainFunction::canEvaluate(const core::TypedExprPtr& expr) {
  return expr->inputs().size() == 2 && expr->inputs()[0]->isConstantKind();
}

MightContainFunction::MightContainFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK_EQ(
      expr->inputs().size(), 2, "might_contain expects exactly 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->isConstantKind(),
      "might_contain bloom filter must be a constant");
  const auto* bloomExpr =
      expr->inputs()[0]->asUnchecked<core::ConstantTypedExpr>();
  const auto bloomValue = bloomExpr->hasValueVector()
      ? bloomExpr->valueVector()
      : bloomExpr->toConstantVector(pool);
  if (bloomValue->isNullAt(0)) {
    bloomIsNull_ = true;
    return;
  }
  auto serialized = bloomValue->as<SimpleVector<StringView>>()->valueAt(0);
  serialized_.assign(serialized.data(), serialized.size());
}

ColumnOrView MightContainFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK_EQ(
      inputColumns.size(),
      1,
      "might_contain receives 1 column input; bloom filter is literal");
  auto inputView = asView(inputColumns[0]);
  const auto numRows = inputView.size();

  if (bloomIsNull_) {
    return cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::BOOL8},
        numRows,
        cudf::mask_state::ALL_NULL,
        stream,
        mr);
  }

  std::vector<int64_t> hostKeys(static_cast<size_t>(numRows));
  if (numRows > 0) {
    copyKeysToHost(inputView, hostKeys, stream);
  }

  std::vector<cudf::bitmask_type> hostNulls;
  const cudf::bitmask_type* deviceNulls = inputView.null_mask();
  if (deviceNulls != nullptr) {
    const auto bytes = cudf::bitmask_allocation_size_bytes(numRows);
    hostNulls.resize(bytes / sizeof(cudf::bitmask_type));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostNulls.data(),
        deviceNulls,
        bytes,
        cudaMemcpyDeviceToHost,
        stream.value()));
  }
  stream.synchronize();

  std::vector<uint8_t> hostResult(static_cast<size_t>(numRows), 0);
  for (cudf::size_type i = 0; i < numRows; ++i) {
    if (!hostNulls.empty() && !cudf::bit_is_set(hostNulls.data(), i)) {
      continue;
    }
    const uint64_t hashed = folly::hasher<int64_t>()(hostKeys[i]);
    hostResult[static_cast<size_t>(i)] =
        BloomFilter<>::mayContain(serialized_.c_str(), hashed) ? 1u : 0u;
  }

  rmm::device_buffer data(
      static_cast<size_t>(numRows) * sizeof(uint8_t), stream, mr);
  if (numRows > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        data.data(),
        hostResult.data(),
        static_cast<size_t>(numRows) * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        stream.value()));
  }

  rmm::device_buffer nullMask{};
  cudf::size_type nullCount = 0;
  if (deviceNulls != nullptr) {
    nullMask = rmm::device_buffer(
        deviceNulls, cudf::bitmask_allocation_size_bytes(numRows), stream, mr);
    nullCount = inputView.null_count();
  }

  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::BOOL8},
      numRows,
      std::move(data),
      std::move(nullMask),
      nullCount);
}

} // namespace facebook::velox::cudf_velox::sparksql
