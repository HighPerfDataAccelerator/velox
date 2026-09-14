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
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/expression/sparksql/HashFunction.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/hashing.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

namespace facebook::velox::cudf_velox::sparksql {
namespace {

cudf::table_view convertToTableView(std::vector<ColumnOrView>& inputColumns) {
  std::vector<cudf::column_view> columns;
  columns.reserve(inputColumns.size());
  for (auto& col : inputColumns) {
    columns.push_back(asView(col));
  }
  return cudf::table_view(columns);
}

} // namespace

bool HashFunction::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() < 2) {
    return false;
  }

  // Multi-column hash_with_seed runs on GPU via cuDF murmurhash3_x86_32, which
  // combines columns with a constant seed (hash_combine) instead of Spark's
  // iterative seed-chaining (rapidsai/cudf#21720). When CPU fallback is
  // allowed, reject this shape so it stays on CPU; when fallback is disabled,
  // preserve the existing forced-GPU behavior.
  const bool hasMultipleDataColumns = expr->inputs().size() > 2;
  return !hasMultipleDataColumns || !CudfConfig::getInstance().allowCpuFallback;
}

HashFunction::HashFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK_GE(expr->inputs().size(), 2, "hash expects at least 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->isConstantKind(), "hash seed must be a constant");
  const auto* seedExpr =
      expr->inputs()[0]->asUnchecked<core::ConstantTypedExpr>();
  const auto vec = seedExpr->hasValueVector()
      ? seedExpr->valueVector()
      : seedExpr->toConstantVector(pool);
  int32_t seedValue = vec->as<SimpleVector<int32_t>>()->valueAt(0);
  VELOX_CHECK_GE(seedValue, 0);
  seedValue_ = seedValue;
}

ColumnOrView HashFunction::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!inputColumns.empty());
  auto inputTableView = convertToTableView(inputColumns);
  return cudf::hashing::murmurhash3_x86_32(
      inputTableView, seedValue_, stream, mr);
}

bool XxHash64Function::canEvaluate(const core::TypedExprPtr& expr) {
  return expr->inputs().size() >= 2 && expr->inputs()[0]->isConstantKind() &&
      expr->inputs()[0]->type()->kind() == TypeKind::BIGINT;
}

XxHash64Function::XxHash64Function(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK_GE(expr->inputs().size(), 2, "xxhash64 expects at least 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->isConstantKind(), "xxhash64 seed must be a constant");
  const auto* seedExpr =
      expr->inputs()[0]->asUnchecked<core::ConstantTypedExpr>();
  const auto vec = seedExpr->hasValueVector()
      ? seedExpr->valueVector()
      : seedExpr->toConstantVector(pool);
  seedValue_ = static_cast<uint64_t>(
      vec->as<SimpleVector<int64_t>>()->valueAt(0));
}

ColumnOrView XxHash64Function::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!inputColumns.empty());
  auto unsignedHash = cudf::hashing::xxhash_64(
      convertToTableView(inputColumns), seedValue_, stream, mr);
  // libcudf exposes the 64-bit hash bits as UINT64. Spark's xxhash64 result is
  // BIGINT, so materialize the signed representation before Arrow/Velox
  // conversion sees the otherwise unsupported unsigned type.
  return cudf::cast(
      unsignedHash->view(),
      cudf::data_type{cudf::type_id::INT64},
      stream,
      mr);
}

} // namespace facebook::velox::cudf_velox::sparksql
