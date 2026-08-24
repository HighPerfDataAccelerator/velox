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
#include "velox/experimental/cudf/expression/sparksql/XxHash64Function.h"

#include "velox/common/memory/Memory.h"
#include "velox/core/Expressions.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/hashing.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>

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

bool XxHash64Function::canEvaluate(const core::TypedExprPtr& expr) {
  if (expr->inputs().size() < 2) {
    return false;
  }

  // Multi-column xxhash64_with_seed runs on GPU via cuDF xxhash_64, which
  // combines columns with a constant seed instead of Spark's iterative
  // seed-chaining (rapidsai/cudf#21720). When CPU fallback is allowed, reject
  // this shape so it stays on CPU; when fallback is disabled, preserve the
  // existing forced-GPU behavior used by hash_with_seed.
  const bool hasMultipleDataColumns = expr->inputs().size() > 2;
  return !hasMultipleDataColumns || !CudfConfig::getInstance().allowCpuFallback;
}

XxHash64Function::XxHash64Function(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  VELOX_CHECK_GE(
      expr->inputs().size(), 2, "xxhash64 expects at least 2 inputs");
  VELOX_CHECK(
      expr->inputs()[0]->isConstantKind(), "xxhash64 seed must be a constant");
  const auto* seedExpr =
      expr->inputs()[0]->asUnchecked<core::ConstantTypedExpr>();
  VELOX_CHECK(!seedExpr->isNull(), "xxhash64 seed must be non-null");
  const auto vec = seedExpr->hasValueVector()
      ? seedExpr->valueVector()
      : seedExpr->toConstantVector(pool);
  seedValue_ = vec->as<SimpleVector<int64_t>>()->valueAt(0);
}

ColumnOrView XxHash64Function::eval(
    std::vector<ColumnOrView>& inputColumns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!inputColumns.empty());
  auto inputTableView = convertToTableView(inputColumns);
  auto hashes = cudf::hashing::xxhash_64(
      inputTableView, static_cast<uint64_t>(seedValue_), stream, mr);
  // Spark hashes a null input to the seed and never returns null.
  if (hashes->nullable() && hashes->null_count() > 0) {
    cudf::numeric_scalar<uint64_t> replacement(
        static_cast<uint64_t>(seedValue_), true, stream, mr);
    hashes = cudf::replace_nulls(hashes->view(), replacement, stream, mr);
  }
  return hashes;
}

} // namespace facebook::velox::cudf_velox::sparksql
