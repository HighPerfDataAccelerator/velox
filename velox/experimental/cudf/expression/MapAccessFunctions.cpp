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

#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/MapAccessFunctions.h"

#include <cudf/binaryop.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/extract.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/transform.hpp>

namespace facebook::velox::cudf_velox {
namespace {

// cuDF physically represents a Velox MAP(K,V) as LIST<STRUCT<K,V>>. Build a
// zero-copy LIST view over one struct child while retaining the map row
// offsets and validity. Keeping the original root offset is important for
// sliced map columns.
cudf::column_view mapChildAsLists(
    const cudf::lists_column_view& map,
    cudf::size_type childIndex) {
  const auto entries = cudf::structs_column_view(map.child());
  VELOX_CHECK_EQ(entries.num_children(), 2);
  return cudf::column_view(
      cudf::data_type{cudf::type_id::LIST},
      map.size(),
      nullptr,
      map.null_mask(),
      map.null_count(),
      map.offset(),
      {map.offsets(), entries.child(childIndex)});
}

bool isScalarMapKey(TypeKind kind) {
  switch (kind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::HUGEINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      return true;
    default:
      return false;
  }
}

class MapAccessFunction : public CudfFunction {
 public:
  MapAccessFunction(const core::TypedExprPtr& expr, memory::MemoryPool* pool) {
    VELOX_CHECK(canEvaluateMapAccess(expr));
    if (expr->inputs()[1]->isConstantKind()) {
      keyIsLiteral_ = true;
      literalKey_ = makeScalarFromConstantExpr(expr->inputs()[1], pool);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        keyIsLiteral_ ? 1 : 2,
        "map access received an unexpected number of input columns");

    const auto map = cudf::lists_column_view(asView(inputColumns[0]));
    const auto keys = mapChildAsLists(map, 0);
    const auto values = mapChildAsLists(map, 1);

    auto indices = keyIsLiteral_
        ? cudf::lists::index_of(
              cudf::lists_column_view(keys),
              *literalKey_,
              cudf::lists::duplicate_find_option::FIND_FIRST,
              stream,
              mr)
        : cudf::lists::index_of(
              cudf::lists_column_view(keys),
              asView(inputColumns[1]),
              cudf::lists::duplicate_find_option::FIND_FIRST,
              stream,
              mr);

    // index_of returns -1 for a missing key, while extract_list_element treats
    // -1 as indexing from the end. Convert only non-negative lookup positions
    // to valid indices; null maps and null keys remain null as well.
    cudf::numeric_scalar<cudf::size_type> zero(0, true, stream, mr);
    auto found = cudf::binary_operation(
        indices->view(),
        zero,
        cudf::binary_operator::GREATER_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto [validity, nullCount] = cudf::bools_to_mask(found->view(), stream, mr);
    indices->set_null_mask(std::move(*validity), nullCount);

    return cudf::lists::extract_list_element(
        cudf::lists_column_view(values), indices->view(), stream, mr);
  }

 private:
  bool keyIsLiteral_{false};
  std::unique_ptr<cudf::scalar> literalKey_;
};

} // namespace

bool canEvaluateMapAccess(const core::TypedExprPtr& expr) {
  if (expr == nullptr || expr->inputs().size() != 2 ||
      expr->inputs()[0]->type()->kind() != TypeKind::MAP ||
      expr->inputs()[0]->isConstantKind()) {
    return false;
  }
  const auto& mapType = expr->inputs()[0]->type();
  return mapType->childAt(0)->equivalent(*expr->inputs()[1]->type()) &&
      mapType->childAt(1)->equivalent(*expr->type()) &&
      isScalarMapKey(mapType->childAt(0)->kind());
}

std::shared_ptr<CudfFunction> makeMapAccessFunction(
    const core::TypedExprPtr& expr,
    memory::MemoryPool* pool) {
  return std::make_shared<MapAccessFunction>(expr, pool);
}

} // namespace facebook::velox::cudf_velox
