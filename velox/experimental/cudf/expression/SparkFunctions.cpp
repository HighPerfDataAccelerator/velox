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

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/expression/sparksql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/sparksql/HashFunction.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/LambdaExpr.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reshape.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace facebook::velox::cudf_velox {
namespace {

std::unique_ptr<cudf::column> makeRepeatedArrayColumn(
    const velox::VectorPtr& arrayVector,
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto literalArray =
      BaseVector::create(arrayVector->type(), 1, arrayVector->pool());
  SelectivityVector singleRow(1);
  std::vector<vector_size_t> sourceRows(1, 0);
  literalArray->copy(arrayVector.get(), singleRow, sourceRows.data());

  auto rowVector = std::make_shared<RowVector>(
      arrayVector->pool(),
      ROW({"literal_array"}, {arrayVector->type()}),
      BufferPtr(nullptr),
      1,
      std::vector<VectorPtr>{literalArray});

  auto table =
      with_arrow::toCudfTable(rowVector, arrayVector->pool(), stream, mr);
  auto columns = table->release();
  VELOX_CHECK_EQ(columns.size(), 1);

  auto repeatedTable =
      cudf::repeat(cudf::table_view{{columns[0]->view()}}, size, stream, mr);
  auto repeatedColumns = repeatedTable->release();
  VELOX_CHECK_EQ(repeatedColumns.size(), 1);
  return std::move(repeatedColumns[0]);
}

std::unique_ptr<cudf::column> makeMapAlignedMask(
    cudf::lists_column_view const& mapView,
    cudf::column_view maskChild,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto offsets = std::make_unique<cudf::column>(mapView.offsets(), stream, mr);
  auto mask = std::make_unique<cudf::column>(maskChild, stream, mr);
  auto nullMask = cudf::copy_bitmask(mapView.parent(), stream, mr);
  return cudf::make_lists_column(
      mapView.size(),
      std::move(offsets),
      std::move(mask),
      mapView.null_count(),
      std::move(nullMask));
}

class MapFilterFunction : public CudfFunction {
 public:
  explicit MapFilterFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    using velox::exec::FieldReference;
    using velox::exec::LambdaExpr;

    VELOX_CHECK_EQ(expr->inputs().size(), 2, "map_filter expects 2 inputs");
    VELOX_CHECK_EQ(
        expr->inputs()[0]->type()->kind(),
        TypeKind::MAP,
        "map_filter expects a MAP input");

    auto lambda = std::dynamic_pointer_cast<LambdaExpr>(expr->inputs()[1]);
    VELOX_CHECK_NOT_NULL(lambda, "map_filter requires a lambda input");

    auto predicate = lambda->body();
    if (predicate->name() == "not") {
      VELOX_CHECK_EQ(
          predicate->inputs().size(),
          1,
          "map_filter NOT predicate expects 1 input");
      keepMatches_ = false;
      predicate = predicate->inputs()[0];
    }

    VELOX_CHECK_EQ(
        predicate->name(),
        "in",
        "cuDF map_filter currently supports only key IN constant set");
    VELOX_CHECK_EQ(
        predicate->inputs().size(),
        2,
        "map_filter key IN predicate expects 2 inputs");

    auto keyField =
        std::dynamic_pointer_cast<FieldReference>(predicate->inputs()[0]);
    VELOX_CHECK_NOT_NULL(
        keyField, "map_filter key IN predicate expects key field input");
    VELOX_CHECK_EQ(
        keyField->field(),
        "k",
        "map_filter key IN predicate expects lambda key field");

    auto keysExpr =
        std::dynamic_pointer_cast<ConstantExpr>(predicate->inputs()[1]);
    VELOX_CHECK_NOT_NULL(
        keysExpr, "map_filter key IN predicate expects a constant key set");
    VELOX_CHECK(
        keysExpr->value()->type()->kind() == TypeKind::ARRAY,
        "map_filter key IN predicate expects an ARRAY constant key set");
    keySetVector_ = keysExpr->value();
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);

    auto mapView = cudf::lists_column_view(asView(inputColumns[0]));
    VELOX_CHECK_EQ(
        mapView.offset(),
        0,
        "cuDF map_filter does not support sliced map columns yet");

    auto entries = mapView.get_sliced_child(stream);
    VELOX_CHECK(
        entries.type().id() == cudf::type_id::STRUCT,
        "cuDF map_filter expects map entries as STRUCT<key,value>");
    VELOX_CHECK_EQ(entries.num_children(), 2);

    auto keyChild =
        cudf::structs_column_view(entries).get_sliced_child(0, stream);
    auto repeatedKeySet =
        makeRepeatedArrayColumn(keySetVector_, keyChild.size(), stream, mr);
    auto containsMask = cudf::lists::contains(
        cudf::lists_column_view(repeatedKeySet->view()),
        keyChild,
        stream,
        mr);
    auto maskLists =
        makeMapAlignedMask(mapView, containsMask->view(), stream, mr);

    if (keepMatches_) {
      return cudf::lists::apply_boolean_mask(
          mapView, cudf::lists_column_view(maskLists->view()), stream, mr);
    }
    return cudf::lists::apply_deletion_mask(
        mapView, cudf::lists_column_view(maskLists->view()), stream, mr);
  }

 private:
  velox::VectorPtr keySetVector_;
  bool keepMatches_{true};
};

void registerSparkArrayAccessFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  // Spark get is 0 based and returns NULL for negative or out-of-bounds
  // indices.
  registerArrayAccessFunction(
      prefix + "get",
      ArrayAccessPolicy{
          .allowNegativeIndices = true,
          .nullOnNegativeIndices = true,
          .allowOutOfBound = true,
          .indexStartsAtOne = false,
      },
      arrayAccessSignatures({"tinyint", "smallint", "integer", "bigint"}));

  // Spark map lookup is lowered as element_at(map, key). Restrict this cuDF
  // evaluator to literal keys for now, which covers generated `map['key']`
  // projections in the benchmark while leaving dynamic-key lookup unsupported.
  registerArrayAccessFunction(
      prefix + "element_at",
      ArrayAccessPolicy{
          .allowNegativeIndices = true,
          .nullOnNegativeIndices = false,
          .allowOutOfBound = true,
          .indexStartsAtOne = true,
      },
      {FunctionSignatureBuilder()
           .typeVariable("K")
           .typeVariable("V")
           .returnType("V")
           .argumentType("map(K,V)")
           .constantArgumentType("K")
           .build()});
}

void registerSparkMapFilterFunction(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "map_filter",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<MapFilterFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("K")
           .typeVariable("V")
           .returnType("map(K,V)")
           .argumentType("map(K,V)")
           .argumentType("function(K,V,boolean)")
           .build()});
}

} // namespace

void registerSparkFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<sparksql::HashFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("integer")
           .argumentType("any")
           .build()});

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<sparksql::DateAddFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("integer")
           .build()});

  registerSparkArrayAccessFunctions(prefix);
  registerSparkMapFilterFunction(prefix);
}

} // namespace facebook::velox::cudf_velox
