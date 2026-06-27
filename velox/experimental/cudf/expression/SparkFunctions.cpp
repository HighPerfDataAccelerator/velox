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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
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
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/json.hpp>
#include <cudf/json/json.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

std::optional<std::string> translateSparkDatetimePattern(
    std::string_view pattern) {
  if (pattern == "yyyyMMdd") {
    return "%Y%m%d";
  }
  if (pattern == "yyyy-MM-dd") {
    return "%Y-%m-%d";
  }
  if (pattern == "yyyy/MM/dd") {
    return "%Y/%m/%d";
  }
  if (pattern == "yyyy-MM-dd HH:mm:ss") {
    return "%Y-%m-%d %H:%M:%S";
  }
  if (pattern == "yyyy-MM-dd HH:mm:ss.SSS") {
    return "%Y-%m-%d %H:%M:%S.%3f";
  }
  if (pattern == "MMM-yyyy") {
    return "%b-%Y";
  }
  return std::nullopt;
}

std::optional<std::string> readConstantString(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  auto constant = std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr);
  VELOX_CHECK_NOT_NULL(constant, "Expected a constant string argument");
  VELOX_CHECK_NOT_NULL(constant->value(), "Expected a constant vector");
  if (constant->value()->isNullAt(0)) {
    return std::nullopt;
  }
  return constant->value()->toString(0);
}

std::string readTranslatedSparkDatetimePattern(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  auto pattern = readConstantString(expr);
  VELOX_CHECK(pattern.has_value(), "Expected a non-null datetime pattern");
  auto translated = translateSparkDatetimePattern(*pattern);
  VELOX_CHECK(
      translated.has_value(),
      "Unsupported Spark datetime pattern for cuDF execution: {}",
      *pattern);
  return *translated;
}

bool needsDatetimeFormatNames(std::string_view format) {
  return format.find("%a") != std::string_view::npos ||
      format.find("%A") != std::string_view::npos ||
      format.find("%b") != std::string_view::npos ||
      format.find("%B") != std::string_view::npos ||
      format.find("%p") != std::string_view::npos;
}

int64_t timestampTicksPerSecond(cudf::data_type timestampType) {
  switch (timestampType.id()) {
    case cudf::type_id::TIMESTAMP_SECONDS:
      return 1;
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return 1'000;
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return 1'000'000;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return 1'000'000'000;
    default:
      VELOX_UNSUPPORTED(
          "Unsupported timestamp type {}",
          static_cast<int32_t>(timestampType.id()));
  }
}

std::unique_ptr<cudf::column> timestampToUnixSeconds(
    cudf::column_view inputCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto int64Type = cudf::data_type{cudf::type_id::INT64};
  auto ticks = cudf::bit_cast(inputCol, int64Type);
  const auto ticksPerSecond = timestampTicksPerSecond(inputCol.type());
  if (ticksPerSecond == 1) {
    return std::make_unique<cudf::column>(ticks, stream, mr);
  }

  cudf::numeric_scalar<int64_t> divisor(ticksPerSecond, true, stream, mr);
  return cudf::binary_operation(
      ticks, divisor, cudf::binary_operator::FLOOR_DIV, int64Type, stream, mr);
}

std::unique_ptr<cudf::column> unixSecondsToTimestamp(
    cudf::column_view inputCol,
    cudf::data_type timestampType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto int64Type = cudf::data_type{cudf::type_id::INT64};
  cudf::numeric_scalar<int64_t> multiplier(
      timestampTicksPerSecond(timestampType), true, stream, mr);
  auto ticks = cudf::binary_operation(
      inputCol, multiplier, cudf::binary_operator::MUL, int64Type, stream, mr);
  return std::make_unique<cudf::column>(
      cudf::bit_cast(ticks->view(), timestampType), stream, mr);
}

bool isSupportedFromJsonRowOfStrings(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  if (expr->inputs().size() != 1 ||
      expr->inputs()[0]->type()->kind() != TypeKind::VARCHAR ||
      expr->type()->kind() != TypeKind::ROW) {
    return false;
  }

  const auto& rowType = expr->type()->asRow();
  return std::all_of(
      rowType.children().begin(),
      rowType.children().end(),
      [](const TypePtr& child) { return child->kind() == TypeKind::VARCHAR; });
}

bool isSupportedDirectJsonLeafType(const TypePtr& type) {
  if (type->isDate() || type->isDecimal()) {
    return false;
  }
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
      return true;
    default:
      return false;
  }
}

bool isSupportedDirectJsonType(const TypePtr& type, bool isRoot) {
  switch (type->kind()) {
    case TypeKind::ROW:
      return std::all_of(
          type->asRow().children().begin(),
          type->asRow().children().end(),
          [](const TypePtr& child) {
            return isSupportedDirectJsonType(child, false);
          });
    case TypeKind::ARRAY:
      return isSupportedDirectJsonType(type->childAt(0), false);
    default:
      return !isRoot && isSupportedDirectJsonLeafType(type);
  }
}

bool isSupportedFromJsonNested(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  return expr->inputs().size() == 1 &&
      expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR &&
      isSupportedDirectJsonType(expr->type(), true);
}

bool isSupportedFromJson(const std::shared_ptr<velox::exec::Expr>& expr) {
  return isSupportedFromJsonRowOfStrings(expr) ||
      isSupportedFromJsonNested(expr);
}

std::string makeBracketJsonPath(std::string_view fieldName) {
  std::string path{"$['"};
  for (const char c : fieldName) {
    if (c == '\\' || c == '\'') {
      path.push_back('\\');
    }
    path.push_back(c);
  }
  path.append("']");
  return path;
}

std::unique_ptr<cudf::column> makeAllNullStringColumn(
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::size_type> offsets(size + 1, 0);
  rmm::device_buffer offsetsBuffer(
      offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(offsets.size()),
      std::move(offsetsBuffer),
      rmm::device_buffer{},
      0);
  return cudf::make_strings_column(
      size,
      std::move(offsetsColumn),
      rmm::device_buffer{},
      size,
      cudf::create_null_mask(size, cudf::mask_state::ALL_NULL, stream, mr));
}

std::unique_ptr<cudf::column> makeAllNullFixedWidthColumn(
    cudf::data_type type,
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::make_fixed_width_column(
      type, size, cudf::mask_state::ALL_NULL, stream, mr);
}

std::unique_ptr<cudf::column> makeZeroOffsetsColumn(
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<cudf::size_type> offsets(size + 1, 0);
  rmm::device_buffer offsetsBuffer(
      offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
  return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(offsets.size()),
      std::move(offsetsBuffer),
      rmm::device_buffer{},
      0);
}

std::unique_ptr<cudf::column> makeEmptyColumnForType(
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (type->kind()) {
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return cudf::make_strings_column(
          0,
          makeZeroOffsetsColumn(0, stream, mr),
          rmm::device_buffer{},
          0,
          rmm::device_buffer{});
    case TypeKind::ARRAY:
      return cudf::make_lists_column(
          0,
          makeZeroOffsetsColumn(0, stream, mr),
          makeEmptyColumnForType(type->childAt(0), stream, mr),
          0,
          rmm::device_buffer{});
    case TypeKind::ROW: {
      std::vector<std::unique_ptr<cudf::column>> children;
      children.reserve(type->size());
      for (size_t i = 0; i < type->size(); ++i) {
        children.push_back(makeEmptyColumnForType(type->childAt(i), stream, mr));
      }
      return cudf::make_structs_column(
          0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
    }
    default:
      return cudf::make_empty_column(cudf_velox::veloxToCudfDataType(type));
  }
}

cudf::io::schema_element makeDirectJsonSchema(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ARRAY: {
      std::map<std::string, cudf::io::schema_element> children;
      children.emplace("element", makeDirectJsonSchema(type->childAt(0)));
      return cudf::io::schema_element{
          cudf::data_type{cudf::type_id::LIST}, std::move(children)};
    }
    case TypeKind::ROW: {
      const auto& rowType = type->asRow();
      std::map<std::string, cudf::io::schema_element> children;
      std::vector<std::string> order;
      children.clear();
      order.reserve(rowType.size());
      for (size_t i = 0; i < rowType.size(); ++i) {
        const auto& name = rowType.nameOf(i);
        children.emplace(name, makeDirectJsonSchema(rowType.childAt(i)));
        order.push_back(name);
      }
      return cudf::io::schema_element{
          cudf::data_type{cudf::type_id::STRUCT},
          std::move(children),
          std::move(order)};
    }
    default:
      return cudf::io::schema_element{cudf_velox::veloxToCudfDataType(type)};
  }
}

cudf::io::schema_element makeWrappedArrayJsonSchema(
    cudf::io::schema_element arraySchema) {
  std::map<std::string, cudf::io::schema_element> children;
  children.emplace("element", std::move(arraySchema));
  std::vector<std::string> order{"element"};
  return cudf::io::schema_element{
      cudf::data_type{cudf::type_id::STRUCT},
      std::move(children),
      std::move(order)};
}

struct PreparedJsonInput {
  rmm::device_buffer data;
  rmm::device_buffer outputNullMask;
  cudf::size_type outputNullCount;
};

PreparedJsonInput prepareJsonInput(
    const cudf::column_view& inputCol,
    bool rootIsArray,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto jsonStrings = cudf::strings_column_view(inputCol);
  const auto replacementText = rootIsArray ? "[]" : "{}";
  cudf::string_scalar replacement(replacementText, true, stream, mr);
  cudf::string_scalar delimiter("\n", true, stream, mr);

  auto stripped = cudf::strings::strip(
      jsonStrings,
      cudf::strings::side_type::BOTH,
      cudf::string_scalar("", true, stream, mr),
      stream,
      mr);
  auto emptyPattern = cudf::strings::regex_program::create("^$");
  auto emptyRows = cudf::strings::contains_re(
      cudf::strings_column_view(stripped->view()),
      *emptyPattern,
      stream,
      mr);

  auto sanitized =
      cudf::copy_if_else(replacement, inputCol, emptyRows->view(), stream, mr);
  if (rootIsArray) {
    cudf::string_scalar prefix("{\"element\":", true, stream, mr);
    cudf::string_scalar suffix("}", true, stream, mr);
    auto prefixCol =
        cudf::make_column_from_scalar(prefix, inputCol.size(), stream, mr);
    auto suffixCol =
        cudf::make_column_from_scalar(suffix, inputCol.size(), stream, mr);

    auto wrapped = cudf::strings::concatenate(
        cudf::table_view{
            {prefixCol->view(), sanitized->view(), suffixCol->view()}},
        cudf::string_scalar("", true, stream, mr),
        replacement,
        cudf::strings::separator_on_nulls::YES,
        stream,
        mr);
    sanitized = std::move(wrapped);
  }

  cudf::string_scalar arrayLineReplacement(
      "{\"element\":[]}", true, stream, mr);
  auto joined = cudf::strings::join_strings(
      cudf::strings_column_view(sanitized->view()),
      delimiter,
      rootIsArray ? arrayLineReplacement : replacement,
      stream,
      mr);
  auto joinedContents = joined->release();

  auto inputNulls = cudf::is_null(inputCol, stream, mr);
  cudf::numeric_scalar<bool> falseScalar(false, true, stream, mr);
  auto emptyRowsNoNulls =
      cudf::replace_nulls(emptyRows->view(), falseScalar, stream, mr);
  auto shouldNull = cudf::binary_operation(
      inputNulls->view(),
      emptyRowsNoNulls->view(),
      cudf::binary_operator::LOGICAL_OR,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto outputIsValid = cudf::unary_operation(
      shouldNull->view(), cudf::unary_operator::NOT, stream, mr);
  auto [outputNullMask, outputNullCount] =
      cudf::bools_to_mask(outputIsValid->view(), stream, mr);

  return PreparedJsonInput{
      std::move(*joinedContents.data),
      std::move(*outputNullMask),
      outputNullCount};
}

std::unique_ptr<cudf::column> applyTopLevelNulls(
    std::unique_ptr<cudf::column> column,
    rmm::device_buffer nullMask,
    cudf::size_type nullCount) {
  const auto type = column->type();
  const auto size = column->size();
  auto contents = column->release();
  return std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data),
      std::move(nullMask),
      nullCount,
      std::move(contents.children));
}

std::unique_ptr<cudf::column> makeEnglishDatetimeFormatNamesColumn(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  static constexpr std::array<std::string_view, 40> kNames{
      "AM",        "PM",      "Sunday",   "Monday",   "Tuesday", "Wednesday",
      "Thursday",  "Friday",  "Saturday", "Sun",      "Mon",     "Tue",
      "Wed",       "Thu",     "Fri",      "Sat",      "January", "February",
      "March",     "April",   "May",      "June",     "July",    "August",
      "September", "October", "November", "December", "Jan",     "Feb",
      "Mar",       "Apr",     "May",      "Jun",      "Jul",     "Aug",
      "Sep",       "Oct",     "Nov",      "Dec"};

  std::vector<cudf::size_type> offsets;
  offsets.reserve(kNames.size() + 1);
  offsets.push_back(0);
  std::string chars;
  for (auto name : kNames) {
    chars.append(name.data(), name.size());
    offsets.push_back(static_cast<cudf::size_type>(chars.size()));
  }

  rmm::device_buffer offsetsBuffer(
      offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(offsets.size()),
      std::move(offsetsBuffer),
      rmm::device_buffer{},
      0);
  rmm::device_buffer charsBuffer(chars.data(), chars.size(), stream, mr);
  return cudf::make_strings_column(
      static_cast<cudf::size_type>(kNames.size()),
      std::move(offsetsColumn),
      std::move(charsBuffer),
      0,
      rmm::device_buffer{});
}

std::unique_ptr<cudf::column> parseTimestampsWithInvalidsAsNulls(
    cudf::column_view inputCol,
    cudf::data_type targetType,
    std::string_view format,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto strings = cudf::strings_column_view(inputCol);
  auto validMask = cudf::strings::is_timestamp(strings, format, stream, mr);
  cudf::numeric_scalar<bool> falseScalar(false, true, stream, mr);
  auto validNoNulls =
      cudf::replace_nulls(validMask->view(), falseScalar, stream, mr);

  auto parsed =
      cudf::strings::to_timestamps(strings, targetType, format, stream, mr);
  auto nullScalar =
      cudf::make_default_constructed_scalar(targetType, stream, mr);
  nullScalar->set_valid_async(false, stream);
  return cudf::copy_if_else(
      parsed->view(), *nullScalar, validNoNulls->view(), stream, mr);
}

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
        cudf::lists_column_view(repeatedKeySet->view()), keyChild, stream, mr);
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

class AddMonthsFunction : public CudfFunction {
 public:
  explicit AddMonthsFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        2,
        "add_months function expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->isDate(),
        "First argument to add_months must be a date");
    VELOX_CHECK_NOT_NULL(
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[1]),
        "Second argument to add_months must be a constant");
    months_ = makeScalarFromConstantExpr(expr->inputs()[1]);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::datetime::add_calendrical_months(
        inputCol, *months_, stream, mr);
  }

 private:
  std::unique_ptr<cudf::scalar> months_;
};

class DateFormatFunction : public CudfFunction {
 public:
  explicit DateFormatFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "date_format expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::TIMESTAMP,
        "date_format input must be TIMESTAMP");
    formatIsNull_ = !readConstantString(expr->inputs()[1]).has_value();
    if (!formatIsNull_) {
      format_ = readTranslatedSparkDatetimePattern(expr->inputs()[1]);
      needsFormatNames_ = needsDatetimeFormatNames(format_);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    auto inputCol = asView(inputColumns[0]);
    if (formatIsNull_) {
      return makeAllNullStringColumn(inputCol.size(), stream, mr);
    }
    if (needsFormatNames_) {
      auto names = makeEnglishDatetimeFormatNamesColumn(stream, mr);
      return cudf::strings::from_timestamps(
          inputCol,
          format_,
          cudf::strings_column_view(names->view()),
          stream,
          mr);
    }
    return cudf::strings::from_timestamps(
        inputCol, format_, cudf::strings_column_view{}, stream, mr);
  }

 private:
  std::string format_;
  bool formatIsNull_{false};
  bool needsFormatNames_{false};
};

class GetTimestampFunction : public CudfFunction {
 public:
  explicit GetTimestampFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "get_timestamp expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR,
        "get_timestamp input must be VARCHAR");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::TIMESTAMP,
        "get_timestamp output must be TIMESTAMP");
    targetType_ = cudf_velox::veloxToCudfDataType(expr->type());
    formatIsNull_ = !readConstantString(expr->inputs()[1]).has_value();
    if (!formatIsNull_) {
      format_ = readTranslatedSparkDatetimePattern(expr->inputs()[1]);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    auto inputCol = asView(inputColumns[0]);
    if (formatIsNull_) {
      return makeAllNullFixedWidthColumn(
          targetType_, inputCol.size(), stream, mr);
    }
    return parseTimestampsWithInvalidsAsNulls(
        inputCol, targetType_, format_, stream, mr);
  }

 private:
  cudf::data_type targetType_{cudf::type_id::TIMESTAMP_NANOSECONDS};
  std::string format_;
  bool formatIsNull_{false};
};

class UnixTimestampFunction : public CudfFunction {
 public:
  explicit UnixTimestampFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 1, "unix_timestamp expects exactly 1 input");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::TIMESTAMP,
        "unix_timestamp input must be TIMESTAMP");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::BIGINT,
        "unix_timestamp output must be BIGINT");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    return timestampToUnixSeconds(asView(inputColumns[0]), stream, mr);
  }
};

class FromUnixTimeFunction : public CudfFunction {
 public:
  explicit FromUnixTimeFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "from_unixtime expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::BIGINT,
        "from_unixtime input must be BIGINT");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "from_unixtime output must be VARCHAR");

    formatIsNull_ = !readConstantString(expr->inputs()[1]).has_value();
    if (!formatIsNull_) {
      format_ = readTranslatedSparkDatetimePattern(expr->inputs()[1]);
      needsFormatNames_ = needsDatetimeFormatNames(format_);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    auto inputCol = asView(inputColumns[0]);
    if (formatIsNull_) {
      return makeAllNullStringColumn(inputCol.size(), stream, mr);
    }

    auto timestampCol = unixSecondsToTimestamp(
        inputCol,
        cudf::data_type{CudfConfig::getInstance().timestampUnit},
        stream,
        mr);
    if (needsFormatNames_) {
      auto names = makeEnglishDatetimeFormatNamesColumn(stream, mr);
      return cudf::strings::from_timestamps(
          timestampCol->view(),
          format_,
          cudf::strings_column_view(names->view()),
          stream,
          mr);
    }
    return cudf::strings::from_timestamps(
        timestampCol->view(), format_, cudf::strings_column_view{}, stream, mr);
  }

 private:
  std::string format_;
  bool formatIsNull_{false};
  bool needsFormatNames_{false};
};

class FromJsonRowOfStringsFunction : public CudfFunction {
 public:
  explicit FromJsonRowOfStringsFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK(
        isSupportedFromJsonRowOfStrings(expr),
        "cuDF from_json currently supports only VARCHAR input to ROW of VARCHAR fields");

    const auto& rowType = expr->type()->asRow();
    fieldPaths_.reserve(rowType.size());
    for (const auto& fieldName : rowType.names()) {
      fieldPaths_.push_back(makeBracketJsonPath(fieldName));
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "from_json expects one non-literal input column");
    auto inputCol = asView(inputColumns[0]);
    auto jsonStrings = cudf::strings_column_view(inputCol);

    cudf::get_json_object_options options;
    options.set_strip_quotes_from_single_strings(true);
    options.set_missing_fields_as_nulls(true);

    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(fieldPaths_.size());
    for (const auto& fieldPath : fieldPaths_) {
      cudf::string_scalar pathScalar(fieldPath, true, stream, mr);
      children.push_back(
          cudf::get_json_object(jsonStrings, pathScalar, options, stream, mr));
    }

    rmm::device_buffer nullMask;
    cudf::size_type nullCount = 0;
    if (inputCol.nullable()) {
      nullMask = cudf::copy_bitmask(inputCol, stream, mr);
      nullCount = inputCol.null_count();
    }

    return cudf::make_structs_column(
        inputCol.size(),
        std::move(children),
        nullCount,
        std::move(nullMask),
        stream,
        mr);
  }

 private:
  std::vector<std::string> fieldPaths_;
};

class FromJsonNestedFunction : public CudfFunction {
 public:
  explicit FromJsonNestedFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK(
        isSupportedFromJsonNested(expr),
        "cuDF from_json currently supports only VARCHAR input to ROW or ARRAY with supported nested leaves");
    outputType_ = expr->type();
    schema_ = makeDirectJsonSchema(outputType_);
    rootIsArray_ = outputType_->kind() == TypeKind::ARRAY;
    if (rootIsArray_) {
      schema_ = makeWrappedArrayJsonSchema(std::move(schema_));
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "from_json expects one non-literal input column");
    auto inputCol = asView(inputColumns[0]);
    if (inputCol.is_empty()) {
      return makeEmptyColumnForType(outputType_, stream, mr);
    }

    auto jsonInput = prepareJsonInput(inputCol, rootIsArray_, stream, mr);

    auto options =
        cudf::io::json_reader_options::builder(
            cudf::io::source_info{cudf::device_span<std::byte const>{
                static_cast<std::byte const*>(jsonInput.data.data()),
                jsonInput.data.size()}})
            .lines(true)
            .delimiter('\n')
            .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
            .normalize_whitespace(true)
            .mixed_types_as_string(true)
            .experimental(true)
            .strict_validation(true)
            .keep_quotes(false)
            .dtypes(schema_)
            .prune_columns(true)
            .build();

    auto parsed = cudf::io::read_json(std::move(options), stream, mr);
    auto columns = parsed.tbl->release();

    std::unique_ptr<cudf::column> result;
    if (rootIsArray_) {
      VELOX_CHECK_EQ(
          columns.size(),
          1,
          "from_json ARRAY schema should produce a single cuDF column");
      result = std::move(columns[0]);
    } else {
      VELOX_CHECK_EQ(
          columns.size(),
          outputType_->size(),
          "from_json ROW schema produced an unexpected number of columns");
      result = cudf::make_structs_column(
          inputCol.size(),
          std::move(columns),
          0,
          rmm::device_buffer{},
          stream,
          mr);
    }

    return applyTopLevelNulls(
        std::move(result),
        std::move(jsonInput.outputNullMask),
        jsonInput.outputNullCount);
  }

 private:
  TypePtr outputType_;
  cudf::io::schema_element schema_;
  bool rootIsArray_{false};
};

class FromJsonFunction : public CudfFunction {
 public:
  explicit FromJsonFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK(
        isSupportedFromJson(expr),
        "Unsupported cuDF from_json shape");
    if (isSupportedFromJsonRowOfStrings(expr)) {
      impl_ = std::make_shared<FromJsonRowOfStringsFunction>(expr);
    } else {
      impl_ = std::make_shared<FromJsonNestedFunction>(expr);
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    return impl_->eval(inputColumns, stream, mr);
  }

 private:
  std::shared_ptr<CudfFunction> impl_;
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

  registerCudfFunction(
      prefix + "add_months",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<AddMonthsFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("integer")
           .build()});

  registerCudfFunction(
      prefix + "date_format",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DateFormatFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("timestamp")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "get_timestamp",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<GetTimestampFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp")
           .argumentType("varchar")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunctions(
      {prefix + "unix_timestamp", prefix + "to_unix_timestamp"},
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<UnixTimestampFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .argumentType("timestamp")
           .build()});

  registerCudfFunction(
      prefix + "from_unixtime",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<FromUnixTimeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("bigint")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "from_json",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<FromJsonFunction>(expr);
      },
      {});

  registerSparkArrayAccessFunctions(prefix);
  registerSparkMapFilterFunction(prefix);
}

} // namespace facebook::velox::cudf_velox
