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

#include "velox/common/base/BloomFilter.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/LambdaExpr.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

#include <cuda_runtime.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/filling.hpp>
#include <cudf/io/json.hpp>
#include <cudf/json/json.hpp>
#include <cudf/lists/contains.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/stream_compaction.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
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
#include <cudf/utilities/error.hpp>

#include <folly/hash/Hash.h>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <map>
#include <optional>
#include <string_view>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

constexpr cudf::size_type kMaxRoundedArrayLength =
    std::numeric_limits<int32_t>::max() - 15;

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

bool isUtf8DecodeCharset(std::string_view charset) {
  if (charset.size() != 5 && charset.size() != 4) {
    return false;
  }

  char normalized[5];
  for (size_t i = 0; i < charset.size(); ++i) {
    normalized[i] =
        static_cast<char>(std::tolower(static_cast<unsigned char>(charset[i])));
  }
  return charset.size() == 5
      ? std::string_view(normalized, 5) == "utf-8"
      : std::string_view(normalized, 4) == "utf8";
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

bool isIntegralSequenceType(const TypePtr& type) {
  if (type == nullptr) {
    return false;
  }
  switch (type->kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return true;
    default:
      return false;
  }
}

std::unique_ptr<cudf::scalar> makeIntegralScalar(
    cudf::data_type type,
    int64_t value,
    bool valid,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  switch (type.id()) {
    case cudf::type_id::INT8:
      return std::make_unique<cudf::numeric_scalar<int8_t>>(
          static_cast<int8_t>(value), valid, stream, mr);
    case cudf::type_id::INT16:
      return std::make_unique<cudf::numeric_scalar<int16_t>>(
          static_cast<int16_t>(value), valid, stream, mr);
    case cudf::type_id::INT32:
      return std::make_unique<cudf::numeric_scalar<int32_t>>(
          static_cast<int32_t>(value), valid, stream, mr);
    case cudf::type_id::INT64:
      return std::make_unique<cudf::numeric_scalar<int64_t>>(
          value, valid, stream, mr);
    default:
      VELOX_UNSUPPORTED(
          "Unsupported sequence integral type {}",
          static_cast<int32_t>(type.id()));
  }
}

std::unique_ptr<cudf::column> makeRepeatedScalarColumn(
    const cudf::scalar& scalar,
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::make_column_from_scalar(scalar, size, stream, mr);
}

std::unique_ptr<cudf::column> makeDefaultSequenceStepColumn(
    cudf::column_view startCol,
    cudf::column_view stopCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto minusOneScalar =
      makeIntegralScalar(startCol.type(), -1, true, stream, mr);
  auto oneScalar = makeIntegralScalar(startCol.type(), 1, true, stream, mr);
  auto minusOne = cudf::make_column_from_scalar(
      *minusOneScalar, startCol.size(), stream, mr);
  auto one =
      cudf::make_column_from_scalar(*oneScalar, startCol.size(), stream, mr);
  auto decreasing = cudf::binary_operation(
      startCol,
      stopCol,
      cudf::binary_operator::GREATER,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  return cudf::copy_if_else(
      minusOne->view(), one->view(), decreasing->view(), stream, mr);
}

bool hasAnyTrue(
    cudf::column_view boolCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (boolCol.is_empty() || boolCol.null_count() == boolCol.size()) {
    return false;
  }
  auto anyAggregation = cudf::make_any_aggregation<cudf::reduce_aggregation>();
  auto anyScalar = cudf::reduce(
      boolCol,
      *anyAggregation,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto const& boolScalar =
      static_cast<cudf::numeric_scalar<bool> const&>(*anyScalar);
  return boolScalar.is_valid(stream) && boolScalar.value(stream);
}

std::unique_ptr<cudf::column> nullToFalse(
    cudf::column_view boolCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!boolCol.has_nulls()) {
    return std::make_unique<cudf::column>(boolCol, stream, mr);
  }
  cudf::numeric_scalar<bool> falseScalar(false, true, stream, mr);
  return cudf::replace_nulls(boolCol, falseScalar, stream, mr);
}

std::unique_ptr<cudf::column> nullToTrue(
    cudf::column_view boolCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!boolCol.has_nulls()) {
    return std::make_unique<cudf::column>(boolCol, stream, mr);
  }
  cudf::numeric_scalar<bool> trueScalar(true, true, stream, mr);
  return cudf::replace_nulls(boolCol, trueScalar, stream, mr);
}

std::unique_ptr<cudf::column> binaryBoolOp(
    cudf::column_view left,
    cudf::column_view right,
    cudf::binary_operator op,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::binary_operation(
      left, right, op, cudf::data_type{cudf::type_id::BOOL8}, stream, mr);
}

std::unique_ptr<cudf::column> binaryInt64Op(
    cudf::column_view left,
    cudf::column_view right,
    cudf::binary_operator op,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  return cudf::binary_operation(
      left, right, op, cudf::data_type{cudf::type_id::INT64}, stream, mr);
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

class DecodeFunction : public CudfFunction {
 public:
  explicit DecodeFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 2, "decode expects two inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARBINARY,
        "decode input must be VARBINARY");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "decode output must be VARCHAR");

    auto charset = readConstantString(expr->inputs()[1]);
    VELOX_CHECK(
        charset.has_value(),
        "decode requires a non-null constant charset argument");
    VELOX_CHECK(
        isUtf8DecodeCharset(*charset),
        "Unsupported charset for cuDF decode: {}",
        *charset);
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(expr->inputs()[0]),
        "decode on a literal binary input is not supported");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1);
    return asView(inputColumns[0]);
  }
};

class MonotonicallyIncreasingIdFunction : public CudfFunction {
 public:
  explicit MonotonicallyIncreasingIdFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(),
        0,
        "monotonically_increasing_id expects no inputs");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::BIGINT,
        "monotonically_increasing_id output must be BIGINT");

    const auto* queryConfig = currentCudfFunctionQueryConfig();
    const auto partitionId = queryConfig == nullptr
        ? 0
        : functions::sparksql::SparkQueryConfig{*queryConfig}.partitionId();
    count_ = static_cast<int64_t>(partitionId) << 33;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    const auto inputRowCount = inputColumns.empty()
        ? cudf::size_type{0}
        : asView(inputColumns.front()).size();
    return eval(inputColumns, inputRowCount, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cudf::size_type inputRowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(inputColumns.empty());
    cudf::numeric_scalar<int64_t> init(count_, true, stream, mr);
    cudf::numeric_scalar<int64_t> step(1, true, stream, mr);
    auto result = cudf::sequence(inputRowCount, init, step, stream, mr);
    count_ += inputRowCount;
    return result;
  }

 private:
  mutable int64_t count_{0};
};

struct SequenceOperand {
  std::unique_ptr<cudf::scalar> literal;
  int32_t inputColumnIndex{-1};
};

class SequenceFunction : public CudfFunction {
 public:
  explicit SequenceFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;

    VELOX_CHECK(
        expr->inputs().size() == 2 || expr->inputs().size() == 3,
        "sequence expects 2 or 3 inputs");
    VELOX_CHECK_EQ(
        expr->type()->kind(), TypeKind::ARRAY, "sequence output must be ARRAY");
    VELOX_CHECK(
        isIntegralSequenceType(expr->type()->childAt(0)),
        "cuDF sequence currently supports integral output arrays only");
    for (const auto& input : expr->inputs()) {
      VELOX_CHECK(
          isIntegralSequenceType(input->type()),
          "cuDF sequence currently supports integral inputs only");
    }

    outputElementType_ =
        cudf_velox::veloxToCudfDataType(expr->type()->childAt(0));
    operands_.reserve(expr->inputs().size());
    int32_t nextInputColumnIndex = 0;
    for (const auto& input : expr->inputs()) {
      SequenceOperand operand;
      if (auto constant = std::dynamic_pointer_cast<ConstantExpr>(input)) {
        operand.literal = makeScalarFromConstantExpr(constant);
      } else {
        operand.inputColumnIndex = nextInputColumnIndex++;
      }
      operands_.push_back(std::move(operand));
    }
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    const auto inputRowCount = inputColumns.empty()
        ? cudf::size_type{0}
        : asView(inputColumns.front()).size();
    return eval(inputColumns, inputRowCount, stream, mr);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      cudf::size_type inputRowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    std::vector<std::unique_ptr<cudf::column>> ownedInputs;
    ownedInputs.reserve(operands_.size());
    std::vector<cudf::column_view> inputViews;
    inputViews.reserve(operands_.size());

    for (const auto& operand : operands_) {
      if (operand.literal != nullptr) {
        ownedInputs.push_back(makeRepeatedScalarColumn(
            *operand.literal, inputRowCount, stream, mr));
        inputViews.push_back(ownedInputs.back()->view());
      } else {
        VELOX_CHECK_GE(operand.inputColumnIndex, 0);
        VELOX_CHECK_LT(operand.inputColumnIndex, inputColumns.size());
        inputViews.push_back(asView(inputColumns[operand.inputColumnIndex]));
      }
    }

    auto startCol = ensureType(inputViews[0], outputElementType_, stream, mr);
    auto stopCol = ensureType(inputViews[1], outputElementType_, stream, mr);
    auto stepCol = operands_.size() == 3
        ? ensureType(inputViews[2], outputElementType_, stream, mr)
        : makeDefaultSequenceStepColumn(
              startCol->view(), stopCol->view(), stream, mr);

    checkSequenceInputs(
        startCol->view(), stopCol->view(), stepCol->view(), stream, mr);
    auto sizes = computeSequenceSizes(
        startCol->view(), stopCol->view(), stepCol->view(), stream, mr);

    const bool hasNulls =
        startCol->has_nulls() || stopCol->has_nulls() || stepCol->has_nulls();
    if (!hasNulls) {
      return cudf::lists::sequences(
          startCol->view(), stepCol->view(), sizes->view(), stream, mr);
    }

    cudf::numeric_scalar<int32_t> zeroSize(0, true, stream, mr);
    auto sizesNoNull = cudf::replace_nulls(sizes->view(), zeroSize, stream, mr);
    auto startNoNull = withoutNulls(startCol->view());
    auto stepNoNull = withoutNulls(stepCol->view());
    auto sequence = cudf::lists::sequences(
        startNoNull, stepNoNull, sizesNoNull->view(), stream, mr);

    auto [nullMask, nullCount] = cudf::bitmask_and(
        cudf::table_view{{startCol->view(), stopCol->view(), stepCol->view()}},
        stream,
        mr);
    auto contents = sequence->release();
    VELOX_CHECK_EQ(contents.children.size(), 2);
    return cudf::make_lists_column(
        inputRowCount,
        std::move(contents.children[0]),
        std::move(contents.children[1]),
        nullCount,
        std::move(nullMask));
  }

 private:
  static std::unique_ptr<cudf::column> ensureType(
      cudf::column_view column,
      cudf::data_type type,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    if (column.type() == type) {
      return std::make_unique<cudf::column>(column, stream, mr);
    }
    return cudf::cast(column, type, stream, mr);
  }

  static cudf::column_view withoutNulls(cudf::column_view column) {
    return cudf::column_view(
        column.type(),
        column.size(),
        column.head<void>(),
        nullptr,
        0,
        column.offset());
  }

  static void checkSequenceInputs(
      cudf::column_view startCol,
      cudf::column_view stopCol,
      cudf::column_view stepCol,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto zero = makeIntegralScalar(stepCol.type(), 0, true, stream, mr);

    auto positive = cudf::binary_operation(
        stepCol,
        *zero,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto startGreaterStop = cudf::binary_operation(
        startCol,
        stopCol,
        cudf::binary_operator::GREATER,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto positiveInvalid = binaryBoolOp(
        positive->view(),
        startGreaterStop->view(),
        cudf::binary_operator::NULL_LOGICAL_AND,
        stream,
        mr);

    auto negative = cudf::binary_operation(
        stepCol,
        *zero,
        cudf::binary_operator::LESS,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto startLessStop = cudf::binary_operation(
        startCol,
        stopCol,
        cudf::binary_operator::LESS,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto negativeInvalid = binaryBoolOp(
        negative->view(),
        startLessStop->view(),
        cudf::binary_operator::NULL_LOGICAL_AND,
        stream,
        mr);

    auto zeroStep = cudf::binary_operation(
        stepCol,
        *zero,
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto startNotEqualStop = cudf::binary_operation(
        startCol,
        stopCol,
        cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto zeroInvalid = binaryBoolOp(
        zeroStep->view(),
        startNotEqualStop->view(),
        cudf::binary_operator::NULL_LOGICAL_AND,
        stream,
        mr);

    auto invalidEither = binaryBoolOp(
        positiveInvalid->view(),
        negativeInvalid->view(),
        cudf::binary_operator::NULL_LOGICAL_OR,
        stream,
        mr);
    auto invalid = binaryBoolOp(
        invalidEither->view(),
        zeroInvalid->view(),
        cudf::binary_operator::NULL_LOGICAL_OR,
        stream,
        mr);
    auto invalidNoNulls = nullToFalse(invalid->view(), stream, mr);
    VELOX_USER_CHECK(
        !hasAnyTrue(invalidNoNulls->view(), stream, mr),
        "Illegal sequence boundaries");
  }

  static std::unique_ptr<cudf::column> computeSequenceSizes(
      cudf::column_view startCol,
      cudf::column_view stopCol,
      cudf::column_view stepCol,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    const auto int64Type = cudf::data_type{cudf::type_id::INT64};
    auto start64 = cudf::cast(startCol, int64Type, stream, mr);
    auto stop64 = cudf::cast(stopCol, int64Type, stream, mr);
    auto step64 = cudf::cast(stepCol, int64Type, stream, mr);

    auto equal = cudf::binary_operation(
        start64->view(),
        stop64->view(),
        cudf::binary_operator::EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    cudf::numeric_scalar<int64_t> one64(1, true, stream, mr);
    auto safeStep =
        cudf::copy_if_else(one64, step64->view(), equal->view(), stream, mr);
    auto delta = binaryInt64Op(
        stop64->view(),
        start64->view(),
        cudf::binary_operator::SUB,
        stream,
        mr);
    auto quotient = binaryInt64Op(
        delta->view(),
        safeStep->view(),
        cudf::binary_operator::FLOOR_DIV,
        stream,
        mr);
    auto size64 = cudf::binary_operation(
        quotient->view(),
        one64,
        cudf::binary_operator::ADD,
        int64Type,
        stream,
        mr);

    if (size64->size() == 0) {
      return cudf::cast(
          size64->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
    }

    cudf::numeric_scalar<int64_t> maxLength(
        kMaxRoundedArrayLength, true, stream, mr);
    auto withinLimit = cudf::binary_operation(
        size64->view(),
        maxLength,
        cudf::binary_operator::LESS_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto withinLimitNoNulls = nullToTrue(withinLimit->view(), stream, mr);
    auto allAggregation = cudf::make_all_aggregation<cudf::reduce_aggregation>();
    auto allScalar = cudf::reduce(
        withinLimitNoNulls->view(),
        *allAggregation,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    auto const& allBool =
        static_cast<cudf::numeric_scalar<bool> const&>(*allScalar);
    VELOX_USER_CHECK(
        allBool.is_valid(stream) && allBool.value(stream),
        "Too long sequence");

    return cudf::cast(
        size64->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  }

  std::vector<SequenceOperand> operands_;
  cudf::data_type outputElementType_{cudf::type_id::EMPTY};
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

// Spark might_contain(varbinary serializedBloomFilter, integral hashKey)
// evaluates a runtime Bloom-filter probe. The serialized Bloom filter is a
// constant, so keep it on the host and probe host copies of each cuDF batch.
class MightContainFunction : public CudfFunction {
 public:
  explicit MightContainFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "might_contain expects exactly 2 inputs");
    auto bloomExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(
        bloomExpr, "might_contain bloom filter must be a constant");
    auto bloomValue = bloomExpr->value();
    if (bloomValue->isNullAt(0)) {
      hasFilter_ = false;
      return;
    }
    auto serialized = bloomValue->as<SimpleVector<StringView>>()->valueAt(0);
    serialized_.assign(serialized.data(), serialized.size());
    hasFilter_ = true;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "might_contain receives 1 column input; bloom filter is literal");
    auto inputView = asView(inputColumns[0]);
    const auto numRows = inputView.size();

    std::vector<int64_t> hostKeys(numRows);
    if (numRows > 0) {
      copyKeysToHost(inputView, hostKeys, stream);
    }

    std::vector<cudf::bitmask_type> hostNulls;
    const cudf::bitmask_type* deviceNulls = inputView.null_mask();
    if (deviceNulls != nullptr) {
      const auto words = cudf::bitmask_allocation_size_bytes(numRows) /
          sizeof(cudf::bitmask_type);
      hostNulls.resize(words);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostNulls.data(),
          deviceNulls,
          words * sizeof(cudf::bitmask_type),
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    stream.synchronize();

    std::vector<uint8_t> hostResult(numRows, 0);
    for (cudf::size_type i = 0; i < numRows; ++i) {
      if (!hasFilter_) {
        continue;
      }
      if (!hostNulls.empty()) {
        const bool valid = (hostNulls[i / 32] >> (i % 32)) & 1u;
        if (!valid) {
          continue;
        }
      }
      const uint64_t hashed = folly::hasher<int64_t>()(hostKeys[i]);
      hostResult[i] =
          velox::BloomFilter<>::mayContain(serialized_.c_str(), hashed) ? 1u
                                                                        : 0u;
    }

    rmm::device_buffer data(numRows * sizeof(uint8_t), stream, mr);
    if (numRows > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          data.data(),
          hostResult.data(),
          numRows * sizeof(uint8_t),
          cudaMemcpyHostToDevice,
          stream.value()));
    }

    rmm::device_buffer nullMask{};
    cudf::size_type nullCount = 0;
    if (deviceNulls != nullptr) {
      nullMask = rmm::device_buffer(
          deviceNulls,
          cudf::bitmask_allocation_size_bytes(numRows),
          stream,
          mr);
      nullCount = inputView.null_count();
    }

    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::BOOL8},
        numRows,
        std::move(data),
        std::move(nullMask),
        nullCount);
  }

 private:
  template <typename T>
  static void copyKeysAsInt64(
      const cudf::column_view& inputView,
      std::vector<int64_t>& hostKeys,
      rmm::cuda_stream_view stream) {
    std::vector<T> tmp(inputView.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        tmp.data(),
        inputView.data<T>(),
        inputView.size() * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    for (cudf::size_type i = 0; i < inputView.size(); ++i) {
      hostKeys[i] = static_cast<int64_t>(tmp[i]);
    }
  }

  static void copyKeysToHost(
      const cudf::column_view& inputView,
      std::vector<int64_t>& hostKeys,
      rmm::cuda_stream_view stream) {
    const auto inputType = inputView.type().id();
    switch (inputType) {
      case cudf::type_id::INT64:
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            hostKeys.data(),
            inputView.data<int64_t>(),
            inputView.size() * sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream.value()));
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
            static_cast<int>(inputType));
    }
  }

  std::string serialized_;
  bool hasFilter_{false};
};

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
      prefix + "might_contain",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<MightContainFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("bigint")
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
      prefix + "decode",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DecodeFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varbinary")
           .constantArgumentType("varchar")
           .build()});

  registerCudfFunction(
      prefix + "monotonically_increasing_id",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<MonotonicallyIncreasingIdFunction>(expr);
      },
      {FunctionSignatureBuilder().returnType("bigint").build()});

  registerCudfFunction(
      prefix + "sequence",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<SequenceFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("array(T)")
           .argumentType("T")
           .argumentType("T")
           .build(),
       FunctionSignatureBuilder()
           .typeVariable("T")
           .returnType("array(T)")
           .argumentType("T")
           .argumentType("T")
           .argumentType("T")
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
