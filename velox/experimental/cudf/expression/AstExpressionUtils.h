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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstUtils.h"
// TODO(kn): in another PR
// #include "velox/experimental/cudf/CudfNoDefaults.h"

#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FieldReference.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <unordered_set>

namespace facebook::velox::cudf_velox {
namespace {

const std::unordered_set<std::string> kFunctionExprNames = {
    "might_contain",
    "xxhash64_with_seed",
    "hash_with_seed",
    "murmur3hash_with_seed",
    "row_constructor",
    "row_constructor_with_null",
    "row_constructor_with_all_null",
};

constexpr const char* kTimestampCastInstruction = "timestamp_cast";

cudf::ast::literal createLiteral(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    size_t atIndex = 0) {
  const auto kind = vector->typeKind();
  const auto& type = vector->type();
  variant value =
      VELOX_DYNAMIC_TYPE_DISPATCH(getVariant, kind, vector, atIndex);
  return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      makeScalarAndLiteral,
      kind,
      type,
      value,
      vector->isNullAt(atIndex),
      scalars);
}

// Helper function to extract literals from array elements based on type
void extractArrayLiterals(
    const ArrayVector* arrayVector,
    std::vector<cudf::ast::literal>& literals,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    vector_size_t offset,
    vector_size_t size) {
  auto elements = arrayVector->elements();

  for (auto i = offset; i < offset + size; ++i) {
    if (elements->isNullAt(i)) {
      // Skip null values for IN expressions
      continue;
    } else {
      literals.emplace_back(createLiteral(elements, scalars, i));
    }
  }
}

// Function to create literals from an array vector
std::vector<cudf::ast::literal> createLiteralsFromArray(
    const VectorPtr& vector,
    std::vector<std::unique_ptr<cudf::scalar>>& scalars) {
  std::vector<cudf::ast::literal> literals;

  // Check if it's a constant vector containing an array
  if (vector->isConstantEncoding()) {
    auto constantVector = vector->asUnchecked<ConstantVector<ComplexType>>();
    if (constantVector->isNullAt(0)) {
      // Return empty vector for null array
      return literals;
    }

    auto valueVector = constantVector->valueVector();
    if (valueVector->encoding() == VectorEncoding::Simple::ARRAY) {
      auto arrayVector = valueVector->as<ArrayVector>();
      auto index = constantVector->index();
      auto size = arrayVector->sizeAt(index);
      if (size == 0) {
        // Return empty vector for empty array
        return literals;
      }

      auto offset = arrayVector->offsetAt(index);
      auto elements = arrayVector->elements();

      // Handle different element types
      if (elements->isScalar()) {
        literals.reserve(size);
        extractArrayLiterals(arrayVector, literals, scalars, offset, size);
      } else if (elements->typeKind() == TypeKind::ARRAY) {
        // Nested arrays not supported in IN expressions
        VELOX_FAIL("Nested arrays not supported in IN expressions");
      } else {
        VELOX_FAIL(
            "Unsupported element type in array: {}",
            elements->type()->toString());
      }
    } else {
      VELOX_FAIL("Expected ARRAY encoding");
    }
  } else {
    VELOX_FAIL("Expected constant vector for IN list");
  }

  return literals;
}

std::string stripPrefix(const std::string& input, const std::string& prefix) {
  if (input.size() >= prefix.size() &&
      input.compare(0, prefix.size(), prefix) == 0) {
    return input.substr(prefix.size());
  }
  return input;
}

using Op = cudf::ast::ast_operator;
const std::unordered_map<std::string, Op> prestoBinaryOps = {
    {"plus", Op::ADD},
    {"minus", Op::SUB},
    {"multiply", Op::MUL},
    {"divide", Op::DIV},
    {"eq", Op::EQUAL},
    {"neq", Op::NOT_EQUAL},
    {"lt", Op::LESS},
    {"gt", Op::GREATER},
    {"lte", Op::LESS_EQUAL},
    {"gte", Op::GREATER_EQUAL},
    {"and", Op::NULL_LOGICAL_AND},
    {"or", Op::NULL_LOGICAL_OR},
    {"mod", Op::MOD},
};

const std::unordered_map<std::string, Op> sparkBinaryOps = {
    {"add", Op::ADD},
    {"subtract", Op::SUB},
    {"multiply", Op::MUL},
    {"divide", Op::DIV},
    {"equalto", Op::EQUAL},
    {"equalnullsafe", Op::NULL_EQUAL},
    {"lessthan", Op::LESS},
    {"greaterthan", Op::GREATER},
    {"lessthanorequal", Op::LESS_EQUAL},
    {"greaterthanorequal", Op::GREATER_EQUAL},
    {"and", Op::NULL_LOGICAL_AND},
    {"or", Op::NULL_LOGICAL_OR},
    {"mod", Op::MOD},
};

const std::unordered_map<std::string, Op> binaryOps = [] {
  std::unordered_map<std::string, Op> merged(
      sparkBinaryOps.begin(), sparkBinaryOps.end());
  merged.insert(prestoBinaryOps.begin(), prestoBinaryOps.end());
  return merged;
}();

const std::unordered_map<std::string, Op> prestoUnaryOps = {
    {"not", Op::NOT},
    {"is_null", Op::IS_NULL},
    // Trigonometric functions
    {"sin", Op::SIN},
    {"cos", Op::COS},
    {"tan", Op::TAN},
    {"asin", Op::ARCSIN},
    {"acos", Op::ARCCOS},
    {"atan", Op::ARCTAN},
    {"cosh", Op::COSH},
    {"tanh", Op::TANH},
    // Exponential and logarithmic functions
    {"exp", Op::EXP},
    {"ln", Op::LOG},
    {"sqrt", Op::SQRT},
    {"cbrt", Op::CBRT},
    // Other functions
    {"abs", Op::ABS},
};

const std::unordered_map<std::string, Op> sparkUnaryOps = {
    {"not", Op::NOT},
    {"is_null", Op::IS_NULL},
    {"isnull", Op::IS_NULL},
    // Trigonometric functions
    {"sin", Op::SIN},
    {"cos", Op::COS},
    {"tan", Op::TAN},
    {"asin", Op::ARCSIN},
    {"acos", Op::ARCCOS},
    {"atan", Op::ARCTAN},
    // Hyperbolic functions
    {"sinh", Op::SINH},
    {"cosh", Op::COSH},
    {"tanh", Op::TANH},
    {"asinh", Op::ARCSINH},
    {"acosh", Op::ARCCOSH},
    {"atanh", Op::ARCTANH},
    // Exponential and logarithmic functions
    {"exp", Op::EXP},
    {"log", Op::LOG},
    {"ln", Op::LOG},
    // Root functions
    {"sqrt", Op::SQRT},
    {"cbrt", Op::CBRT},
    // Rounding functions
    {"ceil", Op::CEIL},
    {"floor", Op::FLOOR},
    {"rint", Op::RINT},
    // Other functions
    {"abs", Op::ABS},
};

const std::unordered_map<std::string, Op> unaryOps = [] {
  std::unordered_map<std::string, Op> merged(
      sparkUnaryOps.begin(), sparkUnaryOps.end());
  merged.insert(prestoUnaryOps.begin(), prestoUnaryOps.end());
  return merged;
}();

namespace detail {

// return the AST operator for the given expression name, if any
std::optional<Op> opFromFunctionName(const std::string& funcName) {
  if (binaryOps.find(funcName) != binaryOps.end()) {
    return binaryOps.at(funcName);
  } else if (unaryOps.find(funcName) != unaryOps.end()) {
    return unaryOps.at(funcName);
  }
  return std::nullopt;
}

bool isOpAndInputsSupported(
    const cudf::ast::ast_operator op,
    const std::vector<cudf::data_type>& inputCudfDataTypes) {
  // check arity
  const auto arity = cudf::ast::detail::ast_operator_arity(op);
  if (arity != static_cast<int>(inputCudfDataTypes.size())) {
    LOG(WARNING) << "Arity mismatch for operator: " << static_cast<int>(op)
                 << " with input types size " << inputCudfDataTypes.size();
    return false;
  }
  // check for a cuDF implementation of this op with these inputs
  try {
    // this will throw if no matching implementation is found
    const auto returnCudfType =
        cudf::ast::detail::ast_operator_return_type(op, inputCudfDataTypes);
    // check it's a sensible type
    return returnCudfType.id() != cudf::type_id::EMPTY;
  } catch (...) {
    // no matching cuDF implementation
  }
  return false;
}

bool isComparisonOp(const cudf::ast::ast_operator op) {
  switch (op) {
    case Op::EQUAL:
    case Op::NULL_EQUAL:
    case Op::NOT_EQUAL:
    case Op::LESS:
    case Op::GREATER:
    case Op::LESS_EQUAL:
    case Op::GREATER_EQUAL:
      return true;
    default:
      return false;
  }
}

bool isTopLevelFieldReference(const std::shared_ptr<velox::exec::Expr>& expr) {
  using velox::exec::FieldReference;
  auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr);
  if (!fieldExpr || !fieldExpr->inputs().empty()) {
    return false;
  }
  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);
  return fieldExpr->field() == name;
}

bool isTimestampFieldComparison(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);
  const auto it = binaryOps.find(name);
  if (it == binaryOps.end() || !isComparisonOp(it->second)) {
    return false;
  }

  const auto& inputs = expr->inputs();
  if (inputs.size() != 2 || !inputs[0]->type()->isTimestamp() ||
      !inputs[1]->type()->isTimestamp() ||
      !isTopLevelFieldReference(inputs[0]) ||
      !isTopLevelFieldReference(inputs[1])) {
    return false;
  }

  try {
    return isOpAndInputsSupported(
        it->second,
        {veloxToCudfDataType(inputs[0]->type()),
         veloxToCudfDataType(inputs[1]->type())});
  } catch (...) {
    return false;
  }
}

// not special form, name = function, so unsupported for astpure
// "in", "between", "isnotnull" are not special form, but supported for astpure
// enum class SpecialFormKind : int32_t {
//   kFieldAccess = 0, supported if not nested column / function
//   kConstant = 1, "literal" for fixed_width and VARCHAR / function
//   kCast = 2, "cast" or "try_cast" to int32, int64, double only / function
//   kCoalesce = 3, unsupported/function
//   kSwitch = 4, unsupported
//   kLambda = 5, unsupported
//   kTry = 6, unsupported
//   kAnd = 7, "and" or "or" with multiple inputs
//   kOr = 8,
//   kCustom = 999,
// };
// check if the expression (name + input types) is supported in AST
bool isAstExprSupported(const std::shared_ptr<velox::exec::Expr>& expr) {
  using velox::exec::FieldReference;
  using Op = cudf::ast::ast_operator;

  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);
  const auto len = expr->inputs().size();

  if (isTimestampFieldComparison(expr)) {
    return true;
  }

  // Reject expressions with types not yet supported in AST/JIT (currently
  // TIMESTAMP and DECIMAL).
  if (containsAstUnsupportedType(expr)) {
    if (cudf_velox::CudfConfig::getInstance().debugEnabled) {
      LOG(WARNING) << "Expression contains a type not supported by AST/JIT: "
                   << expr->toString();
    }
    return false;
  }

  // Literals and top-level field references are always supported in pure
  // AST/JIT. Nested field references are delegated to FunctionExpression so
  // computed ROW values keep Velox's dereference semantics.
  auto isSupportedLiteral = [&](const TypePtr& type) {
    try {
      auto cudfType = veloxToCudfDataType(type);
      return cudf::is_fixed_width(cudfType) ||
          cudfType.id() == cudf::type_id::STRING;
    } catch (...) {
      LOG(WARNING) << "Unsupported type for literal: " << type->toString();
      return false;
    }
  };
  if (name == "literal") {
    auto type = expr->type();
    return isSupportedLiteral(type);
  }
  if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    if (fieldExpr->inputs().empty()) {
      if (fieldExpr->field() == name) {
        return true;
      }
      LOG(WARNING) << "Field " << name << " not found in expression "
                   << expr->toString();
      return false;
    }

    // Nested FieldReferences can reuse the same field name as their parent
    // (e.g. .c1.c1), which makes the pure AST/JIT path misclassify them as a
    // top-level column reference. Keep only true top-level fields here and let
    // FunctionExpression handle nested ROW dereference semantics.
    LOG(WARNING) << "Nested FieldReference is not supported by AST/JIT and "
                 << "will fall back to FunctionExpression: "
                 << expr->toString();
    return false;
  }

  if (kFunctionExprNames.count(name)) {
    return false;
  }

  // Convert input types to CUDF types once
  std::vector<cudf::data_type> inputCudfDataTypes;
  inputCudfDataTypes.reserve(len);
  for (const auto& input : expr->inputs()) {
    try {
      inputCudfDataTypes.push_back(veloxToCudfDataType(input->type()));
    } catch (...) {
      return false;
    }
  }

  // Binary operations
  if (binaryOps.find(name) != binaryOps.end()) {
    // AND/OR can handle multiple inputs by chaining
    if ((name == "and" || name == "or") && len > 2) {
      for (size_t i = 1; i < len; i++) {
        if (!isOpAndInputsSupported(
                binaryOps.at(name),
                {inputCudfDataTypes[0], inputCudfDataTypes[i]})) {
          return false;
        }
      }
      return true;
    }
    return len == 2 &&
        isOpAndInputsSupported(binaryOps.at(name), inputCudfDataTypes);
  }

  // Unary operations (includes both unaryOps and "isnotnull")
  if (unaryOps.find(name) != unaryOps.end()) {
    return isOpAndInputsSupported(unaryOps.at(name), inputCudfDataTypes);
  }
  if (name == "isnotnull" && len == 1) {
    return isOpAndInputsSupported(Op::IS_NULL, inputCudfDataTypes);
  }

  // Between: value >= lower AND value <= upper
  if (name == "between" && len == 3) {
    return isOpAndInputsSupported(
               Op::GREATER_EQUAL,
               {inputCudfDataTypes[0], inputCudfDataTypes[1]}) &&
        isOpAndInputsSupported(
               Op::LESS_EQUAL, {inputCudfDataTypes[0], inputCudfDataTypes[2]});
  }

  // In: chain of EQUAL operations
  if (name == "in") {
    return len == 2 && isSupportedLiteral(expr->inputs()[0]->type()) &&
        isOpAndInputsSupported(
               Op::EQUAL, {inputCudfDataTypes[0], inputCudfDataTypes[0]});
  }

  // Cast operations: BIGINT and DOUBLE are supported in pure AST. cuDF AST only
  // has CAST_TO_INT64, so INTEGER casts must be precomputed to preserve INT32
  // output type before a parent AST comparison sees the value.
  if ((name == "cast" || name == "try_cast") && len == 1) {
    const auto outputKind = expr->type()->kind();
    if (outputKind == TypeKind::INTEGER) {
      return false;
    }
    if (outputKind == TypeKind::BIGINT) {
      return isOpAndInputsSupported(Op::CAST_TO_INT64, inputCudfDataTypes);
    }
    if (outputKind == TypeKind::DOUBLE) {
      return isOpAndInputsSupported(Op::CAST_TO_FLOAT64, inputCudfDataTypes);
    }
    return false;
  }

  LOG(WARNING) << "Unsupported expression by AST: " << expr->toString();
  return false;
}

} // namespace detail

struct AstContext {
  cudf::ast::tree& tree;
  std::vector<std::unique_ptr<cudf::scalar>>& scalars;
  const std::vector<RowTypePtr> inputRowSchema;
  const std::vector<std::reference_wrapper<std::vector<PrecomputeInstruction>>>
      precomputeInstructions;
  const std::shared_ptr<velox::exec::Expr>
      rootExpr; // Track the root expression
  bool allowPureAstOnly;

  cudf::ast::expression const& pushExprToTree(
      const std::shared_ptr<velox::exec::Expr>& expr);
  cudf::ast::expression const& pushFieldReferenceToTree(
      const std::shared_ptr<velox::exec::FieldReference>& fieldExpr);
  cudf::ast::expression const& pushTimestampFieldReferenceToTree(
      const std::shared_ptr<velox::exec::FieldReference>& fieldExpr);
  cudf::ast::expression const& addPrecomputeInstructionOnSide(
      size_t sideIdx,
      size_t columnIndex,
      std::string const& instruction,
      std::string const& fieldName,
      const std::shared_ptr<CudfExpression>& node = nullptr,
      std::optional<cudf::data_type> expectedType = std::nullopt);
  cudf::ast::expression const& addPrecomputeInstruction(
      std::string const& name,
      std::string const& instruction,
      std::string const& fieldName = {},
      const std::shared_ptr<CudfExpression>& node = nullptr);
  cudf::ast::expression const& multipleInputsToPairWise(
      const std::shared_ptr<velox::exec::Expr>& expr);
  static bool canBeEvaluated(const std::shared_ptr<velox::exec::Expr>& expr);
  // Determines which side (0=left, 1=right) an expression references by
  // examining its field references. Returns -1 if no fields found, -2 if spans
  // both sides.
  int findExpressionSide(const std::shared_ptr<velox::exec::Expr>& expr) const;
};

/// Checks if an expression contains sub-expressions that:
/// 1. Cannot be represented natively in cuDF AST (need precomputation)
/// 2. AND reference fields from both sides of a join
/// Returns true only if such problematic sub-expressions exist.
inline bool hasNonAstSubexprSpanningBothSides(
    const std::shared_ptr<velox::exec::Expr>& expr,
    const RowTypePtr& leftSchema,
    const RowTypePtr& rightSchema) {
  // Check if the expression is natively supported by AST
  // If it is, we don't need to precompute it, so cross-side references are fine
  if (detail::isAstExprSupported(expr)) {
    // Recursively check children
    for (const auto& child : expr->inputs()) {
      if (hasNonAstSubexprSpanningBothSides(child, leftSchema, rightSchema)) {
        return true;
      }
    }
    return false;
  }

  // Expression needs precomputation - check if it spans both sides
  bool hasLeft = false, hasRight = false;
  for (const auto* field : expr->distinctFields()) {
    hasLeft |= leftSchema->containsChild(field->field());
    hasRight |= rightSchema->containsChild(field->field());
    if (hasLeft && hasRight) {
      return true;
    }
  }
  return false;
}

// get nested column indices
std::vector<int> getNestedColumnIndices(
    const TypePtr& rowType,
    const std::string& fieldName) {
  std::vector<int> indices;
  auto rowTypePtr = asRowType(rowType);
  if (rowTypePtr->containsChild(fieldName)) {
    auto columnIndex = rowTypePtr->getChildIdx(fieldName);
    indices.push_back(columnIndex);
  }
  return indices;
}

cudf::ast::expression const& AstContext::addPrecomputeInstructionOnSide(
    size_t sideIdx,
    size_t columnIndex,
    std::string const& instruction,
    std::string const& fieldName,
    const std::shared_ptr<CudfExpression>& node,
    std::optional<cudf::data_type> expectedType) {
  auto newColumnIndex = inputRowSchema[sideIdx].get()->size() +
      precomputeInstructions[sideIdx].get().size();
  if (fieldName.empty()) {
    // This custom op should be added to input columns.
    precomputeInstructions[sideIdx].get().emplace_back(
        columnIndex, instruction, newColumnIndex, node, expectedType);
  } else {
    auto nestedIndices = getNestedColumnIndices(
        inputRowSchema[sideIdx].get()->childAt(columnIndex), fieldName);
    precomputeInstructions[sideIdx].get().emplace_back(
        columnIndex,
        instruction,
        newColumnIndex,
        std::move(nestedIndices),
        node,
        expectedType);
  }
  auto side = static_cast<cudf::ast::table_reference>(sideIdx);
  return tree.push(cudf::ast::column_reference(newColumnIndex, side));
}

cudf::ast::expression const& AstContext::addPrecomputeInstruction(
    std::string const& name,
    std::string const& instruction,
    std::string const& fieldName,
    const std::shared_ptr<CudfExpression>& node) {
  for (size_t sideIdx = 0; sideIdx < inputRowSchema.size(); ++sideIdx) {
    if (inputRowSchema[sideIdx].get()->containsChild(name)) {
      auto columnIndex = inputRowSchema[sideIdx].get()->getChildIdx(name);
      return addPrecomputeInstructionOnSide(
          sideIdx, columnIndex, instruction, fieldName, node);
    }
  }
  VELOX_FAIL("Field not found: {}", name);
}

cudf::ast::expression const& AstContext::pushFieldReferenceToTree(
    const std::shared_ptr<velox::exec::FieldReference>& fieldExpr) {
  const auto name = stripPrefix(
      fieldExpr->name(), CudfConfig::getInstance().functionNamePrefix);
  const auto fieldName =
      fieldExpr->inputs().empty() ? name : fieldExpr->inputs()[0]->name();
  for (size_t sideIdx = 0; sideIdx < inputRowSchema.size(); ++sideIdx) {
    auto& schema = inputRowSchema[sideIdx];
    if (schema.get()->containsChild(fieldName)) {
      auto columnIndex = schema.get()->getChildIdx(fieldName);
      // This column may be complex data type like ROW, we need to get the
      // name from row. Push fieldName.name to the tree.
      auto side = static_cast<cudf::ast::table_reference>(sideIdx);
      if (fieldExpr->field() == fieldName) {
        return tree.push(cudf::ast::column_reference(columnIndex, side));
      } else if (!allowPureAstOnly) {
        return addPrecomputeInstruction(
            fieldName, "nested_column", fieldExpr->field());
      } else {
        VELOX_FAIL("Unsupported type for nested column operation");
      }
    }
  }
  VELOX_FAIL("Field not found: {}", name);
}

cudf::ast::expression const& AstContext::pushTimestampFieldReferenceToTree(
    const std::shared_ptr<velox::exec::FieldReference>& fieldExpr) {
  const auto name = stripPrefix(
      fieldExpr->name(), CudfConfig::getInstance().functionNamePrefix);
  const auto fieldName =
      fieldExpr->inputs().empty() ? name : fieldExpr->inputs()[0]->name();
  for (size_t sideIdx = 0; sideIdx < inputRowSchema.size(); ++sideIdx) {
    auto& schema = inputRowSchema[sideIdx];
    if (schema.get()->containsChild(fieldName)) {
      auto columnIndex = schema.get()->getChildIdx(fieldName);
      if (fieldExpr->field() == fieldName) {
        return addPrecomputeInstructionOnSide(
            sideIdx, columnIndex, kTimestampCastInstruction, "");
      }
      VELOX_FAIL("Unsupported nested timestamp field operation");
    }
  }
  VELOX_FAIL("Field not found: {}", name);
}

/// Handles logical AND/OR expressions with multiple inputs by converting them
/// into a chain of binary operations. For example, "a AND b AND c" becomes
/// "(a AND b) AND c".
///
/// @param expr The expression containing multiple inputs for AND/OR operation
/// @return A reference to the resulting AST expression
cudf::ast::expression const& AstContext::multipleInputsToPairWise(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using Operation = cudf::ast::operation;

  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);
  auto len = expr->inputs().size();
  // Create a simple chain of operations
  auto result = &pushExprToTree(expr->inputs()[0]);

  // Chain the rest of the inputs sequentially
  for (size_t i = 1; i < len; i++) {
    auto const& nextInput = pushExprToTree(expr->inputs()[i]);
    result = &tree.push(Operation{binaryOps.at(name), *result, nextInput});
  }
  return *result;
}

/// Pushes an expression into the AST tree and returns a reference to the
/// resulting expression.
///
/// @param expr The expression to push into the AST tree
/// @return A reference to the resulting AST expression
cudf::ast::expression const& AstContext::pushExprToTree(
    const std::shared_ptr<velox::exec::Expr>& expr) {
  using Op = cudf::ast::ast_operator;
  using Operation = cudf::ast::operation;
  using velox::exec::ConstantExpr;
  using velox::exec::FieldReference;

  const auto name =
      stripPrefix(expr->name(), CudfConfig::getInstance().functionNamePrefix);

  if (!detail::isAstExprSupported(expr)) {
    if (canBeEvaluatedByCudf(expr, /*deep=*/false)) {
      // Shallow check: only verify this operation is supported
      // Children will be recursively handled by createCudfExpression
      // Determine which side this expression references
      int sideIdx = findExpressionSide(expr);
      if (sideIdx == -2) {
        // Expression spans both sides of the join - cannot precompute on one
        // side
        VELOX_FAIL(
            "Expression spans both join sides and cannot be precomputed: " +
            name);
      }
      if (sideIdx < 0) {
        sideIdx = 0; // Default to left side if no fields found
      }
      auto node = createCudfExpression(expr, inputRowSchema[sideIdx]);
      return addPrecomputeInstructionOnSide(
          sideIdx, 0, name, "", node, veloxToCudfDataType(expr->type()));
    }
    VELOX_FAIL("Unsupported expression: {}", name);
  }

  auto len = expr->inputs().size();
  auto& type = expr->type();

  if (name == "literal") {
    auto c = dynamic_cast<ConstantExpr*>(expr.get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    VELOX_CHECK(value->isConstantEncoding());

    // Materialize NULL literals via make_column_from_scalar so the output
    // column preserves nullness for downstream operators like count(column).
    //
    // Also keep the standalone literal workaround: cudf::compute_column can
    // produce spurious nulls for root literal expressions. See comment below.
    //
    // TODO: There is a scalar stream synchronization bug that causes
    // cudf::compute_column to produce spurious nulls for standalone
    // literal expressions.  Work around it by materialising via
    // make_column_from_scalar instead.
    if (value->isNullAt(0) || expr == rootExpr) {
      // convert to cudf scalar and store it
      createLiteral(value, scalars);
      // The scalar index is scalars.size() - 1 since we just added it
      std::string fillExpr = "fill " + std::to_string(scalars.size() - 1);
      // For literals, we use the first column just to get the size, but create
      // a new column The new column will be appended after the original input
      // columns
      return addPrecomputeInstruction(inputRowSchema[0]->nameOf(0), fillExpr);
    }

    return tree.push(createLiteral(value, scalars));
  } else if (binaryOps.find(name) != binaryOps.end()) {
    if (name == "and" or name == "or") {
      return multipleInputsToPairWise(expr);
    }
    VELOX_CHECK_EQ(len, 2);
    if (detail::isTimestampFieldComparison(expr)) {
      auto leftField =
          std::dynamic_pointer_cast<FieldReference>(expr->inputs()[0]);
      auto rightField =
          std::dynamic_pointer_cast<FieldReference>(expr->inputs()[1]);
      VELOX_CHECK_NOT_NULL(leftField);
      VELOX_CHECK_NOT_NULL(rightField);
      auto const& op1 = pushTimestampFieldReferenceToTree(leftField);
      auto const& op2 = pushTimestampFieldReferenceToTree(rightField);
      return tree.push(Operation{binaryOps.at(name), op1, op2});
    }
    // libcudf's AST parser cannot currently type a NULL_EQUAL operand that is
    // itself an AST operation (e.g. (string_col = 'x') <=> true), even though
    // that child resolves to BOOL8. Materialize such children as temporary
    // cuDF columns. This preserves Spark null-safe semantics and remains an
    // entirely GPU execution path.
    auto pushNullEqualOperand =
        [&](const std::shared_ptr<velox::exec::Expr>& input)
        -> const cudf::ast::expression& {
      if (binaryOps.at(name) != Op::NULL_EQUAL ||
          std::dynamic_pointer_cast<FieldReference>(input) ||
          std::dynamic_pointer_cast<ConstantExpr>(input)) {
        return pushExprToTree(input);
      }
      int sideIdx = findExpressionSide(input);
      VELOX_CHECK_NE(
          sideIdx,
          -2,
          "NULL_EQUAL child spanning both join sides cannot be precomputed");
      if (sideIdx < 0) {
        sideIdx = 0;
      }
      auto node = createCudfExpression(input, inputRowSchema[sideIdx]);
      return addPrecomputeInstructionOnSide(
          sideIdx,
          0,
          "null_equal_child",
          "",
          node,
          veloxToCudfDataType(input->type()));
    };
    auto const& op1 = pushNullEqualOperand(expr->inputs()[0]);
    auto const& op2 = pushNullEqualOperand(expr->inputs()[1]);
    // libcudf's type inference accepts some mixed numeric signatures, but the
    // AST parser requires the two operands of arithmetic operations to have
    // identical physical types. Spark/Velox performs numeric coercion while
    // resolving the expression (for example DOUBLE / INTEGER -> DOUBLE) and
    // does not always leave an explicit CastExpr in the converted plan. Mirror
    // that resolved type in the AST rather than presenting libcudf with the
    // original mixed operands.
    const bool isArithmetic = name == "add" || name == "plus" ||
        name == "subtract" || name == "minus" || name == "multiply" ||
        name == "divide" || name == "mod";
    if (isArithmetic) {
      auto const targetKind = expr->type()->kind();
      auto castOperand =
          [&](const cudf::ast::expression& operand,
              TypeKind inputKind) -> const cudf::ast::expression& {
        if (inputKind == targetKind) {
          return operand;
        }
        if (targetKind == TypeKind::DOUBLE) {
          return tree.push(Operation{Op::CAST_TO_FLOAT64, operand});
        }
        if (targetKind == TypeKind::BIGINT) {
          return tree.push(Operation{Op::CAST_TO_INT64, operand});
        }
        // INTEGER arithmetic should already have matching INT32 operands;
        // cuDF AST has no CAST_TO_INT32 operator.
        return operand;
      };
      auto const& coerced1 =
          castOperand(op1, expr->inputs()[0]->type()->kind());
      auto const& coerced2 =
          castOperand(op2, expr->inputs()[1]->type()->kind());
      return tree.push(Operation{binaryOps.at(name), coerced1, coerced2});
    }
    return tree.push(Operation{binaryOps.at(name), op1, op2});
  } else if (unaryOps.find(name) != unaryOps.end()) {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    // Spark result type is different with presto, presto is same with cudf
    const auto& op2 = tree.push(Operation{unaryOps.at(name), op1});
    if ((name == "ceil" || name == "floor") &&
        expr->type()->kind() == TypeKind::BIGINT) {
      return tree.push(Operation{Op::CAST_TO_INT64, op2});
    }
    return op2;
  } else if (name == "isnotnull") {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    auto const& nullOp = tree.push(Operation{Op::IS_NULL, op1});
    return tree.push(Operation{Op::NOT, nullOp});
  } else if (name == "between") {
    VELOX_CHECK_EQ(len, 3);
    auto const& value = pushExprToTree(expr->inputs()[0]);
    auto const& lower = pushExprToTree(expr->inputs()[1]);
    auto const& upper = pushExprToTree(expr->inputs()[2]);
    // construct between(op2, op3) using >= and <=
    auto const& geLower = tree.push(Operation{Op::GREATER_EQUAL, value, lower});
    auto const& leUpper = tree.push(Operation{Op::LESS_EQUAL, value, upper});
    return tree.push(Operation{Op::NULL_LOGICAL_AND, geLower, leUpper});
  } else if (name == "in") {
    // number of inputs is variable. >=2
    VELOX_CHECK_EQ(len, 2);
    // actually len is 2, second input is ARRAY
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    auto c = dynamic_cast<ConstantExpr*>(expr->inputs()[1].get());
    VELOX_CHECK_NOT_NULL(c, "literal expression should be ConstantExpr");
    auto value = c->value();
    VELOX_CHECK_NOT_NULL(value, "ConstantExpr value is null");

    // Use the new createLiteralsFromArray function to get literals
    auto literals = createLiteralsFromArray(value, scalars);

    // Create equality expressions for each literal and OR them together
    std::vector<const cudf::ast::expression*> exprVec;
    for (auto& literal : literals) {
      auto const& opi = tree.push(std::move(literal));
      auto const& logicalNode = tree.push(Operation{Op::EQUAL, op1, opi});
      exprVec.push_back(&logicalNode);
    }

    // Handle empty IN list case
    if (exprVec.empty()) {
      // FAIL
      VELOX_FAIL("Empty IN list");
      // Return FALSE for empty IN list
      // auto falseValue = std::make_shared<ConstantVector<bool>>(
      //     value->pool(), 1, false, TypeKind::BOOLEAN, false);
      // return tree.push(createLiteral(falseValue, scalars));
    }

    // OR all logical nodes
    auto* result = exprVec[0];
    for (size_t i = 1; i < exprVec.size(); i++) {
      auto const& treeNode =
          tree.push(Operation{Op::NULL_LOGICAL_OR, *result, *exprVec[i]});
      result = &treeNode;
    }
    return *result;
  } else if (name == "cast" || name == "try_cast") {
    VELOX_CHECK_EQ(len, 1);
    auto const& op1 = pushExprToTree(expr->inputs()[0]);
    if (expr->type()->kind() == TypeKind::INTEGER) {
      VELOX_FAIL("INTEGER casts must be precomputed outside cuDF AST");
    } else if (expr->type()->kind() == TypeKind::BIGINT) {
      return tree.push(Operation{Op::CAST_TO_INT64, op1});
    } else if (expr->type()->kind() == TypeKind::DOUBLE) {
      return tree.push(Operation{Op::CAST_TO_FLOAT64, op1});
    } else {
      VELOX_FAIL("Unsupported type for cast operation");
    }
  } else if (auto fieldExpr = std::dynamic_pointer_cast<FieldReference>(expr)) {
    return pushFieldReferenceToTree(fieldExpr);
  } else {
    VELOX_UNREACHABLE("Unsupported expression: {}", name);
  }
}

// Returns: 0 = left only, 1 = right only, -1 = no fields, -2 = spans both sides
int AstContext::findExpressionSide(
    const std::shared_ptr<velox::exec::Expr>& expr) const {
  int foundSide = -1;
  for (const auto* field : expr->distinctFields()) {
    for (size_t sideIdx = 0; sideIdx < inputRowSchema.size(); ++sideIdx) {
      if (inputRowSchema[sideIdx].get()->containsChild(field->field())) {
        if (foundSide == -1) {
          foundSide = static_cast<int>(sideIdx);
        } else if (foundSide != static_cast<int>(sideIdx)) {
          // Expression spans both sides
          return -2;
        }
        break;
      }
    }
  }
  return foundSide;
}

std::vector<ColumnOrView> precomputeSubexpressions(
    const std::vector<cudf::column_view>& inputColumnViews,
    const std::vector<PrecomputeInstruction>& precomputeInstructions,
    const std::vector<std::unique_ptr<cudf::scalar>>& scalars,
    const RowTypePtr& inputRowSchema,
    rmm::cuda_stream_view stream) {
  std::vector<ColumnOrView> precomputedColumns;
  precomputedColumns.reserve(precomputeInstructions.size());

  auto appendPrecomputed = [&](ColumnOrView result,
                                   const std::optional<cudf::data_type>&
                                       expectedType) {
    if (expectedType && asView(result).type() != *expectedType) {
      result =
          cudf::cast(asView(result), *expectedType, stream, get_output_mr());
    }
    precomputedColumns.push_back(std::move(result));
  };

  for (const auto& instruction : precomputeInstructions) {
    auto
        [dependent_column_index,
         ins_name,
         new_column_index,
         nested_dependent_column_indices,
         cudf_expression,
         expected_type] = instruction;

    // If a compiled cudf node is available, evaluate it directly.
    if (cudf_expression) {
      auto result = cudf_expression->eval(
          inputColumnViews,
          stream,
          get_output_mr(),
          /*finalize=*/true);
      appendPrecomputed(std::move(result), expected_type);
      continue;
    }
    if (ins_name.rfind("fill", 0) == 0) {
      auto scalarIndex =
          std::stoi(ins_name.substr(5)); // "fill " is 5 characters
      auto newColumn = cudf::make_column_from_scalar(
          *scalars[scalarIndex],
          inputColumnViews[dependent_column_index].size(),
          stream,
          get_output_mr());
      appendPrecomputed(std::move(newColumn), expected_type);
    } else if (ins_name == "nested_column") {
      // Nested column already exists in input. Don't materialize.
      auto view = inputColumnViews[dependent_column_index].child(
          nested_dependent_column_indices[0]);
      appendPrecomputed(view, expected_type);
    } else if (ins_name == kTimestampCastInstruction) {
      auto targetType =
          cudf::data_type{CudfConfig::getInstance().timestampUnit};
      auto view = inputColumnViews[dependent_column_index];
      if (view.type() == targetType) {
        appendPrecomputed(view, expected_type);
      } else {
        appendPrecomputed(
            cudf::cast(view, targetType, stream, get_output_mr()),
            expected_type);
      }
    } else {
      VELOX_FAIL("Unsupported precompute operation {}", ins_name);
    }
  }

  return precomputedColumns;
}

} // namespace
} // namespace facebook::velox::cudf_velox
