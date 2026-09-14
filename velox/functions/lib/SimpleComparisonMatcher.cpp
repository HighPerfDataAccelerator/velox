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

#include "velox/functions/lib/SimpleComparisonMatcher.h"
#include "velox/expression/ExprConstants.h"

namespace facebook::velox::functions {

bool Matcher::allMatch(
    const std::vector<core::TypedExprPtr>& exprs,
    std::vector<std::shared_ptr<Matcher>>& matchers) {
  for (auto i = 0; i < exprs.size(); ++i) {
    if (!matchers[i]->match(exprs[i])) {
      return false;
    }
  }
  return true;
}

bool IfMatcher::match(const core::TypedExprPtr& expr) {
  if (auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    if (call->name() == expression::kIf &&
        allMatch(call->inputs(), inputMatchers_)) {
      return true;
    }
  }
  return false;
}

bool ComparisonMatcher::match(const core::TypedExprPtr& expr) {
  if (auto call = dynamic_cast<const core::CallTypedExpr*>(expr.get())) {
    const auto& name = call->name();
    if (exprNameMatch(name)) {
      if (allMatch(call->inputs(), inputMatchers_)) {
        *op_ = name;
        return true;
      }
    }
  }
  return false;
}

bool AnySingleInputMatcher::match(const core::TypedExprPtr& expr) {
  // Check if 'expr' depends on a single column.
  std::unordered_set<core::FieldAccessTypedExprPtr> inputs;
  collectInputs(expr, inputs);

  if (inputs.size() == 1) {
    *expr_ = expr;
    *input_ = *inputs.begin();
    return true;
  }

  return false;
}

void AnySingleInputMatcher::collectInputs(
    const core::TypedExprPtr& expr,
    std::unordered_set<core::FieldAccessTypedExprPtr>& inputs) {
  if (auto field =
          std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr)) {
    if (field->isInputColumn()) {
      inputs.insert(field);
      return;
    }
  }

  for (const auto& input : expr->inputs()) {
    collectInputs(input, inputs);
  }
}

bool ComparisonConstantMatcher::match(const core::TypedExprPtr& expr) {
  if (auto constant = asConstant(expr.get())) {
    auto v = constant.value();
    if (v == 0 || v == 1 || v == -1) {
      *value_ = v;
      return true;
    }
  }
  return false;
}

std::optional<int64_t> ComparisonConstantMatcher::asConstant(
    const core::ITypedExpr* expr) {
  if (auto constant = dynamic_cast<const core::ConstantTypedExpr*>(expr)) {
    if (!constant->isNull()) {
      if (constant->hasValueVector()) {
        const auto& vector = constant->valueVector();
        if (constant->type()->isBigint()) {
          return vector->as<SimpleVector<int64_t>>()->valueAt(0);
        } else if (constant->type()->isInteger()) {
          return vector->as<SimpleVector<int32_t>>()->valueAt(0);
        }
      } else {
        if (constant->value().kind() == TypeKind::BIGINT) {
          return constant->value().value<int64_t>();
        }

        if (constant->value().kind() == TypeKind::INTEGER) {
          return constant->value().value<int32_t>();
        }
      }
    }
  }

  return std::nullopt;
}

bool SimpleComparisonChecker::isLessThen(
    const std::string& prefix,
    const std::string& operation,
    const core::FieldAccessTypedExprPtr& left,
    int64_t result,
    const std::string& inputLeft) {
  std::string op =
      (left->name() == inputLeft) ? operation : invert(prefix, operation);

  if (op == ltName(prefix)) {
    return result < 0;
  }

  return result > 0;
}

std::optional<SimpleComparison> SimpleComparisonChecker::isSimpleComparison(
    const std::string& prefix,
    const core::LambdaTypedExpr& expr) {
  // Spark 4 expands the default array_sort comparator into null guards around
  // an otherwise simple comparison:
  //   if (isnull(x) and isnull(y), 0,
  //     if (isnull(x), 1, if (isnull(y), -1, compare(x, y))))
  // Strip that exact nulls-last wrapper. The rewritten native array_sort has
  // the same null ordering, while avoiding execution of a comparator lambda.
  auto asCall = [](const core::TypedExprPtr& value,
                   const std::string& name,
                   size_t arity) -> const core::CallTypedExpr* {
    const auto* call =
        dynamic_cast<const core::CallTypedExpr*>(value.get());
    return call != nullptr && call->name() == name &&
            call->inputs().size() == arity
        ? call
        : nullptr;
  };
  auto isField = [](const core::TypedExprPtr& value,
                    const std::string& name) {
    const auto* field =
        dynamic_cast<const core::FieldAccessTypedExpr*>(value.get());
    return field != nullptr && field->isInputColumn() &&
        field->name() == name;
  };
  auto isNullOf = [&](const core::TypedExprPtr& value,
                      const std::string& name) {
    const auto* call = asCall(value, prefix + "isnull", 1);
    return call != nullptr && isField(call->inputs()[0], name);
  };
  auto isConstant = [](const core::TypedExprPtr& value, int64_t expected) {
    int64_t actual = 0;
    ComparisonConstantMatcher matcher(&actual);
    return matcher.match(value) && actual == expected;
  };

  if (expr.signature()->size() == 2) {
    const auto& left = expr.signature()->nameOf(0);
    const auto& right = expr.signature()->nameOf(1);
    const auto* outer = asCall(expr.body(), expression::kIf, 3);
    if (outer != nullptr) {
      const auto* bothNull =
          asCall(outer->inputs()[0], expression::kAnd, 2);
      const auto* leftNull =
          asCall(outer->inputs()[2], expression::kIf, 3);
      const auto* rightNull = leftNull == nullptr
          ? nullptr
          : asCall(leftNull->inputs()[2], expression::kIf, 3);
      if (bothNull != nullptr && leftNull != nullptr && rightNull != nullptr &&
          isNullOf(bothNull->inputs()[0], left) &&
          isNullOf(bothNull->inputs()[1], right) &&
          isConstant(outer->inputs()[1], 0) &&
          isNullOf(leftNull->inputs()[0], left) &&
          isConstant(leftNull->inputs()[1], 1) &&
          isNullOf(rightNull->inputs()[0], right) &&
          isConstant(rightNull->inputs()[1], -1)) {
        core::LambdaTypedExpr unwrapped(expr.signature(), rightNull->inputs()[2]);
        return isSimpleComparison(prefix, unwrapped);
      }
    }
  }

  // First, check the shape of the expression.
  // if (x(a) < y(b), c1, if (u(c) > v(d), c2, c3))
  core::FieldAccessTypedExprPtr a, b, c, d;
  core::TypedExprPtr x, y, u, v;
  std::string op1, op2;
  int64_t c1, c2, c3;

  auto matcher = ifelse(
      comparison(prefix, anySingleInput(&x, &a), anySingleInput(&y, &b), &op1),
      comparisonConstant(&c1),
      ifelse(
          comparison(
              prefix, anySingleInput(&u, &c), anySingleInput(&v, &d), &op2),
          comparisonConstant(&c2),
          comparisonConstant(&c3)));

  if (!matcher->match(expr.body())) {
    return std::nullopt;
  }

  // Verify that a != b, c != d.
  if (a == b || c == d) {
    return std::nullopt;
  }

  // Verify that x, y, u, v are the same (except for input column).
  std::unordered_map<std::string, core::TypedExprPtr> inputMapping;
  inputMapping.emplace(
      a->name(),
      std::make_shared<core::FieldAccessTypedExpr>(a->type(), b->name()));
  const auto xRewritten = x->rewriteInputNames(inputMapping);
  if (!(*xRewritten == *y->rewriteInputNames(inputMapping) &&
        *xRewritten == *u->rewriteInputNames(inputMapping) &&
        *xRewritten == *v->rewriteInputNames(inputMapping))) {
    return std::nullopt;
  }

  // Verify all constants are different.
  if (c1 == c2 || c2 == c3 || c1 == c3) {
    return std::nullopt;
  }

  const auto eq = eqName(prefix);

  // Verify that equality comparisons return zero.
  // if (x(a) = y(a), 0,..) is good. if (x(a) = y(a), 1,..) is not good.
  // Also, verify that non-equality comparisons return non-zero.
  // if (x(a) < y(a), 1,..) is good. if (x(a) < y(a), 0,..) is not good.
  if ((op1 == eq && c1 != 0) || (op1 != eq && c1 == 0)) {
    return std::nullopt;
  }

  if ((op2 == eq && c2 != 0) || (op2 != eq && c2 == 0)) {
    return std::nullopt;
  }

  const auto left = expr.signature()->nameOf(0);

  const auto transform = a->name() == left ? x : y;

  if (op1 == eq) {
    // if (x(a) = y(b), 0,...)
    return {{transform, isLessThen(prefix, op2, c, c2, left)}};
  }

  if (op2 == eq) {
    return {{transform, isLessThen(prefix, op1, a, c1, left)}};
  }

  // Make sure op1 and op2 are aligned.
  auto b1 = isLessThen(prefix, op1, a, c1, left);
  auto b2 = isLessThen(prefix, op2, c, c2, left);
  if (b1 != b2) {
    return std::nullopt;
  }

  return {{transform, b1}};
}

} // namespace facebook::velox::functions
