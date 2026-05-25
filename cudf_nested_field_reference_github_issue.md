# Suggested title

cuDF FilterProject fails to evaluate nested FieldReference over ROW/STRUCT columns

## Bug description

Velox cuDF `FilterProject` can fail when a `ProjectNode` dereferences fields from an input `ROW` column. The expression is a nested `velox::exec::FieldReference`, for example:

```text
avg_state.col_0
avg_state.col_1
```

where `avg_state` is a top-level input column of type:

```text
ROW<col_0:DOUBLE,col_1:BIGINT>
```

The evaluator should first evaluate the parent field `avg_state`, then return the corresponding child from the cuDF `STRUCT` column. Instead, the cuDF function-expression path resolves `col_0` against the top-level input row schema and fails because `col_0` is not a top-level column.

Expected behavior: cuDF `FilterProject` should correctly evaluate nested field references over input `ROW`/STRUCT columns, including null propagation from the parent struct.

Actual behavior: cuDF `FilterProject` may throw:

```text
Reason: Field not found: col_0. Available fields are: <top-level input fields>
Function: getChildIdx
```

## General example

A minimal plan shape that demonstrates the issue is:

```text
Project[
  avg_state.col_0 AS sum,
  avg_state.col_1 AS count
]
  Values[
    avg_state: ROW<col_0:DOUBLE,col_1:BIGINT>
  ]
```

For input:

```text
avg_state = [
  {col_0: 1.5,  col_1: 10},
  {col_0: null, col_1: 20},
  null,
  {col_0: 4.5,  col_1: 40}
]
```

The expected output is:

```text
sum   = [1.5, null, null, 4.5]
count = [10, 20, null, 40]
```

The third row verifies that a null parent struct makes all extracted child values null.

A similar shape can also appear when a plan projects fields out of an aggregate intermediate state, e.g. `avg_partial(...)` returning `ROW<sum,count>` followed by a project that flattens this state into separate columns. The problem is not specific to aggregate functions, though; any `ProjectNode` that dereferences an input `ROW` column can hit the same path.

## Reduced unit test

A focused unit test can be added to `velox/experimental/cudf/tests/FilterProjectTest.cpp`:

```cpp
TEST_F(CudfFilterProjectTest, dereferenceInputStruct) {
  auto structVector = makeRowVector(
      {"col_0", "col_1"},
      {makeNullableFlatVector<double>({1.5, std::nullopt, 3.5, 4.5}),
       makeFlatVector<int64_t>({10, 20, 30, 40})},
      [](auto row) { return row == 2; });
  auto data = makeRowVector({"avg_state"}, {structVector});

  auto plan = PlanBuilder()
                  .values({data})
                  .project(
                      {"avg_state.col_0 AS sum", "avg_state.col_1 AS count"})
                  .planNode();

  auto expected = makeRowVector({
      makeNullableFlatVector<double>({1.5, std::nullopt, std::nullopt, 4.5}),
      makeNullableFlatVector<int64_t>({10, 20, std::nullopt, 40}),
  });
  AssertQueryBuilder(plan).assertResults(expected);
}
```

## Relevant log shape

When this fails in a larger plan, the plan contains a project similar to:

```text
-- Project[...][expressions:
   (..., "n102_6"["col_0"]),
   (..., "n102_6"["col_1"])
]
  -- SomeInput[...] -> ..., n102_6:ROW<col_0:DOUBLE,col_1:BIGINT>, ...
```

The exception comes from resolving `col_0` as if it were a top-level field:

```text
Exception: VeloxUserError
Error Source: USER
Error Code: INVALID_ARGUMENT
Reason: Field not found: col_0. Available fields are: n101_10, n101_11, n102_2, n102_3, n102_4, n102_5, n102_6, n102_7, n102_8, n102_9.
Retriable: False
Function: getChildIdx
File: velox/type/Type.cpp
Stack trace:
#3  facebook::velox::RowType::getChildIdx(...)
#4  facebook::velox::cudf_velox::FunctionExpression::eval(...)
#5  facebook::velox::cudf_velox::CudfFilterProject::project(...)
#6  facebook::velox::cudf_velox::CudfFilterProject::doGetOutput()
#7  facebook::velox::cudf_velox::CudfOperatorBase::getOutput()
```

There may also be warnings from the AST-expression support check such as:

```text
AstExpressionUtils.h: Field col_0 not found, in expression (n102_6).col_0
```

This warning is misleading for nested references because `col_0` is not expected to be a top-level input column.

## Suspected root cause

`FunctionExpression::canEvaluate()` returns true for all `velox::exec::FieldReference` expressions. However, the current function-expression implementation only handles top-level references correctly:

```text
top_level_field
```

For nested references:

```text
parent_row.child_field
```

`FunctionExpression::create()` does not compile the parent expression, and `FunctionExpression::eval()` resolves `child_field` directly against `inputRowSchema_`. This calls `RowType::getChildIdx("child_field")` on the top-level input schema instead of on the parent row type.

The AST-expression support check can also reject nested `FieldReference` expressions as if the child field should exist in the top-level input schema. If the expression then falls through to `FunctionExpression`, the crash above occurs.

## Candidate fix

One possible fix is:

1. In `FunctionExpression::create()`, when the expression is a `FieldReference`, compile any non-literal input expressions. This captures the parent expression for nested references.
2. In `FunctionExpression::eval()`:
   - If the `FieldReference` has no inputs, keep the existing top-level lookup.
   - If it has one input, evaluate the parent expression.
   - Verify the parent result is a cuDF `STRUCT` and the parent Velox type is `ROW`.
   - Use the parent row type to map `fieldExpr->field()` to a child index.
   - Return `parentView.child(childIndex)`.
   - If the parent struct has nulls, materialize the child and merge parent nulls into the result.
3. In `AstExpressionUtils::detail::isAstExprSupported()`, avoid treating nested field references as failed top-level lookups. Either support nested field references in AST, or return false without logging a misleading warning.

Files likely involved:

```text
velox/experimental/cudf/expression/ExpressionEvaluator.cpp
velox/experimental/cudf/expression/AstExpressionUtils.h
velox/experimental/cudf/tests/FilterProjectTest.cpp
```

## System information

Observed environment:

```text
OS: Linux 6.8.0-79-generic, amd64
Compiler: GCC 14.2.1
Velox build: release, C++20, cuDF enabled
CUDA/cuDF: system CUDA/cuDF build, cuDF headers under /usr/local/include/cudf
```

Please replace this section with `scripts/info.sh` output from the target environment when filing upstream.

