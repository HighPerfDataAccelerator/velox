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

#include "velox/experimental/cudf/expression/CommonFunctions.h"
#include "velox/experimental/cudf/expression/DateTruncFunction.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/expression/sparksql/DateAddFunction.h"
#include "velox/experimental/cudf/expression/sparksql/HashFunction.h"
#include "velox/experimental/cudf/expression/sparksql/SubStringFunction.h"

#include "velox/expression/FunctionSignature.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strip.hpp>

#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>

#include <atomic>

namespace facebook::velox::cudf_velox {
namespace {

struct MonotonicallyIncreasingIdState {
  std::atomic<int64_t> nextOffset{0};
};

std::shared_ptr<MonotonicallyIncreasingIdState> monotonicallyIncreasingIdState(
    const core::QueryCtx* queryCtx) {
  if (queryCtx == nullptr) {
    return std::make_shared<MonotonicallyIncreasingIdState>();
  }

  static folly::Synchronized<folly::F14FastMap<
      const core::QueryCtx*,
      std::weak_ptr<MonotonicallyIncreasingIdState>>>
      states;
  return states.withWLock([&](auto& stateMap) {
    auto& weakState = stateMap[queryCtx];
    if (auto state = weakState.lock()) {
      return state;
    }
    auto state = std::make_shared<MonotonicallyIncreasingIdState>();
    weakState = state;
    return state;
  });
}

class MonotonicallyIncreasingIdFunction : public CudfFunction {
 public:
  explicit MonotonicallyIncreasingIdFunction(const core::TypedExprPtr& expr) {
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
    partitionBase_ = static_cast<int64_t>(partitionId) << 33;
    state_ = monotonicallyIncreasingIdState(currentCudfFunctionQueryCtx());
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
    constexpr int64_t kRowsPerPartition = int64_t{1} << 33;
    const auto rowCount = static_cast<int64_t>(inputRowCount);
    const auto startOffset =
        state_->nextOffset.fetch_add(rowCount, std::memory_order_relaxed);
    VELOX_CHECK_GE(startOffset, 0);
    VELOX_CHECK_LE(
        startOffset,
        kRowsPerPartition - rowCount,
        "monotonically_increasing_id exceeded 2^33 rows in one partition");
    cudf::numeric_scalar<int64_t> init(
        partitionBase_ + startOffset, true, stream, mr);
    cudf::numeric_scalar<int64_t> step(1, true, stream, mr);
    return cudf::sequence(inputRowCount, init, step, stream, mr);
  }

 private:
  int64_t partitionBase_{0};
  std::shared_ptr<MonotonicallyIncreasingIdState> state_;
};

class TrimFunction : public CudfFunction {
 public:
  explicit TrimFunction(const core::TypedExprPtr& expr) {
    VELOX_CHECK_EQ(expr->inputs().size(), 1, "trim expects one input");
    VELOX_CHECK(
        expr->inputs()[0]->type()->kind() == TypeKind::VARCHAR,
        "trim input must be VARCHAR");
    VELOX_CHECK(
        expr->type()->kind() == TypeKind::VARCHAR,
        "trim output must be VARCHAR");
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(inputColumns.size(), 1, "trim expects one input column");
    cudf::string_scalar space(" ", true, stream, mr);
    return cudf::strings::strip(
        cudf::strings_column_view(asView(inputColumns[0])),
        cudf::strings::side_type::BOTH,
        space,
        stream,
        mr);
  }
};

void registerSparkArrayAccessFunctions(const std::string& prefix) {
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
}

} // namespace

void registerSparkFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  const std::vector<exec::FunctionSignaturePtr> subStringSignatures{
      FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("integer")
          .build(),
      FunctionSignatureBuilder()
          .returnType("varchar")
          .argumentType("varchar")
          .argumentType("integer")
          .argumentType("integer")
          .build()};
  for (const auto& name : {prefix + "substr", prefix + "substring"}) {
    // Route both spellings to the Spark implementation in cuDF. Presto
    // substring is registered only when Presto functions are registered, so
    // Spark runtimes do not need to override an existing candidate.
    registerCudfFunction(
        name,
        [](const std::string&,
           const core::TypedExprPtr& expr,
           memory::MemoryPool* pool) {
          return sparksql::makeSubStringFunction(expr, pool);
        },
        subStringSignatures);
  }

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<sparksql::HashFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("integer")
           .constantArgumentType("integer")
           .argumentType("any")
           .variableArity("any")
           .build()},
      true,
      sparksql::HashFunction::canEvaluate);

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<sparksql::DateAddFunction>(expr, pool);
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
      prefix + "date_trunc",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool* pool) {
        return std::make_shared<DateTruncFunction>(expr, pool);
      },
      {FunctionSignatureBuilder()
           .returnType("timestamp")
           .constantArgumentType("varchar")
           .argumentType("timestamp")
           .build()},
      true,
      DateTruncFunction::canEvaluate);

  registerCudfFunction(
      prefix + "monotonically_increasing_id",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) {
        return std::make_shared<MonotonicallyIncreasingIdFunction>(expr);
      },
      {FunctionSignatureBuilder().returnType("bigint").build()});

  registerCudfFunction(
      prefix + "trim",
      [](const std::string&,
         const core::TypedExprPtr& expr,
         memory::MemoryPool*) { return std::make_shared<TrimFunction>(expr); },
      {FunctionSignatureBuilder()
           .returnType("varchar")
           .argumentType("varchar")
           .build()});

  registerSparkArrayAccessFunctions(prefix);
}

} // namespace facebook::velox::cudf_velox
