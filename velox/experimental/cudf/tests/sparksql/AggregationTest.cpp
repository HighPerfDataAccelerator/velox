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
#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/exec/SparkAggregateFunctions.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"

#include "velox/common/base/BloomFilter.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/vector/SimpleVector.h"

#include <folly/ScopeGuard.h>
#include <folly/hash/Hash.h>

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::exec::sparksql::test {

class AggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    functions::sparksql::registerFunctions("");
    functions::aggregate::sparksql::registerAggregateFunctions("");
    filesystems::registerLocalFileSystem();
    // After register supports function prefix, we could register the function
    // with spark_ to align with sparksql AverageAggregationTest, the
    // function name overwrite may not work well in some condition.
    cudf_velox::registerCudf();
    cudf_velox::registerSparkFunctions("");
    cudf_velox::registerSparkAggregateFunctions("");
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::unregisterAggregateFunctions();
  }

  VectorPtr getSerializedBloomFilter(int32_t capacity) {
    BloomFilter<> bloomFilter;
    bloomFilter.reset(capacity);
    for (auto i = 0; i < 9; ++i) {
      bloomFilter.insert(folly::hasher<int64_t>()(i));
    }
    std::string data;
    data.resize(bloomFilter.serializedSize());
    bloomFilter.serialize(data.data());
    return makeConstant(StringView(data), 1, VARBINARY());
  }
};

TEST_F(AggregationTest, sumReal) {
  // spark sum:
  // exec::AggregateFunctionSignatureBuilder()
  // .returnType("double")
  // .intermediateType("double")
  // .argumentType("real")
  // .build(),
  // presto sum:
  // exec::AggregateFunctionSignatureBuilder()
  //       .returnType("real")
  //       .intermediateType("double")
  //       .argumentType("real")
  //       .build(),
  // The sum(real) final result type is different.
  auto data = makeRowVector({makeFlatVector<float>({3.4028, 3.1})});
  auto vectors = {data};
  auto plan = PlanBuilder()
                  .values(vectors)
                  .partialAggregation({}, {"sum(c0)"})
                  .finalAggregation()
                  .planNode();
  auto expected = makeRowVector({"c0"}, {makeConstant<double>(6.5028, 1)});
  assertQuery(plan, expected);
}

TEST_F(AggregationTest, groupedCollectListOfRow) {
  const auto makeBatch = [&](std::vector<int32_t> keys,
                             std::vector<int64_t> ids,
                             std::vector<std::optional<std::string>> labels,
                             std::function<bool(vector_size_t)> isRowNull) {
    auto rows = makeRowVector(
        {"id", "label"},
        {makeFlatVector<int64_t>(ids),
         makeNullableFlatVector<std::string>(labels)},
        std::move(isRowNull));
    return makeRowVector({"k", "s"}, {makeFlatVector<int32_t>(keys), rows});
  };

  // Null ROW values are ignored, nested field nulls are preserved, and
  // duplicate non-null ROW values are retained. Key 3 has only null input
  // ROWs and must therefore produce an empty ARRAY<ROW>.
  auto batch1 = makeBatch(
      {1, 1, 2, 2, 3},
      {10, -1, 20, 20, -1},
      {"a", "ignored", "x", "x", "ignored"},
      [](auto row) { return row == 1 || row == 4; });
  auto batch2 = makeBatch(
      {1, 2, 3, 3},
      {11, 21, -1, -1},
      {std::nullopt, "y", "ignored", "ignored"},
      [](auto row) { return row == 2 || row == 3; });

  auto expectedRows = makeRowVector(
      {"id", "label"},
      {makeFlatVector<int64_t>({10, 11, 20, 20, 21}),
       makeNullableFlatVector<std::string>(
           {"a", std::nullopt, "x", "x", "y"})});
  auto expected = makeRowVector(
      {"k", "items"},
      {makeFlatVector<int32_t>({1, 2, 3}),
       makeArrayVector({0, 2, 5, 5}, expectedRows)});

  auto single = PlanBuilder()
                    .values({batch1, batch2})
                    .singleAggregation({"k"}, {"collect_list(s) AS items"})
                    .planNode();
  assertQuery(single, expected);

  // Keep two input vectors so partial aggregation consumes multiple batches.
  // Final aggregation must merge its ARRAY<ROW> state without dropping
  // duplicates, nested nulls, or the empty all-null group.
  auto partialFinal =
      PlanBuilder()
          .values({batch1, batch2})
          .partialAggregation({"k"}, {"collect_list(s) AS items"})
          .finalAggregation()
          .planNode();
  assertQuery(partialFinal, expected);
}

TEST_F(AggregationTest, bloomFilterAgg) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto previousFallback = cudfConfig.allowCpuFallback;
  auto restoreFallback =
      folly::makeGuard([&] { cudfConfig.allowCpuFallback = previousFallback; });
  cudfConfig.allowCpuFallback = false;
  auto vectors = makeRowVector({makeFlatVector<int64_t>(
      100, [](vector_size_t row) { return row % 9; })});
  auto expected = makeRowVector({getSerializedBloomFilter(11)});
  auto plan = PlanBuilder()
                  .values({vectors})
                  .partialAggregation({}, {"bloom_filter_agg(c0, 5, 64)"})
                  .finalAggregation()
                  .planNode();
  assertQuery(plan, expected);

  auto single = PlanBuilder()
                    .values({vectors})
                    .singleAggregation({}, {"bloom_filter_agg(c0, 5, 64)"})
                    .planNode();
  assertQuery(single, expected);
}

TEST_F(AggregationTest, bloomFilterAggFromXxHash64) {
  auto& cudfConfig = cudf_velox::CudfConfig::getInstance();
  const auto previousFallback = cudfConfig.allowCpuFallback;
  auto restoreFallback =
      folly::makeGuard([&] { cudfConfig.allowCpuFallback = previousFallback; });
  cudfConfig.allowCpuFallback = false;
  auto keys = makeFlatVector<int64_t>({1, 2, 3, 4, 5, 1, 2});
  auto input = makeRowVector({keys});
  auto plan = PlanBuilder()
                  .values({input})
                  .project({"xxhash64_with_seed(cast(42 as bigint), c0) AS h"})
                  .singleAggregation({}, {"bloom_filter_agg(h, 5, 64)"})
                  .planNode();
  auto gpu = AssertQueryBuilder(plan).copyResults(pool());
  ASSERT_EQ(gpu->size(), 1);
  ASSERT_FALSE(gpu->childAt(0)->isNullAt(0));

  auto containPlan =
      PlanBuilder()
          .values({input})
          .project({"xxhash64_with_seed(cast(42 as bigint), c0) AS h"})
          .planNode();
  auto hashes = AssertQueryBuilder(containPlan).copyResults(pool());
  auto serialized = gpu->childAt(0)->as<SimpleVector<StringView>>()->valueAt(0);
  std::string data(serialized.data(), serialized.size());
  for (auto i = 0; i < hashes->size(); ++i) {
    auto hash = hashes->childAt(0)->as<SimpleVector<int64_t>>()->valueAt(i);
    ASSERT_TRUE(
        BloomFilter<>::mayContain(
            data.c_str(), folly::hasher<int64_t>()(hash)));
  }
  ASSERT_FALSE(
      BloomFilter<>::mayContain(
          data.c_str(), folly::hasher<int64_t>()(int64_t{123456789})));
}

} // namespace facebook::velox::exec::sparksql::test
