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

#include "velox/functions/sparksql/MightContainBatch.h"

#include <random>

#include "velox/common/base/BloomFilter.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/sparksql/registration/Register.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

std::string serializedFilter(
    int32_t capacity,
    const std::vector<int64_t>& keys) {
  BloomFilter filter;
  filter.reset(capacity);
  for (auto key : keys) {
    filter.insert(folly::hasher<int64_t>()(key));
  }
  std::string data;
  data.resize(filter.serializedSize());
  filter.serialize(data.data());
  return data;
}

class MightContainBatchTest : public testing::Test,
                              public velox::test::VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerFunctions("");
  }

  // Evaluates 'name' with a constant (or NULL, or column) filter argument.
  VectorPtr evaluateWith(
      const std::string& name,
      const std::optional<std::string>& serialized,
      const RowVectorPtr& data,
      const SelectivityVector& rows,
      bool filterColumn = false) {
    std::vector<core::TypedExprPtr> args;
    if (filterColumn) {
      args.push_back(
          std::make_shared<core::FieldAccessTypedExpr>(VARBINARY(), "c1"));
    } else if (serialized.has_value()) {
      args.push_back(std::make_shared<core::ConstantTypedExpr>(
          VARBINARY(), variant::binary(serialized.value())));
    } else {
      args.push_back(std::make_shared<core::ConstantTypedExpr>(
          VARBINARY(), variant::null(TypeKind::VARBINARY)));
    }
    args.push_back(
        std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "c0"));
    exec::ExprSet expr(
        {std::make_shared<core::CallTypedExpr>(BOOLEAN(), args, name)},
        &execCtx_);
    exec::EvalCtx context(&execCtx_, &expr, data.get());
    std::vector<VectorPtr> results(1);
    expr.eval(rows, context, results);
    return results[0];
  }

  std::shared_ptr<core::QueryCtx> queryCtx_{core::QueryCtx::create()};
  core::ExecCtx execCtx_{pool_.get(), queryCtx_.get()};

  // Compares the batch function with the scalar simple function on 'rows'.
  void assertSameAsScalar(
      const std::optional<std::string>& serialized,
      const RowVectorPtr& data,
      const SelectivityVector& rows,
      bool filterColumn = false) {
    auto expected =
        evaluateWith("might_contain", serialized, data, rows, filterColumn);
    for (const auto* name :
         {"might_contain_batch", "might_contain_batch_sve"}) {
      auto actual = evaluateWith(name, serialized, data, rows, filterColumn);
      rows.applyToSelected([&](vector_size_t row) {
        ASSERT_EQ(expected->isNullAt(row), actual->isNullAt(row))
            << "row " << row << " " << name;
        if (!expected->isNullAt(row)) {
          ASSERT_TRUE(expected->equalValueAt(actual.get(), row, row))
              << "row " << row << " " << name;
        }
      });
    }
  }
};

TEST_F(MightContainBatchTest, kernelsMatchScalarProbe) {
  std::mt19937_64 random(7);
  for (int32_t capacity : {1, 7, 1'000, 100'000, 4'000'000}) {
    std::vector<int64_t> keys(std::min(capacity, 50'000));
    for (auto& key : keys) {
      key = static_cast<int64_t>(random());
    }
    const auto data = serializedFilter(capacity, keys);
    const BloomFilterView view(data.data());
    for (int32_t size : {0, 1, 2, 3, 31, 32, 33, 1'023, 4'097}) {
      std::vector<int64_t> values(size);
      for (int32_t i = 0; i < size; ++i) {
        values[i] = i % 3 == 0 && !keys.empty()
            ? keys[random() % keys.size()]
            : static_cast<int64_t>(random()) >> (i % 7);
      }
      if (size > 4) {
        values[1] = std::numeric_limits<int64_t>::min();
        values[2] = std::numeric_limits<int64_t>::max();
        values[3] = -1;
        values[4] = 0;
      }
      std::vector<uint8_t> generic(size + 1, 9), sve(size + 1, 9);
      bloom_probe::probeGeneric(
          view.bitsData(), view.size(), values.data(), size, generic.data());
      bloom_probe::probeSve(
          view.bitsData(), view.size(), values.data(), size, sve.data());
      for (int32_t i = 0; i < size; ++i) {
        const bool expected =
            view.mayContain(folly::hasher<int64_t>()(values[i]));
        ASSERT_EQ(generic[i], expected) << capacity << " " << size << " " << i;
        ASSERT_EQ(sve[i], expected) << capacity << " " << size << " " << i;
        ASSERT_EQ(
            bloom_probe::test(view.bitsData(), view.size(), values[i]),
            expected);
      }
      // Neither kernel writes past the requested size.
      ASSERT_EQ(generic[size], 9);
      ASSERT_EQ(sve[size], 9);
    }
  }
}

TEST_F(MightContainBatchTest, flatNullableAndSelectedRows) {
  std::vector<int64_t> keys;
  for (int64_t i = -500; i < 2'000; i += 3) {
    keys.push_back(i);
  }
  const auto serialized = serializedFilter(4'096, keys);
  const vector_size_t size = 5'000;
  auto flat = makeFlatVector<int64_t>(
      size, [](vector_size_t row) { return row - 1'000; });
  auto nullable = makeFlatVector<int64_t>(
      size,
      [](vector_size_t row) { return row * 7 - 3'000; },
      [](vector_size_t row) { return row % 11 == 0; });
  for (const auto& values : {flat, nullable}) {
    auto data = makeRowVector({values});
    SelectivityVector all(size);
    assertSameAsScalar(serialized, data, all);
    SelectivityVector partial(size);
    for (vector_size_t row = 0; row < size; row += 3) {
      partial.setValid(row, false);
    }
    partial.updateBounds();
    assertSameAsScalar(serialized, data, partial);
  }
}

TEST_F(MightContainBatchTest, dictionaryAndConstantInputs) {
  const auto serialized = serializedFilter(64, {1, 2, 3, 40, 50});
  auto base = makeNullableFlatVector<int64_t>({1, 2, std::nullopt, 40, 7, 50});
  auto indices =
      makeIndices(3'000, [](vector_size_t row) { return (row * 5) % 6; });
  auto dictionary = wrapInDictionary(indices, 3'000, base);
  assertSameAsScalar(
      serialized, makeRowVector({dictionary}), SelectivityVector(3'000));
  auto constant = makeConstant<int64_t>(40, 100);
  assertSameAsScalar(
      serialized, makeRowVector({constant}), SelectivityVector(100));
}

TEST_F(MightContainBatchTest, nullAndNonConstantFilters) {
  auto values = makeNullableFlatVector<int64_t>({1, std::nullopt, 3});
  // NULL filter: every row NULL.
  assertSameAsScalar(
      std::nullopt, makeRowVector({values}), SelectivityVector(3));
  // A filter column is not constant: non-NULL rows are false, as in the
  // simple function.
  const auto serialized = serializedFilter(16, {1, 3});
  auto filters = makeFlatVector<StringView>(
      3,
      [&](vector_size_t) { return StringView(serialized); },
      nullptr,
      VARBINARY());
  assertSameAsScalar(
      std::nullopt,
      makeRowVector({values, filters}),
      SelectivityVector(3),
      true);
}

TEST_F(MightContainBatchTest, tooSmallFilterIsUserError) {
  auto data = makeRowVector({makeFlatVector<int64_t>({1})});
  for (const auto* name : {"might_contain_batch", "might_contain_batch_sve"}) {
    VELOX_ASSERT_USER_THROW(
        evaluateWith(name, std::string("ab"), data, SelectivityVector(1)),
        "Serialized BloomFilter is too small");
    auto truncated = serializedFilter(64, {1});
    truncated.pop_back();
    VELOX_ASSERT_USER_THROW(
        evaluateWith(name, truncated, data, SelectivityVector(1)),
        "Truncated serialized BloomFilter");
  }
}

TEST_F(MightContainBatchTest, registeredBesideSimpleFunction) {
  for (const auto* name : {"might_contain_batch", "might_contain_batch_sve"}) {
    auto signatures = exec::getVectorFunctionSignatures(name);
    ASSERT_TRUE(signatures.has_value()) << name;
    EXPECT_EQ(signatures->size(), 1);
  }
  // The original name still resolves to the simple function.
  EXPECT_FALSE(exec::getVectorFunctionSignatures("might_contain").has_value());
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
