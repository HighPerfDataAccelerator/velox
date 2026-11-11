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

#include <algorithm>
#include <numeric>
#include <random>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>
#include <folly/lang/Bits.h>

#include "velox/exec/HashTable.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {
class JoinChainPrefetchBenchmark
    : public facebook::velox::test::VectorTestBase {
 public:
  void run(bool generic, int32_t rowsPerKey, int32_t payloadColumns) {
    folly::BenchmarkSuspender suspender;
    constexpr vector_size_t kNumRows = 65'536;
    const auto numKeys = kNumRows / rowsPerKey;
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    std::vector<VectorPtr> columns;
    for (int32_t key = 0; key < 2; ++key) {
      hashers.push_back(std::make_unique<VectorHasher>(BIGINT(), key));
      columns.push_back(makeFlatVector<int64_t>(kNumRows, [=](auto row) {
        const auto value = row % numKeys;
        return int64_t(key == 0 ? value : value * 7 % numKeys) * 1000;
      }));
    }
    for (int32_t column = 0; column < payloadColumns; ++column) {
      columns.push_back(
          makeFlatVector<int64_t>(kNumRows, [](auto row) { return row; }));
    }
    auto table = HashTable<true>::createForJoin(
        std::move(hashers),
        std::vector<TypePtr>(payloadColumns, BIGINT()),
        true,
        false,
        false,
        1'000,
        pool());
    SelectivityVector allRows(kNumRows);
    raw_vector<uint64_t> valueIds(kNumRows);
    std::vector<DecodedVector> decoded(columns.size());
    for (int32_t column = 0; column < columns.size(); ++column) {
      decoded[column].decode(*columns[column], allRows);
      if (column < 2) {
        auto& hasher = table->hashers()[column];
        hasher->decode(*columns[column], allRows);
        hasher->computeValueIds(allRows, valueIds);
      }
    }
    std::vector<vector_size_t> order(kNumRows);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 randomEngine(51);
    std::shuffle(order.begin(), order.end(), randomEngine);
    auto* rows = table->rows();
    std::vector<char*> storedRows;
    storedRows.reserve(kNumRows);
    for (const auto index : order) {
      auto* row = rows->newRow();
      storedRows.push_back(row);
      folly::storeUnaligned<char*>(
          row + rows->nextOffset(), static_cast<char*>(nullptr));
      for (int32_t column = 0; column < columns.size(); ++column) {
        rows->store(decoded[column], index, row, column);
      }
    }
    if (generic) {
      table->prepareJoinTable(
          {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);
      // Rebuilding requires unlinked rows, as with a fresh join build.
      for (auto* row : storedRows) {
        folly::storeUnaligned<char*>(
            row + rows->nextOffset(), static_cast<char*>(nullptr));
      }
    }
    // Time normalized-key construction or generic-mode rebuilding, including
    // hashing and insertion. Exclude input preparation and destruction.
    suspender.dismiss();
    if (generic) {
      table->forceGenericHashMode(
          BaseHashTable::kNoSpillInputStartPartitionBit);
    } else {
      table->prepareJoinTable(
          {}, BaseHashTable::kNoSpillInputStartPartitionBit, 1'000'000);
    }
    folly::doNotOptimizeAway(table->numDistinct());
    suspender.rehire();
    VELOX_CHECK_EQ(table->rows()->numRows(), kNumRows);
    VELOX_CHECK_EQ(table->hasDuplicateKeys(), rowsPerKey > 1);
    VELOX_CHECK_EQ(
        table->hashMode(),
        generic ? BaseHashTable::HashMode::kHash
                : BaseHashTable::HashMode::kNormalizedKey);
    // Join builds with duplicate keys retain the input row count in
    // numDistinct_, which the slot-count consistency check does not support.
    if (rowsPerKey == 1) {
      table->checkConsistency();
    }
  }
};
} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  JoinChainPrefetchBenchmark benchmark;
  for (const bool generic : {false, true}) {
    for (const auto rowsPerKey : {1, 4, 16}) {
      for (const auto payloadColumns : {0, 16}) {
        const auto name = fmt::format(
            "{}_{}RowsPerKey_{}PayloadColumns",
            generic ? "Hash" : "NormalizedKey",
            rowsPerKey,
            payloadColumns);
        folly::addBenchmark(
            __FILE__, name, [&, generic, rowsPerKey, payloadColumns]() {
              benchmark.run(generic, rowsPerKey, payloadColumns);
              return 1;
            });
      }
    }
  }
  folly::runBenchmarks();
}
