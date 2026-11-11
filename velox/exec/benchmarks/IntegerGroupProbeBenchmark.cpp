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

#include <cstdint>
#include <memory>
#include <vector>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/exec/HashTable.h"
#include "velox/exec/VectorHasher.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {
// Measures generic-mode group probing separately from input decoding/hashing.
class IntegerGroupProbeBenchmark
    : public facebook::velox::test::VectorTestBase {
 public:
  void run(int32_t numKeys, bool nullable, bool hitsOnly) {
    folly::BenchmarkSuspender suspender;
    constexpr vector_size_t kNumRows = 65'536;
    constexpr vector_size_t kBatchSize = 1024;
    const vector_size_t numDistinct = hitsOnly ? kNumRows : kNumRows / 4;
    std::vector<VectorPtr> columns;
    std::vector<std::unique_ptr<VectorHasher>> hashers;
    for (int32_t key = 0; key < numKeys; ++key) {
      const auto isNull = [nullable](auto row) {
        return nullable && row % 11 == 0;
      };
      if (key % 2 == 0) {
        columns.push_back(
            makeFlatVector<int32_t>(
                kNumRows,
                [numDistinct](auto row) { return row % numDistinct; },
                isNull));
      } else {
        columns.push_back(
            makeFlatVector<int64_t>(
                kNumRows,
                [numDistinct](auto row) {
                  return int64_t(row % numDistinct) * 1'000'000'007;
                },
                isNull));
      }
      hashers.push_back(
          std::make_unique<VectorHasher>(columns.back()->type(), key));
    }
    auto input = makeRowVector(columns);
    auto table = HashTable<false>::createForAggregation(
        std::move(hashers), std::vector<Accumulator>{}, pool());
    table->forceGenericHashMode(BaseHashTable::kNoSpillInputStartPartitionBit);
    HashLookup lookup(table->hashers(), pool());
    lookup.reset(kNumRows);
    const SelectivityVector rows(kNumRows);
    for (int32_t key = 0; key < numKeys; ++key) {
      table->hashers()[key]->decode(*input->childAt(key), rows);
      table->hashers()[key]->hash(rows, key > 0, lookup.hashes);
    }
    const auto probe = [&]() {
      for (vector_size_t begin = 0; begin < kNumRows; begin += kBatchSize) {
        lookup.rows.clear();
        lookup.newGroups.clear();
        for (vector_size_t row = begin; row < begin + kBatchSize; ++row) {
          lookup.rows.push_back(row);
        }
        table->groupProbe(
            lookup, BaseHashTable::kNoSpillInputStartPartitionBit);
      }
    };
    if (hitsOnly) {
      probe();
    }
    suspender.dismiss();
    probe();
    folly::doNotOptimizeAway(table->numDistinct());
    suspender.rehire();
    table->checkConsistency();
  }
};
} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  IntegerGroupProbeBenchmark benchmark;
  for (const auto numKeys : {1, 2, 8}) {
    for (const bool nullable : {false, true}) {
      for (const bool hitsOnly : {false, true}) {
        const auto name = fmt::format(
            "{}Keys_{}_{}",
            numKeys,
            nullable ? "Nullable" : "NotNull",
            hitsOnly ? "ExistingGroups" : "NewAndDuplicateGroups");
        folly::addBenchmark(__FILE__, name, [&, numKeys, nullable, hitsOnly]() {
          benchmark.run(numKeys, nullable, hitsOnly);
          return 1;
        });
      }
    }
  }
  folly::runBenchmarks();
}
