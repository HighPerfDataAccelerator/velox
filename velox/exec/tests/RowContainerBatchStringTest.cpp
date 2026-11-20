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
#include "velox/exec/RowContainer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>

DECLARE_bool(velox_row_container_string_extraction_prefetch_enabled);

namespace facebook::velox::exec::test {
using velox::test::assertEqualVectors;

class RowContainerBatchStringTest : public testing::Test,
                                    public velox::test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  std::unique_ptr<RowContainer> container(const TypePtr& type, bool nullable) {
    return std::make_unique<RowContainer>(
        std::vector<TypePtr>{type},
        nullable,
        std::vector<Accumulator>{},
        std::vector<TypePtr>{},
        false,
        false,
        false,
        false,
        false,
        false,
        pool_.get());
  }

  VectorPtr output(const TypePtr& type, vector_size_t size) {
    auto result =
        BaseVector::create<FlatVector<StringView>>(type, size, pool_.get());
    for (vector_size_t i = 0; i < size; ++i) {
      result->set(i, StringView("untouched-output-value"));
      if (i % 3 == 0) {
        result->setNull(i, true);
      }
    }
    return result;
  }
};

TEST_F(RowContainerBatchStringTest, equalityAndOrdering) {
  gflags::FlagSaver savedFlags;
  std::vector<std::optional<std::string>> values{
      std::nullopt,
      "",
      "a",
      "abcd",
      "abcdefghijkl",
      "abcdefghijklm",
      "abcdefghijklN",
      "different-prefix-long",
      std::string("abcd\0abcdefghi", 14),
      std::string("abcd\0abcdefghj", 14),
      "\xE4\xB8\xAD\xE6\x96\x87-long-key",
      std::string(8192, 'x')};
  for (const auto& type : std::vector<TypePtr>{VARCHAR(), VARBINARY()}) {
    auto data = container(type, true);
    auto input = BaseVector::create<FlatVector<StringView>>(
        type, values.size(), pool_.get());
    for (vector_size_t i = 0; i < values.size(); ++i) {
      if (values[i]) {
        input->set(i, StringView(*values[i]));
      } else {
        input->setNull(i, true);
      }
    }
    SelectivityVector all(values.size());
    DecodedVector decoded(*input, all);
    std::vector<char*> rows(values.size());
    for (vector_size_t i = 0; i < values.size(); ++i) {
      rows[i] = data->newRow();
      data->store(decoded, i, rows[i], 0);
    }
    auto indices = makeIndices(
        values.size(), [&](auto i) { return values.size() - 1 - i; });
    auto dictionary =
        BaseVector::wrapInDictionary(nullptr, indices, values.size(), input);
    auto constant = BaseVector::wrapInConstant(values.size(), 5, input);
    for (const auto& vector :
         std::vector<VectorPtr>{input, dictionary, constant}) {
      DecodedVector encoded(*vector, all);
      for (bool ascending : {false, true}) {
        for (bool nullsFirst : {false, true}) {
          CompareFlags flags{nullsFirst, ascending};
          for (bool equalsOnly : {false, true}) {
            flags.equalsOnly = equalsOnly;
            for (vector_size_t i = 0; i < values.size(); ++i) {
              for (vector_size_t j = 0; j < values.size(); ++j) {
                FLAGS_velox_row_container_string_equality_enabled = false;
                auto legacy = data->compare(
                    rows[i], data->columnAt(0), encoded, j, flags);
                auto legacyStored = data->compare(rows[i], rows[j], 0, flags);
                FLAGS_velox_row_container_string_equality_enabled = true;
                auto fast = data->compare(
                    rows[i], data->columnAt(0), encoded, j, flags);
                auto fastStored = data->compare(rows[i], rows[j], 0, flags);
                if (equalsOnly) {
                  EXPECT_EQ(legacy == 0, fast == 0);
                  EXPECT_EQ(legacyStored == 0, fastStored == 0);
                } else {
                  EXPECT_EQ(legacy, fast);
                  EXPECT_EQ(legacyStored, fastStored);
                }
              }
            }
          }
        }
      }
    }
    // No-null dispatch is used by joins which reject null keys.
    auto equality =
        CompareFlags::equality(CompareFlags::NullHandlingMode::kNullAsValue);
    for (vector_size_t i = 1; i < values.size(); ++i) {
      EXPECT_EQ(
          data->compare<false>(
              rows[i], data->columnAt(0), decoded, i, equality),
          0);
    }
  }
}

TEST_F(RowContainerBatchStringTest, equalityOfFragmentedPayloads) {
  gflags::FlagSaver savedFlags;
  FLAGS_velox_row_container_string_equality_enabled = true;
  for (const auto& type : std::vector<TypePtr>{VARCHAR(), VARBINARY()}) {
    auto data = container(type, false);
    // Exceed even a two-MiB allocator slab so newWrite cannot happen to
    // satisfy the fixture contiguously from a large free block.
    std::string bytes(4 * 1024 * 1024 + 3, 'x');
    bytes[90000] = '\0';
    auto unequal = bytes;
    unequal.back() = 'z';
    auto input =
        BaseVector::create<FlatVector<StringView>>(type, 3, pool_.get());
    input->set(0, StringView(bytes));
    input->set(1, StringView(unequal));
    input->set(2, StringView(bytes));
    SelectivityVector all(3);
    DecodedVector decoded(*input, all);
    auto initial =
        BaseVector::create<FlatVector<StringView>>(type, 1, pool_.get());
    initial->set(0, StringView("inline"));
    SelectivityVector initialRow(1);
    DecodedVector initialDecoded(*initial, initialRow);
    std::vector<char*> rows(3);
    for (int i = 0; i < 3; ++i) {
      rows[i] = data->newRow();
      data->store(initialDecoded, 0, rows[i], 0);
      const auto& source = i == 1 ? unequal : bytes;
      ByteOutputStream stream(&data->stringAllocator(), false, false);
      auto position = data->stringAllocator().newWrite(stream, 32);
      stream.appendStringView(StringView(source));
      data->stringAllocator().finishWrite(stream, 0);
      StringView fragmented(position.position, source.size());
      ASSERT_LT(
          HashStringAllocator::headerOf(fragmented.data())->size(),
          fragmented.size());
      std::memcpy(
          rows[i] + data->columnAt(0).offset(),
          &fragmented,
          sizeof(fragmented));
    }
    auto equality =
        CompareFlags::equality(CompareFlags::NullHandlingMode::kNullAsValue);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_EQ(
            data->compare(rows[i], data->columnAt(0), decoded, j, equality) ==
                0,
            (i == 1) == (j == 1));
        EXPECT_EQ(
            data->compare(rows[i], rows[j], 0, equality) == 0,
            (i == 1) == (j == 1));
      }
    }
    // One contiguous stored string on either side must also be handled.
    auto* contiguous = data->newRow();
    data->store(decoded, 0, contiguous, 0);
    EXPECT_EQ(data->compare(contiguous, rows[0], 0, equality), 0);
    EXPECT_EQ(data->compare(rows[0], contiguous, 0, equality), 0);
  }
}

TEST_F(RowContainerBatchStringTest, offsetsSelectionsNullsAndOwnership) {
  gflags::FlagSaver savedFlags;
  std::vector<VectorPtr> legacyResults, batchResults;
  for (const auto& type : std::vector<TypePtr>{VARCHAR(), VARBINARY()}) {
    for (bool nullable : {false, true}) {
      for (const vector_size_t count : {0, 1, 31, 32, 33, 127, 128, 129, 513}) {
        SCOPED_TRACE(fmt::format(
            "{} nullable={} count={}", type->toString(), nullable, count));
        auto data = container(type, nullable);
        auto input = BaseVector::create<FlatVector<StringView>>(
            type, count, pool_.get());
        const int lengths[] = {0, 3, 12, 13, 15, 16, 17, 31, 32, 33, 127, 1024};
        for (vector_size_t i = 0; i < count; ++i) {
          std::string bytes(lengths[i % std::size(lengths)], '\0');
          for (size_t j = 0; j < bytes.size(); ++j) {
            bytes[j] = static_cast<char>('a' + (i + j) % 26);
          }
          if (bytes.size() > 1) {
            bytes[1] = '\0';
          }
          input->set(i, StringView(bytes));
          if (nullable && i % 7 == 0) {
            input->setNull(i, true);
          }
        }
        SelectivityVector all(count);
        DecodedVector decoded(*input, all);
        std::vector<char*> rows(count);
        for (vector_size_t i = 0; i < count; ++i) {
          rows[i] = data->newRow();
          data->store(decoded, i, rows[i], 0);
        }
        // Null row pointers model an unmatched outer-join build row, including
        // columns that themselves have no null flag.
        for (vector_size_t i = 5; i < count; i += 11) {
          rows[i] = nullptr;
        }
        for (bool indexed : {false, true}) {
          std::vector<vector_size_t> selection(count);
          for (vector_size_t i = 0; i < count; ++i) {
            selection[i] = i % 9 == 0 ? -1 : (count - 1 - i) / 2;
          }
          auto legacy = output(type, count + 9);
          auto batch = output(type, count + 9);
          auto alias = batch->as<FlatVector<StringView>>()->values();
          const auto* aliasValues = alias->as<StringView>();
          auto extract = [&](VectorPtr& result, bool enabled) {
            FLAGS_velox_row_container_batch_string_extraction_enabled = enabled;
            if (indexed) {
              RowContainer::extractColumn(
                  rows.data(),
                  folly::Range<const vector_size_t*>(
                      selection.data(), selection.size()),
                  data->columnAt(0),
                  data->columnHasNulls(0),
                  4,
                  result);
            } else {
              RowContainer::extractColumn(
                  rows.data(),
                  count,
                  data->columnAt(0),
                  data->columnHasNulls(0),
                  4,
                  result);
            }
          };
          extract(legacy, false);
          extract(batch, true);
          assertEqualVectors(legacy, batch);
          // mutableRawValues must preserve any shared copy of the old output.
          if (count > 0) {
            EXPECT_EQ(aliasValues[4], StringView("untouched-output-value"));
          }
          legacyResults.push_back(legacy);
          batchResults.push_back(batch);
          // Reuse with a bitmap containing nulls from the previous extraction.
          extract(legacy, false);
          extract(batch, true);
          assertEqualVectors(legacy, batch);
        }
        data.reset();
        input.reset();
      }
    }
  }
  // Every source container and source vector has been destroyed. All string
  // payloads in the optimized output must still be independently owned.
  for (size_t i = 0; i < legacyResults.size(); ++i) {
    assertEqualVectors(legacyResults[i], batchResults[i]);
  }
}

TEST_F(RowContainerBatchStringTest, fragmentedAndLargeBinaryValues) {
  gflags::FlagSaver savedFlags;
  constexpr vector_size_t kCount = 257;
  auto data = container(VARBINARY(), false);
  auto input = BaseVector::create<FlatVector<StringView>>(
      VARBINARY(), kCount, pool_.get());
  for (vector_size_t i = 0; i < kCount; ++i) {
    input->set(i, StringView("inline"));
  }
  SelectivityVector all(kCount);
  DecodedVector decoded(*input, all);
  std::vector<char*> rows(kCount);
  for (vector_size_t i = 0; i < kCount; ++i) {
    rows[i] = data->newRow();
    data->store(decoded, i, rows[i], 0);
  }
  std::string large(200003, 'x');
  large[0] = '\0';
  large[100000] = '\0';
  large.back() = 'z';
  ByteOutputStream stream(&data->stringAllocator(), false, false);
  auto position = data->stringAllocator().newWrite(stream, 32);
  stream.appendStringView(StringView(large));
  data->stringAllocator().finishWrite(stream, 0);
  StringView fragmented(position.position, large.size());
  ASSERT_LT(
      HashStringAllocator::headerOf(fragmented.data())->size(),
      fragmented.size());
  std::memcpy(
      rows[0] + data->columnAt(0).offset(), &fragmented, sizeof(fragmented));
  for (vector_size_t i = 0; i < kCount; i += 17) {
    rows[i] = rows[0];
  }
  auto legacy = output(VARBINARY(), kCount);
  auto batch = output(VARBINARY(), kCount);
  FLAGS_velox_row_container_batch_string_extraction_enabled = false;
  data->extractColumn(rows.data(), kCount, 0, legacy);
  FLAGS_velox_row_container_batch_string_extraction_enabled = true;
  data->extractColumn(rows.data(), kCount, 0, batch);
  assertEqualVectors(legacy, batch);
  data.reset();
  input.reset();
  assertEqualVectors(legacy, batch);
  EXPECT_EQ(batch->as<FlatVector<StringView>>()->valueAt(0), StringView(large));
}

TEST_F(RowContainerBatchStringTest, allNullRowsAndInlineOnly) {
  gflags::FlagSaver savedFlags;
  constexpr vector_size_t kCount = 256;
  auto data = container(VARCHAR(), false);
  std::vector<char*> rows(kCount, nullptr);
  auto result = output(VARCHAR(), kCount);
  FLAGS_velox_row_container_batch_string_extraction_enabled = true;
  data->extractColumn(rows.data(), kCount, 0, result);
  for (vector_size_t i = 0; i < kCount; ++i) {
    EXPECT_TRUE(result->isNullAt(i));
  }
  auto input =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), 1, pool_.get());
  input->set(0, StringView("inline"));
  SelectivityVector all(1);
  DecodedVector decoded(*input, all);
  auto* row = data->newRow();
  data->store(decoded, 0, row, 0);
  std::fill(rows.begin(), rows.end(), row);
  data->extractColumn(rows.data(), kCount, 0, result);
  for (vector_size_t i = 0; i < kCount; ++i) {
    EXPECT_FALSE(result->isNullAt(i));
    EXPECT_EQ(
        result->as<FlatVector<StringView>>()->valueAt(i), StringView("inline"));
  }
}

TEST_F(RowContainerBatchStringTest, freshNullBitmapAndSharedOutputBuffers) {
  gflags::FlagSaver savedFlags;
  constexpr vector_size_t kCount = 256;
  auto data = container(VARCHAR(), false);
  auto input =
      BaseVector::create<FlatVector<StringView>>(VARCHAR(), 1, pool_.get());
  input->set(0, StringView("a long source string for materialization"));
  SelectivityVector all(1);
  DecodedVector decoded(*input, all);
  auto* row = data->newRow();
  data->store(decoded, 0, row, 0);
  std::vector<char*> rows(kCount, row);
  rows[17] = nullptr;
  VectorPtr result = BaseVector::create<FlatVector<StringView>>(
      VARCHAR(), kCount, pool_.get());
  EXPECT_FALSE(result->nulls());
  FLAGS_velox_row_container_batch_string_extraction_enabled = true;
  data->extractColumn(rows.data(), kCount, 0, result);
  EXPECT_TRUE(result->isNullAt(17));
  EXPECT_FALSE(result->isNullAt(16));

  auto* flat = result->as<FlatVector<StringView>>();
  VectorPtr alias = std::make_shared<FlatVector<StringView>>(
      pool_.get(),
      VARCHAR(),
      result->nulls(),
      kCount,
      flat->values(),
      std::vector<BufferPtr>(flat->stringBuffers()));
  input->set(0, StringView("different long payload for the reused output"));
  data->store(decoded, 0, row, 0);
  rows[17] = row;
  rows[18] = nullptr;
  data->extractColumn(rows.data(), kCount, 0, result);
  data.reset();
  input.reset();
  EXPECT_TRUE(alias->isNullAt(17));
  EXPECT_FALSE(alias->isNullAt(18));
  EXPECT_EQ(
      alias->as<FlatVector<StringView>>()->valueAt(18),
      StringView("a long source string for materialization"));
  EXPECT_FALSE(result->isNullAt(17));
  EXPECT_TRUE(result->isNullAt(18));
  EXPECT_EQ(
      flat->valueAt(17),
      StringView("different long payload for the reused output"));
}
TEST_F(
    RowContainerBatchStringTest,
    prefetchPreservesSelectionsFragmentationAndOwnership) {
  gflags::FlagSaver savedFlags;
  FLAGS_velox_row_container_batch_string_extraction_enabled = true;
  constexpr vector_size_t count = 513;
  for (const auto& type : std::vector<TypePtr>{VARCHAR(), VARBINARY()}) {
    auto data = container(type, true);
    auto input =
        BaseVector::create<FlatVector<StringView>>(type, count, pool_.get());
    for (vector_size_t i = 0; i < count; ++i) {
      std::string bytes(i % 3 == 0 ? 4 : 257, 'a' + i % 20);
      bytes[1] = '\0';
      input->set(i, StringView(bytes));
      if (i % 11 == 0) {
        input->setNull(i, true);
      }
    }
    SelectivityVector all(count);
    DecodedVector decoded(*input, all);
    std::vector<char*> rows(count);
    for (vector_size_t i = 0; i < count; ++i) {
      rows[i] = data->newRow();
      data->store(decoded, i, rows[i], 0);
    }
    std::string large(200003, 'x');
    large[100000] = '\0';
    ByteOutputStream stream(&data->stringAllocator(), false, false);
    auto position = data->stringAllocator().newWrite(stream, 32);
    stream.appendStringView(StringView(large));
    data->stringAllocator().finishWrite(stream, 0);
    StringView fragmented(position.position, large.size());
    ASSERT_LT(
        HashStringAllocator::headerOf(fragmented.data())->size(),
        fragmented.size());
    std::memcpy(
        rows[1] + data->columnAt(0).offset(), &fragmented, sizeof(fragmented));
    rows[17] = nullptr;
    std::vector<vector_size_t> selection(count);
    for (vector_size_t i = 0; i < count; ++i) {
      selection[i] = i % 13 == 0 ? -1 : count - 1 - i;
    }
    std::vector<VectorPtr> results;
    for (bool indexed : {false, true}) {
      auto off = output(type, count + 9);
      auto on = output(type, count + 9);
      auto alias = on->as<FlatVector<StringView>>()->values();
      for (bool enabled : {false, true}) {
        FLAGS_velox_row_container_string_extraction_prefetch_enabled = enabled;
        auto& result = enabled ? on : off;
        if (indexed) {
          RowContainer::extractColumn(
              rows.data(),
              folly::Range<const vector_size_t*>(selection.data(), count),
              data->columnAt(0),
              data->columnHasNulls(0),
              4,
              result);
        } else {
          RowContainer::extractColumn(
              rows.data(),
              count,
              data->columnAt(0),
              data->columnHasNulls(0),
              4,
              result);
        }
      }
      assertEqualVectors(off, on);
      EXPECT_EQ(
          alias->as<StringView>()[4], StringView("untouched-output-value"));
      results.push_back(off);
      results.push_back(on);
    }
    data.reset();
    input.reset();
    for (int i = 0; i < results.size(); i += 2) {
      assertEqualVectors(results[i], results[i + 1]);
    }
  }
}
} // namespace facebook::velox::exec::test
