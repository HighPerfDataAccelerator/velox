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
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>
#include <array>
#include <cstring>
#include <numeric>
#include <random>
#include "velox/exec/ArmRowExtraction.h"
#include "velox/exec/RowContainer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec::test {
class RowContainerFixedExtractionTest : public testing::Test,
                                        public velox::test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  template <typename T>
  void check(
      int count,
      int resultOffset,
      bool selected,
      bool nullable,
      int pattern,
      bool existingNulls) {
    gflags::FlagSaver save;
    const int total = std::max(1, count);
    std::vector<std::array<char, 64>> data(total);
    std::vector<const char*> rows(total);
    std::vector<int32_t> indices(total);
    std::mt19937_64 random(count + 0x981f);
    for (int i = 0; i < total; ++i) {
      const T value = static_cast<T>(random());
      memcpy(data[i].data() + 13, &value, sizeof(value));
      data[i][2] =
          nullable && (pattern == 3 || (pattern == 1 && i % 5 == 0) ||
                       (pattern == 5 && i % 17 == 0))
          ? 4
          : 0;
      const bool missing = pattern == 4 || (pattern == 1 && i % 11 == 0) ||
          (pattern == 5 && i % 128 == 64);
      rows[i] = missing ? nullptr : data[i].data();
      indices[i] = pattern == 2 ? (i * 17) % total : total - i - 1;
      if (pattern == 1 && i % 19 == 0)
        indices[i] = -1;
    }
    TypePtr type;
    if constexpr (sizeof(T) == 4)
      type = INTEGER();
    else
      type = BIGINT();
    const RowColumn column(13, nullable ? 20 : RowColumn::kNotNullOffset);
    const auto create = [&]() {
      auto vector = BaseVector::create(type, count + resultOffset, pool());
      auto* flat = vector->template as<FlatVector<T>>();
      for (int i = 0; i < vector->size(); ++i) {
        flat->set(i, T(-777));
        if (existingNulls)
          flat->setNull(i, i % 3 == 0);
      }
      return vector;
    };
    const auto extract = [&](const VectorPtr& vector) {
      if (selected) {
        RowContainer::extractColumn(
            rows.data(),
            folly::Range<const int32_t*>(indices.data(), count),
            column,
            nullable,
            resultOffset,
            vector);
      } else {
        RowContainer::extractColumn(
            rows.data(), count, column, nullable, resultOffset, vector);
      }
    };
    FLAGS_velox_row_container_batch_fixed_extraction_enabled = false;
    FLAGS_velox_arm_sve_row_extraction_enabled = false;
    auto expected = create();
    extract(expected);
    for (int mode : {1, 2, 3}) {
      FLAGS_velox_row_container_batch_fixed_extraction_enabled =
          (mode & 1) != 0;
      FLAGS_velox_arm_sve_row_extraction_enabled = (mode & 2) != 0;
      auto actual = create();
      extract(actual);
      for (int i = 0; i < actual->size(); ++i) {
        ASSERT_TRUE(actual->equalValueAt(expected.get(), i, i))
            << mode << " row " << i;
        if (actual->isNullAt(i)) {
          EXPECT_EQ(
              actual->template as<FlatVector<T>>()->valueAtFast(i), T(-777));
        }
      }
      // The original nullable API may allocate an all-valid bitmap even
      // without null values. ARM-only deliberately retains that fallback.
      // Reject an extra bitmap only when the reference API has none.
      if (!expected->rawNulls()) {
        EXPECT_FALSE(actual->rawNulls());
      }
    }
  }
};

TEST_F(RowContainerFixedExtractionTest, directIntegerValuesAndOutputOffsets) {
  for (int n : {0, 1, 2, 3, 31, 32, 33, 63, 64, 65, 127, 129, 8192})
    for (int offset : {0, 1, 7, 63, 65})
      for (bool existing : {false, true}) {
        check<int32_t>(n, offset, false, false, 0, existing);
        check<int64_t>(n, offset, false, false, 0, existing);
      }
}
TEST_F(RowContainerFixedExtractionTest, nullableColumnsAndMissingRowPointers) {
  for (int n : {0, 1, 31, 32, 33, 63, 64, 65, 129})
    for (int pattern : {1, 3, 4})
      for (bool nullable : {false, true})
        for (bool existing : {false, true}) {
          check<int32_t>(n, 7, false, nullable, pattern, existing);
          check<int64_t>(n, 65, false, nullable, pattern, existing);
        }
}
TEST_F(RowContainerFixedExtractionTest, alternatingNullableBlocks) {
  for (int n : {511, 512, 513, 575, 579}) {
    for (bool existing : {false, true}) {
      check<int32_t>(n, 7, false, true, 5, existing);
      check<int64_t>(n, 65, false, true, 5, existing);
    }
  }
}
TEST_F(
    RowContainerFixedExtractionTest,
    negativeRepeatedAndOutOfOrderSelection) {
  for (int n : {0, 1, 31, 32, 33, 63, 64, 65, 129, 8192})
    for (int pattern : {0, 1, 2, 3, 4})
      for (bool nullable : {false, true})
        for (bool existing : {false, true}) {
          check<int32_t>(n, 7, true, nullable, pattern, existing);
          check<int64_t>(n, 65, true, nullable, pattern, existing);
        }
}
TEST_F(RowContainerFixedExtractionTest, exactWidthGuardPageAndNullSuppression) {
  gflags::FlagSaver save;
  FLAGS_velox_row_container_batch_fixed_extraction_enabled = true;
  FLAGS_velox_arm_sve_row_extraction_enabled = true;
  const auto page = sysconf(_SC_PAGESIZE);
  auto* area = static_cast<char*>(mmap(
      nullptr,
      page * 2,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0));
  ASSERT_NE(area, MAP_FAILED);
  ASSERT_EQ(mprotect(area + page, page, PROT_NONE), 0);
  for (int width : {4, 8})
    for (int n : {31, 32, 33, 63, 64, 65, 127, 511, 512, 513, 577}) {
      std::vector<const char*> rows(n, area + page - width - 13);
      for (int i = 0; i < n; i += 3) {
        if ((i / 64) % 2 == 0) {
          rows[i] = nullptr;
        }
      }
      TypePtr type;
      if (width == 4)
        type = INTEGER();
      else
        type = BIGINT();
      auto actual = BaseVector::create(type, n, pool());
      RowContainer::extractColumn(
          rows.data(),
          n,
          RowColumn(13, RowColumn::kNotNullOffset),
          false,
          actual);
      for (int i = 0; i < n; ++i)
        EXPECT_EQ(actual->isNullAt(i), rows[i] == nullptr);
      // The value address lies in PROT_NONE, but the valid row's null bit
      // requires the scalar nullable path to suppress that value load.
      std::fill(rows.begin(), rows.end(), area + page - 1);
      area[page - 1] = 1;
      RowContainer::extractColumn(
          rows.data(), n, RowColumn(13, 0), true, actual);
      for (int i = 0; i < n; ++i)
        EXPECT_TRUE(actual->isNullAt(i));
    }
  EXPECT_EQ(munmap(area, page * 2), 0);
}
TEST_F(RowContainerFixedExtractionTest, otherTypesRetainExistingExtraction) {
  gflags::FlagSaver save;
  FLAGS_velox_row_container_batch_fixed_extraction_enabled = true;
  FLAGS_velox_arm_sve_row_extraction_enabled = true;
  std::vector<VectorPtr> inputs{
      makeFlatVector<double>(129, [](auto i) { return i / 7.0; }),
      makeFlatVector<float>(129, [](auto i) { return float(i) / 3; }),
      makeFlatVector<int128_t>(129, [](auto i) { return int128_t(i) << 80; }),
      makeFlatVector<std::string>(
          129, [](auto i) { return std::string(i % 27 + 1, 'x'); }),
      makeFlatVector<bool>(129, [](auto i) { return i % 2 == 0; })};
  for (auto input : inputs) {
    input->setNull(7, true);
    RowContainer container({input->type()}, pool());
    SelectivityVector selected(input->size());
    DecodedVector decoded(*input, selected);
    std::vector<char*> rows;
    for (int i = 0; i < input->size(); ++i) {
      rows.push_back(container.newRow());
      container.store(decoded, i, rows.back(), 0);
    }
    auto result = BaseVector::create(input->type(), input->size(), pool());
    container.extractColumn(rows.data(), rows.size(), 0, result);
    for (int i = 0; i < input->size(); ++i)
      EXPECT_TRUE(input->equalValueAt(result.get(), i, i));
  }
}
TEST_F(RowContainerFixedExtractionTest, flagsDefaultOffIndependently) {
  gflags::CommandLineFlagInfo info;
  ASSERT_TRUE(gflags::GetCommandLineFlagInfo(
      "velox_row_container_batch_fixed_extraction_enabled", &info));
  EXPECT_EQ(info.default_value, "false");
  ASSERT_TRUE(gflags::GetCommandLineFlagInfo(
      "velox_arm_sve_row_extraction_enabled", &info));
  EXPECT_EQ(info.default_value, "false");
}
} // namespace facebook::velox::exec::test
