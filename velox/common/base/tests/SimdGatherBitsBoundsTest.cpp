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

#include "velox/common/base/SimdUtil.h"

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <stdexcept>
#include <vector>

namespace facebook::velox::simd {
namespace {

class GuardedBitmap {
 public:
  explicit GuardedBitmap(size_t words) : page_(sysconf(_SC_PAGESIZE)) {
    base_ = static_cast<char*>(mmap(
        nullptr, 2 * page_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (base_ == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }
    if (mprotect(base_, page_, PROT_READ | PROT_WRITE) != 0) {
      munmap(base_, 2 * page_);
      throw std::runtime_error("mprotect failed");
    }
    data_ = reinterpret_cast<uint64_t*>(base_ + page_) - words;
  }

  ~GuardedBitmap() {
    munmap(base_, 2 * page_);
  }

  uint64_t* data() {
    return data_;
  }

 private:
  const size_t page_;
  char* base_;
  uint64_t* data_;
};

TEST(SimdGatherBitsBoundsTest, LastWordAndInactiveLanes) {
  GuardedBitmap bitmap(1);
  constexpr auto lanes = xsimd::batch<int32_t>::size;
  std::array<int32_t, lanes> indices;
  for (uint64_t pattern :
       {0ULL,
        ~0ULL,
        0x5555555555555555ULL,
        0xaaaaaaaaaaaaaaaaULL,
        0x8000000080000000ULL}) {
    bitmap.data()[0] = pattern;
    for (int active = 0; active <= lanes; ++active) {
      indices.fill(1 << 30);
      uint8_t expected = 0;
      for (int i = 0; i < active; ++i) {
        indices[i] = 63 - i;
        expected |= uint8_t(bits::isBitSet(bitmap.data(), indices[i])) << i;
      }
      const auto actual = gather8Bits(
          bitmap.data(), xsimd::load_unaligned(indices.data()), active);
      EXPECT_EQ(actual, expected) << "active=" << active;
    }
  }
}

TEST(SimdGatherBitsBoundsTest, BulkAndPartialBatchesAtGuardPage) {
  constexpr auto lanes = xsimd::batch<int32_t>::size;
  for (int words : {1, 64}) {
    GuardedBitmap bitmap(words);
    for (int i = 0; i < words; ++i) {
      bitmap.data()[i] = 0x8000000087654321ULL ^ uint64_t(i);
    }
    for (int count = 1; count <= 257; ++count) {
      // The API's existing SIMD index loads may read a padded final batch.
      std::vector<int32_t> indices(count + lanes, 1 << 30);
      for (int i = 0; i < count; ++i) {
        indices[i] = words * 64 - 1 - (i % 64);
      }
      std::vector<uint64_t> output((count + 63) / 64, 0);
      gatherBits(
          bitmap.data(),
          folly::Range<const int32_t*>(indices.data(), count),
          output.data());
      for (int i = 0; i < count; ++i) {
        ASSERT_EQ(
            bits::isBitSet(output.data(), i),
            bits::isBitSet(bitmap.data(), indices[i]))
            << "count=" << count << ", bit=" << i;
      }
    }
  }
}

} // namespace
} // namespace facebook::velox::simd
