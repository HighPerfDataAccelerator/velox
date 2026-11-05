/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/dwio/common/ArmSelectedUnpack.h"

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace facebook::velox::dwio::common::arm_sve {
namespace {

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
constexpr bool kHasSve = true;
#else
constexpr bool kHasSve = false;
#endif

class GuardedBuffer {
 public:
  GuardedBuffer() : page_(sysconf(_SC_PAGESIZE)) {
    data_ = static_cast<char*>(mmap(
        nullptr,
        2 * page_,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0));
    if (data_ == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }
    if (mprotect(data_ + page_, page_, PROT_NONE) != 0) {
      munmap(data_, 2 * page_);
      throw std::runtime_error("mprotect failed");
    }
  }

  ~GuardedBuffer() {
    munmap(data_, 2 * page_);
  }

  char* end() const {
    return data_ + page_;
  }

 private:
  size_t page_;
  char* data_;
};

template <typename T>
T reference(const char* input, int32_t offset, int32_t row, uint8_t width) {
  const int64_t bit = offset + static_cast<int64_t>(row) * width;
  uint64_t word;
  std::memcpy(&word, input + bit / 8, sizeof(word));
  return static_cast<T>((word >> (bit & 7)) & ((uint64_t{1} << width) - 1));
}

template <typename T>
void checkRandomSelections() {
  std::mt19937_64 random(0x9ef134d2);
  for (uint8_t width = 1; width <= 56; ++width) {
    for (int32_t offset = 0; offset < 8; ++offset) {
      for (int count : {16, 17, 31, 32, 33, 127, 1024, 8193}) {
        // Arbitrary ordering and repeats are deliberate. The helper requires
        // safety of each active row, without assuming sorted/unique RowSet.
        std::vector<int32_t> rows(count);
        for (auto& row : rows) {
          row = random() % 1024;
        }
        const auto bytes = (offset + 1024 * width + 7) / 8 + 8;
        std::vector<char> storage(bytes + 8);
        for (auto& byte : storage) {
          byte = static_cast<char>(random());
        }
        const int alignment = random() % 8;
        const char* input = storage.data() + alignment;
        constexpr T kCanary = static_cast<T>(0x71b92a4d);
        std::vector<T> output(count + 2, kCanary);
        const auto done = unpackSafePrefix(
            reinterpret_cast<const uint64_t*>(input),
            offset,
            rows.data(),
            count,
            width,
            (uint64_t{1} << width) - 1,
            output.data() + 1);
        ASSERT_EQ(done, kHasSve ? count : 0);
        EXPECT_EQ(output.front(), kCanary);
        EXPECT_EQ(output.back(), kCanary);
        if (done) {
          for (int i = 0; i < count; ++i) {
            ASSERT_EQ(
                output[i + 1], reference<T>(input, offset, rows[i], width))
                << "width=" << static_cast<int>(width) << " offset=" << offset
                << " count=" << count << " row=" << i;
          }
        } else {
          EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](T value) {
            return value == kCanary;
          }));
        }
      }
    }
  }
}

template <typename T>
void checkGuardPages() {
  GuardedBuffer source;
  GuardedBuffer selection;
  GuardedBuffer destination;
  for (uint8_t width : {1, 7, 8, 13, 16, 31, 32, 56}) {
    for (int offset = 0; offset < 8; ++offset) {
      for (int count : {16, 17, 31, 32, 33}) {
        const int32_t lastRow = (count - 1) * 3;
        const int lastByte = (offset + lastRow * width) / 8;
        char* input = source.end() - lastByte - 8;
        std::memset(input, 0xb7, lastByte + 8);
        auto* rows = reinterpret_cast<int32_t*>(selection.end()) - count;
        for (int i = 0; i < count; ++i) {
          rows[i] = i * 3;
        }
        T* output = reinterpret_cast<T*>(destination.end()) - count;
        std::fill(output, output + count, T{0});
        const auto done = unpackSafePrefix(
            reinterpret_cast<const uint64_t*>(input),
            offset,
            rows,
            count,
            width,
            (uint64_t{1} << width) - 1,
            output);
        ASSERT_EQ(done, kHasSve ? count : 0);
        if (done) {
          for (int i = 0; i < count; ++i) {
            EXPECT_EQ(output[i], reference<T>(input, offset, rows[i], width));
          }
        }
      }
    }
  }
}

TEST(ArmSelectedUnpack, AllWidthsAndArbitrarySelections) {
  checkRandomSelections<int32_t>();
  checkRandomSelections<int64_t>();
}

TEST(ArmSelectedUnpack, ExactInputSelectionAndOutputGuardPages) {
  checkGuardPages<int32_t>();
  checkGuardPages<int64_t>();
}

TEST(ArmSelectedUnpack, LargeBitPositionsUse64BitAddresses) {
  if (!kHasSve) {
    GTEST_SKIP() << "Requires SVE";
  }
  // Sparse virtual allocation: only the pages containing selected fields are
  // touched. The last row's bit position exceeds UINT32_MAX.
  constexpr int32_t kLargeRow = 200000000;
  constexpr uint8_t kWidth = 24;
  constexpr int32_t kOffset = 7;
  constexpr size_t kBytes = (int64_t{kLargeRow} * kWidth + kOffset) / 8 + 8;
  auto* data = static_cast<char*>(mmap(
      nullptr,
      kBytes,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0));
  ASSERT_NE(data, MAP_FAILED);
  std::vector<int32_t> rows(33);
  for (int i = 0; i < 33; ++i) {
    rows[i] = i < 16 ? i : kLargeRow;
    const auto byte = (int64_t{rows[i]} * kWidth + kOffset) / 8;
    std::memset(data + byte, 0xb7, 8);
  }
  std::vector<int64_t> output(33, -1);
  const auto done = unpackSafePrefix(
      reinterpret_cast<const uint64_t*>(data),
      kOffset,
      rows.data(),
      rows.size(),
      kWidth,
      (uint64_t{1} << kWidth) - 1,
      output.data());
  EXPECT_EQ(done, rows.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    EXPECT_EQ(output[i], reference<int64_t>(data, kOffset, rows[i], kWidth));
  }
  EXPECT_EQ(munmap(data, kBytes), 0);
}

TEST(ArmSelectedUnpack, FallbackDoesNotAccessMemory) {
  GuardedBuffer inaccessible;
  const auto* input = reinterpret_cast<const uint64_t*>(inaccessible.end());
  const auto* rows = reinterpret_cast<const int32_t*>(inaccessible.end());
  auto* output = reinterpret_cast<int64_t*>(inaccessible.end());
  for (int count : {0, 1, 2, 15}) {
    EXPECT_EQ(unpackSafePrefix(input, 0, rows, count, 12, 4095, output), 0);
  }
  for (uint8_t width : {0, 57, 63, 64}) {
    EXPECT_EQ(
        unpackSafePrefix(input, 0, rows, 16, width, ~uint64_t{0}, output), 0);
  }
  EXPECT_EQ(
      unpackSafePrefix(
          input,
          0,
          rows,
          16,
          12,
          4095,
          reinterpret_cast<int16_t*>(inaccessible.end())),
      0);
  EXPECT_EQ(
      unpackSafePrefix(
          input,
          0,
          rows,
          16,
          12,
          4095,
          reinterpret_cast<__int128_t*>(inaccessible.end())),
      0);
  if (!kHasSve) {
    EXPECT_EQ(unpackSafePrefix(input, 0, rows, 16, 12, 4095, output), 0);
  }
}

} // namespace
} // namespace facebook::velox::dwio::common::arm_sve
