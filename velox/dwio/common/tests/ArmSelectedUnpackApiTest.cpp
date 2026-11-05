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

#include "velox/dwio/common/BitPackDecoder.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

DECLARE_bool(velox_arm_sve_bit_unpack_enabled);
DECLARE_bool(velox_arm_sve_bit_deposit_enabled);

namespace {
namespace common = facebook::velox::dwio::common;
class Guarded {
 public:
  Guarded() : size_(sysconf(_SC_PAGESIZE)) {
    base_ = static_cast<uint8_t*>(mmap(
        nullptr,
        size_ * 2,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0));
    if (base_ == MAP_FAILED) {
      throw std::runtime_error("mmap failed");
    }
    if (mprotect(base_ + size_, size_, PROT_NONE)) {
      munmap(base_, size_ * 2);
      throw std::runtime_error("mprotect failed");
    }
  }
  ~Guarded() {
    munmap(base_, size_ * 2);
  }
  uint8_t* end() const {
    return base_ + size_;
  }

 private:
  size_t size_;
  uint8_t* base_;
};

template <typename T>
void call(
    int mode,
    const uint8_t* data,
    int offset,
    const std::vector<int32_t>& rows,
    int bias,
    uint8_t width,
    const char* end,
    T* output) {
  auto bits = reinterpret_cast<const uint64_t*>(data);
  common::RowSet selected(rows.data(), rows.size());
  FLAGS_velox_arm_sve_bit_unpack_enabled = mode == 1;
  common::unpack(bits, offset, selected, bias, width, end, output);
}

template <typename T>
T reference(const uint8_t* data, int64_t bit, uint8_t width) {
  uint64_t value = 0;
  for (unsigned i = 0; i < std::min<unsigned>(width, sizeof(T) * 8); ++i) {
    value |= uint64_t((data[(bit + i) / 8] >> ((bit + i) % 8)) & 1) << i;
  }
  return static_cast<T>(value);
}

template <typename T>
void checkCase(
    uint8_t width,
    int count,
    int offset,
    int bias,
    int pattern,
    bool guard) {
  std::mt19937_64 random(0x718cae + width * 11 + count);
  std::vector<int32_t> rows(count);
  for (int i = 0; i < count; ++i) {
    rows[i] = bias + (pattern == 0 ? i : pattern == 1 ? i * 3 : i / 3);
  }
  const int last = count ? rows.back() - bias : 0;
  const size_t bytes = (int64_t(last + 1) * width + offset + 7) / 8;
  Guarded inputGuard, outputGuard;
  std::vector<uint64_t> storage((512 + bytes + 32) / 8 + 1);
  uint8_t* data = guard ? inputGuard.end() - bytes
                        : reinterpret_cast<uint8_t*>(storage.data()) + 512 +
          (pattern + offset) % 8;
  for (size_t i = 0; i < bytes; ++i) {
    data[i] = random();
  }
  const char* end =
      count ? reinterpret_cast<const char*>(data + bytes) : nullptr;
  std::vector<T> inheritedWide(count);
  if (width > 56) {
    // This prototype does not change the original wide-field implementation.
    // Check byte-for-byte compatibility for that inherited path separately.
    call<T>(0, data, offset, rows, bias, width, end, inheritedWide.data());
  }
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<T> result(count + 2, T{0x2397});
    T* output = guard ? reinterpret_cast<T*>(outputGuard.end()) - count
                      : result.data() + 1;
    if (guard) {
      std::fill(output - 1, output + count, T{0x2397});
    }
    call<T>(
        mode, width ? data : nullptr, offset, rows, bias, width, end, output);
    if (output[-1] != T{0x2397} || (!guard && output[count] != T{0x2397})) {
      throw std::runtime_error("Output canary overwritten");
    }
    for (int i = 0; i < count; ++i) {
      auto expected = width > 56
          ? inheritedWide[i]
          : reference<T>(data, int64_t(rows[i] - bias) * width + offset, width);
      if (output[i] != expected) {
        throw std::runtime_error(
            "API parity failure width=" + std::to_string(width) + " count=" +
            std::to_string(count) + " mode=" + std::to_string(mode));
      }
    }
  }
}

size_t correctness() {
  size_t cases = 0;
  for (uint8_t width = 0; width <= 64; ++width) {
    for (int count : {0, 1, 7, 15, 16, 17, 31, 128, 8193}) {
      for (int offset = 0; offset < 8; ++offset) {
        for (int pattern = 0; pattern < 3; ++pattern) {
          for (int bias : {0, 29}) {
            if (width <= 56) {
              checkCase<int32_t>(width, count, offset, bias, pattern, false);
              ++cases;
            }
            checkCase<int64_t>(width, count, offset, bias, pattern, false);
            ++cases;
          }
        }
      }
    }
    for (int offset = 0; offset < 8; ++offset) {
      for (int count : {1, 16, 31, 128}) {
        if (width <= 56) {
          checkCase<int32_t>(width, count, offset, 0, 1, true);
          ++cases;
        }
        checkCase<int64_t>(width, count, offset, 0, 1, true);
        ++cases;
        if (width <= 56) {
          checkCase<int16_t>(width, count, offset, 0, 1, true);
          ++cases;
        }
        // The existing >56-bit loadBits<T> path always reads sizeof(T), so an
        // exact-byte guard for a 128-bit T is outside its inherited contract.
        checkCase<__int128_t>(width, count, offset, 0, 1, width <= 56);
        ++cases;
      }
    }
  }
  return cases;
}

TEST(ArmSelectedUnpackApi, WidthsRowBiasSafeTailsAndGuardPages) {
  gflags::FlagSaver savedFlags;
  EXPECT_EQ(correctness(), 60512);
}
TEST(ArmSelectedUnpackApi, LargeRowPositionsAndBias) {
  gflags::FlagSaver savedFlags;
  constexpr int32_t kLargeRow = 200000000;
  constexpr uint8_t kWidth = 24;
  constexpr int kCount = 33;
  constexpr int kOffset = 7;
  const size_t bytes = (int64_t{kLargeRow + kCount} * kWidth + kOffset + 7) / 8;
  auto* data = static_cast<uint8_t*>(mmap(
      nullptr,
      bytes,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0));
  ASSERT_NE(data, MAP_FAILED);
  std::vector<int32_t> rows(kCount);
  for (int i = 0; i < kCount; ++i) {
    rows[i] = kLargeRow + i;
  }
  const auto firstByte = (int64_t{kLargeRow} * kWidth + kOffset) / 8;
  std::fill(data + firstByte, data + bytes, uint8_t{0xb7});
  std::vector<int64_t> output(kCount);
  for (int mode : {0, 1, 2, 3}) {
    FLAGS_velox_arm_sve_bit_deposit_enabled = mode >= 2;
    for (bool bounded : {false, true}) {
      // Leave the unbounded load enough readable padding by excluding the
      // final rows. Both branches still exercise overflowing bit positions.
      auto selected = rows;
      if (!bounded) {
        selected.resize(kCount - 3);
      }
      call(
          mode % 2,
          data,
          kOffset,
          selected,
          0,
          kWidth,
          bounded ? reinterpret_cast<const char*>(data + bytes) : nullptr,
          output.data());
      for (size_t i = 0; i < selected.size(); ++i) {
        EXPECT_EQ(
            output[i],
            reference<int64_t>(
                data, int64_t{selected[i]} * kWidth + kOffset, kWidth));
      }
    }
    // A large rowBias must cancel before multiplication narrows to int32.
    std::vector<uint64_t> small(32, 0xb7b7b7b7b7b7b7b7ULL);
    call(
        mode % 2,
        reinterpret_cast<const uint8_t*>(small.data()),
        kOffset,
        rows,
        kLargeRow,
        kWidth,
        nullptr,
        output.data());
    for (int i = 0; i < kCount; ++i) {
      EXPECT_EQ(
          output[i],
          reference<int64_t>(
              reinterpret_cast<const uint8_t*>(small.data()),
              int64_t{i} * kWidth + kOffset,
              kWidth));
    }
  }
  EXPECT_EQ(munmap(data, bytes), 0);
}

} // namespace
