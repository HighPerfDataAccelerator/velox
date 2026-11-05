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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#if defined(__ARM_FEATURE_SVE2_BITPERM) && defined(__BYTE_ORDER__) && \
    (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) &&                    \
    __has_include(<arm_neon_sve_bridge.h>)
#include <arm_neon_sve_bridge.h>
#define VELOX_ARM_SVE_BDEP_AVAILABLE 1
#else
#define VELOX_ARM_SVE_BDEP_AVAILABLE 0
#endif

namespace facebook::velox::dwio::common::arm_sve_bdep {

#if VELOX_ARM_SVE_BDEP_AVAILABLE

template <unsigned Width>
struct Layout {
  static_assert(Width >= 1 && Width <= 28);
  static constexpr unsigned kSlotBits = Width <= 7 ? 8 : Width <= 14 ? 16 : 32;
  static constexpr unsigned kPerWord = 64 / kSlotBits;
  static constexpr unsigned kPerPair = 2 * kPerWord;
  static constexpr uint64_t kMask = [] {
    uint64_t mask = 0;
    for (unsigned slot = 0; slot < kPerWord; ++slot) {
      mask |= ((uint64_t{1} << Width) - 1) << (slot * kSlotBits);
    }
    return mask;
  }();
};

inline uint64_t loadUnaligned64(const uint8_t* input) {
  uint64_t word;
  std::memcpy(&word, input, sizeof(word));
  return word;
}

template <unsigned Width, unsigned FirstValue>
inline uint64_t loadFields(const uint8_t* input, unsigned startShift) {
  // Caller proves a complete word is readable at every selected row.
  const auto firstBit = FirstValue * Width + startShift;
  const auto word = loadUnaligned64(input + firstBit / 8);
  constexpr auto kFieldBits = Layout<Width>::kPerWord * Width;
  return (word >> (firstBit % 8)) & ((uint64_t{1} << kFieldBits) - 1);
}

template <typename T>
inline void storeFour(T* output, uint32x4_t values) {
  if constexpr (sizeof(T) == 4) {
    vst1q_s32(output, vreinterpretq_s32_u32(values));
  } else {
    vst1q_s64(output, vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(values))));
    vst1q_s64(
        output + 2, vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(values))));
  }
}

template <unsigned Width, unsigned FirstValue, typename T>
inline void emitPair(const uint8_t* input, unsigned startShift, T* output) {
  constexpr auto kPerWord = Layout<Width>::kPerWord;
  const uint64x2_t packed{
      loadFields<Width, FirstValue>(input, startShift),
      loadFields<Width, FirstValue + kPerWord>(input, startShift)};
  const auto expanded = svbdep_u64(
      svset_neonq_u64(svdup_n_u64(0), packed),
      svdup_n_u64(Layout<Width>::kMask));
  const auto words = svget_neonq_u64(expanded);
  if constexpr (Layout<Width>::kSlotBits == 8) {
    const auto bytes = vreinterpretq_u8_u64(words);
    const auto lo = vmovl_u8(vget_low_u8(bytes));
    const auto hi = vmovl_u8(vget_high_u8(bytes));
    storeFour(output + FirstValue, vmovl_u16(vget_low_u16(lo)));
    storeFour(output + FirstValue + 4, vmovl_u16(vget_high_u16(lo)));
    storeFour(output + FirstValue + 8, vmovl_u16(vget_low_u16(hi)));
    storeFour(output + FirstValue + 12, vmovl_u16(vget_high_u16(hi)));
  } else if constexpr (Layout<Width>::kSlotBits == 16) {
    const auto halves = vreinterpretq_u16_u64(words);
    storeFour(output + FirstValue, vmovl_u16(vget_low_u16(halves)));
    storeFour(output + FirstValue + 4, vmovl_u16(vget_high_u16(halves)));
  } else {
    storeFour(output + FirstValue, vreinterpretq_u32_u64(words));
  }
}

template <unsigned Width, typename T, std::size_t... Pair>
inline void unpackWidth(
    const uint8_t* input,
    unsigned startShift,
    T* output,
    std::index_sequence<Pair...>) {
  (emitPair<Width, Pair * Layout<Width>::kPerPair>(input, startShift, output),
   ...);
}

template <unsigned Width, typename T>
inline void unpackWidth(const uint8_t* input, unsigned startShift, T* output) {
  unpackWidth<Width>(
      input,
      startShift,
      output,
      std::make_index_sequence<32 / Layout<Width>::kPerPair>{});
}

#endif

// One complete32-value block inside the existing safe prefix only. Scalar
// signature, independent of xsimd ABI or the thread's SVE vector length.
template <typename T>
inline bool
unpack32(uint8_t width, const uint8_t* input, unsigned startShift, T* output) {
#if VELOX_ARM_SVE_BDEP_AVAILABLE
#define VELOX_ARM_SVE_BDEP_CASE(WIDTH)             \
  case WIDTH:                                      \
    unpackWidth<WIDTH>(input, startShift, output); \
    return true
  switch (width) {
    VELOX_ARM_SVE_BDEP_CASE(1);
    VELOX_ARM_SVE_BDEP_CASE(2);
    VELOX_ARM_SVE_BDEP_CASE(3);
    VELOX_ARM_SVE_BDEP_CASE(4);
    VELOX_ARM_SVE_BDEP_CASE(5);
    VELOX_ARM_SVE_BDEP_CASE(6);
    VELOX_ARM_SVE_BDEP_CASE(7);
    VELOX_ARM_SVE_BDEP_CASE(8);
    VELOX_ARM_SVE_BDEP_CASE(9);
    VELOX_ARM_SVE_BDEP_CASE(10);
    VELOX_ARM_SVE_BDEP_CASE(11);
    VELOX_ARM_SVE_BDEP_CASE(12);
    VELOX_ARM_SVE_BDEP_CASE(13);
    VELOX_ARM_SVE_BDEP_CASE(14);
    VELOX_ARM_SVE_BDEP_CASE(15);
    VELOX_ARM_SVE_BDEP_CASE(16);
    VELOX_ARM_SVE_BDEP_CASE(17);
    VELOX_ARM_SVE_BDEP_CASE(18);
    VELOX_ARM_SVE_BDEP_CASE(19);
    VELOX_ARM_SVE_BDEP_CASE(20);
    VELOX_ARM_SVE_BDEP_CASE(21);
    VELOX_ARM_SVE_BDEP_CASE(22);
    VELOX_ARM_SVE_BDEP_CASE(23);
    VELOX_ARM_SVE_BDEP_CASE(24);
    VELOX_ARM_SVE_BDEP_CASE(25);
    VELOX_ARM_SVE_BDEP_CASE(26);
    VELOX_ARM_SVE_BDEP_CASE(27);
    VELOX_ARM_SVE_BDEP_CASE(28);
    default:
      return false;
  }
#undef VELOX_ARM_SVE_BDEP_CASE
#else
  (void)startShift;
  (void)width;
  (void)input;
  (void)output;
  return false;
#endif
}

template <typename T>
inline int32_t unpackSelectedPrefix(
    const uint64_t* input,
    int64_t bitOffset,
    const int32_t* rows,
    int32_t count,
    uint8_t width,
    T* output) {
#if VELOX_ARM_SVE_BDEP_AVAILABLE
  if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
    if (count < 32 || width == 0 || width > 28) {
      return 0;
    }
    int32_t i = 0;
    for (; i + 32 <= count; i += 32) {
      // Check every row: endpoints alone are insufficient with duplicates.
      uint32x4_t mismatch = vdupq_n_u32(0);
      const int32x4_t offsets{0, 1, 2, 3};
      const auto base = vaddq_s32(vdupq_n_s32(rows[i]), offsets);
      for (int32_t j = 0; j < 32; j += 4) {
        const auto expected = vaddq_s32(base, vdupq_n_s32(j));
        mismatch = vorrq_u32(
            mismatch,
            vreinterpretq_u32_s32(
                vsubq_s32(vld1q_s32(rows + i + j), expected)));
      }
      if (vmaxvq_u32(mismatch) != 0) {
        break;
      }
      const auto firstBit = int64_t{rows[i]} * width + bitOffset;
      unpack32(
          width,
          reinterpret_cast<const uint8_t*>(input) + firstBit / 8,
          firstBit % 8,
          output + i);
    }
    return i;
  }
#endif
  (void)input;
  (void)bitOffset;
  (void)rows;
  (void)count;
  (void)width;
  (void)output;
  return 0;
}

} // namespace facebook::velox::dwio::common::arm_sve_bdep
