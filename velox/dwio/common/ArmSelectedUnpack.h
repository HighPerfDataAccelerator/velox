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

#include <cstdint>
#include <type_traits>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

namespace facebook::velox::dwio::common::arm_sve {

// The caller has already proved that every active selected row can read a
// complete uint64_t. This helper must never include the caller's unsafe tail.
// Arguments and return value are scalar; no SVE type crosses the C++ ABI.
template <typename T>
inline int32_t unpackSafePrefix(
    const uint64_t* input,
    int64_t normalizedBitOffset,
    const int32_t* rows,
    int32_t count,
    uint8_t width,
    uint64_t mask,
    T* output) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
  if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
    if (count < 16 || width == 0 || width > 56) {
      return 0;
    }
    uint64_t i = 0;
#if defined(__AARCH64EL__)
    // Even at bit offset seven, a field of up to 25 bits fits in one word.
    // Use all 32-bit lanes when bit positions fit. Selections may be unordered.
    if (width <= 25 && mask <= 0x1ffffff &&
        static_cast<uint64_t>(normalizedBitOffset) <= UINT32_MAX) {
      const auto offset = static_cast<uint32_t>(normalizedBitOffset);
      // Bound row * width without division, using width <= 32.
      const uint32_t maxRow = (UINT32_MAX - offset) >> 5;
      const uint64_t lanes32 = svcntw();
      const auto all = svptrue_b32();
      for (; i + lanes32 <= static_cast<uint64_t>(count); i += lanes32) {
        const auto selected = svreinterpret_u32_s32(svld1_s32(all, rows + i));
        if (svptest_any(all, svcmpgt_n_u32(all, selected, maxRow))) {
          break;
        }
        const auto bit =
            svmla_n_u32_x(all, svdup_n_u32(offset), selected, width);
        const auto byte = svlsr_n_u32_x(all, bit, 3);
        const auto words = svld1_gather_u32offset_u32(
            all, reinterpret_cast<const uint32_t*>(input), byte);
        const auto values = svand_n_u32_x(
            all,
            svlsr_u32_x(all, words, svand_n_u32_x(all, bit, 7)),
            static_cast<uint32_t>(mask));
        if constexpr (std::is_same_v<T, int32_t>) {
          svst1_s32(all, output + i, svreinterpret_s32_u32(values));
        } else {
          svst1_s64(
              svptrue_b64(),
              output + i,
              svreinterpret_s64_u64(svunpklo_u64(values)));
          svst1_s64(
              svptrue_b64(),
              output + i + svcntd(),
              svreinterpret_s64_u64(svunpkhi_u64(values)));
        }
      }
    }
#endif
    const uint64_t lanes = svcntd();
    for (; i < static_cast<uint64_t>(count); i += lanes) {
      const auto active = svwhilelt_b64(i, static_cast<uint64_t>(count));
      const auto selected = svld1sw_s64(active, rows + i);
      const auto bit = svadd_n_s64_x(
          active, svmul_n_s64_x(active, selected, width), normalizedBitOffset);
      const auto unsignedBit = svreinterpret_u64_s64(bit);
      const auto byte = svlsr_n_u64_x(active, unsignedBit, 3);
      const auto shift = svand_n_u64_x(active, unsignedBit, 7);
      const auto words = svld1_gather_u64offset_u64(active, input, byte);
      const auto values =
          svand_n_u64_x(active, svlsr_u64_x(active, words, shift), mask);
      if constexpr (std::is_same_v<T, int32_t>) {
        svst1w_s64(active, output + i, svreinterpret_s64_u64(values));
      } else {
        svst1_s64(active, output + i, svreinterpret_s64_u64(values));
      }
    }
    return count;
  }
#endif
  (void)input;
  (void)normalizedBitOffset;
  (void)rows;
  (void)count;
  (void)width;
  (void)mask;
  (void)output;
  return 0;
}

} // namespace facebook::velox::dwio::common::arm_sve
