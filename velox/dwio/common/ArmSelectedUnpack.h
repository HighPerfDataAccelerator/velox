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
    const uint64_t lanes = svcntd();
    for (uint64_t i = 0; i < static_cast<uint64_t>(count); i += lanes) {
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
