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
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define VELOX_ARM_ROW_EXTRACTION_AVAILABLE 1
#else
#define VELOX_ARM_ROW_EXTRACTION_AVAILABLE 0
#endif

namespace facebook::velox::exec::detail::row_extraction {
template <int width, bool selected, bool nullable>
uint64_t blockScalar(
    const char* const* rows,
    const int32_t* numbers,
    int count,
    int offset,
    int nullByte,
    uint8_t nullMask,
    uint8_t* output) {
  uint64_t valid = 0;
  for (int i = 0; i < count; ++i) {
    const char* row;
    if constexpr (selected) {
      row = numbers[i] < 0 ? nullptr : rows[numbers[i]];
    } else {
      row = rows[i];
    }
    if (row == nullptr)
      continue;
    if constexpr (nullable) {
      if ((static_cast<uint8_t>(row[nullByte]) & nullMask) != 0)
        continue;
    }
    valid |= uint64_t{1} << i;
    std::memcpy(output + i * width, row + offset, width);
  }
  return valid;
}

// Copy a block of non-null row pointers without the per-vector gather and
// validity-mask reduction used by the 128-bit SVE path.
template <int width>
void blockScalarFour(
    const char* const* rows,
    int count,
    int offset,
    uint8_t* output) {
  int i = 0;
  for (; i + 4 <= count; i += 4) {
    uint8_t values[4 * width];
    std::memcpy(values, rows[i] + offset, width);
    std::memcpy(values + width, rows[i + 1] + offset, width);
    std::memcpy(values + 2 * width, rows[i + 2] + offset, width);
    std::memcpy(values + 3 * width, rows[i + 3] + offset, width);
    std::memcpy(output + i * width, values, sizeof(values));
  }
  for (; i < count; ++i) {
    std::memcpy(output + i * width, rows[i] + offset, width);
  }
}

// All row pointers in the block are present. Check four null flags together
// so groups of valid values can use the same copy as non-nullable columns.
template <int width>
uint64_t blockNullableFour(
    const char* const* rows,
    int count,
    int offset,
    int nullByte,
    uint8_t nullMask,
    uint8_t* output) {
  uint64_t missing = 0;
  const auto copyOne = [&](int i) {
    if (static_cast<uint8_t>(rows[i][nullByte]) & nullMask) {
      missing |= uint64_t{1} << i;
    } else {
      std::memcpy(output + i * width, rows[i] + offset, width);
    }
  };
  int i = 0;
  for (; i + 4 <= count; i += 4) {
    const auto flags = rows[i][nullByte] | rows[i + 1][nullByte] |
        rows[i + 2][nullByte] | rows[i + 3][nullByte];
    if ((flags & nullMask) == 0) [[likely]] {
      blockScalarFour<width>(rows + i, 4, offset, output + i * width);
    } else {
      for (int j = 0; j < 4; ++j) {
        copyOne(i + j);
      }
    }
  }
  for (; i < count; ++i) {
    copyOne(i);
  }
  const auto all = count == 64 ? ~uint64_t{0} : (uint64_t{1} << count) - 1;
  return all & ~missing;
}

template <int width, bool selected, bool nullable>
uint64_t blockSve(
    const char* const* rows,
    const int32_t* numbers,
    int count,
    int offset,
    int nullByte,
    uint8_t nullMask,
    uint8_t* output) {
#if VELOX_ARM_ROW_EXTRACTION_AVAILABLE
  // The ABI is scalar. VL is dynamic; only lanes belonging to this <=64-row
  // block participate, including the last partially populated vector.
  static_assert(sizeof(char*) == sizeof(uint64_t));
  using AliasedAddress = uint64_t __attribute__((may_alias));
  auto accumulated = svdup_u64(0);
  const auto all = svptrue_b64();
  for (uint64_t i = 0; i < static_cast<uint64_t>(count); i += svcntd()) {
    const auto active = svwhilelt_b64(i, static_cast<uint64_t>(count));
    auto valid = active;
    svuint64_t addresses;
    if constexpr (selected) {
      const auto indices = svld1sw_s64(active, numbers + i);
      valid = svcmpge_n_s64(active, indices, 0);
      addresses = svld1_gather_s64index_u64(
          valid, reinterpret_cast<const AliasedAddress*>(rows), indices);
    } else {
      addresses =
          svld1_u64(active, reinterpret_cast<const AliasedAddress*>(rows + i));
    }
    valid = svcmpne_n_u64(valid, addresses, 0);
    if constexpr (nullable) {
      const auto nullBytes =
          svld1ub_gather_u64base_offset_u64(valid, addresses, nullByte);
      valid =
          svcmpeq_n_u64(valid, svand_n_u64_x(valid, nullBytes, nullMask), 0);
    }
    if constexpr (width == 8) {
      const auto values =
          svld1_gather_u64base_offset_u64(valid, addresses, offset);
      svst1_u64(valid, reinterpret_cast<AliasedAddress*>(output) + i, values);
    } else {
      const auto values =
          svld1uw_gather_u64base_offset_u64(valid, addresses, offset);
      svst1w_u64(valid, reinterpret_cast<uint32_t*>(output) + i, values);
    }
    const auto positions = svindex_u64(i, 1);
    const auto bits = svlsl_u64_z(valid, svdup_u64(1), positions);
    accumulated = svorr_u64_x(all, accumulated, bits);
  }
  return svorv_u64(all, accumulated);
#else
  return blockScalar<width, selected, nullable>(
      rows, numbers, count, offset, nullByte, nullMask, output);
#endif
}
} // namespace facebook::velox::exec::detail::row_extraction
