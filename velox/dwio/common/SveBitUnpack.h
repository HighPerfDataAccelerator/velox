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

#include <cstdint>

// SVE kernels for the row-list flavor of dwio::common::unpack(), the Arm
// counterpart of the AVX2 decode1To24() path in BitPackDecoder.cpp. Only
// available when the translation unit is compiled with SVE enabled (for
// example -mcpu=neoverse-v2 or -mcpu=olympus).
#if defined(__ARM_FEATURE_SVE)

namespace facebook::velox::dwio::common::sve {

/// Widest bit field the kernels decode. Each lane loads the 32-bit window that
/// starts at the byte holding the field's first bit, so the field plus its
/// in-byte shift (at most 7) must fit in 32 bits.
constexpr uint8_t kMaxBitWidth = 25;

/// Returns whether unpack() uses unpackRows(), as set by
/// --velox_sve_bit_unpack. Lets one binary A/B the kernels against the scalar
/// loop.
bool bitUnpackEnabled();

/// Decodes the little-endian bit fields selected by 'rows' into 'result'.
/// Field i is 'bitWidth' bits wide and starts at bit
/// 'bitOffset' + rows[i] * 'bitWidth' of 'bits'. 'rows' must be strictly
/// increasing, as a RowSet is. 'bitOffset' must be non-negative and the last
/// field's bit position must fit in 32 bits.
///
/// Never reads before the first byte of the first field nor more than 4 bytes
/// past the first byte of the last field, so any row that is safe for the
/// 8-byte loads of the scalar path in unpack() is safe here.
///
/// Returns the number of leading rows decoded: 'numRows' if 'bitWidth' is at
/// most kMaxBitWidth, otherwise 0 and the caller decodes all rows.
/// T is int16_t, int32_t or int64_t.
template <typename T>
int32_t unpackRows(
    const uint64_t* bits,
    int32_t bitOffset,
    const int32_t* rows,
    int32_t numRows,
    uint8_t bitWidth,
    T* result);

} // namespace facebook::velox::dwio::common::sve

#endif
