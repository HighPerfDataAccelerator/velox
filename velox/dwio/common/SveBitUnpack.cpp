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

#include "velox/dwio/common/SveBitUnpack.h"

#include <gflags/gflags.h>

DEFINE_bool(
    velox_sve_bit_unpack,
    true,
    "Use the SVE kernels for the row-list dwio::common::unpack(). Only read by "
    "builds with SVE.");

#if defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

#include <algorithm>
#include <limits>
#include <type_traits>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"

// Decoding a field takes three steps, and every lane of a 32-bit SVE vector
// decodes one field independently:
//
//   1. Load the field's window: the 4 bytes starting at the byte that holds
//      its first bit.
//   2. Shift the window right by the field's in-byte bit offset (0-7).
//   3. Mask off the bits above the field.
//
// With bitWidth <= kMaxBitWidth (25) a field plus its shift never exceeds the
// 32-bit window. Steps 2 and 3 are the same for every row set; step 1 has two
// implementations:
//
//   - Sparse rows: each lane computes its own address from its row number and
//     the window is fetched with a gather load (LD1W, 32-bit elements,
//     unscaled 32-bit offsets). The Neoverse V2 Software Optimization Guide
//     lists it at latency 9, throughput 1. Loop iterations share no data, so
//     out-of-order execution overlaps the gathers of consecutive iterations;
//     the loop is bound by gather throughput, not its latency, and is not
//     unrolled.
//
//   - Contiguous rows: fields are back to back, so the windows of a whole
//     vector lie in one short byte range. One contiguous load plus a byte
//     shuffle (TBL) builds all windows, avoiding the gather. The shuffle
//     indices and shifts depend only on the in-byte bit offset of the
//     vector's first field, which takes at most two values over a run (see
//     unpackContiguous()), so they are computed once per call.
//
// Each kernel handles a partial last vector through its predicate, so there
// is no scalar tail.

namespace facebook::velox::dwio::common::sve {
namespace {

// Returns the byte offset of the first byte of each field, given the bit
// position of each field's first bit.
inline svuint32_t firstBytes(svbool_t active, svuint32_t bitPositions) {
  return svlsr_n_u32_x(active, bitPositions, 3);
}

// Returns the shift that right-aligns each field in its window: the low 3
// bits of the field's bit position relative to a byte boundary.
inline svuint32_t inByteShifts(svbool_t active, svuint32_t bitPositions) {
  return svand_n_u32_x(active, bitPositions, 7);
}

// Returns the fields held in 'windows', given the shift of each field in its
// window.
inline svuint32_t extractFields(
    svbool_t active,
    svuint32_t windows,
    svuint32_t shifts,
    uint32_t mask) {
  return svand_n_u32_x(active, svlsr_u32_x(active, windows, shifts), mask);
}

// Returns the window of each active lane's field by gathering 4 bytes from
// 'bits' at the field's first byte. 'bitPositions' is relative to 'bits'.
// Unaligned element addresses are allowed for loads from Normal memory.
inline svuint32_t
gatherWindows(svbool_t active, const uint64_t* bits, svuint32_t bitPositions) {
  return svld1_gather_u32offset_u32(
      active,
      reinterpret_cast<const uint32_t*>(bits),
      firstBytes(active, bitPositions));
}

// Returns the TBL indices that move each lane's window into the lane, for a
// vector of bytes loaded from the byte that 'laneBitPositions' is relative to.
inline svuint8_t windowTable(svbool_t active, svuint32_t laneBitPositions) {
  // Byte j of lane k's window is byte firstByte(k) + j of the loaded range.
  // Multiplying firstByte(k) by 0x01010101 copies it into all 4 bytes of the
  // lane and adding 0x03020100 adds j to byte j. firstByte(k) + 3 is below 256
  // for any vector length (at most 200 at 2048 bits), so no byte carries into
  // the next.
  return svreinterpret_u8_u32(svmla_n_u32_x(
      active,
      svdup_n_u32(0x03020100),
      firstBytes(active, laneBitPositions),
      0x01010101));
}

// Returns the number of bytes, counted from the byte holding the first field,
// that covers the windows of 'numFields' consecutive fields whose first field
// starts at bit 'firstShift' of that byte. The last field starts at byte
// (firstShift + (numFields - 1) * bitWidth) / 8 and its window spans 4 bytes.
// With bitWidth <= kMaxBitWidth this never exceeds the vector length in bytes,
// so one load covers every window of a vector.
inline uint32_t
windowBytes(uint32_t firstShift, uint32_t numFields, uint32_t bitWidth) {
  const uint32_t numBytes =
      ((firstShift + (numFields - 1) * bitWidth) >> 3) + 4;
  VELOX_DCHECK_LE(numBytes, svcntb());
  return numBytes;
}

// Returns the windows of fields stored back to back, using a contiguous load
// and a byte shuffle instead of a gather. 'loadBytes' selects the bytes read
// from 'start', which must include every byte of the active windows; bytes
// outside it are not read. 'table' comes from windowTable().
inline svuint32_t
shuffleWindows(svbool_t loadBytes, const uint8_t* start, svuint8_t table) {
  return svreinterpret_u32_u8(svtbl_u8(svld1_u8(loadBytes, start), table));
}

// Stores the active lanes of 'fields' to consecutive elements of 'result',
// narrowing or widening each 32-bit field to T.
template <typename T>
inline void storeFields(svbool_t active, svuint32_t fields, T* result) {
  if constexpr (sizeof(T) == sizeof(uint16_t)) {
    svst1h_u32(active, reinterpret_cast<uint16_t*>(result), fields);
  } else if constexpr (sizeof(T) == sizeof(uint32_t)) {
    svst1_u32(active, reinterpret_cast<uint32_t*>(result), fields);
  } else {
    static_assert(sizeof(T) == sizeof(uint64_t));
    // Zero-extend the low and high halves of the lanes into two vectors of
    // 64-bit lanes, each covering svcntd() consecutive results.
    auto* output = reinterpret_cast<uint64_t*>(result);
    svst1_u64(svunpklo_b(active), output, svunpklo_u64(fields));
    svst1_u64(svunpkhi_b(active), output + svcntd(), svunpkhi_u64(fields));
  }
}

// Decodes arbitrary strictly increasing 'rows' with gathers.
template <typename T>
void unpackSparse(
    const uint64_t* bits,
    uint32_t bitOffset,
    const int32_t* rows,
    int32_t numRows,
    uint32_t bitWidth,
    T* result) {
  const auto mask = static_cast<uint32_t>(bits::lowMask(bitWidth));
  const auto numLanes = static_cast<int32_t>(svcntw());
  for (int32_t i = 0; i < numRows; i += numLanes) {
    const svbool_t active = svwhilelt_b32_s32(i, numRows);
    const svuint32_t rowNumbers =
        svld1_u32(active, reinterpret_cast<const uint32_t*>(rows + i));
    const svuint32_t bitPositions =
        svmla_n_u32_x(active, svdup_n_u32(bitOffset), rowNumbers, bitWidth);
    const svuint32_t windows = gatherWindows(active, bits, bitPositions);
    storeFields(
        active,
        extractFields(
            active, windows, inByteShifts(active, bitPositions), mask),
        result + i);
  }
}

// Decodes the 'numRows' consecutive fields starting at bit 'firstBit' of
// 'bytes' one vector at a time, recomputing the shuffle for each vector. Used
// for the at most two vectors left over by unpackContiguous().
template <typename T>
void unpackContiguousTail(
    const uint8_t* bytes,
    uint64_t firstBit,
    int32_t numRows,
    uint32_t bitWidth,
    uint32_t mask,
    T* result) {
  const auto numLanes = static_cast<int32_t>(svcntw());
  for (int32_t i = 0; i < numRows; i += numLanes) {
    const svbool_t active = svwhilelt_b32_s32(i, numRows);
    const uint64_t vectorBit = firstBit + static_cast<uint64_t>(i) * bitWidth;
    const uint32_t firstShift = vectorBit & 7;
    const svuint32_t laneBitPositions = svindex_u32(firstShift, bitWidth);
    const uint32_t numFields = std::min(numLanes, numRows - i);
    const svuint32_t windows = shuffleWindows(
        svwhilelt_b8_u32(0u, windowBytes(firstShift, numFields, bitWidth)),
        bytes + (vectorBit >> 3),
        windowTable(active, laneBitPositions));
    storeFields(
        active,
        extractFields(
            active, windows, inByteShifts(active, laneBitPositions), mask),
        result + i);
  }
}

// Decodes the 'numRows' consecutive rows starting at 'firstRow' with
// contiguous loads.
//
// A vector of N = svcntw() fields spans N * bitWidth bits. N is a multiple of
// 4 because the vector length is a multiple of 128 bits, so N * bitWidth mod 8
// is 0 or 4: the first field of consecutive vectors starts at in-byte bit
// offsets that alternate between two values, A and B (equal when N *
// bitWidth is a multiple of 8), and a pair of vectors spans a whole number of
// bytes. The main loop therefore decodes pairs of vectors with shuffle
// indices, shifts and load predicates computed once, leaving only the loads,
// shuffles, shifts, masks and stores in the loop.
template <typename T>
void unpackContiguous(
    const uint64_t* bits,
    uint32_t bitOffset,
    int32_t firstRow,
    int32_t numRows,
    uint32_t bitWidth,
    T* result) {
  const auto mask = static_cast<uint32_t>(bits::lowMask(bitWidth));
  const auto numLanes = static_cast<uint32_t>(svcntw());
  const auto* bytes = reinterpret_cast<const uint8_t*>(bits);
  const uint64_t firstBit =
      bitOffset + static_cast<uint64_t>(firstRow) * bitWidth;

  // Vector A starts at 'pairStart' + shiftA bits, vector B at 'pairStart' +
  // offsetB bytes + shiftB bits.
  const uint32_t vectorBits = numLanes * bitWidth;
  const uint32_t shiftA = firstBit & 7;
  const uint32_t shiftB = (shiftA + vectorBits) & 7;
  const uint32_t offsetB = (shiftA + vectorBits) >> 3;
  const uint32_t pairBytes = 2 * vectorBits / 8;

  const svbool_t all = svptrue_b32();
  const svuint32_t laneBitPositionsA = svindex_u32(shiftA, bitWidth);
  const svuint32_t laneBitPositionsB = svindex_u32(shiftB, bitWidth);
  const svuint8_t tableA = windowTable(all, laneBitPositionsA);
  const svuint8_t tableB = windowTable(all, laneBitPositionsB);
  const svuint32_t shiftsA = inByteShifts(all, laneBitPositionsA);
  const svuint32_t shiftsB = inByteShifts(all, laneBitPositionsB);
  const svbool_t loadBytesA =
      svwhilelt_b8_u32(0u, windowBytes(shiftA, numLanes, bitWidth));
  const svbool_t loadBytesB =
      svwhilelt_b8_u32(0u, windowBytes(shiftB, numLanes, bitWidth));

  const int32_t numPairs = numRows / (2 * numLanes);
  const uint8_t* pairStart = bytes + (firstBit >> 3);
  T* output = result;
  for (int32_t pair = 0; pair < numPairs; ++pair) {
    storeFields(
        all,
        extractFields(
            all, shuffleWindows(loadBytesA, pairStart, tableA), shiftsA, mask),
        output);
    storeFields(
        all,
        extractFields(
            all,
            shuffleWindows(loadBytesB, pairStart + offsetB, tableB),
            shiftsB,
            mask),
        output + numLanes);
    pairStart += pairBytes;
    output += 2 * numLanes;
  }

  const int32_t numDone = numPairs * 2 * numLanes;
  unpackContiguousTail(
      bytes,
      firstBit + static_cast<uint64_t>(numDone) * bitWidth,
      numRows - numDone,
      bitWidth,
      mask,
      result + numDone);
}

} // namespace

bool bitUnpackEnabled() {
  return FLAGS_velox_sve_bit_unpack;
}

template <typename T>
int32_t unpackRows(
    const uint64_t* bits,
    int32_t bitOffset,
    const int32_t* rows,
    int32_t numRows,
    uint8_t bitWidth,
    T* result) {
  static_assert(
      std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
      std::is_same_v<T, int64_t>);
  if (numRows == 0 || bitWidth > kMaxBitWidth) {
    return 0;
  }
  VELOX_DCHECK_GE(bitOffset, 0);
  // Bit positions are computed in 32-bit lanes.
  VELOX_DCHECK_LE(
      static_cast<int64_t>(rows[numRows - 1]) * bitWidth + bitOffset,
      std::numeric_limits<uint32_t>::max());

  // Strictly increasing rows form one contiguous range exactly when their
  // span equals their count.
  if (rows[numRows - 1] - rows[0] == numRows - 1) {
    unpackContiguous(bits, bitOffset, rows[0], numRows, bitWidth, result);
  } else {
    unpackSparse(bits, bitOffset, rows, numRows, bitWidth, result);
  }
  return numRows;
}

template int32_t unpackRows(
    const uint64_t* bits,
    int32_t bitOffset,
    const int32_t* rows,
    int32_t numRows,
    uint8_t bitWidth,
    int16_t* result);

template int32_t unpackRows(
    const uint64_t* bits,
    int32_t bitOffset,
    const int32_t* rows,
    int32_t numRows,
    uint8_t bitWidth,
    int32_t* result);

template int32_t unpackRows(
    const uint64_t* bits,
    int32_t bitOffset,
    const int32_t* rows,
    int32_t numRows,
    uint8_t bitWidth,
    int64_t* result);

} // namespace facebook::velox::dwio::common::sve

#endif
