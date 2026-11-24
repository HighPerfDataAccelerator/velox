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

#include "velox/dwio/common/SveDictionaryFilter.h"

#include <gflags/gflags.h>

DEFINE_bool(
    velox_sve_dictionary_filter,
    true,
    "Use the SVE kernel for the filter path of "
    "DictionaryColumnVisitor::processRun(). Only read by builds with SVE.");

#if defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

#include <type_traits>

#include <folly/Likely.h>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/DecoderUtil.h"

// Two kernels, chosen by the dictionary value type from measurements on
// NVIDIA Olympus (gcc 15, -mcpu=olympus, 128-bit SVE), ns per index at Q6
// dictionary sizes and pass rates, independent of both:
//
//            SVE    scalar
//   int32   0.51    0.67    -> filterWithSve()
//   int64   0.97    0.72    -> filterScalar()
//
// For int64 the SVE kernel pays for widening: every vector of 4 indices needs
// two extra 2-element LD1D gathers, predicate and vector unpacks and two more
// COMPACTs, about 28 instructions and 8 element accesses. The scalar kernel
// needs 4 loads, 2 stores and a few integer instructions per index and never
// moves data between vector and general registers.
//
// filterWithSve() handles each vector of svcntw() indices as follows:
//
//   1. Loads the indices as 32-bit lanes.
//   2. Gathers one filter cache byte per lane (LD1B gather).
//   3. Resolves lanes whose cache byte is kUnknown through the caller's
//      filter, then gathers again. This happens once per dictionary entry, so
//      the branch is almost never taken.
//   4. Compares the cache bytes with kSuccess, giving the 'passed' predicate.
//   5. Appends the passing row numbers with COMPACT and a store predicated on
//      the passing count (CNTP).
//   6. Optionally gathers the dictionary values of the passing lanes (LD1W
//      gather) and appends them the same way, or, without a dictionary
//      (string dictionaries), appends the passing indices themselves.
//
// filterScalar() is a branchless compaction: every index stores its row and
// value at the next output slot, and the slot advances only if the index
// passes.
//
// Neither kernel branches on the data, unlike the generic loop, which branches
// on whether none, all or some of each group of values pass.

namespace facebook::velox::dwio::common::sve {
namespace {

constexpr uint32_t kUnknown = FilterResult::kUnknown;
constexpr uint32_t kSuccess = FilterResult::kSuccess;

// Upper bound of svcntw(): 2048-bit vectors hold 64 32-bit lanes.
constexpr int32_t kMaxLanes = 64;

// Returns the dictionary indices of the first min(svcntw(), 'numLeft') slots
// of 'indices' as 32-bit lanes; inactive lanes are 0.
inline svuint32_t loadIndices(const int32_t* indices, int32_t numLeft) {
  return svld1_u32(
      svwhilelt_b32_s32(0, numLeft),
      reinterpret_cast<const uint32_t*>(indices));
}

// Returns the filter cache byte of each active lane's dictionary index.
inline svuint32_t lookupFilterCache(
    svbool_t active,
    const uint8_t* filterCache,
    svuint32_t indices) {
  return svld1ub_gather_u32offset_u32(active, filterCache, indices);
}

// Calls 'resolveUnknown' for each distinct dictionary index among the
// 'unknown' lanes, then returns the reloaded cache bytes of the active lanes.
svuint32_t resolveUnknowns(
    svbool_t active,
    svbool_t unknown,
    svuint32_t indices,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown) {
  uint32_t laneIndices[kMaxLanes];
  uint32_t laneUnknown[kMaxLanes];
  svst1_u32(svptrue_b32(), laneIndices, indices);
  svst1_u32(svptrue_b32(), laneUnknown, svdup_n_u32_z(unknown, 1));
  const auto numLanes = static_cast<int32_t>(svcntw());
  for (int32_t lane = 0; lane < numLanes; ++lane) {
    // A lane repeating an index resolved by an earlier lane is skipped.
    if (laneUnknown[lane] && filterCache[laneIndices[lane]] == kUnknown) {
      resolveUnknown(static_cast<int32_t>(laneIndices[lane]));
    }
  }
  return lookupFilterCache(active, filterCache, indices);
}

// Appends the 'passed' lanes of 'rowNumbers', in lane order, to 'filterHits'.
// 'numPassed' is the number of 'passed' lanes.
inline void appendRows(
    svbool_t passed,
    svint32_t rowNumbers,
    uint64_t numPassed,
    int32_t* filterHits) {
  svst1_s32(
      svwhilelt_b32_u64(0, numPassed),
      filterHits,
      svcompact_s32(passed, rowNumbers));
}

// Appends the 'passed' lanes of 'indices', in lane order, to 'values'.
inline void appendIndices(
    svbool_t passed,
    svuint32_t indices,
    uint64_t numPassed,
    int32_t* values) {
  svst1_s32(
      svwhilelt_b32_u64(0, numPassed),
      values,
      svcompact_s32(passed, svreinterpret_s32_u32(indices)));
}

// Appends dictionary[index] of the 'passed' lanes, in lane order, to 'values'.
inline void appendValues(
    svbool_t passed,
    svuint32_t indices,
    const int32_t* dictionary,
    uint64_t numPassed,
    int32_t* values) {
  const svint32_t translated =
      svld1_gather_u32index_s32(passed, dictionary, indices);
  svst1_s32(
      svwhilelt_b32_u64(0, numPassed),
      values,
      svcompact_s32(passed, translated));
}

// SVE kernel of filterDictionaryRun() for int32_t values.
int32_t filterWithSve(
    const int32_t* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const int32_t* dictionary,
    int32_t* filterHits,
    int32_t* values) {
  const auto numLanes = static_cast<int32_t>(svcntw());
  int32_t numPassed = 0;
  for (int32_t i = 0; i < numIndices; i += numLanes) {
    const int32_t numLeft = numIndices - i;
    const svbool_t active = svwhilelt_b32_s32(0, numLeft);
    // All indices of the vector are loaded before anything is appended, so
    // 'values' may alias 'indices'.
    const svuint32_t dictIndices = loadIndices(indices + i, numLeft);

    svuint32_t cache = lookupFilterCache(active, filterCache, dictIndices);
    const svbool_t unknown = svcmpeq_n_u32(active, cache, kUnknown);
    if (FOLLY_UNLIKELY(svptest_any(active, unknown))) {
      cache = resolveUnknowns(
          active, unknown, dictIndices, filterCache, resolveUnknown);
    }

    const svbool_t passed = svcmpeq_n_u32(active, cache, kSuccess);
    const uint64_t numPassedHere = svcntp_b32(active, passed);
    appendRows(
        passed,
        svld1_s32(active, rows + i),
        numPassedHere,
        filterHits + numPassed);
    if (values != nullptr && dictionary != nullptr) {
      appendValues(
          passed, dictIndices, dictionary, numPassedHere, values + numPassed);
    } else if (values != nullptr) {
      appendIndices(passed, dictIndices, numPassedHere, values + numPassed);
    }
    numPassed += static_cast<int32_t>(numPassedHere);
  }
  return numPassed;
}

// Scalar kernel of filterDictionaryRun() for int64_t values. Every index writes
// its row, and its value if 'kStoreValues', to output slot 'numPassed', which
// advances only if the index passes. The slot never exceeds the index
// position, so 'values' may alias 'indices', and nothing is written past
// 'numIndices' slots.
template <bool kStoreValues>
int32_t filterScalar(
    const int64_t* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const int64_t* dictionary,
    int32_t* filterHits,
    int64_t* values) {
  int32_t numPassed = 0;
  for (int32_t i = 0; i < numIndices; ++i) {
    const auto index = static_cast<int32_t>(indices[i]);
    uint32_t state = filterCache[index];
    if (FOLLY_UNLIKELY(state == kUnknown)) {
      resolveUnknown(index);
      state = filterCache[index];
    }
    filterHits[numPassed] = rows[i];
    if constexpr (kStoreValues) {
      values[numPassed] = dictionary[index];
    }
    // kSuccess is 0x80 and kFailure is 0.
    numPassed += static_cast<int32_t>(state >> 7);
  }
  return numPassed;
}

} // namespace

bool dictionaryFilterEnabled() {
  return FLAGS_velox_sve_dictionary_filter;
}

template <typename T>
int32_t filterDictionaryRun(
    const T* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const T* dictionary,
    int32_t* filterHits,
    T* values) {
  static_assert(kSupportsDictionaryFilter<T>);
  VELOX_DCHECK(
      (std::is_same_v<T, int32_t> || dictionary != nullptr),
      "Only int32_t runs may omit the dictionary");
  if constexpr (std::is_floating_point_v<T>) {
    // Dictionary values are only copied, so a floating-point run is filtered
    // as the integers of the same width holding its bits. Its index slots
    // already hold integers: the index decoder writes make_index<T> values.
    using TBits =
        std::conditional_t<sizeof(T) == sizeof(int64_t), int64_t, int32_t>;
    return filterDictionaryRun<TBits>(
        reinterpret_cast<const TBits*>(indices),
        numIndices,
        rows,
        filterCache,
        resolveUnknown,
        reinterpret_cast<const TBits*>(dictionary),
        filterHits,
        reinterpret_cast<TBits*>(values));
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return filterWithSve(
        indices,
        numIndices,
        rows,
        filterCache,
        resolveUnknown,
        dictionary,
        filterHits,
        values);
  } else if (values != nullptr) {
    return filterScalar<true>(
        indices,
        numIndices,
        rows,
        filterCache,
        resolveUnknown,
        dictionary,
        filterHits,
        values);
  } else {
    return filterScalar<false>(
        indices,
        numIndices,
        rows,
        filterCache,
        resolveUnknown,
        dictionary,
        filterHits,
        values);
  }
}

template int32_t filterDictionaryRun(
    const int32_t* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const int32_t* dictionary,
    int32_t* filterHits,
    int32_t* values);

template int32_t filterDictionaryRun(
    const int64_t* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const int64_t* dictionary,
    int32_t* filterHits,
    int64_t* values);

template int32_t filterDictionaryRun(
    const float* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const float* dictionary,
    int32_t* filterHits,
    float* values);

template int32_t filterDictionaryRun(
    const double* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const double* dictionary,
    int32_t* filterHits,
    double* values);

} // namespace facebook::velox::dwio::common::sve

#endif
