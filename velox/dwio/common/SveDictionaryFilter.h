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
#include <type_traits>

#include <folly/Function.h>

// SVE kernel for the filter path of DictionaryColumnVisitor::processRun(): the
// Arm counterpart of its AVX2-shaped loop, which on NEON emulates gathers,
// movemask and permutes and branches on every group of 4 values. Only
// available when the translation unit is compiled with SVE enabled (for example
// -mcpu=olympus).
#if defined(__ARM_FEATURE_SVE)

namespace facebook::velox::dwio::common::sve {

/// True for the dictionary value types the kernel handles.
template <typename T>
constexpr bool kSupportsDictionaryFilter =
    std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

/// Returns whether DictionaryColumnVisitor uses filterDictionaryRun(), as set
/// by --velox_sve_dictionary_filter. Lets one binary A/B the kernel against the
/// generic loop.
bool dictionaryFilterEnabled();

/// Filters a run of dictionary indices through the dictionary filter cache and
/// appends the passing rows and, optionally, their dictionary values.
///
/// 'indices' holds 'numIndices' dictionary indices, one per T-sized slot, as
/// written by the index decoder. 'rows' holds the strictly increasing row
/// number of each index. 'filterCache' has one byte per dictionary entry:
/// FilterResult::kSuccess, kFailure or kUnknown. For each entry found
/// kUnknown, 'resolveUnknown' is called with its index and must set its byte
/// to kSuccess or kFailure.
///
/// Writes the row numbers of the passing indices to 'filterHits' and, if
/// 'values' is not null, dictionary[index] for each passing index to 'values',
/// both in input order. int32_t runs may pass a null 'dictionary'; 'values'
/// then receives the passing indices themselves, as string dictionary readers
/// keep indices and wrap the dictionary around them later. Like the generic
/// loop, which stores whole vectors, it
/// may write slots past the returned count but never past 'numIndices' slots.
/// 'values' may alias 'indices': a value is never written past the slot of the
/// index it comes from. Returns the number of passing indices.
///
/// int32_t runs use an SVE kernel and int64_t runs a scalar branchless one,
/// the faster of the two for each type on Olympus (see the .cpp). The kernels
/// never interpret dictionary values, they only copy them, and the filter is
/// applied by 'resolveUnknown', so float and double runs reuse the int32_t and
/// int64_t kernels on the same bits.
template <typename T>
int32_t filterDictionaryRun(
    const T* indices,
    int32_t numIndices,
    const int32_t* rows,
    uint8_t* filterCache,
    folly::FunctionRef<void(int32_t)> resolveUnknown,
    const T* dictionary,
    int32_t* filterHits,
    T* values);

} // namespace facebook::velox::dwio::common::sve

#endif
