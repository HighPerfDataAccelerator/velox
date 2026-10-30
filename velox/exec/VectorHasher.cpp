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

#include "velox/exec/VectorHasher.h"
#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Portability.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/serialization/NativeSerdeIO.h"
#include "velox/type/FloatingPointUtil.h"

#include <array>

#include <gflags/gflags.h>

DEFINE_int32(
    velox_vector_hasher_value_id_prefetch_distance,
    0,
    "Rows to look ahead when prefetching distinct-value table chunks for "
    "value-ID lookups; zero disables prefetch");

namespace facebook::velox::exec {

namespace {

constexpr uint32_t kVectorHasherStateMagic = 0x56485331; // VHS1
constexpr uint32_t kVectorHasherStateVersion = 3;

// Runs 'func' on each selected row. With a positive 'distance', first calls
// 'prefetch' on the selected row 'distance' positions ahead so that
// independent cache misses overlap.
template <typename Prefetch, typename Func>
void applyWithPrefetch(
    const SelectivityVector& rows,
    int32_t distance,
    Prefetch&& prefetch,
    Func&& func) {
  if (distance <= 0) {
    rows.applyToSelected(func);
    return;
  }
  if (rows.isAllSelected()) {
    const auto end = rows.end();
    for (auto row = rows.begin(); row < end; ++row) {
      if (row + distance < end) {
        prefetch(row + distance);
      }
      func(row);
    }
    return;
  }
  // Join probes arrive with rows already removed by null keys or a Bloom
  // filter; look ahead along the selected rows instead. 'func' may deselect
  // rows, so walk a copy of the selection.
  thread_local std::vector<vector_size_t> selected;
  selected.clear();
  rows.applyToSelected(
      [&](vector_size_t row) INLINE_LAMBDA { selected.push_back(row); });
  const auto numSelected = static_cast<int32_t>(selected.size());
  for (int32_t i = 0; i < numSelected; ++i) {
    if (i + distance < numSelected) {
      prefetch(selected[i + distance]);
    }
    func(selected[i]);
  }
}

} // namespace

int32_t VectorHasher::valueIdPrefetchDistance() const {
  return isRange_ ? 0 : FLAGS_velox_vector_hasher_value_id_prefetch_distance;
}

#define VALUE_ID_TYPE_DISPATCH(TEMPLATE_FUNC, typeKind, ...)                \
  [&]() {                                                                   \
    switch (typeKind) {                                                     \
      case TypeKind::BOOLEAN: {                                             \
        return TEMPLATE_FUNC<TypeKind::BOOLEAN>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::TINYINT: {                                             \
        return TEMPLATE_FUNC<TypeKind::TINYINT>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::SMALLINT: {                                            \
        return TEMPLATE_FUNC<TypeKind::SMALLINT>(__VA_ARGS__);              \
      }                                                                     \
      case TypeKind::INTEGER: {                                             \
        return TEMPLATE_FUNC<TypeKind::INTEGER>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::BIGINT: {                                              \
        return TEMPLATE_FUNC<TypeKind::BIGINT>(__VA_ARGS__);                \
      }                                                                     \
      case TypeKind::HUGEINT: {                                             \
        return TEMPLATE_FUNC<TypeKind::HUGEINT>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::VARCHAR:                                               \
      case TypeKind::VARBINARY: {                                           \
        return TEMPLATE_FUNC<TypeKind::VARCHAR>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::TIMESTAMP: {                                           \
        return TEMPLATE_FUNC<TypeKind::TIMESTAMP>(__VA_ARGS__);             \
      }                                                                     \
      default:                                                              \
        VELOX_UNREACHABLE(                                                  \
            "Unsupported value ID type: ", TypeKindName::toName(typeKind)); \
    }                                                                       \
  }()

namespace {
template <bool typeProvidesCustomComparison, TypeKind Kind>
uint64_t hashOne(DecodedVector& decoded, vector_size_t index) {
  if constexpr (
      Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
      Kind == TypeKind::MAP) {
    // Virtual function call for complex type.
    return decoded.base()->hashValueAt(decoded.index(index));
  } else {
    // Inlined for scalars.
    using T = typename KindToFlatVector<Kind>::HashRowType;
    T value = decoded.valueAt<T>(index);

    if constexpr (typeProvidesCustomComparison) {
      return static_cast<const CanProvideCustomComparisonType<Kind>*>(
                 decoded.base()->type().get())
          ->hash(value);
    } else if constexpr (std::is_floating_point_v<T>) {
      return util::floating_point::NaNAwareHash<T>()(value);
    } else {
      return folly::hasher<T>()(value);
    }
  }
}
} // namespace

template <bool typeProvidesCustomComparison, TypeKind Kind>
void VectorHasher::hashValues(
    const SelectivityVector& rows,
    bool mix,
    uint64_t* result) {
  if (decoded_.isConstantMapping()) {
    auto hash = decoded_.isNullAt(rows.begin())
        ? kNullHash
        : hashOne<typeProvidesCustomComparison, Kind>(decoded_, rows.begin());
    rows.applyToSelected([&](vector_size_t row) {
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  } else if (groupingSetHashCacheActive_) {
    rows.applyToSelected([&](vector_size_t row) {
      uint64_t hash;
      if (groupingSetHashValid_[row]) {
        hash = groupingSetHashes_[row];
        ++batchReuseStats_.encodedHashReusedRows;
      } else {
        hash = decoded_.isNullAt(row)
            ? kNullHash
            : hashOne<typeProvidesCustomComparison, Kind>(decoded_, row);
        groupingSetHashes_[row] = hash;
        groupingSetHashValid_[row] = 1;
        ++batchReuseStats_.encodedHashComputations;
      }
      ++batchReuseStats_.encodedHashRows;
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  } else if (
      !decoded_.isIdentityMapping() &&
      (rows.countSelected() > decoded_.base()->size() ||
       (batchReuseEnabled_ && hasRepeatedBaseIndices(rows)))) {
    cachedHashes_.resize(decoded_.base()->size());
    std::fill(cachedHashes_.begin(), cachedHashes_.end(), kNullHash);
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded_.isNullAt(row)) {
        result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
        return;
      }
      auto baseIndex = decoded_.index(row);
      uint64_t hash = cachedHashes_[baseIndex];
      if (hash == kNullHash) {
        hash = hashOne<typeProvidesCustomComparison, Kind>(decoded_, row);
        cachedHashes_[baseIndex] = hash;
        if (batchReuseEnabled_) {
          ++batchReuseStats_.encodedHashComputations;
        }
      } else if (batchReuseEnabled_) {
        ++batchReuseStats_.encodedHashReusedRows;
      }
      if (batchReuseEnabled_) {
        ++batchReuseStats_.encodedHashRows;
      }
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  } else {
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded_.isNullAt(row)) {
        result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
        return;
      }
      auto hash = hashOne<typeProvidesCustomComparison, Kind>(decoded_, row);
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  }
}

template <TypeKind Kind>
bool VectorHasher::makeValueIds(
    const SelectivityVector& rows,
    uint64_t* result) {
  using T = typename TypeTraits<Kind>::NativeType;

  if (decoded_.isConstantMapping()) {
    uint64_t id = decoded_.isNullAt(rows.begin())
        ? 0
        : valueId(decoded_.valueAt<T>(rows.begin()));
    if (id == kUnmappable) {
      analyzeValue(decoded_.valueAt<T>(rows.begin()));
      return false;
    }
    rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    });
    return true;
  }

  if (decoded_.isIdentityMapping()) {
    if (decoded_.mayHaveNulls()) {
      return makeValueIdsFlatWithNulls<T>(rows, result);
    } else {
      return makeValueIdsFlatNoNulls<T>(rows, result);
    }
  }

  if (decoded_.mayHaveNulls()) {
    return makeValueIdsDecoded<T, true>(rows, result);
  } else {
    return makeValueIdsDecoded<T, false>(rows, result);
  }
}

template <>
bool VectorHasher::makeValueIdsFlatNoNulls<bool>(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<uint64_t>();
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    bool value = bits::isBitSet(values, row);
    uint64_t id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <>
bool VectorHasher::makeValueIdsFlatWithNulls<bool>(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<uint64_t>();
  const auto* nulls = decoded_.nulls(&rows);
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (bits::isBitNull(nulls, row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
    bool value = bits::isBitSet(values, row);
    uint64_t id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <typename T, bool mayHaveNulls>
void VectorHasher::makeValueIdForOneRow(
    const uint64_t* nulls,
    vector_size_t row,
    const T* values,
    vector_size_t valueRow,
    uint64_t* result,
    bool& success) {
  if constexpr (mayHaveNulls) {
    if (bits::isBitNull(nulls, row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
  }
  T value = values[valueRow];
  if (!success) {
    analyzeValue(value);
    return;
  }
  auto id = valueId(value);
  if (id == kUnmappable) {
    success = false;
    analyzeValue(value);
  } else {
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  }
}

template <typename T>
bool VectorHasher::makeValueIdsFlatNoNulls(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<T>();
  if (isRange_ && tryMapToRange(values, rows, result)) {
    return true;
  }

  bool success = true;
  applyWithPrefetch(
      rows,
      valueIdPrefetchDistance(),
      [&](vector_size_t row) { prefetchValueId(values[row]); },
      [&](vector_size_t row) INLINE_LAMBDA {
        makeValueIdForOneRow<T, false>(
            nullptr, row, values, row, result, success);
      });

  return success;
}

template <typename T>
bool VectorHasher::makeValueIdsFlatWithNulls(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<T>();
  const auto* nulls = decoded_.nulls(&rows);

  bool success = true;
  applyWithPrefetch(
      rows,
      valueIdPrefetchDistance(),
      [&](vector_size_t row) {
        if (nulls == nullptr || !bits::isBitNull(nulls, row)) {
          prefetchValueId(values[row]);
        }
      },
      [&](vector_size_t row) INLINE_LAMBDA {
        makeValueIdForOneRow<T, true>(nulls, row, values, row, result, success);
      });
  return success;
}

template <typename T, bool mayHaveNulls>
bool VectorHasher::makeValueIdsDecoded(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.data<T>();
  bool success = true;

  if (rows.countSelected() <= decoded_.base()->size()) {
    // Cache is not beneficial in this case and we don't use them.
    auto* nulls = decoded_.nulls(&rows);
    applyWithPrefetch(
        rows,
        valueIdPrefetchDistance(),
        [&](vector_size_t row) {
          if (!mayHaveNulls || nulls == nullptr ||
              !bits::isBitNull(nulls, row)) {
            prefetchValueId(values[indices[row]]);
          }
        },
        [&](vector_size_t row) INLINE_LAMBDA {
          makeValueIdForOneRow<T, mayHaveNulls>(
              nulls, row, values, indices[row], result, success);
        });
    return success;
  }

  cachedHashes_.resize(decoded_.base()->size());
  std::fill(cachedHashes_.begin(), cachedHashes_.end(), 0);

  // Prefetch only dense selections, where every row ahead is valid to read.
  const auto distance = rows.isAllSelected() ? valueIdPrefetchDistance() : 0;
  const auto end = rows.end();
  int numCachedHashes = 0;
  rows.testSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (distance > 0 && row + distance < end) {
      const auto ahead = row + distance;
      if ((!mayHaveNulls || !decoded_.isNullAt(ahead)) &&
          cachedHashes_[indices[ahead]] == 0) {
        prefetchValueId(values[indices[ahead]]);
      }
    }
    if constexpr (mayHaveNulls) {
      if (decoded_.isNullAt(row)) {
        if (multiplier_ == 1) {
          result[row] = 0;
        }
        return true;
      }
    }

    auto baseIndex = indices[row];
    uint64_t& id = cachedHashes_[baseIndex];

    if (success) {
      if (id == 0) {
        T value = values[baseIndex];
        id = valueId(value);
        numCachedHashes++;
        if (id == kUnmappable) {
          analyzeValue(value);
          success = false;
        }
      }
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    } else {
      if (id == 0) {
        id = kUnmappable;
        numCachedHashes++;
        analyzeValue(values[baseIndex]);
      }
    }

    return success || numCachedHashes < cachedHashes_.size();
  });

  return success;
}

template <>
bool VectorHasher::makeValueIdsDecoded<bool, true>(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.data<uint64_t>();

  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (decoded_.isNullAt(row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }

    bool value = bits::isBitSet(values, indices[row]);
    auto id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <>
bool VectorHasher::makeValueIdsDecoded<bool, false>(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.data<uint64_t>();

  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    bool value = bits::isBitSet(values, indices[row]);
    auto id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

bool VectorHasher::computeValueIds(
    const SelectivityVector& rows,
    raw_vector<uint64_t>& result) {
  checkTypeSupportsValueIds();

  if (groupingSetValueIdReuseEnabled_ && groupingSetHashCacheActive_ &&
      rangeSize_ > 0 && !isRange_ && !distinctOverflow_ &&
      rows.hasSelections()) {
    return makeGroupingSetValueIds(rows, result);
  }
  if (batchReuseEnabled_ && rows.hasSelections()) {
    batchReuseStats_.valueIdRows += rows.countSelected();
    return VALUE_ID_TYPE_DISPATCH(
        makeValueIdsWithReuse, typeKind_, rows, result.data());
  }

  return VALUE_ID_TYPE_DISPATCH(makeValueIds, typeKind_, rows, result.data());
}

void VectorHasher::configureGroupingSetReuse(
    bool hashReuse,
    bool valueIdReuse,
    uint64_t stringBytes,
    int64_t stringValues) {
  VELOX_CHECK_EQ(numDistinct(), 0, "Configure grouping reuse before analysis");
  VELOX_CHECK_GE(stringBytes, 1 << 20);
  VELOX_CHECK_LE(stringBytes, 64 << 20);
  VELOX_CHECK_GE(stringValues, kMaxDistinct);
  VELOX_CHECK_LE(stringValues, 1000000);
  const bool string =
      typeKind_ == TypeKind::VARCHAR || typeKind_ == TypeKind::VARBINARY;
  groupingSetHashReuseEnabled_ = hashReuse;
  groupingSetValueIdReuseEnabled_ = valueIdReuse && string;
  maxDistinctStringBytes_ = string ? stringBytes : kMaxDistinctStringsBytes;
  maxDistinctStringValues_ = string ? stringValues : kMaxDistinct;
  finishGroupingSetHashInput();
  groupingSetHashEpoch_.reset();
  groupingSetHashInput_.reset();
  cachedValueIdCount_ = 0;
  groupingSetHashValid_.resize(0);
  groupingBufferContainsIds_ = false;
}

int64_t VectorHasher::distinctValueLimit() const {
  return typeKind_ == TypeKind::VARCHAR || typeKind_ == TypeKind::VARBINARY
      ? maxDistinctStringValues_
      : kMaxDistinct;
}

VectorHasher::GroupingSetValueIdStats
VectorHasher::takeGroupingSetValueIdStats() {
  return {
      std::exchange(valueIdEpochRows_, 0),
      std::exchange(valueIdEpochLookups_, 0),
      std::exchange(valueIdEpochReusedRows_, 0),
      std::exchange(valueIdEpochInvalidations_, 0)};
}

void VectorHasher::setGroupingSetHashInput(
    uint64_t epoch,
    const VectorPtr& values,
    memory::MemoryPool* pool) {
  finishGroupingSetHashInput();
  if (!groupingSetHashReuseEnabled_ && !groupingSetValueIdReuseEnabled_) {
    return;
  }
  if (groupingSetHashEpoch_ != epoch) {
    groupingSetHashEpoch_ = epoch;
    groupingSetHashInput_.reset();
    cachedValueIdCount_ = 0;
  }
  // Constant/null projections preserve the source cache for later projections.
  if (values->isConstantEncoding()) {
    return;
  }
  if (!groupingSetHashPool_) {
    groupingSetHashPool_ = pool ? pool : values->pool();
    groupingSetHashes_ = raw_vector<uint64_t>(groupingSetHashPool_);
    groupingSetHashValid_ = raw_vector<uint8_t>(groupingSetHashPool_);
  }
  if (groupingSetHashInput_.get() != values.get() ||
      groupingSetHashes_.size() != values->size()) {
    groupingSetHashInput_ = values;
    groupingSetHashes_.resize(values->size());
    groupingSetHashValid_.resize(values->size());
    std::fill(groupingSetHashValid_.begin(), groupingSetHashValid_.end(), 0);
    cachedValueIdCount_ = 0;
    cachedDictionaryGeneration_ = dictionaryGeneration_;
  }
  // Reuse the established inline decode identity guard for either buffer use.
  groupingSetHashCacheActive_ = true;
}

bool VectorHasher::makeGroupingSetValueIds(
    const SelectivityVector& rows,
    raw_vector<uint64_t>& result) {
  if (!groupingBufferContainsIds_ ||
      cachedDictionaryGeneration_ != dictionaryGeneration_) {
    valueIdEpochInvalidations_ += cachedValueIdCount_ != 0;
    std::fill(groupingSetHashValid_.begin(), groupingSetHashValid_.end(), 0);
    cachedValueIdCount_ = 0;
    cachedDictionaryGeneration_ = dictionaryGeneration_;
  }
  groupingBufferContainsIds_ = true;
  valueIdEpochRows_ += rows.countSelected();
  auto apply = [&](vector_size_t row, uint64_t id) {
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  };
  if (cachedValueIdCount_ == groupingSetHashes_.size()) {
    valueIdEpochReusedRows_ += rows.countSelected();
    rows.applyToSelected(
        [&](vector_size_t row) { apply(row, groupingSetHashes_[row]); });
    return true;
  }
  bool success = true;
  const bool mayHaveNulls = decoded_.mayHaveNulls();
  applyWithPrefetch(
      rows,
      valueIdPrefetchDistance(),
      [&](vector_size_t row) {
        if (!groupingSetHashValid_[row] &&
            (!mayHaveNulls || !decoded_.isNullAt(row))) {
          prefetchValueId(decoded_.valueAt<StringView>(row));
        }
      },
      [&](vector_size_t row) {
        if (!success) {
          if (!mayHaveNulls || !decoded_.isNullAt(row)) {
            analyzeValue(decoded_.valueAt<StringView>(row));
          }
          return;
        }
        uint64_t id;
        if (groupingSetHashValid_[row]) {
          id = groupingSetHashes_[row];
          ++valueIdEpochReusedRows_;
        } else {
          if (mayHaveNulls && decoded_.isNullAt(row)) {
            id = 0;
          } else {
            ++valueIdEpochLookups_;
            const auto value = decoded_.valueAt<StringView>(row);
            id = valueId(value);
            if (id == kUnmappable || id >= rangeSize_) {
              analyzeValue(value);
              success = false;
              return;
            }
          }
          groupingSetHashes_[row] = id;
          groupingSetHashValid_[row] = 1;
          ++cachedValueIdCount_;
        }
        apply(row, id);
      });
  // An overflow may have cleared the codebook while analyzing this batch.
  if (cachedDictionaryGeneration_ != dictionaryGeneration_) {
    valueIdEpochInvalidations_ += cachedValueIdCount_ != 0;
    std::fill(groupingSetHashValid_.begin(), groupingSetHashValid_.end(), 0);
    cachedValueIdCount_ = 0;
    cachedDictionaryGeneration_ = dictionaryGeneration_;
  }
  return success;
}

bool VectorHasher::hasRepeatedBaseIndices(const SelectivityVector& rows) const {
  int samples = 0;
  int repeats = 0;
  vector_size_t previous = -1;
  rows.testSelected([&](vector_size_t row) {
    if (!decoded_.isNullAt(row)) {
      const auto index = decoded_.index(row);
      repeats += index == previous;
      previous = index;
    }
    return ++samples < 32 && repeats < 2;
  });
  return repeats >= 2;
}

template <TypeKind Kind>
bool VectorHasher::makeValueIdsWithReuse(
    const SelectivityVector& rows,
    uint64_t* result) {
  using T = typename TypeTraits<Kind>::NativeType;
  auto legacy = [&]() {
    batchReuseStats_.valueIdLegacyRows += rows.countSelected();
    return makeValueIds<Kind>(rows, result);
  };
  if constexpr (Kind == TypeKind::BOOLEAN) {
    return legacy();
  }
  if (decoded_.isConstantMapping() || isRange_ || rows.countSelected() < 32) {
    return legacy();
  }
  if (!decoded_.isIdentityMapping()) {
    if (!hasRepeatedBaseIndices(rows)) {
      return legacy();
    }
    cachedHashes_.resize(decoded_.base()->size());
    std::fill(cachedHashes_.begin(), cachedHashes_.end(), 0);
    bool success = true;
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded_.isNullAt(row)) {
        if (multiplier_ == 1) {
          result[row] = 0;
        }
        return;
      }
      auto& id = cachedHashes_[decoded_.index(row)];
      if (!success) {
        if (id == 0) {
          id = kUnmappable;
          analyzeValue(decoded_.valueAt<T>(row));
        }
        return;
      }
      if (id == 0) {
        const auto value = decoded_.valueAt<T>(row);
        id = valueId(value);
        if (id == kUnmappable) {
          analyzeValue(value);
          success = false;
          return;
        }
      } else {
        ++batchReuseStats_.valueIdReusedRows;
      }
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    });
    return success;
  }

  // Only select the run loop when local repetition amortizes its comparison.
  T previous{};
  bool havePrevious = false;
  int samples = 0;
  int repeats = 0;
  rows.testSelected([&](vector_size_t row) {
    if (!decoded_.isNullAt(row)) {
      auto value = decoded_.valueAt<T>(row);
      repeats += havePrevious && value == previous;
      previous = value;
      havePrevious = true;
    }
    return ++samples < 32 && repeats < 2;
  });
  if (repeats < 2) {
    return legacy();
  }
  havePrevious = false;
  uint64_t previousId = 0;
  bool success = true;
  rows.applyToSelected([&](vector_size_t row) {
    if (decoded_.isNullAt(row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
    const auto value = decoded_.valueAt<T>(row);
    if (havePrevious && value == previous) {
      if (success) {
        ++batchReuseStats_.valueIdReusedRows;
      }
    } else {
      previous = value;
      havePrevious = true;
      if (success) {
        previousId = valueId(value);
        if (previousId == kUnmappable) {
          success = false;
          analyzeValue(value);
        }
      } else {
        analyzeValue(value);
      }
    }
    if (success) {
      result[row] = multiplier_ == 1 ? previousId
                                     : result[row] + multiplier_ * previousId;
    }
  });
  return success;
}

template <TypeKind Kind>
uint64_t VectorHasher::analyzeDecodedForHashJoin(
    const SelectivityVector& rows) {
  using T = typename TypeTraits<Kind>::NativeType;
  if constexpr (Kind == TypeKind::BOOLEAN) {
    return 0;
  }
  if constexpr (
      std::is_integral_v<T> && sizeof(T) <= sizeof(int64_t) &&
      Kind != TypeKind::BOOLEAN) {
    if (distinctOverflow_ && !rangeOverflow_ && decoded_.isIdentityMapping() &&
        !decoded_.mayHaveNulls() && rows.isAllSelected()) {
      // Range-only integer batches need no per-row dictionary bookkeeping.
      const auto* values = decoded_.data<T>();
      T min = values[rows.begin()];
      T max = min;
      for (auto i = rows.begin() + 1; i < rows.end(); ++i) {
        min = std::min(min, values[i]);
        max = std::max(max, values[i]);
      }
      updateRange(min);
      updateRange(max);
      return 0;
    }
  }

  // Avoid an extra comparison on every low-cardinality, unordered key. This
  // sample selects only a local fast path; it never omits distinct values.
  bool useRuns = false;
  if (decoded_.isIdentityMapping()) {
    T previous{};
    bool hasPrevious = false;
    int32_t samples = 0;
    int32_t repeats = 0;
    rows.testSelected([&](vector_size_t row) {
      if (!decoded_.isNullAt(row)) {
        auto value = decoded_.valueAt<T>(row);
        repeats += hasPrevious && value == previous;
        previous = value;
        hasPrevious = true;
      }
      return ++samples < 32 && repeats < 2;
    });
    useRuns = repeats >= 2;
  }

  if constexpr (Kind == TypeKind::VARCHAR || Kind == TypeKind::VARBINARY) {
    if (!useRuns && !distinctOverflow_ && decoded_.isIdentityMapping() &&
        !decoded_.mayHaveNulls()) {
      // Unordered low-cardinality strings already have an efficient legacy
      // lookup loop. Retain it rather than charging extra per-row branches.
      // Identity mapping does not use cachedHashes_ internally, so it is safe
      // to reuse this buffer for its otherwise-discarded temporary IDs.
      cachedHashes_.resize(rows.end());
      makeValueIds<Kind>(rows, cachedHashes_.data());
      return HashJoinAnalysis::kUnknownLookupCount;
    }
  }

  if constexpr (
      std::is_integral_v<T> && sizeof(T) <= sizeof(int64_t) &&
      Kind != TypeKind::BOOLEAN) {
    if (!useRuns && !distinctOverflow_ && decoded_.isIdentityMapping() &&
        !decoded_.mayHaveNulls() &&
        numDistinct() + rows.countSelected() <= kMaxDistinct) {
      // The whole batch fits even if every selected key is new. Hoist the
      // overflow checks rather than charging them to each low-cardinality
      // lookup. Unlike computeValueIds, no temporary ID array is written.
      const auto* values = decoded_.data<T>();
      rows.applyToSelected([&](vector_size_t row) { valueId(values[row]); });
      if (rangeOverflow_) {
        setRangeOverflow();
      }
      return rows.countSelected();
    }
  }

  auto analyzeRows = [&]<bool useRuns>() -> uint64_t {
    uint64_t dictionaryLookups = 0;
    T previous{};
    bool hasPrevious = false;
    auto analyzeValueForJoin = [&](T value) {
      if constexpr (useRuns) {
        if (hasPrevious && value == previous) {
          return;
        }
        previous = value;
        hasPrevious = true;
      }
      if constexpr (Kind == TypeKind::TIMESTAMP) {
        if (value.getNanos() % Timestamp::kNanosecondsInMillisecond != 0) {
          setRangeOverflow();
          setDistinctOverflow();
          return;
        }
      }
      if (!distinctOverflow_) {
        ++dictionaryLookups;
        // valueId updates bounds and copies strings only for newly inserted
        // values. The temporary ID itself is intentionally discarded.
        if (valueId(value) == kUnmappable) {
          if (rangeOverflow_) {
            // Numeric valueId must not revive an overflowed range.
            setRangeOverflow();
          }
          if (numDistinct() > kMaxDistinct) {
            setDistinctOverflow();
          }
        }
      } else {
        analyzeValue(value);
      }
    };
    if (decoded_.isConstantMapping()) {
      if (!decoded_.isNullAt(rows.begin())) {
        analyzeValueForJoin(decoded_.valueAt<T>(rows.begin()));
      }
    } else if (decoded_.isIdentityMapping()) {
      if constexpr (Kind != TypeKind::BOOLEAN) {
        const auto* values = decoded_.data<T>();
        if (decoded_.mayHaveNulls()) {
          const auto* nulls = decoded_.nulls(&rows);
          rows.applyToSelected([&](vector_size_t row) {
            if (!bits::isBitNull(nulls, row)) {
              analyzeValueForJoin(values[row]);
            }
          });
        } else {
          rows.applyToSelected(
              [&](vector_size_t row) { analyzeValueForJoin(values[row]); });
        }
      }
    } else if (rows.countSelected() <= decoded_.base()->size()) {
      rows.applyToSelected([&](vector_size_t row) {
        if (!decoded_.isNullAt(row)) {
          analyzeValueForJoin(decoded_.valueAt<T>(row));
        }
      });
    } else {
      // Outer nulls must not hide a later valid reference to the same entry.
      cachedHashes_.resize(decoded_.base()->size());
      std::fill(cachedHashes_.begin(), cachedHashes_.end(), 0);
      const auto* indices = decoded_.indices();
      rows.applyToSelected([&](vector_size_t row) {
        if (!decoded_.isNullAt(row) && cachedHashes_[indices[row]] == 0) {
          cachedHashes_[indices[row]] = 1;
          analyzeValueForJoin(decoded_.valueAt<T>(row));
        }
      });
    }
    return dictionaryLookups;
  };
  return useRuns ? analyzeRows.template operator()<true>()
                 : analyzeRows.template operator()<false>();
}

VectorHasher::HashJoinAnalysis VectorHasher::analyzeForHashJoin(
    const SelectivityVector& rows,
    uint64_t numInputRows,
    uint32_t minRows,
    uint32_t minDistinctPct) {
  checkTypeSupportsValueIds();
  VELOX_DCHECK_GE(minRows, 1);
  VELOX_DCHECK_GE(minDistinctPct, 1);
  VELOX_DCHECK_LE(minDistinctPct, 100);
  VELOX_DCHECK(!isRange_);
  const auto capacity = [&]() {
    return typeKind_ == TypeKind::HUGEINT ? uniqueHugeintValues_.bucket_count()
                                          : uniqueValues_.bucket_count();
  };
  const auto before = capacity();
  const auto dictionaryLookups = rows.hasSelections()
      ? VALUE_ID_TYPE_DISPATCH(analyzeDecodedForHashJoin, typeKind_, rows)
      : 0;
  const auto distincts = numDistinct();
  const bool grew = capacity() > before;
  const bool abandon = !distinctOverflow_ && numInputRows >= minRows &&
      distincts * 100 / numInputRows >= minDistinctPct;
  if (abandon) {
    setDistinctOverflow();
  }
  // Validate bounds even before the final hash-mode decision. This lets a
  // wide sparse domain stop analysis once neither representation is viable.
  uint64_t asRange;
  uint64_t asDistinct;
  cardinality(0, asRange, asDistinct);
  return {
      distincts,
      grew,
      abandon,
      distinctOverflow_ && mayUseValueIds(),
      dictionaryLookups};
}

bool VectorHasher::computeValueIdsForRows(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    raw_vector<uint64_t>& result) {
  checkTypeSupportsValueIds();

  return VALUE_ID_TYPE_DISPATCH(
      makeValueIdsForRows,
      typeKind_,
      groups,
      numGroups,
      offset,
      nullByte,
      nullMask,
      result.data());
}

template <>
bool VectorHasher::makeValueIdsForRows<TypeKind::VARCHAR>(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* result) {
  const auto distance = valueIdPrefetchDistance();
  for (int32_t i = 0; i < numGroups; ++i) {
    if (distance > 0) {
      prefetchRowValueIds<StringView>(
          groups, numGroups, i, distance, offset, nullByte, nullMask);
    }
    if (isNullAt(groups[i], nullByte, nullMask)) {
      if (multiplier_ == 1) {
        result[i] = 0;
      }
    } else {
      std::string storage;
      auto id = valueId<StringView>(HashStringAllocator::contiguousString(
          valueAt<StringView>(groups[i], offset), storage));
      if (id == kUnmappable) {
        return false;
      }
      result[i] = multiplier_ == 1 ? id : result[i] + multiplier_ * id;
    }
  }
  return true;
}

template <TypeKind Kind>
void VectorHasher::lookupValueIdsTyped(
    const DecodedVector& decoded,
    SelectivityVector& rows,
    raw_vector<uint64_t>& hashes,
    uint64_t* result) const {
  using T = typename TypeTraits<Kind>::NativeType;
  if (decoded.isConstantMapping()) {
    if (decoded.isNullAt(rows.begin())) {
      if (multiplier_ == 1) {
        rows.applyToSelected([&](vector_size_t row)
                                 INLINE_LAMBDA { result[row] = 0; });
      }
      return;
    }

    uint64_t id = lookupValueId(decoded.valueAt<T>(rows.begin()));
    if (id == kUnmappable) {
      rows.clearAll();
    } else {
      rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
        result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
      });
    }
    return;
  }

  if (decoded.isIdentityMapping() && !decoded.mayHaveNulls()) {
    if (Kind == TypeKind::BIGINT && isRange_) {
      lookupIdsRangeSimd<int64_t>(decoded, rows, result);
      rows.updateBounds();
      return;
    }
    if (Kind == TypeKind::INTEGER && isRange_) {
      lookupIdsRangeSimd<int32_t>(decoded, rows, result);
      rows.updateBounds();
      return;
    }
  }

  if constexpr (
      std::is_integral_v<T> && !std::is_same_v<T, bool> &&
      sizeof(T) <= sizeof(int64_t)) {
    if (!isRange_ && valueIdPrefetchDistance() > 0 &&
        (decoded.isIdentityMapping() ||
         rows.countSelected() <= decoded.base()->size())) {
      // Hash each key once: its F14 token both prefetches the chunk and
      // drives the lookup, and a block of tokens keeps that many misses in
      // flight.
      constexpr int32_t kBlock = 64;
      std::array<vector_size_t, kBlock> blockRows;
      std::array<int64_t, kBlock> blockValues;
      std::array<folly::F14HashToken, kBlock> tokens;
      int32_t numInBlock = 0;
      auto lookupBlock = [&]() {
        for (int32_t i = 0; i < numInBlock; ++i) {
          const auto row = blockRows[i];
          const auto iter =
              uniqueValues_.find(tokens[i], UniqueValue(blockValues[i]));
          if (iter == uniqueValues_.end()) {
            rows.setValid(row, false);
            continue;
          }
          const uint64_t id = iter->id();
          result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
        }
        numInBlock = 0;
      };
      rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
        if (decoded.isNullAt(row)) {
          if (multiplier_ == 1) {
            result[row] = 0;
          }
          return;
        }
        blockValues[numInBlock] = decoded.valueAt<T>(row);
        tokens[numInBlock] =
            uniqueValues_.prehash(UniqueValue(blockValues[numInBlock]));
        uniqueValues_.prefetch(tokens[numInBlock]);
        blockRows[numInBlock] = row;
        if (++numInBlock == kBlock) {
          lookupBlock();
        }
      });
      lookupBlock();
      rows.updateBounds();
      return;
    }
  }

  if (decoded.isIdentityMapping() ||
      rows.countSelected() <= decoded.base()->size()) {
    // Probe rows look up the distinct-value table one at a time, like the
    // build side; prefetch the chunk of the row 'distance' ahead so that the
    // misses of consecutive probes overlap. Rows that a previous key cleared
    // from 'rows' are still valid to read.
    applyWithPrefetch(
        rows,
        valueIdPrefetchDistance(),
        [&](vector_size_t row) {
          if (!decoded.isNullAt(row)) {
            prefetchValueId(decoded.valueAt<T>(row));
          }
        },
        [&](vector_size_t row) INLINE_LAMBDA {
          if (decoded.isNullAt(row)) {
            if (multiplier_ == 1) {
              result[row] = 0;
            }
            return;
          }
          T value = decoded.valueAt<T>(row);
          uint64_t id = lookupValueId(value);
          if (id == kUnmappable) {
            rows.setValid(row, false);
            return;
          }
          result[row] =
              multiplier_ == 1 ? id : result[row] + multiplier_ * id;
        });
    rows.updateBounds();
    return;
  }

  hashes.resize(decoded.base()->size());
  std::fill(hashes.begin(), hashes.end(), 0);
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (decoded.isNullAt(row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
    auto baseIndex = decoded.index(row);
    uint64_t id = hashes[baseIndex];
    if (id == 0) {
      T value = decoded.valueAt<T>(row);
      id = lookupValueId(value);
      if (id == kUnmappable) {
        rows.setValid(row, false);
        return;
      }
      hashes[baseIndex] = id;
    }
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  rows.updateBounds();
}

template <typename T>
void VectorHasher::lookupIdsRangeSimd(
    const DecodedVector& decoded,
    SelectivityVector& rows,
    uint64_t* result) const {
  static_assert(sizeof(T) == 8 || sizeof(T) == 4);
  auto lower = xsimd::batch<T>::broadcast(min_);
  auto upper = xsimd::batch<T>::broadcast(max_);
  auto data = decoded.data<T>();
  uint64_t offset = min_ - 1;
  auto bits = rows.asMutableRange().bits();
  bits::forBatches<xsimd::batch<T>::size>(
      bits, rows.begin(), rows.end(), [&](auto index, auto /*mask*/) {
        auto values = xsimd::batch<T>::load_unaligned(data + index);
        uint64_t outOfRange =
            simd::toBitMask(lower > values) | simd::toBitMask(values > upper);
        if (outOfRange) {
          bits[index / 64] &= ~(outOfRange << (index & 63));
        }
        if (outOfRange != bits::lowMask(xsimd::batch<T>::size)) {
          if constexpr (sizeof(T) == 8) {
            auto unsignedValues =
                simd::reinterpretBatch<typename std::make_unsigned<T>::type>(
                    values);
            if (multiplier_ == 1) {
              (unsignedValues - offset).store_unaligned(result + index);
            } else {
              (xsimd::batch<uint64_t>::load_unaligned(result + index) +
               ((unsignedValues - offset) * multiplier_))
                  .store_unaligned(result + index);
            }
          } else {
            // Widen 8 to 2 x 4 since result is always 64 bits wide.
            auto first4 = simd::reinterpretBatch<uint64_t>(
                              simd::getHalf<int64_t, 0>(values)) -
                offset;
            auto next4 = simd::reinterpretBatch<uint64_t>(
                             simd::getHalf<int64_t, 1>(values)) -
                offset;
            if (multiplier_ == 1) {
              first4.store_unaligned(result + index);
              next4.store_unaligned(result + index + first4.size);
            } else {
              (xsimd::batch<uint64_t>::load_unaligned(result + index) +
               (first4 * multiplier_))
                  .store_unaligned(result + index);
              (xsimd::batch<uint64_t>::load_unaligned(
                   result + index + first4.size) +
               (next4 * multiplier_))
                  .store_unaligned(result + index + first4.size);
            }
          }
        }
      });
}

void VectorHasher::lookupValueIds(
    const BaseVector& values,
    SelectivityVector& rows,
    ScratchMemory& scratchMemory,
    raw_vector<uint64_t>& result) const {
  checkTypeSupportsValueIds();

  scratchMemory.decoded.decode(values, rows);
  VALUE_ID_TYPE_DISPATCH(
      lookupValueIdsTyped,
      typeKind_,
      scratchMemory.decoded,
      rows,
      scratchMemory.hashes,
      result.data());
}

void VectorHasher::hash(
    const SelectivityVector& rows,
    bool mix,
    raw_vector<uint64_t>& result) {
  if (!groupingSetHashReuseEnabled_) {
    groupingSetHashCacheActive_ = false;
  } else if (groupingSetHashCacheActive_ && groupingBufferContainsIds_) {
    std::fill(groupingSetHashValid_.begin(), groupingSetHashValid_.end(), 0);
    groupingBufferContainsIds_ = false;
    cachedValueIdCount_ = 0;
  }
  if (typeKind_ == TypeKind::UNKNOWN) {
    rows.applyToSelected([&](auto row) {
      result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
    });
  } else {
    if (typeProvidesCustomComparison_) {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
          hashValues, true, typeKind_, rows, mix, result.data());
    } else {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
          hashValues, false, typeKind_, rows, mix, result.data());
    }
  }
}

void VectorHasher::hashPrecomputed(
    const SelectivityVector& rows,
    bool mix,
    raw_vector<uint64_t>& result) const {
  rows.applyToSelected([&](vector_size_t row) {
    result[row] =
        mix ? bits::hashMix(result[row], precomputedHash_) : precomputedHash_;
  });
}

void VectorHasher::precompute(const BaseVector& value) {
  if (value.isNullAt(0)) {
    precomputedHash_ = kNullHash;
    return;
  }

  const SelectivityVector rows(1, true);
  decoded_.decode(value, rows);

  if (typeProvidesCustomComparison_) {
    precomputedHash_ = VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
        hashOne, true, typeKind_, decoded_, 0);
  } else {
    precomputedHash_ = VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
        hashOne, false, typeKind_, decoded_, 0);
  }
}

void VectorHasher::analyze(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask) {
  checkTypeSupportsValueIds();

  VALUE_ID_TYPE_DISPATCH(
      analyzeTyped, typeKind_, groups, numGroups, offset, nullByte, nullMask);
}

uint64_t VectorHasher::valueId(int128_t value) {
  if (isRange_) {
    return kUnmappable;
  }

  auto nextId = uniqueHugeintValues_.size() + 1;
  auto [iterator, inserted] = uniqueHugeintValues_.emplace(value, nextId);
  if (!inserted) {
    return iterator->second;
  }

  hasHugeintValue_ = true;
  if (uniqueHugeintValues_.size() >= rangeSize_) {
    return kUnmappable;
  }
  return iterator->second;
}

uint64_t VectorHasher::lookupValueId(int128_t value) const {
  if (isRange_) {
    return kUnmappable;
  }

  auto iterator = uniqueHugeintValues_.find(value);
  if (iterator != uniqueHugeintValues_.end()) {
    return iterator->second;
  }
  return kUnmappable;
}

void VectorHasher::analyzeValue(int128_t value) {
  hasHugeintValue_ = true;
  if (!distinctOverflow_) {
    auto nextId = uniqueHugeintValues_.size() + 1;
    if (uniqueHugeintValues_.emplace(value, nextId).second &&
        uniqueHugeintValues_.size() > kMaxDistinct) {
      setDistinctOverflow();
    }
  }
}

template <>
void VectorHasher::analyzeValue(StringView value) {
  int size = value.size();
  auto data = value.data();
  if (!rangeOverflow_) {
    if (size > kStringASRangeMaxSize) {
      setRangeOverflow();
    } else {
      int64_t number = stringAsNumber(data, size);
      updateRange(number);
    }
  }
  if (!distinctOverflow_) {
    UniqueValue unique(data, size);
    unique.setId(uniqueValues_.size() + 1);
    auto pair = uniqueValues_.insert(unique);
    if (pair.second) {
      if (uniqueValues_.size() > distinctValueLimit()) {
        setDistinctOverflow();
        return;
      }
      copyStringToLocal(&*pair.first);
    }
  }
}

void VectorHasher::copyStringToLocal(const UniqueValue* unique) {
  copyStringToLocalImpl(unique, true);
}

void VectorHasher::copyStringToLocalImpl(
    const UniqueValue* unique,
    bool enforceBudget) {
  auto size = unique->size();
  if (size <= sizeof(int64_t)) {
    return;
  }
  if (enforceBudget && distinctStringsBytes_ > maxDistinctStringBytes_) {
    setDistinctOverflow();
    return;
  }
  if (uniqueValuesStorage_.empty()) {
    uniqueValuesStorage_.emplace_back();
    uniqueValuesStorage_.back().reserve(std::max(kStringBufferUnitSize, size));
    distinctStringsBytes_ += uniqueValuesStorage_.back().capacity();
  }
  auto str = &uniqueValuesStorage_.back();
  if (str->size() + size > str->capacity()) {
    uniqueValuesStorage_.emplace_back();
    uniqueValuesStorage_.back().reserve(std::max(kStringBufferUnitSize, size));
    distinctStringsBytes_ += uniqueValuesStorage_.back().capacity();
    str = &uniqueValuesStorage_.back();
  }
  auto start = str->size();
  str->resize(start + size);
  memcpy(str->data() + start, reinterpret_cast<char*>(unique->data()), size);
  const_cast<UniqueValue*>(unique)->setData(
      reinterpret_cast<int64_t>(str->data() + start));
}

void VectorHasher::setDistinctOverflow() {
  distinctOverflow_ = true;
  clearDistinctValues();
}

void VectorHasher::setRangeOverflow() {
  rangeOverflow_ = true;
  hasRange_ = false;
}

std::unique_ptr<common::Filter> VectorHasher::getFilter(
    bool nullAllowed) const {
  if (typeProvidesCustomComparison_) {
    return nullptr;
  }

  switch (typeKind_) {
    case TypeKind::TINYINT:
      [[fallthrough]];
    case TypeKind::SMALLINT:
      [[fallthrough]];
    case TypeKind::INTEGER:
      [[fallthrough]];
    case TypeKind::BIGINT:
      if (!distinctOverflow_) {
        std::vector<int64_t> values;
        values.reserve(numDistinct());
        for (const auto& value : uniqueValues_) {
          values.emplace_back(value.data());
        }

        return common::createBigintValues(values, nullAllowed);
      }
      [[fallthrough]];
    case TypeKind::HUGEINT:
      if (!distinctOverflow_) {
        std::vector<int128_t> values;
        values.reserve(numDistinct());
        for (const auto& entry : uniqueHugeintValues_) {
          values.emplace_back(entry.first);
        }
        return common::createHugeintValues(values, nullAllowed);
      }
      [[fallthrough]];
    case TypeKind::VARCHAR:
      [[fallthrough]];
    case TypeKind::VARBINARY:
      if (!distinctOverflow_) {
        std::vector<std::string> values;
        values.reserve(numDistinct());
        for (const auto& value : uniqueValues_) {
          values.emplace_back(value.asString());
        }
        return std::make_unique<common::BytesValues>(values, nullAllowed);
      }
      [[fallthrough]];
    default:
      return nullptr;
  }
}

namespace {
template <typename T>
// Adds 'reserve' to either end of the range between 'min' and 'max' while
// staying in the range of T.
void extendRange(int64_t reserve, int64_t& min, int64_t& max) {
  int64_t kMin = std::numeric_limits<T>::min();
  int64_t kMax = std::numeric_limits<T>::max();
  if (kMin + reserve + 1 > min) {
    min = kMin;
  } else {
    min -= reserve;
  }
  if (kMax - reserve < max) {
    max = kMax;
  } else {
    max += reserve;
  }
}

// Adds 'reservePct' % to either end of the range between 'min' and 'max'
// while staying in the range of 'kind'.
void extendRange(
    TypeKind kind,
    int32_t reservePct,
    int64_t& min,
    int64_t& max) {
  // The reserve is 2 + reservePct % of the range. Add 2 to make sure
  // that a non-0 peercentage actually adds something for a small
  // range.
  int64_t reserve =
      reservePct == 0 ? 0 : 2 + (max - min) * (reservePct / 100.0);
  switch (kind) {
    case TypeKind::BOOLEAN:
      break;
    case TypeKind::TINYINT:
      extendRange<int8_t>(reserve, min, max);
      break;
    case TypeKind::SMALLINT:
      extendRange<int16_t>(reserve, min, max);
      break;
    case TypeKind::INTEGER:
      extendRange<int32_t>(reserve, min, max);
      break;
    case TypeKind::BIGINT:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      extendRange<int64_t>(reserve, min, max);
      break;

    default:
      VELOX_FAIL("Unsupported VectorHasher typeKind {}", kind);
  }
}

int64_t addIdReserve(
    size_t numDistinct,
    int32_t reservePct,
    int64_t limit = VectorHasher::kMaxDistinct) {
  // A merge of hashers in a hash join build may end up over the limit, so
  // return that.
  if (numDistinct > limit) {
    return numDistinct;
  }
  if (reservePct == VectorHasher::kNoLimit) {
    return limit;
  }
  // NOTE: 'kMaxDistinct' is a small value so no need to check overflow for
  // reservation here.
  return std::min<int64_t>(limit, numDistinct * (1 + (reservePct / 100.0)));
}
} // namespace

void VectorHasher::cardinality(
    int32_t reservePct,
    uint64_t& asRange,
    uint64_t& asDistincts) {
  if (!typeSupportsValueIds()) {
    asRange = kRangeTooLarge;
    asDistincts = kRangeTooLarge;
    return;
  }

  if (typeKind_ == TypeKind::BOOLEAN) {
    hasRange_ = true;
    asRange = 3;
    asDistincts = 3;
    return;
  }
  if (typeKind_ == TypeKind::HUGEINT) {
    asRange = kRangeTooLarge;
    asDistincts = distinctOverflow_
        ? kRangeTooLarge
        : addIdReserve(numDistinct(), reservePct) + 1;
    return;
  }
  int64_t signedRange;
  if (!hasRange_ || rangeOverflow_) {
    asRange = kRangeTooLarge;
  } else if (__builtin_sub_overflow(max_, min_, &signedRange)) {
    setRangeOverflow();
    asRange = kRangeTooLarge;
  } else if (signedRange < kMaxRange) {
    // We check that after the extension by reservePct the range of max - min
    // will still be in int64_t bounds.
    VELOX_CHECK_GE(100, reservePct);
    static_assert(kMaxRange < std::numeric_limits<uint64_t>::max() / 4);
    // We pad the range by 'reservePct'%, half below and half above,
    // while staying within bounds of the type. We do not pad the
    // limits yet, this is done only when enabling range mode.
    int64_t min = min_;
    int64_t max = max_;
    extendRange(type_->kind(), reservePct, min, max);
    asRange = (max - min) + 2;
  } else {
    setRangeOverflow();
    asRange = kRangeTooLarge;
  }
  if (distinctOverflow_) {
    asDistincts = kRangeTooLarge;
    return;
  }
  // Padded count of values + 1 for null.
  asDistincts =
      addIdReserve(numDistinct(), reservePct, distinctValueLimit()) + 1;
}

uint64_t VectorHasher::enableValueIds(uint64_t multiplier, int32_t reservePct) {
  VELOX_CHECK_NE(
      typeKind_,
      TypeKind::BOOLEAN,
      "A boolean VectorHasher should  always be by range");
  checkTypeSupportsValueIds();

  multiplier_ = multiplier;
  rangeSize_ =
      addIdReserve(numDistinct(), reservePct, distinctValueLimit()) + 1;
  isRange_ = false;
  uint64_t result;
  if (__builtin_mul_overflow(multiplier_, rangeSize_, &result)) {
    return kRangeTooLarge;
  }
  return result;
}

uint64_t VectorHasher::enableValueRange(
    uint64_t multiplier,
    int32_t reservePct) {
  checkTypeSupportsValueIds();

  multiplier_ = multiplier;
  VELOX_CHECK_LE(0, reservePct);
  VELOX_CHECK(hasRange_);
  extendRange(type_->kind(), reservePct, min_, max_);
  isRange_ = true;
  // No overflow because max range is under 63 bits.
  if (typeKind_ == TypeKind::BOOLEAN) {
    rangeSize_ = 3;
  } else {
    rangeSize_ = (max_ - min_) + 2;
  }
  uint64_t result;
  if (__builtin_mul_overflow(multiplier_, rangeSize_, &result)) {
    return kRangeTooLarge;
  }
  return result;
}

void VectorHasher::copyStatsFrom(const VectorHasher& other) {
  hasRange_ = other.hasRange_;
  hasHugeintValue_ = other.hasHugeintValue_;
  rangeOverflow_ = other.rangeOverflow_;
  distinctOverflow_ = other.distinctOverflow_;

  min_ = other.min_;
  max_ = other.max_;
  uniqueValues_ = other.uniqueValues_;
  uniqueHugeintValues_ = other.uniqueHugeintValues_;
}

void VectorHasher::merge(const VectorHasher& other, size_t maxNumDistinct) {
  if (typeKind_ == TypeKind::BOOLEAN) {
    return;
  }
  if (other.empty()) {
    return;
  }
  if (empty()) {
    copyStatsFrom(other);
    return;
  }
  if (hasRange_ && other.hasRange_ && !rangeOverflow_ &&
      !other.rangeOverflow_) {
    min_ = std::min(min_, other.min_);
    max_ = std::max(max_, other.max_);
  } else {
    setRangeOverflow();
  }
  if (distinctOverflow_) {
    return;
  }
  if (other.distinctOverflow_) {
    setDistinctOverflow();
    return;
  }
  if (typeKind_ == TypeKind::HUGEINT) {
    hasHugeintValue_ |= other.hasHugeintValue_;

    for (const auto& entry : other.uniqueHugeintValues_) {
      auto nextId = uniqueHugeintValues_.size() + 1;
      if (uniqueHugeintValues_.emplace(entry.first, nextId).second &&
          uniqueHugeintValues_.size() > maxNumDistinct) {
        setDistinctOverflow();
        break;
      }
    }
    return;
  }
  // Unique values can be merged without dispatch on type. All the
  // merged hashers must stay live for string type columns.
  for (UniqueValue value : other.uniqueValues_) {
    // Assign a new id at end of range for the case 'value' is not
    // in 'uniqueValues_'. We do not set overflow here because the
    // memory is already allocated and there is a known cap on size.
    value.setId(uniqueValues_.size() + 1);
    if (uniqueValues_.insert(value).second &&
        uniqueValues_.size() > maxNumDistinct) {
      setDistinctOverflow();
      break;
    }
  }
}

std::string VectorHasher::serializeState() const {
  const common::BigintValuesUsingBloomFilter* bloomFilter = nullptr;
  std::string bloomFilterData;
  if (bloomFilter_) {
    VELOX_CHECK(
        supportsBloomFilter(),
        "Unexpected bloom filter for VectorHasher type {}",
        type_->toString());
    bloomFilter = dynamic_cast<const common::BigintValuesUsingBloomFilter*>(
        bloomFilter_.get());
    VELOX_CHECK_NOT_NULL(
        bloomFilter, "Unexpected bloom filter implementation in VectorHasher");
    bloomFilterData.reserve(16 + bloomFilter->blocksByteSize());
    bloomFilter->serializeBinary(bloomFilterData);
  }

  std::vector<const UniqueValue*> valuesById;
  if (!distinctOverflow_) {
    valuesById.resize(uniqueValues_.size(), nullptr);
    for (const auto& value : uniqueValues_) {
      VELOX_CHECK_GE(value.id(), 1);
      VELOX_CHECK_LE(value.id(), valuesById.size());
      valuesById[value.id() - 1] = &value;
    }
  }

  std::string data;
  data.reserve(serializedStateSize());
  common::NativeStringWriter writer(data);

  writer.writeValue<uint32_t>(kVectorHasherStateMagic);
  writer.writeValue<uint32_t>(kVectorHasherStateVersion);
  writer.writeValue<uint32_t>(static_cast<uint32_t>(typeKind_));
  writer.writeValue<bool>(typeProvidesCustomComparison_);
  writer.writeValue<uint64_t>(rangeSize_);
  writer.writeValue<uint64_t>(multiplier_);
  writer.writeValue<bool>(isRange_);
  writer.writeValue<bool>(hasRange_);
  writer.writeValue<bool>(rangeOverflow_);
  writer.writeValue<bool>(distinctOverflow_);
  writer.writeValue<int64_t>(min_);
  writer.writeValue<int64_t>(max_);

  writer.writeValue<bool>(bloomFilter != nullptr);
  if (bloomFilter != nullptr) {
    writer.writeValue<uint32_t>(bloomFilterData.size());
    if (!bloomFilterData.empty()) {
      writer.write(bloomFilterData.data(), bloomFilterData.size());
    }
  }

  if (distinctOverflow_) {
    writer.writeValue<uint32_t>(0);
    return data;
  }

  writer.writeValue<uint32_t>(valuesById.size());
  for (const auto* value : valuesById) {
    VELOX_CHECK_NOT_NULL(value, "Missing unique value for serialized id range");
    const auto bytes = value->bytesView();
    writer.writeValue<uint32_t>(bytes.size());
    if (!bytes.empty()) {
      writer.write(bytes.data(), bytes.size());
    }
  }

  return data;
}

uint32_t VectorHasher::serializedStateSize() const {
  uint32_t size = sizeof(kVectorHasherStateMagic) +
      sizeof(kVectorHasherStateVersion) + sizeof(uint32_t) + sizeof(bool) +
      sizeof(rangeSize_) + sizeof(multiplier_) + sizeof(isRange_) +
      sizeof(hasRange_) + sizeof(rangeOverflow_) + sizeof(distinctOverflow_) +
      sizeof(min_) + sizeof(max_) + sizeof(bool) + sizeof(uint32_t);

  if (bloomFilter_) {
    VELOX_CHECK(
        supportsBloomFilter(),
        "Unexpected bloom filter for VectorHasher type {}",
        type_->toString());
    auto* bloomFilter =
        dynamic_cast<const common::BigintValuesUsingBloomFilter*>(
            bloomFilter_.get());
    VELOX_CHECK_NOT_NULL(
        bloomFilter, "Unexpected bloom filter implementation in VectorHasher");
    size += sizeof(bool) + 3 * sizeof(uint32_t) + bloomFilter->blocksByteSize();
  }

  if (!distinctOverflow_) {
    for (const auto& value : uniqueValues_) {
      size += sizeof(uint32_t) + value.size();
    }
  }

  return size;
}

void VectorHasher::serializeState(common::NativeBufferedWriter& writer) const {
  const common::BigintValuesUsingBloomFilter* bloomFilter = nullptr;
  std::string bloomFilterData;
  if (bloomFilter_) {
    VELOX_CHECK(
        supportsBloomFilter(),
        "Unexpected bloom filter for VectorHasher type {}",
        type_->toString());
    bloomFilter = dynamic_cast<const common::BigintValuesUsingBloomFilter*>(
        bloomFilter_.get());
    VELOX_CHECK_NOT_NULL(
        bloomFilter, "Unexpected bloom filter implementation in VectorHasher");
    bloomFilterData.reserve(16 + bloomFilter->blocksByteSize());
    bloomFilter->serializeBinary(bloomFilterData);
  }

  std::vector<const UniqueValue*> valuesById;
  if (!distinctOverflow_) {
    valuesById.resize(uniqueValues_.size(), nullptr);
    for (const auto& value : uniqueValues_) {
      VELOX_CHECK_GE(value.id(), 1);
      VELOX_CHECK_LE(value.id(), valuesById.size());
      valuesById[value.id() - 1] = &value;
    }
  }

  writer.writeValue<uint32_t>(kVectorHasherStateMagic);
  writer.writeValue<uint32_t>(kVectorHasherStateVersion);
  writer.writeValue<uint32_t>(static_cast<uint32_t>(typeKind_));
  writer.writeValue<bool>(typeProvidesCustomComparison_);
  writer.writeValue<uint64_t>(rangeSize_);
  writer.writeValue<uint64_t>(multiplier_);
  writer.writeValue<bool>(isRange_);
  writer.writeValue<bool>(hasRange_);
  writer.writeValue<bool>(rangeOverflow_);
  writer.writeValue<bool>(distinctOverflow_);
  writer.writeValue<int64_t>(min_);
  writer.writeValue<int64_t>(max_);

  writer.writeValue<bool>(bloomFilter != nullptr);
  if (bloomFilter != nullptr) {
    writer.writeValue<uint32_t>(bloomFilterData.size());
    if (!bloomFilterData.empty()) {
      writer.write(bloomFilterData.data(), bloomFilterData.size());
    }
  }

  if (distinctOverflow_) {
    writer.writeValue<uint32_t>(0);
    return;
  }

  writer.writeValue<uint32_t>(valuesById.size());
  for (const auto* value : valuesById) {
    VELOX_CHECK_NOT_NULL(value, "Missing unique value for serialized id range");
    const auto bytes = value->bytesView();
    writer.writeValue<uint32_t>(bytes.size());
    if (!bytes.empty()) {
      writer.write(bytes.data(), bytes.size());
    }
  }
}

void VectorHasher::deserializeState(std::string_view data) {
  common::NativeStringReader reader(data);
  const auto magic = reader.readValue<uint32_t>();
  const auto version = reader.readValue<uint32_t>();
  VELOX_CHECK_EQ(magic, kVectorHasherStateMagic, "Invalid VectorHasher state");
  VELOX_CHECK_EQ(
      version,
      kVectorHasherStateVersion,
      "Unsupported VectorHasher state version: {}",
      version);

  const auto typeKind = static_cast<TypeKind>(reader.readValue<uint32_t>());
  const auto hasCustomComparison = reader.readValue<bool>();
  VELOX_CHECK_EQ(typeKind, typeKind_, "VectorHasher type kind mismatch");
  VELOX_CHECK_EQ(
      hasCustomComparison,
      typeProvidesCustomComparison_,
      "VectorHasher comparison mode mismatch");

  rangeSize_ = reader.readValue<uint64_t>();
  multiplier_ = reader.readValue<uint64_t>();
  isRange_ = reader.readValue<bool>();
  hasRange_ = reader.readValue<bool>();
  rangeOverflow_ = reader.readValue<bool>();
  distinctOverflow_ = reader.readValue<bool>();
  min_ = reader.readValue<int64_t>();
  max_ = reader.readValue<int64_t>();

  bloomFilter_.reset();
  const bool hasBloomFilter = reader.readValue<bool>();
  if (hasBloomFilter) {
    VELOX_CHECK(
        supportsBloomFilter(),
        "Unexpected bloom filter for VectorHasher type {}",
        type_->toString());
    const auto bloomFilterSize = reader.readValue<uint32_t>();
    std::string bloomFilterData(bloomFilterSize, '\0');
    if (bloomFilterSize > 0) {
      reader.read(bloomFilterData.data(), bloomFilterSize);
    }
    bloomFilter_ = common::BigintValuesUsingBloomFilter::deserializeBinary(
        bloomFilterData);
  }

  ++dictionaryGeneration_;
  finishGroupingSetHashInput();
  uniqueValues_.clear();
  uniqueValuesStorage_.clear();
  distinctStringsBytes_ = 0;

  const auto numUniqueValues = reader.readValue<uint32_t>();
  if (!distinctOverflow_) {
    for (uint32_t id = 1; id <= numUniqueValues; ++id) {
      const auto size = reader.readValue<uint32_t>();
      std::string value(size, '\0');
      if (size > 0) {
        reader.read(value.data(), size);
      }

      UniqueValue unique(value.data(), size);
      unique.setId(id);
      auto [iter, inserted] = uniqueValues_.insert(unique);
      VELOX_CHECK(inserted, "Duplicate value in VectorHasher state");
      // Growth limits are query tuning, not serialized state. Restore all
      // existing IDs exactly; apply the configured limits to future growth.
      copyStringToLocalImpl(&*iter, false);
    }
  } else {
    for (uint32_t i = 0; i < numUniqueValues; ++i) {
      const auto size = reader.readValue<uint32_t>();
      if (size > 0) {
        std::string skip(size, '\0');
        reader.read(skip.data(), size);
      }
    }
  }

  VELOX_CHECK(reader.atEnd(), "Trailing bytes in VectorHasher state");
}

std::string VectorHasher::toString() const {
  std::stringstream out;
  out << "VectorHasher channel: " << channel_ << " " << type_->toString()
      << " multiplier: " << multiplier_;
  if (isRange_) {
    out << " range size " << rangeSize_ << ": [" << min_ << ", " << max_ << "]";
  }
  if (!distinctOverflow_) {
    out << " numDistinct: " << numDistinct();
  }
  return out.str();
}

std::vector<std::unique_ptr<VectorHasher>> createVectorHashers(
    const RowTypePtr& rowType,
    const std::vector<core::FieldAccessTypedExprPtr>& keys) {
  const auto numKeys = keys.size();
  std::vector<column_index_t> keyChannels;
  keyChannels.reserve(numKeys);
  for (const auto& key : keys) {
    const auto channel = exprToChannel(key.get(), rowType);
    keyChannels.push_back(channel);
  }
  return createVectorHashers(rowType, keyChannels);
}

std::vector<std::unique_ptr<VectorHasher>> createVectorHashers(
    const RowTypePtr& rowType,
    const std::vector<column_index_t>& keyChannels) {
  const auto numKeys = keyChannels.size();
  std::vector<std::unique_ptr<VectorHasher>> hashers;
  hashers.reserve(numKeys);
  for (const auto& keyChannel : keyChannels) {
    hashers.push_back(
        VectorHasher::create(rowType->childAt(keyChannel), keyChannel));
  }
  return hashers;
}

} // namespace facebook::velox::exec
