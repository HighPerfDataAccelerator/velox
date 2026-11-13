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

#include "velox/exec/ColumnarBroadcastOutput.h"
#include "velox/exec/HashTable.h"
#include "velox/exec/Operator.h"
namespace facebook::velox::exec::detail {
namespace {
// The builder checked the whole slot range against the logical key type.
template <typename T>
inline void materializeRangeKey(
    int64_t bias,
    const vector_size_t* slots,
    vector_size_t count,
    const BufferPtr& nulls,
    const VectorPtr& output) {
  auto* flat = output->as<FlatVector<T>>();
  auto* values = flat->mutableRawValues();
  for (vector_size_t i = 0; i < count; ++i) {
    values[i] = static_cast<T>(bias + int64_t(slots[i]));
  }
  flat->setNulls(nulls);
}
// Cached flat arrays are immutable. NULL slots have initialized zero values.
template <typename S, typename T>
FOLLY_NOINLINE void materializeFlatColumn(
    const VectorPtr& cached,
    const vector_size_t* slots,
    const BufferPtr& rowNulls,
    vector_size_t count,
    const VectorPtr& result,
    vector_size_t stride = 1,
    vector_size_t offset = 0) {
  const auto* __restrict values = cached->as<FlatVector<S>>()->rawValues();
  const auto* sourceNulls = cached->rawNulls();
  auto* output = result->as<FlatVector<T>>();
  auto* __restrict destination = output->mutableRawValues();
  const auto* outer = rowNulls ? rowNulls->as<uint64_t>() : nullptr;
  if (!sourceNulls && !outer) {
    for (vector_size_t i = 0; i < count; ++i) {
      destination[i] = values[slots[i] * stride + offset];
    }
    output->setNulls(nullptr);
    return;
  }
  auto* outputNulls = output->mutableRawNulls();
  bool anyNull = false;
  for (vector_size_t begin = 0; begin < count;) {
    const auto size = std::min<vector_size_t>(64, count - begin);
    for (int i = 0; i < size; ++i) {
      destination[begin + i] = values[slots[begin + i] * stride + offset];
    }
    uint64_t nonNull = ~0ULL;
    if (sourceNulls) {
      // Validated array slots and layout extents are nonnegative and fit int32.
      const auto validBit = [&](int i) -> uint64_t {
        const uint32_t slot =
            uint32_t(slots[begin + i]) * uint32_t(stride) + uint32_t(offset);
        return (sourceNulls[slot >> 6] >> (slot & 63)) & 1;
      };
      nonNull = 0;
      for (int i = 0; i < size; ++i) {
        nonNull |= validBit(i) << i;
      }
    }
    if (outer) {
      nonNull &= outer[begin / 64];
    }
    const uint64_t validMask = size == 64 ? ~0ULL : (1ULL << size) - 1;
    outputNulls[begin / 64] = nonNull | ~validMask;
    anyNull |= (nonNull & validMask) != validMask;
    begin += size;
  }
  if (!anyNull) {
    output->setNulls(nullptr);
  }
}
template <typename T>
void materializeIntegerColumn(
    const VectorPtr& cached,
    const vector_size_t* slots,
    const BufferPtr& rowNulls,
    vector_size_t count,
    const VectorPtr& result,
    vector_size_t stride,
    vector_size_t offset) {
  if (cached->typeKind() == TypeKind::INTEGER) {
    materializeFlatColumn<int32_t, T>(
        cached, slots, rowNulls, count, result, stride, offset);
  } else {
    VELOX_CHECK_EQ(cached->typeKind(), TypeKind::BIGINT);
    materializeFlatColumn<int64_t, T>(
        cached, slots, rowNulls, count, result, stride, offset);
  }
}
} // namespace

// Does not mutate probe row mappings: probe-side output dictionaries may share
// them.
bool tryColumnarBroadcastOutput(
    BaseHashTable* table,
    const HashLookup& lookup,
    const vector_size_t* probeRows,
    folly::Range<char* const*> matches,
    folly::Range<const IdentityProjection*> projections,
    memory::MemoryPool* pool,
    const std::vector<TypePtr>& resultTypes,
    std::vector<VectorPtr>& outputs) {
  const auto* cache = table->columnarBroadcastCache();
  if (!cache || projections.empty() || matches.empty()) {
    return false;
  }
  const auto capacity = table->capacity();
  for (const auto& projection : projections) {
    VELOX_CHECK_LT(projection.inputChannel, cache->columns.size());
    VELOX_CHECK_LT(projection.outputChannel, outputs.size());
    if (!resultTypes[projection.outputChannel]->equivalent(
            *table->rows()->columnTypes()[projection.inputChannel])) {
      return false;
    }
    const auto kind = resultTypes[projection.outputChannel]->kind();
    if (kind != TypeKind::INTEGER && kind != TypeKind::BIGINT) {
      return false;
    }
  }

  const auto count = static_cast<vector_size_t>(matches.size());
  auto indices = AlignedBuffer::allocate<vector_size_t>(count, pool);
  auto* raw = indices->asMutable<vector_size_t>();
  BufferPtr nulls;
  for (vector_size_t i = 0; i < count; ++i) {
    if (matches[i] == nullptr) {
      if (!nulls) {
        nulls = allocateNulls(count, pool);
      }
      bits::setNull(nulls->asMutable<uint64_t>(), i, true);
      raw[i] = 0;
    } else {
      const auto probe = probeRows ? probeRows[i] : i;
      VELOX_CHECK_GE(probe, 0);
      VELOX_CHECK_LT(probe, lookup.hashes.size());
      const auto index = lookup.hashes[probe];
      VELOX_CHECK_LT(index, capacity);
      raw[i] = static_cast<vector_size_t>(index);
    }
  }
  for (const auto& projection : projections) {
    const auto channel = projection.inputChannel;
    auto& output = outputs[projection.outputChannel];
    const auto& type = resultTypes[projection.outputChannel];
    if (!output || !BaseVector::isVectorWritable(output) ||
        !output->isFlatEncoding()) {
      output = BaseVector::create(type, count, pool);
    }
    output->resize(count);
    if (channel == 0 && cache->keyBias.has_value()) {
      if (type->kind() == TypeKind::INTEGER) {
        materializeRangeKey<int32_t>(
            *cache->keyBias, raw, count, nulls, output);
      } else {
        materializeRangeKey<int64_t>(
            *cache->keyBias, raw, count, nulls, output);
      }
      continue;
    }
    const bool packed = channel > 0 && cache->packedPayload;
    const auto& cached =
        packed ? cache->packedPayload : cache->columns[channel];
    VELOX_CHECK_NOT_NULL(cached);
    const auto stride = packed ? cache->columns.size() - 1 : 1;
    const auto offset = packed ? channel - 1 : 0;
    if (type->kind() == TypeKind::INTEGER) {
      materializeIntegerColumn<int32_t>(
          cached, raw, nulls, count, output, stride, offset);
    } else {
      materializeIntegerColumn<int64_t>(
          cached, raw, nulls, count, output, stride, offset);
    }
  }
  return true;
}
} // namespace facebook::velox::exec::detail
