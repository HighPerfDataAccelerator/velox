/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace facebook::velox::cudf_velox::detail {

// Bound a resident restore wave by both chunk count and the existing pinned
// slot's pageable-byte capacity. An individually oversized chunk is returned
// alone so the established fallback can make progress; it must not pull any
// additional chunks into an oversized wave. Pinned chunks may contribute zero.
template <typename PageableBytesAt>
size_t graceRestoreWaveEnd(
    size_t begin,
    size_t end,
    size_t depth,
    uint64_t capacity,
    PageableBytesAt pageableBytesAt) {
  if (begin >= end) {
    return end;
  }
  const auto limit = begin + std::min(end - begin, std::max(size_t{1}, depth));
  uint64_t bytes = 0;
  auto next = begin;
  for (; next < limit; ++next) {
    const uint64_t chunkBytes = pageableBytesAt(next);
    if (chunkBytes > capacity - bytes) {
      return next == begin ? next + 1 : next;
    }
    bytes += chunkBytes;
  }
  return next;
}

// libcudf contiguous_split aligns each packed buffer to 64 bytes. Reserve
// one alignment unit per buffer/partition before estimating slice row counts.
// Actual packed size must still be checked: variable-width rows can be skewed.
inline uint64_t gracePartitionDataBudget(
    uint64_t capacity,
    uint64_t buffers,
    uint64_t partitions) {
  constexpr uint64_t kPackedAlignment = 64;
  if (capacity == 0) {
    return 0;
  }
  if (buffers == 0 || partitions == 0) {
    return capacity;
  }
  if (buffers > (capacity - 1) / kPackedAlignment / partitions) {
    return 1;
  }
  return capacity - buffers * partitions * kPackedAlignment;
}

} // namespace facebook::velox::cudf_velox::detail
