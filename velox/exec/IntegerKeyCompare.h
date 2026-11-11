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
#include <cstring>
#include <vector>

namespace facebook::velox::exec::detail::integer_key_compare {

// Keep decoded vectors and row-column metadata local to one probe batch.
// The caller admits only INTEGER and BIGINT without custom comparison.
// Templates keep the equality kernel independent of decoder and row storage.
template <typename Decoded, typename Column>
struct IntegerKeyColumn {
  // References the decoded input for this batch.
  const Decoded* decoded;
  // Describes the stored key offset and null bit.
  Column column;
};

// Compare one integer width, avoiding reads of NULL payloads.
template <typename T, bool mayHaveNulls, typename Decoded, typename Column>
inline bool equalIntegerColumns(
    const char* group,
    int32_t row,
    const std::vector<IntegerKeyColumn<Decoded, Column>>& columns) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8);
  for (const auto& key : columns) {
    const auto& decoded = *key.decoded;
    if constexpr (mayHaveNulls) {
      const bool groupNull =
          (static_cast<uint8_t>(group[key.column.nullByte()]) &
           key.column.nullMask()) != 0;
      const bool inputNull = decoded.isNullAt(row);
      if (groupNull != inputNull) {
        return false;
      }
      if (groupNull) {
        continue;
      }
    }
    T stored;
    // RowContainer packs key fields without per-field alignment padding.
    std::memcpy(&stored, group + key.column.offset(), sizeof(T));
    if (stored != decoded.template valueAt<T>(row)) {
      return false;
    }
  }
  return true;
}

// Compare both widths with grouping equality semantics for NULL values.
template <bool mayHaveNulls, typename Decoded, typename Column>
inline bool equalFixedIntegerKeys(
    const char* group,
    int32_t row,
    const std::vector<IntegerKeyColumn<Decoded, Column>>& int32Keys,
    const std::vector<IntegerKeyColumn<Decoded, Column>>& int64Keys) {
  return equalIntegerColumns<int32_t, mayHaveNulls>(group, row, int32Keys) &&
      equalIntegerColumns<int64_t, mayHaveNulls>(group, row, int64Keys);
}

} // namespace facebook::velox::exec::detail::integer_key_compare
