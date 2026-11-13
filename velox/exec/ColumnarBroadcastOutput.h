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

#include "velox/exec/HashTable.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec::detail {
/// Materializes build-side output from an explicitly initialized broadcast
/// cache. Returns false when normal row extraction is required. Probe mappings
/// are read-only because probe-side dictionary vectors may share them.
bool tryColumnarBroadcastOutput(
    BaseHashTable* table,
    const HashLookup& lookup,
    const vector_size_t* probeRows,
    folly::Range<char* const*> matches,
    folly::Range<const IdentityProjection*> projections,
    memory::MemoryPool* pool,
    const std::vector<TypePtr>& resultTypes,
    std::vector<VectorPtr>& outputs);
} // namespace facebook::velox::exec::detail
