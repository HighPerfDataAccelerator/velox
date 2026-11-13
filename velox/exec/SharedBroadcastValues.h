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

#include "velox/vector/BaseVector.h"

namespace facebook::velox::exec::detail {
/// Optionally shares immutable integer values through a private, caller-owned
/// directory selected by VELOX_SHARED_BROADCAST_CACHE_DIR on Linux. Cache files
/// are partitioned by NUMA node and content. The caller owns directory cleanup.
/// Returns the original vector if sharing is disabled or unavailable.
VectorPtr shareBroadcastValues(const VectorPtr& input);
} // namespace facebook::velox::exec::detail
