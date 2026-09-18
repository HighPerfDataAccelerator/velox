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

#include <nvtx3/nvtx3.hpp>

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>

namespace facebook::velox::cudf_velox::host_staging {

// Synchronous copy with CPU-only workers shared by Grace and UCX staging.
// Callers retain both buffers (and their existing admission/slot leases) until
// completion. No additional data storage or pinned/device budget is allocated.
// Never call this from one of this executor's workers: they only run memcpy.
inline bool copyPageableHost(
    void* destination,
    const void* source,
    uint64_t bytes,
    bool enabled,
    const char* chunkLabel) {
  constexpr uint64_t kMinParallelBytes = 8ULL << 20;
  constexpr size_t kWorkers = 4;
  if (!enabled || bytes < kMinParallelBytes) {
    if (bytes != 0) {
      std::memcpy(destination, source, bytes);
    }
    return false;
  }
  static folly::CPUThreadPoolExecutor executor(kWorkers);
  std::array<std::future<void>, kWorkers> completions;
  // Join already-submitted copies even on submission/allocation/get failure.
  SCOPE_EXIT {
    for (auto& completion : completions) {
      if (completion.valid()) {
        completion.wait();
      }
    }
  };
  const auto stride = ((bytes + kWorkers - 1) / kWorkers + 63) / 64 * 64;
  for (size_t i = 0; i < kWorkers; ++i) {
    const auto offset = i * stride;
    if (offset >= bytes) {
      break;
    }
    const auto size = std::min<uint64_t>(stride, bytes - offset);
    auto task = std::make_shared<std::packaged_task<void()>>(
        [destination, source, offset, size, chunkLabel] {
          nvtx3::scoped_range range(chunkLabel);
          std::memcpy(
              static_cast<uint8_t*>(destination) + offset,
              static_cast<const uint8_t*>(source) + offset,
              size);
        });
    completions[i] = task->get_future();
    try {
      executor.add([task] { (*task)(); });
    } catch (...) {
      // Rejection must not drop the copy or release its source prematurely.
      (*task)();
    }
  }
  for (auto& completion : completions) {
    if (completion.valid()) {
      completion.get();
    }
  }
  return true;
}

} // namespace facebook::velox::cudf_velox::host_staging
