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

#include <chrono>
#include <cstdint>
#include <utility>

#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

inline constexpr const char* kGpuLockWaitNanos = "gpuLockWaitNanos";
inline constexpr const char* kGpuSyncWaitNanos = "gpuSyncWaitNanos";

namespace detail {
inline thread_local exec::Operator* currentGpuOperator{nullptr};
} // namespace detail

class CurrentGpuOperatorScope {
 public:
  explicit CurrentGpuOperatorScope(exec::Operator* op)
      : previous_(detail::currentGpuOperator) {
    detail::currentGpuOperator = op;
  }

  ~CurrentGpuOperatorScope() {
    detail::currentGpuOperator = previous_;
  }

 private:
  exec::Operator* previous_;
};

inline void addOperatorRuntimeStat(const char* name, uint64_t nanos) {
  if (nanos == 0) {
    return;
  }
  auto* op = detail::currentGpuOperator;
  if (op == nullptr) {
    return;
  }
  op->addRuntimeStat(
      name, RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
}

inline void addGpuLockWaitNanos(uint64_t nanos) {
  addOperatorRuntimeStat(kGpuLockWaitNanos, nanos);
}

inline void addGpuSyncWaitNanos(uint64_t nanos) {
  addOperatorRuntimeStat(kGpuSyncWaitNanos, nanos);
}

template <typename Func>
inline void measureGpuSyncWait(Func&& func) {
  auto start = std::chrono::steady_clock::now();
  try {
    std::forward<Func>(func)();
  } catch (...) {
    addGpuSyncWaitNanos(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    throw;
  }
  addGpuSyncWaitNanos(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

template <typename Stream>
inline void synchronizeStreamAndRecord(Stream stream) {
  measureGpuSyncWait([&]() { stream.synchronize(); });
}

} // namespace facebook::velox::cudf_velox
