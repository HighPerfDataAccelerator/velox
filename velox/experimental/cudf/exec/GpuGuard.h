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

namespace gluten {
void lockGpu();
void unlockGpu();
bool tryLockGpu();
} // namespace gluten

namespace facebook::velox::cudf_velox {

/// RAII guard that limits concurrent GPU usage across Velox pipeline tasks.
/// Forwards to gluten::lockGpu / unlockGpu which are ref-counted per thread.
struct GpuGuard {
  GpuGuard() {
    gluten::lockGpu();
  }
  ~GpuGuard() {
    gluten::unlockGpu();
  }
  GpuGuard(const GpuGuard&) = delete;
  GpuGuard& operator=(const GpuGuard&) = delete;
};

/// Non-blocking RAII guard. Attempts to acquire GPU access without waiting.
/// Check `acquired` (or use `operator bool()`) to see if the lock was obtained.
/// If acquired, automatically releases on destruction.
struct TryGpuGuard {
  bool acquired;
  TryGpuGuard() : acquired(gluten::tryLockGpu()) {}
  ~TryGpuGuard() {
    if (acquired) {
      gluten::unlockGpu();
    }
  }
  explicit operator bool() const {
    return acquired;
  }
  TryGpuGuard(const TryGpuGuard&) = delete;
  TryGpuGuard& operator=(const TryGpuGuard&) = delete;
};

} // namespace facebook::velox::cudf_velox
