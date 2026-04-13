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
} // namespace gluten

namespace facebook::velox::cudf_velox {

/// Pipeline-level GPU region — thread-local boolean.
///
/// beginGpuRegion() is idempotent: first call acquires the semaphore,
/// subsequent calls are no-ops. endGpuRegion() releases once.
/// This lets the semaphore span an entire pipeline iteration [H2D, D2H]
/// even when accumulating operators cause the source to be called N times
/// before the sink produces output.
void beginGpuRegion();
void endGpuRegion();
bool isInGpuRegion();

/// RAII guard for GPU work. Automatically starts a GPU region on
/// construction (idempotent — only the first GpuGuard in a pipeline
/// iteration actually acquires the semaphore). The nested lockGpu/unlockGpu
/// keeps refcount > 0 so the semaphore is never released between operators.
///
/// The region is ended explicitly by calling endGpuRegion() at D2H points
/// (CudfToVelox, HashJoinBuild::addInput, etc.), NOT by ~GpuGuard.
struct GpuGuard {
  GpuGuard() {
    beginGpuRegion();
    gluten::lockGpu();
  }
  ~GpuGuard() {
    gluten::unlockGpu();
  }
  GpuGuard(const GpuGuard&) = delete;
  GpuGuard& operator=(const GpuGuard&) = delete;
};

} // namespace facebook::velox::cudf_velox
