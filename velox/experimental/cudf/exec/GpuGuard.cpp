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

#include "velox/experimental/cudf/exec/GpuGuard.h"
#include <iostream>
#include <thread>

// MPP livelock diagnosis: GpuGuard is bypassed entirely. beginGpuRegion /
// endGpuRegion are no-ops to prove the lock is a symptom, not the root cause
// of the MppNativeQueryExec hang on Q1. Symbols are kept exported so other
// TUs that reference them still link. Stderr markers are compiled out to keep
// executor logs quiet; flip VELOX_GPUGUARD_TRACE to 1 to re-enable.
#ifndef VELOX_GPUGUARD_TRACE
#define VELOX_GPUGUARD_TRACE 0
#endif

namespace facebook::velox::cudf_velox {

void beginGpuRegion() {
  // MPP livelock diagnosis: no-op.
#if VELOX_GPUGUARD_TRACE
  std::cerr << "GPU_REGION [begin-bypass] tid="
            << std::this_thread::get_id() << std::endl;
#endif
}

void endGpuRegion() {
  // MPP livelock diagnosis: no-op.
#if VELOX_GPUGUARD_TRACE
  std::cerr << "GPU_REGION [end-bypass] tid="
            << std::this_thread::get_id() << std::endl;
#endif
}

bool isInGpuRegion() {
  // Reflect the bypass: no region is ever active.
  return false;
}

} // namespace facebook::velox::cudf_velox
