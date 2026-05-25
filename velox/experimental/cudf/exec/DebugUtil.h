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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Exceptions.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <string_view>

namespace facebook::velox::cudf_velox {

inline bool isTruthyCudfTraceEnv(const char* value) {
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string_view flag{value};
  return flag != "0" && flag != "false" && flag != "FALSE" &&
      flag != "off" && flag != "OFF";
}

inline bool perfTraceEnabled() {
  static const bool enabled =
      isTruthyCudfTraceEnv(std::getenv("VELOX_CUDF_PERF_TRACE"));
  return enabled;
}

inline bool perfTraceSyncEnabled() {
  static const bool enabled =
      isTruthyCudfTraceEnv(std::getenv("VELOX_CUDF_PERF_TRACE_SYNC"));
  return enabled;
}

/// Helper function to check for CUDA errors in debug mode.
inline void checkCudaErrorInDebug() {
  if (CudfConfig::getInstance().debugEnabled) {
    cudaError_t err = cudaGetLastError();
    VELOX_CHECK(
        err == cudaSuccess, "CUDA error detected: {}", cudaGetErrorString(err));
  }
}

class DebugUtil {
 public:
  std::string toString(
      const cudf::table_view& table,
      rmm::cuda_stream_view stream,
      vector_size_t from,
      vector_size_t to);

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      rootPool_->addLeafChild("debug_util")};
};

} // namespace facebook::velox::cudf_velox
