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

#include "velox/experimental/cudf/CudfDefaultStreamOverload.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <common/base/Exceptions.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>

namespace facebook::velox::cudf_velox {

cuda::mr::any_resource<cuda::mr::device_accessible> createMemoryResource(
    std::string_view mode,
    int percent) {
  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    return rmm::mr::cuda_async_memory_resource{};
  } else if (mode == "arena") {
    return rmm::mr::arena_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed") {
    return rmm::mr::managed_memory_resource{};
  } else if (mode == "managed_pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::managed_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "managed_async") {
    return rmm::mr::cuda_async_managed_memory_resource{};
  } else if (mode == "prefetch_managed") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::managed_memory_resource{});
  } else if (mode == "prefetch_managed_pool") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::pool_memory_resource(
            rmm::mr::managed_memory_resource{},
            rmm::percent_of_free_device_memory(percent)));
  } else if (mode == "prefetch_managed_async") {
    cudf::prefetch::enable();
    return rmm::mr::prefetch_resource_adaptor(
        rmm::mr::cuda_async_managed_memory_resource{});
  }
  VELOX_FAIL(
      "Unknown memory resource mode: " + std::string(mode) +
      "\nExpecting: cuda, pool, async, arena, managed, prefetch_managed, " +
      "managed_pool, prefetch_managed_pool, managed_async, prefetch_managed_async");
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> output_mr_;
std::optional<rmm::mr::statistics_resource_adaptor> statistics_mr_;
std::optional<rmm::mr::statistics_resource_adaptor> output_statistics_mr_;

rmm::device_async_resource_ref get_output_mr() {
  return output_mr_.value();
}

bool deviceMemoryDiagnosticsEnabled() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_CUDF_DEVICE_MEMORY_DIAGNOSTICS");
    if (value == nullptr) {
      return false;
    }
    std::string normalized{value};
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !normalized.empty() && normalized != "0" && normalized != "false" &&
        normalized != "off" && normalized != "no";
  }();
  return enabled;
}

DeviceMemorySnapshot captureDeviceMemorySnapshot() {
  DeviceMemorySnapshot snapshot;
  snapshot.enabled = deviceMemoryDiagnosticsEnabled();
  if (!snapshot.enabled) {
    return snapshot;
  }

  if (cudaGetDevice(&snapshot.device) == cudaSuccess) {
    snapshot.cudaValid =
        cudaMemGetInfo(&snapshot.freeBytes, &snapshot.totalBytes) ==
        cudaSuccess;
    if (snapshot.cudaValid) {
      snapshot.usedBytes = snapshot.totalBytes - snapshot.freeBytes;
    }
  }

  const auto addStatistics = [&](const auto& statistics) {
    if (!statistics.has_value()) {
      return;
    }
    const auto bytes = statistics->get_bytes_counter();
    const auto allocations = statistics->get_allocations_counter();
    snapshot.rmmCurrentBytes += bytes.value;
    snapshot.rmmPeakBytes += bytes.peak;
    snapshot.rmmTotalBytes += bytes.total;
    snapshot.rmmCurrentAllocations += allocations.value;
    snapshot.rmmPeakAllocations += allocations.peak;
    snapshot.rmmTotalAllocations += allocations.total;
  };
  addStatistics(statistics_mr_);
  addStatistics(output_statistics_mr_);
  return snapshot;
}

void logDeviceMemorySnapshot(
    const std::string& label,
    const DeviceMemorySnapshot& snapshot) {
  if (!snapshot.enabled) {
    return;
  }
  const auto* visibleDevices = std::getenv("CUDA_VISIBLE_DEVICES");
  const auto* executorId = std::getenv("SPARK_EXECUTOR_ID");
  LOG(WARNING) << "CUDF_DEVICE_MEMORY"
               << " label={" << label << "}"
               << " pid=" << static_cast<int64_t>(::getpid())
               << " thread=" << std::this_thread::get_id()
               << " executorId=" << (executorId == nullptr ? "" : executorId)
               << " cudaVisibleDevices="
               << (visibleDevices == nullptr ? "" : visibleDevices)
               << " device=" << snapshot.device
               << " cudaValid=" << snapshot.cudaValid
               << " freeBytes=" << snapshot.freeBytes
               << " totalBytes=" << snapshot.totalBytes
               << " usedBytes=" << snapshot.usedBytes
               << " rmmCurrentBytes=" << snapshot.rmmCurrentBytes
               << " rmmPeakBytes=" << snapshot.rmmPeakBytes
               << " rmmTotalBytes=" << snapshot.rmmTotalBytes
               << " rmmCurrentAllocations=" << snapshot.rmmCurrentAllocations
               << " rmmPeakAllocations=" << snapshot.rmmPeakAllocations
               << " rmmTotalAllocations=" << snapshot.rmmTotalAllocations;
}

CudaAllocationTraceScope::CudaAllocationTraceScope(const std::string& label) {
  using Push = void (*)(const char*);
  static auto push = reinterpret_cast<Push>(
      dlsym(RTLD_DEFAULT, "cuda_alloc_trace_push_context"));
  if (push != nullptr) {
    push(label.c_str());
    active_ = true;
  }
}

CudaAllocationTraceScope::~CudaAllocationTraceScope() {
  if (!active_) {
    return;
  }
  using Pop = void (*)();
  static auto pop = reinterpret_cast<Pop>(
      dlsym(RTLD_DEFAULT, "cuda_alloc_trace_pop_context"));
  if (pop != nullptr) {
    pop();
  }
}

} // namespace facebook::velox::cudf_velox

// This must NOT be in a file that includes CudfNoDefaults.h, because
// CudfNoDefaults.h redeclares cudf::get_default_stream() with
// __attribute__((error)). The overload below calls the real function.
namespace cudf {

rmm::cuda_stream_view const get_default_stream(allow_default_stream_t) {
  return cudf::get_default_stream();
}

} // namespace cudf
