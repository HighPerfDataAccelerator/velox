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

#include "velox/common/memory/MemoryPool.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/exec/Operator.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_async_managed_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/prefetch_resource_adaptor.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cuda_runtime_api.h>

#include <common/base/Exceptions.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox {

struct DeviceMemoryReclaimerState {
  DeviceMemoryReclaimerState(
      exec::Operator* owner,
      memory::MemoryPool* pool,
      int device)
      : owner{owner}, pool{pool}, device{device} {}

  exec::Operator* const owner;
  memory::MemoryPool* const pool;
  std::atomic<int> device;
  std::atomic<uint64_t> reportedReclaimableBytes{0};
  std::atomic<uint64_t> requestedBytes{0};
  std::atomic<uint64_t> requestEpoch{0};
  std::atomic<bool> reclaimInProgress{false};
};

enum class DeviceMemoryWorkspaceRequestStatus : uint8_t {
  kQueued,
  // The scheduler fulfilled this waiter's future, but has not reserved device
  // bytes. The owning Velox driver revalidates physical and virtual headroom
  // atomically when it retries admission. A short advisory lease prevents a
  // second immediate wake from the same headroom snapshot without allowing
  // an unscheduled owner to block the device indefinitely.
  kWakePending,
  kConsumed,
  kCanceled,
};

struct DeviceMemoryWorkspaceRequestState {
  DeviceMemoryWorkspaceRequestState(
      exec::Operator* _requestor,
      memory::MemoryPool* _pool,
      int _device,
      std::size_t _bytes,
      std::size_t _minHeadroomBytes,
      DeviceMemoryWorkspacePriority _priority,
      uint64_t _sequence,
      ContinuePromise _promise)
      : requestor{_requestor},
        pool{_pool},
        device{_device},
        bytes{_bytes},
        minHeadroomBytes{_minHeadroomBytes},
        priority{_priority},
        sequence{_sequence},
        enqueueTime{std::chrono::steady_clock::now()},
        promise{std::move(_promise)} {}

  exec::Operator* const requestor;
  memory::MemoryPool* const pool;
  const int device;
  const std::size_t bytes;
  const std::size_t minHeadroomBytes;
  const DeviceMemoryWorkspacePriority priority;
  const uint64_t sequence;
  const std::chrono::steady_clock::time_point enqueueTime;
  std::chrono::steady_clock::time_point wakeTime{};
  std::chrono::steady_clock::time_point retryNotBefore{};
  std::chrono::steady_clock::time_point nextOutputEscapeSync{};
  DeviceMemoryWorkspaceRequestStatus status{
      DeviceMemoryWorkspaceRequestStatus::kQueued};
  std::optional<ContinuePromise> promise;
};

namespace {
thread_local std::string currentAllocationContext{"unattributed"};
thread_local std::vector<std::string> allocationContextStack;

struct OperatorAttributionResourceImpl {
  struct Allocation {
    std::size_t bytes;
    std::string context;
    uint64_t sequence;
  };

  struct FreedAllocation {
    void* pointer;
    std::size_t bytes;
    std::string context;
    uint64_t allocationSequence;
    uint64_t freeSequence;
  };

  struct ContextStats {
    std::size_t currentBytes{0};
    std::size_t peakBytes{0};
    std::size_t currentAllocations{0};
  };

  explicit OperatorAttributionResourceImpl(
      rmm::device_async_resource_ref upstream)
      : upstream_(upstream) {}

  void* allocate(
      cuda::stream_ref stream,
      std::size_t bytes,
      std::size_t alignment) {
    void* pointer = nullptr;
    try {
      pointer = upstream_.allocate(stream, bytes, alignment);
    } catch (...) {
      dumpFailure(bytes);
      throw;
    }
    recordAllocation(pointer, bytes);
    return pointer;
  }

  void deallocate(
      cuda::stream_ref stream,
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    upstream_.deallocate(stream, pointer, bytes, alignment);
    recordDeallocation(pointer);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    void* pointer = nullptr;
    try {
      pointer = upstream_.allocate_sync(bytes, alignment);
    } catch (...) {
      dumpFailure(bytes);
      throw;
    }
    recordAllocation(pointer, bytes);
    return pointer;
  }

  void deallocate_sync(
      void* pointer,
      std::size_t bytes,
      std::size_t alignment) noexcept {
    upstream_.deallocate_sync(pointer, bytes, alignment);
    recordDeallocation(pointer);
  }

  bool operator==(
      const OperatorAttributionResourceImpl& other) const noexcept {
    return this == &other;
  }

  bool operator!=(
      const OperatorAttributionResourceImpl& other) const noexcept {
    return !(*this == other);
  }

  void dumpLive(const std::string& label, const char* resource) const {
    std::vector<std::pair<std::string, ContextStats>> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot.reserve(contexts_.size());
      for (const auto& entry : contexts_) {
        if (entry.second.currentBytes != 0) {
          snapshot.push_back(entry);
        }
      }
    }
    std::sort(
        snapshot.begin(),
        snapshot.end(),
        [](const auto& left, const auto& right) {
          return left.second.currentBytes > right.second.currentBytes;
        });
    const auto count = std::min<std::size_t>(snapshot.size(), 12);
    for (std::size_t index = 0; index < count; ++index) {
      LOG(WARNING)
          << "CUDF_DEVICE_MEMORY_OWNER label={" << label << "}"
          << " resource=" << resource << " rank=" << (index + 1)
          << " currentBytes=" << snapshot[index].second.currentBytes
          << " peakBytes=" << snapshot[index].second.peakBytes
          << " currentAllocations="
          << snapshot[index].second.currentAllocations
          << " context={" << snapshot[index].first << "}";
    }
  }

  void describePointer(
      const std::string& label,
      const char* resource,
      const void* pointer,
      std::size_t bytes) const {
    const auto requestedBegin = reinterpret_cast<uintptr_t>(pointer);
    const auto requestedEnd =
        requestedBegin + std::max<std::size_t>(bytes, 1);
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [allocationPointer, allocation] : allocations_) {
      const auto allocationBegin =
          reinterpret_cast<uintptr_t>(allocationPointer);
      const auto allocationEnd = allocationBegin + allocation.bytes;
      if (requestedBegin < allocationEnd && requestedEnd > allocationBegin) {
        LOG(ERROR)
            << "CUDF_DEVICE_MEMORY_POINTER label={" << label << "}"
            << " resource=" << resource << " state=live"
            << " requestPtr=" << pointer << " requestBytes=" << bytes
            << " allocationPtr=" << allocationPointer
            << " allocationBytes=" << allocation.bytes
            << " offset="
            << (requestedBegin >= allocationBegin
                    ? requestedBegin - allocationBegin
                    : 0)
            << " allocationSequence=" << allocation.sequence
            << " context={" << allocation.context << "}";
        return;
      }
    }
    for (auto it = recentlyFreed_.rbegin(); it != recentlyFreed_.rend(); ++it) {
      const auto allocationBegin = reinterpret_cast<uintptr_t>(it->pointer);
      const auto allocationEnd = allocationBegin + it->bytes;
      if (requestedBegin < allocationEnd && requestedEnd > allocationBegin) {
        LOG(ERROR)
            << "CUDF_DEVICE_MEMORY_POINTER label={" << label << "}"
            << " resource=" << resource << " state=recently_freed"
            << " requestPtr=" << pointer << " requestBytes=" << bytes
            << " allocationPtr=" << it->pointer
            << " allocationBytes=" << it->bytes
            << " offset="
            << (requestedBegin >= allocationBegin
                    ? requestedBegin - allocationBegin
                    : 0)
            << " allocationSequence=" << it->allocationSequence
            << " freeSequence=" << it->freeSequence
            << " context={" << it->context << "}";
        return;
      }
    }
    LOG(ERROR) << "CUDF_DEVICE_MEMORY_POINTER label={" << label << "}"
               << " resource=" << resource << " state=not_found"
               << " requestPtr=" << pointer << " requestBytes=" << bytes;
  }

 private:
  void recordAllocation(void* pointer, std::size_t bytes) {
    const auto context = currentAllocationContext.empty()
        ? std::string{"unattributed"}
        : currentAllocationContext;
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.emplace(
        pointer, Allocation{bytes, context, ++eventSequence_});
    auto& stats = contexts_[context];
    stats.currentBytes += bytes;
    stats.peakBytes = std::max(stats.peakBytes, stats.currentBytes);
    ++stats.currentAllocations;
  }

  void recordDeallocation(void* pointer) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto allocation = allocations_.find(pointer);
    if (allocation == allocations_.end()) {
      return;
    }
    auto context = contexts_.find(allocation->second.context);
    if (context != contexts_.end()) {
      context->second.currentBytes -= allocation->second.bytes;
      --context->second.currentAllocations;
    }
    recentlyFreed_.push_back(FreedAllocation{
        pointer,
        allocation->second.bytes,
        allocation->second.context,
        allocation->second.sequence,
        ++eventSequence_});
    if (recentlyFreed_.size() > 4096) {
      recentlyFreed_.pop_front();
    }
    allocations_.erase(allocation);
  }

  void dumpFailure(std::size_t requestedBytes) const {
    std::vector<std::pair<std::string, ContextStats>> snapshot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      snapshot.reserve(contexts_.size());
      for (const auto& entry : contexts_) {
        if (entry.second.currentBytes != 0) {
          snapshot.push_back(entry);
        }
      }
    }
    std::sort(
        snapshot.begin(),
        snapshot.end(),
        [](const auto& left, const auto& right) {
          return left.second.currentBytes > right.second.currentBytes;
        });

    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    const auto cudaStatus = cudaMemGetInfo(&freeBytes, &totalBytes);
    LOG(ERROR)
        << "CUDF_DEVICE_OOM triggerContext={" << currentAllocationContext
        << "} requestedBytes=" << requestedBytes
        << " cudaValid=" << (cudaStatus == cudaSuccess)
        << " freeBytes=" << freeBytes << " totalBytes=" << totalBytes
        << " attributedContexts=" << snapshot.size();
    const auto count = std::min<std::size_t>(snapshot.size(), 12);
    for (std::size_t index = 0; index < count; ++index) {
      LOG(ERROR)
          << "CUDF_DEVICE_OOM_OWNER rank=" << (index + 1)
          << " currentBytes=" << snapshot[index].second.currentBytes
          << " peakBytes=" << snapshot[index].second.peakBytes
          << " currentAllocations="
          << snapshot[index].second.currentAllocations
          << " context={" << snapshot[index].first << "}";
    }
  }

  rmm::device_async_resource_ref upstream_;
  mutable std::mutex mutex_;
  std::unordered_map<void*, Allocation> allocations_;
  std::unordered_map<std::string, ContextStats> contexts_;
  std::deque<FreedAllocation> recentlyFreed_;
  uint64_t eventSequence_{0};
};

class OperatorAttributionResource final
    : public cuda::mr::shared_resource<OperatorAttributionResourceImpl> {
  using Base = cuda::mr::shared_resource<OperatorAttributionResourceImpl>;

 public:
  explicit OperatorAttributionResource(
      rmm::device_async_resource_ref upstream)
      : Base(
            cuda::std::in_place_type<OperatorAttributionResourceImpl>,
            upstream) {}

  friend void get_property(
      const OperatorAttributionResource&,
      cuda::mr::device_accessible) noexcept {}
};

std::optional<OperatorAttributionResource> primaryAttributionResource;
std::optional<OperatorAttributionResource> outputAttributionResource;

std::mutex asyncMemoryPoolsMutex;
std::vector<cudaMemPool_t> asyncMemoryPools;
// registerCudf creates the main resource before its optional output resource.
// Remember which device has seen that first resource so an async output-only
// pool is never mistaken for the primary cuDF allocation pool.
std::unordered_set<int> primaryMemoryResourceDevices;
std::unordered_map<int, cudaMemPool_t> primaryAsyncMemoryPools;

std::mutex deviceMemoryAdmissionMutex;
std::unordered_map<int, std::size_t> deviceMemoryAdmissionBytes;

// Transient workspace uses separate accounting from persistent packed state.
// cudaMemGetInfo already includes live persistent allocations, while these
// reservations represent allocations that admitted drivers are about to make.
// Keeping the maps separate avoids double-counting persistent state.
std::mutex deviceMemoryWorkspaceMutex;
std::unordered_map<int, std::size_t> deviceMemoryWorkspaceBytes;
std::unordered_map<
    int,
    std::vector<std::shared_ptr<DeviceMemoryWorkspaceRequestState>>>
    deviceMemoryWorkspaceWaiters;
std::atomic<uint64_t> nextDeviceWorkspaceRequestSequence{1};
constexpr auto kDeviceWorkspacePollInterval = std::chrono::milliseconds(5);
// A fulfilled ContinuePromise is only an advisory: it does not reserve any
// device bytes, and the owning driver may remain unscheduled behind a
// downstream consumer. Give that driver a short exclusive retry window, then
// allow another fitting waiter to be advised. Without this lease, one stale
// kWakePending request can permanently stop the device-level scheduler while
// every GPU is idle.
constexpr auto kDeviceWorkspaceAdvisoryLease =
    std::chrono::milliseconds(20);
// A releasing output edge must not remain below a static physical watermark
// forever. After a bounded wait it may use the lower emergency cushion, but
// the byte-accounted reservation must still fit the same physical snapshot.
constexpr auto kDeviceWorkspaceOutputEscapeAge = std::chrono::seconds(2);

void scheduleDeviceMemoryWorkspaceWaiters(int device) noexcept;
void releaseDeviceMemoryWorkspace(int device, std::size_t bytes) noexcept;

// cudaMallocAsync releases become physically reusable only after their stream
// reaches the deallocation.  That transition does not release a cooperative
// workspace reservation and therefore has no natural Velox promise to fire.
// Poll only while at least one workspace request is queued, and use the
// existing priority/FIFO scheduler to fulfill its normal ContinuePromise as
// soon as the physical headroom becomes available.  The thread is idle for
// the common no-waiter case and avoids one CUDA event/callback per RMM free.
class DeviceWorkspaceWaiterPoller {
 public:
  DeviceWorkspaceWaiterPoller() : thread_{[this] { run(); }} {}

  ~DeviceWorkspaceWaiterPoller() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
      pollRequested_ = true;
    }
    cv_.notify_one();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void notify() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pollRequested_ = true;
    }
    cv_.notify_one();
  }

 private:
  void run() noexcept;

  std::mutex mutex_;
  std::condition_variable cv_;
  bool stopped_{false};
  bool pollRequested_{false};
  std::thread thread_;
};

// Declared after the waiter maps so it is destroyed first. Its destructor
// joins the polling thread before those maps and their mutex are torn down.
DeviceWorkspaceWaiterPoller deviceWorkspaceWaiterPoller;

// Serializes the observe -> reclaim decision within an executor. Shared
// arbitration already serializes pool mutations, but without this boundary two
// operator drivers can observe the same low headroom and both launch a large
// spill wave. Spark's GPU executor model uses one visible device per process;
// a process-wide lock therefore has no cross-device scheduling cost.
std::mutex devicePhysicalPressureMutex;

// The registry contains no callable cross-thread operator pointers. A pressure
// observation selects victims using their last boundary report and writes only
// atomics in DeviceMemoryReclaimerState. The owning Driver is solely
// responsible for invoking Operator::reclaim() at a subsequent safe boundary.
std::mutex deviceReclaimerRegistryMutex;
std::unordered_map<
    exec::Operator*,
    std::weak_ptr<DeviceMemoryReclaimerState>>
    deviceReclaimerRegistry;
std::atomic<uint64_t> nextDeviceReclaimEpoch{1};
std::unordered_map<int, std::chrono::steady_clock::time_point>
    lastNoDeviceVictimLog;

struct MemoryResourceRegistration {
  int device{-1};
  bool primary{false};
};

MemoryResourceRegistration beginMemoryResourceRegistration() {
  MemoryResourceRegistration registration;
  if (cudaGetDevice(&registration.device) != cudaSuccess) {
    return registration;
  }
  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  registration.primary =
      primaryMemoryResourceDevices.insert(registration.device).second;
  return registration;
}

void registerAsyncMemoryPool(
    const MemoryResourceRegistration& registration,
    cudaMemPool_t pool) {
  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  asyncMemoryPools.push_back(pool);
  if (registration.primary && registration.device >= 0) {
    primaryAsyncMemoryPools[registration.device] = pool;
  }
}

uint64_t deviceWorkspaceEffectiveMinHeadroom(
    std::size_t minHeadroomBytes,
    DeviceMemoryWorkspacePriority priority) {
  // The configured watermark is the steady-state cushion for streaming
  // input. A terminal drain releases a complete external partition and must
  // be able to make progress before input work which only grows state. Keep a
  // substantial physical cushion for restore kernels, but do not require the
  // full streaming watermark for work whose completion returns memory.
  switch (priority) {
    case DeviceMemoryWorkspacePriority::kOutput:
      // Output work releases an already-owned source, but the preceding
      // producer can still have asynchronous stream work and an output owner
      // alive after its own admission token is released. Keep enough physical
      // cushion for that overlap, while admitting the releasing output edge
      // before a full device can strand every producer below the streaming
      // watermark. The workspace request itself provides the other bounded
      // part of the output reserve.
      return minHeadroomBytes / 8;
    case DeviceMemoryWorkspacePriority::kDrain:
      // A drain is the next releasing edge once ingestion has closed. Keep
      // most of the former half-watermark cushion, but leave one sixteenth of
      // the configured watermark as scheduling slack. cudaMallocAsync pool
      // accounting moves in allocator-granularity steps; requiring an exact
      // half watermark can otherwise strand a releasing drain a few MiB below
      // its threshold when no material spill victim remains.
      return minHeadroomBytes / 2 - minHeadroomBytes / 16;
    case DeviceMemoryWorkspacePriority::kRestore:
      return minHeadroomBytes - minHeadroomBytes / 4;
    case DeviceMemoryWorkspacePriority::kTransform:
      return minHeadroomBytes - minHeadroomBytes / 4;
    case DeviceMemoryWorkspacePriority::kInput:
      return minHeadroomBytes;
  }
  VELOX_UNREACHABLE();
}

uint64_t deviceWorkspaceRequiredHeadroom(
    std::size_t bytes,
    std::size_t minHeadroomBytes,
    DeviceMemoryWorkspacePriority priority) {
  const auto effectiveMinHeadroom =
      deviceWorkspaceEffectiveMinHeadroom(minHeadroomBytes, priority);
  return effectiveMinHeadroom >
          std::numeric_limits<std::size_t>::max() - bytes
      ? std::numeric_limits<std::size_t>::max()
      : effectiveMinHeadroom + bytes;
}

int64_t effectiveDeviceWorkspacePriority(
    const DeviceMemoryWorkspaceRequestState& waiter,
    std::chrono::steady_clock::time_point /* now */) {
  // Priority classes are strict. In particular, a long-blocked input request
  // must never age above a partition drain: input grows persistent state,
  // while a completed drain releases state and opens the pipeline. Fairness
  // within one class is handled by the bounded starvation policy below, so
  // cross-class aging would only reintroduce the deadlock/backpressure pattern
  // this scheduler is meant to break.
  return static_cast<int64_t>(waiter.priority);
}

bool deviceWorkspaceRequestPrecedes(
    const DeviceMemoryWorkspaceRequestState& left,
    const DeviceMemoryWorkspaceRequestState& right,
    std::chrono::steady_clock::time_point now) {
  const auto leftPriority = effectiveDeviceWorkspacePriority(left, now);
  const auto rightPriority = effectiveDeviceWorkspacePriority(right, now);
  if (leftPriority != rightPriority) {
    return leftPriority > rightPriority;
  }

  // Restore and drain requests are independent operator work units. Before a
  // request becomes old, prefer the smaller fitting unit: it occupies less
  // transient memory and completes a partition sooner, which reduces the
  // number of simultaneous live build/TopN owners. Preserve input FIFO because
  // input is a stream, not a set of interchangeable partitions.
  //
  // Smallest-first alone can starve a skewed partition while short work keeps
  // arriving. Once either request has waited for this bounded interval, switch
  // the class back to FIFO. The original sequence survives scheduler handoff,
  // so this also bounds starvation across cudaMallocAsync headroom rechecks.
  constexpr auto kStarvationAge = std::chrono::seconds(2);
  const auto leftStarved = now - left.enqueueTime >= kStarvationAge;
  const auto rightStarved = now - right.enqueueTime >= kStarvationAge;
  if (leftStarved || rightStarved) {
    if (leftStarved != rightStarved) {
      return leftStarved;
    }
    return left.sequence < right.sequence;
  }
  if (left.priority != DeviceMemoryWorkspacePriority::kInput &&
      left.bytes != right.bytes) {
    return left.bytes < right.bytes;
  }
  return left.sequence < right.sequence;
}

bool deviceWorkspaceRequestFits(
    const DeviceMemoryWorkspaceRequestState& waiter,
    const DeviceAllocationHeadroom& headroom,
    std::size_t reservedBytes,
    std::chrono::steady_clock::time_point now) {
  auto requiredHeadroom = deviceWorkspaceRequiredHeadroom(
      waiter.bytes, waiter.minHeadroomBytes, waiter.priority);
  if (waiter.priority == DeviceMemoryWorkspacePriority::kOutput &&
      now - waiter.enqueueTime >= kDeviceWorkspaceOutputEscapeAge) {
    // kOutput requests include their complete bounded kernel estimate. Once
    // one output is the sole releasing edge on the device, requiring any
    // large watermark can reproduce the same fixed point a few allocator
    // pages lower. A synchronized retry below retires prior async work first;
    // retain a small physical cushion for allocator bookkeeping and callbacks.
    // The normal streaming watermark can be several GiB. Scaling the
    // post-synchronize allocator cushion from that watermark leaves a
    // releasing output stranded for reasons unrelated to its bounded kernel
    // estimate (6 GiB / 32 is 192 MiB in Job 144). Keep a small fixed upper
    // bound instead: the request already accounts for the complete output
    // workspace, the retry synchronizes the device, and reservations are
    // atomically charged against the same physical snapshot below.
    constexpr std::size_t kMaxOutputEscapeCushionBytes = 64ULL << 20;
    const auto emergencyCushion = std::min<std::size_t>(
        waiter.minHeadroomBytes / 32, kMaxOutputEscapeCushionBytes);
    requiredHeadroom = emergencyCushion >
            std::numeric_limits<std::size_t>::max() - waiter.bytes
        ? std::numeric_limits<std::size_t>::max()
        : waiter.bytes + emergencyCushion;
  }
  // cudaMallocAsync allocations can reuse already-reserved free pages in the
  // primary pool without increasing CUDA's physical allocation.  Treat those
  // pages as available for output just as we do for every other workspace.
  // Output safety comes from the complete byte estimate plus the shared
  // reservation; requiring cudaFree alone strands a releasing edge even when
  // the same pool has enough reusable capacity.
  const auto availableBytes = headroom.allocatableBytes();
  return headroom.cudaValid && headroom.device == waiter.device &&
      availableBytes >= requiredHeadroom &&
      reservedBytes <= availableBytes - requiredHeadroom;
}

bool isHighestPriorityDeviceWorkspaceWaiter(
    const std::shared_ptr<DeviceMemoryWorkspaceRequestState>& state) {
  if (state == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
  const auto waitersIt = deviceMemoryWorkspaceWaiters.find(state->device);
  if (waitersIt == deviceMemoryWorkspaceWaiters.end()) {
    return false;
  }
  const auto now = std::chrono::steady_clock::now();
  if (state->status != DeviceMemoryWorkspaceRequestStatus::kQueued ||
      state->retryNotBefore > now) {
    return false;
  }
  for (const auto& waiter : waitersIt->second) {
    if (waiter == nullptr || waiter == state ||
        (waiter->status != DeviceMemoryWorkspaceRequestStatus::kQueued &&
         waiter->status !=
             DeviceMemoryWorkspaceRequestStatus::kWakePending)) {
      continue;
    }
    if (waiter->status == DeviceMemoryWorkspaceRequestStatus::kWakePending) {
      // A recent advisory retains a bounded first chance to retry. An expired
      // one owns no capacity and must not suppress pressure handling for a
      // fitting queued waiter.
      if (now - waiter->wakeTime < kDeviceWorkspaceAdvisoryLease) {
        return false;
      }
      continue;
    }
    if (deviceWorkspaceRequestPrecedes(*waiter, *state, now)) {
      return false;
    }
  }
  return state->status == DeviceMemoryWorkspaceRequestStatus::kQueued;
}

void scheduleDeviceMemoryWorkspaceWaiters(int device) noexcept {
  if (device < 0) {
    return;
  }
  const auto headroom = captureDeviceAllocationHeadroom();
  if (!headroom.cudaValid || headroom.device != device) {
    return;
  }
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
    const auto it = deviceMemoryWorkspaceWaiters.find(device);
    if (it == deviceMemoryWorkspaceWaiters.end()) {
      return;
    }
    auto& waiters = it->second;
    waiters.erase(
        std::remove_if(
            waiters.begin(),
            waiters.end(),
            [](const auto& waiter) {
              return waiter == nullptr ||
                  (waiter->status !=
                       DeviceMemoryWorkspaceRequestStatus::kQueued &&
                   waiter->status !=
                       DeviceMemoryWorkspaceRequestStatus::kWakePending);
            }),
        waiters.end());

    if (waiters.empty()) {
      deviceMemoryWorkspaceWaiters.erase(it);
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    // Keep one fresh scheduler-to-driver advisory wake per device. A wake
    // owns no virtual capacity, so let its lease expire if the Velox driver
    // cannot promptly retry (for example because its downstream consumer is
    // blocked). This preserves the anti-herd gate without letting a stale
    // kWakePending request deadlock all other drain work.
    if (std::any_of(
            waiters.begin(), waiters.end(), [&](const auto& waiter) {
              return waiter->status ==
                      DeviceMemoryWorkspaceRequestStatus::kWakePending &&
                  now - waiter->wakeTime <
                      kDeviceWorkspaceAdvisoryLease;
            })) {
      return;
    }

    const auto reservedIt = deviceMemoryWorkspaceBytes.find(device);
    const auto reservedBytes = reservedIt == deviceMemoryWorkspaceBytes.end()
        ? 0
        : reservedIt->second;
    auto best = waiters.end();
    for (auto candidate = waiters.begin(); candidate != waiters.end();
         ++candidate) {
      if ((*candidate)->status !=
          DeviceMemoryWorkspaceRequestStatus::kQueued) {
        continue;
      }
      if ((*candidate)->retryNotBefore > now) {
        continue;
      }
      if (!deviceWorkspaceRequestFits(
              **candidate, headroom, reservedBytes, now)) {
        continue;
      }
      if (best == waiters.end()) {
        best = candidate;
        continue;
      }
      if (deviceWorkspaceRequestPrecedes(**candidate, **best, now)) {
        best = candidate;
      }
    }
    if (best != waiters.end()) {
      // Publish only an advisory wake. Reserving before the Velox driver
      // actually resumes can retain several GiB of virtual capacity while
      // that driver is blocked by a downstream consumer, creating a cycle in
      // which every GPU is idle but no producer can be admitted. The retry
      // path below preserves fairness by considering only preceding waiters
      // that also fit the then-current headroom.
      auto waiter = *best;
      waiter->status = DeviceMemoryWorkspaceRequestStatus::kWakePending;
      waiter->wakeTime = now;
      if (waiter->promise.has_value()) {
        promises.push_back(std::move(waiter->promise.value()));
        waiter->promise.reset();
      }
      VLOG(1) << "CUDF_DEVICE_WORKSPACE_WAKE device=" << device
              << " operator="
              << (waiter->requestor != nullptr
                      ? waiter->requestor->operatorType()
                      : "CudfConnector")
              << " task="
              << (waiter->requestor != nullptr ? waiter->requestor->taskId()
                                                : "unknown")
              << " node="
              << (waiter->requestor != nullptr
                      ? waiter->requestor->planNodeId()
                      : "unknown")
              << " bytes=" << waiter->bytes
              << " priority=" << static_cast<int>(waiter->priority)
              << " waitUs="
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     now - waiter->enqueueTime)
                     .count()
              << " activeReservedBytes=" << reservedBytes;
    }
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void DeviceWorkspaceWaiterPoller::run() noexcept {
  std::unique_lock<std::mutex> pollLock(mutex_);
  while (true) {
    cv_.wait(pollLock, [&] { return stopped_ || pollRequested_; });
    if (stopped_) {
      return;
    }
    pollRequested_ = false;
    pollLock.unlock();

    while (true) {
      std::this_thread::sleep_for(kDeviceWorkspacePollInterval);
      std::vector<int> devices;
      {
        std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
        devices.reserve(deviceMemoryWorkspaceWaiters.size());
        for (const auto& [device, waiters] :
             deviceMemoryWorkspaceWaiters) {
          const auto hasQueued = std::any_of(
              waiters.begin(), waiters.end(), [](const auto& waiter) {
                return waiter != nullptr &&
                    waiter->status ==
                    DeviceMemoryWorkspaceRequestStatus::kQueued;
              });
          if (hasQueued) {
            devices.push_back(device);
          }
        }
      }
      if (devices.empty()) {
        break;
      }
      for (const auto device : devices) {
        // CUDA current-device state is thread-local. Each Spark executor
        // normally sees one logical device, but setting it explicitly keeps
        // this correct for multi-device native runners as well.
        if (cudaSetDevice(device) == cudaSuccess) {
          scheduleDeviceMemoryWorkspaceWaiters(device);
        }
      }
    }

    pollLock.lock();
  }
}

void notifyDeviceMemoryWorkspaceWaiters(int device) noexcept {
  scheduleDeviceMemoryWorkspaceWaiters(device);
}

void cancelDeviceMemoryWorkspaceRequest(
    const std::shared_ptr<DeviceMemoryWorkspaceRequestState>& state) noexcept {
  if (state == nullptr) {
    return;
  }
  std::optional<ContinuePromise> promise;
  bool shouldSchedule = false;
  {
    std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
    if (state->status == DeviceMemoryWorkspaceRequestStatus::kQueued) {
      state->status = DeviceMemoryWorkspaceRequestStatus::kCanceled;
      if (state->promise.has_value()) {
        promise.emplace(std::move(state->promise.value()));
        state->promise.reset();
      }
      // Remove this canceled waiter promptly and let a lower-priority request
      // use headroom that the canceled request may have been blocking.
      shouldSchedule = true;
    } else if (
        state->status == DeviceMemoryWorkspaceRequestStatus::kWakePending) {
      // The wake is advisory, so cancellation has no virtual bytes to return.
      state->status = DeviceMemoryWorkspaceRequestStatus::kCanceled;
      shouldSchedule = true;
    }
  }
  if (promise.has_value()) {
    promise->setValue();
  }
  if (shouldSchedule) {
    scheduleDeviceMemoryWorkspaceWaiters(state->device);
  }
}

void releaseDeviceMemoryWorkspace(int device, std::size_t bytes) noexcept {
  {
    std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
    const auto it = deviceMemoryWorkspaceBytes.find(device);
    if (it == deviceMemoryWorkspaceBytes.end() || it->second < bytes) {
      LOG(ERROR) << "Invalid device-workspace release device=" << device
                 << " bytes=" << bytes;
      return;
    }
    it->second -= bytes;
    if (it->second == 0) {
      deviceMemoryWorkspaceBytes.erase(it);
    }
  }
  notifyDeviceMemoryWorkspaceWaiters(device);
}

void releaseDeviceMemoryAdmission(int device, std::size_t bytes) noexcept {
  std::lock_guard<std::mutex> lock(deviceMemoryAdmissionMutex);
  const auto it = deviceMemoryAdmissionBytes.find(device);
  if (it == deviceMemoryAdmissionBytes.end() || it->second < bytes) {
    LOG(ERROR) << "Invalid device-memory admission release device=" << device
               << " bytes=" << bytes;
    return;
  }
  it->second -= bytes;
  if (it->second == 0) {
    deviceMemoryAdmissionBytes.erase(it);
  }
  notifyDeviceMemoryWorkspaceWaiters(device);
}

std::shared_ptr<DeviceMemoryReclaimerState> findDeviceReclaimer(
    exec::Operator* owner) {
  if (owner == nullptr) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(deviceReclaimerRegistryMutex);
  const auto it = deviceReclaimerRegistry.find(owner);
  if (it == deviceReclaimerRegistry.end()) {
    return nullptr;
  }
  auto state = it->second.lock();
  if (state == nullptr) {
    deviceReclaimerRegistry.erase(it);
  }
  return state;
}

uint64_t refreshDeviceReclaimableBytes(
    const std::shared_ptr<DeviceMemoryReclaimerState>& state,
    exec::Operator* owner) {
  if (state == nullptr || owner == nullptr || state->owner != owner) {
    return 0;
  }
  if (state->device.load(std::memory_order_relaxed) < 0) {
    int device = -1;
    if (cudaGetDevice(&device) == cudaSuccess) {
      state->device.store(device, std::memory_order_relaxed);
    }
  }
  uint64_t bytes = 0;
  if (!owner->canReclaim() || !owner->reclaimableBytes(bytes)) {
    bytes = 0;
  }
  state->reportedReclaimableBytes.store(bytes, std::memory_order_release);
  return bytes;
}

void serviceDeviceReclaimerState(
    const std::shared_ptr<DeviceMemoryReclaimerState>& state,
    exec::Operator* owner,
    std::string_view safePoint) {
  const auto reportedBefore = refreshDeviceReclaimableBytes(state, owner);
  if (state == nullptr) {
    return;
  }
  if (state->requestedBytes.load(std::memory_order_acquire) == 0) {
    return;
  }
  bool expected = false;
  if (!state->reclaimInProgress.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  const auto requested =
      state->requestedBytes.exchange(0, std::memory_order_acq_rel);
  if (requested == 0) {
    state->reclaimInProgress.store(false, std::memory_order_release);
    return;
  }

  uint64_t reclaimed = 0;
  try {
    if (reportedBefore > 0) {
      // This is a cooperative self-reclaim on the owning Driver. Unlike an
      // external SharedArbitrator shrink, no other Task is paused while it may
      // be inside a long CUDA kernel or transfer.
      exec::Operator::ReclaimableSectionGuard reclaimableSection(owner);
      const auto usedBefore = state->pool->usedBytes();
      memory::MemoryReclaimer::Stats stats;
      owner->reclaim(requested, stats);
      const auto usedAfter = state->pool->usedBytes();
      reclaimed = usedBefore > usedAfter ? usedBefore - usedAfter : 0;
    }
  } catch (...) {
    state->reclaimInProgress.store(false, std::memory_order_release);
    throw;
  }
  const auto reportedAfter = refreshDeviceReclaimableBytes(state, owner);
  const auto headroom = captureDeviceAllocationHeadroom();
  state->reclaimInProgress.store(false, std::memory_order_release);
  if (reclaimed > 0) {
    notifyDeviceMemoryWorkspaceWaiters(
        state->device.load(std::memory_order_relaxed));
  }
  LOG(WARNING)
      << "CUDF_DEVICE_COOPERATIVE_RECLAIM device=" << state->device.load()
      << " epoch=" << state->requestEpoch.load(std::memory_order_relaxed)
      << " safePoint=" << safePoint
      << " operator=" << owner->operatorType()
      << " task=" << owner->taskId()
      << " node=" << owner->planNodeId()
      << " requestedBytes=" << requested
      << " reportedBeforeBytes=" << reportedBefore
      << " reclaimedCapacityBytes=" << reclaimed
      << " reportedAfterBytes=" << reportedAfter
      << " allocatableAfter=" << headroom.allocatableBytes();
}

void requestDeviceMemoryReclaimForPhysicalPressure(
    memory::MemoryPool* pool,
    exec::Operator* requestor,
    uint64_t requiredHeadroomBytes = 0,
    bool requireCompleteExpensiveWave = false) {
  const auto& config = CudfConfig::getInstance();
  const auto minHeadroom = requiredHeadroomBytes == 0
      ? config.deviceMemoryMinHeadroomBytes
      : std::max<uint64_t>(
            requiredHeadroomBytes, config.deviceMemoryMinHeadroomBytes);
  if (minHeadroom == 0) {
    return;
  }

  auto requestorState = findDeviceReclaimer(requestor);
  // Refresh the requestor's estimate at the admission boundary and service a
  // request that another Driver may already have published for it.
  serviceDeviceReclaimerState(
      requestorState, requestor, "deviceAdmission.pending");

  std::lock_guard<std::mutex> lock(devicePhysicalPressureMutex);
  const auto before = captureDeviceAllocationHeadroom();
  if (!before.cudaValid || before.allocatableBytes() >= minHeadroom) {
    return;
  }

  const auto minReclaim = std::max<uint64_t>(
      config.deviceMemoryMinReclaimBytes, 1);
  // Normal persistent-state admission uses hysteresis. An explicit workspace
  // request already includes the kernel's byte estimate, so targeting another
  // full reclaim wave above it would externalize healthy state unnecessarily.
  const auto highWatermark = requiredHeadroomBytes != 0
      ? minHeadroom
      : (minHeadroom > std::numeric_limits<uint64_t>::max() - minReclaim
             ? std::numeric_limits<uint64_t>::max()
             : minHeadroom + minReclaim);
  const auto physicalShortfall =
      highWatermark > before.allocatableBytes()
      ? highWatermark - before.allocatableBytes()
      : 0;
  // A concrete workspace request needs only enough victims to make that
  // request fit. Applying the persistent-state 2-GiB hysteresis here can turn
  // a few-hundred-MiB shortfall into a sweep of several resident hash builds,
  // multiplying the subsequent Grace spill/restore work. Keep hysteresis for
  // background persistent pressure, where amortizing reclaim waves matters.
  const auto requestBytes = requiredHeadroomBytes != 0
      ? std::max<uint64_t>(physicalShortfall, 1)
      : std::max<uint64_t>(minReclaim, physicalShortfall);
  // Ignore tiny owners. Reclaiming one 8-10 MiB TopN chunk for every incoming
  // batch caused hundreds of pressure events without moving the global water
  // level. Wait for a material victim and reclaim in a bucket/build-sized wave.
  const auto minVictimBytes = std::max<uint64_t>(
      64ULL << 20, std::min<uint64_t>(minReclaim, minReclaim / 8));
  // A small resident Join build is a poor cross-Driver victim for ordinary
  // input or a one-batch transform: freeing a fraction of one reclaim wave can
  // externalize tens of GiB of downstream probe traffic. Keep it resident
  // until it can supply at least half of the configured wave. Synchronous
  // self-reclaim remains eligible at every size so this protection cannot
  // create a wait cycle.
  const auto minCrossDriverExpensiveVictimBytes =
      std::max<uint64_t>(minVictimBytes, minReclaim / 2);

  struct Candidate {
    std::shared_ptr<DeviceMemoryReclaimerState> state;
    uint64_t bytes;
    // Reclaiming a resident hash-join build turns the complete downstream
    // probe into a Grace spill/restore pass.  Reclaiming TopN state only
    // externalizes bounded buckets which already have a pipelined pinned
    // restore path. Keep the latter ahead of joins when either can satisfy
    // physical pressure. Blocking workspace requests still handle their
    // current requester first below to preserve the no-wait-cycle contract.
    bool expensiveToRebuild;
  };
  std::vector<Candidate> candidates;
  uint64_t protectedSmallExpensiveVictims = 0;
  bool outstanding = false;
  {
    std::lock_guard<std::mutex> registryLock(deviceReclaimerRegistryMutex);
    for (auto it = deviceReclaimerRegistry.begin();
         it != deviceReclaimerRegistry.end();) {
      auto state = it->second.lock();
      if (state == nullptr) {
        it = deviceReclaimerRegistry.erase(it);
        continue;
      }
      ++it;
      if (state->device.load(std::memory_order_relaxed) != before.device) {
        continue;
      }
      if (state->reclaimInProgress.load(std::memory_order_acquire) ||
          state->requestedBytes.load(std::memory_order_acquire) != 0) {
        outstanding = true;
        continue;
      }
      const auto bytes =
          state->reportedReclaimableBytes.load(std::memory_order_acquire);
      if (bytes >= minVictimBytes) {
        const bool expensiveToRebuild =
            state->owner->operatorType() == "CudfHashJoinBuild";
        if (requireCompleteExpensiveWave && expensiveToRebuild &&
            state != requestorState &&
            bytes < minCrossDriverExpensiveVictimBytes) {
          ++protectedSmallExpensiveVictims;
          continue;
        }
        candidates.push_back(Candidate{
            std::move(state), bytes, expensiveToRebuild});
      }
    }

    const bool requestorIsCandidate = std::any_of(
        candidates.begin(),
        candidates.end(),
        [&](const Candidate& candidate) {
          return candidate.state == requestorState;
        });
    // An outstanding cooperative request suppresses another cross-Driver
    // wave, but it must not suppress synchronous self-reclaim by this
    // workspace requester. Self-reclaim is what breaks a cycle in which every
    // registered owner has blocked before servicing somebody else's request.
    if (!outstanding || requestorIsCandidate) {
      std::sort(
          candidates.begin(),
          candidates.end(),
          [&](const Candidate& left, const Candidate& right) {
            // A workspace requester must not go to sleep while retaining
            // reclaimable device state. Otherwise the global largest victim
            // may itself be waiting for workspace, and every Driver can end
            // up waiting for a victim which cannot reach its next safe point.
            // Prefer the requester so it can reclaim synchronously below.
            const bool leftIsRequestor = left.state == requestorState;
            const bool rightIsRequestor = right.state == requestorState;
            if (leftIsRequestor != rightIsRequestor) {
              return leftIsRequestor;
            }
            // A resident Join build is only hundreds of MiB, but evicting it
            // makes every following probe row cross the Grace host/disk tier.
            // Prefer bounded TopN bucket eviction even when a Join happens to
            // be the larger immediate byte victim.
            if (left.expensiveToRebuild != right.expensiveToRebuild) {
              return !left.expensiveToRebuild;
            }
            return left.bytes > right.bytes;
          });
      // An input waiter grows state, and a transform releases only its source
      // batch. If all currently reclaimable state cannot make either waiter
      // fit, evicting a resident Join build is pure amplification: the waiter
      // remains blocked until transient reservations complete, while the
      // entire downstream probe is irreversibly converted to Grace work.
      // Preserve bounded TopN reclaim and synchronous self-reclaim, but defer
      // cross-Driver Join victims until one wave can close the shortfall.
      uint64_t potentialAssigned = 0;
      bool potentialCrossDriverExpensiveVictim = false;
      if (requireCompleteExpensiveWave) {
        for (const auto& candidate : candidates) {
          if (potentialAssigned >= requestBytes) {
            break;
          }
          if (outstanding && candidate.state != requestorState) {
            break;
          }
          const auto victimRequest = std::min<uint64_t>(
              candidate.bytes,
              std::max<uint64_t>(
                  minVictimBytes, requestBytes - potentialAssigned));
          potentialAssigned += victimRequest;
          potentialCrossDriverExpensiveVictim |=
              candidate.expensiveToRebuild &&
              candidate.state != requestorState;
        }
      }
      const bool deferCrossDriverExpensiveVictims =
          requireCompleteExpensiveWave &&
          potentialCrossDriverExpensiveVictim &&
          potentialAssigned < requestBytes;
      const auto epoch =
          nextDeviceReclaimEpoch.fetch_add(1, std::memory_order_relaxed);
      uint64_t assigned = 0;
      uint64_t selectedVictims = 0;
      uint64_t selectedExpensiveVictims = 0;
      uint64_t deferredExpensiveVictims = 0;
      bool selectedRequestor = false;
      for (auto& candidate : candidates) {
        if (assigned >= requestBytes) {
          break;
        }
        if (outstanding && candidate.state != requestorState) {
          break;
        }
        if (deferCrossDriverExpensiveVictims &&
            candidate.expensiveToRebuild &&
            candidate.state != requestorState) {
          ++deferredExpensiveVictims;
          continue;
        }
        const auto victimRequest = std::min<uint64_t>(
            candidate.bytes,
            std::max<uint64_t>(minVictimBytes, requestBytes - assigned));
        candidate.state->requestEpoch.store(epoch, std::memory_order_relaxed);
        candidate.state->requestedBytes.store(
            victimRequest, std::memory_order_release);
        assigned += victimRequest;
        ++selectedVictims;
        selectedExpensiveVictims += candidate.expensiveToRebuild ? 1 : 0;
        selectedRequestor |= candidate.state == requestorState;
      }
      if (assigned > 0) {
        LOG(WARNING)
            << "CUDF_DEVICE_GLOBAL_ARBITRATION device=" << before.device
            << " mode=publish_cooperative_victims"
            << " epoch=" << epoch
            << " minHeadroomBytes=" << minHeadroom
            << " highWatermarkBytes=" << highWatermark
            << " requestedBytes=" << requestBytes
            << " assignedBytes=" << assigned
            << " victimCount=" << selectedVictims
            << " expensiveVictimCount=" << selectedExpensiveVictims
            << " deferredExpensiveVictimCount=" << deferredExpensiveVictims
            << " selectedRequestor=" << selectedRequestor
            << " allocatableBefore=" << before.allocatableBytes()
            << " cudaFreeBefore=" << before.freeBytes
            << " reusablePoolBefore=" << before.reusablePoolBytes();
      } else if (deferredExpensiveVictims > 0) {
        LOG(WARNING)
            << "CUDF_DEVICE_GLOBAL_ARBITRATION device=" << before.device
            << " mode=defer_incomplete_expensive_wave"
            << " epoch=" << epoch
            << " requestedBytes=" << requestBytes
            << " potentialAssignedBytes=" << potentialAssigned
            << " deferredExpensiveVictimCount="
            << deferredExpensiveVictims
            << " allocatableBytes=" << before.allocatableBytes();
      } else if (protectedSmallExpensiveVictims > 0) {
        LOG(WARNING)
            << "CUDF_DEVICE_GLOBAL_ARBITRATION device=" << before.device
            << " mode=protect_small_expensive_victim"
            << " epoch=" << epoch
            << " requestedBytes=" << requestBytes
            << " minExpensiveVictimBytes="
            << minCrossDriverExpensiveVictimBytes
            << " protectedExpensiveVictimCount="
            << protectedSmallExpensiveVictims
            << " allocatableBytes=" << before.allocatableBytes();
      } else {
        const auto now = std::chrono::steady_clock::now();
        const auto previous = lastNoDeviceVictimLog.find(before.device);
        if (previous == lastNoDeviceVictimLog.end() ||
            now - previous->second >= std::chrono::seconds(1)) {
          lastNoDeviceVictimLog[before.device] = now;
          LOG(WARNING)
              << "CUDF_DEVICE_GLOBAL_ARBITRATION device=" << before.device
              << " mode=no_material_victim"
              << " minVictimBytes=" << minVictimBytes
              << " allocatableBytes=" << before.allocatableBytes();
        }
      }
    }
  }

  // If the requestor was selected, reclaim immediately while it is already at
  // the admission safe point. Other victims will observe their atomic request
  // at their own next operator boundary.
  serviceDeviceReclaimerState(
      requestorState, requestor, "deviceAdmission.selected");
}
} // namespace

cuda::mr::any_resource<cuda::mr::device_accessible>
wrapDeviceMemoryResourceForDiagnostics(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    bool outputResource) {
  auto& statistics =
      outputResource ? output_statistics_mr_ : statistics_mr_;
  auto& attribution = outputResource ? outputAttributionResource
                                     : primaryAttributionResource;
  statistics.emplace(std::move(upstream));
  attribution.emplace(rmm::device_async_resource_ref{statistics.value()});
  return cuda::mr::any_resource<cuda::mr::device_accessible>{
      attribution.value()};
}

cuda::mr::any_resource<cuda::mr::device_accessible> createMemoryResource(
    std::string_view mode,
    int percent) {
  const auto registration = beginMemoryResourceRegistration();
  if (mode == "cuda") {
    return rmm::mr::cuda_memory_resource{};
  } else if (mode == "pool") {
    return rmm::mr::pool_memory_resource(
        rmm::mr::cuda_memory_resource{},
        rmm::percent_of_free_device_memory(percent));
  } else if (mode == "async") {
    // cuda_async_memory_resource otherwise defaults its release threshold to
    // UINT64_MAX. After a transient large join/partition workspace, the CUDA
    // pool then retains almost the entire device even when RMM's live bytes
    // have fallen. UCX receive buffers use a separate cudaMalloc resource and
    // cannot reuse those cached blocks. Apply memoryPercent as the async
    // pool's retention threshold so synchronization returns high-watermark
    // slack while preserving a large cache for steady-state operators.
    auto resource = rmm::mr::cuda_async_memory_resource(
        std::nullopt, rmm::percent_of_free_device_memory(percent));
    registerAsyncMemoryPool(registration, resource.pool_handle());
    return resource;
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

bool trimAsyncMemoryPoolsAtQueryEnd() {
  const auto* value = std::getenv("GLUTEN_CUDF_ASYNC_QUERY_END_TRIM_BYTES");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }

  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    LOG(ERROR) << "Ignoring invalid GLUTEN_CUDF_ASYNC_QUERY_END_TRIM_BYTES='"
               << value << "'";
    return false;
  }
  return trimAsyncMemoryPoolsAtQueryEnd(static_cast<std::size_t>(parsed));
}

bool trimAsyncMemoryPoolsAtQueryEnd(std::size_t bytesToKeep) {
  std::vector<cudaMemPool_t> pools;
  {
    std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
    pools = asyncMemoryPools;
  }
  if (pools.empty()) {
    return false;
  }

  std::size_t freeBefore = 0;
  std::size_t totalBytes = 0;
  cudaMemGetInfo(&freeBefore, &totalBytes);
  const auto syncStatus = cudaDeviceSynchronize();
  if (syncStatus != cudaSuccess) {
    LOG(ERROR) << "CUDF_ASYNC_QUERY_END_TRIM cudaDeviceSynchronize failed: "
               << cudaGetErrorString(syncStatus);
    return false;
  }

  for (const auto pool : pools) {
    const auto trimStatus = cudaMemPoolTrimTo(pool, bytesToKeep);
    if (trimStatus != cudaSuccess) {
      LOG(ERROR) << "CUDF_ASYNC_QUERY_END_TRIM cudaMemPoolTrimTo failed: "
                 << cudaGetErrorString(trimStatus)
                 << " bytesToKeep=" << bytesToKeep;
      return false;
    }
  }

  std::size_t freeAfter = 0;
  cudaMemGetInfo(&freeAfter, &totalBytes);
  LOG(INFO) << "CUDF_ASYNC_QUERY_END_TRIM pools=" << pools.size()
            << " bytesToKeep=" << bytesToKeep << " freeBefore=" << freeBefore
            << " freeAfter=" << freeAfter << " released="
            << (freeAfter >= freeBefore ? freeAfter - freeBefore : 0);
  return true;
}

void clearAsyncMemoryPoolHandles() {
  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  asyncMemoryPools.clear();
  primaryAsyncMemoryPools.clear();
  primaryMemoryResourceDevices.clear();
}

std::size_t DeviceAllocationHeadroom::reusablePoolBytes() const noexcept {
  if (!asyncPoolValid || poolReservedBytes <= poolUsedBytes) {
    return 0;
  }
  return poolReservedBytes - poolUsedBytes;
}

std::size_t DeviceAllocationHeadroom::allocatableBytes() const noexcept {
  if (!cudaValid) {
    return 0;
  }
  const auto reusable = reusablePoolBytes();
  if (freeBytes > std::numeric_limits<std::size_t>::max() - reusable) {
    return std::numeric_limits<std::size_t>::max();
  }
  return freeBytes + reusable;
}

DeviceAllocationHeadroom captureDeviceAllocationHeadroom() {
  DeviceAllocationHeadroom headroom;
  if (cudaGetDevice(&headroom.device) != cudaSuccess) {
    return headroom;
  }

  headroom.cudaValid =
      cudaMemGetInfo(&headroom.freeBytes, &headroom.totalBytes) == cudaSuccess;
  if (!headroom.cudaValid) {
    headroom.freeBytes = 0;
    headroom.totalBytes = 0;
    return headroom;
  }

  std::lock_guard<std::mutex> lock(asyncMemoryPoolsMutex);
  const auto poolIt = primaryAsyncMemoryPools.find(headroom.device);
  if (poolIt == primaryAsyncMemoryPools.end()) {
    return headroom;
  }

  std::uint64_t reservedBytes = 0;
  std::uint64_t usedBytes = 0;
  const auto reservedStatus = cudaMemPoolGetAttribute(
      poolIt->second, cudaMemPoolAttrReservedMemCurrent, &reservedBytes);
  const auto usedStatus = cudaMemPoolGetAttribute(
      poolIt->second, cudaMemPoolAttrUsedMemCurrent, &usedBytes);
  if (reservedStatus != cudaSuccess || usedStatus != cudaSuccess) {
    return headroom;
  }

  headroom.asyncPoolValid = true;
  headroom.poolReservedBytes = static_cast<std::size_t>(reservedBytes);
  headroom.poolUsedBytes = static_cast<std::size_t>(usedBytes);
  return headroom;
}

DeviceMemoryAdmissionReservation::DeviceMemoryAdmissionReservation(
    int device,
    std::size_t bytes) noexcept
    : device_{device}, bytes_{bytes}, active_{true} {}

DeviceMemoryAdmissionReservation::DeviceMemoryAdmissionReservation(
    memory::MemoryPool* pool,
    std::size_t bytes) noexcept
    : pool_{pool}, bytes_{bytes}, active_{true} {
  cudaGetDevice(&device_);
}

DeviceMemoryAdmissionReservation::~DeviceMemoryAdmissionReservation() {
  release();
}

DeviceMemoryAdmissionReservation::DeviceMemoryAdmissionReservation(
    DeviceMemoryAdmissionReservation&& other) noexcept
    : device_{other.device_}, bytes_{other.bytes_}, active_{other.active_} {
  pool_ = other.pool_;
  other.device_ = -1;
  other.pool_ = nullptr;
  other.bytes_ = 0;
  other.active_ = false;
}

DeviceMemoryAdmissionReservation& DeviceMemoryAdmissionReservation::operator=(
    DeviceMemoryAdmissionReservation&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  device_ = other.device_;
  pool_ = other.pool_;
  bytes_ = other.bytes_;
  active_ = other.active_;
  other.device_ = -1;
  other.pool_ = nullptr;
  other.bytes_ = 0;
  other.active_ = false;
  return *this;
}

void DeviceMemoryAdmissionReservation::release() noexcept {
  if (!active_) {
    return;
  }
  if (pool_ != nullptr) {
    pool_->reportExternalFree(static_cast<int64_t>(bytes_));
    notifyDeviceMemoryWorkspaceWaiters(device_);
  } else {
    releaseDeviceMemoryAdmission(device_, bytes_);
  }
  device_ = -1;
  pool_ = nullptr;
  bytes_ = 0;
  active_ = false;
}

DeviceMemoryWorkspaceReservation::DeviceMemoryWorkspaceReservation(
    int device,
    memory::MemoryPool* pool,
    std::size_t bytes) noexcept
    : device_{device}, pool_{pool}, bytes_{bytes}, active_{true} {}

DeviceMemoryWorkspaceReservation::~DeviceMemoryWorkspaceReservation() {
  release();
}

DeviceMemoryWorkspaceRequest::~DeviceMemoryWorkspaceRequest() {
  reset();
}

DeviceMemoryWorkspaceRequest::DeviceMemoryWorkspaceRequest(
    DeviceMemoryWorkspaceRequest&& other) noexcept
    : state_{std::move(other.state_)} {}

DeviceMemoryWorkspaceRequest& DeviceMemoryWorkspaceRequest::operator=(
    DeviceMemoryWorkspaceRequest&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  state_ = std::move(other.state_);
  return *this;
}

void DeviceMemoryWorkspaceRequest::reset() noexcept {
  if (state_ == nullptr) {
    return;
  }
  auto state = std::move(state_);
  cancelDeviceMemoryWorkspaceRequest(state);
}

ReplayableDeviceMemoryWorkspace::Attempt
ReplayableDeviceMemoryWorkspace::tryAcquire(
    memory::MemoryPool* pool,
    exec::Operator* requestor,
    std::size_t bytes,
    std::size_t minHeadroomBytes,
    DeviceMemoryWorkspacePriority priority) {
  VELOX_CHECK(
      !waiting_,
      "A replayable workspace future must be handed to the Driver before "
      "admission is retried");
  auto nextFuture = ContinueFuture::makeEmpty();
  auto reservation = tryAcquireDeviceMemoryWorkspace(
      pool,
      requestor,
      bytes,
      minHeadroomBytes,
      priority,
      &request_,
      &nextFuture);
  if (reservation.has_value()) {
    std::optional<uint64_t> completedWaitMicros;
    if (waitStart_.has_value()) {
      completedWaitMicros =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - waitStart_.value())
              .count();
      waitStart_.reset();
    }
    return {
        std::move(reservation), false, std::move(completedWaitMicros)};
  }

  const bool firstWait = !waitStart_.has_value();
  if (firstWait) {
    waitStart_ = std::chrono::steady_clock::now();
  }
  future_ = std::move(nextFuture);
  waiting_ = true;
  return {std::nullopt, firstWait, std::nullopt};
}

bool ReplayableDeviceMemoryWorkspace::takeFuture(ContinueFuture* future) {
  if (!waiting_) {
    return false;
  }
  VELOX_CHECK_NOT_NULL(future);
  *future = std::move(future_);
  waiting_ = false;
  return true;
}

void ReplayableDeviceMemoryWorkspace::reset() noexcept {
  request_.reset();
  future_ = ContinueFuture::makeEmpty();
  waiting_ = false;
  waitStart_.reset();
}

void ReplayableDeviceMemoryWorkspace::deferReadyForTesting() {
  VELOX_CHECK(!waiting_);
  ContinuePromise promise{"ReplayableDeviceMemoryWorkspace::test"};
  future_ = promise.getSemiFuture();
  promise.setValue();
  if (!waitStart_.has_value()) {
    waitStart_ = std::chrono::steady_clock::now();
  }
  waiting_ = true;
}

DeviceMemoryWorkspaceReservation::DeviceMemoryWorkspaceReservation(
    DeviceMemoryWorkspaceReservation&& other) noexcept
    : device_{other.device_},
      pool_{other.pool_},
      bytes_{other.bytes_},
      active_{other.active_} {
  other.device_ = -1;
  other.pool_ = nullptr;
  other.bytes_ = 0;
  other.active_ = false;
}

DeviceMemoryWorkspaceReservation&
DeviceMemoryWorkspaceReservation::operator=(
    DeviceMemoryWorkspaceReservation&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  device_ = other.device_;
  pool_ = other.pool_;
  bytes_ = other.bytes_;
  active_ = other.active_;
  other.device_ = -1;
  other.pool_ = nullptr;
  other.bytes_ = 0;
  other.active_ = false;
  return *this;
}

void DeviceMemoryWorkspaceReservation::release() noexcept {
  if (!active_) {
    return;
  }
  if (pool_ != nullptr) {
    pool_->reportExternalFree(static_cast<int64_t>(bytes_));
  }
  releaseDeviceMemoryWorkspace(device_, bytes_);
  device_ = -1;
  pool_ = nullptr;
  bytes_ = 0;
  active_ = false;
}

DeviceMemoryReclaimerRegistration registerDeviceMemoryReclaimer(
    exec::Operator* owner,
    memory::MemoryPool* pool) {
  if (owner == nullptr || pool == nullptr) {
    return {};
  }
  int device = -1;
  cudaGetDevice(&device);
  auto state =
      std::make_shared<DeviceMemoryReclaimerState>(owner, pool, device);
  {
    std::lock_guard<std::mutex> lock(deviceReclaimerRegistryMutex);
    deviceReclaimerRegistry[owner] = state;
  }
  return DeviceMemoryReclaimerRegistration{std::move(state)};
}

void serviceDeviceMemoryReclaimer(
    DeviceMemoryReclaimerRegistration& registration,
    exec::Operator* owner,
    std::string_view safePoint,
    bool observePhysicalPressure) {
  serviceDeviceReclaimerState(registration.state_, owner, safePoint);
  if (observePhysicalPressure && registration.state_ != nullptr) {
    requestDeviceMemoryReclaimForPhysicalPressure(
        registration.state_->pool, owner);
  }
}

DeviceMemoryAdmissionReservation acquireDeviceMemoryAdmission(
    memory::MemoryPool* pool,
    std::size_t bytes,
    exec::Operator* requestor) {
  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK_NOT_NULL(requestor);
  VELOX_CHECK_GT(bytes, 0);
  // Persistent state is governed by the byte-accounted Velox device pool.
  // Do not also react to a physical low-watermark here: transient scan,
  // exchange, and kernel allocations legitimately consume the headroom left
  // above that pool, and treating every such peak as persistent pressure
  // repeatedly externalizes healthy TopN/Join state. Operators reserve their
  // transient kernel workspace separately immediately before restore/compute.
  pool->reportExternalAllocation(static_cast<int64_t>(bytes));
  return DeviceMemoryAdmissionReservation{pool, bytes};
}

std::optional<DeviceMemoryAdmissionReservation> tryAcquireDeviceMemoryAdmission(
    int device,
    std::size_t bytes,
    std::size_t capacityBytes) {
  if (device < 0 || bytes > capacityBytes) {
    return std::nullopt;
  }

  {
    std::lock_guard<std::mutex> lock(deviceMemoryAdmissionMutex);
    const auto it = deviceMemoryAdmissionBytes.find(device);
    const auto reserved =
        it == deviceMemoryAdmissionBytes.end() ? 0 : it->second;
    if (reserved > capacityBytes - bytes) {
      return std::nullopt;
    }
    if (it == deviceMemoryAdmissionBytes.end()) {
      deviceMemoryAdmissionBytes.emplace(device, bytes);
    } else {
      it->second += bytes;
    }
  }
  return DeviceMemoryAdmissionReservation{device, bytes};
}

std::size_t deviceMemoryAdmissionReservedBytes(int device) {
  if (device < 0) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(deviceMemoryAdmissionMutex);
  const auto it = deviceMemoryAdmissionBytes.find(device);
  return it == deviceMemoryAdmissionBytes.end() ? 0 : it->second;
}

std::optional<DeviceMemoryWorkspaceReservation>
tryAcquireBackgroundDeviceMemoryWorkspace(
    std::size_t bytes,
    std::size_t minHeadroomBytes,
    DeviceMemoryWorkspacePriority priority) {
  VELOX_CHECK_GT(bytes, 0);

  const auto headroom = captureDeviceAllocationHeadroom();
  if (!headroom.cudaValid) {
    return std::nullopt;
  }

  std::size_t reserved = 0;
  {
    std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
    const auto reservedIt =
        deviceMemoryWorkspaceBytes.find(headroom.device);
    reserved = reservedIt == deviceMemoryWorkspaceBytes.end()
        ? 0
        : reservedIt->second;
    const auto requiredHeadroom =
        deviceWorkspaceRequiredHeadroom(bytes, minHeadroomBytes, priority);
    const auto availableBytes = headroom.allocatableBytes();
    // A background producer owns no Velox future and therefore must not barge
    // ahead of an already queued request of equal or higher priority. It will
    // be retried by its own progress loop after that request makes progress.
    const auto waiterIt = deviceMemoryWorkspaceWaiters.find(headroom.device);
    const auto now = std::chrono::steady_clock::now();
    const bool hasEqualOrHigherPriorityWaiter =
        waiterIt != deviceMemoryWorkspaceWaiters.end() &&
        std::any_of(
            waiterIt->second.begin(),
            waiterIt->second.end(),
            [&](const auto& waiter) {
              return waiter != nullptr &&
                  (waiter->status ==
                       DeviceMemoryWorkspaceRequestStatus::kQueued ||
                   (waiter->status ==
                        DeviceMemoryWorkspaceRequestStatus::kWakePending &&
                    now - waiter->wakeTime <
                        kDeviceWorkspaceAdvisoryLease)) &&
                  static_cast<uint8_t>(waiter->priority) >=
                  static_cast<uint8_t>(priority);
            });
    if (hasEqualOrHigherPriorityWaiter ||
        availableBytes < requiredHeadroom ||
        reserved > availableBytes - requiredHeadroom) {
      return std::nullopt;
    }
    deviceMemoryWorkspaceBytes[headroom.device] = reserved + bytes;
  }

  VLOG(2) << "CUDF_DEVICE_BACKGROUND_WORKSPACE_ADMITTED device="
          << headroom.device << " bytes=" << bytes
          << " priority=" << static_cast<int>(priority)
          << " previouslyReservedBytes=" << reserved
          << " allocatableBytes=" << headroom.allocatableBytes()
          << " minHeadroomBytes=" << minHeadroomBytes;
  return DeviceMemoryWorkspaceReservation{headroom.device, nullptr, bytes};
}

std::optional<DeviceMemoryWorkspaceReservation>
tryAcquireDeviceMemoryWorkspace(
    memory::MemoryPool* pool,
    exec::Operator* requestor,
    std::size_t bytes,
    std::size_t minHeadroomBytes,
    DeviceMemoryWorkspacePriority priority,
    DeviceMemoryWorkspaceRequest* request,
    ContinueFuture* future) {
  VELOX_CHECK(
      requestor != nullptr || pool == nullptr,
      "A connector workspace without an Operator cannot charge a Velox pool");
  VELOX_CHECK_NOT_NULL(request);
  VELOX_CHECK_NOT_NULL(future);
  VELOX_CHECK_GT(bytes, 0);

  if (request->state_ != nullptr &&
      priority == DeviceMemoryWorkspacePriority::kOutput) {
    const auto now = std::chrono::steady_clock::now();
    auto& state = request->state_;
    if (now - state->enqueueTime >= kDeviceWorkspaceOutputEscapeAge &&
        now >= state->nextOutputEscapeSync) {
      state->nextOutputEscapeSync = now + kDeviceWorkspaceOutputEscapeAge;
      const auto syncStatus = cudaDeviceSynchronize();
      LOG(WARNING) << "CUDF_DEVICE_OUTPUT_WORKSPACE_SYNC device="
                   << state->device << " operator="
                   << (requestor != nullptr ? requestor->operatorType()
                                            : "CudfConnector")
                   << " task="
                   << (requestor != nullptr ? requestor->taskId() : "unknown")
                   << " node="
                   << (requestor != nullptr ? requestor->planNodeId()
                                            : "unknown")
                   << " bytes=" << bytes
                   << " status=" << cudaGetErrorString(syncStatus);
    }
  }

  const auto headroom = captureDeviceAllocationHeadroom();
  if (!headroom.cudaValid) {
    return std::nullopt;
  }

  std::size_t reserved = 0;
  bool admitted = false;
  bool queued = false;
  bool outputEscapeAdmitted = false;
  {
    std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
    if (request->state_ != nullptr) {
      auto& state = request->state_;
      VELOX_CHECK(state->requestor == requestor);
      VELOX_CHECK(state->pool == pool);
      VELOX_CHECK_EQ(state->device, headroom.device);
      VELOX_CHECK_EQ(state->bytes, bytes);
      VELOX_CHECK_EQ(state->minHeadroomBytes, minHeadroomBytes);
      VELOX_CHECK(state->priority == priority);
      VELOX_CHECK(
          state->status == DeviceMemoryWorkspaceRequestStatus::kWakePending,
          "Workspace request retried before receiving its advisory wake");

      const auto reservedIt = deviceMemoryWorkspaceBytes.find(headroom.device);
      reserved = reservedIt == deviceMemoryWorkspaceBytes.end()
          ? 0
          : reservedIt->second;
      const auto now = std::chrono::steady_clock::now();
      const auto waiterIt = deviceMemoryWorkspaceWaiters.find(headroom.device);
      const bool precedingFittingWaiter =
          waiterIt != deviceMemoryWorkspaceWaiters.end() &&
          std::any_of(
              waiterIt->second.begin(),
              waiterIt->second.end(),
              [&](const auto& waiter) {
                return waiter != nullptr && waiter != state &&
                    (waiter->status ==
                         DeviceMemoryWorkspaceRequestStatus::kQueued ||
                     (waiter->status ==
                          DeviceMemoryWorkspaceRequestStatus::kWakePending &&
                      now - waiter->wakeTime <
                          kDeviceWorkspaceAdvisoryLease)) &&
                    waiter->retryNotBefore <= now &&
                    deviceWorkspaceRequestFits(
                        *waiter, headroom, reserved, now) &&
                    deviceWorkspaceRequestPrecedes(*waiter, *state, now);
              });
      const bool stateFits =
          deviceWorkspaceRequestFits(*state, headroom, reserved, now);
      if (stateFits && !precedingFittingWaiter) {
        const auto normalRequiredHeadroom = deviceWorkspaceRequiredHeadroom(
            bytes, minHeadroomBytes, priority);
        const auto normalAvailableBytes = headroom.allocatableBytes();
        const bool normalHeadroomFits =
            normalAvailableBytes >= normalRequiredHeadroom &&
            reserved <= normalAvailableBytes - normalRequiredHeadroom;
        outputEscapeAdmitted =
            priority == DeviceMemoryWorkspacePriority::kOutput &&
            now - state->enqueueTime >= kDeviceWorkspaceOutputEscapeAge &&
            !normalHeadroomFits;
        deviceMemoryWorkspaceBytes[headroom.device] = reserved + bytes;
        state->status = DeviceMemoryWorkspaceRequestStatus::kConsumed;
        request->state_.reset();
        admitted = true;
      } else {
        // Physical CUDA/RMM state can change between scheduler polling and a
        // resumed Velox driver. Requeue in place and suppress this waiter for
        // one poll interval so multiple drivers cannot rotate through
        // wake/requeue at CPU speed while the device is idle.
        constexpr auto kRetryBackoff = std::chrono::milliseconds(5);
        ContinuePromise promise{"retryDeviceMemoryWorkspace"};
        *future = promise.getSemiFuture();
        state->promise.emplace(std::move(promise));
        state->retryNotBefore = now + kRetryBackoff;
        state->status = DeviceMemoryWorkspaceRequestStatus::kQueued;
        queued = true;
      }
    } else {
      const auto reservedIt =
          deviceMemoryWorkspaceBytes.find(headroom.device);
      reserved = reservedIt == deviceMemoryWorkspaceBytes.end()
          ? 0
          : reservedIt->second;
      const auto requiredHeadroom =
          deviceWorkspaceRequiredHeadroom(bytes, minHeadroomBytes, priority);
      const auto availableBytes = headroom.allocatableBytes();
      const auto waiterIt = deviceMemoryWorkspaceWaiters.find(headroom.device);
      // A newly-ready output handoff must not take an avoidable Velox
      // block/wake round trip behind lower-priority input or restore work.
      // It already owns the source bucket; delaying the handoff retains that
      // owner, whereas admitting it releases memory and advances downstream.
      // Do not barge ahead of an equal-priority waiter: FIFO within one phase
      // still bounds starvation across drivers.
      const auto now = std::chrono::steady_clock::now();
      const bool hasEqualOrHigherPriorityWaiter =
          waiterIt != deviceMemoryWorkspaceWaiters.end() &&
          std::any_of(
              waiterIt->second.begin(),
              waiterIt->second.end(),
              [&](const auto& waiter) {
                return waiter != nullptr &&
                    (waiter->status ==
                         DeviceMemoryWorkspaceRequestStatus::kQueued ||
                     (waiter->status ==
                          DeviceMemoryWorkspaceRequestStatus::kWakePending &&
                      now - waiter->wakeTime <
                          kDeviceWorkspaceAdvisoryLease)) &&
                    (waiter->status ==
                         DeviceMemoryWorkspaceRequestStatus::kWakePending ||
                     (waiter->retryNotBefore <= now &&
                      deviceWorkspaceRequestFits(
                          *waiter, headroom, reserved, now))) &&
                    static_cast<uint8_t>(waiter->priority) >=
                    static_cast<uint8_t>(priority);
              });
      if (!hasEqualOrHigherPriorityWaiter &&
          availableBytes >= requiredHeadroom &&
          reserved <= availableBytes - requiredHeadroom) {
        deviceMemoryWorkspaceBytes[headroom.device] = reserved + bytes;
        admitted = true;
      } else {
        ContinuePromise promise{"tryAcquireDeviceMemoryWorkspace"};
        *future = promise.getSemiFuture();
        auto state = std::make_shared<DeviceMemoryWorkspaceRequestState>(
            requestor,
            pool,
            headroom.device,
            bytes,
            minHeadroomBytes,
            priority,
            nextDeviceWorkspaceRequestSequence.fetch_add(
                1, std::memory_order_relaxed),
            std::move(promise));
        request->state_ = state;
        deviceMemoryWorkspaceWaiters[headroom.device].push_back(
            std::move(state));
        queued = true;
      }
    }
  }
  if (queued) {
    // First consume already available headroom. Only the highest-priority
    // waiter may create physical pressure and select spill victims. Without
    // this gate, hundreds of low-priority input waiters can each externalize a
    // large TopN bucket while a partition drain is already waiting.
    scheduleDeviceMemoryWorkspaceWaiters(headroom.device);
    // The immediate check above covers reservation releases. Keep a bounded
    // physical-headroom recheck active for cudaMallocAsync/UCX/output frees,
    // which otherwise have no event wired to this ContinuePromise.
    deviceWorkspaceWaiterPoller.notify();
    if (request->state_ != nullptr &&
        isHighestPriorityDeviceWorkspaceWaiter(request->state_)) {
      const auto requiredHeadroom =
          deviceWorkspaceRequiredHeadroom(bytes, minHeadroomBytes, priority);
      // Include active workspace reservations in the physical reclaim target.
      std::size_t reservedForPressure = 0;
      {
        std::lock_guard<std::mutex> lock(deviceMemoryWorkspaceMutex);
        const auto reservedIt =
            deviceMemoryWorkspaceBytes.find(headroom.device);
        if (reservedIt != deviceMemoryWorkspaceBytes.end()) {
          reservedForPressure = reservedIt->second;
        }
      }
      const auto physicalRequiredHeadroom =
          requiredHeadroom > std::numeric_limits<std::size_t>::max() -
                  reservedForPressure
          ? std::numeric_limits<std::size_t>::max()
          : requiredHeadroom + reservedForPressure;
      if (requestor != nullptr) {
        requestDeviceMemoryReclaimForPhysicalPressure(
            pool,
            requestor,
            physicalRequiredHeadroom,
            priority == DeviceMemoryWorkspacePriority::kInput ||
                priority == DeviceMemoryWorkspacePriority::kTransform);
      }
      scheduleDeviceMemoryWorkspaceWaiters(headroom.device);
    }
  }
  if (admitted) {
    // Workspace is virtual until the operator launches its kernels, but it
    // must consume capacity in the same Velox device pool as persistent
    // TopN/Join state. Otherwise persistent state can immediately regrow into
    // the headroom this reservation just protected. Charging the shared pool
    // turns the reservation into a durable reduction of persistent capacity
    // for its lifetime; the matching RAII free restores that capacity.
    try {
      if (pool != nullptr) {
        exec::Operator::ReclaimableSectionGuard reclaimableSection(requestor);
        pool->reportExternalAllocation(static_cast<int64_t>(bytes));
      }
    } catch (...) {
      releaseDeviceMemoryWorkspace(headroom.device, bytes);
      throw;
    }
    if (outputEscapeAdmitted) {
      LOG(WARNING) << "CUDF_DEVICE_OUTPUT_WORKSPACE_ESCAPE device="
                   << headroom.device << " operator="
                   << (requestor != nullptr ? requestor->operatorType()
                                            : "CudfConnector")
                   << " task="
                   << (requestor != nullptr ? requestor->taskId() : "unknown")
                   << " node="
                   << (requestor != nullptr ? requestor->planNodeId()
                                            : "unknown")
                   << " bytes=" << bytes
                   << " cudaFreeBytes=" << headroom.freeBytes
                   << " waitEscapeMs="
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          kDeviceWorkspaceOutputEscapeAge)
                          .count();
    }
    // Normal per-batch admission is intentionally quiet. At WARNING level a
    // large query would serialize thousands of otherwise cheap admission
    // checks through glog. Blocked admissions and actual arbitration remain
    // visible at WARNING level.
    VLOG(2) << "CUDF_DEVICE_WORKSPACE_ADMITTED device=" << headroom.device
            << " operator="
            << (requestor != nullptr ? requestor->operatorType()
                                     : "CudfConnector")
            << " task="
            << (requestor != nullptr ? requestor->taskId() : "unknown")
            << " node="
            << (requestor != nullptr ? requestor->planNodeId() : "unknown")
            << " bytes=" << bytes
            << " priority=" << static_cast<int>(priority)
            << " previouslyReservedBytes=" << reserved
            << " allocatableBytes=" << headroom.allocatableBytes()
            << " minHeadroomBytes=" << minHeadroomBytes;
    return DeviceMemoryWorkspaceReservation{headroom.device, pool, bytes};
  }
  LOG(WARNING) << "CUDF_DEVICE_WORKSPACE_BLOCKED device=" << headroom.device
               << " operator="
               << (requestor != nullptr ? requestor->operatorType()
                                        : "CudfConnector")
               << " task="
               << (requestor != nullptr ? requestor->taskId() : "unknown")
               << " node="
               << (requestor != nullptr ? requestor->planNodeId() : "unknown")
               << " bytes=" << bytes
               << " priority=" << static_cast<int>(priority)
               << " reservedBytes=" << reserved
               << " allocatableBytes=" << headroom.allocatableBytes()
               << " cudaFreeBytes=" << headroom.freeBytes
               << " reusablePoolBytes=" << headroom.reusablePoolBytes()
               << " minHeadroomBytes=" << minHeadroomBytes;
  return std::nullopt;
}

cudf::detail::cuda_stream_pool& cudfGlobalStreamPool() {
  return cudf::detail::global_cuda_stream_pool();
};

std::mutex& cudfCucoMutex() {
  static std::mutex mutex;
  return mutex;
}

std::mutex& cudfHashPartitionMutex() {
  return cudfCucoMutex();
}

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

bool deviceMemoryAttributionEnabled() {
  static const bool enabled = [] {
    const auto* value =
        std::getenv("GLUTEN_CUDF_DEVICE_MEMORY_ATTRIBUTION");
    if (value == nullptr) {
      return false;
    }
    std::string normalized{value};
    std::transform(
        normalized.begin(),
        normalized.end(),
        normalized.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !normalized.empty() && normalized != "0" &&
        normalized != "false" && normalized != "off" && normalized != "no";
  }();
  return enabled;
}

void logLiveDeviceMemoryAttribution(const std::string& label) {
  if (!deviceMemoryAttributionEnabled()) {
    return;
  }
  if (primaryAttributionResource.has_value()) {
    primaryAttributionResource->get().dumpLive(label, "primary");
  }
  if (outputAttributionResource.has_value()) {
    outputAttributionResource->get().dumpLive(label, "output");
  }
}

void logDeviceMemoryPointerAttribution(
    const std::string& label,
    const void* pointer,
    std::size_t bytes) {
  if (!deviceMemoryAttributionEnabled()) {
    return;
  }
  if (primaryAttributionResource.has_value()) {
    primaryAttributionResource->get().describePointer(
        label, "primary", pointer, bytes);
  }
  if (outputAttributionResource.has_value()) {
    outputAttributionResource->get().describePointer(
        label, "output", pointer, bytes);
  }
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
  allocationContextStack.push_back(currentAllocationContext);
  currentAllocationContext = label;
  using Push = void (*)(const char*);
  static auto push = reinterpret_cast<Push>(
      dlsym(RTLD_DEFAULT, "cuda_alloc_trace_push_context"));
  if (push != nullptr) {
    push(label.c_str());
    active_ = true;
  }
}

CudaAllocationTraceScope::~CudaAllocationTraceScope() {
  if (!allocationContextStack.empty()) {
    currentAllocationContext = std::move(allocationContextStack.back());
    allocationContextStack.pop_back();
  } else {
    currentAllocationContext = "unattributed";
  }
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

CudaCallDiagnosticScope::CudaCallDiagnosticScope(std::string label)
    : label_(std::move(label)) {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_CUDF_CALL_DIAGNOSTICS");
    if (value == nullptr) {
      return false;
    }
    const std::string_view setting{value};
    return !setting.empty() && setting != "0" && setting != "false" &&
        setting != "off";
  }();
  if (!enabled) {
    return;
  }
  const auto* filter =
      std::getenv("GLUTEN_CUDF_CALL_DIAGNOSTICS_FILTER");
  if (filter != nullptr && *filter != '\0') {
    const std::string_view filters{filter};
    bool matched = false;
    size_t begin = 0;
    while (begin <= filters.size()) {
      const auto end = filters.find(',', begin);
      const auto token = filters.substr(
          begin,
          end == std::string_view::npos ? filters.size() - begin
                                        : end - begin);
      if (!token.empty() && label_.find(token) != std::string::npos) {
        matched = true;
        break;
      }
      if (end == std::string_view::npos) {
        break;
      }
      begin = end + 1;
    }
    if (!matched) {
      return;
    }
  }
  static std::atomic<uint64_t> nextCallId{1};
  callId_ = nextCallId.fetch_add(1, std::memory_order_relaxed);
  active_ = true;
  LOG(WARNING) << "CUDF_CALL_DIAGNOSTIC phase=BEGIN call=" << callId_
               << " label={" << label_ << "}";
}

CudaCallDiagnosticScope::~CudaCallDiagnosticScope() {
  if (active_) {
    LOG(WARNING) << "CUDF_CALL_DIAGNOSTIC phase=END call=" << callId_
                 << " label={" << label_ << "}";
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
