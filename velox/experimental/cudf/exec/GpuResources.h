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

#include "velox/common/future/VeloxPromise.h"

#include <cudf/detail/utilities/stream_pool.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox {

struct DeviceMemorySnapshot {
  bool enabled{false};
  bool cudaValid{false};
  int device{-1};
  std::size_t freeBytes{0};
  std::size_t totalBytes{0};
  std::size_t usedBytes{0};
  int64_t rmmCurrentBytes{0};
  int64_t rmmPeakBytes{0};
  int64_t rmmTotalBytes{0};
  int64_t rmmCurrentAllocations{0};
  int64_t rmmPeakAllocations{0};
  int64_t rmmTotalAllocations{0};
};

/// A cheap, always-available view of memory that allocations through the
/// current device's primary cuDF resource may be able to use. When the primary
/// resource is not cudaMallocAsync, or its pool attributes cannot be read,
/// allocatableBytes() conservatively reports cudaMemGetInfo's free bytes only.
struct DeviceAllocationHeadroom {
  bool cudaValid{false};
  bool asyncPoolValid{false};
  int device{-1};
  std::size_t freeBytes{0};
  std::size_t totalBytes{0};
  std::size_t poolReservedBytes{0};
  std::size_t poolUsedBytes{0};

  [[nodiscard]] std::size_t reusablePoolBytes() const noexcept;
  [[nodiscard]] std::size_t allocatableBytes() const noexcept;
};

/// Process-wide, per-device admission reservation. This does not allocate GPU
/// memory; it prevents cooperating operators from admitting work against the
/// same headroom snapshot. The reservation is released on destruction.
class DeviceMemoryAdmissionReservation {
 public:
  DeviceMemoryAdmissionReservation() = default;
  ~DeviceMemoryAdmissionReservation();

  DeviceMemoryAdmissionReservation(
      DeviceMemoryAdmissionReservation&& other) noexcept;
  DeviceMemoryAdmissionReservation& operator=(
      DeviceMemoryAdmissionReservation&& other) noexcept;

  DeviceMemoryAdmissionReservation(const DeviceMemoryAdmissionReservation&) =
      delete;
  DeviceMemoryAdmissionReservation& operator=(
      const DeviceMemoryAdmissionReservation&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return active_;
  }

  [[nodiscard]] int device() const noexcept {
    return device_;
  }

  [[nodiscard]] std::size_t reservedBytes() const noexcept {
    return bytes_;
  }

  void release() noexcept;

 private:
  friend std::optional<DeviceMemoryAdmissionReservation>
  tryAcquireDeviceMemoryAdmission(
      int device,
      std::size_t bytes,
      std::size_t capacityBytes);
  friend std::optional<DeviceMemoryAdmissionReservation>
  waitAcquireDeviceMemoryAdmission(
      int device,
      std::size_t bytes,
      std::size_t capacityBytes);
  friend std::optional<DeviceMemoryAdmissionReservation>
  acquireDeviceMemoryAdmissionOrFuture(
      int device,
      std::size_t bytes,
      std::size_t capacityBytes,
      ContinueFuture* future);

  DeviceMemoryAdmissionReservation(int device, std::size_t bytes) noexcept;

  int device_{-1};
  std::size_t bytes_{0};
  bool active_{false};
};

enum class DeviceConcurrencyAdmissionDomain : uint8_t {
  kTransientKernel,
  kRetainedState,
};

/// Process-wide, per-device concurrency reservation. Independent domains let
/// operators bound transient kernels and long-lived retained state separately.
class DeviceConcurrencyAdmissionReservation {
 public:
  DeviceConcurrencyAdmissionReservation() = default;
  ~DeviceConcurrencyAdmissionReservation();

  DeviceConcurrencyAdmissionReservation(
      DeviceConcurrencyAdmissionReservation&& other) noexcept;
  DeviceConcurrencyAdmissionReservation& operator=(
      DeviceConcurrencyAdmissionReservation&& other) noexcept;

  DeviceConcurrencyAdmissionReservation(
      const DeviceConcurrencyAdmissionReservation&) = delete;
  DeviceConcurrencyAdmissionReservation& operator=(
      const DeviceConcurrencyAdmissionReservation&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return active_;
  }

  void release() noexcept;

 private:
  friend std::optional<DeviceConcurrencyAdmissionReservation>
  acquireDeviceConcurrencyAdmissionOrFuture(
      int device,
      std::size_t maxConcurrent,
      ContinueFuture* future,
      DeviceConcurrencyAdmissionDomain domain,
      uint64_t scope);

  explicit DeviceConcurrencyAdmissionReservation(
      int device,
      DeviceConcurrencyAdmissionDomain domain,
      uint64_t scope) noexcept;

  int device_{-1};
  DeviceConcurrencyAdmissionDomain domain_{
      DeviceConcurrencyAdmissionDomain::kTransientKernel};
  uint64_t scope_{0};
  bool active_{false};
};

/// Registers admission bytes that are already reserved for a logical task.
/// Downstream operators can reuse this credit even when intermediate
/// operators replace the input vector.
class DeviceMemoryAdmissionCreditRegistration {
 public:
  DeviceMemoryAdmissionCreditRegistration() = default;
  ~DeviceMemoryAdmissionCreditRegistration();

  DeviceMemoryAdmissionCreditRegistration(
      DeviceMemoryAdmissionCreditRegistration&& other) noexcept;
  DeviceMemoryAdmissionCreditRegistration& operator=(
      DeviceMemoryAdmissionCreditRegistration&& other) noexcept;

  DeviceMemoryAdmissionCreditRegistration(
      const DeviceMemoryAdmissionCreditRegistration&) = delete;
  DeviceMemoryAdmissionCreditRegistration& operator=(
      const DeviceMemoryAdmissionCreditRegistration&) = delete;

  void release() noexcept;

 private:
  friend DeviceMemoryAdmissionCreditRegistration
  registerScopedDeviceMemoryAdmissionCredit(
      std::string scope,
      int device,
      std::size_t bytes,
      std::shared_ptr<void> owner,
      std::function<void()> onConsumed);

  DeviceMemoryAdmissionCreditRegistration(
      std::string scope,
      uint64_t registrationId) noexcept;

  std::string scope_;
  uint64_t registrationId_{0};
};

/// Admission credit attached to a device vector and propagated with its
/// lineage. A consumer may use the credit once and release its owner after
/// preceding work on the consumer stream completes.
class DeviceMemoryAdmissionCredit {
 public:
  DeviceMemoryAdmissionCredit(
      int device,
      std::size_t bytes,
      std::shared_ptr<void> owner,
      std::function<void()> onConsumed);

  int device() const {
    return device_;
  }

  std::size_t availableBytes() const {
    return consumed_.load(std::memory_order_acquire) ? 0 : bytes_;
  }

 private:
  friend std::size_t consumeDeviceMemoryAdmissionCreditAfterStream(
      const std::shared_ptr<DeviceMemoryAdmissionCredit>& credit,
      rmm::cuda_stream_view stream);

  const int device_;
  const std::size_t bytes_;
  std::shared_ptr<void> owner_;
  std::function<void()> onConsumed_;
  std::atomic<bool> consumed_{false};
};

using DeviceMemoryAdmissionCreditPtr =
    std::shared_ptr<DeviceMemoryAdmissionCredit>;

extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
extern std::optional<cuda::mr::any_resource<cuda::mr::device_accessible>>
    output_mr_;
extern std::optional<rmm::mr::statistics_resource_adaptor> statistics_mr_;
extern std::optional<rmm::mr::statistics_resource_adaptor>
    output_statistics_mr_;

/// Returns the memory resource designated for output vector allocations.
rmm::device_async_resource_ref get_output_mr();

/**
 * @brief Creates a memory resource based on the given mode.
 *
 * @param mode rmm::mr::pool_memory_resource mode.
 * @param percent The initial percent of GPU memory to allocate for pool or
 * arena resources, or the retained-memory release threshold for async.
 */
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
createMemoryResource(std::string_view mode, int percent);

/// If GLUTEN_CUDF_ASYNC_QUERY_END_TRIM_BYTES is set, synchronizes the device
/// and releases unused cudaMallocAsync pool memory down to that retained-byte
/// target. Returns true when a trim was requested and completed successfully.
[[nodiscard]] bool trimAsyncMemoryPoolsAtQueryEnd();

/// Explicit query-scoped variant. Unlike the no-argument environment
/// fallback, a zero value is a valid request to release every unused block.
[[nodiscard]] bool trimAsyncMemoryPoolsAtQueryEnd(std::size_t bytesToKeep);

/// Drops native pool handles after their owning RMM resources are destroyed.
void clearAsyncMemoryPoolHandles();

/// Captures current-device free memory and, when the primary cuDF resource is
/// cudaMallocAsync, that pool's currently reusable reserved memory. This API is
/// independent of diagnostic logging and returns a conservative partial
/// snapshot when pool attributes are unavailable.
[[nodiscard]] DeviceAllocationHeadroom captureDeviceAllocationHeadroom();

/// Atomically acquires a cooperative admission reservation for a device when
/// existing reservations plus 'bytes' do not exceed 'capacityBytes'. Returns
/// nullopt for invalid devices or insufficient capacity.
[[nodiscard]] std::optional<DeviceMemoryAdmissionReservation>
tryAcquireDeviceMemoryAdmission(
    int device,
    std::size_t bytes,
    std::size_t capacityBytes);

/// Waits until a cooperative reservation fits within 'capacityBytes'. Returns
/// nullopt only when the request itself is invalid or exceeds the capacity.
[[nodiscard]] std::optional<DeviceMemoryAdmissionReservation>
waitAcquireDeviceMemoryAdmission(
    int device,
    std::size_t bytes,
    std::size_t capacityBytes);

/// Atomically acquires a cooperative reservation or registers a wake-up future
/// that becomes ready after an existing reservation is released. Callers must
/// retry after the future completes; no executor thread is blocked while
/// waiting.
[[nodiscard]] std::optional<DeviceMemoryAdmissionReservation>
acquireDeviceMemoryAdmissionOrFuture(
    int device,
    std::size_t bytes,
    std::size_t capacityBytes,
    ContinueFuture* future);

/// Keeps a reservation alive until all work already submitted to 'stream' has
/// completed. This is required for asynchronous cuDF calls whose temporary
/// allocations outlive their host function scope.
void releaseDeviceMemoryAdmissionAfterStream(
    DeviceMemoryAdmissionReservation reservation,
    rmm::cuda_stream_view stream);

/// Returns bytes currently reserved by cooperative admission clients on a
/// device. This is admission accounting, not live RMM allocation usage.
[[nodiscard]] std::size_t deviceMemoryAdmissionReservedBytes(int device);

/// Acquires one independent GPU-concurrency slot or registers a wake-up
/// future. This does not consume device-memory ownership credit and therefore
/// cannot form a credit cycle with exchange.
[[nodiscard]] std::optional<DeviceConcurrencyAdmissionReservation>
acquireDeviceConcurrencyAdmissionOrFuture(
    int device,
    std::size_t maxConcurrent,
    ContinueFuture* future,
    DeviceConcurrencyAdmissionDomain domain =
        DeviceConcurrencyAdmissionDomain::kTransientKernel,
    uint64_t scope = 0);

/// Releases a concurrency slot after preceding work on 'stream' completes.
void releaseDeviceConcurrencyAdmissionAfterStream(
    DeviceConcurrencyAdmissionReservation reservation,
    rmm::cuda_stream_view stream);

/// Publishes an existing reservation as reusable credit within 'scope'. The
/// owner must keep the underlying reservation alive for the registration's
/// lifetime.
[[nodiscard]] DeviceMemoryAdmissionCreditRegistration
registerScopedDeviceMemoryAdmissionCredit(
    std::string scope,
    int device,
    std::size_t bytes,
    std::shared_ptr<void> owner,
    std::function<void()> onConsumed = {});

/// Returns live admission credit published for 'scope' on 'device'.
[[nodiscard]] std::size_t scopedDeviceMemoryAdmissionCreditBytes(
    std::string_view scope,
    int device);

/// Consumes enough live credit to cover 'bytes'. Credit ownership and its
/// optional callback remain alive until all preceding work on 'stream'
/// completes.
[[nodiscard]] std::size_t consumeScopedDeviceMemoryAdmissionCreditAfterStream(
    std::string_view scope,
    int device,
    std::size_t bytes,
    rmm::cuda_stream_view stream);

[[nodiscard]] std::size_t consumeDeviceMemoryAdmissionCreditAfterStream(
    const DeviceMemoryAdmissionCreditPtr& credit,
    rmm::cuda_stream_view stream);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

/// Enables low-overhead RMM statistics and operator-level CUDA memory
/// diagnostics when GLUTEN_CUDF_DEVICE_MEMORY_DIAGNOSTICS is set to a true
/// value in the executor environment.
[[nodiscard]] bool deviceMemoryDiagnosticsEnabled();

/// Captures both CUDA device-wide usage and allocations made through the
/// cuDF RMM resource. Unlike cudaMemGetInfo-only diagnostics, the snapshot
/// always includes the current CUDA device id.
[[nodiscard]] DeviceMemorySnapshot captureDeviceMemorySnapshot();

/// Emits a structured CUDF_DEVICE_MEMORY log record for later correlation
/// with the operator node and method that triggered the sample.
void logDeviceMemorySnapshot(
    const std::string& label,
    const DeviceMemorySnapshot& snapshot);

inline void logDeviceMemorySnapshot(const std::string& label) {
  if (deviceMemoryDiagnosticsEnabled()) {
    logDeviceMemorySnapshot(label, captureDeviceMemorySnapshot());
  }
}

/// Associates CUDA allocations on the current thread with a native operator
/// when the optional LD_PRELOAD diagnostic tracer is present.
class CudaAllocationTraceScope {
 public:
  explicit CudaAllocationTraceScope(const std::string& label);
  ~CudaAllocationTraceScope();

  CudaAllocationTraceScope(const CudaAllocationTraceScope&) = delete;
  CudaAllocationTraceScope& operator=(const CudaAllocationTraceScope&) = delete;

 private:
  bool active_{false};
};

} // namespace facebook::velox::cudf_velox
