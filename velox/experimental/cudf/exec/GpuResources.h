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

#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::memory {
class MemoryPool;
}

namespace facebook::velox::exec {
class Operator;
}

namespace facebook::velox::cudf_velox {

/// Custom Velox memory-resource tag used for reclaimable persistent GPU state.
/// Transient kernel workspaces are intentionally not charged to this pool.
inline constexpr std::string_view kCudfDeviceMemoryResourceTag{"cudf_device"};

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

enum class DeviceMemoryWorkspacePriority : uint8_t;
class DeviceMemoryWorkspaceRequest;

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
  friend DeviceMemoryAdmissionReservation acquireDeviceMemoryAdmission(
      memory::MemoryPool* pool,
      std::size_t bytes,
      exec::Operator* requestor);

  DeviceMemoryAdmissionReservation(int device, std::size_t bytes) noexcept;
  DeviceMemoryAdmissionReservation(
      memory::MemoryPool* pool,
      std::size_t bytes) noexcept;

  int device_{-1};
  memory::MemoryPool* pool_{nullptr};
  std::size_t bytes_{0};
  bool active_{false};
};

/// A byte-accounted reservation for transient GPU operator workspace. Unlike
/// DeviceMemoryAdmissionReservation, this does not represent persistent state,
/// but it is charged to the same Velox device pool for its lifetime. This
/// reduces the capacity available to persistent state and prevents concurrent
/// drivers from all admitting work against the same cudaMemGetInfo snapshot.
class DeviceMemoryWorkspaceReservation {
 public:
  DeviceMemoryWorkspaceReservation() = default;
  ~DeviceMemoryWorkspaceReservation();

  DeviceMemoryWorkspaceReservation(
      DeviceMemoryWorkspaceReservation&& other) noexcept;
  DeviceMemoryWorkspaceReservation& operator=(
      DeviceMemoryWorkspaceReservation&& other) noexcept;

  DeviceMemoryWorkspaceReservation(const DeviceMemoryWorkspaceReservation&) =
      delete;
  DeviceMemoryWorkspaceReservation& operator=(
      const DeviceMemoryWorkspaceReservation&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return active_;
  }

  [[nodiscard]] std::size_t reservedBytes() const noexcept {
    return bytes_;
  }

  void release() noexcept;

 private:
  friend std::optional<DeviceMemoryWorkspaceReservation>
  tryAcquireDeviceMemoryWorkspace(
      memory::MemoryPool*,
      exec::Operator*,
      std::size_t,
      std::size_t,
      DeviceMemoryWorkspacePriority,
      DeviceMemoryWorkspaceRequest*,
      ContinueFuture*);
  friend std::optional<DeviceMemoryWorkspaceReservation>
  tryAcquireBackgroundDeviceMemoryWorkspace(
      std::size_t,
      std::size_t,
      DeviceMemoryWorkspacePriority);

  DeviceMemoryWorkspaceReservation(
      int device,
      memory::MemoryPool* pool,
      std::size_t bytes) noexcept;

  int device_{-1};
  memory::MemoryPool* pool_{nullptr};
  std::size_t bytes_{0};
  bool active_{false};
};

/// Priority of a transient device-workspace request. These priorities schedule
/// consumers of device memory; they are deliberately separate from Velox
/// MemoryReclaimer priorities, which schedule spill victims.
enum class DeviceMemoryWorkspacePriority : uint8_t {
  /// Streaming input work. Delaying one batch does not prevent an already
  /// complete partition from releasing its external state.
  kInput = 0,
  /// Stateless input replacement. Completing this work releases the source
  /// batch (including an Exchange receive page) instead of growing persistent
  /// operator state, so it must make progress ahead of ordinary ingestion.
  /// It remains below restore/drain work, which releases whole partitions.
  kTransform = 50,
  /// Restores state needed to make progress, but is not yet the terminal drain
  /// of a blocking operator.
  kRestore = 100,
  /// Restores and drains a complete partition. Finishing this work releases
  /// persistent state and unblocks downstream pipelines.
  kDrain = 200,
  /// Hands an already-computed result to the downstream pipeline. This work
  /// must precede another restore: completing it releases the current source
  /// bucket, while starting another restore would create an additional live
  /// device owner and increase pressure without advancing the pipeline.
  kOutput = 250,
};

struct DeviceMemoryWorkspaceRequestState;

/// Cancellation-safe handle for one queued device-workspace request. An
/// operator keeps this object while blocked. Destroying or resetting it removes
/// the waiter. Scheduler wakes are advisory; the owning driver atomically
/// revalidates physical headroom, fitting higher-priority waiters, and active
/// virtual reservations when it retries admission.
class DeviceMemoryWorkspaceRequest {
 public:
  DeviceMemoryWorkspaceRequest() = default;
  ~DeviceMemoryWorkspaceRequest();

  DeviceMemoryWorkspaceRequest(DeviceMemoryWorkspaceRequest&&) noexcept;
  DeviceMemoryWorkspaceRequest& operator=(
      DeviceMemoryWorkspaceRequest&&) noexcept;

  DeviceMemoryWorkspaceRequest(const DeviceMemoryWorkspaceRequest&) = delete;
  DeviceMemoryWorkspaceRequest& operator=(const DeviceMemoryWorkspaceRequest&) =
      delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return state_ != nullptr;
  }

  void reset() noexcept;

 private:
  friend std::optional<DeviceMemoryWorkspaceReservation>
  tryAcquireDeviceMemoryWorkspace(
      memory::MemoryPool*,
      exec::Operator*,
      std::size_t,
      std::size_t,
      DeviceMemoryWorkspacePriority,
      DeviceMemoryWorkspaceRequest*,
      ContinueFuture*);

  std::shared_ptr<DeviceMemoryWorkspaceRequestState> state_;
};

/// Reusable Driver-facing state for a replayable GPU phase. It owns the
/// cancellation-safe scheduler request, advisory future, and wait timing as a
/// single lifecycle. Operators keep their input/spill state intact, return
/// kWaitForArbitration while this object is waiting, and retry only the
/// idempotent phase after the future wakes.
class ReplayableDeviceMemoryWorkspace {
 public:
  struct Attempt {
    std::optional<DeviceMemoryWorkspaceReservation> reservation;
    bool firstWait{false};
    std::optional<uint64_t> completedWaitMicros;
  };

  ReplayableDeviceMemoryWorkspace() = default;
  ~ReplayableDeviceMemoryWorkspace() = default;

  ReplayableDeviceMemoryWorkspace(const ReplayableDeviceMemoryWorkspace&) =
      delete;
  ReplayableDeviceMemoryWorkspace& operator=(
      const ReplayableDeviceMemoryWorkspace&) = delete;

  [[nodiscard]] Attempt tryAcquire(
      memory::MemoryPool* pool,
      exec::Operator* requestor,
      std::size_t bytes,
      std::size_t minHeadroomBytes,
      DeviceMemoryWorkspacePriority priority);

  /// Moves an installed advisory future to the Driver. Returns false when no
  /// workspace request is waiting.
  [[nodiscard]] bool takeFuture(ContinueFuture* future);

  [[nodiscard]] bool waiting() const noexcept {
    return waiting_;
  }

  void reset() noexcept;

  /// Installs an already-ready advisory used to exercise Driver replay without
  /// manufacturing device pressure in an operator test.
  void deferReadyForTesting();

 private:
  DeviceMemoryWorkspaceRequest request_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
  bool waiting_{false};
  std::optional<std::chrono::steady_clock::time_point> waitStart_;
};

struct DeviceMemoryReclaimerState;

/// RAII registration of one cuDF operator with the process-wide device-memory
/// arbitrator. The arbitrator never calls the operator from another thread.
/// It only records a reclaim request; the owning driver services that request
/// at its next operator boundary.
class DeviceMemoryReclaimerRegistration {
 public:
  DeviceMemoryReclaimerRegistration() = default;
  ~DeviceMemoryReclaimerRegistration() = default;

  DeviceMemoryReclaimerRegistration(
      DeviceMemoryReclaimerRegistration&&) noexcept = default;
  DeviceMemoryReclaimerRegistration& operator=(
      DeviceMemoryReclaimerRegistration&&) noexcept = default;

  DeviceMemoryReclaimerRegistration(const DeviceMemoryReclaimerRegistration&) =
      delete;
  DeviceMemoryReclaimerRegistration& operator=(
      const DeviceMemoryReclaimerRegistration&) = delete;

  [[nodiscard]] explicit operator bool() const noexcept {
    return state_ != nullptr;
  }

  void reset() noexcept {
    state_.reset();
  }

 private:
  friend DeviceMemoryReclaimerRegistration registerDeviceMemoryReclaimer(
      exec::Operator*,
      memory::MemoryPool*);
  friend void serviceDeviceMemoryReclaimer(
      DeviceMemoryReclaimerRegistration&,
      exec::Operator*,
      std::string_view,
      bool);

  explicit DeviceMemoryReclaimerRegistration(
      std::shared_ptr<DeviceMemoryReclaimerState> state)
      : state_{std::move(state)} {}

  std::shared_ptr<DeviceMemoryReclaimerState> state_;
};

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

/// Adds RMM statistics and allocation-time operator attribution to a resource.
/// This is only used by the optional device-memory diagnostic path.
[[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
wrapDeviceMemoryResourceForDiagnostics(
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    bool outputResource);

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

/// Charges reclaimable persistent device state to a Velox custom memory-pool.
/// Unlike tryAcquireDeviceMemoryAdmission(), this can invoke Velox's shared
/// memory arbitrator and operator reclaimers before returning. Physical device
/// headroom is deliberately not enforced here: transient workspace is admitted
/// separately at the restore/compute boundary so normal scan and exchange
/// peaks do not churn persistent state. The returned RAII reservation reports
/// the matching external free on destruction.
[[nodiscard]] DeviceMemoryAdmissionReservation acquireDeviceMemoryAdmission(
    memory::MemoryPool* pool,
    std::size_t bytes,
    exec::Operator* requestor);

/// Registers a reclaimable cuDF operator with the per-process/per-device
/// cooperative arbitrator. Registration is cheap and may be used by operators
/// that currently report zero reclaimable bytes.
[[nodiscard]] DeviceMemoryReclaimerRegistration registerDeviceMemoryReclaimer(
    exec::Operator* owner,
    memory::MemoryPool* pool);

/// Reports the owner's current reclaimable bytes and services any pending
/// global pressure request on the owner's driver thread. Call only at an
/// operator boundary where its persistent state is internally consistent.
void serviceDeviceMemoryReclaimer(
    DeviceMemoryReclaimerRegistration& registration,
    exec::Operator* owner,
    std::string_view safePoint,
    bool observePhysicalPressure = false);

/// Returns bytes currently reserved by cooperative admission clients on a
/// device. This is admission accounting, not live RMM allocation usage.
[[nodiscard]] std::size_t deviceMemoryAdmissionReservedBytes(int device);

/// Attempts to reserve transient device workspace on the current CUDA device.
/// The admission decision is serialized per process and accounts for both
/// physical allocatable bytes and other not-yet-allocated workspace
/// reservations. If admission is unavailable, returns nullopt and installs a
/// future which is fulfilled when device memory or another workspace
/// reservation makes progress. 'requestor' may be null only when 'pool' is
/// also null, for a cuDF connector operation driven by a generic Velox
/// operator (for example, TableScan's DataSource::next()).
[[nodiscard]] std::optional<DeviceMemoryWorkspaceReservation>
tryAcquireDeviceMemoryWorkspace(
    memory::MemoryPool* pool,
    exec::Operator* requestor,
    std::size_t bytes,
    std::size_t minHeadroomBytes,
    DeviceMemoryWorkspacePriority priority,
    DeviceMemoryWorkspaceRequest* request,
    ContinueFuture* future);

/// Attempts an immediate transient-workspace reservation for a background GPU
/// producer that is not driven by a Velox Operator (for example, a UCX
/// communicator thread). The allocation must be submitted and made visible to
/// CUDA while the returned reservation is held. Callers retry later when this
/// returns nullopt; this API deliberately does not install an Operator future.
[[nodiscard]] std::optional<DeviceMemoryWorkspaceReservation>
tryAcquireBackgroundDeviceMemoryWorkspace(
    std::size_t bytes,
    std::size_t minHeadroomBytes,
    DeviceMemoryWorkspacePriority priority);

/**
 * @brief Returns the global CUDA stream pool used by cudf.
 */
[[nodiscard]] cudf::detail::cuda_stream_pool& cudfGlobalStreamPool();

/// Serializes cudf::hash_partition calls within an executor process. Callers
/// must hold this mutex until the operation's CUDA stream is synchronized.
[[nodiscard]] std::mutex& cudfHashPartitionMutex();

/// Serializes cuco-backed hash/groupby kernels within an executor process.
/// Callers hold this only across kernel submission and completion on the
/// operation's own stream; unrelated cuDF operators remain concurrent.
[[nodiscard]] std::mutex& cudfCucoMutex();

/// Enables low-overhead RMM statistics and operator-level CUDA memory
/// diagnostics when GLUTEN_CUDF_DEVICE_MEMORY_DIAGNOSTICS is set to a true
/// value in the executor environment.
[[nodiscard]] bool deviceMemoryDiagnosticsEnabled();

/// Enables allocation ownership tracking without enabling periodic
/// CUDF_DEVICE_MEMORY logging.  On allocation failure the resource prints the
/// largest live allocation contexts.  This is useful for long-running
/// workloads where diagnostic logging materially perturbs scheduling.
[[nodiscard]] bool deviceMemoryAttributionEnabled();

/// Captures both CUDA device-wide usage and allocations made through the
/// cuDF RMM resource. Unlike cudaMemGetInfo-only diagnostics, the snapshot
/// always includes the current CUDA device id.
[[nodiscard]] DeviceMemorySnapshot captureDeviceMemorySnapshot();

/// Emits a structured CUDF_DEVICE_MEMORY log record for later correlation
/// with the operator node and method that triggered the sample.
void logDeviceMemorySnapshot(
    const std::string& label,
    const DeviceMemorySnapshot& snapshot);

/// Logs the largest currently-live RMM allocations grouped by the operator
/// context in which they were created. This is a no-op unless
/// GLUTEN_CUDF_DEVICE_MEMORY_ATTRIBUTION is enabled.
void logLiveDeviceMemoryAttribution(const std::string& label);

/// Describes the RMM allocation that owns or most recently owned the supplied
/// device address range. Intended for CUDA/UCX error paths where the failing
/// pointer is more useful than aggregate operator totals.
void logDeviceMemoryPointerAttribution(
    const std::string& label,
    const void* pointer,
    std::size_t bytes);

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

/// Emits a BEGIN/END pair for a narrowly filtered CUDA/cuDF call. This is a
/// diagnostic-only crash locator: a fatal signal leaves the final BEGIN
/// unmatched, preserving the exact native phase that entered the driver.
/// Enabled by GLUTEN_CUDF_CALL_DIAGNOSTICS and optionally filtered by the
/// comma-separated GLUTEN_CUDF_CALL_DIAGNOSTICS_FILTER.
class CudaCallDiagnosticScope {
 public:
  explicit CudaCallDiagnosticScope(std::string label);
  ~CudaCallDiagnosticScope();

  CudaCallDiagnosticScope(const CudaCallDiagnosticScope&) = delete;
  CudaCallDiagnosticScope& operator=(const CudaCallDiagnosticScope&) = delete;

 private:
  std::string label_;
  uint64_t callId_{0};
  bool active_{false};
};

} // namespace facebook::velox::cudf_velox
