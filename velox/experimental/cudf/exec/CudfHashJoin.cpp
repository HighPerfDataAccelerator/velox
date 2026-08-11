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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDeletionHelpers.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfPackedRestore.h"
#include "velox/experimental/cudf/exec/CudfPackedSpill.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/testutil/TestValue.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/Task.h" // NOLINT(misc-unused-headers)
#include "velox/expression/ExprOptimizer.h"
#include "velox/type/TypeUtil.h"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <nvtx3/nvtx3.hpp>

#include <fcntl.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <linux/falloc.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>

namespace facebook::velox::cudf_velox {
namespace {

cudf::device_span<cudf::size_type const> toSpan(
    const rmm::device_uvector<cudf::size_type>& indices) {
  return {indices.data(), indices.size()};
}

uint64_t estimateColumnViewBytes(cudf::column_view column) {
  uint64_t bytes = 0;
  if (cudf::is_fixed_width(column.type())) {
    bytes +=
        static_cast<uint64_t>(column.size()) * cudf::size_of(column.type());
  }
  if (column.nullable()) {
    bytes += cudf::bitmask_allocation_size_bytes(column.size());
  }
  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    bytes += estimateColumnViewBytes(column.child(i));
  }
  return bytes;
}

uint64_t retainedCudfBytes(const CudfVector& vector) {
  const auto ownedBytes = vector.estimateFlatSize();
  if (ownedBytes != 0 || vector.size() == 0) {
    return ownedBytes;
  }
  uint64_t viewBytes = 0;
  const auto table = vector.getTableView();
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    viewBytes += estimateColumnViewBytes(table.column(i));
  }
  return viewBytes;
}

bool graceBulkBuildRestoreEnabled() {
  const auto* value = std::getenv("GLUTEN_CUDF_HASH_JOIN_BULK_BUILD_RESTORE");
  if (value == nullptr) {
    return false;
  }
  return std::string_view(value) == "1" || std::string_view(value) == "true" ||
      std::string_view(value) == "TRUE";
}

bool graceBulkProbeRestoreEnabled() {
  const auto* value = std::getenv("GLUTEN_CUDF_HASH_JOIN_BULK_PROBE_RESTORE");
  if (value == nullptr) {
    return false;
  }
  return std::string_view(value) == "1" || std::string_view(value) == "true" ||
      std::string_view(value) == "TRUE";
}

bool graceAsyncBuildDemoteEnabled() {
  const auto* value = std::getenv("GLUTEN_CUDF_HASH_JOIN_ASYNC_BUILD_DEMOTE");
  if (value == nullptr) {
    return false;
  }
  return std::string_view(value) == "1" || std::string_view(value) == "true" ||
      std::string_view(value) == "TRUE";
}

bool gracePageableRestoreBounceEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_HASH_JOIN_PAGEABLE_RESTORE_BOUNCE");
  if (value == nullptr) {
    return false;
  }
  return std::string_view(value) == "1" || std::string_view(value) == "true" ||
      std::string_view(value) == "TRUE";
}

bool replayableResidentBuildFinalizeEnabled() {
  const auto* value = std::getenv("GLUTEN_CUDF_HASH_JOIN_REPLAYABLE_FINALIZE");
  if (value == nullptr) {
    return true;
  }
  return std::string_view(value) != "0" && std::string_view(value) != "false" &&
      std::string_view(value) != "FALSE";
}

size_t graceBuildDemoteThreads() {
  static const size_t threads = [] {
    const auto* value = std::getenv("CUDF_HASH_JOIN_GRACE_HOST_DEMOTE_THREADS");
    if (value == nullptr) {
      return size_t{2};
    }
    char* end = nullptr;
    const auto requested = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
      return size_t{2};
    }
    return static_cast<size_t>(std::clamp<uint64_t>(requested, 1, 8));
  }();
  return threads;
}

folly::CPUThreadPoolExecutor& graceBuildDemoteExecutor() {
  static folly::CPUThreadPoolExecutor executor(graceBuildDemoteThreads());
  return executor;
}

// Keep the transient hash_partition output and workspace bounded.  In
// particular, an exchange may deliver a single multi-GiB CudfVector even
// after the join has switched to Grace mode.  Slicing by row creates views of
// that vector (not copies), so each slice can be partitioned and packed to
// host before the next slice is submitted.
constexpr uint64_t kGracePartitionBatchBytes = 256ULL << 20;

// Build and probe have already crossed an MPP hash exchange using libcudf's
// default seed. Reusing it for Grace fanout leaves the exchange-selected low
// bits fixed and collapses the effective local partition count. Build and
// probe use this independent seed together, preserving join correctness while
// spreading all local partitions.
constexpr uint32_t kGraceLocalHashSeed = cudf::DEFAULT_HASH_SEED ^ 0x85ebca6bU;

std::atomic<uint64_t> graceHashJoinSpillFileId{0};

std::shared_ptr<void> tryReserveGraceHostMemory(
    uint64_t bytes,
    uint64_t limitBytes) {
  return tryReserveCudfPackedHostMemory(bytes, limitBytes);
}

uint64_t currentGraceHostMemoryReservedBytes() {
  return currentCudfPackedHostMemoryReservedBytes();
}

std::shared_ptr<cudf::hash_join> makeSharedHashJoin(
    cudf::table_view buildKeys,
    cudf::null_equality compareNulls,
    double loadFactor,
    rmm::cuda_stream_view stream) {
  // A hash_join is shared by the bridge and every probe driver.  Its last
  // reference can therefore be released by a probe, the bridge, or task
  // cleanup.  Keep final destruction in the same submission domain as cuco
  // construction/probe calls; otherwise one fragment can tear down cuco
  // storage while another fragment is submitting work.
  return std::shared_ptr<cudf::hash_join>(
      new cudf::hash_join(
          buildKeys,
          cudf::nullable_join::YES,
          compareNulls,
          loadFactor,
          stream),
      [](cudf::hash_join* hash) {
        std::lock_guard<std::mutex> cucoLock(cudfCucoMutex());
        delete hash;
      });
}

HashJoinHostBatch packHashJoinTable(
    cudf::table_view table,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto packed = cudf::pack(table, stream, mr);
  HashJoinHostBatch host;
  host.dataBytes = packed.gpu_data->size();
  host.rows = static_cast<vector_size_t>(table.num_rows());
  host.data = std::shared_ptr<uint8_t>(
      new uint8_t[host.dataBytes], std::default_delete<uint8_t[]>());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      host.data.get(),
      packed.gpu_data->data(),
      host.dataBytes,
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  host.metadata = std::move(packed.metadata);
  return host;
}

struct HashJoinPackTiming {
  uint64_t contiguousSplitMicros{0};
  uint64_t hostAllocateAndCopySubmitMicros{0};
  uint64_t copySynchronizeMicros{0};
  uint64_t pinnedToPageableCopyMicros{0};
  uint64_t storageConsumerMicros{0};
  uint64_t pinnedStagingAcquireMicros{0};
  bool usedPinnedStaging{false};
};

struct CudaPinnedHostDeleter {
  void operator()(uint8_t* data) const {
    if (data != nullptr) {
      // A deleter cannot report an error.  All submitted copies are
      // synchronized before this object is destroyed.
      cudaFreeHost(data);
    }
  }
};

class GracePinnedHostStagingPool;

class GracePinnedHostStagingLease {
 public:
  GracePinnedHostStagingLease() = default;
  GracePinnedHostStagingLease(
      GracePinnedHostStagingPool* pool,
      size_t slot,
      uint8_t* data)
      : pool_(pool), slot_(slot), data_(data) {}
  GracePinnedHostStagingLease(const GracePinnedHostStagingLease&) = delete;
  GracePinnedHostStagingLease& operator=(const GracePinnedHostStagingLease&) =
      delete;
  GracePinnedHostStagingLease(GracePinnedHostStagingLease&& other) noexcept
      : pool_(std::exchange(other.pool_, nullptr)),
        slot_(other.slot_),
        data_(std::exchange(other.data_, nullptr)) {}
  GracePinnedHostStagingLease& operator=(
      GracePinnedHostStagingLease&& other) noexcept;
  ~GracePinnedHostStagingLease();

  uint8_t* data() const {
    return data_;
  }

  explicit operator bool() const {
    return data_ != nullptr;
  }

 private:
  void release();

  GracePinnedHostStagingPool* pool_{nullptr};
  size_t slot_{0};
  uint8_t* data_{nullptr};
};

class GracePinnedHostStagingPool {
 public:
  GracePinnedHostStagingPool(
      const char* slotCountEnvironment,
      size_t defaultSlots)
      : maxSlots_(defaultSlots) {
    if (const auto* value = std::getenv(slotCountEnvironment)) {
      char* end = nullptr;
      const auto requested = std::strtoull(value, &end, 10);
      if (end != value && *end == '\0') {
        maxSlots_ = std::clamp<uint64_t>(requested, 1, 16);
      }
    }
  }

  GracePinnedHostStagingLease acquire(uint64_t requiredBytes) {
    if (requiredBytes == 0) {
      return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    size_t available = slots_.size();
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (!slots_[i]->busy && slots_[i]->capacity >= requiredBytes) {
        slots_[i]->busy = true;
        return {this, i, slots_[i]->data.get()};
      }
      if (!slots_[i]->busy && available == slots_.size()) {
        available = i;
      }
    }
    if (available == slots_.size() && slots_.size() < maxSlots_) {
      slots_.push_back(std::make_unique<Slot>());
      available = slots_.size() - 1;
    }
    if (available == slots_.size()) {
      // Acquisition never waits. Probe prefetch can retain a lease until a
      // later operator turn, and build demotion is independently asynchronous;
      // pageable staging remains the bounded correctness fallback.
      return {};
    }
    auto& slot = *slots_[available];
    uint8_t* allocated = nullptr;
    const auto allocationBytes =
        std::max<uint64_t>(kGracePartitionBatchBytes, requiredBytes);
    const auto status = cudaHostAlloc(
        reinterpret_cast<void**>(&allocated),
        allocationBytes,
        cudaHostAllocDefault);
    if (status != cudaSuccess) {
      // Preserve an existing smaller allocation. Pinned staging is an
      // optimization; pageable memory remains the correctness fallback.
      LOG(WARNING) << "Grace hash join could not allocate " << allocationBytes
                   << " bytes for reusable pinned staging slot " << available
                   << ": " << cudaGetErrorString(status)
                   << "; falling back to pageable copies";
      return {};
    }
    slot.data.reset(allocated);
    slot.capacity = allocationBytes;
    slot.busy = true;
    return {this, available, slot.data.get()};
  }

  std::shared_ptr<uint8_t> acquireShared(uint64_t requiredBytes) {
    auto lease = acquire(requiredBytes);
    if (!lease) {
      return nullptr;
    }
    auto owner =
        std::make_shared<GracePinnedHostStagingLease>(std::move(lease));
    return std::shared_ptr<uint8_t>(owner, owner->data());
  }

  void release(size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK_LT(slot, slots_.size());
    VELOX_CHECK(slots_[slot]->busy);
    slots_[slot]->busy = false;
  }

 private:
  struct Slot {
    std::unique_ptr<uint8_t, CudaPinnedHostDeleter> data;
    uint64_t capacity{0};
    bool busy{false};
  };

  std::mutex mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
  size_t maxSlots_;
};

void GracePinnedHostStagingLease::release() {
  if (pool_ != nullptr) {
    pool_->release(slot_);
    pool_ = nullptr;
    data_ = nullptr;
  }
}

GracePinnedHostStagingLease& GracePinnedHostStagingLease::operator=(
    GracePinnedHostStagingLease&& other) noexcept {
  if (this != &other) {
    release();
    pool_ = std::exchange(other.pool_, nullptr);
    slot_ = other.slot_;
    data_ = std::exchange(other.data_, nullptr);
  }
  return *this;
}

GracePinnedHostStagingLease::~GracePinnedHostStagingLease() {
  release();
}

GracePinnedHostStagingPool& gracePinnedHostStagingPool() {
  static GracePinnedHostStagingPool pool(
      "CUDF_HASH_JOIN_GRACE_PINNED_BUFFER_COUNT", 4);
  return pool;
}

GracePinnedHostStagingPool& graceBuildPinnedHostStagingPool() {
  // Build D2H staging must not be starved by long-lived probe prefetch leases
  // or asynchronous disk appends in the shared restore pool. Four 256-MiB
  // slots permit double buffering for the two Job 144 build drivers while
  // retaining a hard 1-GiB pinned-memory ceiling per executor.
  static GracePinnedHostStagingPool pool(
      "CUDF_HASH_JOIN_GRACE_BUILD_PINNED_BUFFER_COUNT", 4);
  return pool;
}

folly::CPUThreadPoolExecutor& graceSpillReadExecutor() {
  // Pinned staging bounds useful concurrent reads to four per executor.
  // Reuse fixed workers instead of std::async, which created one OS thread
  // for every 25-35 MiB probe chunk.
  static folly::CPUThreadPoolExecutor executor(4);
  return executor;
}

using HashJoinPartitionConsumer =
    std::function<void(size_t, HashJoinHostBatch)>;

void packHashJoinPartitions(
    cudf::table_view table,
    const std::vector<cudf::size_type>& splits,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    const HashJoinPartitionConsumer& consumer,
    HashJoinPackTiming* timing = nullptr,
    bool useDedicatedBuildStaging = false) {
  const auto splitStart = std::chrono::steady_clock::now();
  // contiguous_split packs every partition in one libcudf operation.  Keep
  // all device buffers alive while the D2H copies are submitted, then wait
  // once for the whole batch.  The old split + pack loop synchronized once
  // per partition, which serialized up to 16 pack/copy operations for every
  // Grace input batch.
  auto packedPartitions = cudf::contiguous_split(table, splits, stream, mr);
  const auto splitEnd = std::chrono::steady_clock::now();
  uint64_t totalDataBytes = 0;
  for (const auto& packed : packedPartitions) {
    totalDataBytes += packed.data.gpu_data->size();
  }
  // Reuse one of a small process-wide pool of staging buffers. The storage
  // consumer runs while this lease is held, so a disk-bound partition can
  // be appended directly
  // from pinned memory without first allocating and filling a pageable copy.
  // A host-resident partition is explicitly demoted by the consumer. Multiple
  // Velox drivers can concurrently partition, copy, and write using separate
  // leases instead of serializing behind one process-wide mutex.
  const auto pinnedAcquireStart = std::chrono::steady_clock::now();
  auto pinnedStaging =
      (useDedicatedBuildStaging ? graceBuildPinnedHostStagingPool()
                                : gracePinnedHostStagingPool())
          .acquireShared(totalDataBytes);
  const auto pinnedAcquireEnd = std::chrono::steady_clock::now();
  auto* pinnedData = pinnedStaging.get();
  const bool usePinnedStaging = pinnedData != nullptr;
  uint64_t pinnedOffset = 0;
  std::vector<std::shared_ptr<uint8_t>> pageableFallbacks(
      packedPartitions.size());
  for (size_t i = 0; i < packedPartitions.size(); ++i) {
    auto& packed = packedPartitions[i];
    const auto dataBytes = packed.data.gpu_data->size();
    if (dataBytes > 0) {
      if (!usePinnedStaging) {
        pageableFallbacks[i] = std::shared_ptr<uint8_t>(
            new uint8_t[dataBytes], std::default_delete<uint8_t[]>());
      }
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          usePinnedStaging ? pinnedData + pinnedOffset
                           : pageableFallbacks[i].get(),
          packed.data.gpu_data->data(),
          dataBytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    pinnedOffset += dataBytes;
  }
  const auto copySubmitEnd = std::chrono::steady_clock::now();
  stream.synchronize();
  const auto synchronizeEnd = std::chrono::steady_clock::now();
  uint64_t sourceOffset = 0;
  const auto consumerStart = std::chrono::steady_clock::now();
  for (size_t i = 0; i < packedPartitions.size(); ++i) {
    auto& packed = packedPartitions[i];
    HashJoinHostBatch host;
    host.dataBytes = packed.data.gpu_data->size();
    const auto dataBytes = host.dataBytes;
    host.rows = static_cast<vector_size_t>(packed.table.num_rows());
    host.metadata = std::move(packed.data.metadata);
    host.pinned = usePinnedStaging;
    if (host.dataBytes > 0) {
      if (usePinnedStaging) {
        // Alias the pooled allocation. Asynchronous disk writes retain this
        // owner and return the slot only after the final partition range from
        // this slice has completed.
        host.data =
            std::shared_ptr<uint8_t>(pinnedStaging, pinnedData + sourceOffset);
      } else {
        host.data = std::move(pageableFallbacks[i]);
      }
    }
    consumer(i, std::move(host));
    sourceOffset += dataBytes;
  }
  const auto consumerEnd = std::chrono::steady_clock::now();
  if (timing != nullptr) {
    timing->contiguousSplitMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            splitEnd - splitStart)
            .count();
    timing->hostAllocateAndCopySubmitMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            copySubmitEnd - splitEnd)
            .count();
    timing->copySynchronizeMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            synchronizeEnd - copySubmitEnd)
            .count();
    timing->pinnedToPageableCopyMicros = 0;
    timing->storageConsumerMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            consumerEnd - consumerStart)
            .count();
    timing->pinnedStagingAcquireMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            pinnedAcquireEnd - pinnedAcquireStart)
            .count();
    timing->usedPinnedStaging = usePinnedStaging;
  }
}

uint64_t demotePinnedGraceBatch(HashJoinHostBatch& batch) {
  if (!batch.pinned || batch.dataBytes == 0) {
    return 0;
  }
  const auto start = std::chrono::steady_clock::now();
  auto pageable = std::shared_ptr<uint8_t>(
      new uint8_t[batch.dataBytes], std::default_delete<uint8_t[]>());
  std::memcpy(pageable.get(), batch.data.get(), batch.dataBytes);
  batch.data = std::move(pageable);
  batch.pinned = false;
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

bool demotePinnedGraceBatchAsync(HashJoinHostBatch& batch) {
  if (!batch.pinned || batch.dataBytes == 0) {
    return false;
  }
  auto source = batch.data;
  auto pageable = std::shared_ptr<uint8_t>(
      new uint8_t[batch.dataBytes], std::default_delete<uint8_t[]>());
  const auto dataBytes = batch.dataBytes;
  auto completion = std::make_shared<std::promise<void>>();
  batch.spillWriteFuture = completion->get_future().share();
  auto copy =
      [source = std::move(source), pageable, dataBytes, completion]() mutable {
        try {
          std::memcpy(pageable.get(), source.get(), dataBytes);
          // Release the pinned lease before publishing completion. A
          // packaged_task stores its callable in the future's shared state,
          // which kept source alive until final build publication and starved
          // every reusable staging slot even after memcpy had finished.
          source.reset();
          completion->set_value();
        } catch (...) {
          source.reset();
          completion->set_exception(std::current_exception());
        }
      };
  batch.data = std::move(pageable);
  batch.pinned = false;
  try {
    // Copy the small closure into the executor so the local copy remains
    // usable if submission rejects it.
    graceBuildDemoteExecutor().add(copy);
  } catch (...) {
    // Executor rejection is only an optimization failure. Complete the copy
    // inline so the same future still establishes the consumer ordering.
    copy();
  }
  return true;
}

uint64_t waitForGraceResidentDemotions(GraceHashJoinBuildData& buildData) {
  const auto start = std::chrono::steady_clock::now();
  bool waited = false;
  const auto waitBatches = [&](std::vector<HashJoinHostBatch>& batches) {
    for (auto& batch : batches) {
      if (!batch.spillFile && batch.spillWriteFuture.valid()) {
        batch.spillWriteFuture.get();
        batch.spillWriteFuture = {};
        waited = true;
      }
    }
  };
  for (auto& partition : buildData.partitions) {
    waitBatches(partition);
  }
  waitBatches(buildData.unmatchedNulls);
  if (!waited) {
    return 0;
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

uint64_t appendGraceSpillBatch(
    const std::shared_ptr<HashJoinSpillFile>& spillFile,
    HashJoinHostBatch& batch,
    uint64_t& cumulativeWriteMicros) {
  VELOX_CHECK_NOT_NULL(spillFile);
  VELOX_CHECK_NOT_NULL(batch.data);
  const auto writeStart = std::chrono::steady_clock::now();
  auto [offset, writeFuture] =
      spillFile->appendAsync(batch.data, batch.dataBytes);
  batch.spillWriteFuture = std::move(writeFuture);
  // Pageable fallback must not create an unbounded asynchronous write queue.
  // Waiting here applies backpressure when all bounded pinned slots are busy.
  if (!batch.pinned) {
    batch.spillWriteFuture.get();
  }
  cumulativeWriteMicros +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - writeStart)
          .count();
  return offset;
}

/// Creates extended table view by appending precomputed columns
cudf::table_view createExtendedTableView(
    cudf::table_view originalView,
    std::vector<ColumnOrView>& precomputedColumns) {
  if (precomputedColumns.empty()) {
    return originalView;
  }

  std::vector<cudf::column_view> allViews;
  allViews.reserve(originalView.num_columns() + precomputedColumns.size());

  for (cudf::size_type i = 0; i < originalView.num_columns(); ++i) {
    allViews.push_back(originalView.column(i));
  }
  for (auto& col : precomputedColumns) {
    allViews.push_back(asView(col));
  }

  return cudf::table_view(allViews);
}

vector_size_t filteredOutputNumRows(
    bool zeroColumnOutput,
    cudf::column_view filterColumn,
    const std::vector<std::unique_ptr<cudf::column>>& joinedCols,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref tempMr) {
  if (!zeroColumnOutput) {
    return joinedCols.empty() ? 0 : joinedCols[0]->size();
  }

  auto trueCountScalar = cudf::reduce(
      filterColumn,
      *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::INT32},
      stream,
      tempMr);
  return static_cast<vector_size_t>(
      static_cast<cudf::numeric_scalar<int32_t>*>(trueCountScalar.get())
          ->value(stream));
}

// Selects whether getMaskedIndices keeps rows where the mask is true (kMatched)
// or where the mask is false (kUnmatched).
enum class MaskType { kMatched, kUnmatched };

// Returns row indices selected by mask as a column of size_type. When maskType
// is kMatched, indices where the mask is true are returned; when kUnmatched,
// indices where the mask is false are returned.
std::unique_ptr<cudf::column> getMaskedIndices(
    cudf::column_view mask,
    MaskType maskType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto seq = cudf::sequence(
      mask.size(),
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
      stream,
      mr);

  auto indicesTable = maskType == MaskType::kMatched
      ? cudf::apply_boolean_mask(
            cudf::table_view{{seq->view()}}, mask, stream, mr)
      : cudf::apply_deletion_mask(
            cudf::table_view{{seq->view()}}, mask, stream, mr);

  return std::move(indicesTable->release().front());
}

// Tracks which probe rows have been matched across multiple build batches.
// Maintains a boolean column that accumulates matches via cudf::contains +
// BITWISE_OR, and provides a method to retrieve unmatched probe row indices.
class ProbeMatchTracker {
 public:
  ProbeMatchTracker(
      cudf::size_type numProbeRows,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
    matchCol_ =
        cudf::make_column_from_scalar(falseScalar, numProbeRows, stream, mr);
    probeRowIndices_ = cudf::sequence(
        numProbeRows,
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
        stream,
        mr);
  }

  // Mark probe rows present in matchedLeftIndices as matched.
  void update(
      cudf::column_view matchedLeftIndices,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto matchedInBatch = cudf::contains(
        matchedLeftIndices, probeRowIndices_->view(), stream, mr);
    auto updatedMatch = cudf::binary_operation(
        matchCol_->view(),
        matchedInBatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
    stream.synchronize();
    matchCol_ = std::move(updatedMatch);
  }

  // Returns indices of probe rows that were never matched.
  std::unique_ptr<cudf::column> getUnmatchedIndices(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    return getMaskedIndices(
        matchCol_->view(), MaskType::kUnmatched, stream, mr);
  }

  // Returns indices of probe rows that matched in at least one build batch.
  std::unique_ptr<cudf::column> getMatchedIndices(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    return getMaskedIndices(matchCol_->view(), MaskType::kMatched, stream, mr);
  }

 private:
  std::unique_ptr<cudf::column> matchCol_;
  std::unique_ptr<cudf::column> probeRowIndices_;
};

} // namespace

void CudfHashJoinProbe::doClose() {
  graceWorkspace_.reset();
  graceWorkspaceAdmission_.reset();
  Operator::close();
  filterEvaluator_.reset();
  scalars_.clear();
  tree_ = {};
}

void CudfHashJoinBridge::setHashTable(
    std::optional<CudfHashJoinBridge::hash_type> hashObject,
    std::vector<DeviceMemoryAdmissionReservation> deviceAdmissions) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::setHashTable";
  }
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(
        !hashObject_.has_value(),
        "CudfHashJoinBridge already has a hash table");
    hashObject_ = std::move(hashObject);
    deviceAdmissions_ = std::move(deviceAdmissions);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::optional<CudfHashJoinBridge::hash_type> CudfHashJoinBridge::hashOrFuture(
    ContinueFuture* future) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture";
  }
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return hashObject_;
  }
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture constructing promise";
  }
  promises_.emplace_back("CudfHashJoinBridge::hashOrFuture");
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture getSemiFuture";
  }
  *future = promises_.back().getSemiFuture();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridge::hashOrFuture returning nullopt";
  }
  return std::nullopt;
}

void CudfHashJoinBridge::setGraceBuildData(
    std::shared_ptr<GraceHashJoinBuildData> buildData) {
  VELOX_CHECK_NOT_NULL(buildData);
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(
        !hashObject_.has_value() && !graceBuildData_,
        "CudfHashJoinBridge build result already set");
    graceBuildData_ = std::move(buildData);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

void CudfHashJoinBridge::setGraceActivated() {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (graceActivated_) {
      return;
    }
    graceActivated_ = true;
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::optional<std::shared_ptr<GraceHashJoinBuildData>>
CudfHashJoinBridge::graceOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (graceBuildData_) {
    return graceBuildData_;
  }
  promises_.emplace_back("CudfHashJoinBridge::graceOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

std::optional<CudfHashJoinBridge::BuildResult>
CudfHashJoinBridge::resultOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return BuildResult{hashObject_, nullptr, false};
  }
  if (graceBuildData_) {
    return BuildResult{std::nullopt, graceBuildData_, true};
  }
  if (graceActivated_) {
    return BuildResult{std::nullopt, nullptr, true};
  }
  promises_.emplace_back("CudfHashJoinBridge::resultOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

std::optional<CudfHashJoinBridge::BuildResult>
CudfHashJoinBridge::tryFinalResult() {
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return BuildResult{hashObject_, nullptr, false};
  }
  if (graceBuildData_) {
    return BuildResult{std::nullopt, graceBuildData_, true};
  }
  return std::nullopt;
}

std::optional<CudfHashJoinBridge::BuildResult>
CudfHashJoinBridge::finalResultOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (hashObject_.has_value()) {
    return BuildResult{hashObject_, nullptr, false};
  }
  if (graceBuildData_) {
    return BuildResult{std::nullopt, graceBuildData_, true};
  }
  promises_.emplace_back("CudfHashJoinBridge::finalResultOrFuture");
  *future = promises_.back().getSemiFuture();
  return std::nullopt;
}

void CudfHashJoinBridge::setGracePartitions(
    GraceHashJoinPartitionSet partitions) {
  VELOX_CHECK(!partitions.empty());
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(
      !gracePartitionsSet_, "Grace hash join partitions already published");
  gracePartitions_.insert(std::move(partitions));
  gracePartitionsSet_ = true;
}

void CudfHashJoinBridge::appendGracePartitions(
    GraceHashJoinPartitionSet partitions) {
  VELOX_CHECK(!partitions.empty());
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(gracePartitionsSet_);
  gracePartitions_.insert(std::move(partitions));
}

std::optional<GraceHashJoinPartition> CudfHashJoinBridge::nextGracePartition() {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK(gracePartitionsSet_);
  if (!gracePartitions_.hasNext()) {
    return std::nullopt;
  }
  return gracePartitions_.next();
}

void CudfHashJoinBridge::setBuildStream(rmm::cuda_stream_view buildStream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = buildStream;
}

std::optional<rmm::cuda_stream_view> CudfHashJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

void CudfHashJoinBridge::setBuildReadyEvent(
    std::shared_ptr<CudaEvent> buildReadyEvent) {
  std::lock_guard<std::mutex> l(mutex_);
  buildReadyEvent_ = std::move(buildReadyEvent);
}

std::shared_ptr<CudaEvent> CudfHashJoinBridge::getBuildReadyEvent() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildReadyEvent_;
}

CudfHashJoinBuild::CudfHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    // TODO check outputType should be set or not?
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr, // outputType
          joinNode->id(),
          "CudfHashJoinBuild",
          nvtx3::rgb{65, 105, 225}, // Royal Blue
          NvtxMethodFlag::kAll,
          std::nullopt, // spillConfig
          joinNode),
      joinNode_(joinNode) {
  const auto& config = CudfConfig::getInstance();
  graceThresholdBytes_ = config.hashJoinGraceBuildBytes;
  deviceMemoryPool_ = customPool(kCudfDeviceMemoryResourceTag);
  graceHostBytes_ = config.hashJoinGraceHostBytes;
  gracePartitions_ = config.hashJoinGracePartitions;
  if (const auto* value =
          std::getenv("CUDF_HASH_JOIN_GRACE_EAGER_BUILD_MIN_BYTES")) {
    char* end = nullptr;
    const auto requested = std::strtoull(value, &end, 10);
    if (end != value && *end == '\0') {
      graceEagerProbeBuildMinBytes_ = requested;
    }
  }
  graceEligible_ = graceThresholdBytes_ > 0 && gracePartitions_ >= 2 &&
      (joinNode_->isInnerJoin() || joinNode_->isLeftJoin() ||
       joinNode_->isRightJoin()) &&
      !joinNode_->filter();
  if (graceEligible_) {
    const auto& keys = joinNode_->rightKeys();
    const auto buildType = joinNode_->sources()[1]->outputType();
    buildKeyIndices_.reserve(keys.size());
    for (const auto& key : keys) {
      buildKeyIndices_.push_back(
          static_cast<cudf::size_type>(buildType->getChildIdx(key->name())));
    }
  }
}

void CudfHashJoinBuild::partitionAndPack(CudfVectorPtr input) {
  VELOX_CHECK(graceActive_);
  VELOX_CHECK_NOT_NULL(input);
  auto stream = input->stream();
  const auto inputRows = input->size();
  const auto inputBytes = retainedCudfBytes(*input);
  const uint64_t rowsPerSlice =
      inputRows == 0 || inputBytes <= kGracePartitionBatchBytes
      ? std::max<uint64_t>(inputRows, 1)
      : std::max<uint64_t>(
            1,
            static_cast<uint64_t>(
                static_cast<long double>(inputRows) *
                kGracePartitionBatchBytes / inputBytes));
  for (uint64_t begin = 0; begin < inputRows; begin += rowsPerSlice) {
    const auto end = std::min<uint64_t>(inputRows, begin + rowsPerSlice);
    auto slices = cudf::slice(
        input->getTableView(),
        {static_cast<cudf::size_type>(begin),
         static_cast<cudf::size_type>(end)},
        stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    const auto sliceBytes = inputRows == 0
        ? 0
        : static_cast<uint64_t>(
              static_cast<long double>(inputBytes) * (end - begin) / inputRows);
    auto partitionInput = slices.front();
    std::unique_ptr<cudf::table> nonNullRows;
    std::unique_ptr<cudf::table> nullRows;
    if (joinNode_->isRightJoin() &&
        cudf::has_nulls(partitionInput.select(buildKeyIndices_))) {
      VELOX_CHECK(!buildKeyIndices_.empty());
      auto validMask = cudf::is_valid(
          partitionInput.column(buildKeyIndices_.front()),
          stream,
          get_temp_mr());
      for (size_t key = 1; key < buildKeyIndices_.size(); ++key) {
        auto keyValid = cudf::is_valid(
            partitionInput.column(buildKeyIndices_[key]),
            stream,
            get_temp_mr());
        validMask = cudf::binary_operation(
            validMask->view(),
            keyValid->view(),
            cudf::binary_operator::LOGICAL_AND,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());
      }
      auto nullMask = cudf::unary_operation(
          validMask->view(), cudf::unary_operator::NOT, stream, get_temp_mr());
      nullRows = cudf::apply_boolean_mask(
          partitionInput, nullMask->view(), stream, get_output_mr());
      nonNullRows = cudf::apply_boolean_mask(
          partitionInput, validMask->view(), stream, get_output_mr());
      partitionInput = nonNullRows->view();

      if (nullRows->num_rows() > 0) {
        packHashJoinPartitions(
            nullRows->view(),
            {},
            stream,
            get_output_mr(),
            [&](size_t partition, HashJoinHostBatch batch) {
              VELOX_CHECK_EQ(partition, 0);
              graceBuildData_->unmatchedNullRows += batch.rows;
              storeGraceBuildBatch(
                  std::move(batch),
                  graceBuildData_->unmatchedNulls,
                  gracePartitions_);
            },
            nullptr,
            graceAsyncBuildDemoteEnabled());
      }
    }

    const auto partitionStart = std::chrono::steady_clock::now();
    auto [partitioned, offsets] = [&]() {
      std::lock_guard<std::mutex> lock(cudfHashPartitionMutex());
      auto result = cudf::hash_partition(
          partitionInput,
          buildKeyIndices_,
          gracePartitions_,
          cudf::hash_id::HASH_MURMUR3,
          kGraceLocalHashSeed,
          stream,
          get_temp_mr());
      stream.synchronize();
      return result;
    }();
    const auto partitionEnd = std::chrono::steady_clock::now();
    VELOX_CHECK(
        offsets.size() == gracePartitions_ ||
        offsets.size() == gracePartitions_ + 1);
    VELOX_CHECK_EQ(offsets.front(), 0);
    offsets.erase(offsets.begin());
    if (offsets.size() == gracePartitions_) {
      offsets.pop_back();
    }
    HashJoinPackTiming packTiming;
    uint64_t batchPackedBytes = 0;
    packHashJoinPartitions(
        partitioned->view(),
        offsets,
        stream,
        get_output_mr(),
        [&](size_t partition, HashJoinHostBatch host) {
          VELOX_CHECK_LT(partition, graceBuildData_->partitions.size());
          if (host.rows == 0) {
            return;
          }
          batchPackedBytes += host.dataBytes;
          storeGraceBuildBatch(
              std::move(host),
              graceBuildData_->partitions[partition],
              partition);
        },
        &packTiming,
        graceAsyncBuildDemoteEnabled());
    if (!graceEagerProbeNotified_ && graceEagerProbeBuildMinBytes_ > 0 &&
        graceBuildData_->packedBytes >= graceEagerProbeBuildMinBytes_) {
      auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
          operatorCtx_->driverCtx()->splitGroupId, planNodeId());
      auto cudfHashJoinBridge =
          std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
      VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
      cudfHashJoinBridge->setGraceActivated();
      graceEagerProbeNotified_ = true;
      LOG(WARNING) << "CudfHashJoinBuild task="
                   << operatorCtx_->task()->taskId() << " node=" << planNodeId()
                   << " enabling eager Grace probe at localPackedBytes="
                   << graceBuildData_->packedBytes
                   << " minBytes=" << graceEagerProbeBuildMinBytes_;
    }
    const auto partitionMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            partitionEnd - partitionStart)
            .count();
    LOG(WARNING) << "CudfHashJoinBuild node=" << planNodeId()
                 << " Grace slice rows=" << (end - begin)
                 << " estimatedBytes=" << sliceBytes
                 << " sourceRows=" << inputRows
                 << " sourceEstimatedBytes=" << inputBytes
                 << " packedBytes=" << batchPackedBytes
                 << " hashPartitionUs=" << partitionMicros
                 << " contiguousSplitUs=" << packTiming.contiguousSplitMicros
                 << " hostAllocateCopySubmitUs="
                 << packTiming.hostAllocateAndCopySubmitMicros
                 << " copySynchronizeUs=" << packTiming.copySynchronizeMicros
                 << " pinnedToPageableCopyUs="
                 << packTiming.pinnedToPageableCopyMicros
                 << " storageConsumerUs=" << packTiming.storageConsumerMicros
                 << " pinnedStagingAcquireUs="
                 << packTiming.pinnedStagingAcquireMicros
                 << " usedPinnedStaging=" << packTiming.usedPinnedStaging
                 << " residentHostBytes=" << graceBuildData_->residentHostBytes
                 << " diskBytes=" << graceBuildData_->diskBytes
                 << " executorReservedHostBytes="
                 << currentGraceHostMemoryReservedBytes()
                 << " cumulativeHostDemoteUs=" << graceHostDemoteMicros_
                 << " cumulativeRawSpillWriteUs=" << graceRawSpillWriteMicros_;
  }
}

void CudfHashJoinBuild::storeGraceBuildBatch(
    HashJoinHostBatch batch,
    std::vector<HashJoinHostBatch>& destination,
    size_t spillPartition) {
  if (batch.rows == 0) {
    return;
  }
  graceBuildData_->packedBytes += batch.dataBytes;
  graceBuildData_->rows += batch.rows;
  auto reservation =
      tryReserveGraceHostMemory(batch.dataBytes, graceHostBytes_);
  const bool retainInHost = graceHostBytes_ == 0 || reservation != nullptr;
  if (!retainInHost) {
    VELOX_CHECK_LE(spillPartition, gracePartitions_);
    if (graceSpillFiles_.empty()) {
      graceSpillFiles_.resize(gracePartitions_ + 1);
    }
    auto& spillFile = graceSpillFiles_[spillPartition];
    if (!spillFile) {
      const auto& taskSpillDirectory = operatorCtx_->task()->spillDirectory();
      VELOX_CHECK(
          !taskSpillDirectory.empty(),
          "Grace hash join requires a task spill directory after its "
          "{}-byte host limit is reached",
          graceHostBytes_);
      std::filesystem::create_directories(taskSpillDirectory);
      const auto filename = fmt::format(
          "cudf-hash-join-p{:03}-{:06}.bin",
          spillPartition,
          graceHashJoinSpillFileId.fetch_add(1));
      spillFile = createCudfPackedSpillFile(
          taskSpillDirectory, filename, spillPartition, spillConfig());
      const auto& path = spillFile->path();
      LOG(WARNING) << "CudfHashJoinBuild node=" << planNodeId()
                   << " opened raw packed spill file=" << path
                   << " partition=" << spillPartition
                   << " hostLimitBytes=" << graceHostBytes_;
    }
    batch.fileOffset =
        appendGraceSpillBatch(spillFile, batch, graceRawSpillWriteMicros_);
    batch.spillFile = spillFile;
    batch.data.reset();
    batch.hostReservation.reset();
    batch.pinned = false;
    graceBuildData_->diskBytes += batch.dataBytes;
  } else {
    batch.hostReservation = std::move(reservation);
    if (graceAsyncBuildDemoteEnabled() && demotePinnedGraceBatchAsync(batch)) {
      graceAsyncHostDemoteBytes_ += batch.dataBytes;
      ++graceAsyncHostDemoteTasks_;
    } else {
      graceHostDemoteMicros_ += demotePinnedGraceBatch(batch);
    }
    graceBuildData_->residentHostBytes += batch.dataBytes;
  }
  destination.push_back(std::move(batch));
}

void CudfHashJoinBuild::flushGraceInputBatch() {
  VELOX_CHECK(graceActive_);
  if (inputs_.empty()) {
    return;
  }
  auto batch = std::exchange(inputs_, {});
  retainedBuildBytes_ = 0;
  if (batch.size() == 1) {
    partitionAndPack(std::move(batch.front()));
    return;
  }
  auto stream = batch.front()->stream();
  const auto buildType = joinNode_->sources()[1]->outputType();
  auto table = getConcatenatedTable(
      std::move(batch), buildType, stream, get_output_mr());
  const auto rows = table->num_rows();
  auto combined = std::make_shared<CudfVector>(
      pool(), buildType, rows, std::move(table), stream);
  partitionAndPack(std::move(combined));
}

void CudfHashJoinBuild::queueGraceInput(CudfVectorPtr input) {
  VELOX_CHECK(graceActive_);
  VELOX_CHECK_NOT_NULL(input);
  const auto inputBytes = retainedCudfBytes(*input);
  // Split only when the empty/non-empty chars pattern changes. Uniform
  // all-empty groups remain safe to concatenate and avoid fragmented
  // partition/pack launches.
  if (!inputs_.empty() &&
      !hasSameEmptyStringCharsPattern(
          inputs_.front()->getTableView(), input->getTableView())) {
    flushGraceInputBatch();
  }
  if (!inputs_.empty() &&
      retainedBuildBytes_ + inputBytes > kGracePartitionBatchBytes) {
    flushGraceInputBatch();
  }
  // partitionAndPack() row-slices an oversized vector without copying it.
  if (inputBytes >= kGracePartitionBatchBytes) {
    partitionAndPack(std::move(input));
    return;
  }
  retainedBuildBytes_ += inputBytes;
  inputs_.push_back(std::move(input));
  if (retainedBuildBytes_ >= kGracePartitionBatchBytes) {
    flushGraceInputBatch();
  }
}

void CudfHashJoinBuild::activateGracePath() {
  if (graceActive_) {
    return;
  }
  graceActive_ = true;
  graceBuildData_ = std::make_shared<GraceHashJoinBuildData>();
  graceBuildData_->partitions.resize(gracePartitions_);
  LOG(WARNING) << "CudfHashJoinBuild node=" << planNodeId()
               << " switching to host-first Grace path at retainedBytes="
               << retainedBuildBytes_ << " partitions=" << gracePartitions_;
  auto retainedInputs = std::exchange(inputs_, {});
  retainedBuildBytes_ = 0;
  for (auto& input : retainedInputs) {
    queueGraceInput(std::move(input));
  }
  flushGraceInputBatch();
  // Every retained device input has now been copied into the shared packed
  // host/disk tier. Releasing these charges makes the capacity immediately
  // available to the requesting operator before arbitration resumes drivers.
  deviceAdmissions_.clear();
}

void CudfHashJoinBuild::doAddInput(RowVectorPtr input) {
  // Queue inputs, process all at once.
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    // Count nulls in join key columns
    auto [_, null_count] = cudf::bitmask_and(
        cudfInput->getTableView(), cudfInput->stream(), get_temp_mr());
    // We only retain null_count. The returned device mask is a temporary and
    // is destroyed at the end of this scope. bitmask_and is asynchronous, so
    // without ordering its destruction against the input stream the mask can
    // be recycled while the kernel is still writing it. Under many concurrent
    // joins this corrupts unrelated cuco counter storage and surfaces later as
    // cudaErrorIllegalAddress in a probe, TopN, or scan.
    cudfInput->stream().synchronize();
    {
      // Update statistics for null keys in join operator.
      auto lockedStats = stats_.wlock();
      lockedStats->numNullKeys += null_count;
    }
    const auto inputBytes = retainedCudfBytes(*cudfInput);
    if (graceEligible_ && !graceActive_ &&
        retainedBuildBytes_ + inputBytes > graceThresholdBytes_) {
      activateGracePath();
    }
    if (graceActive_) {
      queueGraceInput(std::move(cudfInput));
    } else {
      std::optional<DeviceMemoryAdmissionReservation> deviceAdmission;
      if (deviceMemoryPool_ != nullptr && inputBytes > 0) {
        // Match CPU Velox's spillable operators: allow arbitration at the
        // reservation boundary even though Driver normally executes addInput
        // in a non-reclaimable section. Existing resident build inputs are a
        // complete state that can safely transition to Grace before this
        // incoming batch is retained.
        Operator::ReclaimableSectionGuard reclaimableSection(this);
        deviceAdmission.emplace(
            acquireDeviceMemoryAdmission(deviceMemoryPool_, inputBytes, this));
      }
      // Arbitration is allowed to select this build as its victim. In that
      // case activateGracePath() ran while the task was paused, so the incoming
      // batch must join the Grace stream instead of recreating resident state.
      if (graceActive_) {
        deviceAdmission.reset();
        queueGraceInput(std::move(cudfInput));
        return;
      }
      retainedBuildBytes_ += inputBytes;
      inputs_.push_back(std::move(cudfInput));
      if (deviceAdmission.has_value()) {
        deviceAdmissions_.push_back(std::move(deviceAdmission.value()));
      }
    }
  }
}

bool CudfHashJoinBuild::canReclaim() const {
  return graceEligible_ && !graceActive_ && !noMoreInput_ && !inputs_.empty();
}

bool CudfHashJoinBuild::reclaimableBytes(uint64_t& reclaimableBytes) const {
  reclaimableBytes = canReclaim() ? retainedBuildBytes_ : 0;
  return true;
}

void CudfHashJoinBuild::reclaim(
    uint64_t /*targetBytes*/,
    memory::MemoryReclaimer::Stats& /*stats*/) {
  if (!canReclaim()) {
    return;
  }
  const auto before = retainedBuildBytes_;
  activateGracePath();
  addRuntimeStat(
      "deviceArbitrationReclaimBytes",
      RuntimeCounter(before, RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "deviceArbitrationReclaimCount",
      RuntimeCounter(1, RuntimeCounter::Unit::kNone));
  LOG(WARNING) << "CudfHashJoinBuild task=" << taskId()
               << " node=" << planNodeId()
               << " swept resident build bytes=" << before
               << " into Grace storage";
}

bool CudfHashJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr CudfHashJoinBuild::doGetOutput() {
  if (buildFinalizePending_) {
    finalizeBuild();
  }
  return nullptr;
}

void CudfHashJoinBuild::doClose() {
  buildFinalizeWorkspace_.reset();
  buildFinalizePending_ = false;
  Operator::close();
}

void CudfHashJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<CudfHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    retainedBuildBytes_ += build->retainedBuildBytes_;
    deviceAdmissions_.insert(
        deviceAdmissions_.end(),
        std::make_move_iterator(build->deviceAdmissions_.begin()),
        std::make_move_iterator(build->deviceAdmissions_.end()));
    build->deviceAdmissions_.clear();
    inputs_.insert(
        inputs_.end(),
        std::make_move_iterator(build->inputs_.begin()),
        std::make_move_iterator(build->inputs_.end()));
    build->inputs_.clear();
    if (build->graceActive_) {
      if (!graceActive_) {
        activateGracePath();
      }
      for (size_t i = 0; i < graceBuildData_->partitions.size(); ++i) {
        auto& source = build->graceBuildData_->partitions[i];
        auto& target = graceBuildData_->partitions[i];
        target.insert(
            target.end(),
            std::make_move_iterator(source.begin()),
            std::make_move_iterator(source.end()));
      }
      graceBuildData_->unmatchedNulls.insert(
          graceBuildData_->unmatchedNulls.end(),
          std::make_move_iterator(
              build->graceBuildData_->unmatchedNulls.begin()),
          std::make_move_iterator(
              build->graceBuildData_->unmatchedNulls.end()));
      graceBuildData_->packedBytes += build->graceBuildData_->packedBytes;
      graceBuildData_->residentHostBytes +=
          build->graceBuildData_->residentHostBytes;
      graceBuildData_->diskBytes += build->graceBuildData_->diskBytes;
      graceBuildData_->rows += build->graceBuildData_->rows;
      graceBuildData_->unmatchedNullRows +=
          build->graceBuildData_->unmatchedNullRows;
      graceHostDemoteMicros_ += build->graceHostDemoteMicros_;
      graceAsyncHostDemoteBytes_ += build->graceAsyncHostDemoteBytes_;
      graceAsyncHostDemoteTasks_ += build->graceAsyncHostDemoteTasks_;
      graceAsyncHostDemoteTailWaitMicros_ +=
          build->graceAsyncHostDemoteTailWaitMicros_;
      graceRawSpillWriteMicros_ += build->graceRawSpillWriteMicros_;
    }
    auto retainedInputBatches = build->inputs_.size();
    common::testutil::TestValue::adjust(
        "facebook::velox::cudf_velox::CudfHashJoinBuild::doNoMoreInput::sourceDriverRetainedInputBatchesAfterTransfer",
        &retainedInputBatches);
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  VELOX_CHECK(!buildFinalizePending_);
  buildFinalizePending_ = true;
  finalizeBuild();
}

uint64_t CudfHashJoinBuild::residentBuildFinalizeWorkspaceBytes() const {
  constexpr uint64_t kFinalizeFixedWorkspaceBytes = 512ULL << 20;
  const auto buildCopiesBytes =
      retainedBuildBytes_ > (std::numeric_limits<uint64_t>::max() -
                             kFinalizeFixedWorkspaceBytes) /
              2
      ? std::numeric_limits<uint64_t>::max()
      : retainedBuildBytes_ * 2 + kFinalizeFixedWorkspaceBytes;
  return std::max<uint64_t>(
      1ULL << 30,
      std::min<uint64_t>(
          buildCopiesBytes, std::numeric_limits<std::size_t>::max()));
}

void CudfHashJoinBuild::finalizeBuild() {
  VELOX_CHECK(buildFinalizePending_);
  VELOX_CHECK(!buildFinalizeWorkspace_.waiting());

  // A resident build owns all of its input columns while concatenating them
  // and constructing the cuco hash table.  Account for that transient image
  // in the same workspace admission domain as TopN/Grace restore.  Without
  // this reservation, several small builds can all finish at once while
  // TopN has admitted restore work, observe the same physical headroom, and
  // fail the first full-size concatenate allocation.
  //
  // Peer collection and build publication are separate one-shot phases. If
  // headroom is unavailable, retain the cancellation-safe request and resume
  // this phase from getOutput() after an advisory scheduler wake. This keeps a
  // short-lived pressure wave from converting a resident build into a full
  // downstream Grace spill/restore pass.
  std::optional<DeviceMemoryWorkspaceReservation>
      residentBuildFinalizeWorkspace;
  if (!graceActive_ && retainedBuildBytes_ > 0 &&
      deviceMemoryPool_ != nullptr) {
    const auto requestedWorkspace = residentBuildFinalizeWorkspaceBytes();
    const auto minHeadroom =
        CudfConfig::getInstance().deviceMemoryMinHeadroomBytes;
    const auto headroom = captureDeviceAllocationHeadroom();
    const auto restoreCushion = minHeadroom - minHeadroom / 4;
    const bool workspaceCanEverFit = headroom.cudaValid &&
        requestedWorkspace <= headroom.totalBytes &&
        restoreCushion <= headroom.totalBytes - requestedWorkspace;
    if (!workspaceCanEverFit) {
      addRuntimeStat(
          "residentBuildFinalizeWorkspaceImpossible",
          RuntimeCounter(1, RuntimeCounter::Unit::kNone));
      LOG(WARNING)
          << "CudfHashJoinBuild task=" << taskId() << " node=" << planNodeId()
          << " cannot ever satisfy resident finalize workspace retainedBytes="
          << retainedBuildBytes_
          << " requestedWorkspaceBytes=" << requestedWorkspace
          << " restoreCushionBytes=" << restoreCushion
          << " deviceTotalBytes=" << headroom.totalBytes
          << " cudaValid=" << headroom.cudaValid;
      if (graceEligible_) {
        activateGracePath();
      }
    } else {
      auto attempt = buildFinalizeWorkspace_.tryAcquire(
          deviceMemoryPool_,
          this,
          static_cast<std::size_t>(requestedWorkspace),
          minHeadroom,
          DeviceMemoryWorkspacePriority::kRestore);
      auto workspace = std::move(attempt.reservation);
      bool testDeferAfterAdmission = false;
      common::testutil::TestValue::adjust(
          "facebook::velox::cudf_velox::CudfHashJoinBuild::finalizeBuild::deferAfterAdmission",
          &testDeferAfterAdmission);
      if (testDeferAfterAdmission && workspace.has_value()) {
        // Exercise the Driver replay path without manufacturing real device
        // pressure in a unit test. A ready advisory models a scheduler wake;
        // the next getOutput() still performs the normal physical-headroom
        // recheck.
        workspace.reset();
        buildFinalizeWorkspace_.deferReadyForTesting();
      }
      if (workspace.has_value()) {
        residentBuildFinalizeWorkspace.emplace(std::move(workspace.value()));
        addRuntimeStat(
            "residentBuildFinalizeWorkspaceBytes",
            RuntimeCounter(requestedWorkspace, RuntimeCounter::Unit::kBytes));
        if (attempt.completedWaitMicros.has_value()) {
          addRuntimeStat(
              "residentBuildFinalizeWorkspaceWaitMicros",
              RuntimeCounter(
                  attempt.completedWaitMicros.value(),
                  RuntimeCounter::Unit::kNone));
        }
      } else {
        if (!replayableResidentBuildFinalizeEnabled()) {
          buildFinalizeWorkspace_.reset();
          addRuntimeStat(
              "residentBuildFinalizeImmediateFallbacks",
              RuntimeCounter(1, RuntimeCounter::Unit::kNone));
          if (graceEligible_) {
            activateGracePath();
          } else {
            LOG(WARNING)
                << "CudfHashJoinBuild task=" << taskId()
                << " node=" << planNodeId()
                << " could not reserve resident finalize workspace and is "
                   "not Grace-eligible retainedBytes="
                << retainedBuildBytes_
                << " requestedWorkspaceBytes=" << requestedWorkspace;
          }
        } else {
          if (attempt.firstWait) {
            addRuntimeStat(
                "residentBuildFinalizeWorkspaceWaits",
                RuntimeCounter(1, RuntimeCounter::Unit::kNone));
            LOG(WARNING)
                << "CudfHashJoinBuild task=" << taskId()
                << " node=" << planNodeId()
                << " deferred resident finalize for workspace retainedBytes="
                << retainedBuildBytes_
                << " requestedWorkspaceBytes=" << requestedWorkspace;
          }
          return;
        }
      }
    }
  }

  if (graceEligible_ && !graceActive_ &&
      retainedBuildBytes_ > graceThresholdBytes_) {
    activateGracePath();
  } else if (graceActive_ && !inputs_.empty()) {
    flushGraceInputBatch();
    deviceAdmissions_.clear();
  }

  if (graceActive_) {
    // Build consumers must never observe a partially demoted pageable image.
    // Waiting only at publication lets the fixed CPU workers overlap the
    // preceding GPU hash-partition and D2H work while bounding the final tail.
    graceAsyncHostDemoteTailWaitMicros_ +=
        waitForGraceResidentDemotions(*graceBuildData_);
    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto cudfHashJoinBridge =
        std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
    LOG(WARNING) << "CudfHashJoinBuild task=" << operatorCtx_->task()->taskId()
                 << " node=" << planNodeId()
                 << " published host-first Grace build rows="
                 << graceBuildData_->rows
                 << " unmatchedNullRows=" << graceBuildData_->unmatchedNullRows
                 << " packedBytes=" << graceBuildData_->packedBytes;
    LOG(WARNING) << "CudfHashJoinBuild node=" << planNodeId()
                 << " Grace storage residentHostBytes="
                 << graceBuildData_->residentHostBytes
                 << " diskBytes=" << graceBuildData_->diskBytes
                 << " executorReservedHostBytes="
                 << currentGraceHostMemoryReservedBytes()
                 << " hostDemoteUs=" << graceHostDemoteMicros_
                 << " asyncHostDemoteBytes=" << graceAsyncHostDemoteBytes_
                 << " asyncHostDemoteTasks=" << graceAsyncHostDemoteTasks_
                 << " asyncHostDemoteTailWaitUs="
                 << graceAsyncHostDemoteTailWaitMicros_
                 << " rawSpillWriteUs=" << graceRawSpillWriteMicros_;
    logLiveDeviceMemoryAttribution(
        fmt::format("CudfHashJoinBuild node={} Grace publish", planNodeId()));
    cudfHashJoinBridge->setGraceBuildData(std::move(graceBuildData_));
    buildFinalizePending_ = false;
    return;
  }

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "CudfHashJoinBuild: build batches count: " << inputs_.size();
    if (!inputs_.empty()) {
      VLOG(1) << "Build batches number of columns: "
              << inputs_[0]->getTableView().num_columns();
    }
    for (auto i = 0; i < inputs_.size(); i++) {
      VLOG(1) << "Build batch " << i
              << ": number of rows: " << inputs_[i]->getTableView().num_rows();
    }
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  const auto& cudfConfig = CudfConfig::getInstance();
  const auto defaultMaxRows = cudfConfig.batchSizeMaxThreshold.value_or(
      std::numeric_limits<int32_t>::max());
  const auto maxRows = operatorCtx_->driverCtx()->queryConfig().get<int32_t>(
      CudfConfig::kCudfBatchSizeMaxThreshold, defaultMaxRows);
  VELOX_CHECK_GT(maxRows, 0, "cuDF hash join max batch size must be positive");
  const auto hashJoinLoadFactor =
      operatorCtx_->driverCtx()->queryConfig().get<double>(
          CudfConfig::kCudfHashJoinLoadFactor, cudfConfig.hashJoinLoadFactor);
  VELOX_CHECK_GT(
      hashJoinLoadFactor, 0, "cuDF hash join load factor must be positive");
  VELOX_CHECK_LE(
      hashJoinLoadFactor, 1, "cuDF hash join load factor must not exceed one");
  // Using output_mr here to allow spilling queued up large tables
  auto tbls = getConcatenatedTableBatched(
      std::exchange(inputs_, {}),
      joinNode_->sources()[1]->outputType(),
      stream,
      get_output_mr(),
      static_cast<size_t>(maxRows));

  for (auto const& tbl : tbls) {
    VELOX_CHECK_NOT_NULL(tbl);
  }
  if (CudfConfig::getInstance().debugEnabled && !tbls.empty()) {
    VLOG(1) << "Build table number of columns: " << tbls[0]->num_columns();
    for (auto i = 0; i < tbls.size(); i++) {
      VLOG(1) << "Build table " << i
              << ": number of rows: " << tbls[i]->num_rows();
    }
  }

  auto rightKeys = joinNode_->rightKeys();

  auto buildKeyIndices = std::vector<cudf::size_type>(rightKeys.size());
  auto buildType = joinNode_->sources()[1]->outputType();
  for (size_t i = 0; i < buildKeyIndices.size(); i++) {
    buildKeyIndices[i] = static_cast<cudf::size_type>(
        buildType->getChildIdx(rightKeys[i]->name()));
  }

  // Construct hash_join object for join types that use hb->inner_join() or
  // hb->left_join(). Semi filter and anti joins use standalone cudf functions
  // (e.g., mixed_left_semi_join, filtered_join) that build hash tables
  // internally, so they don't need this.
  bool buildHashJoin =
      (joinNode_->isInnerJoin() || joinNode_->isLeftJoin() ||
       joinNode_->isRightJoin() || joinNode_->isFullJoin() ||
       joinNode_->isLeftSemiProjectJoin() ||
       joinNode_->isRightSemiProjectJoin());

  std::vector<std::shared_ptr<cudf::hash_join>> hashObjects;
  {
    std::lock_guard<std::mutex> cucoLock(cudfCucoMutex());
    for (auto i = 0; i < tbls.size(); i++) {
      hashObjects.push_back(
          (buildHashJoin) ? makeSharedHashJoin(
                                tbls[i]->view().select(buildKeyIndices),
                                cudf::null_equality::UNEQUAL,
                                hashJoinLoadFactor,
                                stream)
                          : nullptr);
      if (buildHashJoin) {
        VELOX_CHECK_NOT_NULL(hashObjects.back());
      }
      if (CudfConfig::getInstance().debugEnabled) {
        if (hashObjects.back() != nullptr) {
          VLOG(2) << "hashObject " << i << " is not nullptr "
                  << hashObjects.back().get() << "\n";
        } else {
          VLOG(2) << "hashObject " << i << " is *** nullptr\n";
        }
      }
    }
    stream.synchronize();
  }

  auto buildReadyEvent = std::make_shared<CudaEvent>(cudaEventDisableTiming);
  buildReadyEvent->recordFrom(stream);

  std::vector<std::shared_ptr<cudf::table>> shared_tbls;
  for (auto& tbl : tbls) {
    shared_tbls.push_back(std::move(tbl));
  }
  // set hash table to CudfHashJoinBridge
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);

  cudfHashJoinBridge->setBuildStream(stream);
  cudfHashJoinBridge->setBuildReadyEvent(std::move(buildReadyEvent));
  cudfHashJoinBridge->setHashTable(
      std::make_optional(
          std::make_pair(std::move(shared_tbls), std::move(hashObjects))),
      std::move(deviceAdmissions_));
  buildFinalizePending_ = false;
}

exec::BlockingReason CudfHashJoinBuild::isBlocked(ContinueFuture* future) {
  if (buildFinalizeWorkspace_.takeFuture(future)) {
    return exec::BlockingReason::kWaitForArbitration;
  }
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool CudfHashJoinBuild::isFinished() {
  return !future_.valid() && !buildFinalizePending_ && noMoreInput_;
}

CudfHashJoinProbe::CudfHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "CudfHashJoinProbe",
          nvtx3::rgb{0, 128, 128}, // Teal
          NvtxMethodFlag::kAll,
          std::nullopt, // spillConfig
          joinNode),
      joinNode_(joinNode),
      probeType_(joinNode_->sources()[0]->outputType()),
      buildType_(joinNode_->sources()[1]->outputType()),
      cudaEvent_(std::make_unique<CudaEvent>(cudaEventDisableTiming)) {
  auto const& leftKeys = joinNode_->leftKeys(); // probe keys
  auto const& rightKeys = joinNode_->rightKeys(); // build keys

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < probeType_->names().size(); i++) {
      VLOG(1) << "Left column " << i << ": " << probeType_->names()[i];
    }

    for (int i = 0; i < buildType_->names().size(); i++) {
      VLOG(1) << "Right column " << i << ": " << buildType_->names()[i];
    }

    for (int i = 0; i < leftKeys.size(); i++) {
      VLOG(1) << "Left key " << i << ": " << leftKeys[i]->name() << " "
              << leftKeys[i]->type()->kind();
    }

    for (int i = 0; i < rightKeys.size(); i++) {
      VLOG(1) << "Right key " << i << ": " << rightKeys[i]->name() << " "
              << rightKeys[i]->type()->kind();
    }
  }

  auto const probeTableNumColumns = probeType_->size();
  leftKeyIndices_ = std::vector<cudf::size_type>(leftKeys.size());
  for (size_t i = 0; i < leftKeyIndices_.size(); i++) {
    leftKeyIndices_[i] = static_cast<cudf::size_type>(
        probeType_->getChildIdx(leftKeys[i]->name()));
    VELOX_CHECK_LT(leftKeyIndices_[i], probeTableNumColumns);
  }
  auto const buildTableNumColumns = buildType_->size();
  rightKeyIndices_ = std::vector<cudf::size_type>(rightKeys.size());
  for (size_t i = 0; i < rightKeyIndices_.size(); i++) {
    rightKeyIndices_[i] = static_cast<cudf::size_type>(
        buildType_->getChildIdx(rightKeys[i]->name()));
    VELOX_CHECK_LT(rightKeyIndices_[i], buildTableNumColumns);
  }
  const auto& config = CudfConfig::getInstance();
  graceEnabled_ = config.hashJoinGraceBuildBytes > 0 &&
      config.hashJoinGracePartitions >= 2 &&
      (joinNode_->isInnerJoin() || joinNode_->isLeftJoin() ||
       joinNode_->isRightJoin()) &&
      !joinNode_->filter();
  if (graceEnabled_) {
    graceProbePartitions_.resize(config.hashJoinGracePartitions);
    graceProbeHostLimitBytes_ = config.hashJoinGraceHostBytes;
    graceRestoreBuildBytes_ = config.hashJoinGraceRestoreBytes;
    graceRestoreProbeBytes_ = config.hashJoinGraceProbeRestoreBytes;
    if (const auto* value =
            std::getenv("CUDF_HASH_JOIN_GRACE_PREFETCH_DEPTH")) {
      char* end = nullptr;
      const auto requested = std::strtoull(value, &end, 10);
      if (end != value && *end == '\0') {
        graceProbePrefetchDepth_ = std::clamp<uint64_t>(requested, 1, 16);
      }
    }
    if (const auto* value =
            std::getenv("CUDF_HASH_JOIN_GRACE_EAGER_PROBE_BYTES")) {
      char* end = nullptr;
      const auto requested = std::strtoull(value, &end, 10);
      if (end != value && *end == '\0' && requested > 0) {
        graceEagerProbeBufferLimitBytes_ = requested;
      }
    }
  }

  auto outputType = joinNode_->outputType();
  leftColumnIndicesToGather_ = std::vector<cudf::size_type>();
  rightColumnIndicesToGather_ = std::vector<cudf::size_type>();
  leftColumnOutputIndices_ = std::vector<size_t>();
  rightColumnOutputIndices_ = std::vector<size_t>();
  for (int i = 0; i < outputType->names().size(); i++) {
    auto const outputName = outputType->names()[i];
    if (CudfConfig::getInstance().debugEnabled) {
      VLOG(1) << "Output column " << i << ": " << outputName;
    }
    auto channel = probeType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      leftColumnOutputIndices_.push_back(i);
      continue;
    }
    channel = buildType_->getChildIdxIfExists(outputName);
    if (channel.has_value()) {
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(channel.value()));
      rightColumnOutputIndices_.push_back(i);
      continue;
    }
    // For SEMI PROJECT, the last column is the boolean "match" column which is
    // not in probe or build types - skip it here, handled separately.
    if ((isLeftSemiProjectJoin(joinNode_->joinType()) ||
         isRightSemiProjectJoin(joinNode_->joinType())) &&
        i == outputType->size() - 1 &&
        outputType->childAt(i)->kind() == TypeKind::BOOLEAN) {
      continue;
    }
    VELOX_FAIL(
        "Join field {} not in probe or build input", outputType->children()[i]);
  }

  if (CudfConfig::getInstance().debugEnabled) {
    for (int i = 0; i < leftColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Left index to gather " << i << ": "
              << leftColumnIndicesToGather_[i];
    }

    for (int i = 0; i < rightColumnIndicesToGather_.size(); i++) {
      VLOG(1) << "Right index to gather " << i << ": "
              << rightColumnIndicesToGather_[i];
    }
  }
}

void CudfHashJoinProbe::spillGraceProbeBatch(
    HashJoinHostBatch& batch,
    size_t spillPartition) {
  VELOX_CHECK_GT(batch.rows, 0);
  graceProbeBufferedBytes_ += batch.dataBytes;
  auto reservation =
      tryReserveGraceHostMemory(batch.dataBytes, graceProbeHostLimitBytes_);
  if (graceProbeHostLimitBytes_ == 0 || reservation != nullptr) {
    batch.hostReservation = std::move(reservation);
    graceHostDemoteMicros_ += demotePinnedGraceBatch(batch);
    graceProbeResidentHostBytes_ += batch.dataBytes;
    return;
  }

  if (graceProbeSpillFiles_.size() <= spillPartition) {
    graceProbeSpillFiles_.resize(spillPartition + 1);
  }
  auto& spillFile = graceProbeSpillFiles_[spillPartition];
  if (!spillFile) {
    const auto& taskSpillDirectory = operatorCtx_->task()->spillDirectory();
    VELOX_CHECK(
        !taskSpillDirectory.empty(),
        "Grace hash join probe requires a task spill directory after its "
        "{}-byte host limit is reached",
        graceProbeHostLimitBytes_);
    std::filesystem::create_directories(taskSpillDirectory);
    const auto filename = fmt::format(
        "cudf-hash-join-probe-p{:03}-{:06}.bin",
        spillPartition,
        graceHashJoinSpillFileId.fetch_add(1));
    spillFile = createCudfPackedSpillFile(
        taskSpillDirectory, filename, spillPartition, spillConfig());
    const auto& path = spillFile->path();
    LOG(WARNING) << "CudfHashJoinProbe node=" << planNodeId()
                 << " opened raw packed spill file=" << path
                 << " partition=" << spillPartition
                 << " hostLimitBytes=" << graceProbeHostLimitBytes_;
  }
  VELOX_CHECK_NOT_NULL(batch.data);
  batch.fileOffset =
      appendGraceSpillBatch(spillFile, batch, graceRawSpillWriteMicros_);
  batch.spillFile = spillFile;
  batch.data.reset();
  batch.hostReservation.reset();
  batch.pinned = false;
  graceProbeDiskBytes_ += batch.dataBytes;
}

void CudfHashJoinProbe::spillRecursiveGraceBuildBatch(
    HashJoinHostBatch& batch,
    size_t spillPartition) {
  VELOX_CHECK_GT(batch.rows, 0);
  VELOX_CHECK_NOT_NULL(graceBuildData_);
  auto reservation =
      tryReserveGraceHostMemory(batch.dataBytes, graceProbeHostLimitBytes_);
  if (graceProbeHostLimitBytes_ == 0 || reservation != nullptr) {
    batch.hostReservation = std::move(reservation);
    graceHostDemoteMicros_ += demotePinnedGraceBatch(batch);
    graceBuildData_->residentHostBytes += batch.dataBytes;
    return;
  }

  if (graceRecursiveBuildSpillFiles_.size() <= spillPartition) {
    graceRecursiveBuildSpillFiles_.resize(spillPartition + 1);
  }
  auto& spillFile = graceRecursiveBuildSpillFiles_[spillPartition];
  if (!spillFile) {
    const auto& taskSpillDirectory = operatorCtx_->task()->spillDirectory();
    VELOX_CHECK(
        !taskSpillDirectory.empty(),
        "Recursive Grace hash join build requires a task spill directory");
    std::filesystem::create_directories(taskSpillDirectory);
    const auto filename = fmt::format(
        "cudf-hash-join-recursive-build-p{:03}-{:06}.bin",
        spillPartition,
        graceHashJoinSpillFileId.fetch_add(1));
    spillFile = createCudfPackedSpillFile(
        taskSpillDirectory, filename, spillPartition, spillConfig());
    const auto& path = spillFile->path();
    LOG(WARNING) << "CudfHashJoinProbe node=" << planNodeId()
                 << " opened recursive raw packed build spill file=" << path
                 << " partition=" << spillPartition
                 << " hostLimitBytes=" << graceProbeHostLimitBytes_;
  }
  VELOX_CHECK_NOT_NULL(batch.data);
  batch.fileOffset =
      appendGraceSpillBatch(spillFile, batch, graceRawSpillWriteMicros_);
  batch.spillFile = spillFile;
  batch.data.reset();
  batch.hostReservation.reset();
  batch.pinned = false;
  graceBuildData_->diskBytes += batch.dataBytes;
}

void CudfHashJoinProbe::accountConsumedGraceProbeBatch(
    const HashJoinHostBatch& batch) {
  VELOX_CHECK_GE(graceProbeBufferedBytes_, batch.dataBytes);
  graceProbeBufferedBytes_ -= batch.dataBytes;
  if (batch.spillFile) {
    VELOX_CHECK_GE(graceProbeDiskBytes_, batch.dataBytes);
    graceProbeDiskBytes_ -= batch.dataBytes;
  } else {
    VELOX_CHECK_GE(graceProbeResidentHostBytes_, batch.dataBytes);
    graceProbeResidentHostBytes_ -= batch.dataBytes;
  }
}

void CudfHashJoinProbe::accountConsumedGraceBuildBatch(
    const HashJoinHostBatch& batch) {
  VELOX_CHECK_NOT_NULL(graceBuildData_);
  if (batch.spillFile) {
    VELOX_CHECK_GE(graceBuildData_->diskBytes, batch.dataBytes);
    graceBuildData_->diskBytes -= batch.dataBytes;
  } else {
    VELOX_CHECK_GE(graceBuildData_->residentHostBytes, batch.dataBytes);
    graceBuildData_->residentHostBytes -= batch.dataBytes;
  }
}

void CudfHashJoinProbe::partitionAndPackProbe(CudfVectorPtr input) {
  VELOX_CHECK_NOT_NULL(input);
  ++graceProbePartitionBatches_;
  auto stream = input->stream();
  const auto numPartitions = graceProbePartitions_.size();
  const auto inputRows = input->size();
  const auto inputBytes = retainedCudfBytes(*input);
  const uint64_t rowsPerSlice =
      inputRows == 0 || inputBytes <= kGracePartitionBatchBytes
      ? std::max<uint64_t>(inputRows, 1)
      : std::max<uint64_t>(
            1,
            static_cast<uint64_t>(
                static_cast<long double>(inputRows) *
                kGracePartitionBatchBytes / inputBytes));
  for (uint64_t begin = 0; begin < inputRows; begin += rowsPerSlice) {
    const auto end = std::min<uint64_t>(inputRows, begin + rowsPerSlice);
    auto slices = cudf::slice(
        input->getTableView(),
        {static_cast<cudf::size_type>(begin),
         static_cast<cudf::size_type>(end)},
        stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    auto partitionInput = slices.front();
    std::unique_ptr<cudf::table> nonNullProbeRows;
    if (joinNode_->isRightJoin() &&
        cudf::has_nulls(partitionInput.select(leftKeyIndices_))) {
      // Probe is the non-preserved side of a RIGHT join. Rows with any null
      // key cannot match under null_equality::UNEQUAL and produce no output,
      // so do not funnel them into the same skewed hash bucket.
      nonNullProbeRows = cudf::drop_nulls(
          partitionInput, leftKeyIndices_, stream, get_output_mr());
      partitionInput = nonNullProbeRows->view();
    }
    auto [partitioned, offsets] = [&]() {
      std::lock_guard<std::mutex> lock(cudfHashPartitionMutex());
      auto result = cudf::hash_partition(
          partitionInput,
          leftKeyIndices_,
          numPartitions,
          cudf::hash_id::HASH_MURMUR3,
          kGraceLocalHashSeed,
          stream,
          get_temp_mr());
      stream.synchronize();
      return result;
    }();
    VELOX_CHECK(
        offsets.size() == numPartitions || offsets.size() == numPartitions + 1);
    VELOX_CHECK_EQ(offsets.front(), 0);
    offsets.erase(offsets.begin());
    if (offsets.size() == numPartitions) {
      offsets.pop_back();
    }
    packHashJoinPartitions(
        partitioned->view(),
        offsets,
        stream,
        get_output_mr(),
        [&](size_t partition, HashJoinHostBatch host) {
          VELOX_CHECK_LT(partition, graceProbePartitions_.size());
          if (host.rows == 0) {
            return;
          }
          spillGraceProbeBatch(host, partition);
          graceProbePartitions_[partition].push_back(std::move(host));
        });
  }
}

void CudfHashJoinProbe::flushGraceProbeInputBatch() {
  if (graceProbeInputs_.empty()) {
    return;
  }
  auto batch = std::exchange(graceProbeInputs_, {});
  graceProbeInputBytes_ = 0;
  if (batch.size() == 1) {
    partitionAndPackProbe(std::move(batch.front()));
    return;
  }
  const auto stream = batch.front()->stream();
  auto table = getConcatenatedTable(
      std::move(batch), probeType_, stream, get_output_mr());
  const auto rows = table->num_rows();
  auto combined = std::make_shared<CudfVector>(
      pool(), probeType_, rows, std::move(table), stream);
  partitionAndPackProbe(std::move(combined));
}

void CudfHashJoinProbe::queueGraceProbeInput(CudfVectorPtr input) {
  VELOX_CHECK_NOT_NULL(input);
  ++graceProbeSourceBatches_;
  const auto inputBytes = retainedCudfBytes(*input);
  // Split only when the empty/non-empty chars pattern changes. Uniform
  // all-empty groups remain safe to concatenate and avoid fragmented
  // partition/pack launches.
  if (!graceProbeInputs_.empty() &&
      !hasSameEmptyStringCharsPattern(
          graceProbeInputs_.front()->getTableView(), input->getTableView())) {
    flushGraceProbeInputBatch();
  }
  if (!graceProbeInputs_.empty() &&
      inputBytes > kGracePartitionBatchBytes -
              std::min(graceProbeInputBytes_, kGracePartitionBatchBytes)) {
    flushGraceProbeInputBatch();
  }
  // partitionAndPackProbe() row-slices an oversized vector without copying it.
  if (inputBytes >= kGracePartitionBatchBytes) {
    partitionAndPackProbe(std::move(input));
    return;
  }
  graceProbeInputBytes_ += inputBytes;
  graceProbeInputs_.push_back(std::move(input));
  if (graceProbeInputBytes_ >= kGracePartitionBatchBytes) {
    flushGraceProbeInputBatch();
  }
}

CudfVectorPtr CudfHashJoinProbe::restoreHostBatch(
    HashJoinHostBatch& batch,
    const RowTypePtr& type,
    rmm::cuda_stream_view stream,
    bool consume) {
  if (batch.spillWriteFuture.valid()) {
    batch.spillWriteFuture.get();
  }
  auto gpuData = std::make_unique<rmm::device_buffer>(
      batch.dataBytes, stream, get_output_mr());
  // Reserve bounce capacity for storage reads. Host-resident Grace batches
  // were deliberately demoted from the bounded D2H staging pool into pageable
  // memory when they were stored. Copying that complete image back into a
  // pinned slab before every restore adds a second host memcpy and competes
  // with disk prefetch for the same four process-wide slots. The pageable
  // owner remains alive through the synchronization below, so direct H2D is
  // both ownership-safe and faster for this already-materialized case.
  auto pinnedStaging = !batch.pinned && batch.spillFile
      ? gracePinnedHostStagingPool().acquire(batch.dataBytes)
      : GracePinnedHostStagingLease{};
  auto* pinnedData = batch.pinned ? batch.data.get() : pinnedStaging.data();
  std::unique_ptr<uint8_t[]> diskData;
  const uint8_t* sourceData = batch.data.get();
  const auto hostStageStart = std::chrono::steady_clock::now();
  if (batch.spillFile) {
    // A probe read-ahead stores the raw range directly in a pooled pinned
    // buffer and leaves spillFile set so consume still reclaims the range.
    if (batch.pinned && batch.data != nullptr) {
      sourceData = batch.data.get();
    } else if (pinnedData != nullptr) {
      batch.spillFile->read(batch.fileOffset, batch.dataBytes, pinnedData);
      sourceData = pinnedData;
    } else {
      diskData = std::unique_ptr<uint8_t[]>(new uint8_t[batch.dataBytes]);
      batch.spillFile->read(batch.fileOffset, batch.dataBytes, diskData.get());
      sourceData = diskData.get();
    }
  }
  const auto hostStageEnd = std::chrono::steady_clock::now();
  VELOX_CHECK(
      batch.dataBytes == 0 || sourceData != nullptr,
      "Grace hash join batch has neither host data nor a spill file");
  const auto usedPinnedSource = batch.dataBytes > 0 &&
      ((batch.pinned && batch.data != nullptr) || pinnedStaging);
  if (batch.spillFile) {
    graceRestoreDiskBytes_ += batch.dataBytes;
  } else {
    graceRestoreResidentBytes_ += batch.dataBytes;
  }
  if (usedPinnedSource) {
    graceRestorePinnedSourceBytes_ += batch.dataBytes;
  } else {
    graceRestorePageableDirectBytes_ += batch.dataBytes;
  }
  graceRestoreHostStageMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          hostStageEnd - hostStageStart)
          .count();
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      gpuData->data(),
      sourceData,
      batch.dataBytes,
      cudaMemcpyHostToDevice,
      stream.value()));
  const auto copySynchronizeStart = std::chrono::steady_clock::now();
  stream.synchronize();
  graceRestoreCopySynchronizeMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - copySynchronizeStart)
          .count();
  if (consume && batch.spillFile) {
    batch.spillFile->reclaim(batch.fileOffset, batch.dataBytes);
  }
  auto metadata = consume
      ? std::move(batch.metadata)
      : std::make_unique<std::vector<uint8_t>>(*batch.metadata);
  cudf::packed_columns columns{std::move(metadata), std::move(gpuData)};
  auto view = cudf::unpack(columns);
  auto packed = std::make_unique<cudf::packed_table>(
      cudf::packed_table{view, std::move(columns)});
  const auto rows = batch.rows;
  if (consume) {
    batch.data.reset();
    batch.hostReservation.reset();
    batch.spillFile.reset();
    batch.spillWriteFuture = {};
    batch.fileOffset = 0;
    batch.dataBytes = 0;
    batch.rows = 0;
  }
  return std::make_shared<CudfVector>(
      pool(), type, rows, std::move(packed), stream);
}

void CudfHashJoinProbe::scheduleGraceProbePrefetch(
    std::vector<HashJoinHostBatch>& batches,
    size_t chunk) {
  if (chunk >= batches.size()) {
    return;
  }
  VELOX_CHECK(
      std::none_of(
          graceProbePrefetchFutures_.begin(),
          graceProbePrefetchFutures_.end(),
          [&](const auto& prefetch) { return prefetch.first == chunk; }),
      "Grace probe chunk {} was scheduled for prefetch twice",
      chunk);
  auto& batch = batches[chunk];
  if (!batch.spillFile || batch.data != nullptr || batch.dataBytes == 0) {
    return;
  }
  auto pinned = gracePinnedHostStagingPool().acquireShared(batch.dataBytes);
  if (!pinned) {
    return;
  }
  batch.data = pinned;
  batch.pinned = true;
  const auto spillFile = batch.spillFile;
  const auto spillWriteFuture = batch.spillWriteFuture;
  const auto fileOffset = batch.fileOffset;
  const auto dataBytes = batch.dataBytes;
  std::promise<void> completion;
  auto future = completion.get_future();
  graceSpillReadExecutor().add([spillFile,
                                spillWriteFuture,
                                fileOffset,
                                dataBytes,
                                pinned = std::move(pinned),
                                completion = std::move(completion)]() mutable {
    try {
      if (spillWriteFuture.valid()) {
        spillWriteFuture.get();
      }
      spillFile->read(fileOffset, dataBytes, pinned.get());
      completion.set_value();
    } catch (...) {
      completion.set_exception(std::current_exception());
    }
  });
  graceProbePrefetchFutures_.emplace_back(chunk, std::move(future));
}

void CudfHashJoinProbe::waitForGraceProbePrefetch(size_t chunk) {
  const auto prefetch = std::find_if(
      graceProbePrefetchFutures_.begin(),
      graceProbePrefetchFutures_.end(),
      [&](const auto& entry) { return entry.first == chunk; });
  if (prefetch == graceProbePrefetchFutures_.end()) {
    return;
  }
  const auto waitStart = std::chrono::steady_clock::now();
  prefetch->second.get();
  gracePartitionProbePrefetchWaitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - waitStart)
          .count();
  graceProbePrefetchFutures_.erase(prefetch);
}

void CudfHashJoinProbe::finishGraceProbePrefetch() {
  for (auto& [chunk, future] : graceProbePrefetchFutures_) {
    future.get();
  }
  graceProbePrefetchFutures_.clear();
}

void CudfHashJoinProbe::initializeGracePartitionQueue() {
  if (gracePartitionQueueInitialized_) {
    return;
  }
  VELOX_CHECK_NOT_NULL(graceBuildData_);
  VELOX_CHECK_EQ(
      graceBuildData_->partitions.size(), graceProbePartitions_.size());
  VELOX_CHECK_LE(
      graceBuildData_->partitions.size(),
      1U << exec::SpillPartitionId::kMaxPartitionBits,
      "The shared Velox spill lifecycle supports at most {} partitions per "
      "level; use recursive repartition instead of a flat fanout",
      1U << exec::SpillPartitionId::kMaxPartitionBits);

  GraceHashJoinPartitionSet partitions;
  for (uint32_t partition = 0; partition < graceBuildData_->partitions.size();
       ++partition) {
    auto id = exec::SpillPartitionId(partition);
    auto payload = std::make_unique<GraceHashJoinPartition>(id);
    payload->build = std::move(graceBuildData_->partitions[partition]);
    payload->probe = std::move(graceProbePartitions_[partition]);
    partitions.emplace(id, std::move(payload));
  }
  graceBuildData_->partitions.clear();
  graceProbePartitions_.clear();

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
  cudfHashJoinBridge->setGracePartitions(std::move(partitions));
  gracePartitionQueueInitialized_ = true;
}

GraceHashJoinPartitionSet CudfHashJoinProbe::repartitionGracePartition(
    GraceHashJoinPartition& partition,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_LT(
      partition.id().spillLevel(),
      exec::SpillPartitionId::kMaxSpillLevel,
      "Grace hash join partition {} remains larger than {} bytes at the "
      "maximum recursive spill level",
      partition.id(),
      graceRestoreBuildBytes_);
  constexpr uint32_t kChildren = 1U
      << exec::SpillPartitionId::kMaxPartitionBits;
  // Spill files become read-only after the first restore. Each recursive
  // parent therefore writes its children into fresh append-only files. Batches
  // retain shared ownership of older files until their ranges are consumed.
  graceRecursiveBuildSpillFiles_.clear();
  graceProbeSpillFiles_.clear();
  GraceHashJoinPartitionSet children;
  std::array<GraceHashJoinPartition*, kChildren> childPayloads;
  for (uint32_t child = 0; child < kChildren; ++child) {
    auto id = exec::SpillPartitionId(partition.id(), child);
    auto payload = std::make_unique<GraceHashJoinPartition>(id);
    childPayloads[child] = payload.get();
    children.emplace(id, std::move(payload));
  }

  const auto repartition = [&](std::vector<HashJoinHostBatch>& batches,
                               const RowTypePtr& type,
                               const std::vector<cudf::size_type>& keyIndices,
                               bool buildSide) {
    for (auto& batch : batches) {
      if (buildSide) {
        accountConsumedGraceBuildBatch(batch);
      } else {
        accountConsumedGraceProbeBatch(batch);
      }
      auto input = restoreHostBatch(batch, type, stream, true);
      auto [partitioned, offsets] = [&]() {
        std::lock_guard<std::mutex> lock(cudfHashPartitionMutex());
        auto result = cudf::hash_partition(
            input->getTableView(),
            keyIndices,
            kChildren,
            cudf::hash_id::HASH_MURMUR3,
            partition.id().spillLevel() + 1,
            stream,
            get_temp_mr());
        stream.synchronize();
        return result;
      }();
      VELOX_CHECK(
          offsets.size() == kChildren || offsets.size() == kChildren + 1);
      VELOX_CHECK_EQ(offsets.front(), 0);
      offsets.erase(offsets.begin());
      if (offsets.size() == kChildren) {
        offsets.pop_back();
      }
      packHashJoinPartitions(
          partitioned->view(),
          offsets,
          stream,
          get_output_mr(),
          [&](size_t child, HashJoinHostBatch output) {
            VELOX_CHECK_LT(child, kChildren);
            if (output.rows == 0) {
              return;
            }
            if (buildSide) {
              spillRecursiveGraceBuildBatch(output, child);
              childPayloads[child]->build.push_back(std::move(output));
            } else {
              spillGraceProbeBatch(output, child);
              childPayloads[child]->probe.push_back(std::move(output));
            }
          });
    }
    batches.clear();
  };

  uint64_t buildBytes = 0;
  uint64_t probeBytes = 0;
  for (const auto& batch : partition.build) {
    buildBytes += batch.dataBytes;
  }
  for (const auto& batch : partition.probe) {
    probeBytes += batch.dataBytes;
  }
  repartition(partition.build, buildType_, rightKeyIndices_, true);
  repartition(partition.probe, probeType_, leftKeyIndices_, false);
  LOG(WARNING) << "CudfHashJoinProbe node=" << planNodeId()
               << " recursively repartitioned Grace partition="
               << partition.id() << " buildBytes=" << buildBytes
               << " probeBytes=" << probeBytes << " children=" << kChildren;
  return children;
}

void CudfHashJoinProbe::loadGraceBuildPartition(
    GraceHashJoinPartition& partition,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(graceBuildData_);
  auto& hostBatches = partition.build;
  uint64_t buildBytes = 0;
  uint64_t buildRows = 0;
  for (const auto& host : hostBatches) {
    buildBytes += host.dataBytes;
    buildRows += host.rows;
  }
  const auto useBulkRestore = graceBulkBuildRestoreEnabled();
  CudfBulkPackedRestore bulkRestored;
  CudfBulkPackedRestoreStats bulkRestoreStats;
  std::vector<CudfVectorPtr> restored;
  const auto restoreStart = std::chrono::steady_clock::now();
  if (useBulkRestore) {
    // nextGracePartition() transfers one partition to exactly one probe
    // driver. Consume its host batches into one aligned device allocation.
    // Disk materializers fill two bounded pinned slabs in alternation while a
    // dedicated copy stream transfers the previous slab. Resident pageable
    // batches retain their direct H2D path.
    std::vector<CudfPackedHostRestoreChunk> chunks;
    chunks.reserve(hostBatches.size());
    for (auto& host : hostBatches) {
      // Publication normally drains asynchronous resident demotions. Keep
      // the ordering local as well so future producers can safely hand a
      // resident batch directly to the bulk restore path.
      if (!host.spillFile && host.spillWriteFuture.valid()) {
        host.spillWriteFuture.get();
        host.spillWriteFuture = {};
      }
      CudfPackedHostRestoreChunk chunk;
      chunk.metadata = std::move(host.metadata);
      chunk.data = std::move(host.data);
      chunk.dataBytes = host.dataBytes;
      chunk.keepAlive = std::move(host.hostReservation);
      if (host.spillFile) {
        graceRestoreDiskBytes_ += host.dataBytes;
        auto spillFile = std::move(host.spillFile);
        auto spillWriteFuture = std::move(host.spillWriteFuture);
        const auto fileOffset = host.fileOffset;
        const auto dataBytes = host.dataBytes;
        chunk.data.reset();
        chunk.materializeIntoPinned =
            [spillFile = std::move(spillFile),
             spillWriteFuture = std::move(spillWriteFuture),
             fileOffset,
             dataBytes](uint8_t* destination) mutable {
              if (spillWriteFuture.valid()) {
                spillWriteFuture.get();
              }
              spillFile->read(fileOffset, dataBytes, destination);
              spillFile->reclaim(fileOffset, dataBytes);
            };
      } else {
        graceRestoreResidentBytes_ += host.dataBytes;
        if (gracePageableRestoreBounceEnabled() && !host.pinned) {
          chunk.stageResidentPageableThroughPinned = true;
        }
      }
      host.fileOffset = 0;
      host.dataBytes = 0;
      host.rows = 0;
      host.pinned = false;
      chunks.push_back(std::move(chunk));
    }
    bulkRestored = bulkRestoreCudfPackedHostChunks(
        std::move(chunks), stream, get_output_mr());
    bulkRestoreStats = bulkRestored.stats();
    graceRestorePinnedSourceBytes_ += bulkRestoreStats.pinnedBounceBytes;
    graceRestoreResidentBounceBytes_ +=
        bulkRestoreStats.residentPageableBounceBytes;
    graceRestorePageableDirectBytes_ += bulkRestoreStats.pageableDirectBytes;
    graceRestoreHostStageMicros_ += bulkRestoreStats.hostStageMicros;
    graceRestoreCopySynchronizeMicros_ +=
        bulkRestoreStats.copyStreamSynchronizeMicros;
  } else {
    restored.reserve(hostBatches.size());
    // Compatibility path for same-binary A/B. A partition is single-owner,
    // but retain the historical per-batch metadata copy and H2D sync here so
    // the switch isolates only the bulk restore implementation.
    for (auto& host : hostBatches) {
      restored.push_back(restoreHostBatch(host, buildType_, stream, false));
    }
  }
  const auto restoreEnd = std::chrono::steady_clock::now();
  std::unique_ptr<cudf::table> table;
  if (useBulkRestore && !bulkRestored.tables().empty()) {
    table = cudf::concatenate(bulkRestored.tables(), stream, get_output_mr());
  } else {
    table = getConcatenatedTable(
        std::move(restored), buildType_, stream, get_output_mr());
  }
  const auto concatenateEnd = std::chrono::steady_clock::now();
  auto sharedTable = std::shared_ptr<cudf::table>(std::move(table));
  std::shared_ptr<cudf::hash_join> hash;
  {
    // Grace joins construct one hash table per restored partition.  They must
    // obey the same cuco serialization and completion contract as the regular
    // build path; otherwise concurrent fragment builds can corrupt cuco's
    // counter storage before the asynchronous constructor has completed.
    std::lock_guard<std::mutex> cucoLock(cudfCucoMutex());
    const auto hashJoinLoadFactor =
        operatorCtx_->driverCtx()->queryConfig().get<double>(
            CudfConfig::kCudfHashJoinLoadFactor,
            CudfConfig::getInstance().hashJoinLoadFactor);
    hash = makeSharedHashJoin(
        sharedTable->view().select(rightKeyIndices_),
        cudf::null_equality::UNEQUAL,
        hashJoinLoadFactor,
        stream);
    stream.synchronize();
  }
  const auto hashEnd = std::chrono::steady_clock::now();
  hashObject_ = std::make_pair(
      std::vector<std::shared_ptr<cudf::table>>{std::move(sharedTable)},
      std::vector<std::shared_ptr<cudf::hash_join>>{std::move(hash)});
  buildStream_ = stream;
  if (joinNode_->isRightJoin()) {
    initializeRightMatchedFlags(stream);
  }
  LOG(WARNING) << "CudfHashJoinProbe node=" << planNodeId()
               << " loaded Grace build partition=" << partition.id()
               << " batches=" << hostBatches.size() << " rows=" << buildRows
               << " bytes=" << buildBytes << " bulkRestore=" << useBulkRestore
               << " pinnedBounceBytes=" << bulkRestoreStats.pinnedBounceBytes
               << " residentPageableBounceBytes="
               << bulkRestoreStats.residentPageableBounceBytes
               << " pageableDirectBytes="
               << bulkRestoreStats.pageableDirectBytes
               << " pinnedBounceCopies=" << bulkRestoreStats.pinnedBounceCopies
               << " hostStageUs=" << bulkRestoreStats.hostStageMicros
               << " bounceReuseWaitUs="
               << bulkRestoreStats.bounceReuseWaitMicros << " copyStreamSyncUs="
               << bulkRestoreStats.copyStreamSynchronizeMicros
               << " restoreH2DUs="
               << std::chrono::duration_cast<std::chrono::microseconds>(
                      restoreEnd - restoreStart)
                      .count()
               << " concatenateUs="
               << std::chrono::duration_cast<std::chrono::microseconds>(
                      concatenateEnd - restoreEnd)
                      .count()
               << " hashBuildUs="
               << std::chrono::duration_cast<std::chrono::microseconds>(
                      hashEnd - concatenateEnd)
                      .count();
}

uint64_t CudfHashJoinProbe::estimateGraceBuildWorkspaceBytes(
    const GraceHashJoinPartition& partition,
    bool recursiveRepartition) const {
  // Recursive repartition already works on bounded 256-MiB slices. Reserving
  // the complete oversized parent would prevent the operation which is meant
  // to make that parent fit.
  if (recursiveRepartition) {
    return 1ULL << 30;
  }

  uint64_t buildBytes = 0;
  for (const auto& batch : partition.build) {
    buildBytes += batch.dataBytes;
  }
  constexpr uint64_t kFixedWorkspaceBytes = 512ULL << 20;
  if (buildBytes >
      (std::numeric_limits<uint64_t>::max() - kFixedWorkspaceBytes) / 2) {
    return std::numeric_limits<uint64_t>::max();
  }
  // One restored build image, one hash-sized image, plus fixed cuDF/cuco
  // scratch. Probe workspace is admitted separately for each chunk after the
  // hash table is resident.
  return std::max<uint64_t>(1ULL << 30, buildBytes * 2 + kFixedWorkspaceBytes);
}

uint64_t CudfHashJoinProbe::estimateGraceProbeWorkspaceBytes(
    uint64_t probeBytes) const {
  constexpr uint64_t kFixedWorkspaceBytes = 512ULL << 20;
  constexpr uint64_t kWorkspaceCopies = 2;
  const auto boundedProbeBytes = std::min<uint64_t>(probeBytes, 1ULL << 30);
  if (boundedProbeBytes >
      (std::numeric_limits<uint64_t>::max() - kFixedWorkspaceBytes) /
          kWorkspaceCopies) {
    return std::numeric_limits<uint64_t>::max();
  }
  // The restored build and hash table are already live and therefore already
  // reduce physical headroom. Reserve only this probe chunk, its gathered
  // output/scratch image, and fixed cuDF/cuco workspace.
  return std::max<uint64_t>(
      1ULL << 30, boundedProbeBytes * kWorkspaceCopies + kFixedWorkspaceBytes);
}

bool CudfHashJoinProbe::acquireGraceWorkspace(
    uint64_t bytes,
    DeviceMemoryWorkspacePriority priority) {
  if (graceWorkspaceAdmission_.has_value()) {
    return true;
  }
  auto attempt = graceWorkspace_.tryAcquire(
      customPool(kCudfDeviceMemoryResourceTag),
      this,
      bytes,
      CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
      priority);
  auto workspaceAdmission = std::move(attempt.reservation);
  if (!workspaceAdmission.has_value()) {
    return false;
  }
  if (attempt.completedWaitMicros.has_value()) {
    gracePartitionWorkspaceWaitMicros_ += attempt.completedWaitMicros.value();
  }
  graceWorkspaceAdmission_ = std::move(workspaceAdmission.value());
  return true;
}

RowVectorPtr CudfHashJoinProbe::getGraceOutput() {
  if (graceOutputHandoffPending_) {
    gracePartitionDownstreamHandoffMicros_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - graceOutputHandoffStart_)
            .count();
    graceOutputHandoffPending_ = false;
  }
  if (joinNode_->isRightJoin() && (!noMoreInput_ || !isLastDriver_)) {
    return nullptr;
  }
  if ((!noMoreInput_ && !graceProbeDraining_) || graceFinished_) {
    return nullptr;
  }
  initializeGracePartitionQueue();
  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfHashJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfHashJoinBridge);
  auto stream = cudfGlobalStreamPool().get_stream();
  while (true) {
    if (!gracePartition_.has_value()) {
      gracePartition_ = cudfHashJoinBridge->nextGracePartition();
      if (!gracePartition_.has_value()) {
        break;
      }
      graceProbeChunk_ = 0;
    }
    uint64_t buildBytes = 0;
    for (const auto& batch : gracePartition_->build) {
      buildBytes += batch.dataBytes;
    }
    auto& probeBatches = gracePartition_->probe;
    const auto buildEmpty = gracePartition_->build.empty();
    if (probeBatches.empty() && (!joinNode_->isRightJoin() || buildEmpty)) {
      gracePartition_.reset();
      graceProbeChunk_ = 0;
      continue;
    }
    if (joinNode_->isInnerJoin() && buildEmpty) {
      for (const auto& batch : probeBatches) {
        accountConsumedGraceProbeBatch(batch);
      }
      probeBatches.clear();
      gracePartition_.reset();
      graceProbeChunk_ = 0;
      continue;
    }
    if (buildBytes > graceRestoreBuildBytes_) {
      if (!acquireGraceWorkspace(
              estimateGraceBuildWorkspaceBytes(*gracePartition_, true),
              DeviceMemoryWorkspacePriority::kDrain)) {
        return nullptr;
      }
      auto children = repartitionGracePartition(*gracePartition_, stream);
      cudfHashJoinBridge->appendGracePartitions(std::move(children));
      gracePartition_.reset();
      graceProbeChunk_ = 0;
      graceWorkspaceAdmission_.reset();
      continue;
    }
    if (!hashObject_.has_value()) {
      if (!acquireGraceWorkspace(
              estimateGraceBuildWorkspaceBytes(*gracePartition_, false),
              DeviceMemoryWorkspacePriority::kRestore)) {
        return nullptr;
      }
      gracePartitionDrainStart_ = std::chrono::steady_clock::now();
      gracePartitionProbeBytes_ = 0;
      gracePartitionProbeChunks_ = 0;
      gracePartitionProbeGroups_ = 0;
      gracePartitionProbeRestoreMicros_ = 0;
      gracePartitionProbePrefetchWaitMicros_ = 0;
      gracePartitionJoinMicros_ = 0;
      gracePartitionOutputMicros_ = 0;
      gracePartitionBuildLoadMicros_ = 0;
      // Build admission happened before the partition drain clock started.
      // Only account arbitration waits that can explain time inside totalUs.
      gracePartitionWorkspaceWaitMicros_ = 0;
      gracePartitionFinalSyncMicros_ = 0;
      gracePartitionDownstreamHandoffMicros_ = 0;
      gracePartitionUnmatchedEmitted_ = false;
      // Overlap the first sequential probe pread with build restore,
      // concatenate, and hash-table construction.
      for (size_t chunk = 0;
           chunk < std::min(graceProbePrefetchDepth_, probeBatches.size());
           ++chunk) {
        scheduleGraceProbePrefetch(probeBatches, chunk);
      }
      const auto buildLoadStart = std::chrono::steady_clock::now();
      loadGraceBuildPartition(*gracePartition_, stream);
      gracePartitionBuildLoadMicros_ =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - buildLoadStart)
              .count();
      // All build/hash allocations now exist and are visible in physical
      // headroom. Do not retain their transient construction lease while the
      // driver waits on downstream output or probe prefetch.
      graceWorkspaceAdmission_.reset();
    }
    if (graceProbeChunk_ >= probeBatches.size()) {
      if (joinNode_->isRightJoin() && !gracePartitionUnmatchedEmitted_) {
        if (!acquireGraceWorkspace(
                estimateGraceProbeWorkspaceBytes(buildBytes),
                DeviceMemoryWorkspacePriority::kDrain)) {
          return nullptr;
        }
        gracePartitionUnmatchedEmitted_ = true;
        auto unmatched = rightUnmatchedOutput(stream);
        graceWorkspaceAdmission_.reset();
        if (unmatched != nullptr) {
          return unmatched;
        }
      }
      const auto finalSyncStart = std::chrono::steady_clock::now();
      stream.synchronize();
      const auto partitionEnd = std::chrono::steady_clock::now();
      gracePartitionFinalSyncMicros_ =
          std::chrono::duration_cast<std::chrono::microseconds>(
              partitionEnd - finalSyncStart)
              .count();
      const auto totalMicros =
          std::chrono::duration_cast<std::chrono::microseconds>(
              partitionEnd - gracePartitionDrainStart_)
              .count();
      const auto accountedMicros = gracePartitionBuildLoadMicros_ +
          gracePartitionWorkspaceWaitMicros_ +
          gracePartitionProbeRestoreMicros_ +
          gracePartitionProbePrefetchWaitMicros_ + gracePartitionJoinMicros_ +
          gracePartitionOutputMicros_ + gracePartitionDownstreamHandoffMicros_ +
          gracePartitionFinalSyncMicros_;
      LOG(WARNING)
          << "CudfHashJoinProbe node=" << planNodeId()
          << " completed Grace partition=" << gracePartition_->id()
          << " probeChunks=" << gracePartitionProbeChunks_
          << " probeGroups=" << gracePartitionProbeGroups_
          << " probeBytes=" << gracePartitionProbeBytes_
          << " probeRestoreH2DUs=" << gracePartitionProbeRestoreMicros_
          << " probePrefetchWaitUs=" << gracePartitionProbePrefetchWaitMicros_
          << " joinUs=" << gracePartitionJoinMicros_
          << " outputUs=" << gracePartitionOutputMicros_
          << " buildLoadUs=" << gracePartitionBuildLoadMicros_
          << " workspaceWaitUs=" << gracePartitionWorkspaceWaitMicros_
          << " finalSyncUs=" << gracePartitionFinalSyncMicros_
          << " downstreamHandoffUs=" << gracePartitionDownstreamHandoffMicros_
          << " accountedUs=" << accountedMicros << " unaccountedUs="
          << (totalMicros > accountedMicros ? totalMicros - accountedMicros : 0)
          << " totalUs=" << totalMicros;
      hashObject_.reset();
      probeBatches.clear();
      gracePartition_.reset();
      graceProbeChunk_ = 0;
      gracePartitionUnmatchedEmitted_ = false;
      graceWorkspaceAdmission_.reset();
      continue;
    }

    // Keep input partitioning bounded, but drain several adjacent packed
    // chunks as one logical output group. This preserves the partition-major
    // lifecycle while avoiding hundreds of tiny join-to-downstream handoffs.
    // The global device arbitrator admits the complete group at kDrain
    // priority, so two drivers cannot independently overcommit the GPU.
    const auto groupBegin = graceProbeChunk_;
    auto groupEnd = groupBegin;
    uint64_t groupBytes = 0;
    while (groupEnd < probeBatches.size()) {
      const auto batchBytes = probeBatches[groupEnd].dataBytes;
      if (groupEnd > groupBegin &&
          batchBytes > graceRestoreProbeBytes_ -
                  std::min(groupBytes, graceRestoreProbeBytes_)) {
        break;
      }
      groupBytes += batchBytes;
      ++groupEnd;
      if (groupBytes >= graceRestoreProbeBytes_) {
        break;
      }
    }
    VELOX_CHECK_GT(groupEnd, groupBegin);
    if (!acquireGraceWorkspace(
            estimateGraceProbeWorkspaceBytes(groupBytes),
            DeviceMemoryWorkspacePriority::kDrain)) {
      return nullptr;
    }
    vector_size_t zeroColumnRows = 0;
    std::vector<std::unique_ptr<cudf::table>> tables;
    uint64_t groupRestoreMicros = 0;
    uint64_t groupJoinMicros = 0;
    const auto joinProbe = [&](cudf::table_view probeView) {
      const auto joinStart = std::chrono::steady_clock::now();
      std::vector<JoinOutput> outputs;
      {
        std::lock_guard<std::mutex> cucoLock(cudfCucoMutex());
        if (joinNode_->isInnerJoin()) {
          outputs = innerJoin(probeView, stream);
        } else if (joinNode_->isLeftJoin()) {
          outputs = leftJoin(probeView, stream);
        } else {
          VELOX_CHECK(joinNode_->isRightJoin());
          outputs = rightJoin(probeView, stream);
        }
        stream.synchronize();
      }
      groupJoinMicros += std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - joinStart)
                             .count();
      for (auto& output : outputs) {
        zeroColumnRows += output.numRows;
        tables.push_back(std::move(output.table));
      }
    };
    if (!graceBulkProbeRestoreEnabled()) {
      for (; graceProbeChunk_ < groupEnd; ++graceProbeChunk_) {
        waitForGraceProbePrefetch(graceProbeChunk_);
        auto& hostProbe = probeBatches[graceProbeChunk_];
        const auto restoredBytes = hostProbe.dataBytes;
        accountConsumedGraceProbeBatch(hostProbe);
        const auto probeRestoreStart = std::chrono::steady_clock::now();
        auto probe = restoreHostBatch(hostProbe, probeType_, stream, true);
        groupRestoreMicros +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - probeRestoreStart)
                .count();
        // While this GPU join runs, read the next raw range into a different
        // pooled pinned buffer.
        scheduleGraceProbePrefetch(
            probeBatches, graceProbeChunk_ + graceProbePrefetchDepth_);
        joinProbe(probe->getTableView());
        probe.reset();
        gracePartitionProbeBytes_ += restoredBytes;
        ++gracePartitionProbeChunks_;
      }
    } else {
      // Preserve the existing bounded prefetch pipeline, but restore one
      // prefetch-depth wave into a shared aligned device allocation. Four
      // approximately 28-MiB Job 144 chunks therefore need one allocation and
      // one stream synchronization rather than four of each. The joins remain
      // chunk-granular so output cardinality and peak memory do not change.
      while (graceProbeChunk_ < groupEnd) {
        const auto waveBegin = graceProbeChunk_;
        const auto waveEnd =
            std::min<size_t>(groupEnd, waveBegin + graceProbePrefetchDepth_);
        for (auto chunkIndex = waveBegin; chunkIndex < waveEnd; ++chunkIndex) {
          waitForGraceProbePrefetch(chunkIndex);
        }
        const auto hasUnmaterializedDisk = std::any_of(
            probeBatches.begin() + waveBegin,
            probeBatches.begin() + waveEnd,
            [](const auto& batch) {
              return batch.spillFile && batch.data == nullptr;
            });
        if (hasUnmaterializedDisk) {
          // A 4-chunk wave which must still perform host staging is too small
          // to alternate the two 128-MiB generic bounce slabs. Controlled A/B
          // showed that path regresses; preserve the original chunk pipeline
          // so pread of chunk N+depth overlaps the current GPU join.
          for (auto chunkIndex = waveBegin; chunkIndex < waveEnd;
               ++chunkIndex) {
            auto& hostProbe = probeBatches[chunkIndex];
            const auto restoredBytes = hostProbe.dataBytes;
            accountConsumedGraceProbeBatch(hostProbe);
            const auto probeRestoreStart = std::chrono::steady_clock::now();
            auto probe = restoreHostBatch(hostProbe, probeType_, stream, true);
            groupRestoreMicros +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - probeRestoreStart)
                    .count();
            scheduleGraceProbePrefetch(
                probeBatches, chunkIndex + graceProbePrefetchDepth_);
            joinProbe(probe->getTableView());
            probe.reset();
            gracePartitionProbeBytes_ += restoredBytes;
            ++gracePartitionProbeChunks_;
            graceBulkProbeFallbackBytes_ += restoredBytes;
          }
          graceProbeChunk_ = waveEnd;
          continue;
        }
        std::vector<CudfPackedHostRestoreChunk> chunks;
        chunks.reserve(waveEnd - waveBegin);
        uint64_t pinnedDirectBytes = 0;
        for (auto chunkIndex = waveBegin; chunkIndex < waveEnd; ++chunkIndex) {
          auto& hostProbe = probeBatches[chunkIndex];
          const auto restoredBytes = hostProbe.dataBytes;
          accountConsumedGraceProbeBatch(hostProbe);
          CudfPackedHostRestoreChunk chunk;
          chunk.metadata = std::move(hostProbe.metadata);
          chunk.data = std::move(hostProbe.data);
          chunk.dataBytes = restoredBytes;
          chunk.keepAlive = std::move(hostProbe.hostReservation);
          if (hostProbe.spillFile) {
            graceRestoreDiskBytes_ += restoredBytes;
            auto spillFile = std::move(hostProbe.spillFile);
            const auto fileOffset = hostProbe.fileOffset;
            VELOX_CHECK(
                restoredBytes == 0 || chunk.data != nullptr,
                "Bulk Grace probe restore requires a resident or prefetched "
                "host image");
            if (hostProbe.pinned) {
              pinnedDirectBytes += restoredBytes;
            }
            spillFile->reclaim(fileOffset, restoredBytes);
          } else {
            graceRestoreResidentBytes_ += restoredBytes;
            if (gracePageableRestoreBounceEnabled() && !hostProbe.pinned) {
              chunk.stageResidentPageableThroughPinned = true;
            }
          }
          hostProbe.fileOffset = 0;
          hostProbe.spillWriteFuture = {};
          hostProbe.dataBytes = 0;
          hostProbe.rows = 0;
          hostProbe.pinned = false;
          gracePartitionProbeBytes_ += restoredBytes;
          ++gracePartitionProbeChunks_;
          graceBulkProbeRestoreBytes_ += restoredBytes;
          chunks.push_back(std::move(chunk));
        }
        const auto probeRestoreStart = std::chrono::steady_clock::now();
        auto restored = bulkRestoreCudfPackedHostChunks(
            std::move(chunks), stream, get_output_mr());
        const auto& stats = restored.stats();
        VELOX_CHECK_GE(stats.pageableDirectBytes, pinnedDirectBytes);
        graceRestorePinnedSourceBytes_ +=
            stats.pinnedBounceBytes + pinnedDirectBytes;
        graceRestoreResidentBounceBytes_ += stats.residentPageableBounceBytes;
        graceRestorePageableDirectBytes_ +=
            stats.pageableDirectBytes - pinnedDirectBytes;
        graceRestoreHostStageMicros_ += stats.hostStageMicros;
        graceRestoreCopySynchronizeMicros_ += stats.copyStreamSynchronizeMicros;
        groupRestoreMicros +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - probeRestoreStart)
                .count();
        graceProbeChunk_ = waveEnd;
        ++graceBulkProbeRestoreWaves_;
        // bulkRestore has completed the H2D wave and released its host source
        // owners. Refill those bounded prefetch slots while the current wave's
        // GPU joins execute.
        for (auto chunkIndex = waveBegin; chunkIndex < waveEnd; ++chunkIndex) {
          scheduleGraceProbePrefetch(
              probeBatches, chunkIndex + graceProbePrefetchDepth_);
        }
        for (const auto& probeView : restored.tables()) {
          joinProbe(probeView);
        }
      }
    }
    const auto outputStart = std::chrono::steady_clock::now();
    auto output = concatenateTables(std::move(tables), stream, get_output_mr());
    const auto outputEnd = std::chrono::steady_clock::now();
    ++gracePartitionProbeGroups_;
    gracePartitionProbeRestoreMicros_ += groupRestoreMicros;
    gracePartitionJoinMicros_ += groupJoinMicros;
    gracePartitionOutputMicros_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            outputEnd - outputStart)
            .count();
    const auto rows =
        outputType_->size() == 0 ? zeroColumnRows : output->num_rows();
    // Kernel allocations have been made and are now represented by physical
    // device usage. Release the virtual transient lease before handing output
    // to a possibly blocked downstream operator. The returned CudfVector owns
    // its actual buffers independently of this accounting token.
    graceWorkspaceAdmission_.reset();
    if (rows == 0) {
      continue;
    }
    graceOutputHandoffStart_ = std::chrono::steady_clock::now();
    graceOutputHandoffPending_ = true;
    return std::make_shared<CudfVector>(
        pool(), outputType_, rows, std::move(output), stream);
  }
  if (joinNode_->isRightJoin() &&
      graceNullBuildChunk_ < graceBuildData_->unmatchedNulls.size()) {
    const auto bytes =
        graceBuildData_->unmatchedNulls[graceNullBuildChunk_].dataBytes;
    if (!acquireGraceWorkspace(
            estimateGraceProbeWorkspaceBytes(bytes),
            DeviceMemoryWorkspacePriority::kDrain)) {
      return nullptr;
    }
    auto output = graceRightNullUnmatchedOutput(stream);
    graceWorkspaceAdmission_.reset();
    return output;
  }
  hashObject_.reset();
  graceWorkspace_.reset();
  graceWorkspaceAdmission_.reset();
  gracePartition_.reset();
  graceProbeChunk_ = 0;
  graceNullBuildChunk_ = 0;
  gracePartitionUnmatchedEmitted_ = false;
  graceProbeDraining_ = false;
  VELOX_CHECK_EQ(graceProbeBufferedBytes_, 0);
  VELOX_CHECK_EQ(graceProbeResidentHostBytes_, 0);
  VELOX_CHECK_EQ(graceProbeDiskBytes_, 0);
  graceProbeSpillFiles_.clear();
  LOG(WARNING) << "CudfHashJoinProbe node=" << planNodeId()
               << " completed Grace probe drain noMoreInput=" << noMoreInput_
               << " executorReservedHostBytes="
               << currentGraceHostMemoryReservedBytes()
               << " hostDemoteUs=" << graceHostDemoteMicros_
               << " rawSpillWriteUs=" << graceRawSpillWriteMicros_
               << " restoreResidentBytes=" << graceRestoreResidentBytes_
               << " restoreDiskBytes=" << graceRestoreDiskBytes_
               << " restorePinnedSourceBytes=" << graceRestorePinnedSourceBytes_
               << " restoreResidentBounceBytes="
               << graceRestoreResidentBounceBytes_
               << " restorePageableDirectBytes="
               << graceRestorePageableDirectBytes_
               << " restoreHostStageUs=" << graceRestoreHostStageMicros_
               << " restoreCopySyncUs=" << graceRestoreCopySynchronizeMicros_
               << " bulkProbeRestoreBytes=" << graceBulkProbeRestoreBytes_
               << " bulkProbeRestoreWaves=" << graceBulkProbeRestoreWaves_
               << " bulkProbeFallbackBytes=" << graceBulkProbeFallbackBytes_;
  if (noMoreInput_) {
    graceBuildData_.reset();
    graceProbePartitions_.clear();
    graceFinished_ = true;
    finished_ = true;
  }
  return nullptr;
}

void CudfHashJoinProbe::waitForBuildReady(rmm::cuda_stream_view stream) {
  if (buildReadyEvent_ != nullptr) {
    buildReadyEvent_->waitOn(stream);
  }
}

void CudfHashJoinProbe::initialize() {
  Operator::initialize();

  if (!joinNode_->filter()) {
    return;
  }

  auto* const pool = operatorCtx_->pool();

  // Optimize once so the filter evaluator and the two-table AST tree see the
  // same constant-folded form.
  const auto optimizedFilter = expression::optimize(
      joinNode_->filter(), operatorCtx_->execCtx()->queryCtx(), pool);

  // Disable AST-based filtering (and force precomputation) if the filter
  // expression contains a type the AST/JIT evaluator can't handle, using the
  // same shallow check applied during regular expression evaluation.
  if (containsAstUnsupportedType(optimizedFilter)) {
    useAstFilter_ = false;
  }

  // Validate AST filtering for this join type now to avoid run-time error.
  if (joinNode_->isRightSemiFilterJoin() || joinNode_->isLeftSemiFilterJoin() ||
      joinNode_->isAntiJoin()) {
    VELOX_CHECK(
        useAstFilter_,
        "AST expression evaluation must be enabled for semi-filter and anti joins.");
  }

  // Create a reusable evaluator for the filter column. This is expensive to
  // build, and the expression + input schema are stable for the lifetime of
  // the operator instance.
  std::vector<velox::RowTypePtr> filterRowTypes{probeType_, buildType_};
  filterEvaluator_ = createCudfExpression(
      optimizedFilter,
      facebook::velox::type::concatRowTypes(filterRowTypes),
      pool,
      &operatorCtx_->driverCtx()->queryConfig(),
      operatorCtx_->execCtx()->queryCtx());

  // Check if the filter expression spans both join sides (e.g., switch
  // expressions referencing columns from both probe and build). If so, we
  // cannot use AST-based filtering and must fall back to filterEvaluator_.
  if (hasNonAstSubexprSpanningBothSides(
          optimizedFilter, probeType_, buildType_)) {
    VLOG(1) << "Filter expression spans both join sides, using "
               "filterEvaluator_ instead of AST";
    useAstFilter_ = false;
    return;
  }

  // We don't need to get tables that contain conditional comparison columns
  // We'll pass the entire table. The ast will handle finding the required
  // columns. This is required because we build the ast with whole row schema
  // and the column locations in that schema translate to column locations
  // in whole tables

  if (useAstFilter_) {
    // create ast tree
    if (joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin()) {
      createAstTree(
          optimizedFilter,
          tree_,
          scalars_,
          buildType_,
          probeType_,
          rightPrecomputeInstructions_,
          leftPrecomputeInstructions_,
          pool);
    } else {
      createAstTree(
          optimizedFilter,
          tree_,
          scalars_,
          probeType_,
          buildType_,
          leftPrecomputeInstructions_,
          rightPrecomputeInstructions_,
          pool);
    }
  }
}

bool CudfHashJoinProbe::needsInput() const {
  if (joinNode_->isRightSemiFilterJoin()) {
    return !noMoreInput_;
  }
  if (graceEagerActive_ && !graceBuildData_ &&
      graceProbeBufferedBytes_ >= graceEagerProbeBufferLimitBytes_) {
    return false;
  }
  return !noMoreInput_ && !finished_ && input_ == nullptr;
}

void CudfHashJoinProbe::doAddInput(RowVectorPtr input) {
  if (skipInput_) {
    VELOX_CHECK_NULL(input_);
    return;
  }
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);
  if (joinNode_->isRightSemiProjectJoin() && input->size() > 0) {
    probeSideHasRows_ = true;
    if (joinNode_->isNullAware() && !probeSideHasNullKeys_) {
      probeSideHasNullKeys_ =
          cudf::has_nulls(cudfInput->getTableView().select(leftKeyIndices_));
    }
  }
  // Count nulls in join key columns
  auto [_, null_count] = cudf::bitmask_and(
      cudfInput->getTableView(), cudfInput->stream(), get_temp_mr());
  // The returned mask is intentionally used only to obtain null_count and is
  // destroyed at the end of this scope. bitmask_and may still be writing it
  // asynchronously, so complete its own stream before the temporary buffer
  // can be recycled under another cuDF/cuco operation.
  cudfInput->stream().synchronize();
  {
    // Update statistics for null keys in join operator.
    auto lockedStats = stats_.wlock();
    lockedStats->numNullKeys += null_count;
  }
  if (joinNode_->isRightSemiFilterJoin()) {
    // Queue inputs and process all at once
    if (input->size() > 0) {
      inputs_.push_back(std::move(cudfInput));
    }
    return;
  }

  if (input->size() > 0) {
    if (graceBuildData_ || graceEagerActive_) {
      queueGraceProbeInput(std::move(cudfInput));
      return;
    }
    input_ = std::move(input);
  }
}

void CudfHashJoinProbe::doNoMoreInput() {
  if (!graceProbeInputs_.empty()) {
    flushGraceProbeInputBatch();
  }
  if (graceProbeSourceBatches_ > 0) {
    LOG(WARNING) << "CudfHashJoinProbe node=" << planNodeId()
                 << " Grace partition input coalescing sourceBatches="
                 << graceProbeSourceBatches_
                 << " partitionBatches=" << graceProbePartitionBatches_;
  }
  Operator::noMoreInput();
  if (!graceEnabled_ && !joinNode_->isRightJoin() &&
      !joinNode_->isRightSemiFilterJoin() &&
      !joinNode_->isRightSemiProjectJoin() && !joinNode_->isFullJoin()) {
    return;
  }
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  // Only last driver collects all answers
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  SCOPE_EXIT {
    // Realize the promises so that the other Drivers (which were not
    // the last to finish) can continue from the barrier and finish.
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  if (graceEnabled_) {
    isLastDriver_ = true;
    // Standard Grace scheduling is partition-major. Move every driver's probe
    // partitions to the last driver at end-of-input, then restore each build
    // partition exactly once and consume all matching probe chunks. The old
    // bounded-wave path drained while probe input was still arriving and
    // rebuilt every partition hash table for every approximately 512 MiB wave.
    for (auto& peer : peers) {
      if (peer.get() == operatorCtx_->driver()) {
        continue;
      }
      auto* probe =
          dynamic_cast<CudfHashJoinProbe*>(peer->findOperator(planNodeId()));
      if (probe == nullptr || probe == this) {
        continue;
      }
      VELOX_CHECK_EQ(
          probe->graceProbePartitions_.size(), graceProbePartitions_.size());
      for (size_t partition = 0; partition < graceProbePartitions_.size();
           ++partition) {
        auto& source = probe->graceProbePartitions_[partition];
        auto& target = graceProbePartitions_[partition];
        target.insert(
            target.end(),
            std::make_move_iterator(source.begin()),
            std::make_move_iterator(source.end()));
        source.clear();
      }
      graceProbeBufferedBytes_ += probe->graceProbeBufferedBytes_;
      graceProbeResidentHostBytes_ += probe->graceProbeResidentHostBytes_;
      graceProbeDiskBytes_ += probe->graceProbeDiskBytes_;
      graceHostDemoteMicros_ += probe->graceHostDemoteMicros_;
      graceRawSpillWriteMicros_ += probe->graceRawSpillWriteMicros_;
      probe->graceProbeBufferedBytes_ = 0;
      probe->graceProbeResidentHostBytes_ = 0;
      probe->graceProbeDiskBytes_ = 0;
      // Individual batches retain shared ownership of each peer's spill file
      // after they move to the final driver.
      probe->graceProbeSpillFiles_.clear();
      probe->graceBuildData_.reset();
      probe->graceProbePartitions_.clear();
      probe->graceFinished_ = true;
      probe->finished_ = true;
    }
    if (graceBuildData_) {
      graceProbeDraining_ = true;
      LOG(WARNING) << "CudfHashJoinProbe task="
                   << operatorCtx_->task()->taskId() << " node=" << planNodeId()
                   << " starting partition-major Grace drain bytes="
                   << graceProbeBufferedBytes_
                   << " partitions=" << graceProbePartitions_.size()
                   << " executorReservedHostBytes="
                   << currentGraceHostMemoryReservedBytes()
                   << " hostDemoteUs=" << graceHostDemoteMicros_
                   << " rawSpillWriteUs=" << graceRawSpillWriteMicros_;
    }
    return;
  }

  if (joinNode_->isRightJoin() || joinNode_->isFullJoin() ||
      joinNode_->isRightSemiProjectJoin()) {
    isLastDriver_ = true;
    if (hashObject_.has_value()) {
      auto stream = cudfGlobalStreamPool().get_stream();

      // The allPeersFinished barrier above synchronizes CPU threads, but not
      // GPU streams. A driver's CPU thread may return from getOutput() while
      // its GPU work (updating rightMatchedFlags_) is still in flight.
      // join_streams establishes GPU-side ordering so that all probe stream
      // operations complete before the BITWISE_OR reads below.
      // Drivers without lastProbeStream_ (no probe batches) are skipped:
      // their flags are all-false from host-synchronized init with no pending
      // GPU work.
      std::vector<rmm::cuda_stream_view> inputStreams;
      if (lastProbeStream_.has_value()) {
        inputStreams.push_back(lastProbeStream_.value());
      }
      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        auto op = peer->findOperator(operatorCtx_->operatorId());
        auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
        if (probe != nullptr && probe->lastProbeStream_.has_value()) {
          inputStreams.push_back(probe->lastProbeStream_.value());
        }
      }
      if (!inputStreams.empty()) {
        cudf::detail::join_streams(inputStreams, stream);
      }

      for (auto& peer : peers) {
        if (peer.get() == operatorCtx_->driver()) {
          continue;
        }
        auto op = peer->findOperator(operatorCtx_->operatorId());
        auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
        if (probe == nullptr) {
          continue;
        }
        if (joinNode_->isRightSemiProjectJoin()) {
          probeSideHasRows_ = probeSideHasRows_ || probe->probeSideHasRows_;
          probeSideHasNullKeys_ =
              probeSideHasNullKeys_ || probe->probeSideHasNullKeys_;
        }
        VELOX_CHECK_EQ(
            probe->rightMatchedFlags_.size(), rightMatchedFlags_.size());
        // Combine flags per partition using cuDF bitwise OR
        // DM: This needs a relook. This is for when build side exceeds cudf
        // size_type limits. In case of multiple right side chunks, I'm not sure
        // if partitions to combine are in the same place p
        for (size_t p = 0; p < rightMatchedFlags_.size(); ++p) {
          auto or_result = cudf::binary_operation(
              rightMatchedFlags_[p]->view(),
              probe->rightMatchedFlags_[p]->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
          // binary_operation is async on `stream`; the old column destructs via
          // cudaFreeAsync on its allocation stream (not `stream`), so the free
          // can race the kernel. Drain `stream` before the move-assign.
          stream.synchronize();
          rightMatchedFlags_[p] = std::move(or_result);
        }
      }
      stream.synchronize();
    }
    return;
  }

  // Handling RightSemiFilterJoin
  // Collect results from peers
  for (auto& peer : peers) {
    auto op = peer->findOperator(operatorCtx_->operatorId());
    auto* probe = dynamic_cast<CudfHashJoinProbe*>(op);
    VELOX_CHECK_NOT_NULL(probe);
    inputs_.insert(inputs_.end(), probe->inputs_.begin(), probe->inputs_.end());
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  // Using output_mr here to allow spilling queued up large tables
  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}), probeType_, stream, get_output_mr());

  VELOX_CHECK_NOT_NULL(tbl);

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << tbl->num_columns();
    VLOG(1) << "Probe table number of rows: " << tbl->num_rows();
  }

  // Store the concatenated table in input_
  input_ = std::make_shared<CudfVector>(
      operatorCtx_->pool(),
      probeType_,
      tbl->num_rows(),
      std::move(tbl),
      stream);
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::unfilteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    rmm::cuda_stream_view stream) {
  std::vector<std::unique_ptr<cudf::column>> joinedCols;
  auto const numRows = static_cast<vector_size_t>(
      std::max(leftIndicesCol.size(), rightIndicesCol.size()));
  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  auto rightInput = rightTableView.select(rightColumnIndicesToGather_);
  auto leftResult = cudf::gather(
      leftInput, leftIndicesCol, oobPolicy, stream, get_output_mr());
  auto rightResult = cudf::gather(
      rightInput, rightIndicesCol, oobPolicy, stream, get_output_mr());

  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Left result number of columns: " << leftResult->num_columns();
    VLOG(1) << "Right result number of columns: " << rightResult->num_columns();
  }

  auto leftCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.resize(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    joinedCols[leftColumnOutputIndices_[i]] = std::move(leftCols[i]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    joinedCols[rightColumnOutputIndices_[i]] = std::move(rightCols[i]);
  }
  if (buildStream_.has_value()) {
    // Ensure deallocation of build table happens after probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return {std::make_unique<cudf::table>(std::move(joinedCols)), numRows};
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::filteredOutput(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    std::function<std::vector<std::unique_ptr<cudf::column>>(
        std::vector<std::unique_ptr<cudf::column>>&&,
        cudf::column_view)> func,
    rmm::cuda_stream_view stream) {
  auto leftResult = cudf::gather(
      leftTableView, leftIndicesCol, oobPolicy, stream, get_output_mr());
  auto rightResult = cudf::gather(
      rightTableView, rightIndicesCol, oobPolicy, stream, get_output_mr());
  auto leftColsSize = leftResult->num_columns();
  auto rightColsSize = rightResult->num_columns();

  std::vector<std::unique_ptr<cudf::column>> joinedCols = leftResult->release();
  auto rightCols = rightResult->release();
  joinedCols.insert(
      joinedCols.end(),
      std::make_move_iterator(rightCols.begin()),
      std::make_move_iterator(rightCols.end()));

  VELOX_CHECK_NOT_NULL(
      filterEvaluator_,
      "Join filter evaluator must be initialized before filteredOutput()");
  std::vector<cudf::column_view> joinedColViews;
  joinedColViews.reserve(joinedCols.size());
  for (const auto& col : joinedCols) {
    joinedColViews.push_back(col->view());
  }
  auto filterColumns =
      filterEvaluator_->eval(joinedColViews, stream, get_output_mr());
  auto filterColumn = asView(filterColumns);

  joinedCols = func(std::move(joinedCols), filterColumn);
  auto const numRows = filteredOutputNumRows(
      outputType_->size() == 0,
      filterColumn,
      joinedCols,
      stream,
      get_temp_mr());

  auto filteredjoinedCols =
      std::vector<std::unique_ptr<cudf::column>>(outputType_->names().size());
  for (int i = 0; i < leftColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[leftColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColumnIndicesToGather_[i]]);
  }
  for (int i = 0; i < rightColumnOutputIndices_.size(); i++) {
    filteredjoinedCols[rightColumnOutputIndices_[i]] =
        std::move(joinedCols[leftColsSize + rightColumnIndicesToGather_[i]]);
  }
  joinedCols = std::move(filteredjoinedCols);
  if (buildStream_.has_value()) {
    // Ensure any deallocation of join indices is ordered wrt probe gathers
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();
  return {std::make_unique<cudf::table>(std::move(joinedCols)), numRows};
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::filteredOutputIndices(
    cudf::table_view leftTableView,
    cudf::column_view leftIndicesCol,
    cudf::table_view rightTableView,
    cudf::column_view rightIndicesCol,
    cudf::table_view extendedLeftView,
    cudf::table_view extendedRightView,
    cudf::join_kind joinKind,
    rmm::cuda_stream_view stream) {
  // Use extended views (with precomputed columns) for filter evaluation
  auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
      cudf::filter_join_indices(
          extendedLeftView,
          extendedRightView,
          leftIndicesCol,
          rightIndicesCol,
          tree_.back(),
          joinKind,
          stream,
          get_temp_mr());

  auto filteredLeftIndicesSpan = cudf::device_span<cudf::size_type const>{
      filteredLeftJoinIndices->data(), filteredLeftJoinIndices->size()};
  auto filteredRightIndicesSpan = cudf::device_span<cudf::size_type const>{
      filteredRightJoinIndices->data(), filteredRightJoinIndices->size()};
  auto filteredLeftIndicesCol = cudf::column_view{filteredLeftIndicesSpan};
  auto filteredRightIndicesCol = cudf::column_view{filteredRightIndicesSpan};
  // Use original views (without precomputed columns) for gathering output
  return unfilteredOutput(
      leftTableView,
      filteredLeftIndicesCol,
      rightTableView,
      filteredRightIndicesCol,
      stream);
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::innerJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && useAstFilter_ &&
      !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && useAstFilter_ &&
         !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    // left = probe, right = build
    VELOX_CHECK_NOT_NULL(hb);
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        stream,
        get_temp_mr());

    auto leftIndicesSpan = toSpan(*leftJoinIndices);
    auto rightIndicesSpan = toSpan(*rightJoinIndices);
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      if (useAstFilter_) {
        cudfOutputs.push_back(filteredOutputIndices(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            extendedLeftView,
            extendedRightView,
            cudf::join_kind::INNER_JOIN,
            stream));
      } else {
        auto filterFunc =
            [stream](
                std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
                cudf::column_view filterColumn) {
              auto filterTable =
                  std::make_unique<cudf::table>(std::move(joinedCols));
              auto filteredTable = cudf::apply_boolean_mask(
                  *filterTable, filterColumn, stream, get_output_mr());
              return filteredTable->release();
            };
        cudfOutputs.push_back(filteredOutput(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            filterFunc,
            stream));
      }
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::leftJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  auto numProbeRows = leftTableView.num_rows();

  // Track which probe rows matched in any build batch so that unmatched probe
  // rows can be emitted with NULL build columns after the loop.
  ProbeMatchTracker probeTracker(numProbeRows, stream, get_temp_mr());

  // A separate left_join against every independently hashed build table
  // emits an unmatched probe row once per table. Use INNER matches plus one
  // global unmatched pass when concatenate batching split the build.
  if (rightTables.size() > 1) {
    return multiTableLeftOuterJoin(leftTableView, false, stream);
  }

  // Precompute left (probe) table columns if needed (once, outside loop)
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  // Processes a build batch of join indices: applies the filter (if any),
  // updates probeTracker from post-filter left indices, and appends the result
  // to cudfOutputs.
  auto processBatch = [&](cudf::column_view leftIndicesCol,
                          cudf::column_view rightIndicesCol,
                          cudf::table_view rightTableView,
                          size_t buildBatchIdx) {
    if (joinNode_->filter()) {
      cudf::table_view extendedRightView = !rightPrecomputeInstructions_.empty()
          ? cachedExtendedRightViews_[buildBatchIdx]
          : rightTableView;

      if (useAstFilter_) {
        // Inline filter_join_indices so we can access post-filter left indices
        // for match tracking.
        auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
            cudf::filter_join_indices(
                extendedLeftView,
                extendedRightView,
                leftIndicesCol,
                rightIndicesCol,
                tree_.back(),
                cudf::join_kind::INNER_JOIN,
                stream,
                get_temp_mr());

        if (filteredLeftJoinIndices->size() > 0) {
          auto filteredLeftSpan = toSpan(*filteredLeftJoinIndices);
          auto filteredRightSpan = toSpan(*filteredRightJoinIndices);
          auto filteredLeftCol = cudf::column_view{filteredLeftSpan};
          auto filteredRightCol = cudf::column_view{filteredRightSpan};

          probeTracker.update(filteredLeftCol, stream, get_temp_mr());

          cudfOutputs.push_back(unfilteredOutput(
              leftTableView,
              filteredLeftCol,
              rightTableView,
              filteredRightCol,
              stream));
        }
      } else {
        auto leftIndicesSpanCopy =
            cudf::device_span<cudf::size_type const>(leftIndicesCol);
        auto filterFunc =
            [&probeTracker, leftIndicesSpanCopy, stream](
                std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
                cudf::column_view filterColumn) {
              auto filterTable =
                  std::make_unique<cudf::table>(std::move(joinedCols));
              auto filteredTable = cudf::apply_boolean_mask(
                  *filterTable, filterColumn, stream, get_output_mr());
              joinedCols = filteredTable->release();

              // Filter left join indices with the same mask to track which
              // probe rows passed the filter.
              auto leftIdxCol = cudf::column_view{leftIndicesSpanCopy};
              auto filteredIdxTable = cudf::apply_boolean_mask(
                  cudf::table_view{std::vector<cudf::column_view>{leftIdxCol}},
                  filterColumn,
                  stream,
                  get_temp_mr());
              auto filteredLeftIdxCol =
                  std::move(filteredIdxTable->release()[0]);
              probeTracker.update(
                  filteredLeftIdxCol->view(), stream, get_temp_mr());

              return std::move(joinedCols);
            };
        cudfOutputs.push_back(filteredOutput(
            leftTableView,
            leftIndicesCol,
            rightTableView,
            rightIndicesCol,
            filterFunc,
            stream));
      }
    } else {
      probeTracker.update(leftIndicesCol, stream, get_temp_mr());
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  };

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use inner_join to get only real matched pairs. Unmatched probe rows are
    // emitted separately after the loop.
    VELOX_CHECK_NOT_NULL(hb);
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        stream,
        get_temp_mr());

    if (leftJoinIndices->size() == 0) {
      continue;
    }

    auto leftIndicesSpan = toSpan(*leftJoinIndices);
    auto rightIndicesSpan = toSpan(*rightJoinIndices);
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    processBatch(leftIndicesCol, rightIndicesCol, rightTableView, i);
  }

  // Emit unmatched probe rows with JoinNoMatch right indices so that gather
  // with NULLIFY produces NULL build columns.
  auto unmatchedIndices =
      probeTracker.getUnmatchedIndices(stream, get_temp_mr());

  if (unmatchedIndices->size() > 0) {
    auto unmatchedLeftCol = unmatchedIndices->view();
    auto sentinelScalar = cudf::numeric_scalar<cudf::size_type>(
        cudf::JoinNoMatch, true, stream, get_temp_mr());
    auto unmatchedRightIndices = cudf::make_column_from_scalar(
        sentinelScalar, unmatchedIndices->size(), stream, get_temp_mr());
    auto unmatchedRightCol = unmatchedRightIndices->view();

    // Emit unmatched rows directly via unfilteredOutput for two reasons:
    // (1) We cannot use filteredOutputIndices with LEFT_JOIN here because
    //     filter_join_indices(LEFT_JOIN) ensures every row in the left *table*
    //     appears at least once. Since our left indices are a subset of probe
    //     rows (only the unmatched ones), LEFT_JOIN would re-add all the
    //     matched probe rows that are absent from this subset.
    // (2) Using unfilteredOutput is safe because all right indices are
    //     JoinNoMatch. Per filter_join_indices semantics, input pairs with
    //     JoinNoMatch in either position pass through unchanged (the predicate
    //     cannot be evaluated), so filtering would be a no-op anyway.
    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        unmatchedLeftCol,
        rightTables[0]->view(),
        unmatchedRightCol,
        stream));
  }

  return cudfOutputs;
}

CudfHashJoinProbe::JoinOutput CudfHashJoinProbe::leftUnmatchedOutput(
    cudf::table_view leftTableView,
    cudf::column_view probeMatchedFlags,
    rmm::cuda_stream_view stream) {
  auto unmatchedMask = cudf::unary_operation(
      probeMatchedFlags, cudf::unary_operator::NOT, stream, get_temp_mr());
  auto unmatchedCountScalar = cudf::reduce(
      unmatchedMask->view(),
      *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
      cudf::data_type{cudf::type_id::INT32},
      stream,
      get_temp_mr());
  const auto unmatchedRows = static_cast<vector_size_t>(
      static_cast<cudf::numeric_scalar<int32_t>*>(unmatchedCountScalar.get())
          ->value(stream));
  if (unmatchedRows == 0) {
    std::vector<std::unique_ptr<cudf::column>> emptyCols;
    emptyCols.reserve(outputType_->size());
    for (size_t i = 0; i < outputType_->size(); ++i) {
      emptyCols.push_back(makeEmptyColumnForType(
          outputType_->childAt(i), stream, get_output_mr()));
    }
    return {std::make_unique<cudf::table>(std::move(emptyCols)), 0};
  }

  std::vector<std::unique_ptr<cudf::column>> outCols(outputType_->size());
  if (!leftColumnIndicesToGather_.empty()) {
    auto unmatchedLeft = cudf::apply_boolean_mask(
        leftTableView.select(leftColumnIndicesToGather_),
        unmatchedMask->view(),
        stream,
        get_output_mr());
    auto leftCols = unmatchedLeft->release();
    for (size_t i = 0; i < leftColumnOutputIndices_.size(); ++i) {
      outCols[leftColumnOutputIndices_[i]] = std::move(leftCols[i]);
    }
  }
  for (size_t i = 0; i < rightColumnOutputIndices_.size(); ++i) {
    const auto buildChannel = rightColumnIndicesToGather_[i];
    outCols[rightColumnOutputIndices_[i]] = makeAllNullColumnForType(
        buildType_->childAt(buildChannel),
        unmatchedRows,
        stream,
        get_output_mr());
  }
  stream.synchronize();
  return {std::make_unique<cudf::table>(std::move(outCols)), unmatchedRows};
}

std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::multiTableLeftOuterJoin(
    cudf::table_view leftTableView,
    bool trackRightMatches,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;
  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  VELOX_CHECK_GT(rightTables.size(), 1);
  VELOX_CHECK_EQ(rightTables.size(), hbs.size());
  if (trackRightMatches) {
    VELOX_CHECK_EQ(rightTables.size(), rightMatchedFlags_.size());
  }
  addRuntimeStat(
      "multiBuildOuterProbeBatches",
      RuntimeCounter(1, RuntimeCounter::Unit::kNone));
  addRuntimeStat(
      "multiBuildOuterProbeRows",
      RuntimeCounter(leftTableView.num_rows(), RuntimeCounter::Unit::kNone));
  addRuntimeStat(
      "multiBuildOuterTableProbes",
      RuntimeCounter(rightTables.size(), RuntimeCounter::Unit::kNone));

  const auto numProbeRows = leftTableView.num_rows();
  auto falseScalar =
      cudf::numeric_scalar<bool>(false, true, stream, get_temp_mr());
  auto probeMatchedFlags = cudf::make_column_from_scalar(
      falseScalar, numProbeRows, stream, get_temp_mr());
  auto probeRowIndices = cudf::sequence(
      numProbeRows,
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
      stream,
      get_temp_mr());

  const auto markMatchedRows = [&stream, this](
                                   std::unique_ptr<cudf::column>& flags,
                                   cudf::column_view matchedIndices,
                                   cudf::column_view allRowIndices,
                                   bool flagsMayUseAnotherStream) {
    if (matchedIndices.size() == 0) {
      return;
    }
    auto matched =
        cudf::contains(matchedIndices, allRowIndices, stream, get_temp_mr());
    auto updated = cudf::binary_operation(
        flags->view(),
        matched->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    if (flagsMayUseAnotherStream) {
      // Persistent FULL-join flags can have been allocated or last updated on
      // another probe stream. Complete the OR before move-assignment can
      // release that storage. Request-local probe flags use this stream and
      // retain normal stream-ordered deallocation.
      stream.synchronize();
    }
    flags = std::move(updated);
  };

  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  cudfOutputs.reserve(rightTables.size() + 1);
  for (size_t i = 0; i < rightTables.size(); ++i) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];
    VELOX_CHECK_NOT_NULL(hb);
    cudf::table_view extendedRightView =
        (joinNode_->filter() && !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        stream,
        get_temp_mr());
    auto leftIndicesSpan =
        cudf::device_span<cudf::size_type const>{*leftJoinIndices};
    auto rightIndicesSpan =
        cudf::device_span<cudf::size_type const>{*rightJoinIndices};
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    if (joinNode_->filter() && useAstFilter_) {
      auto [filteredLeft, filteredRight] = cudf::filter_join_indices(
          extendedLeftView,
          extendedRightView,
          leftIndicesCol,
          rightIndicesCol,
          tree_.back(),
          cudf::join_kind::INNER_JOIN,
          stream,
          get_temp_mr());
      auto filteredLeftSpan =
          cudf::device_span<cudf::size_type const>{*filteredLeft};
      auto filteredRightSpan =
          cudf::device_span<cudf::size_type const>{*filteredRight};
      auto filteredLeftCol = cudf::column_view{filteredLeftSpan};
      auto filteredRightCol = cudf::column_view{filteredRightSpan};
      markMatchedRows(
          probeMatchedFlags, filteredLeftCol, probeRowIndices->view(), false);
      if (trackRightMatches) {
        auto buildRowIndices = cudf::sequence(
            rightTableView.num_rows(),
            cudf::numeric_scalar<cudf::size_type>(
                0, true, stream, get_temp_mr()),
            cudf::numeric_scalar<cudf::size_type>(
                1, true, stream, get_temp_mr()),
            stream,
            get_temp_mr());
        markMatchedRows(
            rightMatchedFlags_[i],
            filteredRightCol,
            buildRowIndices->view(),
            true);
      }
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          filteredLeftCol,
          rightTableView,
          filteredRightCol,
          stream));
      continue;
    }

    if (joinNode_->filter()) {
      auto filterFunc =
          [&, i, leftIndicesSpan, rightIndicesSpan](
              std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
              cudf::column_view filterColumn) {
            auto filterTable =
                std::make_unique<cudf::table>(std::move(joinedCols));
            auto filteredTable = cudf::apply_boolean_mask(
                *filterTable, filterColumn, stream, get_output_mr());

            auto filteredLeftTable = cudf::apply_boolean_mask(
                cudf::table_view{std::vector<cudf::column_view>{
                    cudf::column_view{leftIndicesSpan}}},
                filterColumn,
                stream,
                get_temp_mr());
            markMatchedRows(
                probeMatchedFlags,
                filteredLeftTable->view().column(0),
                probeRowIndices->view(),
                false);
            if (trackRightMatches) {
              auto filteredRightTable = cudf::apply_boolean_mask(
                  cudf::table_view{std::vector<cudf::column_view>{
                      cudf::column_view{rightIndicesSpan}}},
                  filterColumn,
                  stream,
                  get_temp_mr());
              auto buildRowIndices = cudf::sequence(
                  rightTableView.num_rows(),
                  cudf::numeric_scalar<cudf::size_type>(
                      0, true, stream, get_temp_mr()),
                  cudf::numeric_scalar<cudf::size_type>(
                      1, true, stream, get_temp_mr()),
                  stream,
                  get_temp_mr());
              markMatchedRows(
                  rightMatchedFlags_[i],
                  filteredRightTable->view().column(0),
                  buildRowIndices->view(),
                  true);
            }
            return filteredTable->release();
          };
      cudfOutputs.push_back(filteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          filterFunc,
          stream));
      continue;
    }

    markMatchedRows(
        probeMatchedFlags, leftIndicesCol, probeRowIndices->view(), false);
    if (trackRightMatches) {
      auto buildRowIndices = cudf::sequence(
          rightTableView.num_rows(),
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());
      markMatchedRows(
          rightMatchedFlags_[i],
          rightIndicesCol,
          buildRowIndices->view(),
          true);
    }
    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        leftIndicesCol,
        rightTableView,
        rightIndicesCol,
        stream));
  }

  auto unmatched =
      leftUnmatchedOutput(leftTableView, probeMatchedFlags->view(), stream);
  addRuntimeStat(
      "multiBuildOuterUnmatchedProbeRows",
      RuntimeCounter(unmatched.numRows, RuntimeCounter::Unit::kNone));
  if (unmatched.numRows > 0) {
    cudfOutputs.push_back(std::move(unmatched));
  }
  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::rightJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    VELOX_CHECK_NOT_NULL(hb);
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        stream,
        get_temp_mr());
    if (!joinNode_->filter()) {
      // Mark matched build rows by checking which row indices appear in
      // rightJoinIndices. Use contains to avoid scatter with duplicate indices.
      auto rightIdxCol = cudf::column_view{toSpan(*rightJoinIndices)};

      // Create sequence [0, 1, ..., n-1] for build table row indices
      auto n = rightTableView.num_rows();
      auto rowIndices = cudf::sequence(
          n,
          cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
          cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
          stream,
          get_temp_mr());

      // Check which build row indices are present in the join result
      auto matchedInBatch = cudf::contains(
          rightIdxCol, rowIndices->view(), stream, get_temp_mr());

      // OR with existing flags to accumulate matches across batches
      auto updatedFlags = cudf::binary_operation(
          rightMatchedFlags_[i]->view(),
          matchedInBatch->view(),
          cudf::binary_operator::BITWISE_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());
      // binary_operation is async on `stream`; the old column destructs via
      // cudaFreeAsync on its allocation stream (not `stream`), so the free
      // can race the kernel. Drain `stream` before the move-assign.
      stream.synchronize();
      rightMatchedFlags_[i] = std::move(updatedFlags);
    }

    auto leftIndicesSpan = toSpan(*leftJoinIndices);
    auto rightIndicesSpan = toSpan(*rightJoinIndices);
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
    std::vector<std::unique_ptr<cudf::column>> joinedCols;

    if (joinNode_->filter()) {
      auto& rightMatchedFlags = rightMatchedFlags_[i];
      auto numBuildRows = rightTableView.num_rows();
      auto filterFunc =
          [&rightMatchedFlags, rightIndicesSpan, numBuildRows, stream](
              std::vector<std::unique_ptr<cudf::column>>&& joinedCols,
              cudf::column_view filterColumn) {
            // apply the filter
            auto filterTable =
                std::make_unique<cudf::table>(std::move(joinedCols));
            auto filteredTable = cudf::apply_boolean_mask(
                *filterTable, filterColumn, stream, get_output_mr());
            joinedCols = filteredTable->release();

            // For streaming right join, after applying filter, we record
            // matched right indices filter rightJoinIndices with the same mask
            // to update matched flags
            auto rightIdxCol = cudf::column_view{rightIndicesSpan};
            auto filteredIdxTable = cudf::apply_boolean_mask(
                cudf::table_view{std::vector<cudf::column_view>{rightIdxCol}},
                filterColumn,
                stream,
                get_temp_mr());
            auto filteredCols = filteredIdxTable->release();
            auto filteredRightIdxCol = std::move(filteredCols[0]);

            // Use contains to check which build row indices passed the filter
            auto rowIndices = cudf::sequence(
                numBuildRows,
                cudf::numeric_scalar<cudf::size_type>(
                    0, true, stream, get_temp_mr()),
                cudf::numeric_scalar<cudf::size_type>(
                    1, true, stream, get_temp_mr()),
                stream,
                get_temp_mr());

            auto matchedInBatch = cudf::contains(
                filteredRightIdxCol->view(),
                rowIndices->view(),
                stream,
                get_temp_mr());

            // OR with existing flags to accumulate matches across batches
            auto updatedFlags = cudf::binary_operation(
                rightMatchedFlags->view(),
                matchedInBatch->view(),
                cudf::binary_operator::BITWISE_OR,
                cudf::data_type{cudf::type_id::BOOL8},
                stream,
                get_temp_mr());
            // binary_operation is async on `stream`; the old column destructs
            // via cudaFreeAsync on its allocation stream (not `stream`), so the
            // free can race the kernel. Drain `stream` before the move-assign.
            stream.synchronize();
            rightMatchedFlags = std::move(updatedFlags);
            return std::move(joinedCols);
          };
      cudfOutputs.push_back(filteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          filterFunc,
          stream));
    } else {
      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }
  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::fullJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  auto numProbeRows = leftTableView.num_rows();

  // As for LEFT join, probe preservation must be global across independently
  // hashed build chunks. The helper also updates rightMatchedFlags_ so the
  // existing final FULL-join output still emits each unmatched build row.
  if (rightTables.size() > 1) {
    return multiTableLeftOuterJoin(leftTableView, true, stream);
  }

  // For now, AST support is necessary to filter join output
  if (joinNode_->filter() && !useAstFilter_) {
    VELOX_NYI("Full join requires AST support for filtering");
  }

  // Track which probe rows matched in any build batch so that unmatched probe
  // rows can be emitted with NULL build columns after the loop.
  ProbeMatchTracker probeTracker(numProbeRows, stream, get_temp_mr());

  // Helper to accumulate build-side (right) match flags for a build batch.
  auto updateRightMatchedFlags = [&](size_t batchIdx,
                                     cudf::column_view matchedRightIndices,
                                     cudf::size_type numBuildRows) {
    auto buildRowIndices = cudf::sequence(
        numBuildRows,
        cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
        cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
        stream,
        get_temp_mr());
    auto matchedInBatch = cudf::contains(
        matchedRightIndices, buildRowIndices->view(), stream, get_temp_mr());
    auto updatedFlags = cudf::binary_operation(
        rightMatchedFlags_[batchIdx]->view(),
        matchedInBatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_temp_mr());
    stream.synchronize();
    rightMatchedFlags_[batchIdx] = std::move(updatedFlags);
  };

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use inner_join to get only real matched pairs. Unmatched probe rows are
    // emitted separately after the loop. Unmatched build rows are emitted in
    // doGetOutput via rightMatchedFlags_.
    VELOX_CHECK_NOT_NULL(hb);
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        stream,
        get_temp_mr());

    if (leftJoinIndices->size() == 0) {
      continue;
    }

    auto leftIndicesSpan = toSpan(*leftJoinIndices);
    auto rightIndicesSpan = toSpan(*rightJoinIndices);
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    if (joinNode_->filter()) {
      // Apply filter and keep only pairs where the predicate passes.
      auto [filteredLeftJoinIndices, filteredRightJoinIndices] =
          cudf::filter_join_indices(
              leftTableView,
              rightTableView,
              leftIndicesCol,
              rightIndicesCol,
              tree_.back(),
              cudf::join_kind::INNER_JOIN,
              stream,
              get_temp_mr());

      if (filteredLeftJoinIndices->size() > 0) {
        auto filteredLeftSpan = toSpan(*filteredLeftJoinIndices);
        auto filteredRightSpan = toSpan(*filteredRightJoinIndices);
        auto filteredLeftCol = cudf::column_view{filteredLeftSpan};
        auto filteredRightCol = cudf::column_view{filteredRightSpan};

        probeTracker.update(filteredLeftCol, stream, get_temp_mr());
        updateRightMatchedFlags(i, filteredRightCol, rightTableView.num_rows());

        cudfOutputs.push_back(unfilteredOutput(
            leftTableView,
            filteredLeftCol,
            rightTableView,
            filteredRightCol,
            stream));
      }
    } else {
      probeTracker.update(leftIndicesCol, stream, get_temp_mr());
      updateRightMatchedFlags(i, rightIndicesCol, rightTableView.num_rows());

      cudfOutputs.push_back(unfilteredOutput(
          leftTableView,
          leftIndicesCol,
          rightTableView,
          rightIndicesCol,
          stream));
    }
  }

  // Emit unmatched probe rows with JoinNoMatch right indices so that gather
  // with NULLIFY produces NULL build columns.
  auto unmatchedIndices =
      probeTracker.getUnmatchedIndices(stream, get_temp_mr());

  if (unmatchedIndices->size() > 0) {
    auto unmatchedLeftCol = unmatchedIndices->view();
    auto sentinelScalar = cudf::numeric_scalar<cudf::size_type>(
        cudf::JoinNoMatch, true, stream, get_temp_mr());
    auto unmatchedRightIndices = cudf::make_column_from_scalar(
        sentinelScalar, unmatchedIndices->size(), stream, get_temp_mr());
    auto unmatchedRightCol = unmatchedRightIndices->view();

    // Use unfilteredOutput directly — see the matching comment in leftJoin()
    // for why filteredOutputIndices cannot be used here.
    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        unmatchedLeftCol,
        rightTables[0]->view(),
        unmatchedRightCol,
        stream));
  }

  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::leftSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto numProbeRows = leftTableView.num_rows();

  // Track which probe rows matched across all build chunks so that each
  // probe row is emitted at most once (semi-join semantics).
  ProbeMatchTracker probeTracker(numProbeRows, stream, get_temp_mr());

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftJoinIndices;

    if (joinNode_->filter()) {
      leftJoinIndices = cudf::mixed_left_semi_join(
          leftTableView.select(leftKeyIndices_),
          rightTableView.select(rightKeyIndices_),
          leftTableView,
          rightTableView,
          tree_.back(),
          cudf::null_equality::UNEQUAL,
          stream,
          get_temp_mr());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          stream);
      leftJoinIndices = filter_join.semi_join(
          leftTableView.select(leftKeyIndices_), stream, get_temp_mr());
    }

    if (leftJoinIndices->size() > 0) {
      auto leftIndicesSpan = toSpan(*leftJoinIndices);
      auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
      probeTracker.update(leftIndicesCol, stream, get_temp_mr());
    }
  }

  // Gather the deduplicated set of matched probe rows.
  auto matchedIndices = probeTracker.getMatchedIndices(stream, get_temp_mr());

  if (matchedIndices->size() > 0) {
    auto matchedLeftCol = matchedIndices->view();
    auto sentinelScalar = cudf::numeric_scalar<cudf::size_type>(
        cudf::JoinNoMatch, true, stream, get_temp_mr());
    auto matchedRightIndices = cudf::make_column_from_scalar(
        sentinelScalar, matchedIndices->size(), stream, get_temp_mr());

    cudfOutputs.push_back(unfilteredOutput(
        leftTableView,
        matchedLeftCol,
        rightTables[0]->view(),
        matchedRightIndices->view(),
        stream));
  }

  return cudfOutputs;
}

namespace {
/// Creates a boolean column indicating which rows have NULL in ANY key column.
/// Returns a column where row[i] = true if ANY key column is NULL at row i.
std::unique_ptr<cudf::column> createProbeKeyNullMask(
    cudf::table_view keyView,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numRows = keyView.num_rows();

  if (keyView.num_columns() == 0 || numRows == 0) {
    auto falseScalar = cudf::numeric_scalar<bool>(false, true, stream, mr);
    return cudf::make_column_from_scalar(falseScalar, numRows, stream, mr);
  }

  // Start with first column's null mask
  auto result = cudf::is_null(keyView.column(0), stream, mr);

  // OR with other columns' null masks
  for (cudf::size_type i = 1; i < keyView.num_columns(); i++) {
    auto colIsNull = cudf::is_null(keyView.column(i), stream, mr);
    result = cudf::binary_operation(
        result->view(),
        colIsNull->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        mr);
  }
  return result;
}

/// Applies a null mask to a boolean column.
/// Where nullMask[i] is true, result[i] becomes NULL.
/// Where nullMask[i] is false, result[i] keeps its original value from col.
std::unique_ptr<cudf::column> applyNullMask(
    cudf::column_view col,
    cudf::column_view nullMask,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Create a null scalar (valid=false means NULL)
  auto nullScalar = cudf::numeric_scalar<bool>(false, false, stream, mr);

  // copy_if_else: where nullMask is true, use nullScalar (NULL); else use col
  // value
  return cudf::copy_if_else(nullScalar, col, nullMask, stream, mr);
}

/// Create cross-product of two index columns.
/// Given left = [a, b, c] and right = [x, y], produces:
///   leftOut = [a, a, b, b, c, c]
///   rightOut = [x, y, x, y, x, y]
/// Uses cudf::repeat for left (repeat each element) and cudf::tile for right.
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
createCrossProductIndices(
    cudf::column_view leftIndices,
    cudf::column_view rightIndices,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto numLeft = leftIndices.size();
  auto numRight = rightIndices.size();

  if (numLeft == 0 || numRight == 0) {
    // Return empty columns
    auto emptyLeft = cudf::make_empty_column(cudf::type_id::INT32);
    auto emptyRight = cudf::make_empty_column(cudf::type_id::INT32);
    return {std::move(emptyLeft), std::move(emptyRight)};
  }

  // Repeat each left element numRight times: [a,a,b,b,c,c]
  auto leftRepeated =
      cudf::repeat(cudf::table_view{{leftIndices}}, numRight, stream, mr);

  // Tile the right indices numLeft times: [x,y,x,y,x,y]
  auto rightTiled =
      cudf::tile(cudf::table_view{{rightIndices}}, numLeft, stream, mr);

  return {
      std::move(leftRepeated->release()[0]),
      std::move(rightTiled->release()[0])};
}

} // namespace

// LEFT SEMI PROJECT returns all probe rows with a boolean "match" column
// indicating whether each probe row has at least one matching build row
// (that also passes the filter, if specified). Unlike LEFT SEMI FILTER
// which filters out non-matching rows, this preserves all probe rows.
// Output cardinality always equals probe side cardinality.
//
// Implementation approach:
// 1. Use inner_join to get valid (probe_idx, build_idx) pairs where keys match
// 2. If filter exists, apply filter_join_indices(INNER_JOIN) to keep only
//    pairs where the filter passes
// 3. Use cudf::contains to check which probe row indices appear in the result.
//    This correctly handles duplicate probe indices (when one probe row matches
//    multiple build rows) by returning true if the index appears at least once.
// 4. Accumulate matches across build table batches using BITWISE_OR
// 5. For null-aware mode (without filter): apply null mask based on probe key
//    nullity and build side null keys presence
// 6. Output: all probe columns + match column
std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::leftSemiProjectJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  // For now, AST support is necessary to filter join output
  if (joinNode_->filter() && !useAstFilter_) {
    VELOX_NYI("Left semi project join requires AST support for filtering");
  }

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  auto numProbeRows = leftTableView.num_rows();

  const bool isNullAware = joinNode_->isNullAware();
  const bool hasFilter = joinNode_->filter() != nullptr;
  // For null-aware without filter, we use a different code path (existing)
  // For null-aware with filter, we need to compute indeterminate cases
  const bool isNullAwareWithFilter = isNullAware && hasFilter;
  const bool isNullAwareWithoutFilter = isNullAware && !hasFilter;

  // Create probe row indices sequence: [0, 1, 2, ..., numProbeRows-1]
  // Used with cudf::contains to create the match column
  auto probeRowIndices = cudf::sequence(
      numProbeRows,
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, get_temp_mr()),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, get_temp_mr()),
      stream,
      get_temp_mr());

  // Initialize match column to all false
  auto falseScalar =
      cudf::numeric_scalar<bool>(false, true, stream, get_output_mr());
  auto matchCol = cudf::make_column_from_scalar(
      falseScalar, numProbeRows, stream, get_output_mr());

  // Precompute left (probe) table columns if needed for filter
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (auto i = 0; i < rightTables.size(); i++) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];

    // Use cached precomputed columns for right (build) table
    cudf::table_view extendedRightView =
        (joinNode_->filter() && !rightPrecomputeInstructions_.empty())
        ? cachedExtendedRightViews_[i]
        : rightTableView;

    // Step 1: Inner join to get (probe_idx, build_idx) pairs where keys match.
    // Unlike left_join, inner_join only returns valid pairs (no JoinNoMatch).
    VELOX_CHECK_NOT_NULL(hb);
    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        stream,
        get_temp_mr());

    if (leftJoinIndices->size() == 0) {
      continue; // No matches from this build table
    }

    auto leftIndicesSpan = toSpan(*leftJoinIndices);
    auto rightIndicesSpan = toSpan(*rightJoinIndices);
    auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
    auto rightIndicesCol = cudf::column_view{rightIndicesSpan};

    cudf::column_view matchedProbeIndices;
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> filteredLeftIndices;

    if (joinNode_->filter()) {
      // Step 2: Apply filter to the join pairs. INNER_JOIN mode keeps only
      // pairs where the predicate evaluates to true.
      auto [filteredLeft, filteredRight] = cudf::filter_join_indices(
          extendedLeftView,
          extendedRightView,
          leftIndicesSpan,
          rightIndicesSpan,
          tree_.back(),
          cudf::join_kind::INNER_JOIN,
          stream,
          get_temp_mr());

      filteredLeftIndices = std::move(filteredLeft);
      if (filteredLeftIndices->size() == 0) {
        continue; // No matches passed filter
      }
      auto filteredLeftSpan = toSpan(*filteredLeftIndices);
      matchedProbeIndices = cudf::column_view{filteredLeftSpan};
    } else {
      // No filter - use inner join results directly
      matchedProbeIndices = leftIndicesCol;
    }

    // Step 3: Create match flags using cudf::contains. For each probe row index
    // in [0, numProbeRows), check if it appears in matchedProbeIndices.
    // This handles duplicates correctly - if a probe row matches multiple build
    // rows, it appears multiple times in matchedProbeIndices, but contains()
    // returns true if it appears at least once.
    auto matchedInBatch = cudf::contains(
        matchedProbeIndices, probeRowIndices->view(), stream, get_temp_mr());

    // Step 4: Accumulate matches across build table batches using OR.
    // A probe row's final match value is true if it matched in ANY batch.
    auto updatedMatch = cudf::binary_operation(
        matchCol->view(),
        matchedInBatch->view(),
        cudf::binary_operator::BITWISE_OR,
        cudf::data_type{cudf::type_id::BOOL8},
        stream,
        get_output_mr());
    stream.synchronize();
    matchCol = std::move(updatedMatch);
  }

  // Step 5: Handle null-aware semantics (IN vs EXISTS).
  // For null-aware mode, we need to compute three-valued logic:
  // - TRUE: at least one match passes filter
  // - FALSE: no match passes filter AND no indeterminate cases
  // - NULL: probe key is NULL, OR (no match AND build has null keys that might
  // match)

  if (isNullAwareWithFilter) {
    // Null-aware LEFT SEMI PROJECT with filter implements SQL IN semantics:
    //   SELECT t0 IN (SELECT u0 FROM u WHERE filter) FROM t
    //
    // "Indeterminate" means the result should be NULL (unknown) rather than
    // FALSE. This happens when we cannot definitively say the probe value is
    // NOT IN the subquery because NULL comparisons are involved:
    //
    // - Type B (null probe key): When probe key is NULL, we can't determine
    //   if NULL equals any subquery value. If the filter passes for ANY build
    //   row, result is NULL (might match). If filter fails for ALL build rows,
    //   the subquery is empty, so result is FALSE.
    //
    // - Type A (non-null probe key, no match): When probe key doesn't match
    //   any non-NULL build key, but build has NULL keys where filter passes,
    //   we can't rule out a match (NULL might equal our probe key), so result
    //   is NULL.
    //
    // We evaluate these by creating synthetic (probe, build) index pairs and
    // running the filter to see if any pair passes.

    // Lambda to create device_span from a column
    auto columnToSpan = [](cudf::column_view col) {
      return cudf::device_span<cudf::size_type const>{
          static_cast<cudf::size_type const*>(col.head()),
          static_cast<size_t>(col.size())};
    };

    // Lambda to run filter on synthetic pairs and accumulate indeterminate
    // flags. Creates cross-product of probeIndices × buildIndices, runs the
    // filter, and ORs any passing probe rows into indeterminateCol.
    auto accumulateIndeterminate =
        [&](cudf::column_view probeIndices,
            cudf::column_view buildIndices,
            cudf::table_view extendedRight,
            std::unique_ptr<cudf::column>& indeterminateCol) {
          if (probeIndices.size() == 0 || buildIndices.size() == 0) {
            return;
          }

          auto [syntheticLeft, syntheticRight] = createCrossProductIndices(
              probeIndices, buildIndices, stream, get_temp_mr());

          if (syntheticLeft->size() == 0) {
            return;
          }

          auto [filteredLeft, filteredRight] = cudf::filter_join_indices(
              extendedLeftView,
              extendedRight,
              columnToSpan(syntheticLeft->view()),
              columnToSpan(syntheticRight->view()),
              tree_.back(),
              cudf::join_kind::INNER_JOIN,
              stream,
              get_temp_mr());

          if (filteredLeft->size() == 0) {
            return;
          }

          auto filteredLeftSpan = toSpan(*filteredLeft);
          auto filteredLeftCol = cudf::column_view{filteredLeftSpan};
          auto indeterminate = cudf::contains(
              filteredLeftCol, probeRowIndices->view(), stream, get_temp_mr());

          indeterminateCol = cudf::binary_operation(
              indeterminateCol->view(),
              indeterminate->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
        };

    bool buildSideEmpty = true;
    for (const auto& rt : rightTables) {
      if (rt->num_rows() > 0) {
        buildSideEmpty = false;
        break;
      }
    }

    // For empty build side, IN returns FALSE (already set in matchCol).
    if (!buildSideEmpty) {
      auto probeKeyView = leftTableView.select(leftKeyIndices_);
      bool probeHasNulls = cudf::has_nulls(probeKeyView);

      // Compute probe key null mask upfront
      auto probeKeyNullMask =
          createProbeKeyNullMask(probeKeyView, stream, get_temp_mr());

      // Initialize indeterminate column to all false
      auto falseScalar =
          cudf::numeric_scalar<bool>(false, true, stream, get_temp_mr());
      auto indeterminateCol = cudf::make_column_from_scalar(
          falseScalar, numProbeRows, stream, get_temp_mr());

      // Process each build batch for indeterminate cases
      for (size_t i = 0; i < rightTables.size(); i++) {
        auto rightTableView = rightTables[i]->view();
        auto buildKeyView = rightTableView.select(rightKeyIndices_);
        bool buildBatchHasNullKeys = cudf::has_nulls(buildKeyView);
        auto numBuildRows = rightTableView.num_rows();

        if (numBuildRows == 0) {
          continue;
        }

        // Get extended views for filter evaluation
        cudf::table_view extendedRightView =
            (!rightPrecomputeInstructions_.empty())
            ? cachedExtendedRightViews_[i]
            : rightTableView;

        // Type B: Null probe keys × all build rows
        if (probeHasNulls) {
          auto nullProbeIndices = getMaskedIndices(
              probeKeyNullMask->view(),
              MaskType::kMatched,
              stream,
              get_temp_mr());
          auto allBuildIndices = cudf::sequence(
              numBuildRows,
              cudf::numeric_scalar<cudf::size_type>(
                  0, true, stream, get_temp_mr()),
              cudf::numeric_scalar<cudf::size_type>(
                  1, true, stream, get_temp_mr()),
              stream,
              get_temp_mr());

          accumulateIndeterminate(
              nullProbeIndices->view(),
              allBuildIndices->view(),
              extendedRightView,
              indeterminateCol);
        }

        // Type A: Non-null, non-matching probe keys × null-key build rows
        if (buildBatchHasNullKeys) {
          auto notProbeNull = cudf::unary_operation(
              probeKeyNullMask->view(),
              cudf::unary_operator::NOT,
              stream,
              get_temp_mr());
          auto noMatch = cudf::unary_operation(
              matchCol->view(),
              cudf::unary_operator::NOT,
              stream,
              get_temp_mr());
          auto typeAMask = cudf::binary_operation(
              notProbeNull->view(),
              noMatch->view(),
              cudf::binary_operator::BITWISE_AND,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());

          auto typeAProbeIndices = getMaskedIndices(
              typeAMask->view(), MaskType::kMatched, stream, get_temp_mr());
          auto buildKeyNullMask =
              createProbeKeyNullMask(buildKeyView, stream, get_temp_mr());
          auto nullBuildIndices = getMaskedIndices(
              buildKeyNullMask->view(),
              MaskType::kMatched,
              stream,
              get_temp_mr());

          accumulateIndeterminate(
              typeAProbeIndices->view(),
              nullBuildIndices->view(),
              extendedRightView,
              indeterminateCol);
        }
      }

      // Apply three-valued logic:
      // - Where matchCol is TRUE → keep TRUE (takes precedence)
      // - Where matchCol is FALSE and indeterminateCol is TRUE → set NULL
      // - Where matchCol is FALSE and indeterminateCol is FALSE → keep FALSE
      auto notMatch = cudf::unary_operation(
          matchCol->view(), cudf::unary_operator::NOT, stream, get_temp_mr());
      auto shouldBeNull = cudf::binary_operation(
          notMatch->view(),
          indeterminateCol->view(),
          cudf::binary_operator::BITWISE_AND,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          get_temp_mr());

      matchCol = applyNullMask(
          matchCol->view(), shouldBeNull->view(), stream, get_output_mr());
    }
  } else if (isNullAwareWithoutFilter) {
    // Original null-aware without filter logic
    bool buildSideEmpty = true;
    for (const auto& rt : rightTables) {
      if (rt->num_rows() > 0) {
        buildSideEmpty = false;
        break;
      }
    }

    // For empty build side, IN returns FALSE (already set in matchCol).
    if (!buildSideEmpty) {
      auto probeKeyView = leftTableView.select(leftKeyIndices_);
      bool probeHasNulls = cudf::has_nulls(probeKeyView);

      if (probeHasNulls || buildSideHasNullKeys_) {
        // Compute null mask: true where result should be NULL
        auto probeKeyNullMask =
            createProbeKeyNullMask(probeKeyView, stream, get_temp_mr());

        std::unique_ptr<cudf::column> nullMask;
        if (buildSideHasNullKeys_) {
          // NULL where: probe key is NULL OR no match
          auto noMatchMask = cudf::unary_operation(
              matchCol->view(),
              cudf::unary_operator::NOT,
              stream,
              get_temp_mr());
          nullMask = cudf::binary_operation(
              probeKeyNullMask->view(),
              noMatchMask->view(),
              cudf::binary_operator::BITWISE_OR,
              cudf::data_type{cudf::type_id::BOOL8},
              stream,
              get_temp_mr());
        } else {
          // NULL only where probe key is NULL
          nullMask = std::move(probeKeyNullMask);
        }

        matchCol = applyNullMask(
            matchCol->view(), nullMask->view(), stream, get_output_mr());
      }
    }
  }

  // Step 6: Build output table with all probe columns + match column
  std::vector<std::unique_ptr<cudf::column>> outputCols;
  outputCols.resize(outputType_->names().size());

  // Copy probe columns
  auto leftInput = leftTableView.select(leftColumnIndicesToGather_);
  for (size_t i = 0; i < leftColumnIndicesToGather_.size(); i++) {
    outputCols[leftColumnOutputIndices_[i]] = std::make_unique<cudf::column>(
        leftInput.column(i), stream, get_output_mr());
  }

  // Add match column as the last column
  outputCols.back() = std::move(matchCol);

  if (buildStream_.has_value()) {
    cudaEvent_->recordFrom(stream).waitOn(buildStream_.value());
  }
  stream.synchronize();

  auto output = std::make_unique<cudf::table>(std::move(outputCols));
  cudfOutputs.push_back(
      {std::move(output), static_cast<vector_size_t>(numProbeRows)});
  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::rightSemiFilterJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;

  auto& rightTables = hashObject_.value().first;
  auto rightTableView = rightTables[0]->view();

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for rightSemiFilterJoin");

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightJoinIndices;
  if (joinNode_->filter()) {
    rightJoinIndices = cudf::mixed_left_semi_join(
        rightTableView.select(rightKeyIndices_),
        leftTableView.select(leftKeyIndices_),
        rightTableView,
        leftTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        get_temp_mr());
  } else {
    cudf::filtered_join filter_join(
        leftTableView.select(leftKeyIndices_),
        cudf::null_equality::UNEQUAL,
        stream);
    rightJoinIndices = filter_join.semi_join(
        rightTableView.select(rightKeyIndices_), stream, get_temp_mr());
  }

  auto rightIndicesSpan = toSpan(*rightJoinIndices);
  auto rightIndicesCol = cudf::column_view{rightIndicesSpan};
  auto leftIndicesCol = cudf::empty_like(rightIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol->view(),
      rightTableView,
      rightIndicesCol,
      stream));

  return cudfOutputs;
}

std::vector<CudfHashJoinProbe::JoinOutput>
CudfHashJoinProbe::rightSemiProjectJoin(
    cudf::table_view leftTableView,
    rmm::cuda_stream_view stream) {
  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  VELOX_CHECK_EQ(rightTables.size(), hbs.size());
  VELOX_CHECK_EQ(rightTables.size(), rightMatchedFlags_.size());

  // Precompute probe columns once per input batch when the AST filter needs
  // them. HashJoinNode rejects null-aware RIGHT SEMI PROJECT with a filter.
  std::vector<ColumnOrView> leftPrecomputed;
  cudf::table_view extendedLeftView = leftTableView;
  if (joinNode_->filter() && useAstFilter_ &&
      !leftPrecomputeInstructions_.empty()) {
    auto leftColumnViews = tableViewToColumnViews(leftTableView);
    leftPrecomputed = precomputeSubexpressions(
        leftColumnViews,
        leftPrecomputeInstructions_,
        scalars_,
        probeType_,
        stream);
    extendedLeftView = createExtendedTableView(leftTableView, leftPrecomputed);
  }

  for (size_t i = 0; i < rightTables.size(); ++i) {
    auto rightTableView = rightTables[i]->view();
    auto& hb = hbs[i];
    VELOX_CHECK_NOT_NULL(hb);

    auto [leftJoinIndices, rightJoinIndices] = hb->inner_join(
        leftTableView.select(leftKeyIndices_),
        std::nullopt,
        stream,
        get_temp_mr());

    auto leftIndicesCol = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*leftJoinIndices}};
    auto rightIndicesCol = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*rightJoinIndices}};
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>
        astFilteredRightIndices;
    std::unique_ptr<cudf::column> evaluatedFilteredRightIndices;

    if (joinNode_->filter()) {
      if (useAstFilter_) {
        auto extendedRightView = !rightPrecomputeInstructions_.empty()
            ? cachedExtendedRightViews_[i]
            : rightTableView;
        auto filteredIndices = cudf::filter_join_indices(
            extendedLeftView,
            extendedRightView,
            leftIndicesCol,
            rightIndicesCol,
            tree_.back(),
            cudf::join_kind::INNER_JOIN,
            stream,
            get_temp_mr());
        astFilteredRightIndices = std::move(filteredIndices.second);
        rightIndicesCol = cudf::column_view{
            cudf::device_span<cudf::size_type const>{*astFilteredRightIndices}};
      } else {
        // Evaluate non-AST residuals over candidate pairs, then apply the same
        // mask to the build indices. Null filter results are excluded by
        // apply_boolean_mask, matching SQL join predicate semantics.
        auto leftResult = cudf::gather(
            leftTableView, leftIndicesCol, oobPolicy, stream, get_temp_mr());
        auto rightResult = cudf::gather(
            rightTableView, rightIndicesCol, oobPolicy, stream, get_temp_mr());
        auto joinedCols = leftResult->release();
        auto rightCols = rightResult->release();
        joinedCols.insert(
            joinedCols.end(),
            std::make_move_iterator(rightCols.begin()),
            std::make_move_iterator(rightCols.end()));
        std::vector<cudf::column_view> joinedViews;
        joinedViews.reserve(joinedCols.size());
        for (const auto& col : joinedCols) {
          joinedViews.push_back(col->view());
        }
        VELOX_CHECK_NOT_NULL(filterEvaluator_);
        auto filterColumn =
            filterEvaluator_->eval(joinedViews, stream, get_temp_mr());
        auto filteredTable = cudf::apply_boolean_mask(
            cudf::table_view{{rightIndicesCol}},
            asView(filterColumn),
            stream,
            get_temp_mr());
        evaluatedFilteredRightIndices = std::move(filteredTable->release()[0]);
        rightIndicesCol = evaluatedFilteredRightIndices->view();
      }
    }

    // Mark only the build rows matched by this probe batch. The previous
    // implementation materialized sequence(buildRows), evaluated contains()
    // over the complete build side, and OR'ed a complete BOOL8 column for
    // every probe batch. Large exchange consumers can receive thousands of
    // batches, turning an otherwise incremental semi join into
    // O(buildRows * probeBatches) work. Each driver owns its flag column, and
    // setting TRUE is monotonic and idempotent even when a residual produces
    // duplicate build indices.
    connector::hive::iceberg::scatterDeletesToMask(
        rightMatchedFlags_[i]->mutable_view(),
        cudf::device_span<cudf::size_type const>{
            rightIndicesCol.data<cudf::size_type>(),
            static_cast<std::size_t>(rightIndicesCol.size())},
        stream,
        get_temp_mr());
    // rightIndicesCol can reference a batch-local device buffer. Complete the
    // scatter before its owner is destroyed and before this driver accepts the
    // next probe batch.
    stream.synchronize();
  }

  // RIGHT SEMI PROJECT is build preserving. Probe batches only update state;
  // output is emitted once every driver has completed probing.
  return {};
}

RowVectorPtr CudfHashJoinProbe::rightSemiProjectOutput(
    rmm::cuda_stream_view stream) {
  auto& rightTables = hashObject_.value().first;
  VELOX_CHECK_EQ(rightTables.size(), rightMatchedFlags_.size());

  while (nextBuildOutputIndex_ < rightTables.size()) {
    const auto i = nextBuildOutputIndex_++;
    auto rightTableView = rightTables[i]->view();
    const auto numRows = rightTableView.num_rows();
    if (numRows == 0) {
      continue;
    }

    std::vector<std::unique_ptr<cudf::column>> outputCols(outputType_->size());
    auto rightInput = rightTableView.select(rightColumnIndicesToGather_);
    for (size_t j = 0; j < rightColumnIndicesToGather_.size(); ++j) {
      outputCols[rightColumnOutputIndices_[j]] = std::make_unique<cudf::column>(
          rightInput.column(j), stream, get_output_mr());
    }

    std::unique_ptr<cudf::column> matchColumn;
    if (joinNode_->isNullAware() && probeSideHasRows_) {
      auto noMatch = cudf::unary_operation(
          rightMatchedFlags_[i]->view(),
          cudf::unary_operator::NOT,
          stream,
          get_temp_mr());
      std::unique_ptr<cudf::column> nullMask;
      if (probeSideHasNullKeys_) {
        // Any unmatched build row is indeterminate if the probe side contains
        // a null key. Rows with a true equality match remain true.
        nullMask = std::move(noMatch);
      } else {
        // With a non-empty, non-null probe side, only build rows with a null
        // key are indeterminate.
        auto buildKeyNullMask = createProbeKeyNullMask(
            rightTableView.select(rightKeyIndices_), stream, get_temp_mr());
        nullMask = cudf::binary_operation(
            noMatch->view(),
            buildKeyNullMask->view(),
            cudf::binary_operator::BITWISE_AND,
            cudf::data_type{cudf::type_id::BOOL8},
            stream,
            get_temp_mr());
      }
      matchColumn = applyNullMask(
          rightMatchedFlags_[i]->view(),
          nullMask->view(),
          stream,
          get_output_mr());
    } else {
      // Regular EXISTS semantics, and null-aware IN with an empty probe side,
      // both use the non-nullable accumulated flags directly.
      matchColumn = std::make_unique<cudf::column>(
          rightMatchedFlags_[i]->view(), stream, get_output_mr());
    }
    outputCols.back() = std::move(matchColumn);

    stream.synchronize();
    finished_ = nextBuildOutputIndex_ == rightTables.size();
    return std::make_shared<CudfVector>(
        pool(),
        outputType_,
        static_cast<vector_size_t>(numRows),
        std::make_unique<cudf::table>(std::move(outputCols)),
        stream);
  }

  finished_ = true;
  return nullptr;
}

std::vector<CudfHashJoinProbe::JoinOutput> CudfHashJoinProbe::antiJoin(
    cudf::table_view leftTableViewParam,
    rmm::cuda_stream_view stream) {
  std::vector<JoinOutput> cudfOutputs;
  auto& rightTables = hashObject_.value().first;

  VELOX_CHECK_EQ(
      rightTables.size(),
      1,
      "Multiple right tables not yet supported for antiJoin");

  auto rightTableView = rightTables[0]->view();

  // For the special case where we need to drop nulls, we create a local table.
  // Otherwise, we use the input view directly.
  std::unique_ptr<cudf::table> modifiedLeftTable;
  cudf::table_view leftTableView = leftTableViewParam;

  // Special case for null-aware anti join where
  // build table is not empty, no nulls, and probe table has nulls
  if (joinNode_->isNullAware() and !joinNode_->filter()) {
    auto const leftTableHasNulls =
        cudf::has_nulls(leftTableViewParam.select(leftKeyIndices_));
    auto const rightTableHasNulls =
        cudf::has_nulls(rightTableView.select(rightKeyIndices_));
    if (rightTables[0]->num_rows() > 0 and !rightTableHasNulls and
        leftTableHasNulls) {
      // drop nulls on probe table - creates a new table
      modifiedLeftTable = cudf::drop_nulls(
          leftTableViewParam, leftKeyIndices_, stream, get_temp_mr());
      leftTableView = modifiedLeftTable->view();
    }
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftJoinIndices;
  if (joinNode_->filter()) {
    leftJoinIndices = cudf::mixed_left_anti_join(
        leftTableView.select(leftKeyIndices_),
        rightTableView.select(rightKeyIndices_),
        leftTableView,
        rightTableView,
        tree_.back(),
        cudf::null_equality::UNEQUAL,
        stream,
        get_temp_mr());
  } else {
    auto const rightTableHasNulls =
        cudf::has_nulls(rightTableView.select(rightKeyIndices_));
    if (joinNode_->isNullAware() and rightTableHasNulls) {
      // empty result
      leftJoinIndices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
          0, stream, get_temp_mr());
    } else {
      cudf::filtered_join filter_join(
          rightTableView.select(rightKeyIndices_),
          cudf::null_equality::UNEQUAL,
          stream);
      leftJoinIndices = filter_join.anti_join(
          leftTableView.select(leftKeyIndices_), stream, get_temp_mr());
    }
  }

  auto leftIndicesSpan = toSpan(*leftJoinIndices);
  auto leftIndicesCol = cudf::column_view{leftIndicesSpan};
  auto rightIndicesCol = cudf::empty_like(leftIndicesCol);
  cudfOutputs.push_back(unfilteredOutput(
      leftTableView,
      leftIndicesCol,
      rightTableView,
      rightIndicesCol->view(),
      stream));

  return cudfOutputs;
}

void CudfHashJoinProbe::initializeRightMatchedFlags(
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(hashObject_.has_value());
  auto& rightTables = hashObject_.value().first;
  rightMatchedFlags_.clear();
  rightMatchedFlags_.reserve(rightTables.size());
  for (auto& table : rightTables) {
    auto falseScalar =
        cudf::numeric_scalar<bool>(false, true, stream, get_temp_mr());
    rightMatchedFlags_.push_back(
        cudf::make_column_from_scalar(
            falseScalar, table->num_rows(), stream, get_temp_mr()));
  }
  stream.synchronize();
}

RowVectorPtr CudfHashJoinProbe::rightUnmatchedOutput(
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(hashObject_.has_value());
  auto& rightTables = hashObject_.value().first;
  VELOX_CHECK_EQ(rightTables.size(), rightMatchedFlags_.size());
  std::vector<std::unique_ptr<cudf::table>> toConcat;
  vector_size_t unmatchedRows = 0;
  for (size_t i = 0; i < rightTables.size(); ++i) {
    auto& rightTable = rightTables[i];
    const auto n = rightTable->num_rows();
    if (n == 0) {
      continue;
    }
    auto boolMask = cudf::unary_operation(
        rightMatchedFlags_[i]->view(),
        cudf::unary_operator::NOT,
        stream,
        get_temp_mr());
    auto unmatchedCountScalar = cudf::reduce(
        boolMask->view(),
        *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type{cudf::type_id::INT32},
        stream,
        get_temp_mr());
    const auto unmatched =
        static_cast<cudf::numeric_scalar<int32_t>*>(unmatchedCountScalar.get())
            ->value(stream);
    if (unmatched == 0) {
      continue;
    }
    unmatchedRows += unmatched;

    std::vector<std::unique_ptr<cudf::column>> outCols(outputType_->size());
    for (size_t li = 0; li < leftColumnOutputIndices_.size(); ++li) {
      const auto outIdx = leftColumnOutputIndices_[li];
      const auto probeChannel = leftColumnIndicesToGather_[li];
      outCols[outIdx] = makeAllNullColumnForType(
          probeType_->childAt(probeChannel),
          unmatched,
          stream,
          get_output_mr());
    }
    if (!rightColumnIndicesToGather_.empty()) {
      auto unmatchedRight = cudf::apply_boolean_mask(
          rightTable->view().select(rightColumnIndicesToGather_),
          boolMask->view(),
          stream,
          get_output_mr());
      auto rightCols = unmatchedRight->release();
      for (size_t ri = 0; ri < rightColumnOutputIndices_.size(); ++ri) {
        outCols[rightColumnOutputIndices_[ri]] = std::move(rightCols[ri]);
      }
    }
    toConcat.push_back(std::make_unique<cudf::table>(std::move(outCols)));
  }
  if (toConcat.empty()) {
    return nullptr;
  }
  auto output = concatenateTables(std::move(toConcat), stream, get_output_mr());
  const auto rows =
      outputType_->size() == 0 ? unmatchedRows : output->num_rows();
  if (rows == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(), outputType_, rows, std::move(output), stream);
}

RowVectorPtr CudfHashJoinProbe::graceRightNullUnmatchedOutput(
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(joinNode_->isRightJoin());
  VELOX_CHECK_NOT_NULL(graceBuildData_);
  VELOX_CHECK_LT(graceNullBuildChunk_, graceBuildData_->unmatchedNulls.size());
  auto& host = graceBuildData_->unmatchedNulls[graceNullBuildChunk_++];
  const auto rows = host.rows;
  auto build = restoreHostBatch(host, buildType_, stream, true);

  std::vector<std::unique_ptr<cudf::column>> outCols(outputType_->size());
  for (size_t li = 0; li < leftColumnOutputIndices_.size(); ++li) {
    const auto outIdx = leftColumnOutputIndices_[li];
    const auto probeChannel = leftColumnIndicesToGather_[li];
    outCols[outIdx] = makeAllNullColumnForType(
        probeType_->childAt(probeChannel), rows, stream, get_output_mr());
  }
  if (!rightColumnIndicesToGather_.empty()) {
    auto right = std::make_unique<cudf::table>(
        build->getTableView().select(rightColumnIndicesToGather_),
        stream,
        get_output_mr());
    auto rightCols = right->release();
    for (size_t ri = 0; ri < rightColumnOutputIndices_.size(); ++ri) {
      outCols[rightColumnOutputIndices_[ri]] = std::move(rightCols[ri]);
    }
  }
  auto output = std::make_unique<cudf::table>(std::move(outCols));
  stream.synchronize();
  LOG(WARNING) << "CudfHashJoinProbe node=" << planNodeId()
               << " emitted null-key Grace RIGHT build rows=" << rows
               << " chunk=" << graceNullBuildChunk_ << "/"
               << graceBuildData_->unmatchedNulls.size();
  return std::make_shared<CudfVector>(
      pool(), outputType_, rows, std::move(output), stream);
}

RowVectorPtr CudfHashJoinProbe::doGetOutput() {
  if (graceBuildData_) {
    return getGraceOutput();
  }
  if (finished_ or !hashObject_.has_value()) {
    return nullptr;
  }
  if (!input_) {
    if (joinNode_->isRightSemiProjectJoin() && noMoreInput_ && !finished_ &&
        isLastDriver_) {
      return rightSemiProjectOutput(cudfGlobalStreamPool().get_stream());
    }
    // If no more input, emit unmatched-right rows if needed.
    if ((joinNode_->isRightJoin() || joinNode_->isFullJoin()) && noMoreInput_ &&
        !finished_ && isLastDriver_) {
      auto stream = cudfGlobalStreamPool().get_stream();
      auto output = rightUnmatchedOutput(stream);
      finished_ = true;
      return output;
    }
    return nullptr;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
  VELOX_CHECK_NOT_NULL(cudfInput);
  auto stream = cudfInput->stream();
  waitForBuildReady(stream);
  // Use getTableView() to avoid expensive materialization for packed_table.
  // cudfInput is staying alive until the table view is no longer needed.
  auto leftTableView = cudfInput->getTableView();
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(1) << "Probe table number of columns: " << leftTableView.num_columns();
    VLOG(1) << "Probe table number of rows: " << leftTableView.num_rows();
  }

  auto& rightTables = hashObject_.value().first;
  auto& hbs = hashObject_.value().second;
  for (auto i = 0; i < rightTables.size(); i++) {
    auto& rightTable = rightTables[i];
    auto& hb = hbs[i];
    VELOX_CHECK_NOT_NULL(rightTable);
    if (CudfConfig::getInstance().debugEnabled) {
      if (rightTable != nullptr)
        VLOG(2) << "right_table is not nullptr " << rightTable.get()
                << " hasValue(" << hashObject_.has_value() << ")\n";
      if (hb != nullptr)
        VLOG(2) << "hb is not nullptr " << hb.get() << " hasValue("
                << hashObject_.has_value() << ")\n";
    }
  }

  std::vector<JoinOutput> cudfOutputs;
  {
    std::lock_guard<std::mutex> cucoLock(cudfCucoMutex());
    switch (joinNode_->joinType()) {
      case core::JoinType::kInner:
        cudfOutputs = innerJoin(leftTableView, stream);
        break;
      case core::JoinType::kLeft:
        cudfOutputs = leftJoin(leftTableView, stream);
        break;
      case core::JoinType::kRight:
        cudfOutputs = rightJoin(leftTableView, stream);
        break;
      case core::JoinType::kLeftSemiFilter:
        cudfOutputs = leftSemiFilterJoin(leftTableView, stream);
        break;
      case core::JoinType::kLeftSemiProject:
        cudfOutputs = leftSemiProjectJoin(leftTableView, stream);
        break;
      case core::JoinType::kRightSemiFilter:
        cudfOutputs = rightSemiFilterJoin(leftTableView, stream);
        break;
      case core::JoinType::kRightSemiProject:
        cudfOutputs = rightSemiProjectJoin(leftTableView, stream);
        break;
      case core::JoinType::kAnti:
        cudfOutputs = antiJoin(leftTableView, stream);
        break;
      case core::JoinType::kFull:
        cudfOutputs = fullJoin(leftTableView, stream);
        break;
      default:
        VELOX_FAIL("Unsupported join type: ", joinNode_->joinType());
    }
    // Keep probe inputs and request-local temporaries alive until this
    // submission has completed, without a device-wide barrier.
    stream.synchronize();
  }

  // Record probe stream for cross-driver synchronization in noMoreInput().
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin() ||
      joinNode_->isRightSemiProjectJoin()) {
    lastProbeStream_ = stream;
  }

  // Release input CudfVector to free GPU memory before creating output.
  // This reduces peak memory from (input + output) to max(input, output).
  // cudfInput must be released first since input_.reset() only decrements
  // the refcount while cudfInput still holds a reference.
  cudfInput.reset();
  input_.reset();
  finished_ = noMoreInput_ && !joinNode_->isRightJoin() &&
      !joinNode_->isFullJoin() && !joinNode_->isRightSemiProjectJoin();

  if (joinNode_->isRightSemiProjectJoin()) {
    VELOX_CHECK(cudfOutputs.empty());
    return nullptr;
  }

  vector_size_t zeroColumnOutputRows = 0;
  std::vector<std::unique_ptr<cudf::table>> cudfOutputTables;
  cudfOutputTables.reserve(cudfOutputs.size());
  for (auto& output : cudfOutputs) {
    zeroColumnOutputRows += output.numRows;
    cudfOutputTables.push_back(std::move(output.table));
  }

  if (cudfOutputTables.empty()) {
    return nullptr;
  }

  auto cudfOutput =
      concatenateTables(std::move(cudfOutputTables), stream, get_output_mr());
  auto const size =
      outputType_->size() == 0 ? zeroColumnOutputRows : cudfOutput->num_rows();
  if (size == 0) {
    return nullptr;
  }
  return std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(cudfOutput), stream);
}

bool CudfHashJoinProbe::skipProbeOnEmptyBuild() const {
  auto const joinType = joinNode_->joinType();
  return isInnerJoin(joinType) || isLeftSemiFilterJoin(joinType) ||
      isRightJoin(joinType) || isRightSemiFilterJoin(joinType) ||
      isRightSemiProjectJoin(joinType);
}

exec::BlockingReason CudfHashJoinProbe::isBlocked(ContinueFuture* future) {
  if (graceWorkspace_.takeFuture(future)) {
    return exec::BlockingReason::kWaitForArbitration;
  }
  if (graceEnabled_ && noMoreInput_ && future_.valid()) {
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isRightSemiProjectJoin() || joinNode_->isFullJoin()) &&
      hashObject_.has_value()) {
    if (!future_.valid()) {
      return exec::BlockingReason::kNotBlocked;
    }
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }

  if (hashObject_.has_value() || graceBuildData_) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto cudfJoinBridge =
      std::dynamic_pointer_cast<CudfHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(cudfJoinBridge);
  VELOX_CHECK_NOT_NULL(future);
  if (graceEnabled_) {
    std::optional<CudfHashJoinBridge::BuildResult> result;
    if (graceEagerActive_) {
      result = cudfJoinBridge->tryFinalResult();
      if (!result.has_value()) {
        if (!noMoreInput_ &&
            graceProbeBufferedBytes_ < graceEagerProbeBufferLimitBytes_) {
          return exec::BlockingReason::kNotBlocked;
        }
        result = cudfJoinBridge->finalResultOrFuture(future);
      }
    } else {
      result = cudfJoinBridge->resultOrFuture(future);
    }
    if (!result.has_value()) {
      return exec::BlockingReason::kWaitForJoinBuild;
    }
    if (result->graceActivated && !result->grace && !result->hash.has_value()) {
      if (graceEagerProbeBufferLimitBytes_ > 0) {
        graceEagerActive_ = true;
        LOG(WARNING)
            << "CudfHashJoinProbe task=" << operatorCtx_->task()->taskId()
            << " node=" << planNodeId()
            << " starting eager Grace probe partitioning hostLimitBytes="
            << graceEagerProbeBufferLimitBytes_;
        return exec::BlockingReason::kNotBlocked;
      }
      result = cudfJoinBridge->finalResultOrFuture(future);
      if (!result.has_value()) {
        return exec::BlockingReason::kWaitForJoinBuild;
      }
    }
    if (result->grace) {
      graceBuildData_ = std::move(result->grace);
      if (noMoreInput_ && isLastDriver_) {
        graceProbeDraining_ = true;
        LOG(WARNING) << "CudfHashJoinProbe task="
                     << operatorCtx_->task()->taskId()
                     << " node=" << planNodeId()
                     << " build ready; starting partition-major Grace drain "
                     << "bytes=" << graceProbeBufferedBytes_
                     << " partitions=" << graceProbePartitions_.size();
      }
      return exec::BlockingReason::kNotBlocked;
    }
    VELOX_CHECK_EQ(
        graceProbeBufferedBytes_,
        0,
        "Grace activation cannot publish a regular hash build after eager "
        "probe partitioning");
    hashObject_ = std::move(result->hash);
    // Grace is configured adaptively. A build that remains below the
    // threshold publishes the normal in-device hash table; from this point
    // the probe must follow the normal completion state machine.
    graceEnabled_ = false;
  } else {
    auto hashObject = cudfJoinBridge->hashOrFuture(future);

    if (!hashObject.has_value()) {
      if (CudfConfig::getInstance().debugEnabled) {
        VLOG(2) << "CudfHashJoinProbe is blocked, waiting for join build";
      }
      return exec::BlockingReason::kWaitForJoinBuild;
    }
    hashObject_ = std::move(hashObject);
  }
  buildStream_ = cudfJoinBridge->getBuildStream();
  buildReadyEvent_ = cudfJoinBridge->getBuildReadyEvent();

  // Lazy initialize matched flags only when build side is done
  if (joinNode_->isRightJoin() || joinNode_->isFullJoin() ||
      joinNode_->isRightSemiProjectJoin()) {
    auto initStream = cudfGlobalStreamPool().get_stream();
    waitForBuildReady(initStream);
    initializeRightMatchedFlags(initStream);
  }

  // Precompute right table columns if filter exists (once when build is done)
  if (joinNode_->filter() && !rightPrecomputeInstructions_.empty()) {
    auto& rightTablesInit = hashObject_.value().first;
    cachedRightPrecomputed_.clear();
    cachedExtendedRightViews_.clear();
    cachedRightPrecomputed_.reserve(rightTablesInit.size());
    cachedExtendedRightViews_.reserve(rightTablesInit.size());

    auto initStream = cudfGlobalStreamPool().get_stream();
    waitForBuildReady(initStream);
    for (auto& rt : rightTablesInit) {
      auto rightTableView = rt->view();
      auto rightColumnViews = tableViewToColumnViews(rightTableView);
      auto rightPrecomputed = precomputeSubexpressions(
          rightColumnViews,
          rightPrecomputeInstructions_,
          scalars_,
          buildType_,
          initStream);
      auto extendedView =
          createExtendedTableView(rightTableView, rightPrecomputed);
      cachedRightPrecomputed_.push_back(std::move(rightPrecomputed));
      cachedExtendedRightViews_.push_back(extendedView);
    }
    initStream.synchronize();
  }

  // Check if build side has any null keys (needed for null-aware left semi
  // project)
  if (joinNode_->isLeftSemiProjectJoin() && joinNode_->isNullAware()) {
    auto& rightTablesInit = hashObject_.value().first;
    buildSideHasNullKeys_ = false;
    for (auto& rt : rightTablesInit) {
      auto keyView = rt->view().select(rightKeyIndices_);
      for (cudf::size_type k = 0; k < keyView.num_columns(); k++) {
        if (keyView.column(k).has_nulls()) {
          buildSideHasNullKeys_ = true;
          break;
        }
      }
      if (buildSideHasNullKeys_) {
        break;
      }
    }
  }

  auto& rightTables = hashObject_.value().first;
  // should be rightTable->numDistinct() but it needs compute,
  // so we use num_rows()
  if (rightTables[0]->num_rows() == 0) {
    if (skipProbeOnEmptyBuild()) {
      if (operatorCtx_->driverCtx()
              ->queryConfig()
              .hashProbeFinishEarlyOnEmptyBuild()) {
        noMoreInput();
      } else {
        skipInput_ = true;
      }
    }
  }
  if ((joinNode_->isRightJoin() || joinNode_->isRightSemiFilterJoin() ||
       joinNode_->isRightSemiProjectJoin() || joinNode_->isFullJoin()) &&
      future_.valid()) {
    *future = std::move(future_);
    return exec::BlockingReason::kWaitForJoinProbe;
  }
  return exec::BlockingReason::kNotBlocked;
}

bool CudfHashJoinProbe::isFinished() {
  if (graceBuildData_ || (graceEnabled_ && !graceFinished_)) {
    return graceFinished_;
  }
  // RIGHT SEMI PROJECT can emit one build table per getOutput() call. Keep the
  // last driver alive until all build tables are emitted; peer drivers have no
  // output after the end-of-probe barrier and can finish normally.
  const auto hasNoMoreWork = noMoreInput_ && input_ == nullptr &&
      !(joinNode_->isRightSemiProjectJoin() && isLastDriver_ && !finished_);
  const auto isFinished = finished_ || hasNoMoreWork;

  // Release hashObject_ if finished
  if (isFinished) {
    hashObject_.reset();
    buildReadyEvent_.reset();
    buildStream_.reset();
  }
  return isFinished;
}

std::unique_ptr<exec::Operator> CudfHashJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperator";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return std::make_unique<CudfHashJoinProbe>(id, ctx, joinNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge> CudfHashJoinBridgeTranslator::toJoinBridge(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toJoinBridge";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    auto joinBridge = std::make_unique<CudfHashJoinBridge>();
    return joinBridge;
  }
  return nullptr;
}

exec::OperatorSupplier CudfHashJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  if (CudfConfig::getInstance().debugEnabled) {
    VLOG(2) << "Calling CudfHashJoinBridgeTranslator::toOperatorSupplier";
  }
  if (auto joinNode =
          std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
