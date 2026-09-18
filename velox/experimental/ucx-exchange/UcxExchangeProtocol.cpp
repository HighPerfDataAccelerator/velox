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

#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

#include <cuda_runtime.h>
#include <cudf/utilities/error.hpp>
#include <glog/logging.h>
#include <nvtx3/nvtx3.hpp>
#include <sys/resource.h>
#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include "velox/common/base/Exceptions.h"
#include "velox/experimental/cudf/exec/HostStagingCopy.h"

namespace facebook::velox::ucx_exchange {

namespace {
// Dedicated CPU-only workers: submitting staging threads may wait here, but
// never submit back into their own executor. No additional data buffers or
// pinned leases are allocated. Existing staging admission bounds callers.
bool copyPageableHost(void* destination, const void* source, uint64_t bytes) {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_UCX_PARALLEL_PAGEABLE_COPY");
    return value && std::string_view(value) == "1";
  }();
  return cudf_velox::host_staging::copyPageableHost(
      destination,
      source,
      bytes,
      enabled,
      "UcxHost::parallelPageableCopyChunk");
}

bool pinnedLeaseDiagnostics() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_UCX_PINNED_LEASE_DIAGNOSTICS");
    return value && std::string_view(value) == "1";
  }();
  return enabled;
}

size_t localPinnedQueueLimit() {
  static const size_t limit = [] {
    const auto* value = std::getenv("GLUTEN_UCX_LOCAL_PINNED_QUEUE_LIMIT");
    if (!value) {
      return size_t{0};
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    VELOX_USER_CHECK(
        end != value && *end == '\0' && parsed <= 16,
        "GLUTEN_UCX_LOCAL_PINNED_QUEUE_LIMIT must be in [0, 16]");
    return static_cast<size_t>(parsed);
  }();
  return limit;
}

class UcxPinnedBufferPool {
 public:
  UcxPinnedBufferPool(const char* countEnv, const char* label) : label_{label} {
    if (const char* value = std::getenv(countEnv)) {
      char* end = nullptr;
      const auto requested = std::strtoull(value, &end, 10);
      if (end != value && *end == '\0') {
        maxSlots_ = std::clamp<uint64_t>(requested, 1, 16);
      }
    }
  }

  std::shared_ptr<uint8_t> acquire(
      uint64_t requiredBytes,
      const char* owner = "unspecified",
      std::string_view localQueue = {}) {
    if (requiredBytes == 0) {
      return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // Local pages can retain their transport lease behind a join-build
    // barrier. Bound one queue's ownership, rather than waiting for a slot
    // while the queue needed to unblock that consumer cannot make progress.
    // The caller keeps its existing pageable fallback and host accounting.
    const auto queueLimit = localPinnedQueueLimit();
    if (queueLimit && !localQueue.empty()) {
      size_t queueLeases = 0;
      for (const auto& slot : slots_) {
        if (slot->busy && slot->localQueue == localQueue &&
            ++queueLeases >= queueLimit) {
          if (pinnedLeaseDiagnostics()) {
            auto label = std::string("UcxPinnedQueueLimit pool=") + label_ +
                " owner=" + owner + " bytes=" + std::to_string(requiredBytes) +
                " holders=";
            for (const auto& held : slots_) {
              label += std::string(held->busy ? held->owner : "none") + ",";
            }
            nvtx3::scoped_range limited(label.c_str());
          }
          return {};
        }
      }
    }
    size_t available = slots_.size();
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (!slots_[i]->busy && slots_[i]->capacity >= requiredBytes) {
        slots_[i]->busy = true;
        return makeOwner(i, requiredBytes, owner, localQueue);
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
      if (pinnedLeaseDiagnostics()) {
        std::string label = std::string("UcxPinnedMiss pool=") + label_ +
            " owner=" + owner + " bytes=" + std::to_string(requiredBytes) +
            " holders=";
        for (const auto& slot : slots_) {
          label += std::string(slot->owner) + ",";
        }
        nvtx3::scoped_range miss(label.c_str());
      }
      return {};
    }

    auto& slot = *slots_[available];
    uint8_t* allocation = nullptr;
    const auto allocationBytes = std::max<uint64_t>(64ULL << 20, requiredBytes);
    const auto status = cudaHostAlloc(
        reinterpret_cast<void**>(&allocation),
        allocationBytes,
        cudaHostAllocPortable);
    if (status != cudaSuccess) {
      cudaGetLastError();
      LOG(WARNING) << "Could not allocate " << allocationBytes
                   << " bytes for UCX " << label_ << " pinned bounce slot "
                   << available << ": " << cudaGetErrorString(status)
                   << "; falling back to pageable staging";
      return {};
    }
    if (slot.data != nullptr) {
      cudaFreeHost(slot.data);
    }
    slot.data = allocation;
    slot.capacity = allocationBytes;
    slot.busy = true;
    return makeOwner(available, requiredBytes, owner, localQueue);
  }

  ~UcxPinnedBufferPool() {
    for (auto& slot : slots_) {
      if (slot->data != nullptr) {
        cudaFreeHost(slot->data);
      }
    }
  }

 private:
  struct Slot {
    uint8_t* data{nullptr};
    uint64_t capacity{0};
    bool busy{false};
    const char* owner{"none"};
    std::string localQueue;
  };

  struct Lease {
    Lease(
        UcxPinnedBufferPool* pool,
        size_t slot,
        uint64_t bytes,
        const char* owner)
        : pool(pool), slot(slot) {
      if (pinnedLeaseDiagnostics()) {
        const auto label = std::string("UcxPinnedLease pool=") + pool->label_ +
            " owner=" + owner + " slot=" + std::to_string(slot) +
            " bytes=" + std::to_string(bytes) +
            " capacity=" + std::to_string(pool->slots_[slot]->capacity);
        range = std::make_unique<nvtx3::unique_range>(label.c_str());
      }
    }
    ~Lease() {
      std::lock_guard<std::mutex> lock(pool->mutex_);
      VELOX_CHECK(pool->slots_[slot]->busy);
      range.reset();
      pool->slots_[slot]->owner = "none";
      pool->slots_[slot]->localQueue.clear();
      pool->slots_[slot]->busy = false;
    }
    UcxPinnedBufferPool* pool;
    size_t slot;
    std::unique_ptr<nvtx3::unique_range> range;
  };

  std::shared_ptr<uint8_t> makeOwner(
      size_t slot,
      uint64_t bytes,
      const char* owner,
      std::string_view localQueue) {
    try {
      // Copy before constructing Lease: an allocation failure must not destroy
      // an already-created Lease while this mutex is held.
      slots_[slot]->localQueue = localQueue;
      auto lease = std::make_shared<Lease>(this, slot, bytes, owner);
      slots_[slot]->owner = owner;
      return std::shared_ptr<uint8_t>(lease, slots_[slot]->data);
    } catch (...) {
      slots_[slot]->busy = false;
      slots_[slot]->localQueue.clear();
      throw;
    }
  }

  std::mutex mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
  size_t maxSlots_{4};
  const char* const label_;
};

UcxPinnedBufferPool& ucxPinnedBufferPool() {
  static UcxPinnedBufferPool pool(
      "GLUTEN_UCX_PINNED_BUFFER_COUNT", "transport");
  return pool;
}

UcxPinnedBufferPool& ucxH2DPinnedBufferPool() {
  static UcxPinnedBufferPool pool("GLUTEN_UCX_H2D_PINNED_BUFFER_COUNT", "H2D");
  return pool;
}

UcxPinnedBufferPool& ucxD2HPinnedBufferPool() {
  static UcxPinnedBufferPool pool("GLUTEN_UCX_D2H_PINNED_BUFFER_COUNT", "D2H");
  return pool;
}
} // namespace

std::shared_ptr<uint8_t> acquireUcxPinnedBuffer(uint64_t requiredBytes) {
  return ucxPinnedBufferPool().acquire(requiredBytes);
}

std::shared_ptr<uint8_t> acquireUcxPinnedBufferForStage(
    uint64_t requiredBytes,
    bool intraNode,
    std::string_view localQueue) {
  return ucxPinnedBufferPool().acquire(
      requiredBytes,
      intraNode ? "intra" : "remote",
      intraNode ? localQueue : std::string_view{});
}

std::shared_ptr<uint8_t> acquireUcxH2DPinnedBuffer(uint64_t requiredBytes) {
  return ucxH2DPinnedBufferPool().acquire(requiredBytes);
}

std::shared_ptr<uint8_t> acquireUcxD2HPinnedBuffer() {
  return ucxD2HPinnedBufferPool().acquire(kUcxD2HStagingBytes);
}

void initializeUcxStagingPools() {
  (void)ucxPinnedBufferPool();
  (void)ucxH2DPinnedBufferPool();
  (void)ucxD2HPinnedBufferPool();
}

void copyUcxDeviceToPageableHost(
    void* destination,
    const void* source,
    uint64_t bytes,
    rmm::cuda_stream_view stream) {
  if (bytes == 0) {
    return;
  }
  nvtx3::scoped_range transferRange("UcxHost::shortD2H");
  static const bool diagnostics = [] {
    const auto* value = std::getenv("GLUTEN_UCX_HOST_COPY_DIAGNOSTICS");
    return value && value[0] != '\0' && value[0] != '0';
  }();
  std::shared_ptr<uint8_t> scratch;
  {
    nvtx3::scoped_range acquireRange("UcxHost::acquireD2HSlot");
    scratch = acquireUcxD2HPinnedBuffer();
  }
  VELOX_CHECK_NOT_NULL(
      scratch,
      "Bounded UCX D2H scratch unavailable; refusing pageable CUDA fallback");
  auto* dst = static_cast<uint8_t*>(destination);
  auto* src = static_cast<const uint8_t*>(source);
  for (uint64_t offset = 0; offset < bytes;) {
    const auto size = std::min(kUcxD2HStagingBytes, bytes - offset);
    {
      nvtx3::scoped_range dmaRange("UcxHost::shortDmaAndWait");
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          scratch.get(),
          src + offset,
          size,
          cudaMemcpyDeviceToHost,
          stream.value()));
      stream.synchronize();
    }
    {
      nvtx3::scoped_range copyRange("UcxHost::shortMemcpyToPageable");
      rusage before{}, after{};
      const bool beforeValid =
          diagnostics && getrusage(RUSAGE_THREAD, &before) == 0;
      const auto start = diagnostics ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
      const bool parallel = copyPageableHost(dst + offset, scratch.get(), size);
      if (diagnostics) {
        const auto wallNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        // Parent-thread rusage does not account for the parallel copy workers.
        const bool valid =
            !parallel && getrusage(RUSAGE_THREAD, &after) == 0 && beforeValid;
        const auto micros = [](const timeval& value) {
          return int64_t{value.tv_sec} * 1000000 + value.tv_usec;
        };
        LOG(WARNING)
            << "UCX_HOST_COPY bytes=" << size << " wallNs=" << wallNs
            << " parallel=" << parallel << " rusageValid=" << valid
            << " userUs="
            << (valid ? micros(after.ru_utime) - micros(before.ru_utime) : -1)
            << " systemUs="
            << (valid ? micros(after.ru_stime) - micros(before.ru_stime) : -1)
            << " minorFaults="
            << (valid ? after.ru_minflt - before.ru_minflt : -1)
            << " majorFaults="
            << (valid ? after.ru_majflt - before.ru_majflt : -1);
      }
    }
    offset += size;
  }
}

uint32_t fnv1a_32(std::string_view s) {
  uint32_t hash = 0x811C9DC5u; // FNV offset basis
  for (unsigned char c : s) {
    hash ^= c;
    hash *= 0x01000193u; // FNV prime
  }
  return hash;
}

std::pair<std::shared_ptr<uint8_t>, size_t> MetadataMsg::serialize() {
  uint32_t totalSize = getSerializedSize();

  VELOX_CHECK_LE(
      totalSize,
      kMaxMetaBufSize,
      "Metadata serialized size ({}) exceeds maximum buffer size ({}). "
      "This can happen with extremely wide tables. "
      "Consider reducing table width or increasing kMaxMetaBufSize.",
      totalSize,
      kMaxMetaBufSize);

  auto deleter = [](uint8_t* p) { delete[] p; };
  std::shared_ptr<uint8_t> buffer(new uint8_t[totalSize], deleter);

  uint8_t* ptr = buffer.get();

  std::memcpy(ptr, &kMagicNumber, sizeof(kMagicNumber));
  ptr += sizeof(kMagicNumber);

  std::memcpy(ptr, &totalSize, sizeof(totalSize));
  ptr += sizeof(totalSize);

  WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
  std::memcpy(ptr, &cudfSize, sizeof(cudfSize));
  ptr += sizeof(cudfSize);

  if (cudfSize > 0) {
    std::memcpy(ptr, cudfMetadata->data(), cudfSize);
    ptr += cudfSize;
  }

  std::memcpy(ptr, &dataSizeBytes, sizeof(dataSizeBytes));
  ptr += sizeof(dataSizeBytes);

  WireLengthType numRemaining = remainingBytes.size();
  std::memcpy(ptr, &numRemaining, sizeof(numRemaining));
  ptr += sizeof(numRemaining);

  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(remainingBytes[0]);
    std::memcpy(ptr, remainingBytes.data(), bytesSize);
    ptr += bytesSize;
  }

  uint8_t atEndByte = atEnd ? 1 : 0;
  *ptr = atEndByte;

  return std::make_pair<std::shared_ptr<uint8_t>, size_t>(
      std::move(buffer), totalSize);
}

MetadataMsg MetadataMsg::deserializeMetadataMsg(const uint8_t* buffer) {
  const uint8_t* ptr = buffer;

  MetadataMsg record;

  uint32_t magicNumber = 0;
  std::memcpy(&magicNumber, ptr, sizeof(magicNumber));
  VELOX_CHECK_EQ(magicNumber, kMagicNumber);
  ptr += sizeof(magicNumber);

  uint32_t totalSize = 0;
  std::memcpy(&totalSize, ptr, sizeof(totalSize));
  ptr += sizeof(totalSize);

  const uint8_t* endPtr = buffer + totalSize;

  WireLengthType metaSize = 0;
  if (ptr + sizeof(metaSize) > endPtr)
    throw std::runtime_error("Insufficient data for cudfMetadata size");
  std::memcpy(&metaSize, ptr, sizeof(metaSize));
  ptr += sizeof(metaSize);

  record.cudfMetadata = std::make_unique<std::vector<uint8_t>>(metaSize);
  if (metaSize > 0) {
    if (ptr + metaSize > endPtr)
      throw std::runtime_error("Insufficient data for cudfMetadata bytes");
    std::memcpy(record.cudfMetadata->data(), ptr, metaSize);
    ptr += metaSize;
  }

  if (ptr + sizeof(record.dataSizeBytes) > endPtr)
    throw std::runtime_error("Insufficient data for dataSizeBytes");
  std::memcpy(&record.dataSizeBytes, ptr, sizeof(record.dataSizeBytes));
  ptr += sizeof(record.dataSizeBytes);

  WireLengthType numRemaining = 0;
  if (ptr + sizeof(numRemaining) > endPtr)
    throw std::runtime_error("Insufficient data for remainingBytes count");
  std::memcpy(&numRemaining, ptr, sizeof(numRemaining));
  ptr += sizeof(numRemaining);

  record.remainingBytes.resize(numRemaining);
  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(record.remainingBytes[0]);
    if (ptr + bytesSize > endPtr)
      throw std::runtime_error("Insufficient data for remainingBytes values");
    std::memcpy(record.remainingBytes.data(), ptr, bytesSize);
    ptr += bytesSize;
  }

  if (ptr + 1 > endPtr) {
    throw std::runtime_error("Insufficient data for atEnd flag");
  }
  record.atEnd = (*ptr != 0);

  return record;
}

} // namespace facebook::velox::ucx_exchange
