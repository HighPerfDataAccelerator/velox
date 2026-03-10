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

#include <cudf/io/datasource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_set>

namespace facebook::velox::cudf_velox {

struct PinnedAllocStats {
  std::atomic<uint64_t> pinnedCount{0};
  std::atomic<uint64_t> pinnedBytes{0};
  std::atomic<uint64_t> fallbackCount{0};
  std::atomic<uint64_t> fallbackBytes{0};
  std::atomic<uint64_t> peakInFlight{0};
  std::atomic<uint64_t> curInFlight{0};

  static PinnedAllocStats& instance() {
    static PinnedAllocStats s;
    return s;
  }

  void recordPinned(size_t bytes) {
    auto pc = pinnedCount.fetch_add(1, std::memory_order_relaxed) + 1;
    pinnedBytes.fetch_add(bytes, std::memory_order_relaxed);
    auto cur = curInFlight.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto peak = peakInFlight.load(std::memory_order_relaxed);
    while (cur > peak &&
           !peakInFlight.compare_exchange_weak(
               peak, cur, std::memory_order_relaxed)) {
    }
    auto total = pc + fallbackCount.load(std::memory_order_relaxed);
    if ((total & (total - 1)) == 0 || (total % 100000) == 0) {
      dump("alloc");
    }
  }

  void recordFallback(size_t bytes) {
    auto fc = fallbackCount.fetch_add(1, std::memory_order_relaxed) + 1;
    fallbackBytes.fetch_add(bytes, std::memory_order_relaxed);
    auto total = pinnedCount.load(std::memory_order_relaxed) + fc;
    if ((total & (total - 1)) == 0 || (total % 100000) == 0) {
      dump("fallback");
    }
  }

  void recordFree(size_t bytes) {
    curInFlight.fetch_sub(bytes, std::memory_order_relaxed);
  }

  void dump(const char* event) {
    fprintf(
        stderr,
        "[PinnedAlloc] %s: pinned=%lu/%luMB fallback=%lu/%luMB "
        "inFlight=%luMB peak=%luMB\n",
        event,
        (unsigned long)pinnedCount.load(std::memory_order_relaxed),
        (unsigned long)(pinnedBytes.load(std::memory_order_relaxed) >> 20),
        (unsigned long)fallbackCount.load(std::memory_order_relaxed),
        (unsigned long)(fallbackBytes.load(std::memory_order_relaxed) >> 20),
        (unsigned long)(curInFlight.load(std::memory_order_relaxed) >> 20),
        (unsigned long)(peakInFlight.load(std::memory_order_relaxed) >> 20));
  }
};

/// Unified "preferred-pinned" allocator.  Every host allocation that wants
/// DMA-friendly pinned memory goes through this single layer:
///
///   1. Try cudf's pinned pool  (fast, pool-backed cudaHostAlloc)
///   2. On ANY failure → clear sticky CUDA error, fall back to pageable malloc
///
/// By going straight to malloc on failure we avoid the thundering-herd of
/// direct cudaHostAlloc calls that cudf's own fallback would issue, which
/// under heavy concurrency can exhaust OS pinned-memory limits and poison
/// the CUDA context with cudaErrorIllegalAddress.
///
/// Thread-safe.  The pageable-pointer set is protected by a mutex so that
/// allocate / deallocate can safely happen on different threads.
class PreferredPinnedPool {
 public:
  static PreferredPinnedPool& instance() {
    static PreferredPinnedPool pool;
    return pool;
  }

  /// Allocate `bytes` of host memory, preferring pinned.
  /// Returns {pointer, true} for pinned, {pointer, false} for pageable.
  /// Throws std::bad_alloc only if pageable malloc also fails.
  std::pair<void*, bool> allocate(size_t bytes) {
    if (bytes == 0) {
      return {nullptr, false};
    }

    // Fast path: try pinned pool.
    try {
      auto mr = cudf::get_pinned_memory_resource();
      void* ptr = mr.allocate_sync(bytes);
      PinnedAllocStats::instance().recordPinned(bytes);
      return {ptr, true};
    } catch (...) {
      // Pinned allocation failed (pool exhausted, cudaHostAlloc failed, or
      // CUDA context error).  Clear any sticky CUDA error so that later GPU
      // operations on other streams are not affected.
      cudaGetLastError();
    }

    // Slow path: pageable heap.
    void* ptr = std::malloc(bytes);
    if (!ptr) {
      throw std::bad_alloc();
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      pageablePtrs_.insert(ptr);
    }
    PinnedAllocStats::instance().recordFallback(bytes);
    return {ptr, false};
  }

  /// Deallocate a pointer previously returned by allocate().
  void deallocate(void* ptr, size_t bytes, bool pinned) {
    if (!ptr) {
      return;
    }
    if (pinned) {
      PinnedAllocStats::instance().recordFree(bytes);
      try {
        auto mr = cudf::get_pinned_memory_resource();
        mr.deallocate_sync(ptr, bytes);
      } catch (...) {
        // If deallocation fails (poisoned context), just leak rather than
        // crash.  This is a last-resort safety net.
        cudaGetLastError();
      }
    } else {
      {
        std::lock_guard<std::mutex> lock(mu_);
        pageablePtrs_.erase(ptr);
      }
      std::free(ptr);
    }
  }

  /// Check whether a pointer was allocated as pageable fallback.
  /// Useful when callers only have the pointer and not the pinned flag.
  bool isPageable(void* ptr) const {
    std::lock_guard<std::mutex> lock(mu_);
    return pageablePtrs_.count(ptr) > 0;
  }

 private:
  PreferredPinnedPool() = default;
  mutable std::mutex mu_;
  std::unordered_set<void*> pageablePtrs_;
};

/// RAII host buffer that prefers pinned memory for fast PCIe DMA transfers.
/// Allocates through PreferredPinnedPool: pinned pool → pageable malloc.
class PinnedHostBuffer {
 public:
  explicit PinnedHostBuffer(size_t size) : size_(size) {
    if (size == 0) {
      return;
    }
    auto [ptr, isPinned] = PreferredPinnedPool::instance().allocate(size);
    data_ = static_cast<uint8_t*>(ptr);
    pinned_ = isPinned;
  }

  ~PinnedHostBuffer() {
    if (data_) {
      PreferredPinnedPool::instance().deallocate(data_, size_, pinned_);
    }
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer(PinnedHostBuffer&&) = delete;
  PinnedHostBuffer& operator=(PinnedHostBuffer&&) = delete;

  uint8_t* data() { return data_; }
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  bool isPinned() const { return pinned_; }

 private:
  uint8_t* data_{nullptr};
  size_t size_{0};
  bool pinned_{false};
};

/// cudf::io::datasource::buffer backed by PinnedHostBuffer.
/// Drop-in replacement for datasource::buffer::create(std::vector<uint8_t>).
class PinnedDataSourceBuffer : public cudf::io::datasource::buffer {
 public:
  explicit PinnedDataSourceBuffer(std::shared_ptr<PinnedHostBuffer> buf)
      : buf_(std::move(buf)) {}
  [[nodiscard]] size_t size() const override { return buf_->size(); }
  [[nodiscard]] uint8_t const* data() const override { return buf_->data(); }

 private:
  std::shared_ptr<PinnedHostBuffer> buf_;
};

} // namespace facebook::velox::cudf_velox
