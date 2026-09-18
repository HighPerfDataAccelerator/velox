/* Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

// Dedicated scan staging, not the long-lived cache or exchange pool. Both idle
// and in-flight storage count toward the cap. A released CPU lease may still
// have DMA in flight: its event MUST complete before filling that slot again.
class BufferedInputPinnedPool {
 public:
  explicit BufferedInputPinnedPool(
      size_t slots = 4,
      size_t chunkBytes = 32UL << 20)
      : slots_(slots), chunkBytes_(chunkBytes) {
    if (slots == 0 || chunkBytes == 0) {
      throw std::invalid_argument("Empty BufferedInput pinned pool");
    }
  }

  BufferedInputPinnedPool(const BufferedInputPinnedPool&) = delete;
  BufferedInputPinnedPool& operator=(const BufferedInputPinnedPool&) = delete;

  ~BufferedInputPinnedPool() {
    // Shutdown is after IO workers have joined. No CUDA API runs in a CUDA
    // host callback, including cudaFreeHost or event destruction.
    for (auto& slot : slots_) {
      if (slot.quarantined) {
        continue;
      }
      if (slot.pending && cudaEventSynchronize(slot.event) != cudaSuccess) {
        // Never free a potentially live DMA source on a CUDA failure.
        continue;
      }
      if (slot.event) {
        cudaEventDestroy(slot.event);
      }
      if (slot.data) {
        cudaFreeHost(slot.data);
      }
    }
  }

  static BufferedInputPinnedPool& instance() {
    static BufferedInputPinnedPool pool;
    return pool;
  }

  // fill(offset, size, pinnedDestination) must finish writing before returning.
  // Returns after enqueueing, not after GPU completion, like device_read_async.
  template <typename Fill>
  void copyToDevice(
      uint8_t* destination,
      size_t bytes,
      cudaStream_t stream,
      Fill&& fill) {
    for (size_t offset = 0; offset < bytes;) {
      const auto index = acquire();
      auto& slot = slots_[index];
      bool copySubmitted = false;
      try {
        if (slot.pending) {
          check(cudaEventSynchronize(slot.event));
          slot.pending = false;
        }
        int device;
        check(cudaGetDevice(&device));
        if (slot.event && slot.device != device) {
          check(cudaEventDestroy(slot.event));
          slot.event = nullptr;
        }
        if (!slot.event) {
          check(cudaEventCreateWithFlags(&slot.event, cudaEventDisableTiming));
          slot.device = device;
        }
        if (!slot.data) {
          // Allocation failure is an error, never a hidden pageable fallback.
          check(cudaHostAlloc(
              reinterpret_cast<void**>(&slot.data),
              chunkBytes_,
              cudaHostAllocPortable));
        }
        const auto size = std::min(chunkBytes_, bytes - offset);
        fill(offset, size, slot.data);
        check(cudaMemcpyAsync(
            destination + offset,
            slot.data,
            size,
            cudaMemcpyHostToDevice,
            stream));
        copySubmitted = true;
        check(cudaEventRecord(slot.event, stream));
        slot.pending = true;
        offset += size;
      } catch (...) {
        // If event recording failed, quiesce the submitted copy before reuse.
        // A failed synchronization quarantines the whole pool, not the buffer.
        const bool failed =
            copySubmitted && cudaStreamSynchronize(stream) != cudaSuccess;
        slot.quarantined = failed || slot.pending;
        release(index, failed || slot.pending);
        throw;
      }
      release(index, false);
    }
  }

 private:
  struct Slot {
    uint8_t* data{nullptr};
    cudaEvent_t event{nullptr};
    int device{-1};
    bool pending{false};
    bool leased{false};
    bool quarantined{false};
  };

  static void check(cudaError_t status) {
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("BufferedInput pinned transfer: ") +
          cudaGetErrorString(status));
    }
  }

  size_t acquire() {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [&] {
      return failed_ || std::any_of(slots_.begin(), slots_.end(), [](auto& s) {
               return !s.leased;
             });
    });
    if (failed_) {
      throw std::runtime_error("BufferedInput pinned pool CUDA failure");
    }
    for (size_t n = 0; n < slots_.size(); ++n) {
      const auto index = (next_ + n) % slots_.size();
      if (!slots_[index].leased) {
        slots_[index].leased = true;
        next_ = (index + 1) % slots_.size();
        return index;
      }
    }
    throw std::logic_error("No free BufferedInput pinned slot");
  }

  void release(size_t index, bool failed) {
    {
      std::lock_guard lock(mutex_);
      failed_ |= failed;
      slots_[index].leased = false;
    }
    available_.notify_all();
  }

  std::vector<Slot> slots_;
  const size_t chunkBytes_;
  std::mutex mutex_;
  std::condition_variable available_;
  size_t next_{0};
  bool failed_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
