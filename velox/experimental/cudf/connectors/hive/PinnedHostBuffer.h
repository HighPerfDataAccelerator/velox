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

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

namespace facebook::velox::cudf_velox::connector::hive {

/// Host buffer used by the S3 prefetch path. It first uses CUDA portable
/// pinned memory and falls back to pageable memory without changing
/// correctness.
class PinnedHostBuffer {
 public:
  explicit PinnedHostBuffer(
      size_t size,
      std::shared_ptr<void> lifetime = nullptr)
      : lifetime_(std::move(lifetime)), size_(size) {
    if (size_ == 0) {
      return;
    }
    void* allocation = nullptr;
    const auto result =
        cudaHostAlloc(&allocation, size_, cudaHostAllocPortable);
    if (result == cudaSuccess) {
      data_ = static_cast<uint8_t*>(allocation);
      pinned_ = true;
    } else {
      // Clear the CUDA runtime's per-thread error before falling back. This
      // path must remain usable when the pinned-memory budget is exhausted.
      cudaGetLastError();
      data_ = static_cast<uint8_t*>(std::malloc(size_));
      if (data_ == nullptr) {
        throw std::bad_alloc();
      }
    }
  }

  ~PinnedHostBuffer() {
    if (data_ == nullptr) {
      return;
    }
    if (pinned_) {
      // Direct CUDA allocation keeps the allocation/deallocation pair stable
      // even if cuDF swaps its process-wide pinned memory resource while a
      // reader is alive.
      cudaFreeHost(data_);
    } else {
      std::free(data_);
    }
  }

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  uint8_t* data() {
    return data_;
  }

  const uint8_t* data() const {
    return data_;
  }

  size_t size() const {
    return size_;
  }

 private:
  // Keep any executor-wide byte reservation alive until after the allocation
  // is released by the destructor body.
  std::shared_ptr<void> lifetime_;
  uint8_t* data_{nullptr};
  size_t size_{0};
  bool pinned_{false};
};

} // namespace facebook::velox::cudf_velox::connector::hive
