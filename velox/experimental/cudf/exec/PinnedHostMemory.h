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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// RAII host buffer that prefers pinned memory for fast PCIe DMA transfers.
/// Allocates from cudf's pinned pool; falls back to pageable heap on failure.
class PinnedHostBuffer {
 public:
  explicit PinnedHostBuffer(size_t size) : size_(size) {
    if (size == 0) {
      return;
    }
    try {
      mr_ = cudf::get_pinned_memory_resource();
      data_ = static_cast<uint8_t*>(mr_.allocate_sync(size));
      pinned_ = true;
    } catch (...) {
      fallback_.resize(size);
      data_ = fallback_.data();
      pinned_ = false;
    }
  }

  ~PinnedHostBuffer() {
    if (pinned_ && data_) {
      mr_.deallocate_sync(data_, size_);
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
  rmm::host_device_async_resource_ref mr_{cudf::get_pinned_memory_resource()};
  uint8_t* data_{nullptr};
  size_t size_{0};
  bool pinned_{false};
  std::vector<uint8_t> fallback_;
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
