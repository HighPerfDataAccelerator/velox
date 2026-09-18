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

#include "velox/experimental/cudf/connectors/hive/BoundedWriteQueue.h"

#include <cudf/io/config_utils.hpp>
#include <cudf/io/data_sink.hpp>

#include <kvikio/file_handle.hpp>

#include <nvtx3/nvtx3.hpp>

#include <glog/logging.h>

#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox::connector::hive {

// Same local file backend and codecs as libcudf's file_sink, but submission
// is eager and bounded. Only completion waiting is deferred. No additional
// GPU buffers or pinned pools are created here.
class CudfBoundedFileSink final : public cudf::io::data_sink {
 public:
  explicit CudfBoundedFileSink(const std::string& path, size_t depth = 4)
      : requests_(depth, false) {
    cudf::io::kvikio_integration::set_up_kvikio();
    file_ = kvikio::FileHandle(path, "w");
    if (file_.closed()) {
      throw std::runtime_error("Unable to open bounded cuDF output sink");
    }
  }

  void host_write(const void* data, size_t size) override {
    requests_.drain();
    const auto offset = reserveOffset(size);
    if (size && file_.pwrite(data, size, offset, size).get() != size) {
      throw std::runtime_error("Short host file write");
    }
  }

  void flush() override {
    requests_.drain();
    const auto& stats = requests_.stats();
    LOG(WARNING) << "CUDF_BOUNDED_FILE_QUEUE retireReady=0"
                 << " capacityWaits=" << stats.capacityWaits
                 << " capacityWaitNs=" << stats.capacityWaitNs
                 << " readyBehindBlockedFront=" << stats.readyBehindBlockedFront
                 << " readyBehindRequests=" << stats.readyBehindRequests
                 << " outOfOrderRetired=" << stats.outOfOrderRetired
                 << " peakPending=" << stats.peakPending;
  }

  size_t bytes_written() override {
    return bytesWritten_;
  }

  bool supports_device_write() const override {
    return true;
  }

  bool is_device_write_preferred(size_t /*size*/) const override {
    return true;
  }

  std::future<void> device_write_async(
      const void* data,
      size_t size,
      rmm::cuda_stream_view stream) override {
    const auto offset = reserveOffset(size);
    if (!size) {
      return std::async(std::launch::deferred, [] {});
    }
    // Preserve libcudf's producer-readiness contract. The caller retains the
    // compression buffer until completion (also enforced on future discard).
    stream.synchronize();
    nvtx3::scoped_range admission("CudfWriter::fileSinkAdmission");
    return requests_.submit(
        [this, data, size, offset] {
          // One KvikIO task per admitted request: its completion is also a
          // lifetime fence on failure, rather than a first-failure aggregate
          // that could leave sibling tasks reading this source. Parallelism
          // comes from up to four column chunks, on the existing IO pool.
          return file_.pwrite(data, size, offset, size);
        },
        size);
  }

  void device_write(const void* data, size_t size, rmm::cuda_stream_view stream)
      override {
    device_write_async(data, size, stream).get();
  }

 private:
  size_t reserveOffset(size_t size) {
    if (size > std::numeric_limits<size_t>::max() - bytesWritten_) {
      throw std::overflow_error("Output sink offset overflow");
    }
    const auto offset = bytesWritten_;
    bytesWritten_ += size;
    return offset;
  }

  // Reverse destruction drains requests before closing the file.
  kvikio::FileHandle file_;
  BoundedWriteQueue requests_;
  size_t bytesWritten_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
