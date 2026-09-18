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
#include "velox/common/base/Exceptions.h"
#include "velox/common/file/File.h"

#include <cudf/io/data_sink.hpp>
#include <cudf/utilities/error.hpp>

#include <kvikio/bounce_buffer.hpp>
#include <kvikio/detail/posix_io.hpp>
#include <kvikio/file_handle.hpp>

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
inline size_t eagerFileSinkStagingBytes() {
  static const size_t bytes = [] {
    constexpr size_t kDefaultBytes = 128UL << 20;
    const auto* raw = std::getenv("GLUTEN_CUDF_EAGER_FILE_SINK_STAGING_BYTES");
    if (raw == nullptr || *raw == '\0') {
      return kDefaultBytes;
    }
    char* end = nullptr;
    const auto requested = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || requested == 0 ||
        requested > std::numeric_limits<size_t>::max()) {
      LOG(WARNING) << "CudfHiveDataSink: ignoring invalid "
                   << "GLUTEN_CUDF_EAGER_FILE_SINK_STAGING_BYTES='" << raw
                   << "'";
      return kDefaultBytes;
    }
    return static_cast<size_t>(requested);
  }();
  return bytes;
}

// cuDF's stock file_sink synchronizes the writer stream before submitting each
// device write. KvikIO's compatibility path then schedules every chunk on its
// global thread pool, stages it through another pinned bounce buffer, and
// synchronizes another CUDA stream. Besides serializing encode and transfer,
// that can create hundreds of contending tasks for one Parquet write.
//
// Stage device data on the writer stream instead. One worker per output file
// waits for the stream event and writes the already-pinned buffers directly.
// This preserves cuDF's future as the device-buffer lifetime fence, overlaps
// encoding with D2H and I/O, and preserves the monotonically increasing writes
// required by Mountpoint-S3. A bounded staging window provides backpressure.
class CudfEagerFileSink final : public cudf::io::data_sink {
 public:
  CudfEagerFileSink(std::string path, bool orderedWrites)
      : path_(std::move(path)),
        orderedWrites_(orderedWrites),
        stagingCapacityBytes_(eagerFileSinkStagingBytes()),
        localFile_(std::make_unique<kvikio::FileHandle>(path_, "w")) {
    VELOX_CHECK(!localFile_->closed(), "KvikIO failed to open {}", path_);
    startWorkers();
  }

  CudfEagerFileSink(std::string path, std::unique_ptr<WriteFile> writeFile)
      : path_(std::move(path)),
        orderedWrites_(true),
        stagingCapacityBytes_(eagerFileSinkStagingBytes()),
        remoteFile_(std::move(writeFile)) {
    VELOX_CHECK_NOT_NULL(remoteFile_);
    startWorkers();
  }

  void startWorkers() {
    const auto bounceBufferSize =
        kvikio::CudaPageAlignedPinnedBounceBufferPool::instance().buffer_size();
    VELOX_CHECK_GT(bounceBufferSize, 0, "KvikIO bounce buffer size is zero");
    pinnedBufferCapacity_ =
        std::max<size_t>(1, stagingCapacityBytes_ / bounceBufferSize);
    const auto workerCount = orderedWrites_ ? 1 : kLocalWorkersPerFile;
    workers_.reserve(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
      workers_.emplace_back([this] { runWrites(); });
    }
  }

  ~CudfEagerFileSink() override {
    try {
      closeRemote();
    } catch (const std::exception& error) {
      LOG(ERROR) << "Failed to close eager sink " << path_ << ": "
                 << error.what();
    }
    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      stopping_ = true;
    }
    queueCv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    LOG(WARNING) << "CudfHiveDataSink: eager sink stats path=" << path_
                 << " deviceWrites=" << deviceWriteCount_
                 << " stagingWaits=" << stagingWaitCount_
                 << " stagingWaitNanos=" << stagingWaitNanos_
                 << " stagingCapacityBytes=" << stagingCapacityBytes_
                 << " peakStagedBytes=" << peakStagedBytes_
                 << " pinnedBufferCapacity=" << pinnedBufferCapacity_
                 << " peakPinnedBuffers=" << peakPinnedBuffers_
                 << " cudaEvents=" << createdEventCount_;
    destroyReadyEvents();
  }

  void closeRemote() {
    if (remoteFile_ != nullptr && remoteClosed_) {
      return;
    }
    flush();
    if (remoteFile_ != nullptr) {
      remoteFile_->close();
      remoteClosed_ = true;
    }
  }

  void host_write(const void* data, size_t size) override {
    const auto offset = reserve(size);
    auto request = std::make_unique<WriteRequest>();
    request->offset = offset;
    request->size = size;
    request->hostData.resize(size);
    std::memcpy(request->hostData.data(), data, size);
    submit(std::move(request)).get();
  }

  void flush() override {
    if (workers_.empty()) {
      return;
    }
    if (orderedWrites_) {
      auto request = std::make_unique<WriteRequest>();
      request->barrier = true;
      submit(std::move(request)).get();
      return;
    }
    std::unique_lock<std::mutex> lock(queueMutex_);
    flushCv_.wait(
        lock, [this] { return queue_.empty() && activeWrites_ == 0; });
  }

  size_t bytes_written() override {
    return bytesWritten_;
  }

  [[nodiscard]] bool supports_device_write() const override {
    return true;
  }

  [[nodiscard]] bool is_device_write_preferred(
      size_t /* size */) const override {
    return true;
  }

  std::future<void> device_write_async(
      const void* gpuData,
      size_t size,
      rmm::cuda_stream_view stream) override {
    ++deviceWriteCount_;
    const auto offset = reserve(size);
    auto completionGroup = std::make_shared<CompletionGroup>();
    auto result = completionGroup->getFuture();
    const auto* source = static_cast<const char*>(gpuData);
    size_t copied = 0;
    std::unique_ptr<WriteRequest> activeRequest;
    bool activeRequestAdded = false;
    try {
      const auto bounceBufferSize =
          kvikio::CudaPageAlignedPinnedBounceBufferPool::instance()
              .buffer_size();
      VELOX_CHECK_GT(bounceBufferSize, 0, "KvikIO bounce buffer size is zero");

      while (copied < size) {
        // reserveStaging() intentionally admits one oversized request when the
        // sink is empty. Split here so that no request can take that bypass or
        // need more pinned buffers than the sink's bounded active set.
        const auto firstDataOffset =
            (offset + copied) % kvikio::get_page_size();
        const auto pinnedRequestCapacity =
            pinnedBufferCapacity_ * bounceBufferSize - firstDataOffset;
        const auto requestSize = std::min(
            {stagingCapacityBytes_, size - copied, pinnedRequestCapacity});
        VELOX_CHECK_GT(requestSize, 0);
        reserveStaging(requestSize);

        activeRequest = std::make_unique<WriteRequest>();
        activeRequest->offset = offset + copied;
        activeRequest->size = requestSize;
        activeRequest->stagedBytes = requestSize;
        activeRequest->completionGroup = completionGroup;
        activeRequest->deviceBuffers.reserve(
            (requestSize + bounceBufferSize - 1) / bounceBufferSize + 1);

        size_t requestCopied = 0;
        while (requestCopied < requestSize) {
          auto buffer = acquirePinnedBuffer();
          // Keep the staged pointer congruent with its file offset. After the
          // small unaligned prefix, both pointer and offset then reach the next
          // page boundary together and the large middle can use O_DIRECT
          // without KvikIO allocating and memcpying through another bounce.
          const auto fileOffset = offset + copied + requestCopied;
          const auto dataOffset = fileOffset % kvikio::get_page_size();
          const auto bytes =
              std::min(buffer.size() - dataOffset, requestSize - requestCopied);
          CUDF_CUDA_TRY(cudaMemcpyAsync(
              buffer.get(dataOffset),
              source + copied + requestCopied,
              bytes,
              cudaMemcpyDeviceToHost,
              stream.value()));
          activeRequest->deviceBuffers.push_back(
              DeviceBuffer{std::move(buffer), dataOffset, bytes});
          requestCopied += bytes;
        }
        activeRequest->ready = acquireReadyEvent();
        CUDF_CUDA_TRY(cudaEventRecord(activeRequest->ready, stream.value()));
        completionGroup->addRequest();
        activeRequestAdded = true;
        submit(std::move(activeRequest));
        activeRequestAdded = false;
        copied += requestSize;
      }
      completionGroup->seal();
    } catch (...) {
      // Complete every already-enqueued D2H before the caller may release the
      // source buffer. Submitted workers retain their own staged buffers.
      const auto error = std::current_exception();
      stream.synchronize();
      if (activeRequest != nullptr) {
        if (activeRequest->ready != nullptr) {
          releaseReadyEvent(activeRequest->ready);
          activeRequest->ready = nullptr;
        }
        recyclePinnedBuffers(activeRequest->deviceBuffers);
        if (activeRequest->stagedBytes != 0) {
          releaseStaging(activeRequest->stagedBytes);
          activeRequest->stagedBytes = 0;
        }
        if (activeRequestAdded) {
          completionGroup->finishRequest(error);
          activeRequestAdded = false;
        }
      }
      completionGroup->seal(error);
      throw;
    }
    return result;
  }

  void device_write(
      const void* gpuData,
      size_t size,
      rmm::cuda_stream_view stream) override {
    device_write_async(gpuData, size, stream).get();
  }

 private:
  using PinnedBuffer = kvikio::CudaPageAlignedPinnedBounceBufferPool::Buffer;

  struct DeviceBuffer {
    PinnedBuffer buffer;
    size_t dataOffset;
    size_t bytes;
  };

  // One cuDF device write can be hundreds of MiB.  Keep its externally
  // visible future while splitting the staging work into bounded requests.
  // This prevents a single write from bypassing stagingCapacityBytes_ and
  // registering dozens of pinned bounce buffers before its first request is
  // made visible to the I/O workers.
  class CompletionGroup {
   public:
    std::future<void> getFuture() {
      return completion_.get_future();
    }

    void addRequest() {
      std::lock_guard<std::mutex> lock(mutex_);
      VELOX_CHECK(
          !sealed_, "Cannot add a request to a sealed completion group");
      ++pending_;
    }

    void finishRequest(std::exception_ptr error = nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      VELOX_CHECK_GT(pending_, 0);
      --pending_;
      if (error != nullptr && error_ == nullptr) {
        error_ = error;
      }
      completeIfReady();
    }

    void seal(std::exception_ptr error = nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      sealed_ = true;
      if (error != nullptr && error_ == nullptr) {
        error_ = error;
      }
      completeIfReady();
    }

   private:
    void completeIfReady() {
      if (!sealed_ || pending_ != 0 || completed_) {
        return;
      }
      completed_ = true;
      if (error_ != nullptr) {
        completion_.set_exception(error_);
      } else {
        completion_.set_value();
      }
    }

    std::mutex mutex_;
    size_t pending_{0};
    bool sealed_{false};
    bool completed_{false};
    std::exception_ptr error_;
    std::promise<void> completion_;
  };

  struct WriteRequest {
    size_t size{0};
    size_t offset{0};
    size_t stagedBytes{0};
    bool barrier{false};
    cudaEvent_t ready{nullptr};
    std::vector<DeviceBuffer> deviceBuffers;
    std::vector<char> hostData;
    std::shared_ptr<CompletionGroup> completionGroup;
    std::promise<void> completion;
  };

  static void pwriteAll(int fd, const void* data, size_t size, size_t offset) {
    auto* current = static_cast<const char*>(data);
    while (size > 0) {
      const auto written = ::pwrite(fd, current, size, offset);
      if (written < 0 && errno == EINTR) {
        continue;
      }
      VELOX_CHECK_GT(
          written,
          0,
          "POSIX pwrite failed at offset {}: {}",
          offset,
          std::strerror(errno));
      current += written;
      offset += written;
      size -= written;
    }
  }

  size_t reserve(size_t size) {
    const auto offset = bytesWritten_;
    bytesWritten_ += size;
    return offset;
  }

  void reserveStaging(size_t size) {
    std::unique_lock<std::mutex> lock(stagingMutex_);
    const bool blocked =
        stagedBytes_ != 0 && stagedBytes_ + size > stagingCapacityBytes_;
    const auto waitStart = std::chrono::steady_clock::now();
    stagingCv_.wait(lock, [this, size] {
      return stagedBytes_ == 0 || stagedBytes_ + size <= stagingCapacityBytes_;
    });
    if (blocked) {
      ++stagingWaitCount_;
      stagingWaitNanos_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - waitStart)
                               .count();
    }
    stagedBytes_ += size;
    peakStagedBytes_ = std::max(peakStagedBytes_, stagedBytes_);
  }

  void releaseStaging(size_t size) {
    {
      std::lock_guard<std::mutex> lock(stagingMutex_);
      VELOX_CHECK_GE(stagedBytes_, size);
      stagedBytes_ -= size;
    }
    stagingCv_.notify_all();
  }

  PinnedBuffer acquirePinnedBuffer() {
    std::unique_lock<std::mutex> lock(pinnedBuffersMutex_);
    pinnedBuffersCv_.wait(
        lock, [this] { return pinnedBuffersInUse_ < pinnedBufferCapacity_; });
    ++pinnedBuffersInUse_;
    peakPinnedBuffers_ = std::max(peakPinnedBuffers_, pinnedBuffersInUse_);
    if (!freePinnedBuffers_.empty()) {
      auto buffer = std::move(freePinnedBuffers_.back());
      freePinnedBuffers_.pop_back();
      return buffer;
    }
    lock.unlock();
    try {
      return kvikio::CudaPageAlignedPinnedBounceBufferPool::instance().get();
    } catch (...) {
      lock.lock();
      VELOX_CHECK_GT(pinnedBuffersInUse_, 0);
      --pinnedBuffersInUse_;
      lock.unlock();
      pinnedBuffersCv_.notify_all();
      throw;
    }
  }

  void recyclePinnedBuffers(std::vector<DeviceBuffer>& segments) {
    {
      std::lock_guard<std::mutex> lock(pinnedBuffersMutex_);
      VELOX_CHECK_GE(pinnedBuffersInUse_, segments.size());
      freePinnedBuffers_.reserve(freePinnedBuffers_.size() + segments.size());
      for (auto& segment : segments) {
        freePinnedBuffers_.push_back(std::move(segment.buffer));
      }
      pinnedBuffersInUse_ -= segments.size();
      segments.clear();
    }
    pinnedBuffersCv_.notify_all();
  }

  cudaEvent_t acquireReadyEvent() {
    std::lock_guard<std::mutex> lock(eventMutex_);
    if (!readyEvents_.empty()) {
      const auto event = readyEvents_.back();
      readyEvents_.pop_back();
      return event;
    }
    cudaEvent_t event{nullptr};
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    ++createdEventCount_;
    return event;
  }

  void releaseReadyEvent(cudaEvent_t event) {
    VELOX_CHECK_NOT_NULL(event);
    std::lock_guard<std::mutex> lock(eventMutex_);
    readyEvents_.push_back(event);
  }

  void discardReadyEvent(cudaEvent_t event) {
    if (event == nullptr) {
      return;
    }
    cudaEventDestroy(event);
    std::lock_guard<std::mutex> lock(eventMutex_);
    VELOX_CHECK_GT(createdEventCount_, 0);
    --createdEventCount_;
  }

  void destroyReadyEvents() {
    std::vector<cudaEvent_t> events;
    {
      std::lock_guard<std::mutex> lock(eventMutex_);
      VELOX_CHECK_EQ(readyEvents_.size(), createdEventCount_);
      events.swap(readyEvents_);
      createdEventCount_ = 0;
    }
    for (const auto event : events) {
      const auto status = cudaEventDestroy(event);
      if (status != cudaSuccess) {
        LOG(ERROR) << "Failed to destroy eager sink CUDA event for " << path_
                   << ": " << cudaGetErrorString(status);
      }
    }
  }

  std::future<void> submit(std::unique_ptr<WriteRequest> request) {
    auto result = request->completion.get_future();
    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      VELOX_CHECK(!stopping_, "Write submitted after closing {}", path_);
      queue_.push_back(std::move(request));
    }
    queueCv_.notify_one();
    return result;
  }

  void runWrites() {
    while (true) {
      std::unique_ptr<WriteRequest> request;
      {
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueCv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (stopping_ && queue_.empty()) {
          return;
        }
        request = std::move(queue_.front());
        queue_.pop_front();
        ++activeWrites_;
      }

      try {
        if (!request->barrier && request->ready != nullptr) {
          CUDF_CUDA_TRY(cudaEventSynchronize(request->ready));
          releaseReadyEvent(request->ready);
          request->ready = nullptr;

          size_t remaining = request->size;
          size_t offset = request->offset;
          for (const auto& segment : request->deviceBuffers) {
            if (remoteFile_ != nullptr) {
              VELOX_CHECK_EQ(
                  offset,
                  remoteBytesAppended_,
                  "Out-of-order remote write for {}",
                  path_);
              remoteFile_->append(
                  std::string_view(
                      static_cast<const char*>(
                          segment.buffer.get(segment.dataOffset)),
                      segment.bytes));
              remoteBytesAppended_ += segment.bytes;
            } else {
              const auto written = kvikio::detail::posix_host_write<
                  kvikio::detail::PartialIO::NO>(
                  localFile_->fd(false),
                  segment.buffer.get(segment.dataOffset),
                  segment.bytes,
                  offset,
                  orderedWrites_ ? -1 : localFile_->fd(true));
              VELOX_CHECK_EQ(
                  written,
                  segment.bytes,
                  "Incomplete direct write for {}",
                  path_);
            }
            remaining -= segment.bytes;
            offset += segment.bytes;
          }
          VELOX_CHECK_EQ(remaining, 0, "Incomplete staged write for {}", path_);
        } else if (!request->barrier) {
          if (remoteFile_ != nullptr) {
            VELOX_CHECK_EQ(
                request->offset,
                remoteBytesAppended_,
                "Out-of-order remote host write for {}",
                path_);
            remoteFile_->append(
                std::string_view(
                    request->hostData.data(), request->hostData.size()));
            remoteBytesAppended_ += request->hostData.size();
          } else {
            pwriteAll(
                localFile_->fd(false),
                request->hostData.data(),
                request->size,
                request->offset);
          }
        } else if (remoteFile_ != nullptr) {
          remoteFile_->flush();
        }
        recyclePinnedBuffers(request->deviceBuffers);
        if (request->stagedBytes != 0) {
          releaseStaging(request->stagedBytes);
          request->stagedBytes = 0;
        }
        if (request->completionGroup != nullptr) {
          request->completionGroup->finishRequest();
          request->completionGroup.reset();
        }
        request->completion.set_value();
      } catch (...) {
        const auto error = std::current_exception();
        if (request->ready != nullptr) {
          discardReadyEvent(request->ready);
          request->ready = nullptr;
        }
        recyclePinnedBuffers(request->deviceBuffers);
        if (request->stagedBytes != 0) {
          releaseStaging(request->stagedBytes);
          request->stagedBytes = 0;
        }
        if (request->completionGroup != nullptr) {
          request->completionGroup->finishRequest(error);
          request->completionGroup.reset();
        }
        request->completion.set_exception(error);
      }
      {
        std::lock_guard<std::mutex> lock(queueMutex_);
        VELOX_CHECK_GT(activeWrites_, 0);
        --activeWrites_;
      }
      flushCv_.notify_all();
    }
  }

  static constexpr size_t kLocalWorkersPerFile = 4;
  const std::string path_;
  const bool orderedWrites_;
  const size_t stagingCapacityBytes_;
  size_t bytesWritten_{0};
  size_t remoteBytesAppended_{0};
  std::unique_ptr<kvikio::FileHandle> localFile_;
  std::unique_ptr<WriteFile> remoteFile_;
  bool remoteClosed_{false};
  std::mutex queueMutex_;
  std::condition_variable queueCv_;
  std::condition_variable flushCv_;
  std::deque<std::unique_ptr<WriteRequest>> queue_;
  size_t activeWrites_{0};
  bool stopping_{false};
  std::vector<std::thread> workers_;
  std::mutex stagingMutex_;
  std::condition_variable stagingCv_;
  size_t stagedBytes_{0};
  size_t peakStagedBytes_{0};
  size_t deviceWriteCount_{0};
  size_t stagingWaitCount_{0};
  int64_t stagingWaitNanos_{0};
  std::mutex pinnedBuffersMutex_;
  std::condition_variable pinnedBuffersCv_;
  std::vector<PinnedBuffer> freePinnedBuffers_;
  size_t pinnedBufferCapacity_{1};
  size_t pinnedBuffersInUse_{0};
  size_t peakPinnedBuffers_{0};
  std::mutex eventMutex_;
  std::vector<cudaEvent_t> readyEvents_;
  size_t createdEventCount_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
