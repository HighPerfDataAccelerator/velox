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
#include "velox/experimental/cudf/exchange/LocalGpuExchangeSource.h"

#include <atomic>
#include <thread>

#include "velox/exec/Operator.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/SerializedPage.h"

namespace facebook::velox::cudf_velox {
namespace {

/// ExchangeSource implementation for same-process GPU exchange.
/// Mirrors LocalExchangeSource (velox/exec/tests/utils/LocalExchangeSource.cpp)
/// but registers for the "gpu-local://" task ID prefix.
///
/// The data flow is identical to LocalExchangeSource:
///   - Calls OutputBufferManager::getData() to fetch pages
///   - Enqueues received pages into the ExchangeQueue
///   - A background timer thread fires the result callback on timeout so that
///     the Exchange operator retries even if getData() did not call back
///     immediately (essential for correct multi-fragment execution).
///   - The pages are GpuSerializedPages, but this source is agnostic to
///     the concrete page type -- it handles SerializedPageBase*
class LocalGpuExchangeSource : public exec::ExchangeSource {
 public:
  LocalGpuExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<exec::ExchangeQueue> queue,
      memory::MemoryPool* pool)
      : ExchangeSource(taskId, destination, std::move(queue), pool) {}

  bool supportsMetrics() const override {
    return true;
  }

  folly::F14FastMap<std::string, RuntimeMetric> metrics() const override {
    return {
        {"localGpuExchangeSource.numPages", RuntimeMetric(numPages_)},
        {"localGpuExchangeSource.totalBytes",
         RuntimeMetric(totalBytes_, RuntimeCounter::Unit::kBytes)},
    };
  }

  bool shouldRequestLocked() override {
    if (atEnd_) {
      return false;
    }
    return !requestPending_.exchange(true);
  }

  folly::SemiFuture<Response> request(
      uint32_t maxBytes,
      std::chrono::microseconds maxWait) override {
    LOG(WARNING) << "LocalGpuExchangeSource::request taskId=" << remoteTaskId_
                 << " destination=" << destination_
                 << " sequence=" << sequence_
                 << " maxBytes=" << maxBytes
                 << " maxWaitUs=" << maxWait.count();
    auto promise = VeloxPromise<Response>("LocalGpuExchangeSource::request");
    auto future = promise.getSemiFuture();
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      promise_ = std::move(promise);
    }

    auto buffers = exec::OutputBufferManager::getInstanceRef();
    VELOX_CHECK_NOT_NULL(buffers, "invalid OutputBufferManager");
    VELOX_CHECK(requestPending_);

    auto requestedSequence = sequence_;
    auto self = shared_from_this();

    // Callback invoked by OutputBufferManager when data is available,
    // or by the timer thread on timeout (with empty data).
    // Since this lambda may outlive 'this', we capture a shared_ptr (self).
    auto resultCallback = [self, requestedSequence, buffers, this](
                              std::vector<std::unique_ptr<folly::IOBuf>> data,
                              int64_t sequence,
                              std::vector<int64_t> remainingBytes) {
      LOG(WARNING) << "LocalGpuExchangeSource::resultCallback taskId="
                   << remoteTaskId_ << " destination=" << destination_
                   << " dataSize=" << data.size()
                   << " callbackSeq=" << sequence
                   << " requestedSeq=" << requestedSequence;
      {
        std::lock_guard<std::mutex> l(mutex_);
        auto iter = timeouts_.find(self);
        if (iter == timeouts_.end()) {
          LOG(WARNING) << "LocalGpuExchangeSource::resultCallback "
                       << "DOUBLE-FIRE ignored taskId=" << remoteTaskId_;
          return;
        }
        timeouts_.erase(iter);
      }

      if (requestedSequence > sequence && !data.empty()) {
        int64_t nExtra = requestedSequence - sequence;
        VELOX_CHECK(nExtra < data.size());
        data.erase(data.cbegin(), data.cbegin() + nExtra);
        sequence = requestedSequence;
      }
      if (data.empty()) {
        sequence = requestedSequence;
      }

      // Convert IOBuf data to PrestoSerializedPages and enqueue them into
      // the ExchangeQueue, exactly like the CPU LocalExchangeSource.
      // In the GPU exchange path, the producer enqueues GpuSerializedPages
      // into OutputBufferManager. However, the getData() callback receives
      // IOBuf clones. For a same-process GPU exchange, the pages flowing
      // through here will be PrestoSerializedPage wrappers. A future
      // optimization could bypass IOBuf entirely for GPU pages.
      std::vector<std::unique_ptr<exec::SerializedPageBase>> pages;
      bool atEnd = false;
      int64_t totalBytes = 0;
      for (auto& inputPage : data) {
        if (!inputPage) {
          atEnd = true;
          // Keep looping, there could be extra end markers.
          continue;
        }
        totalBytes += inputPage->length();
        inputPage->unshare();
        pages.push_back(
            std::make_unique<exec::PrestoSerializedPage>(std::move(inputPage)));
        inputPage = nullptr;
      }

      numPages_ += pages.size();
      totalBytes_ += totalBytes;

      VeloxPromise<Response> requestPromise;
      {
        std::vector<ContinuePromise> queuePromises;
        {
          std::lock_guard<std::mutex> l(queue_->mutex());
          requestPending_ = false;
          requestPromise = std::move(promise_);
          for (auto& page : pages) {
            queue_->enqueueLocked(std::move(page), queuePromises);
          }
          if (atEnd) {
            queue_->enqueueLocked(nullptr, queuePromises);
            atEnd_ = true;
          }
          if (!data.empty()) {
            sequence_ = sequence + pages.size();
          }
        }
        for (auto& promise : queuePromises) {
          promise.setValue();
        }
      }

      // Outside of queue mutex.
      if (atEnd_) {
        buffers->deleteResults(remoteTaskId_, destination_);
      }

      if (!requestPromise.isFulfilled()) {
        requestPromise.setValue(Response{totalBytes, atEnd_, remainingBytes});
      }
    };

    registerTimeout(self, resultCallback, maxWait);

    auto ok = buffers->getData(
        remoteTaskId_, destination_, maxBytes, sequence_, resultCallback);
    LOG(WARNING) << "LocalGpuExchangeSource::request getData returned ok="
                 << ok << " for taskId=" << remoteTaskId_
                 << " destination=" << destination_;

    return future;
  }

  folly::SemiFuture<Response> requestDataSizes(
      std::chrono::microseconds maxWait) override {
    return request(0, maxWait);
  }

  void close() override {
    checkSetRequestPromise();

    auto buffers = exec::OutputBufferManager::getInstanceRef();
    buffers->deleteResults(remoteTaskId_, destination_);
  }

  // Invoked to start exchange source to run. It clears 'stop_' which allows
  // the background timeout executor to run on the first exchange source
  // request.
  static void start() {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_NULL(timerThread_);
    VELOX_CHECK(timeouts_.empty());
    stop_ = false;
  }

  // Invoked to stop the exchange source. It sets 'stop_', clears the current
  // pending request 'timeouts_' and joins/destroys the timer thread if
  // created.
  static void stop() {
    {
      std::lock_guard<std::mutex> l(mutex_);
      if (stop_) {
        VELOX_CHECK(timeouts_.empty());
        return;
      }
      stop_ = true;
      if (timerThread_ == nullptr) {
        VELOX_CHECK(timeouts_.empty());
        return;
      }
      timeouts_.clear();
    }
    timerThread_->join();
    timerThread_.reset();
  }

 private:
  using ResultCallback = std::function<void(
      std::vector<std::unique_ptr<folly::IOBuf>> data,
      int64_t sequence,
      std::vector<int64_t> remainingBytes)>;

  static void registerTimeout(
      const std::shared_ptr<ExchangeSource>& self,
      ResultCallback callback,
      std::chrono::microseconds maxWait) {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(!stop_, "Local GPU exchange source has stopped");

    if (timerThread_ == nullptr) {
      timerThread_ = std::make_unique<std::thread>([&]() {
        while (!stop_) {
          auto now = std::chrono::system_clock::now();
          ResultCallback callback = nullptr;
          {
            std::lock_guard<std::mutex> t(mutex_);
            for (auto& pair : timeouts_) {
              if (pair.second.second < now) {
                callback = pair.second.first;
                break;
              }
            }
          }
          if (callback) {
            // Outside of mutex.
            callback({}, 0, {});
            continue;
          }
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      });
      if (!exitInitialized_) {
        exitInitialized_ = true;
        atexit([]() { stop(); });
      }
    }
    timeouts_[self] =
        std::make_pair(callback, std::chrono::system_clock::now() + maxWait);
  }

  bool checkSetRequestPromise() {
    VeloxPromise<Response> promise{VeloxPromise<Response>::makeEmpty()};
    {
      std::lock_guard<std::mutex> l(queue_->mutex());
      promise = std::move(promise_);
    }
    if (promise.valid() && !promise.isFulfilled()) {
      promise.setValue(Response{0, false, {}});
      return true;
    }
    return false;
  }

  static inline std::mutex mutex_;
  static inline folly::F14FastMap<
      std::shared_ptr<ExchangeSource>,
      std::pair<ResultCallback, std::chrono::system_clock::time_point>>
      timeouts_;
  static inline std::unique_ptr<std::thread> timerThread_;
  static inline std::atomic_bool stop_{false};
  static inline bool exitInitialized_{false};

  VeloxPromise<Response> promise_{VeloxPromise<Response>::makeEmpty()};
  std::atomic<int64_t> numPages_{0};
  std::atomic<uint64_t> totalBytes_{0};
};

} // namespace

std::shared_ptr<exec::ExchangeSource> createLocalGpuExchangeSource(
    const std::string& taskId,
    int destination,
    std::shared_ptr<exec::ExchangeQueue> queue,
    memory::MemoryPool* pool) {
  if (strncmp(taskId.c_str(), "gpu-local://", 12) == 0) {
    LOG(WARNING) << "createLocalGpuExchangeSource creating source taskId="
                 << taskId << " destination=" << destination;
    return std::make_shared<LocalGpuExchangeSource>(
        taskId, destination, std::move(queue), pool);
  }
  return nullptr;
}

void testingStartLocalGpuExchangeSource() {
  LocalGpuExchangeSource::start();
}

void testingShutdownLocalGpuExchangeSource() {
  LocalGpuExchangeSource::stop();
}

} // namespace facebook::velox::cudf_velox
