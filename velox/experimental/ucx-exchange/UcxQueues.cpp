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
#include "velox/experimental/ucx-exchange/UcxQueues.h"

#include "velox/experimental/cudf/exec/GpuResources.h"

#include <atomic>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace facebook::velox::ucx_exchange {

namespace {
std::atomic<int64_t> diagnosticGlobalQueuedBytes{0};
std::atomic<int64_t> diagnosticGlobalQueuedColumns{0};
std::atomic<int64_t> diagnosticGlobalQueueGiB{0};

// Lends only bytes above each task's ordinary max output-buffer size. This
// lets hot producers use idle device capacity without multiplying a large
// static allowance by every live fragment.
class AdaptiveQueueBurstCoordinator {
 public:
  bool tryResize(
      const void* owner,
      uint64_t currentBytes,
      uint64_t requestedBytes,
      uint64_t globalLimitBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = grants_.find(owner);
    VELOX_DCHECK_EQ(
        currentBytes, it == grants_.end() ? uint64_t{0} : it->second);
    if (requestedBytes <= currentBytes) {
      grantedBytes_ -= currentBytes - requestedBytes;
      if (requestedBytes == 0) {
        grants_.erase(owner);
      } else {
        grants_[owner] = requestedBytes;
      }
      return true;
    }
    const auto additional = requestedBytes - currentBytes;
    if (additional > globalLimitBytes ||
        grantedBytes_ > globalLimitBytes - additional) {
      return false;
    }
    grantedBytes_ += additional;
    grants_[owner] = requestedBytes;
    return true;
  }

  void addWaiter(const void* owner, ContinuePromise promise) {
    std::lock_guard<std::mutex> lock(mutex_);
    waiters_.push_back(Waiter{owner, std::move(promise)});
  }

  std::vector<ContinuePromise> releaseTo(
      const void* owner,
      uint64_t currentBytes,
      uint64_t requestedBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = grants_.find(owner);
    VELOX_DCHECK_EQ(
        currentBytes, it == grants_.end() ? uint64_t{0} : it->second);
    VELOX_DCHECK_LE(requestedBytes, currentBytes);
    if (requestedBytes == currentBytes) {
      return {};
    }
    grantedBytes_ -= currentBytes - requestedBytes;
    if (requestedBytes == 0) {
      grants_.erase(owner);
    } else {
      grants_[owner] = requestedBytes;
    }
    return takeAllWaitersLocked();
  }

  std::vector<ContinuePromise> removeOwner(
      const void* owner,
      uint64_t currentBytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (const auto it = grants_.find(owner); it != grants_.end()) {
      VELOX_DCHECK_EQ(currentBytes, it->second);
      grantedBytes_ -= it->second;
      grants_.erase(it);
    } else {
      VELOX_DCHECK_EQ(currentBytes, 0);
    }

    std::vector<ContinuePromise> promises;
    for (auto it = waiters_.begin(); it != waiters_.end();) {
      // Releasing credit may make every waiter runnable. With no released
      // credit, wake only this owner so cancellation cannot strand a driver.
      if (currentBytes > 0 || it->owner == owner) {
        promises.push_back(std::move(it->promise));
        it = waiters_.erase(it);
      } else {
        ++it;
      }
    }
    return promises;
  }

  std::vector<ContinuePromise> cancelOwnerWaiters(const void* owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ContinuePromise> promises;
    for (auto it = waiters_.begin(); it != waiters_.end();) {
      if (it->owner == owner) {
        promises.push_back(std::move(it->promise));
        it = waiters_.erase(it);
      } else {
        ++it;
      }
    }
    return promises;
  }

  std::vector<ContinuePromise> notifyProgress() {
    std::lock_guard<std::mutex> lock(mutex_);
    return takeAllWaitersLocked();
  }

 private:
  struct Waiter {
    const void* owner;
    ContinuePromise promise;
  };

  std::vector<ContinuePromise> takeAllWaitersLocked() {
    std::vector<ContinuePromise> promises;
    promises.reserve(waiters_.size());
    for (auto& waiter : waiters_) {
      promises.push_back(std::move(waiter.promise));
    }
    waiters_.clear();
    return promises;
  }

  std::mutex mutex_;
  uint64_t grantedBytes_{0};
  std::unordered_map<const void*, uint64_t> grants_;
  std::vector<Waiter> waiters_;
};

AdaptiveQueueBurstCoordinator& adaptiveQueueBurstCoordinator() {
  static AdaptiveQueueBurstCoordinator coordinator;
  return coordinator;
}

void updateDiagnosticGlobalQueue(
    int64_t bytes,
    int64_t columns,
    const char* event,
    const std::shared_ptr<exec::Task>& task) {
  if (!cudf_velox::deviceMemoryDiagnosticsEnabled()) {
    return;
  }
  const auto currentBytes =
      diagnosticGlobalQueuedBytes.fetch_add(bytes, std::memory_order_relaxed) +
      bytes;
  const auto currentColumns = diagnosticGlobalQueuedColumns.fetch_add(
                                  columns, std::memory_order_relaxed) +
      columns;
  const auto gib = currentBytes >> 30;
  const auto previous =
      diagnosticGlobalQueueGiB.exchange(gib, std::memory_order_relaxed);
  if (gib != previous) {
    LOG(WARNING) << "CUDF_DEVICE_QUEUE_GLOBAL event=" << event
                 << " task=" << (task ? task->taskId() : "n/a")
                 << " queuedBytes=" << currentBytes
                 << " queuedPackedColumns=" << currentColumns;
  }
}
} // namespace

void UcxDestinationQueue::Stats::recordEnqueue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    bytesQueued += data->gpu_data->size();
    packedColumnsQueued++;
  }
}

void UcxDestinationQueue::Stats::recordDequeue(
    const cudf::packed_columns* data) {
  if (data != nullptr) {
    const int64_t size = data->gpu_data->size();

    bytesQueued -= size;
    VELOX_DCHECK_GE(bytesQueued, 0, "bytesQueued must be non-negative");
    --packedColumnsQueued;
    VELOX_DCHECK_GE(
        packedColumnsQueued, 0, "packedColumnsQueued must be non-negative");

    bytesSent += size;
    packedColumnsSent++;
  }
}

void UcxDestinationQueue::enqueueBack(
    std::shared_ptr<cudf::packed_columns> data) {
  // drop duplicate end markers.
  if (data == nullptr && !queue_.empty() && queue_.back() == nullptr) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(data.get());
  }
  queue_.push_back(std::move(data));
}

void UcxDestinationQueue::enqueueFront(
    std::shared_ptr<cudf::packed_columns> data) {
  // ignore nullptr.
  if (data == nullptr) {
    return;
  }

  // insert at the front.
  queue_.push_front(std::move(data));
}

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    UcxDataAvailableCallback notify) {
  return getData(
      std::numeric_limits<uint64_t>::max(),
      sequence_,
      [notify = std::move(notify)](
          std::shared_ptr<cudf::packed_columns> data,
          int64_t /*sequence*/,
          std::vector<int64_t> remainingBytes) mutable {
        if (notify) {
          notify(std::move(data), std::move(remainingBytes));
        }
      });
}

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    uint64_t maxBytes,
    int64_t sequence,
    UcxDataAvailableCallbackV2 notify) {
  if (sequence < sequence_) {
    // A retried/duplicate UCX connection can race with task abort after the
    // original server has already advanced this destination queue.  Treat
    // that request as stale instead of throwing on the communicator thread
    // (an uncaught VeloxException there terminates the whole Spark executor).
    // Returning the current sequence lets UcxExchangeServer identify and
    // close only the stale connection without dequeuing or clearing data.
    LOG(WARNING) << "Ignoring stale UCX queue request: requestedSequence="
                 << sequence << " acknowledgedSequence=" << sequence_;
    return {nullptr, sequence_, {}, true};
  }
  if (notifyV2_ != nullptr && notify != nullptr) {
    // A second server for the same task/destination/sequence must not replace
    // the active server's waiter.  Return a deliberately different sequence
    // so the duplicate UcxExchangeServer follows its stale-connection close
    // path while the original callback remains installed.
    LOG(WARNING) << "Ignoring duplicate UCX queue waiter: sequence=" << sequence
                 << " acknowledgedSequence=" << sequence_;
    return {nullptr, sequence_ + 1, {}, true};
  }
  VELOX_CHECK(
      notify_ == nullptr && notifyV2_ == nullptr,
      "UcxDestinationQueue already has a pending data notification");
  if (sequence > sequence_) {
    // Minimal V2 implementation only supports in-order requests. The full
    // Presto-style ack path can skip prefixes later; for now, install the
    // notify and wait for the requested sequence to become available.
    notifyV2_ = std::move(notify);
    notifySequence_ = sequence;
    notifyMaxBytes_ = maxBytes;
    return {};
  }

  if (queue_.empty()) {
    // delay notification.
    notifyV2_ = std::move(notify);
    notifySequence_ = sequence;
    notifyMaxBytes_ = maxBytes;
    return {};
  }

  // queue is not empty.
  auto data = std::move(queue_.front());
  queue_.pop_front();
  stats_.recordDequeue(data.get());
  const auto resultSequence = sequence_;
  ++sequence_;

  // The current rendezvous protocol does not use the legacy Presto
  // remaining-bytes credit list (UcxExchangeServer sends an empty list in its
  // MetadataMsg). Building it here is O(queue size) for every dequeued packet.
  // A finely chunked shuffle can therefore allocate hundreds of MiB per
  // packet and do O(N^2) work even though the list is immediately discarded.
  std::vector<int64_t> remainingBytes;
  return {std::move(data), resultSequence, std::move(remainingBytes), true};
}

UcxDataAvailable UcxDestinationQueue::deleteResults() {
  for (auto i = 0; i < queue_.size(); ++i) {
    if (queue_[i] == nullptr) {
      VELOX_CHECK_EQ(i, queue_.size() - 1, "null marker found in the middle");
      break;
    }
  }
  queue_.clear();

  UcxDataAvailable result;
  result.callback = std::move(notify_);
  result.callbackV2 = std::move(notifyV2_);
  result.sequence = notifySequence_;
  clearNotify();
  return result;
}

UcxDataAvailable UcxDestinationQueue::getAndClearNotify() {
  if (notify_ == nullptr && notifyV2_ == nullptr) {
    return UcxDataAvailable();
  }
  auto savedV1 = std::move(notify_);
  auto savedV2 = std::move(notifyV2_);
  const auto savedSequence = notifySequence_;
  const auto savedMaxBytes = notifyMaxBytes_;
  clearNotify();

  auto data = getData(
      savedMaxBytes == 0 ? std::numeric_limits<uint64_t>::max() : savedMaxBytes,
      savedSequence,
      nullptr);
  if (!data.immediate) {
    notify_ = std::move(savedV1);
    notifyV2_ = std::move(savedV2);
    notifySequence_ = savedSequence;
    notifyMaxBytes_ = savedMaxBytes;
    return UcxDataAvailable();
  }

  UcxDataAvailable result;
  result.callback = std::move(savedV1);
  result.callbackV2 = std::move(savedV2);
  result.sequence = data.sequence;
  result.data = std::move(data.data);
  result.remainingBytes = std::move(data.remainingBytes);
  return result;
}

void UcxDestinationQueue::clearNotify() {
  notify_ = nullptr;
  notifyV2_ = nullptr;
  notifySequence_ = 0;
  notifyMaxBytes_ = 0;
}

void UcxDestinationQueue::finish() {
  VELOX_CHECK_NULL(notify_, "notify must be cleared before finish");
  VELOX_CHECK_NULL(notifyV2_, "V2 notify must be cleared before finish");
  VELOX_CHECK(queue_.empty(), "data must be fetched before finish");
}

UcxDestinationQueue::Stats UcxDestinationQueue::stats() const {
  return stats_;
}

std::string UcxDestinationQueue::toString() {
  std::stringstream out;
  out << "[available: " << queue_.size() << ", "
      << "sequence: " << sequence_ << ", "
      << (notifyV2_ ? "notifyV2 registered, " : "")
      << (notify_ ? "notify registered, " : "") << this << "]";
  return out.str();
}

// ---------- UcxOutputQueue ----------

UcxOutputQueue::UcxOutputQueue(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind)
    : task_(task), kind_(kind), numDrivers_(numDrivers) {
  if (task_) {
    maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
    continueSize_ = (maxSize_ * kContinuePct) / 100;
    configureAdaptiveBurst(task_->queryCtx()->queryConfig());
    initialized_.store(true, std::memory_order_release);
  } // else: maxSize_ and continueSize_ will be set once the task is created and
    // initialize called.
  // create a queue for each destination.
  queues_.reserve(numDestinations);
  for (int i = 0; i < numDestinations; ++i) {
    // create the destination queues inside the vector using emplace_back.
    queues_.emplace_back(std::make_unique<UcxDestinationQueue>());
  }
}

UcxOutputQueue::~UcxOutputQueue() {
  auto promises = adaptiveQueueBurstCoordinator().removeOwner(
      this, adaptiveBurstGrantedBytes_);
  adaptiveBurstGrantedBytes_ = 0;
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void UcxOutputQueue::configureAdaptiveBurst(
    const core::QueryConfig& queryConfig) {
  adaptiveBurstMaxSize_ = queryConfig.get<uint64_t>(
      kAdaptiveBurstMaxBytesConfig, 0);
  adaptiveGlobalBurstBytes_ = queryConfig.get<uint64_t>(
      kAdaptiveGlobalBurstBytesConfig, 0);
  adaptiveMinDeviceHeadroomBytes_ = queryConfig.get<uint64_t>(
      kAdaptiveMinDeviceHeadroomBytesConfig, 0);
  if (adaptiveBurstMaxSize_ <= maxSize_ ||
      adaptiveGlobalBurstBytes_ == 0) {
    adaptiveBurstMaxSize_ = 0;
    adaptiveGlobalBurstBytes_ = 0;
    adaptiveMinDeviceHeadroomBytes_ = 0;
    return;
  }
  LOG(INFO) << "Adaptive UCX output credit task="
            << (task_ ? task_->taskId() : "n/a")
            << " baseBytes=" << maxSize_
            << " burstMaxBytes=" << adaptiveBurstMaxSize_
            << " globalBurstBytes=" << adaptiveGlobalBurstBytes_
            << " minDeviceHeadroomBytes="
            << adaptiveMinDeviceHeadroomBytes_;
}

bool UcxOutputQueue::initialize(
    std::shared_ptr<exec::Task> task,
    uint32_t numDestinations,
    uint32_t numDrivers,
    core::PartitionedOutputNode::Kind kind) {
  std::lock_guard<std::mutex> l(mutex_);
  if (task_) {
    // already initialized!
    return false;
  }
  kind_ = kind;
  numDrivers_ = numDrivers;
  task_ = task;
  maxSize_ = task_->queryCtx()->queryConfig().maxOutputBufferSize();
  continueSize_ = (maxSize_ * kContinuePct) / 100;
  configureAdaptiveBurst(task_->queryCtx()->queryConfig());
  // Publish task metadata before destination queue expansion. Acceptor only
  // needs task/kind to choose the intra-node path; getData() takes mutex_ and
  // waits for any queue expansion in this function to finish.
  initialized_.store(true, std::memory_order_release);
  // create additional queues if there are more destinations.
  for (int i = queues_.size(); i < numDestinations; ++i) {
    // create the destination queues inside the vector using emplace_back.
    queues_.emplace_back(std::make_unique<UcxDestinationQueue>());
  }
  return true;
}

void UcxOutputQueue::updateNumDrivers(uint32_t newNumDrivers) {
  bool isNoMoreDrivers{false};
  {
    std::lock_guard<std::mutex> l(mutex_);
    numDrivers_ = newNumDrivers;
    // If we finished all drivers, ensure we register that we are 'done'.
    if (numDrivers_ == numFinished_) {
      isNoMoreDrivers = true;
    }
  }
  if (isNoMoreDrivers) {
    noMoreDrivers();
  }
}

void UcxOutputQueue::enqueue(
    int destination,
    std::unique_ptr<cudf::packed_columns> data,
    int32_t numRows) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_NOT_NULL(task_);
  VELOX_CHECK(
      task_->isRunning(), "Task is terminated, cannot add data to output.");
  std::vector<UcxDataAvailable> dataAvailableCallbacks;
  {
    std::lock_guard<std::mutex> l(mutex_);
    auto numBytes = data->gpu_data->size();
    auto sharedData = std::shared_ptr<cudf::packed_columns>(std::move(data));

    bool success = false;
    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      enqueueBroadcastOutputLocked(
          std::move(sharedData), dataAvailableCallbacks);
      // For broadcast, count queuedBytes_ once per active destination so
      // that each destination's dequeue symmetrically decrements it. The
      // total sent stats count the logical data once.
      int numActive = 0;
      for (auto& q : queues_) {
        if (q != nullptr) {
          numActive++;
        }
      }
      updateTotalQueuedBytesMsLocked();
      queuedBytes_ += numBytes * numActive;
      queuedPackedColumns_ += numActive;
      totalBytesSent_ += numBytes;
      totalRowsSent_ += numRows;
      totalPackedColumnsSent_++;
      updateDiagnosticGlobalQueue(
          numBytes * numActive, numActive, "broadcast-enqueue", task_);
      success = true;
    } else {
      VELOX_CHECK_LT(destination, queues_.size());
      success = enqueuePartitionedOutputLocked(
          destination, std::move(sharedData), dataAvailableCallbacks);
      if (success) {
        updateStatsWithEnqueuedLocked(numBytes, numRows);
      }
    }
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
}

bool UcxOutputQueue::checkBlocked(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (future && shouldBlockLocked()) {
    VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
            << " BLOCKED queuedBytes=" << queuedBytes_
            << " maxSize=" << maxSize_
            << " burstMaxSize=" << adaptiveBurstMaxSize_
            << " burstGrantedBytes=" << adaptiveBurstGrantedBytes_
            << " waitingProducers=" << (promises_.size() + 1);
    if (adaptiveBurstMaxSize_ > maxSize_) {
      ++adaptiveBurstBlockedCount_;
      ContinuePromise promise{"UcxOutputQueue::adaptiveCheckBlocked"};
      *future = promise.getSemiFuture();
      adaptiveQueueBurstCoordinator().addWaiter(this, std::move(promise));
    } else {
      promises_.emplace_back("UcxOutputQueue::checkBlocked");
      *future = promises_.back().getSemiFuture();
    }
    return true;
  }
  return false;
}

bool UcxOutputQueue::shouldBlockLocked() {
  VELOX_DCHECK_GE(queuedBytes_, 0);
  const auto queuedBytes = static_cast<uint64_t>(queuedBytes_);
  if (queuedBytes < maxSize_) {
    return false;
  }
  if (adaptiveBurstMaxSize_ <= maxSize_) {
    return true;
  }
  const bool reachedBurstLimit = queuedBytes >= adaptiveBurstMaxSize_;
  const auto accountedBytes = std::min(queuedBytes, adaptiveBurstMaxSize_);
  const auto requestedBurst = accountedBytes - maxSize_;
  if (adaptiveMinDeviceHeadroomBytes_ > 0) {
    const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
    if (!headroom.cudaValid ||
        headroom.allocatableBytes() <= adaptiveMinDeviceHeadroomBytes_) {
      return true;
    }
  }
  if (!adaptiveQueueBurstCoordinator().tryResize(
          this,
          adaptiveBurstGrantedBytes_,
          requestedBurst,
          adaptiveGlobalBurstBytes_)) {
    return true;
  }
  adaptiveBurstGrantedBytes_ = requestedBurst;
  adaptivePeakBurstGrantedBytes_ =
      std::max(adaptivePeakBurstGrantedBytes_, adaptiveBurstGrantedBytes_);
  return reachedBurstLimit;
}

void UcxOutputQueue::shrinkAdaptiveBurstLocked(
    std::vector<ContinuePromise>& promises) {
  if (adaptiveBurstMaxSize_ <= maxSize_) {
    return;
  }
  VELOX_DCHECK_GE(queuedBytes_, 0);
  const auto queuedBytes = static_cast<uint64_t>(queuedBytes_);
  const auto requestedBurst = queuedBytes > maxSize_
      ? std::min(queuedBytes, adaptiveBurstMaxSize_) - maxSize_
      : 0;
  if (requestedBurst >= adaptiveBurstGrantedBytes_) {
    auto adaptivePromises =
        adaptiveQueueBurstCoordinator().notifyProgress();
    promises.insert(
        promises.end(),
        std::make_move_iterator(adaptivePromises.begin()),
        std::make_move_iterator(adaptivePromises.end()));
    return;
  }
  auto adaptivePromises = adaptiveQueueBurstCoordinator().releaseTo(
      this, adaptiveBurstGrantedBytes_, requestedBurst);
  adaptiveBurstGrantedBytes_ = requestedBurst;
  promises.insert(
      promises.end(),
      std::make_move_iterator(adaptivePromises.begin()),
      std::make_move_iterator(adaptivePromises.end()));
}

void UcxOutputQueue::getData(int destination, UcxDataAvailableCallback notify) {
  UcxDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    // If the queue doesn't exist yet, create an empty queue to store
    // the notify callback. The queue will eventually be initialized when
    // the task is being created.
    for (int i = queues_.size(); i <= destination; ++i) {
      // create the destination queues inside the vector using emplace_back.
      queues_.emplace_back(std::make_unique<UcxDestinationQueue>());
    }
    auto* queue = queues_[destination].get();
    // queue can be nullptr here if the task has terminated and results
    // have been removed. In this case, no data is returned.
    if (queue) {
      // Capture weak_ptr instead of raw `this` to prevent use-after-free.
      // The callback fires outside the lock (from enqueue() or terminate()),
      // and concurrent removeTask() can destroy the UcxOutputQueue while
      // the callback is still executing.
      std::weak_ptr<UcxOutputQueue> weakSelf = shared_from_this();
      data = queue->getData([notify, weakSelf](
                                std::shared_ptr<cudf::packed_columns> data,
                                std::vector<int64_t> remainingBytes) {
        std::vector<ContinuePromise> promises;
        int64_t bytes = data ? data->gpu_data->size() : -1L;
        notify(std::move(data), std::move(remainingBytes));
        if (bytes >= 0L) {
          auto self = weakSelf.lock();
          if (!self) {
            // Queue was destroyed by removeTask(), safe to skip stats update.
            return;
          }
          std::lock_guard<std::mutex> l(self->mutex_);
          self->updateStatsWithFreedLocked(bytes, 1L, promises);
        }
        // outside of lock:
        // wake up any producers that are waiting for queue to become less full.
        for (auto& promise : promises) {
          promise.setValue();
        }
      });
      if (data.data) {
        // This implies data.immediate and no notify upcall will be done.
        // Need to update the stats here.
        updateStatsWithFreedLocked(data.data->gpu_data->size(), 1L, promises);
      }
    } else {
      data = UcxDestinationQueue::Data{nullptr, 0, {}, true};
    }
  }
  // outside lock: If we have data, then return it immediately.
  if (data.immediate) {
    notify(std::move(data.data), std::move(data.remainingBytes));
  } else {
    VLOG(2) << "[QUEUE] task=" << (task_ ? task_->taskId() : "n/a")
            << " dest=" << destination
            << " server waiting for data (callback installed)";
  }
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void UcxOutputQueue::getData(
    int destination,
    uint64_t maxBytes,
    int64_t sequence,
    UcxDataAvailableCallbackV2 notify) {
  UcxDestinationQueue::Data data;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    for (int i = queues_.size(); i <= destination; ++i) {
      queues_.emplace_back(std::make_unique<UcxDestinationQueue>());
    }
    auto* queue = queues_[destination].get();
    if (queue) {
      std::weak_ptr<UcxOutputQueue> weakSelf = shared_from_this();
      data = queue->getData(
          maxBytes,
          sequence,
          [notify, weakSelf](
              std::shared_ptr<cudf::packed_columns> data,
              int64_t sequence,
              std::vector<int64_t> remainingBytes) {
            std::vector<ContinuePromise> promises;
            int64_t bytes = data ? data->gpu_data->size() : -1L;
            notify(std::move(data), sequence, std::move(remainingBytes));
            if (bytes >= 0L) {
              auto self = weakSelf.lock();
              if (!self) {
                return;
              }
              std::lock_guard<std::mutex> l(self->mutex_);
              self->updateStatsWithFreedLocked(bytes, 1L, promises);
            }
            for (auto& promise : promises) {
              promise.setValue();
            }
          });
      if (data.data) {
        updateStatsWithFreedLocked(data.data->gpu_data->size(), 1L, promises);
      }
    } else {
      data = UcxDestinationQueue::Data{nullptr, sequence, {}, true};
    }
  }
  if (data.immediate) {
    notify(std::move(data.data), data.sequence, std::move(data.remainingBytes));
  } else {
    VLOG(2) << "[QUEUE] task=" << (task_ ? task_->taskId() : "n/a")
            << " dest=" << destination
            << " server waiting for V2 data (callback installed)"
            << " sequence=" << sequence << " maxBytes=" << maxBytes;
  }
  for (auto& promise : promises) {
    promise.setValue();
  }
}

void UcxOutputQueue::noMoreData() {
  // Increment number of finished drivers.
  checkIfDone(true);
}

void UcxOutputQueue::noMoreDrivers() {
  // Do not increment number of finished drivers.
  checkIfDone(false);
}

void UcxOutputQueue::checkIfDone(bool oneDriverFinished) {
  std::vector<UcxDataAvailable> finished;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (oneDriverFinished) {
      ++numFinished_;
    }
    VELOX_CHECK_LE(
        numFinished_,
        numDrivers_,
        "Each driver should call noMoreData exactly once");
    atEnd_ = numFinished_ == numDrivers_;
    if (!atEnd_) {
      return;
    }
    {
      int64_t avgRows = totalPackedColumnsSent_ > 0
          ? totalRowsSent_ / totalPackedColumnsSent_
          : 0;
      VLOG(1) << "[OUTPUT-STATS] task=" << (task_ ? task_->taskId() : "n/a")
              << " totalRows=" << totalRowsSent_
              << " chunks=" << totalPackedColumnsSent_
              << " avgRowsPerChunk=" << avgRows
              << " totalBytes=" << totalBytesSent_;
      if (adaptiveBurstMaxSize_ > maxSize_) {
        LOG(INFO) << "Adaptive UCX output credit summary task="
                  << (task_ ? task_->taskId() : "n/a")
                  << " peakBurstGrantedBytes="
                  << adaptivePeakBurstGrantedBytes_
                  << " adaptiveBlockedCount=" << adaptiveBurstBlockedCount_;
      }
    }
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr);
        finished.push_back(queue->getAndClearNotify());
      }
    }
  }
  // Notify outside of mutex.
  for (auto& notification : finished) {
    notification.notify();
  }
}

bool UcxOutputQueue::enqueuePartitionedOutputLocked(
    int destination,
    std::shared_ptr<cudf::packed_columns> data,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());
  VELOX_CHECK_LT(destination, queues_.size());
  bool success = false;
  auto* queue = queues_[destination].get();
  if (queue != nullptr) {
    queue->enqueueBack(std::move(data));
    dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    success = true;
  }
  return success;
}

void UcxOutputQueue::enqueueBroadcastOutputLocked(
    std::shared_ptr<cudf::packed_columns> data,
    std::vector<UcxDataAvailable>& dataAvailableCbs) {
  VELOX_DCHECK(dataAvailableCbs.empty());

  for (auto& queue : queues_) {
    if (queue != nullptr) {
      queue->enqueueBack(data);
      dataAvailableCbs.emplace_back(queue->getAndClearNotify());
    }
  }

  // Store for late-arriving destinations (backfill).
  if (!noMoreQueues_) {
    dataToBroadcast_.emplace_back(std::move(data));
  }
}

bool UcxOutputQueue::isFinished() {
  std::lock_guard<std::mutex> l(mutex_);
  return isFinishedLocked();
}

bool UcxOutputQueue::isFinishedLocked() {
  // For broadcast, we can only be finished after receiving the no more
  // (destination) buffers signal, matching OutputBuffer::isFinishedLocked().
  if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast &&
      !noMoreQueues_) {
    return false;
  }
  for (auto& queue : queues_) {
    if (queue != nullptr) {
      return false;
    }
  }
  return true;
}

void UcxOutputQueue::updateOutputBuffers(int numBuffers, bool noMoreBuffers) {
  using Kind = core::PartitionedOutputNode::Kind;
  if (kind_ == Kind::kPartitioned) {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_EQ(queues_.size(), numBuffers);
    VELOX_CHECK(noMoreBuffers);
    noMoreQueues_ = true;
    return;
  }

  VELOX_CHECK_EQ(kind_, Kind::kBroadcast);
  bool isFinished;
  {
    std::lock_guard<std::mutex> l(mutex_);

    if (numBuffers > queues_.size()) {
      // Add new destination queues and backfill with broadcast data.
      int32_t numNewBuffers = numBuffers - queues_.size();
      queues_.reserve(numBuffers);
      for (int32_t i = 0; i < numNewBuffers; ++i) {
        auto buffer = std::make_unique<UcxDestinationQueue>();
        for (const auto& data : dataToBroadcast_) {
          buffer->enqueueBack(data);
          // Account for backfilled data in queuedBytes_ so that dequeue
          // decrements don't drive it negative.
          queuedBytes_ += data->gpu_data->size();
          queuedPackedColumns_++;
          updateDiagnosticGlobalQueue(
              data->gpu_data->size(), 1, "broadcast-backfill", task_);
        }
        if (atEnd_) {
          buffer->enqueueBack(nullptr);
        }
        queues_.emplace_back(std::move(buffer));
      }
    }

    if (!noMoreBuffers) {
      return;
    }

    noMoreQueues_ = true;
    dataToBroadcast_.clear();
    isFinished = isFinishedLocked();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::deleteResults(int destination) {
  bool isFinished;
  UcxDataAvailable dataAvailable;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (destination >= queues_.size()) {
      VLOG(1) << "deleteResults: destination " << destination
              << " out of range (size=" << queues_.size() << "), ignoring";
      return;
    }
    auto* queue = queues_[destination].get();
    if (queue == nullptr) {
      VLOG(1) << "Extra delete received for destination " << destination;
      return;
    }
    // remember destination queue fill stats
    int64_t bytes = queue->stats().bytesQueued;
    int64_t packedCols = queue->stats().packedColumnsQueued;
    dataAvailable = queue->deleteResults();
    queue->finish();
    queues_[destination] = nullptr;
    isFinished = isFinishedLocked();
    // update UcxOutputQueue stats
    if (bytes > 0 || packedCols > 0) {
      updateStatsWithFreedLocked(bytes, packedCols, promises);
    } else {
      promises = std::move(promises_);
    }
  }

  // Outside of mutex.
  dataAvailable.notify();
  // wake up any producers that are waiting for queue to become less full.
  for (auto& promise : promises) {
    promise.setValue();
  }

  if (isFinished && task_) {
    task_->setAllOutputConsumed();
  }
}

void UcxOutputQueue::terminate() {
  std::vector<UcxDataAvailable> pendingCallbacks;
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    if (task_ && task_->isRunning()) {
      LOG(WARNING) << "UcxOutputQueue::terminate() called while task "
                   << task_->taskId() << " is still running";
    }
    // Fire all pending getData callbacks with nullptr to signal end-of-stream.
    // This handles the case where a producer task fails or is cancelled before
    // noMoreData() is called, preventing consumers from being orphaned.
    for (auto& queue : queues_) {
      if (queue != nullptr) {
        queue->enqueueBack(nullptr);
        pendingCallbacks.push_back(queue->getAndClearNotify());
      }
    }
    // Release any outstanding producer-side promises (blocked on queue-full).
    promises = std::move(promises_);
    // Cancellation must wake this task's waiters, but its borrowed credit
    // remains reserved while queued GPU pages are still reachable. The
    // destructor releases the credit together with those pages.
    auto adaptivePromises =
        adaptiveQueueBurstCoordinator().cancelOwnerWaiters(this);
    promises.insert(
        promises.end(),
        std::make_move_iterator(adaptivePromises.begin()),
        std::make_move_iterator(adaptivePromises.end()));
  }

  // Fire callbacks outside of mutex to avoid potential deadlocks.
  for (auto& callback : pendingCallbacks) {
    callback.notify();
  }
  // Unblock any blocked producers.
  for (auto& promise : promises) {
    promise.setValue();
  }
}

exec::OutputBuffer::Stats UcxOutputQueue::stats() {
  std::lock_guard<std::mutex> l(mutex_);
  std::vector<UcxDestinationQueue::Stats> queueStats;

  updateTotalQueuedBytesMsLocked();

  auto stats = exec::OutputBuffer::Stats(
      kind(),
      noMoreQueues_,
      atEnd_,
      isFinishedLocked(),
      queuedBytes_,
      queuedPackedColumns_,
      totalBytesSent_,
      totalRowsSent_,
      totalPackedColumnsSent_,
      getAverageQueueTimeMsLocked(),
      0 /* FIXME: compute num top buffers. */,
      {/* FIXME: transition queueStats to exec::DestinationBuffer::Stats */});
  return stats;
}

void UcxOutputQueue::updateStatsWithEnqueuedLocked(
    int64_t bytes,
    int64_t rows) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ += bytes;
  queuedPackedColumns_++;

  totalBytesSent_ += bytes;
  totalRowsSent_ += rows;
  totalPackedColumnsSent_++;
  updateDiagnosticGlobalQueue(bytes, 1, "enqueue", task_);
  logDeviceQueueResidencyLocked("enqueue");
}

void UcxOutputQueue::updateStatsWithFreedLocked(
    int64_t bytes,
    int64_t numPackedCols,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ -= bytes;
  queuedPackedColumns_ -= numPackedCols;

  VELOX_CHECK_GE(queuedBytes_, 0);
  VELOX_CHECK_GE(queuedPackedColumns_, 0);
  updateDiagnosticGlobalQueue(-bytes, -numPackedCols, "dequeue", task_);
  logDeviceQueueResidencyLocked("dequeue");
  shrinkAdaptiveBurstLocked(promises);

  // Check whether queue is below low-water mark and return outstanding
  // promises
  if (queuedBytes_ <= continueSize_ && !promises_.empty()) {
    VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
            << " UNBLOCKING " << promises_.size() << " producers"
            << " queuedBytes=" << queuedBytes_
            << " continueSize=" << continueSize_;
    auto legacyPromises = std::move(promises_);
    promises.insert(
        promises.end(),
        std::make_move_iterator(legacyPromises.begin()),
        std::make_move_iterator(legacyPromises.end()));
  }
}

void UcxOutputQueue::logDeviceQueueResidencyLocked(const char* event) {
  if (!cudf_velox::deviceMemoryDiagnosticsEnabled()) {
    return;
  }
  constexpr int64_t kBucketBytes = 256LL << 20;
  const auto bucket = queuedBytes_ / kBucketBytes;
  if (bucket == diagnosticQueueBucket_) {
    return;
  }
  diagnosticQueueBucket_ = bucket;
  LOG(WARNING) << "CUDF_DEVICE_QUEUE event=" << event
               << " task=" << (task_ ? task_->taskId() : "n/a")
               << " queuedBytes=" << queuedBytes_
               << " queuedPackedColumns=" << queuedPackedColumns_
               << " maxSize=" << maxSize_ << " continueSize=" << continueSize_;
}

void UcxOutputQueue::updateTotalQueuedBytesMsLocked() {
  const auto nowMs = getCurrentTimeMs();
  if (queuedBytes_ > 0) {
    const auto deltaMs = nowMs - queueStartMs_;
    totalQueuedBytesMs_ += queuedBytes_ * deltaMs;
  }

  queueStartMs_ = nowMs;
}

int64_t UcxOutputQueue::getAverageQueueTimeMsLocked() const {
  if (totalBytesSent_ > 0) {
    return totalQueuedBytesMs_ / totalBytesSent_;
  }

  return 0;
}

} // namespace facebook::velox::ucx_exchange
