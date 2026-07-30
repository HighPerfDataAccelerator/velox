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

#include <malloc.h>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include "cuda_runtime.h"

namespace facebook::velox::ucx_exchange {

namespace {
std::atomic<int64_t> diagnosticGlobalQueuedBytes{0};
std::atomic<int64_t> diagnosticGlobalQueuedColumns{0};
std::atomic<int64_t> diagnosticGlobalQueueGiB{0};

struct PinnedScratch {
  ~PinnedScratch() {
    if (data != nullptr) {
      cudaFreeHost(data);
    }
  }

  uint8_t* ensure(size_t bytes) {
    if (capacity >= bytes) {
      return data;
    }
    if (data != nullptr) {
      CUDF_CUDA_TRY(cudaFreeHost(data));
      data = nullptr;
      capacity = 0;
    }
    CUDF_CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&data), bytes));
    capacity = bytes;
    return data;
  }

  uint8_t* data{nullptr};
  size_t capacity{0};
};

thread_local PinnedScratch pinnedScratch;

uint64_t configuredHostSpoolMaxBytes() {
  const auto* value = std::getenv("GLUTEN_UCX_HOST_SPOOL_MAX_BYTES");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  errno = 0;
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (errno == 0 && end != value && *end == '\0' && parsed > 0) {
    return parsed;
  }
  LOG(WARNING) << "Ignoring invalid GLUTEN_UCX_HOST_SPOOL_MAX_BYTES=" << value;
  return 0;
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

void UcxDestinationQueue::Stats::recordEnqueue(const UcxTransferData* data) {
  if (data != nullptr) {
    bytesQueued += data->size();
    backpressureBytesQueued += data->backpressureSize();
    deviceBytesQueued += data->isHostStaged() ? 0 : data->backpressureSize();
    packedColumnsQueued++;
  }
}

void UcxDestinationQueue::Stats::recordDequeue(const UcxTransferData* data) {
  if (data != nullptr) {
    const int64_t size = data->size();
    const int64_t backpressureSize = data->backpressureSize();
    const int64_t deviceBytes = data->isHostStaged() ? 0 : backpressureSize;

    bytesQueued -= size;
    VELOX_DCHECK_GE(bytesQueued, 0, "bytesQueued must be non-negative");
    backpressureBytesQueued -= backpressureSize;
    VELOX_DCHECK_GE(
        backpressureBytesQueued,
        0,
        "backpressureBytesQueued must be non-negative");
    deviceBytesQueued -= deviceBytes;
    VELOX_DCHECK_GE(
        deviceBytesQueued, 0, "deviceBytesQueued must be non-negative");
    --packedColumnsQueued;
    VELOX_DCHECK_GE(
        packedColumnsQueued, 0, "packedColumnsQueued must be non-negative");

    bytesSent += size;
    packedColumnsSent++;
  }
}

void UcxDestinationQueue::enqueueBack(std::shared_ptr<UcxTransferData> data) {
  // drop duplicate end markers.
  if (data == nullptr && !queue_.empty() && queue_.back() == nullptr) {
    return;
  }

  if (data != nullptr) {
    stats_.recordEnqueue(data.get());
  }
  queue_.push_back(std::move(data));
}

void UcxDestinationQueue::enqueueFront(std::shared_ptr<UcxTransferData> data) {
  // ignore nullptr.
  if (data == nullptr) {
    return;
  }

  // insert at the front.
  queue_.push_front(std::move(data));
}

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    UcxTransferDataAvailableCallback notify) {
  return getData(
      std::numeric_limits<uint64_t>::max(), sequence_, std::move(notify));
}

UcxDestinationQueue::Data UcxDestinationQueue::getData(
    uint64_t maxBytes,
    int64_t sequence,
    UcxTransferDataAvailableCallback notify) {
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
  if (notify_ != nullptr && notify != nullptr) {
    // A second server for the same task/destination/sequence must not replace
    // the active server's waiter.  Return a deliberately different sequence
    // so the duplicate UcxExchangeServer follows its stale-connection close
    // path while the original callback remains installed.
    LOG(WARNING) << "Ignoring duplicate UCX queue waiter: sequence=" << sequence
                 << " acknowledgedSequence=" << sequence_;
    return {nullptr, sequence_ + 1, {}, true};
  }
  VELOX_CHECK(
      notify_ == nullptr,
      "UcxDestinationQueue already has a pending data notification");
  if (sequence > sequence_) {
    // Minimal V2 implementation only supports in-order requests. The full
    // Presto-style ack path can skip prefixes later; for now, install the
    // notify and wait for the requested sequence to become available.
    notify_ = std::move(notify);
    notifySequence_ = sequence;
    notifyMaxBytes_ = maxBytes;
    return {};
  }

  if (queue_.empty()) {
    // delay notification.
    notify_ = std::move(notify);
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
  result.sequence = notifySequence_;
  clearNotify();
  return result;
}

UcxDataAvailable UcxDestinationQueue::getAndClearNotify() {
  if (notify_ == nullptr) {
    return UcxDataAvailable();
  }
  auto saved = std::move(notify_);
  const auto savedSequence = notifySequence_;
  const auto savedMaxBytes = notifyMaxBytes_;
  clearNotify();

  auto data = getData(
      savedMaxBytes == 0 ? std::numeric_limits<uint64_t>::max() : savedMaxBytes,
      savedSequence,
      nullptr);
  if (!data.immediate) {
    notify_ = std::move(saved);
    notifySequence_ = savedSequence;
    notifyMaxBytes_ = savedMaxBytes;
    return UcxDataAvailable();
  }

  UcxDataAvailable result;
  result.callback = std::move(saved);
  result.sequence = data.sequence;
  result.data = std::move(data.data);
  result.remainingBytes = std::move(data.remainingBytes);
  return result;
}

void UcxDestinationQueue::clearNotify() {
  notify_ = nullptr;
  notifySequence_ = 0;
  notifyMaxBytes_ = 0;
}

void UcxDestinationQueue::finish() {
  VELOX_CHECK_NULL(notify_, "notify must be cleared before finish");
  VELOX_CHECK(queue_.empty(), "data must be fetched before finish");
}

UcxDestinationQueue::Stats UcxDestinationQueue::stats() const {
  return stats_;
}

std::string UcxDestinationQueue::toString() {
  std::stringstream out;
  out << "[available: " << queue_.size() << ", "
      << "sequence: " << sequence_ << ", "
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
  if (hostSpooling_) {
    hostSpoolMaxSize_ = std::max(hostSpoolMaxSize_, maxSize_);
    hostSpoolContinueSize_ = (hostSpoolMaxSize_ * kContinuePct) / 100;
  }
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

  bool hostSpooling = false;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK_GE(destination, 0);
    VELOX_CHECK_LT(destination, queues_.size());
    hostSpooling = hostSpooling_;
  }

  const auto numBytes = static_cast<int64_t>(data->gpu_data->size());
  auto transfer = std::make_shared<UcxTransferData>();
  if (hostSpooling) {
    VELOX_CHECK_NE(
        kind_,
        core::PartitionedOutputNode::Kind::kBroadcast,
        "Host spooling is only supported for partitioned output");
    transfer->metadata =
        std::shared_ptr<std::vector<uint8_t>>(std::move(data->metadata));
    auto* hostStaging = pinnedScratch.ensure(static_cast<size_t>(numBytes));
    const auto stream = data->gpu_data->stream();
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    CUDF_CUDA_TRY(cudaMemcpy(
        hostStaging, data->gpu_data->data(), numBytes, cudaMemcpyDeviceToHost));
    transfer->hostData =
        std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(numBytes));
    std::memcpy(transfer->hostData->data(), hostStaging, numBytes);
    data.reset();
  } else {
    transfer->deviceData =
        std::shared_ptr<cudf::packed_columns>(std::move(data));
  }
  const auto hostResidentBytes = transfer->hostResidentSize();

  std::vector<UcxDataAvailable> dataAvailableCallbacks;
  bool trimHostAllocator = false;
  {
    std::lock_guard<std::mutex> l(mutex_);
    bool success = false;
    if (kind_ == core::PartitionedOutputNode::Kind::kBroadcast) {
      VELOX_CHECK_EQ(destination, 0, "Broadcast uses destination 0");
      enqueueBroadcastOutputLocked(
          transfer->deviceData, dataAvailableCallbacks);
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
      deviceQueuedBytes_ += numBytes * numActive;
      queuedPackedColumns_ += numActive;
      totalBytesSent_ += numBytes;
      totalRowsSent_ += numRows;
      totalPackedColumnsSent_++;
      success = true;
    } else {
      VELOX_CHECK_LT(destination, queues_.size());
      success = enqueuePartitionedOutputLocked(
          destination, std::move(transfer), dataAvailableCallbacks);
      if (success) {
        updateStatsWithEnqueuedLocked(
            numBytes,
            hostSpooling ? hostResidentBytes : numBytes,
            hostSpooling ? 0 : numBytes,
            numRows);
        if (hostSpooling) {
          hostSpooledBytes_ += numBytes;
          hostSpoolPeakBytes_ =
              std::max(hostSpoolPeakBytes_, hostSpooledBytes_);
          hostSpoolResidentBytes_ += hostResidentBytes;
          hostSpoolPeakResidentBytes_ =
              std::max(hostSpoolPeakResidentBytes_, hostSpoolResidentBytes_);
          const auto currentGiB = hostSpooledBytes_ >> 30;
          const auto previousGiB = (hostSpooledBytes_ - numBytes) >> 30;
          if (currentGiB != previousGiB) {
            trimHostAllocator = true;
            LOG(WARNING) << "CUDF_UCX_HOST_SPOOL task=" << task_->taskId()
                         << " event=enqueue hostQueuedBytes="
                         << hostSpooledBytes_
                         << " hostPeakBytes=" << hostSpoolPeakBytes_
                         << " hostResidentBytes=" << hostSpoolResidentBytes_
                         << " hostPeakResidentBytes="
                         << hostSpoolPeakResidentBytes_
                         << " packedColumns=" << queuedPackedColumns_;
          }
        }
      }
    }
  }
  // Now that data is enqueued, notify blocked readers (outside of mutex.)
  for (auto& callback : dataAvailableCallbacks) {
    callback.notify();
  }
  if (trimHostAllocator) {
    malloc_trim(0);
  }
}

void UcxOutputQueue::enableHostSpooling() {
  std::lock_guard<std::mutex> l(mutex_);
  VELOX_CHECK_EQ(
      queuedPackedColumns_, 0, "Host spooling must be enabled before enqueue");
  VELOX_CHECK_NE(
      kind_,
      core::PartitionedOutputNode::Kind::kBroadcast,
      "Host spooling is not supported for broadcast output");
  if (hostSpooling_) {
    return;
  }
  hostSpooling_ = true;
  hostSpoolMaxSize_ =
      std::max<uint64_t>(maxSize_, configuredHostSpoolMaxBytes());
  hostSpoolContinueSize_ = (hostSpoolMaxSize_ * kContinuePct) / 100;
  LOG(WARNING) << "CUDF_UCX_HOST_SPOOL task="
               << (task_ ? task_->taskId() : "pending")
               << " event=enabled hostMaxBytes=" << hostSpoolMaxSize_
               << " deviceMaxBytes=" << maxSize_;
}

bool UcxOutputQueue::checkBlocked(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  const bool deviceBlocked = deviceQueuedBytes_ >= maxSize_;
  const bool hostBlocked = hostSpooling_ && queuedBytes_ >= hostSpoolMaxSize_;
  if ((deviceBlocked || hostBlocked) && future) {
    VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
            << " BLOCKED queuedBytes=" << queuedBytes_
            << " hostMaxSize=" << hostSpoolMaxSize_
            << " deviceQueuedBytes=" << deviceQueuedBytes_
            << " deviceMaxSize=" << maxSize_
            << " waitingProducers=" << (promises_.size() + 1);
    promises_.emplace_back("UcxOutputQueue::checkBlocked");
    *future = promises_.back().getSemiFuture();
    return true;
  }
  return false;
}

void UcxOutputQueue::getData(int destination, UcxDataAvailableCallback notify) {
  getTransferData(
      destination,
      std::numeric_limits<uint64_t>::max(),
      -1,
      [notify = std::move(notify)](
          std::shared_ptr<UcxTransferData> data,
          int64_t /*sequence*/,
          std::vector<int64_t> remainingBytes) mutable {
        VELOX_CHECK(
            data == nullptr || data->deviceData != nullptr,
            "Legacy UCX queue fetch cannot consume host-spooled data");
        notify(
            data == nullptr ? nullptr : data->deviceData,
            std::move(remainingBytes));
      });
}

void UcxOutputQueue::getData(
    int destination,
    uint64_t maxBytes,
    int64_t sequence,
    UcxDataAvailableCallbackV2 notify) {
  getTransferData(
      destination,
      maxBytes,
      sequence,
      [notify = std::move(notify)](
          std::shared_ptr<UcxTransferData> data,
          int64_t sequence,
          std::vector<int64_t> remainingBytes) mutable {
        VELOX_CHECK(
            data == nullptr || data->deviceData != nullptr,
            "Legacy UCX queue fetch cannot consume host-spooled data");
        notify(
            data == nullptr ? nullptr : data->deviceData,
            sequence,
            std::move(remainingBytes));
      });
}

void UcxOutputQueue::getTransferData(
    int destination,
    uint64_t maxBytes,
    int64_t sequence,
    UcxTransferDataAvailableCallback notify) {
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
      auto callback = [notify, weakSelf](
                          std::shared_ptr<UcxTransferData> data,
                          int64_t sequence,
                          std::vector<int64_t> remainingBytes) {
        std::vector<ContinuePromise> promises;
        int64_t bytes = data ? data->size() : -1L;
        int64_t backpressureBytes = data ? data->backpressureSize() : -1L;
        const bool hostStaged = data && data->isHostStaged();
        const int64_t hostResidentBytes =
            hostStaged ? data->hostResidentSize() : 0;
        notify(std::move(data), sequence, std::move(remainingBytes));
        if (bytes >= 0L) {
          auto self = weakSelf.lock();
          if (!self) {
            return;
          }
          std::lock_guard<std::mutex> l(self->mutex_);
          self->updateStatsWithFreedLocked(
              backpressureBytes,
              hostStaged ? 0 : backpressureBytes,
              1L,
              promises);
          if (hostStaged) {
            self->hostSpooledBytes_ -= bytes;
            VELOX_CHECK_GE(self->hostSpooledBytes_, 0);
            self->hostSpoolResidentBytes_ -= hostResidentBytes;
            VELOX_CHECK_GE(self->hostSpoolResidentBytes_, 0);
          }
        }
        for (auto& promise : promises) {
          promise.setValue();
        }
      };
      data = sequence < 0
          ? queue->getData(std::move(callback))
          : queue->getData(maxBytes, sequence, std::move(callback));
      if (data.data) {
        const auto bytes = data.data->size();
        updateStatsWithFreedLocked(
            data.data->backpressureSize(),
            data.data->isHostStaged() ? 0 : data.data->backpressureSize(),
            1L,
            promises);
        if (data.data->isHostStaged()) {
          hostSpooledBytes_ -= bytes;
          VELOX_CHECK_GE(hostSpooledBytes_, 0);
          hostSpoolResidentBytes_ -= data.data->hostResidentSize();
          VELOX_CHECK_GE(hostSpoolResidentBytes_, 0);
        }
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
      if (hostSpooling_) {
        LOG(WARNING) << "CUDF_UCX_HOST_SPOOL task="
                     << (task_ ? task_->taskId() : "n/a")
                     << " event=producer_finished"
                     << " hostQueuedBytes=" << hostSpooledBytes_
                     << " hostPeakBytes=" << hostSpoolPeakBytes_
                     << " hostResidentBytes=" << hostSpoolResidentBytes_
                     << " hostPeakResidentBytes=" << hostSpoolPeakResidentBytes_
                     << " totalBytes=" << totalBytesSent_
                     << " packedColumns=" << totalPackedColumnsSent_;
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
    std::shared_ptr<UcxTransferData> data,
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

  auto transfer = std::make_shared<UcxTransferData>();
  transfer->deviceData = data;
  for (auto& queue : queues_) {
    if (queue != nullptr) {
      queue->enqueueBack(transfer);
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
          auto transfer = std::make_shared<UcxTransferData>();
          transfer->deviceData = data;
          buffer->enqueueBack(std::move(transfer));
          // Account for backfilled data in queuedBytes_ so that dequeue
          // decrements don't drive it negative.
          queuedBytes_ += data->gpu_data->size();
          deviceQueuedBytes_ += data->gpu_data->size();
          queuedPackedColumns_++;
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
    int64_t bytes = queue->stats().backpressureBytesQueued;
    int64_t deviceBytes = queue->stats().deviceBytesQueued;
    int64_t packedCols = queue->stats().packedColumnsQueued;
    dataAvailable = queue->deleteResults();
    queue->finish();
    queues_[destination] = nullptr;
    isFinished = isFinishedLocked();
    // update UcxOutputQueue stats
    if (bytes > 0 || packedCols > 0) {
      updateStatsWithFreedLocked(bytes, deviceBytes, packedCols, promises);
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
    int64_t logicalBytes,
    int64_t backpressureBytes,
    int64_t deviceBytes,
    int64_t rows) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ += backpressureBytes;
  deviceQueuedBytes_ += deviceBytes;
  queuedPackedColumns_++;

  totalBytesSent_ += logicalBytes;
  totalRowsSent_ += rows;
  totalPackedColumnsSent_++;
  updateDiagnosticGlobalQueue(backpressureBytes, 1, "enqueue", task_);
  logDeviceQueueResidencyLocked("enqueue");
}

void UcxOutputQueue::updateStatsWithFreedLocked(
    int64_t bytes,
    int64_t deviceBytes,
    int64_t numPackedCols,
    std::vector<ContinuePromise>& promises) {
  updateTotalQueuedBytesMsLocked();

  queuedBytes_ -= bytes;
  deviceQueuedBytes_ -= deviceBytes;
  queuedPackedColumns_ -= numPackedCols;

  VELOX_CHECK_GE(queuedBytes_, 0);
  VELOX_CHECK_GE(deviceQueuedBytes_, 0);
  VELOX_CHECK_GE(queuedPackedColumns_, 0);
  updateDiagnosticGlobalQueue(-bytes, -numPackedCols, "dequeue", task_);
  logDeviceQueueResidencyLocked("dequeue");

  // Check whether queue is below low-water mark and return outstanding
  // promises
  const bool belowDeviceLowWater = deviceQueuedBytes_ <= continueSize_;
  const bool belowHostLowWater =
      !hostSpooling_ || queuedBytes_ <= hostSpoolContinueSize_;
  if (belowDeviceLowWater && belowHostLowWater && !promises_.empty()) {
    VLOG(2) << "[BACKPRESSURE] task=" << (task_ ? task_->taskId() : "n/a")
            << " UNBLOCKING " << promises_.size() << " producers"
            << " queuedBytes=" << queuedBytes_
            << " hostContinueSize=" << hostSpoolContinueSize_
            << " deviceQueuedBytes=" << deviceQueuedBytes_
            << " deviceContinueSize=" << continueSize_;
    promises = std::move(promises_);
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
               << " deviceQueuedBytes=" << deviceQueuedBytes_
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
