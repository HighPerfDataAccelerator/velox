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

#include "velox/experimental/ucx-exchange/LocalDeviceOutputQueueManager.h"

#include <algorithm>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {

namespace {
constexpr uint64_t kContinuePct = 90;

void fulfill(std::vector<ContinuePromise> promises) {
  for (auto& promise : promises) {
    promise.setValue();
  }
}
} // namespace

LocalDeviceOutputQueueManager::TaskQueue::TaskQueue(
    std::shared_ptr<exec::Task> taskIn,
    int numDestinations,
    int numDriversIn)
    : task(std::move(taskIn)),
      queues(numDestinations),
      destinationDeleted(numDestinations, false),
      nextSequences(numDestinations, 0),
      numDrivers(numDriversIn) {
  VELOX_CHECK_NOT_NULL(task);
  VELOX_CHECK_GT(numDestinations, 0);
  VELOX_CHECK_GT(numDrivers, 0);
  // The terminal same-process cache queue has a different producer/consumer
  // geometry from an inter-fragment UCX queue. Keep its larger overlap budget
  // on a dedicated key. Reusing kMaxOutputBufferSize here forces every
  // internal fragment in the same MPP runtime to inherit the cache budget and
  // lets several independent 8 GiB queues exhaust a 32 GiB GPU.
  maxBytes = task->queryCtx()->queryConfig().get<uint64_t>(
      kMaxBytesConfig, task->queryCtx()->queryConfig().maxOutputBufferSize());
  continueBytes = maxBytes * kContinuePct / 100;
  LOG(INFO) << "Local device output queue task=" << task->taskId()
            << " maxBytes=" << maxBytes << " ordinaryMppMaxBytes="
            << task->queryCtx()->queryConfig().maxOutputBufferSize();
}

std::shared_ptr<LocalDeviceOutputQueueManager>
LocalDeviceOutputQueueManager::getInstanceRef() {
  static auto instance = std::make_shared<LocalDeviceOutputQueueManager>();
  return instance;
}

void LocalDeviceOutputQueueManager::registerDirectOutputTask(
    std::string taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto [_, inserted] = directOutputTasks_.insert(std::move(taskId));
  VELOX_CHECK(inserted, "Local direct output task already registered");
}

bool LocalDeviceOutputQueueManager::isDirectOutputTask(
    std::string_view taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  return directOutputTasks_.contains(std::string(taskId));
}

void LocalDeviceOutputQueueManager::initializeTask(
    std::shared_ptr<exec::Task> task,
    int numDestinations,
    int numDrivers) {
  const auto taskId = task->taskId();
  VELOX_CHECK(
      isDirectOutputTask(taskId),
      "Cannot initialize non-root local device output task: {}",
      taskId);
  auto queue = std::make_shared<TaskQueue>(task, numDestinations, numDrivers);
  {
    std::scoped_lock lock(mutex_, accountingMutex_);
    const auto [it, inserted] = queues_.emplace(taskId, std::move(queue));
    VELOX_CHECK(
        inserted, "Local device output task already exists: {}", taskId);
    recomputeAggregateBudgetLocked();
  }
}

bool LocalDeviceOutputQueueManager::updateNumDriversIfExists(
    std::string_view taskId,
    uint32_t newNumDrivers) {
  auto queue = getQueueIfExists(taskId);
  if (!queue) {
    return false;
  }
  std::lock_guard<std::mutex> lock(queue->mutex);
  queue->numDrivers = newNumDrivers;
  return true;
}

void LocalDeviceOutputQueueManager::enqueue(
    std::string_view taskId,
    int destination,
    RowVectorPtr data) {
  VELOX_CHECK_NOT_NULL(data);
  auto queue = getQueue(taskId);
  const auto bytes = data->estimateFlatSize();
  const auto rows = static_cast<uint64_t>(data->size());
  std::lock_guard<std::mutex> accountingLock(accountingMutex_);
  std::lock_guard<std::mutex> lock(queue->mutex);
  VELOX_CHECK(!queue->terminated, "Local device output task is terminated");
  VELOX_CHECK(queue->task->isRunning(), "Task is terminated, cannot add data");
  VELOX_CHECK_GE(destination, 0);
  VELOX_CHECK_LT(destination, queue->queues.size());
  const auto sequence = queue->nextSequences[destination]++;
  queue->queuedBytes += bytes;
  queue->peakQueuedBytes = std::max(queue->peakQueuedBytes, queue->queuedBytes);
  queue->totalEnqueuedBytes += bytes;
  queue->totalEnqueuedRows += rows;
  ++queue->enqueueCalls;
  if (queue->enqueueCalls == 1) {
    LOG(WARNING) << "[LOCAL_DEVICE_OUTPUT_FIRST_ENQUEUE] task=" << taskId
                 << " rows=" << rows << " bytes=" << bytes;
  }
  aggregateQueuedBytes_ += bytes;
  aggregatePeakQueuedBytes_ =
      std::max(aggregatePeakQueuedBytes_, aggregateQueuedBytes_);
  queue->queues[destination].push_back(
      Entry{sequence, bytes, rows, std::move(data)});
}

bool LocalDeviceOutputQueueManager::checkBlocked(
    std::string_view taskId,
    ContinueFuture* future) {
  auto queue = getQueue(taskId);
  std::lock_guard<std::mutex> accountingLock(accountingMutex_);
  std::lock_guard<std::mutex> lock(queue->mutex);
  const bool queueFull = queue->queuedBytes >= queue->maxBytes;
  const bool executorFull =
      aggregateMaxBytes_ > 0 && aggregateQueuedBytes_ >= aggregateMaxBytes_;
  if ((!queueFull && !executorFull) || future == nullptr) {
    return false;
  }
  ++queue->blockedCalls;
  queue->localBudgetBlockedCalls += queueFull ? 1 : 0;
  queue->aggregateBudgetBlockedCalls += executorFull ? 1 : 0;
  aggregatePromises_.emplace_back(
      "LocalDeviceOutputQueueManager::checkBlocked");
  *future = aggregatePromises_.back().getSemiFuture();
  return true;
}

void LocalDeviceOutputQueueManager::noMoreData(std::string_view taskId) {
  auto queue = getQueue(taskId);
  std::lock_guard<std::mutex> lock(queue->mutex);
  VELOX_CHECK_LT(queue->numFinished, queue->numDrivers);
  ++queue->numFinished;
  LOG(WARNING) << "[LOCAL_DEVICE_OUTPUT_DRIVER_END] task=" << taskId
               << " finished=" << queue->numFinished
               << " drivers=" << queue->numDrivers
               << " queuedBytes=" << queue->queuedBytes;
}

LocalDeviceOutputQueueManager::FetchResult
LocalDeviceOutputQueueManager::tryGetData(
    std::string_view taskId,
    int destination,
    int64_t sequence) {
  auto queue = getQueue(taskId);
  std::vector<ContinuePromise> promises;
  FetchResult result;
  {
    std::lock_guard<std::mutex> accountingLock(accountingMutex_);
    std::lock_guard<std::mutex> lock(queue->mutex);
    VELOX_CHECK_GE(destination, 0);
    VELOX_CHECK_LT(destination, queue->queues.size());
    auto& destinationQueue = queue->queues[destination];
    if (!destinationQueue.empty()) {
      auto& front = destinationQueue.front();
      VELOX_CHECK_EQ(
          front.sequence,
          sequence,
          "Unexpected local device output sequence for task {}",
          taskId);
      result.ready = true;
      result.sequence = front.sequence;
      result.data = std::move(front.data);
      VELOX_CHECK_GE(queue->queuedBytes, front.bytes);
      VELOX_CHECK_GE(aggregateQueuedBytes_, front.bytes);
      queue->queuedBytes -= front.bytes;
      queue->totalDequeuedBytes += front.bytes;
      ++queue->dequeueCalls;
      aggregateQueuedBytes_ -= front.bytes;
      destinationQueue.pop_front();
      fulfillAggregateWaitersIfReady(*queue, promises);
    } else if (queue->terminated || queue->numFinished == queue->numDrivers) {
      result.ready = true;
      result.atEnd = true;
      result.sequence = sequence;
    }
  }
  fulfill(std::move(promises));
  return result;
}

void LocalDeviceOutputQueueManager::deleteResults(
    std::string_view taskId,
    int destination) {
  auto queue = getQueueIfExists(taskId);
  if (!queue) {
    return;
  }
  std::vector<ContinuePromise> promises;
  std::shared_ptr<exec::Task> taskToMarkConsumed;
  {
    std::lock_guard<std::mutex> accountingLock(accountingMutex_);
    std::lock_guard<std::mutex> lock(queue->mutex);
    VELOX_CHECK_GE(destination, 0);
    VELOX_CHECK_LT(destination, queue->queues.size());
    for (const auto& entry : queue->queues[destination]) {
      VELOX_CHECK_GE(queue->queuedBytes, entry.bytes);
      VELOX_CHECK_GE(aggregateQueuedBytes_, entry.bytes);
      queue->queuedBytes -= entry.bytes;
      aggregateQueuedBytes_ -= entry.bytes;
    }
    queue->queues[destination].clear();
    queue->destinationDeleted[destination] = true;
    if (!queue->outputConsumed &&
        std::all_of(
            queue->destinationDeleted.begin(),
            queue->destinationDeleted.end(),
            [](bool deleted) { return deleted; })) {
      queue->outputConsumed = true;
      taskToMarkConsumed = queue->task;
    }
    fulfillAggregateWaitersIfReady(*queue, promises);
  }
  fulfill(std::move(promises));
  // Do not call into Task while holding queue/accounting locks.  Completing
  // the task realizes lifecycle futures and can synchronously run callbacks
  // that tear down the same output queue.
  if (taskToMarkConsumed != nullptr) {
    taskToMarkConsumed->setAllOutputConsumed();
  }
}

void LocalDeviceOutputQueueManager::removeTask(std::string_view taskId) {
  std::shared_ptr<TaskQueue> queue;
  std::vector<ContinuePromise> promises;
  {
    std::scoped_lock managerLock(mutex_, accountingMutex_);
    directOutputTasks_.erase(std::string(taskId));
    auto it = queues_.find(std::string(taskId));
    if (it == queues_.end()) {
      return;
    }
    queue = std::move(it->second);
    queues_.erase(it);
    std::lock_guard<std::mutex> lock(queue->mutex);
    queue->terminated = true;
    VELOX_CHECK_GE(aggregateQueuedBytes_, queue->queuedBytes);
    aggregateQueuedBytes_ -= queue->queuedBytes;
    LOG(WARNING) << "[LOCAL_DEVICE_OUTPUT_QUEUE_METRICS] task=" << taskId
                 << " enqueueCalls=" << queue->enqueueCalls
                 << " dequeueCalls=" << queue->dequeueCalls
                 << " blockedCalls=" << queue->blockedCalls
                 << " localBudgetBlockedCalls="
                 << queue->localBudgetBlockedCalls
                 << " aggregateBudgetBlockedCalls="
                 << queue->aggregateBudgetBlockedCalls
                 << " rows=" << queue->totalEnqueuedRows
                 << " enqueuedBytes=" << queue->totalEnqueuedBytes
                 << " dequeuedBytes=" << queue->totalDequeuedBytes
                 << " remainingBytes=" << queue->queuedBytes
                 << " peakQueueBytes=" << queue->peakQueuedBytes
                 << " aggregatePeakBytes=" << aggregatePeakQueuedBytes_
                 << " aggregateRemainingBytes=" << aggregateQueuedBytes_
                 << " aggregateMaxBytes=" << aggregateMaxBytes_;
    queue->queuedBytes = 0;
    for (auto& destination : queue->queues) {
      destination.clear();
    }
    recomputeAggregateBudgetLocked();
    if (aggregateMaxBytes_ == 0 ||
        aggregateQueuedBytes_ < aggregateContinueBytes_) {
      promises.swap(aggregatePromises_);
    }
  }
  fulfill(std::move(promises));
}

void LocalDeviceOutputQueueManager::recomputeAggregateBudgetLocked() {
  aggregateMaxBytes_ = 0;
  for (const auto& [_, queue] : queues_) {
    aggregateMaxBytes_ = aggregateMaxBytes_ == 0
        ? queue->maxBytes
        : std::min(aggregateMaxBytes_, queue->maxBytes);
  }
  aggregateContinueBytes_ = aggregateMaxBytes_ * kContinuePct / 100;
}

std::shared_ptr<LocalDeviceOutputQueueManager::TaskQueue>
LocalDeviceOutputQueueManager::getQueueIfExists(std::string_view taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = queues_.find(std::string(taskId));
  return it == queues_.end() ? nullptr : it->second;
}

std::shared_ptr<LocalDeviceOutputQueueManager::TaskQueue>
LocalDeviceOutputQueueManager::getQueue(std::string_view taskId) {
  auto queue = getQueueIfExists(taskId);
  VELOX_CHECK_NOT_NULL(
      queue, "Local device output queue not found for task {}", taskId);
  return queue;
}

void LocalDeviceOutputQueueManager::fulfillAggregateWaitersIfReady(
    const TaskQueue& changedQueue,
    std::vector<ContinuePromise>& promises) {
  // accountingMutex_ and changedQueue.mutex are held by the caller. Waking all
  // producers is deliberate: futures are advisory and each producer
  // revalidates both its local queue and executor-wide budget on retry.
  if (aggregateQueuedBytes_ < aggregateContinueBytes_ &&
      changedQueue.queuedBytes < changedQueue.continueBytes) {
    promises.swap(aggregatePromises_);
  }
}

} // namespace facebook::velox::ucx_exchange
