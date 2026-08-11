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

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/exec/Task.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::ucx_exchange {

/// Same-process, owning queue for terminal GPU output.
///
/// Internal UCX exchange must publish independently-owned packed columns, but
/// a terminal cache sink lives in the same executor process as its producer.
/// Packing that root output performs an unnecessary full-table D2D copy. This
/// queue transfers the owning RowVector directly while retaining the ordinary
/// output-buffer byte backpressure contract.
class LocalDeviceOutputQueueManager {
 public:
  inline static const std::string kEnabledConfig =
      "cudf.cache_root_direct_device_output";
  inline static const std::string kMaxBytesConfig =
      "cudf.cache_root_output_buffer_bytes";

  struct FetchResult {
    bool ready{false};
    bool atEnd{false};
    int64_t sequence{0};
    RowVectorPtr data;
  };

  static std::shared_ptr<LocalDeviceOutputQueueManager> getInstanceRef();

  /// Marks the one terminal MPP output task whose owning device vectors are
  /// consumed in the same process.  The coordinator knows the root fragment;
  /// the operator cannot infer it from a plan-node suffix (the root is not
  /// necessarily fragment 0).
  void registerDirectOutputTask(std::string taskId);

  bool isDirectOutputTask(std::string_view taskId);

  void initializeTask(
      std::shared_ptr<exec::Task> task,
      int numDestinations,
      int numDrivers);

  bool updateNumDriversIfExists(
      std::string_view taskId,
      uint32_t newNumDrivers);

  void enqueue(std::string_view taskId, int destination, RowVectorPtr data);

  bool checkBlocked(std::string_view taskId, ContinueFuture* future);

  void noMoreData(std::string_view taskId);

  FetchResult
  tryGetData(std::string_view taskId, int destination, int64_t sequence);

  void deleteResults(std::string_view taskId, int destination);

  void removeTask(std::string_view taskId);

 private:
  struct Entry {
    int64_t sequence;
    uint64_t bytes;
    uint64_t rows;
    RowVectorPtr data;
  };

  struct TaskQueue {
    explicit TaskQueue(
        std::shared_ptr<exec::Task> task,
        int numDestinations,
        int numDrivers);

    std::shared_ptr<exec::Task> task;
    std::mutex mutex;
    std::vector<std::deque<Entry>> queues;
    // A Velox task with PartitionedOutput does not become kFinished when its
    // drivers exit.  Every destination must first be deleted/acknowledged by
    // the consumer, at which point setAllOutputConsumed() completes the task.
    std::vector<bool> destinationDeleted;
    std::vector<int64_t> nextSequences;
    uint32_t numDrivers{0};
    uint32_t numFinished{0};
    uint64_t maxBytes{0};
    uint64_t continueBytes{0};
    uint64_t queuedBytes{0};
    uint64_t peakQueuedBytes{0};
    uint64_t totalEnqueuedBytes{0};
    uint64_t totalDequeuedBytes{0};
    uint64_t totalEnqueuedRows{0};
    uint64_t enqueueCalls{0};
    uint64_t dequeueCalls{0};
    uint64_t blockedCalls{0};
    uint64_t localBudgetBlockedCalls{0};
    uint64_t aggregateBudgetBlockedCalls{0};
    bool outputConsumed{false};
    bool terminated{false};
  };

  std::shared_ptr<TaskQueue> getQueue(std::string_view taskId);
  std::shared_ptr<TaskQueue> getQueueIfExists(std::string_view taskId);

  void fulfillAggregateWaitersIfReady(
      const TaskQueue& changedQueue,
      std::vector<ContinuePromise>& promises);

  // mutex_ and accountingMutex_ must both be held.
  void recomputeAggregateBudgetLocked();

  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<TaskQueue>> queues_;
  std::unordered_set<std::string> directOutputTasks_;

  // kMaxBytesConfig is an executor-wide device-output allowance. A root MPP
  // fragment can have several local replicas while Spark exposes only one
  // sequential consumer for them. Applying the allowance independently to
  // every replica lets inactive queues pin N * maxBytes on one GPU without
  // increasing writer parallelism. Account all local root queues against one
  // shared budget instead.
  std::mutex accountingMutex_;
  uint64_t aggregateMaxBytes_{0};
  uint64_t aggregateContinueBytes_{0};
  uint64_t aggregateQueuedBytes_{0};
  uint64_t aggregatePeakQueuedBytes_{0};
  std::vector<ContinuePromise> aggregatePromises_;
};

} // namespace facebook::velox::ucx_exchange
