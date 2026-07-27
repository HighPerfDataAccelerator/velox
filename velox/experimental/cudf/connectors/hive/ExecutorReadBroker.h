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

#include "velox/experimental/cudf/connectors/hive/PinnedHostBuffer.h"

#include <folly/Executor.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

class ExecutorReadBroker;

/// Holds an executor-wide byte reservation. The reservation is released when
/// the last owner drops it, which lets callers acquire the budget before
/// allocating pinned buffers and retain it for the buffers' full lifetime.
class ExecutorReadReservation {
 public:
  ~ExecutorReadReservation();

 private:
  friend class ExecutorReadBroker;

  ExecutorReadReservation(
      std::shared_ptr<ExecutorReadBroker> broker,
      uint64_t bytes)
      : broker_(std::move(broker)), bytes_(bytes) {}

  std::shared_ptr<ExecutorReadBroker> broker_;
  const uint64_t bytes_;
};

struct PrefetchRange {
  uint64_t fileOffset;
  uint64_t size;
  uint64_t bufferOffset;
};

using PrefetchReadFunction =
    std::function<void(uint64_t, uint64_t, uint8_t*)>;
using PrefetchReadFactory = std::function<PrefetchReadFunction()>;

/// Executor-scoped range-read broker. A broker is shared by all scan tasks
/// using the same VeloxBackend IO executor. Individual file requests submit
/// their ranges together; ranges execute on the shared CPU pool while a byte
/// budget bounds host memory and network work in flight.
class ExecutorReadBroker
    : public std::enable_shared_from_this<ExecutorReadBroker> {
 public:
  static std::shared_ptr<ExecutorReadBroker> get(
      folly::Executor* executor,
      uint64_t maxInFlightBytes,
      uint32_t readThreads);

  /// Removes the broker associated with an IO executor. The caller must do
  /// this before destroying the executor that provides the registry key.
  static void erase(folly::Executor* executor);

  /// Acquires the executor-wide byte budget synchronously. A request larger
  /// than the configured budget is admitted only when it is the sole
  /// reservation, matching the broker's existing oversize policy.
  std::shared_ptr<ExecutorReadReservation> reserve(uint64_t bytes);

  std::future<void> read(
      PrefetchReadFunction readFunction,
      uint64_t sourceSize,
      std::vector<PrefetchRange> ranges,
      std::shared_ptr<PinnedHostBuffer> destination,
      std::shared_ptr<ExecutorReadReservation> reservation = nullptr);

  /// Opens one physical source on a broker worker and reads all of its ranges
  /// into a pinned destination. Keeping preparation and reads in one worker
  /// makes RemoteHandle construction parallel across files without creating
  /// more than one handle per file.
  std::future<void> readPrepared(
      PrefetchReadFactory readFactory,
      uint64_t sourceSize,
      std::vector<PrefetchRange> ranges,
      std::shared_ptr<PinnedHostBuffer> destination,
      std::shared_ptr<ExecutorReadReservation> reservation = nullptr);

 private:
  friend class ExecutorReadReservation;

  ExecutorReadBroker(
      folly::Executor* executor,
      uint64_t maxInFlightBytes,
      uint32_t readThreads)
      : executor_(executor),
        readExecutor_(std::make_unique<folly::CPUThreadPoolExecutor>(
            std::max<uint32_t>(1, readThreads))),
        maxInFlightBytes_(std::max<uint64_t>(1, maxInFlightBytes)) {}

  void acquire(uint64_t bytes);
  void release(uint64_t bytes);

  folly::Executor* const executor_;
  // Do not enqueue range work onto 'executor_': prepareSplit() may be called
  // from that same pool and synchronously waits for this request. A dedicated
  // executor-scoped pool prevents all callers from occupying the workers
  // needed to satisfy their own futures.
  std::unique_ptr<folly::CPUThreadPoolExecutor> readExecutor_;
  const uint64_t maxInFlightBytes_;
  std::mutex mutex_;
  std::condition_variable available_;
  uint64_t admittedBytes_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
