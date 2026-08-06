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

#include "velox/experimental/cudf/connectors/hive/ExecutorReadBroker.h"

#include "velox/common/base/Exceptions.h"

#include <folly/futures/Future.h>

#include <algorithm>
#include <atomic>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

std::mutex& brokersMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<folly::Executor*, std::shared_ptr<ExecutorReadBroker>>&
brokers() {
  static std::
      unordered_map<folly::Executor*, std::shared_ptr<ExecutorReadBroker>>
          instances;
  return instances;
}

} // namespace

ExecutorReadReservation::~ExecutorReadReservation() {
  if (broker_) {
    broker_->release(bytes_);
  }
}

std::shared_ptr<ExecutorReadBroker> ExecutorReadBroker::get(
    folly::Executor* executor,
    uint64_t maxInFlightBytes,
    uint32_t readThreads) {
  VELOX_CHECK_NOT_NULL(executor, "ExecutorReadBroker requires an IO executor");
  std::lock_guard<std::mutex> lock(brokersMutex());
  auto& entry = brokers()[executor];
  if (entry) {
    return entry;
  }
  auto broker = std::shared_ptr<ExecutorReadBroker>(
      new ExecutorReadBroker(executor, maxInFlightBytes, readThreads));
  entry = broker;
  return broker;
}

void ExecutorReadBroker::erase(folly::Executor* executor) {
  if (!executor) {
    return;
  }
  std::shared_ptr<ExecutorReadBroker> broker;
  {
    std::lock_guard<std::mutex> lock(brokersMutex());
    const auto it = brokers().find(executor);
    if (it == brokers().end()) {
      return;
    }
    broker = std::move(it->second);
    brokers().erase(it);
  }
  // Destroy outside the registry lock. CPUThreadPoolExecutor destruction joins
  // its workers and can therefore block until outstanding reads finish.
  broker.reset();
}

void ExecutorReadBroker::acquire(uint64_t bytes) {
  std::unique_lock<std::mutex> lock(mutex_);
  available_.wait(lock, [&] {
    return admittedBytes_ == 0 ||
        bytes <=
        maxInFlightBytes_ - std::min(admittedBytes_, maxInFlightBytes_);
  });
  admittedBytes_ += bytes;
}

void ExecutorReadBroker::release(uint64_t bytes) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    admittedBytes_ -= bytes;
  }
  available_.notify_all();
}

std::shared_ptr<ExecutorReadReservation> ExecutorReadBroker::reserve(
    uint64_t bytes) {
  acquire(bytes);
  try {
    return std::shared_ptr<ExecutorReadReservation>(
        new ExecutorReadReservation(shared_from_this(), bytes));
  } catch (...) {
    release(bytes);
    throw;
  }
}

std::future<void> ExecutorReadBroker::read(
    PrefetchReadFunction readFunction,
    uint64_t sourceSize,
    std::vector<PrefetchRange> ranges,
    std::shared_ptr<PinnedHostBuffer> destination,
    std::shared_ptr<ExecutorReadReservation> reservation) {
  VELOX_CHECK(
      static_cast<bool>(readFunction),
      "ExecutorReadBroker requires a range-read function");
  VELOX_CHECK_NOT_NULL(destination);
  for (const auto& range : ranges) {
    VELOX_CHECK_LE(range.bufferOffset + range.size, destination->size());
    VELOX_CHECK_LE(range.fileOffset + range.size, sourceSize);
  }

  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  if (ranges.empty()) {
    promise->set_value();
    return future;
  }

  auto remaining = std::make_shared<std::atomic<size_t>>(ranges.size());
  auto failure = std::make_shared<std::exception_ptr>();
  auto failureMutex = std::make_shared<std::mutex>();
  for (const auto range : ranges) {
    readExecutor_->add([self = shared_from_this(),
                        readFunction,
                        destination,
                        range,
                        remaining,
                        failure,
                        failureMutex,
                        promise,
                        reservation]() {
      if (!reservation) {
        self->acquire(range.size);
      }
      try {
        readFunction(
            range.fileOffset,
            range.size,
            destination->data() + range.bufferOffset);
      } catch (...) {
        std::lock_guard<std::mutex> lock(*failureMutex);
        if (!*failure) {
          *failure = std::current_exception();
        }
      }
      if (!reservation) {
        self->release(range.size);
      }
      if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (*failure) {
          promise->set_exception(*failure);
        } else {
          promise->set_value();
        }
      }
    });
  }
  return future;
}

std::future<void> ExecutorReadBroker::readPrepared(
    PrefetchReadFactory readFactory,
    uint64_t sourceSize,
    std::vector<PrefetchRange> ranges,
    std::shared_ptr<PinnedHostBuffer> destination,
    std::shared_ptr<ExecutorReadReservation> reservation) {
  VELOX_CHECK(
      static_cast<bool>(readFactory),
      "ExecutorReadBroker requires a range-read factory");
  VELOX_CHECK_NOT_NULL(destination);
  for (const auto& range : ranges) {
    VELOX_CHECK_LE(range.bufferOffset + range.size, destination->size());
    VELOX_CHECK_LE(range.fileOffset + range.size, sourceSize);
  }

  auto promise = std::make_shared<std::promise<void>>();
  auto future = promise->get_future();
  if (ranges.empty()) {
    promise->set_value();
    return future;
  }

  auto remaining = std::make_shared<std::atomic<size_t>>(ranges.size());
  auto failure = std::make_shared<std::exception_ptr>();
  auto failureMutex = std::make_shared<std::mutex>();
  for (const auto range : ranges) {
    readExecutor_->add([self = shared_from_this(),
                        readFactory,
                        destination,
                        range,
                        remaining,
                        failure,
                        failureMutex,
                        promise,
                        reservation]() {
      if (!reservation) {
        self->acquire(range.size);
      }
      try {
        auto readFunction = readFactory();
        VELOX_CHECK(
            static_cast<bool>(readFunction),
            "Range-read factory returned an empty function");
        readFunction(
            range.fileOffset,
            range.size,
            destination->data() + range.bufferOffset);
      } catch (...) {
        std::lock_guard<std::mutex> lock(*failureMutex);
        if (!*failure) {
          *failure = std::current_exception();
        }
      }
      if (!reservation) {
        self->release(range.size);
      }
      if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (*failure) {
          promise->set_exception(*failure);
        } else {
          promise->set_value();
        }
      }
    });
  }
  return future;
}

} // namespace facebook::velox::cudf_velox::connector::hive
