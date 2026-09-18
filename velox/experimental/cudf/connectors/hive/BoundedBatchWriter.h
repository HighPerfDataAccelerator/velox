/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

// A single consumer always drains ready input, never waits for a target batch
// size. Bytes/items include BOTH queued and active inputs. Consumer must finish
// using inputs (including asynchronous device work) before returning/throwing.
// Consequently producer backpressure cannot depend on producing another input.
template <typename Input>
class BoundedBatchWriter {
 public:
  struct Stats {
    uint64_t peakBytes{0};
    uint64_t peakItems{0};
    uint64_t batches{0};
    uint64_t inputs{0};
    uint64_t maxBatchItems{0};
  };
  using Consumer = std::function<void(std::vector<Input>&)>;

  BoundedBatchWriter(uint64_t maxBytes, uint64_t maxItems, Consumer consume)
      : maxBytes_(maxBytes), maxItems_(maxItems), consume_(std::move(consume)) {
    if (!maxBytes_ || !maxItems_ || !consume_) {
      throw std::invalid_argument("Invalid batch writer capacity/consumer");
    }
    worker_ = std::thread([this] { run(); });
  }

  BoundedBatchWriter(const BoundedBatchWriter&) = delete;
  BoundedBatchWriter& operator=(const BoundedBatchWriter&) = delete;

  ~BoundedBatchWriter() {
    // Drain on destruction too: an accepted input may own in-flight work.
    // The original error is delivered through submit/drain, never destructor.
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_all();
    worker_.join();
  }

  void submit(Input input, uint64_t bytes) {
    if (bytes > maxBytes_) {
      throw std::invalid_argument("Oversize batch writer input");
    }
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] {
      return failure_ || stopping_ ||
          (items_ < maxItems_ && bytes <= maxBytes_ - bytes_);
    });
    checkError();
    if (stopping_) {
      throw std::runtime_error("Batch writer is stopping");
    }
    queue_.push_back({std::move(input), bytes});
    bytes_ += bytes;
    ++items_;
    stats_.peakBytes = std::max(stats_.peakBytes, bytes_);
    stats_.peakItems = std::max(stats_.peakItems, items_);
    lock.unlock();
    changed_.notify_all();
  }

  void drain() {
    std::unique_lock lock(mutex_);
    changed_.wait(lock, [&] { return items_ == 0; });
    checkError();
  }

  Stats stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
  }

 private:
  struct Entry {
    Input input;
    uint64_t bytes;
  };

  void checkError() const {
    if (failure_) {
      std::rethrow_exception(failure_);
    }
  }

  void run() noexcept {
    // All allocation/callback failures must release accepted owners and wake
    // blocked submit/drain callers, not terminate a background thread.
    try {
      for (;;) {
        std::vector<Input> batch;
        uint64_t batchBytes = 0;
        {
          std::unique_lock lock(mutex_);
          changed_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
          if (queue_.empty()) {
            return;
          }
          batch.reserve(queue_.size());
          while (!queue_.empty()) {
            batchBytes += queue_.front().bytes;
            batch.push_back(std::move(queue_.front().input));
            queue_.pop_front();
          }
        }
        const auto count = batch.size();
        consume_(batch);
        batch.clear(); // Release device owners/leases BEFORE returning credit.
        {
          std::lock_guard lock(mutex_);
          bytes_ -= batchBytes;
          items_ -= count;
          ++stats_.batches;
          stats_.inputs += count;
          stats_.maxBatchItems =
              std::max<uint64_t>(stats_.maxBatchItems, count);
        }
        changed_.notify_all();
      }
    } catch (...) {
      std::deque<Entry> discarded;
      {
        std::lock_guard lock(mutex_);
        failure_ = std::current_exception();
        discarded.swap(queue_);
      }
      discarded.clear();
      {
        std::lock_guard lock(mutex_);
        bytes_ = 0;
        items_ = 0;
      }
      changed_.notify_all();
    }
  }

  const uint64_t maxBytes_;
  const uint64_t maxItems_;
  const Consumer consume_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<Entry> queue_;
  uint64_t bytes_{0};
  uint64_t items_{0};
  bool stopping_{false};
  std::exception_ptr failure_;
  Stats stats_;
  std::thread worker_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
