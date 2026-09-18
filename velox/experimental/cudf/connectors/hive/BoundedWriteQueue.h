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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

// A single submitter admits a bounded number of eagerly submitted IO requests.
// Returned futures may be waited from another thread. Neither the queue nor
// a returned future may abandon an in-flight read of the caller's buffer.
class BoundedWriteQueue {
 public:
  struct Stats {
    size_t capacityWaits{0};
    size_t readyBehindBlockedFront{0};
    size_t readyBehindRequests{0};
    size_t outOfOrderRetired{0};
    size_t peakPending{0};
    uint64_t capacityWaitNs{0};
  };

  explicit BoundedWriteQueue(size_t limit, bool retireReady = false)
      : limit_(limit), retireReady_(retireReady) {
    if (!limit) {
      throw std::invalid_argument("Write queue capacity must be positive");
    }
  }

  ~BoundedWriteQueue() {
    for (const auto& pending : pending_) {
      pending.completion.wait();
    }
  }

  template <typename Submit>
  std::future<void> submit(Submit&& submitIo, size_t expectedBytes) {
    if (pending_.size() >= limit_ && !ready(pending_.front())) {
      size_t behind = 0;
      for (size_t i = 1; i < pending_.size(); ++i) {
        behind += ready(pending_[i]);
      }
      stats_.readyBehindBlockedFront += behind > 0;
      stats_.readyBehindRequests += behind;
    }
    if (retireReady_) {
      // Completion order need not equal submission order. File offsets were
      // already assigned by the caller. Recycle completed requests without
      // adding slots or abandoning any buffer still being read by IO.
      for (auto it = pending_.begin(); it != pending_.end();) {
        if (ready(*it)) {
          stats_.outOfOrderRetired += it != pending_.begin();
          retire(*it);
          it = pending_.erase(it);
        } else {
          ++it;
        }
      }
    }
    while (!pending_.empty() &&
           (pending_.size() >= limit_ || ready(pending_.front()))) {
      if (pending_.size() >= limit_ && !ready(pending_.front())) {
        ++stats_.capacityWaits;
        const auto start = std::chrono::steady_clock::now();
        retireFront();
        stats_.capacityWaitNs +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
      } else {
        retireFront();
      }
    }
    if (error_) {
      drain();
    }
    // Allocate the lifetime guard before starting IO. If queue/future
    // bookkeeping throws afterward, guard destruction still drains the IO.
    auto guard = std::make_unique<CompletionGuard>();
    guard->completion = std::forward<Submit>(submitIo)().share();
    if (!guard->completion.valid()) {
      throw std::runtime_error("IO submission returned an invalid future");
    }
    pending_.push_back({guard->completion, expectedBytes});
    stats_.peakPending = std::max(stats_.peakPending, pending_.size());
    return std::async(
        std::launch::deferred, [guard = std::move(guard), expectedBytes] {
          if (guard->completion.get() != expectedBytes) {
            throw std::runtime_error("Short asynchronous file write");
          }
        });
  }

  void drain() {
    // Wait for *every* request before propagating any failure. A failed first
    // write must not leave later tasks accessing soon-to-be-freed GPU data.
    for (const auto& pending : pending_) {
      pending.completion.wait();
    }
    while (!pending_.empty()) {
      retireFront();
    }
    if (error_) {
      std::rethrow_exception(error_);
    }
  }

  const Stats& stats() const {
    return stats_;
  }

 private:
  struct CompletionGuard {
    std::shared_future<size_t> completion;
    ~CompletionGuard() {
      if (completion.valid()) {
        completion.wait();
      }
    }
  };
  struct Request {
    std::shared_future<size_t> completion;
    size_t expectedBytes;
  };
  static bool ready(const Request& request) {
    return request.completion.wait_for(std::chrono::seconds(0)) ==
        std::future_status::ready;
  }
  void retire(const Request& request) {
    try {
      if (request.completion.get() != request.expectedBytes) {
        throw std::runtime_error("Short asynchronous file write");
      }
    } catch (...) {
      if (!error_) {
        error_ = std::current_exception();
      }
    }
  }
  void retireFront() {
    retire(pending_.front());
    pending_.pop_front();
  }
  const size_t limit_;
  const bool retireReady_;
  std::deque<Request> pending_;
  std::exception_ptr error_;
  Stats stats_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
