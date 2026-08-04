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

#include <list>
#include <memory>
#include <mutex>

template <typename T>
class WorkQueue {
 public:
  WorkQueue() = default;
  ~WorkQueue() = default;

  // non-copyable
  WorkQueue(const WorkQueue&) = delete;
  WorkQueue& operator=(const WorkQueue&) = delete;

  // Push never blocks. It returns false after terminal shutdown has closed the
  // queue, preventing a check-then-push race with closeAndDrain().
  bool push(std::shared_ptr<T> item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    queue_.emplace_back(std::move(item));
    return true;
  }

  // pop never blocks; returns nullptr if empty
  std::shared_ptr<T> pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return nullptr;
    }

    auto item = std::move(queue_.front());
    queue_.pop_front();
    return item;
  }

  // Remove all copies of an element from the queue. Work items can be queued
  // repeatedly, so removing only the first leaves a hidden owning reference.
  bool erase(std::shared_ptr<T> item) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto oldSize = queue_.size();
    queue_.remove(item);
    return queue_.size() != oldSize;
  }

  // Permanently close the queue and remove every queued reference. Swap under
  // the mutex and release shared_ptrs after unlocking so item destruction
  // cannot re-enter this queue while its mutex is held.
  void closeAndDrain() {
    std::list<std::shared_ptr<T>> items;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
      items.swap(queue_);
    }
  }

  // helpers
  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::list<std::shared_ptr<T>> queue_;
  bool closed_{false};
};
