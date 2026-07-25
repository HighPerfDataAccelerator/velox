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

#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"

#include "velox/common/base/Exceptions.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <algorithm>
#include <deque>
#include <future>
#include <mutex>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

constexpr uint64_t kReadRangeBytes = 16ULL << 20;

struct SplitEntry {
  std::string key;
  std::vector<SplitPrefetchFile> files;
  uint64_t bytes{0};
  bool scheduled{false};
  std::shared_ptr<std::promise<std::shared_ptr<SplitPrefetchResult>>> promise{
      std::make_shared<
          std::promise<std::shared_ptr<SplitPrefetchResult>>>()};
  std::shared_future<std::shared_ptr<SplitPrefetchResult>> future{
      promise->get_future().share()};
};

class QueryPrefetchState
    : public std::enable_shared_from_this<QueryPrefetchState> {
 public:
  void registerSplit(
      const std::string& splitKey,
      std::vector<SplitPrefetchFile> files) {
    auto entry = std::make_shared<SplitEntry>();
    entry->key = splitKey;
    entry->files = std::move(files);
    for (const auto& file : entry->files) {
      entry->bytes += file.size;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (entries_.count(splitKey) != 0) {
        return;
      }
      entries_.emplace(splitKey, entry);
      order_.push_back(entry);
    }
    pump();
  }

  bool contains(const std::string& splitKey) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.count(splitKey) != 0;
  }

  void initialize(
      SplitPrefetchReadFactoryBuilder factoryBuilder,
      std::shared_ptr<ExecutorReadBroker> broker,
      uint32_t splitConcurrency,
      uint64_t maxReadyBytes) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (initialized_) {
        return;
      }
      VELOX_CHECK_NOT_NULL(broker);
      readFactory_ = factoryBuilder();
      VELOX_CHECK(
          static_cast<bool>(readFactory_),
          "Split prefetch factory builder returned an empty factory");
      broker_ = std::move(broker);
      concurrency_ = std::max<uint32_t>(1, splitConcurrency);
      maxReadyBytes_ = std::max<uint64_t>(1, maxReadyBytes);
      splitExecutor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
          concurrency_);
      initialized_ = true;
    }
    pump();
  }

  std::shared_ptr<SplitPrefetchResult> take(const std::string& splitKey) {
    std::shared_ptr<SplitEntry> entry;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = entries_.find(splitKey);
      if (it == entries_.end()) {
        return nullptr;
      }
      entry = it->second;
    }
    auto result = entry->future.get();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      entries_.erase(splitKey);
    }
    return result;
  }

 private:
  void pump() {
    std::vector<std::shared_ptr<SplitEntry>> jobs;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!initialized_) {
        return;
      }
      for (const auto& entry : order_) {
        if (active_ >= concurrency_) {
          break;
        }
        if (entry->scheduled) {
          continue;
        }
        const bool fits =
            reservedBytes_ <= maxReadyBytes_ &&
            entry->bytes <= maxReadyBytes_ - reservedBytes_;
        if (!fits && (active_ > 0 || reservedBytes_ > 0)) {
          continue;
        }
        entry->scheduled = true;
        ++active_;
        reservedBytes_ += entry->bytes;
        jobs.push_back(entry);
      }
      // Once a split is scheduled, entries_ and the worker closure provide
      // its lifetime. Keeping it in order_ would also keep the shared_future
      // (and therefore completed pinned buffers) alive after take().
      std::erase_if(
          order_,
          [](const auto& entry) { return entry->scheduled; });
    }
    for (const auto& entry : jobs) {
      splitExecutor_->add(
          [self = shared_from_this(), entry]() { self->run(entry); });
    }
  }

  void run(const std::shared_ptr<SplitEntry>& entry) {
    try {
      auto result = std::make_shared<SplitPrefetchResult>();
      result->reservedBytes = entry->bytes;
      result->buffers.reserve(entry->files.size());

      std::vector<std::future<void>> pending;
      pending.reserve(entry->files.size());

      for (const auto& file : entry->files) {
        auto buffer = std::make_shared<PinnedHostBuffer>(file.size);
        std::vector<PrefetchRange> ranges;
        ranges.reserve((file.size + kReadRangeBytes - 1) / kReadRangeBytes);
        for (uint64_t offset = 0; offset < file.size;
             offset += kReadRangeBytes) {
          ranges.push_back(
              {offset,
               std::min<uint64_t>(kReadRangeBytes, file.size - offset),
               offset});
        }
        auto future = broker_->readPrepared(
            [factory = readFactory_, path = file.path, size = file.size]() {
              return factory(path, size);
            },
            file.size,
            std::move(ranges),
            buffer);
        result->buffers.push_back(std::move(buffer));
        pending.push_back(std::move(future));
      }

      std::exception_ptr failure;
      for (auto& request : pending) {
        try {
          request.get();
        } catch (...) {
          if (!failure) {
            failure = std::current_exception();
          }
        }
      }
      if (failure) {
        std::rethrow_exception(failure);
      }
      std::weak_ptr<QueryPrefetchState> weakSelf = shared_from_this();
      result->release = [weakSelf](uint64_t bytes) {
        if (const auto self = weakSelf.lock()) {
          {
            std::lock_guard<std::mutex> lock(self->mutex_);
            self->reservedBytes_ -= bytes;
          }
          self->pump();
        }
      };
      entry->promise->set_value(std::move(result));
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        reservedBytes_ -= entry->bytes;
      }
      entry->promise->set_exception(std::current_exception());
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      --active_;
    }
    pump();
  }

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<SplitEntry>> entries_;
  std::deque<std::shared_ptr<SplitEntry>> order_;
  SplitPrefetchReadFactory readFactory_;
  std::shared_ptr<ExecutorReadBroker> broker_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> splitExecutor_;
  uint32_t concurrency_{0};
  uint32_t active_{0};
  uint64_t maxReadyBytes_{0};
  uint64_t reservedBytes_{0};
  bool initialized_{false};
};

struct ExecutorState {
  std::mutex mutex;
  std::unordered_map<std::string, std::shared_ptr<QueryPrefetchState>> queries;
};

std::mutex& registryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<folly::Executor*, std::shared_ptr<ExecutorState>>&
registry() {
  static std::unordered_map<
      folly::Executor*,
      std::shared_ptr<ExecutorState>>
      states;
  return states;
}

std::shared_ptr<ExecutorState> getExecutorState(folly::Executor* executor) {
  VELOX_CHECK_NOT_NULL(executor);
  std::lock_guard<std::mutex> lock(registryMutex());
  auto& state = registry()[executor];
  if (!state) {
    state = std::make_shared<ExecutorState>();
  }
  return state;
}

std::shared_ptr<QueryPrefetchState> getQueryState(
    folly::Executor* executor,
    const std::string& queryId,
    bool create) {
  const auto executorState = getExecutorState(executor);
  std::lock_guard<std::mutex> lock(executorState->mutex);
  auto it = executorState->queries.find(queryId);
  if (it != executorState->queries.end()) {
    return it->second;
  }
  if (!create) {
    return nullptr;
  }
  auto state = std::make_shared<QueryPrefetchState>();
  executorState->queries.emplace(queryId, state);
  return state;
}

std::shared_ptr<QueryPrefetchState> findQueryStateAcrossExecutors(
    const std::string& queryId,
    const std::string* splitKey = nullptr) {
  std::lock_guard<std::mutex> registryLock(registryMutex());
  for (const auto& entry : registry()) {
    const auto& executorState = entry.second;
    std::lock_guard<std::mutex> executorLock(executorState->mutex);
    const auto it = executorState->queries.find(queryId);
    if (it == executorState->queries.end()) {
      continue;
    }
    if (splitKey == nullptr || it->second->contains(*splitKey)) {
      return it->second;
    }
  }
  return nullptr;
}

} // namespace

SplitPrefetchResult::~SplitPrefetchResult() {
  if (release) {
    release(reservedBytes);
  }
}

void ExecutorSplitPrefetch::registerSplit(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey,
    std::vector<SplitPrefetchFile> files) {
  if (!executor || files.empty()) {
    return;
  }
  getQueryState(executor, queryId, true)
      ->registerSplit(splitKey, std::move(files));
}

bool ExecutorSplitPrefetch::contains(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey) {
  if (!executor) {
    return false;
  }
  auto state = getQueryState(executor, queryId, false);
  if (!state || !state->contains(splitKey)) {
    state = findQueryStateAcrossExecutors(queryId, &splitKey);
  }
  return state && state->contains(splitKey);
}

void ExecutorSplitPrefetch::initialize(
    folly::Executor* executor,
    const std::string& queryId,
    SplitPrefetchReadFactoryBuilder factoryBuilder,
    std::shared_ptr<ExecutorReadBroker> broker,
    uint32_t splitConcurrency,
    uint64_t maxReadyBytes) {
  auto state = getQueryState(executor, queryId, false);
  if (!state) {
    state = findQueryStateAcrossExecutors(queryId);
  }
  if (!state) {
    state = getQueryState(executor, queryId, true);
  }
  state->initialize(
      std::move(factoryBuilder),
      std::move(broker),
      splitConcurrency,
      maxReadyBytes);
}

std::shared_ptr<SplitPrefetchResult> ExecutorSplitPrefetch::take(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey) {
  if (!executor) {
    return nullptr;
  }
  auto state = getQueryState(executor, queryId, false);
  if (!state || !state->contains(splitKey)) {
    state = findQueryStateAcrossExecutors(queryId, &splitKey);
  }
  return state ? state->take(splitKey) : nullptr;
}

void ExecutorSplitPrefetch::eraseQuery(
    folly::Executor* executor,
    const std::string& queryId) {
  if (!executor) {
    return;
  }
  std::vector<std::shared_ptr<QueryPrefetchState>> queryStates;
  {
    std::lock_guard<std::mutex> registryLock(registryMutex());
    for (const auto& entry : registry()) {
      const auto& executorState = entry.second;
      std::lock_guard<std::mutex> executorLock(executorState->mutex);
      const auto it = executorState->queries.find(queryId);
      if (it == executorState->queries.end()) {
        continue;
      }
      queryStates.push_back(std::move(it->second));
      executorState->queries.erase(it);
    }
  }
  // Destroy outside registry locks. QueryPrefetchState destruction joins its
  // scheduler workers, which may still be finishing broker-backed reads.
  queryStates.clear();
}

void ExecutorSplitPrefetch::erase(folly::Executor* executor) {
  if (!executor) {
    return;
  }
  std::shared_ptr<ExecutorState> state;
  {
    std::lock_guard<std::mutex> lock(registryMutex());
    const auto it = registry().find(executor);
    if (it == registry().end()) {
      return;
    }
    state = std::move(it->second);
    registry().erase(it);
  }
  state.reset();
}

} // namespace facebook::velox::cudf_velox::connector::hive
