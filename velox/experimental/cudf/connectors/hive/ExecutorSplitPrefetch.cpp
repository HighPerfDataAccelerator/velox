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

#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/CachedBufferedInput.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

constexpr uint64_t kReadRangeBytes = 16ULL << 20;

struct SplitEntry {
  std::string key;
  std::vector<SplitPrefetchFile> files;
  uint64_t bytes{0};
  bool requested{false};
  bool scheduled{false};
  std::shared_ptr<std::promise<std::shared_ptr<SplitPrefetchResult>>> promise{
      std::make_shared<std::promise<std::shared_ptr<SplitPrefetchResult>>>()};
  std::shared_future<std::shared_ptr<SplitPrefetchResult>> future{
      promise->get_future().share()};
};

struct CacheHintEntry {
  std::string key;
  uint64_t bytes{0};
  CacheHintRangeStats rangeStats;
  ExecutorSplitPrefetch::CacheHintLoad load;
  bool requested{false};
  bool scheduled{false};
  bool rangeStatsAccounted{false};
  std::shared_ptr<std::promise<void>> promise{
      std::make_shared<std::promise<void>>()};
  std::shared_future<void> future{promise->get_future().share()};
  std::shared_future<void> firstLoadReady;
};

std::atomic<uint64_t> cacheHintPlannedSplits{0};
std::atomic<uint64_t> cacheHintPlannedBytes{0};
std::atomic<uint64_t> cacheHintPrefetchedSplits{0};
std::atomic<uint64_t> cacheHintPrefetchedBytes{0};
std::atomic<uint64_t> cacheHintFallbacks{0};
std::atomic<uint64_t> cacheHintLogicalRanges{0};
std::atomic<uint64_t> cacheHintLogicalBytes{0};
std::atomic<uint64_t> cacheHintUniqueRanges{0};
std::atomic<uint64_t> cacheHintUniqueBytes{0};
std::atomic<uint64_t> cacheHintOverlapBytes{0};
std::atomic<uint64_t> cacheHintDuplicateSuppressedRanges{0};
std::atomic<uint64_t> cacheHintDuplicateSuppressedBytes{0};
std::atomic<uint64_t> cacheHintConsumedRanges{0};
std::atomic<uint64_t> cacheHintConsumedBytes{0};
std::atomic<uint64_t> cacheHintUnusedRanges{0};
std::atomic<uint64_t> cacheHintUnusedBytes{0};
std::atomic<uint64_t> cachePrefetchPlanHits{0};
std::atomic<uint64_t> cachePrefetchPlanMisses{0};
std::atomic<uint64_t> cachePrefetchPlanEntries{0};
std::atomic<uint64_t> cacheHintScanWaits{0};
std::atomic<uint64_t> cacheHintScanReadyAtTake{0};
std::atomic<uint64_t> cacheHintScanWaitWallNanos{0};
std::atomic<uint64_t> cacheHintSplitPreloadWaits{0};
std::atomic<uint64_t> cacheHintSplitPreloadReadyAtTake{0};
std::atomic<uint64_t> cacheHintSplitPreloadWaitWallNanos{0};
std::atomic<uint64_t> cacheHintRangeReadyRequests{0};
std::atomic<uint64_t> cacheHintRangeReadyReleases{0};
std::atomic<uint64_t> cacheHintFirstLoadGroupWaits{0};
std::atomic<uint64_t> cacheHintFirstLoadGroupReadyAtTake{0};
std::atomic<uint64_t> cacheHintFirstLoadGroupWaitWallNanos{0};

void addRangePlan(const CacheHintRangeStats& stats) {
  cacheHintLogicalRanges.fetch_add(
      stats.logicalRanges, std::memory_order_relaxed);
  cacheHintLogicalBytes.fetch_add(
      stats.logicalBytes, std::memory_order_relaxed);
  cacheHintUniqueRanges.fetch_add(
      stats.uniqueRanges, std::memory_order_relaxed);
  cacheHintUniqueBytes.fetch_add(stats.uniqueBytes, std::memory_order_relaxed);
  cacheHintOverlapBytes.fetch_add(
      stats.overlapBytes, std::memory_order_relaxed);
  cacheHintDuplicateSuppressedRanges.fetch_add(
      stats.duplicateSuppressedRanges, std::memory_order_relaxed);
  cacheHintDuplicateSuppressedBytes.fetch_add(
      stats.duplicateSuppressedBytes, std::memory_order_relaxed);
}

void addConsumedRanges(const CacheHintRangeStats& stats) {
  cacheHintConsumedRanges.fetch_add(
      stats.uniqueRanges, std::memory_order_relaxed);
  cacheHintConsumedBytes.fetch_add(
      stats.uniqueBytes, std::memory_order_relaxed);
}

void addUnusedRanges(const CacheHintRangeStats& stats) {
  cacheHintUnusedRanges.fetch_add(
      stats.uniqueRanges, std::memory_order_relaxed);
  cacheHintUnusedBytes.fetch_add(stats.uniqueBytes, std::memory_order_relaxed);
}

void logCacheHintStats(std::string_view event, std::string_view id = {}) {
  const auto demandStats =
      facebook::velox::dwio::common::cacheHintDemandStats();
  const auto asyncLoadStats =
      facebook::velox::dwio::common::cacheHintAsyncLoadStats();
  const auto pageH2dStats = directCachePageH2dStats();
  const auto registrationStats = cachePageRegistrationStats();
  LOG(WARNING)
      << "CUDF_CACHE_HINT_STATS event=" << event << (id.empty() ? "" : " id=")
      << id << " plannedSplits="
      << cacheHintPlannedSplits.load(std::memory_order_relaxed)
      << " plannedBytes="
      << cacheHintPlannedBytes.load(std::memory_order_relaxed)
      << " prefetchedSplits="
      << cacheHintPrefetchedSplits.load(std::memory_order_relaxed)
      << " prefetchedBytes="
      << cacheHintPrefetchedBytes.load(std::memory_order_relaxed)
      << " fallbacks=" << cacheHintFallbacks.load(std::memory_order_relaxed)
      << " planCacheHits="
      << cachePrefetchPlanHits.load(std::memory_order_relaxed)
      << " planCacheMisses="
      << cachePrefetchPlanMisses.load(std::memory_order_relaxed)
      << " planCacheEntries="
      << cachePrefetchPlanEntries.load(std::memory_order_relaxed)
      << " logicalRanges="
      << cacheHintLogicalRanges.load(std::memory_order_relaxed)
      << " logicalBytes="
      << cacheHintLogicalBytes.load(std::memory_order_relaxed)
      << " uniqueRanges="
      << cacheHintUniqueRanges.load(std::memory_order_relaxed)
      << " uniqueBytes=" << cacheHintUniqueBytes.load(std::memory_order_relaxed)
      << " overlapBytes="
      << cacheHintOverlapBytes.load(std::memory_order_relaxed)
      << " consumedRanges="
      << cacheHintConsumedRanges.load(std::memory_order_relaxed)
      << " consumedBytes="
      << cacheHintConsumedBytes.load(std::memory_order_relaxed)
      << " unusedRanges="
      << cacheHintUnusedRanges.load(std::memory_order_relaxed)
      << " unusedBytes=" << cacheHintUnusedBytes.load(std::memory_order_relaxed)
      << " duplicateSuppressedRanges="
      << cacheHintDuplicateSuppressedRanges.load(std::memory_order_relaxed)
      << " duplicateSuppressedBytes="
      << cacheHintDuplicateSuppressedBytes.load(std::memory_order_relaxed)
      << " demandFirstHitRanges=" << demandStats.firstHitRanges
      << " demandFirstHitBytes=" << demandStats.firstHitBytes
      << " demandMissRanges=" << demandStats.missRanges
      << " demandMissBytes=" << demandStats.missBytes
      << " remoteDuplicateRanges=" << demandStats.remoteDuplicateRanges
      << " remoteDuplicateBytes=" << demandStats.remoteDuplicateBytes
      << " sizeMismatchRanges=" << demandStats.sizeMismatchRanges
      << " sizeMismatchBytes=" << demandStats.sizeMismatchBytes
      << " directCachePageH2dCopies=" << pageH2dStats.copies
      << " directCachePageH2dBytes=" << pageH2dStats.bytes
      << " directCachePageH2dPinnedCopies=" << pageH2dStats.pinnedCopies
      << " directCachePageH2dPinnedBytes=" << pageH2dStats.pinnedBytes
      << " cachePageRegistrationAttempts=" << registrationStats.attempts
      << " cachePageRegistrationSuccesses=" << registrationStats.successes
      << " cachePageRegistrationFailures=" << registrationStats.failures
      << " cachePageRegistrationBudgetRejectedBytes="
      << registrationStats.budgetRejectedBytes
      << " cachePageRegisteredRuns=" << registrationStats.registeredRuns
      << " cachePageRegisteredBytes=" << registrationStats.registeredBytes
      << " cachePageCurrentRegisteredBytes=" << registrationStats.currentBytes
      << " cachePagePeakRegisteredBytes=" << registrationStats.peakBytes
      << " cachePageUnregisteredRuns=" << registrationStats.unregisteredRuns
      << " cachePageUnregisteredBytes=" << registrationStats.unregisteredBytes
      << " cachePagePrewarmAttempts=" << registrationStats.prewarmAttempts
      << " cachePagePrewarmSuccesses=" << registrationStats.prewarmSuccesses
      << " cachePagePrewarmFailures=" << registrationStats.prewarmFailures
      << " cachePagePrewarmRuns=" << registrationStats.prewarmRuns
      << " cachePagePrewarmBytes=" << registrationStats.prewarmBytes
      << " cachePagePrewarmCoveredRuns=" << registrationStats.prewarmCoveredRuns
      << " cachePagePrewarmCoveredBytes="
      << registrationStats.prewarmCoveredBytes
      << " cachePageRegistrationWallNanos="
      << registrationStats.registrationWallNanos
      << " cachePagePrewarmWallNanos=" << registrationStats.prewarmWallNanos
      << " cacheHintScanWaits="
      << cacheHintScanWaits.load(std::memory_order_relaxed)
      << " cacheHintScanReadyAtTake="
      << cacheHintScanReadyAtTake.load(std::memory_order_relaxed)
      << " cacheHintScanWaitWallNanos="
      << cacheHintScanWaitWallNanos.load(std::memory_order_relaxed)
      << " cacheHintSplitPreloadWaits="
      << cacheHintSplitPreloadWaits.load(std::memory_order_relaxed)
      << " cacheHintSplitPreloadReadyAtTake="
      << cacheHintSplitPreloadReadyAtTake.load(std::memory_order_relaxed)
      << " cacheHintSplitPreloadWaitWallNanos="
      << cacheHintSplitPreloadWaitWallNanos.load(std::memory_order_relaxed)
      << " cacheHintRangeReadyRequests="
      << cacheHintRangeReadyRequests.load(std::memory_order_relaxed)
      << " cacheHintRangeReadyReleases="
      << cacheHintRangeReadyReleases.load(std::memory_order_relaxed)
      << " cacheHintFirstLoadGroupWaits="
      << cacheHintFirstLoadGroupWaits.load(std::memory_order_relaxed)
      << " cacheHintFirstLoadGroupReadyAtTake="
      << cacheHintFirstLoadGroupReadyAtTake.load(std::memory_order_relaxed)
      << " cacheHintFirstLoadGroupWaitWallNanos="
      << cacheHintFirstLoadGroupWaitWallNanos.load(std::memory_order_relaxed)
      << " cacheHintMakePinsCalls=" << asyncLoadStats.makePinsCalls
      << " cacheHintMakePinsEntries=" << asyncLoadStats.makePinsEntries
      << " cacheHintMakePinsWallNanos=" << asyncLoadStats.makePinsWallNanos
      << " cacheHintMakePinsActive=" << asyncLoadStats.makePinsActive
      << " cacheHintMakePinsMaxActive=" << asyncLoadStats.makePinsMaxActive
      << " cacheHintSubmitCalls=" << asyncLoadStats.submitCalls
      << " cacheHintSubmittedRequests=" << asyncLoadStats.submittedRequests
      << " cacheHintSubmitWallNanos=" << asyncLoadStats.submitWallNanos
      << " cacheHintSubmitActive=" << asyncLoadStats.submitActive
      << " cacheHintSubmitMaxActive=" << asyncLoadStats.submitMaxActive
      << " cacheHintWaitCalls=" << asyncLoadStats.waitCalls
      << " cacheHintWaitWallNanos=" << asyncLoadStats.waitWallNanos
      << " cacheHintWaitActive=" << asyncLoadStats.waitActive
      << " cacheHintWaitMaxActive=" << asyncLoadStats.waitMaxActive;
}

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
      if (stopped_) {
        entry->promise->set_exception(
            std::make_exception_ptr(
                std::runtime_error("Split prefetch query has stopped")));
        return;
      }
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

  bool hasRegisteredSplits() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !entries_.empty();
  }

  bool containsCacheHint(const std::string& splitKey) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cacheEntries_.count(splitKey) != 0;
  }

  void initialize(
      SplitPrefetchReadFactory readFactory,
      std::shared_ptr<ExecutorReadBroker> broker,
      uint32_t splitConcurrency,
      uint64_t maxReadyBytes) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (initialized_ || stopped_) {
        return;
      }
      VELOX_CHECK_NOT_NULL(broker);
      readFactory_ = std::move(readFactory);
      VELOX_CHECK(
          static_cast<bool>(readFactory_),
          "Split prefetch received an empty read factory");
      broker_ = std::move(broker);
      concurrency_ = std::max<uint32_t>(1, splitConcurrency);
      maxReadyBytes_ = std::max<uint64_t>(1, maxReadyBytes);
      splitExecutor_ =
          std::make_unique<folly::CPUThreadPoolExecutor>(concurrency_);
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
      entry->requested = true;
    }
    // A ready-byte window can otherwise fill with speculative splits while
    // every scan driver waits for a different split. Marking demand before
    // pumping lets the requested split bypass that ready-window head of line.
    // Active demand is bounded by the scan-driver count.
    pump();
    auto result = entry->future.get();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      entries_.erase(splitKey);
    }
    return result;
  }

  void registerCacheHint(
      const std::string& splitKey,
      CacheHintRangeStats rangeStats,
      ExecutorSplitPrefetch::CacheHintLoad load,
      uint32_t concurrency,
      uint64_t maxReadyBytes,
      std::shared_future<void> firstLoadReady) {
    VELOX_CHECK(static_cast<bool>(load));
    auto entry = std::make_shared<CacheHintEntry>();
    entry->key = splitKey;
    entry->bytes = rangeStats.uniqueBytes;
    entry->rangeStats = rangeStats;
    entry->load = std::move(load);
    entry->firstLoadReady = std::move(firstLoadReady);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_ || cacheEntries_.count(splitKey) != 0) {
        return;
      }
      if (!cacheExecutor_) {
        cacheConcurrency_ = std::max<uint32_t>(1, concurrency);
        cacheMaxReadyBytes_ = std::max<uint64_t>(1, maxReadyBytes);
        cacheExecutor_ =
            std::make_unique<folly::CPUThreadPoolExecutor>(cacheConcurrency_);
      }
      cacheEntries_.emplace(splitKey, entry);
      cacheOrder_.push_back(entry);
    }
    addRangePlan(rangeStats);
    const auto planned =
        cacheHintPlannedSplits.fetch_add(1, std::memory_order_relaxed) + 1;
    cacheHintPlannedBytes.fetch_add(
        rangeStats.uniqueBytes, std::memory_order_relaxed);
    if (planned == 1) {
      logCacheHintStats("first-planned", splitKey);
    }
    pumpCacheHints();
  }

  void takeCacheHint(const std::string& splitKey, CacheHintWaitMode waitMode) {
    std::shared_ptr<CacheHintEntry> entry;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = cacheEntries_.find(splitKey);
      if (it == cacheEntries_.end()) {
        return;
      }
      entry = it->second;
      entry->requested = true;
      if (!entry->rangeStatsAccounted) {
        entry->rangeStatsAccounted = true;
        addConsumedRanges(entry->rangeStats);
      }
    }
    pumpCacheHints();
    const bool waitForFirstLoad =
        waitMode == CacheHintWaitMode::kFirstLoadGroup &&
        entry->firstLoadReady.valid();
    const auto waitStart = std::chrono::steady_clock::now();
    const auto readyAtTake = waitForFirstLoad
        ? (entry->firstLoadReady.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready ||
           entry->future.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready)
        : entry->future.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready;
    const std::shared_future<void>* waitFuture = &entry->future;
    if (waitForFirstLoad) {
      // A load can fail before its first physical group invokes the readiness
      // callback. In that case full completion is the fallback signal; never
      // strand a scan on an external first-load future that cannot be set.
      while (entry->firstLoadReady.wait_for(std::chrono::milliseconds(1)) !=
             std::future_status::ready) {
        if (entry->future.wait_for(std::chrono::seconds(0)) ==
            std::future_status::ready) {
          break;
        }
      }
      if (entry->firstLoadReady.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
        waitFuture = &entry->firstLoadReady;
      }
    }
    waitFuture->wait();
    const auto waitNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - waitStart)
                               .count();
    if (waitForFirstLoad) {
      cacheHintFirstLoadGroupWaits.fetch_add(1, std::memory_order_relaxed);
      cacheHintFirstLoadGroupReadyAtTake.fetch_add(
          readyAtTake ? 1 : 0, std::memory_order_relaxed);
      cacheHintFirstLoadGroupWaitWallNanos.fetch_add(
          waitNanos, std::memory_order_relaxed);
    } else if (waitMode == CacheHintWaitMode::kSplitPreload) {
      cacheHintSplitPreloadWaits.fetch_add(1, std::memory_order_relaxed);
      cacheHintSplitPreloadReadyAtTake.fetch_add(
          readyAtTake ? 1 : 0, std::memory_order_relaxed);
      cacheHintSplitPreloadWaitWallNanos.fetch_add(
          waitNanos, std::memory_order_relaxed);
    } else {
      cacheHintScanWaits.fetch_add(1, std::memory_order_relaxed);
      cacheHintScanReadyAtTake.fetch_add(
          readyAtTake ? 1 : 0, std::memory_order_relaxed);
      cacheHintScanWaitWallNanos.fetch_add(
          waitNanos, std::memory_order_relaxed);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = cacheEntries_.find(splitKey);
      if (it != cacheEntries_.end()) {
        cacheReservedBytes_ -= entry->bytes;
        cacheEntries_.erase(it);
      }
    }
    pumpCacheHints();
  }

  void requestCacheHint(const std::string& splitKey) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = cacheEntries_.find(splitKey);
      if (it == cacheEntries_.end()) {
        return;
      }
      auto& entry = it->second;
      entry->requested = true;
      if (!entry->rangeStatsAccounted) {
        entry->rangeStatsAccounted = true;
        addConsumedRanges(entry->rangeStats);
      }
    }
    cacheHintRangeReadyRequests.fetch_add(1, std::memory_order_relaxed);
    pumpCacheHints();
  }

  void releaseCacheHint(const std::string& splitKey) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto it = cacheEntries_.find(splitKey);
      if (it == cacheEntries_.end()) {
        return;
      }
      const auto& entry = it->second;
      if (!entry->rangeStatsAccounted) {
        entry->rangeStatsAccounted = true;
        addUnusedRanges(entry->rangeStats);
      }
      if (entry->scheduled) {
        VELOX_CHECK_GE(cacheReservedBytes_, entry->bytes);
        cacheReservedBytes_ -= entry->bytes;
      } else {
        std::erase_if(cacheOrder_, [&](const auto& queued) {
          return queued.get() == entry.get();
        });
      }
      cacheEntries_.erase(it);
    }
    cacheHintRangeReadyReleases.fetch_add(1, std::memory_order_relaxed);
    pumpCacheHints();
  }

  bool useFirstLoadReadyForQuery(uint64_t minRegisteredSplits) {
    if (minRegisteredSplits == 0) {
      return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cacheFirstLoadDecisionMade_) {
      cacheFirstLoadDecisionMade_ = true;
      cacheUseFirstLoadReady_ = entries_.size() >= minRegisteredSplits;
      LOG(WARNING)
          << "CUDF_CACHE_HINT_FIRST_LOAD_QUERY_DECISION useFirstLoadReady="
          << cacheUseFirstLoadReady_
          << " minRegisteredSplits=" << minRegisteredSplits
          << " registeredSplits=" << entries_.size();
    }
    return cacheUseFirstLoadReady_;
  }

  void stop() {
    std::unique_ptr<folly::CPUThreadPoolExecutor> splitExecutor;
    std::unique_ptr<folly::CPUThreadPoolExecutor> cacheExecutor;
    std::vector<std::shared_ptr<SplitEntry>> canceled;
    std::vector<std::shared_ptr<CacheHintEntry>> canceledCacheHints;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopped_) {
        return;
      }
      stopped_ = true;
      initialized_ = false;
      splitExecutor = std::move(splitExecutor_);
      cacheExecutor = std::move(cacheExecutor_);
      for (auto& entry : order_) {
        if (!entry->scheduled) {
          canceled.push_back(entry);
        }
      }
      order_.clear();
      for (auto& entry : cacheOrder_) {
        if (!entry->scheduled) {
          canceledCacheHints.push_back(entry);
        }
      }
      cacheOrder_.clear();
      for (const auto& [key, entry] : cacheEntries_) {
        if (!entry->rangeStatsAccounted) {
          entry->rangeStatsAccounted = true;
          addUnusedRanges(entry->rangeStats);
        }
      }
    }
    const auto failure = std::make_exception_ptr(
        std::runtime_error(
            "Split prefetch canceled because the query stopped"));
    for (const auto& entry : canceled) {
      entry->promise->set_exception(failure);
    }
    for (const auto& entry : canceledCacheHints) {
      entry->promise->set_value();
    }
    // CPUThreadPoolExecutor destruction joins all active reads. This method is
    // called outside registry locks and on the query-cleanup thread, so the
    // final scheduler reference cannot be released by one of its own workers.
    splitExecutor.reset();
    cacheExecutor.reset();
  }

 private:
  void pumpCacheHints() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cacheExecutor_ || stopped_) {
      return;
    }
    auto schedule = [&](const std::shared_ptr<CacheHintEntry>& entry) {
      if (cacheActive_ >= cacheConcurrency_ || entry->scheduled) {
        return;
      }
      const bool fits = cacheReservedBytes_ <= cacheMaxReadyBytes_ &&
          entry->bytes <= cacheMaxReadyBytes_ - cacheReservedBytes_;
      if (!fits && !entry->requested) {
        return;
      }
      entry->scheduled = true;
      ++cacheActive_;
      cacheReservedBytes_ += entry->bytes;
      cacheExecutor_->add([self = shared_from_this(), entry]() {
        try {
          entry->load();
          cacheHintPrefetchedSplits.fetch_add(1, std::memory_order_relaxed);
          cacheHintPrefetchedBytes.fetch_add(
              entry->bytes, std::memory_order_relaxed);
        } catch (const std::exception& e) {
          const auto fallbacks =
              cacheHintFallbacks.fetch_add(1, std::memory_order_relaxed) + 1;
          LOG(WARNING) << "Best-effort cuDF cache hint failed for "
                       << entry->key << ": " << e.what();
          if (fallbacks == 1) {
            logCacheHintStats("first-fallback", entry->key);
          }
        } catch (...) {
          const auto fallbacks =
              cacheHintFallbacks.fetch_add(1, std::memory_order_relaxed) + 1;
          LOG(WARNING) << "Best-effort cuDF cache hint failed for "
                       << entry->key << ": unknown error";
          if (fallbacks == 1) {
            logCacheHintStats("first-fallback", entry->key);
          }
        }
        entry->promise->set_value();
        {
          std::lock_guard<std::mutex> lock(self->mutex_);
          --self->cacheActive_;
        }
        const auto completed =
            cacheHintPrefetchedSplits.load(std::memory_order_relaxed) +
            cacheHintFallbacks.load(std::memory_order_relaxed);
        if (completed == 1 || completed % 64 == 0) {
          logCacheHintStats(
              completed == 1 ? "first-completed" : "periodic", entry->key);
        }
        self->pumpCacheHints();
      });
    };
    for (const auto& entry : cacheOrder_) {
      if (cacheActive_ >= cacheConcurrency_) {
        break;
      }
      if (entry->requested) {
        schedule(entry);
      }
    }
    for (const auto& entry : cacheOrder_) {
      if (cacheActive_ >= cacheConcurrency_) {
        break;
      }
      schedule(entry);
    }
    std::erase_if(
        cacheOrder_, [](const auto& entry) { return entry->scheduled; });
  }

  void pump() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_ || stopped_) {
      return;
    }
    auto schedule = [&](const std::shared_ptr<SplitEntry>& entry) {
      if (active_ >= concurrency_ || entry->scheduled) {
        return;
      }
      const bool fits = reservedBytes_ <= maxReadyBytes_ &&
          entry->bytes <= maxReadyBytes_ - reservedBytes_;
      // An oversized speculative split would become the broker's sole
      // reservation and retain that reservation with its completed buffers.
      // A demanded oversized split could then block forever in reserve()
      // while the speculative result waits for a future consumer.
      if (!fits && !entry->requested) {
        return;
      }
      entry->scheduled = true;
      ++active_;
      reservedBytes_ += entry->bytes;
      splitExecutor_->add(
          [self = shared_from_this(), entry]() { self->run(entry); });
    };
    // Service blocked consumers before speculative lookahead. A demanded
    // split may exceed maxReadyBytes temporarily to break a dependency cycle;
    // concurrency_ bounds the number of such exceptions.
    for (const auto& entry : order_) {
      if (active_ >= concurrency_) {
        break;
      }
      if (entry->requested) {
        schedule(entry);
      }
    }
    for (const auto& entry : order_) {
      if (active_ >= concurrency_) {
        break;
      }
      schedule(entry);
    }
    // Once a split is scheduled, entries_ and the worker closure provide
    // its lifetime. Keeping it in order_ would also keep the shared_future
    // (and therefore completed pinned buffers) alive after take().
    std::erase_if(order_, [](const auto& entry) { return entry->scheduled; });
  }

  void run(const std::shared_ptr<SplitEntry>& entry) {
    try {
      auto result = std::make_shared<SplitPrefetchResult>();
      result->reservedBytes = entry->bytes;
      result->buffers.reserve(entry->files.size());
      auto reservation = broker_->reserve(entry->bytes);

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
        auto readFunction = readFactory_(file.path, file.size);
        VELOX_CHECK(
            static_cast<bool>(readFunction),
            "Split prefetch read factory returned an empty function for {}",
            file.path);
        auto future = broker_->read(
            std::move(readFunction),
            file.size,
            std::move(ranges),
            buffer,
            reservation);
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
      // The broker limits allocation and network I/O in flight. Ready-buffer
      // residency is accounted separately by reservedBytes_ until the
      // consumer releases the result. Retaining the broker reservation here
      // would duplicate that accounting and can deadlock demanded splits
      // behind a completed result that is waiting for its consumer.
      reservation.reset();
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
  std::unordered_map<std::string, std::shared_ptr<CacheHintEntry>>
      cacheEntries_;
  std::deque<std::shared_ptr<CacheHintEntry>> cacheOrder_;
  SplitPrefetchReadFactory readFactory_;
  std::shared_ptr<ExecutorReadBroker> broker_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> splitExecutor_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> cacheExecutor_;
  uint32_t concurrency_{0};
  uint32_t active_{0};
  uint64_t maxReadyBytes_{0};
  uint64_t reservedBytes_{0};
  uint32_t cacheConcurrency_{0};
  uint32_t cacheActive_{0};
  uint64_t cacheMaxReadyBytes_{0};
  uint64_t cacheReservedBytes_{0};
  bool cacheFirstLoadDecisionMade_{false};
  bool cacheUseFirstLoadReady_{false};
  bool initialized_{false};
  bool stopped_{false};
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
  static std::unordered_map<folly::Executor*, std::shared_ptr<ExecutorState>>
      states;
  return states;
}

using QueryExpectedSplitCountMap = std::unordered_map<std::string, uint64_t>;

struct QueryFirstLoadDecisionNode {
  std::string queryId;
  bool present{false};
  bool decision{false};
  QueryFirstLoadDecisionNode* next{nullptr};
};

std::atomic<QueryFirstLoadDecisionNode*>& queryFirstLoadDecisionHead() {
  static std::atomic<QueryFirstLoadDecisionNode*> head{nullptr};
  return head;
}

std::mutex& queryFirstLoadDecisionUpdateMutex() {
  static std::mutex mutex;
  return mutex;
}

std::atomic<std::shared_ptr<const QueryExpectedSplitCountMap>>&
queryExpectedSplitCountSnapshot() {
  static std::atomic<std::shared_ptr<const QueryExpectedSplitCountMap>>
      snapshot{std::make_shared<const QueryExpectedSplitCountMap>()};
  return snapshot;
}

bool findQueryExpectedSplitCount(
    const std::string& queryId,
    uint64_t& expectedSplits) {
  const auto snapshot =
      queryExpectedSplitCountSnapshot().load(std::memory_order_acquire);
  const auto it = snapshot->find(queryId);
  if (it == snapshot->end()) {
    return false;
  }
  expectedSplits = it->second;
  return true;
}

void publishQueryExpectedSplitCount(
    const std::string& queryId,
    uint64_t expectedSplits) {
  std::lock_guard<std::mutex> lock(queryFirstLoadDecisionUpdateMutex());
  const auto snapshot =
      queryExpectedSplitCountSnapshot().load(std::memory_order_acquire);
  auto updated = std::make_shared<QueryExpectedSplitCountMap>(*snapshot);
  (*updated)[queryId] = expectedSplits;
  queryExpectedSplitCountSnapshot().store(
      std::shared_ptr<const QueryExpectedSplitCountMap>(std::move(updated)),
      std::memory_order_release);
}

bool findQueryFirstLoadDecision(const std::string& queryId, bool& decision) {
  auto* node = queryFirstLoadDecisionHead().load(std::memory_order_acquire);
  while (node != nullptr) {
    if (node->queryId == queryId) {
      if (node->present) {
        decision = node->decision;
        return true;
      }
      return false;
    }
    node = node->next;
  }
  return false;
}

void publishQueryFirstLoadDecision(const std::string& queryId, bool decision) {
  auto* node = new QueryFirstLoadDecisionNode{
      .queryId = queryId, .present = true, .decision = decision};
  auto* head = queryFirstLoadDecisionHead().load(std::memory_order_relaxed);
  do {
    node->next = head;
  } while (!queryFirstLoadDecisionHead().compare_exchange_weak(
      head, node, std::memory_order_release, std::memory_order_relaxed));
}

void eraseQueryFirstLoadDecision(const std::string& queryId) {
  // Readers may still hold a raw pointer obtained from the lock-free head.
  // Publish a tombstone and retain old immutable nodes until process exit;
  // query IDs are few and short-lived, so this avoids reclamation barriers on
  // the scan hot path for negligible memory.
  auto* node = new QueryFirstLoadDecisionNode{.queryId = queryId};
  auto* head = queryFirstLoadDecisionHead().load(std::memory_order_relaxed);
  do {
    node->next = head;
  } while (!queryFirstLoadDecisionHead().compare_exchange_weak(
      head, node, std::memory_order_release, std::memory_order_relaxed));
}

void eraseQueryExpectedSplitCount(const std::string& queryId) {
  std::lock_guard<std::mutex> lock(queryFirstLoadDecisionUpdateMutex());
  const auto snapshot =
      queryExpectedSplitCountSnapshot().load(std::memory_order_acquire);
  if (!snapshot->contains(queryId)) {
    return;
  }
  auto updated = std::make_shared<QueryExpectedSplitCountMap>(*snapshot);
  updated->erase(queryId);
  queryExpectedSplitCountSnapshot().store(
      std::shared_ptr<const QueryExpectedSplitCountMap>(std::move(updated)),
      std::memory_order_release);
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

std::shared_ptr<QueryPrefetchState> findRegisteredQueryStateAcrossExecutors(
    const std::string& queryId) {
  std::lock_guard<std::mutex> registryLock(registryMutex());
  for (const auto& entry : registry()) {
    const auto& executorState = entry.second;
    std::lock_guard<std::mutex> executorLock(executorState->mutex);
    const auto it = executorState->queries.find(queryId);
    if (it != executorState->queries.end() &&
        it->second->hasRegisteredSplits()) {
      return it->second;
    }
  }
  return nullptr;
}

std::shared_ptr<QueryPrefetchState> findCacheHintStateAcrossExecutors(
    const std::string& queryId,
    const std::string& splitKey) {
  std::lock_guard<std::mutex> registryLock(registryMutex());
  for (const auto& entry : registry()) {
    const auto& executorState = entry.second;
    std::lock_guard<std::mutex> executorLock(executorState->mutex);
    const auto it = executorState->queries.find(queryId);
    if (it != executorState->queries.end() &&
        it->second->containsCacheHint(splitKey)) {
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
    SplitPrefetchReadFactory readFactory,
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
      std::move(readFactory),
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

void ExecutorSplitPrefetch::registerCacheHint(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey,
    uint64_t plannedBytes,
    CacheHintLoad load,
    uint32_t concurrency,
    uint64_t maxReadyBytes) {
  CacheHintRangeStats stats;
  stats.logicalRanges = 1;
  stats.logicalBytes = plannedBytes;
  stats.uniqueRanges = 1;
  stats.uniqueBytes = plannedBytes;
  registerCacheHint(
      executor,
      queryId,
      splitKey,
      stats,
      std::move(load),
      concurrency,
      maxReadyBytes,
      {});
}

void ExecutorSplitPrefetch::registerCacheHint(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey,
    CacheHintRangeStats rangeStats,
    CacheHintLoad load,
    uint32_t concurrency,
    uint64_t maxReadyBytes) {
  registerCacheHint(
      executor,
      queryId,
      splitKey,
      rangeStats,
      std::move(load),
      concurrency,
      maxReadyBytes,
      {});
}

void ExecutorSplitPrefetch::registerCacheHint(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey,
    CacheHintRangeStats rangeStats,
    CacheHintLoad load,
    uint32_t concurrency,
    uint64_t maxReadyBytes,
    std::shared_future<void> firstLoadReady) {
  if (!executor || rangeStats.uniqueBytes == 0 || !load) {
    return;
  }
  auto state = getQueryState(executor, queryId, false);
  if (!state) {
    state = getQueryState(executor, queryId, true);
  }
  state->registerCacheHint(
      splitKey,
      rangeStats,
      std::move(load),
      concurrency,
      maxReadyBytes,
      std::move(firstLoadReady));
}

void ExecutorSplitPrefetch::takeCacheHint(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey) {
  takeCacheHint(executor, queryId, splitKey, CacheHintWaitMode::kScan);
}

void ExecutorSplitPrefetch::takeCacheHint(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey,
    CacheHintWaitMode waitMode) {
  if (!executor) {
    return;
  }
  auto state = getQueryState(executor, queryId, false);
  if (!state || !state->containsCacheHint(splitKey)) {
    state = findCacheHintStateAcrossExecutors(queryId, splitKey);
  }
  if (state) {
    state->takeCacheHint(splitKey, waitMode);
  }
}

void ExecutorSplitPrefetch::requestCacheHint(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey) {
  if (!executor) {
    return;
  }
  auto state = getQueryState(executor, queryId, false);
  if (!state || !state->containsCacheHint(splitKey)) {
    state = findCacheHintStateAcrossExecutors(queryId, splitKey);
  }
  if (state) {
    state->requestCacheHint(splitKey);
  }
}

void ExecutorSplitPrefetch::releaseCacheHint(
    folly::Executor* executor,
    const std::string& queryId,
    const std::string& splitKey) {
  if (!executor) {
    return;
  }
  auto state = getQueryState(executor, queryId, false);
  if (!state || !state->containsCacheHint(splitKey)) {
    state = findCacheHintStateAcrossExecutors(queryId, splitKey);
  }
  if (state) {
    state->releaseCacheHint(splitKey);
  }
}

void ExecutorSplitPrefetch::setExpectedSplitCount(
    folly::Executor* executor,
    const std::string& queryId,
    uint64_t expectedSplits,
    uint64_t minRegisteredSplits) {
  if (!executor) {
    return;
  }
  publishQueryExpectedSplitCount(queryId, expectedSplits);
  if (minRegisteredSplits == 0) {
    return;
  }
  std::lock_guard<std::mutex> decisionLock(queryFirstLoadDecisionUpdateMutex());
  bool decision{false};
  if (findQueryFirstLoadDecision(queryId, decision)) {
    return;
  }
  decision = expectedSplits >= minRegisteredSplits;
  LOG(WARNING) << "CUDF_CACHE_HINT_FIRST_LOAD_QUERY_DECISION useFirstLoadReady="
               << decision << " minRegisteredSplits=" << minRegisteredSplits
               << " expectedSplits=" << expectedSplits
               << " source=coordinator-prepublished";
  publishQueryFirstLoadDecision(queryId, decision);
}

bool ExecutorSplitPrefetch::useFirstLoadReadyForQuery(
    folly::Executor* executor,
    const std::string& queryId,
    uint64_t minRegisteredSplits) {
  if (!executor || minRegisteredSplits == 0) {
    return true;
  }
  bool decision{false};
  if (findQueryFirstLoadDecision(queryId, decision)) {
    return decision;
  }
  // Split readers are prepared in a wide concurrent wave. Resolve the
  // coordinator-owned registered-split state only once per query, then serve
  // every reader from an immutable snapshot without entering registry or
  // query-state mutexes on the hot path.
  std::lock_guard<std::mutex> decisionLock(queryFirstLoadDecisionUpdateMutex());
  if (findQueryFirstLoadDecision(queryId, decision)) {
    return decision;
  }
  uint64_t expectedSplits{0};
  if (findQueryExpectedSplitCount(queryId, expectedSplits)) {
    decision = expectedSplits >= minRegisteredSplits;
    LOG(WARNING)
        << "CUDF_CACHE_HINT_FIRST_LOAD_QUERY_DECISION useFirstLoadReady="
        << decision << " minRegisteredSplits=" << minRegisteredSplits
        << " expectedSplits=" << expectedSplits
        << " source=coordinator-expected";
    publishQueryFirstLoadDecision(queryId, decision);
    return decision;
  }
  auto state = getQueryState(executor, queryId, false);
  if (!state || !state->hasRegisteredSplits()) {
    state = findRegisteredQueryStateAcrossExecutors(queryId);
  }
  decision = state && state->useFirstLoadReadyForQuery(minRegisteredSplits);
  publishQueryFirstLoadDecision(queryId, decision);
  return decision;
}

void ExecutorSplitPrefetch::recordCacheHintFallback() {
  const auto fallbacks =
      cacheHintFallbacks.fetch_add(1, std::memory_order_relaxed) + 1;
  logCacheHintStats(fallbacks == 1 ? "first-fallback" : "fallback");
}

void ExecutorSplitPrefetch::recordCachePrefetchPlanLookup(bool hit) {
  (hit ? cachePrefetchPlanHits : cachePrefetchPlanMisses)
      .fetch_add(1, std::memory_order_relaxed);
}

void ExecutorSplitPrefetch::recordCachePrefetchPlanEntries(uint64_t entries) {
  cachePrefetchPlanEntries.store(entries, std::memory_order_relaxed);
}

CachePrefetchPlanStats ExecutorSplitPrefetch::cachePrefetchPlanStatsForTest() {
  return {
      cachePrefetchPlanHits.load(std::memory_order_relaxed),
      cachePrefetchPlanMisses.load(std::memory_order_relaxed),
      cachePrefetchPlanEntries.load(std::memory_order_relaxed)};
}

CacheHintRangeStats ExecutorSplitPrefetch::cacheHintRangeStatsForTest() {
  return {
      .logicalRanges = cacheHintLogicalRanges.load(std::memory_order_relaxed),
      .logicalBytes = cacheHintLogicalBytes.load(std::memory_order_relaxed),
      .uniqueRanges = cacheHintUniqueRanges.load(std::memory_order_relaxed),
      .uniqueBytes = cacheHintUniqueBytes.load(std::memory_order_relaxed),
      .overlapBytes = cacheHintOverlapBytes.load(std::memory_order_relaxed),
      .duplicateSuppressedRanges =
          cacheHintDuplicateSuppressedRanges.load(std::memory_order_relaxed),
      .duplicateSuppressedBytes =
          cacheHintDuplicateSuppressedBytes.load(std::memory_order_relaxed),
      .consumedRanges = cacheHintConsumedRanges.load(std::memory_order_relaxed),
      .consumedBytes = cacheHintConsumedBytes.load(std::memory_order_relaxed),
      .unusedRanges = cacheHintUnusedRanges.load(std::memory_order_relaxed),
      .unusedBytes = cacheHintUnusedBytes.load(std::memory_order_relaxed)};
}

CacheHintWaitStats ExecutorSplitPrefetch::cacheHintWaitStatsForTest() {
  return {
      .scanWaits = cacheHintScanWaits.load(std::memory_order_relaxed),
      .scanReadyAtTake =
          cacheHintScanReadyAtTake.load(std::memory_order_relaxed),
      .scanWaitWallNanos =
          cacheHintScanWaitWallNanos.load(std::memory_order_relaxed),
      .splitPreloadWaits =
          cacheHintSplitPreloadWaits.load(std::memory_order_relaxed),
      .splitPreloadReadyAtTake =
          cacheHintSplitPreloadReadyAtTake.load(std::memory_order_relaxed),
      .splitPreloadWaitWallNanos =
          cacheHintSplitPreloadWaitWallNanos.load(std::memory_order_relaxed),
      .rangeReadyRequests =
          cacheHintRangeReadyRequests.load(std::memory_order_relaxed),
      .rangeReadyReleases =
          cacheHintRangeReadyReleases.load(std::memory_order_relaxed),
      .firstLoadGroupWaits =
          cacheHintFirstLoadGroupWaits.load(std::memory_order_relaxed),
      .firstLoadGroupReadyAtTake =
          cacheHintFirstLoadGroupReadyAtTake.load(std::memory_order_relaxed),
      .firstLoadGroupWaitWallNanos =
          cacheHintFirstLoadGroupWaitWallNanos.load(std::memory_order_relaxed)};
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
  for (const auto& queryState : queryStates) {
    queryState->stop();
  }
  queryStates.clear();
  eraseQueryFirstLoadDecision(queryId);
  eraseQueryExpectedSplitCount(queryId);
  logCacheHintStats("query-exit", queryId);
  logNativeS3SchedulerStats("query-exit", queryId);
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
  std::vector<std::shared_ptr<QueryPrefetchState>> queryStates;
  std::vector<std::string> queryIds;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    for (auto& [queryId, queryState] : state->queries) {
      queryIds.push_back(queryId);
      queryStates.push_back(std::move(queryState));
    }
    state->queries.clear();
  }
  for (const auto& queryState : queryStates) {
    queryState->stop();
  }
  queryStates.clear();
  for (const auto& queryId : queryIds) {
    eraseQueryFirstLoadDecision(queryId);
    eraseQueryExpectedSplitCount(queryId);
  }
  state.reset();
  logCacheHintStats("executor-exit");
  logNativeS3SchedulerStats("executor-exit");
}

} // namespace facebook::velox::cudf_velox::connector::hive
