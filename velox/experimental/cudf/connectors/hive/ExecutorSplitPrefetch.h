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

#include "velox/experimental/cudf/connectors/hive/ExecutorReadBroker.h"

#include <folly/Executor.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

struct SplitPrefetchFile {
  std::string path;
  uint64_t size;
};

using SplitPrefetchReadFactory = std::function<
    PrefetchReadFunction(const std::string&, std::optional<std::size_t>)>;

struct SplitPrefetchResult {
  ~SplitPrefetchResult();

  std::vector<std::shared_ptr<PinnedHostBuffer>> buffers;

  // Internal scheduler accounting retained with the result so the ready-byte
  // reservation is released only after the consumer drops all pinned buffers.
  uint64_t reservedBytes{0};
  std::function<void(uint64_t)> release;
};

struct CacheHintRangeStats {
  // Logical ranges come directly from the reader planner. Unique ranges are
  // file-global load-quantum cache keys after canonicalization.
  uint64_t logicalRanges{0};
  uint64_t logicalBytes{0};
  uint64_t uniqueRanges{0};
  uint64_t uniqueBytes{0};
  uint64_t overlapBytes{0};
  uint64_t duplicateSuppressedRanges{0};
  uint64_t duplicateSuppressedBytes{0};
  // "Consumed" means the split demanded and waited for the hint. "Unused"
  // means query cleanup reached a registered hint before its split demanded
  // it. These counters conserve uniqueRanges/uniqueBytes at final query exit.
  uint64_t consumedRanges{0};
  uint64_t consumedBytes{0};
  uint64_t unusedRanges{0};
  uint64_t unusedBytes{0};
};

struct CachePrefetchPlanStats {
  uint64_t hits{0};
  uint64_t misses{0};
  uint64_t entries{0};
};

/// Future-backed one-shot signal for the first physical cache load. The
/// scheduler also signals terminal completion so an early load failure cannot
/// strand a scan. Keeping the wait side on std::shared_future preserves the
/// direct blocking behavior used by the original ready-first implementation.
class CacheHintFirstLoadSignal {
 public:
  CacheHintFirstLoadSignal();

  void signal();

  const std::shared_future<void>& future() const {
    return future_;
  }

 private:
  std::atomic<bool> signaled_{false};
  std::promise<void> promise_;
  std::shared_future<void> future_;
};

enum class CacheHintWaitMode : uint8_t {
  kScan,
  kSplitPreload,
  kFirstLoadGroup,
};

struct CacheHintWaitStats {
  uint64_t scanWaits{0};
  uint64_t scanReadyAtTake{0};
  uint64_t scanWaitWallNanos{0};
  uint64_t splitPreloadWaits{0};
  uint64_t splitPreloadReadyAtTake{0};
  uint64_t splitPreloadWaitWallNanos{0};
  uint64_t rangeReadyRequests{0};
  uint64_t rangeReadyReleases{0};
  uint64_t firstLoadGroupWaits{0};
  uint64_t firstLoadGroupReadyAtTake{0};
  uint64_t firstLoadGroupWaitWallNanos{0};
};

/// Executor-scoped, cross-split prefetch queue. MPP registers all physical
/// file groups before starting Velox tasks. The first reader initializes a
/// shared source factory, after which a dedicated scheduler keeps multiple
/// grouped splits in flight. Readers consume the first completed split without
/// waiting for a wave-wide barrier.
class ExecutorSplitPrefetch {
 public:
  using CacheHintLoad = std::function<void()>;

  static void registerSplit(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey,
      std::vector<SplitPrefetchFile> files);

  static bool contains(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey);

  static void initialize(
      folly::Executor* executor,
      const std::string& queryId,
      SplitPrefetchReadFactory readFactory,
      std::shared_ptr<ExecutorReadBroker> broker,
      uint32_t splitConcurrency,
      uint64_t maxReadyBytes);

  static std::shared_ptr<SplitPrefetchResult> take(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey);

  /// Registers a best-effort cache population hint discovered while a future
  /// split reader parses its footer. Hints are bounded by the same query-level
  /// ready-byte window semantics as whole-split prefetch.
  static void registerCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey,
      uint64_t plannedBytes,
      CacheHintLoad load,
      uint32_t concurrency,
      uint64_t maxReadyBytes);

  static void registerCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey,
      CacheHintRangeStats rangeStats,
      CacheHintLoad load,
      uint32_t concurrency,
      uint64_t maxReadyBytes);

  static void registerCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey,
      CacheHintRangeStats rangeStats,
      CacheHintLoad load,
      uint32_t concurrency,
      uint64_t maxReadyBytes,
      std::shared_ptr<CacheHintFirstLoadSignal> firstLoadReady);

  /// Marks a cache hint as demanded and waits for it. Failures are swallowed:
  /// the regular reader demand path remains the correctness fallback.
  static void takeCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey);

  static void takeCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey,
      CacheHintWaitMode waitMode);

  /// Marks a cache hint as demanded and eligible for scheduling without
  /// waiting for the entire hint. Demand reads then synchronize on the
  /// individual AsyncDataCache keys they actually consume.
  static void requestCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey);

  /// Releases scheduler admission bytes retained by a non-blocking request.
  /// This is safe before the best-effort load completes: the worker owns the
  /// load lifetime independently of the scheduler entry.
  static void releaseCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey);

  /// Publishes the coordinator's complete split count before individual
  /// splits are registered, so readers can choose a query-wide wait policy
  /// without waiting behind the registration wave.
  static void setExpectedSplitCount(
      folly::Executor* executor,
      const std::string& queryId,
      uint64_t expectedSplits,
      uint64_t minRegisteredSplits);

  /// Makes one query-wide first-load decision from the physical splits already
  /// known by the coordinator. The first decision is frozen so every scan in
  /// the query uses the same wait policy.
  static bool useFirstLoadReadyForQuery(
      folly::Executor* executor,
      const std::string& queryId,
      uint64_t minRegisteredSplits);

  static void recordCacheHintFallback();

  static void recordCachePrefetchPlanLookup(bool hit);

  static void recordCachePrefetchPlanEntries(uint64_t entries);

  static CachePrefetchPlanStats cachePrefetchPlanStatsForTest();

  static CacheHintRangeStats cacheHintRangeStatsForTest();

  static CacheHintWaitStats cacheHintWaitStatsForTest();

  static void eraseQuery(folly::Executor* executor, const std::string& queryId);

  static void erase(folly::Executor* executor);
};

} // namespace facebook::velox::cudf_velox::connector::hive
