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

#include <cstdint>
#include <functional>
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

  /// Marks a cache hint as demanded and waits for it. Failures are swallowed:
  /// the regular reader demand path remains the correctness fallback.
  static void takeCacheHint(
      folly::Executor* executor,
      const std::string& queryId,
      const std::string& splitKey);

  static void recordCacheHintFallback();

  static void recordCachePrefetchPlanLookup(bool hit);

  static void recordCachePrefetchPlanEntries(uint64_t entries);

  static CachePrefetchPlanStats cachePrefetchPlanStatsForTest();

  static CacheHintRangeStats cacheHintRangeStatsForTest();

  static void eraseQuery(folly::Executor* executor, const std::string& queryId);

  static void erase(folly::Executor* executor);
};

} // namespace facebook::velox::cudf_velox::connector::hive
