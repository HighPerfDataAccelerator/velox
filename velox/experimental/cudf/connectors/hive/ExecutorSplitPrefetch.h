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

using SplitPrefetchReadFactory = std::function<PrefetchReadFunction(
    const std::string&, std::optional<std::size_t>)>;

struct SplitPrefetchResult {
  ~SplitPrefetchResult();

  std::vector<std::shared_ptr<PinnedHostBuffer>> buffers;

  // Internal scheduler accounting retained with the result so the ready-byte
  // reservation is released only after the consumer drops all pinned buffers.
  uint64_t reservedBytes{0};
  std::function<void(uint64_t)> release;
};

/// Executor-scoped, cross-split prefetch queue. MPP registers all physical
/// file groups before starting Velox tasks. The first reader initializes a
/// shared source factory, after which a dedicated scheduler keeps multiple
/// grouped splits in flight. Readers consume the first completed split without
/// waiting for a wave-wide barrier.
class ExecutorSplitPrefetch {
 public:
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

  static void eraseQuery(
      folly::Executor* executor,
      const std::string& queryId);

  static void erase(folly::Executor* executor);
};

} // namespace facebook::velox::cudf_velox::connector::hive
