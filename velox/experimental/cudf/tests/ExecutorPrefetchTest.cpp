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
#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

using namespace std::chrono_literals;

TEST(ExecutorPrefetchTest, reservesBytesBeforeAllocation) {
  folly::CPUThreadPoolExecutor executor(1);
  auto broker = ExecutorReadBroker::get(&executor, 4, 1);
  auto first = broker->reserve(4);

  auto second =
      std::async(std::launch::async, [&] { return broker->reserve(1); });
  EXPECT_EQ(second.wait_for(50ms), std::future_status::timeout);

  first.reset();
  EXPECT_EQ(second.wait_for(5s), std::future_status::ready);
  auto secondReservation = second.get();
  secondReservation.reset();

  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, splitConcurrencyConfigIsIndependentOfReadThreads) {
  auto connectorConfig = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kPrefetchThreads, "3"}});
  CudfHiveConfig hiveConfig(connectorConfig);

  config::ConfigBase defaultSession(
      std::unordered_map<std::string, std::string>{});
  // Zero/unset preserves the historical min(16, prefetchThreads) behavior.
  EXPECT_EQ(
      hiveConfig.executorSplitPrefetchConcurrencySession(&defaultSession), 3);

  config::ConfigBase explicitSession(
      std::unordered_map<std::string, std::string>{
          {CudfHiveConfig::kPrefetchThreadsSession, "64"},
          {CudfHiveConfig::kExecutorSplitPrefetchConcurrencySession, "5"}});
  EXPECT_EQ(hiveConfig.prefetchThreadsSession(&explicitSession), 64);
  EXPECT_EQ(
      hiveConfig.executorSplitPrefetchConcurrencySession(&explicitSession), 5);
}

TEST(ExecutorPrefetchTest, createsOneReadHandlePerPhysicalPath) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 8, 2);
  std::mutex pathsMutex;
  std::vector<std::string> paths;

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-paths",
      "s3://bucket-a/primary.parquet",
      {{"s3://bucket-a/primary.parquet", 1},
       {"s3://bucket-b/coalesced.parquet", 1}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-paths",
      [&](const std::string& path, std::optional<std::size_t>) {
        {
          std::lock_guard<std::mutex> lock(pathsMutex);
          paths.push_back(path);
        }
        return PrefetchReadFunction{
            [](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
            }};
      },
      broker,
      1,
      8);

  auto result = ExecutorSplitPrefetch::take(
      &executor, "query-paths", "s3://bucket-a/primary.parquet");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->buffers.size(), 2);
  EXPECT_THAT(
      paths,
      testing::UnorderedElementsAre(
          "s3://bucket-a/primary.parquet", "s3://bucket-b/coalesced.parquet"));

  result.reset();
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-paths");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, reusesOneReadSourceAcrossRangesOfFile) {
  constexpr uint64_t kRangeBytes = 16ULL << 20;
  constexpr uint64_t kFileBytes = kRangeBytes + 1;
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, kFileBytes, 2);
  std::atomic<uint32_t> factoryCount{0};
  std::atomic<uint32_t> rangeCount{0};
  std::mutex observationsMutex;
  std::vector<uint32_t> sourceIds;
  std::vector<uint64_t> offsets;

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-multi-range",
      "s3://bucket/multi-range.parquet",
      {{"s3://bucket/multi-range.parquet", kFileBytes}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-multi-range",
      [&](const std::string&, std::optional<std::size_t>) {
        const auto sourceId =
            factoryCount.fetch_add(1, std::memory_order_relaxed) + 1;
        return PrefetchReadFunction{
            [&, sourceId](
                uint64_t offset, uint64_t size, uint8_t* destination) {
              std::memset(destination, static_cast<int>(sourceId), size);
              rangeCount.fetch_add(1, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(observationsMutex);
              sourceIds.push_back(sourceId);
              offsets.push_back(offset);
            }};
      },
      broker,
      1,
      kFileBytes);

  auto result = ExecutorSplitPrefetch::take(
      &executor, "query-multi-range", "s3://bucket/multi-range.parquet");
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->buffers.size(), 1);
  EXPECT_EQ(factoryCount.load(), 1);
  EXPECT_EQ(rangeCount.load(), 2);
  EXPECT_THAT(sourceIds, testing::ElementsAre(1, 1));
  EXPECT_THAT(
      offsets, testing::UnorderedElementsAre(0, kRangeBytes));
  EXPECT_EQ(result->buffers.front()->data()[0], 1);
  EXPECT_EQ(result->buffers.front()->data()[kFileBytes - 1], 1);

  result.reset();
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-multi-range");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, requestedSplitBypassesFullReadyWindow) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 1, 2);
  std::promise<void> firstReadCompleted;
  std::atomic<uint32_t> readCount{0};

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-demand",
      "s3://bucket/first.parquet",
      {{"s3://bucket/first.parquet", 1}});
  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-demand",
      "s3://bucket/second.parquet",
      {{"s3://bucket/second.parquet", 1}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-demand",
      [&](const std::string& path, std::optional<std::size_t>) {
        return PrefetchReadFunction{
            [&, path](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
              ++readCount;
              if (path.ends_with("first.parquet")) {
                firstReadCompleted.set_value();
              }
            }};
      },
      broker,
      2,
      1);

  ASSERT_EQ(
      firstReadCompleted.get_future().wait_for(5s),
      std::future_status::ready);
  // The first completed result still owns the entire ready window. It must
  // not also retain the broker's full I/O budget: a demand for the second
  // split has to make progress without first consuming the ready result.
  auto second = std::async(std::launch::async, [&] {
    return ExecutorSplitPrefetch::take(
        &executor, "query-demand", "s3://bucket/second.parquet");
  });
  ASSERT_EQ(second.wait_for(5s), std::future_status::ready);
  auto secondResult = second.get();
  ASSERT_NE(secondResult, nullptr);
  secondResult.reset();

  auto firstResult = ExecutorSplitPrefetch::take(
      &executor, "query-demand", "s3://bucket/first.parquet");
  ASSERT_NE(firstResult, nullptr);
  firstResult.reset();
  EXPECT_EQ(readCount.load(), 2);

  ExecutorSplitPrefetch::eraseQuery(&executor, "query-demand");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, oversizedSpeculationWaitsForDemand) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 1, 2);
  std::atomic<uint32_t> firstReadCount{0};
  std::atomic<uint32_t> secondReadCount{0};

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-oversized-demand",
      "s3://bucket/first.parquet",
      {{"s3://bucket/first.parquet", 2}});
  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-oversized-demand",
      "s3://bucket/second.parquet",
      {{"s3://bucket/second.parquet", 2}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-oversized-demand",
      [&](const std::string& path, std::optional<std::size_t>) {
        return PrefetchReadFunction{
            [&, path](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
              if (path.ends_with("first.parquet")) {
                ++firstReadCount;
              } else {
                ++secondReadCount;
              }
            }};
      },
      broker,
      2,
      1);

  // Neither oversized split may monopolize the broker speculatively.
  std::this_thread::sleep_for(50ms);
  EXPECT_EQ(firstReadCount.load(), 0);
  EXPECT_EQ(secondReadCount.load(), 0);

  auto secondResult = ExecutorSplitPrefetch::take(
      &executor, "query-oversized-demand", "s3://bucket/second.parquet");
  ASSERT_NE(secondResult, nullptr);
  secondResult.reset();
  EXPECT_EQ(firstReadCount.load(), 0);
  EXPECT_EQ(secondReadCount.load(), 1);

  auto firstResult = ExecutorSplitPrefetch::take(
      &executor, "query-oversized-demand", "s3://bucket/first.parquet");
  ASSERT_NE(firstResult, nullptr);
  firstResult.reset();
  EXPECT_EQ(firstReadCount.load(), 1);

  ExecutorSplitPrefetch::eraseQuery(&executor, "query-oversized-demand");
  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

TEST(ExecutorPrefetchTest, stopsBeforeErasingQueryState) {
  folly::CPUThreadPoolExecutor executor(2);
  auto broker = ExecutorReadBroker::get(&executor, 2, 1);
  std::promise<void> firstReadStarted;
  std::promise<void> unblockFirstRead;
  auto unblock = unblockFirstRead.get_future().share();
  std::atomic<uint32_t> secondReadCount{0};

  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-stop",
      "s3://bucket/first.parquet",
      {{"s3://bucket/first.parquet", 1}});
  ExecutorSplitPrefetch::registerSplit(
      &executor,
      "query-stop",
      "s3://bucket/second.parquet",
      {{"s3://bucket/second.parquet", 1}});
  ExecutorSplitPrefetch::initialize(
      &executor,
      "query-stop",
      [&](const std::string& path, std::optional<std::size_t>) {
        if (path.ends_with("first.parquet")) {
          firstReadStarted.set_value();
          return PrefetchReadFunction{
              [unblock](uint64_t, uint64_t size, uint8_t* destination) {
                unblock.wait();
                std::memset(destination, 0, size);
              }};
        }
        ++secondReadCount;
        return PrefetchReadFunction{
            [](uint64_t, uint64_t size, uint8_t* destination) {
              std::memset(destination, 0, size);
            }};
      },
      broker,
      1,
      1);

  ASSERT_EQ(
      firstReadStarted.get_future().wait_for(5s), std::future_status::ready);
  auto waitingTake = std::async(std::launch::async, [&] {
    return ExecutorSplitPrefetch::take(
        &executor, "query-stop", "s3://bucket/second.parquet");
  });
  ASSERT_EQ(waitingTake.wait_for(50ms), std::future_status::timeout);

  auto cleanup = std::async(std::launch::async, [&] {
    ExecutorSplitPrefetch::eraseQuery(&executor, "query-stop");
  });
  EXPECT_EQ(cleanup.wait_for(50ms), std::future_status::timeout);

  unblockFirstRead.set_value();
  EXPECT_EQ(cleanup.wait_for(5s), std::future_status::ready);
  cleanup.get();
  EXPECT_THROW(waitingTake.get(), std::runtime_error);
  EXPECT_EQ(secondReadCount.load(), 0);

  ExecutorReadBroker::erase(&executor);
  broker.reset();
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
