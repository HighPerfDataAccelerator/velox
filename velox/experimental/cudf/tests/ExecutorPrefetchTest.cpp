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

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

using namespace std::chrono_literals;

TEST(ExecutorPrefetchTest, reservesBytesBeforeAllocation) {
  folly::CPUThreadPoolExecutor executor(1);
  auto broker = ExecutorReadBroker::get(&executor, 4, 1);
  auto first = broker->reserve(4);

  auto second = std::async(std::launch::async, [&] {
    return broker->reserve(1);
  });
  EXPECT_EQ(second.wait_for(50ms), std::future_status::timeout);

  first.reset();
  EXPECT_EQ(second.wait_for(5s), std::future_status::ready);
  auto secondReservation = second.get();
  secondReservation.reset();

  ExecutorReadBroker::erase(&executor);
  broker.reset();
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
          "s3://bucket-a/primary.parquet",
          "s3://bucket-b/coalesced.parquet"));

  result.reset();
  ExecutorSplitPrefetch::eraseQuery(&executor, "query-paths");
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
      firstReadStarted.get_future().wait_for(5s),
      std::future_status::ready);
  auto waitingTake = std::async(std::launch::async, [&] {
    return ExecutorSplitPrefetch::take(
        &executor, "query-stop", "s3://bucket/second.parquet");
  });
  ASSERT_EQ(
      waitingTake.wait_for(50ms), std::future_status::timeout);

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
