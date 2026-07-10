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
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include <gtest/gtest.h>
#include <rmm/cuda_stream_view.hpp>
#include <atomic>

using namespace facebook::velox::ucx_exchange;

namespace {
// Distinct task ids keep tests independent despite the process-wide singleton.
IntraNodeTransferKey makeKey(const std::string& taskId) {
  return IntraNodeTransferKey{taskId, /*destination=*/0, /*sequenceNumber=*/0};
}
} // namespace

// A waiter registered before publish goes dormant (returns false) and is woken
// exactly once when publish lands the data — the event-driven replacement for
// the old poll/self-requeue busy spin that livelocked the single-threaded
// Communicator.
TEST(IntraNodeTransferRegistryTest, registerWaiterWokenByPublish) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const auto key = makeKey("registerWaiterWokenByPublish");

  std::atomic<int> woken{0};
  const bool readyNow = registry->registerWaiter(key, [&woken]() { ++woken; });
  EXPECT_FALSE(readyNow);
  EXPECT_EQ(woken.load(), 0);

  auto future = registry->publish(
      key, /*data=*/nullptr, rmm::cuda_stream_default, /*atEnd=*/false);
  EXPECT_EQ(woken.load(), 1);

  // The woken consumer would now poll and retrieve the data.
  auto result = registry->poll(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->atEnd);
}

// A waiter registered after the data is already published returns true
// (re-poll, no dormancy) and does not leave a stale wakeup behind.
TEST(IntraNodeTransferRegistryTest, registerWaiterReadyReturnsTrue) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const auto key = makeKey("registerWaiterReadyReturnsTrue");

  auto future = registry->publish(
      key, /*data=*/nullptr, rmm::cuda_stream_default, /*atEnd=*/true);

  std::atomic<int> woken{0};
  const bool readyNow = registry->registerWaiter(key, [&woken]() { ++woken; });
  EXPECT_TRUE(readyNow);
  EXPECT_EQ(woken.load(), 0);

  auto result = registry->poll(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);
}

// cancelTask wakes a dormant waiter so it re-polls and observes the atEnd
// result instead of waiting forever for a producer that will never publish.
TEST(IntraNodeTransferRegistryTest, registerWaiterWokenByCancel) {
  auto registry = IntraNodeTransferRegistry::getInstance();
  const std::string taskId = "registerWaiterWokenByCancel";
  const auto key = makeKey(taskId);

  std::atomic<int> woken{0};
  const bool readyNow = registry->registerWaiter(key, [&woken]() { ++woken; });
  EXPECT_FALSE(readyNow);

  registry->cancelTask(taskId);
  EXPECT_EQ(woken.load(), 1);

  auto result = registry->poll(key);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->atEnd);

  // Clear singleton state so the cancelled task does not leak into other tests.
  registry->clearCancelledTask(taskId);
}
