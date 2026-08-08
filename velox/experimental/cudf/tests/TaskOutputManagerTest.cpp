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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/ucx-exchange/LocalDeviceOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <folly/ScopeGuard.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

class TaskOutputManagerTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    auto& config = cudf_velox::CudfConfig::getInstance();
    previousEnabled_ = config.enabled;
    previousExchange_ = config.exchange;
    previousAllowCpuFallback_ = config.allowCpuFallback;
    previousMemoryResource_ = config.memoryResource;
    config.enabled = true;
    config.exchange = true;
    config.allowCpuFallback = true;
    config.memoryResource = "async";
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    auto& config = cudf_velox::CudfConfig::getInstance();
    config.enabled = previousEnabled_;
    config.exchange = previousExchange_;
    config.allowCpuFallback = previousAllowCpuFallback_;
    config.memoryResource = previousMemoryResource_;
    HiveConnectorTestBase::TearDown();
  }

  std::shared_ptr<Task> startTask(
      const std::string& taskId,
      std::string transport,
      core::QueryConfig queryConfig = core::QueryConfig{{}}) {
    auto output = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
        PlanBuilder()
            .tableScan(ROW({"c0"}, {BIGINT()}))
            .partitionedOutputBroadcast()
            .planNode());
    auto plan = core::PartitionedOutputNode::Builder(*output)
                    .transportKind(std::move(transport))
                    .build();
    auto task = Task::create(
        taskId,
        core::PlanFragment{std::move(plan)},
        0,
        core::QueryCtx::create(driverExecutor_.get(), std::move(queryConfig)),
        Task::ExecutionMode::kParallel,
        Consumer{});
    task->start(1, 1);
    return task;
  }

 private:
  bool previousEnabled_{true};
  bool previousExchange_{false};
  bool previousAllowCpuFallback_{true};
  std::string previousMemoryResource_;
};

TEST_F(TaskOutputManagerTest, selectionAndCancellationCleanup) {
  auto queueManager = ucx_exchange::UcxOutputQueueManager::getInstanceRef();
  const std::string ucxTaskId = "ucx-output-manager-lifecycle";
  queueManager->removeTask(ucxTaskId);
  auto cleanup =
      folly::makeGuard([&]() { queueManager->removeTask(ucxTaskId); });

  auto ucxTask = startTask(
      ucxTaskId, std::string{core::TransportKind::kUcx});
  ASSERT_TRUE(queueManager->stats(ucxTaskId).has_value());

  std::atomic_bool cancellationDelivered{false};
  queueManager->getData(
      ucxTaskId,
      0,
      [&](std::shared_ptr<cudf::packed_columns> data, std::vector<int64_t>) {
        EXPECT_EQ(data, nullptr);
        cancellationDelivered = true;
      });
  ucxTask->requestAbort().wait();
  EXPECT_TRUE(cancellationDelivered);
  EXPECT_FALSE(queueManager->stats(ucxTaskId).has_value());
  queueManager->removeTask(ucxTaskId);
  cleanup.dismiss();

  const std::string httpTaskId = "http-output-manager-lifecycle";
  auto httpTask = startTask(
      httpTaskId, std::string{core::TransportKind::kInMemory});
  EXPECT_FALSE(queueManager->stats(httpTaskId).has_value());
  httpTask->requestAbort().wait();

  const std::string disabledTaskId = "disabled-ucx-output-manager-lifecycle";
  auto disabledTask = startTask(
      disabledTaskId,
      std::string{core::TransportKind::kUcx},
      core::QueryConfig(
          std::unordered_map<std::string, std::string>{
              {cudf_velox::CudfConfig::kCudfEnabled, "false"}}));
  // Transport selection is explicit in the plan. Query-level cuDF disablement
  // influences planning/operator adaptation, but it does not rewrite an
  // already selected UCX transport back to in-memory buffering.
  EXPECT_TRUE(queueManager->stats(disabledTaskId).has_value());
  disabledTask->requestAbort().wait();
}

TEST_F(TaskOutputManagerTest, localDeviceRootSharesUcxTaskLifecycle) {
  const std::string taskId = "local-device-root-output-lifecycle";
  auto ucxManager = ucx_exchange::UcxOutputQueueManager::getInstanceRef();
  auto localManager =
      ucx_exchange::LocalDeviceOutputQueueManager::getInstanceRef();
  ucxManager->removeTask(taskId);
  localManager->removeTask(taskId);
  auto cleanup = folly::makeGuard([&]() {
    ucxManager->removeTask(taskId);
    localManager->removeTask(taskId);
  });

  localManager->registerDirectOutputTask(taskId);
  auto task = startTask(
      taskId,
      std::string{core::TransportKind::kUcx},
      core::QueryConfig(std::unordered_map<std::string, std::string>{
          {ucx_exchange::LocalDeviceOutputQueueManager::kEnabledConfig,
           "true"}}));

  EXPECT_TRUE(ucxManager->stats(taskId).has_value());
  EXPECT_TRUE(localManager->isDirectOutputTask(taskId));
  const auto initial = localManager->tryGetData(taskId, 0, 0);
  EXPECT_FALSE(initial.ready);

  task->requestAbort().wait();
  EXPECT_FALSE(ucxManager->stats(taskId).has_value());
  EXPECT_FALSE(localManager->isDirectOutputTask(taskId));
  cleanup.dismiss();
}

} // namespace
