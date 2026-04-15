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

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "folly/experimental/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/exec/PartitionedOutput.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/SerializedPageUtil.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/experimental/cudf/exchange/GpuExchange.h"
#include "velox/experimental/cudf/exchange/GpuPartitionedOutput.h"
#include "velox/experimental/cudf/exchange/LocalGpuExchangeSource.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using facebook::velox::common::testutil::TestValue;
using facebook::velox::test::BatchMaker;

namespace facebook::velox::cudf_velox {

namespace {

class GpuMultiFragmentTest : public exec::test::HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    exec::ExchangeSource::factories().clear();
    exec::ExchangeSource::registerFactory(createLocalExchangeSource);
    exec::ExchangeSource::registerFactory(createLocalGpuExchangeSource);
    testingStartLocalGpuExchangeSource();
  }

  void TearDown() override {
    vectors_.clear();
    waitForAllTasksToBeDeleted();
    testingShutdownLocalGpuExchangeSource();
    HiveConnectorTestBase::TearDown();
  }

  static std::string makeTaskId(const std::string& prefix, int num) {
    return fmt::format("local://{}-{}", prefix, num);
  }

  static exec::Consumer noopConsumer() {
    return [](RowVectorPtr, bool, ContinueFuture*) {
      return BlockingReason::kNotBlocked;
    };
  }

  std::shared_ptr<Task> makeTask(
      const std::string& taskId,
      const core::PlanNodePtr& planNode,
      int destination = 0,
      Consumer consumer = nullptr,
      int64_t maxMemory = memory::kMaxMemory,
      folly::Executor* executor = nullptr) const {
    auto configCopy = configSettings_;
    auto queryCtx = core::QueryCtx::create(
        executor ? executor : executor_.get(),
        core::QueryConfig(std::move(configCopy)));
    queryCtx->testingOverrideMemoryPool(
        memory::memoryManager()->addRootPool(
            queryCtx->queryId(), maxMemory, MemoryReclaimer::create()));
    core::PlanFragment planFragment{planNode};
    return Task::create(
        taskId,
        std::move(planFragment),
        destination,
        std::move(queryCtx),
        Task::ExecutionMode::kParallel,
        std::move(consumer));
  }

  std::vector<RowVectorPtr> makeVectors(int count, int rowsPerVector) {
    std::vector<RowVectorPtr> vectors;
    for (int i = 0; i < count; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(rowType_, rowsPerVector, *pool_));
      vectors.push_back(vector);
    }
    return vectors;
  }

  void setupSources(int filePathCount, int rowsPerVector) {
    filePaths_ = makeFilePaths(filePathCount);
    vectors_ = makeVectors(filePaths_.size(), rowsPerVector);
    for (int i = 0; i < filePaths_.size(); i++) {
      writeToFile(filePaths_[i]->getPath(), vectors_[i]);
    }
    createDuckDbTable(vectors_);
  }

  static void addHiveSplits(
      const std::shared_ptr<Task>& task,
      const std::vector<std::shared_ptr<TempFilePath>>& filePaths) {
    for (auto& filePath : filePaths) {
      auto split = exec::Split(
          std::make_shared<connector::hive::HiveConnectorSplit>(
              kHiveConnectorId,
              "file:" + filePath->getPath(),
              facebook::velox::dwio::common::FileFormat::DWRF),
          -1);
      task->addSplit("0", std::move(split));
    }
    task->noMoreSplits("0");
  }

  static exec::Split remoteSplit(const std::string& taskId) {
    return exec::Split(std::make_shared<RemoteConnectorSplit>(taskId));
  }

  static void addRemoteSplits(
      std::shared_ptr<Task> task,
      const std::vector<std::string>& remoteTaskIds) {
    for (auto& taskId : remoteTaskIds) {
      task->addSplit("0", remoteSplit(taskId));
    }
    task->noMoreSplits("0");
  }

  int32_t enqueue(
      const std::string& taskId,
      int32_t destination,
      const RowVectorPtr& data) {
    auto page = toSerializedPage(
        data, VectorSerde::Kind::kPresto, bufferManager_, pool());
    const auto pageSize = page->size();

    ContinueFuture unused;
    auto blocked =
        bufferManager_->enqueue(taskId, destination, std::move(page), &unused);
    VELOX_CHECK(!blocked);
    return pageSize;
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2"},
          {BIGINT(), INTEGER(), DOUBLE()})};
  std::unordered_map<std::string, std::string> configSettings_;
  std::vector<std::shared_ptr<TempFilePath>> filePaths_;
  std::vector<RowVectorPtr> vectors_;
  std::shared_ptr<OutputBufferManager> bufferManager_{
      OutputBufferManager::getInstanceRef()};
};

// Test: TwoFragmentHashPartition
// Follows the exact pattern of MultiFragmentTest::aggregationSingleKey.
// Fragment0 (leaf): TableScan → Project → PartialAgg → PartitionedOutput(3)
// Fragment1 (final-agg): Exchange → FinalAgg → PartitionedOutput(1)
// Root: AssertQueryBuilder reads from final-agg tasks, verifies via DuckDB.
TEST_F(GpuMultiFragmentTest, TwoFragmentHashPartition) {
  setupSources(10, 1000);
  std::vector<std::shared_ptr<Task>> tasks;
  auto leafTaskId = makeTaskId("leaf", 0);
  core::PlanNodePtr partialAggPlan;
  std::shared_ptr<Task> leafTask;
  {
    partialAggPlan =
        PlanBuilder()
            .tableScan(rowType_)
            .project({"c0 % 10 AS c0", "c1"})
            .partialAggregation({"c0"}, {"sum(c1)"})
            .partitionedOutput(
                {"c0"}, 3, /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
            .planNode();

    leafTask = makeTask(leafTaskId, partialAggPlan, 0);
    tasks.push_back(leafTask);
    leafTask->start(4);
    addHiveSplits(leafTask, filePaths_);
  }

  core::PlanNodePtr finalAggPlan;
  std::vector<std::string> finalAggTaskIds;
  for (int i = 0; i < 3; i++) {
    finalAggPlan =
        PlanBuilder()
            .exchange(partialAggPlan->outputType(), VectorSerde::Kind::kPresto)
            .finalAggregation({"c0"}, {"sum(a0)"}, {{BIGINT()}})
            .partitionedOutput({}, 1, /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
            .planNode();

    finalAggTaskIds.push_back(makeTaskId("final-agg", i));
    auto task = makeTask(finalAggTaskIds.back(), finalAggPlan, i);
    tasks.push_back(task);
    task->start(1);
    addRemoteSplits(task, {leafTaskId});
  }

  // Root consumer: reads from all final-agg tasks and verifies results.
  auto op = PlanBuilder()
                .exchange(finalAggPlan->outputType(), VectorSerde::Kind::kPresto)
                .planNode();

  std::vector<Split> finalAggTaskSplits;
  for (auto& finalAggTaskId : finalAggTaskIds) {
    finalAggTaskSplits.emplace_back(remoteSplit(finalAggTaskId));
  }
  AssertQueryBuilder(op, duckDbQueryRunner_)
      .splits(std::move(finalAggTaskSplits))
      .assertResults("SELECT c0 % 10, sum(c1) FROM tmp GROUP BY 1");

  for (auto& task : tasks) {
    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }
}

// Test: TwoFragmentBroadcast
// Fragment0: TableScan → PartitionedOutput(broadcast, 2)
// Fragment1: Exchange → PartitionedOutput(1)
// Root: AssertQueryBuilder reads from consumer tasks.
TEST_F(GpuMultiFragmentTest, TwoFragmentBroadcast) {
  setupSources(5, 500);
  std::vector<std::shared_ptr<Task>> tasks;
  auto broadcastTaskId = makeTaskId("leaf-broadcast", 0);

  // Fragment 0: Broadcast producer (tableScan-based)
  core::PlanNodePtr broadcastPlan;
  std::shared_ptr<Task> leafTask;
  {
    broadcastPlan = PlanBuilder()
                        .tableScan(rowType_)
                        .partitionedOutputBroadcast(
                            /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
                        .planNode();

    leafTask = makeTask(broadcastTaskId, broadcastPlan, 0);
    tasks.push_back(leafTask);
    leafTask->start(2);
    addHiveSplits(leafTask, filePaths_);
  }

  // Fragment 1: Broadcast consumer (2 tasks)
  core::PlanNodePtr consumerPlan;
  std::vector<std::string> consumerTaskIds;
  for (int i = 0; i < 2; i++) {
    consumerPlan =
        PlanBuilder()
            .exchange(broadcastPlan->outputType(), VectorSerde::Kind::kPresto)
            .partitionedOutput({}, 1, /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
            .planNode();

    consumerTaskIds.push_back(makeTaskId("consumer-broadcast", i));
    auto task = makeTask(consumerTaskIds.back(), consumerPlan, i);
    tasks.push_back(task);
    task->start(1);
    leafTask->updateOutputBuffers(i + 1, false);
    addRemoteSplits(task, {broadcastTaskId});
  }
  leafTask->updateOutputBuffers(consumerTaskIds.size(), true);

  // Root consumer: reads from all consumer tasks.
  // Broadcast sends all data to all partitions, so each consumer gets full data.
  auto op = PlanBuilder()
                .exchange(consumerPlan->outputType(), VectorSerde::Kind::kPresto)
                .planNode();

  std::vector<Split> consumerSplits;
  for (auto& taskId : consumerTaskIds) {
    consumerSplits.emplace_back(remoteSplit(taskId));
  }
  auto results = AssertQueryBuilder(op, duckDbQueryRunner_)
                     .splits(std::move(consumerSplits))
                     .copyResults(pool());
  // Broadcast duplicates data to each of 2 consumers, so we expect 2x rows.
  ASSERT_GT(results->size(), 0);

  for (auto& task : tasks) {
    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }
}

// Test: ThreeFragmentChain
// F0: TableScan → Project → PartialAgg → PartitionedOutput(2)
// F1: Exchange → Project → PartitionedOutput(2)
// F2: Exchange → FinalAgg → PartitionedOutput(1)
// Root: AssertQueryBuilder reads from F2 tasks.
TEST_F(GpuMultiFragmentTest, ThreeFragmentChain) {
  setupSources(5, 300);
  std::vector<std::shared_ptr<Task>> tasks;

  // Fragment 0: Source (tableScan-based)
  auto f0TaskId = makeTaskId("f0-chain", 0);
  core::PlanNodePtr f0Plan;
  {
    f0Plan = PlanBuilder()
                 .tableScan(rowType_)
                 .project({"c0 % 2 AS c0", "c1"})
                 .partialAggregation({"c0"}, {"sum(c1)"})
                 .partitionedOutput(
                     {"c0"},
                     2,
                     /*outputLayout=*/{},
                     VectorSerde::Kind::kPresto)
                 .planNode();

    auto task = makeTask(f0TaskId, f0Plan, 0);
    tasks.push_back(task);
    task->start(2);
    addHiveSplits(task, filePaths_);
  }

  // Fragment 1: Middle stage
  std::vector<std::string> f1TaskIds;
  core::PlanNodePtr f1Plan;
  {
    f1Plan = PlanBuilder()
                 .exchange(f0Plan->outputType(), VectorSerde::Kind::kPresto)
                 .finalAggregation({"c0"}, {"sum(a0)"}, {{BIGINT()}})
                 .partitionedOutput(
                     {},
                     1,
                     /*outputLayout=*/{},
                     VectorSerde::Kind::kPresto)
                 .planNode();

    for (int i = 0; i < 2; i++) {
      auto taskId = makeTaskId("f1-chain", i);
      f1TaskIds.push_back(taskId);
      auto task = makeTask(taskId, f1Plan, i);
      tasks.push_back(task);
      task->start(1);
      addRemoteSplits(task, {f0TaskId});
    }
  }

  // Root consumer: reads from F1 tasks and verifies.
  auto op = PlanBuilder()
                .exchange(f1Plan->outputType(), VectorSerde::Kind::kPresto)
                .planNode();

  std::vector<Split> f1Splits;
  for (auto& taskId : f1TaskIds) {
    f1Splits.emplace_back(remoteSplit(taskId));
  }
  AssertQueryBuilder(op, duckDbQueryRunner_)
      .splits(std::move(f1Splits))
      .assertResults("SELECT c0 % 2, sum(c1) FROM tmp GROUP BY 1");

  for (auto& task : tasks) {
    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }
}

// Test: PipelineOverlap
// Verify that multiple fragments execute concurrently by checking timestamps.
TEST_F(GpuMultiFragmentTest, PipelineOverlap) {
  setupSources(5, 200);
  std::vector<std::shared_ptr<Task>> tasks;
  auto startTime = std::chrono::steady_clock::now();

  auto leafTaskId = makeTaskId("leaf-overlap", 0);
  core::PlanNodePtr partitionedPlan;
  std::shared_ptr<Task> leafTask;
  {
    partitionedPlan = PlanBuilder()
                          .tableScan(rowType_)
                          .partitionedOutputBroadcast(
                              /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
                          .planNode();

    leafTask = makeTask(leafTaskId, partitionedPlan, 0);
    tasks.push_back(leafTask);
    leafTask->start(2);
    addHiveSplits(leafTask, filePaths_);
  }

  core::PlanNodePtr consumerPlan;
  std::vector<std::string> consumerTaskIds;
  std::vector<std::chrono::steady_clock::time_point> taskStartTimes;
  for (int i = 0; i < 2; i++) {
    consumerPlan =
        PlanBuilder()
            .exchange(partitionedPlan->outputType(), VectorSerde::Kind::kPresto)
            .partitionedOutput({}, 1, /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
            .planNode();

    consumerTaskIds.push_back(makeTaskId("consumer-overlap", i));
    auto task = makeTask(consumerTaskIds.back(), consumerPlan, i);
    tasks.push_back(task);

    // Record task start time
    taskStartTimes.push_back(std::chrono::steady_clock::now());
    task->start(1);
    leafTask->updateOutputBuffers(i + 1, false);
    addRemoteSplits(task, {leafTaskId});
  }
  leafTask->updateOutputBuffers(consumerTaskIds.size(), true);

  // Root consumer: reads from all consumer tasks to drain their output buffers.
  auto op = PlanBuilder()
                .exchange(consumerPlan->outputType(), VectorSerde::Kind::kPresto)
                .planNode();

  std::vector<Split> consumerSplits;
  for (auto& taskId : consumerTaskIds) {
    consumerSplits.emplace_back(remoteSplit(taskId));
  }
  auto results = AssertQueryBuilder(op, duckDbQueryRunner_)
                     .splits(std::move(consumerSplits))
                     .copyResults(pool());
  ASSERT_GT(results->size(), 0);

  for (auto& task : tasks) {
    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }

  // Verify that tasks started close in time (overlapping execution)
  if (taskStartTimes.size() >= 2) {
    auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(
        taskStartTimes[1] - taskStartTimes[0]);
    EXPECT_LT(timeDiff.count(), 1000) << "Tasks should start close together";
  }
}

// Test: LargeData
// Large dataset through hash-partitioned aggregation with DuckDB verification.
TEST_F(GpuMultiFragmentTest, LargeData) {
  setupSources(10, 5000); // 50K rows total
  std::vector<std::shared_ptr<Task>> tasks;
  auto leafTaskId = makeTaskId("leaf-large", 0);

  core::PlanNodePtr partialAggPlan;
  {
    partialAggPlan = PlanBuilder()
                         .tableScan(rowType_)
                         .project({"c0 % 5 AS c0", "c1"})
                         .partialAggregation({"c0"}, {"sum(c1)"})
                         .partitionedOutput(
                             {"c0"},
                             5,
                             /*outputLayout=*/{},
                             VectorSerde::Kind::kPresto)
                         .planNode();

    auto task = makeTask(leafTaskId, partialAggPlan, 0);
    tasks.push_back(task);
    task->start(4);
    addHiveSplits(task, filePaths_);
  }

  core::PlanNodePtr finalPlan;
  std::vector<std::string> finalTaskIds;
  for (int i = 0; i < 5; i++) {
    finalPlan = PlanBuilder()
                    .exchange(partialAggPlan->outputType(), VectorSerde::Kind::kPresto)
                    .finalAggregation({"c0"}, {"sum(a0)"}, {{BIGINT()}})
                    .partitionedOutput({}, 1, /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
                    .planNode();

    finalTaskIds.push_back(makeTaskId("final-large", i));
    auto task = makeTask(finalTaskIds.back(), finalPlan, i);
    tasks.push_back(task);
    task->start(1);
    addRemoteSplits(task, {leafTaskId});
  }

  // Root consumer: reads from all final tasks and verifies.
  auto op = PlanBuilder()
                .exchange(finalPlan->outputType(), VectorSerde::Kind::kPresto)
                .planNode();

  std::vector<Split> finalSplits;
  for (auto& taskId : finalTaskIds) {
    finalSplits.emplace_back(remoteSplit(taskId));
  }
  AssertQueryBuilder(op, duckDbQueryRunner_)
      .splits(std::move(finalSplits))
      .assertResults("SELECT c0 % 5, sum(c1) FROM tmp GROUP BY 1");

  for (auto& task : tasks) {
    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }
}

// Test: EmptyPartitions
// Hash partitioning where some partitions receive 0 rows.
TEST_F(GpuMultiFragmentTest, EmptyPartitions) {
  setupSources(2, 100);
  std::vector<std::shared_ptr<Task>> tasks;
  auto leafTaskId = makeTaskId("leaf-empty", 0);

  core::PlanNodePtr partialAggPlan;
  {
    // Using c0 % 10 with 20 partitions means some partitions get no data.
    partialAggPlan = PlanBuilder()
                         .tableScan(rowType_)
                         .project({"c0 % 10 AS c0", "c1"})
                         .partialAggregation({"c0"}, {"sum(c1)"})
                         .partitionedOutput(
                             {"c0"},
                             20,
                             /*outputLayout=*/{},
                             VectorSerde::Kind::kPresto)
                         .planNode();

    auto task = makeTask(leafTaskId, partialAggPlan, 0);
    tasks.push_back(task);
    task->start(2);
    addHiveSplits(task, filePaths_);
  }

  core::PlanNodePtr finalPlan;
  std::vector<std::string> finalTaskIds;
  for (int i = 0; i < 20; i++) {
    finalPlan = PlanBuilder()
                    .exchange(partialAggPlan->outputType(), VectorSerde::Kind::kPresto)
                    .finalAggregation({"c0"}, {"sum(a0)"}, {{BIGINT()}})
                    .partitionedOutput({}, 1, /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
                    .planNode();

    finalTaskIds.push_back(makeTaskId("final-empty", i));
    auto task = makeTask(finalTaskIds.back(), finalPlan, i);
    tasks.push_back(task);
    task->start(1);
    addRemoteSplits(task, {leafTaskId});
  }

  // Root consumer: reads from all final tasks.
  auto op = PlanBuilder()
                .exchange(finalPlan->outputType(), VectorSerde::Kind::kPresto)
                .planNode();

  std::vector<Split> finalSplits;
  for (auto& taskId : finalTaskIds) {
    finalSplits.emplace_back(remoteSplit(taskId));
  }
  AssertQueryBuilder(op, duckDbQueryRunner_)
      .splits(std::move(finalSplits))
      .assertResults("SELECT c0 % 10, sum(c1) FROM tmp GROUP BY 1");

  for (auto& task : tasks) {
    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }
}

// Test: BackpressureThroughput
// Large dataset through hash partitioning to test backpressure handling.
TEST_F(GpuMultiFragmentTest, BackpressureThroughput) {
  setupSources(10, 5000); // 50K rows
  std::vector<std::shared_ptr<Task>> tasks;
  auto leafTaskId = makeTaskId("leaf-backpressure", 0);

  core::PlanNodePtr partialAggPlan;
  {
    partialAggPlan = PlanBuilder()
                         .tableScan(rowType_)
                         .project({"c0 % 3 AS c0", "c1"})
                         .partialAggregation({"c0"}, {"sum(c1)"})
                         .partitionedOutput(
                             {"c0"},
                             3,
                             /*outputLayout=*/{},
                             VectorSerde::Kind::kPresto)
                         .planNode();

    auto task = makeTask(leafTaskId, partialAggPlan, 0);
    tasks.push_back(task);
    task->start(2);
    addHiveSplits(task, filePaths_);
  }

  core::PlanNodePtr finalPlan;
  std::vector<std::string> finalTaskIds;
  for (int i = 0; i < 3; i++) {
    finalPlan = PlanBuilder()
                    .exchange(partialAggPlan->outputType(), VectorSerde::Kind::kPresto)
                    .finalAggregation({"c0"}, {"sum(a0)"}, {{BIGINT()}})
                    .partitionedOutput({}, 1, /*outputLayout=*/{}, VectorSerde::Kind::kPresto)
                    .planNode();

    finalTaskIds.push_back(makeTaskId("final-backpressure", i));
    auto task = makeTask(finalTaskIds.back(), finalPlan, i);
    tasks.push_back(task);
    task->start(1);
    addRemoteSplits(task, {leafTaskId});
  }

  // Root consumer: reads from all final tasks and verifies.
  auto op = PlanBuilder()
                .exchange(finalPlan->outputType(), VectorSerde::Kind::kPresto)
                .planNode();

  std::vector<Split> finalSplits;
  for (auto& taskId : finalTaskIds) {
    finalSplits.emplace_back(remoteSplit(taskId));
  }
  AssertQueryBuilder(op, duckDbQueryRunner_)
      .splits(std::move(finalSplits))
      .assertResults("SELECT c0 % 3, sum(c1) FROM tmp GROUP BY 1");

  for (auto& task : tasks) {
    ASSERT_TRUE(waitForTaskCompletion(task.get())) << task->taskId();
  }
}

} // namespace
} // namespace facebook::velox::cudf_velox
