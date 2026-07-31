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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

#include <unistd.h>

#include <filesystem>
#include <future>
#include <set>
#include <string>
#include <string_view>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

namespace fs = std::filesystem;

constexpr std::string_view kSpillDirectoryPrefix{"velox-cudf-topn-spill-"};

bool hasSpillDirectoryPrefix(const fs::path& path) {
  return path.filename().string().rfind(kSpillDirectoryPrefix, 0) == 0;
}

std::vector<fs::path> findSpillDirectories(const fs::path& root) {
  std::vector<fs::path> paths;
  std::error_code error;
  if (!fs::exists(root, error)) {
    return paths;
  }

  fs::recursive_directory_iterator iterator(
      root, fs::directory_options::skip_permission_denied, error);
  const fs::recursive_directory_iterator end;
  while (iterator != end) {
    if (!error && iterator->is_directory(error) &&
        hasSpillDirectoryPrefix(iterator->path())) {
      paths.push_back(iterator->path());
    }
    error.clear();
    iterator.increment(error);
  }
  return paths;
}

std::set<fs::path> findTopLevelProcessSpillDirectories() {
  const auto processPrefix =
      std::string(kSpillDirectoryPrefix) + std::to_string(::getpid()) + "-";
  std::set<fs::path> paths;
  std::error_code error;
  fs::directory_iterator iterator(
      fs::temp_directory_path(),
      fs::directory_options::skip_permission_denied,
      error);
  const fs::directory_iterator end;
  while (iterator != end) {
    if (!error && iterator->is_directory(error) &&
        iterator->path().filename().string().rfind(processPrefix, 0) == 0) {
      paths.insert(iterator->path());
    }
    error.clear();
    iterator.increment(error);
  }
  return paths;
}

bool usedCudfTopNRowNumber(const TaskStats& stats) {
  for (const auto& pipelineStats : stats.pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType == "CudfTopNRowNumber") {
        return true;
      }
    }
  }
  return false;
}

const OperatorStats* cudfTopNRowNumberStats(const TaskStats& stats) {
  for (const auto& pipelineStats : stats.pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType == "CudfTopNRowNumber") {
        return &operatorStats;
      }
    }
  }
  return nullptr;
}

class CudfTopNRowNumberTest : public OperatorTestBase {
 protected:
  static constexpr vector_size_t kNumRows = 32;

  void SetUp() override {
    OperatorTestBase::SetUp();
    filesystems::registerLocalFileSystem();
    previousAllowCpuFallback_ =
        cudf_velox::CudfConfig::getInstance().allowCpuFallback;
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback =
        previousAllowCpuFallback_;
    OperatorTestBase::TearDown();
  }

  core::PlanNodePtr makePlan() {
    auto data = makeRowVector(
        {"p", "s", "v"},
        {makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return row; }),
         makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return kNumRows - row; }),
         makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return row * 10; })});

    return PlanBuilder()
        .values({data})
        .topNRowNumber({"p"}, {"s"}, 1, true)
        .planNode();
  }

  RowVectorPtr makeExpectedResult() {
    return makeRowVector(
        {"p", "s", "v", "row_number"},
        {makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return row; }),
         makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return kNumRows - row; }),
         makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return row * 10; }),
         makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t /*row*/) { return 1; })});
  }

  std::unique_ptr<TaskCursor> makeSpillingCursor(const std::string& spillRoot) {
    CursorParameters params;
    params.planNode = makePlan();
    params.maxDrivers = 1;
    params.serialExecution = true;
    params.spillDirectory = spillRoot;
    params.queryConfigs = {
        {cudf_velox::CudfConfig::kCudfEnabled, "true"},
        {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
        {cudf_velox::CudfConfig::kCudfTopNRowNumberForceSpill, "true"}};
    return TaskCursor::create(params);
  }

 private:
  bool previousAllowCpuFallback_{true};
};

TEST_F(CudfTopNRowNumberTest, scopedAdmissionCreditLifetime) {
  const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
  ASSERT_TRUE(headroom.cudaValid);
  const std::string scope{"topn-credit-lifetime-test"};
  auto owner = std::make_shared<int>(1);
  std::atomic<bool> consumed{false};
  auto registration = cudf_velox::registerScopedDeviceMemoryAdmissionCredit(
      scope, headroom.device, 123, owner, [&] {
        consumed.store(true, std::memory_order_release);
      });
  EXPECT_EQ(
      cudf_velox::scopedDeviceMemoryAdmissionCreditBytes(
          scope, headroom.device),
      123);

  auto moved = std::move(registration);
  EXPECT_EQ(
      cudf_velox::scopedDeviceMemoryAdmissionCreditBytes(
          scope, headroom.device),
      123);
  const auto stream = cudf_velox::cudfGlobalStreamPool().get_stream();
  EXPECT_EQ(
      cudf_velox::consumeScopedDeviceMemoryAdmissionCreditAfterStream(
          scope, headroom.device, 123, stream),
      123);
  EXPECT_EQ(
      cudf_velox::scopedDeviceMemoryAdmissionCreditBytes(
          scope, headroom.device),
      0);
  stream.synchronize();
  EXPECT_TRUE(consumed.load(std::memory_order_acquire));
  moved.release();
  EXPECT_EQ(
      cudf_velox::scopedDeviceMemoryAdmissionCreditBytes(
          scope, headroom.device),
      0);
}

TEST_F(CudfTopNRowNumberTest, spillUsesTaskRootAndCleansUp) {
  const auto spillRoot = TempDirectoryPath::create();
  const auto preexistingTopLevelSpills = findTopLevelProcessSpillDirectories();
  auto cursor = makeSpillingCursor(spillRoot->getPath());
  ASSERT_TRUE(cursor->moveNext());

  const fs::path taskSpillRoot(cursor->task()->spillDirectory());
  EXPECT_EQ(taskSpillRoot.parent_path(), fs::path(spillRoot->getPath()));

  const auto activeSpillDirectories =
      findSpillDirectories(spillRoot->getPath());
  ASSERT_EQ(activeSpillDirectories.size(), 1);
  EXPECT_EQ(activeSpillDirectories.front().parent_path(), taskSpillRoot);
  EXPECT_TRUE(hasSpillDirectoryPrefix(activeSpillDirectories.front()));
  EXPECT_EQ(findTopLevelProcessSpillDirectories(), preexistingTopLevelSpills)
      << "CudfTopNRowNumber created a spill directory outside the Task root";

  std::vector<RowVectorPtr> actualResults{cursor->current()};
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  assertEqualResults({makeExpectedResult()}, actualResults);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
  EXPECT_EQ(findTopLevelProcessSpillDirectories(), preexistingTopLevelSpills);

  auto task = cursor->task();
  EXPECT_TRUE(usedCudfTopNRowNumber(task->taskStats()));

  actualResults.clear();
  cursor.reset();
  task.reset();
  EXPECT_FALSE(fs::exists(taskSpillRoot));
  EXPECT_TRUE(fs::is_empty(spillRoot->getPath()));
}

TEST_F(CudfTopNRowNumberTest, earlyCloseCleansUp) {
  const auto spillRoot = TempDirectoryPath::create();
  const auto preexistingTopLevelSpills = findTopLevelProcessSpillDirectories();
  auto cursor = makeSpillingCursor(spillRoot->getPath());
  ASSERT_TRUE(cursor->moveNext());

  auto task = cursor->task();
  const fs::path taskSpillRoot(task->spillDirectory());
  EXPECT_EQ(taskSpillRoot.parent_path(), fs::path(spillRoot->getPath()));
  const auto activeSpillDirectories =
      findSpillDirectories(spillRoot->getPath());
  ASSERT_EQ(activeSpillDirectories.size(), 1);
  EXPECT_EQ(activeSpillDirectories.front().parent_path(), taskSpillRoot);
  EXPECT_EQ(findTopLevelProcessSpillDirectories(), preexistingTopLevelSpills);

  // SingleThreadedTaskCursor synchronously cancels the Task in its destructor.
  // The Task closes off-thread Drivers before requestCancel().wait() returns.
  cursor.reset();
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
  EXPECT_EQ(findTopLevelProcessSpillDirectories(), preexistingTopLevelSpills);
  EXPECT_TRUE(usedCudfTopNRowNumber(task->taskStats()));

  task.reset();
  EXPECT_FALSE(fs::exists(taskSpillRoot));
  EXPECT_TRUE(fs::is_empty(spillRoot->getPath()));
}

TEST_F(CudfTopNRowNumberTest, retainsCandidatesWhenHeadroomIsSafe) {
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = makePlan();
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"}};

  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actualResults;
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  assertEqualResults({makeExpectedResult()}, actualResults);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());

  const auto taskStats = cursor->task()->taskStats();
  const auto* operatorStats = cudfTopNRowNumberStats(taskStats);
  ASSERT_NE(operatorStats, nullptr);
  EXPECT_GT(
      operatorStats->runtimeStats.at("topNPressureRetainedMerges").sum, 0);
  EXPECT_EQ(operatorStats->runtimeStats.at("topNSpillRuns").sum, 0);
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNPressurePreMergeSpills").sum, 0);
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNPressurePostMergeSpills").sum, 0);
}

TEST_F(CudfTopNRowNumberTest, geometricallyMergesCandidateRuns) {
  constexpr int32_t kBatches = 16;
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kBatches);
  for (int32_t batch = 0; batch < kBatches; ++batch) {
    inputs.push_back(makeRowVector(
        {"p", "score", "payload"},
        {makeFlatVector<int64_t>(
             kNumRows,
             [batch](vector_size_t row) { return batch * kNumRows + row; }),
         makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return row; }),
         makeFlatVector<int64_t>(kNumRows, [batch](vector_size_t row) {
           return batch * kNumRows + row;
         })}));
  }

  const auto plan = PlanBuilder()
                        .values(inputs)
                        .topNRowNumber({"p"}, {"score"}, 1, false)
                        .planNode();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"}};

  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actualResults;
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  assertEqualResults(inputs, actualResults);

  const auto taskStats = cursor->task()->taskStats();
  const auto* operatorStats = cudfTopNRowNumberStats(taskStats);
  ASSERT_NE(operatorStats, nullptr);
  for (const auto* name :
       {"topNCandidateLevelMerges",
        "topNCandidateReductionCalls",
        "topNCandidateRowsReduced",
        "topNSpillRuns"}) {
    ASSERT_TRUE(operatorStats->runtimeStats.contains(name)) << name;
  }
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNCandidateLevelMerges").sum,
      kBatches - 1);
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNCandidateReductionCalls").sum,
      2 * kBatches - 1);
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNCandidateRowsReduced").sum,
      5 * kBatches * kNumRows);
  EXPECT_EQ(operatorStats->runtimeStats.at("topNSpillRuns").sum, 0);
}

TEST_F(CudfTopNRowNumberTest, geometricallyMergesBelowCandidateRunThreshold) {
  constexpr int32_t kBatches = 16;
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kBatches);
  for (int32_t batch = 0; batch < kBatches; ++batch) {
    inputs.push_back(makeRowVector(
        {"p", "score"},
        {makeFlatVector<int64_t>(
             kNumRows,
             [batch](vector_size_t row) { return batch * kNumRows + row; }),
         makeFlatVector<int64_t>(
             kNumRows, [](vector_size_t row) { return row; })}));
  }

  const auto plan = PlanBuilder()
                        .values(inputs)
                        .topNRowNumber({"p"}, {"score"}, 1, false)
                        .planNode();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
       std::to_string(1ULL << 30)}};

  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actualResults;
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  assertEqualResults(inputs, actualResults);

  const auto taskStats = cursor->task()->taskStats();
  const auto* operatorStats = cudfTopNRowNumberStats(taskStats);
  ASSERT_NE(operatorStats, nullptr);
  ASSERT_TRUE(operatorStats->runtimeStats.contains("topNCandidateLevelMerges"));
  ASSERT_TRUE(operatorStats->runtimeStats.contains("topNCandidateRowsReduced"));
  ASSERT_TRUE(operatorStats->runtimeStats.contains("topNSpillRuns"));
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNCandidateLevelMerges").sum,
      kBatches - 1);
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNCandidateRowsReduced").sum,
      5 * kBatches * kNumRows);
  EXPECT_EQ(operatorStats->runtimeStats.at("topNSpillRuns").sum, 0);
}

TEST_F(CudfTopNRowNumberTest, geometricMergePreservesRowNumberInputOrder) {
  std::vector<RowVectorPtr> inputs;
  for (int64_t batch = 0; batch < 7; ++batch) {
    inputs.push_back(makeRowVector(
        {"p", "score", "payload"},
        {makeFlatVector<int64_t>({1}),
         makeFlatVector<int64_t>({10}),
         makeFlatVector<int64_t>({batch})}));
  }

  const auto plan = PlanBuilder()
                        .values(inputs)
                        .topNRowNumber({"p"}, {"score"}, 1, false)
                        .planNode();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"}};

  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actualResults;
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  auto expected = makeRowVector(
      {"p", "score", "payload"},
      {makeFlatVector<int64_t>({1}),
       makeFlatVector<int64_t>({10}),
       makeFlatVector<int64_t>({0})});
  assertEqualResults({expected}, actualResults);
}

TEST_F(CudfTopNRowNumberTest, batchCandidatesFeedExactFinalRank) {
  auto batch1 = makeRowVector(
      {"p", "score", "payload"},
      {makeFlatVector<int64_t>({1, 1, 2, 3}),
       makeFlatVector<int64_t>({10, 8, 5, 7}),
       makeFlatVector<std::string>({"old", "discard", "winner", "tie-a"})});
  auto batch2 = makeRowVector(
      {"p", "score", "payload"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({12, 4, 7}),
       makeFlatVector<std::string>({"new", "discard", "tie-b"})});

  auto source = PlanBuilder().values({batch1, batch2}).planNode();
  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys{
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "p")};
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys{
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "score")};
  const std::vector<core::SortOrder> sortingOrders{core::kDescNullsLast};
  auto partial = std::make_shared<core::TopNRowNumberNode>(
      "partial",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      std::nullopt,
      1,
      source,
      true);
  auto final = std::make_shared<core::TopNRowNumberNode>(
      "final",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      "rank",
      1,
      partial);

  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = final;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"}};

  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actualResults;
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  auto expected = makeRowVector(
      {"p", "score", "payload", "rank"},
      {makeFlatVector<int64_t>({1, 2, 3, 3}),
       makeFlatVector<int64_t>({12, 5, 7, 7}),
       makeFlatVector<std::string>({"new", "winner", "tie-a", "tie-b"}),
       makeFlatVector<int64_t>({1, 1, 1, 1})});
  assertEqualResults({expected}, actualResults);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());

  uint64_t batchCandidateBatches = 0;
  uint64_t batchCandidateInputBatches = 0;
  uint64_t batchCandidateFlushes = 0;
  uint64_t spillRuns = 0;
  uint64_t candidateLevelMerges = 0;
  uint64_t rankMembershipFilterCalls = 0;
  for (const auto& pipelineStats : cursor->task()->taskStats().pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType != "CudfTopNRowNumber") {
        continue;
      }
      batchCandidateBatches +=
          operatorStats.runtimeStats.at("topNBatchCandidateBatches").sum;
      batchCandidateInputBatches +=
          operatorStats.runtimeStats.at("topNBatchCandidateInputBatches").sum;
      batchCandidateFlushes +=
          operatorStats.runtimeStats.at("topNBatchCandidateFlushes").sum;
      spillRuns += operatorStats.runtimeStats.at("topNSpillRuns").sum;
      candidateLevelMerges +=
          operatorStats.runtimeStats.at("topNCandidateLevelMerges").sum;
      rankMembershipFilterCalls +=
          operatorStats.runtimeStats.at("topNRankMembershipFilterCalls").sum;
    }
  }
  EXPECT_EQ(batchCandidateBatches, 1);
  EXPECT_EQ(batchCandidateInputBatches, 2);
  EXPECT_EQ(batchCandidateFlushes, 1);
  EXPECT_EQ(spillRuns, 0);
  EXPECT_EQ(candidateLevelMerges, 0);
  EXPECT_GT(rankMembershipFilterCalls, 0);
}

TEST_F(CudfTopNRowNumberTest, batchCandidatesBypassUnderPressure) {
  auto input = makeRowVector(
      {"p", "score", "payload"},
      {makeFlatVector<int64_t>({1, 1, 2, 2}),
       makeFlatVector<int64_t>({10, 8, 7, 6}),
       makeFlatVector<std::string>({"keep", "also-keep", "a", "b"})});
  auto source = PlanBuilder().values({input}).planNode();
  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys{
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "p")};
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys{
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "score")};
  const std::vector<core::SortOrder> sortingOrders{core::kDescNullsLast};
  auto partial = std::make_shared<core::TopNRowNumberNode>(
      "partial",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      std::nullopt,
      1,
      source,
      true);

  const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
  ASSERT_TRUE(headroom.cudaValid);
  const auto allocatableBytes = headroom.allocatableBytes();
  ASSERT_GT(allocatableBytes, 0);
  auto pressureReservation = cudf_velox::tryAcquireDeviceMemoryAdmission(
      headroom.device, allocatableBytes, allocatableBytes);
  ASSERT_TRUE(pressureReservation);

  CursorParameters params;
  params.planNode = partial;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.queryConfigs = {{cudf_velox::CudfConfig::kCudfEnabled, "true"}};

  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actualResults;
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  assertEqualResults({input}, actualResults);

  const auto taskStats = cursor->task()->taskStats();
  const auto* operatorStats = cudfTopNRowNumberStats(taskStats);
  ASSERT_NE(operatorStats, nullptr);
  EXPECT_EQ(operatorStats->runtimeStats.at("topNPressureBypassBatches").sum, 1);
  EXPECT_EQ(
      operatorStats->runtimeStats.at("topNPressureBypassRows").sum,
      input->size());
  EXPECT_GT(operatorStats->runtimeStats.at("topNPressureBypassBytes").sum, 0);
  EXPECT_EQ(operatorStats->runtimeStats.at("topNBatchCandidateBatches").sum, 0);
}

TEST_F(
    CudfTopNRowNumberTest,
    conditionalPassthroughBypassesWorkspaceAdmission) {
  auto input = makeRowVector(
      {"__gluten_mpp_topn_active_test", "p", "singleton", "score", "payload"},
      {makeFlatVector<bool>({false, false, false}),
       makeFlatVector<int64_t>({9, 9, 9}),
       makeFlatVector<int64_t>({101, 102, 103}),
       makeFlatVector<int64_t>({100, 1, 999}),
       makeFlatVector<std::string>({"keep-a", "keep-b", "keep-c"})});
  auto source = PlanBuilder().values({input}).planNode();
  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys{
      std::make_shared<core::FieldAccessTypedExpr>(
          BOOLEAN(), "__gluten_mpp_topn_active_test"),
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "p"),
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "singleton")};
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys{
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "score")};
  const std::vector<core::SortOrder> sortingOrders{core::kDescNullsLast};
  auto topN = std::make_shared<core::TopNRowNumberNode>(
      "topn",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      std::nullopt,
      1,
      source,
      false,
      0);

  const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
  ASSERT_TRUE(headroom.cudaValid);
  const auto allocatableBytes = headroom.allocatableBytes();
  ASSERT_GT(allocatableBytes, 0);
  auto pressureReservation = cudf_velox::tryAcquireDeviceMemoryAdmission(
      headroom.device, allocatableBytes, allocatableBytes);
  ASSERT_TRUE(pressureReservation);

  auto future = std::async(std::launch::async, [topN] {
    CursorParameters params;
    params.planNode = topN;
    params.maxDrivers = 1;
    params.serialExecution = true;
    params.queryConfigs = {{cudf_velox::CudfConfig::kCudfEnabled, "true"}};
    auto cursor = TaskCursor::create(params);
    vector_size_t rows = 0;
    while (cursor->moveNext()) {
      rows += cursor->current()->size();
    }
    const auto taskStats = cursor->task()->taskStats();
    const auto* operatorStats = cudfTopNRowNumberStats(taskStats);
    VELOX_CHECK_NOT_NULL(operatorStats);
    return std::pair{
        rows,
        std::pair{
            operatorStats->runtimeStats.at("topNConditionalBlockingAdmissions")
                .sum,
            operatorStats->runtimeStats
                .at("topNConditionalPassthroughAdmissionBypasses")
                .sum}};
  });

  const auto status = future.wait_for(std::chrono::seconds(30));
  pressureReservation.reset();
  ASSERT_EQ(status, std::future_status::ready);
  const auto [rows, admissionStats] = future.get();
  EXPECT_EQ(rows, input->size());
  EXPECT_EQ(admissionStats.first, 0);
  EXPECT_EQ(admissionStats.second, 1);
}

TEST_F(
    CudfTopNRowNumberTest,
    conditionalBatchCandidatesPreservePassthroughRows) {
  auto inactiveBatch = makeRowVector(
      {"__gluten_mpp_topn_active_test", "p", "singleton", "score", "payload"},
      {makeFlatVector<bool>({false, false}),
       makeFlatVector<int64_t>({9, 9}),
       makeFlatVector<int64_t>({101, 102}),
       makeFlatVector<int64_t>({100, 1}),
       makeFlatVector<std::string>({"keep-a", "keep-b"})});
  auto activeBatch = makeRowVector(
      {"__gluten_mpp_topn_active_test", "p", "singleton", "score", "payload"},
      {makeFlatVector<bool>({true, true, true}),
       makeFlatVector<int64_t>({1, 1, 2}),
       makeFlatVector<int64_t>({0, 0, 0}),
       makeFlatVector<int64_t>({10, 8, 7}),
       makeFlatVector<std::string>({"old", "discard", "tie-a"})});
  auto mixedBatch = makeRowVector(
      {"__gluten_mpp_topn_active_test", "p", "singleton", "score", "payload"},
      {makeFlatVector<bool>({true, true, false}),
       makeFlatVector<int64_t>({1, 2, 9}),
       makeFlatVector<int64_t>({0, 0, 103}),
       makeFlatVector<int64_t>({12, 7, 999}),
       makeFlatVector<std::string>({"new", "tie-b", "keep-c"})});

  auto source =
      PlanBuilder().values({inactiveBatch, activeBatch, mixedBatch}).planNode();
  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys{
      std::make_shared<core::FieldAccessTypedExpr>(
          BOOLEAN(), "__gluten_mpp_topn_active_test"),
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "p"),
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "singleton")};
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys{
      std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "score")};
  const std::vector<core::SortOrder> sortingOrders{core::kDescNullsLast};
  auto partial = std::make_shared<core::TopNRowNumberNode>(
      "partial",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      std::nullopt,
      1,
      source,
      true,
      0);
  auto final = std::make_shared<core::TopNRowNumberNode>(
      "final",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      std::nullopt,
      1,
      partial,
      false,
      0);

  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = final;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
       "1073741824"}};

  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actualResults;
  while (cursor->moveNext()) {
    actualResults.push_back(cursor->current());
  }
  auto expected = makeRowVector(
      {"__gluten_mpp_topn_active_test", "p", "singleton", "score", "payload"},
      {makeFlatVector<bool>({false, false, false, true, true, true}),
       makeFlatVector<int64_t>({9, 9, 9, 1, 2, 2}),
       makeFlatVector<int64_t>({101, 102, 103, 0, 0, 0}),
       makeFlatVector<int64_t>({100, 1, 999, 12, 7, 7}),
       makeFlatVector<std::string>(
           {"keep-a", "keep-b", "keep-c", "new", "tie-a", "tie-b"})});
  assertEqualResults({expected}, actualResults);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());

  uint64_t batchCandidateBatches = 0;
  uint64_t batchCandidateInputBatches = 0;
  uint64_t batchCandidateFlushes = 0;
  uint64_t spillRuns = 0;
  uint64_t candidateLevelMerges = 0;
  for (const auto& pipelineStats : cursor->task()->taskStats().pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType != "CudfTopNRowNumber") {
        continue;
      }
      batchCandidateBatches +=
          operatorStats.runtimeStats.at("topNBatchCandidateBatches").sum;
      batchCandidateInputBatches +=
          operatorStats.runtimeStats.at("topNBatchCandidateInputBatches").sum;
      batchCandidateFlushes +=
          operatorStats.runtimeStats.at("topNBatchCandidateFlushes").sum;
      spillRuns += operatorStats.runtimeStats.at("topNSpillRuns").sum;
      candidateLevelMerges +=
          operatorStats.runtimeStats.at("topNCandidateLevelMerges").sum;
    }
  }
  EXPECT_EQ(batchCandidateBatches, 1);
  EXPECT_EQ(batchCandidateInputBatches, 2);
  EXPECT_EQ(batchCandidateFlushes, 1);
  EXPECT_EQ(spillRuns, 0);
  EXPECT_EQ(candidateLevelMerges, 0);
}

TEST_F(CudfTopNRowNumberTest, concurrentConditionalNestedPayload) {
  constexpr vector_size_t kRows = 4096;
  constexpr int32_t kBatches = 4;
  std::vector<RowVectorPtr> inputs;
  inputs.reserve(kBatches);
  for (int32_t batch = 0; batch < kBatches; ++batch) {
    inputs.push_back(makeRowVector(
        {"account",
         "profile",
         "title",
         "score",
         "payload",
         "__gluten_mpp_topn_active_test"},
        {makeFlatVector<std::string>(
             kRows,
             [batch](vector_size_t row) {
               return row % 2 == 0 ? fmt::format("account-{}", row % 256)
                                   : fmt::format("inactive-{}-{}", batch, row);
             }),
         makeFlatVector<std::string>(
             kRows,
             [](vector_size_t row) {
               return fmt::format("profile-{}", row % 8);
             }),
         makeFlatVector<std::string>(
             kRows,
             [](vector_size_t row) {
               return fmt::format("title-{}", row % 4);
             }),
         makeFlatVector<double>(
             kRows, [batch](vector_size_t row) { return batch * kRows + row; }),
         makeArrayVector<std::string>(
             kRows,
             [](vector_size_t row) { return row % 3 + 1; },
             [batch](vector_size_t row) {
               return fmt::format("payload-{}-{}", batch, row);
             }),
         makeFlatVector<bool>(
             kRows, [](vector_size_t row) { return row % 2 == 0; })}));
  }

  auto source = PlanBuilder().values(inputs).planNode();
  const std::vector<core::FieldAccessTypedExprPtr> partitionKeys{
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "account"),
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "profile"),
      std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "title"),
      std::make_shared<core::FieldAccessTypedExpr>(
          BOOLEAN(), "__gluten_mpp_topn_active_test")};
  const std::vector<core::FieldAccessTypedExprPtr> sortingKeys{
      std::make_shared<core::FieldAccessTypedExpr>(DOUBLE(), "score")};
  const std::vector<core::SortOrder> sortingOrders{core::kDescNullsLast};
  auto partial = std::make_shared<core::TopNRowNumberNode>(
      "partial",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      std::nullopt,
      1,
      source,
      true,
      5);
  auto final = std::make_shared<core::TopNRowNumberNode>(
      "final",
      core::TopNRowNumberNode::RankFunction::kRank,
      partitionKeys,
      sortingKeys,
      sortingOrders,
      std::nullopt,
      1,
      partial,
      false,
      5);

  const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
  ASSERT_TRUE(headroom.cudaValid);
  const auto allocatableBytes = headroom.allocatableBytes();
  ASSERT_GT(allocatableBytes, 0);
  auto pressureReservation = cudf_velox::tryAcquireDeviceMemoryAdmission(
      headroom.device, allocatableBytes, allocatableBytes);
  ASSERT_TRUE(pressureReservation);

  constexpr int32_t kConcurrentTasks = 96;
  std::vector<std::future<std::pair<vector_size_t, uint64_t>>> futures;
  futures.reserve(kConcurrentTasks);
  for (int32_t task = 0; task < kConcurrentTasks; ++task) {
    futures.push_back(std::async(std::launch::async, [final] {
      CursorParameters params;
      params.planNode = final;
      params.maxDrivers = 1;
      params.serialExecution = true;
      params.queryConfigs = {
          {cudf_velox::CudfConfig::kCudfEnabled, "true"},
          {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
           "1073741824"}};
      auto cursor = TaskCursor::create(params);
      vector_size_t rows = 0;
      while (cursor->moveNext()) {
        rows += cursor->current()->size();
      }
      uint64_t conditionalAdmissions = 0;
      for (const auto& pipelineStats :
           cursor->task()->taskStats().pipelineStats) {
        for (const auto& operatorStats : pipelineStats.operatorStats) {
          if (operatorStats.operatorType == "CudfTopNRowNumber") {
            conditionalAdmissions +=
                operatorStats.runtimeStats
                    .at("topNConditionalBlockingAdmissions")
                    .sum;
          }
        }
      }
      return std::pair{rows, conditionalAdmissions};
    }));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  pressureReservation.reset();

  uint64_t totalConditionalAdmissions = 0;
  for (auto& future : futures) {
    const auto [rows, conditionalAdmissions] = future.get();
    EXPECT_EQ(rows, kRows * kBatches / 2 + 128);
    totalConditionalAdmissions += conditionalAdmissions;
  }
  EXPECT_GT(totalConditionalAdmissions, 0);
}

} // namespace
