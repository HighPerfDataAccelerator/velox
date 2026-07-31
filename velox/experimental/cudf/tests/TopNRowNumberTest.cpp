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

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
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

std::vector<const OperatorStats*> cudfTopNStats(const TaskStats& stats) {
  std::vector<const OperatorStats*> result;
  for (const auto& pipelineStats : stats.pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType == "CudfTopNRowNumber") {
        result.push_back(&operatorStats);
      }
    }
  }
  return result;
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
        {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"}};
    return TaskCursor::create(params);
  }

  core::PlanNodePtr makeTopN(
      core::PlanNodePtr source,
      core::TopNRowNumberNode::RankFunction function,
      bool partialOutput) {
    return std::make_shared<core::TopNRowNumberNode>(
        partialOutput ? "partial" : "final",
        function,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "p")},
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "s")},
        std::vector<core::SortOrder>{core::SortOrder(false, false)},
        std::nullopt,
        1,
        std::move(source),
        partialOutput);
  }

  core::PlanNodePtr makeConditionalTopN(
      core::PlanNodePtr source,
      bool partialOutput) {
    return std::make_shared<core::TopNRowNumberNode>(
        partialOutput ? "conditional-partial" : "conditional-final",
        core::TopNRowNumberNode::RankFunction::kRank,
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(
                BOOLEAN(), "__gluten_mpp_topn_active"),
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "p")},
        std::vector<core::FieldAccessTypedExprPtr>{
            std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "s")},
        std::vector<core::SortOrder>{core::SortOrder(false, false)},
        std::nullopt,
        1,
        std::move(source),
        partialOutput);
  }

 private:
  bool previousAllowCpuFallback_{true};
};

TEST_F(CudfTopNRowNumberTest, partialOutputIsBoundedByInputBatch) {
  auto first = makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>({1, 1, 2}),
       makeFlatVector<int64_t>({10, 11, 1}),
       makeFlatVector<int64_t>({100, 101, 200})});
  auto second = makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int64_t>({20, 1}),
       makeFlatVector<int64_t>({102, 201})});

  auto values = PlanBuilder().values({first, second}).planNode();
  auto partial =
      makeTopN(values, core::TopNRowNumberNode::RankFunction::kRowNumber, true);

  CursorParameters params;
  params.planNode = partial;
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"}};
  auto [cursor, rows] = readCursor(params);

  assertEqualResults(
      {makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2, 1, 2}),
           makeFlatVector<int64_t>({11, 1, 20, 1}),
           makeFlatVector<int64_t>({101, 200, 102, 201})})},
      rows);
  const auto stats = cudfTopNStats(cursor->task()->taskStats());
  ASSERT_EQ(stats.size(), 1);
  EXPECT_EQ(stats.front()->outputVectors, 2);
  EXPECT_EQ(stats.front()->spilledBytes, 0);
}

TEST_F(CudfTopNRowNumberTest, partialAndFinalPreserveCrossBatchRankTies) {
  auto first = makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>({1, 1, 2}),
       makeFlatVector<int64_t>({10, 10, 1}),
       makeFlatVector<int64_t>({100, 101, 200})});
  auto second = makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>({1, 1, 2}),
       makeFlatVector<int64_t>({20, 20, 1}),
       makeFlatVector<int64_t>({102, 103, 201})});

  auto values = PlanBuilder().values({first, second}).planNode();
  auto partial =
      makeTopN(values, core::TopNRowNumberNode::RankFunction::kRank, true);
  auto final =
      makeTopN(partial, core::TopNRowNumberNode::RankFunction::kRank, false);

  CursorParameters params;
  params.planNode = final;
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
       std::to_string(1ULL << 30)}};
  auto [cursor, rows] = readCursor(params);

  assertEqualResults(
      {makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 1, 2, 2}),
           makeFlatVector<int64_t>({20, 20, 1, 1}),
           makeFlatVector<int64_t>({102, 103, 200, 201})})},
      rows);
  const auto stats = cudfTopNStats(cursor->task()->taskStats());
  ASSERT_EQ(stats.size(), 2);
  EXPECT_TRUE(std::any_of(stats.begin(), stats.end(), [](const auto* stat) {
    return stat->outputVectors == 2;
  }));
  EXPECT_TRUE(std::all_of(stats.begin(), stats.end(), [](const auto* stat) {
    return stat->spilledBytes == 0;
  }));
}

TEST_F(CudfTopNRowNumberTest, conditionalPartialPreservesInactiveRows) {
  auto first = makeRowVector(
      {"__gluten_mpp_topn_active", "p", "s", "v"},
      {makeFlatVector<bool>({true, true, false, false}),
       makeFlatVector<int64_t>({1, 1, 1, 1}),
       makeFlatVector<int64_t>({10, 20, 100, 200}),
       makeFlatVector<int64_t>({10, 20, 100, 200})});
  auto second = makeRowVector(
      {"__gluten_mpp_topn_active", "p", "s", "v"},
      {makeFlatVector<bool>({true, false}),
       makeFlatVector<int64_t>({1, 1}),
       makeFlatVector<int64_t>({30, 300}),
       makeFlatVector<int64_t>({30, 300})});

  auto values = PlanBuilder().values({first, second}).planNode();
  auto partial = makeConditionalTopN(values, true);
  auto final = makeConditionalTopN(partial, false);

  CursorParameters params;
  params.planNode = final;
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
       std::to_string(1ULL << 30)}};
  auto [cursor, rows] = readCursor(params);

  assertEqualResults(
      {makeRowVector(
          {"__gluten_mpp_topn_active", "p", "s", "v"},
          {makeFlatVector<bool>({false, false, false, true}),
           makeFlatVector<int64_t>({1, 1, 1, 1}),
           makeFlatVector<int64_t>({100, 200, 300, 30}),
           makeFlatVector<int64_t>({100, 200, 300, 30})})},
      rows);
  const auto stats = cudfTopNStats(cursor->task()->taskStats());
  ASSERT_EQ(stats.size(), 2);
  EXPECT_TRUE(std::all_of(stats.begin(), stats.end(), [](const auto* stat) {
    return stat->spilledBytes == 0;
  }));
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

} // namespace
