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
#include <limits>
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

int64_t topNRuntimeStat(const TaskStats& stats, std::string_view name) {
  int64_t sum = 0;
  for (const auto& pipelineStats : stats.pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType != "CudfTopNRowNumber") {
        continue;
      }
      const auto it = operatorStats.runtimeStats.find(std::string(name));
      if (it != operatorStats.runtimeStats.end()) {
        sum += it->second.sum;
      }
    }
  }
  return sum;
}

std::vector<RowVectorPtr> readAll(TaskCursor& cursor) {
  std::vector<RowVectorPtr> results;
  while (cursor.moveNext()) {
    results.push_back(cursor.current());
  }
  return results;
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

  std::unique_ptr<TaskCursor> makeCursor(
      core::PlanNodePtr plan,
      const std::string& spillRoot,
      uint64_t candidateRunBytes,
      bool cudfEnabled = true) {
    CursorParameters params;
    params.planNode = std::move(plan);
    params.maxDrivers = 1;
    params.serialExecution = true;
    params.spillDirectory = spillRoot;
    params.queryConfigs = {
        {cudf_velox::CudfConfig::kCudfEnabled, cudfEnabled ? "true" : "false"},
        {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
         std::to_string(candidateRunBytes)}};
    return TaskCursor::create(params);
  }

  std::unique_ptr<TaskCursor> makeSpillingCursor(const std::string& spillRoot) {
    return makeCursor(makePlan(), spillRoot, 1);
  }

  void registerWithCpuFallback(bool allowCpuFallback) {
    cudf_velox::unregisterCudf();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback = allowCpuFallback;
    cudf_velox::registerCudf();
  }

 private:
  bool previousAllowCpuFallback_{true};
};

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

TEST_F(CudfTopNRowNumberTest, noSpillRowNumberAcrossBatches) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeNullableFlatVector<int64_t>({10, std::nullopt, 5}),
           makeFlatVector<int64_t>({101, 201, 301})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2, 3}),
           makeNullableFlatVector<int64_t>({20, 7, std::nullopt}),
           makeFlatVector<int64_t>({102, 202, 302})})};
  auto plan = PlanBuilder()
                  .values(data)
                  .topNRank("row_number", {"p"}, {"s DESC NULLS LAST"}, 1, true)
                  .planNode();
  auto expected = makeRowVector(
      {"p", "s", "v", "row_number"},
      {makeFlatVector<int64_t>({1, 2, 3}),
       makeFlatVector<int64_t>({20, 7, 5}),
       makeFlatVector<int64_t>({102, 202, 301}),
       makeFlatVector<int64_t>({1, 1, 1})});

  const auto spillRoot = TempDirectoryPath::create();
  auto cursor = makeCursor(
      plan, spillRoot->getPath(), std::numeric_limits<uint64_t>::max());
  auto actual = readAll(*cursor);
  auto task = cursor->task();

  assertEqualResults({expected}, actual);
  EXPECT_TRUE(usedCudfTopNRowNumber(task->taskStats()));
  EXPECT_GT(
      topNRuntimeStat(task->taskStats(), "topNRowNumberCandidateMerges"), 0);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberNoSpillFastPath"), 1);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberPressureChecks"), 0);
  EXPECT_EQ(topNRuntimeStat(task->taskStats(), "topNRowNumberSpillRuns"), 0);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(CudfTopNRowNumberTest, rankTiesAcrossHostCandidateBucketsMatchCpu) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<int64_t>({10, 5}),
           makeFlatVector<int64_t>({101, 201})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<int64_t>({10, 7}),
           makeFlatVector<int64_t>({102, 202})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<int64_t>({9, 7}),
           makeFlatVector<int64_t>({103, 203})})};
  auto plan = PlanBuilder()
                  .values(data)
                  .topNRank("rank", {"p"}, {"s DESC"}, 1, true)
                  .planNode();

  MaterializedRowMultiset cpuResults;
  {
    registerWithCpuFallback(true);
    const auto cpuRoot = TempDirectoryPath::create();
    auto cpuCursor = makeCursor(
        plan, cpuRoot->getPath(), std::numeric_limits<uint64_t>::max(), false);
    cpuResults = materialize(readAll(*cpuCursor));
    EXPECT_FALSE(usedCudfTopNRowNumber(cpuCursor->task()->taskStats()));
  }
  registerWithCpuFallback(false);

  const auto spillRoot = TempDirectoryPath::create();
  auto cursor = makeCursor(plan, spillRoot->getPath(), 1);
  auto actual = readAll(*cursor);
  auto task = cursor->task();

  assertEqualResults(cpuResults, plan->outputType(), actual);
  EXPECT_GE(
      topNRuntimeStat(task->taskStats(), "topNRowNumberHostCandidateBatches"),
      3);
  EXPECT_GT(
      topNRuntimeStat(task->taskStats(), "topNRowNumberHostOutputBuckets"), 0);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberHostCandidatePath"), 1);
  EXPECT_GT(
      topNRuntimeStat(task->taskStats(), "topNRowNumberHostCandidateBytes"), 0);
  EXPECT_GT(
      topNRuntimeStat(
          task->taskStats(), "topNRowNumberHostCandidateUncompressedBytes"),
      0);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberPressureSpills"), 0);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberThresholdSpills"), 0);
  EXPECT_EQ(topNRuntimeStat(task->taskStats(), "topNRowNumberSpillRuns"), 0);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(CudfTopNRowNumberTest, earlyCloseReleasesHostCandidateState) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2, 3, 4}),
           makeFlatVector<int64_t>({10, 20, 30, 40}),
           makeFlatVector<int64_t>({101, 201, 301, 401})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2, 3, 4}),
           makeFlatVector<int64_t>({11, 21, 31, 41}),
           makeFlatVector<int64_t>({102, 202, 302, 402})})};
  auto plan = PlanBuilder()
                  .values(data)
                  .topNRank("rank", {"p"}, {"s DESC"}, 1, true)
                  .planNode();

  const auto spillRoot = TempDirectoryPath::create();
  auto cursor = makeCursor(plan, spillRoot->getPath(), 1);
  ASSERT_TRUE(cursor->moveNext());
  auto task = cursor->task();

  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberHostCandidatePath"), 1);
  EXPECT_EQ(topNRuntimeStat(task->taskStats(), "topNRowNumberSpillRuns"), 0);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());

  cursor.reset();
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
  task.reset();
  EXPECT_TRUE(fs::is_empty(spillRoot->getPath()));
}

TEST_F(CudfTopNRowNumberTest, pressurePreMergeSpillIsExercised) {
  std::vector<RowVectorPtr> data{
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<int64_t>({10, 5}),
           makeFlatVector<int64_t>({101, 201})}),
      makeRowVector(
          {"p", "s", "v"},
          {makeFlatVector<int64_t>({1, 2}),
           makeFlatVector<int64_t>({20, 7}),
           makeFlatVector<int64_t>({102, 202})})};
  auto plan = PlanBuilder()
                  .values(data)
                  .topNRank("row_number", {"p"}, {"s DESC"}, 1, true)
                  .planNode();
  auto expected = makeRowVector(
      {"p", "s", "v", "row_number"},
      {makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int64_t>({20, 7}),
       makeFlatVector<int64_t>({102, 202}),
       makeFlatVector<int64_t>({1, 1})});

  const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
  ASSERT_TRUE(headroom.cudaValid);
  ASSERT_GT(headroom.allocatableBytes(), 0);
  auto admissionBlocker = cudf_velox::tryAcquireDeviceMemoryAdmission(
      headroom.device,
      headroom.allocatableBytes(),
      std::numeric_limits<size_t>::max());
  ASSERT_TRUE(admissionBlocker.has_value());

  const auto spillRoot = TempDirectoryPath::create();
  auto cursor = makeCursor(plan, spillRoot->getPath(), 72);
  auto actual = readAll(*cursor);
  auto task = cursor->task();

  assertEqualResults({expected}, actual);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberPressureChecks"), 1);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberPressureSpills"), 1);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberThresholdSpills"), 0);
  EXPECT_EQ(topNRuntimeStat(task->taskStats(), "topNRowNumberSpillRuns"), 2);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(CudfTopNRowNumberTest, pressureSplitsOversizedInputBeforeReduction) {
  std::vector<RowVectorPtr> data{makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>({1, 2, 1, 2}),
       makeFlatVector<int64_t>({10, 5, 20, 7}),
       makeFlatVector<int64_t>({101, 201, 102, 202})})};
  auto plan = PlanBuilder()
                  .values(data)
                  .topNRank("row_number", {"p"}, {"s DESC"}, 1, true)
                  .planNode();
  auto expected = makeRowVector(
      {"p", "s", "v", "row_number"},
      {makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int64_t>({20, 7}),
       makeFlatVector<int64_t>({102, 202}),
       makeFlatVector<int64_t>({1, 1})});

  const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
  ASSERT_TRUE(headroom.cudaValid);
  ASSERT_GT(headroom.allocatableBytes(), 0);
  auto admissionBlocker = cudf_velox::tryAcquireDeviceMemoryAdmission(
      headroom.device,
      headroom.allocatableBytes(),
      std::numeric_limits<size_t>::max());
  ASSERT_TRUE(admissionBlocker.has_value());

  const auto spillRoot = TempDirectoryPath::create();
  auto cursor = makeCursor(plan, spillRoot->getPath(), 72);
  auto actual = readAll(*cursor);
  auto task = cursor->task();

  assertEqualResults({expected}, actual);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberInputPressureChecks"),
      1);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberInputPressureSplits"),
      1);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberInputPressureChunks"),
      2);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberInputPressureSpills"),
      2);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberThresholdSpills"), 0);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(CudfTopNRowNumberTest, pressureSplitsRankInputIntoHostCandidateBuckets) {
  std::vector<RowVectorPtr> data{makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>({1, 2, 1, 2}),
       makeFlatVector<int64_t>({10, 5, 20, 7}),
       makeFlatVector<int64_t>({101, 201, 102, 202})})};
  auto plan = PlanBuilder()
                  .values(data)
                  .topNRank("rank", {"p"}, {"s DESC"}, 1, true)
                  .planNode();
  auto expected = makeRowVector(
      {"p", "s", "v", "rank"},
      {makeFlatVector<int64_t>({1, 2}),
       makeFlatVector<int64_t>({20, 7}),
       makeFlatVector<int64_t>({102, 202}),
       makeFlatVector<int64_t>({1, 1})});

  const auto headroom = cudf_velox::captureDeviceAllocationHeadroom();
  ASSERT_TRUE(headroom.cudaValid);
  ASSERT_GT(headroom.allocatableBytes(), 0);
  constexpr uint64_t kTestRemainingAdmissionBytes = 32ULL << 20;
  const auto reserveBytes = std::max<uint64_t>(
      1ULL << 30, static_cast<uint64_t>(headroom.totalBytes) / 50);
  const auto admissionCapacity = headroom.allocatableBytes() > reserveBytes
      ? headroom.allocatableBytes() - reserveBytes
      : headroom.allocatableBytes() / 2;
  ASSERT_GT(admissionCapacity, kTestRemainingAdmissionBytes);
  auto admissionBlocker = cudf_velox::tryAcquireDeviceMemoryAdmission(
      headroom.device,
      admissionCapacity - kTestRemainingAdmissionBytes,
      std::numeric_limits<size_t>::max());
  ASSERT_TRUE(admissionBlocker.has_value());

  const auto spillRoot = TempDirectoryPath::create();
  auto cursor = makeCursor(plan, spillRoot->getPath(), 72);
  auto actual = readAll(*cursor);
  auto task = cursor->task();

  assertEqualResults({expected}, actual);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberInputPressureSplits"),
      1);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberInputPressureChunks"),
      2);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberHostCandidateBatches"),
      2);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberHostCandidatePath"), 1);
  EXPECT_EQ(
      topNRuntimeStat(task->taskStats(), "topNRowNumberInputPressureSpills"),
      0);
  EXPECT_EQ(topNRuntimeStat(task->taskStats(), "topNRowNumberSpillRuns"), 0);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(CudfTopNRowNumberTest, noSpillPerformancePath) {
  constexpr vector_size_t kRowsPerBatch = 131'072;
  constexpr vector_size_t kNumPartitions = 32'768;
  constexpr int32_t kNumBatches = 48;
  std::vector<RowVectorPtr> data;
  data.reserve(kNumBatches);
  for (int32_t batch = 0; batch < kNumBatches; ++batch) {
    data.push_back(makeRowVector(
        {"p", "s", "v"},
        {makeFlatVector<int64_t>(
             kRowsPerBatch,
             [](vector_size_t row) { return row % kNumPartitions; }),
         makeFlatVector<int64_t>(
             kRowsPerBatch,
             [batch](vector_size_t row) {
               return static_cast<int64_t>(batch) * kRowsPerBatch + row;
             }),
         makeFlatVector<int64_t>(
             kRowsPerBatch, [](vector_size_t row) { return row; })}));
  }

  auto plan = PlanBuilder()
                  .values(data)
                  .topNRank("row_number", {"p"}, {"s DESC NULLS LAST"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
       "1073741824"}};
  auto cursor = TaskCursor::create(params);

  vector_size_t outputRows = 0;
  while (cursor->moveNext()) {
    outputRows += cursor->current()->size();
  }

  EXPECT_EQ(outputRows, kNumPartitions);
  EXPECT_TRUE(usedCudfTopNRowNumber(cursor->task()->taskStats()));
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

} // namespace
