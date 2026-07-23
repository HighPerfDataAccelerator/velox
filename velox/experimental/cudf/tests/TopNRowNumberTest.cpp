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

} // namespace
