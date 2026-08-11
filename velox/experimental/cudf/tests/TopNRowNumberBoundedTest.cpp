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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
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

bool usedBoundedTop1Backend(const TaskStats& stats) {
  for (const auto& pipelineStats : stats.pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType == "CudfTopNRowNumber" &&
          operatorStats.runtimeStats.count("topNBoundedTop1Backend") != 0) {
        return true;
      }
    }
  }
  return false;
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* key, const char* value) : key_(key) {
    if (const auto* existing = std::getenv(key); existing != nullptr) {
      oldValue_ = std::string(existing);
    }
    setenv(key, value, 1);
  }

  ~ScopedEnvVar() {
    if (oldValue_.has_value()) {
      setenv(key_.c_str(), oldValue_->c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

 private:
  std::string key_;
  std::optional<std::string> oldValue_;
};

class CudfTopNRowNumberBoundedTest : public OperatorTestBase {
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

TEST_F(CudfTopNRowNumberBoundedTest, rowNumberHostStateDoesNotUseDisk) {
  const auto spillRoot = TempDirectoryPath::create();
  const auto preexistingTopLevelSpills = findTopLevelProcessSpillDirectories();
  auto cursor = makeSpillingCursor(spillRoot->getPath());
  ASSERT_TRUE(cursor->moveNext());

  const fs::path taskSpillRoot(cursor->task()->spillDirectory());
  EXPECT_EQ(taskSpillRoot.parent_path(), fs::path(spillRoot->getPath()));

  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty())
      << "ROW_NUMBER=1 externalized candidates to disk";
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
  EXPECT_TRUE(usedBoundedTop1Backend(task->taskStats()));

  actualResults.clear();
  cursor.reset();
  task.reset();
  EXPECT_FALSE(fs::exists(taskSpillRoot));
  EXPECT_TRUE(fs::is_empty(spillRoot->getPath()));
}

TEST_F(CudfTopNRowNumberBoundedTest, earlyCloseCleansHostState) {
  const auto spillRoot = TempDirectoryPath::create();
  const auto preexistingTopLevelSpills = findTopLevelProcessSpillDirectories();
  auto cursor = makeSpillingCursor(spillRoot->getPath());
  ASSERT_TRUE(cursor->moveNext());

  auto task = cursor->task();
  const fs::path taskSpillRoot(task->spillDirectory());
  EXPECT_EQ(taskSpillRoot.parent_path(), fs::path(spillRoot->getPath()));
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
  EXPECT_EQ(findTopLevelProcessSpillDirectories(), preexistingTopLevelSpills);

  // SingleThreadedTaskCursor synchronously cancels the Task in its destructor.
  // The Task closes off-thread Drivers before requestCancel().wait() returns.
  cursor.reset();
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
  EXPECT_EQ(findTopLevelProcessSpillDirectories(), preexistingTopLevelSpills);
  EXPECT_TRUE(usedBoundedTop1Backend(task->taskStats()));

  task.reset();
  EXPECT_FALSE(fs::exists(taskSpillRoot));
  EXPECT_TRUE(fs::is_empty(spillRoot->getPath()));
}

TEST_F(CudfTopNRowNumberBoundedTest, groupedDescendingNullsLastAcrossBatches) {
  // Exercise the grouped scalar ARGMAX path used by Job 144, including null
  // partition keys and an all-null sort-key group.
  auto first = makeRowVector(
      {"p", "s", "v"},
      {makeNullableFlatVector<std::string>(
           {"a", "b", "a", "c", std::nullopt, "b"}),
       makeNullableFlatVector<int64_t>(
           {1, 7, 5, std::nullopt, 2, std::nullopt}),
       makeFlatVector<std::string>(
           {"a1", "b7", "a5", "c-null", "null-2", "b-null"})});
  auto second = makeRowVector(
      {"p", "s", "v"},
      {makeNullableFlatVector<std::string>(
           {"b", "a", "c", std::nullopt, "d", "a"}),
       makeNullableFlatVector<int64_t>(
           {9, std::nullopt, 3, 4, std::nullopt, 8}),
       makeFlatVector<std::string>(
           {"b9", "a-null", "c3", "null-4", "d-null", "a8"})});

  auto plan = PlanBuilder()
                  .values({first, second})
                  .topNRowNumber({"p"}, {"s DESC NULLS LAST"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "4"},
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes, "1"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "v", "row_number"},
      {makeNullableFlatVector<std::string>({"a", "b", "c", std::nullopt, "d"}),
       makeNullableFlatVector<int64_t>({8, 9, 3, 4, std::nullopt}),
       makeFlatVector<std::string>({"a8", "b9", "c3", "null-4", "d-null"}),
       makeFlatVector<int64_t>(5, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  EXPECT_TRUE(usedBoundedTop1Backend(cursor->task()->taskStats()));
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(CudfTopNRowNumberBoundedTest, multiKeyArgMaxIsStableAcrossBatches) {
  auto first = makeRowVector(
      {"p", "s1", "s2", "v"},
      {makeFlatVector<std::string>({"a", "a", "a", "b", "b"}),
       makeNullableFlatVector<int64_t>({5, 5, 5, 7, 6}),
       makeNullableFlatVector<int64_t>({3, 3, 2, std::nullopt, 9}),
       makeFlatVector<std::string>(
           {"a-first", "a-same-batch-later", "a-lower", "b-null", "b6"})});
  auto second = makeRowVector(
      {"p", "s1", "s2", "v"},
      {makeFlatVector<std::string>({"a", "a", "b", "c", "c"}),
       makeNullableFlatVector<int64_t>({5, 4, 7, std::nullopt, std::nullopt}),
       makeNullableFlatVector<int64_t>({3, 9, 4, 5, 7}),
       makeFlatVector<std::string>({"a-later-batch", "a4", "b7", "c5", "c7"})});

  auto plan =
      PlanBuilder()
          .values({first, second})
          .topNRowNumber(
              {"p"}, {"s1 DESC NULLS LAST", "s2 DESC NULLS LAST"}, 1, true)
          .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "4"},
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes, "1"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s1", "s2", "v", "row_number"},
      {makeFlatVector<std::string>({"a", "b", "c"}),
       makeNullableFlatVector<int64_t>({5, 7, std::nullopt}),
       makeNullableFlatVector<int64_t>({3, 4, 7}),
       makeFlatVector<std::string>({"a-first", "b7", "c7"}),
       makeFlatVector<int64_t>(3, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  EXPECT_TRUE(usedBoundedTop1Backend(cursor->task()->taskStats()));
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(
    CudfTopNRowNumberBoundedTest,
    uniquePartitionFastPathIsExactAndFallsBack) {
  ScopedEnvVar uniqueFastPath(
      "GLUTEN_CUDF_TOPN_UNIQUE_PARTITION_FAST_PATH", "1");
  // This small semantic fixture is intentionally duplicate-heavy. Force the
  // sparse continuation so it continues to cover exact NULL/NaN duplicate
  // filtering independently of the production profitability guard.
  ScopedEnvVar sparseMaxCandidatePct(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_MAX_CANDIDATE_PCT", "100");
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  auto data = makeRowVector(
      {"p", "s", "v"},
      {makeNullableFlatVector<double>(
           {1.0, 2.0, 1.0, nan, nan, std::nullopt, std::nullopt}),
       makeFlatVector<int64_t>({3, 4, 9, 5, 8, 6, 10}),
       makeFlatVector<std::string>(
           {"one-3",
            "two-4",
            "one-9",
            "nan-5",
            "nan-8",
            "null-6",
            "null-10"})});
  auto plan = PlanBuilder()
                  .values({data})
                  .topNRowNumber({"p"}, {"s DESC NULLS LAST"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "2"},
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes, "1"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "v", "row_number"},
      {makeNullableFlatVector<double>({1.0, 2.0, nan, std::nullopt}),
       makeFlatVector<int64_t>({9, 4, 8, 10}),
       makeFlatVector<std::string>({"one-9", "two-4", "nan-8", "null-10"}),
       makeFlatVector<int64_t>(4, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  const auto planStats = exec::toPlanStats(cursor->task()->taskStats());
  const auto& topNStats = planStats.at(plan->id());
  EXPECT_GT(topNStats.customStats.at("topNUniquePartitionFallbackRows").sum, 0);
  const auto* sequential =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_UNIQUE_FAST_PATH");
  if (sequential != nullptr && std::string_view(sequential) == "1") {
    const auto* sparse =
        std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_DUPLICATES");
    const bool sparseEnabled = sparse == nullptr ||
        (std::string_view(sparse) != "0" &&
         std::string_view(sparse) != "false" &&
         std::string_view(sparse) != "FALSE");
    if (sparseEnabled) {
      EXPECT_EQ(
          topNStats.customStats.at("topNSequentialDuplicateKeyRows").sum, 3);
      EXPECT_EQ(
          topNStats.customStats.at("topNSequentialSparseCandidateRows").sum, 6);
      EXPECT_EQ(
          topNStats.customStats.at("topNSequentialSingletonOutputRows").sum, 1);
      EXPECT_EQ(
          topNStats.customStats.count("topNSequentialUniqueFallbackRows"), 0);
    } else {
      EXPECT_GT(
          topNStats.customStats.at("topNSequentialUniqueFallbackRows").sum, 0);
    }
  }
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(
    CudfTopNRowNumberBoundedTest,
    sequentialSparseDuplicatePathFallsBackForDenseCandidates) {
  ScopedEnvVar uniqueFastPath(
      "GLUTEN_CUDF_TOPN_UNIQUE_PARTITION_FAST_PATH", "1");
  ScopedEnvVar sequentialFastPath(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_UNIQUE_FAST_PATH", "1");
  ScopedEnvVar sparseDuplicates(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_DUPLICATES", "1");
  ScopedEnvVar sparseMaxCandidatePct(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_MAX_CANDIDATE_PCT", "10");
  ScopedEnvVar denseRawRewrite(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_DENSE_RAW_REWRITE", "1");
  constexpr vector_size_t kRows = 4096;
  constexpr vector_size_t kKeys = 128;
  auto data = makeRowVector(
      {"p", "s", "payload"},
      {makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row % kKeys; }),
       makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<std::string>(kRows, [](vector_size_t row) {
         return std::string(256, static_cast<char>('a' + row % 26));
       })});
  auto plan = PlanBuilder()
                  .values({data})
                  .topNRowNumber({"p"}, {"s DESC"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "4"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberFinalizeInputBytes, "65536"},
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes, "1"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "payload", "row_number"},
      {makeFlatVector<int64_t>(kKeys, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kKeys, [](vector_size_t row) { return kRows - kKeys + row; }),
       makeFlatVector<std::string>(
           kKeys,
           [](vector_size_t row) {
             const auto sourceRow = kRows - kKeys + row;
             return std::string(256, static_cast<char>('a' + sourceRow % 26));
           }),
       makeFlatVector<int64_t>(kKeys, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  const auto planStats = exec::toPlanStats(cursor->task()->taskStats());
  const auto& topNStats = planStats.at(plan->id());
  EXPECT_EQ(
      topNStats.customStats.at("topNSequentialProvenDuplicateCandidateRows")
          .sum,
      kRows);
  EXPECT_EQ(
      topNStats.customStats.at("topNSequentialSparseDenseFallbackRows").sum,
      kRows);
  EXPECT_EQ(
      topNStats.customStats.at("topNSequentialSparseDenseFallbackCandidateRows")
          .sum,
      kRows);
  EXPECT_EQ(
      topNStats.customStats.at("topNSequentialUniqueFallbackRows").sum, kRows);
  EXPECT_GT(
      topNStats.customStats.at("topNSequentialDenseRawRewriteBytes").sum, 0);
  EXPECT_GT(
      topNStats.customStats.at("topNSequentialDenseRawRewriteChunks").sum, 0);
  EXPECT_GT(topNStats.customStats.at("topNSpillRawChunks").sum, 0);
  EXPECT_EQ(
      topNStats.customStats.count("topNSequentialSparseCandidateRows"), 0);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(CudfTopNRowNumberBoundedTest, uniquePartitionFastPathSkipsReduction) {
  ScopedEnvVar uniqueFastPath(
      "GLUTEN_CUDF_TOPN_UNIQUE_PARTITION_FAST_PATH", "1");
  auto data = makeRowVector(
      {"p", "s", "v"},
      {makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt}),
       makeFlatVector<int64_t>({9, 4, 8, 5}),
       makeFlatVector<std::string>({"one", "two", "three", "null"})});
  auto plan = PlanBuilder()
                  .values({data})
                  .topNRowNumber({"p"}, {"s DESC NULLS LAST"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "2"},
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes, "1"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "v", "row_number"},
      {makeNullableFlatVector<int64_t>({1, 2, 3, std::nullopt}),
       makeFlatVector<int64_t>({9, 4, 8, 5}),
       makeFlatVector<std::string>({"one", "two", "three", "null"}),
       makeFlatVector<int64_t>(4, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  const auto planStats = exec::toPlanStats(cursor->task()->taskStats());
  const auto& topNStats = planStats.at(plan->id());
  const auto* sequential =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_UNIQUE_FAST_PATH");
  if (sequential != nullptr && std::string_view(sequential) == "1") {
    EXPECT_GT(
        topNStats.customStats.at("topNSequentialUniqueConfirmedRows").sum, 0);
    EXPECT_GT(
        topNStats.customStats.at("topNSequentialUniqueOutputRows").sum, 0);
    EXPECT_EQ(
        topNStats.customStats.count("topNSequentialUniqueFallbackRows"), 0);
  } else {
    EXPECT_GT(
        topNStats.customStats.at("topNUniquePartitionFastPathRows").sum, 0);
    EXPECT_EQ(
        topNStats.customStats.count("topNUniquePartitionFallbackRows"), 0);
  }
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

// Manual A/B probe for the wide-payload path. Keep disabled in the default
// suite because it intentionally materializes roughly 256 MiB of strings.
TEST_F(CudfTopNRowNumberBoundedTest, DISABLED_widePayloadLateGatherBenchmark) {
  constexpr vector_size_t kRows = 262'144;
  constexpr vector_size_t kDuplicateEvery = 80;
  const auto* sparseValue =
      std::getenv("CUDF_TOPN_BENCHMARK_SPARSE_DUPLICATES");
  const bool sparseDuplicates =
      sparseValue != nullptr && std::string_view(sparseValue) == "1";
  const std::string payload(240, 'x');
  auto data = makeRowVector(
      {"p", "s1", "s2", "v0", "v1", "v2", "v3"},
      {makeFlatVector<int64_t>(
           kRows,
           [sparseDuplicates](auto row) {
             return sparseDuplicates &&
                     row % kDuplicateEvery == kDuplicateEvery - 1
                 ? row - 1
                 : row;
           }),
       makeFlatVector<int64_t>(kRows, [](auto row) { return kRows - row; }),
       makeFlatVector<int64_t>(kRows, [](auto row) { return row % 101; }),
       makeFlatVector<std::string>(
           kRows, [&](auto row) { return payload + std::to_string(row % 97); }),
       makeFlatVector<std::string>(
           kRows, [&](auto row) { return payload + std::to_string(row % 89); }),
       makeFlatVector<std::string>(
           kRows, [&](auto row) { return payload + std::to_string(row % 83); }),
       makeFlatVector<std::string>(kRows, [&](auto row) {
         return payload + std::to_string(row % 79);
       })});

  auto plan =
      PlanBuilder()
          .values({data})
          .topNRowNumber(
              {"p"}, {"s1 DESC NULLS LAST", "s2 DESC NULLS LAST"}, 1, false)
          .planNode();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  const auto spillRoot = TempDirectoryPath::create();
  params.spillDirectory = spillRoot->getPath();
  const auto* benchmarkHostBytes =
      std::getenv("CUDF_TOPN_BENCHMARK_HOST_BYTES");
  const auto* benchmarkOutputBytes =
      std::getenv("CUDF_TOPN_BENCHMARK_OUTPUT_CHUNK_BYTES");
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      // Mirror the Job-144 final path: partition the low-reduction wide input,
      // externalize every candidate run, force packed disk ownership, then
      // restore through the pinned bounce pipeline before the final gather.
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      // Two scaled buckets keep each restore wave just below the 128-MiB
      // bounce capacity while raising each bucket to 131,072 rows. This is a
      // closer sort/group scaling probe than many tiny buckets; sixty-four
      // buckets here would fall below the 16-MiB bounce threshold.
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "2"},
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes,
       benchmarkHostBytes == nullptr ? "1" : benchmarkHostBytes},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberOutputChunkBytes,
       benchmarkOutputBytes == nullptr ? "33554432" : benchmarkOutputBytes}};
  auto cursor = TaskCursor::create(params);
  uint64_t outputRows = 0;
  while (cursor->moveNext()) {
    outputRows += cursor->current()->size();
  }
  const auto expectedRows =
      sparseDuplicates ? kRows - kRows / kDuplicateEvery : kRows;
  EXPECT_EQ(outputRows, expectedRows);
  const auto taskStats = cursor->task()->taskStats();
  EXPECT_TRUE(usedBoundedTop1Backend(taskStats));
  for (const auto& pipelineStats : taskStats.pipelineStats) {
    for (const auto& operatorStats : pipelineStats.operatorStats) {
      if (operatorStats.operatorType == "CudfTopNRowNumber" &&
          operatorStats.runtimeStats.count("topNBoundedTop1Backend") != 0) {
        const auto runtimeSum = [&](std::string_view name) -> uint64_t {
          const auto it = operatorStats.runtimeStats.find(std::string(name));
          return it == operatorStats.runtimeStats.end() ? 0 : it->second.sum;
        };
        std::cout
            << "wide_payload_topn sparse_duplicates=" << sparseDuplicates
            << " add_input_wall_ms="
            << operatorStats.addInputTiming.wallNanos / 1'000'000.0
            << " add_input_cpu_ms="
            << operatorStats.addInputTiming.cpuNanos / 1'000'000.0
            << " get_output_wall_ms="
            << operatorStats.getOutputTiming.wallNanos / 1'000'000.0
            << " pinned_restore_mib="
            << runtimeSum("topNPinnedBounceBytes") / (1024.0 * 1024.0)
            << " pageable_restore_mib="
            << runtimeSum("topNPageableDirectRestoreBytes") / (1024.0 * 1024.0)
            << " host_stage_ms="
            << runtimeSum("topNPinnedBounceHostStageNanos") / 1'000'000.0
            << " copy_sync_ms="
            << runtimeSum("topNPinnedBounceCopySyncNanos") / 1'000'000.0
            << " finalize_restore_ms="
            << runtimeSum("topNFinalizeRestoreNanos") / 1'000'000.0
            << " finalize_concat_ms="
            << runtimeSum("topNFinalizeConcatenateNanos") / 1'000'000.0
            << " finalize_reduce_ms="
            << runtimeSum("topNFinalizeReduceNanos") / 1'000'000.0
            << " unique_probe_ms="
            << runtimeSum("topNUniquePartitionProbeNanos") / 1'000'000.0
            << " unique_fast_rows="
            << runtimeSum("topNUniquePartitionFastPathRows")
            << " unique_fallback_rows="
            << runtimeSum("topNUniquePartitionFallbackRows")
            << " sequential_probe_ms="
            << runtimeSum("topNSequentialUniqueProbeNanos") / 1'000'000.0
            << " sequential_key_mib="
            << runtimeSum("topNSequentialKeyBytes") / (1024.0 * 1024.0)
            << " sequential_confirmed_rows="
            << runtimeSum("topNSequentialUniqueConfirmedRows")
            << " sequential_output_rows="
            << runtimeSum("topNSequentialUniqueOutputRows")
            << " sequential_fallback_rows="
            << runtimeSum("topNSequentialUniqueFallbackRows")
            << " sequential_duplicate_keys="
            << runtimeSum("topNSequentialDuplicateKeyRows")
            << " sequential_sparse_candidates="
            << runtimeSum("topNSequentialSparseCandidateRows")
            << " sequential_singleton_output="
            << runtimeSum("topNSequentialSingletonOutputRows")
            << " sequential_sparse_pinned_mib="
            << runtimeSum("topNSequentialSparsePinnedBounceBytes") /
                (1024.0 * 1024.0)
            << " sequential_sparse_pageable_mib="
            << runtimeSum("topNSequentialSparsePageableDirectBytes") /
                (1024.0 * 1024.0)
            << " sequential_sparse_host_stage_ms="
            << runtimeSum("topNSequentialSparseHostStageNanos") / 1'000'000.0
            << " sequential_sparse_copy_sync_ms="
            << runtimeSum("topNSequentialSparseCopySyncNanos") / 1'000'000.0
            << " sequential_sparse_split_ms="
            << runtimeSum("topNSequentialSparseSplitNanos") / 1'000'000.0
            << " sequential_direct_chunk_mib="
            << runtimeSum("topNSequentialDirectChunkBytes") / (1024.0 * 1024.0)
            << " sequential_direct_chunk_batches="
            << runtimeSum("topNSequentialDirectChunkBatches")
            << " spill_input_mib="
            << runtimeSum("topNSpillInputBytes") / (1024.0 * 1024.0)
            << " spill_stored_mib="
            << runtimeSum("topNSpillStoredBytes") / (1024.0 * 1024.0)
            << " spill_compression_ms="
            << runtimeSum("topNSpillCompressionNanos") / 1'000'000.0
            << " spill_compressed_chunks="
            << runtimeSum("topNSpillCompressedChunks")
            << " spill_raw_chunks=" << runtimeSum("topNSpillRawChunks")
            << std::endl;
      }
    }
  }
}

TEST_F(CudfTopNRowNumberBoundedTest, abandonsLowReductionPartial) {
  constexpr vector_size_t kRows = 128;
  auto first = makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row + 1000; }),
       makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; })});
  auto second = makeRowVector(
      {"p", "s", "v"},
      {makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row + kRows; }),
       makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row + kRows; })});

  auto basePartial = std::dynamic_pointer_cast<const core::TopNRowNumberNode>(
      PlanBuilder()
          .values({first, second})
          .topNRowNumber({"p"}, {"s"}, 1, false)
          .planNode());
  ASSERT_NE(basePartial, nullptr);
  auto partial = core::TopNRowNumberNode::Builder(*basePartial)
                     .id("partial")
                     .partialOutput(true)
                     .build();
  auto final = core::TopNRowNumberNode::Builder(*basePartial)
                   .id("final")
                   .rowNumberColumnName("row_number")
                   .source(partial)
                   .partialOutput(false)
                   .build();

  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = final;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {core::QueryConfig::kAbandonPartialTopNRowNumberMinRows, "64"},
      {core::QueryConfig::kAbandonPartialTopNRowNumberMinPct, "80"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "v", "row_number"},
      {makeFlatVector<int64_t>(
           2 * kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           2 * kRows,
           [](vector_size_t row) {
             return row < kRows ? row + 1000 : row - kRows;
           }),
       makeFlatVector<int64_t>(
           2 * kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(2 * kRows, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);

  const auto planStats = exec::toPlanStats(cursor->task()->taskStats());
  const auto& partialStats = planStats.at("partial");
  ASSERT_EQ(partialStats.customStats.at("abandonedPartial").sum, 1);
  // The partial decision must come from the configured bounded sample rather
  // than a full reduction of the 128-row GPU input batch.
  ASSERT_EQ(
      partialStats.customStats.at("partialAbandonSampleInputRows").sum, 64);
  ASSERT_EQ(
      partialStats.customStats.at("partialAbandonSampleOutputRows").sum, 64);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(
    CudfTopNRowNumberBoundedTest,
    sequentialEarlyDenseProbeSwitchesToOnePassHashExactly) {
  ScopedEnvVar uniqueFastPath(
      "GLUTEN_CUDF_TOPN_UNIQUE_PARTITION_FAST_PATH", "1");
  ScopedEnvVar sequentialFastPath(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_UNIQUE_FAST_PATH", "1");
  ScopedEnvVar earlyDenseBatches(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_EARLY_DENSE_BATCHES", "2");
  ScopedEnvVar earlyDenseDistinctPct(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_EARLY_DENSE_MAX_DISTINCT_PCT", "95");
  ScopedEnvVar denseRawRewrite(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_DENSE_RAW_REWRITE", "0");
  constexpr vector_size_t kRowsPerBatch = 128;
  auto first = makeRowVector(
      {"p", "s", "payload"},
      {makeFlatVector<int64_t>(
           kRowsPerBatch, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRowsPerBatch, [](vector_size_t row) { return row; }),
       makeFlatVector<std::string>(kRowsPerBatch, [](vector_size_t row) {
         return "first-" + std::to_string(row);
       })});
  auto second = makeRowVector(
      {"p", "s", "payload"},
      {makeFlatVector<int64_t>(
           kRowsPerBatch, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRowsPerBatch,
           [](vector_size_t row) { return kRowsPerBatch + row; }),
       makeFlatVector<std::string>(kRowsPerBatch, [](vector_size_t row) {
         return "second-" + std::to_string(row);
       })});
  auto plan = PlanBuilder()
                  .values({first, second})
                  .topNRowNumber({"p"}, {"s DESC"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      // Preserve the two Values vectors as two GPU input batches. The
      // early-dense probe is deliberately batch-count based; the default
      // CPU-to-GPU conversion target would merge this small fixture.
      {cudf_velox::CudfFromVelox::kGpuBatchSizeRows,
       std::to_string(kRowsPerBatch)},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "4"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberFinalizeInputBytes, "65536"},
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes, "1"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "payload", "row_number"},
      {makeFlatVector<int64_t>(
           kRowsPerBatch, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRowsPerBatch,
           [](vector_size_t row) { return kRowsPerBatch + row; }),
       makeFlatVector<std::string>(
           kRowsPerBatch,
           [](vector_size_t row) { return "second-" + std::to_string(row); }),
       makeFlatVector<int64_t>(
           kRowsPerBatch, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  const auto planStats = exec::toPlanStats(cursor->task()->taskStats());
  const auto& topNStats = planStats.at(plan->id());
  EXPECT_EQ(
      topNStats.customStats.at("topNSequentialEarlyDenseSwitches").sum, 1);
  EXPECT_EQ(
      topNStats.customStats.at("topNSequentialEarlyDensePrefixRows").sum,
      2 * kRowsPerBatch);
  EXPECT_LT(
      topNStats.customStats.at("topNSequentialEarlyDenseDistinctRows").sum,
      topNStats.customStats.at("topNSequentialEarlyDenseProbeRows").sum);
  EXPECT_EQ(topNStats.customStats.count("topNSequentialUniqueFallbackRows"), 0);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(
    CudfTopNRowNumberBoundedTest,
    oversizedFinalizeBucketIsRepartitionedFromPackedSpill) {
  constexpr vector_size_t kRows = 4096;
  constexpr vector_size_t kBatchRows = 128;
  std::vector<RowVectorPtr> data;
  for (vector_size_t begin = 0; begin < kRows; begin += kBatchRows) {
    data.push_back(makeRowVector(
        {"p", "s", "payload"},
        {makeFlatVector<int64_t>(
             kBatchRows,
             [begin](vector_size_t row) {
               const auto globalRow = begin + row;
               // Duplicate partition zero in the final input batch. The
               // The duplicate is deliberately in a different input batch.
               // Sparse sequential mode must retain both rows for exact
               // Top-1 while sending every singleton directly to output.
               return globalRow == kRows - 1 ? 0 : globalRow;
             }),
         makeFlatVector<int64_t>(
             kBatchRows,
             [begin](vector_size_t row) { return kRows - (begin + row); }),
         makeFlatVector<std::string>(kBatchRows, [begin](vector_size_t row) {
           const auto globalRow = begin + row;
           return std::string(256, static_cast<char>('a' + globalRow % 26)) +
               std::to_string(globalRow);
         })}));
  }
  auto plan = PlanBuilder()
                  .values(data)
                  .topNRowNumber({"p"}, {"s"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      // Keep each source vector as an independent packed chunk so this test
      // exercises multi-chunk bulk restore coalescing, not just recursive
      // repartitioning of one conversion-merged input.
      {cudf_velox::CudfFromVelox::kGpuBatchSizeRows,
       std::to_string(kBatchRows)},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "2"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberFinalizeInputBytes, "65536"},
      // Force every externalized chunk through the shared async packed spill
      // backend before recursive repartition and final restore.
      {cudf_velox::CudfConfig::kCudfOrderByHostSpillBytes, "1"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "payload", "row_number"},
      {makeFlatVector<int64_t>(
           kRows - 1, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRows - 1,
           [](vector_size_t row) { return row == 0 ? 1 : kRows - row; }),
       makeFlatVector<std::string>(
           kRows - 1,
           [](vector_size_t row) {
             const auto sourceRow = row == 0 ? kRows - 1 : row;
             return std::string(256, static_cast<char>('a' + sourceRow % 26)) +
                 std::to_string(sourceRow);
           }),
       makeFlatVector<int64_t>(kRows - 1, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  EXPECT_GT(actual.size(), 2)
      << "oversized two-bucket input was not split into bounded outputs";
  EXPECT_TRUE(usedBoundedTop1Backend(cursor->task()->taskStats()));
  const auto planStats = exec::toPlanStats(cursor->task()->taskStats());
  const auto& topNStats = planStats.at(plan->id());
  const auto* sequential =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_UNIQUE_FAST_PATH");
  const auto* sparse =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_DUPLICATES");
  const bool sparseEnabled = sequential != nullptr &&
      std::string_view(sequential) == "1" &&
      (sparse == nullptr ||
       (std::string_view(sparse) != "0" &&
        std::string_view(sparse) != "false" &&
        std::string_view(sparse) != "FALSE"));
  if (sparseEnabled) {
    EXPECT_EQ(
        topNStats.customStats.at("topNSequentialDuplicateKeyRows").sum, 1);
    EXPECT_EQ(
        topNStats.customStats.at("topNSequentialSparseCandidateRows").sum, 2);
    EXPECT_EQ(
        topNStats.customStats.at("topNSequentialSingletonOutputRows").sum,
        kRows - 2);
    EXPECT_EQ(topNStats.customStats.count("topNRecursiveSourceChunks"), 0);
  } else {
    const auto sourceChunks =
        topNStats.customStats.at("topNRecursiveSourceChunks").sum;
    const auto restoreGroups =
        topNStats.customStats.at("topNRecursiveRestoreGroups").sum;
    EXPECT_GT(sourceChunks, restoreGroups)
        << "recursive repartition did not coalesce multiple packed source "
           "chunks into bulk restore groups";
    EXPECT_GT(
        topNStats.customStats.at("topNRecursiveChildChunks").sum,
        restoreGroups);
  }
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

TEST_F(
    CudfTopNRowNumberBoundedTest,
    largeFinalizeBucketHasBoundedOutputBatches) {
  constexpr vector_size_t kRows = 300000;
  auto data = makeRowVector(
      {"p", "s", "payload"},
      {makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return kRows - row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row * 10; })});
  auto plan = PlanBuilder()
                  .values({data})
                  .topNRowNumber({"p"}, {"s"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "8"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberFinalizeInputBytes,
       "1073741824"},
      // Retain only part of the partitioned state on device. This exercises
      // mixed device/host buckets and largest-bucket eviction before direct
      // device-aware finalize.
      {cudf_velox::CudfConfig::kCudfTopNRowNumberDeviceResidentBytes,
       "2097152"},
      // A lower process-wide budget exercises admission rejection in
      // addition to the per-operator largest-bucket watermark.
      {cudf_velox::CudfConfig::kCudfDeviceResidentCapacityBytes, "1048576"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    EXPECT_LE(cursor->current()->size(), 262144);
    actual.push_back(cursor->current());
  }

  auto expected = makeRowVector(
      {"p", "s", "payload", "row_number"},
      {makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return kRows - row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row * 10; }),
       makeFlatVector<int64_t>(kRows, [](vector_size_t) { return 1; })});
  EXPECT_GT(actual.size(), 1);
  assertEqualResults({expected}, actual);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
  const auto planStats = exec::toPlanStats(cursor->task()->taskStats());
  const auto& topNStats = planStats.at(plan->id());
  const auto* sequential =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_UNIQUE_FAST_PATH");
  if (sequential != nullptr && std::string_view(sequential) == "1") {
    EXPECT_GT(topNStats.customStats.at("topNSequentialStoredBytes").sum, 0);
    EXPECT_EQ(
        topNStats.customStats.at("topNSequentialUniqueConfirmedRows").sum,
        kRows);
    EXPECT_EQ(
        topNStats.customStats.at("topNSequentialUniqueOutputRows").sum, kRows);
  } else {
    EXPECT_GT(topNStats.customStats.at("topNDeviceSpillBytes").sum, 0);
    EXPECT_GT(topNStats.customStats.at("topNDeviceSpillBuckets").sum, 0);
    EXPECT_GT(topNStats.customStats.at("topNDeviceResidentBytes").sum, 0);
    EXPECT_GT(topNStats.customStats.at("topNFinalizePreflushBytes").sum, 0);
    EXPECT_GT(
        topNStats.customStats.at("topNGlobalAdmissionRejectedBytes").sum, 0);
  }
}

TEST_F(
    CudfTopNRowNumberBoundedTest,
    deviceOutputStagingHonorsConfiguredBounds) {
  // Device staging avoids the device -> host -> device round trip, but it
  // must not bypass the operator's row/byte ownership boundary. In
  // particular, downstream exchange backpressure must never keep a complete
  // oversized variable-width bucket alive behind a bounded queue.
  ScopedEnvVar deviceOutputStaging(
      "GLUTEN_CUDF_TOPN_DEVICE_OUTPUT_STAGING_ENABLED", "true");
  constexpr vector_size_t kRows = 600000;
  auto data = makeRowVector(
      {"p", "s", "payload"},
      {makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return kRows - row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row * 10; })});
  auto plan = PlanBuilder()
                  .values({data})
                  .topNRowNumber({"p"}, {"s"}, 1, true)
                  .planNode();
  const auto spillRoot = TempDirectoryPath::create();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 1;
  params.serialExecution = true;
  params.spillDirectory = spillRoot->getPath();
  params.queryConfigs = {
      {cudf_velox::CudfConfig::kCudfEnabled, "true"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberCandidateRunBytes, "1"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberHostPartitions, "2"},
      {cudf_velox::CudfConfig::kCudfTopNRowNumberFinalizeInputBytes,
       "1073741824"}};
  auto cursor = TaskCursor::create(params);
  std::vector<RowVectorPtr> actual;
  while (cursor->moveNext()) {
    actual.push_back(cursor->current());
  }

  ASSERT_GT(actual.size(), 2);
  vector_size_t actualRows = 0;
  for (const auto& batch : actual) {
    EXPECT_LE(batch->size(), 262144);
    actualRows += batch->size();
  }
  EXPECT_EQ(actualRows, kRows);
  auto expected = makeRowVector(
      {"p", "s", "payload", "row_number"},
      {makeFlatVector<int64_t>(kRows, [](vector_size_t row) { return row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return kRows - row; }),
       makeFlatVector<int64_t>(
           kRows, [](vector_size_t row) { return row * 10; }),
       makeFlatVector<int64_t>(kRows, [](vector_size_t) { return 1; })});
  assertEqualResults({expected}, actual);
  EXPECT_TRUE(findSpillDirectories(spillRoot->getPath()).empty());
}

} // namespace
