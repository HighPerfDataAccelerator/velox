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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/tests/SubfieldFiltersBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::connector;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::common::test;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec::test;

class CudfHiveDataSource2Test : public virtual CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();

    auto config = std::unordered_map<std::string, std::string>{
        {facebook::velox::cudf_velox::connector::hive::CudfHiveConfig::
             kUseMetadataPrefilter,
         "true"}};
    resetCudfHiveConnector(
        std::make_shared<facebook::velox::config::ConfigBase>(
            std::move(config)));
  }

  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
  }

  std::vector<RowVectorPtr> makeVectors(
      int32_t count,
      int32_t rowsPerVector,
      const RowTypePtr& rowType = nullptr) {
    auto inputs = rowType ? rowType : rowType_;
    return CudfHiveConnectorTestBase::makeVectors(inputs, count, rowsPerVector);
  }

  core::PlanNodePtr tableScanNode() {
    return tableScanNode(rowType_);
  }

  core::PlanNodePtr tableScanNode(const RowTypePtr& outputType) {
    auto tableHandle = makeTableHandle();
    return PlanBuilder(pool_.get())
        .startTableScan()
        .outputType(outputType)
        .tableHandle(tableHandle)
        .endTableScan()
        .planNode();
  }

  static PlanNodeStats getTableScanStats(const std::shared_ptr<Task>& task) {
    auto planStats = toPlanStats(task->taskStats());
    return std::move(planStats.at("0"));
  }

  RowTypePtr rowType_{
      ROW({"c0", "c1", "c2", "c3", "c4", "c5", "c6"},
          {INTEGER(),
           VARCHAR(),
           TINYINT(),
           DOUBLE(),
           BIGINT(),
           VARCHAR(),
           REAL()})};
};

TEST_F(CudfHiveDataSource2Test, allColumns) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto splits = makeCudfHiveConnectorSplits({filePath});

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(splits)
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  ASSERT_TRUE(it->second.dynamicFilterStats.empty());
}

TEST_F(CudfHiveDataSource2Test, allColumnsWithHiveConnectorSplit) {
  auto vectors = makeVectors(10, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  auto plan = tableScanNode();

  auto splits = std::vector<std::shared_ptr<ConnectorSplit>>{
      facebook::velox::connector::hive::HiveConnectorSplitBuilder(
          filePath->getPath())
          .connectorId(kCudfHiveConnectorId)
          .fileFormat(dwio::common::FileFormat::PARQUET)
          .build()};

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults("SELECT * FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, columnProjection) {
  auto vectors = makeVectors(5, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  auto outputType = ROW({"c0", "c4"}, {INTEGER(), BIGINT()});
  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(outputType)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  auto splits = makeCudfHiveConnectorSplits({filePath});
  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults("SELECT c0, c4 FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, multipleFiles) {
  auto vectors = makeVectors(10, 1'000);
  auto filePaths = makeFilePaths(5);
  for (int i = 0; i < 5; ++i) {
    writeToFile(
        filePaths[i]->getPath(),
        {vectors[i * 2], vectors[i * 2 + 1]},
        "c");
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto splits = makeCudfHiveConnectorSplits(filePaths);

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults("SELECT * FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, multipleFilesOnePerSplit) {
  auto vectors = makeVectors(5, 1'000);
  auto filePaths = makeFilePaths(5);
  for (int i = 0; i < 5; ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i], "c");
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto splits = makeCudfHiveConnectorSplits(filePaths);

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults("SELECT * FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, filterPushdown) {
  auto rowType =
      ROW({"c0", "c1", "c2", "c3"}, {TINYINT(), BIGINT(), DOUBLE(), BOOLEAN()});
  auto filePaths = makeFilePaths(10);
  auto vectors = makeVectors(10, 1'000, rowType);
  for (int32_t i = 0; i < vectors.size(); i++) {
    writeToFile(filePaths[i]->getPath(), vectors[i]);
  }
  createDuckDbTable(vectors);

  common::SubfieldFilters subfieldFilters =
      SubfieldFiltersBuilder()
          .add(
              "c1",
              std::make_unique<common::BigintRange>(
                  int64_t(0), std::numeric_limits<int64_t>::max(), true))
          .add("c3", std::make_unique<common::BoolValue>(true, false))
          .build();

  auto tableHandle = makeTableHandle(
      "parquet_table", rowType, true, std::move(subfieldFilters), nullptr);

  auto assignments =
      HiveConnectorTestBase::allRegularColumns(rowType);

  auto plan =
      PlanBuilder()
          .startTableScan()
          .outputType(ROW({"c1", "c3", "c0"}, {BIGINT(), BOOLEAN(), TINYINT()}))
          .tableHandle(tableHandle)
          .assignments(assignments)
          .endTableScan()
          .planNode();

  auto task = assertQuery(
      plan,
      makeCudfHiveConnectorSplits(filePaths),
      "SELECT c1, c3, c0 FROM tmp WHERE (c1 >= 0) AND c3");

  auto tableScanStats = getTableScanStats(task);
  EXPECT_EQ(tableScanStats.inputRows, tableScanStats.outputRows);
}

TEST_F(CudfHiveDataSource2Test, multipleRowGroups) {
  auto vectors = makeVectors(5, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto splits = makeCudfHiveConnectorSplits({filePath});

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults("SELECT * FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, withSplitPreload) {
  auto vectors = makeVectors(5, 1'000);
  auto filePaths = makeFilePaths(5);
  for (int i = 0; i < 5; ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i], "c");
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto splits = makeCudfHiveConnectorSplits(filePaths);

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .config(
                      QueryConfig::kMaxSplitPreloadPerDriver,
                      std::to_string(2))
                  .splits(splits)
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
}

TEST_F(CudfHiveDataSource2Test, singleColumn) {
  auto vectors = makeVectors(3, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  auto outputType = ROW({"c0"}, {INTEGER()});
  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(outputType)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  auto splits = makeCudfHiveConnectorSplits({filePath});
  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults("SELECT c0 FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, columnAliases) {
  auto vectors = makeVectors(1, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  std::string tableName = "t";
  std::unordered_map<std::string, std::string> aliases = {{"a", "c0"}};
  auto outputType = ROW({"a"}, {INTEGER()});
  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .tableName(tableName)
                  .outputType(outputType)
                  .columnAliases(aliases)
                  .endTableScan()
                  .planNode();

  assertQuery(plan, {filePath}, "SELECT c0 FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, smallData) {
  auto vector = makeRowVector({
      makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
      makeFlatVector<double>({1.1, 2.2, 3.3, 4.4, 5.5}),
  });
  auto rowType = ROW({"c0", "c1"}, {INTEGER(), DOUBLE()});
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vector, "c");
  createDuckDbTable({vector});

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(rowType)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  auto splits = makeCudfHiveConnectorSplits({filePath});
  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults("SELECT * FROM tmp");
}

TEST_F(CudfHiveDataSource2Test, directBufferInputRawInputBytes) {
  constexpr int kSize = 10;
  auto vector = makeRowVector({
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
      makeFlatVector<int64_t>(kSize, folly::identity),
  });
  auto filePath = TempFilePath::create();
  createDuckDbTable({vector});
  writeToFile(filePath->getPath(), {vector}, "c");

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .tableHandle(tableHandle)
                  .outputType(ROW({"c0", "c2"}, {BIGINT(), BIGINT()}))
                  .endTableScan()
                  .planNode();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .splits(makeCudfHiveConnectorSplits({filePath}))
                  .assertResults("SELECT c0, c2 FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
  auto rawInputBytes = it->second.rawInputBytes;
  ASSERT_GE(rawInputBytes, 0);
}

TEST_F(CudfHiveDataSource2Test, multipleFilesWithPreload) {
  auto vectors = makeVectors(10, 500);
  auto filePaths = makeFilePaths(10);
  for (int i = 0; i < 10; ++i) {
    writeToFile(filePaths[i]->getPath(), vectors[i], "c");
  }
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto splits = makeCudfHiveConnectorSplits(filePaths);

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .config(
                      QueryConfig::kMaxSplitPreloadPerDriver,
                      std::to_string(4))
                  .splits(splits)
                  .assertResults("SELECT * FROM tmp");

  auto planStats = toPlanStats(task->taskStats());
  auto scanNodeId = plan->id();
  auto it = planStats.find(scanNodeId);
  ASSERT_TRUE(it != planStats.end());
}

TEST_F(CudfHiveDataSource2Test, sameFileTwice) {
  auto vectors = makeVectors(3, 1'000);
  auto filePath = TempFilePath::create();
  writeToFile(filePath->getPath(), vectors, "c");
  createDuckDbTable(vectors);

  auto plan = tableScanNode();
  auto splits = makeCudfHiveConnectorSplits({filePath, filePath});

  AssertQueryBuilder(duckDbQueryRunner_)
      .plan(plan)
      .splits(splits)
      .assertResults(
          "SELECT * FROM tmp UNION ALL SELECT * FROM tmp");
}
