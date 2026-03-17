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
#include "velox/experimental/cudf/connectors/hive/CpuCoalescedDecoder.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gtest/gtest.h>

#include <thread>

using namespace facebook::velox;
using namespace facebook::velox::connector;
using namespace facebook::velox::core;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::exec::test;
using namespace facebook::velox::cudf_velox::connector::hive;

class CpuCoalescedDecoderTest : public CudfHiveConnectorTestBase {
 protected:
  void SetUp() override {
    CudfHiveConnectorTestBase::SetUp();
  }

  static void SetUpTestCase() {
    CudfHiveConnectorTestBase::SetUpTestCase();
  }

  RowTypePtr fixedWidthType_{
      ROW({"a", "b", "c", "d"},
          {BIGINT(), DOUBLE(), INTEGER(), REAL()})};
};

TEST_F(CpuCoalescedDecoderTest, allFieldsFixedWidthCheck) {
  EXPECT_TRUE(allFieldsFixedWidth(
      ROW({"a", "b"}, {BIGINT(), DOUBLE()})));
  EXPECT_TRUE(allFieldsFixedWidth(
      ROW({"a", "b", "c", "d"}, {INTEGER(), BIGINT(), REAL(), DOUBLE()})));
  EXPECT_FALSE(allFieldsFixedWidth(
      ROW({"a", "b"}, {BIGINT(), VARCHAR()})));
  EXPECT_FALSE(allFieldsFixedWidth(
      ROW({"a"}, {BOOLEAN()})));
  EXPECT_FALSE(allFieldsFixedWidth(
      ROW({"a", "b"}, {BIGINT(), ARRAY(INTEGER())})));
}

TEST_F(CpuCoalescedDecoderTest, cpuFallbackWithSaturatedGpu) {
  // Enable CPU decode fallback.
  CudfConfig::getInstance().cpuDecodeFallback = true;

  const int numFiles = 3;
  const int rowsPerFile = 1'000;

  auto vectors = makeVectors(fixedWidthType_, numFiles, rowsPerFile);
  std::vector<std::shared_ptr<TempFilePath>> filePaths;
  for (int i = 0; i < numFiles; i++) {
    auto fp = TempFilePath::create();
    writeToFile(fp->getPath(), vectors[i]);
    filePaths.push_back(fp);
  }

  createDuckDbTable(vectors);

  // Build a coalesced split: primary file + coalesced files.
  CudfHiveConnectorSplitBuilder splitBuilder(filePaths[0]->getPath());
  for (int i = 1; i < numFiles; i++) {
    splitBuilder.addCoalescedFile(
        CoalescedFileRange{filePaths[i]->getPath(), 0, 0, {}});
  }
  auto split = splitBuilder.build();

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(fixedWidthType_)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  // Saturate the GPU semaphore so tryLockGpu() fails.
  // Default concurrency is 1, so one lockGpu() call saturates it.
  gluten::lockGpu();

  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .split(Split(split))
                  .assertResults("SELECT * FROM tmp");

  gluten::unlockGpu();

  // Restore config.
  CudfConfig::getInstance().cpuDecodeFallback = false;
}

TEST_F(CpuCoalescedDecoderTest, cpuFallbackDisabledByConfig) {
  // Ensure fallback is disabled.
  CudfConfig::getInstance().cpuDecodeFallback = false;

  const int rowsPerFile = 500;
  auto vectors = makeVectors(fixedWidthType_, 2, rowsPerFile);
  std::vector<std::shared_ptr<TempFilePath>> filePaths;
  for (int i = 0; i < 2; i++) {
    auto fp = TempFilePath::create();
    writeToFile(fp->getPath(), vectors[i]);
    filePaths.push_back(fp);
  }

  createDuckDbTable(vectors);

  CudfHiveConnectorSplitBuilder splitBuilder(filePaths[0]->getPath());
  splitBuilder.addCoalescedFile(
      CoalescedFileRange{filePaths[1]->getPath(), 0, 0, {}});
  auto split = splitBuilder.build();

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(fixedWidthType_)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  // Even with GPU saturated, fallback won't activate because config is off.
  // The GPU path will eventually acquire the semaphore (blocking).
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .split(Split(split))
                  .assertResults("SELECT * FROM tmp");
}

TEST_F(CpuCoalescedDecoderTest, cpuFallbackSkippedForStringColumns) {
  CudfConfig::getInstance().cpuDecodeFallback = true;

  auto mixedType = ROW({"a", "b"}, {BIGINT(), VARCHAR()});
  auto vectors = makeVectors(mixedType, 2, 500);

  std::vector<std::shared_ptr<TempFilePath>> filePaths;
  for (int i = 0; i < 2; i++) {
    auto fp = TempFilePath::create();
    writeToFile(fp->getPath(), vectors[i]);
    filePaths.push_back(fp);
  }

  createDuckDbTable(vectors);

  CudfHiveConnectorSplitBuilder splitBuilder(filePaths[0]->getPath());
  splitBuilder.addCoalescedFile(
      CoalescedFileRange{filePaths[1]->getPath(), 0, 0, {}});
  auto split = splitBuilder.build();

  auto tableHandle = makeTableHandle();
  auto plan = PlanBuilder(pool_.get())
                  .startTableScan()
                  .outputType(mixedType)
                  .tableHandle(tableHandle)
                  .endTableScan()
                  .planNode();

  // allFieldsFixedWidth returns false for VARCHAR, so GPU path is used.
  auto task = AssertQueryBuilder(duckDbQueryRunner_)
                  .plan(plan)
                  .split(Split(split))
                  .assertResults("SELECT * FROM tmp");

  CudfConfig::getInstance().cpuDecodeFallback = false;
}
