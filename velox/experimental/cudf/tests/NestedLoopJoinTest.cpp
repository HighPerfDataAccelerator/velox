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

#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace {

class CudfNestedLoopJoinTest : public HiveConnectorTestBase {
 public:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    cudf_velox::CudfConfig::getInstance().allowCpuFallback =
        false;
    cudf_velox::registerCudf();
  }

  void TearDown() override {
    cudf_velox::unregisterCudf();
    HiveConnectorTestBase::TearDown();
  }
};

TEST_F(CudfNestedLoopJoinTest, crossJoin) {
  auto probeVectors = {makeRowVector(
      {"t0", "t1"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
          makeFlatVector<int64_t>({10, 20, 30}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0"},
      {
          makeFlatVector<int32_t>({100, 200}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(probeVectors)
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildVectors)
                          .planNode(),
                      {"t0", "t1", "u0"})
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, t.t1, u.u0 FROM t CROSS JOIN u");
}

TEST_F(CudfNestedLoopJoinTest, innerJoinWithCondition) {
  auto probeVectors = {makeRowVector(
      {"t0", "t1"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0", "u1"},
      {
          makeFlatVector<int32_t>({2, 4, 6}),
          makeFlatVector<int64_t>({200, 400, 600}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(probeVectors)
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator)
                  .values(buildVectors)
                  .planNode(),
              "t0 < u0",
              {"t0", "t1", "u0", "u1"},
              core::JoinType::kInner)
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, t.t1, u.u0, u.u1 "
          "FROM t INNER JOIN u ON t.t0 < u.u0");
}

TEST_F(CudfNestedLoopJoinTest, leftJoin) {
  auto probeVectors = {makeRowVector(
      {"t0", "t1"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0"},
      {
          makeFlatVector<int32_t>({2, 4}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(probeVectors)
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator)
                  .values(buildVectors)
                  .planNode(),
              "t0 = u0",
              {"t0", "t1", "u0"},
              core::JoinType::kLeft)
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, t.t1, u.u0 "
          "FROM t LEFT JOIN u ON t.t0 = u.u0");
}

TEST_F(CudfNestedLoopJoinTest, rightJoin) {
  auto probeVectors = {makeRowVector(
      {"t0"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0", "u1"},
      {
          makeFlatVector<int32_t>({2, 4, 6}),
          makeFlatVector<int64_t>({20, 40, 60}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(probeVectors)
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator)
                  .values(buildVectors)
                  .planNode(),
              "t0 = u0",
              {"t0", "u0", "u1"},
              core::JoinType::kRight)
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, u.u0, u.u1 "
          "FROM t RIGHT JOIN u ON t.t0 = u.u0");
}

TEST_F(CudfNestedLoopJoinTest, fullJoin) {
  auto probeVectors = {makeRowVector(
      {"t0"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0"},
      {
          makeFlatVector<int32_t>({2, 4}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(probeVectors)
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator)
                  .values(buildVectors)
                  .planNode(),
              "t0 = u0",
              {"t0", "u0"},
              core::JoinType::kFull)
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, u.u0 "
          "FROM t FULL OUTER JOIN u ON t.t0 = u.u0");
}

TEST_F(CudfNestedLoopJoinTest, leftSemiProject) {
  auto probeVectors = {makeRowVector(
      {"t0", "t1"},
      {
          makeFlatVector<int32_t>({1, 2, 3, 4, 5}),
          makeFlatVector<int64_t>({10, 20, 30, 40, 50}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0"},
      {
          makeFlatVector<int32_t>({2, 4, 6}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(probeVectors)
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator)
                  .values(buildVectors)
                  .planNode(),
              "t0 = u0",
              {"t0", "t1", "match"},
              core::JoinType::kLeftSemiProject)
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, t.t1, "
          "EXISTS (SELECT 1 FROM u WHERE t.t0 = u.u0) "
          "FROM t");
}

TEST_F(CudfNestedLoopJoinTest, emptyBuild) {
  auto probeVectors = {makeRowVector(
      {"t0"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0"},
      {
          makeFlatVector<int32_t>({}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  // Inner join with empty build -> empty result
  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(probeVectors)
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildVectors)
                          .planNode(),
                      {"t0", "u0"})
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, u.u0 FROM t CROSS JOIN u");

  // Left join with empty build -> all left rows, null build
  planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  plan = PlanBuilder(planNodeIdGenerator)
             .values(probeVectors)
             .nestedLoopJoin(
                 PlanBuilder(planNodeIdGenerator)
                     .values(buildVectors)
                     .planNode(),
                 "t0 = u0",
                 {"t0", "u0"},
                 core::JoinType::kLeft)
             .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, u.u0 "
          "FROM t LEFT JOIN u ON t.t0 = u.u0");
}

TEST_F(CudfNestedLoopJoinTest, emptyProbe) {
  auto probeVectors = {makeRowVector(
      {"t0"},
      {
          makeFlatVector<int32_t>({}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0"},
      {
          makeFlatVector<int32_t>({1, 2, 3}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .values(probeVectors)
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator)
                  .values(buildVectors)
                  .planNode(),
              "t0 = u0",
              {"t0", "u0"},
              core::JoinType::kRight)
          .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, u.u0 "
          "FROM t RIGHT JOIN u ON t.t0 = u.u0");
}

TEST_F(CudfNestedLoopJoinTest, crossJoinWithNulls) {
  auto probeVectors = {makeRowVector(
      {"t0"},
      {
          makeNullableFlatVector<int32_t>(
              {1, std::nullopt, 3}),
      })};

  auto buildVectors = {makeRowVector(
      {"u0"},
      {
          makeNullableFlatVector<int32_t>(
              {std::nullopt, 20}),
      })};

  createDuckDbTable("t", probeVectors);
  createDuckDbTable("u", buildVectors);

  auto planNodeIdGenerator =
      std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .values(probeVectors)
                  .nestedLoopJoin(
                      PlanBuilder(planNodeIdGenerator)
                          .values(buildVectors)
                          .planNode(),
                      {"t0", "u0"})
                  .planNode();

  AssertQueryBuilder(plan, duckDbQueryRunner_)
      .assertResults(
          "SELECT t.t0, u.u0 FROM t CROSS JOIN u");
}

} // namespace
