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

#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/tests/utils/VectorMaker.h"

#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

namespace {

class DtoHPackedTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::SharedArbitrator::registerFactory();
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("DtoHPackedTest");
    maker_ = std::make_unique<test::VectorMaker>(pool_.get());
  }

  RowVectorPtr makeRowVector(int numRows) {
    std::vector<int64_t> col0(numRows);
    std::iota(col0.begin(), col0.end(), 0);

    std::vector<double> col1(numRows);
    for (int i = 0; i < numRows; ++i) {
      col1[i] = i * 1.1;
    }

    std::vector<std::string> col2(numRows);
    for (int i = 0; i < numRows; ++i) {
      col2[i] = "row_" + std::to_string(i % 1000);
    }

    return maker_->rowVector({
        maker_->flatVector<int64_t>(col0),
        maker_->flatVector<double>(col1),
        maker_->flatVector<StringView>(
            numRows,
            [&](auto i) { return StringView(col2[i]); }),
    });
  }

  RowVectorPtr makeRowVectorWithNulls(int numRows) {
    auto col0 = maker_->flatVector<int64_t>(
        numRows,
        std::function<int64_t(vector_size_t)>(
            [](vector_size_t i) -> int64_t { return i; }),
        std::function<bool(vector_size_t)>(
            [](vector_size_t i) -> bool { return i % 7 == 0; }));

    auto col1 = maker_->flatVector<double>(
        numRows,
        std::function<double(vector_size_t)>(
            [](vector_size_t i) -> double { return i * 2.5; }),
        std::function<bool(vector_size_t)>(
            [](vector_size_t i) -> bool { return i % 5 == 0; }));

    std::vector<std::optional<std::string>> strs(numRows);
    for (int i = 0; i < numRows; ++i) {
      if (i % 3 == 0) {
        strs[i] = std::nullopt;
      } else {
        strs[i] = "str_" + std::to_string(i % 500);
      }
    }
    auto col2 = maker_->flatVectorNullable(strs);

    std::vector<VectorPtr> children = {col0, col1, col2};
    return maker_->rowVector(children);
  }

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<test::VectorMaker> maker_;
};

} // namespace

TEST_F(DtoHPackedTest, roundTripInts) {
  auto stream = rmm::cuda_stream_view{};
  auto input = makeRowVector(1024);

  auto cudfTable = with_arrow::toCudfTable(input, pool_.get(), stream);
  auto output = with_arrow::toVeloxColumn(
      cudfTable->view(), pool_.get(), "c", stream);

  ASSERT_EQ(output->size(), input->size());
  ASSERT_EQ(output->childrenSize(), input->childrenSize());

  auto inCol0 = input->childAt(0)->asFlatVector<int64_t>();
  auto outCol0 = output->childAt(0)->asFlatVector<int64_t>();
  for (int i = 0; i < input->size(); ++i) {
    ASSERT_EQ(inCol0->valueAt(i), outCol0->valueAt(i)) << "row " << i;
  }

  auto inCol1 = input->childAt(1)->asFlatVector<double>();
  auto outCol1 = output->childAt(1)->asFlatVector<double>();
  for (int i = 0; i < input->size(); ++i) {
    ASSERT_DOUBLE_EQ(inCol1->valueAt(i), outCol1->valueAt(i)) << "row " << i;
  }
}

TEST_F(DtoHPackedTest, roundTripStrings) {
  auto stream = rmm::cuda_stream_view{};
  auto input = makeRowVector(2048);

  auto cudfTable = with_arrow::toCudfTable(input, pool_.get(), stream);
  auto output = with_arrow::toVeloxColumn(
      cudfTable->view(), pool_.get(), "c", stream);

  ASSERT_EQ(output->size(), input->size());
  auto inCol = input->childAt(2)->asFlatVector<StringView>();
  auto outCol = output->childAt(2)->asFlatVector<StringView>();
  for (int i = 0; i < input->size(); ++i) {
    ASSERT_EQ(inCol->valueAt(i).str(), outCol->valueAt(i).str()) << "row " << i;
  }
}

TEST_F(DtoHPackedTest, roundTripWithNulls) {
  auto stream = rmm::cuda_stream_view{};
  auto input = makeRowVectorWithNulls(4096);

  auto cudfTable = with_arrow::toCudfTable(input, pool_.get(), stream);
  auto output = with_arrow::toVeloxColumn(
      cudfTable->view(), pool_.get(), "c", stream);

  ASSERT_EQ(output->size(), input->size());
  for (int c = 0; c < 3; ++c) {
    auto inChild = input->childAt(c);
    auto outChild = output->childAt(c);
    for (int i = 0; i < input->size(); ++i) {
      ASSERT_EQ(inChild->isNullAt(i), outChild->isNullAt(i))
          << "col " << c << " row " << i;
    }
  }

  auto inCol0 = input->childAt(0)->asFlatVector<int64_t>();
  auto outCol0 = output->childAt(0)->asFlatVector<int64_t>();
  for (int i = 0; i < input->size(); ++i) {
    if (!inCol0->isNullAt(i)) {
      ASSERT_EQ(inCol0->valueAt(i), outCol0->valueAt(i)) << "row " << i;
    }
  }
}

TEST_F(DtoHPackedTest, emptyTable) {
  auto stream = rmm::cuda_stream_view{};
  auto input = makeRowVector(0);

  auto cudfTable = with_arrow::toCudfTable(input, pool_.get(), stream);
  auto output = with_arrow::toVeloxColumn(
      cudfTable->view(), pool_.get(), "c", stream);

  ASSERT_EQ(output->size(), 0);
}

TEST_F(DtoHPackedTest, largeBatch) {
  constexpr int kRows = 100000;
  auto stream = rmm::cuda_stream_view{};
  auto input = makeRowVector(kRows);

  auto cudfTable = with_arrow::toCudfTable(input, pool_.get(), stream);

  auto start = std::chrono::high_resolution_clock::now();
  auto output = with_arrow::toVeloxColumn(
      cudfTable->view(), pool_.get(), "c", stream);
  auto end = std::chrono::high_resolution_clock::now();
  auto ms =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  std::cout << "D2H (packed) " << kRows << " rows x 3 cols: "
            << ms / 1000.0 << " ms\n";

  ASSERT_EQ(output->size(), kRows);

  auto inCol0 = input->childAt(0)->asFlatVector<int64_t>();
  auto outCol0 = output->childAt(0)->asFlatVector<int64_t>();
  EXPECT_EQ(inCol0->valueAt(0), outCol0->valueAt(0));
  EXPECT_EQ(inCol0->valueAt(kRows - 1), outCol0->valueAt(kRows - 1));
}

TEST_F(DtoHPackedTest, manySmallColumns) {
  constexpr int kRows = 500;
  constexpr int kCols = 20;
  auto stream = rmm::cuda_stream_view{};

  std::vector<VectorPtr> columns;
  for (int c = 0; c < kCols; ++c) {
    columns.push_back(maker_->flatVector<int32_t>(
        kRows, [c](auto i) { return static_cast<int32_t>(i + c * 1000); }));
  }
  auto input = maker_->rowVector(columns);

  auto cudfTable = with_arrow::toCudfTable(input, pool_.get(), stream);
  auto output = with_arrow::toVeloxColumn(
      cudfTable->view(), pool_.get(), "c", stream);

  ASSERT_EQ(output->size(), kRows);
  ASSERT_EQ(output->childrenSize(), kCols);

  for (int c = 0; c < kCols; ++c) {
    auto inCol = input->childAt(c)->asFlatVector<int32_t>();
    auto outCol = output->childAt(c)->asFlatVector<int32_t>();
    ASSERT_EQ(inCol->valueAt(0), outCol->valueAt(0)) << "col " << c;
    ASSERT_EQ(inCol->valueAt(kRows - 1), outCol->valueAt(kRows - 1))
        << "col " << c;
  }
}
