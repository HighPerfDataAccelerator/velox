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

class HtoDPerfTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    memory::SharedArbitrator::registerFactory();
    memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool("HtoDPerfTest");
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

  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<test::VectorMaker> maker_;
};

} // namespace

TEST_F(HtoDPerfTest, singleVsBatched) {
  constexpr int kBatchRows = 64;
  constexpr int kNumBatches = 5000;
  constexpr int kWarmup = 3;

  auto stream = rmm::cuda_stream_view{};

  std::vector<RowVectorPtr> batches;
  batches.reserve(kNumBatches);
  for (int i = 0; i < kNumBatches; ++i) {
    batches.push_back(makeRowVector(kBatchRows));
  }

  int64_t totalRows = static_cast<int64_t>(kBatchRows) * kNumBatches;
  std::cout << "Config: " << kNumBatches << " batches x " << kBatchRows
            << " rows = " << totalRows << " total rows\n";

  // Warmup
  for (int w = 0; w < kWarmup; ++w) {
    auto t = with_arrow::toCudfTable(batches[0], pool_.get(), stream);
  }

  // Benchmark: individual toCudfTable calls (old path)
  auto start1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < kNumBatches; ++i) {
    auto t = with_arrow::toCudfTable(batches[i], pool_.get(), stream);
  }
  auto end1 = std::chrono::high_resolution_clock::now();
  auto ms1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1)
          .count();

  // Warmup batched
  {
    std::vector<RowVectorPtr> warmBatch(batches.begin(), batches.begin() + 100);
    auto t = with_arrow::toCudfTableBatched(warmBatch, pool_.get(), stream);
  }

  // Benchmark: toCudfTableBatched (new path)
  auto start2 = std::chrono::high_resolution_clock::now();
  auto batchedResult =
      with_arrow::toCudfTableBatched(batches, pool_.get(), stream);
  auto end2 = std::chrono::high_resolution_clock::now();
  auto ms2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2)
          .count();

  std::cout << "\n=== HtoD Performance Comparison ===\n";
  std::cout << "Individual toCudfTable x" << kNumBatches << ": "
            << ms1 / 1000.0 << " ms  ("
            << static_cast<double>(ms1) / kNumBatches << " us/call)\n";
  std::cout << "Batched toCudfTableBatched:        " << ms2 / 1000.0 << " ms\n";
  std::cout << "Speedup: " << static_cast<double>(ms1) / ms2 << "x\n";
  std::cout << "Batched result: " << batchedResult->num_rows() << " rows x "
            << batchedResult->num_columns() << " cols\n";

  EXPECT_EQ(batchedResult->num_rows(), totalRows);
  EXPECT_EQ(batchedResult->num_columns(), 3);
}

TEST_F(HtoDPerfTest, varyingBatchSize) {
  auto stream = rmm::cuda_stream_view{};

  std::vector<int> batchSizes = {10, 50, 100, 500, 1000};
  constexpr int kTotalRows = 100000;

  std::cout << "\n=== Varying Batch Size (total " << kTotalRows
            << " rows) ===\n";
  std::cout << "BatchSize | NumBatches | Individual(ms) | Batched(ms) | Speedup\n";
  std::cout << "----------|------------|----------------|-------------|--------\n";

  for (int batchSize : batchSizes) {
    int numBatches = kTotalRows / batchSize;

    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    for (int i = 0; i < numBatches; ++i) {
      batches.push_back(makeRowVector(batchSize));
    }

    // Individual
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numBatches; ++i) {
      auto t = with_arrow::toCudfTable(batches[i], pool_.get(), stream);
    }
    auto e1 = std::chrono::high_resolution_clock::now();
    double ms1 =
        std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count() /
        1000.0;

    // Batched
    auto s2 = std::chrono::high_resolution_clock::now();
    auto result =
        with_arrow::toCudfTableBatched(batches, pool_.get(), stream);
    auto e2 = std::chrono::high_resolution_clock::now();
    double ms2 =
        std::chrono::duration_cast<std::chrono::microseconds>(e2 - s2).count() /
        1000.0;

    printf(
        "%9d | %10d | %14.2f | %11.2f | %6.2fx\n",
        batchSize,
        numBatches,
        ms1,
        ms2,
        ms1 / ms2);

    EXPECT_EQ(result->num_rows(), kTotalRows);
  }
}
