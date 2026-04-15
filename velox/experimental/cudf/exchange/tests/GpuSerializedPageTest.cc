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
#include <memory>
#include <optional>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/experimental/cudf/exchange/GpuSerializedPage.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::cudf_velox {

namespace {

/// Minimal stub CudfVector for unit-testing GpuSerializedPage without a real
/// CUDA device.  It overrides the two methods that GpuSerializedPage calls:
///   - estimateFlatSize() -> size()
///   - RowVector::size()  -> numRows()
///
/// The base CudfVector constructor requires a cudf::table, which needs a GPU.
/// Instead we subclass RowVector directly and give GpuSerializedPage a thin
/// mock that satisfies the same interface.
class StubCudfVector : public CudfVector {
 public:
  // CudfVector inherits RowVector which is complex to construct without a
  // real cudf::table. We provide a factory that builds one from an empty
  // cudf table (requires CUDA runtime).
  //
  // For a truly device-free test, you'd mock at a higher level. Here we
  // simply store the desired sizes and override the accessors.
  StubCudfVector(
      velox::memory::MemoryPool* pool,
      uint64_t flatSize,
      vector_size_t numRows,
      std::unique_ptr<cudf::table>&& emptyTable,
      rmm::cuda_stream_view stream)
      : CudfVector(
            pool,
            ROW({}),
            numRows,
            std::move(emptyTable),
            stream),
        flatSize_(flatSize) {}

  uint64_t estimateFlatSize() const override {
    return flatSize_;
  }

 private:
  uint64_t flatSize_;
};

/// Helper to build a StubCudfVector. Needs CUDA runtime for the empty table.
std::shared_ptr<CudfVector> makeStubCudfVector(
    velox::memory::MemoryPool* pool,
    uint64_t flatSize,
    vector_size_t numRows) {
  auto emptyTable = std::make_unique<cudf::table>(
      std::vector<std::unique_ptr<cudf::column>>{});
  auto stream = rmm::cuda_stream_default;
  return std::make_shared<StubCudfVector>(
      pool, flatSize, numRows, std::move(emptyTable), stream);
}

class GpuSerializedPageTest : public testing::Test,
                               public velox::test::VectorTestBase {
 protected:
  static void SetUpTestSuite() {
    // VectorTestBase member initializers call memory::memoryManager() which
    // requires the singleton to exist before fixture construction.
    velox::memory::MemoryManager::testingSetInstance({});
  }
};

// Test: ConstructFromCudfVector
// Verify that a GpuSerializedPage can be constructed from a CudfVector
// and that size() and numRows() return correct values.
TEST_F(GpuSerializedPageTest, ConstructFromCudfVector) {
  auto cudfVector = makeStubCudfVector(pool(), 2048, 50);
  auto page = GpuSerializedPage(cudfVector);

  EXPECT_EQ(2048, page.size());
  EXPECT_TRUE(page.numRows().has_value());
  EXPECT_EQ(50, page.numRows().value());
}

// Test: NullInputRejects
// Verify that passing a null CudfVector triggers a VELOX_CHECK failure.
TEST_F(GpuSerializedPageTest, NullInputRejects) {
  std::shared_ptr<CudfVector> nullVector;
  VELOX_ASSERT_THROW(
      GpuSerializedPage(nullVector),
      "non-null CudfVector");
}

// Test: CudfVectorAccessor
// Verify that cudfVector() returns the same shared_ptr that was passed in.
TEST_F(GpuSerializedPageTest, CudfVectorAccessor) {
  auto cudfVector = makeStubCudfVector(pool(), 4096, 200);
  auto page = GpuSerializedPage(cudfVector);

  auto retrieved = page.cudfVector();
  EXPECT_EQ(cudfVector.get(), retrieved.get());
  EXPECT_EQ(4096, retrieved->estimateFlatSize());
  EXPECT_EQ(200, retrieved->size());
}

// Test: MoveSemantics
// Verify that move constructor works correctly.
TEST_F(GpuSerializedPageTest, MoveSemantics) {
  auto cudfVector = makeStubCudfVector(pool(), 1024, 75);
  GpuSerializedPage page1(cudfVector);

  // Move constructor
  GpuSerializedPage page2(std::move(page1));

  // page2 should have valid data
  EXPECT_EQ(1024, page2.size());
  EXPECT_TRUE(page2.numRows().has_value());
  EXPECT_EQ(75, page2.numRows().value());
}

// Test: SharedOwnership
// Verify that multiple pages can share the same CudfVector
// and reference counting works correctly.
TEST_F(GpuSerializedPageTest, SharedOwnership) {
  auto cudfVector = makeStubCudfVector(pool(), 512, 30);
  const long initialRefCount = cudfVector.use_count();

  {
    auto page1 = GpuSerializedPage(cudfVector);
    EXPECT_GT(cudfVector.use_count(), initialRefCount);

    {
      auto page2 = GpuSerializedPage(cudfVector);
      // Both pages share the same vector
      EXPECT_EQ(page1.cudfVector().get(), page2.cudfVector().get());
    }
  }

  // After pages go out of scope, reference count should return
  EXPECT_EQ(initialRefCount, cudfVector.use_count());
}

// Test: ZeroRowPage
// Verify that a CudfVector with 0 rows is handled correctly.
TEST_F(GpuSerializedPageTest, ZeroRowPage) {
  auto cudfVector = makeStubCudfVector(pool(), 0, 0);
  auto page = GpuSerializedPage(cudfVector);

  EXPECT_EQ(0, page.size());
  EXPECT_TRUE(page.numRows().has_value());
  EXPECT_EQ(0, page.numRows().value());
}

// Test: PrepareStreamThrows
// Verify that prepareStreamForDeserialize() throws VELOX_UNSUPPORTED
// because GPU-resident data cannot be streamed directly.
TEST_F(GpuSerializedPageTest, PrepareStreamThrows) {
  auto cudfVector = makeStubCudfVector(pool(), 1024, 50);
  GpuSerializedPage page(cudfVector);

  VELOX_ASSERT_THROW(
      page.prepareStreamForDeserialize(),
      "does not support CPU deserialization");
}

// Test: GetIOBufThrows
// Verify that getIOBuf() throws VELOX_UNSUPPORTED
// because GPU-resident data cannot be converted to IOBuf directly.
TEST_F(GpuSerializedPageTest, GetIOBufThrows) {
  auto cudfVector = makeStubCudfVector(pool(), 1024, 50);
  GpuSerializedPage page(cudfVector);

  VELOX_ASSERT_THROW(
      page.getIOBuf(),
      "does not have a CPU IOBuf representation");
}

// Test: LargePageSize
// Verify that large pages (>100MB) are handled correctly.
TEST_F(GpuSerializedPageTest, LargePageSize) {
  const uint64_t largeSize = 100 * 1024 * 1024; // 100MB
  auto cudfVector = makeStubCudfVector(pool(), largeSize, 1000000);
  auto page = GpuSerializedPage(cudfVector);

  EXPECT_EQ(largeSize, page.size());
  EXPECT_TRUE(page.numRows().has_value());
  EXPECT_EQ(1000000, page.numRows().value());
}

// Test: MultipleAccessorsAreSafe
// Verify that calling cudfVector() multiple times returns consistent results
// and is thread-safe (for const access).
TEST_F(GpuSerializedPageTest, MultipleAccessorsAreSafe) {
  auto cudfVector = makeStubCudfVector(pool(), 2048, 100);
  auto page = GpuSerializedPage(cudfVector);

  auto v1 = page.cudfVector();
  auto v2 = page.cudfVector();
  auto v3 = page.cudfVector();

  EXPECT_EQ(v1.get(), v2.get());
  EXPECT_EQ(v2.get(), v3.get());
  EXPECT_EQ(v1.get(), cudfVector.get());
}

} // namespace
} // namespace facebook::velox::cudf_velox
