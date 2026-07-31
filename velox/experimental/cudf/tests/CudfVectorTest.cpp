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

#include "velox/experimental/cudf/exec/PackedTableHostBuffer.h"
#include "velox/experimental/cudf/tests/utils/CudfStreamTestUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Exceptions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;
using namespace facebook::velox::cudf_velox::test;
using namespace facebook::velox::test;

namespace {

class TestCudaStream {
 public:
  TestCudaStream() {
    auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    VELOX_CHECK(
        status == cudaSuccess,
        "cudaStreamCreateWithFlags failed: {} ({})",
        cudaGetErrorString(status),
        static_cast<int>(status));
  }

  ~TestCudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  rmm::cuda_stream_view view() const {
    return rmm::cuda_stream_view{stream_};
  }

  cudaStream_t value() const {
    return stream_;
  }

 private:
  cudaStream_t stream_{nullptr};
};

std::unique_ptr<cudf::table> makeTable(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::array<int32_t, 4> values{{1, 2, 3, 4}};
  rmm::device_buffer data(values.size() * sizeof(int32_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      data.data(),
      values.data(),
      values.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      static_cast<cudf::size_type>(values.size()),
      std::move(data),
      rmm::device_buffer{},
      0));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::packed_table> makePackedTable(
    rmm::cuda_stream_view stream,
    RecordingAsyncDeviceResource& resource) {
  auto table = makeTable(stream, cudf::get_current_device_resource_ref());
  auto packedColumns = cudf::pack(
      table->view(),
      stream,
      rmm::to_device_async_resource_ref_checked(&resource));
  // CudfVector does not join producer streams. Synchronize the packing stream
  // before handing the packed table to CudfVector.
  stream.synchronize();
  auto tableView = cudf::unpack(packedColumns);
  return std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});
}

std::unique_ptr<cudf::packed_table> makeRepeatedPackedTable(
    cudf::size_type rows,
    rmm::cuda_stream_view stream,
    RecordingAsyncDeviceResource& resource) {
  rmm::device_buffer data(
      static_cast<std::size_t>(rows) * sizeof(int32_t),
      stream,
      rmm::to_device_async_resource_ref_checked(&resource));
  CUDF_CUDA_TRY(cudaMemsetAsync(data.data(), 0, data.size(), stream.value()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      rows,
      std::move(data),
      rmm::device_buffer{},
      0));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  auto packedColumns = cudf::pack(
      table->view(),
      stream,
      rmm::to_device_async_resource_ref_checked(&resource));
  stream.synchronize();
  auto tableView = cudf::unpack(packedColumns);
  return std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});
}

std::unique_ptr<cudf::packed_table> makeRepeatedStringPackedTable(
    cudf::size_type rows,
    rmm::cuda_stream_view stream,
    RecordingAsyncDeviceResource& resource) {
  constexpr std::string_view kValue = "repeated";
  std::vector<int32_t> offsets(rows + 1);
  for (cudf::size_type i = 0; i <= rows; ++i) {
    offsets[i] = i * kValue.size();
  }
  rmm::device_buffer offsetsData(
      offsets.size() * sizeof(int32_t),
      stream,
      rmm::to_device_async_resource_ref_checked(&resource));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsetsData.data(),
      offsets.data(),
      offsetsData.size(),
      cudaMemcpyHostToDevice,
      stream.value()));
  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      offsets.size(),
      std::move(offsetsData),
      rmm::device_buffer{},
      0);

  std::vector<char> chars(rows * kValue.size());
  for (cudf::size_type i = 0; i < rows; ++i) {
    std::memcpy(chars.data() + i * kValue.size(), kValue.data(), kValue.size());
  }
  rmm::device_buffer charsData(
      chars.size(),
      stream,
      rmm::to_device_async_resource_ref_checked(&resource));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      charsData.data(),
      chars.data(),
      charsData.size(),
      cudaMemcpyHostToDevice,
      stream.value()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_strings_column(
      rows,
      std::move(offsetsColumn),
      std::move(charsData),
      0,
      rmm::device_buffer{}));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  auto packedColumns = cudf::pack(
      table->view(),
      stream,
      rmm::to_device_async_resource_ref_checked(&resource));
  stream.synchronize();
  auto tableView = cudf::unpack(packedColumns);
  return std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});
}

class CudfVectorTest : public ::testing::Test, public VectorTestBase {
 protected:
  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
};

TEST_F(CudfVectorTest, rebindOwnedTableDeallocationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;

  auto table = makeTable(
      allocationStream.view(),
      rmm::to_device_async_resource_ref_checked(&resource));
  allocationStream.view().synchronize();

  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      table->num_rows(),
      std::move(table),
      allocationStream.view());
  resource.reset();

  ASSERT_TRUE(vector->rebindStream(targetStream.view()));
  vector.reset();

  EXPECT_GT(resource.deallocationCount(), 0);
  EXPECT_EQ(resource.lastDeallocationStream(), targetStream.value());
}

TEST_F(CudfVectorTest, rebindPackedTableDeallocationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(allocationStream.view(), resource);

  // Model the intra-node UCX path: the packed buffer was allocated on
  // allocationStream, but downstream work is associated with targetStream.
  // The CudfVector logical stream is already targetStream, but the packed
  // buffer's deallocation stream is still allocationStream. rebindStream must
  // update the packed buffer even when stream_ already matches targetStream.
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      targetStream.view());
  resource.reset();

  ASSERT_TRUE(vector->rebindStream(targetStream.view()));
  vector.reset();

  EXPECT_GT(resource.deallocationCount(), 0);
  EXPECT_EQ(resource.lastDeallocationStream(), targetStream.value());
}

TEST_F(CudfVectorTest, packedResidencyOwnerOutlivesDeviceBuffer) {
  TestCudaStream stream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(stream.view(), resource);
  resource.reset();

  bool ownerReleased = false;
  bool bufferReleasedBeforeOwner = false;
  auto owner = std::shared_ptr<void>(new int(1), [&](void* value) {
    ownerReleased = true;
    bufferReleasedBeforeOwner = resource.deallocationCount() > 0;
    delete static_cast<int*>(value);
  });
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      stream.view(),
      std::move(owner));

  EXPECT_FALSE(ownerReleased);
  vector.reset();
  EXPECT_TRUE(ownerReleased);
  EXPECT_TRUE(bufferReleasedBeforeOwner);
}

TEST_F(CudfVectorTest, exposesPackedStorageWithoutMaterializing) {
  TestCudaStream stream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(stream.view(), resource);
  const auto* expectedData = packedTable->data.gpu_data->data();
  const auto expectedDataSize = packedTable->data.gpu_data->size();
  const auto expectedMetadata = *packedTable->data.metadata;

  CudfVector packedVector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      stream.view());
  const auto* packedColumns = packedVector.getPackedColumns();
  ASSERT_NE(packedColumns, nullptr);
  EXPECT_EQ(packedColumns->gpu_data->data(), expectedData);
  EXPECT_EQ(packedColumns->gpu_data->size(), expectedDataSize);
  EXPECT_EQ(*packedColumns->metadata, expectedMetadata);

  auto table =
      makeTable(stream.view(), cudf::get_current_device_resource_ref());
  stream.view().synchronize();
  CudfVector tableVector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      table->num_rows(),
      std::move(table),
      stream.view());
  EXPECT_EQ(tableVector.getPackedColumns(), nullptr);
}

TEST_F(CudfVectorTest, packedHostRoundTripDoesNotRepack) {
  TestCudaStream stream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(stream.view(), resource);
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      stream.view());

  PackedTableHostBufferStats stats;
  auto hostBuffer = PackedTableHostBuffer::fromVector(
      std::move(vector),
      pool_.get(),
      rmm::to_device_async_resource_ref_checked(&resource),
      stats);
  EXPECT_EQ(stats.packedInputBatches, 1);
  EXPECT_EQ(stats.repackedInputBatches, 0);
  EXPECT_GT(stats.deviceToHostBytes, 0);

  auto restored = hostBuffer.toVector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      stream.view(),
      rmm::to_device_async_resource_ref_checked(&resource),
      stats);
  ASSERT_NE(restored->getPackedColumns(), nullptr);
  EXPECT_EQ(restored->size(), 4);
  EXPECT_EQ(stats.hostToDeviceBytes, stats.deviceToHostBytes);

  std::array<int32_t, 4> actual{};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      actual.data(),
      restored->getTableView().column(0).data<int32_t>(),
      actual.size() * sizeof(int32_t),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.view().synchronize();
  EXPECT_EQ(actual, (std::array<int32_t, 4>{{1, 2, 3, 4}}));
}

TEST_F(CudfVectorTest, compressedPackedHostRoundTrip) {
  TestCudaStream stream;
  RecordingAsyncDeviceResource resource;
  constexpr cudf::size_type kRows = 1 << 18;
  auto packedTable = makeRepeatedPackedTable(kRows, stream.view(), resource);
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      stream.view());

  PackedTableHostBufferStats stats;
  auto hostBuffer = PackedTableHostBuffer::fromVector(
      std::move(vector),
      pool_.get(),
      rmm::to_device_async_resource_ref_checked(&resource),
      stats,
      true);
  EXPECT_EQ(stats.hostUncompressedBytes, hostBuffer.uncompressedSize());
  EXPECT_EQ(stats.hostCompressedBytes, hostBuffer.size());
  EXPECT_LT(hostBuffer.size(), hostBuffer.uncompressedSize());
  EXPECT_GT(stats.hostCompressionNanos, 0);

  auto restored = hostBuffer.toVector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      stream.view(),
      rmm::to_device_async_resource_ref_checked(&resource),
      stats);
  ASSERT_NE(restored->getPackedColumns(), nullptr);
  EXPECT_EQ(restored->size(), kRows);
  EXPECT_GT(stats.hostDecompressionNanos, 0);
  EXPECT_EQ(stats.hostToDeviceBytes, stats.hostUncompressedBytes);

  int32_t lastValue = -1;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &lastValue,
      restored->getTableView().column(0).data<int32_t>() + kRows - 1,
      sizeof(lastValue),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.view().synchronize();
  EXPECT_EQ(lastValue, 0);
}

TEST_F(CudfVectorTest, dictionaryEncodedStringHostRoundTrip) {
  TestCudaStream stream;
  RecordingAsyncDeviceResource resource;
  constexpr cudf::size_type kRows = 1 << 18;
  auto packedTable =
      makeRepeatedStringPackedTable(kRows, stream.view(), resource);
  auto vector = std::make_shared<CudfVector>(
      pool_.get(),
      ROW({"c0"}, {VARCHAR()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      stream.view());

  PackedTableHostBufferStats stats;
  auto hostBuffer = PackedTableHostBuffer::fromVector(
      std::move(vector),
      pool_.get(),
      rmm::to_device_async_resource_ref_checked(&resource),
      stats,
      false,
      true);
  EXPECT_EQ(stats.dictionaryCandidateBatches, 1);
  EXPECT_EQ(stats.dictionaryEncodedBatches, 1);
  EXPECT_LT(stats.dictionaryOutputBytes, stats.dictionaryInputBytes);
  EXPECT_EQ(stats.dictionaryOutputBytes, stats.hostUncompressedBytes);
  EXPECT_EQ(hostBuffer.size(), stats.dictionaryOutputBytes);

  auto restored = hostBuffer.toVector(
      pool_.get(),
      ROW({"c0"}, {VARCHAR()}),
      stream.view(),
      rmm::to_device_async_resource_ref_checked(&resource),
      stats);
  EXPECT_EQ(restored->size(), kRows);
  EXPECT_EQ(
      restored->getTableView().column(0).type().id(), cudf::type_id::STRING);
  EXPECT_GT(stats.dictionaryDecodeNanos, 0);

  cudf::strings_column_view strings(restored->getTableView().column(0));
  int32_t finalOffset = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &finalOffset,
      strings.offsets().data<int32_t>() + kRows,
      sizeof(finalOffset),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.view().synchronize();
  EXPECT_EQ(finalOffset, kRows * std::string_view("repeated").size());
}

TEST_F(CudfVectorTest, packedTableReleaseUsesMaterializationStream) {
  TestCudaStream allocationStream;
  TestCudaStream targetStream;
  RecordingAsyncDeviceResource resource;
  auto packedTable = makePackedTable(allocationStream.view(), resource);

  // CudfVector::release materializes from a table_view that points into the
  // packed buffer. The packed buffer must be freed on the materialization
  // stream, not the original packing stream.
  CudfVector vector(
      pool_.get(),
      ROW({"c0"}, {INTEGER()}),
      packedTable->table.num_rows(),
      std::move(packedTable),
      targetStream.view());
  resource.reset();

  auto materialized = vector.release();

  EXPECT_GT(resource.deallocationCount(), 0);
  EXPECT_EQ(resource.lastDeallocationStream(), targetStream.value());
  targetStream.view().synchronize();
  EXPECT_EQ(materialized->num_columns(), 1);
  EXPECT_EQ(materialized->num_rows(), 4);
}

} // namespace
