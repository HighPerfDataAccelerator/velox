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

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <chrono>
#include <optional>

namespace facebook::velox::cudf_velox {
namespace {

uint64_t nowNanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

} // namespace

PackedTableHostBuffer PackedTableHostBuffer::fromVector(
    CudfVectorPtr input,
    memory::MemoryPool* pool,
    rmm::device_async_resource_ref mr,
    PackedTableHostBufferStats& stats) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_NOT_NULL(pool);
  auto stream = input->stream();

  std::optional<cudf::packed_columns> repacked;
  auto* packedColumns = input->getPackedColumns();
  if (packedColumns == nullptr) {
    const auto start = nowNanos();
    repacked.emplace(cudf::pack(input->getTableView(), stream, mr));
    stream.synchronize();
    stats.repackNanos += nowNanos() - start;
    ++stats.repackedInputBatches;
    packedColumns = &*repacked;
  } else {
    ++stats.packedInputBatches;
  }

  auto hostData = AlignedBuffer::allocateExact<uint8_t>(
      packedColumns->gpu_data->size(), pool);
  const auto bytes = packedColumns->gpu_data->size();
  if (bytes > 0) {
    const auto start = nowNanos();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostData->asMutable<uint8_t>(),
        packedColumns->gpu_data->data(),
        bytes,
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    stats.deviceToHostNanos += nowNanos() - start;
    stats.deviceToHostBytes += bytes;
  }

  return PackedTableHostBuffer{
      *packedColumns->metadata, std::move(hostData), input->size()};
}

CudfVectorPtr PackedTableHostBuffer::toVector(
    memory::MemoryPool* pool,
    const RowTypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    PackedTableHostBufferStats& stats) {
  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK_NOT_NULL(type);
  VELOX_CHECK_NOT_NULL(data_);

  const auto bytes = data_->size();
  auto gpuData = std::make_unique<rmm::device_buffer>(bytes, stream, mr);
  if (bytes > 0) {
    const auto start = nowNanos();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        gpuData->data(),
        data_->as<uint8_t>(),
        bytes,
        cudaMemcpyHostToDevice,
        stream.value()));
    stream.synchronize();
    stats.hostToDeviceNanos += nowNanos() - start;
    stats.hostToDeviceBytes += bytes;
  }

  auto metadata = std::make_unique<std::vector<uint8_t>>(std::move(metadata_));
  cudf::packed_columns packedColumns{std::move(metadata), std::move(gpuData)};
  auto tableView = cudf::unpack(packedColumns);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedColumns)});
  auto output = std::make_shared<CudfVector>(
      pool, type, numRows_, std::move(packedTable), stream);
  data_.reset();
  numRows_ = 0;
  return output;
}

} // namespace facebook::velox::cudf_velox
