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

#include "velox/common/compression/Compression.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
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
    PackedTableHostBufferStats& stats,
    bool compress,
    bool dictionaryEncodeStrings) {
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

  struct DictionaryCandidate {
    column_index_t columnIndex;
    cudf::packed_columns keys;
  };
  std::optional<cudf::packed_columns> dictionaryMainPacked;
  std::vector<DictionaryCandidate> dictionaryCandidates;
  std::vector<std::unique_ptr<cudf::column>> dictionaryOwners;
  if (dictionaryEncodeStrings) {
    const auto tableView = input->getTableView();
    std::vector<cudf::column_view> encodedViews;
    encodedViews.reserve(tableView.num_columns());
    dictionaryOwners.reserve(tableView.num_columns());
    const auto start = nowNanos();
    for (column_index_t i = 0; i < tableView.num_columns(); ++i) {
      const auto column = tableView.column(i);
      if (column.type().id() == cudf::type_id::STRING) {
        auto encoded = cudf::dictionary::encode(
            column, cudf::data_type{cudf::type_id::INT32}, stream, mr);
        const cudf::dictionary_column_view dictionary{encoded->view()};
        encodedViews.push_back(dictionary.get_indices_annotated());
        dictionaryCandidates.push_back(
            {i, cudf::pack(cudf::table_view{{dictionary.keys()}}, stream, mr)});
        dictionaryOwners.push_back(std::move(encoded));
      } else {
        encodedViews.push_back(column);
      }
    }
    if (!dictionaryCandidates.empty()) {
      ++stats.dictionaryCandidateBatches;
      dictionaryMainPacked.emplace(
          cudf::pack(cudf::table_view{encodedViews}, stream, mr));
      stream.synchronize();
      stats.dictionaryEncodeNanos += nowNanos() - start;
      const auto dictionaryBytes = std::accumulate(
          dictionaryCandidates.begin(),
          dictionaryCandidates.end(),
          dictionaryMainPacked->gpu_data->size(),
          [](uint64_t sum, const auto& candidate) {
            return sum + candidate.keys.gpu_data->size();
          });
      stats.dictionaryInputBytes += packedColumns->gpu_data->size();
      stats.dictionaryOutputBytes += dictionaryBytes;
      if (dictionaryBytes < packedColumns->gpu_data->size()) {
        ++stats.dictionaryEncodedBatches;
        packedColumns = &*dictionaryMainPacked;
      } else {
        dictionaryCandidates.clear();
        dictionaryOwners.clear();
        dictionaryMainPacked.reset();
      }
    }
  }

  const auto bytes = packedColumns->gpu_data->size();
  std::vector<DictionaryHostColumn> dictionaryColumns;
  dictionaryColumns.reserve(dictionaryCandidates.size());
  uint64_t dictionaryBytes = 0;
  for (const auto& candidate : dictionaryCandidates) {
    const auto keyBytes = candidate.keys.gpu_data->size();
    memory::ContiguousAllocation keyData;
    if (keyBytes > 0) {
      pool->allocateContiguous(
          memory::AllocationTraits::numPages(keyBytes), keyData);
    }
    dictionaryBytes += keyBytes;
    dictionaryColumns.push_back(DictionaryHostColumn{
        candidate.columnIndex,
        *candidate.keys.metadata,
        std::move(keyData),
        keyBytes});
  }
  memory::ContiguousAllocation hostData;
  if (bytes > 0) {
    pool->allocateContiguous(
        memory::AllocationTraits::numPages(bytes), hostData);
  }
  if (bytes > 0 || dictionaryBytes > 0) {
    const auto start = nowNanos();
    if (bytes > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostData.data(),
          packedColumns->gpu_data->data(),
          bytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    for (size_t i = 0; i < dictionaryCandidates.size(); ++i) {
      const auto keyBytes = dictionaryColumns[i].dataSize;
      if (keyBytes > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dictionaryColumns[i].data.data(),
            dictionaryCandidates[i].keys.gpu_data->data(),
            keyBytes,
            cudaMemcpyDeviceToHost,
            stream.value()));
      }
    }
    stream.synchronize();
    stats.deviceToHostNanos += nowNanos() - start;
    stats.deviceToHostBytes += bytes + dictionaryBytes;
  }

  stats.hostUncompressedBytes += bytes + dictionaryBytes;
  uint64_t hostDataSize = bytes;
  if (compress && bytes > 0) {
    auto codecResult =
        common::Codec::create(common::CompressionKind::CompressionKind_LZ4);
    VELOX_CHECK(
        codecResult.hasValue(),
        "Failed to create LZ4 codec: {}",
        codecResult.error().toString());
    auto codec = std::move(codecResult).value();
    const auto maxCompressedSize = codec->maxCompressedLength(bytes);
    memory::ContiguousAllocation compressed;
    pool->allocateContiguous(
        memory::AllocationTraits::numPages(maxCompressedSize), compressed);
    const auto start = nowNanos();
    auto compressedSizeResult = codec->compress(
        hostData.data(), bytes, compressed.data(), compressed.size());
    VELOX_CHECK(
        compressedSizeResult.hasValue(),
        "Failed to compress packed table: {}",
        compressedSizeResult.error().toString());
    const auto compressedSize = compressedSizeResult.value();
    stats.hostCompressionNanos += nowNanos() - start;
    if (compressedSize < bytes) {
      memory::ContiguousAllocation exactCompressed;
      pool->allocateContiguous(
          memory::AllocationTraits::numPages(compressedSize), exactCompressed);
      std::memcpy(exactCompressed.data(), compressed.data(), compressedSize);
      hostData.pool()->freeContiguous(hostData);
      hostData = std::move(exactCompressed);
      hostDataSize = compressedSize;
    }
  }
  stats.hostCompressedBytes += hostDataSize + dictionaryBytes;

  return PackedTableHostBuffer{
      *packedColumns->metadata,
      std::move(hostData),
      hostDataSize,
      input->size(),
      bytes,
      std::move(dictionaryColumns)};
}

CudfVectorPtr PackedTableHostBuffer::toVector(
    memory::MemoryPool* pool,
    const RowTypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    PackedTableHostBufferStats& stats) {
  VELOX_CHECK_NOT_NULL(pool);
  VELOX_CHECK_NOT_NULL(type);
  VELOX_CHECK(dataSize_ == 0 || !data_.empty());

  memory::ContiguousAllocation uncompressedData;
  if (dataSize_ < uncompressedSize_) {
    auto codecResult =
        common::Codec::create(common::CompressionKind::CompressionKind_LZ4);
    VELOX_CHECK(
        codecResult.hasValue(),
        "Failed to create LZ4 codec: {}",
        codecResult.error().toString());
    auto codec = std::move(codecResult).value();
    pool->allocateContiguous(
        memory::AllocationTraits::numPages(uncompressedSize_),
        uncompressedData);
    const auto start = nowNanos();
    auto decompressedSizeResult = codec->decompress(
        data_.data(),
        dataSize_,
        uncompressedData.data(),
        uncompressedData.size());
    VELOX_CHECK(
        decompressedSizeResult.hasValue(),
        "Failed to decompress packed table: {}",
        decompressedSizeResult.error().toString());
    VELOX_CHECK_EQ(decompressedSizeResult.value(), uncompressedSize_);
    stats.hostDecompressionNanos += nowNanos() - start;
  }

  const auto bytes = uncompressedSize_;
  const auto* hostData =
      uncompressedData.empty() ? data_.data() : uncompressedData.data();
  auto gpuData = std::make_unique<rmm::device_buffer>(bytes, stream, mr);
  std::vector<std::pair<column_index_t, cudf::packed_columns>>
      deviceDictionaries;
  deviceDictionaries.reserve(dictionaryColumns_.size());
  uint64_t dictionaryBytes = 0;
  for (auto& dictionary : dictionaryColumns_) {
    auto dictionaryGpuData =
        std::make_unique<rmm::device_buffer>(dictionary.dataSize, stream, mr);
    auto dictionaryMetadata =
        std::make_unique<std::vector<uint8_t>>(std::move(dictionary.metadata));
    dictionaryBytes += dictionary.dataSize;
    deviceDictionaries.emplace_back(
        dictionary.columnIndex,
        cudf::packed_columns{
            std::move(dictionaryMetadata), std::move(dictionaryGpuData)});
  }
  if (bytes > 0 || dictionaryBytes > 0) {
    const auto start = nowNanos();
    if (bytes > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          gpuData->data(),
          hostData,
          bytes,
          cudaMemcpyHostToDevice,
          stream.value()));
    }
    for (size_t i = 0; i < dictionaryColumns_.size(); ++i) {
      const auto dictionarySize = dictionaryColumns_[i].dataSize;
      if (dictionarySize > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            deviceDictionaries[i].second.gpu_data->data(),
            dictionaryColumns_[i].data.data(),
            dictionarySize,
            cudaMemcpyHostToDevice,
            stream.value()));
      }
    }
    stream.synchronize();
    stats.hostToDeviceNanos += nowNanos() - start;
    stats.hostToDeviceBytes += bytes + dictionaryBytes;
  }

  auto metadata = std::make_unique<std::vector<uint8_t>>(std::move(metadata_));
  cudf::packed_columns packedColumns{std::move(metadata), std::move(gpuData)};
  auto tableView = cudf::unpack(packedColumns);
  CudfVectorPtr output;
  if (dictionaryColumns_.empty()) {
    auto packedTable = std::make_unique<cudf::packed_table>(
        cudf::packed_table{tableView, std::move(packedColumns)});
    output = std::make_shared<CudfVector>(
        pool, type, numRows_, std::move(packedTable), stream);
  } else {
    std::vector<bool> isDictionaryColumn(tableView.num_columns(), false);
    for (const auto& dictionary : dictionaryColumns_) {
      VELOX_CHECK_LT(dictionary.columnIndex, tableView.num_columns());
      isDictionaryColumn[dictionary.columnIndex] = true;
    }
    std::vector<std::unique_ptr<cudf::column>> decodedColumns;
    decodedColumns.reserve(tableView.num_columns());
    const auto start = nowNanos();
    for (column_index_t i = 0; i < tableView.num_columns(); ++i) {
      const auto column = tableView.column(i);
      if (isDictionaryColumn[i]) {
        const auto dictionary = std::find_if(
            deviceDictionaries.begin(),
            deviceDictionaries.end(),
            [&](const auto& entry) { return entry.first == i; });
        VELOX_CHECK(
            dictionary != deviceDictionaries.end(),
            "Missing packed dictionary keys for column {}",
            i);
        const auto keysTable = cudf::unpack(dictionary->second);
        VELOX_CHECK_EQ(keysTable.num_columns(), 1);
        auto gathered = cudf::gather(
            keysTable,
            column,
            cudf::out_of_bounds_policy::DONT_CHECK,
            cudf::negative_index_policy::NOT_ALLOWED,
            stream,
            mr);
        auto columns = gathered->release();
        decodedColumns.push_back(std::move(columns[0]));
      } else {
        decodedColumns.push_back(
            std::make_unique<cudf::column>(column, stream, mr));
      }
    }
    stream.synchronize();
    stats.dictionaryDecodeNanos += nowNanos() - start;
    output = std::make_shared<CudfVector>(
        pool,
        type,
        numRows_,
        std::make_unique<cudf::table>(std::move(decodedColumns)),
        stream);
  }
  if (!data_.empty()) {
    data_.pool()->freeContiguous(data_);
  }
  for (auto& dictionary : dictionaryColumns_) {
    if (!dictionary.data.empty()) {
      dictionary.data.pool()->freeContiguous(dictionary.data);
    }
  }
  dataSize_ = 0;
  numRows_ = 0;
  uncompressedSize_ = 0;
  dictionaryColumns_.clear();
  return output;
}

} // namespace facebook::velox::cudf_velox
