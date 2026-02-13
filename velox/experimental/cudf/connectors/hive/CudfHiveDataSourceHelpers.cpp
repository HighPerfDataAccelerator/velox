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

#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSourceHelpers.hpp"

#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>

#include <rmm/device_buffer.hpp>

#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include <folly/futures/Future.h>

#include <filesystem>
#include <fstream>
#include <list>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using facebook::velox::cudf_velox::PinnedDataSourceBuffer;
using facebook::velox::cudf_velox::PinnedHostBuffer;

namespace {
template <typename T>
std::future<T> toStdFuture(folly::Future<T> follyFuture) {
  auto promise = std::make_shared<std::promise<T>>();
  auto stdFuture = promise->get_future();

  std::move(follyFuture).thenTry([promise](folly::Try<T>&& result) mutable {
    if (result.hasValue()) {
      promise->set_value(std::move(result.value()));
    } else {
      promise->set_exception(result.exception().to_exception_ptr());
    }
  });

  return stdFuture;
}
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  pendingChunks_.push_back(
      {std::shared_ptr(std::move(inputStream)), size, dst});
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  input_->load(velox::dwio::common::LogType::FILE);

  if (pendingChunks_.empty()) {
    return;
  }

  // Phase 1 (no GPU needed): read all chunks into one contiguous pinned buffer.
  uint64_t totalBytes = 0;
  for (const auto& chunk : pendingChunks_) {
    totalBytes += chunk.size;
  }

  auto pinnedBuf = std::make_shared<PinnedHostBuffer>(totalBytes);
  if (pinnedBuf->isPinned()) {
    pinnedAllocBytes_.fetch_add(totalBytes, std::memory_order_relaxed);
  } else {
    pageableAllocBytes_.fetch_add(totalBytes, std::memory_order_relaxed);
  }

  uint64_t hostOffset = 0;
  for (auto& chunk : pendingChunks_) {
    chunk.stream->readFully(
        reinterpret_cast<char*>(pinnedBuf->data() + hostOffset), chunk.size);
    hostOffset += chunk.size;
  }

  // Phase 2 (GPU): one bulk H2D into a staging device buffer, then D2D scatter.
  rmm::device_buffer staging(totalBytes, stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      staging.data(),
      pinnedBuf->data(),
      totalBytes,
      cudaMemcpyHostToDevice,
      stream.value()));

  hostOffset = 0;
  for (const auto& chunk : pendingChunks_) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        chunk.deviceDst,
        static_cast<uint8_t*>(staging.data()) + hostOffset,
        chunk.size,
        cudaMemcpyDeviceToDevice,
        stream.value()));
    hostOffset += chunk.size;
  }

  pendingChunks_.clear();
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  auto buf = std::make_shared<PinnedHostBuffer>(readSize);
  if (buf->isPinned()) {
    pinnedAllocBytes_.fetch_add(readSize, std::memory_order_relaxed);
  } else {
    pageableAllocBytes_.fetch_add(readSize, std::memory_order_relaxed);
  }
  readContiguous(offset, readSize, buf->data());
  return std::make_unique<PinnedDataSourceBuffer>(std::move(buf));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  auto future = folly::via(input_->executor())
                    .thenValue([this, offset, size, dst, stream](auto&&) {
                      auto hostBuffer = this->host_read(offset, size);
                      CUDF_CUDA_TRY(cudaMemcpyAsync(
                          dst,
                          hostBuffer->data(),
                          hostBuffer->size(),
                          cudaMemcpyHostToDevice,
                          stream.value()));
                      return hostBuffer->size();
                    });
  return toStdFuture(std::move(future));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

std::unique_ptr<cudf::io::datasource::buffer> fetchFooterBytes(
    std::shared_ptr<cudf::io::datasource> dataSource) {
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_footer_to_host(*dataSource);
}

std::unique_ptr<cudf::io::datasource::buffer> fetchPageIndexBytes(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::io::text::byte_range_info const pageIndexBytes) {
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_page_index_to_host(
      *dataSource, pageIndexBytes);
}

std::tuple<
    std::vector<rmm::device_buffer>,
    std::vector<cudf::device_span<uint8_t const>>,
    std::future<void>>
fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<cudf::io::text::byte_range_info const> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Using libcudf utility but may have custom implementation in the future
  return cudf::io::parquet::fetch_byte_ranges_to_device_async(
      *dataSource, byteRanges, stream, mr);
}

namespace {

namespace pqthrift = facebook::velox::parquet::thrift;

// Build parent→children index from the flat Parquet schema, which is stored
// in depth-first pre-order with each node carrying a num_children count.
std::vector<std::vector<int>> buildChildrenIdx(
    const std::vector<pqthrift::SchemaElement>& schema) {
  std::vector<std::vector<int>> childrenIdx(schema.size());
  if (schema.empty()) {
    return childrenIdx;
  }
  // Stack of (node_index, remaining_children_to_assign).
  std::stack<std::pair<int, int>> parentStack;
  int nc = schema[0].__isset.num_children ? schema[0].num_children : 0;
  if (nc > 0) {
    parentStack.push({0, nc});
  }
  for (size_t i = 1; i < schema.size(); ++i) {
    VELOX_CHECK(!parentStack.empty(), "Malformed Parquet schema");
    auto& [parent, remaining] = parentStack.top();
    childrenIdx[parent].push_back(static_cast<int>(i));
    if (--remaining == 0) {
      parentStack.pop();
    }
    int childNc =
        schema[i].__isset.num_children ? schema[i].num_children : 0;
    if (childNc > 0) {
      parentStack.push({static_cast<int>(i), childNc});
    }
  }
  return childrenIdx;
}

struct ColumnChunkRange {
  int64_t offset;
  int64_t size;
};

ColumnChunkRange getColumnChunkByteRange(
    const pqthrift::ColumnMetaData& meta) {
  int64_t start = meta.data_page_offset;
  if (meta.__isset.dictionary_page_offset &&
      meta.dictionary_page_offset > 0 &&
      meta.dictionary_page_offset < start) {
    start = meta.dictionary_page_offset;
  }
  return {start, meta.total_compressed_size};
}

std::string getTopLevelColumnName(
    const std::vector<std::string>& pathInSchema) {
  if (pathInSchema.empty()) {
    return "";
  }
  return pathInSchema[0];
}

} // anonymous namespace

std::shared_ptr<PinnedHostBuffer> selectiveParquetRead(
    const std::string& filePath,
    const std::vector<std::string>& readColumnNames,
    uint64_t splitStart) {
  const auto fileSize = std::filesystem::file_size(filePath);

  auto fullRead = [&]() {
    auto buf = std::make_shared<PinnedHostBuffer>(fileSize);
    std::ifstream f(filePath, std::ios::binary);
    VELOX_CHECK(f.good(), "Failed to open file: {}", filePath);
    f.read(reinterpret_cast<char*>(buf->data()), fileSize);
    return buf;
  };

  // Fall back to full read if no column projection or sub-file split
  if (readColumnNames.empty() || splitStart > 0) {
    return fullRead();
  }

  constexpr uint32_t kParquetMagic =
      (('P' << 0) | ('A' << 8) | ('R' << 16) | ('1' << 24));
  constexpr size_t kHeaderLen = 4; // "PAR1"
  constexpr size_t kEnderLen = 8; // footer_len(4) + magic(4)

  VELOX_CHECK(
      fileSize > kHeaderLen + kEnderLen,
      "Parquet file too small: {}",
      filePath);

  // Step 1: Read and parse the Parquet footer using Velox Thrift types,
  // avoiding any dependency on cuDF internal APIs.
  pqthrift::FileMetaData metadata;
  {
    std::ifstream file(filePath, std::ios::binary);
    VELOX_CHECK(file.good(), "Failed to open file: {}", filePath);

    file.seekg(
        static_cast<std::streamoff>(fileSize) -
        static_cast<std::streamoff>(kEnderLen));
    uint32_t footerLen = 0, magic = 0;
    file.read(reinterpret_cast<char*>(&footerLen), sizeof(footerLen));
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    VELOX_CHECK(
        magic == kParquetMagic, "Invalid Parquet magic in: {}", filePath);
    VELOX_CHECK(
        footerLen > 0 && footerLen <= fileSize - kHeaderLen - kEnderLen,
        "Invalid footer length in: {}",
        filePath);

    std::vector<uint8_t> footerBytes(footerLen);
    file.seekg(
        static_cast<std::streamoff>(fileSize) -
        static_cast<std::streamoff>(kEnderLen) -
        static_cast<std::streamoff>(footerLen));
    file.read(reinterpret_cast<char*>(footerBytes.data()), footerLen);

    auto transport =
        std::make_shared<apache::thrift::transport::TMemoryBuffer>(
            footerBytes.data(),
            static_cast<uint32_t>(footerLen),
            apache::thrift::transport::TMemoryBuffer::OBSERVE);
    apache::thrift::protocol::TCompactProtocolT<
        apache::thrift::transport::TMemoryBuffer>
        protocol(transport);
    metadata.read(&protocol);
  }

  // Step 2: Build children indices for schema tree navigation.
  const auto childrenIdx = buildChildrenIdx(metadata.schema);

  // Step 3: Build the set of needed column names
  std::unordered_set<std::string> neededColumns(
      readColumnNames.begin(), readColumnNames.end());

  // Step 4: For each row group, identify needed column chunks and their
  // byte ranges. Also build the clipped metadata.
  struct ChunkReadInfo {
    int64_t srcOffset;
    int64_t size;
  };
  std::vector<ChunkReadInfo> chunksToRead;
  int64_t totalChunkBytes = 0;

  pqthrift::FileMetaData clippedMetadata;
  clippedMetadata.version = metadata.version;
  clippedMetadata.num_rows = metadata.num_rows;
  if (metadata.__isset.created_by) {
    clippedMetadata.created_by = metadata.created_by;
    clippedMetadata.__isset.created_by = true;
  }

  // Copy key_value_metadata but strip ARROW:schema since it describes all
  // original columns and would mismatch the clipped schema.
  if (metadata.__isset.key_value_metadata) {
    for (const auto& kv : metadata.key_value_metadata) {
      if (kv.key != "ARROW:schema") {
        clippedMetadata.key_value_metadata.push_back(kv);
      }
    }
    clippedMetadata.__isset.key_value_metadata = true;
  }

  // column_orders has one entry per leaf column in schema order.  After
  // clipping, the indices no longer match, so omit it to let cuDF use
  // the default (safe) sort order for statistics interpretation.

  // Clip the schema: keep root + needed top-level columns and their children.
  // The Parquet schema is a flattened tree. Element 0 is root, then children
  // follow in depth-first order.
  {
    const auto& origSchema = metadata.schema;
    VELOX_CHECK(!origSchema.empty(), "Empty Parquet schema");

    pqthrift::SchemaElement clippedRoot = origSchema[0];
    clippedRoot.num_children = 0;
    clippedRoot.__isset.num_children = true;
    std::vector<pqthrift::SchemaElement> clippedSchema;
    clippedSchema.push_back(clippedRoot);

    const auto& rootChildren = childrenIdx[0];
    for (int childIdx : rootChildren) {
      const auto& childElem = origSchema[childIdx];
      if (neededColumns.count(childElem.name) == 0) {
        continue;
      }

      // Collect this column's entire sub-tree (DFS)
      std::vector<int> subtreeIndices;
      std::vector<int> dfsStack = {childIdx};
      while (!dfsStack.empty()) {
        int idx = dfsStack.back();
        dfsStack.pop_back();
        subtreeIndices.push_back(idx);
        const auto& children = childrenIdx[idx];
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
          dfsStack.push_back(*it);
        }
      }

      clippedSchema[0].num_children++;
      for (int si : subtreeIndices) {
        clippedSchema.push_back(origSchema[si]);
      }
    }

    clippedMetadata.schema = std::move(clippedSchema);
  }

  for (auto& rg : metadata.row_groups) {
    pqthrift::RowGroup clippedRg;
    clippedRg.total_byte_size = 0;
    clippedRg.num_rows = rg.num_rows;
    clippedRg.total_compressed_size = 0;
    clippedRg.__isset.total_compressed_size = true;
    if (rg.__isset.ordinal) {
      clippedRg.ordinal = rg.ordinal;
      clippedRg.__isset.ordinal = true;
    }

    for (auto& cc : rg.columns) {
      const auto colName =
          getTopLevelColumnName(cc.meta_data.path_in_schema);
      if (neededColumns.count(colName) == 0) {
        continue;
      }

      auto range = getColumnChunkByteRange(cc.meta_data);
      chunksToRead.push_back({range.offset, range.size});
      totalChunkBytes += range.size;

      clippedRg.columns.push_back(cc);
      clippedRg.total_byte_size += cc.meta_data.total_uncompressed_size;
      clippedRg.total_compressed_size += cc.meta_data.total_compressed_size;
    }

    clippedMetadata.row_groups.push_back(std::move(clippedRg));
  }

  // Check if selective read saves significant IO (>20% savings)
  const int64_t fullReadCost = static_cast<int64_t>(fileSize);
  if (totalChunkBytes > fullReadCost * 80 / 100) {
    return fullRead();
  }

  // Step 5: Build compact Parquet buffer.
  // Layout: [PAR1 header][column chunks contiguously][clipped footer][ender]
  // Update column chunk offsets in clippedMetadata to point to new positions.
  int64_t writeOffset = static_cast<int64_t>(kHeaderLen);
  size_t chunkIdx = 0;
  for (auto& rg : clippedMetadata.row_groups) {
    rg.file_offset = writeOffset;
    rg.__isset.file_offset = true;
    for (auto& cc : rg.columns) {
      auto& meta = cc.meta_data;
      const int64_t dictOffset = meta.__isset.dictionary_page_offset
          ? meta.dictionary_page_offset
          : 0;
      const int64_t dataOffset = meta.data_page_offset;

      if (dictOffset > 0 && dictOffset < dataOffset) {
        meta.dictionary_page_offset = writeOffset;
        meta.__isset.dictionary_page_offset = true;
        meta.data_page_offset = writeOffset + (dataOffset - dictOffset);
      } else {
        meta.data_page_offset = writeOffset;
        meta.__isset.dictionary_page_offset = false;
      }
      meta.__isset.index_page_offset = false;

      cc.file_offset = meta.data_page_offset + meta.total_compressed_size;
      cc.__isset.offset_index_offset = false;
      cc.__isset.offset_index_length = false;
      cc.__isset.column_index_offset = false;
      cc.__isset.column_index_length = false;

      writeOffset += chunksToRead[chunkIdx].size;
      ++chunkIdx;
    }
  }

  // Step 6: Serialize the clipped footer using Velox Thrift Compact Protocol.
  auto outTransport =
      std::make_shared<apache::thrift::transport::TMemoryBuffer>();
  {
    apache::thrift::protocol::TCompactProtocolT<
        apache::thrift::transport::TMemoryBuffer>
        outProtocol(outTransport);
    clippedMetadata.write(&outProtocol);
  }
  uint8_t* newFooterData = nullptr;
  uint32_t newFooterSize = 0;
  outTransport->getBuffer(&newFooterData, &newFooterSize);

  const size_t totalBufSize =
      kHeaderLen + totalChunkBytes + newFooterSize + kEnderLen;
  auto buf = std::make_shared<PinnedHostBuffer>(totalBufSize);
  uint8_t* dst = buf->data();

  // Write PAR1 header
  const uint32_t magic = kParquetMagic;
  std::memcpy(dst, &magic, kHeaderLen);
  size_t dstOffset = kHeaderLen;

  // Step 7: Read needed column chunks from file and write to buffer
  std::ifstream file(filePath, std::ios::binary);
  VELOX_CHECK(file.good(), "Failed to open file: {}", filePath);
  for (const auto& chunk : chunksToRead) {
    file.seekg(chunk.srcOffset);
    file.read(reinterpret_cast<char*>(dst + dstOffset), chunk.size);
    dstOffset += chunk.size;
  }

  // Write footer
  std::memcpy(dst + dstOffset, newFooterData, newFooterSize);
  dstOffset += newFooterSize;

  // Write ender (footer_len + magic)
  std::memcpy(dst + dstOffset, &newFooterSize, sizeof(newFooterSize));
  dstOffset += sizeof(newFooterSize);
  std::memcpy(dst + dstOffset, &magic, sizeof(magic));

  LOG(INFO) << "Selective Parquet read: " << filePath
            << " full=" << fileSize
            << " selective=" << totalBufSize
            << " saved=" << (fileSize - totalBufSize)
            << " (" << (100 - totalBufSize * 100 / fileSize) << "%)";

  return buf;
}

} // namespace facebook::velox::cudf_velox::connector::hive
