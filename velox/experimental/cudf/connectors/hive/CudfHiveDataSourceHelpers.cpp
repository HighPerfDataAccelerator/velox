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

#include <folly/futures/Future.h>

#include <string>
#include <unordered_map>
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

std::vector<std::unique_ptr<cudf::io::datasource>>
makeDataSourcesFromSourceInfo(
    const cudf::io::source_info& info,
    size_t offset,
    size_t maxSizeEstimate) {
  return cudf::io::make_datasources(info, offset, maxSizeEstimate);
}

} // namespace facebook::velox::cudf_velox::connector::hive
