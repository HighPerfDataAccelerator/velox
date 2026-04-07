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

#pragma once

#include "velox/dwio/common/BufferedInput.h"

#include <atomic>
#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox {
class ReadFile;
} // namespace facebook::velox

namespace facebook::velox::cudf_velox {
class PinnedHostBuffer;
} // namespace facebook::velox::cudf_velox

namespace facebook::velox::cudf_velox::connector::hive {

// ---------------- Internal helper ----------------
// A cudf::io::datasource that serves bytes via Velox BufferedInput so that
// reads benefit from AsyncDataCache / SSD cache and are always returned as
// contiguous buffers.
class BufferedInputDataSource : public cudf::io::datasource {
 public:
  explicit BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      rmm::cuda_stream_view stream) override;

  // Use the enqueue API from dwio::common::BufferedInput.
  // Pass a device buffer to copy to after load.
  void enqueueForDevice(uint64_t offset, uint64_t size, uint8_t* dst);

  // loads and copies to device.
  void load(rmm::cuda_stream_view stream);

  uint64_t pinnedAllocBytes() const {
    return pinnedAllocBytes_.load(std::memory_order_relaxed);
  }
  uint64_t pageableAllocBytes() const {
    return pageableAllocBytes_.load(std::memory_order_relaxed);
  }

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  struct PendingChunk {
    std::shared_ptr<facebook::velox::dwio::common::SeekableInputStream> stream;
    uint64_t size;
    uint8_t* deviceDst;
  };

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  std::vector<PendingChunk> pendingChunks_;
  std::atomic<uint64_t> pinnedAllocBytes_{0};
  std::atomic<uint64_t> pageableAllocBytes_{0};
};

// ---------------- Internal helper ----------------
// Convert a filter expression such that all `ast::column_reference`s are
// replaced with `ast::column_name_reference`s from the selected columns or
// the schema tree.
// TODO(mh): Remove this once https://github.com/rapidsai/cudf/pull/20604 is
// merged
class referenceToNameConverter
    : public cudf::ast::detail::expression_transformer {
 public:
  explicit referenceToNameConverter(
      std::optional<std::reference_wrapper<const cudf::ast::expression>> expr,
      const std::vector<cudf::io::parquet::SchemaElement>& schemaTree,
      cudf::host_span<const std::string> readColumnNames);

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::literal& expr) override;

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::column_reference& expr) override;

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::column_name_reference& expr) override;

  std::reference_wrapper<const cudf::ast::expression> visit(
      const cudf::ast::operation& expr) override;

  // Returns the converted AST expression
  [[nodiscard]] std::reference_wrapper<const cudf::ast::expression>
  convertedExpression() const;

 private:
  std::vector<std::reference_wrapper<const cudf::ast::expression>>
  visitOperands(
      cudf::host_span<const std::reference_wrapper<const cudf::ast::expression>>
          operands);
  cudf::ast::tree convertedExpr_;
  std::unordered_map<cudf::size_type, std::string> indicesToColumnNames_;
};

// Fetch a host buffer containing parquet source footer from a data source.
std::unique_ptr<cudf::io::datasource::buffer> fetchFooterBytes(
    std::shared_ptr<cudf::io::datasource> dataSource);

std::vector<std::unique_ptr<cudf::io::datasource>>
makeDataSourcesFromSourceInfo(
    const cudf::io::source_info& info,
    size_t offset = 0,
    size_t maxSizeEstimate = 0);

/// Read a Parquet file selectively: parse footer, identify needed column
/// chunks based on readColumnNames, read only those byte ranges, and
/// reconstruct a compact valid Parquet buffer in pinned host memory.
/// Falls back to reading the full file if readColumnNames is empty,
/// splitStart > 0 (sub-file splits), or selective read would not save IO.
/// Uses the Velox ReadFile abstraction, supporting local, HDFS, S3, etc.
std::shared_ptr<facebook::velox::cudf_velox::PinnedHostBuffer>
selectiveParquetRead(
    facebook::velox::ReadFile* readFile,
    const std::vector<std::string>& readColumnNames,
    uint64_t splitStart = 0);

} // namespace facebook::velox::cudf_velox::connector::hive
