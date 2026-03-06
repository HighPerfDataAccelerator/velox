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

#include "velox/dwio/parquet/thrift/ParquetThriftTypes.h"
#include "velox/type/Filter.h"
#include "velox/type/Type.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::parquet {

/// Reusable Parquet metadata utilities for footer parsing, column pruning,
/// and byte-range computation. Extracted from the Velox Parquet reader
/// internals so they can be used by external components (e.g., GPU data
/// sources) without pulling in the full reader pipeline.
class ParquetMetadataUtil {
 public:
  /// Deserialize a Parquet FileMetaData from raw Thrift compact bytes.
  static std::unique_ptr<thrift::FileMetaData> loadFileMetaData(
      const char* buffer,
      uint32_t length);

  /// Read the Parquet footer from a file on disk and deserialize it.
  /// Reads trailing bytes to find footer length, then deserializes via Thrift.
  /// If the footer is larger than footerEstimatedSize, performs a second read.
  static std::unique_ptr<thrift::FileMetaData> loadFileMetaDataFromFile(
      const std::string& filePath,
      uint64_t footerEstimatedSize = 16UL * 1024);

  /// Resolve column names to their leaf column indices in the Parquet schema.
  /// Uses the first row group's ColumnChunk metadata to match path_in_schema[0]
  /// against the requested names. Returns indices in the order of columnNames.
  static std::vector<uint32_t> resolveColumnIndices(
      const thrift::FileMetaData& metadata,
      const std::vector<std::string>& columnNames);

  /// Byte range within a Parquet file.
  struct ByteRange {
    uint64_t offset;
    uint64_t size;
  };

  /// Compute byte ranges for specific columns in specific RowGroups.
  /// Each range covers the full column chunk (dict + data pages).
  /// Returned in rowGroup × column iteration order.
  static std::vector<ByteRange> getColumnChunkByteRanges(
      const thrift::FileMetaData& metadata,
      const std::vector<uint32_t>& rowGroupIndices,
      const std::vector<uint32_t>& columnIndices);

  /// Serialize a FileMetaData into Thrift compact binary format.
  static std::string serializeFileMetaData(
      const thrift::FileMetaData& metadata);

  /// Build a clipped FileMetaData containing only the specified RowGroups
  /// and columns, with column chunk offsets adjusted for a packed buffer.
  /// byteRanges must be in the same rowGroup × column order as returned
  /// by getColumnChunkByteRanges(). dataOffset is the byte position of
  /// the first chunk in the output buffer (typically 4 for leading "PAR1").
  static thrift::FileMetaData buildClippedFileMetaData(
      const thrift::FileMetaData& original,
      const std::vector<uint32_t>& rowGroupIndices,
      const std::vector<uint32_t>& columnIndices,
      const std::vector<ByteRange>& byteRanges,
      uint64_t dataOffset);

  /// Filter candidate row groups using column chunk statistics (min/max).
  /// Tests each filter against the statistics of the corresponding column
  /// chunk in each row group. Row groups whose statistics prove the filter
  /// cannot match are excluded.
  ///
  /// @param metadata         The full Parquet FileMetaData.
  /// @param candidateRowGroups Row group indices to consider.
  /// @param columnFilters    Pairs of (column_name, Filter*) to evaluate.
  /// @param dataColumnsType  RowType describing all data columns for type
  ///                         resolution when decoding statistics.
  /// @return Row group indices from candidateRowGroups that were not
  ///         excluded by statistics.
  static std::vector<uint32_t> filterRowGroupsByStatistics(
      const thrift::FileMetaData& metadata,
      const std::vector<uint32_t>& candidateRowGroups,
      const std::vector<std::pair<std::string, const common::Filter*>>&
          columnFilters,
      const RowTypePtr& dataColumnsType);
};

} // namespace facebook::velox::parquet
