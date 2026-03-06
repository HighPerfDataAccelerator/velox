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

#include "velox/dwio/parquet/reader/ParquetMetadataUtil.h"
#include "velox/dwio/parquet/thrift/ThriftTransport.h"

#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/Statistics.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace facebook::velox::parquet {

// ===================================================================
// Anonymous-namespace helpers for statistics-based row group filtering.
// Adapted from Metadata.cpp (getMin/getMax, buildColumnStatisticsFromThrift)
// and ScanSpec.cpp (testIntFilter, testDoubleFilter, testStringFilter,
// testBoolFilter, testFilter).
// ===================================================================

namespace {

template <typename T>
inline T loadValue(const char* ptr) {
  T ret;
  std::memcpy(&ret, ptr, sizeof(ret));
  return ret;
}

template <typename T>
inline std::optional<T> getMin(const thrift::Statistics& stats) {
  if (stats.__isset.min_value) {
    return loadValue<T>(stats.min_value.data());
  }
  if (stats.__isset.min) {
    return loadValue<T>(stats.min.data());
  }
  return std::nullopt;
}

template <typename T>
inline std::optional<T> getMax(const thrift::Statistics& stats) {
  if (stats.__isset.max_value) {
    return loadValue<T>(stats.max_value.data());
  }
  if (stats.__isset.max) {
    return loadValue<T>(stats.max.data());
  }
  return std::nullopt;
}

template <>
inline std::optional<std::string> getMin<std::string>(
    const thrift::Statistics& stats) {
  if (stats.__isset.min_value) {
    return stats.min_value;
  }
  if (stats.__isset.min) {
    return stats.min;
  }
  return std::nullopt;
}

template <>
inline std::optional<std::string> getMax<std::string>(
    const thrift::Statistics& stats) {
  if (stats.__isset.max_value) {
    return stats.max_value;
  }
  if (stats.__isset.max) {
    return stats.max;
  }
  return std::nullopt;
}

std::unique_ptr<dwio::common::ColumnStatistics>
buildColumnStatisticsFromThrift(
    const thrift::Statistics& stats,
    const velox::Type& type,
    uint64_t numRowsInRowGroup) {
  std::optional<uint64_t> nullCount = stats.__isset.null_count
      ? std::optional<uint64_t>(stats.null_count)
      : std::nullopt;
  std::optional<uint64_t> valueCount = nullCount.has_value()
      ? std::optional<uint64_t>(numRowsInRowGroup - nullCount.value())
      : std::nullopt;
  std::optional<bool> hasNull = stats.__isset.null_count
      ? std::optional<bool>(stats.null_count > 0)
      : std::nullopt;

  switch (type.kind()) {
    case TypeKind::BOOLEAN:
      return std::make_unique<dwio::common::BooleanColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt, std::nullopt);
    case TypeKind::TINYINT:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt,
          getMin<int8_t>(stats), getMax<int8_t>(stats), std::nullopt);
    case TypeKind::SMALLINT:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt,
          getMin<int16_t>(stats), getMax<int16_t>(stats), std::nullopt);
    case TypeKind::INTEGER:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt,
          getMin<int32_t>(stats), getMax<int32_t>(stats), std::nullopt);
    case TypeKind::BIGINT:
      return std::make_unique<dwio::common::IntegerColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt,
          getMin<int64_t>(stats), getMax<int64_t>(stats), std::nullopt);
    case TypeKind::REAL:
      return std::make_unique<dwio::common::DoubleColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt,
          getMin<float>(stats), getMax<float>(stats), std::nullopt);
    case TypeKind::DOUBLE:
      return std::make_unique<dwio::common::DoubleColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt,
          getMin<double>(stats), getMax<double>(stats), std::nullopt);
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return std::make_unique<dwio::common::StringColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt,
          getMin<std::string>(stats), getMax<std::string>(stats),
          std::nullopt);
    default:
      return std::make_unique<dwio::common::ColumnStatistics>(
          valueCount, hasNull, std::nullopt, std::nullopt);
  }
}

bool testIntFilter(
    const common::Filter* filter,
    dwio::common::IntegerColumnStatistics* intStats,
    bool mayHaveNull) {
  if (!intStats) {
    return true;
  }
  if (intStats->getMinimum().has_value() &&
      intStats->getMaximum().has_value()) {
    return filter->testInt64Range(
        intStats->getMinimum().value(),
        intStats->getMaximum().value(),
        mayHaveNull);
  }
  if (intStats->getMinimum().has_value()) {
    return filter->testInt64Range(
        intStats->getMinimum().value(),
        std::numeric_limits<int64_t>::max(),
        mayHaveNull);
  }
  if (intStats->getMaximum().has_value()) {
    return filter->testInt64Range(
        std::numeric_limits<int64_t>::min(),
        intStats->getMaximum().value(),
        mayHaveNull);
  }
  return true;
}

bool testDoubleFilter(
    const common::Filter* filter,
    dwio::common::DoubleColumnStatistics* doubleStats,
    bool mayHaveNull) {
  if (!doubleStats) {
    return true;
  }
  if (doubleStats->getMinimum().has_value() &&
      doubleStats->getMaximum().has_value()) {
    return filter->testDoubleRange(
        doubleStats->getMinimum().value(),
        doubleStats->getMaximum().value(),
        mayHaveNull);
  }
  if (doubleStats->getMinimum().has_value()) {
    return filter->testDoubleRange(
        doubleStats->getMinimum().value(),
        std::numeric_limits<double>::max(),
        mayHaveNull);
  }
  if (doubleStats->getMaximum().has_value()) {
    return filter->testDoubleRange(
        std::numeric_limits<double>::lowest(),
        doubleStats->getMaximum().value(),
        mayHaveNull);
  }
  return true;
}

bool testStringFilter(
    const common::Filter* filter,
    dwio::common::StringColumnStatistics* stringStats,
    bool mayHaveNull) {
  if (!stringStats) {
    return true;
  }
  if (stringStats->getMinimum().has_value() &&
      stringStats->getMaximum().has_value()) {
    const auto& min = stringStats->getMinimum().value();
    const auto& max = stringStats->getMaximum().value();
    return filter->testBytesRange(min, max, mayHaveNull);
  }
  if (stringStats->getMinimum().has_value()) {
    const auto& min = stringStats->getMinimum().value();
    return filter->testBytesRange(min, std::nullopt, mayHaveNull);
  }
  if (stringStats->getMaximum().has_value()) {
    const auto& max = stringStats->getMaximum().value();
    return filter->testBytesRange(std::nullopt, max, mayHaveNull);
  }
  return true;
}

bool testBoolFilter(
    const common::Filter* filter,
    dwio::common::BooleanColumnStatistics* boolStats) {
  const auto trueCount = boolStats->getTrueCount();
  const auto falseCount = boolStats->getFalseCount();
  if (trueCount.has_value() && falseCount.has_value()) {
    if (trueCount.value() == 0) {
      if (!filter->testBool(false)) {
        return false;
      }
    } else if (falseCount.value() == 0) {
      if (!filter->testBool(true)) {
        return false;
      }
    }
  }
  return true;
}

bool testFilter(
    const common::Filter* filter,
    dwio::common::ColumnStatistics* stats,
    uint64_t totalRows,
    const TypePtr& type) {
  bool mayHaveNull{true};

  if (stats->getNumberOfValues().has_value()) {
    if (stats->getNumberOfValues().value() == 0) {
      return filter->testNull();
    }
    mayHaveNull = stats->getNumberOfValues().value() < totalRows;
  }

  if (!mayHaveNull && filter->kind() == common::FilterKind::kIsNull) {
    return false;
  }

  if (mayHaveNull && filter->testNull()) {
    return true;
  }

  if (type->isDecimal()) {
    return true;
  }

  switch (type->kind()) {
    case TypeKind::BIGINT:
    case TypeKind::INTEGER:
    case TypeKind::SMALLINT:
    case TypeKind::TINYINT: {
      auto* intStats =
          dynamic_cast<dwio::common::IntegerColumnStatistics*>(stats);
      return testIntFilter(filter, intStats, mayHaveNull);
    }
    case TypeKind::REAL:
    case TypeKind::DOUBLE: {
      auto* doubleStats =
          dynamic_cast<dwio::common::DoubleColumnStatistics*>(stats);
      return testDoubleFilter(filter, doubleStats, mayHaveNull);
    }
    case TypeKind::BOOLEAN: {
      auto* boolStats =
          dynamic_cast<dwio::common::BooleanColumnStatistics*>(stats);
      return testBoolFilter(filter, boolStats);
    }
    case TypeKind::VARCHAR: {
      auto* stringStats =
          dynamic_cast<dwio::common::StringColumnStatistics*>(stats);
      return testStringFilter(filter, stringStats, mayHaveNull);
    }
    default:
      break;
  }
  return true;
}

} // namespace

// --- loadFileMetaData ---

std::unique_ptr<thrift::FileMetaData> ParquetMetadataUtil::loadFileMetaData(
    const char* buffer,
    uint32_t length) {
  auto thriftTransport =
      std::make_shared<thrift::ThriftBufferedTransport>(buffer, length);
  auto thriftProtocol = std::make_unique<
      apache::thrift::protocol::TCompactProtocolT<thrift::ThriftTransport>>(
      thriftTransport);
  auto fileMetaData = std::make_unique<thrift::FileMetaData>();
  fileMetaData->read(thriftProtocol.get());
  return fileMetaData;
}

// --- loadFileMetaDataFromFile ---
// Adapted from ReaderBase::loadFileMetaData() in ParquetReader.cpp.

std::unique_ptr<thrift::FileMetaData>
ParquetMetadataUtil::loadFileMetaDataFromFile(
    const std::string& filePath,
    uint64_t footerEstimatedSize) {
  const auto fileLength =
      static_cast<uint64_t>(std::filesystem::file_size(filePath));
  VELOX_CHECK_GT(fileLength, 0, "Parquet file is empty: {}", filePath);
  VELOX_CHECK_GE(fileLength, 12, "Parquet file is too small: {}", filePath);

  // Read the trailing bytes of the file (at least footerEstimatedSize).
  uint64_t readSize = std::min(fileLength, footerEstimatedSize);
  std::vector<char> buffer(readSize);

  {
    std::ifstream file(filePath, std::ios::binary);
    VELOX_CHECK(file.good(), "Failed to open file: {}", filePath);
    file.seekg(
        static_cast<std::streamoff>(fileLength - readSize), std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(readSize));
    VELOX_CHECK(file.good(), "Failed to read footer from: {}", filePath);
  }

  // Validate trailing magic bytes "PAR1".
  VELOX_CHECK(
      std::strncmp(buffer.data() + readSize - 4, "PAR1", 4) == 0,
      "No magic bytes found at end of Parquet file: {}",
      filePath);

  // Read footer length from the 4 bytes before the trailing magic.
  uint32_t footerLength;
  std::memcpy(&footerLength, buffer.data() + readSize - 8, sizeof(uint32_t));
  VELOX_CHECK_LE(
      footerLength + 12,
      fileLength,
      "Footer length {} exceeds file size {} in: {}",
      footerLength,
      fileLength,
      filePath);

  // If the initial read didn't capture the full footer, read the missing part.
  int64_t footerOffsetInBuffer =
      static_cast<int64_t>(readSize) - 8 - static_cast<int64_t>(footerLength);
  if (footerOffsetInBuffer < 0) {
    auto missingLength =
        static_cast<uint64_t>(-footerOffsetInBuffer);
    buffer.resize(footerLength);
    // Shift existing data to make room for the missing prefix.
    std::memmove(
        buffer.data() + missingLength, buffer.data(), readSize - 8);

    std::ifstream file(filePath, std::ios::binary);
    VELOX_CHECK(file.good(), "Failed to reopen file: {}", filePath);
    file.seekg(
        static_cast<std::streamoff>(fileLength - footerLength - 8),
        std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(missingLength));
    VELOX_CHECK(
        file.good(), "Failed to read extended footer from: {}", filePath);
    footerOffsetInBuffer = 0;
  }

  return loadFileMetaData(
      buffer.data() + footerOffsetInBuffer, footerLength);
}

// --- resolveColumnIndices ---

std::vector<uint32_t> ParquetMetadataUtil::resolveColumnIndices(
    const thrift::FileMetaData& metadata,
    const std::vector<std::string>& columnNames) {
  if (columnNames.empty() || metadata.row_groups.empty()) {
    return {};
  }

  // Build a map from top-level column name to its leaf column index using
  // the first row group's column chunks. Each ColumnChunk's
  // meta_data.path_in_schema[0] gives the top-level column name.
  const auto& firstRowGroup = metadata.row_groups[0];
  std::unordered_map<std::string, uint32_t> nameToIndex;
  for (int i = 0; i < static_cast<int>(firstRowGroup.columns.size()); ++i) {
    const auto& chunk = firstRowGroup.columns[i];
    if (chunk.__isset.meta_data &&
        !chunk.meta_data.path_in_schema.empty()) {
      const auto& topLevelName = chunk.meta_data.path_in_schema[0];
      // Only store the first occurrence (handles nested columns that share
      // the same top-level name — we want the leaf index).
      nameToIndex.emplace(topLevelName, static_cast<uint32_t>(i));
    }
  }

  std::vector<uint32_t> indices;
  indices.reserve(columnNames.size());
  for (const auto& name : columnNames) {
    auto it = nameToIndex.find(name);
    if (it != nameToIndex.end()) {
      indices.push_back(it->second);
    }
    // Silently skip columns not found in the file (schema evolution).
  }
  return indices;
}

// --- getColumnChunkByteRanges ---
// Adapted from ParquetData::enqueueRowGroup() in ParquetData.cpp.

std::vector<ParquetMetadataUtil::ByteRange>
ParquetMetadataUtil::getColumnChunkByteRanges(
    const thrift::FileMetaData& metadata,
    const std::vector<uint32_t>& rowGroupIndices,
    const std::vector<uint32_t>& columnIndices) {
  std::vector<ByteRange> ranges;
  ranges.reserve(rowGroupIndices.size() * columnIndices.size());

  for (auto rgIdx : rowGroupIndices) {
    VELOX_CHECK_LT(
        rgIdx,
        metadata.row_groups.size(),
        "Row group index {} out of range",
        rgIdx);
    const auto& rowGroup = metadata.row_groups[rgIdx];

    for (auto colIdx : columnIndices) {
      VELOX_CHECK_LT(
          colIdx,
          rowGroup.columns.size(),
          "Column index {} out of range in row group {}",
          colIdx,
          rgIdx);
      const auto& chunk = rowGroup.columns[colIdx];
      VELOX_CHECK(
          chunk.__isset.meta_data,
          "ColumnMetaData missing for column {} in row group {}",
          colIdx,
          rgIdx);

      const auto& meta = chunk.meta_data;

      // Start of chunk: dictionary page offset if present, else data page
      // offset. This mirrors ParquetData::enqueueRowGroup().
      uint64_t chunkOffset = static_cast<uint64_t>(meta.data_page_offset);
      if (meta.__isset.dictionary_page_offset &&
          meta.dictionary_page_offset >= 4) {
        chunkOffset = static_cast<uint64_t>(meta.dictionary_page_offset);
      }

      // Size: use compressed size if compression is active, otherwise
      // uncompressed. This is the total on-disk size of the column chunk.
      uint64_t chunkSize =
          (meta.codec == thrift::CompressionCodec::UNCOMPRESSED)
          ? static_cast<uint64_t>(meta.total_uncompressed_size)
          : static_cast<uint64_t>(meta.total_compressed_size);

      ranges.push_back({chunkOffset, chunkSize});
    }
  }

  return ranges;
}

// --- serializeFileMetaData ---

std::string ParquetMetadataUtil::serializeFileMetaData(
    const thrift::FileMetaData& metadata) {
  auto memBuffer =
      std::make_shared<apache::thrift::transport::TMemoryBuffer>();
  auto protocol = std::make_unique<
      apache::thrift::protocol::TCompactProtocolT<
          apache::thrift::transport::TMemoryBuffer>>(memBuffer);
  metadata.write(protocol.get());

  std::string result;
  memBuffer->appendBufferToString(result);
  return result;
}

// --- buildClippedFileMetaData ---

thrift::FileMetaData ParquetMetadataUtil::buildClippedFileMetaData(
    const thrift::FileMetaData& original,
    const std::vector<uint32_t>& rowGroupIndices,
    const std::vector<uint32_t>& columnIndices,
    const std::vector<ByteRange>& byteRanges,
    uint64_t dataOffset) {
  thrift::FileMetaData clipped;
  clipped.version = original.version;
  clipped.schema = original.schema;
  clipped.created_by = original.created_by;
  if (original.__isset.column_orders) {
    clipped.column_orders = original.column_orders;
    clipped.__isset.column_orders = true;
  }

  // Track the running offset in the output buffer where each chunk's data
  // will be placed. Starts at dataOffset (after leading "PAR1").
  uint64_t currentOffset = dataOffset;
  size_t rangeIdx = 0;
  int64_t totalRows = 0;

  for (auto rgIdx : rowGroupIndices) {
    const auto& srcRowGroup = original.row_groups[rgIdx];
    thrift::RowGroup clippedRG;
    clippedRG.num_rows = srcRowGroup.num_rows;
    totalRows += srcRowGroup.num_rows;

    int64_t rgTotalByteSize = 0;
    int64_t rgTotalCompressedSize = 0;

    for (auto colIdx : columnIndices) {
      VELOX_CHECK_LT(
          rangeIdx,
          byteRanges.size(),
          "byteRanges exhausted at rangeIdx {}",
          rangeIdx);
      const auto& range = byteRanges[rangeIdx];
      ++rangeIdx;

      thrift::ColumnChunk clippedChunk = srcRowGroup.columns[colIdx];
      auto& meta = clippedChunk.meta_data;

      // Compute offset delta: how much to shift the data_page_offset.
      // In the original file, the chunk starts at range.offset.
      // In the packed buffer, it starts at currentOffset.
      int64_t originalChunkStart =
          static_cast<int64_t>(range.offset);
      int64_t newChunkStart = static_cast<int64_t>(currentOffset);
      int64_t delta = newChunkStart - originalChunkStart;

      meta.data_page_offset += delta;
      if (meta.__isset.dictionary_page_offset) {
        meta.dictionary_page_offset += delta;
      }
      if (meta.__isset.index_page_offset) {
        meta.index_page_offset += delta;
      }

      // Clear column/offset index references — they are not in the packed
      // buffer and would point to invalid offsets.
      clippedChunk.__isset.offset_index_offset = false;
      clippedChunk.__isset.offset_index_length = false;
      clippedChunk.__isset.column_index_offset = false;
      clippedChunk.__isset.column_index_length = false;

      rgTotalByteSize += meta.total_uncompressed_size;
      rgTotalCompressedSize += meta.total_compressed_size;

      clippedRG.columns.push_back(std::move(clippedChunk));
      currentOffset += range.size;
    }

    clippedRG.total_byte_size = rgTotalByteSize;
    if (srcRowGroup.__isset.total_compressed_size) {
      clippedRG.total_compressed_size = rgTotalCompressedSize;
      clippedRG.__isset.total_compressed_size = true;
    }
    // file_offset points to first page in this row group.
    if (!clippedRG.columns.empty()) {
      const auto& firstMeta = clippedRG.columns[0].meta_data;
      clippedRG.file_offset =
          firstMeta.__isset.dictionary_page_offset
          ? firstMeta.dictionary_page_offset
          : firstMeta.data_page_offset;
      clippedRG.__isset.file_offset = true;
    }

    clipped.row_groups.push_back(std::move(clippedRG));
  }

  clipped.num_rows = totalRows;

  // Copy key_value_metadata (e.g., ARROW:schema, pandas metadata).
  if (original.__isset.key_value_metadata) {
    clipped.key_value_metadata = original.key_value_metadata;
    clipped.__isset.key_value_metadata = true;
  }

  return clipped;
}

// --- filterRowGroupsByStatistics ---

std::vector<uint32_t> ParquetMetadataUtil::filterRowGroupsByStatistics(
    const thrift::FileMetaData& metadata,
    const std::vector<uint32_t>& candidateRowGroups,
    const std::vector<std::pair<std::string, const common::Filter*>>&
        columnFilters,
    const RowTypePtr& dataColumnsType) {
  if (columnFilters.empty() || candidateRowGroups.empty() ||
      metadata.row_groups.empty()) {
    return candidateRowGroups;
  }

  // Build name -> column index map from the first row group's column chunks.
  const auto& firstRowGroup = metadata.row_groups[0];
  std::unordered_map<std::string, uint32_t> nameToColIdx;
  for (uint32_t i = 0; i < static_cast<uint32_t>(firstRowGroup.columns.size());
       ++i) {
    const auto& chunk = firstRowGroup.columns[i];
    if (chunk.__isset.meta_data && !chunk.meta_data.path_in_schema.empty()) {
      nameToColIdx.emplace(chunk.meta_data.path_in_schema[0], i);
    }
  }

  // Pre-resolve each filter to its Parquet column index and Velox type.
  struct ResolvedFilter {
    uint32_t colIdx;
    const common::Filter* filter;
    TypePtr type;
  };
  std::vector<ResolvedFilter> resolved;
  resolved.reserve(columnFilters.size());
  for (const auto& [colName, filter] : columnFilters) {
    auto idxIt = nameToColIdx.find(colName);
    if (idxIt == nameToColIdx.end()) {
      continue;
    }
    TypePtr colType;
    if (dataColumnsType->containsChild(colName)) {
      colType = dataColumnsType->findChild(colName);
    } else {
      continue;
    }
    resolved.push_back({idxIt->second, filter, std::move(colType)});
  }

  if (resolved.empty()) {
    return candidateRowGroups;
  }

  std::vector<uint32_t> surviving;
  surviving.reserve(candidateRowGroups.size());

  for (auto rgIdx : candidateRowGroups) {
    const auto& rowGroup = metadata.row_groups[rgIdx];
    bool matches = true;

    for (const auto& rf : resolved) {
      if (rf.colIdx >= static_cast<uint32_t>(rowGroup.columns.size())) {
        continue;
      }
      const auto& chunk = rowGroup.columns[rf.colIdx];
      if (!chunk.__isset.meta_data || !chunk.meta_data.__isset.statistics) {
        continue;
      }

      auto colStats = buildColumnStatisticsFromThrift(
          chunk.meta_data.statistics,
          *rf.type,
          static_cast<uint64_t>(rowGroup.num_rows));
      if (!testFilter(
              rf.filter,
              colStats.get(),
              static_cast<uint64_t>(rowGroup.num_rows),
              rf.type)) {
        matches = false;
        break;
      }
    }

    if (matches) {
      surviving.push_back(rgIdx);
    }
  }

  return surviving;
}

} // namespace facebook::velox::parquet
