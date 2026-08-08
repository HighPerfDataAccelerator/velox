/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#include "velox/experimental/cudf/connectors/hive/SelectiveParquetReader.h"

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/parquet/thrift/ParquetThrift.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stack>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

namespace pqthrift = facebook::velox::parquet::thrift;

#define VELOX_NVTX_SCOPED(name) \
  do {                          \
  } while (false)

void recordPreloadBufferStats(
    IoStats* ioStats,
    const PinnedHostBuffer& buffer) {
  if (!ioStats) {
    return;
  }
  const auto prefix =
      buffer.isPinned() ? "cudfPinnedPreload" : "cudfPageablePreload";
  ioStats->addCounter(
      std::string(prefix) + "Allocations", RuntimeCounter(1));
  ioStats->addCounter(
      std::string(prefix) + "Bytes",
      RuntimeCounter(buffer.size(), RuntimeCounter::Unit::kBytes));
}

void brokerRead(
    const std::shared_ptr<ExecutorReadBroker>& broker,
    const PrefetchReadFunction& readFunction,
    uint64_t sourceSize,
    std::vector<PrefetchRange> ranges,
    const std::shared_ptr<PinnedHostBuffer>& destination,
    IoStats* ioStats) {
  uint64_t bytes = 0;
  for (const auto& range : ranges) {
    bytes += range.size;
  }
  const auto numRanges = ranges.size();
  auto requestStats = std::make_shared<ExecutorReadRequestStats>();
  broker->read(
            readFunction,
            sourceSize,
            std::move(ranges),
            destination,
            requestStats)
      .get();
  if (!ioStats) {
    return;
  }
  ioStats->addCounter("cudfPrefetchRequests", RuntimeCounter(1));
  ioStats->addCounter(
      "cudfPrefetchRanges", RuntimeCounter(numRanges));
  ioStats->addCounter(
      "cudfPrefetchBytes",
      RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
  ioStats->addCounter(
      "cudfPrefetchWaitNanos",
      RuntimeCounter(
          requestStats->waitNanos.load(std::memory_order_relaxed),
          RuntimeCounter::Unit::kNanos));
  ioStats->addCounter(
      "cudfPrefetchReadNanos",
      RuntimeCounter(
          requestStats->readNanos.load(std::memory_order_relaxed),
          RuntimeCounter::Unit::kNanos));
}

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
  int nc = schema[0].num_children().value_or(0);
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
    int childNc = schema[i].num_children().value_or(0);
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
  int64_t start = *meta.data_page_offset();
  if (meta.dictionary_page_offset().has_value() &&
      *meta.dictionary_page_offset() > 0 &&
      *meta.dictionary_page_offset() < start) {
    start = *meta.dictionary_page_offset();
  }
  return {start, *meta.total_compressed_size()};
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
    uint64_t fileSize,
    PrefetchReadFunction readFunction,
    const std::vector<std::string>& readColumnNames,
    uint64_t splitStart,
    uint64_t splitLength,
    std::shared_ptr<ExecutorReadBroker> broker,
    IoStats* ioStats) {
  VELOX_CHECK(
      static_cast<bool>(readFunction),
      "Selective Parquet read requires a range-read function");

  // Clamp the split range to the file: splitLength may arrive as sentinel
  // max or oversize when the caller only has the file path.
  const uint64_t splitEndExclusive =
      (splitStart >= fileSize) ? fileSize
      : (splitLength > fileSize - splitStart) ? fileSize
      : splitStart + splitLength;

  auto fullRead = [&]() {
    VELOX_NVTX_SCOPED("SelectiveRead::FullRead");
    std::shared_ptr<PinnedHostBuffer> buf;
    {
      VELOX_NVTX_SCOPED("SelectiveRead::AllocPinnedBuf");
      buf = std::make_shared<PinnedHostBuffer>(fileSize);
      recordPreloadBufferStats(ioStats, *buf);
    }
    if (broker) {
      constexpr uint64_t kFullReadRangeBytes = 16ULL << 20;
      std::vector<PrefetchRange> ranges;
      for (uint64_t offset = 0; offset < fileSize;
           offset += kFullReadRangeBytes) {
        ranges.push_back(
            {offset,
             std::min<uint64_t>(kFullReadRangeBytes, fileSize - offset),
             offset});
      }
      brokerRead(
          broker,
          readFunction,
          fileSize,
          std::move(ranges),
          buf,
          ioStats);
    } else {
      readFunction(0, fileSize, buf->data());
    }
    return buf;
  };

  // Fall back to full read only when there is no column projection. Byte-
  // range splits (splitStart > 0) are handled via row group filtering below;
  // the old fallback also triggered on splitStart > 0 and produced incorrect
  // results for byte-range splits, so we drop it here.
  if (readColumnNames.empty()) {
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
    VELOX_NVTX_SCOPED("SelectiveRead::ReadFooter");
    // Read the ender (footer_len + magic) from the end of the file.
    uint8_t enderBuf[kEnderLen];
    readFunction(fileSize - kEnderLen, kEnderLen, enderBuf);
    uint32_t footerLen = 0, magic = 0;
    std::memcpy(&footerLen, enderBuf, sizeof(footerLen));
    std::memcpy(&magic, enderBuf + sizeof(footerLen), sizeof(magic));
    VELOX_CHECK(
        magic == kParquetMagic, "Invalid Parquet magic in: {}", filePath);
    VELOX_CHECK(
        footerLen > 0 && footerLen <= fileSize - kHeaderLen - kEnderLen,
        "Invalid footer length in: {}",
        filePath);

    std::vector<uint8_t> footerBytes(footerLen);
    readFunction(
        fileSize - kEnderLen - footerLen, footerLen, footerBytes.data());

    pqthrift::deserialize(
        &metadata,
        std::string_view(
            reinterpret_cast<const char*>(footerBytes.data()), footerLen));
  }

  // Step 2: Build children indices for schema tree navigation.
  static NvtxRegisteredStringT const clipName{"SelectiveRead::ClipMetadata"};
  std::optional<::nvtx3::scoped_range_in<VeloxDomain>> clipRange;
  clipRange.emplace(::nvtx3::event_attributes{clipName});
  const auto childrenIdx = buildChildrenIdx(*metadata.schema());

  // Step 3: Build the set of needed column names
  std::unordered_set<std::string> neededColumns;
  for (const auto& name : readColumnNames) {
    const auto nestedSeparator = name.find('.');
    neededColumns.insert(name.substr(0, nestedSeparator));
  }

  // Step 4: For each row group, identify needed column chunks and their
  // byte ranges. Also build the clipped metadata.
  struct ChunkReadInfo {
    int64_t srcOffset;
    int64_t size;
  };
  std::vector<ChunkReadInfo> chunksToRead;
  int64_t totalChunkBytes = 0;

  pqthrift::FileMetaData clippedMetadata;
  clippedMetadata.version() = *metadata.version();
  clippedMetadata.num_rows() = *metadata.num_rows();
  if (metadata.created_by().has_value()) {
    clippedMetadata.created_by() = *metadata.created_by();
  }

  // Copy key_value_metadata but strip ARROW:schema since it describes all
  // original columns and would mismatch the clipped schema.
  if (metadata.key_value_metadata().has_value()) {
    for (const auto& kv : *metadata.key_value_metadata()) {
      if (*kv.key() != "ARROW:schema") {
        clippedMetadata.key_value_metadata().ensure().push_back(kv);
      }
    }
  }

  // column_orders has one entry per leaf column in schema order.  After
  // clipping, the indices no longer match, so omit it to let cuDF use
  // the default (safe) sort order for statistics interpretation.

  // Clip the schema: keep root + needed top-level columns and their children.
  // The Parquet schema is a flattened tree. Element 0 is root, then children
  // follow in depth-first order.
  {
    const auto& origSchema = *metadata.schema();
    VELOX_CHECK(!origSchema.empty(), "Empty Parquet schema");

    pqthrift::SchemaElement clippedRoot = origSchema[0];
    clippedRoot.num_children() = 0;
    std::vector<pqthrift::SchemaElement> clippedSchema;
    clippedSchema.push_back(clippedRoot);

    const auto& rootChildren = childrenIdx[0];
    for (int childIdx : rootChildren) {
      const auto& childElem = origSchema[childIdx];
      if (neededColumns.count(*childElem.name()) == 0) {
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

      ++*clippedSchema[0].num_children();
      for (int si : subtreeIndices) {
        clippedSchema.push_back(origSchema[si]);
      }
    }

    clippedMetadata.schema() = std::move(clippedSchema);
  }

  // Determine a row group's starting byte offset in the source file.
  // Prefers the optional RowGroup.file_offset when set by the writer; falls
  // back to the minimum offset across its column chunks (dictionary page
  // offset or data page offset, whichever comes first on disk).
  auto rowGroupStartOffset = [](const pqthrift::RowGroup& rg) -> int64_t {
    if (rg.file_offset().has_value() && *rg.file_offset() > 0) {
      return *rg.file_offset();
    }
    int64_t minOffset = std::numeric_limits<int64_t>::max();
    for (const auto& cc : *rg.columns()) {
      const auto& meta = *cc.meta_data();
      int64_t s = *meta.data_page_offset();
      if (meta.dictionary_page_offset().has_value() &&
          *meta.dictionary_page_offset() > 0 &&
          *meta.dictionary_page_offset() < s) {
        s = *meta.dictionary_page_offset();
      }
      if (s < minOffset) {
        minOffset = s;
      }
    }
    return minOffset;
  };

  int64_t totalKeptRows = 0;
  for (auto& rg : *metadata.row_groups()) {
    // Row group ownership: keep a row group iff its starting offset falls
    // in [splitStart, splitEndExclusive). This matches how Spark and cuDF
    // on-demand reader assign row groups to byte-range splits, so each
    // row group is owned by exactly one split across the whole job.
    const int64_t rgStart = rowGroupStartOffset(rg);
    if (rgStart < static_cast<int64_t>(splitStart) ||
        rgStart >= static_cast<int64_t>(splitEndExclusive)) {
      continue;
    }

    pqthrift::RowGroup clippedRg;
    clippedRg.total_byte_size() = 0;
    clippedRg.num_rows() = *rg.num_rows();
    clippedRg.total_compressed_size() = 0;
    if (rg.ordinal().has_value()) {
      clippedRg.ordinal() = *rg.ordinal();
    }

    for (auto& cc : *rg.columns()) {
      const auto& meta = *cc.meta_data();
      const auto colName = getTopLevelColumnName(*meta.path_in_schema());
      if (neededColumns.count(colName) == 0) {
        continue;
      }

      auto range = getColumnChunkByteRange(meta);
      chunksToRead.push_back({range.offset, range.size});
      totalChunkBytes += range.size;

      clippedRg.columns()->push_back(cc);
      *clippedRg.total_byte_size() += *meta.total_uncompressed_size();
      *clippedRg.total_compressed_size() += *meta.total_compressed_size();
    }

    totalKeptRows += *rg.num_rows();
    clippedMetadata.row_groups()->push_back(std::move(clippedRg));
  }

  // When row groups were filtered by byte range, num_rows in the clipped
  // footer must reflect only the kept row groups, not the original total.
  clippedMetadata.num_rows() = totalKeptRows;

  // Check if selective read saves significant IO (>20% savings). Only
  // valid when this split covers the whole file — for byte-range splits
  // (splitStart > 0) fullRead would bypass row group filtering and
  // return all rows, breaking correctness. So skip this optimization
  // whenever the split is not the whole file.
  const bool coversWholeFile =
      (splitStart == 0) && (splitEndExclusive == fileSize);
  const int64_t fullReadCost = static_cast<int64_t>(fileSize);
  if (coversWholeFile && totalChunkBytes > fullReadCost * 80 / 100) {
    return fullRead();
  }

  // Step 5: Build compact Parquet buffer.
  // Layout: [PAR1 header][column chunks contiguously][clipped footer][ender]
  // Update column chunk offsets in clippedMetadata to point to new positions.
  int64_t writeOffset = static_cast<int64_t>(kHeaderLen);
  size_t chunkIdx = 0;
  for (auto& rg : *clippedMetadata.row_groups()) {
    rg.file_offset() = writeOffset;
    for (auto& cc : *rg.columns()) {
      auto& meta = *cc.meta_data();
      const int64_t dictOffset = meta.dictionary_page_offset().has_value()
          ? *meta.dictionary_page_offset()
          : 0;
      const int64_t dataOffset = *meta.data_page_offset();

      if (dictOffset > 0 && dictOffset < dataOffset) {
        meta.dictionary_page_offset() = writeOffset;
        meta.data_page_offset() = writeOffset + (dataOffset - dictOffset);
      } else {
        meta.data_page_offset() = writeOffset;
        meta.dictionary_page_offset().reset();
      }
      meta.index_page_offset().reset();

      cc.file_offset() =
          *meta.data_page_offset() + *meta.total_compressed_size();
      cc.offset_index_offset().reset();
      cc.offset_index_length().reset();
      cc.column_index_offset().reset();
      cc.column_index_length().reset();

      writeOffset += chunksToRead[chunkIdx].size;
      ++chunkIdx;
    }
  }

  // End ClipMetadata range before moving on to serialization.
  clipRange.reset();

  // Step 6: Serialize the clipped footer using Velox Thrift Compact Protocol.
  folly::IOBufQueue serializedFooterQueue;
  {
    VELOX_NVTX_SCOPED("SelectiveRead::SerializeFooter");
    pqthrift::serialize(clippedMetadata, &serializedFooterQueue);
  }
  auto serializedFooter = serializedFooterQueue.move();
  serializedFooter->coalesce();
  const auto* newFooterData = serializedFooter->data();
  const auto newFooterSize =
      static_cast<uint32_t>(serializedFooter->length());

  const size_t totalBufSize =
      kHeaderLen + totalChunkBytes + newFooterSize + kEnderLen;
  std::shared_ptr<PinnedHostBuffer> buf;
  {
    VELOX_NVTX_SCOPED("SelectiveRead::AllocPinnedBuf");
    buf = std::make_shared<PinnedHostBuffer>(totalBufSize);
    recordPreloadBufferStats(ioStats, *buf);
  }
  uint8_t* dst = buf->data();

  // Write PAR1 header
  const uint32_t magic = kParquetMagic;
  std::memcpy(dst, &magic, kHeaderLen);
  size_t dstOffset = kHeaderLen;

  // Step 7: Submit every selected chunk from this file as one broker request.
  // The broker executes ranges on the executor-wide CPU pool and applies a
  // shared in-flight byte budget across Spark tasks and Velox drivers.
  std::vector<PrefetchRange> prefetchRanges;
  prefetchRanges.reserve(chunksToRead.size());
  for (const auto& chunk : chunksToRead) {
    prefetchRanges.push_back(
        {static_cast<uint64_t>(chunk.srcOffset),
         static_cast<uint64_t>(chunk.size),
         static_cast<uint64_t>(dstOffset)});
    dstOffset += chunk.size;
  }
  if (broker && !prefetchRanges.empty()) {
    VELOX_NVTX_SCOPED("SelectiveRead::BrokerReadChunks");
    brokerRead(
        broker,
        readFunction,
        fileSize,
        std::move(prefetchRanges),
        buf,
        ioStats);
  } else {
    VELOX_NVTX_SCOPED("SelectiveRead::SequentialReadChunks");
    for (size_t i = 0; i < chunksToRead.size(); ++i) {
      const auto& chunk = chunksToRead[i];
      readFunction(
          chunk.srcOffset,
          chunk.size,
          buf->data() + kHeaderLen +
              std::accumulate(
                  chunksToRead.begin(),
                  chunksToRead.begin() + i,
                  size_t{0},
                  [](size_t bytes, const auto& value) {
                    return bytes + value.size;
                  }));
    }
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
