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

#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSource2.h"
#include "velox/dwio/parquet/reader/ParquetMetadataUtil.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SubfieldFiltersToAst.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/time/Timer.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/HiveDataSource.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/expression/FieldReference.h"

#include <cudf/aggregation.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;
using namespace facebook::velox::connector::hive;

namespace {

/// Return the byte offset of the first page in a row group.
/// Prefers the row group's file_offset when available; otherwise falls
/// back to the minimum data/dictionary page offset across all columns.
int64_t getRowGroupStartOffset(
    const facebook::velox::parquet::thrift::RowGroup& rowGroup) {
  if (rowGroup.__isset.file_offset) {
    return rowGroup.file_offset;
  }
  int64_t minOffset = std::numeric_limits<int64_t>::max();
  for (const auto& col : rowGroup.columns) {
    if (col.__isset.meta_data) {
      int64_t offset = col.meta_data.data_page_offset;
      if (col.meta_data.__isset.dictionary_page_offset) {
        offset = std::min(offset, col.meta_data.dictionary_page_offset);
      }
      minOffset = std::min(minOffset, offset);
    }
  }
  return minOffset;
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor — mirrors CudfHiveDataSource but much simpler (no coalescing).
// ---------------------------------------------------------------------------

CudfHiveDataSource2::CudfHiveDataSource2(
    const RowTypePtr& outputType,
    const ConnectorTableHandlePtr& tableHandle,
    const ColumnHandleMap& columnHandles,
    facebook::velox::FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig)
    : NvtxHelper(
          nvtx3::rgb{80, 200, 120}, // green for DS2
          std::nullopt,
          fmt::format("[DS2:{}]", tableHandle->name())),
      cudfHiveConfig_(cudfHiveConfig),
      fileHandleFactory_(fileHandleFactory),
      executor_(executor),
      connectorQueryCtx_(connectorQueryCtx),
      pool_(connectorQueryCtx->memoryPool()),
      outputType_(outputType),
      expressionEvaluator_(connectorQueryCtx->expressionEvaluator()) {
  // Build column projection.
  for (const auto& outputName : outputType_->names()) {
    auto it = columnHandles.find(outputName);
    VELOX_CHECK(
        it != columnHandles.end(),
        "ColumnHandle is missing for output column: {}",
        outputName);
    auto* handle =
        static_cast<const hive::HiveColumnHandle*>(it->second.get());
    readColumnSet_.emplace(handle->name());
    readColumnNames_.emplace_back(handle->name());
  }

  tableHandle_ =
      std::dynamic_pointer_cast<const hive::HiveTableHandle>(tableHandle);
  VELOX_CHECK_NOT_NULL(
      tableHandle_, "TableHandle must be an instance of HiveTableHandle");

  // Copy subfield filters and ensure filter columns are in read set.
  for (const auto& [k, v] : tableHandle_->subfieldFilters()) {
    subfieldFilters_.emplace(k.clone(), v->clone());
    for (const auto& [field, _] : subfieldFilters_) {
      if (readColumnSet_.count(field.toString()) == 0) {
        readColumnSet_.emplace(field.toString());
        readColumnNames_.emplace_back(field.toString());
      }
    }
  }

  // Create remaining filter expression.
  auto remainingFilter = tableHandle_->remainingFilter();
  if (remainingFilter) {
    remainingFilterExprSet_ = expressionEvaluator_->compile(remainingFilter);
    for (const auto& field : remainingFilterExprSet_->distinctFields()) {
      if (readColumnSet_.count(field->name()) == 0) {
        readColumnSet_.emplace(field->name());
        readColumnNames_.emplace_back(field->name());
      }
    }

    const RowTypePtr remainingFilterType = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> names;
        std::vector<TypePtr> types;
        for (const auto& name : readColumnNames_) {
          names.emplace_back(name);
          types.push_back(tableHandle_->dataColumns()->findChild(name));
        }
        return ROW(std::move(names), std::move(types));
      }
      return outputType_;
    }();

    cudfExpressionEvaluator_ = velox::cudf_velox::createCudfExpression(
        remainingFilterExprSet_->exprs()[0], remainingFilterType);
  }

  // Build combined AST for subfield filters (query-constant).
  if (!subfieldFilters_.empty()) {
    const RowTypePtr readerFilterType = [&] {
      if (tableHandle_->dataColumns()) {
        std::vector<std::string> names;
        std::vector<TypePtr> types;
        for (const auto& name : readColumnNames_) {
          names.emplace_back(name);
          types.push_back(tableHandle_->dataColumns()->findChild(name));
        }
        return ROW(std::move(names), std::move(types));
      }
      return outputType_;
    }();

    subfieldFilterExpr_ = &createAstFromSubfieldFilters(
        subfieldFilters_, subfieldTree_, subfieldScalars_, readerFilterType);
  }

  VELOX_CHECK_NOT_NULL(fileHandleFactory_, "No FileHandleFactory present");

  ioStatistics_ = std::make_shared<io::IoStatistics>();
}

// ---------------------------------------------------------------------------
// castToCudfSplit — same logic as CudfHiveDataSource::addSplit's lambda.
// ---------------------------------------------------------------------------

std::shared_ptr<CudfHiveConnectorSplit> CudfHiveDataSource2::castToCudfSplit(
    const std::shared_ptr<ConnectorSplit>& split) {
  if (auto cudfSplit =
          std::dynamic_pointer_cast<CudfHiveConnectorSplit>(split)) {
    return cudfSplit;
  }
  if (auto hiveSplit =
          std::dynamic_pointer_cast<hive::HiveConnectorSplit>(split)) {
    std::string cleanedPath = hiveSplit->filePath;
    constexpr std::string_view kFilePrefix = "file:";
    constexpr std::string_view kS3APrefix = "s3a:";
    if (cleanedPath.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
      cleanedPath = cleanedPath.substr(kFilePrefix.size());
    } else if (cleanedPath.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
      cleanedPath.erase(kS3APrefix.size() - 2, 1);
    }
    return CudfHiveConnectorSplitBuilder(cleanedPath)
        .start(hiveSplit->start)
        .length(hiveSplit->length)
        .connectorId(hiveSplit->connectorId)
        .splitWeight(hiveSplit->splitWeight)
        .build();
  }
  VELOX_FAIL("Unsupported split type: {}", split->toString());
}

// ---------------------------------------------------------------------------
// addSplit — the core preload work. Runs on IO executor thread.
//
// 1. Read footer
// 2. Resolve column indices
// 3. (Future: row group filtering)
// 4. Compute byte ranges for surviving column chunks
// 5. Build clipped footer with adjusted offsets
// 6. Allocate PinnedHostBuffer for the full self-contained Parquet buffer
// 7. Read byte ranges directly into the pinned buffer + append footer
// ---------------------------------------------------------------------------

void CudfHiveDataSource2::addSplit(std::shared_ptr<ConnectorSplit> split) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  split_ = castToCudfSplit(split);
  splitReader_.reset();
  splitExhausted_ = true; // Assume exhausted until proven otherwise.

  VLOG(1) << "CudfHiveDataSource2::addSplit " << split_->toString();

  // Step 1: Read footer from file.
  auto footerMeta = parquet::ParquetMetadataUtil::loadFileMetaDataFromFile(
      split_->filePath);

  if (footerMeta->row_groups.empty()) {
    return; // No row groups — nothing to read.
  }

  // Step 2: Resolve requested column names to leaf column indices.
  auto columnIndices = parquet::ParquetMetadataUtil::resolveColumnIndices(
      *footerMeta, readColumnNames_);

  if (columnIndices.empty()) {
    return; // No matching columns found.
  }

  // Step 3: Filter row groups by the split's byte range.
  // A row group is owned by this split when its first-page offset falls
  // within [split_start, split_start + split_length).
  std::vector<uint32_t> survivingRowGroups;
  survivingRowGroups.reserve(footerMeta->row_groups.size());
  for (uint32_t i = 0;
       i < static_cast<uint32_t>(footerMeta->row_groups.size());
       ++i) {
    auto rgOffset =
        static_cast<uint64_t>(getRowGroupStartOffset(footerMeta->row_groups[i]));
    if (rgOffset >= split_->start &&
        (rgOffset - split_->start) < split_->length) {
      survivingRowGroups.push_back(i);
    }
  }

  if (survivingRowGroups.empty()) {
    return; // No row groups in this split's byte range.
  }

  // Step 3b: Filter row groups by column chunk statistics (predicate pushdown).
  const auto rowGroupsBeforeStats =
      static_cast<uint32_t>(survivingRowGroups.size());
  if (!subfieldFilters_.empty()) {
    std::vector<std::pair<std::string, const common::Filter*>> columnFilters;
    columnFilters.reserve(subfieldFilters_.size());
    for (const auto& [subfield, filter] : subfieldFilters_) {
      columnFilters.emplace_back(subfield.toString(), filter.get());
    }
    const auto dataColumnsType = tableHandle_->dataColumns()
        ? tableHandle_->dataColumns()
        : outputType_;
    survivingRowGroups =
        parquet::ParquetMetadataUtil::filterRowGroupsByStatistics(
            *footerMeta, survivingRowGroups, columnFilters, dataColumnsType);
    if (survivingRowGroups.empty()) {
      return;
    }
  }

  // Step 4: Compute byte ranges for surviving column chunks.
  auto byteRanges = parquet::ParquetMetadataUtil::getColumnChunkByteRanges(
      *footerMeta, survivingRowGroups, columnIndices);

  // Step 5: Build clipped footer with adjusted offsets.
  // Done before data read — only needs byte-range metadata, not actual data.
  constexpr uint64_t kMagicSize = 4;
  auto clippedMeta = parquet::ParquetMetadataUtil::buildClippedFileMetaData(
      *footerMeta,
      survivingRowGroups,
      columnIndices,
      byteRanges,
      /*dataOffset=*/kMagicSize);
  auto serializedFooter =
      parquet::ParquetMetadataUtil::serializeFileMetaData(clippedMeta);

  // Step 6: Allocate PinnedHostBuffer for the self-contained Parquet buffer.
  //   Layout: PAR1 | packed chunk data | clipped footer | footer_len | PAR1
  size_t totalDataSize = 0;
  for (const auto& range : byteRanges) {
    totalDataSize += range.size;
  }

  const auto footerLen = static_cast<uint32_t>(serializedFooter.size());
  const size_t bufferSize =
      kMagicSize + totalDataSize + footerLen + 4 + kMagicSize;

  const auto originalFileSize =
      std::filesystem::file_size(split_->filePath);
  LOG(WARNING) << "CudfHiveDataSource2::addSplit " << split_->filePath
               << " ratio=" << (static_cast<double>(bufferSize) / split_->length)
               << " clippedBuffer=" << succinctBytes(bufferSize)
               << " splitRange=[" << split_->start << ", "
               << split_->start + split_->length << ")"
               << " rowGroups=" << survivingRowGroups.size()
               << "/" << footerMeta->row_groups.size()
               << " (statsFiltered=" << (rowGroupsBeforeStats - survivingRowGroups.size()) << ")"
               << " columns=" << columnIndices.size()
               << "/" << footerMeta->row_groups[0].columns.size();

  pinnedBuffer_ = std::make_shared<cudf_velox::PinnedHostBuffer>(bufferSize);
  uint8_t* ptr = pinnedBuffer_->data();

  // Leading magic.
  std::memcpy(ptr, "PAR1", 4);
  ptr += 4;

  // Step 7: Read byte ranges directly into the pinned buffer.
  {
    std::ifstream file(split_->filePath, std::ios::binary);
    VELOX_CHECK(
        file.good(), "Failed to open file: {}", split_->filePath);
    for (const auto& range : byteRanges) {
      file.seekg(static_cast<std::streamoff>(range.offset), std::ios::beg);
      file.read(
          reinterpret_cast<char*>(ptr),
          static_cast<std::streamsize>(range.size));
      VELOX_CHECK(
          file.good(),
          "Failed to read byte range [{}, {}) from: {}",
          range.offset,
          range.offset + range.size,
          split_->filePath);
      ptr += range.size;
    }
  }

  // Clipped Thrift footer.
  std::memcpy(ptr, serializedFooter.data(), footerLen);
  ptr += footerLen;
  // Footer length (little-endian uint32).
  std::memcpy(ptr, &footerLen, 4);
  ptr += 4;
  // Trailing magic.
  std::memcpy(ptr, "PAR1", 4);

  completedBytes_ += bufferSize;
  splitExhausted_ = false;
}

// ---------------------------------------------------------------------------
// setFromDataSource — transfer preloaded state from IO executor DataSource.
// ---------------------------------------------------------------------------

void CudfHiveDataSource2::setFromDataSource(
    std::unique_ptr<DataSource> sourceUnique) {
  auto* source = dynamic_cast<CudfHiveDataSource2*>(sourceUnique.get());
  VELOX_CHECK_NOT_NULL(
      source,
      "setFromDataSource expects CudfHiveDataSource2");

  split_ = std::move(source->split_);
  completedBytes_ += source->completedBytes_;

  if (source->pinnedBuffer_) {
    pendingBytes_ += static_cast<int64_t>(source->pinnedBuffer_->size());
    pendingBuffers_.push_back(std::move(source->pinnedBuffer_));
    ++numFilesCoalesced_;
  }

  splitReader_.reset();
  splitExhausted_ = source->splitExhausted_;

  // Merge IO statistics.
  if (source->ioStatistics_) {
    ioStatistics_->merge(*source->ioStatistics_);
  }
}

// ---------------------------------------------------------------------------
// wantsMoreSplits — signal to driver whether to fetch more splits.
// ---------------------------------------------------------------------------

bool CudfHiveDataSource2::wantsMoreSplits() const {
  const auto threshold = cudfHiveConfig_->coalesceAccBufferSizeInBytes();
  if (threshold <= 0) {
    return false;
  }
  return !pendingBuffers_.empty() && pendingBytes_ < threshold;
}

// ---------------------------------------------------------------------------
// createMultiSourceReader — drain pendingBuffers_ into a multi-source reader.
// ---------------------------------------------------------------------------

void CudfHiveDataSource2::createMultiSourceReader() {
  VELOX_CHECK(!pendingBuffers_.empty(), "No pending buffers to create reader");

  activeBuffers_ = std::move(pendingBuffers_);
  pendingBuffers_.clear();
  pendingBytes_ = 0;

  stream_ = cudfGlobalStreamPool().get_stream();

  std::vector<cudf::host_span<const std::byte>> spans;
  spans.reserve(activeBuffers_.size());
  for (const auto& buf : activeBuffers_) {
    spans.emplace_back(
        reinterpret_cast<const std::byte*>(buf->data()), buf->size());
  }

  auto sourceInfo = cudf::io::source_info(
      cudf::host_span<cudf::host_span<const std::byte>>(
          spans.data(), spans.size()));

  uint64_t totalBufferBytes = 0;
  for (const auto& buf : activeBuffers_) {
    totalBufferBytes += buf->size();
  }
  LOG(WARNING) << "CudfHiveDataSource2::createMultiSourceReader"
               << " numSplits=" << activeBuffers_.size()
               << " totalBufferSize=" << succinctBytes(totalBufferBytes);

  auto opts =
      cudf::io::parquet_reader_options::builder(std::move(sourceInfo))
          .use_pandas_metadata(cudfHiveConfig_->isUsePandasMetadata())
          .use_arrow_schema(cudfHiveConfig_->isUseArrowSchema())
          .allow_mismatched_pq_schemas(
              cudfHiveConfig_->isAllowMismatchedCudfHiveSchemas())
          .timestamp_type(cudfHiveConfig_->timestampType())
          .build();

  if (!readColumnNames_.empty()) {
    opts.set_column_names(readColumnNames_);
  }
  if (subfieldFilterExpr_ != nullptr) {
    opts.set_filter(*subfieldFilterExpr_);
  }

  splitReader_ = std::make_unique<cudf::io::chunked_parquet_reader>(
      cudfHiveConfig_->maxChunkReadLimit(),
      cudfHiveConfig_->maxPassReadLimit(),
      opts,
      stream_,
      cudf::get_current_device_resource_ref());
}

// ---------------------------------------------------------------------------
// next — driver thread. Creates cuDF reader from pending buffers, performs
// a one-shot read of all coalesced files, applies filter, returns result.
//
// With chunk_read_limit=0 and pass_read_limit=0 (the current defaults),
// cuDF's chunked_parquet_reader returns all rows from all source files in
// a single read_chunk() call. We assert this invariant rather than
// accumulating across multiple chunks.
// ---------------------------------------------------------------------------

std::optional<RowVectorPtr> CudfHiveDataSource2::next(
    uint64_t /*size*/,
    velox::ContinueFuture& /*future*/) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  // For non-preloaded initial split: move pinnedBuffer_ into pendingBuffers_.
  if (pinnedBuffer_ && !splitExhausted_) {
    pendingBytes_ += static_cast<int64_t>(pinnedBuffer_->size());
    pendingBuffers_.push_back(std::move(pinnedBuffer_));
    pinnedBuffer_.reset();
  }

  if (splitExhausted_ && pendingBuffers_.empty() && !splitReader_) {
    return nullptr;
  }

  auto startTimeUs = getCurrentTimeMicro();

  // Lazily create the reader from accumulated pending buffers.
  if (!splitReader_) {
    if (pendingBuffers_.empty()) {
      return nullptr;
    }
    createMultiSourceReader();
  }

  if (!splitReader_->has_next()) {
    splitReader_.reset();
    activeBuffers_.clear();
    splitExhausted_ = true;
    return nullptr;
  }

  // One-shot read: with both chunk/pass limits at 0, cuDF returns all rows
  // from all coalesced source files in a single call.
  auto tableWithMetadata = splitReader_->read_chunk();
  auto cudfTable = std::move(tableWithMetadata.tbl);

  VELOX_CHECK(
      !splitReader_->has_next(),
      "chunked_parquet_reader produced multiple chunks with "
      "chunk_read_limit=0 and pass_read_limit=0; "
      "set non-zero limits or add accumulation loop");

  // Reader fully consumed — release it and the backing buffers.
  splitReader_.reset();
  activeBuffers_.clear();
  splitExhausted_ = true;

  if (!cudfTable || cudfTable->num_rows() == 0) {
    return nullptr;
  }

  // Record scan timing via CUDA host callback.
  auto* callbackData =
      new TotalScanTimeCallbackData{startTimeUs, ioStatistics_};
  cudaLaunchHostFunc(
      stream_.value(),
      &CudfHiveDataSource2::totalScanTimeCalculator,
      callbackData);

  // Apply remaining filter if present.
  const auto rowsBefore = cudfTable->num_rows();
  const auto bytesBefore = estimateTableBytes(cudfTable);
  uint64_t filterTimeUs{0};
  if (remainingFilterExprSet_) {
    MicrosecondTimer filterTimer(&filterTimeUs);
    auto cols = cudfTable->release();
    std::vector<cudf::column_view> views;
    views.reserve(cols.size());
    for (const auto& col : cols) {
      views.push_back(col->view());
    }
    auto filterResult = cudfExpressionEvaluator_->eval(
        views, stream_, cudf::get_current_device_resource_ref());
    auto filterView = asView(filterResult);
    bool shouldFilter = [&]() {
      if (filterView.has_nulls()) {
        return true;
      }
      auto allTrue = cudf::reduce(
          filterView,
          *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
          cudf::data_type(cudf::type_id::BOOL8),
          stream_,
          cudf::get_current_device_resource_ref());
      using ScalarType = cudf::scalar_type_t<bool>;
      auto* result = static_cast<ScalarType*>(allTrue.get());
      return !(result->is_valid(stream_) && result->value(stream_));
    }();
    if (shouldFilter) {
      auto origTable = std::make_unique<cudf::table>(std::move(cols));
      cudfTable = cudf::apply_boolean_mask(
          *origTable,
          filterView,
          stream_,
          cudf::get_current_device_resource_ref());
    } else {
      cudfTable = std::make_unique<cudf::table>(std::move(cols));
    }
  }
  totalRemainingFilterTime_.fetch_add(
      filterTimeUs * 1000, std::memory_order_relaxed);

  const auto nRows = cudfTable->num_rows();
  LOG(WARNING) << "CudfHiveDataSource2::getOutput"
               << " beforeFilter: rows=" << rowsBefore
               << " size=" << succinctBytes(bytesBefore)
               << " afterFilter: rows=" << nRows
               << " size=" << succinctBytes(estimateTableBytes(cudfTable));
  if (nRows == 0) {
    return nullptr;
  }

  // Keep only output columns (drop filter-only columns).
  if (outputType_->size() < cudfTable->num_columns()) {
    auto cols = cudfTable->release();
    std::vector<std::unique_ptr<cudf::column>> outputCols;
    outputCols.reserve(outputType_->size());
    std::move(
        cols.begin(),
        cols.begin() + outputType_->size(),
        std::back_inserter(outputCols));
    cudfTable = std::make_unique<cudf::table>(std::move(outputCols));
  }

  stream_.synchronize();

  auto output = cudfIsRegistered()
      ? std::make_shared<CudfVector>(
            pool_, outputType_, nRows, std::move(cudfTable), stream_)
      : with_arrow::toVeloxColumn(
            cudfTable->view(), pool_, outputType_->names(), stream_);

  VELOX_CHECK_NOT_NULL(output, "cuDF to Velox conversion yielded nullptr");
  completedRows_ += output->size();
  return output;
}

// ---------------------------------------------------------------------------
// getRuntimeStats
// ---------------------------------------------------------------------------

std::unordered_map<std::string, RuntimeMetric>
CudfHiveDataSource2::getRuntimeStats() {
  std::unordered_map<std::string, RuntimeMetric> res;
  res.emplace(
      std::string(
          ::facebook::velox::connector::hive::HiveDataSource::kTotalScanTime),
      RuntimeMetric(
          ioStatistics_->totalScanTime(), RuntimeCounter::Unit::kNanos));
  res.emplace(
      Connector::kTotalRemainingFilterTime,
      RuntimeMetric(
          totalRemainingFilterTime_.load(std::memory_order_relaxed),
          RuntimeCounter::Unit::kNanos));
  if (numFilesCoalesced_ > 0) {
    res.emplace(
        "numFilesCoalesced",
        RuntimeMetric(numFilesCoalesced_, RuntimeCounter::Unit::kNone));
  }
  auto bufferTimeNs =
      totalCoalesceBufferTimeNs_.load(std::memory_order_relaxed);
  if (bufferTimeNs > 0) {
    res.emplace(
        "totalCoalesceBufferTimeNs",
        RuntimeMetric(bufferTimeNs, RuntimeCounter::Unit::kNanos));
  }
  return res;
}

// ---------------------------------------------------------------------------
// totalScanTimeCalculator — CUDA host callback.
// ---------------------------------------------------------------------------

void CudfHiveDataSource2::totalScanTimeCalculator(void* userData) {
  auto* data = static_cast<TotalScanTimeCallbackData*>(userData);
  auto endTimeUs = getCurrentTimeMicro();
  auto elapsedNs = (endTimeUs - data->startTimeUs) * 1000;
  data->ioStatistics->incTotalScanTime(elapsedNs);
  delete data;
}

} // namespace facebook::velox::cudf_velox::connector::hive
