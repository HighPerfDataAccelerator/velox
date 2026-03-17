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

#include "velox/experimental/cudf/connectors/hive/CpuCoalescedDecoder.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/file/File.h"
#include "velox/common/time/Timer.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/parquet/reader/ParquetReader.h"

#include <cudf/utilities/error.hpp>
#include <glog/logging.h>

#include <cuda_runtime.h>
#include <rmm/device_buffer.hpp>

#include <cstring>

namespace facebook::velox::cudf_velox::connector::hive {

bool allFieldsFixedWidth(const RowTypePtr& type) {
  for (auto i = 0; i < type->size(); i++) {
    switch (type->childAt(i)->kind()) {
      case TypeKind::INTEGER:
      case TypeKind::BIGINT:
      case TypeKind::REAL:
      case TypeKind::DOUBLE:
        break;
      default:
        return false;
    }
  }
  return true;
}

CpuCoalescedDecoder::CpuCoalescedDecoder(
    std::vector<AsyncFileRead>& asyncFileReads,
    const RowTypePtr& readerOutputType,
    const RowTypePtr& outputType,
    const std::vector<std::string>& readColumnNames,
    const common::SubfieldFilters& subfieldFilters,
    exec::ExprSet* remainingFilterExprSet,
    core::ExpressionEvaluator* expressionEvaluator,
    memory::MemoryPool* pool,
    rmm::cuda_stream_view stream,
    int64_t targetBytes)
    : asyncFileReads_(asyncFileReads),
      readerOutputType_(readerOutputType),
      outputType_(outputType),
      readColumnNames_(readColumnNames),
      remainingFilterExprSet_(remainingFilterExprSet),
      expressionEvaluator_(expressionEvaluator),
      pool_(pool),
      stream_(stream) {
  // Build ScanSpec with subfieldFilters for row-group pruning and in-scan
  // predicate pushdown.
  scanSpec_ = std::make_shared<common::ScanSpec>("root");
  for (const auto& [subfield, filter] : subfieldFilters) {
    auto* fieldSpec = scanSpec_->getOrCreateChild(
        common::Subfield(subfield.toString()));
    fieldSpec->setFilter(filter->clone());
  }
  for (const auto& col : readColumnNames) {
    scanSpec_->getOrCreateChild(common::Subfield(col));
  }

  // Compute row byte width and pinned capacity.
  size_t rowByteWidth = 0;
  const auto numCols = readerOutputType->size();
  pinnedSlots_.resize(numCols);
  decodedColumns_.resize(numCols);
  for (size_t i = 0; i < numCols; i++) {
    auto& s = pinnedSlots_[i];
    s.dtype =
        cudf::data_type{veloxToCudfTypeId(readerOutputType->childAt(i))};
    s.elementSize = cudf::size_of(s.dtype);
    rowByteWidth += s.elementSize;
  }
  pinnedCapacity_ = static_cast<vector_size_t>(
      std::max<int64_t>(1, targetBytes / static_cast<int64_t>(rowByteWidth)));

  for (size_t i = 0; i < numCols; i++) {
    auto& s = pinnedSlots_[i];
    s.data = std::make_shared<PinnedHostBuffer>(
        s.elementSize * pinnedCapacity_);
    s.nulls = std::make_shared<PinnedHostBuffer>(
        (pinnedCapacity_ + 7) / 8);
    std::memset(s.nulls->data(), 0xFF, s.nulls->size());
    s.nullCount = 0;
  }
}

bool CpuCoalescedDecoder::advanceToNextFile() {
  if (nextFileIndex_ >= asyncFileReads_.size()) {
    return false;
  }

  auto pinnedBuf = asyncFileReads_[nextFileIndex_].future.get();
  currentPinnedBuf_ = std::move(pinnedBuf);

  auto readFile = std::make_shared<InMemoryReadFile>(std::string_view(
      reinterpret_cast<const char*>(currentPinnedBuf_->data()),
      currentPinnedBuf_->size()));

  dwio::common::ReaderOptions readerOpts(pool_);
  readerOpts.setFileFormat(dwio::common::FileFormat::PARQUET);

  auto input = std::make_unique<dwio::common::BufferedInput>(
      std::move(readFile), *pool_);
  currentReader_ = std::make_unique<parquet::ParquetReader>(
      std::move(input), readerOpts);

  dwio::common::RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(scanSpec_);
  rowReaderOpts.setRequestedType(readerOutputType_);

  currentRowReader_ = currentReader_->createRowReader(rowReaderOpts);
  LOG(WARNING) << "CPU decode fallback: file " << nextFileIndex_ << "/"
               << asyncFileReads_.size() << ", " << currentPinnedBuf_->size()
               << " bytes";
  ++nextFileIndex_;
  ++cpuDecodedFiles_;
  return true;
}

void CpuCoalescedDecoder::appendRows(
    const RowVectorPtr& rv,
    const vector_size_t* selectedIndices,
    vector_size_t numSelected) {
  SelectivityVector allRows(rv->size());

  for (size_t col = 0; col < pinnedSlots_.size(); col++) {
    auto& slot = pinnedSlots_[col];
    auto& decoded = decodedColumns_[col];
    decoded.decode(*rv->childAt(col), allRows);

    uint8_t* dstData =
        slot.data->data() + pinnedRowCount_ * slot.elementSize;
    auto* dstNulls = reinterpret_cast<uint64_t*>(slot.nulls->data());
    const auto* srcData =
        reinterpret_cast<const uint8_t*>(decoded.data<char>());

    if (!selectedIndices && decoded.isIdentityMapping()) {
      std::memcpy(dstData, srcData, numSelected * slot.elementSize);
      if (decoded.mayHaveNulls()) {
        bits::copyBits(
            decoded.nulls(), 0, dstNulls, pinnedRowCount_, numSelected);
        auto validBits =
            bits::countBits(decoded.nulls(), 0, numSelected);
        slot.nullCount += (numSelected - validBits);
      }
    } else {
      // Typed gather avoids per-element memcpy overhead; the compiler
      // emits a single mov for each 4-byte or 8-byte assignment.
      auto gatherTyped = [&](auto* dst, const auto* src) {
        for (vector_size_t j = 0; j < numSelected; j++) {
          auto topRow = selectedIndices ? selectedIndices[j] : j;
          dst[j] = src[decoded.index(topRow)];
          if (decoded.isNullAt(topRow)) {
            bits::clearBit(dstNulls, pinnedRowCount_ + j);
            ++slot.nullCount;
          }
        }
      };
      if (slot.elementSize == 8) {
        gatherTyped(
            reinterpret_cast<int64_t*>(dstData),
            reinterpret_cast<const int64_t*>(srcData));
      } else {
        gatherTyped(
            reinterpret_cast<int32_t*>(dstData),
            reinterpret_cast<const int32_t*>(srcData));
      }
    }
  }
  pinnedRowCount_ += numSelected;
}

void CpuCoalescedDecoder::resetPinnedSlots() {
  pinnedRowCount_ = 0;
  for (auto& slot : pinnedSlots_) {
    std::memset(slot.nulls->data(), 0xFF, slot.nulls->size());
    slot.nullCount = 0;
  }
}

std::optional<RowVectorPtr> CpuCoalescedDecoder::decodeNext() {
  uint64_t decodeTimeUs{0};
  {
    MicrosecondTimer timer(&decodeTimeUs);

    while (pinnedRowCount_ < pinnedCapacity_) {
      if (!currentRowReader_ && !advanceToNextFile()) {
        break;
      }

      VectorPtr result;
      auto scanned = currentRowReader_->next(kBatchRows, result);
      if (scanned == 0) {
        currentRowReader_.reset();
        currentReader_.reset();
        currentPinnedBuf_.reset();
        continue;
      }

      auto rv = std::dynamic_pointer_cast<RowVector>(result);
      vector_size_t numPassed = rv->size();
      const vector_size_t* selected = nullptr;

      if (remainingFilterExprSet_) {
        filterRows_.resize(rv->size());
        filterRows_.setAll();
        expressionEvaluator_->evaluate(
            remainingFilterExprSet_, filterRows_, *rv, filterResult_);
        numPassed = exec::processFilterResults(
            filterResult_, filterRows_, filterEvalCtx_, pool_);
        if (numPassed == 0) {
          continue;
        }
        if (numPassed < rv->size()) {
          selected =
              filterEvalCtx_.selectedIndices->as<vector_size_t>();
        }
      }

      auto space = pinnedCapacity_ - pinnedRowCount_;
      if (numPassed > space) {
        numPassed = space;
      }

      appendRows(rv, selected, numPassed);
    }
  }
  cpuDecodeTimeNs_.fetch_add(
      decodeTimeUs * 1000, std::memory_order_relaxed);

  if (pinnedRowCount_ == 0) {
    return std::nullopt;
  }

  cpuDecodedRows_ += pinnedRowCount_;
  LOG(WARNING) << "CPU decode fallback: flushing " << pinnedRowCount_
               << " rows (" << pinnedSlots_.size() << " cols) to GPU";

  // H2D transfer under GPU guard.
  GpuGuard guard;
  const auto numCols = pinnedSlots_.size();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(numCols);

  for (auto& slot : pinnedSlots_) {
    size_t dataBytes = pinnedRowCount_ * slot.elementSize;
    rmm::device_buffer devData(dataBytes, stream_);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        devData.data(),
        slot.data->data(),
        dataBytes,
        cudaMemcpyHostToDevice,
        stream_.value()));

    rmm::device_buffer devNulls;
    if (slot.nullCount > 0) {
      size_t nullBytes =
          static_cast<size_t>((pinnedRowCount_ + 7) / 8);
      devNulls = rmm::device_buffer(nullBytes, stream_);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          devNulls.data(),
          slot.nulls->data(),
          nullBytes,
          cudaMemcpyHostToDevice,
          stream_.value()));
    }

    columns.push_back(std::make_unique<cudf::column>(
        slot.dtype,
        static_cast<cudf::size_type>(pinnedRowCount_),
        std::move(devData),
        std::move(devNulls),
        slot.nullCount));
  }

  stream_.synchronize();

  auto nRows = pinnedRowCount_;
  auto cudfTable = std::make_unique<cudf::table>(std::move(columns));

  resetPinnedSlots();

  // Trim to outputType columns if readerOutputType had extra filter columns.
  if (outputType_->size() < static_cast<uint32_t>(cudfTable->num_columns())) {
    auto cols = cudfTable->release();
    cols.resize(outputType_->size());
    cudfTable = std::make_unique<cudf::table>(std::move(cols));
  }

  return std::make_shared<CudfVector>(
      pool_, outputType_, nRows, std::move(cudfTable), stream_);
}

} // namespace facebook::velox::cudf_velox::connector::hive
