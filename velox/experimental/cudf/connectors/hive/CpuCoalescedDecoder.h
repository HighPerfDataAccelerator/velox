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

#include "velox/dwio/common/Reader.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/expression/Expr.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Returns true if every child of `type` is one of INTEGER, BIGINT, REAL,
/// DOUBLE. The CPU fallback decoder only supports these fixed-width types.
bool allFieldsFixedWidth(const RowTypePtr& type);

/// Async file read descriptor shared by CudfHiveDataSource and
/// CpuCoalescedDecoder.
struct AsyncFileRead {
  std::shared_future<std::shared_ptr<PinnedHostBuffer>> future;
  size_t fileSize;
  uint64_t start;
  uint64_t length;
};

/// CPU-side Parquet decoder for the coalesced multi-source path.
///
/// Reads PinnedHostBuffers (compact Parquet files from selectiveParquetRead)
/// using Velox's ParquetReader with ScanSpec filter pushdown, evaluates
/// the remaining filter on CPU, then copies passing rows into per-column
/// PinnedHostBuffer slots. Flushed to GPU via direct cudaMemcpyAsync and
/// assembled into cudf::column / cudf::table / CudfVector.
class CpuCoalescedDecoder {
 public:
  CpuCoalescedDecoder(
      std::vector<AsyncFileRead>& asyncFileReads,
      const RowTypePtr& readerOutputType,
      const RowTypePtr& outputType,
      const std::vector<std::string>& readColumnNames,
      const common::SubfieldFilters& subfieldFilters,
      exec::ExprSet* remainingFilterExprSet,
      core::ExpressionEvaluator* expressionEvaluator,
      memory::MemoryPool* pool,
      rmm::cuda_stream_view stream,
      int64_t targetBytes);

  ~CpuCoalescedDecoder() = default;

  /// Decode rows until the pinned buffer is full or all files are consumed.
  /// Returns a CudfVector, or std::nullopt when finished.
  std::optional<RowVectorPtr> decodeNext();

  // Metrics accessors.
  uint64_t cpuDecodeTimeNs() const {
    return cpuDecodeTimeNs_.load(std::memory_order_relaxed);
  }
  int64_t cpuDecodedFiles() const {
    return cpuDecodedFiles_;
  }
  int64_t cpuDecodedRows() const {
    return cpuDecodedRows_;
  }

 private:
  struct ColumnPinnedSlot {
    std::shared_ptr<PinnedHostBuffer> data;
    std::shared_ptr<PinnedHostBuffer> nulls;
    cudf::data_type dtype;
    size_t elementSize;
    cudf::size_type nullCount{0};
  };

  bool advanceToNextFile();
  void appendRows(
      const RowVectorPtr& rv,
      const vector_size_t* selectedIndices,
      vector_size_t numSelected);
  void resetPinnedSlots();

  static constexpr vector_size_t kBatchRows = 10'000;

  std::vector<AsyncFileRead>& asyncFileReads_;
  const RowTypePtr readerOutputType_;
  const RowTypePtr outputType_;
  const std::vector<std::string>& readColumnNames_;
  exec::ExprSet* const remainingFilterExprSet_;
  core::ExpressionEvaluator* const expressionEvaluator_;
  memory::MemoryPool* const pool_;
  rmm::cuda_stream_view stream_;

  size_t nextFileIndex_{0};
  std::shared_ptr<PinnedHostBuffer> currentPinnedBuf_;
  std::unique_ptr<dwio::common::Reader> currentReader_;
  std::unique_ptr<dwio::common::RowReader> currentRowReader_;

  std::vector<ColumnPinnedSlot> pinnedSlots_;
  vector_size_t pinnedRowCount_{0};
  vector_size_t pinnedCapacity_{0};

  std::shared_ptr<common::ScanSpec> scanSpec_;

  VectorPtr filterResult_;
  SelectivityVector filterRows_;
  exec::FilterEvalCtx filterEvalCtx_;

  std::vector<DecodedVector> decodedColumns_;

  // Metrics
  std::atomic<uint64_t> cpuDecodeTimeNs_{0};
  int64_t cpuDecodedFiles_{0};
  int64_t cpuDecodedRows_{0};
};

} // namespace facebook::velox::cudf_velox::connector::hive
