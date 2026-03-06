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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnectorSplit.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/io/IoStatistics.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/type/Type.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox {
class PinnedHostBuffer;
} // namespace facebook::velox::cudf_velox

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

/// GPU data source with Parquet footer pre-filtering and selective I/O.
///
/// Designed for the Velox split preload path:
///   - addSplit() runs on an IO executor thread: reads footer, prunes
///     columns, filters row groups, reads only surviving byte ranges,
///     and assembles a self-contained Parquet buffer (PAR1 + chunks +
///     clipped footer + footer_len + PAR1).
///   - setFromDataSource() transfers the preloaded state to the driver.
///     When coalescing is enabled, multiple preloaded buffers are
///     accumulated in pendingBuffers_ until the size threshold is met.
///   - next() creates a cuDF chunked_parquet_reader from all pending
///     buffers as multiple source_info entries.
class CudfHiveDataSource2 : public DataSource, public NvtxHelper {
 public:
  CudfHiveDataSource2(
      const RowTypePtr& outputType,
      const ConnectorTableHandlePtr& tableHandle,
      const ColumnHandleMap& columnHandles,
      facebook::velox::FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig);

  /// Runs on IO executor during preload: reads footer, prunes columns,
  /// performs selective I/O, assembles self-contained Parquet buffer.
  void addSplit(std::shared_ptr<ConnectorSplit> split) override;

  void addDynamicFilter(
      column_index_t /*outputChannel*/,
      const std::shared_ptr<facebook::velox::common::Filter>& /*filter*/)
      override {
    VELOX_NYI("Dynamic filters not yet implemented by CudfHiveDataSource2.");
  }

  /// Runs on driver thread: creates cuDF reader from preloaded buffer,
  /// reads chunks, applies remaining filter.
  std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& future) override;

  /// Transfer preloaded state from a DataSource created on the IO executor.
  /// When coalescing is enabled, accumulates the source's pinned buffer.
  void setFromDataSource(std::unique_ptr<DataSource> source) override;

  /// Returns true while accumulated pending buffers are below the configured
  /// coalesce threshold, signaling the driver to fetch more splits.
  bool wantsMoreSplits() const override;

  uint64_t getCompletedRows() override {
    return completedRows_;
  }

  uint64_t getCompletedBytes() override {
    return completedBytes_;
  }

  std::unordered_map<std::string, RuntimeMetric> getRuntimeStats() override;

 private:
  /// Cast a generic ConnectorSplit to CudfHiveConnectorSplit, converting
  /// from HiveConnectorSplit if needed.
  std::shared_ptr<CudfHiveConnectorSplit> castToCudfSplit(
      const std::shared_ptr<ConnectorSplit>& split);

  const RowVectorPtr& getEmptyOutput() {
    if (!emptyOutput_) {
      emptyOutput_ = RowVector::createEmpty(outputType_, pool_);
    }
    return emptyOutput_;
  }

  RowVectorPtr emptyOutput_;

  std::shared_ptr<CudfHiveConnectorSplit> split_;
  std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
      tableHandle_;

  const std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  facebook::velox::FileHandleFactory* const fileHandleFactory_;
  folly::Executor* const executor_;
  const ConnectorQueryCtx* const connectorQueryCtx_;

  memory::MemoryPool* const pool_;

  // The row type for output, not including filter-only columns.
  const RowTypePtr outputType_;

  // Columns to read from the Parquet file.
  std::unordered_set<std::string> readColumnSet_;
  std::vector<std::string> readColumnNames_;

  // Expression evaluator for remaining filter.
  core::ExpressionEvaluator* const expressionEvaluator_;
  std::unique_ptr<exec::ExprSet> remainingFilterExprSet_;
  std::shared_ptr<velox::cudf_velox::CudfExpression> cudfExpressionEvaluator_;

  // Expression evaluator for subfield filter.
  std::vector<std::unique_ptr<cudf::scalar>> subfieldScalars_;
  cudf::ast::tree subfieldTree_;
  common::SubfieldFilters subfieldFilters_;
  cudf::ast::expression const* subfieldFilterExpr_{nullptr};

  /// Drain pendingBuffers_ into activeBuffers_ and create a multi-source
  /// cudf::io::chunked_parquet_reader.
  void createMultiSourceReader();

  // --- Preloaded state (populated by addSplit on IO executor) ---

  /// Self-contained Parquet buffer: PAR1 + surviving chunks + clipped footer
  /// + footer_len + PAR1.  Used for the initial non-preloaded split path.
  std::shared_ptr<cudf_velox::PinnedHostBuffer> pinnedBuffer_;

  /// True if the file was entirely filtered out (zero data I/O).
  bool splitExhausted_{true};

  // --- Coalescing state ---

  /// Buffers accumulated via setFromDataSource(), awaiting reader creation.
  std::vector<std::shared_ptr<cudf_velox::PinnedHostBuffer>> pendingBuffers_;
  int64_t pendingBytes_{0};

  /// Buffers currently being read by the chunked reader (kept alive for
  /// pointer stability).
  std::vector<std::shared_ptr<cudf_velox::PinnedHostBuffer>> activeBuffers_;

  // --- Driver-thread state ---

  /// cuDF chunked reader created lazily on first next() call.
  std::unique_ptr<cudf::io::chunked_parquet_reader> splitReader_;
  rmm::cuda_stream_view stream_{};

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  size_t completedRows_{0};
  size_t completedBytes_{0};

  // Timing & coalesce stats.
  std::atomic<uint64_t> totalRemainingFilterTime_{0};
  int64_t numFilesCoalesced_{0};
  std::atomic<uint64_t> totalCoalesceBufferTimeNs_{0};
  struct TotalScanTimeCallbackData {
    uint64_t startTimeUs;
    std::shared_ptr<io::IoStatistics> ioStatistics;
  };
  static void totalScanTimeCalculator(void* userData);
};

} // namespace facebook::velox::cudf_velox::connector::hive
