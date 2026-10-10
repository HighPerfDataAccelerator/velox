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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderByteFetch.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorSplitPrefetch.h"
#include "velox/experimental/cudf/connectors/hive/PinnedHostBuffer.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"

#include "velox/common/io/IoStatistics.h"
#include "velox/common/io/Options.h"
#include "velox/connectors/Connector.h"
#include "velox/connectors/hive/FileHandle.h"
#include "velox/connectors/hive/TableHandle.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/type/Type.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/experimental/hybrid_scan_multifile.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/types.hpp>

#include <functional>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>

namespace facebook::velox::cudf_velox::connector::hive {

using namespace facebook::velox::connector;

using CudfParquetReader =
    cudf::io::parquet::experimental::hybrid_scan_multifile;
using CudfParquetReaderPtr = std::unique_ptr<CudfParquetReader>;
using CudfChunkedParquetReader = cudf::io::chunked_parquet_reader;
using CudfChunkedParquetReaderPtr = std::unique_ptr<CudfChunkedParquetReader>;
using CudfHybridScanReader =
    cudf::io::parquet::experimental::hybrid_scan_reader;
using CudfHybridScanReaderPtr = std::unique_ptr<CudfHybridScanReader>;

/// Normalizes decimal columns, recursively, to their logical Velox types.
/// When columnTypes is non-empty, it must describe every column after
/// numPrependedColumns, which are left unchanged. An empty columnTypes span
/// leaves cuDF's all-columns result unchanged for a zero-column projection.
/// Casts and buffer releases use the supplied stream.
std::unique_ptr<cudf::table> castDecimalColumnsToVeloxTypes(
    std::unique_ptr<cudf::table>&& table,
    std::span<const TypePtr> columnTypes,
    size_t numPrependedColumns,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr);

class CudfSplitReader : public NvtxHelper {
 public:
  CudfSplitReader(
      std::shared_ptr<CudfHiveConnectorSplit> split,
      std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
          tableHandle,
      const RowTypePtr& outputType,
      const std::vector<std::string>& readColumnNames,
      FileHandleFactory* fileHandleFactory,
      folly::Executor* executor,
      const ConnectorQueryCtx* connectorQueryCtx,
      const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
      const std::shared_ptr<io::IoStatistics>& ioStatistics,
      const std::shared_ptr<IoStats>& ioStats,
      const cudf::ast::expression* subfieldFilterAst);

  virtual ~CudfSplitReader();

  using PushdownFilterBuilder = std::function<cudf::ast::expression const*(
      const cudf::io::parquet::FileMetaData&)>;

  /// Sets a builder for a split-specific pushdown filter. The builder is
  /// invoked after the Parquet footer is read and before reader options are
  /// configured. The returned expression must remain alive while the split is
  /// being read.
  void setPushdownFilterBuilder(PushdownFilterBuilder builder) {
    pushdownFilterBuilder_ = std::move(builder);
  }

  /// Prepare the split: open cudf reader, set up data source and options.
  /// @param runtimeStats Reference to the DataSource's runtime statistics
  void prepareSplit(dwio::common::RuntimeStats& runtimeStats);

  /// Rebind state owned by the DataSource after an asynchronously prepared
  /// reader is moved to the driver thread.
  virtual void setDataSourceContext(
      const ConnectorQueryCtx* connectorQueryCtx,
      dwio::common::RuntimeStats& runtimeStats,
      cudf::ast::expression const* subfieldFilterExpr);

  /// A table chunk and its row count
  struct TableChunk {
    std::unique_ptr<cudf::table> table;
    vector_size_t numRows{0};
  };

  /// Read the next raw cudf table chunk. Returns nullopt when done.
  virtual std::optional<TableChunk> next(uint64_t size);

  /// Rebinds the query context of a reader prepared in the background to the
  /// context owned by the driver that reads it. Must be called before reading
  /// from a reader that outlives the context it was prepared with.
  void setConnectorQueryCtx(const ConnectorQueryCtx* connectorQueryCtx);

  /// Start the reads of the column chunks of the current pass without waiting
  /// for them. Does nothing when they are already in flight or complete.
  void startColumnChunkFetch();

  /// Get the stream.
  cuda::stream_ref stream() const {
    return stream_;
  }

  /// Releases the reader, data source, parquet metadata and any pending
  /// column chunk fetches.
  virtual void resetSplit();

 protected:
  // Performs split-specific setup after base reader state is reset.
  virtual void prepareSplitInternal(dwio::common::RuntimeStats& runtimeStats);

  // Returns whether the split is skipped.
  virtual bool isSplitSkipped() const;

  // Return the split-specific filter to push down to the cuDF reader.
  virtual cudf::ast::expression const* pushdownFilter() const;

  // Determine the output memory resource for the cuDF reader.
  virtual rmm::device_async_resource_ref determineCudfMemoryResource() const;

  // Read the next table chunk from the parquet reader. Returns nullopt when no
  // more data. All read decimals, including nested,filter-only and
  // equality-delete key columns, have their logical Velox scale and storage
  // width (DECIMAL64 for short decimals, DECIMAL128 for long decimals) before
  // deferred filters or equality deletes consume the table. A prepended
  // row-index column is not part of the logical read schema.
  virtual std::optional<TableChunk> readNextChunk();

  // Setup the cuDF data source
  void setupCudfDataSource();

  // Replaces grouped file-backed sources with complete pinned-host Parquet
  // buffers. All files in the group are submitted before waiting, and cuDF
  // applies the final projection while decoding them.
  void setupSelectivePreloadDataSources();

  // Return the size of the primary physical file after its data source has
  // been initialized.
  uint64_t primaryDataSourceSize() const;

  // Create a data source for one physical file.
  std::shared_ptr<cudf::io::datasource> createCudfDataSource(
      const std::string& filePath,
      std::optional<std::size_t> fileSize = std::nullopt);

  // Return non-owning wrappers for every source in this split.
  std::vector<std::unique_ptr<cudf::io::datasource>> makeDataSourceViews();

  // Read file metadatas.
  void fileMetaDatas();

  // Create the chunked parquet reader.
  void createCudfReader();

  // Create the experimental hybrid scan reader.
  // Requires exactly one footer.
  void createExperimentalReader();

  // Resolve row groups and fetch projected byte ranges for the experimental
  // reader. This is safe to call from split preparation or first next().
  void setupExperimentalScan();

  // Resolve projected ranges and fetch them into host memory during split
  // preload, leaving device allocation, H2D, and reader setup to the driver.
  void prepareExperimentalHostRead();

  // Registers metadata-only projected ranges as a best-effort AsyncDataCache
  // hint for a future regular-reader split.
  void setupCachePrefetchHint();

  void waitForCachePrefetchHint();

  void waitForCachePrefetchHint(bool splitPreload);

  void releaseCachePrefetchHint();

  // Whether to use the experimental cuDF reader.
  bool useExperimentalCudfReader() const;

  // Read file metadatas and cache `fileColumnNames_`, `baseReadOffset_` and
  // `splitRowCount_` from the Parquet footer.
  void cacheSchemaFromMetadata();

  // Returns the {start row, row count} covered by the split.
  std::pair<std::size_t, std::size_t> computeSplitRowRange() const;

  // Return the logical subfield filter AST used after reading.
  const cudf::ast::expression* subfieldFilterAst() const;

  // Return whether the pushdown filter was built for the current split.
  bool hasSplitSpecificPushdownFilter() const;

  std::shared_ptr<CudfHiveConnectorSplit> split_;
  std::shared_ptr<const ::facebook::velox::connector::hive::HiveTableHandle>
      tableHandle_;
  const RowTypePtr outputType_;
  std::vector<std::string> readColumnNames_;
  // Logical types aligned with readColumnNames_, including hidden read columns.
  std::vector<TypePtr> readColumnTypes_;

  FileHandleFactory* fileHandleFactory_;
  folly::Executor* executor_;
  const ConnectorQueryCtx* connectorQueryCtx_;

  std::shared_ptr<io::IoStatistics> ioStatistics_;
  std::shared_ptr<IoStats> ioStats_;

  cuda::stream_ref stream_{cudaStream_t{cudaStreamDefault}};

  // Parquet metadata(s) for the current split(s).
  std::vector<cudf::io::parquet::FileMetaData> fileMetaData_;

  // Whether to prepend a row index column to the output.
  bool prependRowIndex_{false};

  // Whether Parquet column names are matched case-insensitively.
  bool caseInsensitiveColumnNames_{false};

  // Top-level column names from the file metadata.
  std::unordered_set<std::string> fileColumnNames_;

  // Whether `readColumnNames_` is just file columns in the same order.
  bool readAllFileColumns_{false};

  // Tracks the absolute row range covered by the split.
  std::size_t baseReadOffset_{0};
  std::size_t splitRowCount_{0};

  // Whether the split reads no columns, so no cuDF reader exists and the row
  // count comes from the Parquet footer.
  bool noColumnsToRead_{false};

 private:
  // Stores row group indices for one Parquet source.
  using RowGroupIndices = std::vector<cudf::size_type>;

  // Stores row group indices by split.
  using RowGroupIndicesBySplit = std::vector<RowGroupIndices>;

  // Stores per-split row group indices by read pass.
  using RowGroupPasses = std::vector<RowGroupIndicesBySplit>;

  // Tracks how far the row group passes of the current split have been read.
  struct RowGroupPassState {
    // Row groups to read, indexed as passes[pass][split][rowGroup], in read
    // order.
    RowGroupPasses passes;

    // The pass being materialized.
    size_t currentPass{0};

    // Whether the chunked read of the current pass has been set up.
    bool isChunkingSetup{false};

    // Owns the device data of the current pass.
    ByteRangeFetch fetch;
  };

  // Setup the cuDF reader options
  void setupReaderOptions();

  // Setup Parquet column and offset indexes for the cudf split reader.
  void setupPageIndexes();

  // Return the row groups to read, grouped into passes bounded by the pass
  // read limit. Empty when the split has no row groups left after pruning.
  RowGroupPasses selectRowGroupPasses();

  // Wait for the column chunks of the current pass, fetching them first if
  // that has not started yet, and set up its chunked read.
  void setupChunkingForCurrentPass(rmm::device_async_resource_ref mr);

  // Wait for any reads of the current pass, then release its column chunk data.
  void releaseCurrentPassData();
  std::shared_ptr<CudfHiveConfig> cudfHiveConfig_;
  memory::MemoryPool* pool_;

  // cuDF split reader stuff.
  std::shared_ptr<cudf::io::datasource> dataSource_;
  std::vector<std::shared_ptr<cudf::io::datasource>> coalescedDataSources_;
  std::vector<std::shared_ptr<PinnedHostBuffer>> selectivePreloadBuffers_;
  std::shared_ptr<SplitPrefetchResult> selectivePreloadResult_;
  cudf::io::parquet_reader_options readerOptions_;
  CudfParquetReaderPtr splitReader_;
  CudfChunkedParquetReaderPtr chunkedSplitReader_;
  CudfHybridScanReaderPtr exptSplitReader_;
  std::unique_ptr<HybridScanState> hybridScanState_;
  bool useExperimentalCudfReader_{false};
  std::optional<std::string> cachePrefetchQueryId_;
  std::optional<std::string> cachePrefetchHintKey_;
  bool cachePrefetchHintWaited_{false};
  bool cachePrefetchDemandPrioritized_{false};
  bool cachePrefetchNonBlockingRequested_{false};
  bool cachePrefetchFirstLoadPolicyEnabled_{false};
  bool cachePrefetchFirstLoadReady_{false};

  // Row group passes and byte fetch states.
  std::unique_ptr<RowGroupPassState> passState_;
  bool serializeIoRequests_{false};

  // Chunk and pass read limits resolved from the session when the reader is
  // created.
  std::size_t chunkReadLimit_{0};
  std::size_t passReadLimit_{0};

  dwio::common::ReaderOptions baseReaderOpts_;
  const cudf::ast::expression* subfieldFilterAst_;
  cudf::ast::expression const* pushdownFilterExpr_;
  PushdownFilterBuilder pushdownFilterBuilder_;
  bool hasSplitSpecificPushdownFilter_{false};

  struct TotalScanTimeCallbackData {
    uint64_t startTimeUs;
    std::shared_ptr<io::IoStatistics> ioStatistics;
  };

  static void totalScanTimeCalculator(void* userData);
};

} // namespace facebook::velox::cudf_velox::connector::hive
