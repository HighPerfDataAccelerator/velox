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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/BoundedBatchWriter.h"
#include "velox/experimental/cudf/connectors/hive/CudfBoundedFileSink.h"
#include "velox/experimental/cudf/connectors/hive/CudfEagerFileSink.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSink.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/Fs.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/OperatorUtils.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtx3/nvtx3.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <chrono>
#include <cstdlib>

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

bool asyncWriterEnabled() {
  const auto* value = std::getenv("GLUTEN_CUDF_ASYNC_TABLE_WRITE");
  return value != nullptr && std::string_view(value) == "1";
}

bool batchWriterEnabled() {
  const auto* value = std::getenv("GLUTEN_CUDF_ASYNC_TABLE_WRITE_COALESCE");
  return asyncWriterEnabled() && value != nullptr &&
      std::string_view(value) == "1";
}

uint64_t asyncWriterMaxInputBytes() {
  constexpr uint64_t kDefault = 1ULL << 30;
  const auto* value =
      std::getenv("GLUTEN_CUDF_ASYNC_TABLE_WRITE_MAX_INPUT_BYTES");
  if (value == nullptr) {
    return kDefault;
  }
  char* end = nullptr;
  const auto bytes = std::strtoull(value, &end, 10);
  return end != value && *end == '\0' && bytes > 0 && bytes <= kDefault
      ? bytes
      : kDefault;
}

std::unordered_map<LocationHandle::TableType, std::string> tableTypeNames() {
  return {
      {LocationHandle::TableType::kNew, "kNew"},
  };
}

template <typename K, typename V>
std::unordered_map<V, K> invertMap(const std::unordered_map<K, V>& mapping) {
  std::unordered_map<V, K> inverted;
  for (const auto& [key, value] : mapping) {
    inverted.emplace(value, key);
  }
  return inverted;
}

uint64_t getFinishTimeSliceLimitMsFromCudfHiveConfig(
    const std::shared_ptr<const CudfHiveConfig>& config,
    const config::ConfigBase* sessions) {
  const uint64_t flushTimeSliceLimitMsFromConfig =
      config->sortWriterFinishTimeSliceLimitMs(sessions);
  // NOTE: if the flush time slice limit is set to 0, then we treat it as no
  // limit.
  return flushTimeSliceLimitMsFromConfig == 0
      ? std::numeric_limits<uint64_t>::max()
      : flushTimeSliceLimitMsFromConfig;
}

std::string makeUuid() {
  return boost::lexical_cast<std::string>(boost::uuids::random_generator()());
}

std::string cudfWritePath(std::string path) {
  // Hadoop passes local task-attempt directories as file:/... URIs. libcudf's
  // local sink expects an operating-system path; otherwise it creates a
  // literal "file:" directory below the executor working directory, outside
  // the FileOutputCommitter staging tree.
  constexpr std::string_view kFilePrefix{"file:"};
  if (path.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    path.erase(0, kFilePrefix.size());
  }
  return path;
}

cudf::io::compression_type getCompressionType(
    facebook::velox::common::CompressionKind name) {
  using CompressionType = cudf::io::compression_type;

  static std::unordered_map<
      facebook::velox::common::CompressionKind,
      CompressionType> const kMap = {
      {facebook::velox::common::CompressionKind::CompressionKind_NONE,
       CompressionType::NONE},
      {facebook::velox::common::CompressionKind::CompressionKind_SNAPPY,
       CompressionType::SNAPPY},
      {facebook::velox::common::CompressionKind::CompressionKind_LZ4,
       CompressionType::LZ4},
      {facebook::velox::common::CompressionKind::CompressionKind_ZSTD,
       CompressionType::ZSTD}};

  VELOX_CHECK(
      kMap.find(name) != kMap.end(),
      "Unsupported compression type requested. Supported compression types are: "
      "NONE, SNAPPY, LZ4, ZSTD");

  return kMap.at(name);
}

std::shared_ptr<memory::MemoryPool> createSinkPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sink", writerPool->name()));
}

std::shared_ptr<memory::MemoryPool> createSortPool(
    const std::shared_ptr<memory::MemoryPool>& writerPool) {
  return writerPool->addLeafChild(fmt::format("{}.sort", writerPool->name()));
}

} // namespace

struct CudfWriterBatchState {
  struct Input {
    DeviceMemoryWorkspaceReservation reservation;
    std::shared_ptr<CudfVector> owner;
    CudaEvent ready;
    uint64_t bytes;
  };

  explicit CudfWriterBatchState(BoundedBatchWriter<Input>::Consumer consume)
      : queue(asyncWriterMaxInputBytes(), 64, std::move(consume)) {}

  BoundedBatchWriter<Input> queue;
  uint64_t coalescedBatches{0};
  uint64_t coalescedInputs{0};
  uint64_t admissionFallbackBatches{0};
};

const std::string LocationHandle::tableTypeName(
    LocationHandle::TableType type) {
  static const auto kTableTypes = tableTypeNames();
  return kTableTypes.at(type);
}

LocationHandle::TableType LocationHandle::tableTypeFromName(
    const std::string& name) {
  static const auto kNameTableTypes = invertMap(tableTypeNames());
  return kNameTableTypes.at(name);
}

CudfHiveDataSink::CudfHiveDataSink(
    RowTypePtr inputType,
    std::shared_ptr<const CudfHiveInsertTableHandle> insertTableHandle,
    const ConnectorQueryCtx* connectorQueryCtx,
    CommitStrategy commitStrategy,
    const std::shared_ptr<const CudfHiveConfig>& parquetConfig)
    : inputType_(std::move(inputType)),
      insertTableHandle_(std::move(insertTableHandle)),
      connectorQueryCtx_(connectorQueryCtx),
      commitStrategy_(commitStrategy),
      parquetConfig_(parquetConfig),
      spillConfig_(connectorQueryCtx->spillConfig()),
      sortWriterFinishTimeSliceLimitMs_(
          getFinishTimeSliceLimitMsFromCudfHiveConfig(
              parquetConfig_,
              connectorQueryCtx->sessionProperties())) {
  VELOX_USER_CHECK(
      (commitStrategy_ == CommitStrategy::kNoCommit) ||
          (commitStrategy_ == CommitStrategy::kTaskCommit),
      "Unsupported commit strategy: {}",
      CommitStrategyName::toName(commitStrategy_));

  const auto& writerOptions = dynamic_cast<CudfHiveWriterOptions*>(
      insertTableHandle_->writerOptions().get());

  if (writerOptions != nullptr) {
    sortingColumns_ = std::move(writerOptions->sortingColumns);
  }
}

void CudfHiveDataSink::appendData(RowVectorPtr input) {
  nvtx3::scoped_range appendRange("CudfWriter::appendData");
  checkRunning();
  auto deviceInput = std::dynamic_pointer_cast<CudfVector>(input);
  // The experimental consumer serializes the writer independently. Legacy
  // futures and host-converted inputs must still drain before writer access.
  if (!batchWriterEnabled() || !deviceInput || pendingWrite_.valid()) {
    awaitPendingWrite();
  }

  // Preserve a device-resident pipeline into libcudf's file writer. The
  // generic Arrow conversion below materializes a CudfVector on the host and
  // copies the same columns back to the GPU. Wide TopN output made that round
  // trip and its synchronization dominate table-write tasks.
  if (deviceInput) {
    const auto inputStream = deviceInput->stream();
    const auto inputView = deviceInput->getTableView();
    if (!writer_.has_value()) {
      writerStream_ = asyncWriterEnabled() ? cudfGlobalStreamPool().get_stream()
                                           : inputStream;
      writer_ = createCudfWriter(inputView, *writerStream_);
    }
    VELOX_CHECK(writerStream_.has_value());

    const auto inputBytes = deviceInput->estimateFlatSize();
    auto workspace = asyncWriterEnabled() && inputBytes > 0 &&
            inputBytes <= asyncWriterMaxInputBytes()
        ? tryAcquireBackgroundDeviceMemoryWorkspace(
              2 * inputBytes,
              512ULL << 20,
              DeviceMemoryWorkspacePriority::kOutput)
        : std::optional<DeviceMemoryWorkspaceReservation>{};
    if (workspace.has_value() && batchWriterEnabled()) {
      if (!batchWriter_) {
        int device = 0;
        CUDF_CUDA_TRY(cudaGetDevice(&device));
        batchWriter_ = std::make_unique<CudfWriterBatchState>(
            [this, device](std::vector<CudfWriterBatchState::Input>& inputs) {
              CUDF_CUDA_TRY(cudaSetDevice(device));
              nvtx3::scoped_range workerRange("CudfWriter::asyncBatch");
              // Keep temporaries alive until the exception handler has also
              // stopped any already-submitted CUDA work.
              std::unique_ptr<cudf::table> combined;
              std::optional<DeviceMemoryWorkspaceReservation> concatWorkspace;
              try {
                std::vector<cudf::table_view> views;
                uint64_t bytes = 0;
                for (const auto& input : inputs) {
                  input.ready.waitOn(*writerStream_);
                  views.push_back(input.owner->getTableView());
                  bytes += input.bytes;
                }
                if (inputs.size() > 1) {
                  // Each accepted input already owns its original 2x writer
                  // workspace credit. Concatenation needs EXTRA credit; if
                  // unavailable, drain individually, never wait for upstream.
                  concatWorkspace = tryAcquireBackgroundDeviceMemoryWorkspace(
                      bytes,
                      512ULL << 20,
                      DeviceMemoryWorkspacePriority::kOutput);
                  if (concatWorkspace) {
                    nvtx3::scoped_range range(
                        "CudfWriter::coalesceReadyInputs");
                    combined =
                        cudf::concatenate(views, *writerStream_, get_temp_mr());
                    ++batchWriter_->coalescedBatches;
                    batchWriter_->coalescedInputs += inputs.size();
                  } else {
                    ++batchWriter_->admissionFallbackBatches;
                  }
                }
                if (combined) {
                  writeCudf(combined->view());
                  ++asyncWriteBatches_;
                } else {
                  for (const auto& view : views) {
                    writeCudf(view);
                    ++asyncWriteBatches_;
                  }
                }
                {
                  nvtx3::scoped_range finishRange("CudfWriter::finishStream");
                  writerStream_->synchronize();
                }
                combined.reset();
                // Input owners are destroyed by the queue before it returns
                // byte/item credit. Their workspace tokens remain live too.
              } catch (...) {
                cudaStreamSynchronize(writerStream_->value());
                throw;
              }
            });
      }
      CudaEvent ready(cudaEventDisableTiming);
      ready.recordFrom(inputStream);
      const auto start = std::chrono::steady_clock::now();
      {
        nvtx3::scoped_range waitRange("CudfWriter::awaitBatchCapacity");
        batchWriter_->queue.submit(
            {std::move(*workspace), deviceInput, std::move(ready), inputBytes},
            inputBytes);
      }
      asyncWriteWaitMicros_ +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
    } else if (workspace.has_value()) {
      int device = 0;
      CUDF_CUDA_TRY(cudaGetDevice(&device));
      CudaEvent ready(cudaEventDisableTiming);
      ready.recordFrom(inputStream).waitOn(*writerStream_);
      // The strong vector owner survives the complete write and CUDA tail.
      // The cooperative workspace lease protects concurrent join restore
      // admission; unavailable admission takes the original synchronous path.
      pendingWrite_ = std::async(
          std::launch::async,
          [this,
           device,
           owner = deviceInput,
           reservation = std::move(*workspace)]() mutable {
            CUDF_CUDA_TRY(cudaSetDevice(device));
            nvtx3::scoped_range workerRange("CudfWriter::asyncBatch");
            try {
              writeCudf(owner->getTableView());
              {
                nvtx3::scoped_range finishRange("CudfWriter::finishStream");
                writerStream_->synchronize();
              }
              {
                nvtx3::scoped_range releaseRange("CudfWriter::releaseInput");
                owner.reset();
              }
              {
                nvtx3::scoped_range releaseRange(
                    "CudfWriter::releaseWorkspace");
                reservation.release();
              }
            } catch (...) {
              // Keep the source alive until already-submitted work stops,
              // including on exception. Preserve the original write error.
              cudaStreamSynchronize(writerStream_->value());
              owner.reset();
              reservation.release();
              throw;
            }
          });
      ++asyncWriteBatches_;
    } else {
      // Oversize or admission failure: the synchronous fallback cannot touch
      // a writer that still has accepted background batches.
      awaitPendingWrite();
      if (writerStream_->value() == inputStream.value()) {
        // The writer work and CudfVector's stream-ordered destruction use the
        // same stream, so the input remains live until the writer consumes it.
        writeCudf(inputView);
      } else {
        // Inputs from a different producer stream need ordering in both
        // directions: writer waits for production, then producer waits before
        // its stream-ordered input destruction. Neither wait blocks the host.
        CudaEvent inputReady(cudaEventDisableTiming);
        CudaEvent inputConsumed(cudaEventDisableTiming);
        inputReady.recordFrom(inputStream).waitOn(*writerStream_);
        writeCudf(inputView);
        inputConsumed.recordFrom(*writerStream_).waitOn(inputStream);
      }
    }

    if (!loggedDeviceInput_) {
      LOG(WARNING) << "CudfHiveDataSink writing CudfVector without host "
                      "Arrow round trip";
      loggedDeviceInput_ = true;
    }
    TestValue::adjust(
        "facebook::velox::cudf_velox::connector::hive::"
        "CudfHiveDataSink::appendDeviceData",
        this);
    writerInfo_->inputSizeInBytes += input->estimateFlatSize();
    writerInfo_->numWrittenRows += input->size();
    return;
  }

  // Convert the input RowVectorPtr to cudf::table
  auto stream = cudfGlobalStreamPool().get_stream();
  auto cudfInput =
      with_arrow::toCudfTable(input, input->pool(), stream, get_temp_mr());
  stream.synchronize();
  VELOX_CHECK_NOT_NULL(
      cudfInput, "Failed to convert input RowVectorPtr to cudf::table");

  // Check if the writer doesn't already exist
  if (!writer_.has_value()) {
    writer_ = createCudfWriter(cudfInput->view(), stream);
    writerStream_ = stream;
  }

  // Write the table to the sink
  writeCudf(cudfInput->view());
  writerInfo_->inputSizeInBytes += input->estimateFlatSize();
  writerInfo_->numWrittenRows += input->size();
}

CudfHiveDataSink::~CudfHiveDataSink() {
  // A cancelled task may destroy the sink without calling close(). Do not
  // let a worker outlive this object or its writer. Normal finish/close/abort
  // consumes errors; destructors must not throw during exception unwinding.
  if (pendingWrite_.valid()) {
    pendingWrite_.wait();
  }
  if (batchWriter_) {
    try {
      batchWriter_->queue.drain();
    } catch (...) {
      // drain propagates only after all accepted owners have been released.
    }
    batchWriter_.reset();
  }
}

void CudfHiveDataSink::awaitPendingWrite() {
  if (batchWriter_) {
    const auto start = std::chrono::steady_clock::now();
    {
      nvtx3::scoped_range range("CudfWriter::drainBatchedWrites");
      batchWriter_->queue.drain();
    }
    asyncWriteWaitMicros_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start)
            .count();
  }
  if (!pendingWrite_.valid()) {
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  {
    // This is the producer's exposed wait, not the worker's full write time.
    nvtx3::scoped_range waitRange("CudfWriter::awaitPreviousWrite");
    pendingWrite_.get();
  }
  asyncWriteWaitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
}

CudfHiveDataSink::CudfWriter CudfHiveDataSink::createCudfWriter(
    cudf::table_view cudfTable,
    rmm::cuda_stream_view stream) {
  // Create a table_input_metadata from the input
  auto tableInputMetadata = createCudfTableInputMetadata(cudfTable);

  auto compressionKind =
      getCompressionType(insertTableHandle_->compressionKind().value_or(
          facebook::velox::common::CompressionKind::CompressionKind_NONE));

  // Create a sink and writer
  const auto& locationHandle = insertTableHandle_->locationHandle();
  const auto fileFormat = insertTableHandle_->storageFormat();
  const auto defaultExtension =
      fileFormat == dwio::common::FileFormat::ORC ? ".orc" : ".parquet";
  const auto targetFileName = locationHandle->targetFileName().empty()
      ? fmt::format("{}{}", makeUuid(), defaultExtension)
      : locationHandle->targetFileName();

  auto writerParameters = CudfHiveWriterParameters(
      CudfHiveWriterParameters::UpdateMode::kNew,
      targetFileName,
      locationHandle->targetPath());

  const auto localWriteDirectory =
      cudfWritePath(writerParameters.writeDirectory());
  const bool isS3 = localWriteDirectory.rfind("s3://", 0) == 0 ||
      localWriteDirectory.rfind("s3a://", 0) == 0 ||
      localWriteDirectory.rfind("s3n://", 0) == 0;
  const auto* directS3 = std::getenv("GLUTEN_CUDF_DIRECT_S3_SINK");
  VELOX_CHECK(
      !isS3 || (directS3 && std::string_view(directS3) == "1"),
      "Native S3 writes require GLUTEN_CUDF_DIRECT_S3_SINK=1");
  // filesystem::path can normalize away a slash in the URI scheme.
  const auto writePath = isS3
      ? fmt::format(
            "{}/{}", localWriteDirectory, writerParameters.writeFileName())
      : (fs::path(localWriteDirectory) / writerParameters.writeFileName())
            .string();

  // Spark's FileCommitProtocol can hand each native writer a task-attempt
  // directory below the final output root. libcudf/kvikio opens the target
  // file directly and does not create missing parents, unlike Velox's generic
  // WriteFile path. Create the complete directory here before constructing the
  // file sink. mkdir is recursive and idempotent for concurrent writers.
  if (!isS3) {
    filesystems::getFileSystem(
        writerParameters.writeDirectory(), parquetConfig_->config())
        ->mkdir(localWriteDirectory);
  }

  makeWriterOptions(writerParameters);

  // Create writer options for the given sink
  const auto* eagerSink = std::getenv("GLUTEN_CUDF_BOUNDED_EAGER_FILE_SINK");
  if (isS3) {
    auto fileSystem =
        filesystems::getFileSystem(writePath, parquetConfig_->config());
    filesystems::FileOptions options;
    options.pool = writerInfo_->sinkPool.get();
    boundedFileSink_ = std::make_unique<CudfEagerFileSink>(
        writePath, fileSystem->openFileForWrite(writePath, options));
    LOG(WARNING) << "CudfHiveDataSink: direct S3 sink for " << writePath;
  } else if (eagerSink && std::string_view(eagerSink) == "1") {
    boundedFileSink_ = std::make_unique<CudfBoundedFileSink>(writePath);
    LOG(WARNING) << "CUDF_BOUNDED_EAGER_FILE_SINK depth=4";
  }
  const auto sinkInfo = boundedFileSink_
      ? cudf::io::sink_info(boundedFileSink_.get())
      : cudf::io::sink_info(writePath);
  if (fileFormat == dwio::common::FileFormat::ORC) {
    VELOX_CHECK(
        cudf::io::is_supported_write_orc(compressionKind),
        "Unsupported libcudf ORC compression codec: {}",
        static_cast<int>(compressionKind));
    auto cudfWriterOptions =
        cudf::io::chunked_orc_writer_options::builder(sinkInfo)
            .metadata(std::move(tableInputMetadata))
            .compression(compressionKind)
            .build();
    return std::make_unique<cudf::io::orc_chunked_writer>(
        cudfWriterOptions, stream);
  }

  auto cudfWriterOptions =
      cudf::io::chunked_parquet_writer_options::builder(sinkInfo)
          .metadata(tableInputMetadata)
          .utc_timestamps(parquetConfig_->writeTimestampsAsUTC())
          .write_arrow_schema(parquetConfig_->writeArrowSchema())
          .write_v2_headers(parquetConfig_->writev2PageHeaders())
          .compression(compressionKind)
          .build();

  const auto& writerOptions = dynamic_cast<CudfHiveWriterOptions*>(
      insertTableHandle_->writerOptions().get());

  // If non-null writerOptions were passed, pass them to the chunked parquet
  // writer options
  if (writerOptions != nullptr) {
    // Set encoding for all columns
    std::for_each(
        tableInputMetadata.column_metadata.begin(),
        tableInputMetadata.column_metadata.end(),
        [=](auto& colMeta) { colMeta.set_encoding(writerOptions->encoding); });

    cudfWriterOptions.set_row_group_size_bytes(
        writerOptions->rowGroupSizeBytes);
    cudfWriterOptions.set_row_group_size_rows(writerOptions->rowGroupSizeRows);
    cudfWriterOptions.set_max_page_size_bytes(writerOptions->maxPageSizeBytes);
    cudfWriterOptions.set_max_page_size_rows(writerOptions->maxPageSizeRows);
    cudfWriterOptions.set_dictionary_policy(writerOptions->dictionaryPolicy);
    cudfWriterOptions.set_max_dictionary_size(writerOptions->maxDictionarySize);
    cudfWriterOptions.enable_int96_timestamps(
        writerOptions->writeTimestampsAsInt96);

    // Enable if enabled in the session or the writerOptions
    cudfWriterOptions.enable_utc_timestamps(
        parquetConfig_->writeTimestampsAsUTC() or
        writerOptions->writeTimestampsAsUTC);
    cudfWriterOptions.enable_write_arrow_schema(
        parquetConfig_->writeArrowSchema() or writerOptions->writeArrowSchema);
    cudfWriterOptions.enable_write_v2_headers(
        parquetConfig_->writev2PageHeaders() or writerOptions->v2PageHeaders);
    cudfWriterOptions.set_stats_level(writerOptions->statsLevel);

    if (writerOptions->maxPageFragmentSize.has_value()) {
      cudfWriterOptions.set_max_page_fragment_size(
          writerOptions->maxPageFragmentSize.value());
    }
    // Get compression stats if needed
    if (writerOptions->compressionStats != nullptr) {
      cudfWriterOptions.set_compression_statistics(
          writerOptions->compressionStats);
    }
    // Write sorting columns if available
    if (!sortingColumns_.empty()) {
      cudfWriterOptions.set_sorting_columns(sortingColumns_);
    }
  }

  return std::make_unique<cudf::io::chunked_parquet_writer>(
      cudfWriterOptions, stream);
}

void CudfHiveDataSink::writeCudf(cudf::table_view cudfTable) {
  nvtx3::scoped_range writeRange("CudfWriter::writeBatch");
  VELOX_CHECK(writer_.has_value());
  std::visit(
      [&](auto& writer) {
        VELOX_CHECK_NOT_NULL(writer);
        writer->write(cudfTable);
      },
      *writer_);
}

void CudfHiveDataSink::closeCudf() {
  nvtx3::scoped_range closeRange("CudfWriter::close");
  VELOX_CHECK(writer_.has_value());
  std::visit(
      [](auto& writer) {
        VELOX_CHECK_NOT_NULL(writer);
        writer->close();
      },
      *writer_);
}

cudf::io::table_input_metadata CudfHiveDataSink::createCudfTableInputMetadata(
    cudf::table_view cudfTable) {
  auto tableInputMetadata = cudf::io::table_input_metadata(cudfTable);
  auto inputColumns = insertTableHandle_->inputColumns();

  // Check if equal number of columns in the input and
  // CudfHiveInsertTableHandle
  VELOX_CHECK_EQ(
      tableInputMetadata.column_metadata.size(),
      inputColumns.size(),
      "Unequal number of columns in the input and CudfHiveInsertTableHandle");

  std::function<void(
      cudf::io::column_in_metadata&, const CudfHiveColumnHandle&)>
      setColumnName = [&](cudf::io::column_in_metadata& colMeta,
                          const CudfHiveColumnHandle& columnHandle) {
        // Check if equal number of children
        const auto& childrenHandles = columnHandle.children();

        // Warn if the mismatch in the number of child cols in CudfHive
        // table_metadata and columnHandles
        if (colMeta.num_children() != childrenHandles.size()) {
          LOG(WARNING) << fmt::format(
              "({} vs {}): Unequal number of child columns in CudfHive table_metadata and ColumnHandles",
              colMeta.num_children(),
              childrenHandles.size());
        }

        // Set children's names
        for (int32_t i = 0; i <
             std::min<int32_t>(colMeta.num_children(), childrenHandles.size());
             ++i) {
          setColumnName(colMeta.child(i), childrenHandles[i]);
        }
        // Set this column's name
        colMeta.set_name(columnHandle.name());
      };

  // Set names for all columns and their children
  for (int32_t i = 0; i < tableInputMetadata.column_metadata.size(); ++i) {
    setColumnName(tableInputMetadata.column_metadata[i], *inputColumns[i]);
  }

  return tableInputMetadata;
}

std::string CudfHiveDataSink::stateString(State state) {
  switch (state) {
    case State::kRunning:
      return "RUNNING";
    case State::kFinishing:
      return "FLUSHING";
    case State::kClosed:
      return "CLOSED";
    case State::kAborted:
      return "ABORTED";
    default:
      VELOX_UNREACHABLE("BAD STATE: {}", static_cast<int>(state));
  }
}

DataSink::Stats CudfHiveDataSink::stats() const {
  Stats stats;
  if (state_ == State::kAborted) {
    return stats;
  }

  int64_t numWrittenBytes{0};
  int64_t writeIOTimeUs{0};

  if (state_ == State::kClosed) {
    numWrittenBytes = writtenBytes_;
  } else if (ioStatistics_ != nullptr) {
    numWrittenBytes = ioStatistics_->rawBytesWritten();
    writeIOTimeUs = ioStatistics_->writeIOTimeUs();
  }

  stats.numWrittenBytes = numWrittenBytes;
  stats.writeIOTimeUs = writeIOTimeUs;

  if (state_ != State::kClosed) {
    return stats;
  }

  if (writerInfo_ == nullptr) {
    return stats;
  }
  stats.numWrittenFiles = 1;
  if (!writerInfo_->spillStats->empty()) {
    stats.spillStats += *writerInfo_->spillStats;
  }

  return stats;
}

void CudfHiveDataSink::setState(State newState) {
  checkStateTransition(state_, newState);
  state_ = newState;
}

/// Validates the state transition from 'oldState' to 'newState'.
void CudfHiveDataSink::checkStateTransition(State oldState, State newState) {
  switch (oldState) {
    case State::kRunning:
      if (newState == State::kAborted || newState == State::kFinishing) {
        return;
      }
      break;
    case State::kFinishing:
      if (newState == State::kAborted || newState == State::kClosed ||
          // The finishing state is reentry state if we yield in the
          // middle of finish processing if a single run takes too long.
          newState == State::kFinishing) {
        return;
      }
      [[fallthrough]];
    case State::kAborted:
    case State::kClosed:
    default:
      break;
  }
  VELOX_FAIL("Unexpected state transition from {} to {}", oldState, newState);
}

bool CudfHiveDataSink::finish() {
  setState(State::kFinishing);
  awaitPendingWrite();
  return true;
}

std::vector<std::string> CudfHiveDataSink::close() {
  setState(State::kClosed);
  closeInternal();

  std::vector<std::string> partitionUpdates{};

  // An empty writer task does not create a data file or a commit fragment.
  if (writerInfo_ == nullptr) {
    return partitionUpdates;
  }

  partitionUpdates.reserve(1);
  // clang-format off
    auto partitionUpdateJson = folly::toJson(
     folly::dynamic::object
        ("writePath", writerInfo_->writerParameters.writeDirectory())
        ("targetPath", writerInfo_->writerParameters.targetDirectory())
        ("fileWriteInfos", folly::dynamic::array(
          folly::dynamic::object
            ("writeFileName", writerInfo_->writerParameters.writeFileName())
            ("targetFileName", writerInfo_->writerParameters.targetFileName())
            ("fileSize", writtenBytes_)))
        ("rowCount", writerInfo_->numWrittenRows)
        ("inMemoryDataSizeInBytes", writerInfo_->inputSizeInBytes)
        ("onDiskDataSizeInBytes", writtenBytes_)
        ("containsNumberedFileNames", true));
  // clang-format on
  partitionUpdates.emplace_back(partitionUpdateJson);

  return partitionUpdates;
}

void CudfHiveDataSink::abort() {
  setState(State::kAborted);
  closeInternal();
}

void CudfHiveDataSink::closeInternal() {
  VELOX_CHECK_NE(state_, State::kRunning);
  VELOX_CHECK_NE(state_, State::kFinishing);
  awaitPendingWrite();
  if (!writer_.has_value()) {
    return;
  }

  TestValue::adjust(
      "facebook::velox::connector::hive::CudfHiveDataSink::closeInternal",
      this);

  // Close cudf writer
  closeCudf();
  if (asyncWriteBatches_ > 0) {
    LOG(WARNING) << "CUDF_ASYNC_TABLE_WRITE batches=" << asyncWriteBatches_
                 << " waitUs=" << asyncWriteWaitMicros_
                 << " maxInFlightInputs=" << (batchWriter_ ? 64 : 1)
                 << " maxInputBytes=" << asyncWriterMaxInputBytes();
  }
  if (batchWriter_) {
    const auto stats = batchWriter_->queue.stats();
    LOG(WARNING) << "CUDF_WRITER_READY_BATCHES inputs=" << stats.inputs
                 << " batches=" << stats.batches
                 << " maxBatchItems=" << stats.maxBatchItems
                 << " peakBytes=" << stats.peakBytes
                 << " peakItems=" << stats.peakItems
                 << " maxOutstandingBytes=" << asyncWriterMaxInputBytes()
                 << " coalescedBatches=" << batchWriter_->coalescedBatches
                 << " coalescedInputs=" << batchWriter_->coalescedInputs
                 << " admissionFallbackBatches="
                 << batchWriter_->admissionFallbackBatches;
  }

  if (auto* remoteSink =
          dynamic_cast<CudfEagerFileSink*>(boundedFileSink_.get())) {
    // Complete the S3 upload while failures can still fail the Spark task.
    remoteSink->closeRemote();
    writtenBytes_ = remoteSink->bytes_written();
  } else {
    const auto& parameters = writerInfo_->writerParameters;
    const auto writePath =
        fs::path(cudfWritePath(parameters.writeDirectory())) /
        parameters.writeFileName();
    writtenBytes_ = fs::file_size(writePath);
  }

  // Reset the unique pointers to Cudf writer and options
  writer_.reset();
}

std::shared_ptr<memory::MemoryPool> CudfHiveDataSink::createWriterPool() {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  return connectorPool->addAggregateChild(
      fmt::format("{}.{}", connectorPool->name(), "cudf-file-writer"));
}

void CudfHiveDataSink::makeWriterOptions(
    CudfHiveWriterParameters writerParameters) {
  auto writerPool = createWriterPool();
  auto sinkPool = createSinkPool(writerPool);
  std::shared_ptr<memory::MemoryPool> sortPool{nullptr};
  if (sortWrite()) {
    sortPool = createSortPool(writerPool);
  }

  writerInfo_ = std::make_shared<CudfHiveWriterInfo>(
      std::move(writerParameters),
      std::move(writerPool),
      std::move(sinkPool),
      std::move(sortPool));

  ioStatistics_ = std::make_shared<io::IoStatistics>();

  // Take the writer options provided by the user as a starting point,
  // or allocate a new one.
  auto options = insertTableHandle_->writerOptions();
  if (!options) {
    options = std::make_unique<CudfHiveWriterOptions>();
  }

  const auto* connectorSessionProperties =
      connectorQueryCtx_->sessionProperties();

  if (options->memoryPool == nullptr) {
    options->memoryPool = writerInfo_->writerPool.get();
  }

  if (!options->compressionKind) {
    options->compressionKind = insertTableHandle_->compressionKind();
  }

  const auto& sessionTimeZoneName = connectorQueryCtx_->sessionTimezone();
  if (!sessionTimeZoneName.empty()) {
    options->sessionTimezoneName = sessionTimeZoneName;
  }
  options->adjustTimestampToTimezone =
      connectorQueryCtx_->adjustTimestampToTimezone();
}

folly::dynamic CudfHiveInsertTableHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "CudfHiveInsertTableHandle";
  folly::dynamic arr = folly::dynamic::array;
  for (const auto& ic : inputColumns_) {
    arr.push_back(ic->serialize());
  }

  obj["inputColumns"] = arr;
  obj["locationHandle"] = locationHandle_->serialize();
  obj["tableStorageFormat"] =
      dwio::common::FileFormatName::toName(storageFormat_);

  if (compressionKind_.has_value()) {
    obj["compressionKind"] = common::compressionKindToString(*compressionKind_);
  }
  folly::dynamic serde = folly::dynamic::object;
  for (const auto& [key, value] : serdeParameters_) {
    serde[key] = value;
  }
  obj["serdeParameters"] = std::move(serde);

  return obj;
}

CudfHiveInsertTableHandlePtr CudfHiveInsertTableHandle::create(
    const folly::dynamic& obj) {
  auto inputColumns =
      ISerializable::deserialize<std::vector<CudfHiveColumnHandle>>(
          obj["inputColumns"]);
  auto locationHandle =
      ISerializable::deserialize<LocationHandle>(obj["locationHandle"]);
  std::optional<common::CompressionKind> compressionKind = std::nullopt;
  if (obj.count("compressionKind") > 0) {
    compressionKind =
        common::stringToCompressionKind(obj["compressionKind"].asString());
  }
  std::unordered_map<std::string, std::string> serdeParameters;
  if (obj.count("serdeParameters") > 0) {
    for (const auto& pair : obj["serdeParameters"].items()) {
      serdeParameters.emplace(pair.first.asString(), pair.second.asString());
    }
  }
  const auto storageFormat = obj.count("tableStorageFormat") > 0
      ? dwio::common::toFileFormat(obj["tableStorageFormat"].asString())
      : dwio::common::FileFormat::PARQUET;
  return std::make_shared<CudfHiveInsertTableHandle>(
      inputColumns,
      locationHandle,
      compressionKind,
      serdeParameters,
      nullptr,
      storageFormat);
}

std::string CudfHiveInsertTableHandle::toString() const {
  std::ostringstream out;
  out << "CudfHiveInsertTableHandle ["
      << dwio::common::FileFormatName::toName(storageFormat_);
  if (compressionKind_.has_value()) {
    out << " " << common::compressionKindToString(compressionKind_.value());
  } else {
    out << " none";
  }
  out << "], [inputColumns: [";
  for (const auto& i : inputColumns_) {
    out << " " << i->toString();
  }
  out << " ], locationHandle: " << locationHandle_->toString();

  out << "]";
  return out.str();
}

void CudfHiveInsertTableHandle::registerSerDe() {
  auto& registry = DeserializationRegistryForSharedPtr();
  registry.Register("HiveInsertTableHandle", CudfHiveInsertTableHandle::create);
}

std::string LocationHandle::toString() const {
  return fmt::format(
      "LocationHandle [targetPath: {}, tableType: {},",
      targetPath_,
      tableTypeName(tableType_));
}

folly::dynamic LocationHandle::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "LocationHandle";
  obj["targetPath"] = targetPath_;
  obj["tableType"] = tableTypeName(tableType_);
  return obj;
}

LocationHandlePtr LocationHandle::create(const folly::dynamic& obj) {
  auto targetPath = obj["targetPath"].asString();
  auto tableType = tableTypeFromName(obj["tableType"].asString());
  return std::make_shared<LocationHandle>(targetPath, tableType);
}

} // namespace facebook::velox::cudf_velox::connector::hive
