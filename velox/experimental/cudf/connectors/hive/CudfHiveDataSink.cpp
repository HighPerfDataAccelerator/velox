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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveDataSink.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/Counters.h"
#include "velox/common/base/Fs.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/dwio/common/Options.h"
#include "velox/exec/OperatorUtils.h"

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <cstdlib>

using facebook::velox::common::testutil::TestValue;

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

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

int parallelDirectWriteLanes() {
  const auto* raw = std::getenv("GLUTEN_CUDF_PARALLEL_DIRECT_WRITE_LANES");
  if (raw == nullptr || *raw == '\0') {
    return 1;
  }
  char* end = nullptr;
  const auto requested = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || requested < 1) {
    LOG(WARNING) << "CudfHiveDataSink: ignoring invalid "
                 << "GLUTEN_CUDF_PARALLEL_DIRECT_WRITE_LANES='" << raw << "'";
    return 1;
  }
  return static_cast<int>(std::clamp<long>(requested, 1, 4));
}

std::string addWriterLaneSuffix(std::string fileName, int driverId) {
  const auto suffix = fmt::format("-flux-lane-{:02d}", driverId);
  const auto extension = fileName.rfind('.');
  if (extension == std::string::npos) {
    return fileName + suffix;
  }
  fileName.insert(extension, suffix);
  return fileName;
}

bool destinationWriteLanesEnabled(const config::ConfigBase* sessions) {
  return sessions != nullptr &&
      sessions->get<bool>(
          "spark.gluten.sql.columnar.backend.velox.flux."
          "keyedFinalDestinationLanes",
          false);
}

bool hasFluxTaskReplicaSuffix(const std::string& taskId) {
  const auto marker = taskId.rfind("-p");
  if (marker == std::string::npos || marker + 2 == taskId.size()) {
    return false;
  }
  return std::all_of(
      taskId.begin() + marker + 2, taskId.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
      });
}

std::string addFluxTaskReplicaSuffix(
    std::string fileName,
    const std::string& taskId) {
  // Flux task IDs end in "-p<replica>".  Different destination-owner tasks
  // share Spark's one injected target filename, so include that replica in
  // the native filename.  Peer identity is already unique at the Spark task
  // (and attempt-directory) level.
  const auto marker = taskId.rfind("-p");
  VELOX_CHECK_NE(
      marker,
      std::string::npos,
      "Destination-lane TableWrite task has no Flux replica suffix: {}",
      taskId);
  const auto replica = taskId.substr(marker + 2);
  VELOX_CHECK(
      !replica.empty() &&
          std::all_of(
              replica.begin(),
              replica.end(),
              [](unsigned char c) { return std::isdigit(c) != 0; }),
      "Invalid Flux task replica suffix in task ID: {}",
      taskId);
  const auto suffix = fmt::format("-flux-task-p{}", replica);
  const auto extension = fileName.rfind('.');
  if (extension == std::string::npos) {
    return fileName + suffix;
  }
  fileName.insert(extension, suffix);
  return fileName;
}

std::string localWriteDirectory(std::string path) {
  if (path.rfind("file://", 0) == 0) {
    path.erase(0, 7);
  } else if (path.rfind("file:", 0) == 0) {
    path.erase(0, 5);
  } else if (path.rfind("fuse://", 0) == 0) {
    path.erase(0, 7);
  } else if (path.rfind("fuse:", 0) == 0) {
    path.erase(0, 5);
  }
  return path;
}

bool isLocalWriteDirectory(const std::string& path) {
  return path.find("://") == std::string::npos ||
      path.rfind("file://", 0) == 0 || path.rfind("file:", 0) == 0 ||
      path.rfind("fuse://", 0) == 0 || path.rfind("fuse:", 0) == 0;
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
  checkRunning();

  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  auto stream =
      cudfVector ? cudfVector->stream() : cudfGlobalStreamPool().get_stream();
  std::unique_ptr<cudf::table> convertedInput;
  cudf::table_view inputView;
  if (cudfVector) {
    // CudfVector already owns a device-resident table. Keep the write path on
    // device. The chunked writer is stream ordered, so synchronizing the host
    // here only inserts a producer/writer bubble. Converting through Arrow
    // would materialize on host and defeats the purpose of the cuDF connector.
    inputView = cudfVector->getTableView();
  } else {
    convertedInput =
        with_arrow::toCudfTable(input, input->pool(), stream, get_temp_mr());
    VELOX_CHECK_NOT_NULL(
        convertedInput, "Failed to convert input RowVectorPtr to cudf::table");
    inputView = convertedInput->view();
  }

  // Check if the writer doesn't already exist
  if (writer_ == nullptr) {
    writerStream_ = stream;
    writer_ = createCudfWriter(inputView, stream);
  } else if (stream.value() != writerStream_->value()) {
    // Operators normally keep a driver lane on one CUDA stream. Preserve
    // correctness for a cross-stream input without blocking the CPU: make the
    // writer stream wait for the input producer on the device.
    const std::vector<rmm::cuda_stream_view> inputStreams{stream};
    cudf::detail::join_streams(inputStreams, *writerStream_);
  }

  // Write the table to the sink
  writer_->write(inputView);
  writerInfo_->inputSizeInBytes += input->estimateFlatSize();
  writerInfo_->numWrittenRows += input->size();
}

std::unique_ptr<cudf::io::chunked_parquet_writer>
CudfHiveDataSink::createCudfWriter(
    cudf::table_view cudfTable,
    rmm::cuda_stream_view stream) {
  // Create a table_input_metadata from the input
  auto tableInputMetadata = createCudfTableInputMetadata(cudfTable);

  auto compressionKind =
      getCompressionType(insertTableHandle_->compressionKind().value_or(
          facebook::velox::common::CompressionKind::CompressionKind_NONE));

  // Create a sink and writer
  const auto& locationHandle = insertTableHandle_->locationHandle();
  auto targetFileName = locationHandle->targetFileName().empty()
      ? fmt::format("{}{}", makeUuid(), ".parquet")
      : locationHandle->targetFileName();
  const auto& taskId = connectorQueryCtx_->taskId();
  // Connector session properties are intentionally narrower than the task's
  // QueryConfig and may omit the Flux destination-lane switch.  The native
  // task ID is the authoritative replica identity, so use it whenever it is
  // present.  This also makes the single-replica p0 case harmlessly unique.
  if (destinationWriteLanesEnabled(connectorQueryCtx_->sessionProperties()) ||
      hasFluxTaskReplicaSuffix(taskId)) {
    targetFileName =
        addFluxTaskReplicaSuffix(std::move(targetFileName), taskId);
    LOG(WARNING) << "CudfHiveDataSink: destination-owner task " << taskId
                 << " -> " << targetFileName;
  }
  if (parallelDirectWriteLanes() > 1) {
    targetFileName = addWriterLaneSuffix(
        std::move(targetFileName), connectorQueryCtx_->driverId());
    LOG(WARNING) << "CudfHiveDataSink: parallel direct-write driver "
                 << connectorQueryCtx_->driverId() << " -> " << targetFileName;
  }

  auto writerParameters = CudfHiveWriterParameters(
      CudfHiveWriterParameters::UpdateMode::kNew,
      targetFileName,
      locationHandle->targetPath());

  auto sinkDirectory = writerParameters.writeDirectory();
  if (isLocalWriteDirectory(sinkDirectory)) {
    sinkDirectory = localWriteDirectory(std::move(sinkDirectory));
    fs::create_directories(sinkDirectory);
  }

  makeWriterOptions(writerParameters);

  // Create writer options for the given sink
  const auto sinkInfo =
      cudf::io::sink_info(fmt::format("{}/{}", sinkDirectory, targetFileName));
  LOG(WARNING) << "CudfHiveDataSink: opening output " << sinkDirectory << "/"
               << targetFileName;
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
    if (sortingColumns_.empty()) {
      cudfWriterOptions.set_sorting_columns(sortingColumns_);
    }
  }

  return std::make_unique<cudf::io::chunked_parquet_writer>(
      cudfWriterOptions, stream);
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

  numWrittenBytes += ioStatistics_->rawBytesWritten();
  writeIOTimeUs += ioStatistics_->writeIOTimeUs();

  stats.numWrittenBytes = numWrittenBytes;
  stats.writeIOTimeUs = writeIOTimeUs;

  if (state_ != State::kClosed) {
    return stats;
  }

  stats.numWrittenFiles = 1;
  VELOX_CHECK_NOT_NULL(writerInfo_);
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
  VELOX_CHECK_NOT_NULL(writer_, "CudfHiveDataSink has no writer");

  setState(State::kFinishing);
  return true;
}

std::vector<std::string> CudfHiveDataSink::close() {
  setState(State::kClosed);
  closeInternal();

  std::vector<std::string> partitionUpdates{};

  partitionUpdates.reserve(1);
  VELOX_CHECK_NOT_NULL(writerInfo_);
  // clang-format off
    auto partitionUpdateJson = folly::toJson(
     folly::dynamic::object
        ("name", "")
        ("updateMode", CudfHiveWriterParameters::updateModeToString(
          writerInfo_->writerParameters.updateMode()))
        ("writePath", writerInfo_->writerParameters.writeDirectory())
        ("targetPath", writerInfo_->writerParameters.targetDirectory())
        ("fileWriteInfos", folly::dynamic::array(
          folly::dynamic::object
            ("writeFileName", writerInfo_->writerParameters.writeFileName())
            ("targetFileName", writerInfo_->writerParameters.targetFileName())
            ("fileSize", ioStatistics_->rawBytesWritten())))
        ("rowCount", writerInfo_->numWrittenRows)
        ("inMemoryDataSizeInBytes", writerInfo_->inputSizeInBytes)
        ("onDiskDataSizeInBytes", ioStatistics_->rawBytesWritten())
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
  if (writer_ == nullptr) {
    VELOX_CHECK_EQ(state_, State::kAborted);
    return;
  }

  TestValue::adjust(
      "facebook::velox::connector::hive::CudfHiveDataSink::closeInternal",
      this);

  // Close cudf writer
  writer_->close();

  // Reset the unique pointers to Cudf writer and options
  writer_.reset();
  writerStream_.reset();
}

std::shared_ptr<memory::MemoryPool> CudfHiveDataSink::createWriterPool() {
  auto* connectorPool = connectorQueryCtx_->connectorMemoryPool();
  return connectorPool->addAggregateChild(
      fmt::format("{}.{}", connectorPool->name(), "parquet-writer"));
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
  for (const auto& pair : obj["serdeParameters"].items()) {
    serdeParameters.emplace(pair.first.asString(), pair.second.asString());
  }
  return std::make_shared<CudfHiveInsertTableHandle>(
      inputColumns, locationHandle, compressionKind, serdeParameters);
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
