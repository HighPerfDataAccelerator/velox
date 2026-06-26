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
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergDataSource.h"
#include "velox/experimental/cudf/connectors/hive/iceberg/CudfIcebergSplitReader.h"

#include "velox/common/Casts.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/connectors/hive/iceberg/IcebergSplit.h"

namespace facebook::velox::cudf_velox::connector::hive::iceberg {

namespace velox_connector = ::facebook::velox::connector;
namespace velox_hive = ::facebook::velox::connector::hive;
namespace velox_iceberg = ::facebook::velox::connector::hive::iceberg;

namespace {

std::string cleanPathForCudf(std::string path) {
  constexpr std::string_view kFilePrefix = "file:";
  constexpr std::string_view kS3APrefix = "s3a:";
  if (path.compare(0, kFilePrefix.size(), kFilePrefix) == 0) {
    path = path.substr(kFilePrefix.size());
  } else if (path.compare(0, kS3APrefix.size(), kS3APrefix) == 0) {
    // KvikIO does not support the "s3a:" prefix.
    path.erase(kS3APrefix.size() - 2, 1);
  }
  return path;
}

std::vector<velox_iceberg::IcebergDeleteFile> copyDeleteFiles(
    const std::vector<velox_iceberg::IcebergDeleteFile>& deleteFiles) {
  std::vector<velox_iceberg::IcebergDeleteFile> copy;
  copy.reserve(deleteFiles.size());
  for (const auto& deleteFile : deleteFiles) {
    copy.emplace_back(deleteFile);
  }
  return copy;
}

std::optional<uint64_t> getFileSize(
    const velox_iceberg::HiveIcebergSplit& split,
    FileHandleFactory* fileHandleFactory,
    const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
    IoStats* ioStats) {
  if (split.properties && split.properties->fileSize.has_value()) {
    return static_cast<uint64_t>(*split.properties->fileSize);
  }

  const auto fileHandleKey = FileHandleKey{
      .filename = cleanPathForCudf(split.filePath),
      .tokenProvider = connectorQueryCtx->fsTokenProvider()};
  auto fileProperties = FileProperties{};
  auto fileHandleCachePtr =
      fileHandleFactory->generate(fileHandleKey, &fileProperties, ioStats);
  VELOX_CHECK_NOT_NULL(fileHandleCachePtr.get());
  VELOX_CHECK_NOT_NULL(fileHandleCachePtr->file.get());
  return fileHandleCachePtr->file->size();
}

} // namespace

CudfIcebergDataSource::CudfIcebergDataSource(
    const RowTypePtr& outputType,
    const velox_connector::ConnectorTableHandlePtr& tableHandle,
    const velox_connector::ColumnHandleMap& columnHandles,
    FileHandleFactory* fileHandleFactory,
    folly::Executor* executor,
    const velox_connector::ConnectorQueryCtx* connectorQueryCtx,
    const std::shared_ptr<CudfHiveConfig>& cudfHiveConfig,
    const std::shared_ptr<const velox_hive::HiveConfig>& hiveConfig)
    : CudfHiveDataSource(
          outputType,
          tableHandle,
          columnHandles,
          fileHandleFactory,
          executor,
          connectorQueryCtx,
          cudfHiveConfig),
      hiveConfig_(hiveConfig) {}

void CudfIcebergDataSource::convertSplit(
    std::shared_ptr<velox_connector::ConnectorSplit> split) {
  // Convert `ConnectorSplit` to `HiveIcebergSplit`
  icebergSplit_ =
      checkedPointerCast<const velox_iceberg::HiveIcebergSplit>(split);

  if (icebergSplit_->start != 0) {
    const auto fileSize = getFileSize(
        *icebergSplit_,
        fileHandleFactory_,
        connectorQueryCtx_,
        ioStats_ ? ioStats_.get() : nullptr);
    const auto splitEnd =
        icebergSplit_->length == std::numeric_limits<uint64_t>::max()
        ? *fileSize
        : icebergSplit_->start + icebergSplit_->length;
    const auto isWholeParquetSplit = icebergSplit_->fileFormat ==
            dwio::common::FileFormat::PARQUET &&
        icebergSplit_->start == 4 && splitEnd == *fileSize;
    VELOX_CHECK(
        isWholeParquetSplit,
        "Sub-splits are not yet supported in CudfIcebergDataSource. "
        "Split start: {}, length: {}, file size: {}",
        icebergSplit_->start,
        icebergSplit_->length,
        *fileSize);

    auto normalizedSplit = std::make_shared<velox_iceberg::HiveIcebergSplit>(
        icebergSplit_->connectorId,
        icebergSplit_->filePath,
        icebergSplit_->fileFormat,
        0,
        *fileSize,
        icebergSplit_->partitionKeys,
        icebergSplit_->tableBucketNumber,
        icebergSplit_->customSplitInfo,
        icebergSplit_->extraFileInfo,
        icebergSplit_->cacheable,
        copyDeleteFiles(icebergSplit_->deleteFiles),
        icebergSplit_->infoColumns,
        icebergSplit_->properties,
        icebergSplit_->dataSequenceNumber);
    icebergSplit_ = normalizedSplit;
    split = normalizedSplit;
  }

  // Convert `ConnectorSplit` to `CudfHiveConnectorSplit`
  CudfHiveDataSource::convertSplit(split);
}

std::unique_ptr<CudfSplitReader>
CudfIcebergDataSource::createCudfSplitReader() {
  return std::make_unique<CudfIcebergSplitReader>(
      split_,
      icebergSplit_,
      tableHandle_,
      outputType_,
      readColumnNames_,
      fileHandleFactory_,
      executor_,
      connectorQueryCtx_,
      cudfHiveConfig_,
      hiveConfig_,
      ioStatistics_,
      ioStats_,
      useExperimentalCudfReader_,
      subfieldFilterExpr_);
}

} // namespace facebook::velox::cudf_velox::connector::hive::iceberg
