/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#pragma once

#include "velox/common/file/File.h"
#include "velox/experimental/cudf/connectors/hive/ExecutorReadBroker.h"
#include "velox/experimental/cudf/connectors/hive/PinnedHostBuffer.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

std::shared_ptr<PinnedHostBuffer> selectiveParquetRead(
    const std::string& filePath,
    uint64_t fileSize,
    PrefetchReadFunction readFunction,
    const std::vector<std::string>& readColumnNames,
    uint64_t splitStart = 0,
    uint64_t splitLength = std::numeric_limits<uint64_t>::max(),
    std::shared_ptr<ExecutorReadBroker> broker = nullptr,
    IoStats* ioStats = nullptr);

} // namespace facebook::velox::cudf_velox::connector::hive
