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

#include "velox/exec/ExchangeSource.h"

namespace facebook::velox::cudf_velox {

/// Factory function that creates a LocalGpuExchangeSource for task IDs
/// starting with "gpu-local://". Returns nullptr if the task ID does not
/// match the prefix, allowing the ExchangeSource factory chain to try
/// other factories.
///
/// The LocalGpuExchangeSource is identical in behavior to the CPU
/// LocalExchangeSource (velox/exec/tests/utils/LocalExchangeSource.cpp)
/// but registered under the "gpu-local://" prefix. It fetches pages from
/// the local OutputBufferManager and enqueues them into the ExchangeQueue.
/// The pages flowing through are GpuSerializedPages, but the source itself
/// is agnostic to the page type -- it handles SerializedPageBase*.
std::shared_ptr<exec::ExchangeSource> createLocalGpuExchangeSource(
    const std::string& taskId,
    int destination,
    std::shared_ptr<exec::ExchangeQueue> queue,
    memory::MemoryPool* pool);

/// Sets the local GPU exchange source to start by clearing 'stop_'. This is
/// used when we run multiple test cases sequentially and restarts the local
/// GPU exchange source between tests.
void testingStartLocalGpuExchangeSource();

/// Ensures that there are no references to ExchangeSource callbacks,
/// e.g. while waiting for timing out. Call this before end of unit
/// tests to ensure no ASAN errors at exit.
void testingShutdownLocalGpuExchangeSource();

} // namespace facebook::velox::cudf_velox
