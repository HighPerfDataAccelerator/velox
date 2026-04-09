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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/process/StackTrace.h"
#include "velox/experimental/cudf/exec/SyncWait.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {
namespace {

void logDefaultStreamIfNeeded(
    rmm::cuda_stream_view stream,
    const char* constructorName) {
  if (stream.value() != rmm::cuda_stream_default.value()) {
    return;
  }
  LOG(WARNING) << constructorName
               << " constructed with default CUDA stream. Backtrace:\n"
               << process::StackTrace().toString();
}

} // namespace

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::table>&& table,
    rmm::cuda_stream_view stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(table)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(table)");
  auto& tablePtr = std::get<std::unique_ptr<cudf::table>>(tableStorage_);
  flatSize_ = estimateTableBytes(tablePtr);
  tabView_ = tablePtr->view();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::packed_table>&& packedTable,
    rmm::cuda_stream_view stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(packedTable)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(packed_table)");
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  tabView_ = packedPtr->table;
  // For packed table, flatSize is the size of the GPU data buffer
  flatSize_ = packedPtr->data.gpu_data->size();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    cudf::table_view view,
    std::shared_ptr<cudf::table> sourceRef,
    rmm::cuda_stream_view stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(sourceRef)},
      tabView_{view},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(borrowed)");
  flatSize_ = 0; // borrowed, don't count towards memory
}

std::unique_ptr<cudf::table> CudfVector::release() {
  flatSize_ = 0;
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    // Constructed from owned table - just move it out
    return std::move(*tablePtr);
  }
  // Constructed from packed_table or borrowed — materialize a table
  // from the view. This copies the data.
  auto mr = cudf::get_current_device_resource_ref();
  auto materializedTable = std::make_unique<cudf::table>(tabView_, stream_, mr);
  synchronizeStreamAndRecord(stream_);
  // Clear the source storage
  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_)) {
    packedPtr->reset();
  } else if (auto* sharedPtr =
                 std::get_if<std::shared_ptr<cudf::table>>(&tableStorage_)) {
    sharedPtr->reset();
  }
  return materializedTable;
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

} // namespace facebook::velox::cudf_velox
