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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/buffer/Buffer.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/process/StackTrace.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/table/table.hpp>

#include <mutex>
#include <unordered_map>

namespace facebook::velox::cudf_velox {
namespace {

std::mutex& streamOwnerRegistryMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<cudaStream_t, std::shared_ptr<rmm::cuda_stream>>&
streamOwnerRegistry() {
  // Intentionally keep both the registry and registered streams alive for the
  // process lifetime.  Device buffers can outlive the producing operator after
  // publication to asynchronous exchange/consumer queues.  Tying stream
  // destruction to the last visible CudfVector is therefore insufficient:
  // libcudf/RMM may still query the buffer's deallocation stream from a
  // background thread.  A process owns only a small bounded set of these
  // operator output streams, so retaining them is preferable to destroying a
  // stream while an asynchronously published buffer still references it.
  static auto* registry = new std::unordered_map<
      cudaStream_t,
      std::shared_ptr<rmm::cuda_stream>>();
  return *registry;
}

std::shared_ptr<rmm::cuda_stream> retainRegisteredStreamOwner(
    rmm::cuda_stream_view stream) {
  std::lock_guard<std::mutex> lock(streamOwnerRegistryMutex());
  auto& registry = streamOwnerRegistry();
  const auto it = registry.find(stream.value());
  if (it == registry.end()) {
    return nullptr;
  }
  return it->second;
}

/// Calculates the total memory size in bytes of a cudf column and reconstructs
/// it.
///
/// This function disassembles a cudf column to access its underlying memory
/// buffers, calculates the total size including children columns (for nested
/// types), and then reassembles the column.
///
/// @return A pair containing the total size in bytes and the reconstructed
/// column
std::pair<uint64_t, std::unique_ptr<cudf::column>> getColumnSize(
    std::unique_ptr<cudf::column> column) {
  // Store column metadata (type, null count, and size) before releasing it,
  // as the release() operation transfers ownership of the underlying buffers
  // and invalidates access to these properties.
  auto type = column->type();
  auto nullCount = column->null_count();
  auto size = column->size();

  auto contents = column->release();
  auto bytes = contents.data->size() + contents.null_mask->size();

  // Recursively get the size of the children columns.
  std::vector<std::unique_ptr<cudf::column>> children;
  for (auto& child : contents.children) {
    auto [childBytes, childColumn] = getColumnSize(std::move(child));
    bytes += childBytes;
    children.push_back(std::move(childColumn));
  }

  // Reassemble the column with the original metadata.
  auto reconstitutedColumn = std::make_unique<cudf::column>(
      type,
      size,
      std::move(*contents.data.release()),
      std::move(*contents.null_mask.release()),
      nullCount,
      std::move(children));

  return std::make_pair(bytes, std::move(reconstitutedColumn));
}

/// Calculates the total memory size in bytes of a cudf table and reconstructs
/// it.
///
/// This function disassembles a cudf table to access its underlying columns,
/// calculates the total size, and then reassembles the table.
///
/// @note This is a workaround because cudf::table doesn't have an API to get
/// this information without involving estimation and d->h copies.
/// @see https://github.com/rapidsai/cudf/issues/18462
///
/// @return A pair containing the total size in bytes and the reconstructed
/// table
std::pair<uint64_t, std::unique_ptr<cudf::table>> getTableSize(
    std::unique_ptr<cudf::table>&& table) {
  auto columns = table->release();
  std::vector<std::unique_ptr<cudf::column>> columnsOut;
  uint64_t totalBytes = 0;

  for (auto& column : columns) {
    auto [bytes, columnOut] = getColumnSize(std::move(column));
    totalBytes += bytes;
    columnsOut.push_back(std::move(columnOut));
  }
  return std::make_pair(
      totalBytes, std::make_unique<cudf::table>(std::move(columnsOut)));
}

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

void validatePhysicalSchema(const TypePtr& type, cudf::table_view table) {
  const auto rowType = asRowType(type);
  VELOX_CHECK_EQ(
      rowType->size(),
      table.num_columns(),
      "CudfVector physical column count does not match RowType");
  for (size_t i = 0; i < rowType->size(); ++i) {
    const auto kind = rowType->childAt(i)->kind();
    if (kind == TypeKind::ARRAY || kind == TypeKind::MAP ||
        kind == TypeKind::ROW || kind == TypeKind::UNKNOWN) {
      continue;
    }
    const auto expected = veloxToCudfDataType(rowType->childAt(i));
    VELOX_CHECK(
        expected == table.column(i).type(),
        "CudfVector schema mismatch at column {} ({}): Velox {} -> cuDF {}, actual cuDF {}. The producing GPU operator emitted columns in the wrong order.",
        i,
        rowType->nameOf(i),
        rowType->childAt(i)->toString(),
        static_cast<int>(expected.id()),
        static_cast<int>(table.column(i).type().id()));
  }
}

} // namespace

void CudfVector::registerStreamOwner(
    const std::shared_ptr<rmm::cuda_stream>& owner) {
  VELOX_CHECK_NOT_NULL(owner);
  std::lock_guard<std::mutex> lock(streamOwnerRegistryMutex());
  streamOwnerRegistry()[owner->value()] = owner;
}

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
      streamOwner_{retainRegisteredStreamOwner(stream)},
      tableStorage_{std::move(table)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(table)");
  auto& tablePtr = std::get<std::unique_ptr<cudf::table>>(tableStorage_);
  auto [bytes, tableOut] = getTableSize(std::move(tablePtr));
  flatSize_ = bytes;
  tablePtr = std::move(tableOut);
  tabView_ = tablePtr->view();
  validatePhysicalSchema(type_, tabView_);
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
      streamOwner_{retainRegisteredStreamOwner(stream)},
      tableStorage_{std::move(packedTable)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(packed_table)");
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  tabView_ = packedPtr->table;
  validatePhysicalSchema(type_, tabView_);
  // For packed table, flatSize is the size of the GPU data buffer
  flatSize_ = packedPtr->data.gpu_data->size();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    cudf::table_view tableView,
    rmm::cuda_stream_view stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      streamOwner_{retainRegisteredStreamOwner(stream)},
      tableStorage_{std::unique_ptr<cudf::table>()},
      tabView_{tableView},
      stream_{stream} {
  validatePhysicalSchema(type_, tabView_);
  flatSize_ = 0;
}

std::unique_ptr<cudf::table> CudfVector::release() {
  flatSize_ = 0;
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    if (*tablePtr) {
      // Constructed from owned table - just move it out.
      return std::move(*tablePtr);
    }
    // A non-owning view materializes only when a consuming API explicitly
    // requests ownership.
    auto materializedTable =
        std::make_unique<cudf::table>(tabView_, stream_, get_temp_mr());
    stream_.synchronize();
    return materializedTable;
  }
  // Constructed from packed_table - materialize a table from the view.
  // This copies the data since the view references the packed buffer.
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  auto mr = packedPtr->data.gpu_data->memory_resource();
  packedPtr->data.gpu_data->set_stream(stream_);
  auto materializedTable = std::make_unique<cudf::table>(tabView_, stream_, mr);
  stream_.synchronize();
  packedPtr.reset();
  return materializedTable;
}

std::unique_ptr<cudf::packed_table> CudfVector::releasePacked() {
  auto* packedPtr =
      std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_);
  if (packedPtr == nullptr || !*packedPtr) {
    return nullptr;
  }
  flatSize_ = 0;
  return std::move(*packedPtr);
}

bool CudfVector::rebindStream(rmm::cuda_stream_view stream) {
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    if (!*tablePtr) {
      return false;
    }

    if (stream_.value() == stream.value()) {
      return true;
    }

    auto columns = (*tablePtr)->release();
    for (auto& column : columns) {
      column = cudf::rebind_stream(std::move(*column), stream);
    }

    *tablePtr = std::make_unique<cudf::table>(std::move(columns));
    tabView_ = (*tablePtr)->view();
    stream_ = stream;
    streamOwner_ = retainRegisteredStreamOwner(stream);
    return true;
  }

  if (auto* packedPtr =
          std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_)) {
    if (!*packedPtr) {
      return false;
    }

    (*packedPtr)->data.gpu_data->set_stream(stream);
    stream_ = stream;
    streamOwner_ = retainRegisteredStreamOwner(stream);
    return true;
  }

  return false;
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

} // namespace facebook::velox::cudf_velox
