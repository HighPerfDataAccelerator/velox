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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

bool hasEmptyStringChars(const cudf::column_view& column) {
  if (column.type().id() == cudf::type_id::STRING) {
    // A zero-row column is harmless because concatenate skips it. In current
    // cuDF the chars buffer is the STRING column's own data, not a
    // second child. The sole child is offsets. Checking child(1) therefore
    // classified every STRING as empty and made this safety grouping a no-op.
    return column.size() > 0 && column.head<char>() == nullptr;
  }
  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    if (hasEmptyStringChars(column.child(i))) {
      return true;
    }
  }
  return false;
}

bool hasSameEmptyStringCharsPattern(
    const cudf::column_view& left,
    const cudf::column_view& right) {
  if (left.type().id() == cudf::type_id::STRING ||
      right.type().id() == cudf::type_id::STRING) {
    if (left.type().id() != right.type().id()) {
      return false;
    }
    return hasEmptyStringChars(left) == hasEmptyStringChars(right);
  }
  if (left.num_children() != right.num_children()) {
    return false;
  }
  for (cudf::size_type i = 0; i < left.num_children(); ++i) {
    if (!hasSameEmptyStringCharsPattern(left.child(i), right.child(i))) {
      return false;
    }
  }
  return true;
}

void validateVariableWidthColumnLayout(
    const cudf::column_view& column,
    std::string_view path,
    rmm::cuda_stream_view stream,
    std::ostringstream& layout) {
  const auto type = column.type().id();
  layout << " " << path << "{type=" << static_cast<int>(type)
         << ",size=" << column.size() << ",offset=" << column.offset();
  if ((type == cudf::type_id::STRING || type == cudf::type_id::LIST) &&
      column.size() > 0) {
    VELOX_CHECK_GE(column.num_children(), 1);
    const auto offsets = column.child(0);
    const auto beginIndex = column.offset();
    const auto endIndex = column.offset() + column.size();
    VELOX_CHECK_GE(beginIndex, 0, "Negative parent offset at {}", path);
    VELOX_CHECK_LT(
        endIndex,
        offsets.size() + offsets.offset(),
        "Offsets child is too short at {}",
        path);
    cudf::size_type edgeOffsets[2]{};
    const auto* offsetsBase = offsets.head<cudf::size_type>();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        edgeOffsets,
        offsetsBase + beginIndex,
        sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        edgeOffsets + 1,
        offsetsBase + endIndex,
        sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    const bool isString = type == cudf::type_id::STRING;
    VELOX_CHECK(isString || column.num_children() >= 2);
    const auto payloadBegin =
        isString ? edgeOffsets[0] : column.child(1).offset();
    const auto payloadEnd = isString
        ? edgeOffsets[1]
        : column.child(1).offset() + column.child(1).size();
    layout << ",edgeOffsets=[" << edgeOffsets[0] << "," << edgeOffsets[1]
           << "],payloadRange=[" << payloadBegin << "," << payloadEnd << "]";
    VELOX_CHECK_GE(edgeOffsets[0], 0, "Negative first offset at {}", path);
    VELOX_CHECK_GE(
        edgeOffsets[1],
        edgeOffsets[0],
        "Non-monotonic offsets at {}: first={}, last={}",
        path,
        edgeOffsets[0],
        edgeOffsets[1]);
    VELOX_CHECK_GE(edgeOffsets[0], payloadBegin);
    VELOX_CHECK_LE(
        edgeOffsets[1],
        payloadEnd,
        "Offset exceeds payload at {}: last={}, payloadEnd={}",
        path,
        edgeOffsets[1],
        payloadEnd);
  }
  layout << "}";
  for (cudf::size_type i = 0; i < column.num_children(); ++i) {
    validateVariableWidthColumnLayout(
        column.child(i), fmt::format("{}.{}", path, i), stream, layout);
  }
}

int getNumCudaDevices() {
  int numDevices{};
  CUDF_CUDA_TRY(cudaGetDeviceCount(&numDevices));
  return numDevices;
}

int getCurrentCudaDevice() {
  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  return device;
}

CudaEvent& eventForThread() {
  // Intentionally leak per-thread, per-device events to avoid CUDA calls from
  // thread-local destructors after CUDA context teardown.
  thread_local static std::vector<CudaEvent*> events(getNumCudaDevices());
  auto const device = getCurrentCudaDevice();
  VELOX_CHECK_GE(device, 0);
  auto const deviceIndex = static_cast<size_t>(device);
  VELOX_CHECK_LT(deviceIndex, events.size());

  if (events[deviceIndex] == nullptr) {
    events[deviceIndex] = new CudaEvent(cudaEventDisableTiming);
  }
  return *events[deviceIndex];
}

size_t maxBatchRows() {
  const auto& cudfConfig = CudfConfig::getInstance();
  if (cudfConfig.batchSizeMaxThreshold) {
    VELOX_CHECK_GT(
        cudfConfig.batchSizeMaxThreshold.value(),
        0,
        "cuDF max batch size must be positive");
    return static_cast<size_t>(cudfConfig.batchSizeMaxThreshold.value());
  }
  return static_cast<size_t>(std::numeric_limits<cudf::size_type>::max());
}

vector_size_t checkedVectorSize(size_t rowCount) {
  VELOX_CHECK_LE(
      rowCount,
      static_cast<size_t>(std::numeric_limits<vector_size_t>::max()),
      "cuDF vector row count exceeds Velox vector size limit");
  return static_cast<vector_size_t>(rowCount);
}
} // namespace

std::string validateVariableWidthTableLayout(
    const cudf::table_view& table,
    rmm::cuda_stream_view stream) {
  std::ostringstream layout;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    validateVariableWidthColumnLayout(
        table.column(i), fmt::format("c{}", i), stream, layout);
  }
  return layout.str();
}

bool hasEmptyStringChars(const cudf::table_view& table) {
  for (const auto& column : table) {
    if (hasEmptyStringChars(column)) {
      return true;
    }
  }
  return false;
}

bool hasSameEmptyStringCharsPattern(
    const cudf::table_view& left,
    const cudf::table_view& right) {
  if (left.num_columns() != right.num_columns()) {
    return false;
  }
  for (cudf::size_type i = 0; i < left.num_columns(); ++i) {
    if (!hasSameEmptyStringCharsPattern(left.column(i), right.column(i))) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<cudf::table> concatenateTables(
    std::vector<std::unique_ptr<cudf::table>> tables,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  VELOX_CHECK_GT(tables.size(), 0);

  if (tables.size() == 1) {
    return std::move(tables[0]);
  }
  std::vector<cudf::table_view> tableViews;
  tableViews.reserve(tables.size());
  std::transform(
      tables.begin(),
      tables.end(),
      std::back_inserter(tableViews),
      [&](const auto& tbl) { return tbl->view(); });
  auto result = cudf::concatenate(tableViews, stream, mr);
  // cudf::concatenate is asynchronous, while this helper owns and destroys
  // every source table on return.  Hash-join output can therefore recycle a
  // source buffer before concatenate has finished reading it.  Complete the
  // producing stream before releasing those sources; the returned result may
  // safely continue on the same stream in downstream operators.
  stream.synchronize();
  return result;
}

std::unique_ptr<cudf::column> makeEmptyColumnForType(
    const TypePtr& type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto zeroOffsets = [&]() {
    cudf::size_type zero = 0;
    rmm::device_buffer offsets(&zero, sizeof(zero), stream, mr);
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        1,
        std::move(offsets),
        rmm::device_buffer{},
        0);
  };
  switch (type->kind()) {
    case TypeKind::UNKNOWN:
      // Velox uses UNKNOWN for an untyped NULL literal. cuDF has no physical
      // UNKNOWN type, but an empty column carries no values whose type could
      // be observed. Keep the same INT8 placeholder used by the row
      // constructor for all-null UNKNOWN fields. This is required when an
      // empty hash-join build side contains such a field and must synthesize
      // its physical schema in noMoreInput().
      return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT8});
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return cudf::make_strings_column(
          0, zeroOffsets(), rmm::device_buffer{}, 0, rmm::device_buffer{});
    case TypeKind::ARRAY:
      return cudf::make_lists_column(
          0,
          zeroOffsets(),
          makeEmptyColumnForType(type->childAt(0), stream, mr),
          0,
          rmm::device_buffer{});
    case TypeKind::MAP: {
      std::vector<std::unique_ptr<cudf::column>> entries;
      entries.push_back(makeEmptyColumnForType(type->childAt(0), stream, mr));
      entries.push_back(makeEmptyColumnForType(type->childAt(1), stream, mr));
      auto entryStruct = cudf::make_structs_column(
          0, std::move(entries), 0, rmm::device_buffer{}, stream, mr);
      return cudf::make_lists_column(
          0, zeroOffsets(), std::move(entryStruct), 0, rmm::device_buffer{});
    }
    case TypeKind::ROW: {
      std::vector<std::unique_ptr<cudf::column>> children;
      children.reserve(type->size());
      for (size_t i = 0; i < type->size(); ++i) {
        children.push_back(
            makeEmptyColumnForType(type->childAt(i), stream, mr));
      }
      return cudf::make_structs_column(
          0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
    }
    default:
      return cudf::make_empty_column(cudf_velox::veloxToCudfDataType(type));
  }
}

std::unique_ptr<cudf::column> makeAllNullColumnForType(
    const TypePtr& type,
    cudf::size_type size,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_GE(size, 0);
  auto zeroOffsets = [&]() {
    std::vector<cudf::size_type> offsets(static_cast<size_t>(size) + 1, 0);
    rmm::device_buffer offsetsBuffer(
        offsets.data(), offsets.size() * sizeof(cudf::size_type), stream, mr);
    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(offsets.size()),
        std::move(offsetsBuffer),
        rmm::device_buffer{},
        0);
  };
  auto allNullMask = [&]() {
    return cudf::create_null_mask(size, cudf::mask_state::ALL_NULL, stream, mr);
  };

  switch (type->kind()) {
    case TypeKind::UNKNOWN:
      return cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_id::INT8},
          size,
          cudf::mask_state::ALL_NULL,
          stream,
          mr);
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return cudf::make_strings_column(
          size, zeroOffsets(), rmm::device_buffer{}, size, allNullMask());
    case TypeKind::ARRAY:
      return cudf::make_lists_column(
          size,
          zeroOffsets(),
          makeEmptyColumnForType(type->childAt(0), stream, mr),
          size,
          allNullMask());
    case TypeKind::MAP: {
      std::vector<std::unique_ptr<cudf::column>> entries;
      entries.push_back(makeEmptyColumnForType(type->childAt(0), stream, mr));
      entries.push_back(makeEmptyColumnForType(type->childAt(1), stream, mr));
      auto entryStruct = cudf::make_structs_column(
          0, std::move(entries), 0, rmm::device_buffer{}, stream, mr);
      return cudf::make_lists_column(
          size, zeroOffsets(), std::move(entryStruct), size, allNullMask());
    }
    case TypeKind::ROW: {
      std::vector<std::unique_ptr<cudf::column>> children;
      children.reserve(type->size());
      for (size_t i = 0; i < type->size(); ++i) {
        children.push_back(
            makeAllNullColumnForType(type->childAt(i), size, stream, mr));
      }
      return cudf::make_structs_column(
          size, std::move(children), size, allNullMask(), stream, mr);
    }
    default:
      return cudf::make_fixed_width_column(
          veloxToCudfDataType(type),
          size,
          cudf::mask_state::ALL_NULL,
          stream,
          mr);
  }
}

std::unique_ptr<cudf::table> makeEmptyTable(
    TypePtr const& inputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::column>> emptyColumns;
  for (size_t i = 0; i < inputType->size(); ++i) {
    emptyColumns.push_back(
        makeEmptyColumnForType(inputType->childAt(i), stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(emptyColumns));
}

std::unique_ptr<cudf::table> getConcatenatedTable(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Check for empty vector
  if (tables.size() == 0) {
    return makeEmptyTable(tableType, stream, mr);
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  try {
    // Even for a single input table we must concatenate (copy) rather than
    // release in-place: the output is owned by `stream` but the input buffer
    // was allocated on a different stream, so releasing it would bind
    // deallocation to the wrong stream.
    auto output = cudf::concatenate(tableViews, stream, mr);

    orderCudfVectorDeallocationsAfterStream(tables, inputStreams, stream);
    // Input tables are deallocated here when 'tables' goes out of scope.
    return output;
  } catch (...) {
    // concatenate can enqueue work for early columns before a later allocation
    // fails. Finish that work before owners unwind on their original streams.
    stream.synchronize();
    throw;
  }
}

std::vector<std::unique_ptr<cudf::table>> getConcatenatedTableBatched(
    std::vector<CudfVectorPtr>&& tables,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::optional<size_t> maxRowsOverride) {
  std::vector<std::unique_ptr<cudf::table>> concatTables;
  // Check for empty vector
  if (tables.size() == 0) {
    concatTables.push_back(makeEmptyTable(tableType, stream, mr));
    return concatTables;
  }

  auto inputStreams = std::vector<rmm::cuda_stream_view>();
  auto tableViews = std::vector<cudf::table_view>();

  inputStreams.reserve(tables.size());
  tableViews.reserve(tables.size());

  for (const auto& table : tables) {
    VELOX_CHECK_NOT_NULL(table);
    tableViews.push_back(table->getTableView());
    inputStreams.push_back(table->stream());
  }

  cudf::detail::join_streams(inputStreams, stream);

  try {
    std::vector<std::unique_ptr<cudf::table>> outputTables;
    const auto maxRows = maxRowsOverride.value_or(maxBatchRows());
    VELOX_CHECK_GT(maxRows, 0, "cuDF max batch size must be positive");
    std::vector<cudf::table_view> boundedViews;
    std::vector<uint64_t> boundedEstimatedBytes;
    boundedViews.reserve(tableViews.size());
    boundedEstimatedBytes.reserve(tableViews.size());
    for (size_t tableIndex = 0; tableIndex < tableViews.size(); ++tableIndex) {
      const auto& tableView = tableViews[tableIndex];
      const auto numRows = static_cast<size_t>(tableView.num_rows());
      const auto estimatedBytes =
          static_cast<uint64_t>(tables[tableIndex]->estimateFlatSize());
      if (numRows <= maxRows) {
        boundedViews.push_back(tableView);
        boundedEstimatedBytes.push_back(estimatedBytes);
        continue;
      }
      for (size_t start = 0; start < numRows;) {
        const auto end = start + std::min(maxRows, numRows - start);
        auto slices = cudf::slice(
            tableView,
            {static_cast<cudf::size_type>(start),
             static_cast<cudf::size_type>(end)},
            stream);
        VELOX_CHECK_EQ(slices.size(), 1);
        boundedViews.push_back(slices.front());
        const auto sliceRows = static_cast<uint64_t>(end - start);
        const auto rowCount = static_cast<uint64_t>(numRows);
        // Scale the flat-size estimate without overflowing the multiplication.
        boundedEstimatedBytes.push_back(
            (estimatedBytes / rowCount) * sliceRows +
            ((estimatedBytes % rowCount) * sliceRows + rowCount - 1) /
                rowCount);
        start = end;
      }
    }

    size_t startpos = 0;
    size_t runningRows = 0;
    uint64_t runningBytes = 0;
    for (size_t i = 0; i < boundedViews.size(); ++i) {
      auto const numRows = static_cast<size_t>(boundedViews[i].num_rows());
      const auto estimatedBytes = boundedEstimatedBytes[i];
      // If adding this table would exceed the limit, flush current batch
      // [startpos, i). cuDF uses signed 32-bit offsets for STRING/LIST
      // children, so row count alone cannot make concatenate safe.
      if (runningRows > 0 &&
          (runningRows + numRows > maxRows ||
           !hasSameEmptyStringCharsPattern(
               boundedViews[startpos], boundedViews[i]) ||
           estimatedBytes > kMaxSafeConcatEstimatedBytes -
                   std::min(runningBytes, kMaxSafeConcatEstimatedBytes))) {
        outputTables.push_back(
            cudf::concatenate(
                std::vector<cudf::table_view>(
                    boundedViews.begin() + startpos, boundedViews.begin() + i),
                stream,
                mr));
        startpos = i;
        runningRows = 0;
        runningBytes = 0;
      }
      runningRows += numRows;
      runningBytes += estimatedBytes;
    }
    // Flush the final batch [startpos, end).
    if (startpos < boundedViews.size()) {
      outputTables.push_back(
          cudf::concatenate(
              std::vector<cudf::table_view>(
                  boundedViews.begin() + startpos, boundedViews.end()),
              stream,
              mr));
    }
    orderCudfVectorDeallocationsAfterStream(tables, inputStreams, stream);

    // Input tables are deallocated here when 'tables' goes out of scope.
    return outputTables;
  } catch (...) {
    // A failed later batch may leave earlier concatenate kernels in flight.
    stream.synchronize();
    throw;
  }
}

std::vector<CudfVectorPtr> getConcatenatedCudfVectorsBatched(
    memory::MemoryPool* pool,
    std::vector<CudfVectorPtr>&& vectors,
    const TypePtr& tableType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(pool);

  std::vector<CudfVectorPtr> outputVectors;
  if (tableType->size() > 0) {
    auto tables =
        getConcatenatedTableBatched(std::move(vectors), tableType, stream, mr);
    outputVectors.reserve(tables.size());
    for (auto& table : tables) {
      VELOX_CHECK_NOT_NULL(table);
      const auto rowCount =
          checkedVectorSize(static_cast<size_t>(table->num_rows()));
      outputVectors.push_back(
          std::make_shared<CudfVector>(
              pool, tableType, rowCount, std::move(table), stream));
    }
    return outputVectors;
  }

  size_t remainingRows = 0;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    VELOX_CHECK_EQ(vector->getTableView().num_columns(), 0);
    const auto rowCount = static_cast<size_t>(vector->size());
    VELOX_CHECK_LE(
        rowCount,
        std::numeric_limits<size_t>::max() - remainingRows,
        "zero-column cuDF vector row count overflow");
    remainingRows += rowCount;
  }

  const auto maxRows = maxBatchRows();
  do {
    const auto chunkRows = std::min(remainingRows, maxRows);
    outputVectors.push_back(
        std::make_shared<CudfVector>(
            pool,
            tableType,
            checkedVectorSize(chunkRows),
            makeEmptyTable(tableType, stream, mr),
            stream));
    remainingRows -= chunkRows;
  } while (remainingRows > 0);

  return outputVectors;
}

void streamsWaitForStream(
    CudaEvent& event,
    std::span<const rmm::cuda_stream_view> streams,
    rmm::cuda_stream_view stream) {
  event.recordFrom(stream);
  for (const auto& strm : streams) {
    event.waitOn(strm);
  }
}

CudaEvent::CudaEvent(unsigned int flags) {
  cudaEvent_t ev{};
  CUDF_CUDA_TRY(cudaEventCreateWithFlags(&ev, flags));
  event_ = ev;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

const CudaEvent& CudaEvent::recordFrom(rmm::cuda_stream_view stream) const {
  CUDF_CUDA_TRY(cudaEventRecord(event_, stream.value()));
  return *this;
}

const CudaEvent& CudaEvent::waitOn(rmm::cuda_stream_view stream) const {
  CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.value(), event_, 0));
  return *this;
}

std::string getBaseFunctionName(const std::string& fullName) {
  auto pos = fullName.rfind('.');
  return pos == std::string::npos ? fullName : fullName.substr(pos + 1);
}

std::string stripFunctionPrefix(
    const std::string& name,
    const std::string& prefix) {
  auto base = getBaseFunctionName(name);
  if (!prefix.empty() && base.find(prefix) == 0) {
    return base.substr(prefix.size());
  }
  return base;
}

void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    std::span<const rmm::cuda_stream_view> inputStreams,
    rmm::cuda_stream_view stream) {
  bool allRebound = true;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    allRebound &= vector->rebindStream(stream);
  }

  if (!allRebound) {
    streamsWaitForStream(eventForThread(), inputStreams, stream);
  }
}

} // namespace facebook::velox::cudf_velox
