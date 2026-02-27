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

#include "velox/experimental/cudf/exec/PinnedArrowInterop.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/common/memory/Memory.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/arrow/Bridge.h"

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <arrow/c/bridge.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

#include <cstring>

namespace facebook::velox::cudf_velox {

cudf::type_id veloxToCudfTypeId(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return cudf::type_id::BOOL8;
    case TypeKind::TINYINT:
      return cudf::type_id::INT8;
    case TypeKind::SMALLINT:
      return cudf::type_id::INT16;
    case TypeKind::INTEGER:
      // TODO: handle interval types (durations?)
      // if (type->isIntervalYearMonth()) {
      //   return cudf::type_id::...;
      // }
      if (type->isDate()) {
        return cudf::type_id::TIMESTAMP_DAYS;
      }
      return cudf::type_id::INT32;
    case TypeKind::BIGINT:
      return cudf::type_id::INT64;
    case TypeKind::REAL:
      return cudf::type_id::FLOAT32;
    case TypeKind::DOUBLE:
      return cudf::type_id::FLOAT64;
    case TypeKind::VARCHAR:
      return cudf::type_id::STRING;
    case TypeKind::VARBINARY:
      return cudf::type_id::STRING;
    case TypeKind::TIMESTAMP:
      return cudf::type_id::TIMESTAMP_NANOSECONDS;
    // case TypeKind::HUGEINT: return cudf::type_id::DURATION_DAYS;
    // TODO: DATE was converted to a logical type:
    // https://github.com/facebookincubator/velox/commit/e480f5c03a6c47897ef4488bd56918a89719f908
    // case TypeKind::DATE: return cudf::type_id::DURATION_DAYS;
    // case TypeKind::INTERVAL_DAY_TIME: return cudf::type_id::EMPTY;
    // TODO: Decimals are now logical types:
    // https://github.com/facebookincubator/velox/commit/73d2f935b55f084d30557c7be94b9768efb8e56f
    // case TypeKind::SHORT_DECIMAL: return cudf::type_id::DECIMAL64;
    // case TypeKind::LONG_DECIMAL: return cudf::type_id::DECIMAL128;
    case TypeKind::ARRAY:
      return cudf::type_id::LIST;
    // case TypeKind::MAP: return cudf::type_id::EMPTY;
    case TypeKind::ROW:
      return cudf::type_id::STRUCT;
    // case TypeKind::UNKNOWN: return cudf::type_id::EMPTY;
    // case TypeKind::FUNCTION: return cudf::type_id::EMPTY;
    // case TypeKind::OPAQUE: return cudf::type_id::EMPTY;
    // case TypeKind::INVALID: return cudf::type_id::EMPTY;
    default:
      CUDF_FAIL(
          "Unsupported Velox type: " +
          std::string(TypeKindName::toName(type->kind())));
      return cudf::type_id::EMPTY;
  }
}

namespace with_arrow {

namespace {

/// Compute the byte size of the i-th buffer in a C Data Interface ArrowArray,
/// given the Arrow format string from ArrowSchema.
/// `buffersOnHost` must be true when buffers are readable from CPU (HtoD path).
int64_t arrowBufferSize(
    const ArrowArray* array,
    const char* format,
    int bufIdx,
    bool buffersOnHost) {
  const int64_t len = array->length + array->offset;

  // Buffer 0: validity bitmap for all types.
  if (bufIdx == 0) {
    if (array->null_count == 0 && array->buffers[0] == nullptr) {
      return 0;
    }
    return (len + 7) / 8;
  }

  switch (format[0]) {
    case 'n':
      return 0; // null type
    case 'b':
      return (bufIdx == 1) ? (len + 7) / 8 : 0; // bool (bit-packed)
    case 'c':
    case 'C':
      return (bufIdx == 1) ? len : 0; // int8/uint8
    case 's':
    case 'S':
      return (bufIdx == 1) ? len * 2 : 0; // int16/uint16
    case 'i':
    case 'I':
      return (bufIdx == 1) ? len * 4 : 0; // int32/uint32
    case 'l':
    case 'L':
      return (bufIdx == 1) ? len * 8 : 0; // int64/uint64
    case 'e':
      return (bufIdx == 1) ? len * 2 : 0; // float16
    case 'f':
      return (bufIdx == 1) ? len * 4 : 0; // float32
    case 'g':
      return (bufIdx == 1) ? len * 8 : 0; // float64
    case 'u': // utf8
      if (bufIdx == 1)
        return static_cast<int64_t>(len + 1) * sizeof(int32_t);
      if (bufIdx == 2 && buffersOnHost && array->buffers[1]) {
        return static_cast<const int32_t*>(array->buffers[1])[len];
      }
      return 0;
    case 'U': // large utf8
      if (bufIdx == 1)
        return static_cast<int64_t>(len + 1) * sizeof(int64_t);
      if (bufIdx == 2 && buffersOnHost && array->buffers[1]) {
        return static_cast<const int64_t*>(array->buffers[1])[len];
      }
      return 0;
    case 'z': // binary
      if (bufIdx == 1)
        return static_cast<int64_t>(len + 1) * sizeof(int32_t);
      if (bufIdx == 2 && buffersOnHost && array->buffers[1]) {
        return static_cast<const int32_t*>(array->buffers[1])[len];
      }
      return 0;
    case 't': // temporal
      if (format[1] == 'd' && format[2] == 'D')
        return (bufIdx == 1) ? len * 4 : 0; // date32
      if (format[1] == 'd' && format[2] == 'm')
        return (bufIdx == 1) ? len * 8 : 0; // date64
      return (bufIdx == 1) ? len * 8 : 0; // timestamp variants
    case '+': // nested
      if (format[1] == 's')
        return 0; // struct: only validity (buf 0)
      if (format[1] == 'l') // list
        return (bufIdx == 1) ? static_cast<int64_t>(len + 1) * sizeof(int32_t)
                             : 0;
      if (format[1] == 'L') // large list
        return (bufIdx == 1) ? static_cast<int64_t>(len + 1) * sizeof(int64_t)
                             : 0;
      return 0;
    case 'd': // decimal128
      return (bufIdx == 1) ? len * 16 : 0;
    default:
      return 0;
  }
}

/// Recursively copy pageable ArrowArray buffers to pinned host memory.
/// The ArrowArray buffer pointers are replaced in-place.  The PinnedHostBuffers
/// are kept alive via `pinnedBufs` — caller must not clear it until
/// cudf::from_arrow + stream sync is done.
void pinArrowBuffersRecursive(
    ArrowArray* array,
    ArrowSchema* schema,
    std::vector<std::shared_ptr<PinnedHostBuffer>>& pinnedBufs) {
  for (int64_t i = 0; i < array->n_buffers; ++i) {
    if (array->buffers[i] == nullptr) {
      continue;
    }
    int64_t size = arrowBufferSize(array, schema->format, i, /*buffersOnHost=*/true);
    if (size <= 0) {
      continue;
    }
    auto pinned = std::make_shared<PinnedHostBuffer>(size);
    std::memcpy(pinned->data(), array->buffers[i], size);
    array->buffers[i] = pinned->data();
    pinnedBufs.push_back(std::move(pinned));
  }
  for (int64_t i = 0; i < array->n_children; ++i) {
    pinArrowBuffersRecursive(
        array->children[i], schema->children[i], pinnedBufs);
  }
  if (array->dictionary && schema->dictionary) {
    pinArrowBuffersRecursive(
        array->dictionary, schema->dictionary, pinnedBufs);
  }
}

} // anonymous namespace

std::unique_ptr<cudf::table> toCudfTable(
    const facebook::velox::RowVectorPtr& veloxTable,
    facebook::velox::memory::MemoryPool* pool,
    rmm::cuda_stream_view stream) {
  ArrowOptions arrowOptions{true, true};
  ArrowArray arrowArray;
  exportToArrow(
      std::dynamic_pointer_cast<facebook::velox::BaseVector>(veloxTable),
      arrowArray,
      pool,
      arrowOptions);
  ArrowSchema arrowSchema;
  exportToArrow(
      std::dynamic_pointer_cast<facebook::velox::BaseVector>(veloxTable),
      arrowSchema,
      arrowOptions);

  // Copy pageable Arrow buffers to pinned host memory so that
  // cudf::from_arrow's cudaMemcpyDefault uses DMA instead of staging.
  std::vector<std::shared_ptr<PinnedHostBuffer>> pinnedBufs;
  pinArrowBuffersRecursive(&arrowArray, &arrowSchema, pinnedBufs);

  auto tbl = cudf::from_arrow(&arrowSchema, &arrowArray, stream);

  // Wait for HtoD copies to complete before freeing pinned buffers.
  stream.synchronize();
  pinnedBufs.clear();

  if (arrowArray.release) {
    arrowArray.release(&arrowArray);
  }
  if (arrowSchema.release) {
    arrowSchema.release(&arrowSchema);
  }
  return tbl;
}

namespace {

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<cudf::column_metadata>& metadata,
    rmm::cuda_stream_view stream) {
  // Use pinned host memory for DtoH instead of cudf::to_arrow_host (pageable).
  auto arrowDeviceArray = pinnedToArrowHost(table, stream);
  auto& arrowArray = arrowDeviceArray->array;

  auto arrowSchema = cudf::to_arrow_schema(table, metadata);
  auto veloxTable = importFromArrowAsOwner(*arrowSchema, arrowArray, pool);
  auto castedPtr =
      std::dynamic_pointer_cast<facebook::velox::RowVector>(veloxTable);
  VELOX_CHECK_NOT_NULL(castedPtr);
  return castedPtr;
}

template <typename Iterator>
std::vector<cudf::column_metadata>
getMetadata(Iterator begin, Iterator end, const std::string& namePrefix) {
  std::vector<cudf::column_metadata> metadata;
  int i = 0;
  for (auto c = begin; c < end; c++) {
    metadata.push_back(cudf::column_metadata(namePrefix + std::to_string(i)));
    metadata.back().children_meta = getMetadata(
        c->child_begin(), c->child_end(), namePrefix + std::to_string(i));
    i++;
  }
  return metadata;
}

} // namespace

facebook::velox::RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    facebook::velox::memory::MemoryPool* pool,
    std::string namePrefix,
    rmm::cuda_stream_view stream) {
  auto metadata = getMetadata(table.begin(), table.end(), namePrefix);
  return toVeloxColumn(table, pool, metadata, stream);
}

RowVectorPtr toVeloxColumn(
    const cudf::table_view& table,
    memory::MemoryPool* pool,
    const std::vector<std::string>& columnNames,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::column_metadata> metadata;
  for (auto name : columnNames) {
    metadata.emplace_back(cudf::column_metadata(name));
  }
  return toVeloxColumn(table, pool, metadata, stream);
}

} // namespace with_arrow
} // namespace facebook::velox::cudf_velox
