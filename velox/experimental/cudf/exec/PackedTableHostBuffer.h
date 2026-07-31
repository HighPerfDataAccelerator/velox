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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/memory/Allocation.h"

#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <vector>

namespace facebook::velox::cudf_velox {

struct PackedTableHostBufferStats {
  uint64_t packedInputBatches{0};
  uint64_t repackedInputBatches{0};
  uint64_t repackNanos{0};
  uint64_t deviceToHostBytes{0};
  uint64_t deviceToHostNanos{0};
  uint64_t hostUncompressedBytes{0};
  uint64_t hostCompressedBytes{0};
  uint64_t hostCompressionNanos{0};
  uint64_t hostDecompressionNanos{0};
  uint64_t dictionaryCandidateBatches{0};
  uint64_t dictionaryEncodedBatches{0};
  uint64_t dictionaryInputBytes{0};
  uint64_t dictionaryOutputBytes{0};
  uint64_t dictionaryEncodeNanos{0};
  uint64_t dictionaryDecodeNanos{0};
  uint64_t hostToDeviceBytes{0};
  uint64_t hostToDeviceNanos{0};
};

/// Pageable-host backing for a cudf::packed_table. This preserves libcudf's
/// packed metadata and contiguous data representation without materializing a
/// Velox or Arrow column tree.
class PackedTableHostBuffer {
 public:
  PackedTableHostBuffer() = default;

  static PackedTableHostBuffer fromVector(
      CudfVectorPtr input,
      memory::MemoryPool* pool,
      rmm::device_async_resource_ref mr,
      PackedTableHostBufferStats& stats,
      bool compress = false,
      bool dictionaryEncodeStrings = false);

  CudfVectorPtr toVector(
      memory::MemoryPool* pool,
      const RowTypePtr& type,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      PackedTableHostBufferStats& stats);

  uint64_t size() const {
    uint64_t bytes = dataSize_;
    for (const auto& dictionary : dictionaryColumns_) {
      bytes += dictionary.dataSize;
    }
    return bytes;
  }

  uint64_t uncompressedSize() const {
    return uncompressedSize_;
  }

  vector_size_t numRows() const {
    return numRows_;
  }

 private:
  struct DictionaryHostColumn {
    column_index_t columnIndex;
    std::vector<uint8_t> metadata;
    memory::ContiguousAllocation data;
    uint64_t dataSize;
  };

  PackedTableHostBuffer(
      std::vector<uint8_t> metadata,
      memory::ContiguousAllocation data,
      uint64_t dataSize,
      vector_size_t numRows,
      uint64_t uncompressedSize,
      std::vector<DictionaryHostColumn> dictionaryColumns)
      : metadata_(std::move(metadata)),
        data_(std::move(data)),
        dataSize_(dataSize),
        numRows_(numRows),
        uncompressedSize_(uncompressedSize),
        dictionaryColumns_(std::move(dictionaryColumns)) {}

  std::vector<uint8_t> metadata_;
  memory::ContiguousAllocation data_;
  uint64_t dataSize_{0};
  vector_size_t numRows_{0};
  uint64_t uncompressedSize_{0};
  std::vector<DictionaryHostColumn> dictionaryColumns_;
};

} // namespace facebook::velox::cudf_velox
