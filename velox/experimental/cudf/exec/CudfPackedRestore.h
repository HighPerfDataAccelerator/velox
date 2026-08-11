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

#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// One uncompressed packed-cuDF image ready to restore from host memory.
struct CudfPackedHostRestoreChunk {
  std::unique_ptr<std::vector<uint8_t>> metadata;
  std::shared_ptr<uint8_t> data;
  uint64_t dataBytes{0};
  // Optional accounting/lifetime token retained until the batched H2D copy
  // has synchronized.
  std::shared_ptr<void> keepAlive;
  // Optional one-shot producer which writes the uncompressed packed image
  // directly into a pinned bounce slab. Disk-backed TopN chunks use this to
  // avoid first materializing a complete pageable bucket. If the bounded
  // pinned pool is busy, bulk restore invokes the same producer into a
  // temporary pageable allocation and preserves the original behavior.
  std::function<void(uint8_t*)> materializeIntoPinned;
  // Optional optimization for an already-materialized pageable image. When
  // two bounded pinned slabs are available, copy this chunk into a slab and
  // submit H2D from page-locked memory. Unlike materializeIntoPinned, failure
  // to acquire both slabs must preserve direct H2D from data: making another
  // pageable copy would add work without enabling the CUDA copy engine.
  bool stageResidentPageableThroughPinned{false};
};

/// Host-side diagnostics for one packed restore wave.
///
/// The pinned path materializes disk-backed chunks directly into two reusable
/// slabs. While one slab is copied by CUDA, the caller fills the other on the
/// CPU. Resident pageable chunks keep their direct-copy path, and disk-backed
/// chunks fall back without waiting when two bounded pool slots are not
/// available. Distinct materializers within one slab may run concurrently
/// when CUDF_PACKED_RESTORE_HOST_THREADS is greater than one.
struct CudfBulkPackedRestoreStats {
  uint64_t pinnedBounceBytes{0};
  uint64_t residentPageableBounceBytes{0};
  uint64_t pageableDirectBytes{0};
  uint64_t pinnedBounceCopies{0};
  uint64_t hostStageMicros{0};
  uint64_t bounceReuseWaitMicros{0};
  uint64_t copyStreamSynchronizeMicros{0};
  uint64_t parallelHostStageGroups{0};
  uint64_t parallelHostStageChunks{0};
  uint64_t pinnedHostThreadLimit{0};

  bool usedPinnedBounce() const {
    return pinnedBounceBytes > 0;
  }
};

/// Per-call controls for packed restore. A zero pinnedHostThreads value keeps
/// the process-wide CUDF_PACKED_RESTORE_HOST_THREADS limit. A non-zero value
/// selects an independently bounded call-local limit, while all pinned-source
/// restore tasks still share one process-wide eight-thread ceiling.
struct CudfBulkPackedRestoreOptions {
  size_t pinnedHostThreads{0};
};

/// Owns one device allocation shared by all table views restored in a batch.
///
/// cudf::packed_columns cannot represent this ownership: it requires a unique
/// rmm::device_buffer per packed table. cudf::unpack(metadata, gpuData) is the
/// supported non-owning API and lets each table view point at an aligned
/// subrange of gpuData_ instead.
class CudfBulkPackedRestore {
 public:
  CudfBulkPackedRestore() = default;
  CudfBulkPackedRestore(CudfBulkPackedRestore&&) noexcept = default;
  CudfBulkPackedRestore& operator=(CudfBulkPackedRestore&&) noexcept = default;
  CudfBulkPackedRestore(const CudfBulkPackedRestore&) = delete;
  CudfBulkPackedRestore& operator=(const CudfBulkPackedRestore&) = delete;

  const std::vector<cudf::table_view>& tables() const {
    return tables_;
  }

  const rmm::device_buffer* deviceBuffer() const {
    return gpuData_.get();
  }

  const CudfBulkPackedRestoreStats& stats() const {
    return stats_;
  }

 private:
  friend CudfBulkPackedRestore bulkRestoreCudfPackedHostChunks(
      std::vector<CudfPackedHostRestoreChunk> chunks,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      CudfBulkPackedRestoreOptions options);

  std::unique_ptr<rmm::device_buffer> gpuData_;
  std::vector<std::unique_ptr<std::vector<uint8_t>>> metadata_;
  std::vector<cudf::table_view> tables_;
  CudfBulkPackedRestoreStats stats_;
};

/// Restores all chunks using one device allocation, one H2D submission wave,
/// and one stream synchronization. Each chunk starts at an RMM-aligned offset
/// so metadata created for an independently allocated packed image remains
/// valid.
CudfBulkPackedRestore bulkRestoreCudfPackedHostChunks(
    std::vector<CudfPackedHostRestoreChunk> chunks,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    CudfBulkPackedRestoreOptions options = {});

} // namespace facebook::velox::cudf_velox
