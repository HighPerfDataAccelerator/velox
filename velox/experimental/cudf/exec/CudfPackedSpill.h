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

#include "velox/common/base/SpillConfig.h"

#include <folly/Executor.h>

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace facebook::velox::cudf_velox {

struct CudfPackedSpillWriteResult {
  uint64_t fileOffset{0};
  uint64_t storedBytes{0};
  uint64_t compressionMicros{0};
  bool compressed{false};
};

/// Raw-file backend for packed cuDF payloads.
///
/// Velox's SpillConfig remains the control plane (task directory, spill
/// executor and query spill accounting), while this class deliberately avoids
/// SpillWriter's RowVector/Presto-page serialization. Offset-local I/O allows
/// independent GPU operators and partitions to share the same bounded async
/// pipeline.
class CudfPackedSpillFile
    : public std::enable_shared_from_this<CudfPackedSpillFile> {
 public:
  CudfPackedSpillFile(
      std::string path,
      folly::Executor* executor,
      common::UpdateAndCheckSpillLimitCB updateSpillLimit = nullptr);
  ~CudfPackedSpillFile();

  std::pair<uint64_t, std::shared_future<void>> appendAsync(
      std::shared_ptr<uint8_t> data,
      uint64_t bytes);

  /// Optionally compresses one independent packed-cuDF block and writes it on
  /// the Velox spill executor. The source owner remains alive until the write
  /// completes, allowing a pinned staging lease to flow directly from D2H to
  /// spill without a pageable intermediate copy.
  std::shared_future<CudfPackedSpillWriteResult> appendCompressedAsync(
      std::shared_ptr<uint8_t> data,
      uint64_t bytes,
      bool enableCompression = true);

  void read(uint64_t offset, uint64_t bytes, uint8_t* destination);
  void reclaim(uint64_t offset, uint64_t bytes);

  const std::string& path() const {
    return path_;
  }

 private:
  void ensureOpenLocked();
  uint64_t reserveOffset(uint64_t bytes);
  void write(uint64_t offset, uint64_t bytes, const uint8_t* source);

  const std::string path_;
  folly::Executor* const executor_;
  const common::UpdateAndCheckSpillLimitCB updateSpillLimit_;
  std::mutex mutex_;
  int dataFd_{-1};
  uint64_t nextOffset_{0};
  std::atomic<uint64_t> writeCalls_{0};
  std::atomic<uint64_t> writeMicros_{0};
  uint64_t reclaimedBytes_{0};
  uint64_t nextReclaimLogBytes_{16ULL << 30};
  bool reclaimWarningLogged_{false};
};

/// Selects a striped physical path while retaining Velox's task spill
/// directory as the lifecycle/identity root. CUDF_PACKED_SPILL_DIRECTORIES is
/// shared by all GPU operators. The former Grace-join variable is accepted as
/// a compatibility fallback.
std::string makeCudfPackedSpillPath(
    const std::string& taskSpillDirectory,
    std::string_view filename,
    size_t stripeKey);

/// Creates a packed file using the executor and spill-limit callback from the
/// operator's Velox SpillConfig when available.
std::shared_ptr<CudfPackedSpillFile> createCudfPackedSpillFile(
    const std::string& taskSpillDirectory,
    std::string_view filename,
    size_t stripeKey,
    const common::SpillConfig* spillConfig);

/// Executor-wide host budget shared by Grace Join, TopN and future GPU
/// spillable operators. The returned token releases its reservation.
std::shared_ptr<void> tryReserveCudfPackedHostMemory(
    uint64_t bytes,
    uint64_t limitBytes);

uint64_t currentCudfPackedHostMemoryReservedBytes();

/// Acquires one reusable pinned D2H/H2D staging buffer. The aliasing shared
/// pointer returns the slot to the bounded process-wide pool when its final
/// owner (including asynchronous writes) releases it. Returns nullptr instead
/// of blocking when all slots are busy.
std::shared_ptr<uint8_t> acquireCudfPackedPinnedBuffer(
    uint64_t requiredBytes);

} // namespace facebook::velox::cudf_velox
