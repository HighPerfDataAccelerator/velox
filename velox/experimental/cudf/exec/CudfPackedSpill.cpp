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

#include "velox/experimental/cudf/exec/CudfPackedSpill.h"

#include "velox/common/base/Exceptions.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <cuda_runtime_api.h>
#include <lz4.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <linux/falloc.h>
#include <limits>
#include <sys/syscall.h>
#include <unistd.h>

#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

folly::CPUThreadPoolExecutor& fallbackPackedSpillExecutor() {
  static folly::CPUThreadPoolExecutor executor(8);
  return executor;
}

std::vector<std::string> parseRoots(const char* value) {
  std::vector<std::string> result;
  if (value == nullptr || *value == '\0') {
    return result;
  }
  std::string input(value);
  size_t begin = 0;
  while (begin <= input.size()) {
    const auto end = input.find(',', begin);
    auto root = input.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    const auto first = root.find_first_not_of(" \t");
    const auto last = root.find_last_not_of(" \t");
    if (first != std::string::npos) {
      result.push_back(root.substr(first, last - first + 1));
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return result;
}

const std::vector<std::string>& packedSpillRoots() {
  static const auto roots = [] {
    auto result = parseRoots(std::getenv("CUDF_PACKED_SPILL_DIRECTORIES"));
    if (result.empty()) {
      result = parseRoots(
          std::getenv("CUDF_HASH_JOIN_GRACE_SPILL_DIRECTORIES"));
    }
    return result;
  }();
  return roots;
}

std::atomic<uint64_t>& packedHostMemoryReservedBytes() {
  static std::atomic<uint64_t> bytes{0};
  return bytes;
}

class PackedHostMemoryReservation {
 public:
  explicit PackedHostMemoryReservation(uint64_t bytes) : bytes_(bytes) {}
  ~PackedHostMemoryReservation() {
    const auto previous = packedHostMemoryReservedBytes().fetch_sub(
        bytes_, std::memory_order_acq_rel);
    VELOX_DCHECK_GE(previous, bytes_);
  }

 private:
  const uint64_t bytes_;
};

class PackedPinnedBufferPool {
 public:
  PackedPinnedBufferPool() {
    const auto* value = std::getenv("CUDF_PACKED_SPILL_PINNED_BUFFER_COUNT");
    if (value == nullptr) {
      value = std::getenv("CUDF_HASH_JOIN_GRACE_PINNED_BUFFER_COUNT");
    }
    if (value != nullptr) {
      char* end = nullptr;
      const auto requested = std::strtoull(value, &end, 10);
      if (end != value && *end == '\0') {
        maxSlots_ = std::clamp<uint64_t>(requested, 1, 16);
      }
    }
  }

  std::shared_ptr<uint8_t> acquire(uint64_t requiredBytes) {
    if (requiredBytes == 0) {
      return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    size_t available = slots_.size();
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (!slots_[i]->busy && slots_[i]->capacity >= requiredBytes) {
        slots_[i]->busy = true;
        return makeOwner(i);
      }
      if (!slots_[i]->busy && available == slots_.size()) {
        available = i;
      }
    }
    if (available == slots_.size() && slots_.size() < maxSlots_) {
      slots_.push_back(std::make_unique<Slot>());
      available = slots_.size() - 1;
    }
    if (available == slots_.size()) {
      return {};
    }

    auto& slot = *slots_[available];
    uint8_t* allocated = nullptr;
    const auto allocationBytes =
        std::max<uint64_t>(256ULL << 20, requiredBytes);
    const auto status = cudaHostAlloc(
        reinterpret_cast<void**>(&allocated),
        allocationBytes,
        cudaHostAllocDefault);
    if (status != cudaSuccess) {
      LOG(WARNING) << "Could not allocate " << allocationBytes
                   << " bytes for packed cuDF pinned staging slot "
                   << available << ": " << cudaGetErrorString(status)
                   << "; falling back to pageable staging";
      return {};
    }
    if (slot.data != nullptr) {
      cudaFreeHost(slot.data);
    }
    slot.data = allocated;
    slot.capacity = allocationBytes;
    slot.busy = true;
    return makeOwner(available);
  }

  ~PackedPinnedBufferPool() {
    for (auto& slot : slots_) {
      if (slot->data != nullptr) {
        cudaFreeHost(slot->data);
      }
    }
  }

 private:
  struct Slot {
    uint8_t* data{nullptr};
    uint64_t capacity{0};
    bool busy{false};
  };

  struct Lease {
    Lease(PackedPinnedBufferPool* pool, size_t slot)
        : pool(pool), slot(slot) {}
    ~Lease() {
      std::lock_guard<std::mutex> lock(pool->mutex_);
      VELOX_CHECK(pool->slots_[slot]->busy);
      pool->slots_[slot]->busy = false;
    }
    PackedPinnedBufferPool* pool;
    size_t slot;
  };

  std::shared_ptr<uint8_t> makeOwner(size_t slot) {
    auto lease = std::make_shared<Lease>(this, slot);
    return std::shared_ptr<uint8_t>(lease, slots_[slot]->data);
  }

  std::mutex mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
  size_t maxSlots_{4};
};

PackedPinnedBufferPool& packedPinnedBufferPool() {
  static PackedPinnedBufferPool pool;
  return pool;
}

} // namespace

CudfPackedSpillFile::CudfPackedSpillFile(
    std::string path,
    folly::Executor* executor,
    common::UpdateAndCheckSpillLimitCB updateSpillLimit)
    : path_(std::move(path)),
      executor_(
          executor == nullptr ? &fallbackPackedSpillExecutor() : executor),
      updateSpillLimit_(std::move(updateSpillLimit)) {}

CudfPackedSpillFile::~CudfPackedSpillFile() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (nextOffset_ > 0) {
    LOG(INFO) << "Packed cuDF spill file completed path=" << path_
              << " bytes=" << nextOffset_
              << " writeCalls=" << writeCalls_.load()
              << " cumulativePwriteUs=" << writeMicros_.load();
  }
  if (dataFd_ >= 0) {
    ::close(dataFd_);
    dataFd_ = -1;
  }
  std::error_code error;
  std::filesystem::remove(path_, error);
  if (error) {
    LOG(WARNING) << "Failed to remove packed cuDF spill file " << path_
                 << ": " << error.message();
  }
  error.clear();
  std::filesystem::remove(std::filesystem::path(path_).parent_path(), error);
}

std::pair<uint64_t, std::shared_future<void>>
CudfPackedSpillFile::appendAsync(
    std::shared_ptr<uint8_t> data,
    uint64_t bytes) {
  VELOX_CHECK(data != nullptr || bytes == 0);
  const auto offset = reserveOffset(bytes);
  std::promise<void> completion;
  auto future = completion.get_future().share();
  auto self = shared_from_this();
  executor_->add(
      [self = std::move(self),
       data = std::move(data),
       offset,
       bytes,
       completion = std::move(completion)]() mutable {
        try {
          self->write(offset, bytes, data.get());
          completion.set_value();
        } catch (...) {
          completion.set_exception(std::current_exception());
        }
      });
  return {offset, std::move(future)};
}

std::shared_future<CudfPackedSpillWriteResult>
CudfPackedSpillFile::appendCompressedAsync(
    std::shared_ptr<uint8_t> data,
    uint64_t bytes,
    bool enableCompression) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_LE(
      bytes,
      static_cast<uint64_t>(std::numeric_limits<int>::max()),
      "Packed cuDF spill block exceeds the LZ4 block size limit");
  std::promise<CudfPackedSpillWriteResult> completion;
  auto future = completion.get_future().share();
  auto self = shared_from_this();
  executor_->add(
      [self = std::move(self),
       data = std::move(data),
       bytes,
       enableCompression,
        completion = std::move(completion)]() mutable {
        try {
          std::vector<uint8_t> compressed;
          int compressedBytes = 0;
          uint64_t compressionMicros = 0;
          if (enableCompression) {
            const auto compressionStart = std::chrono::steady_clock::now();
            compressed.resize(static_cast<size_t>(
                LZ4_compressBound(static_cast<int>(bytes))));
            compressedBytes = LZ4_compress_default(
                reinterpret_cast<const char*>(data.get()),
                reinterpret_cast<char*>(compressed.data()),
                static_cast<int>(bytes),
                static_cast<int>(compressed.size()));
            compressionMicros =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - compressionStart)
                    .count();
          }
          const bool useCompressed = enableCompression &&
              compressedBytes > 0 &&
              static_cast<uint64_t>(compressedBytes) < bytes;
          const auto storedBytes = useCompressed
              ? static_cast<uint64_t>(compressedBytes)
              : bytes;
          const auto offset = self->reserveOffset(storedBytes);
          self->write(
              offset,
              storedBytes,
              useCompressed ? compressed.data() : data.get());
          // The promise is the write-completion boundary used by restore.
          // Release a pooled pinned D2H source before waking that restore so
          // it can immediately reuse the slot as an H2D bounce slab.
          data.reset();
          completion.set_value(CudfPackedSpillWriteResult{
              offset,
              storedBytes,
              static_cast<uint64_t>(compressionMicros),
              useCompressed});
        } catch (...) {
          completion.set_exception(std::current_exception());
        }
      });
  return future;
}

void CudfPackedSpillFile::read(
    uint64_t offset,
    uint64_t bytes,
    uint8_t* destination) {
  int fd;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureOpenLocked();
    fd = dataFd_;
  }
  uint64_t completed = 0;
  while (completed < bytes) {
    const auto result = ::pread(
        fd,
        destination + completed,
        bytes - completed,
        static_cast<off_t>(offset + completed));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    VELOX_CHECK_GT(
        result,
        0,
        "Packed cuDF spill file {} is truncated at offset {}: {}",
        path_,
        offset + completed,
        result < 0 ? std::strerror(errno) : "unexpected EOF");
    completed += static_cast<uint64_t>(result);
  }
}

void CudfPackedSpillFile::reclaim(uint64_t offset, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  ensureOpenLocked();
  const auto rc = ::syscall(
      SYS_fallocate,
      dataFd_,
      FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
      static_cast<off_t>(offset),
      static_cast<off_t>(bytes));
  const int savedErrno = errno;
  if (rc != 0) {
    if (!reclaimWarningLogged_) {
      reclaimWarningLogged_ = true;
      LOG(WARNING) << "Failed to reclaim packed cuDF spill range file="
                   << path_ << " offset=" << offset << " bytes=" << bytes
                   << ": " << std::strerror(savedErrno);
    }
    return;
  }
  reclaimedBytes_ += bytes;
  if (reclaimedBytes_ >= nextReclaimLogBytes_) {
    LOG(INFO) << "Packed cuDF spill reclaimed ranges file=" << path_
              << " reclaimedBytes=" << reclaimedBytes_;
    nextReclaimLogBytes_ += 16ULL << 30;
  }
}

void CudfPackedSpillFile::ensureOpenLocked() {
  if (dataFd_ >= 0) {
    return;
  }
  std::filesystem::create_directories(
      std::filesystem::path(path_).parent_path());
  dataFd_ =
      ::open(path_.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0600);
  VELOX_CHECK_GE(
      dataFd_,
      0,
      "Failed to open packed cuDF spill file {}: {}",
      path_,
      std::strerror(errno));
}

uint64_t CudfPackedSpillFile::reserveOffset(uint64_t bytes) {
  if (updateSpillLimit_) {
    updateSpillLimit_(bytes);
  }
  std::lock_guard<std::mutex> lock(mutex_);
  ensureOpenLocked();
  const auto offset = nextOffset_;
  nextOffset_ += bytes;
  return offset;
}

void CudfPackedSpillFile::write(
    uint64_t offset,
    uint64_t bytes,
    const uint8_t* source) {
  const auto writeStart = std::chrono::steady_clock::now();
  int fd;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureOpenLocked();
    fd = dataFd_;
  }
  uint64_t completed = 0;
  while (completed < bytes) {
    const auto result = ::pwrite(
        fd,
        source + completed,
        bytes - completed,
        static_cast<off_t>(offset + completed));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    VELOX_CHECK_GT(
        result,
        0,
        "Failed to write packed cuDF spill file {} at offset {}: {}",
        path_,
        offset + completed,
        result < 0 ? std::strerror(errno) : "zero-length write");
    completed += static_cast<uint64_t>(result);
  }
  writeCalls_.fetch_add(1, std::memory_order_relaxed);
  writeMicros_.fetch_add(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - writeStart)
          .count(),
      std::memory_order_relaxed);
}

std::string makeCudfPackedSpillPath(
    const std::string& taskSpillDirectory,
    std::string_view filename,
    size_t stripeKey) {
  const auto& roots = packedSpillRoots();
  if (roots.empty()) {
    return fmt::format("{}/{}", taskSpillDirectory, filename);
  }
  const auto taskHash = static_cast<uint64_t>(
      std::hash<std::string>{}(taskSpillDirectory));
  const auto& root = roots[stripeKey % roots.size()];
  const auto taskKey = fmt::format("task-{:016x}", taskHash);
  return (std::filesystem::path(root) / taskKey / filename).string();
}

std::shared_ptr<CudfPackedSpillFile> createCudfPackedSpillFile(
    const std::string& taskSpillDirectory,
    std::string_view filename,
    size_t stripeKey,
    const common::SpillConfig* spillConfig) {
  return std::make_shared<CudfPackedSpillFile>(
      makeCudfPackedSpillPath(taskSpillDirectory, filename, stripeKey),
      spillConfig == nullptr ? nullptr : spillConfig->executor,
      spillConfig == nullptr ? nullptr : spillConfig->updateAndCheckSpillLimitCb);
}

std::shared_ptr<void> tryReserveCudfPackedHostMemory(
    uint64_t bytes,
    uint64_t limitBytes) {
  if (limitBytes == 0 || bytes == 0) {
    return {};
  }
  auto& reserved = packedHostMemoryReservedBytes();
  auto current = reserved.load(std::memory_order_acquire);
  while (current <= limitBytes && bytes <= limitBytes - current) {
    if (reserved.compare_exchange_weak(
            current,
            current + bytes,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      return std::make_shared<PackedHostMemoryReservation>(bytes);
    }
  }
  return {};
}

uint64_t currentCudfPackedHostMemoryReservedBytes() {
  return packedHostMemoryReservedBytes().load(std::memory_order_acquire);
}

std::shared_ptr<uint8_t> acquireCudfPackedPinnedBuffer(
    uint64_t requiredBytes) {
  return packedPinnedBufferPool().acquire(requiredBytes);
}

} // namespace facebook::velox::cudf_velox
