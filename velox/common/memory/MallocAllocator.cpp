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

#include "velox/common/memory/MallocAllocator.h"
#include <folly/system/HardwareConcurrency.h>
#include "velox/common/memory/Memory.h"

#include <sys/mman.h>

#include <deque>
#include <mutex>

#include <gflags/gflags.h>

DECLARE_int64(velox_memory_contiguous_reuse_cache_bytes);

namespace facebook::velox::memory {
namespace {

// Process-wide cache of mmap'ed contiguous regions freed by any
// MallocAllocator. Hash tables and RowContainer runs are allocated and freed
// per task with the same power-of-two reservation sizes; returning the region
// to the kernel with munmap() and mapping a fresh one makes every reuse pay a
// 4KB page fault plus kernel zeroing per touched page. Keeping a bounded
// number of freed regions mapped and handing them back for an identical
// reservation size avoids those faults. Regions handed out from the cache are
// not zeroed, which matches the mallocContiguousEnabled (aligned_alloc) path.
// The bound is on bytes that were in use when the region was freed, which is
// what stays resident.
class ContiguousReuseCache {
 public:
  static ContiguousReuseCache& instance() {
    static auto* cache = new ContiguousReuseCache();
    return *cache;
  }

  // Returns a cached region of exactly 'maxBytes' or nullptr.
  void* take(size_t maxBytes) {
    if (FLAGS_velox_memory_contiguous_reuse_cache_bytes <= 0) {
      return nullptr;
    }
    std::lock_guard<std::mutex> l(mutex_);
    // Newest first: most recently used pages are most likely still in cache.
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
      if (it->maxBytes == maxBytes) {
        void* data = it->data;
        residentBytes_ -= it->residentBytes;
        entries_.erase(std::next(it).base());
        return data;
      }
    }
    return nullptr;
  }

  // Takes ownership of the mapping if it fits; returns false if the caller
  // must unmap it.
  bool put(void* data, size_t maxBytes, size_t residentBytes) {
    const int64_t limit = FLAGS_velox_memory_contiguous_reuse_cache_bytes;
    if (limit <= 0 || residentBytes > static_cast<size_t>(limit) / 2) {
      return false;
    }
    std::vector<Entry> evicted;
    {
      std::lock_guard<std::mutex> l(mutex_);
      entries_.push_back({data, maxBytes, residentBytes});
      residentBytes_ += residentBytes;
      while (residentBytes_ > static_cast<size_t>(limit)) {
        evicted.push_back(entries_.front());
        residentBytes_ -= entries_.front().residentBytes;
        entries_.pop_front();
      }
    }
    for (const auto& e : evicted) {
      ::munmap(e.data, e.maxBytes);
    }
    return true;
  }

 private:
  struct Entry {
    void* data;
    size_t maxBytes;
    size_t residentBytes;
  };

  std::mutex mutex_;
  std::deque<Entry> entries_;
  size_t residentBytes_{0};
};

} // namespace

MallocAllocator::MallocAllocator(const Options& options)
    : kind_(MemoryAllocator::Kind::kMalloc),
      mallocContiguousEnabled_(options.mallocContiguousEnabled),
      capacity_(options.capacity),
      reservationByteLimit_(options.reservationByteLimit),
      reserveFunc_(
          [this](uint32_t& counter, uint32_t increment, std::mutex& lock) {
            return incrementUsageWithReservationFunc(counter, increment, lock);
          }),
      releaseFunc_(
          [&](uint32_t& counter, uint32_t decrement, std::mutex& lock) {
            decrementUsageWithReservationFunc(counter, decrement, lock);
            return true;
          }),
      reservations_(folly::available_concurrency()) {}

MallocAllocator::~MallocAllocator() {
  // TODO: Remove the check when memory leak issue is resolved.
  if (FLAGS_velox_memory_leak_check_enabled) {
    VELOX_CHECK(
        ((allocatedBytes_ - reservations_.read()) == 0) &&
            (numAllocated_ == 0) && (numMapped_ == 0),
        "{}",
        toString());
  }
}

bool MallocAllocator::allocateNonContiguousWithoutRetry(
    const SizeMix& sizeMix,
    Allocation& out) {
  freeNonContiguous(out);
  if (sizeMix.totalPages == 0) {
    return true;
  }
  const auto totalBytes = AllocationTraits::pageBytes(sizeMix.totalPages);
  if (testingHasInjectedFailure(InjectedFailure::kCap) ||
      !incrementUsage(totalBytes)) {
    const auto errorMsg = fmt::format(
        "Exceeded memory allocator limit when allocating {} new pages"
        ", the memory allocator capacity is {}, the allocated bytes is {}",
        sizeMix.totalPages,
        succinctBytes(capacity_),
        succinctBytes(allocatedBytes_));
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000) << errorMsg;
    setAllocatorFailureMessage(errorMsg);
    return false;
  }

  std::vector<void*> buffers;
  buffers.reserve(sizeMix.numSizes);
  for (int32_t i = 0; i < sizeMix.numSizes; ++i) {
    MachinePageCount numSizeClassPages =
        sizeMix.sizeCounts[i] * sizeClassSizes_[sizeMix.sizeIndices[i]];
    void* ptr = nullptr;
    // Trigger allocation failure by skipping malloc
    if (!testingHasInjectedFailure(InjectedFailure::kAllocate)) {
      stats_.recordAllocate(
          AllocationTraits::pageBytes(sizeClassSizes_[sizeMix.sizeIndices[i]]),
          sizeMix.sizeCounts[i],
          [&]() {
            ptr = ::malloc(
                AllocationTraits::pageBytes(numSizeClassPages)); // NOLINT
          });
    }
    if (ptr == nullptr) {
      // Failed to allocate memory from memory.
      const auto errorMsg = fmt::format(
          "Malloc failed to allocate {} of memory while allocating for "
          "non-contiguous allocation of {} pages",
          succinctBytes(AllocationTraits::pageBytes(numSizeClassPages)),
          sizeMix.totalPages);
      VELOX_MEM_LOG(WARNING) << errorMsg;
      setAllocatorFailureMessage(errorMsg);
      break;
    }
    buffers.push_back(ptr);
    out.append(reinterpret_cast<uint8_t*>(ptr), numSizeClassPages); // NOLINT
  }

  if (buffers.size() != sizeMix.numSizes) {
    // Failed to allocate memory using malloc. Free any malloced pages and
    // return false.
    for (auto* buffer : buffers) {
      ::free(buffer);
    }
    out.clear();
    VELOX_MEM_LOG(WARNING)
        << "Failed to allocate memory for non-contiguous allocation of "
        << sizeMix.totalPages << " pages";
    decrementUsage(totalBytes);
    return false;
  }

  // Successfully allocated all pages.
  numAllocated_.fetch_add(sizeMix.totalPages);
  return true;
}

bool MallocAllocator::allocateContiguousWithoutRetry(
    MachinePageCount numPages,
    Allocation* collateral,
    ContiguousAllocation& allocation,
    MachinePageCount maxPages) {
  bool result;
  stats_.recordAllocate(AllocationTraits::pageBytes(numPages), 1, [&]() {
    result = allocateContiguousImpl(numPages, collateral, allocation, maxPages);
  });
  return result;
}

bool MallocAllocator::allocateContiguousImpl(
    MachinePageCount numPages,
    Allocation* collateral,
    ContiguousAllocation& allocation,
    MachinePageCount maxPages) {
  if (maxPages == 0) {
    maxPages = numPages;
  } else {
    VELOX_CHECK_LE(numPages, maxPages);
  }
  if (collateral != nullptr) {
    freeNonContiguous(*collateral);
  }
  auto numContiguousCollateralPages = allocation.numPages();
  if (numContiguousCollateralPages > 0) {
    dispatchFreeContiguous(allocation);
    numMapped_.fetch_sub(numContiguousCollateralPages);
    numAllocated_.fetch_sub(numContiguousCollateralPages);
    numExternalMapped_.fetch_sub(numContiguousCollateralPages);
    decrementUsage(AllocationTraits::pageBytes(numContiguousCollateralPages));
    allocation.clear();
  }
  if (numPages == 0) {
    return true;
  }

  const auto totalBytes = AllocationTraits::pageBytes(numPages);
  if (testingHasInjectedFailure(InjectedFailure::kCap) ||
      !incrementUsage(totalBytes)) {
    const auto errorMsg = fmt::format(
        "Exceeded memory allocator limit when allocating {} new pages, the "
        "memory allocator capacity is {}",
        numPages,
        succinctBytes(capacity_));
    setAllocatorFailureMessage(errorMsg);
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000) << errorMsg;
    return false;
  }
  numAllocated_.fetch_add(numPages);
  numMapped_.fetch_add(numPages);
  numExternalMapped_.fetch_add(numPages);
  const auto maxBytes = AllocationTraits::pageBytes(maxPages);
  void* data = dispatchAllocateContiguous(maxBytes);
  if (FOLLY_UNLIKELY(data == nullptr)) {
    numAllocated_.fetch_sub(numPages);
    numMapped_.fetch_sub(numPages);
    numExternalMapped_.fetch_sub(numPages);
    decrementUsage(static_cast<int64_t>(AllocationTraits::pageBytes(numPages)));
    const auto errorMsg = fmt::format(
        "Failed to allocate {} of contiguous memory", succinctBytes(maxBytes));
    VELOX_MEM_LOG(WARNING) << errorMsg;
    setAllocatorFailureMessage(errorMsg);
    return false;
  }
  allocation.set(data, AllocationTraits::pageBytes(numPages), maxBytes);
  if (!mallocContiguousEnabled_) {
    useHugePages(allocation, true);
  }
  return true;
}

int64_t MallocAllocator::freeNonContiguous(Allocation& allocation) {
  if (allocation.empty()) {
    return 0;
  }
  MachinePageCount freedPages{0};
  for (int32_t i = 0; i < allocation.numRuns(); ++i) {
    Allocation::PageRun run = allocation.runAt(i);
    void* ptr = run.data();
    const int64_t numPages = run.numPages();
    freedPages += numPages;
    stats_.recordFree(AllocationTraits::pageBytes(numPages), [&]() {
      ::free(ptr); // NOLINT
    });
  }

  const auto freedBytes = AllocationTraits::pageBytes(freedPages);
  decrementUsage(freedBytes);
  numAllocated_.fetch_sub(freedPages);
  allocation.clear();
  return freedBytes;
}

void MallocAllocator::freeContiguous(ContiguousAllocation& allocation) {
  stats_.recordFree(
      allocation.size(), [&]() { freeContiguousImpl(allocation); });
}

void MallocAllocator::freeContiguousImpl(ContiguousAllocation& allocation) {
  if (allocation.empty()) {
    return;
  }
  const auto bytes = allocation.size();
  const auto numPages = allocation.numPages();
  dispatchFreeContiguous(allocation);
  numMapped_.fetch_sub(numPages);
  numAllocated_.fetch_sub(numPages);
  numExternalMapped_.fetch_sub(numPages);
  decrementUsage(bytes);
  allocation.clear();
}

void* MallocAllocator::dispatchAllocateContiguous(size_t maxBytes) {
  if (testingHasInjectedFailure(InjectedFailure::kAllocate)) {
    return nullptr;
  }
  if (mallocContiguousEnabled_) {
    return ::aligned_alloc(AllocationTraits::kPageSize, maxBytes);
  }
  if (void* cached = ContiguousReuseCache::instance().take(maxBytes)) {
    return cached;
  }
  // TODO: add handling of MAP_FAILED.
  return ::mmap(
      nullptr,
      maxBytes,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANONYMOUS,
      -1,
      0);
}

void MallocAllocator::dispatchFreeContiguous(ContiguousAllocation& allocation) {
  if (mallocContiguousEnabled_) {
    ::free(allocation.data());
  } else {
    useHugePages(allocation, false);
    if (ContiguousReuseCache::instance().put(
            allocation.data(), allocation.maxSize(), allocation.size())) {
      return;
    }
    if (::munmap(allocation.data(), allocation.maxSize()) < 0) {
      VELOX_MEM_LOG(ERROR) << "Error for munmap(" << allocation.data() << ", "
                           << succinctBytes(allocation.size()) << "): '"
                           << folly::errnoStr(errno) << "'";
    }
  }
}

bool MallocAllocator::growContiguousWithoutRetry(
    MachinePageCount increment,
    ContiguousAllocation& allocation) {
  if (!incrementUsage(AllocationTraits::pageBytes(increment))) {
    const auto errorMsg = fmt::format(
        "Exceeded memory allocator limit when allocating {} new pages for "
        "total allocation of {} pages, the memory allocator capacity is {}",
        increment,
        allocation.numPages(),
        succinctBytes(capacity_));
    setAllocatorFailureMessage(errorMsg);
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000) << errorMsg;
    return false;
  }
  numAllocated_ += increment;
  numMapped_ += increment;
  numExternalMapped_ += increment;
  allocation.set(
      allocation.data(),
      allocation.size() + AllocationTraits::kPageSize * increment,
      allocation.maxSize());
  return true;
}

void* MallocAllocator::allocateBytesWithoutRetry(
    uint64_t bytes,
    uint16_t alignment) {
  if (!incrementUsage(bytes)) {
    auto errorMsg = fmt::format(
        "Failed to allocateBytes {}: Exceeded memory allocator "
        "limit of {}",
        succinctBytes(bytes),
        succinctBytes(capacity_));
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000) << errorMsg;
    setAllocatorFailureMessage(errorMsg);
    return nullptr;
  }
  if (!isAlignmentValid(bytes, alignment)) {
    decrementUsage(bytes);
    VELOX_FAIL(
        "Alignment check failed, allocateBytes {}, alignmentBytes {}",
        bytes,
        alignment);
  }
  void* result = (alignment > kMinAlignment) ? ::aligned_alloc(alignment, bytes)
                                             : ::malloc(bytes);
  if (FOLLY_UNLIKELY(result == nullptr)) {
    VELOX_MEM_LOG(ERROR) << "Failed to allocateBytes " << succinctBytes(bytes)
                         << " with " << alignment << " alignment";
  }
  return result;
}

void* MallocAllocator::allocateZeroFilledWithoutRetry(uint64_t bytes) {
  if (!incrementUsage(bytes)) {
    auto errorMsg = fmt::format(
        "Failed to allocateZeroFilled {}: Exceeded memory allocator "
        "limit of {}",
        succinctBytes(bytes),
        succinctBytes(capacity_));
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000) << errorMsg;
    setAllocatorFailureMessage(errorMsg);
    return nullptr;
  }
  void* result = std::calloc(1, bytes);
  if (FOLLY_UNLIKELY(result == nullptr)) {
    VELOX_MEM_LOG(ERROR) << "Failed to allocateZeroFilled "
                         << succinctBytes(bytes);
  }
  return result;
}

void* MallocAllocator::reallocateBytesWithoutRetry(
    void* p,
    uint64_t oldSize,
    uint64_t newSize,
    uint16_t alignment) {
  if (p == nullptr || alignment > kMinAlignment ||
      (reinterpret_cast<uintptr_t>(p) % alignment) != 0) {
    return nullptr;
  }
  if (!isAlignmentValid(newSize, alignment)) {
    VELOX_FAIL(
        "Alignment check failed, reallocateBytes {}, alignmentBytes {}",
        newSize,
        alignment);
  }
  const auto delta =
      static_cast<int64_t>(newSize) - static_cast<int64_t>(oldSize);
  if (delta > 0 && !incrementUsage(delta)) {
    auto errorMsg = fmt::format(
        "Failed to reallocate: exceeded memory allocator limit of {}. "
        "Old size: {}, new size: {}",
        succinctBytes(capacity_),
        succinctBytes(oldSize),
        succinctBytes(newSize));
    VELOX_MEM_LOG_EVERY_MS(WARNING, 1000) << errorMsg;
    setAllocatorFailureMessage(errorMsg);
    return nullptr;
  }
  void* result = ::realloc(p, newSize); // NOLINT
  if (result == nullptr) {
    // realloc failed. The original pointer is still valid.
    if (delta > 0) {
      decrementUsage(delta);
    }
    VELOX_MEM_LOG(ERROR) << "Failed to reallocateBytes from "
                         << succinctBytes(oldSize) << " to "
                         << succinctBytes(newSize);
    return nullptr;
  }
  // ::realloc() must return a pointer aligned to the requested alignment.
  // The C standard guarantees alignof(max_align_t), and we already enforced
  // alignment <= kMinAlignment above, so this should always hold for a
  // conformant allocator. Fail loudly if it doesn't (e.g., a non-conformant
  // LD_PRELOAD interposer).
  VELOX_CHECK_EQ(
      reinterpret_cast<uintptr_t>(result) % alignment,
      0,
      "::realloc returned a pointer not aligned to requested alignment: {}",
      alignment);
  if (delta < 0) {
    decrementUsage(-delta);
  }
  return result;
}

void MallocAllocator::freeBytes(void* p, uint64_t bytes) noexcept {
  ::free(p); // NOLINT
  decrementUsage(bytes);
}

bool MallocAllocator::checkConsistency() const {
  const auto allocatedBytes = allocatedBytes_.load();
  return allocatedBytes >= 0 && allocatedBytes <= capacity_;
}

std::string MallocAllocator::toString() const {
  std::stringstream out;
  out << "Memory Allocator[" << kindString(kind_) << " capacity "
      << ((capacity_ == kMaxMemory) ? "UNLIMITED" : succinctBytes(capacity_))
      << " allocated bytes " << allocatedBytes_ << " allocated pages "
      << numAllocated_ << " mapped pages " << numMapped_ << "]";
  return out.str();
}
} // namespace facebook::velox::memory
