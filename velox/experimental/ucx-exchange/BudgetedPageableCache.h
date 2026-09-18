/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace facebook::velox::ucx_exchange {

// A budget token charges the FULL backing allocation, not the latest request
// size. The token survives active and idle ownership; eviction deletes memory
// before releasing it. The caller must reserve from its existing shared budget
// before acquire(), and evict idle blocks when that reservation cannot fit.
class BudgetedPageableCache
    : public std::enable_shared_from_this<BudgetedPageableCache> {
 public:
  struct Stats {
    uint64_t requests{0}, hits{0}, hitPayloadBytes{0}, allocatedBytes{0};
    uint64_t idleBytes{0}, peakIdleBytes{0}, evictedBytes{0}, idleBlocks{0};
  };
  using Allocator = std::unique_ptr<uint8_t[]> (*)(uint64_t);

  explicit BudgetedPageableCache(
      uint64_t maxIdleBytes,
      uint64_t maxIdleBlocks = 64,
      Allocator allocate = allocateUninitialized)
      : maxIdleBytes_(maxIdleBytes),
        maxIdleBlocks_(maxIdleBlocks),
        allocate_(allocate) {
    if (!maxIdleBlocks || !allocate) {
      throw std::invalid_argument("Invalid pageable cache configuration");
    }
  }

  std::shared_ptr<uint8_t> acquire(
      uint64_t bytes,
      std::shared_ptr<void>& requestCredit) {
    if (!bytes || !requestCredit) {
      throw std::invalid_argument("Pageable cache requires bytes and budget");
    }
    std::unique_ptr<Block> block;
    {
      std::lock_guard lock(mutex_);
      auto best = idle_.end();
      for (auto it = idle_.begin(); it != idle_.end(); ++it) {
        if ((*it)->capacity >= bytes &&
            (best == idle_.end() || (*it)->capacity < (*best)->capacity)) {
          best = it;
        }
      }
      ++stats_.requests;
      if (best != idle_.end()) {
        block = std::move(*best);
        idle_.erase(best);
        stats_.idleBytes -= block->capacity;
        ++stats_.hits;
        stats_.hitPayloadBytes += bytes;
      }
    }
    if (!block) {
      block = std::make_unique<Block>();
      block->credit = requestCredit;
      block->data = allocate_(bytes); // No prefault/zeroing or hidden warmup.
      if (!block->data) {
        throw std::bad_alloc();
      }
      block->capacity = bytes;
      std::lock_guard lock(mutex_);
      stats_.allocatedBytes += bytes;
    }
    auto lease = std::make_shared<Lease>(shared_from_this(), std::move(block));
    auto result = std::shared_ptr<uint8_t>(lease, lease->block->data.get());
    // Miss: the block now owns this token. Hit: retain the old block's FULL
    // token, release the new request's redundant token. Never re-charge a
    // large reused allocation as only the smaller request size.
    requestCredit.reset();
    return result;
  }

  uint64_t evictAtLeast(uint64_t bytes) {
    uint64_t evicted = 0;
    while (evicted < bytes) {
      std::unique_ptr<Block> block;
      {
        std::lock_guard lock(mutex_);
        if (idle_.empty()) {
          break;
        }
        block = std::move(idle_.front());
        idle_.erase(idle_.begin());
        stats_.idleBytes -= block->capacity;
        stats_.evictedBytes += block->capacity;
      }
      evicted += block->capacity;
      block.reset(); // No destructor/budget callback under the cache mutex.
    }
    return evicted;
  }

  Stats stats() const {
    std::lock_guard lock(mutex_);
    auto result = stats_;
    result.idleBlocks = idle_.size();
    return result;
  }

 private:
  static std::unique_ptr<uint8_t[]> allocateUninitialized(uint64_t bytes) {
    return std::unique_ptr<uint8_t[]>(new uint8_t[bytes]);
  }

  struct Block {
    // Reverse member destruction order is intentional: memory before credit.
    std::shared_ptr<void> credit;
    std::unique_ptr<uint8_t[]> data;
    uint64_t capacity{0};
  };
  struct Lease {
    Lease(
        std::shared_ptr<BudgetedPageableCache> owner,
        std::unique_ptr<Block> value)
        : cache(std::move(owner)), block(std::move(value)) {}
    ~Lease() {
      cache->recycle(std::move(block));
    }
    std::shared_ptr<BudgetedPageableCache> cache;
    std::unique_ptr<Block> block;
  };

  void recycle(std::unique_ptr<Block> block) noexcept {
    try {
      std::lock_guard lock(mutex_);
      if (block->capacity <= maxIdleBytes_ - stats_.idleBytes &&
          idle_.size() < maxIdleBlocks_) {
        const auto bytes = block->capacity;
        idle_.push_back(std::move(block));
        stats_.idleBytes += bytes;
        stats_.peakIdleBytes = std::max(stats_.peakIdleBytes, stats_.idleBytes);
      }
    } catch (...) {
      // Failed cache bookkeeping must not throw from an owner's destructor.
      // The still-owned block is discarded after the mutex unlocks.
    }
  }

  const uint64_t maxIdleBytes_, maxIdleBlocks_;
  const Allocator allocate_;
  mutable std::mutex mutex_;
  Stats stats_;
  std::vector<std::unique_ptr<Block>> idle_;
};

} // namespace facebook::velox::ucx_exchange
