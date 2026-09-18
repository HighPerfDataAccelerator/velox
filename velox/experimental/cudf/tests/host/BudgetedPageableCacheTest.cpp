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

#include "velox/experimental/ucx-exchange/BudgetedPageableCache.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>

using facebook::velox::ucx_exchange::BudgetedPageableCache;
using Cache = BudgetedPageableCache;
struct Budget {
  std::atomic<int64_t> charged{0};
  std::shared_ptr<void> reserve(uint64_t bytes) {
    struct Ticket {
      Budget* budget;
      uint64_t bytes;
      ~Ticket() {
        budget->charged -= bytes;
      }
    };
    // Build before charging: allocation failure cannot leak budget.
    auto ticket = std::shared_ptr<Ticket>(new Ticket{this, 0});
    charged += bytes;
    ticket->bytes = bytes;
    return ticket;
  }
};

void bestFitAndCredit() {
  Budget b;
  auto cache = std::make_shared<Cache>(1024);
  auto c = b.reserve(256);
  auto large = cache->acquire(256, c);
  assert(!c && b.charged == 256);
  auto* first = large.get();
  std::memset(first, 42, 256);
  auto alias = large;
  large.reset();
  assert(cache->stats().idleBytes == 0); // Deferred consumer still owns it.
  c = b.reserve(128);
  auto small = cache->acquire(128, c);
  auto* second = small.get();
  assert(second != first && b.charged == 384);
  alias.reset();
  small.reset();
  c = b.reserve(100);
  auto reused = cache->acquire(100, c);
  assert(reused.get() == second); // Best fit, not oldest/largest.
  assert(b.charged == 384 && !c); // FULL 128-byte credit survives reuse.
  assert(cache->evictAtLeast(1) == 256 && b.charged == 128);
  assert(reused.get() == second);
  reused.reset();
  assert(cache->stats().hits == 1 && cache->stats().hitPayloadBytes == 100);
  cache.reset();
  assert(b.charged == 0);
}

void boundsAndLastOwner() {
  Budget b;
  auto cache = std::make_shared<Cache>(256, 1);
  auto c = b.reserve(128), d = b.reserve(128), e = b.reserve(512);
  auto x = cache->acquire(128, c), y = cache->acquire(128, d);
  auto oversized = cache->acquire(512, e);
  x.reset();
  y.reset();
  oversized.reset();
  assert(cache->stats().idleBlocks == 1 && b.charged == 128);
  assert(cache->evictAtLeast(4096) == 128 && b.charged == 0);
  c = b.reserve(128);
  auto last = cache->acquire(128, c);
  std::weak_ptr<Cache> weak = cache;
  cache.reset();
  assert(!weak.expired() && b.charged == 128);
  last.reset();
  assert(weak.expired() && b.charged == 0);
}

std::unique_ptr<uint8_t[]> failAllocate(uint64_t) {
  throw std::bad_alloc();
}
void failures() {
  Budget b;
  auto cache = std::make_shared<Cache>(256, 1, failAllocate);
  auto c = b.reserve(128);
  bool failed = false;
  try {
    cache->acquire(128, c);
  } catch (const std::bad_alloc&) {
    failed = true;
  }
  assert(failed && c && b.charged == 128 && cache->stats().idleBytes == 0);
  c.reset();
  assert(b.charged == 0);
  failed = false;
  try {
    cache->acquire(0, c);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  assert(failed);
}

void pressureAndConcurrentOwners() {
  Budget b;
  auto cache = std::make_shared<Cache>(16 * 1024, 8);
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&, thread] {
      for (int step = 0; step < 2000; ++step) {
        const auto size = 1024 + (step % 7) * 128;
        auto c = b.reserve(size);
        auto data = cache->acquire(size, c);
        std::memset(data.get(), thread + 1, size);
        std::this_thread::yield();
        for (int i = 0; i < size; ++i)
          assert(data.get()[i] == thread + 1);
        if (step % 13 == 0)
          cache->evictAtLeast(2048);
      }
    });
  }
  for (auto& t : threads)
    t.join();
  auto s = cache->stats();
  assert(s.requests == 16000 && s.hits > 0);
  assert(s.peakIdleBytes <= 16 * 1024 && s.idleBlocks <= 8);
  assert(b.charged == s.idleBytes);
  cache->evictAtLeast(UINT64_MAX);
  assert(b.charged == 0);
}

int main() {
  bestFitAndCredit();
  boundsAndLastOwner();
  failures();
  pressureAndConcurrentOwners();
  std::cout
      << "PASS: best fit, full-capacity credit, final owner, bounds, failure, concurrent integrity\n";
}
