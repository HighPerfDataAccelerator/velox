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

#include "velox/experimental/cudf/connectors/hive/BoundedBatchWriter.h"

#include <atomic>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>

using facebook::velox::cudf_velox::connector::hive::BoundedBatchWriter;
using namespace std::chrono_literals;
void require(bool value) {
  if (!value)
    throw std::runtime_error("batch writer assertion failed");
}
template <typename F>
void throws(F action) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed);
}

int main() {
  {
    // The first small input is consumed without waiting for another. While
    // that callback is held, three ready inputs must become one next batch.
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    std::vector<std::vector<int>> observed;
    BoundedBatchWriter<int> queue(4, 4, [&](auto& batch) {
      observed.push_back(batch);
      if (observed.size() == 1) {
        entered.set_value();
        gate.wait();
      }
    });
    queue.submit(0, 1);
    entered.get_future().get();
    queue.submit(1, 1);
    queue.submit(2, 1);
    queue.submit(3, 1);
    auto fifth = std::async(std::launch::async, [&] { queue.submit(4, 1); });
    const bool blocked = fifth.wait_for(30ms) == std::future_status::timeout;
    release.set_value();
    fifth.get();
    queue.drain();
    require(blocked);
    require(observed.size() >= 2);
    require(observed[0] == std::vector<int>{0});
    require(observed[1].size() >= 3);
    std::vector<int> all;
    for (auto& batch : observed)
      all.insert(all.end(), batch.begin(), batch.end());
    require(all == std::vector<int>({0, 1, 2, 3, 4}));
    require(queue.stats().peakBytes == 4 && queue.stats().peakItems == 4);
    require(queue.stats().inputs == 5);
    throws([&] { queue.submit(9, 5); });
    queue.submit(5, 4);
    queue.drain(); // Oversize rejection did not poison it.
  }
  {
    // Item bound still applies to zero-byte inputs; byte bound includes the
    // active callback, not just the waiting queue.
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    std::atomic<int> calls{0};
    BoundedBatchWriter<int> queue(8, 2, [&](auto&) {
      if (++calls == 1) {
        entered.set_value();
        gate.wait();
      }
    });
    queue.submit(0, 8);
    entered.get_future().get();
    queue.submit(1, 0);
    auto more = std::async(std::launch::async, [&] { queue.submit(2, 0); });
    const bool blocked = more.wait_for(30ms) == std::future_status::timeout;
    release.set_value();
    more.get();
    queue.drain();
    require(
        blocked && queue.stats().peakItems == 2 &&
        queue.stats().peakBytes == 8);
  }
  {
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    BoundedBatchWriter<std::shared_ptr<int>> queue(2, 2, [&](auto&) {
      entered.set_value();
      gate.wait();
      throw std::runtime_error("write failure");
    });
    auto first = std::make_shared<int>(0), second = std::make_shared<int>(1);
    std::weak_ptr<int> firstLife = first, secondLife = second;
    queue.submit(std::move(first), 1);
    entered.get_future().get();
    queue.submit(std::move(second), 1);
    auto blocked = std::async(std::launch::async, [&] {
      throws([&] { queue.submit(std::make_shared<int>(2), 1); });
    });
    const bool waited = blocked.wait_for(30ms) == std::future_status::timeout;
    release.set_value();
    blocked.get();
    throws([&] { queue.drain(); });
    require(waited && firstLife.expired() && secondLife.expired());
    throws([&] { queue.submit({}, 0); });
  }
  {
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    std::atomic<int> consumed{0};
    auto queue =
        std::make_unique<BoundedBatchWriter<int>>(3, 3, [&](auto& batch) {
          if (consumed == 0) {
            entered.set_value();
            gate.wait();
          }
          consumed += batch.size();
        });
    queue->submit(1, 1);
    entered.get_future().get();
    queue->submit(2, 1);
    auto destroy = std::async(std::launch::async, [&] { queue.reset(); });
    const bool waited = destroy.wait_for(30ms) == std::future_status::timeout;
    release.set_value();
    destroy.get();
    require(waited && consumed == 2);
  }
  throws([] { BoundedBatchWriter<int> bad(0, 1, [](auto&) {}); });
  std::cout
      << "PASS immediate progress, ready-input batching, FIFO, active+queued "
         "byte/item bounds, oversize, zero bytes, failure wakeup, owner release, "
         "destructor draining\n";
}
