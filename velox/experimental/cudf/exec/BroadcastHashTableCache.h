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

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>

#include <glog/logging.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

// Process-wide cache for broadcast hash tables shared across Spark tasks.
//
// In broadcast hash join, every Spark task independently builds the same
// hash table from identical broadcast data. This cache ensures only one
// task (the leader) builds the hash table; all other tasks wait and
// reuse the result.
//
// Leader election uses a token pattern:
//   - First task to call acquire() for a given key becomes the leader.
//   - The leader must call complete() after building, or fail() on error.
//   - Other tasks block in acquire() until the leader finishes.
//   - If the leader fails, one waiting task is promoted to new leader.
//
// Cache key: (planNodeId, first-column device pointer). With Level 1
// broadcast sharing, all tasks point to the same GPU data, so their
// keys match. Non-broadcast joins have different data per task, so
// keys differ and the cache is bypassed naturally.
class BroadcastHashTableCache {
 public:
  using hash_type = std::pair<
      std::vector<std::shared_ptr<cudf::table>>,
      std::vector<std::shared_ptr<cudf::hash_join>>>;

  static BroadcastHashTableCache& instance() {
    static BroadcastHashTableCache cache;
    return cache;
  }

  struct AcquireResult {
    // Set when cached hash table is available (follower path).
    std::optional<hash_type> hashTable;
    // True when this caller must build and call complete()/fail().
    bool isLeader;
  };

  AcquireResult acquire(const std::string& key) {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(mapMutex_);
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        entry = std::make_shared<Entry>();
        entry->building = true;
        entries_[key] = entry;
        LOG(WARNING) << "BroadcastHashTableCache: leader acquired key=" << key;
        return {std::nullopt, true};
      }
      entry = it->second;
    }
    // Entry exists -- wait for leader to finish, fail, or cancel.
    std::unique_lock<std::mutex> lock(entry->mtx);
    entry->cv.wait(lock, [&] {
      return !entry->building || entry->cancelled;
    });
    if (entry->cancelled) {
      // Cache was cleared (e.g. SQL execution ended). Caller
      // falls through to independent build without using cache.
      LOG(WARNING) << "BroadcastHashTableCache: cancelled, independent build"
                   " key=" << key;
      return {std::nullopt, true};
    }
    if (entry->ready) {
      LOG(WARNING) << "BroadcastHashTableCache: follower reusing key=" << key;
      return {entry->hashTable, false};
    }
    // Leader failed -- this thread becomes the new leader.
    entry->building = true;
    LOG(WARNING) << "BroadcastHashTableCache: new leader after failure key="
              << key;
    return {std::nullopt, true};
  }

  void complete(const std::string& key, hash_type result) {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(mapMutex_);
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        return;
      }
      entry = it->second;
    }
    {
      std::lock_guard<std::mutex> lock(entry->mtx);
      entry->hashTable = std::move(result);
      entry->ready = true;
      entry->building = false;
    }
    entry->cv.notify_all();
    LOG(WARNING) << "BroadcastHashTableCache: build complete key=" << key;
  }

  void fail(const std::string& key) {
    std::shared_ptr<Entry> entry;
    {
      std::lock_guard<std::mutex> lock(mapMutex_);
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        return;
      }
      entry = it->second;
    }
    {
      std::lock_guard<std::mutex> lock(entry->mtx);
      entry->building = false;
      // Do not set ready -- next waiter becomes leader.
    }
    entry->cv.notify_one();
    LOG(WARNING) << "BroadcastHashTableCache: leader failed key=" << key;
  }

  void clear() {
    std::vector<std::shared_ptr<Entry>> snapshot;
    {
      std::lock_guard<std::mutex> lock(mapMutex_);
      snapshot.reserve(entries_.size());
      for (auto& [k, v] : entries_) {
        snapshot.push_back(v);
      }
      entries_.clear();
    }
    // Wake any threads blocked in acquire(). They will see
    // cancelled=true and fall through to independent build.
    for (auto& entry : snapshot) {
      std::lock_guard<std::mutex> lock(entry->mtx);
      entry->cancelled = true;
      entry->building = false;
      entry->cv.notify_all();
    }
    if (!snapshot.empty()) {
      LOG(WARNING) << "BroadcastHashTableCache: cleared " << snapshot.size()
                << " entries";
    }
  }

 private:
  struct Entry {
    std::mutex mtx;
    std::condition_variable cv;
    std::optional<hash_type> hashTable;
    bool building = false;
    bool ready = false;
    bool cancelled = false;
  };

  std::mutex mapMutex_;
  std::unordered_map<std::string, std::shared_ptr<Entry>> entries_;
};

// RAII guard for leader builds. Calls fail() if not completed
// before destruction (exception safety).
class BroadcastBuildGuard {
 public:
  explicit BroadcastBuildGuard(std::string key)
      : key_(std::move(key)), completed_(false) {}

  ~BroadcastBuildGuard() {
    if (!completed_) {
      BroadcastHashTableCache::instance().fail(key_);
    }
  }

  BroadcastBuildGuard(const BroadcastBuildGuard&) = delete;
  BroadcastBuildGuard& operator=(const BroadcastBuildGuard&) = delete;

  void complete(BroadcastHashTableCache::hash_type result) {
    BroadcastHashTableCache::instance().complete(key_, std::move(result));
    completed_ = true;
  }

 private:
  std::string key_;
  bool completed_;
};

} // namespace facebook::velox::cudf_velox
