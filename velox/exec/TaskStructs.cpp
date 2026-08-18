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

#include "velox/exec/TaskStructs.h"

namespace facebook::velox::exec {

void SplitsStore::addSplit(
    Split split,
    std::vector<ContinuePromise>& promises) {
  VELOX_CHECK(!noMoreSplits_);
  VELOX_CHECK(!(remoteSplit_ && split.isBarrier()));
  VELOX_CHECK(barrierSplits_.empty());
  if (split.isBarrier()) {
    for (auto i = 0; i < split.barrier->numDrivers; ++i) {
      barrierSplits_[i] = Split::createBarrier();
    }
    VELOX_CHECK_LE(promises_.size(), split.barrier->numDrivers);
    // A barrier is assigned to every driver; wake up all currently blocked
    // drivers to process it.
    std::move(promises_.begin(), promises_.end(), std::back_inserter(promises));
    promises_.clear();
  } else {
    splits_.push_back(std::move(split));
    if (!promises_.empty()) {
      promises.push_back(std::move(promises_.back()));
      promises_.pop_back();
    }
  }
}

ContinueFuture SplitsStore::makeFuture() {
  auto [promise, future] =
      makeVeloxContinuePromiseContract("SplitsStore::makeFuture");
  promises_.push_back(std::move(promise));
  return std::move(future);
}

bool SplitsStore::getSplit(
    int maxPreloadSplits,
    const ConnectorSplitPreloadFunc& preload,
    Split& split) {
  int readySplitIndex = -1;
  if (maxPreloadSplits > 0) {
    maxPreloadSplits = preloadSplits(maxPreloadSplits, preload);
    for (int i = 0, end = std::min<size_t>(maxPreloadSplits, splits_.size());
         i < end;
         ++i) {
      if (splits_[i].isBarrier()) {
        VELOX_CHECK(!remoteSplit_);
        continue;
      }
      auto& connectorSplit = splits_[i].connectorSplit;
      if (readySplitIndex == -1 && connectorSplit->dataSource->hasValue()) {
        readySplitIndex = i;
        preloadingSplits_->erase(connectorSplit);
      }
    }
    // Do not bind a scan driver to an arbitrary in-flight preload. The
    // completion path wakes a waiter, which retries and takes whichever split
    // is ready first. This keeps I/O and compute pipelined without
    // head-of-line blocking on the queue front.
    if (readySplitIndex == -1) {
      return false;
    }
  }
  if (readySplitIndex == -1) {
    readySplitIndex = 0;
  }
  VELOX_CHECK(!splits_.empty());
  split = std::move(splits_[readySplitIndex]);
  splits_.erase(splits_.begin() + readySplitIndex);
  --taskStats_->numQueuedSplits;
  ++taskStats_->numRunningSplits;
  if (!remoteSplit_ && split.connectorSplit) {
    --taskStats_->numQueuedTableScanSplits;
    ++taskStats_->numRunningTableScanSplits;
    taskStats_->queuedTableScanSplitWeights -=
        split.connectorSplit->splitWeight;
    taskStats_->runningTableScanSplitWeights +=
        split.connectorSplit->splitWeight;
  }
  taskStats_->lastSplitStartTimeMs = getCurrentTimeMs();
  if (taskStats_->firstSplitStartTimeMs == 0) {
    taskStats_->firstSplitStartTimeMs = taskStats_->lastSplitStartTimeMs;
  }
  return true;
}

int SplitsStore::preloadSplits(
    int maxPreloadSplits,
    const ConnectorSplitPreloadFunc& preload) {
  if (maxPreloadSplits <= 0 || !preload) {
    return 0;
  }
  VELOX_CHECK_NOT_NULL(preloadingSplits_);
  for (int i = 0, end = std::min<size_t>(maxPreloadSplits, splits_.size());
       i < end;
       ++i) {
    if (splits_[i].isBarrier()) {
      VELOX_CHECK(!remoteSplit_);
      continue;
    }
    auto& connectorSplit = splits_[i].connectorSplit;
    if (!connectorSplit->dataSource) {
      preload(connectorSplit);
      preloadingSplits_->insert(connectorSplit);
    }
  }
  return maxPreloadSplits;
}

bool SplitsStore::tryGetBarrier(
    std::optional<uint32_t> driverId,
    Split& split) {
  if (!driverId.has_value()) {
    barrierSplits_.clear();
    return false;
  }
  // Delivers a barrier exactly once for each driver from the same plan node.
  auto it = barrierSplits_.find(*driverId);
  if (it == barrierSplits_.end()) {
    return false;
  }
  split = it->second;
  barrierSplits_.erase(it);
  return true;
}

} // namespace facebook::velox::exec
