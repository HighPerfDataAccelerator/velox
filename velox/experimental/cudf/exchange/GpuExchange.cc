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
#include "velox/experimental/cudf/exchange/GpuExchange.h"

#include "velox/common/Casts.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cudf/exchange/GpuSerializedPage.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

GpuExchange::GpuExchange(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const GpuExchangeNode>& gpuExchangeNode,
    std::shared_ptr<exec::ExchangeClient> exchangeClient)
    : SourceOperator(
          driverCtx,
          gpuExchangeNode->outputType(),
          operatorId,
          gpuExchangeNode->id(),
          "GpuExchange"),
      preferredOutputBatchBytes_{
          driverCtx->queryConfig().preferredOutputBatchBytes()},
      processSplits_{operatorCtx_->driverCtx()->driverId == 0},
      driverId_{driverCtx->driverId},
      exchangeClient_{std::move(exchangeClient)} {}

void GpuExchange::addRemoteTaskIds(std::vector<std::string>& remoteTaskIds) {
  std::shuffle(std::begin(remoteTaskIds), std::end(remoteTaskIds), rng_);
  for (const std::string& taskId : remoteTaskIds) {
    exchangeClient_->addRemoteTaskId(taskId);
  }
  stats_.wlock()->numSplits += remoteTaskIds.size();
}

void GpuExchange::getSplits(ContinueFuture* future) {
  if (!processSplits_) {
    return;
  }
  if (noMoreSplits_) {
    return;
  }
  std::vector<std::string> remoteTaskIds;
  for (;;) {
    exec::Split split;
    const auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId(), split, *future);
    if (reason != exec::BlockingReason::kNotBlocked) {
      addRemoteTaskIds(remoteTaskIds);
      return;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit =
          checkedPointerCast<exec::RemoteConnectorSplit>(split.connectorSplit);
      remoteTaskIds.push_back(remoteSplit->taskId);
      continue;
    }

    addRemoteTaskIds(remoteTaskIds);
    exchangeClient_->noMoreRemoteTasks();
    noMoreSplits_ = true;
    if (atEnd_) {
      operatorCtx_->task()->multipleSplitsFinished(
          false, stats_.rlock()->numSplits, 0);
      recordExchangeClientStats();
    }
    return;
  }
}

exec::BlockingReason GpuExchange::isBlocked(ContinueFuture* future) {
  if (currentPageIdx_ < currentPages_.size() || atEnd_) {
    return exec::BlockingReason::kNotBlocked;
  }

  // Start fetching data right away. Do not wait for all splits to be
  // available.
  if (!splitFuture_.valid()) {
    getSplits(&splitFuture_);
  }

  ContinueFuture dataFuture;
  currentPageIdx_ = 0;
  currentPages_ = exchangeClient_->next(
      driverId_, preferredOutputBatchBytes_, &atEnd_, &dataFuture);
  if (!currentPages_.empty() || atEnd_) {
    if (atEnd_ && noMoreSplits_) {
      const auto numSplits = stats_.rlock()->numSplits;
      operatorCtx_->task()->multipleSplitsFinished(false, numSplits, 0);
    }
    recordExchangeClientStats();
    return exec::BlockingReason::kNotBlocked;
  }

  // We have a dataFuture and we may also have a splitFuture_.
  if (splitFuture_.valid()) {
    // Block until data becomes available or more splits arrive.
    std::vector<ContinueFuture> futures;
    futures.push_back(std::move(splitFuture_));
    futures.push_back(std::move(dataFuture));
    *future = folly::collectAny(futures).unit();
    return exec::BlockingReason::kWaitForSplit;
  }

  // Block until data becomes available.
  VELOX_CHECK(dataFuture.valid());
  *future = std::move(dataFuture);
  return exec::BlockingReason::kWaitForProducer;
}

bool GpuExchange::isFinished() {
  return atEnd_ && currentPageIdx_ >= currentPages_.size();
}

RowVectorPtr GpuExchange::getOutput() {
  if (currentPages_.empty()) {
    return nullptr;
  }

  // Process one page per getOutput() call. Each GpuSerializedPage contains
  // a complete CudfVector that we unwrap with zero copy. Remaining pages
  // are left in currentPages_ for subsequent getOutput() calls; isBlocked()
  // will see the non-empty vector and return kNotBlocked immediately.
  while (currentPageIdx_ < currentPages_.size()) {
    auto& page = currentPages_[currentPageIdx_];
    ++currentPageIdx_;

    if (!page) {
      continue;
    }

    // GPU fast path: if the page is a GpuSerializedPage, unwrap the
    // CudfVector directly (zero-copy).
    auto* gpuPage = dynamic_cast<GpuSerializedPage*>(page.get());
    if (gpuPage != nullptr) {
      const uint64_t rawInputBytes = gpuPage->size();
      auto result = gpuPage->cudfVector();

      // Release the page now that we have taken its vector.
      page.reset();

      // If we've consumed all pages, release the vector.
      if (currentPageIdx_ >= currentPages_.size()) {
        currentPages_.clear();
        currentPageIdx_ = 0;
      }

      // Record stats.
      {
        auto lockedStats = stats_.wlock();
        lockedStats->rawInputBytes += rawInputBytes;
        lockedStats->rawInputPositions += result->size();
        lockedStats->addInputVector(result->estimateFlatSize(), result->size());
      }

      return result;
    }

    // Fallback: CPU page. This should be rare in a GPU exchange but can
    // happen if the upstream mixes CPU and GPU pages. Throw unsupported
    // for now -- a real fallback would require CPU serde machinery.
    VELOX_UNSUPPORTED(
        "GpuExchange received a non-GPU page. "
        "CPU deserialization fallback is not yet implemented.");
  }

  // All remaining pages were null (should not normally happen).
  currentPages_.clear();
  currentPageIdx_ = 0;
  return nullptr;
}

void GpuExchange::close() {
  exec::SourceOperator::close();
  currentPages_.clear();
  currentPageIdx_ = 0;

  if (exchangeClient_) {
    recordExchangeClientStats();
    exchangeClient_->close();
  }
  exchangeClient_ = nullptr;
}

void GpuExchange::recordExchangeClientStats() {
  if (!processSplits_) {
    return;
  }

  auto lockedStats = stats_.wlock();
  const auto exchangeClientStats = exchangeClient_->stats();
  for (const auto& [name, value] : exchangeClientStats) {
    lockedStats->runtimeStats.erase(name);
    lockedStats->runtimeStats.insert({name, value});
  }

  const auto iter =
      exchangeClientStats.find(exec::Operator::kBackgroundCpuTimeNanos);
  if (iter != exchangeClientStats.end()) {
    const CpuWallTiming backgroundTiming{
        static_cast<uint64_t>(iter->second.count),
        0,
        static_cast<uint64_t>(iter->second.sum)};
    lockedStats->backgroundTiming.clear();
    lockedStats->backgroundTiming.add(backgroundTiming);
  }
}

} // namespace facebook::velox::cudf_velox
