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

#include <random>

#include "velox/exec/ExchangeClient.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exchange/GpuExchangeNode.h"

namespace facebook::velox::cudf_velox {

class GpuSerializedPage;

/// A SourceOperator that reads GPU data from an exchange. Similar to
/// exec::Exchange but returns CudfVector (wrapped in RowVector) instead of
/// deserializing from a CPU serde format.
///
/// Data path:
///   1. ExchangeClient::next() returns SerializedPageBase* pages.
///   2. If the page is a GpuSerializedPage, unwrap the CudfVector (zero-copy).
///   3. If the page is a regular (CPU) page, fall back to CPU deserialization
///      using the base Exchange logic.
///   4. If no data is available yet, return nullptr from getOutput().
class GpuExchange : public exec::SourceOperator {
 public:
  GpuExchange(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const GpuExchangeNode>& gpuExchangeNode,
      std::shared_ptr<exec::ExchangeClient> exchangeClient);

  ~GpuExchange() override {
    close();
  }

  RowVectorPtr getOutput() override;

  void close() override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 private:
  /// Fetches splits from the task and passes them to the exchange client.
  /// Only the first driver (driverId == 0) processes splits.
  void getSplits(ContinueFuture* future);

  /// Adds remote task IDs to the exchange client in shuffled order.
  void addRemoteTaskIds(std::vector<std::string>& remoteTaskIds);

  /// Fetches runtime stats from ExchangeClient and records them.
  void recordExchangeClientStats();

  const uint64_t preferredOutputBatchBytes_;

  /// True if this operator is responsible for fetching splits from the Task
  /// and passing these to ExchangeClient.
  const bool processSplits_;

  const int driverId_;

  bool noMoreSplits_ = false;

  std::shared_ptr<exec::ExchangeClient> exchangeClient_;

  /// A future received from Task::getSplitOrFuture(). It will be complete
  /// when there are more splits available or no-more-splits signal has
  /// arrived.
  ContinueFuture splitFuture_{ContinueFuture::makeEmpty()};

  /// Pages retrieved from ExchangeClient, pending processing.
  std::vector<std::unique_ptr<exec::SerializedPageBase>> currentPages_;

  /// Index of the next page to process in currentPages_.
  size_t currentPageIdx_{0};

  bool atEnd_{false};

  std::default_random_engine rng_{std::random_device{}()};
};

} // namespace facebook::velox::cudf_velox
