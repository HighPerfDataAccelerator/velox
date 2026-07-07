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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// GPU-native ordered UCX exchange.
///
/// A private UcxExchangeClient is created for every remote producer. Keeping
/// the clients separate preserves source identity, per-source order, EOS, and
/// backpressure without changing the unordered UCX exchange path.
///
/// Each producer must emit one monotonic stream sorted using the ordering from
/// MergeExchangeNode. Once every unfinished source has a head batch, the
/// operator emits a globally safe prefix:
///
///   frontier = min(last-row(head[i]))
///   split[i] = upper_bound(head[i], frontier)
///   output = merge(head[i][0:split[i]])
///
/// At least one head is consumed per round. Remaining suffixes stay attached
/// to their source for the next round.
class CudfMergeExchange : public exec::SourceOperator, public NvtxHelper {
 public:
  CudfMergeExchange(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::MergeExchangeNode> mergeExchangeNode);

  ~CudfMergeExchange() override;

  [[nodiscard]] exec::BlockingReason isBlocked(ContinueFuture* future) override;

  [[nodiscard]] bool isFinished() override;

  [[nodiscard]] RowVectorPtr getOutput() override;

  void close() override;

 private:
  struct SourceState {
    std::string remoteTaskId;
    std::shared_ptr<ucx_exchange::UcxExchangeClient> client;
    CudfVectorPtr head;
    cudf::size_type beginOffset{0};
    bool atEnd{false};

    // One copied key row: the last row of the most recently received batch.
    // This is used to validate monotonicity across batch boundaries without
    // retaining the entire preceding batch.
    std::unique_ptr<cudf::table> previousLastKey;
  };

  exec::BlockingReason collectSplits(ContinueFuture* future);

  void initializeSources();

  /// Fetches a head for each unfinished source. Returns true when all sources
  /// have either a non-empty head or EOS. Futures for missing heads are added
  /// to 'futures'.
  bool fetchMissingHeads(std::vector<ContinueFuture>& futures);

  CudfVectorPtr makeVector(
      ucx_exchange::PackedTableWithStreamPtr data,
      SourceState& source);

  void validateAndRememberHead(SourceState& source);

  RowVectorPtr mergeSafePrefix(const std::vector<size_t>& activeSources);

  RowVectorPtr passThroughSingleSource(SourceState& source);

  bool allSourcesAtEnd() const;

  const std::shared_ptr<const core::MergeExchangeNode> mergeExchangeNode_;
  std::vector<std::string> remoteTaskIds_;
  std::vector<SourceState> sources_;

  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::size_type> normalizedSortKeys_;
  std::vector<cudf::order> columnOrder_;
  std::vector<cudf::null_order> nullOrder_;

  rmm::cuda_stream_view mergeStream_;

  bool sourcesInitialized_{false};
  bool noMoreSplits_{false};
  bool finished_{false};
  bool closed_{false};
};

} // namespace facebook::velox::cudf_velox
