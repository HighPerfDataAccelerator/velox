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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfMergeExchange.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/Exchange.h"
#include "velox/exec/Task.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/merge.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <fmt/format.h>

#include <algorithm>
#include <utility>

namespace facebook::velox::cudf_velox {

namespace {

std::vector<cudf::table_view> singleRowSlices(
    const std::vector<cudf::table_view>& tables,
    bool first,
    rmm::cuda_stream_view stream) {
  std::vector<cudf::table_view> rows;
  rows.reserve(tables.size());
  for (const auto& table : tables) {
    VELOX_CHECK_GT(table.num_rows(), 0);
    const auto row = first ? 0 : table.num_rows() - 1;
    auto slices = cudf::slice(table, {row, row + 1}, stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    rows.push_back(slices.front());
  }
  return rows;
}

} // namespace

CudfMergeExchange::CudfMergeExchange(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::MergeExchangeNode> mergeExchangeNode)
    : SourceOperator(
          driverCtx,
          mergeExchangeNode->outputType(),
          operatorId,
          mergeExchangeNode->id(),
          "CudfMergeExchange"),
      NvtxHelper(
          nvtx3::rgb{0, 191, 255},
          operatorId,
          fmt::format("[{}]", mergeExchangeNode->id())),
      mergeExchangeNode_(std::move(mergeExchangeNode)),
      mergeStream_(cudfGlobalStreamPool().get_stream()) {
  VELOX_CHECK_NOT_NULL(mergeExchangeNode_);
  VELOX_CHECK_EQ(
      operatorCtx_->driverCtx()->driverId,
      0,
      "MergeExchange must execute with one driver");
  VELOX_CHECK_EQ(
      mergeExchangeNode_->sortingKeys().size(),
      mergeExchangeNode_->sortingOrders().size());
  VELOX_CHECK_GT(
      mergeExchangeNode_->sortingKeys().size(),
      0,
      "MergeExchange requires at least one sorting key");

  sortKeys_.reserve(mergeExchangeNode_->sortingKeys().size());
  normalizedSortKeys_.reserve(mergeExchangeNode_->sortingKeys().size());
  columnOrder_.reserve(mergeExchangeNode_->sortingKeys().size());
  nullOrder_.reserve(mergeExchangeNode_->sortingKeys().size());

  for (size_t i = 0; i < mergeExchangeNode_->sortingKeys().size(); ++i) {
    const auto channel = exec::exprToChannel(
        mergeExchangeNode_->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK_NE(
        channel,
        kConstantChannel,
        "MergeExchange doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    normalizedSortKeys_.push_back(static_cast<cudf::size_type>(i));

    const auto& order = mergeExchangeNode_->sortingOrders()[i];
    columnOrder_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    // Match CudfOrderBy's conversion. cuDF null precedence is interpreted
    // relative to the physical column order.
    nullOrder_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

CudfMergeExchange::~CudfMergeExchange() {
  close();
}

exec::BlockingReason CudfMergeExchange::collectSplits(ContinueFuture* future) {
  if (noMoreSplits_) {
    return exec::BlockingReason::kNotBlocked;
  }

  for (;;) {
    exec::Split split;
    auto reason = operatorCtx_->task()->getSplitOrFuture(
        operatorCtx_->driverCtx()->driverId,
        operatorCtx_->driverCtx()->splitGroupId,
        planNodeId(),
        /*maxPreloadSplits=*/0,
        /*preload=*/nullptr,
        split,
        *future);
    if (reason != exec::BlockingReason::kNotBlocked) {
      return reason;
    }

    if (split.hasConnectorSplit()) {
      auto remoteSplit = std::dynamic_pointer_cast<exec::RemoteConnectorSplit>(
          split.connectorSplit);
      VELOX_CHECK_NOT_NULL(remoteSplit, "Wrong split type for MergeExchange");
      remoteTaskIds_.push_back(remoteSplit->taskId);
      continue;
    }

    noMoreSplits_ = true;
    initializeSources();
    operatorCtx_->task()->multipleSplitsFinished(
        false, remoteTaskIds_.size(), 0);
    return exec::BlockingReason::kNotBlocked;
  }
}

void CudfMergeExchange::initializeSources() {
  VELOX_CHECK(noMoreSplits_);
  VELOX_CHECK(!sourcesInitialized_);
  sourcesInitialized_ = true;
  sources_.reserve(remoteTaskIds_.size());

  auto task = operatorCtx_->task();
  for (const auto& remoteTaskId : remoteTaskIds_) {
    auto client = std::make_shared<ucx_exchange::UcxExchangeClient>(
        task->taskId(), task->destination(), 1);
    client->addRemoteTaskId(remoteTaskId);
    client->noMoreRemoteTasks();

    SourceState source;
    source.remoteTaskId = remoteTaskId;
    source.client = std::move(client);
    sources_.push_back(std::move(source));
  }
  stats_.wlock()->numSplits += remoteTaskIds_.size();
}

CudfVectorPtr CudfMergeExchange::makeVector(
    ucx_exchange::PackedTableWithStreamPtr data,
    SourceState& source) {
  VELOX_CHECK_NOT_NULL(data);
  VELOX_CHECK_NOT_NULL(data->packedTable);
  const auto numRows = data->packedTable->table.num_rows();
  if (numRows == 0) {
    return nullptr;
  }

  auto vector = std::make_shared<CudfVector>(
      pool(), outputType_, numRows, std::move(data->packedTable), data->stream);

  {
    auto lockedStats = stats_.wlock();
    lockedStats->rawInputPositions += numRows;
    lockedStats->rawInputBytes += vector->estimateFlatSize();
    lockedStats->addInputVector(vector->estimateFlatSize(), numRows);
  }

  source.beginOffset = 0;
  return vector;
}

void CudfMergeExchange::validateAndRememberHead(SourceState& source) {
  VELOX_CHECK_NOT_NULL(source.head);
  const auto table = source.head->getTableView();
  VELOX_CHECK_GT(table.num_rows(), 0);

  const std::vector<rmm::cuda_stream_view> inputStreams{source.head->stream()};
  cudf::detail::join_streams(inputStreams, mergeStream_);

  const auto keys = table.select(sortKeys_);
  VELOX_CHECK(
      cudf::is_sorted(keys, columnOrder_, nullOrder_, mergeStream_),
      "MERGE_SINGLE producer {} emitted a batch that is not sorted",
      source.remoteTaskId);

  auto firstRows = singleRowSlices({keys}, true, mergeStream_);
  if (source.previousLastKey) {
    std::vector<cudf::table_view> boundaryRows{
        source.previousLastKey->view(), firstRows.front()};
    auto boundary =
        cudf::concatenate(boundaryRows, mergeStream_, get_temp_mr());
    VELOX_CHECK(
        cudf::is_sorted(
            boundary->view(), columnOrder_, nullOrder_, mergeStream_),
        "MERGE_SINGLE producer {} is not monotonic across batches",
        source.remoteTaskId);
  }

  auto lastRows = singleRowSlices({keys}, false, mergeStream_);
  source.previousLastKey = std::make_unique<cudf::table>(
      lastRows.front(), mergeStream_, get_output_mr());

  // The copied boundary key reads from the received head on mergeStream_.
  // Rebind the head's eventual deallocation to that stream before the head can
  // be passed through or released by a merge round.
  const std::vector<CudfVectorPtr> headVector{source.head};
  orderCudfVectorDeallocationsAfterStream(
      headVector, inputStreams, mergeStream_);
}

bool CudfMergeExchange::fetchMissingHeads(
    std::vector<ContinueFuture>& futures) {
  bool allReady = true;
  for (auto& source : sources_) {
    if (source.head || source.atEnd) {
      continue;
    }

    for (;;) {
      bool atEnd = false;
      ContinueFuture dataFuture = ContinueFuture::makeEmpty();
      auto data = source.client->next(0, &atEnd, &dataFuture);
      if (data) {
        source.head = makeVector(std::move(data), source);
        if (!source.head) {
          // Empty input batches carry no ordering information. Keep polling
          // this source until data, EOS, or a future is returned.
          continue;
        }
        validateAndRememberHead(source);
        break;
      }

      if (atEnd) {
        source.atEnd = true;
        break;
      }

      VELOX_CHECK(dataFuture.valid());
      futures.push_back(std::move(dataFuture));
      allReady = false;
      break;
    }
  }
  return allReady;
}

exec::BlockingReason CudfMergeExchange::isBlocked(ContinueFuture* future) {
  if (finished_ || closed_) {
    return exec::BlockingReason::kNotBlocked;
  }

  if (!sourcesInitialized_) {
    const auto reason = collectSplits(future);
    if (reason != exec::BlockingReason::kNotBlocked) {
      return reason;
    }
  }

  std::vector<ContinueFuture> dataFutures;
  if (!fetchMissingHeads(dataFutures)) {
    VELOX_CHECK(!dataFutures.empty());
    if (dataFutures.size() == 1) {
      *future = std::move(dataFutures.front());
    } else {
      *future = folly::collectAny(dataFutures).unit();
    }
    return exec::BlockingReason::kWaitForProducer;
  }

  if (allSourcesAtEnd()) {
    finished_ = true;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr CudfMergeExchange::passThroughSingleSource(SourceState& source) {
  VELOX_CHECK_NOT_NULL(source.head);
  auto input = std::exchange(source.head, nullptr);
  const auto begin = std::exchange(source.beginOffset, 0);
  if (begin == 0) {
    return input;
  }

  const auto inputStream = input->stream();
  const std::vector<rmm::cuda_stream_view> inputStreams{inputStream};
  cudf::detail::join_streams(inputStreams, mergeStream_);
  auto slices = cudf::slice(
      input->getTableView(),
      {begin, input->getTableView().num_rows()},
      mergeStream_);
  VELOX_CHECK_EQ(slices.size(), 1);
  auto output = std::make_unique<cudf::table>(
      slices.front(), mergeStream_, get_output_mr());
  const auto outputRows = output->num_rows();
  const std::vector<CudfVectorPtr> inputVectors{input};
  orderCudfVectorDeallocationsAfterStream(
      inputVectors, inputStreams, mergeStream_);
  return std::make_shared<CudfVector>(
      pool(), outputType_, outputRows, std::move(output), mergeStream_);
}

RowVectorPtr CudfMergeExchange::mergeSafePrefix(
    const std::vector<size_t>& activeSources) {
  VELOX_CHECK_GT(activeSources.size(), 1);

  std::vector<CudfVectorPtr> activeHeads;
  std::vector<rmm::cuda_stream_view> inputStreams;
  std::vector<cudf::table_view> lastKeyViews;
  activeHeads.reserve(activeSources.size());
  inputStreams.reserve(activeSources.size());
  lastKeyViews.reserve(activeSources.size());

  for (const auto index : activeSources) {
    auto& source = sources_[index];
    VELOX_CHECK_NOT_NULL(source.head);
    VELOX_CHECK_NOT_NULL(source.previousLastKey);
    activeHeads.push_back(source.head);
    inputStreams.push_back(source.head->stream());
    lastKeyViews.push_back(source.previousLastKey->view());
  }

  cudf::detail::join_streams(inputStreams, mergeStream_);

  // Each last-key table contains one row and is therefore sorted. Merging them
  // puts the earliest last key under the SQL ordering at row zero.
  auto sortedLastKeys = cudf::merge(
      lastKeyViews,
      normalizedSortKeys_,
      columnOrder_,
      nullOrder_,
      mergeStream_,
      get_temp_mr());
  auto frontierSlices =
      cudf::slice(sortedLastKeys->view(), {0, 1}, mergeStream_);
  VELOX_CHECK_EQ(frontierSlices.size(), 1);
  const auto frontier = frontierSlices.front();

  std::vector<cudf::size_type> splitOffsets(activeSources.size());
  std::vector<std::unique_ptr<cudf::column>> splitColumns;
  splitColumns.reserve(activeSources.size());

  for (size_t i = 0; i < activeSources.size(); ++i) {
    auto& source = sources_[activeSources[i]];
    const auto table = source.head->getTableView();
    auto effectiveSlices = cudf::slice(
        table, {source.beginOffset, table.num_rows()}, mergeStream_);
    VELOX_CHECK_EQ(effectiveSlices.size(), 1);
    const auto effectiveKeys = effectiveSlices.front().select(sortKeys_);
    auto split = cudf::upper_bound(
        effectiveKeys,
        frontier,
        columnOrder_,
        nullOrder_,
        mergeStream_,
        get_temp_mr());
    VELOX_CHECK_EQ(split->size(), 1);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &splitOffsets[i],
        split->view().data<cudf::size_type>(),
        sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        mergeStream_.value()));
    splitColumns.push_back(std::move(split));
  }

  // V0 intentionally has one synchronization point per output round. The
  // public cuDF search API returns device columns while cudf::slice takes host
  // offsets. A future fused bounds/merge kernel can remove this boundary.
  mergeStream_.synchronize();

  std::vector<cudf::table_view> prefixes;
  prefixes.reserve(activeSources.size());
  cudf::size_type outputRows = 0;
  for (size_t i = 0; i < activeSources.size(); ++i) {
    auto& source = sources_[activeSources[i]];
    const auto split = splitOffsets[i];
    VELOX_CHECK_GE(split, 0);
    const auto remaining =
        source.head->getTableView().num_rows() - source.beginOffset;
    VELOX_CHECK_LE(split, remaining);
    if (split == 0) {
      continue;
    }
    auto slices = cudf::slice(
        source.head->getTableView(),
        {source.beginOffset, source.beginOffset + split},
        mergeStream_);
    VELOX_CHECK_EQ(slices.size(), 1);
    prefixes.push_back(slices.front());
    outputRows += split;
  }

  VELOX_CHECK_GT(
      outputRows,
      0,
      "MERGE_SINGLE safe-prefix round must consume at least one row");
  auto merged = cudf::merge(
      prefixes,
      sortKeys_,
      columnOrder_,
      nullOrder_,
      mergeStream_,
      get_output_mr());

  // Make stream-ordered deallocation of received packed tables occur after the
  // merge has consumed all prefix views.
  orderCudfVectorDeallocationsAfterStream(
      activeHeads, inputStreams, mergeStream_);

  for (size_t i = 0; i < activeSources.size(); ++i) {
    auto& source = sources_[activeSources[i]];
    source.beginOffset += splitOffsets[i];
    if (source.beginOffset == source.head->getTableView().num_rows()) {
      source.head.reset();
      source.beginOffset = 0;
    }
  }

  VELOX_CHECK_EQ(merged->num_rows(), outputRows);
  return std::make_shared<CudfVector>(
      pool(), outputType_, outputRows, std::move(merged), mergeStream_);
}

RowVectorPtr CudfMergeExchange::getOutput() {
  if (finished_ || closed_ || !sourcesInitialized_) {
    return nullptr;
  }

  std::vector<size_t> activeSources;
  activeSources.reserve(sources_.size());
  for (size_t i = 0; i < sources_.size(); ++i) {
    if (sources_[i].head) {
      activeSources.push_back(i);
    } else if (!sources_[i].atEnd) {
      // Driver must call isBlocked() to fetch all missing heads before output.
      return nullptr;
    }
  }

  if (activeSources.empty()) {
    finished_ = allSourcesAtEnd();
    return nullptr;
  }

  RowVectorPtr output;
  if (activeSources.size() == 1) {
    output = passThroughSingleSource(sources_[activeSources.front()]);
  } else {
    output = mergeSafePrefix(activeSources);
  }

  if (output) {
    stats_.wlock()->addOutputVector(output->estimateFlatSize(), output->size());
  }
  return output;
}

bool CudfMergeExchange::allSourcesAtEnd() const {
  if (!sourcesInitialized_) {
    return false;
  }
  return std::all_of(sources_.begin(), sources_.end(), [](const auto& source) {
    return source.atEnd && !source.head;
  });
}

bool CudfMergeExchange::isFinished() {
  return finished_ || closed_;
}

void CudfMergeExchange::close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  for (auto& source : sources_) {
    source.head.reset();
    source.previousLastKey.reset();
    if (source.client) {
      source.client->close();
      source.client.reset();
    }
  }
  sources_.clear();
  SourceOperator::close();
}

} // namespace facebook::velox::cudf_velox
