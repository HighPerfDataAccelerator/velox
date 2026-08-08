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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/CudfPackedRestore.h"
#include "velox/experimental/cudf/exec/CudfPackedSpill.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"
#include "velox/exec/TopNRowNumber.h"

#include <folly/ScopeGuard.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/merge.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/replace.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <malloc.h>
#include <lz4.h>
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kSortedRunBytes = 3ULL << 30;
constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
constexpr size_t kMergeFanIn = 4;
// Complete TopN output can feed expressions which amplify nested data before
// applying a selective filter (Job 144 has JSON array extraction followed by
// Unnest). Keep the handoff independently bounded by the same runtime output
// controls as OrderBy. This avoids a hidden 8 MiB/32K-row limit and lets a
// query tune the boundary after downstream expression CSE reduces its working
// set, without changing the finalize bucket or the Top-1 state algorithm.
constexpr std::string_view kConditionalTopNMarker = "__gluten_mpp_topn_active";
// MPP exchange already partitions the same key with libcudf's default seed.
// Reusing that seed locally preserves the exchange's low hash bits, so a
// four-way exchange followed by 64 local buckets only uses 16 buckets per
// executor. A fixed independent seed keeps a logical key stable across input
// batches while decorrelating the operator-local fanout from the exchange.
constexpr uint32_t kTopNLocalHashSeed =
    cudf::DEFAULT_HASH_SEED ^ 0x9e3779b9U;
std::atomic<uint64_t> spillDirectorySequence{0};

bool topNPhaseSyncEnabled(
    std::string_view nodeId,
    uint64_t inputBatches) {
  const auto* enabled =
      std::getenv("GLUTEN_CUDF_TOPN_PHASE_SYNC");
  if (enabled == nullptr ||
      (std::string_view(enabled) != "1" &&
       std::string_view(enabled) != "true" &&
       std::string_view(enabled) != "TRUE")) {
    return false;
  }
  const auto* nodeFilter =
      std::getenv("GLUTEN_CUDF_TOPN_PHASE_SYNC_NODE");
  if (nodeFilter != nullptr && *nodeFilter != '\0' &&
      std::string_view(nodeFilter) != nodeId) {
    return false;
  }
  const auto* minBatchValue =
      std::getenv("GLUTEN_CUDF_TOPN_PHASE_SYNC_MIN_BATCH");
  if (minBatchValue != nullptr && *minBatchValue != '\0') {
    char* end = nullptr;
    const auto minBatch = std::strtoull(minBatchValue, &end, 10);
    if (end != minBatchValue && *end == '\0' && inputBatches < minBatch) {
      return false;
    }
  }
  return true;
}

void synchronizeTopNPhase(
    rmm::cuda_stream_view stream,
    std::string_view nodeId,
    uint64_t inputBatches,
    std::string_view phase,
    bool always = false) {
  if (!always && !topNPhaseSyncEnabled(nodeId, inputBatches)) {
    return;
  }
  const auto error = cudaStreamSynchronize(stream.value());
  if (error != cudaSuccess) {
    VELOX_FAIL(
        "CudfTopNRowNumber node={} inputBatches={} phase={} CUDA stream "
        "synchronization failed: {}",
        nodeId,
        inputBatches,
        phase,
        cudaGetErrorString(error));
  }
}

bool topNDeviceOutputStagingEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_DEVICE_OUTPUT_STAGING_ENABLED");
  if (value == nullptr) {
    return false;
  }
  return std::string_view(value) == "1" ||
      std::string_view(value) == "true" ||
      std::string_view(value) == "TRUE";
}

bool topNLatePayloadGatherEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_LATE_PAYLOAD_GATHER");
  if (value == nullptr) {
    return true;
  }
  return std::string_view(value) != "0" &&
      std::string_view(value) != "false" &&
      std::string_view(value) != "FALSE";
}

bool topNLatePayloadSortEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_LATE_PAYLOAD_ALGORITHM");
  return value != nullptr && std::string_view(value) == "sort";
}

bool topNUniquePartitionFastPathEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_UNIQUE_PARTITION_FAST_PATH");
  if (value == nullptr) {
    return false;
  }
  return std::string_view(value) == "1" ||
      std::string_view(value) == "true" ||
      std::string_view(value) == "TRUE";
}

bool topNSequentialUniqueFastPathEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_UNIQUE_FAST_PATH");
  if (value == nullptr) {
    return false;
  }
  return std::string_view(value) == "1" ||
      std::string_view(value) == "true" ||
      std::string_view(value) == "TRUE";
}

bool topNSequentialDirectChunkOutputEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_DIRECT_CHUNK_OUTPUT");
  // The sequential fast path is itself opt-in. Keep direct bounded chunk
  // output on by default inside that mode while retaining a same-binary A/B
  // switch for endpoint attribution.
  return value == nullptr || std::string_view(value) != "0";
}

bool topNSequentialSparseDuplicatePathEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_DUPLICATES");
  // Sequential mode is already an explicit opt-in. Keep the exact sparse
  // continuation enabled by default inside it, with a same-binary escape
  // hatch for endpoint attribution or emergency rollback.
  return value == nullptr ||
      (std::string_view(value) != "0" &&
       std::string_view(value) != "false" &&
       std::string_view(value) != "FALSE");
}

size_t topNDenseRestoreHostThreads() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_DENSE_RESTORE_HOST_THREADS");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed > 8) {
    LOG(WARNING)
        << "Ignoring invalid "
           "GLUTEN_CUDF_TOPN_DENSE_RESTORE_HOST_THREADS="
        << value << "; using the shared packed-restore default";
    return 0;
  }
  // Zero is an explicit same-binary disable switch. Non-zero calls receive a
  // local bound while the packed restore backend retains its global cap.
  return static_cast<size_t>(parsed);
}

uint32_t topNSequentialSparseMaxCandidatePct() {
  constexpr uint32_t kDefaultMaxCandidatePct = 10;
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_MAX_CANDIDATE_PCT");
  if (value == nullptr || *value == '\0') {
    return kDefaultMaxCandidatePct;
  }
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed > 100) {
    LOG(WARNING)
        << "Ignoring invalid "
           "GLUTEN_CUDF_TOPN_SEQUENTIAL_SPARSE_MAX_CANDIDATE_PCT="
        << value << "; using " << kDefaultMaxCandidatePct;
    return kDefaultMaxCandidatePct;
  }
  return static_cast<uint32_t>(parsed);
}

uint64_t topNSequentialEarlyDenseProbeBatches() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_EARLY_DENSE_BATCHES");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0') {
    LOG(WARNING)
        << "Ignoring invalid "
           "GLUTEN_CUDF_TOPN_SEQUENTIAL_EARLY_DENSE_BATCHES="
        << value;
    return 0;
  }
  return std::clamp<uint64_t>(parsed, 2, 1024);
}

uint32_t topNSequentialEarlyDenseMaxDistinctPct() {
  constexpr uint32_t kDefaultMaxDistinctPct = 95;
  const auto* value = std::getenv(
      "GLUTEN_CUDF_TOPN_SEQUENTIAL_EARLY_DENSE_MAX_DISTINCT_PCT");
  if (value == nullptr || *value == '\0') {
    return kDefaultMaxDistinctPct;
  }
  char* end = nullptr;
  const auto parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed > 100) {
    LOG(WARNING)
        << "Ignoring invalid "
           "GLUTEN_CUDF_TOPN_SEQUENTIAL_EARLY_DENSE_MAX_DISTINCT_PCT="
        << value << "; using " << kDefaultMaxDistinctPct;
    return kDefaultMaxDistinctPct;
  }
  return static_cast<uint32_t>(parsed);
}

bool topNSequentialDenseRawRewriteEnabled() {
  const auto* value =
      std::getenv("GLUTEN_CUDF_TOPN_SEQUENTIAL_DENSE_RAW_REWRITE");
  // r811 proved that skipping the second LZ4 pass increases critical-path
  // NVMe and restore work more than it saves compression work. Keep the
  // implementation as an explicit diagnostic A/B only.
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false" &&
      std::string_view(value) != "FALSE";
}

bool isSupportedKeyType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::ARRAY:
    case TypeKind::MAP:
    case TypeKind::ROW:
    case TypeKind::UNKNOWN:
      return false;
    default:
      return true;
  }
}

std::unique_ptr<cudf::table> copyTableSlice(
    cudf::table_view input,
    cudf::size_type begin,
    cudf::size_type end,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_LE(begin, end);
  auto slices = cudf::slice(input, {begin, end}, stream);
  VELOX_CHECK_EQ(slices.size(), 1);
  return std::make_unique<cudf::table>(slices.front(), stream, mr);
}

cudf::size_type firstSearchPosition(
    cudf::column_view positions,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_EQ(positions.size(), 1);
  cudf::size_type result{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &result,
      positions.data<cudf::size_type>(),
      sizeof(result),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return result;
}

} // namespace

bool CudfTopNRowNumber::shouldReplace(
    const std::shared_ptr<const core::TopNRowNumberNode>& node) {
  if (node == nullptr || node->limit() != 1) {
    return false;
  }
  const auto rankFunction = node->rankFunction();
  if (rankFunction != core::TopNRowNumberNode::RankFunction::kRowNumber &&
      rankFunction != core::TopNRowNumberNode::RankFunction::kRank &&
      rankFunction != core::TopNRowNumberNode::RankFunction::kDenseRank) {
    return false;
  }
  if (rankFunction != core::TopNRowNumberNode::RankFunction::kRowNumber &&
      node->sortingKeys().empty()) {
    return false;
  }

  for (const auto& key : node->partitionKeys()) {
    if (!isSupportedKeyType(key->type())) {
      return false;
    }
  }

  for (const auto& key : node->sortingKeys()) {
    if (!isSupportedKeyType(key->type())) {
      return false;
    }
  }

  return true;
}

CudfTopNRowNumber::CudfTopNRowNumber(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::TopNRowNumberNode>& node)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          node->outputType(),
          node->id(),
          "CudfTopNRowNumber",
          nvtx3::rgb{255, 140, 0},
          NvtxMethodFlag::kAll,
          std::nullopt,
          node),
      limit_(node->limit()),
      rankFunction_(node->rankFunction()),
      generateRowNumber_(node->generateRowNumber()),
      partialOutput_(node->partialOutput()),
      inputType_(node->inputType()),
      diagnosticNodeId_(node->id()),
      candidateRunBytes_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
          CudfConfig::getInstance().topNRowNumberCandidateRunBytes)),
      inputWorkspaceBytes_(std::max<uint64_t>(
          1ULL << 30,
          std::min<uint64_t>(candidateRunBytes_, 1ULL << 30) * 2)),
      hostPartitionCount_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfTopNRowNumberHostPartitions,
          CudfConfig::getInstance().topNRowNumberHostPartitions)),
      finalizeInputBytes_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfTopNRowNumberFinalizeInputBytes,
          CudfConfig::getInstance().topNRowNumberFinalizeInputBytes)),
      deviceResidentBytesLimit_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfTopNRowNumberDeviceResidentBytes,
          CudfConfig::getInstance().topNRowNumberDeviceResidentBytes)),
      globalDeviceResidentCapacityBytes_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfDeviceResidentCapacityBytes,
          CudfConfig::getInstance().deviceResidentCapacityBytes)),
      hostResidentBytesLimit_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfOrderByHostSpillBytes,
          CudfConfig::getInstance().orderByHostSpillBytes)),
      outputChunkBytes_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfTopNRowNumberOutputChunkBytes,
          CudfConfig::getInstance().topNRowNumberOutputChunkBytes)),
      maxOutputRows_(driverCtx->queryConfig().get<int32_t>(
          CudfConfig::kCudfTopNRowNumberMaxOutputRows,
          CudfConfig::getInstance().topNRowNumberMaxOutputRows)),
      deviceOutputStagingEnabled_(topNDeviceOutputStagingEnabled()),
      abandonPartialMinRows_(
          driverCtx->queryConfig().abandonPartialTopNRowNumberMinRows()),
      abandonPartialMinPct_(
          driverCtx->queryConfig().abandonPartialTopNRowNumberMinPct()) {
  VELOX_CHECK_EQ(limit_, 1, "CudfTopNRowNumber only supports limit=1");
  VELOX_CHECK_GT(
      candidateRunBytes_,
      0,
      "CudfTopNRowNumber candidate run bytes must be greater than zero");
  VELOX_CHECK_GE(
      hostPartitionCount_,
      2,
      "CudfTopNRowNumber host partition count must be at least two");
  VELOX_CHECK_GT(
      finalizeInputBytes_,
      0,
      "CudfTopNRowNumber finalize input bytes must be greater than zero");
  VELOX_CHECK_GT(
      hostResidentBytesLimit_,
      0,
      "CudfTopNRowNumber host resident byte limit must be greater than zero");
  VELOX_CHECK_GT(
      outputChunkBytes_,
      0,
      "CudfTopNRowNumber output chunk bytes must be greater than zero");
  VELOX_CHECK_GT(
      maxOutputRows_,
      0,
      "CudfTopNRowNumber max output rows must be greater than zero");
  VELOX_CHECK(
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber ||
          rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank ||
          rankFunction_ == core::TopNRowNumberNode::RankFunction::kDenseRank,
      "CudfTopNRowNumber only supports row_number, rank, or dense_rank");
  LOG(INFO) << "CudfTopNRowNumber node=" << diagnosticNodeId_
            << " candidateRunBytes=" << candidateRunBytes_
            << " inputWorkspaceBytes=" << inputWorkspaceBytes_
            << " hostPartitions=" << hostPartitionCount_
            << " finalizeInputBytes=" << finalizeInputBytes_
            << " deviceResidentBytesLimit=" << deviceResidentBytesLimit_
            << " globalDeviceResidentCapacityBytes="
            << globalDeviceResidentCapacityBytes_
            << " hostResidentBytesLimit=" << hostResidentBytesLimit_
            << " outputChunkBytes=" << outputChunkBytes_
            << " maxOutputRows=" << maxOutputRows_
            << " deviceOutputStagingEnabled="
            << deviceOutputStagingEnabled_
            << " abandonPartialMinRows=" << abandonPartialMinRows_
            << " abandonPartialMinPct=" << abandonPartialMinPct_
            << " partialOutput=" << partialOutput_;

  for (const auto& key : node->partitionKeys()) {
    const auto channel = exec::exprToChannel(key.get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "TopNRowNumber doesn't allow constant partition keys");
    partitionKeys_.push_back(channel);
    const auto& keyName = inputType_->nameOf(channel);
    if (keyName.compare(
            0, kConditionalTopNMarker.size(), kConditionalTopNMarker) == 0) {
      VELOX_CHECK(
          !passthroughKey_.has_value(),
          "TopNRowNumber allows only one conditional pass-through key");
      VELOX_CHECK(
          inputType_->childAt(channel)->kind() == TypeKind::BOOLEAN,
          "Conditional TopNRowNumber marker must be boolean");
      passthroughKey_ = channel;
    }
  }

  const auto& sortingKeys = node->sortingKeys();
  const auto& sortingOrders = node->sortingOrders();

  for (const auto& key : sortingKeys) {
    const auto channel = exec::exprToChannel(key.get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "TopNRowNumber doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
  }

  allKeyIndices_ = partitionKeys_;
  allKeyIndices_.insert(
      allKeyIndices_.end(), sortKeys_.begin(), sortKeys_.end());

  for (size_t i = 0; i < partitionKeys_.size(); ++i) {
    columnOrders_.push_back(cudf::order::ASCENDING);
    nullOrders_.push_back(cudf::null_order::BEFORE);
  }

  for (const auto& order : sortingOrders) {
    columnOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    nullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

exec::BlockingReason CudfTopNRowNumber::isBlocked(
    ContinueFuture* future) {
  if (!waitingForDeviceWorkspace_) {
    if (pendingInput_ != nullptr &&
        !(partialOutput_ && abandonedPartial_) &&
        !inputWorkspaceAdmission_.has_value()) {
      VELOX_CHECK_NOT_NULL(future);
      ContinueFuture workspaceFuture;
      auto workspaceAdmission = tryAcquireDeviceMemoryWorkspace(
          customPool(kCudfDeviceMemoryResourceTag),
          this,
          inputWorkspaceBytes_,
          CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
          DeviceMemoryWorkspacePriority::kInput,
          &deviceWorkspaceRequest_,
          &workspaceFuture);
      if (!workspaceAdmission.has_value()) {
        *future = std::move(workspaceFuture);
        return exec::BlockingReason::kWaitForArbitration;
      }
      inputWorkspaceAdmission_ = std::move(workspaceAdmission.value());
    }
    return exec::BlockingReason::kNotBlocked;
  }
  VELOX_CHECK_NOT_NULL(future);
  *future = std::move(deviceWorkspaceFuture_);
  waitingForDeviceWorkspace_ = false;
  return exec::BlockingReason::kWaitForArbitration;
}

void CudfTopNRowNumber::doAddInput(RowVectorPtr input) {
  VELOX_CHECK_NULL(
      pendingInput_, "TopN received a second input before processing the first");
  VELOX_CHECK(
      !inputWorkspaceAdmission_.has_value(),
      "TopN input workspace must not be reserved before input ownership");
  pendingInput_ = std::move(input);
}

void CudfTopNRowNumber::processPendingInput(RowVectorPtr input) {
  // The next batch must take a new physical-memory snapshot. This scope guard
  // also covers early returns and exceptions. An abandoned partial is a pure
  // ownership handoff and deliberately reaches this method without an
  // admission: it launches no GPU work and must not wait for 1 GiB of
  // workspace while retaining its input batch.
  SCOPE_EXIT {
    inputWorkspaceAdmission_.reset();
  };
  if (input->size() == 0) {
    return;
  }
  const auto addInputStart = std::chrono::steady_clock::now();
  const auto inputRows = input->size();

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "Expected CudfVector input");

  if (partialOutput_ && abandonedPartial_) {
    partialOutputs_.push_back(std::move(cudfInput));
    ++diagnosticInputBatches_;
    diagnosticInputRows_ += inputRows;
    diagnosticPartialOutputRows_ += inputRows;
    ++diagnosticPartialWorkspaceBypassBatches_;
    return;
  }

  VELOX_CHECK(
      inputWorkspaceAdmission_.has_value(),
      "TopN GPU input processing requires a live device workspace admission");

  if (passthroughKey_.has_value()) {
    auto stream = cudfInput->stream();
    auto inputView = cudfInput->getTableView();
    auto activeMask = inputView.column(*passthroughKey_);
    VELOX_CHECK(
        activeMask.type().id() == cudf::type_id::BOOL8,
        "Conditional TopNRowNumber marker must be BOOL8");

    auto inactiveMask = cudf::unary_operation(
        activeMask, cudf::unary_operator::NOT, stream, get_temp_mr());
    auto inactive = cudf::apply_boolean_mask(
        inputView, inactiveMask->view(), stream, get_output_mr());
    if (inactive->num_rows() > 0) {
      passthroughOutputs_.push_back(
          std::make_shared<CudfVector>(
              pool(),
              inputType_,
              inactive->num_rows(),
              std::move(inactive),
              stream));
    }

    auto active = cudf::apply_boolean_mask(
        inputView, activeMask, stream, get_output_mr());
    if (active->num_rows() == 0) {
      return;
    }
    cudfInput = std::make_shared<CudfVector>(
        pool(), inputType_, active->num_rows(), std::move(active), stream);
  }

  auto stream = cudfInput->stream();
  if (candidates_ && stateStream_.value() != stream.value()) {
    stateStream_.synchronize();
  }
  stateStream_ = stream;
  auto mr = get_output_mr();

  // CPU Velox checks whether a partial TopN should be abandoned only after it
  // has processed enough input rows. That is reasonable for CPU-sized input
  // vectors, but a single GPU vector can be close to a GiB. Fully reducing
  // such a vector just to discover that the partial retains nearly every row
  // creates a large and unnecessary sort/gather working set.
  //
  // Partial TopN is only an optimization in front of an exact final TopN. On
  // the first sufficiently large input, use a bounded prefix to estimate its
  // reduction ratio. If reduction is poor, pass the original (unmodified)
  // input through. A false-positive abandon can only send extra rows to the
  // final TopN; it cannot change the result.
  if (partialOutput_ && !generateRowNumber_ && diagnosticInputRows_ == 0 &&
      abandonPartialMinRows_ > 0 &&
      static_cast<uint64_t>(inputRows) >= abandonPartialMinRows_) {
    const auto sampleRows = static_cast<cudf::size_type>(
        std::min<uint64_t>(inputRows, abandonPartialMinRows_));
    const auto originalInputBytes = cudfInput->estimateFlatSize();
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfTopNRowNumber node={} state=partialSample.begin "
        "originalInputRows={} originalInputBytes={} sampleRows={}",
        diagnosticNodeId_,
        inputRows,
        originalInputBytes,
        sampleRows));
    auto sampleViews =
        cudf::slice(cudfInput->getTableView(), {0, sampleRows}, stream);
    VELOX_CHECK_EQ(sampleViews.size(), 1);
    auto sampleCandidates = reduceToCandidates(sampleViews.front(), stream, mr);
    synchronizeTopNPhase(
        stream,
        diagnosticNodeId_,
        diagnosticInputBatches_,
        "reduce-partial-abandon-sample");
    const auto sampleOutputRows = sampleCandidates->num_rows();
    logDeviceMemorySnapshot(fmt::format(
        "operator=CudfTopNRowNumber node={} state=partialSample.reduced "
        "originalInputRows={} originalInputBytes={} sampleInputRows={} "
        "sampleOutputRows={}",
        diagnosticNodeId_,
        inputRows,
        originalInputBytes,
        sampleRows,
        sampleOutputRows));
    if (static_cast<uint64_t>(sampleOutputRows) * 100 >=
        static_cast<uint64_t>(sampleRows) * abandonPartialMinPct_) {
      // Release the probe result before publishing the original input. The
      // partial output queue must retain only the original owner, never both
      // the diagnostic sample candidates and the pass-through batch.
      sampleCandidates.reset();
      logDeviceMemorySnapshot(fmt::format(
          "operator=CudfTopNRowNumber node={} state=partialSample.released "
          "logicalOutputOwnerBytes={}",
          diagnosticNodeId_,
          originalInputBytes));
      abandonedPartial_ = true;
      partialOutputs_.push_back(std::move(cudfInput));
      ++diagnosticInputBatches_;
      diagnosticInputRows_ += inputRows;
      diagnosticPartialOutputRows_ += inputRows;
      diagnosticAddInputMicros_ +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - addInputStart)
              .count();
      {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            std::string(exec::TopNRowNumber::kAbandonedPartial),
            RuntimeCounter(1));
        lockedStats->addRuntimeStat(
            "partialAbandonSampleInputRows", RuntimeCounter(sampleRows));
        lockedStats->addRuntimeStat(
            "partialAbandonSampleOutputRows",
            RuntimeCounter(sampleOutputRows));
      }
      LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                   << " abandoned low-reduction partial from bounded sample"
                   << " sampleInputRows=" << sampleRows
                   << " sampleOutputRows=" << sampleOutputRows
                   << " retainedPct="
                   << (100.0 * sampleOutputRows / sampleRows)
                   << " originalBatchRows=" << inputRows
                   << " thresholdPct=" << abandonPartialMinPct_;
      return;
    }
  }

  // When explicitly selected, preserve each final input run in arrival form
  // and collect only its narrow partition keys for a bounded exact
  // end-of-input uniqueness proof. This avoids hashing and scattering the
  // wide payload on the measured all-unique path. Any duplicate later
  // consolidates these untouched runs and enters the historical recursive
  // exact path.
  const bool sequentialUniqueInput =
      !partialOutput_ && topNUniquePartitionFastPathEnabled() &&
      topNSequentialUniqueFastPathEnabled() &&
      !sequentialDenseInputMode_ &&
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber &&
      !partitionKeys_.empty();
  if (sequentialUniqueInput) {
    const auto estimatedBytes = cudfInput->estimateFlatSize();
    externalizeSequentialUniqueCandidates(
        cudfInput->release(), estimatedBytes, stream);
    addRuntimeStat("topNSequentialInputRows", RuntimeCounter(inputRows));
    addRuntimeStat("topNSequentialInputBatches", RuntimeCounter(1));
    ++diagnosticInputBatches_;
    diagnosticInputRows_ += inputRows;
    diagnosticAddInputMicros_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - addInputStart)
            .count();
    return;
  }

  // The final partitioned ROW_NUMBER path is exact over the complete local
  // hash bucket. With the opt-in uniqueness path, retain every input row and
  // defer reduction to that final bucket instead of probing/sorting each
  // batch first. Duplicates are therefore never discarded speculatively; the
  // final exact probe either reuses a unique bucket or falls back to Top-1.
  const bool deferInputReduction =
      !partialOutput_ && topNUniquePartitionFastPathEnabled() &&
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber &&
      !partitionKeys_.empty();
  auto batchCandidates = deferInputReduction
      ? cudfInput->release()
      : reduceToCandidates(cudfInput->getTableView(), stream, mr);
  if (deferInputReduction) {
    addRuntimeStat("topNDeferredInputRows", RuntimeCounter(inputRows));
    addRuntimeStat("topNDeferredInputBatches", RuntimeCounter(1));
  }
  synchronizeTopNPhase(
      stream,
      diagnosticNodeId_,
      diagnosticInputBatches_,
      "reduce-input-batch");
  if (partialOutput_) {
    const auto rows = batchCandidates->num_rows();
    if (rows > 0) {
      partialOutputs_.push_back(std::make_shared<CudfVector>(
          pool(),
          inputType_,
          rows,
          std::move(batchCandidates),
          stream));
    }
    ++diagnosticInputBatches_;
    diagnosticInputRows_ += inputRows;
    diagnosticPartialOutputRows_ += rows;
    diagnosticAddInputMicros_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - addInputStart)
            .count();
    if (!generateRowNumber_ &&
        diagnosticInputRows_ >= abandonPartialMinRows_ &&
        diagnosticPartialOutputRows_ * 100 >=
            diagnosticInputRows_ * abandonPartialMinPct_) {
      abandonedPartial_ = true;
      {
        auto lockedStats = stats_.wlock();
        lockedStats->addRuntimeStat(
            std::string(exec::TopNRowNumber::kAbandonedPartial),
            RuntimeCounter(1));
      }
      LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                   << " abandoned low-reduction partial after inputRows="
                   << diagnosticInputRows_
                   << " outputRows=" << diagnosticPartialOutputRows_
                   << " retainedPct="
                   << (100.0 * diagnosticPartialOutputRows_ /
                       diagnosticInputRows_)
                   << " thresholdPct=" << abandonPartialMinPct_;
    }
    return;
  }

  if (candidates_ && candidates_->num_rows() > 0) {
    std::vector<cudf::table_view> pieces{
        candidates_->view(), batchCandidates->view()};
    auto merged = cudf::concatenate(pieces, stream, mr);
    synchronizeTopNPhase(
        stream,
        diagnosticNodeId_,
        diagnosticInputBatches_,
        "concatenate-candidates");
    candidates_ = deferInputReduction
        ? std::move(merged)
        : reduceOwnedCandidates(std::move(merged), stream, mr);
    synchronizeTopNPhase(
        stream,
        diagnosticNodeId_,
        diagnosticInputBatches_,
        "reduce-merged-candidates");
  } else {
    candidates_ = std::move(batchCandidates);
  }

  auto candidateVector = std::make_shared<CudfVector>(
      pool(),
      inputType_,
      candidates_->num_rows(),
      std::move(candidates_),
      stream);
  const auto candidateBytes = candidateVector->estimateFlatSize();
  if (candidateBytes >= candidateRunBytes_) {
    if (rankFunction_ ==
            core::TopNRowNumberNode::RankFunction::kRowNumber &&
        !partitionKeys_.empty()) {
      externalizeRowNumberCandidates(candidateVector->release(), stream);
    } else {
      inputs_.push_back(std::move(candidateVector));
      bufferedBytes_ = candidateBytes;
      spillSortedRun();
    }
  } else {
    candidates_ = candidateVector->release();
  }
  ++diagnosticInputBatches_;
  diagnosticInputRows_ += inputRows;
  diagnosticAddInputMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - addInputStart)
          .count();
  if (diagnosticInputBatches_ == 1 ||
      diagnosticInputBatches_ % 64 == 0) {
    LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                 << " progress inputBatches=" << diagnosticInputBatches_
                 << " inputRows=" << diagnosticInputRows_
                 << " addInputUs=" << diagnosticAddInputMicros_
                 << " candidateRows="
                 << (candidates_ ? candidates_->num_rows() : 0)
                 << " partitionedMode=" << partitionedRowNumberMode_
                 << " deviceBytes=" << partitionedDeviceBytes_
                 << " hostBytes=" << partitionedHostBytes_
                 << " diskBytes=" << partitionedDiskBytes_
                 << " externalizeCalls=" << diagnosticExternalizeCalls_
                 << " hashPartitionUs=" << diagnosticHashPartitionMicros_
                 << " contiguousSplitUs="
                 << diagnosticContiguousSplitMicros_
                 << " d2hSubmitUs=" << diagnosticD2hSubmitMicros_
                 << " d2hSynchronizeUs="
                 << diagnosticD2hSynchronizeMicros_
                 << " compressionUs=" << diagnosticCompressionMicros_
                 << " spillEnqueueUs=" << diagnosticSpillEnqueueMicros_
                 << " storeUs=" << diagnosticStoreMicros_;
  }
}

void CudfTopNRowNumber::doNoMoreInput() {
  Operator::noMoreInput();
  LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
               << " noMoreInput inputBatches=" << diagnosticInputBatches_
               << " inputRows=" << diagnosticInputRows_
               << " addInputUs=" << diagnosticAddInputMicros_
               << " partialOutputRows=" << diagnosticPartialOutputRows_
               << " abandonedPartial=" << abandonedPartial_
               << " partialWorkspaceBypassBatches="
               << diagnosticPartialWorkspaceBypassBatches_
               << " candidateRows="
               << (candidates_ ? candidates_->num_rows() : 0)
               << " partitionedMode=" << partitionedRowNumberMode_
               << " deviceBytes=" << partitionedDeviceBytes_
               << " hostBytes=" << partitionedHostBytes_
               << " diskBytes=" << partitionedDiskBytes_
               << " externalizeCalls=" << diagnosticExternalizeCalls_
               << " hashPartitionUs=" << diagnosticHashPartitionMicros_
               << " contiguousSplitUs=" << diagnosticContiguousSplitMicros_
               << " d2hSubmitUs=" << diagnosticD2hSubmitMicros_
               << " d2hSynchronizeUs="
               << diagnosticD2hSynchronizeMicros_
               << " compressionUs=" << diagnosticCompressionMicros_
               << " spillEnqueueUs=" << diagnosticSpillEnqueueMicros_
               << " storeUs=" << diagnosticStoreMicros_;
  if (partialOutput_) {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "partialWorkspaceBypassBatches",
        RuntimeCounter(diagnosticPartialWorkspaceBypassBatches_));
    finished_ = partialOutputs_.empty() && passthroughOutputs_.empty();
    return;
  }
  if (partitionedRowNumberMode_ && candidates_) {
    externalizeRowNumberCandidates(
        std::exchange(candidates_, nullptr), stateStream_);
  } else if (spilled_ && candidates_) {
    auto stream = stateStream_;
    auto candidateVector = std::make_shared<CudfVector>(
        pool(),
        inputType_,
        candidates_->num_rows(),
        std::move(candidates_),
        stream);
    bufferedBytes_ = candidateVector->estimateFlatSize();
    inputs_.push_back(std::move(candidateVector));
    spillSortedRun();
  }
  // A large partitioned TopN can reach noMoreInput with tens of GiB already
  // externalized while still retaining several GiB of packed candidates on
  // device.  Keeping that persistent cache during finalize competes with the
  // restore, concatenate, reduction, and output-split working set.  Multiple
  // finalizing TopNs on the same GPU can therefore exhaust the device even
  // though each operator individually respects its resident-state limit.
  //
  // Once any state has entered the host/disk tier, flush the remaining device
  // cache before finalize.  This costs one D2H per retained byte and avoids
  // repeatedly evicting it under allocation pressure.  Keep small all-device
  // TopNs unchanged.  Spill in bounded bucket-major waves so this transition
  // does not require a multi-GiB pinned/pageable staging allocation.
  const auto finalizeDeviceResidentBytes = partitionedDeviceBytes_;
  if (partitionedRowNumberMode_ && partitionedDeviceBytes_ > 0 &&
      (partitionedHostBytes_ > 0 || partitionedDiskBytes_ > 0)) {
    const auto waveBudget = std::max<uint64_t>(finalizeInputBytes_, 1);
    std::vector<size_t> wave;
    uint64_t waveBytes = 0;
    size_t spilledBuckets = 0;
    for (size_t bucket = 0;
         bucket < partitionedRowNumberDeviceBytes_.size(); ++bucket) {
      const auto bucketBytes = partitionedRowNumberDeviceBytes_[bucket];
      if (bucketBytes == 0) {
        continue;
      }
      if (!wave.empty() && waveBytes + bucketBytes > waveBudget) {
        spilledBuckets += wave.size();
        spillDevicePartitions(wave, stateStream_);
        wave.clear();
        waveBytes = 0;
      }
      wave.push_back(bucket);
      waveBytes += bucketBytes;
    }
    if (!wave.empty()) {
      spilledBuckets += wave.size();
      spillDevicePartitions(wave, stateStream_);
    }
    VELOX_CHECK_EQ(partitionedDeviceBytes_, 0);
    LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                 << " flushed mixed-tier device state before finalize bytes="
                 << finalizeDeviceResidentBytes
                 << " buckets=" << spilledBuckets
                 << " hostBytes=" << partitionedHostBytes_
                 << " residentHostBytes=" << partitionedResidentHostBytes_
                 << " diskBytes=" << partitionedDiskBytes_;
  }
  if (spilled_) {
    compactSortedRunsForMerge();
    initializeSortedRunReaders();
  }
  if (partitionedRowNumberMode_) {
    // Capture the state after the tail candidate batch has been bucketed.
    // Recording this before externalization under-counted resident and evicted
    // bytes whenever candidates_ was non-null at noMoreInput().
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "topNDeviceResidentBytes",
        RuntimeCounter(finalizeDeviceResidentBytes));
    lockedStats->addRuntimeStat(
        "topNFinalizePreflushBytes",
        RuntimeCounter(finalizeDeviceResidentBytes - partitionedDeviceBytes_));
    lockedStats->addRuntimeStat(
        "topNDeviceSpillBytes",
        RuntimeCounter(diagnosticDeviceSpillBytes_));
    lockedStats->addRuntimeStat(
        "topNDeviceSpillBuckets",
        RuntimeCounter(diagnosticDeviceSpillBuckets_));
    lockedStats->addRuntimeStat(
        "topNGlobalAdmissionRejectedBytes",
        RuntimeCounter(diagnosticGlobalAdmissionRejectedBytes_));
  }
  if (!candidates_ && passthroughOutputs_.empty()) {
    finished_ = !spilled_ && !partitionedRowNumberMode_;
  }
}

RowVectorPtr CudfTopNRowNumber::doGetOutput() {
  if (pendingInput_ != nullptr) {
    VELOX_CHECK(
        inputWorkspaceAdmission_.has_value() ||
            (partialOutput_ && abandonedPartial_),
        "TopN pending input reached getOutput without workspace admission");
    processPendingInput(std::exchange(pendingInput_, nullptr));
  }

  if (!passthroughOutputs_.empty()) {
    auto output = std::move(passthroughOutputs_.front());
    passthroughOutputs_.pop_front();
    return output;
  }

  if (!partialOutputs_.empty()) {
    auto output = std::move(partialOutputs_.front());
    partialOutputs_.pop_front();
    if (noMoreInput_ && partialOutputs_.empty() &&
        passthroughOutputs_.empty()) {
      finished_ = true;
    }
    return output;
  }

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (partitionedRowNumberMode_) {
    auto result = computeNextPartitionedRowNumberOutput();
    if (result.status == PartitionedOutputStatus::kOutput) {
      VELOX_CHECK_NOT_NULL(result.output);
      return std::move(result.output);
    }
    if (result.status == PartitionedOutputStatus::kBlocked) {
      VELOX_CHECK_NULL(result.output);
      VELOX_CHECK(
          waitingForDeviceWorkspace_,
          "Partitioned TopN reported blocked without a workspace future");
      return nullptr;
    }
    VELOX_CHECK(
        result.status == PartitionedOutputStatus::kFinished,
        "Unexpected partitioned TopN output status");
    VELOX_CHECK_NULL(result.output);
    VELOX_CHECK(
        !waitingForDeviceWorkspace_,
        "Partitioned TopN cannot finish while waiting for device workspace");
    finished_ = true;
    clearPartitionedRowNumberState();
    return nullptr;
  }

  if (spilled_) {
    auto result = computeNextSortedOutput();
    if (result != nullptr) {
      return result;
    }
    finished_ = true;
    cleanupSpillFiles();
    return nullptr;
  }

  if (!candidates_) {
    finished_ = true;
    return nullptr;
  }

  auto stream = stateStream_;
  auto mr = get_output_mr();
  auto input = std::exchange(candidates_, nullptr);
  auto result =
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
      ? computeLimitOneRowNumber(input->view(), stream, mr)
      : computeLimitOneRankLike(input->view(), stream, mr);
  finished_ = true;
  return result;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::reduceToCandidates(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto reduced =
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
      ? computeLimitOneRowNumber(input, stream, mr)
      : computeLimitOneRankLike(input, stream, mr);
  auto table = reduced->release();
  if (generateRowNumber_) {
    auto columns = table->release();
    VELOX_CHECK_EQ(
        columns.size(),
        inputType_->size() + 1,
        "Incremental TopN candidate has unexpected generated rank column");
    columns.pop_back();
    table = std::make_unique<cudf::table>(std::move(columns));
  }
  return table;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::reduceOwnedCandidates(
    std::unique_ptr<cudf::table> input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_NOT_NULL(input);
  if (!topNUniquePartitionFastPathEnabled() ||
      rankFunction_ != core::TopNRowNumberNode::RankFunction::kRowNumber ||
      partitionKeys_.empty()) {
    return reduceToCandidates(input->view(), stream, mr);
  }

  const auto inputRows = input->num_rows();
  const auto probeStart = std::chrono::steady_clock::now();
  bool unique = inputRows <= 1;
  if (!unique) {
    // TopN partition equality follows SQL grouping semantics: null keys form
    // one group and Spark normalizes all NaNs into the same grouping value.
    // Probe only the narrow partition-key view. distinct_count uses the same
    // NaN-equal row comparator as stream compaction, but avoids materializing
    // an index column that this boolean decision would immediately discard.
    // If every key is distinct, the exact
    // Top-1 result is the original table with row_number=1 and no ordering or
    // payload gather is necessary. Any duplicate takes the historical exact
    // reduction path below.
    const auto distinctRows = cudf::distinct_count(
        input->view().select(partitionKeys_),
        cudf::null_equality::EQUAL,
        stream);
    unique = distinctRows == inputRows;
  }
  const auto probeNanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - probeStart)
          .count();
  addRuntimeStat(
      "topNUniquePartitionProbeNanos",
      RuntimeCounter(probeNanos, RuntimeCounter::Unit::kNanos));
  addRuntimeStat("topNUniquePartitionProbeRows", RuntimeCounter(inputRows));
  if (unique) {
    addRuntimeStat(
        "topNUniquePartitionFastPathRows", RuntimeCounter(inputRows));
    addRuntimeStat("topNUniquePartitionFastPathBatches", RuntimeCounter(1));
    return input;
  }

  addRuntimeStat(
      "topNUniquePartitionFallbackRows", RuntimeCounter(inputRows));
  addRuntimeStat("topNUniquePartitionFallbackBatches", RuntimeCounter(1));
  return reduceToCandidates(input->view(), stream, mr);
}

void CudfTopNRowNumber::externalizeSequentialUniqueCandidates(
    std::unique_ptr<cudf::table> candidates,
    uint64_t estimatedBytes,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(candidates);
  if (candidates->num_rows() == 0) {
    return;
  }
  VELOX_CHECK(
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber);
  VELOX_CHECK(!partitionKeys_.empty());

  if (!partitionedRowNumberMode_) {
    partitionedRowNumberMode_ = true;
    partitionedRowNumberHost_.resize(hostPartitionCount_);
    partitionedRowNumberDevice_.resize(hostPartitionCount_);
    partitionedRowNumberDeviceBytes_.resize(hostPartitionCount_, 0);
    partitionedRowNumberSpillFiles_.resize(hostPartitionCount_);
    partitionedRowNumberHashDepths_.resize(hostPartitionCount_, 0);
  }
  if (sequentialUniqueKeyPartitions_.empty()) {
    sequentialUniqueKeyPartitions_.resize(hostPartitionCount_);
  }
  sequentialUniqueMode_ = true;

  auto mr = get_output_mr();
  const auto packStart = std::chrono::steady_clock::now();
  constexpr uint64_t kSequentialPackedChunkBytes = 96ULL << 20;
  const auto inputRows = candidates->num_rows();
  const auto bytesPerRow = std::max<uint64_t>(
      1, (estimatedBytes + inputRows - 1) / inputRows);
  const auto rowsPerChunk = static_cast<cudf::size_type>(
      std::max<uint64_t>(1, kSequentialPackedChunkBytes / bytesPerRow));
  std::vector<cudf::size_type> splitOffsets;
  for (cudf::size_type offset = rowsPerChunk; offset < inputRows;
       offset += rowsPerChunk) {
    splitOffsets.push_back(offset);
  }
  auto packedInputs =
      cudf::contiguous_split(candidates->view(), splitOffsets, stream, mr);
  // Hash only the narrow partition keys. Equal keys, including equal
  // NULL/NaN patterns supported by the historical exact path, are guaranteed
  // to reach the same local bucket. This preserves arrival order for the wide
  // payload while making the final uniqueness proof bounded by one of the
  // existing host partitions instead of the complete multi-GiB key set.
  auto keyView = candidates->view().select(partitionKeys_);
  std::vector<cudf::size_type> localKeyColumns(keyView.num_columns());
  std::iota(localKeyColumns.begin(), localKeyColumns.end(), 0);
  const auto keyHashStart = std::chrono::steady_clock::now();
  auto [partitionedKeys, keyOffsets] = [&]() {
    std::lock_guard<std::mutex> lock(cudfHashPartitionMutex());
    return cudf::hash_partition(
        keyView,
        localKeyColumns,
        hostPartitionCount_,
        cudf::hash_id::HASH_MURMUR3,
        kTopNLocalHashSeed,
        stream,
        get_temp_mr());
  }();
  VELOX_CHECK(
      keyOffsets.size() == hostPartitionCount_ ||
      keyOffsets.size() == hostPartitionCount_ + 1);
  VELOX_CHECK_EQ(keyOffsets.front(), 0);
  keyOffsets.erase(keyOffsets.begin());
  if (keyOffsets.size() == hostPartitionCount_) {
    keyOffsets.pop_back();
  }
  auto packedKeyPartitions =
      cudf::contiguous_split(partitionedKeys->view(), keyOffsets, stream, mr);
  partitionedKeys.reset();
  diagnosticHashPartitionMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - keyHashStart)
          .count();
  VELOX_CHECK_EQ(packedKeyPartitions.size(), hostPartitionCount_);
  uint64_t inputBytes = 0;
  for (const auto& packed : packedInputs) {
    VELOX_CHECK_LE(
        packed.data.gpu_data->size(),
        std::numeric_limits<uint64_t>::max() - inputBytes,
        "Sequential TopN packed input size overflow");
    inputBytes += packed.data.gpu_data->size();
  }
  uint64_t keyBytes = 0;
  for (const auto& packed : packedKeyPartitions) {
    VELOX_CHECK_LE(
        packed.data.gpu_data->size(),
        std::numeric_limits<uint64_t>::max() - keyBytes,
        "Sequential TopN packed key size overflow");
    keyBytes += packed.data.gpu_data->size();
  }
  VELOX_CHECK_LE(
      inputBytes,
      std::numeric_limits<uint64_t>::max() - keyBytes,
      "Sequential TopN packed input size overflow");
  const auto totalBytes = inputBytes + keyBytes;

  auto hostStorage = acquireCudfPackedPinnedBuffer(totalBytes);
  const bool usedPinnedStaging = hostStorage != nullptr;
  if (!hostStorage) {
    hostStorage = std::shared_ptr<uint8_t>(
        new uint8_t[totalBytes], std::default_delete<uint8_t[]>());
  }
  const auto d2hSubmitStart = std::chrono::steady_clock::now();
  uint64_t hostOffset = 0;
  for (const auto& packed : packedInputs) {
    const auto bytes = packed.data.gpu_data->size();
    if (bytes > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostStorage.get() + hostOffset,
          packed.data.gpu_data->data(),
          bytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    hostOffset += bytes;
  }
  uint64_t keyHostOffset = inputBytes;
  for (const auto& packed : packedKeyPartitions) {
    const auto bytes = packed.data.gpu_data->size();
    if (bytes > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostStorage.get() + keyHostOffset,
          packed.data.gpu_data->data(),
          bytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    keyHostOffset += bytes;
  }
  const auto d2hSubmitEnd = std::chrono::steady_clock::now();
  stream.synchronize();
  const auto d2hSynchronizeEnd = std::chrono::steady_clock::now();

  auto keyStorage = std::shared_ptr<uint8_t>(
      new uint8_t[keyBytes], std::default_delete<uint8_t[]>());
  if (keyBytes > 0) {
    std::memcpy(
        keyStorage.get(), hostStorage.get() + inputBytes, keyBytes);
  }
  std::vector<HostPackedChunk> keyChunks(hostPartitionCount_);
  keyHostOffset = inputBytes;
  for (size_t bucket = 0; bucket < packedKeyPartitions.size(); ++bucket) {
    auto& packed = packedKeyPartitions[bucket];
    auto& keyChunk = keyChunks[bucket];
    keyChunk.metadata = std::move(packed.data.metadata);
    keyChunk.dataBytes = packed.data.gpu_data->size();
    keyChunk.storedBytes = keyChunk.dataBytes;
    keyChunk.rows = packed.table.num_rows();
    if (keyChunk.rows > 0) {
      keyChunk.data = std::shared_ptr<uint8_t>(
          keyStorage, keyStorage.get() + keyHostOffset - inputBytes);
    }
    keyHostOffset += keyChunk.dataBytes;
  }

  const auto storeStart = std::chrono::steady_clock::now();
  hostOffset = 0;
  for (auto& packed : packedInputs) {
    HostPackedChunk inputChunk;
    inputChunk.metadata = std::move(packed.data.metadata);
    inputChunk.data = std::shared_ptr<uint8_t>(
        hostStorage, hostStorage.get() + hostOffset);
    inputChunk.dataBytes = packed.data.gpu_data->size();
    inputChunk.storedBytes = inputChunk.dataBytes;
    inputChunk.rows = packed.table.num_rows();
    inputChunk.pinned = usedPinnedStaging;
    hostOffset += inputChunk.dataBytes;
    const auto bucket =
        sequentialUniqueNextBucket_++ % hostPartitionCount_;
    storeHostPartitionChunk(std::move(inputChunk), bucket);
  }
  for (size_t bucket = 0; bucket < keyChunks.size(); ++bucket) {
    if (keyChunks[bucket].rows > 0) {
      sequentialUniqueKeyPartitions_[bucket].push_back(
          std::move(keyChunks[bucket]));
    }
  }
  sequentialUniqueKeyBytes_ += keyBytes;
  sequentialUniqueKeyRows_ += inputRows;

  diagnosticD2hSubmitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          d2hSubmitEnd - d2hSubmitStart)
          .count();
  diagnosticD2hSynchronizeMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          d2hSynchronizeEnd - d2hSubmitEnd)
          .count();
  diagnosticStoreMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - storeStart)
          .count();
  ++diagnosticExternalizeCalls_;
  addRuntimeStat(
      "topNSequentialStoredBytes",
      RuntimeCounter(inputBytes, RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNSequentialKeyBytes",
      RuntimeCounter(keyBytes, RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNSequentialPackSubmitNanos",
      RuntimeCounter(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              d2hSubmitStart - packStart)
              .count(),
          RuntimeCounter::Unit::kNanos));

  // The host copies above are complete, so release the transient input and
  // packed device owners before a possible prefix conversion allocates its
  // bounded restore/hash workspace.
  packedInputs.clear();
  packedKeyPartitions.clear();
  candidates.reset();
  ++sequentialEarlyDenseInputBatches_;
  maybeSwitchSequentialDenseInputMode(stream);
}

void CudfTopNRowNumber::maybeSwitchSequentialDenseInputMode(
    rmm::cuda_stream_view stream) {
  const auto probeEvery = topNSequentialEarlyDenseProbeBatches();
  if (probeEvery == 0 || sequentialDenseInputMode_ ||
      sequentialEarlyDenseInputBatches_ % probeEvery != 0 ||
      sequentialUniqueKeyPartitions_.empty()) {
    return;
  }

  // Probe one deterministic hash bucket. Equal keys always reach the same
  // bucket, so its exact distinct ratio is a representative and very small
  // signal for whether the wide arrival-order optimization is still useful.
  // The original key chunks remain untouched for the eventual global proof
  // if this performance-only probe does not select the dense route.
  size_t probeBucket = 0;
  while (probeBucket < sequentialUniqueKeyPartitions_.size() &&
         sequentialUniqueKeyPartitions_[probeBucket].empty()) {
    ++probeBucket;
  }
  if (probeBucket == sequentialUniqueKeyPartitions_.size()) {
    return;
  }

  const auto& keyChunks = sequentialUniqueKeyPartitions_[probeBucket];
  std::vector<CudfPackedHostRestoreChunk> prepared;
  prepared.reserve(keyChunks.size());
  uint64_t probeRows = 0;
  for (const auto& chunk : keyChunks) {
    VELOX_CHECK_NOT_NULL(chunk.metadata);
    VELOX_CHECK(
        chunk.dataBytes == 0 || chunk.data != nullptr,
        "Sequential early-dense key chunk has no host data");
    probeRows += chunk.rows;
    auto owner = chunk.data;
    CudfPackedHostRestoreChunk copy;
    copy.metadata =
        std::make_unique<std::vector<uint8_t>>(*chunk.metadata);
    copy.dataBytes = chunk.dataBytes;
    copy.keepAlive = owner;
    copy.materializeIntoPinned =
        [owner = std::move(owner), bytes = chunk.dataBytes](
            uint8_t* destination) {
          VELOX_CHECK(
              bytes == 0 || destination != nullptr,
              "Sequential early-dense key probe has no destination");
          if (bytes > 0) {
            std::memcpy(destination, owner.get(), bytes);
          }
        };
    prepared.push_back(std::move(copy));
  }
  if (probeRows == 0) {
    return;
  }

  const auto probeStart = std::chrono::steady_clock::now();
  auto restored = bulkRestoreCudfPackedHostChunks(
      std::move(prepared), stream, get_output_mr());
  auto merged = cudf::concatenate(
      restored.tables(), stream, get_output_mr());
  const auto distinctRows = static_cast<uint64_t>(cudf::distinct_count(
      merged->view(), cudf::null_equality::EQUAL, stream));
  const auto probeNanos =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - probeStart)
          .count();
  sequentialEarlyDenseProbeRows_ += probeRows;
  sequentialEarlyDenseDistinctRows_ += distinctRows;
  addRuntimeStat(
      "topNSequentialEarlyDenseProbeRows", RuntimeCounter(probeRows));
  addRuntimeStat(
      "topNSequentialEarlyDenseDistinctRows", RuntimeCounter(distinctRows));
  addRuntimeStat(
      "topNSequentialEarlyDenseProbeNanos",
      RuntimeCounter(probeNanos, RuntimeCounter::Unit::kNanos));

  const auto maxDistinctPct =
      topNSequentialEarlyDenseMaxDistinctPct();
  const bool dense =
      static_cast<unsigned __int128>(distinctRows) * 100 <=
      static_cast<unsigned __int128>(probeRows) * maxDistinctPct;
  LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
               << " sequential early-dense probe inputBatches="
               << sequentialEarlyDenseInputBatches_
               << " bucket=" << probeBucket
               << " rows=" << probeRows
               << " distinctRows=" << distinctRows
               << " distinctPct="
               << (100.0 * distinctRows / probeRows)
               << " maxDistinctPct=" << maxDistinctPct
               << " dense=" << dense;
  merged.reset();
  restored = CudfBulkPackedRestore{};
  if (dense) {
    switchSequentialDenseInputMode(stream);
  }
}

void CudfTopNRowNumber::switchSequentialDenseInputMode(
    rmm::cuda_stream_view /*stream*/) {
  VELOX_CHECK(!sequentialDenseInputMode_);
  VELOX_CHECK_EQ(partitionedDeviceBytes_, 0);

  uint64_t prefixRows = 0;
  uint64_t prefixBytes = 0;
  VELOX_CHECK(sequentialDensePrefixHost_.empty());
  for (auto& bucket : partitionedRowNumberHost_) {
    for (auto& chunk : bucket) {
      prefixRows += chunk.rows;
      prefixBytes += chunk.dataBytes;
      sequentialDensePrefixHost_.push_back(std::move(chunk));
    }
    bucket.clear();
  }

  partitionedRowNumberHost_.clear();
  partitionedRowNumberHost_.resize(hostPartitionCount_);
  partitionedRowNumberDevice_.clear();
  partitionedRowNumberDevice_.resize(hostPartitionCount_);
  partitionedRowNumberDeviceBytes_.assign(hostPartitionCount_, 0);
  partitionedRowNumberSpillFiles_.clear();
  partitionedRowNumberSpillFiles_.resize(hostPartitionCount_);
  partitionedRowNumberHashDepths_.assign(hostPartitionCount_, 0);
  nextHostPartition_ = 0;

  sequentialUniqueKeyPartitions_.clear();
  sequentialDuplicateKeyTables_.clear();
  sequentialDuplicateKeys_.reset();
  sequentialDuplicateFilter_.reset();
  sequentialUniqueKeyBytes_ = 0;
  sequentialUniqueKeyRows_ = 0;
  sequentialUniqueMode_ = false;
  sequentialUniqueProbeComplete_ = false;
  sequentialUniqueConfirmed_ = false;
  sequentialDenseInputMode_ = true;
  sequentialDensePrefixRows_ = prefixRows;
  sequentialDensePrefixBytes_ = prefixBytes;
  sequentialDenseConvertedRows_ = 0;
  sequentialDenseConvertedBytes_ = 0;

  addRuntimeStat(
      "topNSequentialEarlyDenseSwitches", RuntimeCounter(1));
  addRuntimeStat(
      "topNSequentialEarlyDensePrefixRows", RuntimeCounter(prefixRows));
  addRuntimeStat(
      "topNSequentialEarlyDensePrefixBytes",
      RuntimeCounter(prefixBytes, RuntimeCounter::Unit::kBytes));
  LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
               << " deferred sequential prefix for bounded dense hash drain"
               << " inputBatches=" << sequentialEarlyDenseInputBatches_
               << " prefixChunks=" << sequentialDensePrefixHost_.size()
               << " prefixRows=" << prefixRows
               << " prefixBytes=" << prefixBytes
               << " hostBytes=" << partitionedHostBytes_
               << " deviceBytes=" << partitionedDeviceBytes_
               << " diskBytes=" << partitionedDiskBytes_;
}

void CudfTopNRowNumber::spillSequentialDensePrefixDeviceState(
    rmm::cuda_stream_view stream,
    bool flushAll) {
  // Prefix conversion happens after noMoreInput(), so retaining its output on
  // device cannot accelerate a later input batch.  More importantly, Job 144
  // has two dense Final TopN drivers per GPU.  Letting each reuse the normal
  // 8-GiB input-phase resident cache left 10--13 GiB live just as sibling JSON
  // projections resumed.  Bound the combined transient state to roughly one
  // GiB per GPU and finish every prefix drain at zero resident bytes.
  constexpr uint64_t kDenseDrainDeviceWaveBytes = 512ULL << 20;
  if (partitionedDeviceBytes_ == 0 ||
      (!flushAll &&
       partitionedDeviceBytes_ <= kDenseDrainDeviceWaveBytes)) {
    return;
  }

  auto remainingBytes = partitionedRowNumberDeviceBytes_;
  auto projectedBytes = partitionedDeviceBytes_;
  const uint64_t targetBytes = flushAll ? 0 : kDenseDrainDeviceWaveBytes / 2;
  std::vector<size_t> victims;
  while (projectedBytes > targetBytes) {
    const auto largest =
        std::max_element(remainingBytes.begin(), remainingBytes.end());
    VELOX_CHECK(
        largest != remainingBytes.end() && *largest > 0,
        "Dense TopN prefix has resident bytes without a spillable bucket");
    const auto bucket =
        static_cast<size_t>(std::distance(remainingBytes.begin(), largest));
    projectedBytes -= *largest;
    *largest = 0;
    victims.push_back(bucket);
  }
  const auto before = partitionedDeviceBytes_;
  spillDevicePartitions(victims, stream);
  const auto spilledBytes = before - partitionedDeviceBytes_;
  addRuntimeStat(
      "topNSequentialEarlyDenseDeviceSpillBytes",
      RuntimeCounter(spilledBytes, RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNSequentialEarlyDenseDeviceSpillWaves", RuntimeCounter(1));
}

void CudfTopNRowNumber::externalizeRowNumberCandidates(
    std::unique_ptr<cudf::table> candidates,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(candidates);
  if (candidates->num_rows() == 0) {
    return;
  }
  VELOX_CHECK(
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber);
  VELOX_CHECK(!partitionKeys_.empty());

  if (!partitionedRowNumberMode_) {
    partitionedRowNumberMode_ = true;
    partitionedRowNumberHost_.resize(hostPartitionCount_);
    partitionedRowNumberDevice_.resize(hostPartitionCount_);
    partitionedRowNumberDeviceBytes_.resize(hostPartitionCount_, 0);
    partitionedRowNumberSpillFiles_.resize(hostPartitionCount_);
    partitionedRowNumberHashDepths_.resize(hostPartitionCount_, 0);
  }

  auto mr = get_output_mr();
  const auto hashPartitionStart = std::chrono::steady_clock::now();
  auto [partitioned, offsets] = [&]() {
    std::lock_guard<std::mutex> lock(cudfHashPartitionMutex());
    auto result = cudf::hash_partition(
        candidates->view(),
        partitionKeys_,
        hostPartitionCount_,
        cudf::hash_id::HASH_MURMUR3,
        kTopNLocalHashSeed,
        stream,
        get_temp_mr());
    synchronizeTopNPhase(
        stream,
        diagnosticNodeId_,
        diagnosticInputBatches_,
        "externalize-hash-partition",
        true);
    return result;
  }();
  const auto hashPartitionEnd = std::chrono::steady_clock::now();
  diagnosticHashPartitionMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          hashPartitionEnd - hashPartitionStart)
          .count();
  VELOX_CHECK(
      offsets.size() == hostPartitionCount_ ||
      offsets.size() == hostPartitionCount_ + 1);
  VELOX_CHECK_EQ(offsets.front(), 0);
  offsets.erase(offsets.begin());
  if (offsets.size() == hostPartitionCount_) {
    offsets.pop_back();
  }
  // Pack all buckets below the input concat/partition boundary in one libcudf
  // operation. The previous split + pack + synchronize loop issued hundreds
  // of tiny kernels and D2H synchronizations for every candidate run.
  const auto contiguousSplitStart = std::chrono::steady_clock::now();
  auto packedPartitions =
      cudf::contiguous_split(partitioned->view(), offsets, stream, mr);
  synchronizeTopNPhase(
      stream,
      diagnosticNodeId_,
      diagnosticInputBatches_,
      "externalize-contiguous-split");
  partitioned.reset();
  const auto contiguousSplitEnd = std::chrono::steady_clock::now();
  diagnosticContiguousSplitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          contiguousSplitEnd - contiguousSplitStart)
          .count();
  VELOX_CHECK_EQ(packedPartitions.size(), hostPartitionCount_);

  if (deviceResidentBytesLimit_ > 0) {
    uint64_t retainedRows = 0;
    uint64_t retainedBytes = 0;
    for (size_t bucket = 0; bucket < packedPartitions.size(); ++bucket) {
      if (packedPartitions[bucket].table.num_rows() == 0) {
        continue;
      }
      retainedRows += packedPartitions[bucket].table.num_rows();
      retainedBytes += packedPartitions[bucket].data.gpu_data->size();
      retainDevicePartitionChunk(
          std::move(packedPartitions[bucket]), bucket, stream);
    }
    synchronizeTopNPhase(
        stream,
        diagnosticNodeId_,
        diagnosticInputBatches_,
        "externalize-retain-device-partitions");
    evictLargestDevicePartitions(stream);
    synchronizeTopNPhase(
        stream,
        diagnosticNodeId_,
        diagnosticInputBatches_,
        "externalize-evict-device-partitions");
    ++diagnosticExternalizeCalls_;
    LOG(INFO) << "CudfTopNRowNumber node=" << diagnosticNodeId_
              << " retained partitioned ROW_NUMBER=1 candidates rows="
              << retainedRows << " bytes=" << retainedBytes
              << " deviceBytes=" << partitionedDeviceBytes_
              << " hostBytes=" << partitionedHostBytes_
              << " residentHostBytes=" << partitionedResidentHostBytes_
              << " diskBytes=" << partitionedDiskBytes_
              << " evictedDeviceBytes=" << diagnosticDeviceSpillBytes_
              << " evictedBuckets=" << diagnosticDeviceSpillBuckets_
              << " buckets=" << hostPartitionCount_;
    return;
  }

  uint64_t packedBytes = 0;
  for (const auto& packed : packedPartitions) {
    packedBytes += packed.data.gpu_data->size();
  }
  auto hostStorage = acquireCudfPackedPinnedBuffer(packedBytes);
  const bool usedPinnedStaging = hostStorage != nullptr;
  if (!hostStorage) {
    hostStorage = std::shared_ptr<uint8_t>(
        new uint8_t[packedBytes], std::default_delete<uint8_t[]>());
  }
  const auto d2hSubmitStart = std::chrono::steady_clock::now();
  uint64_t hostOffset = 0;
  std::vector<HostPackedChunk> chunks(packedPartitions.size());
  for (size_t bucket = 0; bucket < packedPartitions.size(); ++bucket) {
    auto& packed = packedPartitions[bucket];
    auto& chunk = chunks[bucket];
    chunk.dataBytes = packed.data.gpu_data->size();
    chunk.storedBytes = chunk.dataBytes;
    chunk.rows = packed.table.num_rows();
    chunk.metadata = std::move(packed.data.metadata);
    chunk.pinned = usedPinnedStaging;
    if (chunk.dataBytes > 0) {
      chunk.data =
          std::shared_ptr<uint8_t>(hostStorage, hostStorage.get() + hostOffset);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          chunk.data.get(),
          packed.data.gpu_data->data(),
          chunk.dataBytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    hostOffset += chunk.dataBytes;
  }
  const auto d2hSubmitEnd = std::chrono::steady_clock::now();
  stream.synchronize();
  const auto d2hSynchronizeEnd = std::chrono::steady_clock::now();
  diagnosticD2hSubmitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          d2hSubmitEnd - d2hSubmitStart)
          .count();
  diagnosticD2hSynchronizeMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          d2hSynchronizeEnd - d2hSubmitEnd)
          .count();

  const auto storeStart = std::chrono::steady_clock::now();
  uint64_t externalizedBytes = 0;
  uint64_t externalizedRows = 0;
  for (size_t bucket = 0; bucket < chunks.size(); ++bucket) {
    if (chunks[bucket].rows == 0) {
      continue;
    }
    const auto hostBytesBefore = partitionedHostBytes_;
    externalizedRows += chunks[bucket].rows;
    storeHostPartitionChunk(std::move(chunks[bucket]), bucket);
    externalizedBytes += partitionedHostBytes_ - hostBytesBefore;
  }
  diagnosticStoreMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - storeStart)
          .count();
  ++diagnosticExternalizeCalls_;
  LOG(INFO) << "CudfTopNRowNumber node=" << diagnosticNodeId_
            << " externalized ROW_NUMBER=1 candidates rows="
            << externalizedRows << " bytes=" << externalizedBytes
            << " hostBytes=" << partitionedHostBytes_
            << " residentHostBytes=" << partitionedResidentHostBytes_
            << " diskBytes=" << partitionedDiskBytes_
            << " diskUncompressedBytes="
            << partitionedDiskUncompressedBytes_
            << " buckets=" << hostPartitionCount_;
}

void CudfTopNRowNumber::storeHostPartitionChunk(
    cudf::table_view partition,
    size_t bucket,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_LT(bucket, partitionedRowNumberHost_.size());
  auto packed = cudf::pack(partition, stream, mr);
  HostPackedChunk chunk;
  chunk.dataBytes = packed.gpu_data->size();
  chunk.storedBytes = chunk.dataBytes;
  chunk.rows = partition.num_rows();
  // Pageable host memory is deliberate: complete Top1 state can be tens of
  // GiB and must not consume the process-wide pinned-memory budget.
  chunk.data = std::shared_ptr<uint8_t>(
      new uint8_t[chunk.dataBytes], std::default_delete<uint8_t[]>());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      chunk.data.get(),
      packed.gpu_data->data(),
      chunk.dataBytes,
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  chunk.metadata = std::move(packed.metadata);
  storeHostPartitionChunk(std::move(chunk), bucket);
}

void CudfTopNRowNumber::retainDevicePartitionChunk(
    cudf::packed_table packed,
    size_t bucket,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_LT(bucket, partitionedRowNumberDevice_.size());
  DevicePackedChunk chunk;
  chunk.dataBytes = packed.data.gpu_data->size();
  chunk.rows = packed.table.num_rows();
  chunk.producerStream = stream;
  chunk.packed =
      std::make_unique<cudf::packed_table>(std::move(packed));

  if (auto* devicePool = customPool(kCudfDeviceMemoryResourceTag);
      devicePool != nullptr && chunk.dataBytes > 0) {
    // reportExternalAllocation() enters Velox shared arbitration when this
    // executor's persistent device-state budget is exhausted. The arbitrator
    // pauses victim tasks and invokes their operator reclaimers, which sweep
    // whole buckets rather than forcing smaller input batches.
    // Driver execution marks operator methods non-reclaimable by default. As
    // with CPU Velox spill operators, make only the reservation boundary
    // reclaimable. The incoming chunk has not entered our state yet, so the
    // arbitrator can safely sweep existing buckets from this same TopN when
    // all other victims are temporarily busy.
    Operator::ReclaimableSectionGuard reclaimableSection(this);
    chunk.admission =
        acquireDeviceMemoryAdmission(devicePool, chunk.dataBytes, this);
  } else if (globalDeviceResidentCapacityBytes_ > 0) {
    int device = -1;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    chunk.admission = tryAcquireDeviceMemoryAdmission(
        device, chunk.dataBytes, globalDeviceResidentCapacityBytes_);
    if (!chunk.admission.has_value()) {
      // Count pressure at the admission boundary even if cooperative reclaim
      // below creates enough headroom for the retry. This metric describes
      // demand rejected by the first attempt, not unrecoverable spill bytes.
      diagnosticGlobalAdmissionRejectedBytes_ += chunk.dataBytes;
      // Reclaim in one bucket-major wave down to a global low watermark.
      // Spilling only the rejected incoming chunk turns steady pressure into
      // thousands of tiny D2H submissions because every following chunk is
      // rejected again. Evicting the largest local buckets amortizes the copy
      // and creates admission headroom for subsequent candidate runs.
      const auto reserved = deviceMemoryAdmissionReservedBytes(device);
      const auto lowWatermark = globalDeviceResidentCapacityBytes_ / 2;
      const auto targetFree =
          reserved > lowWatermark ? reserved - lowWatermark : chunk.dataBytes;
      auto remainingBytes = partitionedRowNumberDeviceBytes_;
      uint64_t selectedBytes = 0;
      std::vector<size_t> victims;
      while (selectedBytes < targetFree) {
        const auto largest = std::max_element(
            remainingBytes.begin(), remainingBytes.end());
        if (largest == remainingBytes.end() || *largest == 0) {
          break;
        }
        const auto victim = static_cast<size_t>(
            std::distance(remainingBytes.begin(), largest));
        selectedBytes += *largest;
        *largest = 0;
        victims.push_back(victim);
      }
      if (!victims.empty()) {
        spillDevicePartitions(victims, stream);
        chunk.admission = tryAcquireDeviceMemoryAdmission(
            device, chunk.dataBytes, globalDeviceResidentCapacityBytes_);
      }
    }
    if (!chunk.admission.has_value()) {
      // The packed chunk already exists on device, so make it visible to the
      // normal bucket spill path and externalize it immediately. This avoids
      // admitting persistent state beyond the shared per-GPU budget while
      // preserving the batched D2H/store implementation.
      partitionedDeviceBytes_ += chunk.dataBytes;
      partitionedRowNumberDeviceBytes_[bucket] += chunk.dataBytes;
      partitionedRowNumberDevice_[bucket].push_back(std::move(chunk));
      spillDevicePartition(bucket, stream);
      return;
    }
  }
  partitionedDeviceBytes_ += chunk.dataBytes;
  partitionedRowNumberDeviceBytes_[bucket] += chunk.dataBytes;
  partitionedRowNumberDevice_[bucket].push_back(std::move(chunk));
}

bool CudfTopNRowNumber::canReclaim() const {
  return partitionedDeviceBytes_ > 0;
}

bool CudfTopNRowNumber::reclaimableBytes(
    uint64_t& reclaimableBytes) const {
  reclaimableBytes = partitionedDeviceBytes_;
  return true;
}

void CudfTopNRowNumber::reclaim(
    uint64_t targetBytes,
    memory::MemoryReclaimer::Stats& /*stats*/) {
  if (partitionedDeviceBytes_ == 0) {
    return;
  }

  const auto reclaimTarget = targetBytes == 0
      ? partitionedDeviceBytes_
      : std::min<uint64_t>(targetBytes, partitionedDeviceBytes_);
  auto remainingBytes = partitionedRowNumberDeviceBytes_;
  uint64_t selectedBytes = 0;
  std::vector<size_t> victims;
  while (selectedBytes < reclaimTarget) {
    const auto largest =
        std::max_element(remainingBytes.begin(), remainingBytes.end());
    if (largest == remainingBytes.end() || *largest == 0) {
      break;
    }
    const auto bucket =
        static_cast<size_t>(std::distance(remainingBytes.begin(), largest));
    selectedBytes += *largest;
    *largest = 0;
    victims.push_back(bucket);
  }
  if (victims.empty()) {
    return;
  }

  const auto before = partitionedDeviceBytes_;
  auto stream = cudfGlobalStreamPool().get_stream();
  spillDevicePartitions(victims, stream);
  const auto reclaimed = before - partitionedDeviceBytes_;
  addRuntimeStat(
      "deviceArbitrationReclaimBytes",
      RuntimeCounter(reclaimed, RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "deviceArbitrationReclaimCount",
      RuntimeCounter(1, RuntimeCounter::Unit::kNone));
  LOG(WARNING) << "CudfTopNRowNumber task=" << taskId()
               << " node=" << planNodeId()
               << " swept persistent device state bytes=" << reclaimed
               << " requested=" << targetBytes
               << " remaining=" << partitionedDeviceBytes_;
}

void CudfTopNRowNumber::handoffDevicePartitionChunk(
    DevicePackedChunk& chunk,
    rmm::cuda_stream_view consumerStream) {
  VELOX_CHECK_NOT_NULL(chunk.packed);
  VELOX_CHECK_NOT_NULL(chunk.packed->data.gpu_data);
  if (chunk.producerStream.value() != consumerStream.value()) {
    cudaEvent_t producerReady{nullptr};
    CUDF_CUDA_TRY(
        cudaEventCreateWithFlags(&producerReady, cudaEventDisableTiming));
    try {
      CUDF_CUDA_TRY(
          cudaEventRecord(producerReady, chunk.producerStream.value()));
      CUDF_CUDA_TRY(
          cudaStreamWaitEvent(consumerStream.value(), producerReady, 0));
    } catch (...) {
      cudaEventDestroy(producerReady);
      throw;
    }
    // CUDA retains an event until already-submitted waits complete, so it is
    // safe to release the host handle immediately after cudaStreamWaitEvent.
    CUDF_CUDA_TRY(cudaEventDestroy(producerReady));
  }

  // The packed buffer is consumed on consumerStream.  Rebind its future RMM
  // deallocation to the same stream before the owning vector is cleared;
  // otherwise the old producer stream can free/reuse the storage while
  // concatenate or D2H is still reading it.
  chunk.packed->data.gpu_data->set_stream(consumerStream);
  chunk.producerStream = consumerStream;
}

void CudfTopNRowNumber::spillDevicePartition(
    size_t bucket,
    rmm::cuda_stream_view stream) {
  spillDevicePartitions({bucket}, stream);
}

void CudfTopNRowNumber::spillDevicePartitions(
    const std::vector<size_t>& buckets,
    rmm::cuda_stream_view stream) {
  struct PendingHostChunk {
    size_t bucket;
    HostPackedChunk chunk;
  };

  uint64_t bytes = 0;
  size_t chunkCount = 0;
  for (const auto bucket : buckets) {
    VELOX_CHECK_LT(bucket, partitionedRowNumberDevice_.size());
    bytes += partitionedRowNumberDeviceBytes_[bucket];
    chunkCount += partitionedRowNumberDevice_[bucket].size();
  }
  if (bytes == 0) {
    return;
  }

  auto hostStorage = acquireCudfPackedPinnedBuffer(bytes);
  const bool usedPinnedStaging = hostStorage != nullptr;
  if (!hostStorage) {
    hostStorage = std::shared_ptr<uint8_t>(
        new uint8_t[bytes], std::default_delete<uint8_t[]>());
  }

  std::vector<PendingHostChunk> hostChunks;
  hostChunks.reserve(chunkCount);
  uint64_t hostOffset = 0;
  const auto d2hSubmitStart = std::chrono::steady_clock::now();
  for (const auto bucket : buckets) {
    for (auto& deviceChunk : partitionedRowNumberDevice_[bucket]) {
      handoffDevicePartitionChunk(deviceChunk, stream);
      HostPackedChunk hostChunk;
      hostChunk.dataBytes = deviceChunk.dataBytes;
      hostChunk.storedBytes = deviceChunk.dataBytes;
      hostChunk.rows = deviceChunk.rows;
      hostChunk.metadata = std::move(deviceChunk.packed->data.metadata);
      hostChunk.pinned = usedPinnedStaging;
      if (hostChunk.dataBytes > 0) {
        hostChunk.data = std::shared_ptr<uint8_t>(
            hostStorage, hostStorage.get() + hostOffset);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            hostChunk.data.get(),
            deviceChunk.packed->data.gpu_data->data(),
            hostChunk.dataBytes,
            cudaMemcpyDeviceToHost,
            stream.value()));
      }
      hostOffset += hostChunk.dataBytes;
      hostChunks.push_back(
          PendingHostChunk{bucket, std::move(hostChunk)});
    }
  }
  const auto d2hSubmitEnd = std::chrono::steady_clock::now();
  stream.synchronize();
  const auto d2hSynchronizeEnd = std::chrono::steady_clock::now();
  diagnosticD2hSubmitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          d2hSubmitEnd - d2hSubmitStart)
          .count();
  diagnosticD2hSynchronizeMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          d2hSynchronizeEnd - d2hSubmitEnd)
          .count();

  VELOX_CHECK_GE(partitionedDeviceBytes_, bytes);
  partitionedDeviceBytes_ -= bytes;
  diagnosticDeviceSpillBytes_ += bytes;
  for (const auto bucket : buckets) {
    if (partitionedRowNumberDeviceBytes_[bucket] == 0) {
      continue;
    }
    partitionedRowNumberDevice_[bucket].clear();
    partitionedRowNumberDeviceBytes_[bucket] = 0;
    ++diagnosticDeviceSpillBuckets_;
  }

  const auto storeStart = std::chrono::steady_clock::now();
  for (auto& pending : hostChunks) {
    storeHostPartitionChunk(std::move(pending.chunk), pending.bucket);
  }
  diagnosticStoreMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - storeStart)
          .count();
}

void CudfTopNRowNumber::evictLargestDevicePartitions(
    rmm::cuda_stream_view stream) {
  if (deviceResidentBytesLimit_ == 0 ||
      partitionedDeviceBytes_ <= deviceResidentBytesLimit_) {
    return;
  }
  const auto lowWatermark = deviceResidentBytesLimit_ / 2;
  auto remainingBytes = partitionedRowNumberDeviceBytes_;
  auto projectedDeviceBytes = partitionedDeviceBytes_;
  std::vector<size_t> victims;
  while (projectedDeviceBytes > lowWatermark) {
    const auto largest = std::max_element(
        remainingBytes.begin(), remainingBytes.end());
    VELOX_CHECK(
        largest != remainingBytes.end() && *largest > 0,
        "TopN device residency exceeds its low watermark without a "
        "resident bucket to evict");
    const auto bucket =
        static_cast<size_t>(std::distance(remainingBytes.begin(), largest));
    projectedDeviceBytes -= *largest;
    *largest = 0;
    victims.push_back(bucket);
  }
  spillDevicePartitions(victims, stream);
}

uint64_t CudfTopNRowNumber::partitionBytes(size_t bucket) const {
  VELOX_CHECK_LT(bucket, partitionedRowNumberHost_.size());
  const auto hostBytes = std::accumulate(
      partitionedRowNumberHost_[bucket].begin(),
      partitionedRowNumberHost_[bucket].end(),
      uint64_t{0},
      [](uint64_t bytes, const HostPackedChunk& chunk) {
        return bytes + chunk.dataBytes;
      });
  const auto deviceBytes =
      bucket < partitionedRowNumberDeviceBytes_.size()
      ? partitionedRowNumberDeviceBytes_[bucket]
      : 0;
  return hostBytes + deviceBytes;
}

void CudfTopNRowNumber::storeHostPartitionChunk(
    HostPackedChunk chunk,
    size_t bucket,
    bool enableCompression) {
  VELOX_CHECK_LT(bucket, partitionedRowNumberHost_.size());
  VELOX_CHECK_LE(
      chunk.dataBytes,
      static_cast<uint64_t>(std::numeric_limits<int>::max()),
      "Packed cuDF spill chunk exceeds the LZ4 block size limit");
  auto hostReservation = tryReserveCudfPackedHostMemory(
      chunk.dataBytes, hostResidentBytesLimit_);
  if (!hostReservation) {
    auto& spillFile = partitionedRowNumberSpillFiles_[bucket];
    if (!spillFile) {
      const auto& taskSpillDirectory =
          operatorCtx_->task()->getOrCreateSpillDirectory();
      spillFile = createCudfPackedSpillFile(
          taskSpillDirectory,
          fmt::format(
              "cudf-topn-{}-p{:06}-{:06}.bin",
              diagnosticNodeId_,
              bucket,
              spillDirectorySequence.fetch_add(1)),
          bucket,
          spillConfig());
      LOG(INFO) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                << " opened packed spill file=" << spillFile->path()
                << " bucket=" << bucket;
    }
    const auto spillEnqueueStart = std::chrono::steady_clock::now();
    auto writeFuture = spillFile->appendCompressedAsync(
        chunk.data, chunk.dataBytes, enableCompression);
    diagnosticSpillEnqueueMicros_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - spillEnqueueStart)
            .count();
    chunk.spillFile = spillFile;
    chunk.spillWriteFuture = std::move(writeFuture);
    chunk.data.reset();
    // Until the asynchronous compression resolves, account the conservative
    // uncompressed upper bound. restoreHostChunk replaces it with the exact
    // stored size.
    partitionedDiskBytes_ += chunk.dataBytes;
    partitionedDiskUncompressedBytes_ += chunk.dataBytes;
  } else {
    if (chunk.pinned && chunk.dataBytes > 0) {
      auto pageable = std::shared_ptr<uint8_t>(
          new uint8_t[chunk.dataBytes], std::default_delete<uint8_t[]>());
      std::memcpy(pageable.get(), chunk.data.get(), chunk.dataBytes);
      chunk.data = std::move(pageable);
      chunk.pinned = false;
    }
    chunk.hostReservation = std::move(hostReservation);
    partitionedResidentHostBytes_ += chunk.dataBytes;
    chunk.accountedResidentHost = true;
  }
  partitionedHostBytes_ += chunk.dataBytes;
  partitionedRowNumberHost_[bucket].push_back(std::move(chunk));
}

CudfPackedHostRestoreChunk
CudfTopNRowNumber::prepareHostChunkForBulkRestore(HostPackedChunk chunk) {
  if (chunk.data) {
    return materializeHostChunk(std::move(chunk));
  }
  VELOX_CHECK_NOT_NULL(
      chunk.spillFile,
      "Packed cuDF chunk has neither host data nor spill input");
  // Resolve every write while the caller is still assembling the restore
  // wave. appendCompressedAsync releases its pinned D2H source before making
  // this future ready, so by the time bulk restore acquires two bounce slabs
  // it is not competing with the bucket's own completed spill writes.
  if (chunk.spillWriteFuture.valid()) {
    const auto result = chunk.spillWriteFuture.get();
    VELOX_CHECK_GE(partitionedDiskBytes_, chunk.dataBytes);
    partitionedDiskBytes_ -= chunk.dataBytes;
    partitionedDiskBytes_ += result.storedBytes;
    chunk.fileOffset = result.fileOffset;
    chunk.storedBytes = result.storedBytes;
    chunk.compressed = result.compressed;
    chunk.spillWriteFuture = {};
    diagnosticCompressionMicros_ += result.compressionMicros;
    addRuntimeStat(
        "topNSpillInputBytes",
        RuntimeCounter(chunk.dataBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSpillStoredBytes",
        RuntimeCounter(result.storedBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSpillCompressionNanos",
        RuntimeCounter(
            result.compressionMicros * 1'000, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        result.compressed ? "topNSpillCompressedChunks"
                          : "topNSpillRawChunks",
        RuntimeCounter(1));
  }
  auto owner = std::make_shared<HostPackedChunk>(std::move(chunk));
  CudfPackedHostRestoreChunk prepared;
  prepared.metadata = std::move(owner->metadata);
  prepared.dataBytes = owner->dataBytes;
  prepared.keepAlive = owner;
  prepared.materializeIntoPinned =
      [this, owner = std::move(owner)](uint8_t* destination) mutable {
        materializeDiskChunkInto(std::move(*owner), destination);
        owner.reset();
      };
  return prepared;
}

void CudfTopNRowNumber::materializeDiskChunkInto(
    HostPackedChunk chunk,
    uint8_t* destination) {
  VELOX_CHECK_NULL(chunk.data);
  VELOX_CHECK_NOT_NULL(chunk.spillFile);
  VELOX_CHECK(
      chunk.dataBytes == 0 || destination != nullptr,
      "Packed cuDF disk chunk has {} bytes but no restore destination",
      chunk.dataBytes);
  if (chunk.spillWriteFuture.valid()) {
    const auto result = chunk.spillWriteFuture.get();
    {
      std::lock_guard<std::mutex> lock(partitionedDiskAccountingMutex_);
      VELOX_CHECK_GE(partitionedDiskBytes_, chunk.dataBytes);
      partitionedDiskBytes_ -= chunk.dataBytes;
      partitionedDiskBytes_ += result.storedBytes;
      diagnosticCompressionMicros_ += result.compressionMicros;
    }
    chunk.fileOffset = result.fileOffset;
    chunk.storedBytes = result.storedBytes;
    chunk.compressed = result.compressed;
    addRuntimeStat(
        "topNSpillInputBytes",
        RuntimeCounter(chunk.dataBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSpillStoredBytes",
        RuntimeCounter(result.storedBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSpillCompressionNanos",
        RuntimeCounter(
            result.compressionMicros * 1'000, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        result.compressed ? "topNSpillCompressedChunks"
                          : "topNSpillRawChunks",
        RuntimeCounter(1));
  }

  std::unique_ptr<uint8_t[]> compressed;
  auto* readDestination = destination;
  if (chunk.compressed) {
    compressed = std::make_unique<uint8_t[]>(chunk.storedBytes);
    readDestination = compressed.get();
  }
  chunk.spillFile->read(
      chunk.fileOffset, chunk.storedBytes, readDestination);
  chunk.spillFile->reclaim(chunk.fileOffset, chunk.storedBytes);
  {
    std::lock_guard<std::mutex> lock(partitionedDiskAccountingMutex_);
    VELOX_CHECK_GE(partitionedDiskBytes_, chunk.storedBytes);
    partitionedDiskBytes_ -= chunk.storedBytes;
    VELOX_CHECK_GE(partitionedDiskUncompressedBytes_, chunk.dataBytes);
    partitionedDiskUncompressedBytes_ -= chunk.dataBytes;
  }

  if (chunk.compressed) {
    const auto decompressedBytes = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressed.get()),
        reinterpret_cast<char*>(destination),
        static_cast<int>(chunk.storedBytes),
        static_cast<int>(chunk.dataBytes));
    VELOX_CHECK_EQ(
        decompressedBytes,
        static_cast<int>(chunk.dataBytes),
        "Failed to decompress packed cuDF spill input at offset {}",
        chunk.fileOffset);
  }
}

CudfPackedHostRestoreChunk CudfTopNRowNumber::materializeHostChunk(
    HostPackedChunk chunk) {
  if (chunk.data) {
    if (chunk.accountedResidentHost) {
      VELOX_CHECK_GE(partitionedResidentHostBytes_, chunk.storedBytes);
      partitionedResidentHostBytes_ -= chunk.storedBytes;
    }
  } else {
    VELOX_CHECK_NOT_NULL(
        chunk.spillFile,
        "Packed cuDF chunk has neither host data nor spill input");
    if (chunk.spillWriteFuture.valid()) {
      const auto result = chunk.spillWriteFuture.get();
      VELOX_CHECK_GE(partitionedDiskBytes_, chunk.dataBytes);
      partitionedDiskBytes_ -= chunk.dataBytes;
      partitionedDiskBytes_ += result.storedBytes;
      chunk.fileOffset = result.fileOffset;
      chunk.storedBytes = result.storedBytes;
      chunk.compressed = result.compressed;
      diagnosticCompressionMicros_ += result.compressionMicros;
      addRuntimeStat(
          "topNSpillInputBytes",
          RuntimeCounter(chunk.dataBytes, RuntimeCounter::Unit::kBytes));
      addRuntimeStat(
          "topNSpillStoredBytes",
          RuntimeCounter(result.storedBytes, RuntimeCounter::Unit::kBytes));
      addRuntimeStat(
          "topNSpillCompressionNanos",
          RuntimeCounter(
              result.compressionMicros * 1'000,
              RuntimeCounter::Unit::kNanos));
      addRuntimeStat(
          result.compressed ? "topNSpillCompressedChunks"
                            : "topNSpillRawChunks",
          RuntimeCounter(1));
    }
    chunk.data = std::shared_ptr<uint8_t>(
        new uint8_t[chunk.storedBytes], std::default_delete<uint8_t[]>());
    chunk.spillFile->read(
        chunk.fileOffset, chunk.storedBytes, chunk.data.get());
    chunk.spillFile->reclaim(chunk.fileOffset, chunk.storedBytes);
    VELOX_CHECK_GE(partitionedDiskBytes_, chunk.storedBytes);
    partitionedDiskBytes_ -= chunk.storedBytes;
    VELOX_CHECK_GE(partitionedDiskUncompressedBytes_, chunk.dataBytes);
    partitionedDiskUncompressedBytes_ -= chunk.dataBytes;
  }
  if (chunk.compressed) {
    auto decompressed = std::shared_ptr<uint8_t>(
        new uint8_t[chunk.dataBytes], std::default_delete<uint8_t[]>());
    const auto decompressedBytes = LZ4_decompress_safe(
        reinterpret_cast<const char*>(chunk.data.get()),
        reinterpret_cast<char*>(decompressed.get()),
        static_cast<int>(chunk.storedBytes),
        static_cast<int>(chunk.dataBytes));
    VELOX_CHECK_EQ(
        decompressedBytes,
        static_cast<int>(chunk.dataBytes),
        "Failed to decompress packed cuDF spill input at offset {}",
        chunk.fileOffset);
    chunk.data = std::move(decompressed);
  }
  return CudfPackedHostRestoreChunk{
      std::move(chunk.metadata),
      std::move(chunk.data),
      chunk.dataBytes,
      std::move(chunk.hostReservation)};
}

std::unique_ptr<cudf::packed_table>
CudfTopNRowNumber::restoreHostChunk(
    HostPackedChunk chunk,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    std::vector<std::shared_ptr<uint8_t>>* asyncHostKeepAlive) {
  auto hostChunk = materializeHostChunk(std::move(chunk));
  auto gpuData =
      std::make_unique<rmm::device_buffer>(hostChunk.dataBytes, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      gpuData->data(),
      hostChunk.data.get(),
      hostChunk.dataBytes,
      cudaMemcpyHostToDevice,
      stream.value()));
  if (asyncHostKeepAlive != nullptr) {
    // The caller submits all chunks in one bucket and synchronizes once.
    // Retain pageable/decompressed sources until that synchronization.
    asyncHostKeepAlive->push_back(hostChunk.data);
  } else {
    stream.synchronize();
  }
  cudf::packed_columns columns{
      std::move(hostChunk.metadata), std::move(gpuData)};
  auto view = cudf::unpack(columns);
  return std::make_unique<cudf::packed_table>(
      cudf::packed_table{view, std::move(columns)});
}

void CudfTopNRowNumber::repartitionOversizedHostBucket(
    size_t bucket,
    uint64_t bucketBytes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_LT(bucket, partitionedRowNumberHost_.size());
  VELOX_CHECK_GT(bucketBytes, finalizeInputBytes_);

  const auto depth = partitionedRowNumberHashDepths_[bucket];
  const auto requiredFanout =
      (bucketBytes + finalizeInputBytes_ - 1) / finalizeInputBytes_;
  const auto fanout = static_cast<size_t>(
      std::clamp<uint64_t>(requiredFanout, 2, 64));
  const auto childStart = partitionedRowNumberHost_.size();

  // Move the source state before growing the parallel vectors, since resize
  // may invalidate references into them.
  auto sourceChunks = std::move(partitionedRowNumberHost_[bucket]);
  auto sourceSpillFile =
      std::move(partitionedRowNumberSpillFiles_[bucket]);

  partitionedRowNumberHost_.resize(childStart + fanout);
  partitionedRowNumberDevice_.resize(childStart + fanout);
  partitionedRowNumberDeviceBytes_.resize(childStart + fanout, 0);
  partitionedRowNumberSpillFiles_.resize(childStart + fanout);
  partitionedRowNumberHashDepths_.resize(childStart + fanout, depth + 1);
  for (size_t child = childStart; child < childStart + fanout; ++child) {
    partitionedRowNumberHashDepths_[child] = depth + 1;
  }

  uint64_t sourceRows = 0;
  uint64_t rewrittenBytes = 0;
  uint64_t restoreGroups = 0;
  uint64_t childChunks = 0;
  const auto sourceChunkCount = sourceChunks.size();
  const auto hashSeed =
      static_cast<uint32_t>(0x9e3779b9U * static_cast<uint32_t>(depth + 1));

  // Repartition partition-major groups, not individual 4-8 MiB source
  // chunks. Each group is bounded by the same input limit as normal finalize,
  // and bulk restore gives the group one device allocation, one H2D submission
  // wave and one synchronization.
  size_t sourceIndex = 0;
  while (sourceIndex < sourceChunks.size()) {
    std::vector<CudfPackedHostRestoreChunk> hostChunks;
    uint64_t groupBytes = 0;
    while (sourceIndex < sourceChunks.size()) {
      auto& chunk = sourceChunks[sourceIndex];
      if (!hostChunks.empty() &&
          chunk.dataBytes > finalizeInputBytes_ - groupBytes) {
        break;
      }
      const auto sourceBytes = chunk.dataBytes;
      sourceRows += chunk.rows;
      groupBytes += sourceBytes;
      hostChunks.push_back(
          prepareHostChunkForBulkRestore(std::move(chunk)));
      VELOX_CHECK_GE(partitionedHostBytes_, sourceBytes);
      partitionedHostBytes_ -= sourceBytes;
      ++sourceIndex;
      // A legacy chunk may itself exceed the new bound. Process it alone.
      if (groupBytes >= finalizeInputBytes_) {
        break;
      }
    }
    VELOX_CHECK(!hostChunks.empty());
    auto restored =
        bulkRestoreCudfPackedHostChunks(std::move(hostChunks), stream, mr);
    VELOX_CHECK(!restored.tables().empty());
    auto merged = cudf::concatenate(restored.tables(), stream, mr);
    // Release the bulk packed allocation before hash_partition allocates its
    // output. merged owns a compact copy of the complete bounded group.
    restored = CudfBulkPackedRestore{};

    std::unique_ptr<cudf::table> partitioned;
    std::vector<cudf::size_type> offsets;
    {
      std::lock_guard<std::mutex> lock(cudfHashPartitionMutex());
      auto result = cudf::hash_partition(
          merged->view(),
          partitionKeys_,
          fanout,
          cudf::hash_id::HASH_MURMUR3,
          hashSeed,
          stream,
          get_temp_mr());
      stream.synchronize();
      partitioned = std::move(result.first);
      offsets = std::move(result.second);
    }
    merged.reset();
    VELOX_CHECK(
        offsets.size() == fanout || offsets.size() == fanout + 1);
    VELOX_CHECK_EQ(offsets.front(), 0);
    offsets.erase(offsets.begin());
    if (offsets.size() == fanout) {
      offsets.pop_back();
    }
    rewrittenBytes += storeRepartitionedChildrenWave(
        std::move(partitioned),
        std::move(offsets),
        childStart,
        stream,
        mr,
        childChunks);
    ++restoreGroups;
  }
  sourceChunks.clear();
  sourceSpillFile.reset();

  {
    auto lockedStats = stats_.wlock();
    lockedStats->addRuntimeStat(
        "topNRecursiveSourceChunks", RuntimeCounter(sourceChunkCount));
    lockedStats->addRuntimeStat(
        "topNRecursiveRestoreGroups", RuntimeCounter(restoreGroups));
    lockedStats->addRuntimeStat(
        "topNRecursiveChildChunks", RuntimeCounter(childChunks));
    if (sequentialDenseRawRewrite_) {
      lockedStats->addRuntimeStat(
          "topNSequentialDenseRawRewriteBytes",
          RuntimeCounter(rewrittenBytes, RuntimeCounter::Unit::kBytes));
      lockedStats->addRuntimeStat(
          "topNSequentialDenseRawRewriteChunks",
          RuntimeCounter(childChunks));
    }
  }
  LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
               << " repartitioned oversized finalize bucket=" << bucket
               << " depth=" << depth << " inputRows=" << sourceRows
               << " inputBytes=" << bucketBytes
               << " finalizeInputBytes=" << finalizeInputBytes_
               << " fanout=" << fanout
               << " sourceChunks=" << sourceChunkCount
               << " restoreGroups=" << restoreGroups
               << " childChunks=" << childChunks
               << " rewrittenBytes=" << rewrittenBytes
               << " childStart=" << childStart;
}

uint64_t CudfTopNRowNumber::storeRepartitionedChildrenWave(
    std::unique_ptr<cudf::table> partitioned,
    std::vector<cudf::size_type> offsets,
    size_t childStart,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    uint64_t& childChunks) {
  const auto fanout = partitionedRowNumberHost_.size() - childStart;
  VELOX_CHECK_EQ(offsets.size() + 1, fanout);

  const auto contiguousSplitStart = std::chrono::steady_clock::now();
  auto packedChildren =
      cudf::contiguous_split(partitioned->view(), offsets, stream, mr);
  partitioned.reset();
  diagnosticContiguousSplitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - contiguousSplitStart)
          .count();
  VELOX_CHECK_EQ(packedChildren.size(), fanout);

  uint64_t packedBytes = 0;
  for (const auto& packed : packedChildren) {
    packedBytes += packed.data.gpu_data->size();
  }
  auto hostStorage = acquireCudfPackedPinnedBuffer(packedBytes);
  const bool usedPinnedStaging = hostStorage != nullptr;
  if (!hostStorage) {
    hostStorage = std::shared_ptr<uint8_t>(
        new uint8_t[packedBytes], std::default_delete<uint8_t[]>());
  }

  const auto d2hSubmitStart = std::chrono::steady_clock::now();
  uint64_t hostOffset = 0;
  std::vector<HostPackedChunk> chunks(packedChildren.size());
  for (size_t child = 0; child < packedChildren.size(); ++child) {
    auto& packed = packedChildren[child];
    auto& chunk = chunks[child];
    chunk.dataBytes = packed.data.gpu_data->size();
    chunk.storedBytes = chunk.dataBytes;
    chunk.rows = packed.table.num_rows();
    chunk.metadata = std::move(packed.data.metadata);
    chunk.pinned = usedPinnedStaging;
    if (chunk.dataBytes > 0) {
      chunk.data =
          std::shared_ptr<uint8_t>(hostStorage, hostStorage.get() + hostOffset);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          chunk.data.get(),
          packed.data.gpu_data->data(),
          chunk.dataBytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    hostOffset += chunk.dataBytes;
  }
  const auto d2hSubmitEnd = std::chrono::steady_clock::now();
  stream.synchronize();
  diagnosticD2hSubmitMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          d2hSubmitEnd - d2hSubmitStart)
          .count();
  diagnosticD2hSynchronizeMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - d2hSubmitEnd)
          .count();
  packedChildren.clear();

  const auto storeStart = std::chrono::steady_clock::now();
  uint64_t storedBytes = 0;
  for (size_t child = 0; child < chunks.size(); ++child) {
    if (chunks[child].rows == 0) {
      continue;
    }
    storedBytes += chunks[child].dataBytes;
    storeHostPartitionChunk(
        std::move(chunks[child]),
        childStart + child,
        !sequentialDenseRawRewrite_);
    ++childChunks;
  }
  diagnosticStoreMicros_ +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - storeStart)
          .count();
  return storedBytes;
}

CudfTopNRowNumber::PartitionedOutputResult
CudfTopNRowNumber::computeNextPartitionedRowNumberOutput() {
  // Finalize completion can race the next output request. Reclaim before
  // taking another workspace reservation so completed input owners do not
  // artificially reduce admission headroom.
  reclaimCompletedFinalizeOwnerships();
  // The early dense classifier runs in addInput, but converting its complete
  // accumulated prefix there would exceed that call's fixed 1-GiB admission.
  // Drain one 96-MiB packed prefix chunk per admitted wave instead.  This
  // preserves the performance benefit (only the small prefix is rewritten)
  // while making restore/hash/contiguous_split participate in the same global
  // backpressure as normal TopN finalization.
  while (!sequentialDensePrefixHost_.empty()) {
    const auto packedBytes = sequentialDensePrefixHost_.back().dataBytes;
    const auto rows = static_cast<uint64_t>(
        sequentialDensePrefixHost_.back().rows);
    const auto workspaceBytes = estimateFinalizeWorkspaceBytes(packedBytes);
    ContinueFuture workspaceFuture;
    auto workspaceAdmission = tryAcquireDeviceMemoryWorkspace(
        customPool(kCudfDeviceMemoryResourceTag),
        this,
        workspaceBytes,
        CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
        DeviceMemoryWorkspacePriority::kDrain,
        &deviceWorkspaceRequest_,
        &workspaceFuture);
    if (!workspaceAdmission.has_value()) {
      VELOX_CHECK(!waitingForDeviceWorkspace_);
      deviceWorkspaceFuture_ = std::move(workspaceFuture);
      waitingForDeviceWorkspace_ = true;
      return {PartitionedOutputStatus::kBlocked, nullptr};
    }

    auto stream = cudfGlobalStreamPool().get_stream();
    auto chunk = std::move(sequentialDensePrefixHost_.back());
    sequentialDensePrefixHost_.pop_back();
    VELOX_CHECK_GE(partitionedHostBytes_, packedBytes);
    partitionedHostBytes_ -= packedBytes;
    std::vector<CudfPackedHostRestoreChunk> prepared;
    prepared.push_back(prepareHostChunkForBulkRestore(std::move(chunk)));
    const auto convertStart = std::chrono::steady_clock::now();
    auto restored = bulkRestoreCudfPackedHostChunks(
        std::move(prepared), stream, get_output_mr());
    const auto restoreStats = restored.stats();
    auto wide =
        cudf::concatenate(restored.tables(), stream, get_output_mr());
    externalizeRowNumberCandidates(std::move(wide), stream);
    restored = CudfBulkPackedRestore{};
    spillSequentialDensePrefixDeviceState(stream, false);
    sequentialDenseConvertedRows_ += rows;
    sequentialDenseConvertedBytes_ += packedBytes;
    addRuntimeStat(
        "topNSequentialEarlyDenseConvertedRows", RuntimeCounter(rows));
    addRuntimeStat(
        "topNSequentialEarlyDenseConvertedBytes",
        RuntimeCounter(packedBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSequentialEarlyDenseConvertNanos",
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - convertStart)
                .count(),
            RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNSequentialEarlyDensePinnedBounceBytes",
        RuntimeCounter(
            restoreStats.pinnedBounceBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSequentialEarlyDensePageableDirectBytes",
        RuntimeCounter(
            restoreStats.pageableDirectBytes, RuntimeCounter::Unit::kBytes));
    workspaceAdmission.reset();
  }
  if (sequentialDensePrefixRows_ > 0 &&
      sequentialDenseConvertedRows_ == sequentialDensePrefixRows_) {
    auto stream = cudfGlobalStreamPool().get_stream();
    spillSequentialDensePrefixDeviceState(stream, true);
    VELOX_CHECK_EQ(partitionedDeviceBytes_, 0);
    VELOX_CHECK_EQ(
        sequentialDenseConvertedBytes_, sequentialDensePrefixBytes_);
    LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                 << " completed bounded sequential dense prefix drain"
                 << " rows=" << sequentialDenseConvertedRows_
                 << " bytes=" << sequentialDenseConvertedBytes_
                 << " hostBytes=" << partitionedHostBytes_
                 << " deviceBytes=" << partitionedDeviceBytes_
                 << " diskBytes=" << partitionedDiskBytes_;
    // Use zero as the one-shot completion/log sentinel. Runtime stats retain
    // the original prefix totals.
    sequentialDensePrefixRows_ = 0;
    sequentialDensePrefixBytes_ = 0;
  }
  if (sequentialUniqueMode_ && !sequentialUniqueProbeComplete_) {
    constexpr uint64_t kProbeFixedWorkspace = 512ULL << 20;
    constexpr uint64_t kProbeCopies = 5;
    if (!sequentialSparseStream_) {
      sequentialSparseStream_ = std::make_shared<rmm::cuda_stream>(
          rmm::cuda_stream::flags::non_blocking);
      CudfVector::registerStreamOwner(sequentialSparseStream_);
    }
    while (!sequentialUniqueProbeComplete_) {
      while (sequentialUniqueProbeBucket_ <
                 sequentialUniqueKeyPartitions_.size() &&
             sequentialUniqueKeyPartitions_[sequentialUniqueProbeBucket_]
                 .empty()) {
        ++sequentialUniqueProbeBucket_;
      }
      if (sequentialUniqueProbeBucket_ ==
          sequentialUniqueKeyPartitions_.size()) {
        sequentialUniqueConfirmed_ =
            sequentialDuplicateKeyTables_.empty();
        sequentialUniqueProbeComplete_ = true;
        break;
      }

      auto& keyChunks =
          sequentialUniqueKeyPartitions_[sequentialUniqueProbeBucket_];
      uint64_t bucketBytes = 0;
      uint64_t bucketRows = 0;
      for (const auto& chunk : keyChunks) {
        VELOX_CHECK_LE(
            chunk.dataBytes,
            std::numeric_limits<uint64_t>::max() - bucketBytes,
            "Sequential TopN key bucket size overflow");
        bucketBytes += chunk.dataBytes;
        bucketRows += chunk.rows;
      }
      const auto probeWorkspaceBytes =
          bucketBytes >
              (std::numeric_limits<uint64_t>::max() - kProbeFixedWorkspace) /
                  kProbeCopies
          ? std::numeric_limits<uint64_t>::max()
          : std::max<uint64_t>(
                1ULL << 30,
                bucketBytes * kProbeCopies + kProbeFixedWorkspace);
      sequentialUniqueProbePeakWorkspaceBytes_ = std::max(
          sequentialUniqueProbePeakWorkspaceBytes_, probeWorkspaceBytes);
      ContinueFuture workspaceFuture;
      auto probeWorkspaceAdmission = tryAcquireDeviceMemoryWorkspace(
          customPool(kCudfDeviceMemoryResourceTag),
          this,
          probeWorkspaceBytes,
          CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
          DeviceMemoryWorkspacePriority::kDrain,
          &deviceWorkspaceRequest_,
          &workspaceFuture);
      if (!probeWorkspaceAdmission.has_value()) {
        VELOX_CHECK(!waitingForDeviceWorkspace_);
        deviceWorkspaceFuture_ = std::move(workspaceFuture);
        waitingForDeviceWorkspace_ = true;
        return {PartitionedOutputStatus::kBlocked, nullptr};
      }

      // Every retained duplicate-key table and the persistent filtered join
      // use one operator-owned stream. This gives construction and all later
      // probes an explicit lifetime/ordering domain instead of relying on
      // unrelated global-pool streams being reused in a particular order.
      auto probeStream = sequentialSparseStream_->view();
      auto probeMr = get_output_mr();
      const auto probeStart = std::chrono::steady_clock::now();
      std::vector<CudfPackedHostRestoreChunk> preparedKeys;
      preparedKeys.reserve(keyChunks.size());
      for (auto& chunk : keyChunks) {
        auto keyOwner = std::move(chunk.data);
        CudfPackedHostRestoreChunk prepared;
        prepared.metadata = std::move(chunk.metadata);
        prepared.dataBytes = chunk.dataBytes;
        prepared.keepAlive = keyOwner;
        prepared.materializeIntoPinned =
            [keyOwner = std::move(keyOwner), bytes = chunk.dataBytes](
                uint8_t* destination) {
              VELOX_CHECK(
                  bytes == 0 || destination != nullptr,
                  "Sequential TopN key restore has no destination");
              if (bytes > 0) {
                std::memcpy(destination, keyOwner.get(), bytes);
              }
            };
        preparedKeys.push_back(std::move(prepared));
      }
      keyChunks.clear();

      auto restoredKeys = bulkRestoreCudfPackedHostChunks(
          std::move(preparedKeys), probeStream, probeMr);
      const auto keyRestoreStats = restoredKeys.stats();
      VELOX_CHECK(!restoredKeys.tables().empty());
      auto mergedKeys =
          cudf::concatenate(restoredKeys.tables(), probeStream, probeMr);
      std::vector<cudf::size_type> keyColumns(
          mergedKeys->num_columns());
      std::iota(keyColumns.begin(), keyColumns.end(), 0);
      auto distinctKeys = cudf::distinct(
          mergedKeys->view(),
          keyColumns,
          cudf::duplicate_keep_option::KEEP_ANY,
          cudf::null_equality::EQUAL,
          cudf::nan_equality::ALL_EQUAL,
          probeStream,
          probeMr);
      const auto distinctRows = distinctKeys->num_rows();
      const auto bucketUnique =
          static_cast<uint64_t>(distinctRows) == bucketRows;
      if (!bucketUnique && topNSequentialSparseDuplicatePathEnabled()) {
        // KEEP_NONE returns exactly the keys with one occurrence. Subtracting
        // those from the one-row-per-key table yields one exact representative
        // for every duplicated key, including NULL and all NaN spellings.
        auto singletonKeys = cudf::distinct(
            mergedKeys->view(),
            keyColumns,
            cudf::duplicate_keep_option::KEEP_NONE,
            cudf::null_equality::EQUAL,
            cudf::nan_equality::ALL_EQUAL,
            probeStream,
            probeMr);
        VELOX_CHECK_LE(
            static_cast<uint64_t>(singletonKeys->num_rows()),
            bucketRows,
            "Sequential TopN singleton-key count exceeds bucket rows");
        // Every singleton key contributes exactly one input row. Therefore
        // this subtraction is the exact number of wide rows that the sparse
        // continuation would gather and feed back into the historical TopN
        // path. It catches both many duplicate pairs and one very hot key;
        // duplicate-key cardinality alone cannot distinguish the latter.
        sequentialProvenDuplicateCandidateRows_ +=
            bucketRows - singletonKeys->num_rows();
        std::unique_ptr<cudf::table> duplicateKeys;
        if (singletonKeys->num_rows() == 0) {
          duplicateKeys = std::move(distinctKeys);
        } else {
          cudf::filtered_join singletonFilter(
              singletonKeys->view(),
              cudf::null_equality::EQUAL,
              probeStream);
          auto duplicateIndices = singletonFilter.anti_join(
              distinctKeys->view(), probeStream, get_temp_mr());
          auto duplicateSpan =
              cudf::device_span<cudf::size_type const>{*duplicateIndices};
          auto duplicateIndexColumn = cudf::column_view{duplicateSpan};
          duplicateKeys = cudf::gather(
              distinctKeys->view(),
              duplicateIndexColumn,
              cudf::out_of_bounds_policy::DONT_CHECK,
              probeStream,
              probeMr);
        }
        sequentialDuplicateKeyRows_ += duplicateKeys->num_rows();
        sequentialDuplicateKeyTables_.push_back(std::move(duplicateKeys));
      }
      const auto probeNanos =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - probeStart)
              .count();
      sequentialUniqueProbedRows_ += bucketRows;
      sequentialUniqueProbedDistinctRows_ += distinctRows;
      sequentialUniqueProbeNanos_ += probeNanos;
      addRuntimeStat(
          "topNSequentialUniqueProbeNanos",
          RuntimeCounter(probeNanos, RuntimeCounter::Unit::kNanos));
      addRuntimeStat(
          "topNSequentialUniqueProbeRows", RuntimeCounter(bucketRows));
      addRuntimeStat(
          "topNSequentialKeyPinnedBounceBytes",
          RuntimeCounter(
              keyRestoreStats.pinnedBounceBytes,
              RuntimeCounter::Unit::kBytes));
      addRuntimeStat(
          "topNSequentialKeyPageableDirectBytes",
          RuntimeCounter(
              keyRestoreStats.pageableDirectBytes,
              RuntimeCounter::Unit::kBytes));
      addRuntimeStat(
          "topNSequentialKeyHostStageNanos",
          RuntimeCounter(
              keyRestoreStats.hostStageMicros * 1'000,
              RuntimeCounter::Unit::kNanos));
      addRuntimeStat(
          "topNSequentialKeyCopySyncNanos",
          RuntimeCounter(
              keyRestoreStats.copyStreamSynchronizeMicros * 1'000,
              RuntimeCounter::Unit::kNanos));
      ++sequentialUniqueProbeBucket_;
      if (!bucketUnique &&
          !topNSequentialSparseDuplicatePathEnabled()) {
        sequentialUniqueConfirmed_ = false;
        sequentialUniqueProbeComplete_ = true;
      }
    }

    addRuntimeStat(
        "topNSequentialUniqueProbeRestoreBytes",
        RuntimeCounter(
            sequentialUniqueKeyBytes_, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSequentialProbeWorkspaceBytes",
        RuntimeCounter(
            sequentialUniqueProbePeakWorkspaceBytes_,
            RuntimeCounter::Unit::kBytes));
    if (sequentialUniqueConfirmed_) {
      addRuntimeStat(
          "topNSequentialUniqueConfirmedRows",
          RuntimeCounter(sequentialUniqueKeyRows_));
    }

    const auto sparseMaxCandidatePct =
        topNSequentialSparseMaxCandidatePct();
    const bool hasProvenDuplicateKeys =
        !sequentialDuplicateKeyTables_.empty();
    const bool sparseCandidateRatioAccepted =
        static_cast<unsigned __int128>(
            sequentialProvenDuplicateCandidateRows_) *
            100 <=
        static_cast<unsigned __int128>(sequentialUniqueKeyRows_) *
            sparseMaxCandidatePct;
    if (hasProvenDuplicateKeys) {
      addRuntimeStat(
          "topNSequentialDuplicateKeyRows",
          RuntimeCounter(sequentialDuplicateKeyRows_));
      addRuntimeStat(
          "topNSequentialProvenDuplicateCandidateRows",
          RuntimeCounter(sequentialProvenDuplicateCandidateRows_));
    }

    if (!sequentialUniqueConfirmed_ &&
        topNSequentialSparseDuplicatePathEnabled() &&
        hasProvenDuplicateKeys && sparseCandidateRatioAccepted) {
      std::vector<cudf::table_view> duplicateKeyViews;
      duplicateKeyViews.reserve(sequentialDuplicateKeyTables_.size());
      for (const auto& duplicateKeys : sequentialDuplicateKeyTables_) {
        duplicateKeyViews.push_back(duplicateKeys->view());
      }
      auto sparseStream = sequentialSparseStream_->view();
      sequentialDuplicateKeys_ = duplicateKeyViews.size() == 1
          ? std::move(sequentialDuplicateKeyTables_.front())
          : cudf::concatenate(
                duplicateKeyViews, sparseStream, get_output_mr());
      sequentialDuplicateKeyTables_.clear();
      VELOX_CHECK_EQ(
          sequentialDuplicateKeys_->num_rows(),
          sequentialDuplicateKeyRows_,
          "Sequential TopN duplicate key buckets must be disjoint");
      sequentialDuplicateFilter_ = std::make_unique<cudf::filtered_join>(
          sequentialDuplicateKeys_->view(),
          cudf::null_equality::EQUAL,
          sparseStream);

      // Arrival-order storage buckets have no key semantics. Move them out of
      // the historical candidate vectors so those vectors can receive only
      // the sparse duplicate rows produced during the wide-run drain.
      VELOX_CHECK_EQ(partitionedDeviceBytes_, 0);
      sequentialWideHost_ = std::move(partitionedRowNumberHost_);
      partitionedRowNumberHost_.clear();
      partitionedRowNumberHost_.resize(hostPartitionCount_);
      partitionedRowNumberSpillFiles_.clear();
      partitionedRowNumberSpillFiles_.resize(hostPartitionCount_);
      sequentialWideDrainBucket_ = 0;
      sequentialSparseMode_ = true;
      addRuntimeStat(
          "topNSequentialSparseClassifiedRows",
          RuntimeCounter(sequentialUniqueKeyRows_));
    } else if (!sequentialUniqueConfirmed_) {
      sequentialDenseRawRewrite_ =
          topNSequentialDenseRawRewriteEnabled();
      if (hasProvenDuplicateKeys && !sparseCandidateRatioAccepted) {
        addRuntimeStat(
            "topNSequentialSparseDenseFallbackRows",
            RuntimeCounter(sequentialUniqueKeyRows_));
        addRuntimeStat(
            "topNSequentialSparseDenseFallbackCandidateRows",
            RuntimeCounter(sequentialProvenDuplicateCandidateRows_));
      }
      // Duplicate-key tables were produced asynchronously on the operator
      // stream. Dense fallback does not use them; finish their construction
      // before releasing the stream-owned buffers and returning their device
      // memory to the shared pool.
      if (sequentialSparseStream_ && hasProvenDuplicateKeys) {
        sequentialSparseStream_->view().synchronize();
      }
      sequentialDuplicateFilter_.reset();
      sequentialDuplicateKeys_.reset();
      sequentialDuplicateKeyTables_.clear();

      // Arrival-order storage buckets have no key semantics. Consolidate the
      // untouched wide chunks so the existing oversized-bucket recursion
      // re-hashes the complete input before any per-bucket exact reduction.
      auto& fallback = partitionedRowNumberHost_.front();
      for (size_t bucket = 1;
           bucket < partitionedRowNumberHost_.size(); ++bucket) {
        auto& source = partitionedRowNumberHost_[bucket];
        fallback.insert(
            fallback.end(),
            std::make_move_iterator(source.begin()),
            std::make_move_iterator(source.end()));
        source.clear();
        partitionedRowNumberSpillFiles_[bucket].reset();
      }
      addRuntimeStat(
          "topNSequentialUniqueFallbackRows",
          RuntimeCounter(sequentialUniqueKeyRows_));
      sequentialUniqueMode_ = false;
    }
    LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                 << " sequential unique proof rows="
                 << sequentialUniqueKeyRows_
                 << " probedRows=" << sequentialUniqueProbedRows_
                 << " keyBytes=" << sequentialUniqueKeyBytes_
                 << " probedDistinctRows="
                 << sequentialUniqueProbedDistinctRows_
                 << " duplicateKeyRows="
                 << sequentialDuplicateKeyRows_
                 << " provenDuplicateCandidateRows="
                 << sequentialProvenDuplicateCandidateRows_
                 << " sparseMaxCandidatePct="
                 << sparseMaxCandidatePct
                 << " unique=" << sequentialUniqueConfirmed_
                 << " sparse=" << sequentialSparseMode_
                 << " denseRawRewrite=" << sequentialDenseRawRewrite_
                 << " probeBuckets=" << sequentialUniqueProbeBucket_
                 << " peakWorkspaceBytes="
                 << sequentialUniqueProbePeakWorkspaceBytes_
                 << " probeUs=" << sequentialUniqueProbeNanos_ / 1'000;
    sequentialUniqueKeyPartitions_.clear();
    sequentialUniqueKeyBytes_ = 0;
    if (!sequentialSparseMode_) {
      sequentialUniqueKeyRows_ = 0;
    }
  }
  if (pendingPartitionedDeviceOutput_ ||
      !pendingPartitionedOutputChunks_.empty()) {
    // Finalize admission protects restore/concatenate/reduce, but the
    // resulting bucket can outlive that reservation while downstream consumes
    // bounded output slices. copyTableSlice() and host-chunk restore both
    // allocate a new device result. Without a fresh admission here, several
    // finalizing TopNs can each retain a source bucket and then launch output
    // copies against the same last few MiB of physical headroom.
    //
    // Reserve a conservative slice-sized workspace before every allocating
    // handoff. This is drain work: completing it advances downstream and the
    // final slice releases the retained source bucket. Use a smaller steady
    // state cushion than a full bucket restore so output can still make
    // progress under pressure, while the shared scheduler and cooperative
    // reclaimer prevent an unaccounted cudaMallocAsync OOM.
    uint64_t outputWorkspaceBytes = 0;
    if (pendingPartitionedDeviceOutput_) {
      const auto totalRows = pendingPartitionedDeviceOutput_->num_rows();
      VELOX_CHECK_LT(pendingPartitionedDeviceOutputOffset_, totalRows);
      const auto bytesPerRow = std::max<uint64_t>(
          1, (pendingPartitionedDeviceOutputBytes_ + totalRows - 1) /
              totalRows);
      const auto remainingRows =
          totalRows - pendingPartitionedDeviceOutputOffset_;
      // Transfer the finalized bucket without a copy only when it already
      // satisfies both configured output bounds. The former 1-GiB exception
      // handed 3.8M-row/143-MiB Job 144 buckets to PartitionedOutput even
      // though this operator was configured for 262K rows and 64 MiB. That
      // forced the exchange to slice/concatenate the same variable-width
      // owner under backpressure and exposed a nondeterministic hash-partition
      // launch failure. Oversized buckets must cross this ownership boundary
      // as independent, bounded device tables.
      const bool wholeBucketHandoff =
          pendingPartitionedDeviceOutputOffset_ == 0 &&
          pendingPartitionedDeviceOutputBytes_ <= outputChunkBytes_ &&
          totalRows <= maxOutputRows_;
      const auto byteBoundRows = std::max<uint64_t>(
          1, outputChunkBytes_ / bytesPerRow);
      const auto rows = wholeBucketHandoff
          ? static_cast<uint64_t>(totalRows)
          : static_cast<uint64_t>(std::min<uint64_t>(
                remainingRows,
                std::min<uint64_t>(maxOutputRows_, byteBoundRows)));
      // A whole-bucket ownership handoff is zero-copy.
      if (pendingPartitionedDeviceOutputOffset_ != 0 || rows != totalRows) {
        const auto sliceBytes = rows >
                std::numeric_limits<uint64_t>::max() / bytesPerRow
            ? pendingPartitionedDeviceOutputBytes_
            : std::min<uint64_t>(
                  pendingPartitionedDeviceOutputBytes_, bytesPerRow * rows);
        constexpr uint64_t kOutputCopyFixedWorkspace = 256ULL << 20;
        outputWorkspaceBytes =
            sliceBytes >
                (std::numeric_limits<uint64_t>::max() -
                 kOutputCopyFixedWorkspace) /
                    2
            ? std::numeric_limits<uint64_t>::max()
            : sliceBytes * 2 + kOutputCopyFixedWorkspace;
      }
    } else {
      VELOX_CHECK(!pendingPartitionedOutputChunks_.empty());
      const auto packedBytes =
          pendingPartitionedOutputChunks_.front().dataBytes;
      constexpr uint64_t kOutputRestoreFixedWorkspace = 256ULL << 20;
      outputWorkspaceBytes =
          packedBytes >
              (std::numeric_limits<uint64_t>::max() -
               kOutputRestoreFixedWorkspace) /
                  2
          ? std::numeric_limits<uint64_t>::max()
          : packedBytes * 2 + kOutputRestoreFixedWorkspace;
    }

    std::optional<DeviceMemoryWorkspaceReservation> outputWorkspaceAdmission;
    if (outputWorkspaceBytes > 0) {
      ContinueFuture workspaceFuture;
      const auto outputMinHeadroom = std::min<uint64_t>(
          CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
          1ULL << 30);
      outputWorkspaceAdmission = tryAcquireDeviceMemoryWorkspace(
          customPool(kCudfDeviceMemoryResourceTag),
          this,
          outputWorkspaceBytes,
          outputMinHeadroom,
          DeviceMemoryWorkspacePriority::kOutput,
          &deviceWorkspaceRequest_,
          &workspaceFuture);
      if (!outputWorkspaceAdmission.has_value()) {
        VELOX_CHECK(!waitingForDeviceWorkspace_);
        deviceWorkspaceFuture_ = std::move(workspaceFuture);
        waitingForDeviceWorkspace_ = true;
        return {PartitionedOutputStatus::kBlocked, nullptr};
      }
    }
    return {
        PartitionedOutputStatus::kOutput,
        takeNextPartitionedOutputBatch()};
  }

  // The exact duplicate-key filter is typically tiny relative to the wide
  // Final TopN input. Restore one original packed run, probe only its narrow
  // partition keys, and route the two disjoint row-index sets:
  //   * singleton keys are already final and receive row_number=1 directly;
  //   * duplicated keys alone enter the historical exact Top-1 buckets.
  // This preserves correctness without hash-collision false negatives while
  // avoiding a hash/scatter/restore/reduce cycle for the dominant singleton
  // payload.
  while (sequentialSparseMode_) {
    while (sequentialWideDrainBucket_ < sequentialWideHost_.size() &&
           sequentialWideHost_[sequentialWideDrainBucket_].empty()) {
      ++sequentialWideDrainBucket_;
    }
    if (sequentialWideDrainBucket_ == sequentialWideHost_.size()) {
      VELOX_CHECK_NOT_NULL(sequentialSparseStream_);
      // filtered_join does not own its build table. Complete the last probe
      // before releasing both objects and switching the same operator to the
      // historical sparse-candidate finalize path.
      sequentialSparseStream_->view().synchronize();
      sequentialDuplicateFilter_.reset();
      sequentialDuplicateKeys_.reset();
      sequentialDuplicateKeyTables_.clear();
      sequentialWideHost_.clear();
      sequentialSparseStream_.reset();
      sequentialSparseMode_ = false;
      sequentialUniqueMode_ = false;
      VELOX_CHECK_EQ(
          sequentialSparseCandidateRows_ + sequentialSingletonOutputRows_,
          sequentialUniqueKeyRows_,
          "Sequential TopN sparse classification must preserve every row");
      VELOX_CHECK_EQ(
          sequentialSparseCandidateRows_,
          sequentialProvenDuplicateCandidateRows_,
          "Sequential TopN sparse drain must match the exact narrow-key "
          "proof");
      addRuntimeStat(
          "topNSequentialSparseCandidateRows",
          RuntimeCounter(sequentialSparseCandidateRows_));
      addRuntimeStat(
          "topNSequentialSingletonOutputRows",
          RuntimeCounter(sequentialSingletonOutputRows_));
      LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                   << " completed sparse duplicate drain inputRows="
                   << sequentialUniqueKeyRows_
                   << " duplicateKeyRows=" << sequentialDuplicateKeyRows_
                   << " duplicateCandidateRows="
                   << sequentialSparseCandidateRows_
                   << " singletonOutputRows="
                   << sequentialSingletonOutputRows_
                   << " remainingCandidateHostBytes="
                   << partitionedHostBytes_
                   << " remainingCandidateDeviceBytes="
                   << partitionedDeviceBytes_;
      sequentialUniqueKeyRows_ = 0;
      break;
    }

    auto& wideChunks =
        sequentialWideHost_[sequentialWideDrainBucket_];
    VELOX_CHECK(!wideChunks.empty());
    const auto packedBytes = wideChunks.back().dataBytes;
    const auto workspaceBytes = estimateFinalizeWorkspaceBytes(packedBytes);
    ContinueFuture workspaceFuture;
    auto workspaceAdmission = tryAcquireDeviceMemoryWorkspace(
        customPool(kCudfDeviceMemoryResourceTag),
        this,
        workspaceBytes,
        CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
        DeviceMemoryWorkspacePriority::kDrain,
        &deviceWorkspaceRequest_,
        &workspaceFuture);
    if (!workspaceAdmission.has_value()) {
      VELOX_CHECK(!waitingForDeviceWorkspace_);
      deviceWorkspaceFuture_ = std::move(workspaceFuture);
      waitingForDeviceWorkspace_ = true;
      return {PartitionedOutputStatus::kBlocked, nullptr};
    }

    auto sparseStream = sequentialSparseStream_->view();
    auto sparseMr = get_output_mr();
    auto wideChunk = std::move(wideChunks.back());
    wideChunks.pop_back();
    const auto wideRows = static_cast<uint64_t>(wideChunk.rows);
    VELOX_CHECK_GE(partitionedHostBytes_, packedBytes);
    partitionedHostBytes_ -= packedBytes;
    std::vector<CudfPackedHostRestoreChunk> preparedWide;
    preparedWide.push_back(
        prepareHostChunkForBulkRestore(std::move(wideChunk)));
    const auto splitStart = std::chrono::steady_clock::now();
    auto restoredWide = bulkRestoreCudfPackedHostChunks(
        std::move(preparedWide), sparseStream, sparseMr);
    const auto sparseRestoreStats = restoredWide.stats();
    VELOX_CHECK_EQ(restoredWide.tables().size(), 1);
    const auto wideView = restoredWide.tables().front();
    VELOX_CHECK_NOT_NULL(sequentialDuplicateFilter_);
    const auto keyView = wideView.select(partitionKeys_);
    auto tempMr = get_temp_mr();
    auto duplicateIndices = sequentialDuplicateFilter_->semi_join(
        keyView, sparseStream, tempMr);
    auto singletonIndices = sequentialDuplicateFilter_->anti_join(
        keyView, sparseStream, tempMr);
    const auto duplicateRows =
        static_cast<uint64_t>(duplicateIndices->size());
    const auto singletonRows =
        static_cast<uint64_t>(singletonIndices->size());
    VELOX_CHECK_LE(
        duplicateRows,
        wideRows,
        "Sequential TopN duplicate filter returned too many input rows");
    VELOX_CHECK_EQ(
        duplicateRows + singletonRows,
        wideRows,
        "Sequential TopN combined semi/anti join did not classify every "
        "input row");

    std::unique_ptr<cudf::table> duplicateCandidates;
    if (duplicateRows > 0) {
      auto duplicateIndexColumn = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*duplicateIndices}};
      duplicateCandidates = cudf::gather(
          wideView,
          duplicateIndexColumn,
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          sparseStream,
          sparseMr);
    }
    std::unique_ptr<cudf::table> singletonOutput;
    if (singletonRows > 0) {
      auto singletonSpan =
          cudf::device_span<cudf::size_type const>{*singletonIndices};
      auto singletonIndexColumn = cudf::column_view{singletonSpan};
      singletonOutput = cudf::gather(
          wideView,
          singletonIndexColumn,
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          sparseStream,
          sparseMr);
    }

    if (duplicateCandidates) {
      externalizeRowNumberCandidates(
          std::move(duplicateCandidates), sparseStream);
    }
    sequentialSparseCandidateRows_ += duplicateRows;
    sequentialSingletonOutputRows_ += singletonRows;
    const auto splitNanos =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - splitStart)
            .count();
    addRuntimeStat(
        "topNSequentialSparseSplitNanos",
        RuntimeCounter(splitNanos, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNSequentialSparseRestoredBytes",
        RuntimeCounter(packedBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSequentialSparsePinnedBounceBytes",
        RuntimeCounter(
            sparseRestoreStats.pinnedBounceBytes,
            RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSequentialSparsePageableDirectBytes",
        RuntimeCounter(
            sparseRestoreStats.pageableDirectBytes,
            RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNSequentialSparsePinnedBounceCopies",
        RuntimeCounter(sparseRestoreStats.pinnedBounceCopies));
    addRuntimeStat(
        "topNSequentialSparseHostStageNanos",
        RuntimeCounter(
            sparseRestoreStats.hostStageMicros * 1'000,
            RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNSequentialSparseCopySyncNanos",
        RuntimeCounter(
            sparseRestoreStats.copyStreamSynchronizeMicros * 1'000,
            RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNSequentialSparseInputRows", RuntimeCounter(wideRows));

    if (!singletonOutput) {
      workspaceAdmission.reset();
      continue;
    }
    if (generateRowNumber_) {
      auto one = cudf::numeric_scalar<int64_t>(
          1, true, sparseStream, sparseMr);
      auto rowNumber = cudf::make_column_from_scalar(
          one,
          singletonOutput->num_rows(),
          sparseStream,
          sparseMr);
      auto columns = singletonOutput->release();
      columns.push_back(std::move(rowNumber));
      singletonOutput =
          std::make_unique<cudf::table>(std::move(columns));
    }
    stagePartitionedOutput(
        std::move(singletonOutput), sparseStream, sparseMr);
    workspaceAdmission.reset();
    return {
        PartitionedOutputStatus::kOutput,
        takeNextPartitionedOutputBatch()};
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();

  while (true) {
    // Partition ids carry no output ordering semantics for ROW_NUMBER=1.
    // Restore the smallest ready bucket first so a temporarily constrained
    // device can finish useful work and release external state instead of
    // blocking behind an arbitrary large numeric bucket.
    size_t bucket = partitionedRowNumberHost_.size();
    uint64_t bucketBytes = std::numeric_limits<uint64_t>::max();
    for (size_t candidate = 0;
         candidate < partitionedRowNumberHost_.size();
         ++candidate) {
      const auto candidateBytes = partitionBytes(candidate);
      if (candidateBytes > 0 && candidateBytes < bucketBytes) {
        bucket = candidate;
        bucketBytes = candidateBytes;
      }
    }
    if (bucket == partitionedRowNumberHost_.size()) {
      break;
    }

    // Once global uniqueness is proven, every stored arrival-order chunk is
    // already an exact output run. Restore only one bounded packed chunk at a
    // time instead of concatenating all round-robin chunks in a storage
    // bucket and then copying that larger table back into bounded output
    // slices. Storage still uses the fixed bucket/file fanout, so this avoids
    // creating hundreds of spill files for a large Job-144 input.
    const bool sequentialDirectChunk =
        sequentialUniqueMode_ && sequentialUniqueConfirmed_ &&
        topNSequentialDirectChunkOutputEnabled() &&
        !partitionedRowNumberHost_[bucket].empty();
    const auto finalizeBytes = sequentialDirectChunk
        ? partitionedRowNumberHost_[bucket].back().dataBytes
        : bucketBytes;

    // Restore, concatenate, grouped reduction and output packing overlap in
    // libcudf. Account for that transient working set before starting any of
    // them. The reservation is byte based and process global, so two Velox
    // drivers cannot both admit a multi-GiB bucket against the same physical
    // headroom snapshot. A rejected driver blocks through normal Velox
    // isBlocked() instead of shrinking the input batch or holding a GPU lock.
    const auto workspaceBytes = estimateFinalizeWorkspaceBytes(
        std::min<uint64_t>(finalizeBytes, finalizeInputBytes_));
    ContinueFuture workspaceFuture;
    auto workspaceAdmission = tryAcquireDeviceMemoryWorkspace(
        customPool(kCudfDeviceMemoryResourceTag),
        this,
        workspaceBytes,
        CudfConfig::getInstance().deviceMemoryMinHeadroomBytes,
        DeviceMemoryWorkspacePriority::kDrain,
        &deviceWorkspaceRequest_,
        &workspaceFuture);
    if (!workspaceAdmission.has_value()) {
      VELOX_CHECK(!waitingForDeviceWorkspace_);
      deviceWorkspaceFuture_ = std::move(workspaceFuture);
      waitingForDeviceWorkspace_ = true;
      // A temporary arbitration miss is not end-of-stream. Returning an
      // unqualified nullptr used to make doGetOutput() mark the operator
      // finished and clear every unprocessed bucket. Under Job 144 pressure,
      // one replica consequently emitted four of 64 buckets and silently
      // dropped the remaining rows. Preserve the distinction explicitly so
      // the driver observes isBlocked(), waits, and retries this bucket.
      return {PartitionedOutputStatus::kBlocked, nullptr};
    }
    ++nextHostPartition_;
    if (!sequentialDirectChunk && bucketBytes > finalizeInputBytes_) {
      // Recursive repartition currently consumes host chunks. Preserve the
      // bounded path for a genuinely oversized/skewed bucket, but do not
      // externalize resident device buckets that already fit finalize.
      if (bucket < partitionedRowNumberDevice_.size() &&
          !partitionedRowNumberDevice_[bucket].empty()) {
        spillDevicePartition(bucket, stream);
      }
      repartitionOversizedHostBucket(bucket, bucketBytes, stream, mr);
      continue;
    }
    auto& chunks = partitionedRowNumberHost_[bucket];
    auto& spillFile = partitionedRowNumberSpillFiles_[bucket];
    const auto bucketStart = std::chrono::steady_clock::now();
    std::vector<CudfPackedHostRestoreChunk> hostChunks;
    hostChunks.reserve(sequentialDirectChunk ? 1 : chunks.size());
    uint64_t restoredRows = 0;
    uint64_t restoredBytes = 0;
    const auto prepareChunk = [&](HostPackedChunk chunk) {
      const auto hostBytes = chunk.dataBytes;
      restoredRows += chunk.rows;
      hostChunks.push_back(
          prepareHostChunkForBulkRestore(std::move(chunk)));
      VELOX_CHECK_GE(partitionedHostBytes_, hostBytes);
      partitionedHostBytes_ -= hostBytes;
      restoredBytes += hostBytes;
    };
    if (sequentialDirectChunk) {
      auto chunk = std::move(chunks.back());
      chunks.pop_back();
      prepareChunk(std::move(chunk));
    } else {
      for (auto& chunk : chunks) {
        prepareChunk(std::move(chunk));
      }
      chunks.clear();
    }
    auto restored = bulkRestoreCudfPackedHostChunks(
        std::move(hostChunks),
        stream,
        mr,
        CudfBulkPackedRestoreOptions{
            .pinnedHostThreads = topNDenseRestoreHostThreads()});
    const auto restoreStats = restored.stats();
    const auto restoreEnd = std::chrono::steady_clock::now();
    spillFile.reset();

    auto& deviceChunks = partitionedRowNumberDevice_[bucket];
    VELOX_CHECK(
        !sequentialDirectChunk || deviceChunks.empty(),
        "Sequential unique output cannot mix arrival-order host chunks with "
        "hash-partitioned device chunks");
    std::vector<cudf::table_view> inputTables = restored.tables();
    for (auto& deviceChunk : deviceChunks) {
      handoffDevicePartitionChunk(deviceChunk, stream);
      restoredRows += deviceChunk.rows;
      restoredBytes += deviceChunk.dataBytes;
      inputTables.push_back(deviceChunk.packed->table);
    }

    if (inputTables.empty()) {
      continue;
    }

    // A bucket is bounded by total candidate bytes / hostPartitionCount
    // (roughly 275-300 MiB for Job 144).  Restore the bounded bucket and
    // concatenate once, then perform one grouped reduction.  The former
    // chunk-at-a-time accumulator repeatedly copied and regrouped an
    // ever-growing, mostly-distinct table, making high-cardinality Top1
    // finalize approximately quadratic in the number of host chunks.
    const auto inputChunks = inputTables.size();
    auto merged = cudf::concatenate(inputTables, stream, mr);
    const auto residentDeviceBytes = partitionedRowNumberDeviceBytes_[bucket];
    const auto concatenateEnd = std::chrono::steady_clock::now();
    auto accumulator =
        sequentialUniqueMode_ && sequentialUniqueConfirmed_
        ? std::move(merged)
        : reduceOwnedCandidates(std::move(merged), stream, mr);
    const auto reduceEnd = std::chrono::steady_clock::now();
    if (sequentialUniqueMode_ && sequentialUniqueConfirmed_) {
      addRuntimeStat(
          "topNSequentialUniqueOutputRows", RuntimeCounter(restoredRows));
      addRuntimeStat("topNSequentialUniqueOutputBatches", RuntimeCounter(1));
      if (sequentialDirectChunk) {
        addRuntimeStat(
            "topNSequentialDirectChunkBytes",
            RuntimeCounter(restoredBytes, RuntimeCounter::Unit::kBytes));
        addRuntimeStat(
            "topNSequentialDirectChunkRows", RuntimeCounter(restoredRows));
        addRuntimeStat(
            "topNSequentialDirectChunkBatches", RuntimeCounter(1));
      }
    }
    const auto restoreMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            restoreEnd - bucketStart)
            .count();
    const auto concatenateMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            concatenateEnd - restoreEnd)
            .count();
    const auto reduceMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            reduceEnd - concatenateEnd)
            .count();
    LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                 << " finalized host bucket=" << bucket
                 << " inputChunks=" << inputChunks
                 << " inputRows=" << restoredRows
                 << " inputBytes=" << restoredBytes
                 << " outputRows=" << accumulator->num_rows()
                 << " pinnedBounceBytes="
                 << restoreStats.pinnedBounceBytes
                 << " pageableDirectBytes="
                 << restoreStats.pageableDirectBytes
                 << " pinnedBounceCopies="
                 << restoreStats.pinnedBounceCopies
                 << " bounceHostStageUs="
                 << restoreStats.hostStageMicros
                 << " bounceReuseWaitUs="
                 << restoreStats.bounceReuseWaitMicros
                 << " copyStreamSynchronizeUs="
                 << restoreStats.copyStreamSynchronizeMicros
                 << " parallelHostStageGroups="
                 << restoreStats.parallelHostStageGroups
                 << " parallelHostStageChunks="
                 << restoreStats.parallelHostStageChunks
                 << " pinnedHostThreadLimit="
                 << restoreStats.pinnedHostThreadLimit
                 << " restoreUs=" << restoreMicros
                 << " concatenateUs=" << concatenateMicros
                 << " reduceUs=" << reduceMicros;
    addRuntimeStat(
        "topNFinalizeInputBytes",
        RuntimeCounter(restoredBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNFinalizeRestoreNanos",
        RuntimeCounter(restoreMicros * 1'000, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNFinalizeConcatenateNanos",
        RuntimeCounter(
            concatenateMicros * 1'000, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNFinalizeReduceNanos",
        RuntimeCounter(reduceMicros * 1'000, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNPinnedBounceBytes",
        RuntimeCounter(
            restoreStats.pinnedBounceBytes,
            RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNPageableDirectRestoreBytes",
        RuntimeCounter(
            restoreStats.pageableDirectBytes,
            RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "topNPinnedBounceCopies",
        RuntimeCounter(restoreStats.pinnedBounceCopies));
    addRuntimeStat(
        "topNPinnedBounceHostStageNanos",
        RuntimeCounter(
            restoreStats.hostStageMicros * 1'000,
            RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNPinnedBounceReuseWaitNanos",
        RuntimeCounter(
            restoreStats.bounceReuseWaitMicros * 1'000,
            RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNPinnedBounceCopySyncNanos",
        RuntimeCounter(
            restoreStats.copyStreamSynchronizeMicros * 1'000,
            RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "topNParallelHostStageGroups",
        RuntimeCounter(restoreStats.parallelHostStageGroups));
    addRuntimeStat(
        "topNParallelHostStageChunks",
        RuntimeCounter(restoreStats.parallelHostStageChunks));
    addRuntimeStat(
        "topNPinnedHostThreadLimit",
        RuntimeCounter(restoreStats.pinnedHostThreadLimit));

    if (generateRowNumber_) {
      auto one = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
      auto rowNumber = cudf::make_column_from_scalar(
          one, accumulator->num_rows(), stream, mr);
      auto columns = accumulator->release();
      columns.push_back(std::move(rowNumber));
      accumulator = std::make_unique<cudf::table>(std::move(columns));
    }

    // The shared device-memory resource is not a sufficient lifetime fence:
    // concatenate, grouped reduction, and row-number generation are all
    // asynchronous, while clearing the packed inputs and merged table can
    // make their allocations immediately reclaimable by another native
    // driver. Under Job 144's concurrent Final TopN drain this occasionally
    // let another stream reuse an input allocation before the consumer kernel
    // completed, surfacing later as CUDA launch failure 719 in an unrelated
    // operator. Record completion instead of synchronizing the driver for
    // every bucket. stagePartitionedOutput() separately establishes the
    // producer -> output-stream dependency; this deferred owner protects only
    // the inputs read by the producer kernels.
    cudaEvent_t finalizeComplete{nullptr};
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(
        &finalizeComplete, cudaEventDisableTiming));
    try {
      CUDF_CUDA_TRY(cudaEventRecord(finalizeComplete, stream.value()));
      deferredFinalizeOwnerships_.emplace_back(
          std::move(restored),
          std::move(deviceChunks),
          std::move(merged),
          finalizeComplete);
    } catch (...) {
      cudaEventDestroy(finalizeComplete);
      throw;
    }
    VELOX_CHECK_GE(partitionedDeviceBytes_, residentDeviceBytes);
    partitionedDeviceBytes_ -= residentDeviceBytes;
    partitionedRowNumberDeviceBytes_[bucket] = 0;

    stagePartitionedOutput(std::move(accumulator), stream, mr);
    // Restore/concatenate/reduce allocations have now been made and are
    // visible through cudaMemGetInfo. Release the transient construction
    // lease before returning output to a potentially blocked downstream
    // pipeline. Holding the multi-GiB estimate until every small output slice
    // was consumed double-counted the resident bucket and starved unrelated
    // restores on the same executor.
    workspaceAdmission.reset();
    return {
        PartitionedOutputStatus::kOutput,
        takeNextPartitionedOutputBatch()};
  }
  return {PartitionedOutputStatus::kFinished, nullptr};
}

uint64_t CudfTopNRowNumber::estimateFinalizeWorkspaceBytes(
    uint64_t bucketBytes) const {
  constexpr uint64_t kFixedWorkspaceBytes = 512ULL << 20;
  constexpr uint64_t kWorkspaceCopies = 3;
  if (bucketBytes >
      (std::numeric_limits<uint64_t>::max() - kFixedWorkspaceBytes) /
          kWorkspaceCopies) {
    return std::numeric_limits<uint64_t>::max();
  }
  return std::max<uint64_t>(
      1ULL << 30,
      bucketBytes * kWorkspaceCopies + kFixedWorkspaceBytes);
}

void CudfTopNRowNumber::stagePartitionedOutput(
    std::unique_ptr<cudf::table> output,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(pendingPartitionedOutputChunks_.empty());
  VELOX_CHECK_NULL(pendingPartitionedDeviceOutput_);
  const auto totalRows = output->num_rows();
  VELOX_CHECK_GT(totalRows, 0);
  auto outputVector = std::make_shared<CudfVector>(
      pool(), outputType_, totalRows, std::move(output), stream);
  const auto outputBytes = outputVector->estimateFlatSize();
  const auto bytesPerRow = std::max<uint64_t>(
      1, (outputBytes + totalRows - 1) / totalRows);
  const auto byteBoundRows = std::max<uint64_t>(
      1, outputChunkBytes_ / bytesPerRow);
  const auto batchRows = static_cast<cudf::size_type>(
      std::min<uint64_t>(maxOutputRows_, byteBoundRows));

  if (deviceOutputStagingEnabled_) {
    if (!deviceOutputStream_) {
      deviceOutputStream_ = std::make_shared<rmm::cuda_stream>(
          rmm::cuda_stream::flags::non_blocking);
      CudfVector::registerStreamOwner(deviceOutputStream_);
    }
    const auto outputStream = deviceOutputStream_->view();
    // The finalized bucket was produced on 'stream'. Establish an explicit
    // asynchronous producer -> output handoff without blocking the driver.
    cudaEvent_t producerReady{nullptr};
    CUDF_CUDA_TRY(
        cudaEventCreateWithFlags(&producerReady, cudaEventDisableTiming));
    try {
      CUDF_CUDA_TRY(cudaEventRecord(producerReady, stream.value()));
      CUDF_CUDA_TRY(
          cudaStreamWaitEvent(outputStream.value(), producerReady, 0));
    } catch (...) {
      cudaEventDestroy(producerReady);
      throw;
    }
    CUDF_CUDA_TRY(cudaEventDestroy(producerReady));
    // The event orders the already-submitted producer work before future
    // output-stream reads, but it does not change the stream used by RMM when
    // the table's buffers are eventually released.  Keep both sides of the
    // ownership handoff on the output stream.  Otherwise a downstream stall
    // can destroy a completed output on its old producer stream while a slice
    // copy submitted to outputStream is still reading the same allocation.
    VELOX_CHECK(outputVector->rebindStream(outputStream));
    pendingPartitionedDeviceOutput_ = outputVector->release();
    pendingPartitionedDeviceOutputOffset_ = 0;
    pendingPartitionedDeviceOutputBytes_ = outputBytes;
    pendingPartitionedDeviceOutputStream_ = outputStream;
    LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
                 << " retained finalized output on device rows=" << totalRows
                 << " estimatedBytes=" << outputBytes
                 << " maxChunkRows=" << batchRows;
    return;
  }
  std::vector<cudf::size_type> splitOffsets;
  for (cudf::size_type offset = batchRows; offset < totalRows;
       offset += batchRows) {
    splitOffsets.push_back(offset);
  }

  // release() is zero-copy for a CudfVector constructed from cudf::table.
  // contiguous_split packs all bounded output slices in one libcudf call.
  auto outputTable = outputVector->release();
  outputVector.reset();
  auto packedOutputs =
      cudf::contiguous_split(outputTable->view(), splitOffsets, stream, mr);
  outputTable.reset();

  uint64_t packedBytes = 0;
  for (const auto& packed : packedOutputs) {
    packedBytes += packed.data.gpu_data->size();
  }
  auto pinnedStorage = acquireCudfPackedPinnedBuffer(packedBytes);
  const bool usedPinnedStaging = pinnedStorage != nullptr;
  if (!pinnedStorage) {
    pinnedStorage = std::shared_ptr<uint8_t>(
        new uint8_t[packedBytes], std::default_delete<uint8_t[]>());
  }
  uint64_t hostOffset = 0;
  for (auto& packed : packedOutputs) {
    HostPackedChunk chunk;
    chunk.dataBytes = packed.data.gpu_data->size();
    chunk.storedBytes = chunk.dataBytes;
    chunk.rows = packed.table.num_rows();
    chunk.metadata = std::move(packed.data.metadata);
    chunk.pinned = usedPinnedStaging;
    if (chunk.dataBytes > 0) {
      chunk.data = std::shared_ptr<uint8_t>(
          pinnedStorage, pinnedStorage.get() + hostOffset);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          chunk.data.get(),
          packed.data.gpu_data->data(),
          chunk.dataBytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    hostOffset += chunk.dataBytes;
    pendingPartitionedOutputChunks_.push_back(std::move(chunk));
  }
  stream.synchronize();
  packedOutputs.clear();

  // Pinned memory is scarce and shared with active partition/spill traffic.
  // Demote the complete finalized bucket with one contiguous host copy, while
  // keeping each chunk as an alias into the shared pageable allocation.
  if (usedPinnedStaging && packedBytes > 0) {
    auto pageableStorage = std::shared_ptr<uint8_t>(
        new uint8_t[packedBytes], std::default_delete<uint8_t[]>());
    std::memcpy(pageableStorage.get(), pinnedStorage.get(), packedBytes);
    uint64_t pageableOffset = 0;
    for (auto& chunk : pendingPartitionedOutputChunks_) {
      chunk.data = std::shared_ptr<uint8_t>(
          pageableStorage, pageableStorage.get() + pageableOffset);
      chunk.pinned = false;
      pageableOffset += chunk.dataBytes;
    }
  }
  LOG(WARNING) << "CudfTopNRowNumber node=" << diagnosticNodeId_
               << " staged finalized output rows=" << totalRows
               << " estimatedBytes=" << outputBytes
               << " packedBytes=" << packedBytes
               << " outputChunks=" << pendingPartitionedOutputChunks_.size()
               << " maxChunkRows=" << batchRows;
}

CudfVectorPtr CudfTopNRowNumber::takeNextPartitionedOutputBatch() {
  reclaimCompletedDeviceOutputs();
  if (pendingPartitionedDeviceOutput_) {
    const auto totalRows = pendingPartitionedDeviceOutput_->num_rows();
    VELOX_CHECK_LT(pendingPartitionedDeviceOutputOffset_, totalRows);
    const auto bytesPerRow = std::max<uint64_t>(
        1, (pendingPartitionedDeviceOutputBytes_ + totalRows - 1) /
            totalRows);
    const auto remainingRows =
        totalRows - pendingPartitionedDeviceOutputOffset_;
    const bool wholeBucketHandoff =
        pendingPartitionedDeviceOutputOffset_ == 0 &&
        pendingPartitionedDeviceOutputBytes_ <= outputChunkBytes_ &&
        totalRows <= maxOutputRows_;
    const auto byteBoundRows = std::max<uint64_t>(
        1, outputChunkBytes_ / bytesPerRow);
    const auto rows = static_cast<cudf::size_type>(wholeBucketHandoff
            ? totalRows
            : std::min<uint64_t>(
                  remainingRows,
                  std::min<uint64_t>(maxOutputRows_, byteBoundRows)));
    // The finalized table was produced asynchronously on this stream. Keep
    // every slice copy and the final release on that same stream; switching
    // to an arbitrary pool stream can read or free the bucket before its
    // producer kernels complete.
    auto stream = pendingPartitionedDeviceOutputStream_;

    if (pendingPartitionedDeviceOutputOffset_ == 0 && rows == totalRows) {
      auto output = std::make_shared<CudfVector>(
          pool(),
          outputType_,
          totalRows,
          std::move(pendingPartitionedDeviceOutput_),
          stream);
      pendingPartitionedDeviceOutputOffset_ = 0;
      pendingPartitionedDeviceOutputBytes_ = 0;
      pendingPartitionedDeviceOutputStream_ = rmm::cuda_stream_view{};
      return output;
    }

    auto chunk = copyTableSlice(
        pendingPartitionedDeviceOutput_->view(),
        pendingPartitionedDeviceOutputOffset_,
        pendingPartitionedDeviceOutputOffset_ + rows,
        stream,
        get_output_mr());
    pendingPartitionedDeviceOutputOffset_ += rows;
    if (pendingPartitionedDeviceOutputOffset_ == totalRows) {
      // copyTableSlice is asynchronous. Keep the source bucket alive behind a
      // CUDA event rather than synchronizing here: a per-bucket synchronize
      // turns the output pipeline into a serial restore/copy/consume loop.
      cudaEvent_t completionEvent{nullptr};
      CUDF_CUDA_TRY(cudaEventCreateWithFlags(
          &completionEvent, cudaEventDisableTiming));
      try {
        CUDF_CUDA_TRY(cudaEventRecord(completionEvent, stream.value()));
        deferredDeviceOutputs_.emplace_back(
            std::move(pendingPartitionedDeviceOutput_),
            completionEvent);
      } catch (...) {
        cudaEventDestroy(completionEvent);
        throw;
      }
      pendingPartitionedDeviceOutputOffset_ = 0;
      pendingPartitionedDeviceOutputBytes_ = 0;
      pendingPartitionedDeviceOutputStream_ = rmm::cuda_stream_view{};
    }
    return std::make_shared<CudfVector>(
        pool(), outputType_, rows, std::move(chunk), stream);
  }

  VELOX_CHECK(!pendingPartitionedOutputChunks_.empty());
  auto chunk = std::move(pendingPartitionedOutputChunks_.front());
  pendingPartitionedOutputChunks_.pop_front();
  const auto rows = chunk.rows;
  auto stream = cudfGlobalStreamPool().get_stream();
  auto packed =
      restoreHostChunk(std::move(chunk), stream, get_output_mr());
  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      rows,
      std::move(packed),
      stream);
}

void CudfTopNRowNumber::reclaimCompletedDeviceOutputs() {
  for (auto it = deferredDeviceOutputs_.begin();
       it != deferredDeviceOutputs_.end();) {
    const auto status = cudaEventQuery(it->event);
    if (status == cudaSuccess) {
      it = deferredDeviceOutputs_.erase(it);
      continue;
    }
    if (status != cudaErrorNotReady) {
      CUDF_CUDA_TRY(status);
    }
    ++it;
  }
}

void CudfTopNRowNumber::reclaimCompletedFinalizeOwnerships() {
  for (auto it = deferredFinalizeOwnerships_.begin();
       it != deferredFinalizeOwnerships_.end();) {
    const auto status = cudaEventQuery(it->event);
    if (status == cudaSuccess) {
      it = deferredFinalizeOwnerships_.erase(it);
      continue;
    }
    if (status != cudaErrorNotReady) {
      CUDF_CUDA_TRY(status);
    }
    ++it;
  }
}

void CudfTopNRowNumber::clearPartitionedRowNumberState() {
  pendingPartitionedOutputChunks_.clear();
  pendingPartitionedDeviceOutput_.reset();
  pendingPartitionedDeviceOutputOffset_ = 0;
  pendingPartitionedDeviceOutputBytes_ = 0;
  pendingPartitionedDeviceOutputStream_ = rmm::cuda_stream_view{};
  // DeferredDeviceOutput synchronizes its event before releasing the source
  // table. This only blocks during close/error cleanup; the normal output path
  // uses non-blocking cudaEventQuery above.
  deferredDeviceOutputs_.clear();
  // The same close/error rule applies to inputs retained for asynchronous
  // finalization. Normal output polling removes only completed owners.
  deferredFinalizeOwnerships_.clear();
  deviceOutputStream_.reset();
  partitionedRowNumberHost_.clear();
  partitionedRowNumberDevice_.clear();
  partitionedRowNumberDeviceBytes_.clear();
  partitionedRowNumberSpillFiles_.clear();
  partitionedRowNumberHashDepths_.clear();
  sequentialUniqueKeyPartitions_.clear();
  sequentialWideHost_.clear();
  sequentialDensePrefixHost_.clear();
  sequentialDuplicateFilter_.reset();
  sequentialDuplicateKeys_.reset();
  sequentialDuplicateKeyTables_.clear();
  sequentialSparseStream_.reset();
  sequentialUniqueKeyBytes_ = 0;
  sequentialUniqueKeyRows_ = 0;
  sequentialUniqueNextBucket_ = 0;
  sequentialUniqueProbeBucket_ = 0;
  sequentialUniqueProbedRows_ = 0;
  sequentialUniqueProbedDistinctRows_ = 0;
  sequentialUniqueProbeNanos_ = 0;
  sequentialUniqueProbePeakWorkspaceBytes_ = 0;
  sequentialDuplicateKeyRows_ = 0;
  sequentialProvenDuplicateCandidateRows_ = 0;
  sequentialSparseCandidateRows_ = 0;
  sequentialSingletonOutputRows_ = 0;
  sequentialWideDrainBucket_ = 0;
  sequentialEarlyDenseInputBatches_ = 0;
  sequentialEarlyDenseProbeRows_ = 0;
  sequentialEarlyDenseDistinctRows_ = 0;
  sequentialDensePrefixRows_ = 0;
  sequentialDensePrefixBytes_ = 0;
  sequentialDenseConvertedRows_ = 0;
  sequentialDenseConvertedBytes_ = 0;
  sequentialUniqueMode_ = false;
  sequentialUniqueProbeComplete_ = false;
  sequentialUniqueConfirmed_ = false;
  sequentialSparseMode_ = false;
  sequentialDenseInputMode_ = false;
  sequentialDenseRawRewrite_ = false;
  nextHostPartition_ = 0;
  partitionedHostBytes_ = 0;
  partitionedResidentHostBytes_ = 0;
  partitionedDiskBytes_ = 0;
  partitionedDiskUncompressedBytes_ = 0;
  partitionedDeviceBytes_ = 0;
  partitionedRowNumberMode_ = false;
  if (!spilled_ && !spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    if (!error) {
      spillDirectory_.clear();
    }
  }
  ::malloc_trim(0);
}

void CudfTopNRowNumber::ensureSpillDirectory() {
  if (!spillDirectory_.empty()) {
    return;
  }
  namespace fs = std::filesystem;
  const auto& taskSpillRoot =
      operatorCtx_->task()->getOrCreateSpillDirectory();
  VELOX_CHECK(
      !taskSpillRoot.empty(),
      "CudfTopNRowNumber requires an explicit Task spill directory");
  const auto sequence = spillDirectorySequence.fetch_add(1);
  spillDirectory_ =
      (fs::path(taskSpillRoot) /
       fmt::format(
           "velox-cudf-topn-spill-{}-{}",
           static_cast<int64_t>(::getpid()),
           sequence))
          .string();
  fs::create_directories(spillDirectory_);
}

void CudfTopNRowNumber::spillSortedRun() {
  if (inputs_.empty()) {
    return;
  }

  if (!spilled_) {
    ensureSpillDirectory();
    spilled_ = true;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfTopNRowNumber node={} state=sortRun.concatenate.begin "
          "bufferedBytes={} bufferedInputs={}",
          diagnosticNodeId_,
          bufferedBytes_,
          inputs_.size()));
  auto input =
      getConcatenatedTable(std::exchange(inputs_, {}), inputType_, stream, mr);
  bufferedBytes_ = 0;

  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfTopNRowNumber node={} state=sortRun.sort.begin rows={}",
          diagnosticNodeId_,
          input->num_rows()));
  auto sorted = cudf::sort_by_key(
      input->view(),
      input->view().select(allKeyIndices_),
      columnOrders_,
      nullOrders_,
      stream,
      mr);
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfTopNRowNumber node={} state=sortRun.sort.end rows={}",
          diagnosticNodeId_,
          input->num_rows()));

  auto path = fmt::format(
      "{}/run-{:06}.parquet", spillDirectory_, spillFileSequence_++);
  auto options = cudf::io::parquet_writer_options::builder(
                     cudf::io::sink_info{path}, sorted->view())
                     .build();
  cudf::io::write_parquet(options, stream);
  sortedRuns_.push_back({std::move(path), nullptr});
  ::malloc_trim(0);
}

void CudfTopNRowNumber::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  for (auto& run : sortedRuns_) {
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{run.path})
                       .build();
    run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        kMergeChunkBytes, 0, options, stream, mr);
  }
  readersInitialized_ = true;
}

void CudfTopNRowNumber::compactSortedRunsForMerge() {
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();

  while (sortedRuns_.size() > kMergeFanIn) {
    std::vector<SortedRun> nextLevel;
    nextLevel.reserve((sortedRuns_.size() + kMergeFanIn - 1) / kMergeFanIn);

    for (size_t begin = 0; begin < sortedRuns_.size(); begin += kMergeFanIn) {
      const auto end = std::min(sortedRuns_.size(), begin + kMergeFanIn);
      if (end - begin == 1) {
        nextLevel.push_back(std::move(sortedRuns_[begin]));
        continue;
      }

      std::vector<std::unique_ptr<cudf::io::chunked_parquet_reader>> readers;
      readers.reserve(end - begin);
      for (size_t index = begin; index < end; ++index) {
        auto options = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info{sortedRuns_[index].path})
                           .build();
        readers.push_back(
            std::make_unique<cudf::io::chunked_parquet_reader>(
                kMergeChunkBytes, 0, options, stream, mr));
      }

      const auto outputPath = fmt::format(
          "{}/merge-{:06}.parquet", spillDirectory_, spillFileSequence_++);
      auto writerOptions = cudf::io::chunked_parquet_writer_options::builder(
                               cudf::io::sink_info{outputPath})
                               .build();
      cudf::io::chunked_parquet_writer writer(writerOptions, stream);
      std::unique_ptr<cudf::table> carry;

      while (true) {
        std::vector<std::unique_ptr<cudf::table>> chunks;
        std::vector<cudf::table_view> mergeViews;
        std::vector<cudf::table_view> boundaryRows;
        if (carry && carry->num_rows() > 0) {
          mergeViews.push_back(carry->view());
        }
        for (auto& reader : readers) {
          if (!reader->has_next()) {
            continue;
          }
          auto chunk = reader->read_chunk();
          if (chunk.tbl->num_rows() == 0) {
            continue;
          }
          chunks.push_back(std::move(chunk.tbl));
          mergeViews.push_back(chunks.back()->view());
          if (reader->has_next()) {
            auto last = cudf::slice(
                chunks.back()->view(),
                {chunks.back()->num_rows() - 1, chunks.back()->num_rows()},
                stream);
            boundaryRows.push_back(last.front());
          }
        }

        if (mergeViews.empty()) {
          break;
        }
        std::unique_ptr<cudf::table> merged = mergeViews.size() == 1
            ? std::make_unique<cudf::table>(mergeViews.front(), stream, mr)
            : cudf::merge(
                  mergeViews,
                  allKeyIndices_,
                  columnOrders_,
                  nullOrders_,
                  stream,
                  mr);
        carry.reset();
        if (boundaryRows.empty()) {
          writer.write(merged->view());
          break;
        }

        auto boundaryCandidates = cudf::concatenate(boundaryRows, stream, mr);
        auto sortedBoundaries = cudf::sort_by_key(
            boundaryCandidates->view(),
            boundaryCandidates->view().select(allKeyIndices_),
            columnOrders_,
            nullOrders_,
            stream,
            mr);
        auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);
        auto positions = cudf::upper_bound(
            merged->view().select(allKeyIndices_),
            boundary.front().select(allKeyIndices_),
            columnOrders_,
            nullOrders_,
            stream,
            mr);
        const auto safeEnd = firstSearchPosition(positions->view(), stream);
        carry = copyTableSlice(
            merged->view(), safeEnd, merged->num_rows(), stream, mr);
        if (safeEnd > 0) {
          auto safe = cudf::slice(merged->view(), {0, safeEnd}, stream);
          writer.write(safe.front());
        }
      }
      writer.close();

      for (size_t index = begin; index < end; ++index) {
        std::error_code error;
        std::filesystem::remove(sortedRuns_[index].path, error);
      }
      nextLevel.push_back({outputPath, nullptr});
    }
    sortedRuns_ = std::move(nextLevel);
  }
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::mergeNextSortedBatch(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finalBatch) {
  // Once all readers are exhausted, a subsequent call may exist solely to
  // drain partitionCarry_. Mark it final so the last partition is emitted
  // instead of being retained forever as a possibly-incomplete peer group.
  finalBatch = mergeFinished_;
  while (!mergeFinished_) {
    std::vector<std::unique_ptr<cudf::table>> chunks;
    std::vector<cudf::table_view> mergeViews;
    std::vector<cudf::table_view> boundaryRows;
    if (mergeCarry_ && mergeCarry_->num_rows() > 0) {
      mergeViews.push_back(mergeCarry_->view());
    }
    for (auto& run : sortedRuns_) {
      if (!run.reader || !run.reader->has_next()) {
        continue;
      }
      auto chunk = run.reader->read_chunk();
      if (chunk.tbl->num_rows() == 0) {
        continue;
      }
      chunks.push_back(std::move(chunk.tbl));
      mergeViews.push_back(chunks.back()->view());
      if (run.reader->has_next()) {
        auto last = cudf::slice(
            chunks.back()->view(),
            {chunks.back()->num_rows() - 1, chunks.back()->num_rows()},
            stream);
        boundaryRows.push_back(last.front());
      }
    }
    if (mergeViews.empty()) {
      mergeFinished_ = true;
      finalBatch = true;
      return std::exchange(mergeCarry_, nullptr);
    }

    std::unique_ptr<cudf::table> merged;
    if (mergeViews.size() == 1) {
      merged = std::make_unique<cudf::table>(mergeViews.front(), stream, mr);
    } else {
      merged = cudf::merge(
          mergeViews, allKeyIndices_, columnOrders_, nullOrders_, stream, mr);
    }
    mergeCarry_.reset();
    if (boundaryRows.empty()) {
      mergeFinished_ = true;
      finalBatch = true;
      return merged;
    }

    auto boundaryCandidates = cudf::concatenate(boundaryRows, stream, mr);
    auto sortedBoundaries = cudf::sort_by_key(
        boundaryCandidates->view(),
        boundaryCandidates->view().select(allKeyIndices_),
        columnOrders_,
        nullOrders_,
        stream,
        mr);
    auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);
    // Future rows are >= each run's current tail. Rows equal to the minimum
    // tail are therefore safe to emit as well (the sort is not required to be
    // stable across runs). Keeping them in carry makes low-cardinality keys
    // grow without bound.
    auto positions = cudf::upper_bound(
        merged->view().select(allKeyIndices_),
        boundary.front().select(allKeyIndices_),
        columnOrders_,
        nullOrders_,
        stream,
        mr);
    const auto safeEnd = firstSearchPosition(positions->view(), stream);
    mergeCarry_ =
        copyTableSlice(merged->view(), safeEnd, merged->num_rows(), stream, mr);
    if (safeEnd > 0) {
      return copyTableSlice(merged->view(), 0, safeEnd, stream, mr);
    }
  }
  return nullptr;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::takeCompletePartitions(
    std::unique_ptr<cudf::table> sorted,
    bool finalBatch,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (partitionCarry_ && partitionCarry_->num_rows() > 0) {
    if (sorted && sorted->num_rows() > 0) {
      std::vector<cudf::table_view> pieces{
          partitionCarry_->view(), sorted->view()};
      sorted = cudf::concatenate(pieces, stream, mr);
      partitionCarry_.reset();
    } else {
      sorted = std::exchange(partitionCarry_, nullptr);
    }
  }
  if (!sorted || sorted->num_rows() == 0) {
    return nullptr;
  }

  auto partitionColumns = sorted->view().select(partitionKeys_);
  std::vector<cudf::order> orders(
      partitionKeys_.size(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> nullOrders(
      partitionKeys_.size(), cudf::null_order::BEFORE);
  cudf::size_type completeEnd = sorted->num_rows();
  if (!finalBatch) {
    auto lastPartition = cudf::slice(
        partitionColumns, {sorted->num_rows() - 1, sorted->num_rows()}, stream);
    auto positions = cudf::lower_bound(
        partitionColumns,
        lastPartition.front(),
        orders,
        nullOrders,
        stream,
        mr);
    completeEnd = firstSearchPosition(positions->view(), stream);
  }

  // Keep each downstream Top-N reduction/gather bounded. Select a partition
  // boundary at or before the row target so a rank peer group is never split.
  cudf::size_type emitEnd = completeEnd;
  if (completeEnd > maxOutputRows_) {
    auto boundaryPartition = cudf::slice(
        partitionColumns,
        {maxOutputRows_, maxOutputRows_ + 1},
        stream);
    auto positions = cudf::lower_bound(
        partitionColumns,
        boundaryPartition.front(),
        orders,
        nullOrders,
        stream,
        mr);
    const auto boundary = firstSearchPosition(positions->view(), stream);
    // A single giant peer group must remain intact for rank semantics.
    emitEnd = boundary > 0 ? boundary : completeEnd;
  }

  partitionCarry_ =
      copyTableSlice(sorted->view(), emitEnd, sorted->num_rows(), stream, mr);
  if (emitEnd == 0) {
    return nullptr;
  }
  return copyTableSlice(sorted->view(), 0, emitEnd, stream, mr);
}

CudfVectorPtr CudfTopNRowNumber::computeNextSortedOutput() {
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  while (!mergeFinished_ || mergeCarry_ || partitionCarry_) {
    bool finalBatch = false;
    auto sorted = mergeNextSortedBatch(stream, mr, finalBatch);
    sorted = takeCompletePartitions(std::move(sorted), finalBatch, stream, mr);
    if (!sorted || sorted->num_rows() == 0) {
      if (finalBatch) {
        return nullptr;
      }
      continue;
    }
    // The merge output is already globally ordered. The existing limit-one
    // helpers are reused initially for exact rank/tie semantics; they operate
    // on a bounded complete-partition batch.
    return rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
        ? computeLimitOneRowNumber(sorted->view(), stream, mr)
        : computeLimitOneRankLike(sorted->view(), stream, mr);
  }
  return nullptr;
}

void CudfTopNRowNumber::cleanupSpillFiles() {
  sortedRuns_.clear();
  mergeCarry_.reset();
  partitionCarry_.reset();
  if (spillDirectory_.empty()) {
    return;
  }
  std::error_code error;
  std::filesystem::remove_all(spillDirectory_, error);
  if (error) {
    LOG(ERROR) << "Failed to remove CudfTopNRowNumber spill directory '"
               << spillDirectory_ << "': " << error.message();
  } else {
    spillDirectory_.clear();
  }
  ::malloc_trim(0);
}

void CudfTopNRowNumber::doClose() {
  deviceWorkspaceRequest_.reset();
  inputWorkspaceAdmission_.reset();
  pendingInput_.reset();
  inputs_.clear();
  candidates_.reset();
  passthroughOutputs_.clear();
  clearPartitionedRowNumberState();
  cleanupSpillFiles();
  Operator::close();
}

CudfVectorPtr CudfTopNRowNumber::computeLimitOneRowNumber(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::lock_guard<std::mutex> cucoLock(cudfCucoMutex());
  std::unique_ptr<cudf::table> result;

  if (input.num_rows() == 0) {
    result = std::make_unique<cudf::table>(input, stream, mr);
  } else if (partitionKeys_.empty()) {
    auto keyView = input.select(sortKeys_);
    std::vector<cudf::order> sortOrders(
        columnOrders_.begin() + partitionKeys_.size(), columnOrders_.end());
    std::vector<cudf::null_order> sortNullOrders(
        nullOrders_.begin() + partitionKeys_.size(), nullOrders_.end());

    auto sortedIndices = cudf::stable_sorted_order(
        keyView, sortOrders, sortNullOrders, stream, mr);
    auto firstIndex = cudf::split(sortedIndices->view(), {1}, stream).front();
    result = cudf::gather(
        input,
        firstIndex,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
  } else if (
      !sortKeys_.empty() &&
      std::all_of(
          sortKeys_.begin(),
          sortKeys_.end(),
          [&](auto key) {
            const auto index =
                std::find(sortKeys_.begin(), sortKeys_.end(), key) -
                sortKeys_.begin();
            const auto order = columnOrders_[partitionKeys_.size() + index];
            const auto nullOrder =
                nullOrders_[partitionKeys_.size() + index];
            // Spark NULLS LAST maps to AFTER for ascending and BEFORE for
            // descending in the cuDF ordering representation used here.
            return order == cudf::order::ASCENDING
                ? nullOrder == cudf::null_order::AFTER
                : nullOrder == cudf::null_order::BEFORE;
          })) {
    // A full stable sort is unnecessary for ROW_NUMBER=1. Reduce
    // lexicographically: keep the min/max rows for the first ordering key,
    // then repeat over the surviving ties for each later key. Finally choose
    // one arbitrary peer per partition; SQL ROW_NUMBER does not define the
    // order among rows whose complete ordering keys are equal.
    result = computeGroupedLexicographicTop(input, stream, mr, false);
  } else {
    auto allKeysView = input.select(allKeyIndices_);
    auto sortedIndices = cudf::stable_sorted_order(
        allKeysView, columnOrders_, nullOrders_, stream, mr);
    auto sortedTable = cudf::gather(
        input,
        sortedIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    result = cudf::unique(
        sortedTable->view(),
        partitionKeys_,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        stream,
        mr);
  }

  if (generateRowNumber_) {
    auto one = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
    auto rowNumber =
        cudf::make_column_from_scalar(one, result->num_rows(), stream, mr);
    auto columns = result->release();
    columns.push_back(std::move(rowNumber));
    result = std::make_unique<cudf::table>(std::move(columns));
  }

  stream.synchronize();
  return std::make_shared<CudfVector>(
      pool(), outputType_, result->num_rows(), std::move(result), stream);
}

std::unique_ptr<cudf::table>
CudfTopNRowNumber::computeGroupedLexicographicTop(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool preserveTies) {
  VELOX_CHECK(!partitionKeys_.empty());
  VELOX_CHECK(!sortKeys_.empty());

  // Keep the historical full-row path as a diagnostic fallback. The narrow
  // path below is semantically equivalent, but the switch makes an A/B run or
  // an emergency rollback possible without rebuilding the native library.
  if (!topNLatePayloadGatherEnabled()) {
    std::unique_ptr<cudf::table> candidates;
    for (size_t sortIndex = 0; sortIndex < sortKeys_.size(); ++sortIndex) {
      const auto current = candidates ? candidates->view() : input;
      cudf::groupby::groupby grouper(
          current.select(partitionKeys_), cudf::null_policy::INCLUDE);
      std::vector<cudf::groupby::aggregation_request> requests(1);
      requests[0].values = current.column(sortKeys_[sortIndex]);
      if (columnOrders_[partitionKeys_.size() + sortIndex] ==
          cudf::order::ASCENDING) {
        requests[0].aggregations.push_back(
            cudf::make_min_aggregation<cudf::groupby_aggregation>());
      } else {
        requests[0].aggregations.push_back(
            cudf::make_max_aggregation<cudf::groupby_aggregation>());
      }
      auto [groupKeys, aggregateResults] =
          grouper.aggregate(requests, stream, mr);
      VELOX_CHECK_EQ(aggregateResults.size(), 1);
      VELOX_CHECK_EQ(aggregateResults[0].results.size(), 1);
      auto topKeyColumns = groupKeys->release();
      topKeyColumns.push_back(std::move(aggregateResults[0].results[0]));
      auto topKeys = std::make_unique<cudf::table>(std::move(topKeyColumns));

      auto probeKeyIndices = partitionKeys_;
      probeKeyIndices.push_back(sortKeys_[sortIndex]);
      cudf::hash_join lookup(
          topKeys->view(),
          cudf::nullable_join::YES,
          cudf::null_equality::EQUAL,
          0.5,
          stream);
      auto joinIndices = lookup.inner_join(
          current.select(probeKeyIndices), std::nullopt, stream, mr);
      auto probeIndices = cudf::column_view{
          cudf::device_span<cudf::size_type const>{*joinIndices.first}};

      candidates = cudf::gather(
          current,
          probeIndices,
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          mr);
    }

    if (!preserveTies && candidates->num_rows() > 0) {
      candidates = cudf::distinct(
          candidates->view(),
          partitionKeys_,
          cudf::duplicate_keep_option::KEEP_FIRST,
          cudf::null_equality::EQUAL,
          cudf::nan_equality::ALL_EQUAL,
          stream,
          mr);
    }
    return candidates;
  }

  // Carry only partition/sort keys and original row indices through the
  // iterative grouped reductions. In particular, do not gather wide payload
  // columns (MAP/ARRAY/STRING) once per ordering key and again for distinct.
  // The final row set is gathered from the input exactly once.
  const auto keyInput = input.select(allKeyIndices_);
  std::vector<cudf::size_type> narrowPartitionKeys(partitionKeys_.size());
  std::iota(
      narrowPartitionKeys.begin(), narrowPartitionKeys.end(), 0);

  if (!preserveTies && topNLatePayloadSortEnabled()) {
    auto sortedIndices = cudf::stable_sorted_order(
        keyInput, columnOrders_, nullOrders_, stream, mr);
    auto sortedPartitionKeys = cudf::gather(
        input.select(partitionKeys_),
        sortedIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    auto narrowColumns = sortedPartitionKeys->release();
    narrowColumns.push_back(std::move(sortedIndices));
    auto narrowCandidates =
        std::make_unique<cudf::table>(std::move(narrowColumns));
    auto distinctCandidates = cudf::distinct(
        narrowCandidates->view(),
        narrowPartitionKeys,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL,
        stream,
        mr);
    auto distinctColumns = distinctCandidates->release();
    VELOX_CHECK_EQ(
        distinctColumns.size(), narrowPartitionKeys.size() + 1);
    auto finalIndices = std::move(distinctColumns.back());
    return cudf::gather(
        input,
        finalIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
  }

  auto candidateIndices = cudf::sequence(
      input.num_rows(),
      cudf::numeric_scalar<cudf::size_type>(0, true, stream, mr),
      cudf::numeric_scalar<cudf::size_type>(1, true, stream, mr),
      stream,
      mr);

  std::unique_ptr<cudf::table> gatheredKeys;
  for (size_t sortIndex = 0; sortIndex < sortKeys_.size(); ++sortIndex) {
    const auto current = [&]() {
      if (sortIndex == 0) {
        return keyInput;
      }
      gatheredKeys = cudf::gather(
          keyInput,
          candidateIndices->view(),
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          mr);
      return gatheredKeys->view();
    }();
    cudf::groupby::groupby grouper(
        current.select(narrowPartitionKeys), cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values =
        current.column(partitionKeys_.size() + sortIndex);
    if (columnOrders_[partitionKeys_.size() + sortIndex] ==
        cudf::order::ASCENDING) {
      requests[0].aggregations.push_back(
          cudf::make_min_aggregation<cudf::groupby_aggregation>());
    } else {
      requests[0].aggregations.push_back(
          cudf::make_max_aggregation<cudf::groupby_aggregation>());
    }
    auto [groupKeys, aggregateResults] =
        grouper.aggregate(requests, stream, mr);
    VELOX_CHECK_EQ(aggregateResults.size(), 1);
    VELOX_CHECK_EQ(aggregateResults[0].results.size(), 1);
    auto topKeyColumns = groupKeys->release();
    topKeyColumns.push_back(std::move(aggregateResults[0].results[0]));
    auto topKeys = std::make_unique<cudf::table>(std::move(topKeyColumns));

    auto probeKeyIndices = narrowPartitionKeys;
    probeKeyIndices.push_back(partitionKeys_.size() + sortIndex);
    cudf::hash_join lookup(
        topKeys->view(),
        cudf::nullable_join::YES,
        cudf::null_equality::EQUAL,
        0.5,
        stream);
    auto joinIndices = lookup.inner_join(
        current.select(probeKeyIndices), std::nullopt, stream, mr);
    auto probeIndices = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*joinIndices.first}};

    auto remappedIndices = cudf::gather(
        cudf::table_view{{candidateIndices->view()}},
        probeIndices,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    auto remappedColumns = remappedIndices->release();
    VELOX_CHECK_EQ(remappedColumns.size(), 1);
    candidateIndices = std::move(remappedColumns.front());
  }

  if (!preserveTies && candidateIndices->size() > 0) {
    auto candidatePartitionKeys = cudf::gather(
        input.select(partitionKeys_),
        candidateIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    auto narrowColumns = candidatePartitionKeys->release();
    narrowColumns.push_back(std::move(candidateIndices));
    auto narrowCandidates =
        std::make_unique<cudf::table>(std::move(narrowColumns));
    auto distinctCandidates = cudf::distinct(
        narrowCandidates->view(),
        narrowPartitionKeys,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        cudf::nan_equality::ALL_EQUAL,
        stream,
        mr);
    auto distinctColumns = distinctCandidates->release();
    VELOX_CHECK_EQ(
        distinctColumns.size(), narrowPartitionKeys.size() + 1);
    candidateIndices = std::move(distinctColumns.back());
  }

  return cudf::gather(
      input,
      candidateIndices->view(),
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
}

CudfVectorPtr CudfTopNRowNumber::computeLimitOneRankLike(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::lock_guard<std::mutex> cucoLock(cudfCucoMutex());
  VELOX_CHECK(!sortKeys_.empty(), "Rank-like TopNRowNumber requires sort keys");

  std::unique_ptr<cudf::table> result;
  if (input.num_rows() == 0) {
    result = std::make_unique<cudf::table>(input, stream, mr);
  } else if (
      !partitionKeys_.empty() &&
      std::all_of(
          sortKeys_.begin(),
          sortKeys_.end(),
          [&](auto key) {
            const auto index =
                std::find(sortKeys_.begin(), sortKeys_.end(), key) -
                sortKeys_.begin();
            const auto order = columnOrders_[partitionKeys_.size() + index];
            const auto nullOrder =
                nullOrders_[partitionKeys_.size() + index];
            return order == cudf::order::ASCENDING
                ? nullOrder == cudf::null_order::AFTER
                : nullOrder == cudf::null_order::BEFORE;
          })) {
    // Preserve every peer that ties on the complete lexicographic ordering
    // key, which is exactly RANK/DENSE_RANK limit=1 semantics.
    result = computeGroupedLexicographicTop(input, stream, mr, true);
  } else {
    auto allKeysView = input.select(allKeyIndices_);
    auto sortedIndices = cudf::stable_sorted_order(
        allKeysView, columnOrders_, nullOrders_, stream, mr);
    auto sortedTable = cudf::gather(
        input,
        sortedIndices->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);

    std::unique_ptr<cudf::table> topRows;
    if (partitionKeys_.empty()) {
      auto firstIndex = cudf::split(sortedIndices->view(), {1}, stream).front();
      topRows = cudf::gather(
          input,
          firstIndex,
          cudf::out_of_bounds_policy::DONT_CHECK,
          cudf::negative_index_policy::NOT_ALLOWED,
          stream,
          mr);
    } else {
      topRows = cudf::unique(
          sortedTable->view(),
          partitionKeys_,
          cudf::duplicate_keep_option::KEEP_FIRST,
          cudf::null_equality::EQUAL,
          stream,
          mr);
    }

    auto topKeyView = topRows->view().select(allKeyIndices_);
    auto probeKeyView = sortedTable->view().select(allKeyIndices_);
    cudf::hash_join lookup(
        topKeyView,
        cudf::nullable_join::YES,
        cudf::null_equality::EQUAL,
        0.5,
        stream);
    auto joinIndices =
        lookup.inner_join(probeKeyView, std::nullopt, stream, mr);
    auto leftIndicesCol = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*joinIndices.first}};
    result = cudf::gather(
        sortedTable->view(),
        leftIndicesCol,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
  }

  if (generateRowNumber_) {
    auto one = cudf::numeric_scalar<int64_t>(1, true, stream, mr);
    auto rowNumber =
        cudf::make_column_from_scalar(one, result->num_rows(), stream, mr);
    auto columns = result->release();
    columns.push_back(std::move(rowNumber));
    result = std::make_unique<cudf::table>(std::move(columns));
  }

  stream.synchronize();
  return std::make_shared<CudfVector>(
      pool(), outputType_, result->num_rows(), std::move(result), stream);
}

} // namespace facebook::velox::cudf_velox
