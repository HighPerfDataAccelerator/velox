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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Task.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
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
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <malloc.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kSortedRunBytes = 3ULL << 30;
constexpr uint64_t kDefaultCandidateRunBytes = 128ULL << 20;
constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
constexpr size_t kMergeFanIn = 4;
constexpr int kHostCandidatePartitions = 512;
constexpr cudf::size_type kMaxCompleteOutputRows = 262144;
constexpr uint64_t kCandidateMergeWorkspaceMultiplier = 8;
constexpr uint64_t kCandidateMergeWorkspaceOverhead = 64ULL << 20;
constexpr uint64_t kDeviceHeadroomReserveBytes = 1ULL << 30;
constexpr uint64_t kMaxUnadmittedCandidateWorkspaceBytes = 1ULL << 30;
constexpr uint64_t kCandidatePartitionWorkspaceMultiplier = 2;
constexpr uint64_t kCandidatePartitionWorkspaceOverhead = 16ULL << 20;
constexpr uint64_t kMinimumHostPressureTrimTriggerBytes = 2ULL << 30;
constexpr std::string_view kConditionalTopNMarker = "__gluten_mpp_topn_active";
std::atomic<uint64_t> spillDirectorySequence{0};

uint64_t saturatingAdd(uint64_t left, uint64_t right) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return std::numeric_limits<uint64_t>::max();
  }
  return left + right;
}

uint64_t saturatingMultiply(uint64_t value, uint64_t multiplier) {
  if (value > std::numeric_limits<uint64_t>::max() / multiplier) {
    return std::numeric_limits<uint64_t>::max();
  }
  return value * multiplier;
}

struct CandidateWorkspaceAdmission {
  std::optional<DeviceMemoryAdmissionReservation> reservation;
  DeviceAllocationHeadroom headroom;
  uint64_t estimatedWorkspaceBytes;
  uint64_t admissionCapacityBytes;
  uint64_t alreadyReservedBytes;

  [[nodiscard]] uint64_t availableCapacityBytes() const {
    return admissionCapacityBytes > alreadyReservedBytes
        ? admissionCapacityBytes - alreadyReservedBytes
        : 0;
  }
};

CandidateWorkspaceAdmission acquireCandidateWorkspaceAdmission(
    uint64_t candidateBytes,
    uint64_t workspaceMultiplier = kCandidateMergeWorkspaceMultiplier,
    uint64_t workspaceOverhead = kCandidateMergeWorkspaceOverhead) {
  const auto estimatedWorkspaceBytes = saturatingAdd(
      saturatingMultiply(candidateBytes, workspaceMultiplier),
      workspaceOverhead);
  auto headroom = captureDeviceAllocationHeadroom();
  const auto allocatableBytes =
      static_cast<uint64_t>(headroom.allocatableBytes());
  const auto reserveBytes = headroom.totalBytes == 0
      ? kDeviceHeadroomReserveBytes
      : std::max<uint64_t>(
            kDeviceHeadroomReserveBytes,
            static_cast<uint64_t>(headroom.totalBytes) / uint64_t{50});
  const auto admissionCapacity = allocatableBytes > reserveBytes
      ? allocatableBytes - reserveBytes
      : allocatableBytes / uint64_t{2};
  const auto alreadyReserved =
      deviceMemoryAdmissionReservedBytes(headroom.device);
  std::optional<DeviceMemoryAdmissionReservation> reservation;
  if (headroom.cudaValid) {
    reservation = tryAcquireDeviceMemoryAdmission(
        headroom.device, estimatedWorkspaceBytes, admissionCapacity);
  }
  return {
      std::move(reservation),
      std::move(headroom),
      estimatedWorkspaceBytes,
      admissionCapacity,
      alreadyReserved};
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
      inputType_(node->inputType()),
      diagnosticNodeId_(node->id()),
      candidateRunBytes_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
          kDefaultCandidateRunBytes)),
      spillStream_(cudfGlobalStreamPool().get_stream()) {
  VELOX_CHECK_EQ(limit_, 1, "CudfTopNRowNumber only supports limit=1");
  VELOX_CHECK_GT(
      candidateRunBytes_,
      0,
      "CudfTopNRowNumber candidate run bytes must be greater than zero");
  VELOX_CHECK(
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber ||
          rankFunction_ == core::TopNRowNumberNode::RankFunction::kRank ||
          rankFunction_ == core::TopNRowNumberNode::RankFunction::kDenseRank,
      "CudfTopNRowNumber only supports row_number, rank, or dense_rank");

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

void CudfTopNRowNumber::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "Expected CudfVector input");

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
  auto mr = get_output_mr();
  const auto inputView = cudfInput->getTableView();
  const auto inputFlatBytes = cudfInput->estimateFlatSize();
  if (inputFlatBytes >= candidateRunBytes_) {
    auto inputAdmission = acquireCandidateWorkspaceAdmission(inputFlatBytes);
    addRuntimeStat(
        "topNRowNumberInputPressureChecks", RuntimeCounter(int64_t{1}));
    if (!inputAdmission.reservation) {
      bool retainOnHost = !spilled_ && supportsHostCandidateBuckets();
      if (candidates_ && retainOnHost) {
        retainOnHost = tryRetainCurrentCandidatesOnHost(mr);
      }
      if (candidates_) {
        spillCandidates(SpillReason::kPressure);
      }

      const auto maxCandidateBytes = (kMaxUnadmittedCandidateWorkspaceBytes -
                                      kCandidateMergeWorkspaceOverhead) /
          kCandidateMergeWorkspaceMultiplier;
      auto targetFlatBytes = std::min(candidateRunBytes_, maxCandidateBytes);
      const auto availableCapacity = inputAdmission.availableCapacityBytes();
      if (availableCapacity > kCandidateMergeWorkspaceOverhead) {
        targetFlatBytes = std::min(
            targetFlatBytes,
            (availableCapacity - kCandidateMergeWorkspaceOverhead) /
                kCandidateMergeWorkspaceMultiplier);
      }
      const auto inputRows = static_cast<uint64_t>(inputView.num_rows());
      const auto averageRowBytes = inputFlatBytes / inputRows +
          static_cast<uint64_t>(inputFlatBytes % inputRows != 0);
      const auto targetRows = std::max<uint64_t>(
          1, std::min(inputRows, targetFlatBytes / averageRowBytes));
      const auto numChunks = inputRows / targetRows +
          static_cast<uint64_t>(inputRows % targetRows != 0);
      addRuntimeStat(
          "topNRowNumberInputPressureSplits", RuntimeCounter(int64_t{1}));
      addRuntimeStat(
          "topNRowNumberInputPressureChunks",
          RuntimeCounter(saturateCast(numChunks)));
      LOG(INFO) << "CudfTopNRowNumber splitting input under pressure node="
                << diagnosticNodeId_ << " inputRows=" << inputRows
                << " inputFlatBytes=" << inputFlatBytes
                << " targetRows=" << targetRows
                << " targetFlatBytes=" << targetFlatBytes
                << " chunks=" << numChunks << " estimatedWorkspaceBytes="
                << inputAdmission.estimatedWorkspaceBytes
                << " cudaFreeBytes=" << inputAdmission.headroom.freeBytes
                << " poolReusableBytes="
                << inputAdmission.headroom.reusablePoolBytes()
                << " admissionCapacityBytes="
                << inputAdmission.admissionCapacityBytes
                << " alreadyReservedBytes="
                << inputAdmission.alreadyReservedBytes;

      for (uint64_t begin = 0; begin < inputRows; begin += targetRows) {
        const auto end = std::min(inputRows, begin + targetRows);
        auto slices = cudf::slice(
            inputView,
            {static_cast<cudf::size_type>(begin),
             static_cast<cudf::size_type>(end)},
            stream);
        VELOX_CHECK_EQ(slices.size(), 1);
        const auto chunkFlatBytes =
            averageRowBytes * static_cast<uint64_t>(end - begin);
        auto chunkAdmission =
            acquireCandidateWorkspaceAdmission(chunkFlatBytes);
        auto batchCandidates = reduceToCandidates(
            slices[0], stream, mr, chunkFlatBytes, ReductionSource::kInput);
        chunkAdmission.reservation.reset();
        if (retainOnHost &&
            tryRetainHostCandidateBatch(batchCandidates, stream, mr)) {
          continue;
        }
        if (retainOnHost) {
          spillHostCandidateBatches(SpillReason::kInputPressure);
          retainOnHost = false;
        }
        candidates_ = std::move(batchCandidates.table);
        candidateBytes_ = batchCandidates.flatBytes;
        candidateStream_ = stream;
        spillCandidates(SpillReason::kInputPressure);
      }
      return;
    }

    auto batchCandidates = reduceToCandidates(
        inputView, stream, mr, inputFlatBytes, ReductionSource::kInput);
    inputAdmission.reservation.reset();
    addBatchCandidates(std::move(batchCandidates), stream, mr);
    return;
  }

  addBatchCandidates(
      reduceToCandidates(
          inputView, stream, mr, inputFlatBytes, ReductionSource::kInput),
      stream,
      mr);
}

void CudfTopNRowNumber::addBatchCandidates(
    CandidateTable batchCandidates,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (hostCandidateMode_) {
    if (tryRetainHostCandidateBatch(batchCandidates, stream, mr)) {
      return;
    }
    spillHostCandidateBatches(SpillReason::kPressure);
    candidates_ = std::move(batchCandidates.table);
    candidateBytes_ = batchCandidates.flatBytes;
    candidateStream_ = stream;
    spillCandidates(SpillReason::kPressure);
    return;
  }

  if (candidates_ && candidates_->num_rows() > 0) {
    bool pressureSpilled = false;
    const auto projectedCandidateBytes =
        saturatingAdd(candidateBytes_, batchCandidates.flatBytes);
    std::optional<DeviceMemoryAdmissionReservation> memoryAdmission;
    if (projectedCandidateBytes >= candidateRunBytes_) {
      auto admission =
          acquireCandidateWorkspaceAdmission(projectedCandidateBytes);
      memoryAdmission = std::move(admission.reservation);
      addRuntimeStat("topNRowNumberPressureChecks", RuntimeCounter(int64_t{1}));
      VLOG(2) << "CudfTopNRowNumber candidate merge admission node="
              << diagnosticNodeId_
              << " retainedCandidateBytes=" << candidateBytes_
              << " batchCandidateBytes=" << batchCandidates.flatBytes
              << " projectedCandidateBytes=" << projectedCandidateBytes
              << " estimatedWorkspaceBytes="
              << admission.estimatedWorkspaceBytes
              << " cudaFreeBytes=" << admission.headroom.freeBytes
              << " poolReusableBytes=" << admission.headroom.reusablePoolBytes()
              << " admissionCapacityBytes=" << admission.admissionCapacityBytes
              << " alreadyReservedBytes=" << admission.alreadyReservedBytes
              << " admitted=" << memoryAdmission.has_value();
      if (!memoryAdmission) {
        if (!spilled_ && supportsHostCandidateBuckets() &&
            tryRetainCurrentCandidatesOnHost(mr) &&
            tryRetainHostCandidateBatch(batchCandidates, stream, mr)) {
          addRuntimeStat(
              "topNRowNumberHostPressureTransitions",
              RuntimeCounter(int64_t{1}));
          return;
        }
        if (hostCandidateMode_) {
          spillHostCandidateBatches(SpillReason::kPressure);
          candidates_ = std::move(batchCandidates.table);
          candidateBytes_ = batchCandidates.flatBytes;
          candidateStream_ = stream;
          spillCandidates(SpillReason::kPressure);
          return;
        }
        LOG(INFO)
            << "CudfTopNRowNumber spilling retained candidates before merge "
               "under pressure node="
            << diagnosticNodeId_
            << " retainedCandidateBytes=" << candidateBytes_
            << " batchCandidateBytes=" << batchCandidates.flatBytes
            << " projectedCandidateBytes=" << projectedCandidateBytes
            << " estimatedWorkspaceBytes=" << admission.estimatedWorkspaceBytes
            << " cudaFreeBytes=" << admission.headroom.freeBytes
            << " poolReusableBytes=" << admission.headroom.reusablePoolBytes();
        spillCandidates(SpillReason::kPressure);
        candidates_ = std::move(batchCandidates.table);
        candidateBytes_ = batchCandidates.flatBytes;
        candidateStream_ = stream;
        pressureSpilled = true;
      }
    }

    if (!pressureSpilled) {
      if (candidateStream_->value() != stream.value()) {
        std::vector<rmm::cuda_stream_view> candidateStreams{*candidateStream_};
        cudf::detail::join_streams(candidateStreams, stream);
        auto columns = candidates_->release();
        for (auto& column : columns) {
          column = cudf::rebind_stream(std::move(*column), stream);
        }
        candidates_ = std::make_unique<cudf::table>(std::move(columns));
      }
      std::vector<cudf::table_view> pieces{
          candidates_->view(), batchCandidates.table->view()};
      auto merged = cudf::concatenate(pieces, stream, mr);
      auto reduced = reduceToCandidates(
          merged->view(),
          stream,
          mr,
          projectedCandidateBytes,
          ReductionSource::kCandidateMerge);
      candidates_ = std::move(reduced.table);
      candidateBytes_ = reduced.flatBytes;
      candidateStream_ = stream;
      addRuntimeStat(
          "topNRowNumberCandidateMerges", RuntimeCounter(int64_t{1}));
    }
  } else {
    candidates_ = std::move(batchCandidates.table);
    candidateBytes_ = batchCandidates.flatBytes;
    candidateStream_ = stream;
  }

  if (candidateBytes_ >= candidateRunBytes_) {
    if (!spilled_ && supportsHostCandidateBuckets() &&
        tryRetainCurrentCandidatesOnHost(mr)) {
      addRuntimeStat(
          "topNRowNumberHostThresholdTransitions", RuntimeCounter(int64_t{1}));
      return;
    }
    spillCandidates(SpillReason::kThreshold);
  }
}

bool CudfTopNRowNumber::supportsHostCandidateBuckets() const {
  return rankFunction_ != core::TopNRowNumberNode::RankFunction::kRowNumber &&
      !partitionKeys_.empty();
}

bool CudfTopNRowNumber::tryRetainHostCandidateBatch(
    CandidateTable& batchCandidates,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(supportsHostCandidateBuckets());
  if (!batchCandidates.table || batchCandidates.table->num_rows() == 0) {
    return true;
  }

  auto admission = acquireCandidateWorkspaceAdmission(
      batchCandidates.flatBytes,
      kCandidatePartitionWorkspaceMultiplier,
      kCandidatePartitionWorkspaceOverhead);
  if (!admission.reservation) {
    addRuntimeStat(
        "topNRowNumberHostHeadroomRejects", RuntimeCounter(int64_t{1}));
    return false;
  }

  auto stateStream = spillStream_;
  if (stream.value() != stateStream.value()) {
    std::vector<rmm::cuda_stream_view> inputStreams{stream};
    cudf::detail::join_streams(inputStreams, stateStream);
    auto columns = batchCandidates.table->release();
    for (auto& column : columns) {
      column = cudf::rebind_stream(std::move(*column), stateStream);
    }
    batchCandidates.table = std::make_unique<cudf::table>(std::move(columns));
  }

  auto [partitioned, offsets] = cudf::hash_partition(
      batchCandidates.table->view(),
      partitionKeys_,
      kHostCandidatePartitions,
      cudf::hash_id::HASH_MURMUR3,
      cudf::DEFAULT_HASH_SEED,
      stateStream,
      mr);
  VELOX_CHECK_EQ(
      offsets.size(), static_cast<size_t>(kHostCandidatePartitions + 1));
  std::vector<cudf::size_type> splitPoints(
      offsets.begin() + 1, offsets.end() - 1);
  auto packedBuckets =
      cudf::contiguous_split(partitioned->view(), splitPoints, stateStream, mr);
  VELOX_CHECK_EQ(
      packedBuckets.size(), static_cast<size_t>(kHostCandidatePartitions));
  if (hostCandidateBuckets_.empty()) {
    hostCandidateBuckets_.resize(kHostCandidatePartitions);
  }
  if (!hostCandidateCodec_) {
    auto codec = common::Codec::create(common::CompressionKind_LZ4);
    if (!codec.hasValue()) {
      VELOX_FAIL(
          "Failed to create LZ4 codec for TopN candidate state: {}",
          codec.error().message());
    }
    hostCandidateCodec_ = std::move(codec.value());
  }

  const auto totalRows =
      static_cast<uint64_t>(batchCandidates.table->num_rows());
  struct PendingHostCandidateChunk {
    size_t bucket;
    std::unique_ptr<std::vector<uint8_t>> metadata;
    BufferPtr data;
    cudf::size_type rows;
    uint64_t flatBytes;
  };
  std::vector<PendingHostCandidateChunk> pendingChunks;
  pendingChunks.reserve(packedBuckets.size());
  uint64_t packedBatchBytes = 0;
  for (size_t bucket = 0; bucket < packedBuckets.size(); ++bucket) {
    auto& packed = packedBuckets[bucket];
    const auto rows = static_cast<uint64_t>(packed.table.num_rows());
    if (rows == 0) {
      continue;
    }
    const auto packedBytes = packed.data.gpu_data->size();
    auto hostData = AlignedBuffer::allocate<uint8_t>(packedBytes, pool());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostData->asMutable<uint8_t>(),
        packed.data.gpu_data->data(),
        packedBytes,
        cudaMemcpyDeviceToHost,
        stateStream.value()));
    const auto bucketFlatBytes = batchCandidates.flatBytes * rows / totalRows;
    pendingChunks.push_back(
        {bucket,
         std::move(packed.data.metadata),
         std::move(hostData),
         packed.table.num_rows(),
         bucketFlatBytes});
    packedBatchBytes = saturatingAdd(packedBatchBytes, packedBytes);
  }
  stateStream.synchronize();
  packedBuckets.clear();
  partitioned.reset();
  batchCandidates.table.reset();
  admission.reservation.reset();
  maybeTrimDeviceCacheUnderHostPressure();

  uint64_t compressedBatchBytes = 0;
  for (auto& pending : pendingChunks) {
    const auto uncompressedBytes = pending.data->size();
    auto compressionScratch = AlignedBuffer::allocate<uint8_t>(
        hostCandidateCodec_->maxCompressedLength(uncompressedBytes), pool());
    auto compressedBytes =
        hostCandidateCodec_
            ->compress(
                pending.data->as<uint8_t>(),
                uncompressedBytes,
                compressionScratch->asMutable<uint8_t>(),
                compressionScratch->size())
            .thenOrThrow(folly::identity, [](const Status& status) {
              VELOX_FAIL(
                  "Failed to compress TopN candidate "
                  "state: {}",
                  status.message());
            });
    auto compressed = AlignedBuffer::allocate<uint8_t>(compressedBytes, pool());
    std::memcpy(
        compressed->asMutable<uint8_t>(),
        compressionScratch->as<uint8_t>(),
        compressedBytes);
    const auto retainedBytes = compressed->capacity();
    hostCandidateBuckets_[pending.bucket].push_back(
        {std::move(pending.metadata),
         std::move(compressed),
         uncompressedBytes,
         pending.rows,
         pending.flatBytes});
    hostCandidateBytes_ = saturatingAdd(hostCandidateBytes_, retainedBytes);
    maxHostCandidateBytes_ =
        std::max(maxHostCandidateBytes_, hostCandidateBytes_);
    hostCandidatePayloadBytes_ =
        saturatingAdd(hostCandidatePayloadBytes_, compressedBytes);
    maxHostCandidatePayloadBytes_ =
        std::max(maxHostCandidatePayloadBytes_, hostCandidatePayloadBytes_);
    hostCandidateUncompressedBytes_ =
        saturatingAdd(hostCandidateUncompressedBytes_, uncompressedBytes);
    maxHostCandidateUncompressedBytes_ = std::max(
        maxHostCandidateUncompressedBytes_, hostCandidateUncompressedBytes_);
    compressedBatchBytes = saturatingAdd(compressedBatchBytes, compressedBytes);
  }
  hostCandidateRows_ = saturatingAdd(hostCandidateRows_, totalRows);
  ++hostCandidateBatches_;
  batchCandidates.flatBytes = 0;
  hostCandidateMode_ = true;
  addRuntimeStat(
      "topNRowNumberHostCandidateBatches", RuntimeCounter(int64_t{1}));
  addRuntimeStat(
      "topNRowNumberHostCandidateBytes",
      RuntimeCounter(
          saturateCast(compressedBatchBytes), RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNRowNumberHostCandidateUncompressedBytes",
      RuntimeCounter(
          saturateCast(packedBatchBytes), RuntimeCounter::Unit::kBytes));
  return true;
}

bool CudfTopNRowNumber::tryRetainCurrentCandidatesOnHost(
    rmm::device_async_resource_ref mr) {
  if (!candidates_ || candidates_->num_rows() == 0) {
    return true;
  }
  VELOX_CHECK(candidateStream_.has_value());
  const auto retainedStream = *candidateStream_;
  CandidateTable retained{std::move(candidates_), candidateBytes_};
  candidates_.reset();
  candidateBytes_ = 0;
  candidateStream_.reset();
  if (tryRetainHostCandidateBatch(retained, retainedStream, mr)) {
    return true;
  }
  candidates_ = std::move(retained.table);
  candidateBytes_ = retained.flatBytes;
  candidateStream_ = retainedStream;
  return false;
}

void CudfTopNRowNumber::maybeTrimDeviceCacheUnderHostPressure() {
  const auto headroom = captureDeviceAllocationHeadroom();
  if (!headroom.cudaValid) {
    return;
  }
  const auto triggerBytes = std::max<uint64_t>(
      kMinimumHostPressureTrimTriggerBytes,
      static_cast<uint64_t>(headroom.totalBytes) / uint64_t{8});
  if (headroom.freeBytes >= triggerBytes) {
    return;
  }

  addRuntimeStat(
      "topNRowNumberHostDeviceTrimChecks", RuntimeCounter(int64_t{1}));
  if (!trimAsyncMemoryPools(0)) {
    return;
  }
  ++hostDeviceCacheTrims_;
  addRuntimeStat(
      "topNRowNumberHostDeviceCacheTrims", RuntimeCounter(int64_t{1}));
}

void CudfTopNRowNumber::spillHostCandidateBatches(SpillReason reason) {
  if (hostCandidateBuckets_.empty()) {
    hostCandidateMode_ = false;
    hostCandidateBytes_ = 0;
    hostCandidatePayloadBytes_ = 0;
    hostCandidateUncompressedBytes_ = 0;
    nextCandidatePartition_ = 0;
    return;
  }

  addRuntimeStat("topNRowNumberHostFallbackSpills", RuntimeCounter(int64_t{1}));
  while (auto output = computeNextHostCandidateOutput()) {
    candidateBytes_ = output->estimateFlatSize();
    candidates_ = output->release();
    candidateStream_ = output->stream();
    spillCandidates(reason);
  }
  hostCandidateBuckets_.clear();
  hostCandidateBytes_ = 0;
  hostCandidatePayloadBytes_ = 0;
  hostCandidateUncompressedBytes_ = 0;
  nextCandidatePartition_ = 0;
  hostCandidateMode_ = false;
}

CudfVectorPtr CudfTopNRowNumber::computeNextHostCandidateOutput() {
  auto stream = spillStream_;
  auto mr = get_output_mr();
  while (nextCandidatePartition_ <
         static_cast<size_t>(kHostCandidatePartitions)) {
    const auto partition = nextCandidatePartition_++;
    auto& hostPieces = hostCandidateBuckets_[partition];
    if (hostPieces.empty()) {
      continue;
    }

    std::vector<cudf::packed_columns> deviceOwners;
    std::vector<cudf::table_view> pieces;
    std::vector<BufferPtr> uncompressedPieces;
    deviceOwners.reserve(hostPieces.size());
    pieces.reserve(hostPieces.size());
    uncompressedPieces.reserve(hostPieces.size());
    for (auto& hostPiece : hostPieces) {
      auto uncompressed =
          AlignedBuffer::allocate<uint8_t>(hostPiece.uncompressedBytes, pool());
      auto decompressedBytes =
          hostCandidateCodec_
              ->decompress(
                  hostPiece.data->as<uint8_t>(),
                  hostPiece.data->size(),
                  uncompressed->asMutable<uint8_t>(),
                  uncompressed->size())
              .thenOrThrow(folly::identity, [](const Status& status) {
                VELOX_FAIL(
                    "Failed to decompress TopN candidate state: {}",
                    status.message());
              });
      VELOX_CHECK_EQ(decompressedBytes, hostPiece.uncompressedBytes);
      auto deviceData =
          std::make_unique<rmm::device_buffer>(decompressedBytes, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          deviceData->data(),
          uncompressed->as<uint8_t>(),
          decompressedBytes,
          cudaMemcpyHostToDevice,
          stream.value()));
      uncompressedPieces.push_back(std::move(uncompressed));
      deviceOwners.emplace_back(
          std::move(hostPiece.metadata), std::move(deviceData));
      pieces.push_back(cudf::unpack(deviceOwners.back()));
    }
    stream.synchronize();
    hostCandidateBytes_ = hostCandidateBytes_ > 0
        ? hostCandidateBytes_ -
            std::min<uint64_t>(
                hostCandidateBytes_,
                std::accumulate(
                    hostPieces.begin(),
                    hostPieces.end(),
                    uint64_t{0},
                    [](uint64_t bytes, const HostCandidateChunk& piece) {
                      return saturatingAdd(bytes, piece.data->capacity());
                    }))
        : 0;
    hostCandidateUncompressedBytes_ = hostCandidateUncompressedBytes_ > 0
        ? hostCandidateUncompressedBytes_ -
            std::min<uint64_t>(
                hostCandidateUncompressedBytes_,
                std::accumulate(
                    hostPieces.begin(),
                    hostPieces.end(),
                    uint64_t{0},
                    [](uint64_t bytes, const HostCandidateChunk& piece) {
                      return saturatingAdd(bytes, piece.uncompressedBytes);
                    }))
        : 0;
    hostCandidatePayloadBytes_ = hostCandidatePayloadBytes_ > 0
        ? hostCandidatePayloadBytes_ -
            std::min<uint64_t>(
                hostCandidatePayloadBytes_,
                std::accumulate(
                    hostPieces.begin(),
                    hostPieces.end(),
                    uint64_t{0},
                    [](uint64_t bytes, const HostCandidateChunk& piece) {
                      return saturatingAdd(bytes, piece.data->size());
                    }))
        : 0;
    hostPieces.clear();
    uncompressedPieces.clear();

    auto bucket = pieces.size() == 1
        ? std::make_unique<cudf::table>(pieces.front(), stream, mr)
        : cudf::concatenate(pieces, stream, mr);
    addRuntimeStat(
        "topNRowNumberHostOutputBuckets", RuntimeCounter(int64_t{1}));
    ++hostOutputBuckets_;
    return computeLimitOneRankLike(bucket->view(), stream, mr);
  }

  hostCandidateBuckets_.clear();
  hostCandidateBytes_ = 0;
  hostCandidatePayloadBytes_ = 0;
  hostCandidateUncompressedBytes_ = 0;
  nextCandidatePartition_ = 0;
  hostCandidateMode_ = false;
  return nullptr;
}

void CudfTopNRowNumber::doNoMoreInput() {
  Operator::noMoreInput();
  if (spilled_ && candidates_) {
    spillCandidates(SpillReason::kFinal);
  }
  if (spilled_) {
    compactSortedRunsForMerge();
    initializeSortedRunReaders();
  } else if (hostCandidateMode_) {
    addRuntimeStat(
        "topNRowNumberHostCandidatePath", RuntimeCounter(int64_t{1}));
  } else {
    addRuntimeStat("topNRowNumberNoSpillFastPath", RuntimeCounter(int64_t{1}));
  }
  if (!candidates_ && !hostCandidateMode_ && passthroughOutputs_.empty()) {
    finished_ = !spilled_;
    if (finished_) {
      logCandidateObservations();
    }
  }
}

RowVectorPtr CudfTopNRowNumber::doGetOutput() {
  if (!passthroughOutputs_.empty()) {
    auto output = std::move(passthroughOutputs_.front());
    passthroughOutputs_.pop_front();
    return output;
  }

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (hostCandidateMode_) {
    auto result = computeNextHostCandidateOutput();
    if (result != nullptr) {
      recordFinalCandidateOutput(result);
      return result;
    }
    finished_ = true;
    logCandidateObservations();
    return nullptr;
  }

  if (spilled_) {
    auto result = computeNextSortedOutput();
    if (result != nullptr) {
      recordFinalCandidateOutput(result);
      return result;
    }
    finished_ = true;
    cleanupSpillFiles();
    logCandidateObservations();
    return nullptr;
  }

  if (!candidates_) {
    finished_ = true;
    logCandidateObservations();
    return nullptr;
  }

  VELOX_CHECK(candidateStream_.has_value());
  auto stream = *candidateStream_;
  auto mr = get_output_mr();
  auto input = std::exchange(candidates_, nullptr);
  candidateBytes_ = 0;
  candidateStream_.reset();
  auto result =
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
      ? computeLimitOneRowNumber(input->view(), stream, mr)
      : computeLimitOneRankLike(input->view(), stream, mr);
  recordFinalCandidateOutput(result);
  finished_ = true;
  logCandidateObservations();
  return result;
}

CudfTopNRowNumber::CandidateTable CudfTopNRowNumber::reduceToCandidates(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    uint64_t inputFlatBytes,
    ReductionSource source) {
  auto reduced =
      rankFunction_ == core::TopNRowNumberNode::RankFunction::kRowNumber
      ? computeLimitOneRowNumber(input, stream, mr)
      : computeLimitOneRankLike(input, stream, mr);
  auto flatBytes = reduced->estimateFlatSize();
  const auto numRows = reduced->size();
  auto table = reduced->release();
  if (generateRowNumber_) {
    auto columns = table->release();
    VELOX_CHECK_EQ(
        columns.size(),
        inputType_->size() + 1,
        "Incremental TopN candidate has unexpected generated rank column");
    columns.pop_back();
    table = std::make_unique<cudf::table>(std::move(columns));
    const auto generatedBytes =
        static_cast<uint64_t>(numRows) * sizeof(int64_t);
    flatBytes = flatBytes > generatedBytes ? flatBytes - generatedBytes : 0;
  }
  if (source == ReductionSource::kInput) {
    inputReductionRows_ = saturatingAdd(inputReductionRows_, input.num_rows());
    inputReductionBytes_ = saturatingAdd(inputReductionBytes_, inputFlatBytes);
    inputCandidateRows_ = saturatingAdd(inputCandidateRows_, numRows);
    inputCandidateBytes_ = saturatingAdd(inputCandidateBytes_, flatBytes);
    maxInputCandidateRows_ =
        std::max<uint64_t>(maxInputCandidateRows_, numRows);
    maxInputCandidateBytes_ = std::max(maxInputCandidateBytes_, flatBytes);
  } else {
    candidateMergeRows_ = saturatingAdd(candidateMergeRows_, input.num_rows());
    candidateMergeBytes_ = saturatingAdd(candidateMergeBytes_, inputFlatBytes);
  }
  return {std::move(table), flatBytes};
}

void CudfTopNRowNumber::spillCandidates(SpillReason reason) {
  VELOX_CHECK_NOT_NULL(candidates_);
  VELOX_CHECK(candidateStream_.has_value());
  spilledCandidateRows_ =
      saturatingAdd(spilledCandidateRows_, candidates_->num_rows());
  auto candidateVector = std::make_shared<CudfVector>(
      pool(),
      inputType_,
      candidates_->num_rows(),
      std::move(candidates_),
      *candidateStream_);
  inputs_.push_back(std::move(candidateVector));
  bufferedBytes_ = candidateBytes_;
  candidateBytes_ = 0;
  candidateStream_.reset();
  spillSortedRun(reason);
}

void CudfTopNRowNumber::spillSortedRun(SpillReason reason) {
  if (inputs_.empty()) {
    return;
  }
  const auto runBytes = bufferedBytes_;

  namespace fs = std::filesystem;
  if (!spilled_) {
    const auto& taskSpillRoot =
        operatorCtx_->task()->getOrCreateSpillDirectory();
    VELOX_CHECK(
        !taskSpillRoot.empty(),
        "CudfTopNRowNumber requires an explicit Task spill directory");
    const auto sequence = spillDirectorySequence.fetch_add(1);
    spillDirectory_ = (fs::path(taskSpillRoot) /
                       fmt::format(
                           "velox-cudf-topn-spill-{}-{}",
                           static_cast<int64_t>(::getpid()),
                           sequence))
                          .string();
    fs::create_directories(spillDirectory_);
    spilled_ = true;
  }

  auto stream = spillStream_;
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
  ++spillRuns_;
  addRuntimeStat("topNRowNumberSpillRuns", RuntimeCounter(int64_t{1}));
  addRuntimeStat(
      "topNRowNumberSpilledCandidateBytes",
      RuntimeCounter(saturateCast(runBytes), RuntimeCounter::Unit::kBytes));
  switch (reason) {
    case SpillReason::kThreshold:
      addRuntimeStat(
          "topNRowNumberThresholdSpills", RuntimeCounter(int64_t{1}));
      break;
    case SpillReason::kPressure:
      addRuntimeStat("topNRowNumberPressureSpills", RuntimeCounter(int64_t{1}));
      break;
    case SpillReason::kInputPressure:
      addRuntimeStat(
          "topNRowNumberInputPressureSpills", RuntimeCounter(int64_t{1}));
      break;
    case SpillReason::kFinal:
      addRuntimeStat("topNRowNumberFinalSpills", RuntimeCounter(int64_t{1}));
      break;
  }
  VLOG(1) << "CudfTopNRowNumber wrote candidate run node=" << diagnosticNodeId_
          << " reason="
          << (reason == SpillReason::kThreshold           ? "threshold"
                  : reason == SpillReason::kPressure      ? "pressure"
                  : reason == SpillReason::kInputPressure ? "input-pressure"
                                                          : "final")
          << " rows=" << input->num_rows() << " candidateBytes=" << runBytes
          << " totalRuns=" << sortedRuns_.size();
  bufferedBytes_ = 0;
  ::malloc_trim(0);
}

void CudfTopNRowNumber::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  auto mr = get_output_mr();
  for (auto& run : sortedRuns_) {
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{run.path})
                       .build();
    run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        kMergeChunkBytes, 0, options, spillStream_, mr);
  }
  readersInitialized_ = true;
}

void CudfTopNRowNumber::compactSortedRunsForMerge() {
  auto stream = spillStream_;
  auto mr = get_output_mr();

  while (sortedRuns_.size() > kMergeFanIn) {
    std::vector<SortedRun> nextLevel;
    nextLevel.reserve((sortedRuns_.size() + kMergeFanIn - 1) / kMergeFanIn);
    std::vector<std::string> obsoletePaths;

    for (size_t begin = 0; begin < sortedRuns_.size(); begin += kMergeFanIn) {
      const auto end = std::min(sortedRuns_.size(), begin + kMergeFanIn);
      if (end - begin == 1) {
        nextLevel.push_back(std::move(sortedRuns_[begin]));
        continue;
      }

      std::vector<SortedRun*> runs;
      runs.reserve(end - begin);
      for (size_t index = begin; index < end; ++index) {
        auto options = cudf::io::parquet_reader_options::builder(
                           cudf::io::source_info{sortedRuns_[index].path})
                           .build();
        auto& run = sortedRuns_[index];
        run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
            kMergeChunkBytes, 0, options, stream, mr);
        run.chunk.reset();
        run.chunkOffset = 0;
        runs.push_back(&run);
      }

      const auto outputPath = fmt::format(
          "{}/merge-{:06}.parquet", spillDirectory_, spillFileSequence_++);
      auto writerOptions = cudf::io::chunked_parquet_writer_options::builder(
                               cudf::io::sink_info{outputPath})
                               .build();
      cudf::io::chunked_parquet_writer writer(writerOptions, stream);
      bool groupFinished{false};
      while (!groupFinished) {
        auto merged = mergeNextPausedBatch(runs, stream, mr, groupFinished);
        if (merged && merged->num_rows() > 0) {
          writer.write(merged->view());
        }
      }
      writer.close();

      for (size_t index = begin; index < end; ++index) {
        auto& run = sortedRuns_[index];
        run.reader.reset();
        run.chunk.reset();
        run.chunkOffset = 0;
        obsoletePaths.push_back(run.path);
      }
      SortedRun outputRun;
      outputRun.path = outputPath;
      nextLevel.push_back(std::move(outputRun));
    }
    stream.synchronize();
    for (const auto& path : obsoletePaths) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    sortedRuns_ = std::move(nextLevel);
  }
}

bool CudfTopNRowNumber::loadPausedChunk(SortedRun& run) {
  VELOX_CHECK_NOT_NULL(run.reader);
  if (run.chunk && run.chunkOffset < run.chunk->num_rows()) {
    return true;
  }
  run.chunk.reset();
  run.chunkOffset = 0;
  while (run.reader->has_next()) {
    auto chunk = run.reader->read_chunk();
    addRuntimeStat(
        "topNRowNumberMergeSourceChunks", RuntimeCounter(int64_t{1}));
    if (chunk.tbl->num_rows() == 0) {
      continue;
    }
    run.chunk = std::move(chunk.tbl);
    return true;
  }
  return false;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::mergeNextPausedBatch(
    std::vector<SortedRun*>& runs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finished) {
  if (finished) {
    return nullptr;
  }

  std::vector<SortedRun*> activeRuns;
  std::vector<cudf::table_view> remainingViews;
  activeRuns.reserve(runs.size());
  remainingViews.reserve(runs.size());
  for (auto* run : runs) {
    VELOX_CHECK_NOT_NULL(run);
    if (!loadPausedChunk(*run)) {
      continue;
    }
    auto slices = cudf::slice(
        run->chunk->view(), {run->chunkOffset, run->chunk->num_rows()}, stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    activeRuns.push_back(run);
    remainingViews.push_back(slices.front());
  }

  if (activeRuns.empty()) {
    finished = true;
    return nullptr;
  }

  std::vector<cudf::table_view> safeViews;
  std::vector<cudf::size_type> consumed(activeRuns.size(), 0);
  if (activeRuns.size() == 1) {
    safeViews.push_back(remainingViews.front());
    consumed.front() = remainingViews.front().num_rows();
  } else {
    std::vector<cudf::table_view> boundaryRows;
    boundaryRows.reserve(remainingViews.size());
    for (const auto& view : remainingViews) {
      auto last =
          cudf::slice(view, {view.num_rows() - 1, view.num_rows()}, stream);
      boundaryRows.push_back(last.front());
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
    for (size_t index = 0; index < remainingViews.size(); ++index) {
      auto positions = cudf::upper_bound(
          remainingViews[index].select(allKeyIndices_),
          boundary.front().select(allKeyIndices_),
          columnOrders_,
          nullOrders_,
          stream,
          mr);
      consumed[index] = firstSearchPosition(positions->view(), stream);
      if (consumed[index] == 0) {
        continue;
      }
      auto safe =
          cudf::slice(remainingViews[index], {0, consumed[index]}, stream);
      safeViews.push_back(safe.front());
    }
  }
  VELOX_CHECK(!safeViews.empty(), "Paused TopN merge made no progress");
  auto output = safeViews.size() == 1
      ? std::make_unique<cudf::table>(safeViews.front(), stream, mr)
      : cudf::merge(
            safeViews, allKeyIndices_, columnOrders_, nullOrders_, stream, mr);
  for (size_t index = 0; index < activeRuns.size(); ++index) {
    activeRuns[index]->chunkOffset += consumed[index];
  }
  addRuntimeStat("topNRowNumberMergeOutputBatches", RuntimeCounter(int64_t{1}));
  finished = true;
  for (auto* run : runs) {
    if ((run->chunk && run->chunkOffset < run->chunk->num_rows()) ||
        (run->reader && run->reader->has_next())) {
      finished = false;
      break;
    }
  }
  return output;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::mergeNextSortedBatch(
    bool& finalBatch) {
  std::vector<SortedRun*> runs;
  runs.reserve(sortedRuns_.size());
  for (auto& run : sortedRuns_) {
    runs.push_back(&run);
  }
  auto output =
      mergeNextPausedBatch(runs, spillStream_, get_output_mr(), mergeFinished_);
  finalBatch = mergeFinished_;
  return output;
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
  if (completeEnd > kMaxCompleteOutputRows) {
    auto boundaryPartition = cudf::slice(
        partitionColumns,
        {kMaxCompleteOutputRows, kMaxCompleteOutputRows + 1},
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
  auto stream = spillStream_;
  auto mr = get_output_mr();
  while (!mergeFinished_ || partitionCarry_) {
    bool finalBatch = false;
    auto sorted = mergeNextSortedBatch(finalBatch);
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
  spillStream_.synchronize();
  sortedRuns_.clear();
  partitionCarry_.reset();
  spillStream_.synchronize();
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
  logCandidateObservations();
  try {
    spillStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfTopNRowNumber close pre-destruction cleanup failed: "
                 << error.what();
  }
  inputs_.clear();
  candidates_.reset();
  candidateBytes_ = 0;
  candidateStream_.reset();
  hostCandidateBuckets_.clear();
  hostCandidateBytes_ = 0;
  hostCandidatePayloadBytes_ = 0;
  hostCandidateUncompressedBytes_ = 0;
  hostCandidateCodec_.reset();
  nextCandidatePartition_ = 0;
  hostCandidateMode_ = false;
  passthroughOutputs_.clear();
  sortedRuns_.clear();
  partitionCarry_.reset();
  try {
    spillStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfTopNRowNumber close post-destruction cleanup failed: "
                 << error.what();
  }
  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    if (error) {
      LOG(WARNING) << "Failed to remove CudfTopNRowNumber spill directory '"
                   << spillDirectory_ << "': " << error.message();
    } else {
      spillDirectory_.clear();
    }
  }
  ::malloc_trim(0);
  Operator::close();
}

void CudfTopNRowNumber::recordFinalCandidateOutput(
    const CudfVectorPtr& output) {
  VELOX_CHECK_NOT_NULL(output);
  finalCandidateRows_ = saturatingAdd(finalCandidateRows_, output->size());
  finalCandidateBytes_ =
      saturatingAdd(finalCandidateBytes_, output->estimateFlatSize());
}

void CudfTopNRowNumber::logCandidateObservations() {
  if (candidateObservationsLogged_) {
    return;
  }
  candidateObservationsLogged_ = true;

  addRuntimeStat(
      "topNRowNumberInputReductionRows",
      RuntimeCounter(saturateCast(inputReductionRows_)));
  addRuntimeStat(
      "topNRowNumberInputReductionBytes",
      RuntimeCounter(
          saturateCast(inputReductionBytes_), RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNRowNumberInputCandidateRows",
      RuntimeCounter(saturateCast(inputCandidateRows_)));
  addRuntimeStat(
      "topNRowNumberInputCandidateBytes",
      RuntimeCounter(
          saturateCast(inputCandidateBytes_), RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNRowNumberCandidateMergeRows",
      RuntimeCounter(saturateCast(candidateMergeRows_)));
  addRuntimeStat(
      "topNRowNumberCandidateMergeBytes",
      RuntimeCounter(
          saturateCast(candidateMergeBytes_), RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNRowNumberFinalCandidateRows",
      RuntimeCounter(saturateCast(finalCandidateRows_)));
  addRuntimeStat(
      "topNRowNumberFinalCandidateBytes",
      RuntimeCounter(
          saturateCast(finalCandidateBytes_), RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "topNRowNumberSpilledCandidateRows",
      RuntimeCounter(saturateCast(spilledCandidateRows_)));
  addRuntimeStat(
      "topNRowNumberHostCandidateRows",
      RuntimeCounter(saturateCast(hostCandidateRows_)));
  addRuntimeStat(
      "topNRowNumberHostOutputBucketCount",
      RuntimeCounter(saturateCast(hostOutputBuckets_)));

  if (inputReductionBytes_ >= candidateRunBytes_ || spilled_) {
    LOG(WARNING) << "CudfTopNRowNumber candidate observations node="
                 << diagnosticNodeId_ << " inputRows=" << inputReductionRows_
                 << " inputFlatBytes=" << inputReductionBytes_
                 << " localCandidateRows=" << inputCandidateRows_
                 << " localCandidateFlatBytes=" << inputCandidateBytes_
                 << " maxLocalCandidateRows=" << maxInputCandidateRows_
                 << " maxLocalCandidateFlatBytes=" << maxInputCandidateBytes_
                 << " candidateMergeRows=" << candidateMergeRows_
                 << " candidateMergeFlatBytes=" << candidateMergeBytes_
                 << " spilledCandidateRows=" << spilledCandidateRows_
                 << " hostCandidateRows=" << hostCandidateRows_
                 << " maxHostCandidateAllocatedBytes=" << maxHostCandidateBytes_
                 << " maxHostCandidateCompressedBytes="
                 << maxHostCandidatePayloadBytes_
                 << " maxHostCandidateUncompressedBytes="
                 << maxHostCandidateUncompressedBytes_
                 << " hostCandidateBatches=" << hostCandidateBatches_
                 << " hostOutputBuckets=" << hostOutputBuckets_
                 << " hostDeviceCacheTrims=" << hostDeviceCacheTrims_
                 << " finalCandidateRows=" << finalCandidateRows_
                 << " finalCandidateFlatBytes=" << finalCandidateBytes_
                 << " spillRuns=" << spillRuns_;
  }
}

CudfVectorPtr CudfTopNRowNumber::computeLimitOneRowNumber(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
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
      sortKeys_.size() == 1 &&
      nullOrders_[partitionKeys_.size()] == cudf::null_order::AFTER) {
    auto partitionView = input.select(partitionKeys_);
    cudf::groupby::groupby grouper(partitionView, cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = input.column(sortKeys_.front());
    if (columnOrders_[partitionKeys_.size()] == cudf::order::ASCENDING) {
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
    auto probeKeys = input.select(allKeyIndices_);
    cudf::hash_join lookup(
        topKeys->view(),
        cudf::nullable_join::YES,
        cudf::null_equality::EQUAL,
        0.5,
        stream);
    auto joinIndices = lookup.inner_join(probeKeys, std::nullopt, stream, mr);
    auto probeIndices = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*joinIndices.first}};
    auto bestPeers = cudf::gather(
        input,
        probeIndices,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    result = cudf::unique(
        bestPeers->view(),
        partitionKeys_,
        cudf::duplicate_keep_option::KEEP_FIRST,
        cudf::null_equality::EQUAL,
        stream,
        mr);
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

  return std::make_shared<CudfVector>(
      pool(), outputType_, result->num_rows(), std::move(result), stream);
}

CudfVectorPtr CudfTopNRowNumber::computeLimitOneRankLike(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK(!sortKeys_.empty(), "Rank-like TopNRowNumber requires sort keys");

  std::unique_ptr<cudf::table> result;
  if (input.num_rows() == 0) {
    result = std::make_unique<cudf::table>(input, stream, mr);
  } else if (
      !partitionKeys_.empty() && sortKeys_.size() == 1 &&
      nullOrders_[partitionKeys_.size()] == cudf::null_order::AFTER) {
    // Fast grouped Top-1 for the common single scalar order key. A full sort
    // is unnecessary: compute each partition's best key, then join that key
    // back to the input. The inner join intentionally preserves every peer of
    // the best key, which is exactly rank/dense_rank limit=1 semantics.
    auto partitionView = input.select(partitionKeys_);
    cudf::groupby::groupby grouper(partitionView, cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = input.column(sortKeys_.front());
    if (columnOrders_[partitionKeys_.size()] == cudf::order::ASCENDING) {
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
    auto probeKeys = input.select(allKeyIndices_);
    cudf::hash_join lookup(
        topKeys->view(),
        cudf::nullable_join::YES,
        cudf::null_equality::EQUAL,
        0.5,
        stream);
    auto joinIndices = lookup.inner_join(probeKeys, std::nullopt, stream, mr);
    auto probeIndices = cudf::column_view{
        cudf::device_span<cudf::size_type const>{*joinIndices.first}};
    result = cudf::gather(
        input,
        probeIndices,
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
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

  return std::make_shared<CudfVector>(
      pool(), outputType_, result->num_rows(), std::move(result), stream);
}

} // namespace facebook::velox::cudf_velox
