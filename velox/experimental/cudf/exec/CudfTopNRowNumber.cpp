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
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/merge.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/reduction.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>

#include <malloc.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kSortedRunBytes = 3ULL << 30;
constexpr uint64_t kDefaultCandidateRunBytes = 128ULL << 20;
constexpr uint64_t kMinDeviceReserveBytes = 2ULL << 30;
constexpr uint64_t kWorkspaceFixedBytes = 256ULL << 20;
constexpr uint64_t kReductionWorkspaceMultiplier = 8;
constexpr uint64_t kBatchCandidateWorkspaceMultiplier = 16;
constexpr uint64_t kMinPressureReductionInputBytes = 1ULL << 20;
constexpr uint64_t kBatchCandidateCoalesceBytes = 32ULL << 20;
constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
constexpr size_t kMergeFanIn = 4;
constexpr cudf::size_type kMaxCompleteOutputRows = 262144;
constexpr std::string_view kConditionalTopNMarker = "__gluten_mpp_topn_active";
std::atomic<uint64_t> spillDirectorySequence{0};

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

uint64_t estimatedWorkspaceBytes(uint64_t inputBytes, uint64_t multiplier) {
  const auto maxBeforeMultiply =
      (std::numeric_limits<uint64_t>::max() - kWorkspaceFixedBytes) /
      multiplier;
  return inputBytes > maxBeforeMultiply
      ? std::numeric_limits<uint64_t>::max()
      : inputBytes * multiplier + kWorkspaceFixedBytes;
}

uint64_t admissionCapacity(const DeviceAllocationHeadroom& headroom) {
  const auto allocatableBytes = headroom.allocatableBytes();
  const auto reserveBytes = std::max<uint64_t>(
      kMinDeviceReserveBytes, static_cast<uint64_t>(headroom.totalBytes) / 8);
  return allocatableBytes > reserveBytes ? allocatableBytes - reserveBytes
                                         : allocatableBytes / 2;
}

std::optional<DeviceMemoryAdmissionReservation> acquireWorkspaceAdmission(
    uint64_t projectedBytes,
    uint64_t multiplier) {
  const auto headroom = captureDeviceAllocationHeadroom();
  if (!headroom.cudaValid || headroom.totalBytes == 0) {
    return std::nullopt;
  }
  return tryAcquireDeviceMemoryAdmission(
      headroom.device,
      estimatedWorkspaceBytes(projectedBytes, multiplier),
      admissionCapacity(headroom));
}

void deferWorkspaceAdmission(
    std::optional<DeviceMemoryAdmissionReservation>& admission,
    rmm::cuda_stream_view stream) {
  if (!admission) {
    return;
  }
  releaseDeviceMemoryAdmissionAfterStream(std::move(*admission), stream);
  admission.reset();
}

void diagnosticSynchronize(
    std::string_view state,
    rmm::cuda_stream_view stream) {
  if (!deviceMemoryDiagnosticsEnabled()) {
    return;
  }
  LOG(WARNING) << "CUDF_TOPN_DIAGNOSTIC_SYNC state=" << state;
  stream.synchronize();
}

std::unique_ptr<cudf::table> filterRowsMatchingKeys(
    cudf::table_view input,
    cudf::table_view probeKeys,
    cudf::table_view matchingKeys,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  cudf::filtered_join lookup(matchingKeys, cudf::null_equality::EQUAL, stream);
  auto probeIndices = lookup.semi_join(probeKeys, stream, mr);
  // filtered_join owns device state used by asynchronous probe work. Complete
  // the probe before the local lookup is destroyed.
  stream.synchronize();
  auto probeIndexView = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*probeIndices}};
  auto result = cudf::gather(
      input,
      probeIndexView,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
  diagnosticSynchronize("membership_gather", stream);
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

bool CudfTopNRowNumber::hasConditionalPassthrough(
    const std::shared_ptr<const core::TopNRowNumberNode>& node) {
  if (!node) {
    return false;
  }
  if (node->conditionalPassthroughKey().has_value()) {
    return true;
  }
  const auto& inputType = node->inputType();
  for (const auto& key : node->partitionKeys()) {
    const auto channel = exec::exprToChannel(key.get(), inputType);
    if (channel == kConstantChannel) {
      continue;
    }
    const auto& keyName = inputType->nameOf(channel);
    if (keyName.compare(
            0, kConditionalTopNMarker.size(), kConditionalTopNMarker) == 0) {
      return true;
    }
  }
  return false;
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
      emitBatchCandidates_(node->emitBatchCandidates()),
      inputType_(node->inputType()),
      diagnosticNodeId_(node->id()),
      admissionScope_(driverCtx->task->uuid()),
      stateStream_(cudfGlobalStreamPool().get_stream()),
      candidateRunBytes_(driverCtx->queryConfig().get<uint64_t>(
          CudfConfig::kCudfTopNRowNumberCandidateRunBytes,
          kDefaultCandidateRunBytes)),
      forceSpill_(driverCtx->queryConfig().get<bool>(
          CudfConfig::kCudfTopNRowNumberForceSpill,
          false)) {
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
  VELOX_CHECK(
      !emitBatchCandidates_ || !generateRowNumber_,
      "Batch candidate TopNRowNumber cannot generate a rank column");

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
  if (node->conditionalPassthroughKey().has_value()) {
    passthroughKey_ = *node->conditionalPassthroughKey();
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
  input.reset();
  prepareInputForStateStream(cudfInput);
  diagnosticSynchronize("input_ready", stateStream_);
  const auto inputBytes = cudfInput->estimateFlatSize();

  auto conditionalInputMode = ConditionalInputMode::kNone;
  if (passthroughKey_.has_value()) {
    conditionalInputRows_ += cudfInput->size();
    const auto inputView = cudfInput->getTableView();
    const auto activeMask = inputView.column(*passthroughKey_);
    VELOX_CHECK(
        activeMask.type().id() == cudf::type_id::BOOL8,
        "Conditional TopNRowNumber marker must be BOOL8");

    auto allActiveScalar = cudf::reduce(
        activeMask,
        *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
        cudf::data_type(cudf::type_id::BOOL8),
        stateStream_,
        get_temp_mr());
    diagnosticSynchronize("conditional_all", stateStream_);
    auto* allActive =
        static_cast<cudf::scalar_type_t<bool>*>(allActiveScalar.get());
    if (allActive->is_valid(stateStream_) && allActive->value(stateStream_)) {
      conditionalInputMode = ConditionalInputMode::kAllActive;
    } else {
      auto anyActiveScalar = cudf::reduce(
          activeMask,
          *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
          cudf::data_type(cudf::type_id::BOOL8),
          stateStream_,
          get_temp_mr());
      diagnosticSynchronize("conditional_any", stateStream_);
      auto* anyActive =
          static_cast<cudf::scalar_type_t<bool>*>(anyActiveScalar.get());
      if (!anyActive->is_valid(stateStream_) ||
          !anyActive->value(stateStream_)) {
        conditionalPassthroughRows_ += cudfInput->size();
        ++conditionalPassthroughAdmissionBypasses_;
        passthroughOutputs_.push_back(std::move(cudfInput));
        return;
      }
      conditionalInputMode = ConditionalInputMode::kMixed;
    }
  }

  std::optional<DeviceMemoryAdmissionReservation> batchCandidateAdmission;
  uint64_t batchCandidateTaskScopedCreditBytes = 0;
  if (emitBatchCandidates_) {
    bool admitted = false;
    const auto headroom = captureDeviceAllocationHeadroom();
    if (headroom.cudaValid && headroom.totalBytes > 0) {
      const auto capacity = admissionCapacity(headroom);
      const auto requestedBytes = std::min<uint64_t>(
          estimatedWorkspaceBytes(
              inputBytes, kBatchCandidateWorkspaceMultiplier),
          capacity);
      if (!taskScopedAdmissionCreditClaimed_) {
        batchCandidateTaskScopedCreditBytes =
            scopedDeviceMemoryAdmissionCreditBytes(
                admissionScope_, headroom.device);
      }
      const auto additionalBytes =
          requestedBytes > batchCandidateTaskScopedCreditBytes
          ? requestedBytes - batchCandidateTaskScopedCreditBytes
          : 0;
      if (additionalBytes == 0) {
        admitted = true;
      } else {
        batchCandidateAdmission = tryAcquireDeviceMemoryAdmission(
            headroom.device, additionalBytes, capacity);
        admitted = batchCandidateAdmission.has_value();
      }
    } else {
      batchCandidateAdmission = acquireWorkspaceAdmission(
          inputBytes, kBatchCandidateWorkspaceMultiplier);
      admitted = batchCandidateAdmission.has_value();
    }
    if (!admitted) {
      ++pressureBypassBatches_;
      pressureBypassRows_ += cudfInput->size();
      pressureBypassBytes_ += inputBytes;
      passthroughOutputs_.push_back(std::move(cudfInput));
      return;
    }
    if (batchCandidateTaskScopedCreditBytes > 0) {
      taskScopedAdmissionCreditClaimed_ = true;
    }
  }

  std::optional<DeviceMemoryAdmissionReservation> conditionalAdmission;
  std::vector<DeviceMemoryAdmissionCreditPtr> inputAdmissionCredits;
  uint64_t inputAdmissionCreditBytes = 0;
  if (passthroughKey_.has_value() && !emitBatchCandidates_) {
    const auto headroom = captureDeviceAllocationHeadroom();
    if (headroom.cudaValid && headroom.totalBytes > 0) {
      const auto capacity = admissionCapacity(headroom);
      if (capacity > 0) {
        const auto requestedBytes = std::min<uint64_t>(
            estimatedWorkspaceBytes(inputBytes, kReductionWorkspaceMultiplier),
            capacity);
        uint64_t availableCreditBytes = 0;
        for (const auto& credit : cudfInput->deviceMemoryAdmissionCredits()) {
          if (credit->device() == headroom.device &&
              credit->availableBytes() > 0) {
            inputAdmissionCredits.push_back(credit);
            availableCreditBytes += credit->availableBytes();
          }
        }
        inputAdmissionCreditBytes =
            std::min<uint64_t>(requestedBytes, availableCreditBytes);
        scopedAdmissionCreditBytes_ += inputAdmissionCreditBytes;
        const auto additionalBytes = requestedBytes - inputAdmissionCreditBytes;
        if (additionalBytes > 0) {
          conditionalAdmission = tryAcquireDeviceMemoryAdmission(
              headroom.device, additionalBytes, capacity);
          if (!conditionalAdmission) {
            pendingInput_ = std::move(cudfInput);
            pendingAdmissionDevice_ = headroom.device;
            pendingAdmissionBytes_ = additionalBytes;
            pendingAdmissionCapacity_ = capacity;
            pendingConditionalInputMode_ = conditionalInputMode;
            pendingInputAdmissionCredits_ = std::move(inputAdmissionCredits);
            pendingInputAdmissionCreditBytes_ = inputAdmissionCreditBytes;
            pendingAdmissionStart_ = std::chrono::steady_clock::now();
            ++conditionalBlockingAdmissions_;
            return;
          }
        }
      }
    }
  }

  processInput(
      std::move(cudfInput),
      std::move(batchCandidateAdmission),
      std::move(conditionalAdmission),
      conditionalInputMode,
      std::move(inputAdmissionCredits),
      inputAdmissionCreditBytes,
      batchCandidateTaskScopedCreditBytes);
}

exec::BlockingReason CudfTopNRowNumber::isBlocked(ContinueFuture* future) {
  if (!pendingInput_) {
    return exec::BlockingReason::kNotBlocked;
  }
  if (pendingAdmission_) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto admission = acquireDeviceMemoryAdmissionOrFuture(
      pendingAdmissionDevice_,
      pendingAdmissionBytes_,
      pendingAdmissionCapacity_,
      future);
  if (!admission) {
    return exec::BlockingReason::kWaitForMemory;
  }

  conditionalAdmissionWaitNanos_ +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - pendingAdmissionStart_)
          .count();
  pendingAdmission_ = std::move(admission);
  return exec::BlockingReason::kNotBlocked;
}

void CudfTopNRowNumber::processInput(
    CudfVectorPtr cudfInput,
    std::optional<DeviceMemoryAdmissionReservation> batchCandidateAdmission,
    std::optional<DeviceMemoryAdmissionReservation> conditionalAdmission,
    ConditionalInputMode conditionalInputMode,
    std::vector<DeviceMemoryAdmissionCreditPtr> inputAdmissionCredits,
    uint64_t inputAdmissionCreditBytes,
    uint64_t batchCandidateTaskScopedCreditBytes) {
  std::vector<DeviceMemoryAdmissionCreditPtr> batchCandidateAdmissionCredits;
  if (emitBatchCandidates_) {
    for (const auto& credit : cudfInput->deviceMemoryAdmissionCredits()) {
      if (credit->availableBytes() == 0) {
        continue;
      }
      const auto duplicate = std::find_if(
          batchCandidateAdmissionCredits.begin(),
          batchCandidateAdmissionCredits.end(),
          [&](const auto& existing) { return existing.get() == credit.get(); });
      if (duplicate == batchCandidateAdmissionCredits.end()) {
        batchCandidateAdmissionCredits.push_back(credit);
      }
    }
  }

  CudfVectorPtr conditionalInput;
  if (conditionalInputMode == ConditionalInputMode::kAllActive) {
    conditionalActiveRows_ += cudfInput->size();
  } else if (conditionalInputMode == ConditionalInputMode::kMixed) {
    VELOX_CHECK(passthroughKey_.has_value());
    conditionalInput = cudfInput;
    const auto stream = stateStream_;
    const auto inputView = cudfInput->getTableView();
    const auto activeMask = inputView.column(*passthroughKey_);
    auto active = cudf::apply_boolean_mask(
        inputView, activeMask, stream, get_output_mr());
    diagnosticSynchronize("conditional_active_rows", stream);
    conditionalActiveRows_ += active->num_rows();
    cudfInput = std::make_shared<CudfVector>(
        pool(), inputType_, active->num_rows(), std::move(active), stream);
  }

  const auto stream = stateStream_;
  const auto mr = get_output_mr();
  bool shouldFlushBatchCandidateInputs = false;
  if (emitBatchCandidates_) {
    ++batchCandidateInputBatches_;
    const auto inputBytes = cudfInput->estimateFlatSize();
    pendingBatchCandidateInputBytes_ = pendingBatchCandidateInputBytes_ >
            std::numeric_limits<uint64_t>::max() - inputBytes
        ? std::numeric_limits<uint64_t>::max()
        : pendingBatchCandidateInputBytes_ + inputBytes;
    pendingBatchCandidateInputs_.push_back(std::move(cudfInput));
    for (auto& credit : batchCandidateAdmissionCredits) {
      const auto duplicate = std::find_if(
          pendingBatchCandidateAdmissionCredits_.begin(),
          pendingBatchCandidateAdmissionCredits_.end(),
          [&](const auto& existing) { return existing.get() == credit.get(); });
      if (duplicate == pendingBatchCandidateAdmissionCredits_.end()) {
        pendingBatchCandidateAdmissionCredits_.push_back(std::move(credit));
      }
    }
    pendingBatchCandidateTaskScopedCreditBytes_ +=
        batchCandidateTaskScopedCreditBytes;
    shouldFlushBatchCandidateInputs =
        pendingBatchCandidateInputBytes_ >= kBatchCandidateCoalesceBytes;
  } else {
    std::vector<std::unique_ptr<cudf::table>> reducedBatches;
    if (conditionalAdmission || inputAdmissionCreditBytes > 0) {
      reducedBatches.push_back(
          reduceToCandidates(cudfInput->getTableView(), stream, mr));
    } else {
      reducedBatches = reduceToCandidatesBounded(cudfInput, stream, mr);
    }
    for (auto& batchCandidates : reducedBatches) {
      mergeBatchCandidates(std::move(batchCandidates), stream, mr);
    }
    cudfInput.reset();
  }

  if (conditionalInput) {
    const auto inputView = conditionalInput->getTableView();
    const auto activeMask = inputView.column(*passthroughKey_);
    auto inactiveMask = cudf::unary_operation(
        activeMask, cudf::unary_operator::NOT, stream, get_temp_mr());
    auto inactive = cudf::apply_boolean_mask(
        inputView, inactiveMask->view(), stream, get_output_mr());
    if (inactive->num_rows() > 0) {
      conditionalPassthroughRows_ += inactive->num_rows();
      passthroughOutputs_.push_back(std::make_shared<CudfVector>(
          pool(),
          inputType_,
          inactive->num_rows(),
          std::move(inactive),
          stream));
    }
    diagnosticSynchronize("conditional_outputs", stream);
  }
  deferWorkspaceAdmission(batchCandidateAdmission, stream);
  deferWorkspaceAdmission(conditionalAdmission, stream);
  if (inputAdmissionCreditBytes > 0) {
    uint64_t consumedBytes = 0;
    for (const auto& credit : inputAdmissionCredits) {
      consumedBytes +=
          consumeDeviceMemoryAdmissionCreditAfterStream(credit, stream);
      if (consumedBytes >= inputAdmissionCreditBytes) {
        break;
      }
    }
    VELOX_CHECK_GE(
        consumedBytes,
        inputAdmissionCreditBytes,
        "Input admission credit disappeared before TopN consumed it");
  }
  if (shouldFlushBatchCandidateInputs) {
    this->flushBatchCandidateInputs(stream, mr);
  }
}

void CudfTopNRowNumber::prepareInputForStateStream(const CudfVectorPtr& input) {
  const auto inputStream = input->stream();
  if (inputStream.value() != stateStream_.value()) {
    cudf::detail::join_streams(
        std::vector<rmm::cuda_stream_view>{inputStream}, stateStream_);
  }
  VELOX_CHECK(
      input->rebindStream(stateStream_),
      "CudfTopNRowNumber cannot rebind its input to the state stream");
}

std::vector<std::unique_ptr<cudf::table>>
CudfTopNRowNumber::reduceToCandidatesBounded(
    const CudfVectorPtr& input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto inputBytes = input->estimateFlatSize();
  const auto headroom = captureDeviceAllocationHeadroom();
  if (!headroom.cudaValid || headroom.totalBytes == 0) {
    std::vector<std::unique_ptr<cudf::table>> result;
    result.push_back(reduceToCandidates(input->getTableView(), stream, mr));
    return result;
  }

  const auto capacity = admissionCapacity(headroom);
  auto admission = tryAcquireDeviceMemoryAdmission(
      headroom.device,
      estimatedWorkspaceBytes(inputBytes, kReductionWorkspaceMultiplier),
      capacity);
  if (admission) {
    std::vector<std::unique_ptr<cudf::table>> result;
    result.push_back(reduceToCandidates(input->getTableView(), stream, mr));
    deferWorkspaceAdmission(admission, stream);
    return result;
  }

  const auto pressureBudget = capacity / 2;
  const auto targetInputBytes = std::max<uint64_t>(
      kMinPressureReductionInputBytes,
      pressureBudget > kWorkspaceFixedBytes
          ? (pressureBudget - kWorkspaceFixedBytes) /
              kReductionWorkspaceMultiplier
          : kMinPressureReductionInputBytes);
  const auto numRows = static_cast<uint64_t>(input->size());
  auto targetRows = std::max<uint64_t>(
      1,
      inputBytes == 0 ? numRows
                      : std::min<uint64_t>(
                            numRows,
                            static_cast<unsigned __int128>(targetInputBytes) *
                                numRows / inputBytes));
  if (numRows > 1) {
    targetRows = std::min<uint64_t>(targetRows, (numRows + 1) / 2);
  }

  ++pressureSplitBatches_;
  std::vector<std::unique_ptr<cudf::table>> result;
  for (uint64_t begin = 0; begin < numRows; begin += targetRows) {
    const auto end = std::min<uint64_t>(numRows, begin + targetRows);
    auto slices = cudf::slice(
        input->getTableView(),
        {static_cast<cudf::size_type>(begin),
         static_cast<cudf::size_type>(end)},
        stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    const auto sliceBytes = inputBytes == 0
        ? 0
        : static_cast<uint64_t>(
              static_cast<unsigned __int128>(inputBytes) * (end - begin) /
              numRows);
    const auto requestedBytes =
        estimatedWorkspaceBytes(sliceBytes, kReductionWorkspaceMultiplier);
    const auto waitStart = std::chrono::steady_clock::now();
    auto sliceAdmission = waitAcquireDeviceMemoryAdmission(
        headroom.device, requestedBytes, capacity);
    pressureAdmissionWaitNanos_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - waitStart)
            .count();
    ++pressureBlockingAdmissions_;
    VELOX_CHECK(
        sliceAdmission.has_value(),
        "TopNRowNumber pressure chunk requires {} bytes but admission "
        "capacity is only {} bytes",
        requestedBytes,
        capacity);
    result.push_back(reduceToCandidates(slices.front(), stream, mr));
    deferWorkspaceAdmission(sliceAdmission, stream);
    ++pressureSplitChunks_;
  }
  return result;
}

void CudfTopNRowNumber::mergeBatchCandidates(
    std::unique_ptr<cudf::table> batchCandidates,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto batchCandidateBytes =
      estimateCandidateBytes(batchCandidates, stream);
  addCandidateRun(std::move(batchCandidates), batchCandidateBytes, stream, mr);
}

void CudfTopNRowNumber::flushBatchCandidateInputs(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (pendingBatchCandidateInputs_.empty()) {
    return;
  }

  auto flushInputs = std::exchange(pendingBatchCandidateInputs_, {});
  auto flushAdmissionCredits =
      std::exchange(pendingBatchCandidateAdmissionCredits_, {});
  const auto flushTaskScopedCreditBytes =
      std::exchange(pendingBatchCandidateTaskScopedCreditBytes_, uint64_t{0});
  const auto flushBytes =
      std::exchange(pendingBatchCandidateInputBytes_, uint64_t{0});
  const auto headroom = captureDeviceAllocationHeadroom();
  std::optional<DeviceMemoryAdmissionReservation> flushAdmission;
  bool admitted = false;
  if (headroom.cudaValid && headroom.totalBytes > 0) {
    const auto capacity = admissionCapacity(headroom);
    const auto requestedBytes = std::min<uint64_t>(
        estimatedWorkspaceBytes(flushBytes, kBatchCandidateWorkspaceMultiplier),
        capacity);
    const auto additionalBytes = requestedBytes > flushTaskScopedCreditBytes
        ? requestedBytes - flushTaskScopedCreditBytes
        : 0;
    if (additionalBytes == 0) {
      admitted = true;
    } else {
      flushAdmission = tryAcquireDeviceMemoryAdmission(
          headroom.device, additionalBytes, capacity);
      admitted = flushAdmission.has_value();
    }
  } else {
    flushAdmission = acquireWorkspaceAdmission(
        flushBytes, kBatchCandidateWorkspaceMultiplier);
    admitted = flushAdmission.has_value();
  }
  if (!admitted) {
    if (flushTaskScopedCreditBytes > 0) {
      taskScopedAdmissionCreditClaimed_ = false;
    }
    for (auto& input : flushInputs) {
      ++pressureBypassBatches_;
      pressureBypassRows_ += input->size();
      pressureBypassBytes_ += input->estimateFlatSize();
      passthroughOutputs_.push_back(std::move(input));
    }
    return;
  }

  std::vector<cudf::table_view> pieces;
  pieces.reserve(flushInputs.size());
  for (const auto& input : flushInputs) {
    pieces.push_back(input->getTableView());
  }
  std::unique_ptr<cudf::table> combined;
  if (pieces.size() == 1) {
    combined = flushInputs.front()->release();
  } else {
    combined = cudf::concatenate(pieces, stream, mr);
  }
  flushInputs.clear();

  auto candidates = reduceToCandidates(combined->view(), stream, mr);
  auto output = std::make_shared<CudfVector>(
      pool(),
      inputType_,
      candidates->num_rows(),
      std::move(candidates),
      stream);
  if (output->size() > 0) {
    ++batchCandidateBatches_;
    ++batchCandidateFlushes_;
    batchCandidateRows_ += output->size();
    batchCandidateBytes_ += output->estimateFlatSize();
    passthroughOutputs_.push_back(std::move(output));
  }
  for (const auto& credit : flushAdmissionCredits) {
    scopedAdmissionCreditBytes_ +=
        consumeDeviceMemoryAdmissionCreditAfterStream(credit, stream);
  }
  if (flushTaskScopedCreditBytes > 0) {
    taskScopedAdmissionCreditBytes_ +=
        consumeScopedDeviceMemoryAdmissionCreditAfterStream(
            admissionScope_,
            headroom.device,
            flushTaskScopedCreditBytes,
            stream);
  }
  deferWorkspaceAdmission(flushAdmission, stream);
}

uint64_t CudfTopNRowNumber::estimateCandidateBytes(
    std::unique_ptr<cudf::table>& candidateRun,
    rmm::cuda_stream_view stream) {
  auto vector = std::make_shared<CudfVector>(
      pool(),
      inputType_,
      candidateRun->num_rows(),
      std::move(candidateRun),
      stream);
  const auto bytes = vector->estimateFlatSize();
  candidateRun = vector->release();
  return bytes;
}

void CudfTopNRowNumber::updateCandidateStatePeak() {
  peakCandidateRows_ = std::max(peakCandidateRows_, candidateRows_);
  peakCandidateBytes_ = std::max(peakCandidateBytes_, candidateBytes_);
}

void CudfTopNRowNumber::spillCandidateRun(
    std::unique_ptr<cudf::table> candidateRun,
    uint64_t candidateRunBytes,
    rmm::cuda_stream_view stream) {
  if (!candidateRun || candidateRun->num_rows() == 0) {
    return;
  }
  inputs_.push_back(std::make_shared<CudfVector>(
      pool(),
      inputType_,
      candidateRun->num_rows(),
      std::move(candidateRun),
      stream));
  bufferedBytes_ = candidateRunBytes;
  spillSortedRun();
}

void CudfTopNRowNumber::addCandidateRun(
    std::unique_ptr<cudf::table> candidateRun,
    uint64_t candidateRunBytes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!candidateRun || candidateRun->num_rows() == 0) {
    return;
  }

  uint64_t level = 0;
  while (true) {
    if (level == candidateLevels_.size()) {
      candidateLevels_.emplace_back();
    }
    auto& slot = candidateLevels_[level];
    if (!slot.table) {
      if (level == 0 && candidateRunBytes >= candidateRunBytes_) {
        auto retentionAdmission = forceSpill_
            ? std::optional<DeviceMemoryAdmissionReservation>{}
            : acquireWorkspaceAdmission(candidateRunBytes, 3);
        if (!retentionAdmission) {
          ++pressurePostMergeSpills_;
          spillCandidateRun(std::move(candidateRun), candidateRunBytes, stream);
          return;
        }
        ++pressureRetainedMerges_;
        deferWorkspaceAdmission(retentionAdmission, stream);
      }
      candidateRows_ += candidateRun->num_rows();
      candidateBytes_ = candidateBytes_ >
              std::numeric_limits<uint64_t>::max() - candidateRunBytes
          ? std::numeric_limits<uint64_t>::max()
          : candidateBytes_ + candidateRunBytes;
      slot.table = std::move(candidateRun);
      slot.bytes = candidateRunBytes;
      updateCandidateStatePeak();
      return;
    }

    const auto existingRows = slot.table->num_rows();
    const auto existingBytes = slot.bytes;
    const auto projectedCandidateBytes =
        existingBytes > std::numeric_limits<uint64_t>::max() - candidateRunBytes
        ? std::numeric_limits<uint64_t>::max()
        : existingBytes + candidateRunBytes;
    std::optional<DeviceMemoryAdmissionReservation> mergeAdmission;
    if (projectedCandidateBytes >= candidateRunBytes_ && !forceSpill_) {
      mergeAdmission = acquireWorkspaceAdmission(projectedCandidateBytes, 3);
    }
    if (projectedCandidateBytes >= candidateRunBytes_ && !mergeAdmission) {
      candidateRows_ -= existingRows;
      candidateBytes_ -= existingBytes;
      auto retained = std::move(slot.table);
      slot.bytes = 0;
      ++pressurePreMergeSpills_;
      spillCandidateRun(std::move(retained), existingBytes, stream);
      continue;
    }
    if (mergeAdmission) {
      ++pressureRetainedMerges_;
    }

    candidateRows_ -= existingRows;
    candidateBytes_ -= existingBytes;
    std::vector<cudf::table_view> pieces{
        slot.table->view(), candidateRun->view()};
    auto merged = cudf::concatenate(pieces, stream, mr);
    slot.table.reset();
    slot.bytes = 0;
    candidateRun = reduceToCandidates(merged->view(), stream, mr);
    candidateRunBytes = estimateCandidateBytes(candidateRun, stream);
    ++candidateLevelMerges_;
    deferWorkspaceAdmission(mergeAdmission, stream);
    ++level;
  }
}

void CudfTopNRowNumber::finalizeCandidateLevels(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (candidateLevels_.empty()) {
    return;
  }
  if (spilled_) {
    for (auto& level : candidateLevels_) {
      if (level.table) {
        spillCandidateRun(std::move(level.table), level.bytes, stream);
      }
    }
    candidateLevels_.clear();
    candidateRows_ = 0;
    candidateBytes_ = 0;
    return;
  }

  std::unique_ptr<cudf::table> carry;
  uint64_t carryBytes = 0;
  for (auto levelIt = candidateLevels_.rbegin();
       levelIt != candidateLevels_.rend();
       ++levelIt) {
    auto& level = *levelIt;
    if (!level.table) {
      continue;
    }
    candidateRows_ -= level.table->num_rows();
    candidateBytes_ -= level.bytes;
    if (!carry) {
      carry = std::move(level.table);
      carryBytes = std::exchange(level.bytes, 0);
      continue;
    }

    const auto projectedCandidateBytes =
        carryBytes > std::numeric_limits<uint64_t>::max() - level.bytes
        ? std::numeric_limits<uint64_t>::max()
        : carryBytes + level.bytes;
    auto mergeAdmission = projectedCandidateBytes >= candidateRunBytes_
        ? acquireWorkspaceAdmission(projectedCandidateBytes, 3)
        : std::optional<DeviceMemoryAdmissionReservation>{};
    if (projectedCandidateBytes >= candidateRunBytes_ && !mergeAdmission) {
      ++pressurePreMergeSpills_;
      spillCandidateRun(std::move(carry), carryBytes, stream);
      spillCandidateRun(std::move(level.table), level.bytes, stream);
      level.bytes = 0;
      for (auto& remaining : candidateLevels_) {
        if (remaining.table) {
          spillCandidateRun(
              std::move(remaining.table), remaining.bytes, stream);
          remaining.bytes = 0;
        }
      }
      candidateLevels_.clear();
      candidateRows_ = 0;
      candidateBytes_ = 0;
      return;
    }
    if (mergeAdmission) {
      ++pressureRetainedMerges_;
    }

    std::vector<cudf::table_view> pieces{carry->view(), level.table->view()};
    auto merged = cudf::concatenate(pieces, stream, mr);
    level.table.reset();
    level.bytes = 0;
    carry = reduceToCandidates(merged->view(), stream, mr);
    carryBytes = estimateCandidateBytes(carry, stream);
    ++candidateLevelMerges_;
    deferWorkspaceAdmission(mergeAdmission, stream);
  }
  candidateLevels_.clear();
  candidates_ = std::move(carry);
  candidateBytes_ = carryBytes;
  candidateRows_ = candidates_ ? candidates_->num_rows() : 0;
  updateCandidateStatePeak();
}

void CudfTopNRowNumber::doNoMoreInput() {
  Operator::noMoreInput();
  if (emitBatchCandidates_) {
    flushBatchCandidateInputs(stateStream_, get_output_mr());
    finished_ = true;
    recordRuntimeStats();
    return;
  }
  auto stream = stateStream_;
  auto mr = get_output_mr();
  finalizeCandidateLevels(stream, mr);
  if (spilled_ && candidates_) {
    auto candidateVector = std::make_shared<CudfVector>(
        pool(),
        inputType_,
        candidates_->num_rows(),
        std::move(candidates_),
        stream);
    bufferedBytes_ = candidateVector->estimateFlatSize();
    candidateBytes_ = 0;
    inputs_.push_back(std::move(candidateVector));
    spillSortedRun();
  }
  if (spilled_) {
    compactSortedRunsForMerge();
    initializeSortedRunReaders();
  }
  if (!candidates_ && passthroughOutputs_.empty()) {
    finished_ = !spilled_;
    if (finished_) {
      recordRuntimeStats();
    }
  }
}

RowVectorPtr CudfTopNRowNumber::doGetOutput() {
  if (pendingInput_) {
    VELOX_CHECK(
        pendingAdmission_.has_value(),
        "TopNRowNumber pending input requires device-memory admission");
    auto input = std::exchange(pendingInput_, nullptr);
    pendingAdmissionDevice_ = -1;
    pendingAdmissionBytes_ = 0;
    pendingAdmissionCapacity_ = 0;
    auto inputAdmissionCredits =
        std::exchange(pendingInputAdmissionCredits_, {});
    const auto inputAdmissionCreditBytes =
        std::exchange(pendingInputAdmissionCreditBytes_, 0);
    const auto conditionalInputMode = std::exchange(
        pendingConditionalInputMode_, ConditionalInputMode::kNone);
    processInput(
        std::move(input),
        std::nullopt,
        std::move(pendingAdmission_),
        conditionalInputMode,
        std::move(inputAdmissionCredits),
        inputAdmissionCreditBytes,
        0);
    pendingAdmission_.reset();
  }

  if (!passthroughOutputs_.empty()) {
    auto output = std::move(passthroughOutputs_.front());
    passthroughOutputs_.pop_front();
    return output;
  }

  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (spilled_) {
    auto result = computeNextSortedOutput();
    if (result != nullptr) {
      return result;
    }
    finished_ = true;
    cleanupSpillFiles();
    recordRuntimeStats();
    return nullptr;
  }

  if (!candidates_) {
    finished_ = true;
    recordRuntimeStats();
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
  recordRuntimeStats();
  return result;
}

std::unique_ptr<cudf::table> CudfTopNRowNumber::reduceToCandidates(
    cudf::table_view input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  ++candidateReductionCalls_;
  candidateRowsReduced_ += input.num_rows();
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

void CudfTopNRowNumber::spillSortedRun() {
  if (inputs_.empty()) {
    return;
  }

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

  auto stream = stateStream_;
  auto mr = get_output_mr();
  logDeviceMemorySnapshot(fmt::format(
      "operator=CudfTopNRowNumber node={} state=sortRun.concatenate.begin "
      "bufferedBytes={} bufferedInputs={}",
      diagnosticNodeId_,
      bufferedBytes_,
      inputs_.size()));
  auto input =
      getConcatenatedTable(std::exchange(inputs_, {}), inputType_, stream, mr);
  bufferedBytes_ = 0;

  logDeviceMemorySnapshot(fmt::format(
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
  logDeviceMemorySnapshot(fmt::format(
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
  spillBytes_ += input->num_rows() == 0
      ? 0
      : std::filesystem::file_size(sortedRuns_.back().path);
  ::malloc_trim(0);
}

void CudfTopNRowNumber::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  auto stream = stateStream_;
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
  auto stream = stateStream_;
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
        readers.push_back(std::make_unique<cudf::io::chunked_parquet_reader>(
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
  auto stream = stateStream_;
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

void CudfTopNRowNumber::recordRuntimeStats() {
  if (runtimeStatsRecorded_) {
    return;
  }
  runtimeStatsRecorded_ = true;
  auto lockedStats = stats_.wlock();
  lockedStats->addRuntimeStat(
      "topNPressureRetainedMerges", RuntimeCounter(pressureRetainedMerges_));
  lockedStats->addRuntimeStat(
      "topNPressurePreMergeSpills", RuntimeCounter(pressurePreMergeSpills_));
  lockedStats->addRuntimeStat(
      "topNPressurePostMergeSpills", RuntimeCounter(pressurePostMergeSpills_));
  lockedStats->addRuntimeStat("topNSpillRuns", RuntimeCounter(spillRuns_));
  lockedStats->addRuntimeStat("topNSpillBytes", RuntimeCounter(spillBytes_));
  lockedStats->addRuntimeStat(
      "topNBatchCandidateBatches", RuntimeCounter(batchCandidateBatches_));
  lockedStats->addRuntimeStat(
      "topNBatchCandidateInputBatches",
      RuntimeCounter(batchCandidateInputBatches_));
  lockedStats->addRuntimeStat(
      "topNBatchCandidateFlushes", RuntimeCounter(batchCandidateFlushes_));
  lockedStats->addRuntimeStat(
      "topNBatchCandidateRows", RuntimeCounter(batchCandidateRows_));
  lockedStats->addRuntimeStat(
      "topNBatchCandidateBytes", RuntimeCounter(batchCandidateBytes_));
  lockedStats->addRuntimeStat(
      "topNPressureSplitBatches", RuntimeCounter(pressureSplitBatches_));
  lockedStats->addRuntimeStat(
      "topNPressureSplitChunks", RuntimeCounter(pressureSplitChunks_));
  lockedStats->addRuntimeStat(
      "topNPressureBlockingAdmissions",
      RuntimeCounter(pressureBlockingAdmissions_));
  lockedStats->addRuntimeStat(
      "topNPressureAdmissionWaitNanos",
      RuntimeCounter(pressureAdmissionWaitNanos_));
  lockedStats->addRuntimeStat(
      "topNConditionalBlockingAdmissions",
      RuntimeCounter(conditionalBlockingAdmissions_));
  lockedStats->addRuntimeStat(
      "topNConditionalAdmissionWaitNanos",
      RuntimeCounter(conditionalAdmissionWaitNanos_));
  lockedStats->addRuntimeStat(
      "topNScopedAdmissionCreditBytes",
      RuntimeCounter(scopedAdmissionCreditBytes_));
  lockedStats->addRuntimeStat(
      "topNTaskScopedAdmissionCreditBytes",
      RuntimeCounter(taskScopedAdmissionCreditBytes_));
  lockedStats->addRuntimeStat(
      "topNPeakCandidateRows", RuntimeCounter(peakCandidateRows_));
  lockedStats->addRuntimeStat(
      "topNPeakCandidateBytes", RuntimeCounter(peakCandidateBytes_));
  lockedStats->addRuntimeStat(
      "topNCandidateLevelMerges", RuntimeCounter(candidateLevelMerges_));
  lockedStats->addRuntimeStat(
      "topNCandidateReductionCalls", RuntimeCounter(candidateReductionCalls_));
  lockedStats->addRuntimeStat(
      "topNCandidateRowsReduced", RuntimeCounter(candidateRowsReduced_));
  lockedStats->addRuntimeStat(
      "topNRankMembershipFilterCalls",
      RuntimeCounter(rankMembershipFilterCalls_));
  lockedStats->addRuntimeStat(
      "topNConditionalInputRows", RuntimeCounter(conditionalInputRows_));
  lockedStats->addRuntimeStat(
      "topNConditionalActiveRows", RuntimeCounter(conditionalActiveRows_));
  lockedStats->addRuntimeStat(
      "topNConditionalPassthroughRows",
      RuntimeCounter(conditionalPassthroughRows_));
  lockedStats->addRuntimeStat(
      "topNConditionalPassthroughAdmissionBypasses",
      RuntimeCounter(conditionalPassthroughAdmissionBypasses_));
  lockedStats->addRuntimeStat(
      "topNPressureBypassBatches", RuntimeCounter(pressureBypassBatches_));
  lockedStats->addRuntimeStat(
      "topNPressureBypassRows", RuntimeCounter(pressureBypassRows_));
  lockedStats->addRuntimeStat(
      "topNPressureBypassBytes", RuntimeCounter(pressureBypassBytes_));
}

void CudfTopNRowNumber::doClose() {
  pendingInput_.reset();
  pendingAdmission_.reset();
  pendingAdmissionDevice_ = -1;
  pendingAdmissionBytes_ = 0;
  pendingAdmissionCapacity_ = 0;
  pendingConditionalInputMode_ = ConditionalInputMode::kNone;
  pendingInputAdmissionCredits_.clear();
  pendingInputAdmissionCreditBytes_ = 0;
  inputs_.clear();
  candidateLevels_.clear();
  pendingBatchCandidateInputs_.clear();
  pendingBatchCandidateAdmissionCredits_.clear();
  pendingBatchCandidateTaskScopedCreditBytes_ = 0;
  pendingBatchCandidateInputBytes_ = 0;
  candidateRows_ = 0;
  candidates_.reset();
  candidateBytes_ = 0;
  passthroughOutputs_.clear();
  cleanupSpillFiles();
  recordRuntimeStats();
  Operator::close();
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
    diagnosticSynchronize("row_number_groupby", stream);
    VELOX_CHECK_EQ(aggregateResults.size(), 1);
    VELOX_CHECK_EQ(aggregateResults[0].results.size(), 1);
    auto topKeyColumns = groupKeys->release();
    topKeyColumns.push_back(std::move(aggregateResults[0].results[0]));
    auto topKeys = std::make_unique<cudf::table>(std::move(topKeyColumns));
    auto probeKeys = input.select(allKeyIndices_);
    auto bestPeers =
        filterRowsMatchingKeys(input, probeKeys, topKeys->view(), stream, mr);
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
    diagnosticSynchronize("rank_like_groupby", stream);
    VELOX_CHECK_EQ(aggregateResults.size(), 1);
    VELOX_CHECK_EQ(aggregateResults[0].results.size(), 1);
    auto topKeyColumns = groupKeys->release();
    topKeyColumns.push_back(std::move(aggregateResults[0].results[0]));
    auto topKeys = std::make_unique<cudf::table>(std::move(topKeyColumns));
    auto probeKeys = input.select(allKeyIndices_);
    ++rankMembershipFilterCalls_;
    result =
        filterRowsMatchingKeys(input, probeKeys, topKeys->view(), stream, mr);
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
    ++rankMembershipFilterCalls_;
    result = filterRowsMatchingKeys(
        sortedTable->view(), probeKeyView, topKeyView, stream, mr);
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
