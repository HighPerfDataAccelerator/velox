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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"

#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

/// GPU TopNRowNumber for limit=1 rank-like windows.
///
/// row_number keeps the single first row in each partition. rank and dense_rank
/// keep every row in the first peer group in each partition.
class CudfTopNRowNumber : public CudfOperatorBase {
 public:
  CudfTopNRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

  bool needsInput() const override {
    return !noMoreInput_ && !pendingInput_ && passthroughOutputs_.empty();
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return finished_ && !pendingInput_ && passthroughOutputs_.empty();
  }

  static bool shouldReplace(
      const std::shared_ptr<const core::TopNRowNumberNode>& node);
  static bool hasConditionalPassthrough(
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  void spillSortedRun();
  void compactSortedRunsForMerge();
  void initializeSortedRunReaders();
  std::unique_ptr<cudf::table> mergeNextSortedBatch(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool& finalBatch);
  std::unique_ptr<cudf::table> takeCompletePartitions(
      std::unique_ptr<cudf::table> sorted,
      bool finalBatch,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr computeNextSortedOutput();
  void cleanupSpillFiles();
  void recordRuntimeStats();

  CudfVectorPtr computeLimitOneRowNumber(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr computeLimitOneRankLike(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  std::unique_ptr<cudf::table> reduceToCandidates(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  std::vector<std::unique_ptr<cudf::table>> reduceToCandidatesBounded(
      const CudfVectorPtr& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void prepareInputForStateStream(const CudfVectorPtr& input);
  enum class ConditionalInputMode {
    kNone,
    kAllActive,
    kMixed,
  };
  void processInput(
      CudfVectorPtr input,
      std::optional<DeviceMemoryAdmissionReservation> batchCandidateAdmission,
      std::optional<DeviceMemoryAdmissionReservation> conditionalAdmission,
      ConditionalInputMode conditionalInputMode,
      std::vector<DeviceMemoryAdmissionCreditPtr> inputAdmissionCredits,
      uint64_t inputAdmissionCreditBytes,
      uint64_t batchCandidateTaskScopedCreditBytes);
  void mergeBatchCandidates(
      std::unique_ptr<cudf::table> batchCandidates,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void flushBatchCandidateInputs(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void addCandidateRun(
      std::unique_ptr<cudf::table> candidateRun,
      uint64_t candidateRunBytes,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void finalizeCandidateLevels(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void spillCandidateRun(
      std::unique_ptr<cudf::table> candidateRun,
      uint64_t candidateRunBytes,
      rmm::cuda_stream_view stream);
  uint64_t estimateCandidateBytes(
      std::unique_ptr<cudf::table>& candidateRun,
      rmm::cuda_stream_view stream);
  void updateCandidateStatePeak();

  const int32_t limit_;
  const core::TopNRowNumberNode::RankFunction rankFunction_;
  const bool generateRowNumber_;
  const bool emitBatchCandidates_;
  const RowTypePtr inputType_;
  const core::PlanNodeId diagnosticNodeId_;
  const std::string admissionScope_;
  const rmm::cuda_stream_view stateStream_;
  const uint64_t candidateRunBytes_;
  const bool forceSpill_;

  std::vector<cudf::size_type> partitionKeys_;
  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::size_type> allKeyIndices_;
  std::vector<cudf::order> columnOrders_;
  std::vector<cudf::null_order> nullOrders_;

  // A boolean partition key whose name starts with this marker makes Top-N
  // conditional: false rows are known singleton/pass-through partitions and
  // can be emitted immediately; only true rows enter rank state.  This keeps
  // one input scan while avoiding state for high-volume unaffected rows.
  std::optional<cudf::size_type> passthroughKey_;
  std::deque<CudfVectorPtr> passthroughOutputs_;
  CudfVectorPtr pendingInput_;
  std::optional<DeviceMemoryAdmissionReservation> pendingAdmission_;
  int pendingAdmissionDevice_{-1};
  uint64_t pendingAdmissionBytes_{0};
  uint64_t pendingAdmissionCapacity_{0};
  ConditionalInputMode pendingConditionalInputMode_{
      ConditionalInputMode::kNone};
  std::vector<DeviceMemoryAdmissionCreditPtr> pendingInputAdmissionCredits_;
  uint64_t pendingInputAdmissionCreditBytes_{0};
  std::chrono::steady_clock::time_point pendingAdmissionStart_;

  std::vector<CudfVectorPtr> inputs_;
  struct CandidateLevel {
    std::unique_ptr<cudf::table> table;
    uint64_t bytes{0};
  };
  // Reduced runs use a binary carry: level N represents 2^N input runs.
  // This bounds repeated reduction work to O(rows * log(input batches)).
  std::vector<CandidateLevel> candidateLevels_;
  std::vector<CudfVectorPtr> pendingBatchCandidateInputs_;
  std::vector<DeviceMemoryAdmissionCreditPtr>
      pendingBatchCandidateAdmissionCredits_;
  uint64_t pendingBatchCandidateTaskScopedCreditBytes_{0};
  uint64_t pendingBatchCandidateInputBytes_{0};
  uint64_t candidateRows_{0};
  std::unique_ptr<cudf::table> candidates_;
  uint64_t candidateBytes_{0};
  uint64_t peakCandidateRows_{0};
  uint64_t peakCandidateBytes_{0};
  uint64_t candidateLevelMerges_{0};
  uint64_t candidateReductionCalls_{0};
  uint64_t candidateRowsReduced_{0};
  uint64_t rankMembershipFilterCalls_{0};
  uint64_t bufferedBytes_{0};
  uint64_t nextDiagnosticBufferedBytes_{512ULL << 20};
  uint64_t pressureRetainedMerges_{0};
  uint64_t pressurePreMergeSpills_{0};
  uint64_t pressurePostMergeSpills_{0};
  uint64_t batchCandidateBatches_{0};
  uint64_t batchCandidateInputBatches_{0};
  uint64_t batchCandidateFlushes_{0};
  uint64_t batchCandidateRows_{0};
  uint64_t batchCandidateBytes_{0};
  uint64_t pressureSplitBatches_{0};
  uint64_t pressureSplitChunks_{0};
  uint64_t pressureBlockingAdmissions_{0};
  uint64_t pressureAdmissionWaitNanos_{0};
  uint64_t conditionalBlockingAdmissions_{0};
  uint64_t conditionalAdmissionWaitNanos_{0};
  uint64_t scopedAdmissionCreditBytes_{0};
  uint64_t taskScopedAdmissionCreditBytes_{0};
  uint64_t conditionalInputRows_{0};
  uint64_t conditionalActiveRows_{0};
  uint64_t conditionalPassthroughRows_{0};
  uint64_t conditionalPassthroughAdmissionBypasses_{0};
  uint64_t pressureBypassBatches_{0};
  uint64_t pressureBypassRows_{0};
  uint64_t pressureBypassBytes_{0};
  uint64_t spillRuns_{0};
  uint64_t spillBytes_{0};
  bool taskScopedAdmissionCreditClaimed_{false};
  bool runtimeStatsRecorded_{false};
  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
  };
  std::vector<SortedRun> sortedRuns_;
  std::string spillDirectory_;
  uint64_t spillFileSequence_{0};
  std::unique_ptr<cudf::table> mergeCarry_;
  std::unique_ptr<cudf::table> partitionCarry_;
  bool readersInitialized_{false};
  bool mergeFinished_{false};
  bool spilled_{false};
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
