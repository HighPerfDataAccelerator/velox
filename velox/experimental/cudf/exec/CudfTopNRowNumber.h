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

#include "velox/common/compression/Compression.h"
#include "velox/core/PlanNode.h"

#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

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
    return !noMoreInput_ && passthroughOutputs_.empty();
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_ && passthroughOutputs_.empty();
  }

  static bool shouldReplace(
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  enum class ReductionSource {
    kInput,
    kCandidateMerge,
  };

  struct CandidateTable {
    std::unique_ptr<cudf::table> table;
    uint64_t flatBytes;
  };

  struct HostCandidateChunk {
    std::unique_ptr<std::vector<uint8_t>> metadata;
    BufferPtr data;
    uint64_t uncompressedBytes;
    cudf::size_type rows;
    uint64_t flatBytes;
  };

  struct SortedRun;

  void spillSortedRun();
  void spillCandidates();
  void compactSortedRunsForMerge();
  void initializeSortedRunReaders();
  bool loadPausedChunk(SortedRun& run);
  std::unique_ptr<cudf::table> mergeNextPausedBatch(
      std::vector<SortedRun*>& runs,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool& finished);
  std::unique_ptr<cudf::table> mergeNextSortedBatch(bool& finalBatch);
  std::unique_ptr<cudf::table> takeCompletePartitions(
      std::unique_ptr<cudf::table> sorted,
      bool finalBatch,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr computeNextSortedOutput();
  void cleanupSpillFiles();

  CudfVectorPtr computeLimitOneRowNumber(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr computeLimitOneRankLike(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CandidateTable reduceToCandidates(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      uint64_t inputFlatBytes,
      ReductionSource source);
  void addBatchCandidates(
      CandidateTable batchCandidates,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  bool supportsHostCandidateBuckets() const;
  bool tryRetainHostCandidateBatch(
      CandidateTable& batchCandidates,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  bool tryRetainCurrentCandidatesOnHost(rmm::device_async_resource_ref mr);
  CudfVectorPtr computeNextHostCandidateOutput();
  void recordFinalCandidateOutput(const CudfVectorPtr& output);
  void logCandidateObservations();

  const int32_t limit_;
  const core::TopNRowNumberNode::RankFunction rankFunction_;
  const bool generateRowNumber_;
  const RowTypePtr inputType_;
  const core::PlanNodeId diagnosticNodeId_;
  const uint64_t candidateRunBytes_;
  const rmm::cuda_stream_view spillStream_;

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

  std::vector<CudfVectorPtr> inputs_;
  // Incremental per-partition Top-1 state. Every input batch is reduced first,
  // then merged with this already-reduced state and reduced again. For
  // row_number this is at most one row per partition; rank/dense_rank retain
  // only ties for the current best sort key. Device residency therefore
  // depends on candidate cardinality, not total input rows.
  std::unique_ptr<cudf::table> candidates_;
  uint64_t candidateBytes_{0};
  std::optional<rmm::cuda_stream_view> candidateStream_;
  std::vector<std::vector<HostCandidateChunk>> hostCandidateBuckets_;
  uint64_t hostCandidateBytes_{0};
  uint64_t maxHostCandidateBytes_{0};
  uint64_t hostCandidatePayloadBytes_{0};
  uint64_t maxHostCandidatePayloadBytes_{0};
  uint64_t hostCandidateUncompressedBytes_{0};
  uint64_t maxHostCandidateUncompressedBytes_{0};
  std::unique_ptr<common::Codec> hostCandidateCodec_;
  size_t nextCandidatePartition_{0};
  bool hostCandidateMode_{false};
  uint64_t bufferedBytes_{0};
  uint64_t nextDiagnosticBufferedBytes_{512ULL << 20};
  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
    std::unique_ptr<cudf::table> chunk;
    cudf::size_type chunkOffset{0};
  };
  std::vector<SortedRun> sortedRuns_;
  std::string spillDirectory_;
  uint64_t spillFileSequence_{0};
  std::unique_ptr<cudf::table> partitionCarry_;
  bool readersInitialized_{false};
  bool mergeFinished_{false};
  bool spilled_{false};
  bool finished_{false};
  uint64_t inputReductionRows_{0};
  uint64_t inputReductionBytes_{0};
  uint64_t inputCandidateRows_{0};
  uint64_t inputCandidateBytes_{0};
  uint64_t maxInputCandidateRows_{0};
  uint64_t maxInputCandidateBytes_{0};
  uint64_t candidateMergeRows_{0};
  uint64_t candidateMergeBytes_{0};
  uint64_t finalCandidateRows_{0};
  uint64_t finalCandidateBytes_{0};
  uint64_t spilledCandidateRows_{0};
  uint64_t spillRuns_{0};
  uint64_t hostCandidateRows_{0};
  uint64_t hostCandidateBatches_{0};
  uint64_t hostOutputBuckets_{0};
  bool candidateObservationsLogged_{false};
};

} // namespace facebook::velox::cudf_velox
