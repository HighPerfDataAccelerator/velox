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
#include "velox/experimental/cudf/exec/CudfPackedRestore.h"
#include "velox/experimental/cudf/exec/CudfPackedSpill.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"

#include <cudf/io/parquet.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/packed_types.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudaEvent;

/// GPU-accelerated TopNRowNumber: partitioned top-N with row_number, or with
/// dense_rank when there is one ordering key. Partitioned row_number limit=1
/// plans use the same operator's bounded host/disk state when their candidate
/// set outgrows the configured device tier.
///
/// Retained state is bounded to O(limit * distinct partitions) rather than
/// the full input: each input batch is locally reduced to its own top-`limit`
/// rows per partition, then merged with the running `candidates_` (which
/// already holds the top-`limit` rows per partition across all prior
/// batches) and pruned back down to `limit` rows per partition. See
/// reduceBatchToLocalCandidates() and mergeAndPruneCandidates().
class CudfTopNRowNumber : public CudfOperatorBase {
 public:
  CudfTopNRowNumber(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

  bool needsInput() const override {
    return !noMoreInput_ && passthroughOutputs_.empty() &&
        partialOutputs_.empty() && pendingInput_ == nullptr;
  }

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override {
    return finished_ && passthroughOutputs_.empty() && partialOutputs_.empty();
  }

  bool canReclaim() const override;

  bool reclaimableBytes(uint64_t& reclaimableBytes) const override;

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

  static bool useBoundedTop1(
      const std::shared_ptr<const core::TopNRowNumberNode>& node);

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  CudfVectorPtr reduceBatchToLocalCandidates(
      const CudfVectorPtr& cudfInput,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr mergeAndPruneCandidates(
      const CudfVectorPtr& previous,
      const CudfVectorPtr& incoming,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void addInputGeneral(RowVectorPtr input);
  void noMoreInputGeneral();
  RowVectorPtr getOutputGeneral();

  struct HostPackedChunk {
    std::unique_ptr<std::vector<uint8_t>> metadata;
    // May alias one shared allocation containing every partition from a
    // contiguous_split. This lets partition-to-host submit all D2H copies and
    // synchronize once instead of allocating and synchronizing per bucket.
    std::shared_ptr<uint8_t> data;
    // Shares the executor-wide packed-cuDF host tier with Grace Join and
    // other spillable GPU operators.
    std::shared_ptr<void> hostReservation;
    std::shared_ptr<CudfPackedSpillFile> spillFile;
    std::shared_future<CudfPackedSpillWriteResult> spillWriteFuture;
    uint64_t fileOffset{0};
    uint64_t dataBytes{0};
    uint64_t storedBytes{0};
    cudf::size_type rows{0};
    bool compressed{false};
    bool pinned{false};
    // True only for chunks retained in partitionedRowNumberHost_ through the
    // shared packed-cuDF host-memory arbitrator. Finalized output slices use
    // short-lived pageable host storage and must not change that accounting.
    bool accountedResidentHost{false};
  };

  struct DevicePackedChunk {
    std::unique_ptr<cudf::packed_table> packed;
    std::optional<DeviceMemoryAdmissionReservation> admission;
    uint64_t dataBytes{0};
    cudf::size_type rows{0};
    // contiguous_split may produce this chunk on a different stream from the
    // later spill/finalize consumer.  Keep the producer stream so the
    // consumer can establish an explicit dependency before reading it.
    rmm::cuda_stream_view producerStream{};
  };

  // Keeps a finalized device bucket alive until the last asynchronous slice
  // copy has finished.  Synchronizing every bucket in getOutput serializes
  // the Velox pipeline, so normal reclamation polls the event instead.  The
  // destructor is only a close/error-path safety net.
  struct DeferredDeviceOutput {
    DeferredDeviceOutput(
        std::unique_ptr<cudf::table> sourceTable,
        cudaEvent_t completionEvent)
        : table(std::move(sourceTable)), event(completionEvent) {}

    DeferredDeviceOutput(const DeferredDeviceOutput&) = delete;
    DeferredDeviceOutput& operator=(const DeferredDeviceOutput&) = delete;

    DeferredDeviceOutput(DeferredDeviceOutput&& other) noexcept
        : table(std::move(other.table)), event(other.event) {
      other.event = nullptr;
    }

    DeferredDeviceOutput& operator=(DeferredDeviceOutput&& other) noexcept {
      if (this != &other) {
        release();
        table = std::move(other.table);
        event = other.event;
        other.event = nullptr;
      }
      return *this;
    }

    ~DeferredDeviceOutput() {
      release();
    }

    void release() noexcept {
      if (event != nullptr) {
        // No CUDA call is allowed to escape a destructor.  In normal
        // execution reclaimCompletedDeviceOutputs() has already observed the
        // event as complete, so this is non-blocking.
        cudaEventSynchronize(event);
        cudaEventDestroy(event);
        event = nullptr;
      }
      table.reset();
    }

    std::unique_ptr<cudf::table> table;
    cudaEvent_t event{nullptr};
  };

  // Keeps the inputs to one finalized bucket alive until the asynchronous
  // concatenate/reduce/row-number work has stopped reading them. Output has
  // its own producer event in stagePartitionedOutput(); this owner replaces a
  // host-side stream synchronize whose only purpose was input lifetime.
  struct DeferredFinalizeOwnership {
    DeferredFinalizeOwnership(
        CudfBulkPackedRestore restoredInput,
        std::vector<DevicePackedChunk> residentInput,
        std::unique_ptr<cudf::table> mergedInput,
        cudaEvent_t completionEvent)
        : restored(std::move(restoredInput)),
          deviceChunks(std::move(residentInput)),
          merged(std::move(mergedInput)),
          event(completionEvent) {}

    DeferredFinalizeOwnership(const DeferredFinalizeOwnership&) = delete;
    DeferredFinalizeOwnership& operator=(const DeferredFinalizeOwnership&) =
        delete;

    DeferredFinalizeOwnership(DeferredFinalizeOwnership&& other) noexcept
        : restored(std::move(other.restored)),
          deviceChunks(std::move(other.deviceChunks)),
          merged(std::move(other.merged)),
          event(other.event) {
      other.event = nullptr;
    }

    DeferredFinalizeOwnership& operator=(
        DeferredFinalizeOwnership&& other) noexcept {
      if (this != &other) {
        release();
        restored = std::move(other.restored);
        deviceChunks = std::move(other.deviceChunks);
        merged = std::move(other.merged);
        event = other.event;
        other.event = nullptr;
      }
      return *this;
    }

    ~DeferredFinalizeOwnership() {
      release();
    }

    void release() noexcept {
      if (event != nullptr) {
        // Normal output polling has already observed this event as complete.
        // Synchronize only for close/error cleanup, where releasing an input
        // owner early would be unsafe.
        cudaEventSynchronize(event);
        cudaEventDestroy(event);
        event = nullptr;
      }
      merged.reset();
      deviceChunks.clear();
      restored = CudfBulkPackedRestore{};
    }

    CudfBulkPackedRestore restored;
    std::vector<DevicePackedChunk> deviceChunks;
    std::unique_ptr<cudf::table> merged;
    cudaEvent_t event{nullptr};
  };

  void externalizeRowNumberCandidates(
      std::unique_ptr<cudf::table> candidates,
      rmm::cuda_stream_view stream);
  void externalizeSequentialUniqueCandidates(
      std::unique_ptr<cudf::table> candidates,
      uint64_t estimatedBytes,
      rmm::cuda_stream_view stream);
  void maybeSwitchSequentialDenseInputMode(rmm::cuda_stream_view stream);
  void switchSequentialDenseInputMode(rmm::cuda_stream_view stream);
  void spillSequentialDensePrefixDeviceState(
      rmm::cuda_stream_view stream,
      bool flushAll);
  void processPendingInput(RowVectorPtr input);

  void handoffDevicePartitionChunk(
      DevicePackedChunk& chunk,
      rmm::cuda_stream_view consumerStream);
  void storeHostPartitionChunk(
      cudf::table_view partition,
      size_t bucket,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void storeHostPartitionChunk(
      HostPackedChunk chunk,
      size_t bucket,
      bool enableCompression = true);
  void retainDevicePartitionChunk(
      cudf::packed_table packed,
      size_t bucket,
      rmm::cuda_stream_view stream);
  void spillDevicePartition(size_t bucket, rmm::cuda_stream_view stream);
  void spillDevicePartitions(
      const std::vector<size_t>& buckets,
      rmm::cuda_stream_view stream);
  void evictLargestDevicePartitions(rmm::cuda_stream_view stream);
  uint64_t partitionBytes(size_t bucket) const;
  void repartitionOversizedHostBucket(
      size_t bucket,
      uint64_t bucketBytes,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  uint64_t storeRepartitionedChildrenWave(
      std::unique_ptr<cudf::table> partitioned,
      std::vector<cudf::size_type> offsets,
      size_t childStart,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      uint64_t& childChunks);
  CudfPackedHostRestoreChunk prepareHostChunkForBulkRestore(
      HostPackedChunk chunk);
  CudfPackedHostRestoreChunk materializeHostChunk(HostPackedChunk chunk);
  void materializeDiskChunkInto(HostPackedChunk chunk, uint8_t* destination);
  std::unique_ptr<cudf::packed_table> restoreHostChunk(
      HostPackedChunk chunk,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      std::vector<std::shared_ptr<uint8_t>>* asyncHostKeepAlive = nullptr);
  enum class PartitionedOutputStatus : uint8_t {
    kOutput,
    kBlocked,
    kFinished,
  };

  struct PartitionedOutputResult {
    PartitionedOutputStatus status;
    CudfVectorPtr output;
  };

  PartitionedOutputResult computeNextPartitionedRowNumberOutput();
  uint64_t estimateFinalizeWorkspaceBytes(uint64_t bucketBytes) const;
  void stagePartitionedOutput(
      std::unique_ptr<cudf::table> output,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  CudfVectorPtr takeNextPartitionedOutputBatch();
  void reclaimCompletedDeviceOutputs();
  void reclaimCompletedFinalizeOwnerships();
  void clearPartitionedRowNumberState();
  void ensureSpillDirectory();
  void cleanupSpillFiles();

  CudfVectorPtr computeLimitOneRowNumber(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  std::unique_ptr<cudf::table> computeGroupedLexicographicTop(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr,
      bool preserveTies);
  std::unique_ptr<cudf::table> reduceToCandidates(
      cudf::table_view input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  std::unique_ptr<cudf::table> reduceOwnedCandidates(
      std::unique_ptr<cudf::table> input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);

  const core::TopNRowNumberNode::RankFunction rankFunction_;
  const int32_t limit_;
  const bool generateRowNumber_;
  const bool partialOutput_;
  const RowTypePtr inputType_;
  const bool boundedTop1_;
  const core::PlanNodeId diagnosticNodeId_;
  const uint64_t candidateRunBytes_;
  // A TopN input reduction can temporarily hold the input, per-batch
  // candidates, a concatenated candidate table and the reduced result. The
  // reservation is acquired immediately before each addInput and released
  // when that call finishes. Keeping one reservation for the whole input
  // phase makes the physical-headroom snapshot stale while also charging idle
  // and backpressured drivers for workspace they are not using.
  const uint64_t inputWorkspaceBytes_;
  const size_t hostPartitionCount_;
  const uint64_t finalizeInputBytes_;
  const uint64_t deviceResidentBytesLimit_;
  const uint64_t globalDeviceResidentCapacityBytes_;
  const uint64_t hostResidentBytesLimit_;
  // Runtime-bounded handoff shared with OrderBy. Finalized TopN buckets can
  // be much larger, but only one output slice is restored to device at once.
  const uint64_t outputChunkBytes_;
  const cudf::size_type maxOutputRows_;
  const bool deviceOutputStagingEnabled_;
  const uint64_t abandonPartialMinRows_;
  const uint64_t abandonPartialMinPct_;

  std::vector<cudf::size_type> partitionKeys_;
  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::size_type> allKeyIndices_;
  std::vector<cudf::order> columnOrders_;
  std::vector<cudf::null_order> nullOrders_;
  std::vector<cudf::size_type> generalPartitionKeys_;
  CudfVectorPtr generalCandidates_;
  std::unique_ptr<CudaEvent> generalCudaEvent_;

  // A boolean partition key whose name starts with this marker makes Top-N
  // conditional: false rows are known singleton/pass-through partitions and
  // can be emitted immediately; only true rows enter rank state.  This keeps
  // one input scan while avoiding state for high-volume unaffected rows.
  std::optional<cudf::size_type> passthroughKey_;
  std::deque<CudfVectorPtr> passthroughOutputs_;
  std::deque<CudfVectorPtr> partialOutputs_;
  // Partial Top-N is only a pre-exchange optimization. If it retains most of
  // the input, continuing to hash and reduce every batch costs more than
  // sending the rows directly to the exact final Top-N. Match CPU Velox's
  // adaptive policy and turn the partial operator into a pass-through after
  // observing insufficient reduction.
  bool abandonedPartial_{false};
  uint64_t diagnosticPartialOutputRows_{0};
  uint64_t diagnosticPartialWorkspaceBypassBatches_{0};

  // Driver calls isBlocked() before asking the upstream operator for output.
  // Therefore workspace cannot be reserved there until this operator actually
  // owns an input batch: an upstream-starved pipeline would otherwise hold a
  // reservation indefinitely and block runnable siblings. doAddInput only
  // takes ownership; the next driver iteration arbitrates this exact batch,
  // and doGetOutput processes it after admission.
  RowVectorPtr pendingInput_;
  // Incremental per-partition Top-1 state. Every input batch is reduced first,
  // then merged with this already-reduced state and reduced again. For
  // row_number this is at most one row per partition. Device residency
  // therefore depends on candidate cardinality, not total input rows.
  std::unique_ptr<cudf::table> candidates_;
  // ROW_NUMBER=1 is associative: each batch can first retain one row per
  // logical partition, then those candidates can be reduced again. Once the
  // device threshold is reached, hash-partition the reduced state and pack
  // each bucket into host memory. At end-of-input, restore and finalize one
  // bucket at a time. This avoids both a full-device sort and Parquet I/O.
  // Keep the stream that produced candidates_: a cudf::table does not carry
  // stream provenance, so consuming it on an unrelated pool stream can race
  // the kernels that populate its columns.
  rmm::cuda_stream_view stateStream_{};
  std::vector<std::vector<HostPackedChunk>> partitionedRowNumberHost_;
  std::vector<std::vector<DevicePackedChunk>> partitionedRowNumberDevice_;
  std::vector<uint64_t> partitionedRowNumberDeviceBytes_;
  std::vector<std::shared_ptr<CudfPackedSpillFile>>
      partitionedRowNumberSpillFiles_;
  std::vector<uint32_t> partitionedRowNumberHashDepths_;
  size_t nextHostPartition_{0};
  uint64_t partitionedHostBytes_{0};
  uint64_t partitionedResidentHostBytes_{0};
  uint64_t partitionedDiskBytes_{0};
  uint64_t partitionedDiskUncompressedBytes_{0};
  // Parallel pinned-bounce materializers update only these two accounting
  // values (and the late write-completion diagnostic). Normal operator access
  // remains single-threaded and waits for every materializer before resuming.
  std::mutex partitionedDiskAccountingMutex_;
  uint64_t partitionedDeviceBytes_{0};
  uint64_t diagnosticDeviceSpillBytes_{0};
  uint64_t diagnosticDeviceSpillBuckets_{0};
  uint64_t diagnosticGlobalAdmissionRejectedBytes_{0};
  // The host path is a bounded fallback for devices which cannot retain one
  // finalized bucket while downstream consumes it. The normal path keeps one
  // bucket on device and copies bounded output slices directly, avoiding a
  // bucket-wide D2H followed by one H2D restore per slice.
  std::deque<HostPackedChunk> pendingPartitionedOutputChunks_;
  // Retain at most one finalized bucket and emit bounded device copies. This
  // avoids a bucket-wide D2H/pageable copy followed by H2D per output slice.
  std::unique_ptr<cudf::table> pendingPartitionedDeviceOutput_;
  std::optional<DeviceMemoryWorkspaceReservation> inputWorkspaceAdmission_;
  ReplayableDeviceMemoryWorkspace deviceWorkspace_;
  cudf::size_type pendingPartitionedDeviceOutputOffset_{0};
  uint64_t pendingPartitionedDeviceOutputBytes_{0};
  rmm::cuda_stream_view pendingPartitionedDeviceOutputStream_{};
  // A pool stream must not be retained across many getOutput calls: another
  // fragment can acquire and enqueue unrelated work on the same stream. Use
  // an operator-owned stream for the direct-output pipeline instead.
  std::shared_ptr<rmm::cuda_stream> deviceOutputStream_;
  std::deque<DeferredDeviceOutput> deferredDeviceOutputs_;
  std::deque<DeferredFinalizeOwnership> deferredFinalizeOwnerships_;
  bool partitionedRowNumberMode_{false};
  // Opt-in exact path for low-reduction final ROW_NUMBER=1. Wide input runs
  // are stored without a local hash partition while narrow partition-key
  // runs classify keys. Globally singleton keys are emitted directly; only
  // rows whose exact key occurs more than once enter the historical
  // hash-partition/reduction path.
  // Narrow partition keys are hash-partitioned independently of the wide
  // arrival-order payload. Equal keys always land in the same bucket, so
  // exact distinct_count can prove uniqueness one bounded bucket at a time.
  std::vector<std::vector<HostPackedChunk>> sequentialUniqueKeyPartitions_;
  // Once duplicate keys are known, move the untouched arrival-order payload
  // out of the normal candidate buckets. This frees those buckets to receive
  // only sparse duplicate rows while the wide runs are restored one chunk at
  // a time and split by sequentialDuplicateFilter_.
  std::vector<std::vector<HostPackedChunk>> sequentialWideHost_;
  // An early dense decision is made while addInput owns only the normal
  // input-workspace lease.  Re-hashing the complete accumulated wide prefix
  // in that call creates an unbounded allocation burst.  Move the prefix
  // aside and convert one bounded packed chunk at a time during drain, where
  // every restore/hash wave first acquires the shared drain-workspace lease.
  std::vector<HostPackedChunk> sequentialDensePrefixHost_;
  std::vector<std::unique_ptr<cudf::table>> sequentialDuplicateKeyTables_;
  std::unique_ptr<cudf::table> sequentialDuplicateKeys_;
  std::unique_ptr<cudf::filtered_join> sequentialDuplicateFilter_;
  std::shared_ptr<rmm::cuda_stream> sequentialSparseStream_;
  uint64_t sequentialUniqueKeyBytes_{0};
  uint64_t sequentialUniqueKeyRows_{0};
  size_t sequentialUniqueNextBucket_{0};
  size_t sequentialUniqueProbeBucket_{0};
  uint64_t sequentialUniqueProbedRows_{0};
  uint64_t sequentialUniqueProbedDistinctRows_{0};
  uint64_t sequentialUniqueProbeNanos_{0};
  uint64_t sequentialUniqueProbePeakWorkspaceBytes_{0};
  uint64_t sequentialDuplicateKeyRows_{0};
  // Exact input-row count whose keys occur more than once, computed from the
  // narrow key proof before any wide payload is restored. Sparse continuation
  // is profitable only when this is a small fraction of the full input.
  uint64_t sequentialProvenDuplicateCandidateRows_{0};
  uint64_t sequentialSparseCandidateRows_{0};
  uint64_t sequentialSingletonOutputRows_{0};
  size_t sequentialWideDrainBucket_{0};
  uint64_t sequentialEarlyDenseInputBatches_{0};
  uint64_t sequentialEarlyDenseProbeRows_{0};
  uint64_t sequentialEarlyDenseDistinctRows_{0};
  uint64_t sequentialDensePrefixRows_{0};
  uint64_t sequentialDensePrefixBytes_{0};
  uint64_t sequentialDenseConvertedRows_{0};
  uint64_t sequentialDenseConvertedBytes_{0};
  bool sequentialUniqueMode_{false};
  bool sequentialUniqueProbeComplete_{false};
  bool sequentialUniqueConfirmed_{false};
  bool sequentialSparseMode_{false};
  // An early narrow-key probe can prove that the accumulating input is dense
  // enough that preserving arrival-order wide runs is no longer profitable.
  // Switching to the historical exact hash buckets is performance-only: a
  // false positive still produces the same exact Top-1 result.
  bool sequentialDenseInputMode_{false};
  // A dense sequential proof must hash the already externalized arrival-order
  // rows into exact Top-1 buckets. Keep only that second-generation rewrite
  // raw so the same wide payload is not compressed twice.
  bool sequentialDenseRawRewrite_{false};
  uint64_t diagnosticInputBatches_{0};
  uint64_t diagnosticInputRows_{0};
  uint64_t diagnosticAddInputMicros_{0};
  uint64_t diagnosticExternalizeCalls_{0};
  uint64_t diagnosticHashPartitionMicros_{0};
  uint64_t diagnosticContiguousSplitMicros_{0};
  uint64_t diagnosticD2hSubmitMicros_{0};
  uint64_t diagnosticD2hSynchronizeMicros_{0};
  uint64_t diagnosticCompressionMicros_{0};
  uint64_t diagnosticSpillEnqueueMicros_{0};
  uint64_t diagnosticStoreMicros_{0};
  std::string spillDirectory_;
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
