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
#include "velox/experimental/cudf/exec/CudfPackedSpill.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/AstExpressionUtils.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Spill.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <chrono>
#include <deque>
#include <future>
#include <limits>
#include <memory>

namespace facebook::velox::cudf_velox {

class CudaEvent;
class CudfExpression;
using HashJoinSpillFile = CudfPackedSpillFile;

/// One cuDF table slice stored in host memory. The metadata describes the
/// packed contiguous device image in data. data may alias a larger shared
/// pinned allocation that contains all partitions from one input slice.
struct HashJoinHostBatch {
  std::unique_ptr<std::vector<uint8_t>> metadata;
  std::shared_ptr<uint8_t> data;
  // Keeps this batch's share of the executor-wide Grace host budget reserved
  // for exactly as long as its pageable payload remains resident.
  std::shared_ptr<void> hostReservation;
  std::shared_ptr<HashJoinSpillFile> spillFile;
  // Completion of the host payload producer: either an offset-local raw spill
  // write or an asynchronous pinned-to-pageable resident demotion. Keeping
  // this future in the batch lets partitioning continue while the producer
  // runs, while publication, restore and read-ahead still establish the
  // required producer-before-consumer ordering.
  std::shared_future<void> spillWriteFuture;
  uint64_t fileOffset{0};
  uint64_t dataBytes{0};
  vector_size_t rows{0};
  bool pinned{false};
};

/// Build-side hash buckets shared by the build and probe operators. Both sides
/// use HASH_MURMUR3 with seed zero, so a matching key is restored into exactly
/// one GPU-resident bucket.
struct GraceHashJoinBuildData {
  std::vector<std::vector<HashJoinHostBatch>> partitions;
  // RIGHT join build rows with any null join key can never match under SQL
  // null semantics. Keep them outside the hash partitions and emit them
  // directly as unmatched preserved-side rows.
  std::vector<HashJoinHostBatch> unmatchedNulls;
  uint64_t packedBytes{0};
  uint64_t residentHostBytes{0};
  uint64_t diskBytes{0};
  uint64_t rows{0};
  uint64_t unmatchedNullRows{0};
};

/// Backend-native payload scheduled by the same hierarchical partition
/// iterator as CPU Velox hash join spill. The iterator consumes these objects;
/// raw packed batches are not copied or converted to RowVector.
struct GraceHashJoinPartition {
  explicit GraceHashJoinPartition(exec::SpillPartitionId partitionId)
      : partitionId(std::move(partitionId)) {}

  const exec::SpillPartitionId& id() const {
    return partitionId;
  }

  exec::SpillPartitionId partitionId;
  std::vector<HashJoinHostBatch> build;
  std::vector<HashJoinHostBatch> probe;
};

using GraceHashJoinPartitionSet =
    std::map<exec::SpillPartitionId, std::unique_ptr<GraceHashJoinPartition>>;

/**
 * @brief Bridge for transferring build-side hash tables between build and probe
 * operators.
 *
 * This bridge manages the lifecycle of CUDF hash join objects and ensures
 * proper synchronization between build and probe phases. It stores the
 * constructed hash tables and hash join objects created from build-side data,
 * making them available to probe operators across different driver threads.
 *
 * The bridge handles batched hash tables when build data exceeds
 * cudf::size_type limits, and manages CUDA stream coordination between build
 * and probe operations.
 */
class CudfHashJoinBridge : public exec::JoinBridge {
 public:
  // The bridge transfers all build side batches and the hash join objects
  // constructed from them to the probe operator
  /** @brief Hash tables paired with their corresponding join objects for
   * batched processing */
  using hash_type = std::pair<
      std::vector<std::shared_ptr<cudf::table>>,
      std::vector<std::shared_ptr<cudf::hash_join>>>;
  struct BuildResult {
    std::optional<hash_type> hash;
    std::shared_ptr<GraceHashJoinBuildData> grace;
    bool graceActivated{false};
  };

  void setHashTable(
      std::optional<hash_type> hashObject,
      std::vector<DeviceMemoryAdmissionReservation> deviceAdmissions = {});

  std::optional<hash_type> hashOrFuture(ContinueFuture* future);

  void setGraceBuildData(std::shared_ptr<GraceHashJoinBuildData> buildData);

  void setGraceActivated();

  std::optional<std::shared_ptr<GraceHashJoinBuildData>> graceOrFuture(
      ContinueFuture* future);

  std::optional<BuildResult> resultOrFuture(ContinueFuture* future);

  std::optional<BuildResult> tryFinalResult();

  std::optional<BuildResult> finalResultOrFuture(ContinueFuture* future);

  /// Publishes aligned build/probe partitions after all probe drivers reach
  /// end-of-input. Partition selection and recursive child insertion follow
  /// CPU Velox's IterableSpillPartitionSet ordering.
  void setGracePartitions(GraceHashJoinPartitionSet partitions);

  void appendGracePartitions(GraceHashJoinPartitionSet partitions);

  std::optional<GraceHashJoinPartition> nextGracePartition();

  // Store and retrieve the CUDA stream used for building the hash join.
  void setBuildStream(rmm::cuda_stream_view buildStream);

  std::optional<rmm::cuda_stream_view> getBuildStream();

  void setBuildReadyEvent(std::shared_ptr<CudaEvent> buildReadyEvent);

  std::shared_ptr<CudaEvent> getBuildReadyEvent();

 private:
  /** @brief Hash tables and join objects transferred from build to probe
   * operators */
  std::optional<hash_type> hashObject_;
  // Keeps persistent build-state charges live for exactly as long as the
  // bridge keeps the corresponding device tables/hash objects alive.
  std::vector<DeviceMemoryAdmissionReservation> deviceAdmissions_;
  std::shared_ptr<GraceHashJoinBuildData> graceBuildData_;
  exec::IterableSpillPartitionSetBase<GraceHashJoinPartition, false>
      gracePartitions_;
  bool gracePartitionsSet_{false};
  bool graceActivated_{false};
  /** @brief CUDA stream used by build operator for proper synchronization */
  std::optional<rmm::cuda_stream_view> buildStream_;
  /** @brief Event recorded after build-side CUDA work is ready for probes */
  std::shared_ptr<CudaEvent> buildReadyEvent_;
};

/**
 * @brief Build operator that constructs CUDF hash tables from build-side input
 * data.
 *
 * This operator accumulates all build-side input batches and constructs hash
 * tables when all input is received. It handles batching when data exceeds
 * cudf::size_type limits and coordinates with other driver threads to ensure
 * only one driver performs the final hash table construction. The constructed
 * hash tables are transferred to probe operators via CudfHashJoinBridge.
 */
class CudfHashJoinBuild : public CudfOperatorBase {
 public:
  CudfHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  bool needsInput() const override;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  bool canReclaim() const override;

  bool reclaimableBytes(uint64_t& reclaimableBytes) const override;

  void reclaim(uint64_t targetBytes, memory::MemoryReclaimer::Stats& stats)
      override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  void finalizeBuild();
  uint64_t residentBuildFinalizeWorkspaceBytes() const;
  void partitionAndPack(CudfVectorPtr input);
  void storeGraceBuildBatch(
      HashJoinHostBatch batch,
      std::vector<HashJoinHostBatch>& destination,
      size_t spillPartition);
  void queueGraceInput(CudfVectorPtr input);
  void flushGraceInputBatch();
  void activateGracePath();

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  std::vector<CudfVectorPtr> inputs_;
  std::vector<DeviceMemoryAdmissionReservation> deviceAdmissions_;
  memory::MemoryPool* deviceMemoryPool_{nullptr};
  std::vector<cudf::size_type> buildKeyIndices_;
  std::shared_ptr<GraceHashJoinBuildData> graceBuildData_;
  uint64_t retainedBuildBytes_{0};
  uint64_t graceThresholdBytes_{0};
  uint64_t graceHostBytes_{0};
  uint64_t graceHostDemoteMicros_{0};
  uint64_t graceAsyncHostDemoteBytes_{0};
  uint64_t graceAsyncHostDemoteTasks_{0};
  uint64_t graceAsyncHostDemoteTailWaitMicros_{0};
  uint64_t graceRawSpillWriteMicros_{0};
  uint64_t graceEagerProbeBuildMinBytes_{0};
  // One append-only stream per hash partition keeps partition-major restore
  // sequential. The final slot is reserved for unmatched NULL build rows.
  std::vector<std::shared_ptr<HashJoinSpillFile>> graceSpillFiles_;
  int32_t gracePartitions_{0};
  bool graceEligible_{false};
  bool graceActive_{false};
  bool graceEagerProbeNotified_{false};
  // The last build Driver owns all peer inputs after the barrier. Finalizing
  // that resident image is replayable from getOutput(): a workspace wake only
  // retries admission and never repeats peer collection or bridge publication.
  bool buildFinalizePending_{false};
  ReplayableDeviceMemoryWorkspace buildFinalizeWorkspace_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

/**
 * @brief Probe operator that performs CUDF hash join operations on probe-side
 * input.
 *
 * This operator receives hash tables from CudfHashJoinBuild via the bridge and
 * performs join operations on probe-side input data. It supports all standard
 * join types (inner, left, right, anti, semi) with optional filter conditions.
 * The operator handles stream synchronization between build and probe phases,
 * manages right join state across multiple drivers, and supports batched
 * processing for large datasets.
 */
class CudfHashJoinProbe : public CudfOperatorBase {
 public:
  using hash_type = CudfHashJoinBridge::hash_type;

  CudfHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  void initialize() override;

  bool needsInput() const override;

  bool skipProbeOnEmptyBuild() const;

  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  /// Returns true if the join type is supported by cudf hash join.
  /// Supported types:
  /// - Inner, Left, Right, Full joins
  /// - Left/Right Semi Filter joins
  /// - Left/Right Semi Project joins (null-aware right semi project with a
  ///   filter is rejected by HashJoinNode validation)
  /// - Anti join (non-null-aware, or null-aware without filter)
  static bool isSupportedJoinType(core::JoinType joinType) {
    return joinType == core::JoinType::kInner ||
        joinType == core::JoinType::kLeft ||
        joinType == core::JoinType::kAnti ||
        joinType == core::JoinType::kLeftSemiFilter ||
        joinType == core::JoinType::kLeftSemiProject ||
        joinType == core::JoinType::kRight ||
        joinType == core::JoinType::kRightSemiFilter ||
        joinType == core::JoinType::kRightSemiProject ||
        joinType == core::JoinType::kFull;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  void waitForBuildReady(rmm::cuda_stream_view stream);
  void partitionAndPackProbe(CudfVectorPtr input);
  void queueGraceProbeInput(CudfVectorPtr input);
  void flushGraceProbeInputBatch();
  void spillGraceProbeBatch(HashJoinHostBatch& batch, size_t spillPartition);
  void spillRecursiveGraceBuildBatch(
      HashJoinHostBatch& batch,
      size_t spillPartition);
  void accountConsumedGraceProbeBatch(const HashJoinHostBatch& batch);
  void accountConsumedGraceBuildBatch(const HashJoinHostBatch& batch);
  void scheduleGraceProbePrefetch(
      std::vector<HashJoinHostBatch>& batches,
      size_t chunk);
  void waitForGraceProbePrefetch(size_t chunk);
  void finishGraceProbePrefetch();
  CudfVectorPtr restoreHostBatch(
      HashJoinHostBatch& batch,
      const RowTypePtr& type,
      rmm::cuda_stream_view stream,
      bool consume);
  void initializeGracePartitionQueue();
  GraceHashJoinPartitionSet repartitionGracePartition(
      GraceHashJoinPartition& partition,
      rmm::cuda_stream_view stream);
  void loadGraceBuildPartition(
      GraceHashJoinPartition& partition,
      rmm::cuda_stream_view stream);
  uint64_t estimateGraceBuildWorkspaceBytes(
      const GraceHashJoinPartition& partition,
      bool recursiveRepartition) const;
  uint64_t estimateGraceProbeWorkspaceBytes(uint64_t probeBytes) const;
  bool acquireGraceWorkspace(
      uint64_t bytes,
      DeviceMemoryWorkspacePriority priority);
  void initializeRightMatchedFlags(rmm::cuda_stream_view stream);
  RowVectorPtr rightUnmatchedOutput(rmm::cuda_stream_view stream);
  RowVectorPtr graceRightNullUnmatchedOutput(rmm::cuda_stream_view stream);
  RowVectorPtr getGraceOutput();

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  /** @brief Hash tables and join objects received from build operator */
  std::optional<hash_type> hashObject_;

  // Filter related members
  /** @brief Whether to use AST-based filtering (false if filter spans both
   * sides or if filter deals with decimal types) */
  bool useAstFilter_{true};
  /** @brief CUDF AST tree for join filter evaluation */
  cudf::ast::tree tree_;
  /** @brief Scalar values used in filter expressions */
  std::vector<std::unique_ptr<cudf::scalar>> scalars_;
  /** @brief Precompute instructions for left (probe) table columns */
  std::vector<PrecomputeInstruction> leftPrecomputeInstructions_;
  /** @brief Precompute instructions for right (build) table columns */
  std::vector<PrecomputeInstruction> rightPrecomputeInstructions_;
  /** @brief Row type for probe table (needed for precomputation) */
  RowTypePtr probeType_;
  /** @brief Row type for build table (needed for precomputation) */
  RowTypePtr buildType_;
  /** @brief Cached evaluator for post-join filter column */
  std::shared_ptr<CudfExpression> filterEvaluator_;

  bool rightPrecomputed_{false};

  // Batched probe inputs needed for right join
  std::vector<CudfVectorPtr> inputs_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  /** @brief Column indices for join keys in left (probe) table */
  std::vector<cudf::size_type> leftKeyIndices_;
  /** @brief Column indices for join keys in right (build) table */
  std::vector<cudf::size_type> rightKeyIndices_;
  /** @brief Column indices to gather from left table for output */
  std::vector<cudf::size_type> leftColumnIndicesToGather_;
  /** @brief Column indices to gather from right table for output */
  std::vector<cudf::size_type> rightColumnIndicesToGather_;
  /** @brief Output column positions for left table columns */
  std::vector<size_t> leftColumnOutputIndices_;
  /** @brief Output column positions for right table columns */
  std::vector<size_t> rightColumnOutputIndices_;
  bool finished_{false};

  std::shared_ptr<GraceHashJoinBuildData> graceBuildData_;
  // Coalesce small device inputs before hash partition + pack + D2H. This
  // amortizes partition/pack kernels without changing the bounded host tier.
  std::vector<CudfVectorPtr> graceProbeInputs_;
  uint64_t graceProbeInputBytes_{0};
  uint64_t graceProbeSourceBatches_{0};
  uint64_t graceProbePartitionBatches_{0};
  std::vector<std::vector<HashJoinHostBatch>> graceProbePartitions_;
  std::optional<GraceHashJoinPartition> gracePartition_;
  bool gracePartitionQueueInitialized_{false};
  size_t graceProbeChunk_{0};
  size_t graceNullBuildChunk_{0};
  uint64_t graceProbeBufferedBytes_{0};
  uint64_t graceProbeResidentHostBytes_{0};
  uint64_t graceProbeDiskBytes_{0};
  uint64_t graceProbeHostLimitBytes_{0};
  uint64_t graceHostDemoteMicros_{0};
  uint64_t graceRawSpillWriteMicros_{0};
  uint64_t graceRestoreResidentBytes_{0};
  uint64_t graceRestoreDiskBytes_{0};
  uint64_t graceRestorePinnedSourceBytes_{0};
  uint64_t graceRestoreResidentBounceBytes_{0};
  uint64_t graceRestorePageableDirectBytes_{0};
  uint64_t graceRestoreHostStageMicros_{0};
  uint64_t graceRestoreCopySynchronizeMicros_{0};
  uint64_t graceBulkProbeRestoreBytes_{0};
  uint64_t graceBulkProbeRestoreWaves_{0};
  uint64_t graceBulkProbeFallbackBytes_{0};
  uint64_t graceRestoreBuildBytes_{0};
  uint64_t graceRestoreProbeBytes_{0};
  uint64_t graceEagerProbeBufferLimitBytes_{0};
  // Root and recursive partition-local append streams. Batches retain shared
  // ownership, so resetting these vectors never invalidates queued ranges.
  std::vector<std::shared_ptr<HashJoinSpillFile>> graceProbeSpillFiles_;
  std::vector<std::shared_ptr<HashJoinSpillFile>>
      graceRecursiveBuildSpillFiles_;
  bool graceProbeDraining_{false};
  bool graceEnabled_{false};
  bool graceEagerActive_{false};
  bool graceFinished_{false};
  bool gracePartitionUnmatchedEmitted_{false};
  std::chrono::steady_clock::time_point gracePartitionDrainStart_;
  uint64_t gracePartitionProbeBytes_{0};
  uint64_t gracePartitionProbeChunks_{0};
  uint64_t gracePartitionProbeGroups_{0};
  uint64_t gracePartitionProbeRestoreMicros_{0};
  uint64_t gracePartitionProbePrefetchWaitMicros_{0};
  uint64_t gracePartitionJoinMicros_{0};
  uint64_t gracePartitionOutputMicros_{0};
  uint64_t gracePartitionBuildLoadMicros_{0};
  uint64_t gracePartitionWorkspaceWaitMicros_{0};
  uint64_t gracePartitionFinalSyncMicros_{0};
  // Wall time between handing a Grace output group to the downstream
  // pipeline and the next getOutput call. This separates actual
  // restore/join/concatenate work from downstream backpressure and driver
  // scheduling, which otherwise appear as unexplained partition drain time.
  uint64_t gracePartitionDownstreamHandoffMicros_{0};
  bool graceOutputHandoffPending_{false};
  std::chrono::steady_clock::time_point graceOutputHandoffStart_;
  // Workspace leases cover one GPU work unit (build restore, recursive
  // partition, or one probe chunk), not the complete partition lifetime.
  // The restored build/hash table is real device memory and is already
  // reflected by cudaMemGetInfo. Holding its transient lease while an output
  // batch is blocked downstream double-counts that memory and can starve the
  // next drain for tens of seconds.
  std::optional<DeviceMemoryWorkspaceReservation> graceWorkspaceAdmission_;
  ReplayableDeviceMemoryWorkspace graceWorkspace_;
  size_t graceProbePrefetchDepth_{4};
  std::deque<std::pair<size_t, std::future<void>>> graceProbePrefetchFutures_;

  /// True if any build table has NULL values in join key columns.
  /// Used for null-aware LEFT SEMI PROJECT to determine match column
  /// nullability.
  bool buildSideHasNullKeys_{false};

  /// Whether this probe driver has seen any non-empty probe input and whether
  /// any such input has a null join key. These are reduced across drivers at
  /// the end-of-probe barrier for null-aware RIGHT SEMI PROJECT joins.
  bool probeSideHasRows_{false};
  bool probeSideHasNullKeys_{false};

  // Copied from HashProbe.h
  // Indicates whether to skip probe input data processing or not. It only
  // applies for a specific set of join types (see skipProbeOnEmptyBuild()), and
  // the build table is empty and the probe input is read from non-spilled
  // source. This ensures the hash probe operator keeps running until all the
  // probe input from the sources have been processed. It prevents the exchange
  // hanging problem at the producer side caused by the early query finish.
  bool skipInput_{false};

  /** @brief CUDA stream from build operator for synchronization */
  std::optional<rmm::cuda_stream_view> buildStream_;
  /** @brief Event recorded after build-side CUDA work is ready for probes */
  std::shared_ptr<CudaEvent> buildReadyEvent_;
  /** @brief CUDA event for coordinating stream synchronization */
  std::unique_ptr<CudaEvent> cudaEvent_;

  // Streaming right join state
  // Per-build-table flags indicating whether a build row has had at least one
  // left match.
  /** @brief Flags tracking which build rows have been matched for right/full
   * and right-semi-project joins. */
  std::vector<std::unique_ptr<cudf::column>> rightMatchedFlags_;

  /// Cached precomputed columns for right (build) tables
  std::vector<std::vector<ColumnOrView>> cachedRightPrecomputed_;
  /// Cached extended views for right tables (original + precomputed columns)
  std::vector<cudf::table_view> cachedExtendedRightViews_;

  // For right/full and right-semi-project joins, only one driver combines the
  // build-row match masks and emits build-side output. This value is set true
  // only for that driver. See noMoreInput().
  bool isLastDriver_{false};

  /// Next build table to emit for RIGHT SEMI PROJECT. Build tables are emitted
  /// one at a time so the multi-table path does not concatenate past the cuDF
  /// size_type limit.
  size_t nextBuildOutputIndex_{0};

  /// CUDA stream used during the last getOutput() probe operation. Set only
  /// for right/full/right-semi-project joins, and only for drivers that process
  /// at least one probe batch. Used in noMoreInput() to synchronize GPU streams
  /// across drivers before combining rightMatchedFlags_. Drivers with no probe
  /// input are safe to skip: the driver loop guarantees all addInput batches
  /// are consumed by getOutput() before noMoreInput() fires, and unset flags
  /// remain in their host-synchronized all-false init state with no pending GPU
  /// work.
  std::optional<rmm::cuda_stream_view> lastProbeStream_;

  static constexpr auto oobPolicy = cudf::out_of_bounds_policy::NULLIFY;

  struct JoinOutput {
    std::unique_ptr<cudf::table> table;
    vector_size_t numRows;
  };

  /**
   * @brief Performs inner join between probe table and all build tables.
   * @param leftTable Probe-side table to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> innerJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs left join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> leftJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs right join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> rightJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs full outer join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> fullJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Implements the probe-preserving half of LEFT/FULL joins when the
   * build was split into multiple independently hashed tables.
   *
   * Each build table contributes only real INNER matches. Probe rows that did
   * not match any build table (after the residual filter) are emitted once at
   * the end. For FULL joins, matched build rows are also accumulated in
   * rightMatchedFlags_ for the existing end-of-probe unmatched-build output.
   */
  std::vector<JoinOutput> multiTableLeftOuterJoin(
      cudf::table_view leftTableView,
      bool trackRightMatches,
      rmm::cuda_stream_view stream);
  /** @brief Emits probe rows not selected by probeMatchedFlags with NULL build
   * columns. */
  JoinOutput leftUnmatchedOutput(
      cudf::table_view leftTableView,
      cudf::column_view probeMatchedFlags,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs left semi filter join between probe table and all build
   * tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> leftSemiFilterJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs left semi project join between probe table and all build
   * tables. Returns all probe rows with a boolean match column indicating
   * whether each row has a match on the build side.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> leftSemiProjectJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Performs right semi filter join between probe table and all build
   * tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> rightSemiFilterJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Accumulates per-build-row matches for a right semi project join.
   * This produces no rows while probing; the build rows and match column are
   * emitted after all probe drivers reach the end-of-input barrier.
   */
  std::vector<JoinOutput> rightSemiProjectJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);

  /// Emits the next build-side batch for RIGHT SEMI PROJECT.
  RowVectorPtr rightSemiProjectOutput(rmm::cuda_stream_view stream);
  /**
   * @brief Performs anti join between probe table and all build tables.
   * @param leftTableView Probe-side table view to join
   * @param stream CUDA stream for operations
   * @return Vector of result tables (multiple if build data was batched)
   */
  std::vector<JoinOutput> antiJoin(
      cudf::table_view leftTableView,
      rmm::cuda_stream_view stream);
  /**
   * @brief Constructs join output table without applying filter conditions.
   * @param leftTableView Input probe table view
   * @param leftIndicesCol Column of indices into left table
   * @param rightTableView Input build table view
   * @param rightIndicesCol Column of indices into right table
   * @param stream CUDA stream for operations
   * @return Join result table with its logical row count
   */
  JoinOutput unfilteredOutput(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      rmm::cuda_stream_view stream);
  /**
   * @brief Constructs join output table with filter condition applied.
   * @param leftTableView Input probe table view
   * @param leftIndicesCol Column of indices into left table
   * @param rightTableView Input build table view
   * @param rightIndicesCol Column of indices into right table
   * @param func Filter function to apply to joined data
   * @param stream CUDA stream for operations
   * @return Filtered join result table with its logical row count
   */
  JoinOutput filteredOutput(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      std::function<std::vector<std::unique_ptr<cudf::column>>(
          std::vector<std::unique_ptr<cudf::column>>&&,
          cudf::column_view)> func,
      rmm::cuda_stream_view stream);

  JoinOutput filteredOutputIndices(
      cudf::table_view leftTableView,
      cudf::column_view leftIndicesCol,
      cudf::table_view rightTableView,
      cudf::column_view rightIndicesCol,
      cudf::table_view extendedLeftView,
      cudf::table_view extendedRightView,
      cudf::join_kind joinKind,
      rmm::cuda_stream_view stream);
};

/**
 * @brief Factory for creating CUDF hash join operators and bridges from plan
 * nodes.
 *
 * This translator converts HashJoinNode plan nodes into the appropriate
 * CUDF-specific operators and bridges. It creates CudfHashJoinProbe operators
 * for probe-side processing, CudfHashJoinBuild operators for build-side
 * processing, and CudfHashJoinBridge instances for coordinating between them.
 */
class CudfHashJoinBridgeTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator>
  toOperator(exec::DriverCtx* ctx, int32_t id, const core::PlanNodePtr& node);

  std::unique_ptr<exec::JoinBridge> toJoinBridge(const core::PlanNodePtr& node);

  exec::OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node);
};

} // namespace facebook::velox::cudf_velox
