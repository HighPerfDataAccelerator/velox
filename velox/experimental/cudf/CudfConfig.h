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

#include <cudf/types.hpp>

#include <optional>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfJitExpressionEnabled{
      "cudf.jit_expression_enabled"};
  static constexpr const char* kCudfJitExpressionPriority{
      "cudf.jit_expression_priority"};
  static constexpr const char* kCudfOutputMr{"cudf.output_mr"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfBatchSizeMinThreshold{
      "cudf.batch_size_min_threshold"};
  static constexpr const char* kCudfExchangeBatchSizeMinThreshold{
      "cudf.exchange_batch_size_min_threshold"};
  static constexpr const char* kCudfExchangeBatchSizeMinThresholdBytes{
      "cudf.exchange_batch_size_min_threshold_bytes"};
  static constexpr const char* kCudfBatchSizeMinThresholdBytes{
      "cudf.batch_size_min_threshold_bytes"};
  static constexpr const char* kCudfBatchSizeMaxThreshold{
      "cudf.batch_size_max_threshold"};
  static constexpr const char* kCudfHashJoinLoadFactor{
      "cudf.hash_join_load_factor"};
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  static constexpr const char* kCudfGroupbyStreamingMaxDistinctKeys{
      "cudf.groupby_streaming_max_distinct_keys"};
  static constexpr const char* kCudfPartialStreamingGroupby{
      "cudf.partial_streaming_groupby"};
  static constexpr const char* kCudfPartialStreamingMaxDistinctKeys{
      "cudf.partial_streaming_max_distinct_keys"};
  static constexpr const char* kCudfPartialIdentityAggregation{
      "cudf.partial_identity_aggregation"};
  static constexpr const char* kCudfOrderBySortedRunBytes{
      "cudf.order_by_sorted_run_bytes"};
  static constexpr const char* kCudfOrderByHostSpillBytes{
      "cudf.order_by_host_spill_bytes"};
  static constexpr const char* kCudfOrderByMergeFanIn{
      "cudf.order_by_merge_fan_in"};
  static constexpr const char* kCudfWindowSortedRunBytes{
      "cudf.window.sorted_run_bytes"};
  static constexpr const char* kCudfOrderByOutputChunkBytes{
      "cudf.order_by_output_chunk_bytes"};
  static constexpr const char* kCudfOrderByMaxOutputRows{
      "cudf.order_by_max_output_rows"};
  static constexpr const char* kCudfExchangeConcatOptimizationEnabled{
      "cudf.exchange_concat_optimization_enabled"};
  static constexpr const char* kCudfHashJoinGraceBuildBytes{
      "cudf.hash_join.grace_build_bytes"};
  static constexpr const char* kCudfHashJoinGracePartitions{
      "cudf.hash_join.grace_partitions"};
  static constexpr const char* kCudfHashJoinGraceHostBytes{
      "cudf.hash_join.grace_host_bytes"};
  static constexpr const char* kCudfHashJoinGraceRestoreBytes{
      "cudf.hash_join.grace_restore_bytes"};
  static constexpr const char* kCudfHashJoinGraceProbeRestoreBytes{
      "cudf.hash_join.grace_probe_restore_bytes"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  // The value could be either spark or presto.
  static constexpr const char* kCudfFunctionEngine{"cudf.function_engine"};
  /// Query session configs for the cuDF Operators.
  static constexpr const char* kCudfTopNBatchSize{"cudf.topk_batch_size"};
  // Maximum retained candidate bytes before CudfTopNRowNumber externalizes a
  // run. ROW_NUMBER=1 hash-partitions reduced candidates into packed host
  // memory; rank-like functions retain the sorted-run spill implementation.
  // The value must be a plain unsigned decimal byte count with no unit suffix;
  // the default is 1073741824 bytes (1 GiB). This amortizes packed-host
  // externalization while keeping the device working set bounded.
  static constexpr const char* kCudfTopNRowNumberCandidateRunBytes{
      "cudf.topn_row_number.candidate_run_bytes"};
  // Number of hash buckets used by the ROW_NUMBER=1 host-memory Top1 state.
  // A partition key belongs to exactly one bucket, so buckets can be finalized
  // independently with bounded device memory. The default is 16; must be at
  // least 2.
  static constexpr const char* kCudfTopNRowNumberHostPartitions{
      "cudf.topn_row_number.host_partitions"};
  // Maximum packed input bytes restored for one ROW_NUMBER=1 finalize unit.
  // Larger host buckets are re-hash-partitioned a chunk at a time before any
  // concatenate/reduce kernel is admitted. This bounds restore, input,
  // output, and kernel scratch independently from the initial hash fan-out.
  static constexpr const char* kCudfTopNRowNumberFinalizeInputBytes{
      "cudf.topn_row_number.finalize_input_bytes"};
  // Maximum packed ROW_NUMBER=1 bucket bytes retained on device after the
  // operator enters partitioned mode. Zero preserves the legacy all-host
  // path. Above the limit, the largest resident buckets are externalized
  // until residency falls to half the limit.
  static constexpr const char* kCudfTopNRowNumberDeviceResidentBytes{
      "cudf.topn_row_number.device_resident_bytes"};
  // Output ownership bounds for ROW_NUMBER=1. Keep these separate from
  // OrderBy: TopN hands sliced variable-width tables across an exchange,
  // whereas root OrderBy drains an already ordered result.
  static constexpr const char* kCudfTopNRowNumberOutputChunkBytes{
      "cudf.topn_row_number.output_chunk_bytes"};
  static constexpr const char* kCudfTopNRowNumberMaxOutputRows{
      "cudf.topn_row_number.max_output_rows"};
  // Process-wide per-GPU budget shared by cooperating persistent device
  // states. Zero disables global admission accounting. This is deliberately
  // separate from the per-operator residency limit: operators may use a high
  // local limit while the shared budget prevents their aggregate state from
  // exhausting the device.
  static constexpr const char* kCudfDeviceResidentCapacityBytes{
      "cudf.device_resident_capacity_bytes"};
  // Physical allocation headroom maintained by the executor-wide device
  // memory arbitrator. Persistent-state admission triggers global spill when
  // allocatable device memory falls below this watermark. Zero disables the
  // physical-pressure trigger while preserving logical-capacity arbitration.
  static constexpr const char* kCudfDeviceMemoryMinHeadroomBytes{
      "cudf.device_memory.min_headroom_bytes"};
  // Minimum bytes requested from one physical-pressure arbitration. This
  // provides hysteresis so a stream of small persistent chunks does not cause
  // one arbitration per chunk.
  static constexpr const char* kCudfDeviceMemoryMinReclaimBytes{
      "cudf.device_memory.min_reclaim_bytes"};
  // Hard limit for the complete build table retained by CudfNestedLoopJoin.
  // The default is intentionally unbounded for non-MPP callers; Gluten MPP
  // supplies a conservative value for replicated Cartesian joins.
  static constexpr const char* kCudfNestedLoopJoinMaxBuildBytes{
      "cudf.nested_loop_join.max_build_bytes"};
  static constexpr const char* kCudfSkipOutputToVelox{
      "velox.cudf.skip_output_to_velox"};

  static constexpr const char* kUcxExchange{"cudf.exchange"};
  static constexpr const char* kUcxxErrorHandling{"ucxx.error_handling"};
  static constexpr const char* kUcxIntraNodeExchange{
      "cudf.intra_node_exchange"};
  static constexpr const char* kUcxxBlockingPolling{"ucxx.blocking_polling"};
  static constexpr const char* kUcxExchangeLogLevel{"cudf.exchange_log_level"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources, and the retained-memory release threshold for async.
  int32_t memoryPercent{50};

  /// Memory resource for output vectors. When set to a value different from
  /// memoryResource, a separate MR is created for output allocations.
  /// When empty, the main memoryResource is used.
  std::string outputMemoryResource;

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Register Spark or Presto specific cuDF functions.
  std::string functionEngine{"presto"};

  /// Enable AST in expression evaluation.
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation.
  bool jitExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Priority of JIT expression.
  int jitExpressionPriority{101};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};

  /// Whether to insert CudfBatchConcat before supported Cudf operators.
  /// This can improve performance by reducing the number of cuda kernel
  /// launches on addInput of certain operators by collecting a minimum number
  /// of rows before concatenating and passing on to the next operator.
  /// This batch size is determined by batchSizeMinThreshold and
  /// batchSizeMaxThreshold
  bool concatOptimizationEnabled{false};

  /// Maximum distinct keys retained by FINAL streaming groupby. Zero keeps
  /// the all-GPU levelled aggregation path and does not construct streaming
  /// state.
  int32_t groupbyStreamingMaxDistinctKeys{0};

  /// Approximate bytes buffered before OrderBy spills an independently sorted
  /// run.
  uint64_t orderBySortedRunBytes{256ULL << 20};

  /// Process-wide bytes available to packed-cuDF OrderBy runs in host memory.
  /// Zero disables the host tier and writes runs directly to disk.
  uint64_t orderByHostSpillBytes{64ULL << 30};

  /// Number of sorted runs compacted by one OrderBy merge group.
  int32_t orderByMergeFanIn{8};

  /// Approximate bytes buffered before a partitioned Window writes an
  /// independently sorted run.
  uint64_t windowSortedRunBytes{3ULL << 30};

  /// Retained ROW_NUMBER=1 candidate bytes before packed-host
  /// externalization, and the number of host hash buckets used for that
  /// bounded state.
  uint64_t topNRowNumberCandidateRunBytes{1ULL << 30};
  uint64_t topNRowNumberHostPartitions{16};
  uint64_t topNRowNumberFinalizeInputBytes{512ULL << 20};
  uint64_t topNRowNumberDeviceResidentBytes{0};
  uint64_t topNRowNumberOutputChunkBytes{32ULL << 20};
  int32_t topNRowNumberMaxOutputRows{262144};
  uint64_t deviceResidentCapacityBytes{0};
  uint64_t deviceMemoryMinHeadroomBytes{6ULL << 30};
  uint64_t deviceMemoryMinReclaimBytes{2ULL << 30};

  /// Maximum bytes and rows returned by one OrderBy output batch.
  uint64_t orderByOutputChunkBytes{32ULL << 20};
  int32_t orderByMaxOutputRows{262144};

  /// Minimum rows to accumulate before GPU-side concatenation in
  /// `CudfBatchConcat` (default 100k).
  int32_t batchSizeMinThreshold{100000};

  /// Target rows for CudfBatchConcat immediately after a UCX Exchange. Keep
  /// this below the aggregation target because a fragment can buffer several
  /// inbound exchanges concurrently while its join builds are still live.
  int32_t exchangeBatchSizeMinThreshold{100000};

  /// Target bytes to accumulate immediately after a UCX Exchange. Zero
  /// disables the exchange-specific byte threshold.
  uint64_t exchangeBatchSizeMinThresholdBytes{0};

  /// Target bytes to accumulate before GPU-side concatenation. Zero disables
  /// the byte threshold. When both row and byte thresholds are configured,
  /// reaching either one flushes the batch.
  uint64_t batchSizeMinThresholdBytes{0};

  /// Maximum rows allowed in a concatenated batch (user configurable).
  /// When not set, cuDF's own `size_type::max()` is used.
  std::optional<int32_t> batchSizeMaxThreshold;

  /// Hash-table occupancy used by cuDF hash joins. Higher values reduce
  /// retained build memory at the cost of additional probe collisions.
  double hashJoinLoadFactor{0.5};
  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;

  /// Whether UCX exchange is enabled.
  bool exchange{false};

  /// Whether to enable error handling in UCXX endpoints.
  bool ucxxErrorHandling{true};

  /// Whether intra-node exchange optimization is enabled.
  bool intraNodeExchange{false};

  /// Whether to use blocking polling in UCXX.
  bool ucxxBlockingPolling{true};

  /// VLOG level for ucx-exchange source files.
  int32_t exchangeLogLevel{0};

  /// Whether to insert CudfBatchConcat immediately after ordinary UCX
  /// Exchange sources. Keep this independent from aggregation-side concat.
  /// This field is intentionally appended so incremental builds that reuse
  /// stable UCX objects preserve the offsets of every pre-existing field.
  bool exchangeConcatOptimizationEnabled{true};

  /// Switch eligible INNER/LEFT/RIGHT hash joins from device-resident build
  /// accumulation to packed-cuDF host partitions after this many build bytes.
  /// Zero disables the Grace path.
  uint64_t hashJoinGraceBuildBytes{0};

  /// Number of same-hash buckets used by the bounded Grace hash join.
  int32_t hashJoinGracePartitions{8};

  /// Executor-process-wide budget for packed Grace build and probe bytes
  /// retained in pageable host memory. All concurrent joins share this
  /// budget; additional packed bytes spill directly from reusable pinned
  /// staging to a raw local file. Zero keeps all packed bytes in host memory.
  uint64_t hashJoinGraceHostBytes{0};

  /// Maximum packed build bytes restored into one device hash table. A larger
  /// partition is recursively split using the next SpillPartitionId level.
  uint64_t hashJoinGraceRestoreBytes{4ULL << 30};

  /// Maximum packed probe bytes processed as one partition-major drain group.
  /// Input partitioning remains independently bounded; grouping here reduces
  /// tiny join/output handoffs while the device arbitrator serializes large
  /// restore groups when physical headroom is tight.
  uint64_t hashJoinGraceProbeRestoreBytes{1ULL << 30};
};

} // namespace facebook::velox::cudf_velox
