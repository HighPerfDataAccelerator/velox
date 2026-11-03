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

#include <gflags/gflags.h>

// Used in velox/common/memory/Memory.cpp

DEFINE_int32(
    velox_memory_num_shared_leaf_pools,
    32,
    "Number of shared leaf memory pools per process");

DEFINE_bool(
    velox_time_allocations,
    false,
    "Record time and volume for large allocation/free");

// Used in exec/RowContainer.cpp. Configure before starting worker threads.
DEFINE_bool(
    velox_row_container_batch_string_extraction_enabled,
    true,
    "Batch RowContainer string extraction into output-owned buffers; set only "
    "during process initialization");

// Used in exec/RowContainer.cpp. Configure before starting worker threads.
DEFINE_bool(
    velox_row_container_string_equality_enabled,
    true,
    "Use cached size/prefix and equality comparison for RowContainer string "
    "keys; set only during process initialization");

DEFINE_bool(
    velox_row_container_string_extraction_prefetch_enabled,
    true,
    "Prefetch row metadata and out-of-line string headers during batch extraction; "
    "set only during process initialization");

// Used in common/base/VeloxException.cpp
DEFINE_bool(
    velox_exception_user_stacktrace_enabled,
    false,
    "Enable the stacktrace for user type of VeloxException");

DEFINE_bool(
    velox_exception_system_stacktrace_enabled,
    true,
    "Enable the stacktrace for system type of VeloxException");

DEFINE_int32(
    velox_exception_user_stacktrace_rate_limit_ms,
    0, // effectively turns off rate-limiting
    "Min time interval in milliseconds between stack traces captured in"
    " user type of VeloxException; off when set to 0 (the default)");

DEFINE_int32(
    velox_exception_system_stacktrace_rate_limit_ms,
    0, // effectively turns off rate-limiting
    "Min time interval in milliseconds between stack traces captured in"
    " system type of VeloxException; off when set to 0 (the default)");

// Used in common/base/ProcessBase.cpp

DEFINE_bool(avx2, true, "Enables use of AVX2 when available");

DEFINE_bool(bmi2, true, "Enables use of BMI2 when available");

// Used in exec/Expr.cpp

DEFINE_string(
    velox_save_input_on_expression_any_failure_path,
    "",
    "Enable saving input vector and expression SQL on any failure during "
    "expression evaluation. Specifies the directory to use for storing the "
    "vectors and expression SQL strings.");

DEFINE_string(
    velox_save_input_on_expression_system_failure_path,
    "",
    "Enable saving input vector and expression SQL on system failure during "
    "expression evaluation. Specifies the directory to use for storing the "
    "vectors and expression SQL strings. This flag is ignored if "
    "velox_save_input_on_expression_any_failure_path is set.");

DEFINE_bool(
    force_eval_simplified,
    false,
    "Whether to overwrite queryCtx and force the "
    "use of simplified expression evaluation path.");

DEFINE_bool(
    velox_experimental_save_input_on_fatal_signal,
    false,
    "This is an experimental flag only to be used for debugging "
    "purposes. If set to true, serializes the input vector data and "
    "all the SQL expressions in the ExprSet that is currently "
    "executing, whenever a fatal signal is encountered. Enabling "
    "this flag makes the signal handler async signal unsafe, so it "
    "should only be used for debugging purposes. The vector and SQLs "
    "are serialized to files in directories specified by either "
    "'velox_save_input_on_expression_any_failure_path' or "
    "'velox_save_input_on_expression_system_failure_path'");

// TODO: deprecate this once all the memory leak issues have been fixed in
// existing meta internal use cases.
DEFINE_bool(
    velox_memory_leak_check_enabled,
    false,
    "If true, check fails on any memory leaks in memory pool and memory manager");

// TODO: deprecate this after solves all the use cases that can cause
// significant performance regression by memory usage tracking.
DEFINE_bool(
    velox_enable_memory_usage_track_in_default_memory_pool,
    false,
    "If true, enable memory usage tracking in the default memory pool");

DEFINE_bool(
    velox_suppress_memory_capacity_exceeding_error_message,
    false,
    "If true, suppress the verbose error message in memory capacity exceeded "
    "exception. This is only used by test to control the test error output size");

DEFINE_bool(velox_memory_use_hugepages, true, "Use explicit huge pages");

// Used in common/memory/MallocAllocator.cpp.
DEFINE_int64(
    velox_memory_contiguous_reuse_cache_bytes,
    1LL << 30,
    "Bytes of freed mmap'ed contiguous allocations (hash tables, RowContainer "
    "runs) that MallocAllocator keeps mapped per process for reuse by a later "
    "allocation of the same size, avoiding page faults. 0 disables.");

DEFINE_int32(
    cache_prefetch_min_pct,
    80,
    "Minimum percentage of actual uses over references to a column for prefetching. No prefetch if > 100");

DEFINE_bool(velox_ssd_odirect, true, "Use O_DIRECT for SSD cache IO");

DEFINE_bool(
    velox_ssd_verify_write,
    false,
    "Read back data after writing to SSD");

// Used in /connectors/tpch
DEFINE_int32(
    velox_tpch_text_pool_size_mb,
    300,
    "TPC-H DBGen text pool size in MB");


DEFINE_bool(
    velox_hash_join_fixed_width_output_enabled,
    true,
    "Skip per-hit byte accounting when a fixed-width output batch is row-limited.");

// Used in exec/HashTable.cpp.
DEFINE_bool(
    velox_hash_join_flatten_duplicate_chains,
    true,
    "Copy long duplicate-key chains of a finished join table into contiguous "
    "arrays so probes list matches without chasing next-row pointers.");

// Used in exec/HashTable.cpp.
DEFINE_int64(
    velox_hash_join_flat_chains_max_bytes,
    64LL << 20,
    "Largest join table, in bytes of row pointers (8 per build row), whose "
    "duplicate-key chains are flattened.");

// Used in velox/exec/VectorHasher.cpp.
DEFINE_bool(
    velox_vector_hasher_dense_value_ids,
    true,
    "Map integer keys that use distinct-value ids (hash join and group by) to "
    "ids with a direct-indexed array in front of the hash set.");
DEFINE_int64(
    velox_vector_hasher_dense_value_ids_max_range,
    4LL << 20,
    "Largest max - min + 1 of distinct key values for "
    "velox_vector_hasher_dense_value_ids.");
DEFINE_bool(
    velox_vector_hasher_batched_value_id_lookup,
    true,
    "Probe the distinct-value set of integer join keys in prefetched batches.");
DEFINE_int64(
    velox_vector_hasher_oa_value_ids_max_slots,
    1LL << 19,
    "Largest open-addressing table that maps wide-range integer keys (group "
    "by and finished join tables) to distinct-value ids in front of the hash "
    "set.");
