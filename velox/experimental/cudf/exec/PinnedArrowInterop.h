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

#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace facebook::velox::cudf_velox {

/// DtoH: Convert cudf table_view to host ArrowDeviceArray using pinned memory.
///
/// Uses cudf::pack() to consolidate all column buffers into a single
/// contiguous device buffer, then transfers the entire buffer to pinned
/// host memory in one cudaMemcpyAsync call.  This eliminates the CPU
/// overhead of issuing per-buffer D2H copies (~47µs each).
///
/// Falls back to per-buffer D2H if to_arrow_device produces pointers
/// outside the packed range (e.g. offset-type conversion for large strings).
///
/// The returned ArrowDeviceArray has device_type=CPU and owns the pinned
/// buffer through shared_ptr ref-counting in nanoarrow's ArrowBuffer
/// deallocator.
cudf::unique_device_array_t pinnedToArrowHost(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox
