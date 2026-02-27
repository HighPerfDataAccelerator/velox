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
/// Replacement for cudf::to_arrow_host() that allocates destination buffers
/// from cudf's pinned host memory pool instead of pageable malloc. This
/// enables CUDA DMA (cudaMemcpyDefault detects pinned → direct PCIe transfer)
/// instead of the slower staged-copy path used for pageable memory.
///
/// Internally calls cudf::to_arrow_device(table_view) to get device-side
/// ArrowArray, then copies every buffer to pinned host via cudaMemcpyAsync.
/// The returned ArrowDeviceArray has device_type=CPU and owns the pinned
/// buffers through nanoarrow's ArrowBuffer deallocator mechanism.
cudf::unique_device_array_t pinnedToArrowHost(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox
