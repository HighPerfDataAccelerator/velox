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

// Separate translation unit for pinned-memory Arrow interop.
// Includes nanoarrow (cudf's private dependency) — must NOT include
// Apache Arrow C++ headers in this file to avoid struct redefinition.

#include "velox/experimental/cudf/exec/PinnedArrowInterop.h"
#include "velox/experimental/cudf/exec/PinnedHostMemory.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/interop.hpp>
#include <cudf/utilities/error.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow_device.h>

#include <cuda_runtime.h>

namespace facebook::velox::cudf_velox {

namespace {

/// Recursively copy device buffers in a nanoarrow ArrowArray to pinned host.
/// (Legacy per-buffer path — kept as fallback.)
void copyDeviceBuffersToPinnedHost(
    ArrowArray* array,
    rmm::cuda_stream_view stream) {
  for (int64_t i = 0; i < array->n_buffers; ++i) {
    auto* buf = ArrowArrayBuffer(array, i);
    if (buf->data == nullptr || buf->size_bytes <= 0) {
      continue;
    }

    auto* pinned = new PinnedHostBuffer(buf->size_bytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        pinned->data(),
        buf->data,
        buf->size_bytes,
        cudaMemcpyDeviceToHost,
        stream.value()));

    buf->data = pinned->data();
    buf->allocator.reallocate = nullptr;
    buf->allocator.free = [](ArrowBufferAllocator* alloc,
                             uint8_t* /*ptr*/,
                             int64_t /*size*/) {
      delete static_cast<PinnedHostBuffer*>(alloc->private_data);
    };
    buf->allocator.private_data = pinned;
    array->buffers[i] = pinned->data();
  }

  for (int64_t i = 0; i < array->n_children; ++i) {
    copyDeviceBuffersToPinnedHost(array->children[i], stream);
  }
  if (array->dictionary) {
    copyDeviceBuffersToPinnedHost(array->dictionary, stream);
  }
}

/// Recursively rebase ArrowArray buffer pointers from a device address range
/// into a contiguous pinned host buffer.  Each nanoarrow ArrowBuffer gets a
/// shared_ptr ref to keep the host buffer alive until all consumers release.
/// Returns false if any buffer pointer falls outside the packed device range
/// (caller should fall back to per-buffer D2H).
bool rebaseArrowBuffersToHost(
    ArrowArray* array,
    const uint8_t* devBase,
    size_t devSize,
    const std::shared_ptr<PinnedHostBuffer>& hostBuf) {
  uint8_t* hostBase = hostBuf->data();

  for (int64_t i = 0; i < array->n_buffers; ++i) {
    auto* buf = ArrowArrayBuffer(array, i);
    if (buf->data == nullptr || buf->size_bytes <= 0) {
      continue;
    }

    auto* devPtr = static_cast<const uint8_t*>(buf->data);
    ptrdiff_t offset = devPtr - devBase;
    if (offset < 0 ||
        static_cast<size_t>(offset + buf->size_bytes) > devSize) {
      return false;
    }

    buf->data = hostBase + offset;
    array->buffers[i] = hostBase + offset;

    auto* ref = new std::shared_ptr<PinnedHostBuffer>(hostBuf);
    buf->allocator.reallocate = nullptr;
    buf->allocator.free = [](ArrowBufferAllocator* alloc,
                             uint8_t* /*ptr*/,
                             int64_t /*size*/) {
      delete static_cast<std::shared_ptr<PinnedHostBuffer>*>(
          alloc->private_data);
    };
    buf->allocator.private_data = ref;
  }

  for (int64_t i = 0; i < array->n_children; ++i) {
    if (!rebaseArrowBuffersToHost(
            array->children[i], devBase, devSize, hostBuf)) {
      return false;
    }
  }
  if (array->dictionary) {
    if (!rebaseArrowBuffersToHost(
            array->dictionary, devBase, devSize, hostBuf)) {
      return false;
    }
  }
  return true;
}

} // namespace

cudf::unique_device_array_t pinnedToArrowHost(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream) {
  if (table.num_rows() == 0 || table.num_columns() == 0) {
    auto devArray = cudf::to_arrow_device(table, stream);
    devArray->device_type = ARROW_DEVICE_CPU;
    return devArray;
  }

  // --- Packed path: single D2H instead of per-buffer copies ---
  //
  // 1. cudf::pack  — GPU kernel copies all column buffers into one
  //                   contiguous device buffer (D2D).
  // 2. cudf::unpack — zero-copy: creates table_view into packed buffer.
  // 3. to_arrow_device on the packed view — ArrowArray whose buffer
  //                   pointers all fall inside the packed range.
  // 4. Single cudaMemcpyAsync D2H of the whole packed buffer.
  // 5. Pointer rebase — swap device pointers for host offsets.
  //
  // If to_arrow_device allocates buffers outside the packed range
  // (e.g. offset-type conversion for large strings), the rebase
  // detects this and we fall back to the legacy per-buffer path.

  auto packed = cudf::pack(table, stream);
  auto packedView = cudf::unpack(packed);
  auto devArray = cudf::to_arrow_device(packedView, stream);

  auto* devBase = static_cast<const uint8_t*>(packed.gpu_data->data());
  size_t devSize = packed.gpu_data->size();

  auto hostBuf = std::make_shared<PinnedHostBuffer>(devSize);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostBuf->data(),
      devBase,
      devSize,
      cudaMemcpyDeviceToHost,
      stream.value()));

  stream.synchronize();

  if (rebaseArrowBuffersToHost(
          &devArray->array, devBase, devSize, hostBuf)) {
    devArray->device_type = ARROW_DEVICE_CPU;
    return devArray;
  }

  // Fallback: some buffer was outside the packed range.
  // Re-create from scratch using the original per-buffer D2H path.
  auto devArrayFallback = cudf::to_arrow_device(table, stream);
  copyDeviceBuffersToPinnedHost(&devArrayFallback->array, stream);
  stream.synchronize();
  devArrayFallback->device_type = ARROW_DEVICE_CPU;
  return devArrayFallback;
}

} // namespace facebook::velox::cudf_velox
