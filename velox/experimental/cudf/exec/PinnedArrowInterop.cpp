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

#include <cudf/interop.hpp>
#include <cudf/utilities/error.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow_device.h>

#include <cuda_runtime.h>

namespace facebook::velox::cudf_velox {

namespace {

/// Recursively copy device buffers in a nanoarrow ArrowArray to pinned host.
/// After this call, every non-null buffer in `array` points to pinned host
/// memory and the ArrowBuffer's deallocator will free it on ArrowArrayRelease.
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

    // Replace the device pointer with pinned host pointer and install
    // a deallocator that frees the PinnedHostBuffer on release.
    buf->data = pinned->data();
    buf->allocator.reallocate = nullptr;
    buf->allocator.free = [](ArrowBufferAllocator* alloc,
                             uint8_t* /*ptr*/,
                             int64_t /*size*/) {
      delete static_cast<PinnedHostBuffer*>(alloc->private_data);
    };
    buf->allocator.private_data = pinned;

    // nanoarrow keeps a separate buffer_data[] array that backs the C Data
    // Interface array->buffers pointer.  ArrowArrayBuffer() updates the
    // internal ArrowBuffer::data but NOT buffer_data[i], so consumers that
    // read array->buffers[i] (e.g. Velox importFromArrowAsOwner) would still
    // see the old device pointer.  Sync it here.
    array->buffers[i] = pinned->data();
  }

  for (int64_t i = 0; i < array->n_children; ++i) {
    copyDeviceBuffersToPinnedHost(array->children[i], stream);
  }
  if (array->dictionary) {
    copyDeviceBuffersToPinnedHost(array->dictionary, stream);
  }
}

} // namespace

cudf::unique_device_array_t pinnedToArrowHost(
    cudf::table_view const& table,
    rmm::cuda_stream_view stream) {
  // Step 1: get device-side ArrowArray (zero-copy views of GPU buffers).
  auto devArray = cudf::to_arrow_device(table, stream);

  // Step 2: copy every buffer from device to pinned host memory.
  copyDeviceBuffersToPinnedHost(&devArray->array, stream);

  // Step 3: wait for all DtoH copies to finish.
  stream.synchronize();

  // Step 4: mark the array as CPU-resident.
  devArray->device_type = ARROW_DEVICE_CPU;

  return devArray;
}

} // namespace facebook::velox::cudf_velox
