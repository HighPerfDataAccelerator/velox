/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace facebook::velox::cudf_velox::grace_transfer {

struct GracePinnedTransferStats {
  uint64_t bytes{0};
  uint64_t hostCopyMicros{0};
  uint64_t deviceCopyMicros{0};
  uint64_t slotAcquireMicros{0};
};

// Synchronous transfer using a dedicated bounded scratch pool. No lease can
// escape this call into a future, queued batch or downstream vector. Pool
// waiters therefore depend only on independently completing copy calls, never
// a later turn of the waiting operator. Producer ordering is caller-owned.
// Large images are chunked; allocation failure is not a pageable CUDA fallback.
GracePinnedTransferStats copyGraceViaPinned(
    void* device,
    void* host,
    uint64_t bytes,
    bool hostToDevice,
    rmm::cuda_stream_view stream);

} // namespace facebook::velox::cudf_velox::grace_transfer
