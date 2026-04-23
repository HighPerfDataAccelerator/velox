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

#include "velox/experimental/cudf/CudfConfig.h"

#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace facebook::velox::cudf_velox {

/// Accumulates GPU-side elapsed time across multiple start/stop
/// intervals using CUDA events.  Events are lightweight stream
/// markers; recording them does NOT synchronize the host.
///
/// Typical usage inside a cuDF operator:
///
///   gpuTimer_.start(stream);
///   /* cudf API calls that enqueue GPU kernels */
///   gpuTimer_.stop(stream);
///
/// At close() time, call totalNanos() to get the accumulated GPU time
/// and report it as a RuntimeStat.  totalNanos() may trigger a
/// partial sync (cudaEventSynchronize) on the last pending interval.
///
/// Gated by CudfConfig::profilingTimersEnabled (default true).  When
/// disabled, no cudaEvent* calls are issued and totalNanos() returns 0.
/// Measured ~7-8% warm-state speedup on q3/q7 sf1000 4-GPU when off,
/// due to driver-lock contention under dense cudaEventRecord.
class GpuTimer {
 public:
  GpuTimer() {
    if (!kEnabled()) {
      return;
    }
    cudaEventCreateWithFlags(&startEvt_, cudaEventDefault);
    cudaEventCreateWithFlags(&stopEvt_, cudaEventDefault);
  }

  ~GpuTimer() {
    if (startEvt_) {
      cudaEventDestroy(startEvt_);
    }
    if (stopEvt_) {
      cudaEventDestroy(stopEvt_);
    }
  }

  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;

  /// Record the start of a GPU timing interval on the given stream.
  /// If there is a pending (unresolved) previous interval, it is
  /// resolved first via cudaEventQuery (non-blocking when the GPU
  /// has already caught up, which is the common case).
  void start(rmm::cuda_stream_view stream) {
    if (!kEnabled()) {
      return;
    }
    resolvePending();
    cudaEventRecord(startEvt_, stream.value());
    started_ = true;
  }

  /// Record the end of a GPU timing interval.
  void stop(rmm::cuda_stream_view stream) {
    if (!kEnabled()) {
      return;
    }
    if (!started_) {
      return;
    }
    cudaEventRecord(stopEvt_, stream.value());
    pending_ = true;
    started_ = false;
  }

  /// Return the total accumulated GPU-side nanoseconds.
  /// Forces resolution of any pending interval (may sync).
  uint64_t totalNanos() {
    if (!kEnabled()) {
      return 0;
    }
    resolvePending();
    return accumulatedNs_;
  }

 private:
  /// Read CudfConfig::profilingTimersEnabled once at first call and cache.
  /// The config is initialized early in VeloxBackend init before any cudf
  /// operator runs, so this cached value is stable for the lifetime of
  /// the process.  Branch-predictable — every timer method exits early
  /// when false, incurring no cudaEvent* calls.
  static bool kEnabled() {
    static const bool enabled =
        CudfConfig::getInstance().profilingTimersEnabled;
    return enabled;
  }

  void resolvePending() {
    if (!pending_) {
      return;
    }
    auto status = cudaEventQuery(stopEvt_);
    if (status == cudaErrorNotReady) {
      cudaEventSynchronize(stopEvt_);
    }
    float ms = 0;
    cudaEventElapsedTime(&ms, startEvt_, stopEvt_);
    if (ms > 0) {
      accumulatedNs_ +=
          static_cast<uint64_t>(static_cast<double>(ms) * 1e6);
    }
    pending_ = false;
  }

  cudaEvent_t startEvt_{};
  cudaEvent_t stopEvt_{};
  bool started_ = false;
  bool pending_ = false;
  uint64_t accumulatedNs_ = 0;
};

/// Name used for the RuntimeStat reported by GPU-timed operators.
inline constexpr const char* kGpuComputeNanos = "gpuComputeNanos";

} // namespace facebook::velox::cudf_velox
