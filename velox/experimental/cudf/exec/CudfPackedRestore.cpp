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

#include "velox/experimental/cudf/exec/CudfPackedRestore.h"

#include "velox/experimental/cudf/exec/CudfPackedSpill.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/error.hpp>

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <semaphore>

namespace facebook::velox::cudf_velox {
namespace {

uint64_t alignedOffset(uint64_t offset) {
  constexpr auto kAlignment =
      static_cast<uint64_t>(rmm::CUDA_ALLOCATION_ALIGNMENT);
  static_assert((kAlignment & (kAlignment - 1)) == 0);
  VELOX_CHECK_LE(
      offset,
      std::numeric_limits<uint64_t>::max() - (kAlignment - 1),
      "Packed cuDF bulk restore size overflow");
  return (offset + kAlignment - 1) & ~(kAlignment - 1);
}

constexpr uint64_t kDefaultPinnedBounceBytes = 128ULL << 20;
constexpr uint64_t kMinPinnedBounceRestoreBytes = 16ULL << 20;

uint64_t pinnedBounceBytes() {
  static const uint64_t bytes = [] {
    const auto* value =
        std::getenv("CUDF_PACKED_RESTORE_PINNED_BOUNCE_BYTES");
    if (value == nullptr) {
      return kDefaultPinnedBounceBytes;
    }
    char* end = nullptr;
    const auto requested = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
      return kDefaultPinnedBounceBytes;
    }
    // Zero is the explicit A/B disable switch. Keep a non-zero slab large
    // enough to amortize event and submission overhead, and bounded so two
    // concurrent restore drivers cannot silently grow the pinned pool.
    return requested == 0
        ? uint64_t{0}
        : std::clamp<uint64_t>(requested, 16ULL << 20, 256ULL << 20);
  }();
  return bytes;
}

size_t pinnedBounceHostThreads() {
  static const size_t threads = [] {
    const auto* value =
        std::getenv("CUDF_PACKED_RESTORE_HOST_THREADS");
    if (value == nullptr) {
      return size_t{1};
    }
    char* end = nullptr;
    const auto requested = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
      return size_t{1};
    }
    return static_cast<size_t>(
        std::clamp<uint64_t>(requested, 1, 8));
  }();
  return threads;
}

folly::CPUThreadPoolExecutor& pinnedBounceHostExecutor() {
  // Preserve the proven default queue and worker topology for Grace and
  // sparse TopN restore calls.
  static folly::CPUThreadPoolExecutor executor(pinnedBounceHostThreads());
  return executor;
}

folly::CPUThreadPoolExecutor& expandedPinnedBounceHostExecutor() {
  // Constructed only when an explicitly expanded call is first used. Active
  // callbacks from both pools are jointly limited by pinnedBounceGlobalSlots.
  static folly::CPUThreadPoolExecutor executor(8);
  return executor;
}

std::counting_semaphore<8>& pinnedBounceGlobalSlots() {
  static std::counting_semaphore<8> slots(8);
  return slots;
}

size_t residentPageableBounceHostThreads() {
  static const size_t threads = [] {
    const auto* value =
        std::getenv("CUDF_HASH_JOIN_GRACE_PAGEABLE_RESTORE_THREADS");
    if (value == nullptr) {
      return size_t{4};
    }
    char* end = nullptr;
    const auto requested = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
      return size_t{4};
    }
    return static_cast<size_t>(
        std::clamp<uint64_t>(requested, 1, 8));
  }();
  return threads;
}

folly::CPUThreadPoolExecutor& residentPageableBounceHostExecutor() {
  static folly::CPUThreadPoolExecutor executor(
      residentPageableBounceHostThreads());
  return executor;
}

bool needsPinnedStage(const CudfPackedHostRestoreChunk& chunk) {
  return static_cast<bool>(chunk.materializeIntoPinned) ||
      chunk.stageResidentPageableThroughPinned;
}

std::function<void(uint8_t*)> takePinnedStage(
    CudfPackedHostRestoreChunk& chunk) {
  if (chunk.materializeIntoPinned) {
    return std::move(chunk.materializeIntoPinned);
  }
  VELOX_CHECK(chunk.stageResidentPageableThroughPinned);
  VELOX_CHECK(
      chunk.dataBytes == 0 || chunk.data != nullptr,
      "Resident pageable packed restore has no source data");
  auto source = chunk.data;
  const auto bytes = chunk.dataBytes;
  return [source = std::move(source), bytes](uint8_t* destination) {
    std::memcpy(destination, source.get(), bytes);
  };
}

class ScopedCudaEvent {
 public:
  ScopedCudaEvent() {
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~ScopedCudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  ScopedCudaEvent(const ScopedCudaEvent&) = delete;
  ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;

  cudaEvent_t get() const {
    return event_;
  }

 private:
  cudaEvent_t event_{nullptr};
};

uint64_t elapsedMicros(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

CudfBulkPackedRestore bulkRestoreCudfPackedHostChunks(
    std::vector<CudfPackedHostRestoreChunk> chunks,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    CudfBulkPackedRestoreOptions options) {
  uint64_t diagnosticBytes = 0;
  for (const auto& chunk : chunks) {
    diagnosticBytes += chunk.dataBytes;
  }
  CudaCallDiagnosticScope callDiagnostic(fmt::format(
      "phase=bulkPackedRestore chunks={} bytes={} stream={}",
      chunks.size(),
      diagnosticBytes,
      static_cast<const void*>(stream.value())));
  CudfBulkPackedRestore result;
  if (chunks.empty()) {
    return result;
  }

  std::vector<uint64_t> offsets;
  offsets.reserve(chunks.size());
  uint64_t totalBytes = 0;
  for (const auto& chunk : chunks) {
    VELOX_CHECK_NOT_NULL(chunk.metadata);
    VELOX_CHECK(
        chunk.dataBytes == 0 || chunk.data != nullptr ||
            chunk.materializeIntoPinned,
        "Packed cuDF restore chunk has {} bytes but no host data or "
        "materializer",
        chunk.dataBytes);
    totalBytes = alignedOffset(totalBytes);
    offsets.push_back(totalBytes);
    VELOX_CHECK_LE(
        chunk.dataBytes,
        std::numeric_limits<uint64_t>::max() - totalBytes,
        "Packed cuDF bulk restore size overflow");
    totalBytes += chunk.dataBytes;
  }
  VELOX_CHECK_LE(
      totalBytes,
      std::numeric_limits<size_t>::max(),
      "Packed cuDF bulk restore exceeds addressable device buffer size");

  // A zero-byte producer can still own a completed spill extent whose
  // accounting must be reclaimed. Resolve it before the copy loops, which
  // otherwise intentionally skip empty device ranges.
  for (auto& chunk : chunks) {
    if (chunk.dataBytes == 0 && chunk.materializeIntoPinned) {
      const auto stageStart = std::chrono::steady_clock::now();
      chunk.materializeIntoPinned(nullptr);
      chunk.materializeIntoPinned = {};
      result.stats_.hostStageMicros += elapsedMicros(stageStart);
    }
  }

  result.gpuData_ = std::make_unique<rmm::device_buffer>(
      static_cast<size_t>(totalBytes), stream, mr);
  auto* const gpuBase =
      static_cast<uint8_t*>(result.gpuData_->data());

  const auto hasPinnedStages = std::any_of(
      chunks.begin(), chunks.end(), [](const auto& chunk) {
        return needsPinnedStage(chunk);
      });
  const auto bounceCapacity = pinnedBounceBytes();
  const auto useCallLocalPinnedLimit = options.pinnedHostThreads > 0;
  const auto pinnedHostThreads = useCallLocalPinnedLimit
      ? std::clamp<size_t>(options.pinnedHostThreads, 1, 8)
      : pinnedBounceHostThreads();
  std::counting_semaphore<8> callLocalPinnedSlots(
      static_cast<std::ptrdiff_t>(pinnedHostThreads));
  std::array<std::shared_ptr<uint8_t>, 2> bounceBuffers;
  size_t bounceBufferCount = 0;
  if (hasPinnedStages &&
      totalBytes >= kMinPinnedBounceRestoreBytes && bounceCapacity > 0) {
    bounceBuffers[0] = acquireCudfPackedPinnedBuffer(bounceCapacity);
    if (bounceBuffers[0]) {
      // One slab is sufficient when every non-empty chunk belongs to one
      // complete staging wave. This is the early-dense prefix shape (one
      // <=96-MiB packed chunk): requiring an unused second slab needlessly
      // fell back to pageable H2D while D2H spill occupied the shared pool.
      // Multi-wave restore still requires two slabs so it cannot convoy by
      // synchronizing the same slab after every submission.
      const auto singleStageWave = totalBytes <= bounceCapacity &&
          std::all_of(chunks.begin(), chunks.end(), [](const auto& chunk) {
            return chunk.dataBytes == 0 || needsPinnedStage(chunk);
          });
      if (singleStageWave) {
        bounceBufferCount = 1;
      } else {
        bounceBuffers[1] = acquireCudfPackedPinnedBuffer(bounceCapacity);
        if (bounceBuffers[1]) {
          bounceBufferCount = 2;
        } else {
          bounceBuffers = {};
        }
      }
    }
  }

  if (bounceBufferCount > 0) {
    // Keep allocation/deallocation ordered on the caller's compute stream,
    // but run the H2D wave on a separate pooled stream so its copy-engine work
    // may overlap kernels from other admitted drivers. The final host wait is
    // also the lifetime boundary for the reusable bounce leases.
    auto copyStream = cudfGlobalStreamPool().get_stream();
    ScopedCudaEvent allocationReady;
    CUDF_CUDA_TRY(cudaEventRecord(allocationReady.get(), stream.value()));
    CUDF_CUDA_TRY(cudaStreamWaitEvent(
        copyStream.value(), allocationReady.get(), 0));

    std::array<std::unique_ptr<ScopedCudaEvent>, 2> reusableEvents;
    std::array<bool, 2> pending{false, false};
    for (size_t slot = 0; slot < bounceBufferCount; ++slot) {
      reusableEvents[slot] = std::make_unique<ScopedCudaEvent>();
    }

    size_t next = 0;
    size_t nextSlot = 0;
    while (next < chunks.size()) {
      while (next < chunks.size() && chunks[next].dataBytes == 0) {
        ++next;
      }
      if (next == chunks.size()) {
        break;
      }

      if (!needsPinnedStage(chunks[next])) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            gpuBase + offsets[next],
            chunks[next].data.get(),
            chunks[next].dataBytes,
            cudaMemcpyHostToDevice,
            copyStream.value()));
        result.stats_.pageableDirectBytes += chunks[next].dataBytes;
        ++next;
        continue;
      }

      if (chunks[next].dataBytes > bounceCapacity) {
        if (chunks[next].materializeIntoPinned) {
          chunks[next].data = std::shared_ptr<uint8_t>(
              new uint8_t[chunks[next].dataBytes],
              std::default_delete<uint8_t[]>());
          const auto stageStart = std::chrono::steady_clock::now();
          chunks[next].materializeIntoPinned(chunks[next].data.get());
          result.stats_.hostStageMicros += elapsedMicros(stageStart);
          chunks[next].materializeIntoPinned = {};
        }
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            gpuBase + offsets[next],
            chunks[next].data.get(),
            chunks[next].dataBytes,
            cudaMemcpyHostToDevice,
            copyStream.value()));
        result.stats_.pageableDirectBytes += chunks[next].dataBytes;
        ++next;
        continue;
      }

      const auto groupBegin = next;
      const auto deviceBegin = offsets[groupBegin];
      uint64_t deviceEnd = deviceBegin;
      while (next < chunks.size()) {
        if (chunks[next].dataBytes == 0) {
          ++next;
          continue;
        }
        if (!needsPinnedStage(chunks[next])) {
          break;
        }
        const auto candidateEnd = offsets[next] + chunks[next].dataBytes;
        if (candidateEnd - deviceBegin > bounceCapacity) {
          break;
        }
        deviceEnd = candidateEnd;
        ++next;
      }
      VELOX_CHECK_GT(deviceEnd, deviceBegin);

      if (pending[nextSlot]) {
        const auto waitStart = std::chrono::steady_clock::now();
        CUDF_CUDA_TRY(
            cudaEventSynchronize(reusableEvents[nextSlot]->get()));
        result.stats_.bounceReuseWaitMicros += elapsedMicros(waitStart);
      }

      const auto stageStart = std::chrono::steady_clock::now();
      const auto materializerCount = std::count_if(
          chunks.begin() + groupBegin,
          chunks.begin() + next,
          [](const auto& chunk) {
            return chunk.dataBytes > 0 &&
                needsPinnedStage(chunk);
          });
      const auto residentPageableOnly = std::all_of(
          chunks.begin() + groupBegin,
          chunks.begin() + next,
          [](const auto& chunk) {
            return chunk.dataBytes == 0 ||
                (chunk.stageResidentPageableThroughPinned &&
                 !chunk.materializeIntoPinned);
          });
      const auto hostThreads = residentPageableOnly
          ? residentPageableBounceHostThreads()
          : pinnedHostThreads;
      if (!residentPageableOnly) {
        result.stats_.pinnedHostThreadLimit = std::max<uint64_t>(
            result.stats_.pinnedHostThreadLimit, hostThreads);
      }
      if (hostThreads > 1 && materializerCount > 1) {
        // Job 144 buckets contain roughly 33 independently stored chunks.
        // Fill the non-overlapping regions of one pinned slab concurrently;
        // the existing H2D copy still consumes the slab as one contiguous
        // transfer. packaged_task preserves exceptions, and every scheduled
        // callback is joined before the slab is submitted or released.
        std::vector<std::future<void>> completions;
        completions.reserve(materializerCount);
        for (size_t i = groupBegin; i < next; ++i) {
          if (chunks[i].dataBytes == 0) {
            continue;
          }
          auto materialize = takePinnedStage(chunks[i]);
          auto* const destination = bounceBuffers[nextSlot].get() +
              offsets[i] - deviceBegin;
          auto task = std::make_shared<std::packaged_task<void()>>(
              [materialize = std::move(materialize), destination]() mutable {
                materialize(destination);
              });
          completions.push_back(task->get_future());
          try {
            if (residentPageableOnly) {
              residentPageableBounceHostExecutor().add(
                  [task]() mutable { (*task)(); });
            } else {
              // Keep submission non-blocking, matching the proven default
              // implementation: callers enqueue a complete group and then
              // join its futures. The ordinary pool still owns exactly the
              // configured default worker count. Expanded dense TopN work
              // uses a separate pool, while the callback-side semaphore caps
              // active materializers from both pools at eight in aggregate.
              auto runTask = [task,
                              &callLocalPinnedSlots,
                              useCallLocalPinnedLimit]() mutable {
                if (useCallLocalPinnedLimit) {
                  callLocalPinnedSlots.acquire();
                }
                pinnedBounceGlobalSlots().acquire();
                (*task)();
                pinnedBounceGlobalSlots().release();
                if (useCallLocalPinnedLimit) {
                  callLocalPinnedSlots.release();
                }
              };
              try {
                if (useCallLocalPinnedLimit) {
                  expandedPinnedBounceHostExecutor().add(runTask);
                } else {
                  pinnedBounceHostExecutor().add(runTask);
                }
              } catch (...) {
                runTask();
              }
            }
          } catch (...) {
            // Preserve correctness if executor admission itself fails. The
            // packaged task records any callback failure in its future.
            (*task)();
          }
        }
        std::exception_ptr firstError;
        for (auto& completion : completions) {
          try {
            completion.get();
          } catch (...) {
            if (!firstError) {
              firstError = std::current_exception();
            }
          }
        }
        if (firstError) {
          std::rethrow_exception(firstError);
        }
        ++result.stats_.parallelHostStageGroups;
        result.stats_.parallelHostStageChunks += materializerCount;
      } else {
        for (size_t i = groupBegin; i < next; ++i) {
          if (chunks[i].dataBytes == 0) {
            continue;
          }
          auto materialize = takePinnedStage(chunks[i]);
          materialize(
              bounceBuffers[nextSlot].get() + offsets[i] - deviceBegin);
        }
      }
      for (size_t i = groupBegin; i < next; ++i) {
        result.stats_.pinnedBounceBytes += chunks[i].dataBytes;
        if (chunks[i].stageResidentPageableThroughPinned) {
          result.stats_.residentPageableBounceBytes += chunks[i].dataBytes;
        }
      }
      result.stats_.hostStageMicros += elapsedMicros(stageStart);

      CUDF_CUDA_TRY(cudaMemcpyAsync(
          gpuBase + deviceBegin,
          bounceBuffers[nextSlot].get(),
          deviceEnd - deviceBegin,
          cudaMemcpyHostToDevice,
          copyStream.value()));
      CUDF_CUDA_TRY(cudaEventRecord(
          reusableEvents[nextSlot]->get(), copyStream.value()));
      pending[nextSlot] = true;
      ++result.stats_.pinnedBounceCopies;
      nextSlot = (nextSlot + 1) % bounceBufferCount;
    }

    const auto syncStart = std::chrono::steady_clock::now();
    copyStream.synchronize();
    result.stats_.copyStreamSynchronizeMicros = elapsedMicros(syncStart);
  } else {
    for (size_t i = 0; i < chunks.size(); ++i) {
      if (chunks[i].dataBytes == 0) {
        continue;
      }
      if (chunks[i].materializeIntoPinned) {
        chunks[i].data = std::shared_ptr<uint8_t>(
            new uint8_t[chunks[i].dataBytes],
            std::default_delete<uint8_t[]>());
        const auto stageStart = std::chrono::steady_clock::now();
        chunks[i].materializeIntoPinned(chunks[i].data.get());
        result.stats_.hostStageMicros += elapsedMicros(stageStart);
        chunks[i].materializeIntoPinned = {};
      }
      // A resident pageable bounce is opportunistic. If the two-slab pool is
      // unavailable, use the original source directly and do not perform an
      // otherwise useless pageable-to-pageable staging copy.
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          gpuBase + offsets[i],
          chunks[i].data.get(),
          chunks[i].dataBytes,
          cudaMemcpyHostToDevice,
          stream.value()));
      result.stats_.pageableDirectBytes += chunks[i].dataBytes;
    }
    // Host payload owners may be released after this function. Complete the
    // entire H2D wave once before unpacking and releasing them.
    const auto syncStart = std::chrono::steady_clock::now();
    stream.synchronize();
    result.stats_.copyStreamSynchronizeMicros = elapsedMicros(syncStart);
  }

  result.metadata_.reserve(chunks.size());
  result.tables_.reserve(chunks.size());
  for (size_t i = 0; i < chunks.size(); ++i) {
    result.metadata_.push_back(std::move(chunks[i].metadata));
    const auto& metadata = result.metadata_.back();
    result.tables_.push_back(
        metadata->empty()
            ? cudf::table_view{}
            : cudf::unpack(metadata->data(), gpuBase + offsets[i]));
  }
  return result;
}

} // namespace facebook::velox::cudf_velox
