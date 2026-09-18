/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include <cudf/contiguous_split.hpp>
#include <atomic>
#include <memory>
#include <optional>
#include "velox/experimental/cudf/exec/GpuResources.h"

namespace facebook::velox::ucx_exchange {

// The producer and independent consumer clone are both charged at admission.
// A split reservation permits retiring just the producer after a completed
// clone. The consumer half remains owned by the downstream vector.
struct IntraNodeDeviceLease {
  explicit IntraNodeDeviceLease(std::atomic<uint64_t>& counter)
      : counter(counter) {}

  std::atomic<uint64_t>& counter;
  std::shared_ptr<cudf::packed_columns> producer;
  std::optional<cudf_velox::DeviceMemoryWorkspaceReservation> workspace;
  std::optional<cudf_velox::DeviceMemoryWorkspaceReservation> producerWorkspace;
  uint64_t bytes{0};
  bool splitReservation{false};

  ~IntraNodeDeviceLease() {
    producer.reset();
    producerWorkspace.reset();
    workspace.reset();
    counter.fetch_sub(bytes, std::memory_order_acq_rel);
  }
};

struct IntraNodeDeviceLeaseOwner {
  std::shared_ptr<IntraNodeDeviceLease> lease;
  void operator()(cudf::packed_columns*) noexcept {
    lease.reset();
  }
};

// PRECONDITION: the D2D clone has completed and metadata has been copied. No
// subsequent operation may dereference 'data'. Shared aliases (including a
// retained producer elsewhere) conservatively retain the full reservation.
inline std::shared_ptr<void> finishIntraNodeDeviceClone(
    std::shared_ptr<cudf::packed_columns> data) {
  const auto* owner = std::get_deleter<IntraNodeDeviceLeaseOwner>(data);
  if (owner == nullptr || data.use_count() != 1 ||
      !owner->lease->splitReservation ||
      owner->lease->producer.use_count() != 1) {
    return data;
  }
  auto lease = owner->lease;
  data.reset();
  const auto releasedBytes = lease->bytes / 2;
  lease->producer.reset();
  lease->producerWorkspace.reset();
  lease->bytes -= releasedBytes;
  lease->counter.fetch_sub(releasedBytes, std::memory_order_acq_rel);
  return lease;
}

} // namespace facebook::velox::ucx_exchange
