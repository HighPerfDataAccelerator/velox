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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Physical location of one opaque packed-cuDF image. The host scheduler never
/// interprets metadata or payload bytes: metadataOwner and storageOwner only
/// establish lifetime. offset is relative to storageOwner (host) or to the
/// spill file represented by storageOwner (disk).
struct CudfPackedMicroBucketExtent {
  enum class Tier {
    kHost,
    kDisk,
  };

  Tier tier{Tier::kHost};
  std::shared_ptr<const void> metadataOwner;
  std::shared_ptr<void> storageOwner;
  uint64_t offset{0};
  uint64_t storedBytes{0};
  uint64_t restoreBytes{0};
};

/// Identifies the exact partitioning contract. Join build/probe descriptors
/// may only be aligned by prefix when all fields match. schemaFingerprint is
/// for normalized hash-key types, not the full (and usually different) build
/// or probe payload schema.
struct CudfPackedPartitioningIdentity {
  uint32_t hashVersion{0};
  uint32_t hashSeed{0};
  uint64_t keySchemaFingerprint{0};
  uint64_t generation{0};

  bool operator==(const CudfPackedPartitioningIdentity& other) const = default;
};

/// A logical hash range whose payload is already divided into independently
/// restorable packed images. hashPrefix contains hashBitCount low-order bits
/// starting at hashBitOffset. A future operator can initially materialize many
/// fine-grained descriptors, then combine them in host metadata without a
/// host-side gather or a GPU restore/repartition/spill round trip.
struct CudfPackedMicroBucketDescriptor {
  uint64_t id{0};
  CudfPackedPartitioningIdentity partitioning;
  uint32_t hashPrefix{0};
  uint8_t hashBitOffset{0};
  uint8_t hashBitCount{0};
  uint64_t rows{0};
  uint64_t restoreBytes{0};
  uint64_t storedBytes{0};
  std::vector<CudfPackedMicroBucketExtent> extents;
};

/// One admission wave. Buckets are never split: all their packed images must
/// be available to the operator together. oversized is explicit so the caller
/// can select an operator-specific skew path instead of accidentally exceeding
/// the device budget.
struct CudfPackedMicroBucketRestoreWave {
  std::vector<size_t> bucketIndices;
  /// Actual packed payload bytes copied to device.
  uint64_t restoreBytes{0};
  /// restoreBytes plus per-bucket workspace. fixedReserveBytes remains global.
  uint64_t admissionBytes{0};
  bool oversized{false};
};

/// A consecutive extent range for operators that can consume a logical bucket
/// incrementally (for example ROW_NUMBER=1 local reduction or Join probe).
/// Extents are atomic; no packed image is split or decoded on host.
struct CudfPackedMicroBucketRestoreSlice {
  size_t bucketIndex{0};
  size_t extentBegin{0};
  size_t extentEnd{0};
  uint64_t restoreBytes{0};
  uint64_t admissionBytes{0};
  bool first{false};
  bool last{false};
  bool oversized{false};
};

struct CudfPackedMicroBucketPlanOptions {
  /// Space reserved for concatenate, hash-table or operator output workspace.
  uint64_t fixedReserveBytes{0};
  /// Additional workspace charged for every bucket admitted in a wave.
  uint64_t perBucketReserveBytes{0};
  /// Zero means unlimited. This can bound metadata/kernel launch fan-in.
  size_t maxBucketsPerWave{0};
  /// Process lower-footprint waves first to reduce early device pressure.
  bool smallestWaveFirst{true};
};

/// Validates descriptor accounting without looking into packed payload bytes.
/// Throws std::invalid_argument for malformed hash ranges or byte totals.
void validateCudfPackedMicroBucket(
    const CudfPackedMicroBucketDescriptor& bucket);

/// Rejects build/probe or partial/final descriptors that cannot be matched by
/// host metadata alone. This check never inspects payload bytes.
void validateCompatibleCudfPackedMicroBuckets(
    const CudfPackedMicroBucketDescriptor& left,
    const CudfPackedMicroBucketDescriptor& right);

/// Chooses a power-of-two initial fanout from observed packed bytes. The
/// result is the number of hash bits, not the partition count. This is an
/// estimate only: operators still need a skew path for a single dominant key.
uint8_t chooseCudfPackedMicroBucketHashBits(
    uint64_t packedBytes,
    uint64_t targetBucketBytes,
    uint8_t minimumBits,
    uint8_t maximumBits);

/// Best-fit-decreasing bin packing under a dynamic restore budget. The plan is
/// deterministic, does not copy payloads and never splits a logical bucket.
/// Buckets that cannot fit after fixed/per-bucket reserves are emitted alone
/// with oversized=true for operator-specific skew handling.
std::vector<CudfPackedMicroBucketRestoreWave>
planCudfPackedMicroBucketRestoreWaves(
    const std::vector<CudfPackedMicroBucketDescriptor>& buckets,
    uint64_t deviceBudgetBytes,
    const CudfPackedMicroBucketPlanOptions& options = {});

/// Plans extent-preserving slices for a streaming operator. Buckets are
/// visited smallest first when requested, but extent order within each bucket
/// is unchanged. A single extent larger than the available budget is surfaced
/// as oversized for operator fallback.
std::vector<CudfPackedMicroBucketRestoreSlice>
planCudfPackedMicroBucketStreamingRestore(
    const std::vector<CudfPackedMicroBucketDescriptor>& buckets,
    uint64_t deviceBudgetBytes,
    const CudfPackedMicroBucketPlanOptions& options = {});

} // namespace facebook::velox::cudf_velox
