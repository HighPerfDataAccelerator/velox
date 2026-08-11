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

#include "velox/experimental/cudf/exec/CudfPackedMicroBucket.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace facebook::velox::cudf_velox {
namespace {

uint64_t checkedAdd(uint64_t left, uint64_t right, const char* context) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::overflow_error(
        std::string("Packed micro-bucket ") + context +
        " byte accounting overflow");
  }
  return left + right;
}

uint64_t chargedBytes(
    const CudfPackedMicroBucketDescriptor& bucket,
    const CudfPackedMicroBucketPlanOptions& options) {
  return checkedAdd(
      bucket.restoreBytes, options.perBucketReserveBytes, "restore");
}

} // namespace

void validateCudfPackedMicroBucket(
    const CudfPackedMicroBucketDescriptor& bucket) {
  if (static_cast<uint32_t>(bucket.hashBitOffset) + bucket.hashBitCount > 32) {
    throw std::invalid_argument(
        "Packed micro-bucket hash range exceeds a 32-bit hash");
  }
  if (bucket.hashBitCount < 32) {
    const auto prefixLimit = uint64_t{1} << bucket.hashBitCount;
    if (bucket.hashPrefix >= prefixLimit) {
      throw std::invalid_argument(
          "Packed micro-bucket hash prefix does not fit in its bit range");
    }
  }

  uint64_t restoreBytes = 0;
  uint64_t storedBytes = 0;
  for (const auto& extent : bucket.extents) {
    if ((extent.restoreBytes != 0 || extent.storedBytes != 0) &&
        extent.storageOwner == nullptr) {
      throw std::invalid_argument(
          "Non-empty packed micro-bucket extent has no storage owner");
    }
    if (extent.restoreBytes != 0 && extent.metadataOwner == nullptr) {
      throw std::invalid_argument(
          "Non-empty packed micro-bucket extent has no metadata owner");
    }
    restoreBytes =
        checkedAdd(restoreBytes, extent.restoreBytes, "restore extent");
    storedBytes = checkedAdd(storedBytes, extent.storedBytes, "stored extent");
  }
  if (restoreBytes != bucket.restoreBytes) {
    throw std::invalid_argument(
        "Packed micro-bucket restore byte total differs from its extents");
  }
  if (storedBytes != bucket.storedBytes) {
    throw std::invalid_argument(
        "Packed micro-bucket stored byte total differs from its extents");
  }
}

void validateCompatibleCudfPackedMicroBuckets(
    const CudfPackedMicroBucketDescriptor& left,
    const CudfPackedMicroBucketDescriptor& right) {
  validateCudfPackedMicroBucket(left);
  validateCudfPackedMicroBucket(right);
  if (left.partitioning != right.partitioning) {
    throw std::invalid_argument(
        "Packed micro-buckets use different partitioning identities");
  }
  if (left.hashBitOffset != right.hashBitOffset ||
      left.hashBitCount != right.hashBitCount ||
      left.hashPrefix != right.hashPrefix) {
    throw std::invalid_argument(
        "Packed micro-buckets describe different hash ranges");
  }
}

uint8_t chooseCudfPackedMicroBucketHashBits(
    uint64_t packedBytes,
    uint64_t targetBucketBytes,
    uint8_t minimumBits,
    uint8_t maximumBits) {
  if (targetBucketBytes == 0) {
    throw std::invalid_argument(
        "Packed micro-bucket target size must be non-zero");
  }
  if (minimumBits > maximumBits || maximumBits > 32) {
    throw std::invalid_argument("Invalid packed micro-bucket hash bit range");
  }

  const auto requiredBuckets = packedBytes / targetBucketBytes +
      static_cast<uint64_t>(packedBytes % targetBucketBytes != 0);
  uint8_t bits = 0;
  uint64_t capacity = 1;
  while (capacity < requiredBuckets && bits < maximumBits) {
    capacity <<= 1;
    ++bits;
  }
  return std::clamp(bits, minimumBits, maximumBits);
}

std::vector<CudfPackedMicroBucketRestoreWave>
planCudfPackedMicroBucketRestoreWaves(
    const std::vector<CudfPackedMicroBucketDescriptor>& buckets,
    uint64_t deviceBudgetBytes,
    const CudfPackedMicroBucketPlanOptions& options) {
  if (deviceBudgetBytes <= options.fixedReserveBytes) {
    throw std::invalid_argument(
        "Packed micro-bucket restore budget must exceed the fixed reserve");
  }
  const auto waveBudget = deviceBudgetBytes - options.fixedReserveBytes;

  std::vector<size_t> order(buckets.size());
  std::iota(order.begin(), order.end(), 0);
  for (const auto& bucket : buckets) {
    validateCudfPackedMicroBucket(bucket);
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    return buckets[left].restoreBytes > buckets[right].restoreBytes;
  });

  std::vector<CudfPackedMicroBucketRestoreWave> waves;
  for (const auto index : order) {
    const auto bytes = chargedBytes(buckets[index], options);
    if (bytes > waveBudget) {
      waves.push_back({{index}, buckets[index].restoreBytes, bytes, true});
      continue;
    }

    size_t best = waves.size();
    uint64_t leastRemaining = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < waves.size(); ++i) {
      auto& wave = waves[i];
      if (wave.oversized ||
          (options.maxBucketsPerWave != 0 &&
           wave.bucketIndices.size() >= options.maxBucketsPerWave) ||
          wave.admissionBytes > waveBudget - bytes) {
        continue;
      }
      const auto remaining = waveBudget - wave.admissionBytes - bytes;
      if (remaining < leastRemaining) {
        leastRemaining = remaining;
        best = i;
      }
    }
    if (best == waves.size()) {
      waves.push_back({{index}, buckets[index].restoreBytes, bytes, false});
    } else {
      waves[best].bucketIndices.push_back(index);
      waves[best].restoreBytes = checkedAdd(
          waves[best].restoreBytes,
          buckets[index].restoreBytes,
          "restore wave");
      waves[best].admissionBytes += bytes;
    }
  }

  if (options.smallestWaveFirst) {
    std::stable_sort(
        waves.begin(), waves.end(), [](const auto& left, const auto& right) {
          if (left.oversized != right.oversized) {
            return !left.oversized;
          }
          return left.admissionBytes < right.admissionBytes;
        });
  }
  return waves;
}

std::vector<CudfPackedMicroBucketRestoreSlice>
planCudfPackedMicroBucketStreamingRestore(
    const std::vector<CudfPackedMicroBucketDescriptor>& buckets,
    uint64_t deviceBudgetBytes,
    const CudfPackedMicroBucketPlanOptions& options) {
  if (deviceBudgetBytes <= options.fixedReserveBytes) {
    throw std::invalid_argument(
        "Packed micro-bucket restore budget must exceed the fixed reserve");
  }
  const auto waveBudget = deviceBudgetBytes - options.fixedReserveBytes;
  if (waveBudget <= options.perBucketReserveBytes) {
    throw std::invalid_argument(
        "Packed micro-bucket streaming budget must exceed per-bucket reserve");
  }
  const auto payloadBudget = waveBudget - options.perBucketReserveBytes;

  std::vector<size_t> order(buckets.size());
  std::iota(order.begin(), order.end(), 0);
  for (const auto& bucket : buckets) {
    validateCudfPackedMicroBucket(bucket);
  }
  if (options.smallestWaveFirst) {
    std::stable_sort(
        order.begin(), order.end(), [&](size_t left, size_t right) {
          return buckets[left].restoreBytes < buckets[right].restoreBytes;
        });
  }

  std::vector<CudfPackedMicroBucketRestoreSlice> slices;
  for (const auto bucketIndex : order) {
    const auto& bucket = buckets[bucketIndex];
    size_t begin = 0;
    while (begin < bucket.extents.size()) {
      size_t end = begin;
      uint64_t bytes = 0;
      while (end < bucket.extents.size()) {
        const auto extentBytes = bucket.extents[end].restoreBytes;
        if (end != begin && extentBytes > payloadBudget - bytes) {
          break;
        }
        bytes = checkedAdd(bytes, extentBytes, "streaming restore slice");
        ++end;
        if (bytes >= payloadBudget) {
          break;
        }
      }
      const auto admissionBytes = checkedAdd(
          bytes, options.perBucketReserveBytes, "streaming admission");
      slices.push_back(
          CudfPackedMicroBucketRestoreSlice{
              bucketIndex,
              begin,
              end,
              bytes,
              admissionBytes,
              begin == 0,
              end == bucket.extents.size(),
              admissionBytes > waveBudget});
      begin = end;
    }
  }
  return slices;
}

} // namespace facebook::velox::cudf_velox
