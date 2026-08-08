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

#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::ucx_exchange {

namespace {
class UcxPinnedBufferPool {
 public:
  UcxPinnedBufferPool(const char* countEnv, const char* label)
      : label_{label} {
    if (const char* value = std::getenv(countEnv)) {
      char* end = nullptr;
      const auto requested = std::strtoull(value, &end, 10);
      if (end != value && *end == '\0') {
        maxSlots_ = std::clamp<uint64_t>(requested, 1, 16);
      }
    }
  }

  std::shared_ptr<uint8_t> acquire(uint64_t requiredBytes) {
    if (requiredBytes == 0) {
      return {};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    size_t available = slots_.size();
    for (size_t i = 0; i < slots_.size(); ++i) {
      if (!slots_[i]->busy && slots_[i]->capacity >= requiredBytes) {
        slots_[i]->busy = true;
        return makeOwner(i);
      }
      if (!slots_[i]->busy && available == slots_.size()) {
        available = i;
      }
    }
    if (available == slots_.size() && slots_.size() < maxSlots_) {
      slots_.push_back(std::make_unique<Slot>());
      available = slots_.size() - 1;
    }
    if (available == slots_.size()) {
      return {};
    }

    auto& slot = *slots_[available];
    uint8_t* allocation = nullptr;
    const auto allocationBytes =
        std::max<uint64_t>(64ULL << 20, requiredBytes);
    const auto status = cudaHostAlloc(
        reinterpret_cast<void**>(&allocation),
        allocationBytes,
        cudaHostAllocPortable);
    if (status != cudaSuccess) {
      cudaGetLastError();
      LOG(WARNING) << "Could not allocate " << allocationBytes
                   << " bytes for UCX " << label_
                   << " pinned bounce slot " << available
                   << ": " << cudaGetErrorString(status)
                   << "; falling back to pageable staging";
      return {};
    }
    if (slot.data != nullptr) {
      cudaFreeHost(slot.data);
    }
    slot.data = allocation;
    slot.capacity = allocationBytes;
    slot.busy = true;
    return makeOwner(available);
  }

  ~UcxPinnedBufferPool() {
    for (auto& slot : slots_) {
      if (slot->data != nullptr) {
        cudaFreeHost(slot->data);
      }
    }
  }

 private:
  struct Slot {
    uint8_t* data{nullptr};
    uint64_t capacity{0};
    bool busy{false};
  };

  struct Lease {
    Lease(UcxPinnedBufferPool* pool, size_t slot) : pool(pool), slot(slot) {}
    ~Lease() {
      std::lock_guard<std::mutex> lock(pool->mutex_);
      VELOX_CHECK(pool->slots_[slot]->busy);
      pool->slots_[slot]->busy = false;
    }
    UcxPinnedBufferPool* pool;
    size_t slot;
  };

  std::shared_ptr<uint8_t> makeOwner(size_t slot) {
    auto lease = std::make_shared<Lease>(this, slot);
    return std::shared_ptr<uint8_t>(lease, slots_[slot]->data);
  }

  std::mutex mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
  size_t maxSlots_{4};
  const char* const label_;
};

UcxPinnedBufferPool& ucxPinnedBufferPool() {
  static UcxPinnedBufferPool pool(
      "GLUTEN_UCX_PINNED_BUFFER_COUNT", "transport");
  return pool;
}

UcxPinnedBufferPool& ucxH2DPinnedBufferPool() {
  static UcxPinnedBufferPool pool(
      "GLUTEN_UCX_H2D_PINNED_BUFFER_COUNT", "H2D");
  return pool;
}
} // namespace

std::shared_ptr<uint8_t> acquireUcxPinnedBuffer(uint64_t requiredBytes) {
  return ucxPinnedBufferPool().acquire(requiredBytes);
}

std::shared_ptr<uint8_t> acquireUcxH2DPinnedBuffer(uint64_t requiredBytes) {
  return ucxH2DPinnedBufferPool().acquire(requiredBytes);
}

uint32_t fnv1a_32(std::string_view s) {
  uint32_t hash = 0x811C9DC5u; // FNV offset basis
  for (unsigned char c : s) {
    hash ^= c;
    hash *= 0x01000193u; // FNV prime
  }
  return hash;
}

std::pair<std::shared_ptr<uint8_t>, size_t> MetadataMsg::serialize() {
  uint32_t totalSize = getSerializedSize();

  VELOX_CHECK_LE(
      totalSize,
      kMaxMetaBufSize,
      "Metadata serialized size ({}) exceeds maximum buffer size ({}). "
      "This can happen with extremely wide tables. "
      "Consider reducing table width or increasing kMaxMetaBufSize.",
      totalSize,
      kMaxMetaBufSize);

  auto deleter = [](uint8_t* p) { delete[] p; };
  std::shared_ptr<uint8_t> buffer(new uint8_t[totalSize], deleter);

  uint8_t* ptr = buffer.get();

  std::memcpy(ptr, &kMagicNumber, sizeof(kMagicNumber));
  ptr += sizeof(kMagicNumber);

  std::memcpy(ptr, &totalSize, sizeof(totalSize));
  ptr += sizeof(totalSize);

  WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
  std::memcpy(ptr, &cudfSize, sizeof(cudfSize));
  ptr += sizeof(cudfSize);

  if (cudfSize > 0) {
    std::memcpy(ptr, cudfMetadata->data(), cudfSize);
    ptr += cudfSize;
  }

  std::memcpy(ptr, &dataSizeBytes, sizeof(dataSizeBytes));
  ptr += sizeof(dataSizeBytes);

  WireLengthType numRemaining = remainingBytes.size();
  std::memcpy(ptr, &numRemaining, sizeof(numRemaining));
  ptr += sizeof(numRemaining);

  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(remainingBytes[0]);
    std::memcpy(ptr, remainingBytes.data(), bytesSize);
    ptr += bytesSize;
  }

  uint8_t atEndByte = atEnd ? 1 : 0;
  *ptr = atEndByte;

  return std::make_pair<std::shared_ptr<uint8_t>, size_t>(
      std::move(buffer), totalSize);
}

MetadataMsg MetadataMsg::deserializeMetadataMsg(const uint8_t* buffer) {
  const uint8_t* ptr = buffer;

  MetadataMsg record;

  uint32_t magicNumber = 0;
  std::memcpy(&magicNumber, ptr, sizeof(magicNumber));
  VELOX_CHECK_EQ(magicNumber, kMagicNumber);
  ptr += sizeof(magicNumber);

  uint32_t totalSize = 0;
  std::memcpy(&totalSize, ptr, sizeof(totalSize));
  ptr += sizeof(totalSize);

  const uint8_t* endPtr = buffer + totalSize;

  WireLengthType metaSize = 0;
  if (ptr + sizeof(metaSize) > endPtr)
    throw std::runtime_error("Insufficient data for cudfMetadata size");
  std::memcpy(&metaSize, ptr, sizeof(metaSize));
  ptr += sizeof(metaSize);

  record.cudfMetadata = std::make_unique<std::vector<uint8_t>>(metaSize);
  if (metaSize > 0) {
    if (ptr + metaSize > endPtr)
      throw std::runtime_error("Insufficient data for cudfMetadata bytes");
    std::memcpy(record.cudfMetadata->data(), ptr, metaSize);
    ptr += metaSize;
  }

  if (ptr + sizeof(record.dataSizeBytes) > endPtr)
    throw std::runtime_error("Insufficient data for dataSizeBytes");
  std::memcpy(&record.dataSizeBytes, ptr, sizeof(record.dataSizeBytes));
  ptr += sizeof(record.dataSizeBytes);

  WireLengthType numRemaining = 0;
  if (ptr + sizeof(numRemaining) > endPtr)
    throw std::runtime_error("Insufficient data for remainingBytes count");
  std::memcpy(&numRemaining, ptr, sizeof(numRemaining));
  ptr += sizeof(numRemaining);

  record.remainingBytes.resize(numRemaining);
  if (numRemaining > 0) {
    auto bytesSize = numRemaining * sizeof(record.remainingBytes[0]);
    if (ptr + bytesSize > endPtr)
      throw std::runtime_error("Insufficient data for remainingBytes values");
    std::memcpy(record.remainingBytes.data(), ptr, bytesSize);
    ptr += bytesSize;
  }

  if (ptr + 1 > endPtr) {
    throw std::runtime_error("Insufficient data for atEnd flag");
  }
  record.atEnd = (*ptr != 0);

  return record;
}

} // namespace facebook::velox::ucx_exchange
