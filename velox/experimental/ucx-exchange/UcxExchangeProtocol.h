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

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <memory>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

/// Definitions needed for the Ucx exchange protocol.
///
/// Byte order: all multi-byte fields are serialized with std::memcpy, which
/// preserves host byte order. The protocol assumes matching endianness between
/// peers (little-endian on x86 and ARM). Cross-endian transfers are not
/// supported.

namespace facebook::velox::ucx_exchange {

// Data and metadata tags are a uint64_t split into 3 fields, most-significant
// first:
// - Bits 63..32 (4 bytes): FNV-1a hash of the producing taskId, which is
//   unique within a cluster.
// - Bits 31..24 (1 byte): Operation type (metadata, data, or handshake
//   response).
// - Bits 23..0  (3 bytes): Sequence number of the chunk exchanged between 2
//   tasks.

// Definition of the operations.
constexpr uint64_t METADATA_TAG = 0x02000000;
constexpr uint64_t DATA_TAG = 0x03000000;
constexpr uint64_t HANDSHAKE_RESPONSE_TAG = 0x04000000;

// A CUDA-capable UCP context does not guarantee that every endpoint lane can
// execute the eager "short" protocol directly from device memory. TCP/SM can
// select uct_*_am_short for a tiny packed table and call memcpy on the device
// address before cuda_copy participates. Sender and receiver must make the
// same host-staging decision.
constexpr int64_t kDeviceEagerHostStageBytes = 64 << 10;

inline int64_t maxDirectDeviceTransferBytes() {
  if (const char* value =
          std::getenv("GLUTEN_UCX_DIRECT_DEVICE_MAX_BYTES")) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 0) {
      return parsed;
    }
  }
  return 0;
}

inline bool shouldHostStageDeviceTransfer(
    bool hasCudaTransport,
    int64_t bytes) {
  if (const char* value = std::getenv("GLUTEN_UCX_FORCE_HOST_STAGING");
      value != nullptr && value[0] != '\0') {
    if (value[0] != '0') {
      return true;
    }
  }
  const auto directLimit = maxDirectDeviceTransferBytes();
  return !hasCudaTransport || bytes <= kDeviceEagerHostStageBytes ||
      (directLimit > 0 && bytes > directLimit);
}

/// Acquires one reusable pinned host buffer dedicated to UCX bounce traffic.
/// The exchange pool is intentionally separate from packed spill/restore: a
/// long rendezvous send must not consume the small pool that Grace Join and
/// TopN need to overlap disk decode with H2D. Returns nullptr when every
/// bounded slot is busy so callers can preserve correctness with pageable
/// staging.
std::shared_ptr<uint8_t> acquireUcxPinnedBuffer(uint64_t requiredBytes);

/// Acquires a pinned scratch buffer used only for deferred host-to-device
/// copies. Remote UCX receive buffers deliberately remain pageable and owned
/// by their retained Request so a wireup replay cannot overwrite a recycled
/// pool slot. Keeping this pool separate also prevents long D2H sends from
/// starving the short-lived H2D copy engine bounce.
std::shared_ptr<uint8_t> acquireUcxH2DPinnedBuffer(uint64_t requiredBytes);

inline bool exchangeVariableWidthValidationEnabled() {
  const char* value =
      std::getenv("GLUTEN_CUDF_BATCH_CONCAT_VALIDATE_VARIABLE_WIDTH");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string_view setting(value);
  return setting != "0" && setting != "false" && setting != "FALSE";
}

// Implementation of the fowler-noll-vo hash function for 32 bits.
uint32_t fnv1a_32(std::string_view s);

// Low-overhead diagnostic fingerprint for large exchange buffers. It samples
// up to 4,096 evenly spaced 64-bit words, which is sufficient to distinguish
// cross-wired pages without adding a full extra memory-bandwidth pass.
inline uint64_t diagnosticBufferFingerprint(const void* data, size_t size) {
  constexpr uint64_t kOffsetBasis = 1469598103934665603ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  if (data == nullptr || size == 0) {
    return kOffsetBasis ^ size;
  }
  const auto* bytes = static_cast<const uint8_t*>(data);
  const size_t words = size / sizeof(uint64_t);
  const size_t samples = std::min<size_t>(words, 4096);
  uint64_t hash = kOffsetBasis ^ size;
  for (size_t i = 0; i < samples; ++i) {
    const size_t wordIndex = samples == words ? i : (i * words) / samples;
    uint64_t value;
    std::memcpy(&value, bytes + wordIndex * sizeof(uint64_t), sizeof(value));
    hash = (hash ^ value) * kPrime;
  }
  return hash;
}

// Gets the tag used for metadata communication
// Note: taskHash and sequenceNumber are implicitly converted to 64 bits.
inline uint64_t getMetadataTag(uint64_t taskHash, uint64_t sequenceNumber) {
  return (taskHash << 32) | METADATA_TAG | sequenceNumber;
}

// Gets the tag used for data communication
// Note: taskHash and sequenceNumber are implicitly converted to 64 bits.
inline uint64_t getDataTag(uint64_t taskHash, uint64_t sequenceNumber) {
  return (taskHash << 32) | DATA_TAG | sequenceNumber;
}

// Gets the tag used for handshake response communication.
// Note: taskHash is implicitly converted to 64 bits.
inline uint64_t getHandshakeResponseTag(uint64_t taskHash) {
  return (taskHash << 32) | HANDSHAKE_RESPONSE_TAG;
}

/// @brief Request that is sent from the client (UcxExchangeSource) to the
/// server (UcxExchangeServer) after connection.
///
/// The handshake establishes the partition key for data exchange.
/// The workerId identifies the source's Communicator instance (process).
/// If the server's workerId matches, both are in the same process, enabling
/// intra-node transfer via IntraNodeTransferRegistry instead of UCXX.
struct HandshakeMsg {
  char taskId[256];
  uint32_t destination;
  /// Unique identifier for the source's Communicator instance.
  /// Generated randomly at Communicator startup. The server compares this
  /// against its own workerId to detect same-process (intra-node) transfers.
  uint64_t workerId{0};
};

/// Marks an active message as a request to discard one destination's output.
/// Real destination IDs are restricted to [0, 65536), leaving the high bit
/// available without changing the wire layout.
constexpr uint32_t kCancelDestinationFlag = 0x80000000U;

/// @brief Response sent from server to source after handshake.
/// Informs the source whether intra-node transfer optimization is available,
/// allowing the source to bypass UCXX for all subsequent data transfers.
struct HandshakeResponse {
  /// True if server and source are on the same node (same Communicator).
  /// When true, source should use IntraNodeTransferRegistry instead of UCXX.
  bool isIntraNodeTransfer{false};
  /// Padding for alignment
  uint8_t padding[7]{};
};

constexpr uint32_t kMagicNumber = 0x12345678;
/// Maximum metadata buffer size for receiving. This should be large enough
/// to handle tables with many columns. 1MB allows for ~10,000+ columns.
/// The sender allocates exact size needed; receiver pre-allocates this max.
constexpr uint32_t kMaxMetaBufSize = 1024 * 1024; // 1MB

/// Minimum header size needed to read the totalSize field.
/// Format: [magic (4 bytes)][totalSize (4 bytes)]
constexpr uint32_t kMetaHeaderSize = sizeof(kMagicNumber) + sizeof(uint32_t);

/// Wire-format types for MetadataMsg serialization. Using shared type aliases
/// ensures serialize() and deserializeMetadataMsg() agree on field widths.
using WireLengthType = uint64_t;
using WireDataSizeType = int64_t;
using WireRemainingElementType = int64_t;

struct MetadataMsg {
  std::unique_ptr<std::vector<uint8_t>> cudfMetadata;
  WireDataSizeType dataSizeBytes;
  std::vector<WireRemainingElementType> remainingBytes;
  bool atEnd;

  uint32_t getSerializedSize() const {
    // The header: the magic number and the metadata length.
    uint32_t totalSize = sizeof(kMagicNumber) + sizeof(totalSize);
    // cudfMetadata: length info and then the data.
    WireLengthType cudfSize = cudfMetadata ? cudfMetadata->size() : 0;
    totalSize += sizeof(cudfSize);
    totalSize += cudfSize;
    // dataSizeBytes
    totalSize += sizeof(dataSizeBytes);
    // remainingBytes: length and then the data.
    totalSize += sizeof(WireLengthType); // for numRemaining count
    totalSize += remainingBytes.size() * sizeof(remainingBytes[0]);
    // atEnd, encoded in a byte.
    totalSize += sizeof(uint8_t);

    return totalSize;
  }

  /// Serializes this metadata record into a newly allocated buffer.
  std::pair<std::shared_ptr<uint8_t>, size_t> serialize();

  /// Deserializes a MetadataMsg from a buffer produced by serialize().
  static MetadataMsg deserializeMetadataMsg(const uint8_t* buffer);
};

} // namespace facebook::velox::ucx_exchange
