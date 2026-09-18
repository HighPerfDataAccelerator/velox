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

#include <sys/mman.h>
#include <cstdint>
#include <cstring>
#include <memory>

#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

// Request-owned transport storage, never pooled or shared with another receive.
// UCXX can retain the request and replay a receive after its first completion.
// On successful receive only the serialized prefix is meaningful. Anonymous
// mappings preserve zero-filled tail semantics without eagerly touching all
// 1 MiB. In particular, do not replace this with uninitialized malloc storage:
// UCXX does not expose the received tag length to the completion callback.
class MetadataReceiveBuffer {
 private:
  struct Deleter {
    bool mapped{false};
    void operator()(uint8_t* bytes) const {
      if (mapped) {
        ::munmap(bytes, kMaxMetaBufSize);
      } else {
        delete[] bytes;
      }
    }
  };

  using Owner = std::unique_ptr<uint8_t[], Deleter>;

  static Owner allocate(bool initializePayload) {
    if (!initializePayload) {
      auto* mapping = ::mmap(
          nullptr,
          kMaxMetaBufSize,
          PROT_READ | PROT_WRITE,
          MAP_PRIVATE | MAP_ANONYMOUS,
          -1,
          0);
      if (mapping != MAP_FAILED) {
        return Owner(static_cast<uint8_t*>(mapping), Deleter{true});
      }
    }
    // Same initialized semantics on allocation fallback and in the control.
    return Owner(new uint8_t[kMaxMetaBufSize](), Deleter{false});
  }

 public:
  explicit MetadataReceiveBuffer(bool initializePayload)
      : bytes_(allocate(initializePayload)) {}

  uint8_t* data() const {
    return bytes_.get();
  }

  static constexpr size_t size() {
    return kMaxMetaBufSize;
  }

  static constexpr size_t kHeaderBytes =
      sizeof(kMagicNumber) + sizeof(uint32_t);

 private:
  Owner bytes_;
};

} // namespace facebook::velox::ucx_exchange
