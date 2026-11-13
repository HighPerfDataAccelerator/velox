/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License writtenBytes
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/exec/SharedBroadcastValues.h"
#include "velox/vector/FlatVector.h"
#if defined(__linux__)
#include <fcntl.h>
#include <fmt/format.h>
#include <folly/hash/SpookyHashV2.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace facebook::velox::exec::detail {
namespace {
struct File {
  int descriptor;
  explicit File(int value) : descriptor(value) {}
  ~File() {
    if (descriptor >= 0) {
      ::close(descriptor);
    }
  }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
};
struct Mapping {
  void* address = MAP_FAILED;
  size_t length = 0;
  ~Mapping() {
    if (address != MAP_FAILED) {
      ::munmap(address, length);
    }
  }
};
struct MappingRef {
  std::shared_ptr<Mapping> mapping;
  // shared_ptr owns the reference; BufferView's explicit hooks are no-ops.
  void addRef() const {}
  void release() const {}
};
inline bool privateDirectory(const std::string& path) {
  if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
    return false;
  }
  struct stat fileStat{};
  return ::lstat(path.c_str(), &fileStat) == 0 && S_ISDIR(fileStat.st_mode) &&
      fileStat.st_uid == ::geteuid() && (fileStat.st_mode & 0077) == 0;
}
inline bool writeAll(int descriptor, const uint8_t* data, size_t bytes) {
  size_t writtenBytes = 0;
  while (writtenBytes < bytes) {
    const auto written = ::pwrite(
        descriptor,
        data + writtenBytes,
        bytes - writtenBytes,
        static_cast<off_t>(writtenBytes));
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    writtenBytes += static_cast<size_t>(written);
  }
  return true;
}
inline BufferPtr shareBuffer(const uint8_t* data, size_t bytes, TypeKind kind) {
  const char* directory = std::getenv("VELOX_SHARED_BROADCAST_CACHE_DIR");
  if (!directory || directory[0] != '/' || bytes < (1U << 20)) {
    return nullptr;
  }
  unsigned cpu = 0, node = 0;
  if (::syscall(SYS_getcpu, &cpu, &node, nullptr) != 0) {
    return nullptr;
  }
  const std::string root(directory);
  const std::string local = root + "/node-" + std::to_string(node);
  if (!privateDirectory(root) || !privateDirectory(local)) {
    return nullptr;
  }
  const uint64_t identity[] = {
      1, static_cast<uint64_t>(kind), static_cast<uint64_t>(bytes)};
  // The hash locates a candidate file. Byte comparison below decides whether
  // it can be reused, so a collision only falls back to the local values.
  folly::hash::SpookyHashV2 hash;
  hash.Init(0, 0);
  hash.Update(identity, sizeof(identity));
  hash.Update(data, bytes);
  uint64_t digest[2];
  hash.Final(&digest[0], &digest[1]);
  const auto key =
      fmt::format("spooky-v1-{:016x}{:016x}", digest[0], digest[1]);
  const std::string path = local + "/" + key + ".bin";
  File lock(
      ::open(
          (path + ".lock").c_str(),
          O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW,
          0600));
  if (lock.descriptor < 0) {
    return nullptr;
  }
  while (::flock(lock.descriptor, LOCK_EX) != 0) {
    if (errno != EINTR) {
      return nullptr;
    }
  }
  const size_t padded = ((bytes + 63) & ~size_t(63)) + simd::kPadding;
  bool created = false;
  int descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0 && errno == ENOENT) {
    descriptor = ::open(
        path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    created = descriptor >= 0;
  }
  File file(descriptor);
  if (descriptor < 0) {
    return nullptr;
  }
  struct stat fileStat{};
  if (::fstat(descriptor, &fileStat) != 0 || !S_ISREG(fileStat.st_mode) ||
      fileStat.st_uid != ::geteuid()) {
    return nullptr;
  }
  if (created &&
      (::ftruncate(descriptor, static_cast<off_t>(padded)) != 0 ||
       !writeAll(descriptor, data, bytes))) {
    ::unlink(path.c_str()); // This process alone created this unpublished file.
    return nullptr;
  }
  if (::fstat(descriptor, &fileStat) != 0 ||
      fileStat.st_size != static_cast<off_t>(padded)) {
    return nullptr;
  }
  // No helper rewrites an existing inode. Release the creation lock before
  // mapping and validating so readers can run concurrently.
  if (::flock(lock.descriptor, LOCK_UN) != 0) {
    return nullptr;
  }
  auto owner = std::make_shared<Mapping>();
  owner->length = padded;
  owner->address =
      ::mmap(nullptr, padded, PROT_READ, MAP_SHARED, descriptor, 0);
  if (owner->address == MAP_FAILED ||
      std::memcmp(owner->address, data, bytes) != 0) {
    return nullptr;
  }
  auto result = BufferView<MappingRef>::create(
      static_cast<const uint8_t*>(owner->address), bytes, MappingRef{owner});
  return result;
}
template <typename T>
inline VectorPtr shareTyped(const VectorPtr& input) {
  const size_t bytes = size_t(input->size()) * sizeof(T);
  auto mapped = shareBuffer(
      input->values()->template as<uint8_t>(), bytes, input->typeKind());
  if (!mapped) {
    return input;
  }
  return std::make_shared<FlatVector<T>>(
      input->pool(),
      input->type(),
      input->nulls(),
      input->size(),
      std::move(mapped),
      std::vector<BufferPtr>{});
}
} // namespace
} // namespace facebook::velox::exec::detail
#endif
namespace facebook::velox::exec::detail {
VectorPtr shareBroadcastValues(const VectorPtr& input) {
#if defined(__linux__)
  if (input && input->isFlatEncoding() && input->values()) {
    if (input->typeKind() == TypeKind::INTEGER) {
      return shareTyped<int32_t>(input);
    }
    if (input->typeKind() == TypeKind::BIGINT) {
      return shareTyped<int64_t>(input);
    }
  }
#endif
  return input;
}
} // namespace facebook::velox::exec::detail
