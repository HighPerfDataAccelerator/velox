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

#include "velox/connectors/hive/storage_adapters/s3fs/S3ReadFile.h"
#include "velox/common/base/StatsReporter.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Counters.h"
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

#include <algorithm>

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>

namespace facebook::velox::filesystems {

namespace {

// By default, the AWS SDK reads object data into an auto-growing StringStream.
// To avoid copies, read directly into a pre-allocated buffer instead.
// See https://github.com/aws/aws-sdk-cpp/issues/64 for an alternative but
// functionally similar recipe.
Aws::IOStreamFactory AwsWriteableStreamFactory(void* data, int64_t nbytes) {
  return [=]() { return Aws::New<StringViewStream>("", data, nbytes); };
}

// A streambuf that scatters sequential bytes into multiple target buffers.
// Bytes falling in null (gap) ranges are consumed but discarded.
class ScatterBuf : public std::streambuf {
 public:
  explicit ScatterBuf(const std::vector<folly::Range<char*>>& buffers)
      : buffers_(buffers) {}

 protected:
  std::streamsize xsputn(const char_type* s, std::streamsize count) override {
    std::streamsize consumed = 0;
    while (consumed < count && rangeIdx_ < buffers_.size()) {
      const auto& range = buffers_[rangeIdx_];
      auto remaining =
          static_cast<std::streamsize>(range.size()) - rangePos_;
      auto n = std::min(remaining, count - consumed);
      if (range.data()) {
        ::memcpy(range.data() + rangePos_, s + consumed, n);
      }
      consumed += n;
      rangePos_ += n;
      if (rangePos_ >= static_cast<std::streamsize>(range.size())) {
        ++rangeIdx_;
        rangePos_ = 0;
      }
    }
    return consumed;
  }

  int_type overflow(int_type ch) override {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      return traits_type::not_eof(ch);
    }
    char_type c = traits_type::to_char_type(ch);
    return xsputn(&c, 1) == 1 ? ch : traits_type::eof();
  }

 private:
  const std::vector<folly::Range<char*>>& buffers_;
  size_t rangeIdx_ = 0;
  std::streamsize rangePos_ = 0;
};

// An iostream wrapping ScatterBuf for use as an Aws::IOStream.
// Follows the StringViewStream pattern: the streambuf is embedded via
// multiple inheritance to avoid a separate heap allocation.
class ScatterStream : ScatterBuf, public std::iostream {
 public:
  explicit ScatterStream(const std::vector<folly::Range<char*>>& buffers)
      : ScatterBuf(buffers), std::iostream(this) {}
};

} // namespace

class S3ReadFile ::Impl {
 public:
  explicit Impl(std::string_view path, Aws::S3::S3Client* client)
      : client_(client) {
    getBucketAndKeyFromPath(path, bucket_, key_);
  }

  // Gets the length of the file.
  // Checks if there are any issues reading the file.
  void initialize(const filesystems::FileOptions& options) {
    if (options.fileSize.has_value()) {
      VELOX_CHECK_GE(
          options.fileSize.value(), 0, "File size must be non-negative");
      length_ = options.fileSize.value();
    }

    // Make it a no-op if invoked twice.
    if (length_ != -1) {
      return;
    }

    Aws::S3::Model::HeadObjectRequest request;
    request.SetBucket(awsString(bucket_));
    request.SetKey(awsString(key_));

    RECORD_METRIC_VALUE(kMetricS3MetadataCalls);
    auto outcome = client_->HeadObject(request);
    if (!outcome.IsSuccess()) {
      RECORD_METRIC_VALUE(kMetricS3GetMetadataErrors);
    }
    RECORD_METRIC_VALUE(kMetricS3GetMetadataRetries, outcome.GetRetryCount());
    VELOX_CHECK_AWS_OUTCOME(
        outcome, "Failed to get metadata for S3 object", bucket_, key_);
    length_ = outcome.GetResult().GetContentLength();
    VELOX_CHECK_GE(length_, 0);
  }

  std::string_view pread(
      uint64_t offset,
      uint64_t length,
      void* buffer,
      const FileIoContext& context) const {
    preadInternal(offset, length, static_cast<char*>(buffer));
    return {static_cast<char*>(buffer), length};
  }

  std::string
  pread(uint64_t offset, uint64_t length, const FileIoContext& context) const {
    std::string result(length, 0);
    char* position = result.data();
    preadInternal(offset, length, position);
    return result;
  }

  uint64_t preadv(
      uint64_t offset,
      const std::vector<folly::Range<char*>>& buffers,
      const FileIoContext& context) const {
    // 'buffers' contains Ranges(data, size) with gaps (data == nullptr).
    // AWS S3 GetObject does not support multi-range, so we issue a single
    // GET spanning all ranges.  A ScatterStream writes the response
    // directly into the caller's buffers, eliminating the temporary
    // allocation + memcpy that the previous implementation required.
    size_t totalSpan = 0;
    size_t bytesNeeded = 0;
    for (const auto& range : buffers) {
      totalSpan += range.size();
      if (range.data()) {
        bytesNeeded += range.size();
      }
    }
    if (bytesNeeded == 0) {
      return 0;
    }
    preadInternal(offset, totalSpan, [&buffers]() {
      return Aws::New<ScatterStream>("S3ScatterStream", buffers);
    });
    return bytesNeeded;
  }

  uint64_t size() const {
    return length_;
  }

  uint64_t memoryUsage() const {
    // TODO: Check if any buffers are being used by the S3 library
    return sizeof(Aws::S3::S3Client) + kS3MaxKeySize + 2 * sizeof(std::string) +
        sizeof(int64_t);
  }

  bool shouldCoalesce() const {
    return true;
  }

  std::string getName() const {
    return fmt::format("s3://{}/{}", bucket_, key_);
  }

 private:
  // Issues an S3 GET for [offset, offset+length) using the given stream
  // factory to receive the response body.
  void preadInternal(
      uint64_t offset,
      uint64_t length,
      Aws::IOStreamFactory streamFactory) const {
    Aws::S3::Model::GetObjectRequest request;
    request.SetBucket(awsString(bucket_));
    request.SetKey(awsString(key_));
    std::stringstream ss;
    ss << "bytes=" << offset << "-" << offset + length - 1;
    request.SetRange(awsString(ss.str()));
    request.SetResponseStreamFactory(std::move(streamFactory));
    RECORD_METRIC_VALUE(kMetricS3ActiveConnections);
    RECORD_METRIC_VALUE(kMetricS3GetObjectCalls);
    auto outcome = client_->GetObject(request);
    if (!outcome.IsSuccess()) {
      RECORD_METRIC_VALUE(kMetricS3GetObjectErrors);
    }
    RECORD_METRIC_VALUE(kMetricS3GetObjectRetries, outcome.GetRetryCount());
    RECORD_METRIC_VALUE(kMetricS3ActiveConnections, -1);
    VELOX_CHECK_AWS_OUTCOME(outcome, "Failed to get S3 object", bucket_, key_);
  }

  // Convenience overload: reads into a pre-allocated contiguous buffer.
  void preadInternal(uint64_t offset, uint64_t length, char* position) const {
    preadInternal(offset, length, AwsWriteableStreamFactory(position, length));
  }

  Aws::S3::S3Client* client_;
  std::string bucket_;
  std::string key_;
  int64_t length_ = -1;
};

S3ReadFile::S3ReadFile(
    std::string_view path,
    Aws::S3::S3Client* client,
    folly::Executor* executor)
    : executor_(executor) {
  impl_ = std::make_shared<Impl>(path, client);
}

S3ReadFile::~S3ReadFile() = default;

void S3ReadFile::initialize(const filesystems::FileOptions& options) {
  return impl_->initialize(options);
}

std::string_view S3ReadFile::pread(
    uint64_t offset,
    uint64_t length,
    void* buf,
    const FileIoContext& context) const {
  return impl_->pread(offset, length, buf, context);
}

std::string S3ReadFile::pread(
    uint64_t offset,
    uint64_t length,
    const FileIoContext& context) const {
  return impl_->pread(offset, length, context);
}

uint64_t S3ReadFile::preadv(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
  return impl_->preadv(offset, buffers, context);
}

folly::SemiFuture<uint64_t> S3ReadFile::preadvAsync(
    uint64_t offset,
    const std::vector<folly::Range<char*>>& buffers,
    const FileIoContext& context) const {
  if (!executor_) {
    return ReadFile::preadvAsync(offset, buffers, context);
  }
  auto impl = impl_; // prevent Impl destruction while async in-flight
  auto [promise, future] = folly::makePromiseContract<uint64_t>();
  executor_->add([impl,
                  _promise = std::move(promise),
                  _offset = offset,
                  _buffers = buffers,
                  _context = context]() mutable {
    try {
      auto result = impl->preadv(_offset, _buffers, _context);
      _promise.setValue(result);
    } catch (...) {
      _promise.setException(
          folly::exception_wrapper(std::current_exception()));
    }
  });
  return std::move(future);
}

bool S3ReadFile::hasPreadvAsync() const {
  return executor_ != nullptr;
}

uint64_t S3ReadFile::size() const {
  return impl_->size();
}

uint64_t S3ReadFile::memoryUsage() const {
  return impl_->memoryUsage();
}

bool S3ReadFile::shouldCoalesce() const {
  return impl_->shouldCoalesce();
}

std::string S3ReadFile::getName() const {
  return impl_->getName();
}

} // namespace facebook::velox::filesystems
