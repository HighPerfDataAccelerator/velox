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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderHelpers.h"

#include "velox/common/Casts.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/types.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <folly/futures/Future.h>

#include <curl/curl.h>

#ifdef VELOX_ENABLE_S3
#include "velox/connectors/hive/storage_adapters/s3fs/S3Util.h"

#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/model/GetObjectRequest.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

extern "C" bool glutenCrtS3RangeReaderAvailable();
extern "C" uint64_t glutenCrtS3ObjectSize(const char* uri);
extern "C" uint64_t glutenCrtS3ReadRanges(
    const char* uri,
    uint8_t* destination,
    const uint64_t* offsets,
    const uint64_t* lengths,
    const uint64_t* destinationOffsets,
    size_t count);

namespace {

#ifdef VELOX_ENABLE_S3
kvikio::RemoteHandle makeKvikioS3Handle(
    const std::string& filePath,
    const std::string& accessKeyId,
    const std::string& secretAccessKey,
    const std::string& sessionToken,
    std::optional<std::string> region,
    std::optional<std::string> endpoint,
    std::optional<std::size_t> fileSize) {
  auto bucketAndObject = kvikio::S3Endpoint::parse_s3_url(filePath);
  auto s3Endpoint = std::make_unique<kvikio::S3Endpoint>(
      std::move(bucketAndObject),
      std::move(region),
      accessKeyId,
      secretAccessKey,
      std::move(endpoint),
      sessionToken.empty() ? std::nullopt
                           : std::optional<std::string>{sessionToken});
  if (fileSize.has_value()) {
    return kvikio::RemoteHandle(std::move(s3Endpoint), fileSize.value());
  }
  return kvikio::RemoteHandle(std::move(s3Endpoint));
}
#endif

/**
 * @brief Static mutex to serialize batches of IO operations across drivers
 *
 * Mutex to ensure no interleaving of IO operations across drivers to ensure
 * drivers can move ahead without waiting for other drivers to finish their IO.
 */
std::mutex& ioBatchMutex() {
  static std::mutex mutex;
  return mutex;
}

std::size_t envBytesOrZero(const char* name) {
  const auto* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(value, &end, 10);
  VELOX_CHECK(
      errno == 0 && end != value && *end == '\0',
      "Invalid byte count {}='{}'",
      name,
      value);
  return static_cast<std::size_t>(parsed);
}

class NativeS3H2dGate {
 public:
  static NativeS3H2dGate& instance() {
    static NativeS3H2dGate gate;
    return gate;
  }

  void acquire() {
    if (capacity_ == 0) {
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return active_ < capacity_; });
    ++active_;
  }

  void release() {
    if (capacity_ == 0) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      VELOX_CHECK_GT(active_, 0);
      --active_;
    }
    condition_.notify_one();
  }

  bool enabled() const {
    return capacity_ != 0;
  }

 private:
  NativeS3H2dGate()
      : capacity_(envBytesOrZero("GLUTEN_CPP_S3_H2D_SLOTS")) {}

  const size_t capacity_;
  size_t active_{0};
  std::mutex mutex_;
  std::condition_variable condition_;
};

class NativeS3H2dPermit {
 public:
  explicit NativeS3H2dPermit(bool enabled)
      : gate_(enabled ? &NativeS3H2dGate::instance() : nullptr) {
    if (gate_ != nullptr && gate_->enabled()) {
      gate_->acquire();
      acquired_ = true;
    }
  }

  ~NativeS3H2dPermit() {
    release();
  }

  NativeS3H2dPermit(const NativeS3H2dPermit&) = delete;
  NativeS3H2dPermit& operator=(const NativeS3H2dPermit&) = delete;

  void release() {
    if (acquired_) {
      gate_->release();
      acquired_ = false;
    }
  }

 private:
  NativeS3H2dGate* gate_;
  bool acquired_{false};
};

#ifdef VELOX_ENABLE_S3
class NativeS3MultiScheduler {
 public:
  struct Auth {
    std::string url;
    std::string awsSigV4;
    std::string userPassword;
    std::string sessionToken;
  };

  static NativeS3MultiScheduler& instance() {
    static NativeS3MultiScheduler scheduler;
    return scheduler;
  }

  std::future<size_t> submit(
      const Auth& auth,
      size_t offset,
      size_t size,
      uint8_t* destination) {
    auto request = std::make_shared<Request>();
    request->auth = auth;
    request->offset = offset;
    request->size = size;
    request->destination = destination;
    auto future = request->promise.get_future();

    auto& shard = *shards_[
        nextShard_.fetch_add(1, std::memory_order_relaxed) % shards_.size()];
    {
      std::lock_guard<std::mutex> lock(shard.mutex);
      if (size <= metadataPriorityBytes_) {
        shard.priorityQueue.push_back(std::move(request));
      } else {
        shard.queue.push_back(std::move(request));
      }
    }
    shard.condition.notify_one();
    curl_multi_wakeup(shard.multi);
    submittedBytes_.fetch_add(size, std::memory_order_relaxed);
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    const auto inflight =
        inflightRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
    auto peak = peakInflightRequests_.load(std::memory_order_relaxed);
    while (inflight > peak &&
           !peakInflightRequests_.compare_exchange_weak(
               peak, inflight, std::memory_order_relaxed)) {
    }
    return future;
  }

  size_t rangeBytes() const {
    return rangeBytes_;
  }

  NativeS3MultiScheduler(const NativeS3MultiScheduler&) = delete;
  NativeS3MultiScheduler& operator=(const NativeS3MultiScheduler&) = delete;

 private:
  struct Request {
    Auth auth;
    size_t offset{0};
    size_t size{0};
    uint8_t* destination{nullptr};
    size_t written{0};
    CURL* easy{nullptr};
    curl_slist* headers{nullptr};
    std::string range;
    size_t retryCount{0};
    std::promise<size_t> promise;
  };

  struct Shard {
    CURLM* multi{nullptr};
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<Request>> priorityQueue;
    std::deque<std::shared_ptr<Request>> queue;
    std::multimap<
        std::chrono::steady_clock::time_point,
        std::shared_ptr<Request>>
        retryQueue;
    std::unordered_map<CURL*, std::shared_ptr<Request>> active;
    bool stopping{false};
    std::thread worker;
  };

  static size_t writeCallback(
      char* data,
      size_t elementSize,
      size_t elementCount,
      void* opaque) {
    auto* request = static_cast<Request*>(opaque);
    const auto bytes = elementSize * elementCount;
    if (request->written + bytes > request->size) {
      return CURL_WRITEFUNC_ERROR;
    }
    std::memcpy(
        request->destination + request->written, data, bytes);
    request->written += bytes;
    return bytes;
  }

  static size_t envOrDefault(const char* name, size_t defaultValue) {
    const auto parsed = envBytesOrZero(name);
    return parsed == 0 ? defaultValue : parsed;
  }

  NativeS3MultiScheduler()
      : concurrencyPerShard_(
            envOrDefault("GLUTEN_CPP_S3_CONCURRENCY_PER_SHARD", 16)),
        rangeBytes_(envOrDefault("GLUTEN_CPP_S3_RANGE_BYTES", 4UL << 20)),
        metadataPriorityBytes_(
            envOrDefault("GLUTEN_CPP_S3_METADATA_PRIORITY_BYTES", 64UL << 10)),
        maxRetries_(envOrDefault("GLUTEN_CPP_S3_MAX_RETRIES", 5)),
        retryBaseDelayMs_(
            envOrDefault("GLUTEN_CPP_S3_RETRY_BASE_DELAY_MS", 50)),
        retryMaxDelayMs_(
            envOrDefault("GLUTEN_CPP_S3_RETRY_MAX_DELAY_MS", 1'000)),
        diagnostics_(
            envBytesOrZero("GLUTEN_CPP_S3_DIAGNOSTICS") != 0),
        started_(std::chrono::steady_clock::now()) {
    const auto shardCount =
        envOrDefault("GLUTEN_CPP_S3_SHARDS", 8);
    VELOX_CHECK_GT(shardCount, 0);
    VELOX_CHECK_GT(concurrencyPerShard_, 0);
    VELOX_CHECK_GT(rangeBytes_, 0);
    curl_global_init(CURL_GLOBAL_ALL);
    shards_.reserve(shardCount);
    for (size_t index = 0; index < shardCount; ++index) {
      auto shard = std::make_unique<Shard>();
      shard->multi = curl_multi_init();
      VELOX_CHECK_NOT_NULL(shard->multi);
      curl_multi_setopt(
          shard->multi,
          CURLMOPT_MAX_TOTAL_CONNECTIONS,
          static_cast<long>(concurrencyPerShard_));
      curl_multi_setopt(
          shard->multi,
          CURLMOPT_MAX_HOST_CONNECTIONS,
          static_cast<long>(concurrencyPerShard_));
      shards_.push_back(std::move(shard));
    }
    for (auto& shard : shards_) {
      shard->worker =
          std::thread([this, rawShard = shard.get()] { run(*rawShard); });
    }
  }

  ~NativeS3MultiScheduler() {
    for (auto& shard : shards_) {
      {
        std::lock_guard<std::mutex> lock(shard->mutex);
        shard->stopping = true;
      }
      shard->condition.notify_one();
      curl_multi_wakeup(shard->multi);
    }
    for (auto& shard : shards_) {
      if (shard->worker.joinable()) {
        shard->worker.join();
      }
      curl_multi_cleanup(shard->multi);
    }
  }

  void startRequest(Shard& shard, std::shared_ptr<Request> request) {
    request->easy = curl_easy_init();
    VELOX_CHECK_NOT_NULL(request->easy);
    request->written = 0;
    request->range = std::to_string(request->offset) + "-" +
        std::to_string(request->offset + request->size - 1);
    if (!request->auth.sessionToken.empty()) {
      const auto header =
          "x-amz-security-token: " + request->auth.sessionToken;
      request->headers =
          curl_slist_append(request->headers, header.c_str());
      VELOX_CHECK_NOT_NULL(request->headers);
    }
    curl_easy_setopt(
        request->easy, CURLOPT_URL, request->auth.url.c_str());
    curl_easy_setopt(
        request->easy, CURLOPT_AWS_SIGV4, request->auth.awsSigV4.c_str());
    curl_easy_setopt(
        request->easy,
        CURLOPT_USERPWD,
        request->auth.userPassword.c_str());
    if (request->headers != nullptr) {
      curl_easy_setopt(
          request->easy, CURLOPT_HTTPHEADER, request->headers);
    }
    curl_easy_setopt(
        request->easy, CURLOPT_RANGE, request->range.c_str());
    curl_easy_setopt(
        request->easy, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(request->easy, CURLOPT_WRITEDATA, request.get());
    curl_easy_setopt(request->easy, CURLOPT_PRIVATE, request.get());
    curl_easy_setopt(request->easy, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(request->easy, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(request->easy, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(
        request->easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl_easy_setopt(request->easy, CURLOPT_BUFFERSIZE, 512L * 1024L);
    const auto code = curl_multi_add_handle(shard.multi, request->easy);
    VELOX_CHECK(
        code == CURLM_OK,
        "curl_multi_add_handle failed: {}",
        curl_multi_strerror(code));
    shard.active.emplace(request->easy, std::move(request));
  }

  void fillActive(Shard& shard) {
    while (shard.active.size() < concurrencyPerShard_) {
      std::shared_ptr<Request> request;
      {
        std::lock_guard<std::mutex> lock(shard.mutex);
        const auto now = std::chrono::steady_clock::now();
        if (!shard.retryQueue.empty() &&
            shard.retryQueue.begin()->first <= now) {
          request = std::move(shard.retryQueue.begin()->second);
          shard.retryQueue.erase(shard.retryQueue.begin());
        } else {
          auto* source = !shard.priorityQueue.empty()
              ? &shard.priorityQueue
              : &shard.queue;
          if (source->empty()) {
            return;
          }
          request = std::move(source->front());
          source->pop_front();
        }
      }
      startRequest(shard, std::move(request));
    }
  }

  static bool isRetryable(
      CURLcode result,
      long responseCode,
      size_t written,
      size_t expected) {
    if (responseCode == 408 || responseCode == 429 ||
        responseCode == 500 || responseCode == 502 ||
        responseCode == 503 || responseCode == 504) {
      return true;
    }
    if (responseCode == 206 && written != expected) {
      return true;
    }
    switch (result) {
      case CURLE_COULDNT_RESOLVE_HOST:
      case CURLE_COULDNT_CONNECT:
      case CURLE_OPERATION_TIMEDOUT:
      case CURLE_SEND_ERROR:
      case CURLE_RECV_ERROR:
      case CURLE_PARTIAL_FILE:
      case CURLE_GOT_NOTHING:
      case CURLE_SSL_CONNECT_ERROR:
        return true;
      default:
        return false;
    }
  }

  std::chrono::milliseconds retryDelay(const Request& request) const {
    const auto exponent = std::min<size_t>(request.retryCount - 1, 8);
    const auto exponential = retryBaseDelayMs_ * (size_t{1} << exponent);
    const auto capped = std::min(exponential, retryMaxDelayMs_);
    const auto jitterRange = std::max<size_t>(retryBaseDelayMs_, 1);
    const auto jitter =
        (request.offset ^ request.size ^ request.retryCount) % jitterRange;
    return std::chrono::milliseconds(
        std::min(capped + jitter, retryMaxDelayMs_));
  }

  void finishRequest(
      Shard& shard,
      CURL* easy,
      CURLcode result) {
    auto active = shard.active.find(easy);
    VELOX_CHECK(active != shard.active.end());
    auto request = std::move(active->second);
    shard.active.erase(active);
    long responseCode = 0;
    curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &responseCode);
    curl_off_t totalTimeUs = 0;
    curl_off_t connectTimeUs = 0;
    curl_off_t startTransferTimeUs = 0;
    curl_easy_getinfo(easy, CURLINFO_TOTAL_TIME_T, &totalTimeUs);
    curl_easy_getinfo(easy, CURLINFO_CONNECT_TIME_T, &connectTimeUs);
    curl_easy_getinfo(
        easy, CURLINFO_STARTTRANSFER_TIME_T, &startTransferTimeUs);
    curl_multi_remove_handle(shard.multi, easy);
    curl_easy_cleanup(easy);
    request->easy = nullptr;
    if (request->headers != nullptr) {
      curl_slist_free_all(request->headers);
      request->headers = nullptr;
    }

    completedAttempts_.fetch_add(1, std::memory_order_relaxed);
    totalTimeUs_.fetch_add(totalTimeUs, std::memory_order_relaxed);
    connectTimeUs_.fetch_add(connectTimeUs, std::memory_order_relaxed);
    startTransferTimeUs_.fetch_add(
        startTransferTimeUs, std::memory_order_relaxed);

    const auto succeeded =
        result == CURLE_OK && responseCode == 206 &&
        request->written == request->size;
    if (!succeeded &&
        request->retryCount < maxRetries_ &&
        isRetryable(
            result, responseCode, request->written, request->size)) {
      ++request->retryCount;
      const auto delay = retryDelay(*request);
      retryAttempts_.fetch_add(1, std::memory_order_relaxed);
      if (diagnostics_) {
        LOG(WARNING) << "CPP_S3_MULTI retry=" << request->retryCount
                     << " curl=" << curl_easy_strerror(result)
                     << " http=" << responseCode
                     << " expected=" << request->size
                     << " received=" << request->written
                     << " delayMs=" << delay.count();
      }
      request->written = 0;
      {
        std::lock_guard<std::mutex> lock(shard.mutex);
        shard.retryQueue.emplace(
            std::chrono::steady_clock::now() + delay,
            std::move(request));
      }
      shard.condition.notify_one();
      curl_multi_wakeup(shard.multi);
      return;
    }

    const auto completed =
        completedRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (succeeded) {
      completedBytes_.fetch_add(request->written, std::memory_order_relaxed);
    }
    inflightRequests_.fetch_sub(1, std::memory_order_relaxed);
    if (diagnostics_ && completed % 256 == 0) {
      const auto elapsed =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - started_)
              .count();
      const auto bytes =
          completedBytes_.load(std::memory_order_relaxed);
      const auto requests =
          completedRequests_.load(std::memory_order_relaxed);
      const auto attempts =
          completedAttempts_.load(std::memory_order_relaxed);
      LOG(WARNING) << "CPP_S3_MULTI completed=" << requests
                << " submitted="
                << submittedRequests_.load(std::memory_order_relaxed)
                << " attempts=" << attempts
                << " retries="
                << retryAttempts_.load(std::memory_order_relaxed)
                << " inflight="
                << inflightRequests_.load(std::memory_order_relaxed)
                << " peakInflight="
                << peakInflightRequests_.load(std::memory_order_relaxed)
                << " bytes=" << bytes
                << " elapsedSeconds=" << elapsed
                << " payloadGbps="
                << (elapsed == 0 ? 0.0 : bytes * 8.0 / elapsed / 1e9)
                << " avgTotalUs="
                << totalTimeUs_.load(std::memory_order_relaxed) / attempts
                << " avgConnectUs="
                << connectTimeUs_.load(std::memory_order_relaxed) / attempts
                << " avgStartTransferUs="
                << startTransferTimeUs_.load(std::memory_order_relaxed) /
            attempts;
    }

    if (succeeded) {
      request->promise.set_value(request->written);
      return;
    }
    std::ostringstream message;
    message << "Native S3 curl-multi range failed: curl="
            << curl_easy_strerror(result)
            << " http=" << responseCode
            << " expected=" << request->size
            << " received=" << request->written;
    request->promise.set_exception(
        std::make_exception_ptr(std::runtime_error(message.str())));
  }

  void failQueued(Shard& shard, const std::string& message) {
    std::deque<std::shared_ptr<Request>> requests;
    {
      std::lock_guard<std::mutex> lock(shard.mutex);
      requests.swap(shard.priorityQueue);
      requests.insert(
          requests.end(),
          std::make_move_iterator(shard.queue.begin()),
          std::make_move_iterator(shard.queue.end()));
      shard.queue.clear();
      for (auto& retry : shard.retryQueue) {
        requests.push_back(std::move(retry.second));
      }
      shard.retryQueue.clear();
    }
    for (auto& request : requests) {
      request->promise.set_exception(
          std::make_exception_ptr(std::runtime_error(message)));
    }
  }

  void run(Shard& shard) {
    try {
      while (true) {
        fillActive(shard);
        {
          std::unique_lock<std::mutex> lock(shard.mutex);
          if (shard.active.empty() && shard.priorityQueue.empty() &&
              shard.queue.empty() && shard.retryQueue.empty()) {
            if (shard.stopping) {
              break;
            }
            shard.condition.wait(lock, [&shard] {
              return shard.stopping || !shard.priorityQueue.empty() ||
                  !shard.queue.empty() || !shard.retryQueue.empty();
            });
            continue;
          }
          if (shard.active.empty() && shard.priorityQueue.empty() &&
              shard.queue.empty() && !shard.retryQueue.empty()) {
            shard.condition.wait_until(
                lock, shard.retryQueue.begin()->first);
            continue;
          }
        }

        int running = 0;
        auto code = curl_multi_perform(shard.multi, &running);
        VELOX_CHECK(
            code == CURLM_OK,
            "curl_multi_perform failed: {}",
            curl_multi_strerror(code));
        int remaining = 0;
        while (auto* message =
                   curl_multi_info_read(shard.multi, &remaining)) {
          if (message->msg == CURLMSG_DONE) {
            finishRequest(
                shard, message->easy_handle, message->data.result);
          }
        }
        fillActive(shard);
        if (!shard.active.empty()) {
          int descriptors = 0;
          code = curl_multi_poll(
              shard.multi, nullptr, 0, 100, &descriptors);
          VELOX_CHECK(
              code == CURLM_OK,
              "curl_multi_poll failed: {}",
              curl_multi_strerror(code));
        }
      }
    } catch (const std::exception& error) {
      failQueued(shard, error.what());
      for (auto& [easy, request] : shard.active) {
        curl_multi_remove_handle(shard.multi, easy);
        curl_easy_cleanup(easy);
        if (request->headers != nullptr) {
          curl_slist_free_all(request->headers);
        }
        request->promise.set_exception(std::current_exception());
      }
      shard.active.clear();
    }
  }

  const size_t concurrencyPerShard_;
  const size_t rangeBytes_;
  const size_t metadataPriorityBytes_;
  const size_t maxRetries_;
  const size_t retryBaseDelayMs_;
  const size_t retryMaxDelayMs_;
  const bool diagnostics_;
  const std::chrono::steady_clock::time_point started_;
  std::atomic<size_t> nextShard_{0};
  std::atomic<uint64_t> submittedRequests_{0};
  std::atomic<uint64_t> submittedBytes_{0};
  std::atomic<uint64_t> completedRequests_{0};
  std::atomic<uint64_t> completedAttempts_{0};
  std::atomic<uint64_t> retryAttempts_{0};
  std::atomic<uint64_t> completedBytes_{0};
  std::atomic<uint64_t> inflightRequests_{0};
  std::atomic<uint64_t> peakInflightRequests_{0};
  std::atomic<uint64_t> totalTimeUs_{0};
  std::atomic<uint64_t> connectTimeUs_{0};
  std::atomic<uint64_t> startTransferTimeUs_{0};
  std::vector<std::unique_ptr<Shard>> shards_;
};

// Executor-global admission and priority queue for the official AWS SDK paths.
// The scheduler owns one S3CrtClient per executor when CRT is selected. The
// classic path uses S3ReadFile instances that share S3FileSystem's S3Client.
// Both paths retain official credentials/retry behavior and write directly
// into the caller-provided pinned destination.
class NativeS3SdkScheduler {
 public:
  static NativeS3SdkScheduler& instance() {
    static NativeS3SdkScheduler scheduler;
    return scheduler;
  }

  std::future<size_t> submit(
      std::shared_ptr<facebook::velox::ReadFile> readFile,
      size_t offset,
      size_t size,
      uint8_t* destination) {
    VELOX_CHECK_NOT_NULL(readFile);
    auto request = std::make_shared<Request>();
    request->readFile = std::move(readFile);
    request->offset = offset;
    request->size = size;
    request->destination = destination;
    auto future = request->promise.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& queue =
          size <= metadataPriorityBytes_ ? priorityQueue_ : queue_;
      queue.push_back(std::move(request));
    }
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    submittedBytes_.fetch_add(size, std::memory_order_relaxed);
    condition_.notify_one();
    return future;
  }

  std::future<size_t> submitCrt(
      std::string bucket,
      std::string key,
      size_t offset,
      size_t size,
      uint8_t* destination) {
    VELOX_CHECK(useCrt_);
    auto request = std::make_shared<Request>();
    request->bucket = std::move(bucket);
    request->key = std::move(key);
    request->offset = offset;
    request->size = size;
    request->destination = destination;
    auto future = request->promise.get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& queue =
          size <= metadataPriorityBytes_ ? priorityQueue_ : queue_;
      queue.push_back(std::move(request));
    }
    submittedRequests_.fetch_add(1, std::memory_order_relaxed);
    submittedBytes_.fetch_add(size, std::memory_order_relaxed);
    dispatchCrt();
    return future;
  }

  size_t rangeBytes() const {
    return rangeBytes_;
  }

  NativeS3SdkScheduler(const NativeS3SdkScheduler&) = delete;
  NativeS3SdkScheduler& operator=(const NativeS3SdkScheduler&) = delete;

 private:
  struct Request {
    std::shared_ptr<facebook::velox::ReadFile> readFile;
    std::string bucket;
    std::string key;
    size_t offset{0};
    size_t size{0};
    uint8_t* destination{nullptr};
    std::promise<size_t> promise;
  };

  static size_t envOrDefault(const char* name, size_t defaultValue) {
    const auto parsed = envBytesOrZero(name);
    return parsed == 0 ? defaultValue : parsed;
  }

  NativeS3SdkScheduler()
      : useCrt_(envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0),
        concurrency_(
            envOrDefault("GLUTEN_CPP_S3_SDK_CONCURRENCY", 128)),
        rangeBytes_(
            envOrDefault("GLUTEN_CPP_S3_SDK_RANGE_BYTES", 4UL << 20)),
        metadataPriorityBytes_(
            envOrDefault(
                "GLUTEN_CPP_S3_METADATA_PRIORITY_BYTES", 64UL << 10)),
        diagnostics_(
            envBytesOrZero("GLUTEN_CPP_S3_DIAGNOSTICS") != 0),
        started_(std::chrono::steady_clock::now()) {
    VELOX_CHECK_GT(concurrency_, 0);
    VELOX_CHECK_GT(rangeBytes_, 0);
    if (useCrt_) {
      Aws::S3Crt::ClientConfiguration config;
      if (const auto* region = std::getenv("AWS_REGION");
          region != nullptr && region[0] != '\0') {
        config.region = region;
      } else if (const auto* region = std::getenv("AWS_DEFAULT_REGION");
                 region != nullptr && region[0] != '\0') {
        config.region = region;
      }
      config.throughputTargetGbps = static_cast<double>(
          envOrDefault("GLUTEN_CPP_S3_CRT_TARGET_GBPS", 25));
      config.partSize =
          envOrDefault("GLUTEN_CPP_S3_CRT_PART_BYTES", 8UL << 20);
      config.downloadMemoryUsageWindow = envOrDefault(
          "GLUTEN_CPP_S3_CRT_DOWNLOAD_WINDOW_BYTES", 1UL << 30);
      config.crtRetryStrategyConfig.crtRetryStrategyType =
          Aws::S3Crt::S3CrtClientConfiguration::CrtRetryStrategyConfig::
              CrtRetryStrategyType::EXPONENTIAL_BACKOFF;
      config.crtRetryStrategyConfig.config.maxRetries =
          envOrDefault("GLUTEN_CPP_S3_CRT_MAX_RETRIES", 5);
      auto credentialsProvider =
          facebook::velox::filesystems::
              makeSynchronizedCachingCredentialsProvider(
                  std::make_shared<
                      Aws::Auth::DefaultAWSCredentialsProviderChain>());
      crtClient_ = std::make_shared<Aws::S3Crt::S3CrtClient>(
          std::move(credentialsProvider), config);
    }
    if (!useCrt_) {
      workers_.reserve(concurrency_);
      for (size_t index = 0; index < concurrency_; ++index) {
        workers_.emplace_back([this] { run(); });
      }
    }
  }

  ~NativeS3SdkScheduler() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void dispatchCrt() {
    while (true) {
      std::shared_ptr<Request> request;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ ||
            inflightRequests_.load(std::memory_order_relaxed) >=
                concurrency_ ||
            (priorityQueue_.empty() && queue_.empty())) {
          return;
        }
        auto& source =
            !priorityQueue_.empty() ? priorityQueue_ : queue_;
        request = std::move(source.front());
        source.pop_front();
        const auto inflight =
            inflightRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
        auto peak = peakInflightRequests_.load(std::memory_order_relaxed);
        while (inflight > peak &&
               !peakInflightRequests_.compare_exchange_weak(
                   peak, inflight, std::memory_order_relaxed)) {
        }
      }

      auto get =
          std::make_shared<Aws::S3Crt::Model::GetObjectRequest>();
      get->SetBucket(request->bucket.c_str());
      get->SetKey(request->key.c_str());
      get->SetRange(
          fmt::format(
              "bytes={}-{}",
              request->offset,
              request->offset + request->size - 1)
              .c_str());
      const auto destination = request->destination;
      const auto size = request->size;
      get->SetResponseStreamFactory([destination, size]() {
        return Aws::New<
            facebook::velox::filesystems::StringViewStream>(
            "NativeS3CrtScheduler", destination, size);
      });
      const auto started = std::chrono::steady_clock::now();
      crtClient_->GetObjectAsync(
          *get,
          [this, request, get, started](
              const Aws::S3Crt::S3CrtClient*,
              const Aws::S3Crt::Model::GetObjectRequest&,
              Aws::S3Crt::Model::GetObjectOutcome outcome,
              const std::shared_ptr<
                  const Aws::Client::AsyncCallerContext>&) {
            bool succeeded = false;
            try {
              if (outcome.IsSuccess()) {
                retryAttempts_.fetch_add(
                    outcome.GetRetryCount(), std::memory_order_relaxed);
              }
              VELOX_CHECK(
                  outcome.IsSuccess(),
                  "Native S3 CRT range failed for s3://{}/{}: {}",
                  request->bucket,
                  request->key,
                  outcome.GetError().GetMessage());
              request->promise.set_value(request->size);
              succeeded = true;
            } catch (...) {
              request->promise.set_exception(std::current_exception());
            }
            const auto elapsedUs =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count();
            totalTimeUs_.fetch_add(
                elapsedUs, std::memory_order_relaxed);
            if (succeeded) {
              completedBytes_.fetch_add(
                  request->size, std::memory_order_relaxed);
            }
            inflightRequests_.fetch_sub(1, std::memory_order_relaxed);
            const auto completed =
                completedRequests_.fetch_add(
                    1, std::memory_order_relaxed) +
                1;
            logProgress(completed);
            dispatchCrt();
          });
    }
  }

  void logProgress(uint64_t completed) {
    if (!diagnostics_ || completed % 256 != 0) {
      return;
    }
    const auto elapsed =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_)
            .count();
    const auto bytes = completedBytes_.load(std::memory_order_relaxed);
    LOG(WARNING) << (useCrt_ ? "CPP_S3_CRT" : "CPP_S3_SDK")
                 << " completed=" << completed
                 << " submitted="
                 << submittedRequests_.load(std::memory_order_relaxed)
                 << " inflight="
                 << inflightRequests_.load(std::memory_order_relaxed)
                 << " peakInflight="
                 << peakInflightRequests_.load(std::memory_order_relaxed)
                 << " bytes=" << bytes
                 << " retries="
                 << retryAttempts_.load(std::memory_order_relaxed)
                 << " elapsedSeconds=" << elapsed
                 << " payloadGbps="
                 << (elapsed == 0 ? 0.0 : bytes * 8.0 / elapsed / 1e9)
                 << " avgTotalUs="
                 << totalTimeUs_.load(std::memory_order_relaxed) /
            completed;
  }

  void run() {
    while (true) {
      std::shared_ptr<Request> request;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
          return stopping_ || !priorityQueue_.empty() || !queue_.empty();
        });
        if (stopping_ && priorityQueue_.empty() && queue_.empty()) {
          return;
        }
        auto& source =
            !priorityQueue_.empty() ? priorityQueue_ : queue_;
        request = std::move(source.front());
        source.pop_front();
      }

      const auto inflight =
          inflightRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
      auto peak = peakInflightRequests_.load(std::memory_order_relaxed);
      while (inflight > peak &&
             !peakInflightRequests_.compare_exchange_weak(
                 peak, inflight, std::memory_order_relaxed)) {
      }
      const auto started = std::chrono::steady_clock::now();
      bool succeeded = false;
      try {
        const auto view = request->readFile->pread(
            request->offset, request->size, request->destination);
        VELOX_CHECK_EQ(
            view.size(), request->size, "Short AWS SDK S3 range read");
        request->promise.set_value(view.size());
        succeeded = true;
      } catch (...) {
        request->promise.set_exception(std::current_exception());
      }
      const auto elapsedUs =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - started)
              .count();
      totalTimeUs_.fetch_add(elapsedUs, std::memory_order_relaxed);
      if (succeeded) {
        completedBytes_.fetch_add(request->size, std::memory_order_relaxed);
      }
      inflightRequests_.fetch_sub(1, std::memory_order_relaxed);
      const auto completed =
          completedRequests_.fetch_add(1, std::memory_order_relaxed) + 1;
      logProgress(completed);
    }
  }

  const bool useCrt_;
  const size_t concurrency_;
  const size_t rangeBytes_;
  const size_t metadataPriorityBytes_;
  const bool diagnostics_;
  const std::chrono::steady_clock::time_point started_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::shared_ptr<Request>> priorityQueue_;
  std::deque<std::shared_ptr<Request>> queue_;
  bool stopping_{false};
  std::vector<std::thread> workers_;
  std::shared_ptr<Aws::S3Crt::S3CrtClient> crtClient_;
  std::atomic<uint64_t> submittedRequests_{0};
  std::atomic<uint64_t> submittedBytes_{0};
  std::atomic<uint64_t> completedRequests_{0};
  std::atomic<uint64_t> completedBytes_{0};
  std::atomic<uint64_t> inflightRequests_{0};
  std::atomic<uint64_t> peakInflightRequests_{0};
  std::atomic<uint64_t> totalTimeUs_{0};
  std::atomic<uint64_t> retryAttempts_{0};
};
#endif

template <typename T>
std::future<T> toStdFuture(folly::Future<T> follyFuture) {
  auto promise = std::make_shared<std::promise<T>>();
  auto stdFuture = promise->get_future();

  std::move(follyFuture).thenTry([promise](folly::Try<T>&& result) mutable {
    if (result.hasValue()) {
      promise->set_value(std::move(result.value()));
    } else {
      promise->set_exception(result.exception().to_exception_ptr());
    }
  });

  return stdFuture;
}
} // namespace

namespace facebook::velox::cudf_velox::connector::hive {

#ifdef VELOX_ENABLE_S3
bool crtS3RangeReaderAvailable() {
  return glutenCrtS3RangeReaderAvailable();
}

CrtS3DataSource::CrtS3DataSource(
    std::string filePath,
    std::optional<std::size_t> fileSize)
    : filePath_(std::move(filePath)),
      fileSize_(
          fileSize.has_value()
              ? fileSize.value()
              : static_cast<size_t>(glutenCrtS3ObjectSize(filePath_.c_str()))),
      metadataReadAheadBytes_(
          envBytesOrZero("GLUTEN_CRT_S3_METADATA_READAHEAD_BYTES")) {
  VELOX_CHECK(
      glutenCrtS3RangeReaderAvailable(),
      "AWS CRT S3 range bridge is not available");
}

size_t CrtS3DataSource::clampedReadSize(
    size_t offset,
    size_t requestedSize) const {
  if (offset >= fileSize_) {
    return 0;
  }
  return std::min(requestedSize, fileSize_ - offset);
}

size_t CrtS3DataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer> CrtS3DataSource::host_read(
    size_t offset,
    size_t requestedSize) {
  auto data = std::vector<uint8_t>(clampedReadSize(offset, requestedSize));
  if (!data.empty()) {
    const auto bytes = host_read(offset, data.size(), data.data());
    VELOX_CHECK_EQ(bytes, data.size(), "Short AWS CRT S3 read");
  }
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t CrtS3DataSource::host_read(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return 0;
  }
  // This datasource is constructed only for the cuDF Parquet reader. Parquet
  // always starts with the four-byte PAR1 magic, and the real footer magic is
  // still fetched and validated from S3. Avoid one latency-dominated 4-byte
  // GET per file when explicitly enabled.
  if (offset == 0 && readSize == 4 &&
      envBytesOrZero("GLUTEN_CRT_S3_SYNTHESIZE_PARQUET_MAGIC") != 0) {
    static constexpr uint8_t kParquetMagic[] = {'P', 'A', 'R', '1'};
    std::memcpy(dst, kParquetMagic, sizeof(kParquetMagic));
    return sizeof(kParquetMagic);
  }
  const auto tailReadSize = std::min(metadataReadAheadBytes_, fileSize_);
  const auto tailOffset = fileSize_ - tailReadSize;
  if (tailReadSize != 0 && readSize <= tailReadSize && offset >= tailOffset) {
    std::call_once(tailCacheOnce_, [this, tailOffset, tailReadSize]() {
      tailCacheOffset_ = tailOffset;
      tailCache_.resize(tailReadSize);
      const auto bytes =
          readRanges({tailOffset}, {tailReadSize}, tailCache_.data(), {0});
      VELOX_CHECK_EQ(
          bytes, tailReadSize, "Short AWS CRT S3 metadata read-ahead");
    });
    VELOX_CHECK_LE(
        readSize,
        tailCache_.size() - (offset - tailCacheOffset_),
        "AWS CRT S3 metadata read exceeds tail cache");
    std::memcpy(
        dst, tailCache_.data() + offset - tailCacheOffset_, readSize);
    return readSize;
  }
  return readRanges({offset}, {readSize}, dst, {0});
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
CrtS3DataSource::host_read_async(size_t offset, size_t requestedSize) {
  return std::async(std::launch::deferred, [this, offset, requestedSize]() {
    return host_read(offset, requestedSize);
  });
}

std::future<size_t> CrtS3DataSource::host_read_async(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst) {
  return std::async(
      std::launch::deferred,
      [this, offset, requestedSize, dst]() {
        return host_read(offset, requestedSize, dst);
      });
}

bool CrtS3DataSource::supports_device_read() const {
  return false;
}

size_t CrtS3DataSource::readRanges(
    const std::vector<size_t>& offsets,
    const std::vector<size_t>& sizes,
    uint8_t* destination,
    const std::vector<size_t>& destinationOffsets) const {
  VELOX_CHECK_EQ(offsets.size(), sizes.size());
  VELOX_CHECK_EQ(sizes.size(), destinationOffsets.size());
  std::vector<uint64_t> rangeOffsets(offsets.begin(), offsets.end());
  std::vector<uint64_t> rangeSizes(sizes.begin(), sizes.end());
  std::vector<uint64_t> outputOffsets(
      destinationOffsets.begin(), destinationOffsets.end());
  return static_cast<size_t>(glutenCrtS3ReadRanges(
      filePath_.c_str(),
      destination,
      rangeOffsets.data(),
      rangeSizes.data(),
      outputOffsets.data(),
      rangeOffsets.size()));
}

KvikioS3DataSource::KvikioS3DataSource(
    const std::string& filePath,
    const std::string& accessKeyId,
    const std::string& secretAccessKey,
    const std::string& sessionToken,
    std::optional<std::string> region,
    std::optional<std::string> endpoint,
    std::shared_ptr<ReadFile> nativeS3ReadFile,
    std::optional<std::size_t> fileSize)
    : handle_(makeKvikioS3Handle(
          filePath,
          accessKeyId,
          secretAccessKey,
          sessionToken,
          region,
          endpoint,
          fileSize)),
      nativeS3ReadFile_(std::move(nativeS3ReadFile)),
      metadataReadAheadBytes_(
          envBytesOrZero("GLUTEN_CPP_S3_METADATA_READAHEAD_BYTES")) {
  VELOX_CHECK_NOT_NULL(nativeS3ReadFile_);
  auto bucketAndObject = kvikio::S3Endpoint::parse_s3_url(filePath);
  nativeS3Bucket_ = bucketAndObject.first;
  nativeS3Key_ = bucketAndObject.second;
  nativeS3Url_ = kvikio::S3Endpoint::url_from_bucket_and_object(
      bucketAndObject.first,
      bucketAndObject.second,
      region,
      endpoint);
  std::string resolvedRegion;
  if (region.has_value()) {
    resolvedRegion = region.value();
  } else if (const auto* environmentRegion =
                 std::getenv("AWS_DEFAULT_REGION")) {
    resolvedRegion = environmentRegion;
  }
  VELOX_CHECK(
      !resolvedRegion.empty(), "Native S3 curl-multi requires AWS region");
  nativeAwsSigV4_ = "aws:amz:" + resolvedRegion + ":s3";
  nativeUserPassword_ = accessKeyId + ":" + secretAccessKey;
  nativeSessionToken_ = sessionToken;
}

size_t KvikioS3DataSource::clampedReadSize(size_t offset, size_t requestedSize)
    const {
  if (offset >= size()) {
    return 0;
  }
  return std::min(requestedSize, size() - offset);
}

size_t KvikioS3DataSource::size() const {
  return handle_.nbytes();
}

std::unique_ptr<cudf::io::datasource::buffer> KvikioS3DataSource::host_read(
    size_t offset,
    size_t requestedSize) {
  auto data = std::vector<uint8_t>(clampedReadSize(offset, requestedSize));
  if (!data.empty()) {
    handle_.pread(data.data(), data.size(), offset).get();
  }
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t KvikioS3DataSource::host_read(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return 0;
  }
  if (offset == 0 && readSize == 4 &&
      envBytesOrZero("GLUTEN_CPP_S3_SYNTHESIZE_PARQUET_MAGIC") != 0) {
    static constexpr uint8_t kParquetMagic[] = {'P', 'A', 'R', '1'};
    std::memcpy(dst, kParquetMagic, sizeof(kParquetMagic));
    return sizeof(kParquetMagic);
  }
  const auto fileSize = size();
  const auto tailReadSize = std::min(metadataReadAheadBytes_, fileSize);
  const auto tailOffset = fileSize - tailReadSize;
  if (tailReadSize != 0 && readSize <= tailReadSize && offset >= tailOffset) {
    std::call_once(tailCacheOnce_, [this, tailOffset, tailReadSize]() {
      tailCacheOffset_ = tailOffset;
      tailCache_.resize(tailReadSize);
      const auto bytes =
          readRanges({tailOffset}, {tailReadSize}, tailCache_.data(), {0});
      VELOX_CHECK_EQ(
          bytes, tailReadSize, "Short native S3 metadata read-ahead");
    });
    VELOX_CHECK_LE(
        readSize,
        tailCache_.size() - (offset - tailCacheOffset_),
        "Native S3 metadata read exceeds tail cache");
    std::memcpy(
        dst, tailCache_.data() + offset - tailCacheOffset_, readSize);
    return readSize;
  }
  return readRanges({offset}, {readSize}, dst, {0});
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
KvikioS3DataSource::host_read_async(size_t offset, size_t requestedSize) {
  auto data = std::vector<uint8_t>(clampedReadSize(offset, requestedSize));
  if (data.empty()) {
    return std::async(
        std::launch::deferred, [data = std::move(data)]() mutable {
          return cudf::io::datasource::buffer::create(std::move(data));
        });
  }
  auto readFuture = handle_.pread(data.data(), data.size(), offset);
  return std::async(
      std::launch::deferred,
      [data = std::move(data), readFuture = std::move(readFuture)]() mutable {
        readFuture.get();
        return cudf::io::datasource::buffer::create(std::move(data));
      });
}

std::future<size_t> KvikioS3DataSource::host_read_async(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return std::async(std::launch::deferred, [] { return size_t{0}; });
  }
  auto readFuture = handle_.pread(dst, readSize, offset);
  return std::async(
      std::launch::deferred, [readFuture = std::move(readFuture)]() mutable {
        return readFuture.get();
      });
}

size_t KvikioS3DataSource::readRanges(
    const std::vector<size_t>& offsets,
    const std::vector<size_t>& sizes,
    uint8_t* destination,
    const std::vector<size_t>& destinationOffsets) {
  VELOX_CHECK_EQ(offsets.size(), sizes.size());
  VELOX_CHECK_EQ(sizes.size(), destinationOffsets.size());
  std::vector<std::future<size_t>> pending;
  pending.reserve(offsets.size());
  size_t expectedBytes = 0;
  const auto useCrt = envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0;
  const auto useAwsSdk =
      !useCrt && envBytesOrZero("GLUTEN_CPP_S3_AWS_SDK") != 0;
  const auto useCurlMulti =
      !useCrt && !useAwsSdk &&
      envBytesOrZero("GLUTEN_CPP_S3_CURL_MULTI") != 0;
  auto* sdkScheduler = (useCrt || useAwsSdk)
      ? &NativeS3SdkScheduler::instance()
      : nullptr;
  auto* scheduler = useCurlMulti
      ? &NativeS3MultiScheduler::instance()
      : nullptr;
  const NativeS3MultiScheduler::Auth auth{
      nativeS3Url_,
      nativeAwsSigV4_,
      nativeUserPassword_,
      nativeSessionToken_};
  for (size_t index = 0; index < offsets.size(); ++index) {
    const auto readSize = clampedReadSize(offsets[index], sizes[index]);
    VELOX_CHECK_EQ(
        readSize,
        sizes[index],
        "Native S3 range extends beyond end of object");
    if (readSize == 0) {
      continue;
    }
    if (sdkScheduler != nullptr) {
      size_t scheduled = 0;
      while (scheduled < readSize) {
        const auto chunkSize = std::min(
            sdkScheduler->rangeBytes(), readSize - scheduled);
        if (useCrt) {
          pending.emplace_back(sdkScheduler->submitCrt(
              nativeS3Bucket_,
              nativeS3Key_,
              offsets[index] + scheduled,
              chunkSize,
              destination + destinationOffsets[index] + scheduled));
        } else {
          pending.emplace_back(sdkScheduler->submit(
              nativeS3ReadFile_,
              offsets[index] + scheduled,
              chunkSize,
              destination + destinationOffsets[index] + scheduled));
        }
        scheduled += chunkSize;
      }
    } else if (scheduler == nullptr) {
      pending.emplace_back(handle_.pread(
          destination + destinationOffsets[index],
          readSize,
          offsets[index]));
    } else {
      size_t scheduled = 0;
      while (scheduled < readSize) {
        const auto chunkSize = std::min(
            scheduler->rangeBytes(), readSize - scheduled);
        pending.emplace_back(scheduler->submit(
            auth,
            offsets[index] + scheduled,
            chunkSize,
            destination + destinationOffsets[index] + scheduled));
        scheduled += chunkSize;
      }
    }
    expectedBytes += readSize;
  }
  size_t actualBytes = 0;
  for (auto& read : pending) {
    actualBytes += read.get();
  }
  VELOX_CHECK_EQ(actualBytes, expectedBytes, "Short native S3 range batch");
  return actualBytes;
}

bool KvikioS3DataSource::supports_device_read() const {
  return true;
}

bool KvikioS3DataSource::is_device_read_preferred(
    size_t /* requestedSize */) const {
  return true;
}

std::future<size_t> KvikioS3DataSource::device_read_async(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  const auto readSize = clampedReadSize(offset, requestedSize);
  if (readSize == 0) {
    return std::async(std::launch::deferred, [] { return size_t{0}; });
  }
  if (envBytesOrZero("GLUTEN_CPP_S3_REGULAR_DEVICE") != 0 &&
      (envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0 ||
       envBytesOrZero("GLUTEN_CPP_S3_AWS_SDK") != 0 ||
       envBytesOrZero("GLUTEN_CPP_S3_CURL_MULTI") != 0)) {
    auto hostBuffer = std::make_shared<PinnedHostBuffer>(readSize);
    const auto useCrt =
        envBytesOrZero("GLUTEN_CPP_S3_CRT") != 0;
    const auto useAwsSdk =
        !useCrt && envBytesOrZero("GLUTEN_CPP_S3_AWS_SDK") != 0;
    auto* sdkScheduler = (useCrt || useAwsSdk)
        ? &NativeS3SdkScheduler::instance()
        : nullptr;
    auto* scheduler = (useCrt || useAwsSdk)
        ? nullptr
        : &NativeS3MultiScheduler::instance();
    const NativeS3MultiScheduler::Auth auth{
        nativeS3Url_,
        nativeAwsSigV4_,
        nativeUserPassword_,
        nativeSessionToken_};
    std::vector<std::future<size_t>> reads;
    const auto rangeBytes = (useCrt || useAwsSdk)
        ? sdkScheduler->rangeBytes()
        : scheduler->rangeBytes();
    reads.reserve((readSize + rangeBytes - 1) / rangeBytes);
    size_t scheduled = 0;
    while (scheduled < readSize) {
      const auto chunkSize =
          std::min(rangeBytes, readSize - scheduled);
      if (useCrt) {
        reads.emplace_back(sdkScheduler->submitCrt(
            nativeS3Bucket_,
            nativeS3Key_,
            offset + scheduled,
            chunkSize,
            hostBuffer->data() + scheduled));
      } else if (useAwsSdk) {
        reads.emplace_back(sdkScheduler->submit(
            nativeS3ReadFile_,
            offset + scheduled,
            chunkSize,
            hostBuffer->data() + scheduled));
      } else {
        reads.emplace_back(scheduler->submit(
            auth,
            offset + scheduled,
            chunkSize,
            hostBuffer->data() + scheduled));
      }
      scheduled += chunkSize;
    }
    // cuDF invokes device_read_async for the projected ranges before waiting
    // on their futures. Submission above therefore fills the executor-global
    // curl-multi window immediately; deferred completion avoids one host
    // thread per range batch.
    return std::async(
        std::launch::deferred,
        [reads = std::move(reads),
         hostBuffer = std::move(hostBuffer),
         dst,
         stream,
         readSize]() mutable {
          size_t actual = 0;
          for (auto& read : reads) {
            actual += read.get();
          }
          VELOX_CHECK_EQ(actual, readSize, "Short regular native S3 read");
          if (envBytesOrZero(
                  "GLUTEN_CPP_S3_REGULAR_STREAM_ORDERED_H2D") != 0) {
            using Completion = std::pair<
                std::shared_ptr<PinnedHostBuffer>,
                std::unique_ptr<NativeS3H2dPermit>>;
            auto completion = std::make_unique<Completion>(
                std::move(hostBuffer),
                std::make_unique<NativeS3H2dPermit>(/*enabled=*/true));
            CUDF_CUDA_TRY(cudaMemcpyAsync(
                dst,
                completion->first->data(),
                readSize,
                cudaMemcpyHostToDevice,
                stream.value()));
            CUDF_CUDA_TRY(cudaLaunchHostFunc(
                stream.value(),
                [](void* opaque) {
                  delete static_cast<Completion*>(opaque);
                },
                completion.get()));
            completion.release();
            return actual;
          }
          NativeS3H2dPermit h2dPermit(/*enabled=*/true);
          CUDF_CUDA_TRY(cudaMemcpyAsync(
              dst,
              hostBuffer->data(),
              readSize,
              cudaMemcpyHostToDevice,
              stream.value()));
          stream.synchronize();
          h2dPermit.release();
          return actual;
        });
  }
  // KvikIO owns the worker streams used by remote device reads. Keep the H2D
  // copy on those internal streams and use pread()'s completion as the device
  // visibility boundary. The consumer stream is not exposed to KvikIO.
  auto readFuture = handle_.pread(dst, readSize, offset);
  return std::async(
      std::launch::deferred, [readFuture = std::move(readFuture)]() mutable {
        return readFuture.get();
      });
}

size_t KvikioS3DataSource::device_read(
    size_t offset,
    size_t requestedSize,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  return device_read_async(offset, requestedSize, dst, stream).get();
}

std::unique_ptr<cudf::io::datasource::buffer> KvikioS3DataSource::device_read(
    size_t offset,
    size_t requestedSize,
    rmm::cuda_stream_view stream) {
  rmm::device_buffer data(clampedReadSize(offset, requestedSize), stream);
  const auto readSize = device_read(
      offset, requestedSize, static_cast<uint8_t*>(data.data()), stream);
  data.resize(readSize, stream);
  return cudf::io::datasource::buffer::create(std::move(data));
}
#endif

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input)
    : input_(std::move(input)), fileSize_(input_->getReadFile()->size()) {}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

void BufferedInputDataSource::enqueueForDevice(
    uint64_t offset,
    uint64_t size,
    uint8_t* dst) {
  auto inputStream = input_->enqueue({offset, size});
  std::shared_ptr sharedStream(std::move(inputStream));
  pendingDeviceLoads_.push_back(
      [dst, size, sharedStream](rmm::cuda_stream_view stream) {
        std::vector<uint8_t> buffer(size);
        sharedStream->readFully(reinterpret_cast<char*>(buffer.data()), size);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst, buffer.data(), size, cudaMemcpyHostToDevice, stream.value()));
      });
}

void BufferedInputDataSource::load(rmm::cuda_stream_view stream) {
  input_->load(velox::dwio::common::LogType::FILE);
  std::lock_guard<std::mutex> lock(ioBatchMutex());
  for (auto& deviceLoad : pendingDeviceLoads_) {
    deviceLoad(stream);
  }
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  auto future = folly::via(input_->executor())
                    .thenValue([this, offset, size, dst, stream](auto&&) {
                      auto hostBuffer = this->host_read(offset, size);
                      CUDF_CUDA_TRY(cudaMemcpyAsync(
                          dst,
                          hostBuffer->data(),
                          hostBuffer->size(),
                          cudaMemcpyHostToDevice,
                          stream.value()));
                      return hostBuffer->size();
                    });
  return toStdFuture(std::move(future));
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

std::shared_ptr<PreparedHostByteRanges> prepareByteRangesToHost(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges) {
#ifdef VELOX_ENABLE_S3
  using PackedHostRead = std::function<size_t(
      const std::vector<size_t>&,
      const std::vector<size_t>&,
      uint8_t*,
      const std::vector<size_t>&)>;
  PackedHostRead packedHostRead;
  const char* pageableHostBufferEnv = nullptr;
  bool nativeCppHostBatch = false;
  if (auto* crtSource = dynamic_cast<CrtS3DataSource*>(dataSource.get())) {
    packedHostRead =
        [crtSource](
            const std::vector<size_t>& offsets,
            const std::vector<size_t>& sizes,
            uint8_t* destination,
            const std::vector<size_t>& destinationOffsets) {
          return crtSource->readRanges(
              offsets, sizes, destination, destinationOffsets);
        };
    pageableHostBufferEnv = "GLUTEN_CRT_S3_PAGEABLE_HOST_BUFFER";
  } else if (
      envBytesOrZero("GLUTEN_CPP_S3_HOST_BATCH") != 0) {
    if (auto* nativeSource =
            dynamic_cast<KvikioS3DataSource*>(dataSource.get())) {
      packedHostRead =
          [nativeSource](
              const std::vector<size_t>& offsets,
              const std::vector<size_t>& sizes,
              uint8_t* destination,
              const std::vector<size_t>& destinationOffsets) {
            return nativeSource->readRanges(
                offsets, sizes, destination, destinationOffsets);
          };
      pageableHostBufferEnv = "GLUTEN_CPP_S3_PAGEABLE_HOST_BUFFER";
      nativeCppHostBatch = true;
    }
  }
  if (!packedHostRead) {
    return nullptr;
  }

  auto prepared = std::make_shared<PreparedHostByteRanges>();
  prepared->hostSourceOffsets.resize(byteRanges.size());
  prepared->coalesceGapBytes =
      envBytesOrZero("GLUTEN_CRT_S3_RANGE_COALESCE_GAP_BYTES");
  prepared->nativeCppHostBatch = nativeCppHostBatch;
  prepared->totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        return acc + static_cast<size_t>(byteRange.size());
      });

  std::vector<size_t> ioOffsets;
  std::vector<size_t> ioSizes;
  std::vector<size_t> destinationOffsets;
  size_t hostBufferSize = 0;
  for (size_t chunk = 0; chunk < byteRanges.size();) {
    const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
    auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
    const auto destinationOffset = hostBufferSize;
    prepared->hostSourceOffsets[chunk] = destinationOffset;
    size_t nextChunk = chunk + 1;
    while (nextChunk < byteRanges.size()) {
      const size_t nextOffset = byteRanges[nextChunk].offset();
      const size_t nextSize = byteRanges[nextChunk].size();
      VELOX_CHECK_LE(
          ioSize,
          std::numeric_limits<size_t>::max() - ioOffset,
          "S3 coalesced range end overflow");
      const auto ioEnd = ioOffset + ioSize;
      if (nextOffset < ioEnd ||
          nextOffset - ioEnd > prepared->coalesceGapBytes) {
        break;
      }
      VELOX_CHECK_LE(
          nextSize,
          std::numeric_limits<size_t>::max() - nextOffset,
          "S3 next range end overflow");
      const auto nextEnd = nextOffset + nextSize;
      ioSize = nextEnd - ioOffset;
      prepared->hostSourceOffsets[nextChunk] =
          destinationOffset + nextOffset - ioOffset;
      ++nextChunk;
    }
    if (ioSize != 0) {
      ioOffsets.push_back(ioOffset);
      ioSizes.push_back(ioSize);
      destinationOffsets.push_back(destinationOffset);
      VELOX_CHECK_LE(
          ioSize,
          std::numeric_limits<size_t>::max() - hostBufferSize,
          "S3 host buffer size overflow");
      hostBufferSize += ioSize;
    }
    chunk = nextChunk;
  }

  VELOX_CHECK_NOT_NULL(pageableHostBufferEnv);
  prepared->hostBuffer = std::make_shared<PinnedHostBuffer>(
      hostBufferSize,
      nullptr,
      envBytesOrZero(pageableHostBufferEnv) == 0);
  const auto bytes = packedHostRead(
      ioOffsets,
      ioSizes,
      prepared->hostBuffer->data(),
      destinationOffsets);
  const auto expected =
      std::accumulate(ioSizes.begin(), ioSizes.end(), size_t{0});
  VELOX_CHECK_EQ(bytes, expected, "Short packed S3 range batch");
  return prepared;
#else
  return nullptr;
#endif
}

FetchedDeviceByteRanges copyPreparedByteRangesToDevice(
    std::shared_ptr<PreparedHostByteRanges> prepared,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto totalStart = std::chrono::steady_clock::now();
  constexpr auto kBufferPaddingMultiple = 8;
  VELOX_CHECK_NOT_NULL(prepared);
  VELOX_CHECK_NOT_NULL(prepared->hostBuffer);
  VELOX_CHECK_EQ(prepared->hostSourceOffsets.size(), byteRanges.size());

  NativeS3H2dPermit h2dPermit(prepared->nativeCppHostBatch);
  const auto allocationBytes = cudf::util::round_up_safe<size_t>(
      prepared->totalSize, kBufferPaddingMultiple);
  std::optional<DeviceMemoryAdmissionReservation> deviceMemoryAdmission;
  const auto admissionCapacity =
      envBytesOrZero("GLUTEN_CRT_S3_DEVICE_ADMISSION_BYTES");
  if (admissionCapacity != 0) {
    VELOX_CHECK_LE(
        allocationBytes,
        admissionCapacity,
        "S3 device allocation {} exceeds admission capacity {}",
        allocationBytes,
        admissionCapacity);
    int device = -1;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    const auto reserveBytes =
        envBytesOrZero("GLUTEN_CRT_S3_DEVICE_HEADROOM_BYTES");
    const auto started = std::chrono::steady_clock::now();
    constexpr auto kAdmissionTimeout = std::chrono::seconds(120);
    for (;;) {
      const auto headroom = captureDeviceAllocationHeadroom();
      const auto requiredHeadroom =
          allocationBytes >
              std::numeric_limits<std::size_t>::max() - reserveBytes
          ? std::numeric_limits<std::size_t>::max()
          : allocationBytes + reserveBytes;
      if (headroom.cudaValid &&
          headroom.allocatableBytes() >= requiredHeadroom) {
        deviceMemoryAdmission = tryAcquireDeviceMemoryAdmission(
            device, allocationBytes, admissionCapacity);
        if (deviceMemoryAdmission.has_value()) {
          break;
        }
      }
      if (std::chrono::steady_clock::now() - started >=
          kAdmissionTimeout) {
        VELOX_FAIL(
            "Timed out waiting for S3 device admission: bytes={}, "
            "capacityBytes={}, reserveBytes={}, allocatableBytes={}",
            allocationBytes,
            admissionCapacity,
            reserveBytes,
            headroom.allocatableBytes());
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  std::vector<rmm::device_buffer> columnChunkBuffers;
  columnChunkBuffers.emplace_back(allocationBytes, stream, mr);
  const auto allocationDone = std::chrono::steady_clock::now();
  auto* bufferData =
      static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::vector<cudf::device_span<const uint8_t>> columnChunkData;
  columnChunkData.reserve(byteRanges.size());
  size_t deviceOffset = 0;
  for (size_t chunk = 0; chunk < byteRanges.size(); ++chunk) {
    const auto chunkSize = static_cast<size_t>(byteRanges[chunk].size());
    columnChunkData.emplace_back(bufferData + deviceOffset, chunkSize);
    if (prepared->coalesceGapBytes != 0 && chunkSize != 0) {
      std::memmove(
          prepared->hostBuffer->data() + deviceOffset,
          prepared->hostBuffer->data() +
              prepared->hostSourceOffsets[chunk],
          chunkSize);
    }
    deviceOffset += chunkSize;
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      bufferData,
      prepared->hostBuffer->data(),
      prepared->totalSize,
      cudaMemcpyHostToDevice,
      stream.value()));
  stream.synchronize();
  const auto copyDone = std::chrono::steady_clock::now();
  // Return pooled pinned memory before publishing the device spans. Timing
  // this separately catches CUDA-wide synchronization in host unregister.
  prepared.reset();
  const auto recycleDone = std::chrono::steady_clock::now();
  h2dPermit.release();
  static std::atomic<uint64_t> completed{0};
  static std::atomic<uint64_t> bytes{0};
  static std::atomic<uint64_t> allocationNanos{0};
  static std::atomic<uint64_t> copyNanos{0};
  static std::atomic<uint64_t> recycleNanos{0};
  static std::atomic<uint64_t> totalNanos{0};
  const auto count = completed.fetch_add(1, std::memory_order_relaxed) + 1;
  bytes.fetch_add(allocationBytes, std::memory_order_relaxed);
  allocationNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          allocationDone - totalStart)
          .count(),
      std::memory_order_relaxed);
  copyNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          copyDone - allocationDone)
          .count(),
      std::memory_order_relaxed);
  recycleNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          recycleDone - copyDone)
          .count(),
      std::memory_order_relaxed);
  totalNanos.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          recycleDone - totalStart)
          .count(),
      std::memory_order_relaxed);
  if (envBytesOrZero("GLUTEN_CPP_S3_DIAGNOSTICS") != 0 &&
      count % 64 == 0) {
    LOG(WARNING)
        << "CPP_S3_H2D completed=" << count
        << " bytes=" << bytes.load(std::memory_order_relaxed)
        << " avgAllocationMs="
        << allocationNanos.load(std::memory_order_relaxed) / count / 1e6
        << " avgCopySyncMs="
        << copyNanos.load(std::memory_order_relaxed) / count / 1e6
        << " avgRecycleMs="
        << recycleNanos.load(std::memory_order_relaxed) / count / 1e6
        << " avgTotalMs="
        << totalNanos.load(std::memory_order_relaxed) / count / 1e6;
  }
  return {
      std::move(columnChunkBuffers),
      std::move(columnChunkData),
      std::async(std::launch::deferred, [] {}),
      std::move(deviceMemoryAdmission)};
}

FetchedDeviceByteRanges fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  // Pad buffer sizes to be a multiple of 8 bytes. Required by
  // `decode_page_data_kernel` in cuDF Parquet reader.
  constexpr auto kBufferPaddingMultiple = 8;

  // Allocate device spans for each column chunk
  std::vector<cudf::device_span<const uint8_t>> columnChunkData{};
  columnChunkData.reserve(byteRanges.size());

  // Total IO size across all byte ranges
  auto totalSize = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) { return acc + byteRange.size(); });

  if (auto prepared = prepareByteRangesToHost(dataSource, byteRanges)) {
    return copyPreparedByteRangesToDevice(
        std::move(prepared), byteRanges, stream, mr);
  }

  using PackedHostRead = std::function<size_t(
      const std::vector<size_t>&,
      const std::vector<size_t>&,
      uint8_t*,
      const std::vector<size_t>&)>;
  PackedHostRead packedHostRead;
  const char* pageableHostBufferEnv = nullptr;
  bool nativeCppHostBatch = false;
  if (auto* crtSource = dynamic_cast<CrtS3DataSource*>(dataSource.get())) {
    packedHostRead =
        [crtSource](
            const std::vector<size_t>& offsets,
            const std::vector<size_t>& sizes,
            uint8_t* destination,
            const std::vector<size_t>& destinationOffsets) {
          return crtSource->readRanges(
              offsets, sizes, destination, destinationOffsets);
        };
    pageableHostBufferEnv = "GLUTEN_CRT_S3_PAGEABLE_HOST_BUFFER";
  } else if (
      envBytesOrZero("GLUTEN_CPP_S3_HOST_BATCH") != 0) {
    if (auto* nativeSource =
            dynamic_cast<KvikioS3DataSource*>(dataSource.get())) {
      packedHostRead =
          [nativeSource](
              const std::vector<size_t>& offsets,
              const std::vector<size_t>& sizes,
              uint8_t* destination,
              const std::vector<size_t>& destinationOffsets) {
            return nativeSource->readRanges(
                offsets, sizes, destination, destinationOffsets);
          };
      pageableHostBufferEnv = "GLUTEN_CPP_S3_PAGEABLE_HOST_BUFFER";
      nativeCppHostBatch = true;
    }
  }

  // Remote range sources read directly into their final packed host offsets.
  // Do not reserve the matching device buffer until S3 has completed: at high
  // scan parallelism, holding a device allocation per in-flight remote file
  // exhausts RMM before any file can reach decode. One H2D copy publishes the
  // complete projected batch after all native range futures finish.
  if (packedHostRead) {
    std::vector<size_t> ioOffsets;
    std::vector<size_t> ioSizes;
    std::vector<size_t> destinationOffsets;
    std::vector<size_t> hostSourceOffsets(byteRanges.size());
    const auto coalesceGapBytes =
        envBytesOrZero("GLUTEN_CRT_S3_RANGE_COALESCE_GAP_BYTES");
    size_t hostBufferSize = 0;
    for (size_t chunk = 0; chunk < byteRanges.size();) {
      const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
      auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
      const auto destinationOffset = hostBufferSize;
      hostSourceOffsets[chunk] = destinationOffset;
      size_t nextChunk = chunk + 1;
      while (nextChunk < byteRanges.size()) {
        const size_t nextOffset = byteRanges[nextChunk].offset();
        const size_t nextSize = byteRanges[nextChunk].size();
        VELOX_CHECK_LE(
            ioSize,
            std::numeric_limits<size_t>::max() - ioOffset,
            "CRT S3 coalesced range end overflow");
        const auto ioEnd = ioOffset + ioSize;
        if (nextOffset < ioEnd ||
            nextOffset - ioEnd > coalesceGapBytes) {
          break;
        }
        VELOX_CHECK_LE(
            nextSize,
            std::numeric_limits<size_t>::max() - nextOffset,
            "CRT S3 next range end overflow");
        const auto nextEnd = nextOffset + nextSize;
        ioSize = nextEnd - ioOffset;
        hostSourceOffsets[nextChunk] =
            destinationOffset + nextOffset - ioOffset;
        ++nextChunk;
      }
      if (ioSize != 0) {
        ioOffsets.push_back(ioOffset);
        ioSizes.push_back(ioSize);
        destinationOffsets.push_back(destinationOffset);
        VELOX_CHECK_LE(
            ioSize,
            std::numeric_limits<size_t>::max() - hostBufferSize,
            "CRT S3 host buffer size overflow");
        hostBufferSize += ioSize;
      }
      chunk = nextChunk;
    }

    VELOX_CHECK_NOT_NULL(pageableHostBufferEnv);
    const auto preferPinned =
        envBytesOrZero(pageableHostBufferEnv) == 0;
    auto hostBuffer = std::make_shared<PinnedHostBuffer>(
        hostBufferSize, nullptr, preferPinned);
    const auto bytes = packedHostRead(
        ioOffsets, ioSizes, hostBuffer->data(), destinationOffsets);
    const auto expected =
        std::accumulate(ioSizes.begin(), ioSizes.end(), size_t{0});
    VELOX_CHECK_EQ(bytes, expected, "Short AWS CRT S3 range batch");

    // Keep remote host reads independent from GPU backpressure. A large Velox
    // IO pool can therefore keep curl-multi full, while only a small number of
    // completed host batches allocate device memory and copy concurrently.
    // The permit ends immediately after H2D synchronization; it is not held
    // through parquet decode or the lifetime of the returned device buffer.
    NativeS3H2dPermit h2dPermit(nativeCppHostBatch);
    const auto allocationBytes = cudf::util::round_up_safe<size_t>(
        totalSize, kBufferPaddingMultiple);
    std::optional<DeviceMemoryAdmissionReservation> deviceMemoryAdmission;
    const auto admissionCapacity =
        envBytesOrZero("GLUTEN_CRT_S3_DEVICE_ADMISSION_BYTES");
    if (admissionCapacity != 0) {
      VELOX_CHECK_LE(
          allocationBytes,
          admissionCapacity,
          "CRT S3 device allocation {} exceeds admission capacity {}",
          allocationBytes,
          admissionCapacity);
      int device = -1;
      CUDF_CUDA_TRY(cudaGetDevice(&device));
      const auto reserveBytes =
          envBytesOrZero("GLUTEN_CRT_S3_DEVICE_HEADROOM_BYTES");
      const auto started = std::chrono::steady_clock::now();
      constexpr auto kAdmissionTimeout = std::chrono::seconds(120);
      for (;;) {
        const auto headroom = captureDeviceAllocationHeadroom();
        const auto requiredHeadroom =
            allocationBytes >
                std::numeric_limits<std::size_t>::max() - reserveBytes
            ? std::numeric_limits<std::size_t>::max()
            : allocationBytes + reserveBytes;
        if (headroom.cudaValid &&
            headroom.allocatableBytes() >= requiredHeadroom) {
          deviceMemoryAdmission = tryAcquireDeviceMemoryAdmission(
              device, allocationBytes, admissionCapacity);
          if (deviceMemoryAdmission.has_value()) {
            const auto waitedMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count();
            if (waitedMs > 0) {
              LOG(INFO) << "CRT_S3_DEVICE_ADMISSION acquired bytes="
                        << allocationBytes
                        << " capacityBytes=" << admissionCapacity
                        << " reserveBytes=" << reserveBytes
                        << " waitedMs=" << waitedMs
                        << " allocatableBytes="
                        << headroom.allocatableBytes();
            }
            break;
          }
        }
        if (std::chrono::steady_clock::now() - started >=
            kAdmissionTimeout) {
          VELOX_FAIL(
              "Timed out waiting {} ms for CRT S3 device admission: "
              "bytes={}, capacityBytes={}, reserveBytes={}, "
              "allocatableBytes={}",
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  kAdmissionTimeout)
                  .count(),
              allocationBytes,
              admissionCapacity,
              reserveBytes,
              headroom.allocatableBytes());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    }

    std::vector<rmm::device_buffer> columnChunkBuffers;
    columnChunkBuffers.emplace_back(allocationBytes, stream, mr);
    auto* bufferData =
        static_cast<uint8_t*>(columnChunkBuffers.back().data());
    std::vector<cudf::device_span<const uint8_t>> columnChunkData;
    columnChunkData.reserve(byteRanges.size());
    size_t deviceOffset = 0;
    for (size_t chunk = 0; chunk < byteRanges.size(); ++chunk) {
      const auto chunkSize = static_cast<size_t>(byteRanges[chunk].size());
      columnChunkData.emplace_back(bufferData + deviceOffset, chunkSize);
      if (coalesceGapBytes != 0 && chunkSize != 0) {
        // Coalesced S3 responses include the gaps between projected column
        // chunks. Compact the selected chunks in place so the host layout
        // matches the packed device layout. Source offsets are never before
        // their packed destinations, so forward memmove is safe and avoids
        // allocating a second host buffer.
        std::memmove(
            hostBuffer->data() + deviceOffset,
            hostBuffer->data() + hostSourceOffsets[chunk],
            chunkSize);
      }
      deviceOffset += chunkSize;
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        bufferData,
        hostBuffer->data(),
        totalSize,
        cudaMemcpyHostToDevice,
        stream.value()));
    stream.synchronize();
    h2dPermit.release();
    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::async(std::launch::deferred, [] {}),
        std::move(deviceMemoryAdmission)};
  }

  // Allocate single device buffer for all column chunks
  std::vector<rmm::device_buffer> columnChunkBuffers{};
  columnChunkBuffers.emplace_back(
      cudf::util::round_up_safe<size_t>(totalSize, kBufferPaddingMultiple),
      stream,
      mr);

  // Compute device spans for each column chunk
  auto bufferData = static_cast<uint8_t*>(columnChunkBuffers.back().data());
  std::ignore = std::accumulate(
      byteRanges.begin(),
      byteRanges.end(),
      std::size_t{0},
      [&](auto acc, const auto& byteRange) {
        columnChunkData.emplace_back(
            bufferData + acc, static_cast<size_t>(byteRange.size()));
        return acc + byteRange.size();
      });

  // For BufferedInputDataSource, enqueue reads into the buffer and launch the
  // actual load asynchronously.
  if (auto bufferedInput =
          dynamic_cast<BufferedInputDataSource*>(dataSource.get())) {
    auto iter =
        cuda::make_zip_iterator(byteRanges.begin(), columnChunkData.begin());
    std::for_each(
        iter, iter + byteRanges.size(), [bufferedInput](const auto& tuple) {
          const auto& byteRange = cuda::std::get<0>(tuple);
          const auto& destination = cuda::std::get<1>(tuple);
          bufferedInput->enqueueForDevice(
              static_cast<uint64_t>(byteRange.offset()),
              static_cast<uint64_t>(byteRange.size()),
              const_cast<uint8_t*>(destination.data()));
        });

    // load buffered input data source
    auto syncFunction = [](std::shared_ptr<cudf::io::datasource> dataSource,
                           rmm::cuda_stream_view stream) {
      auto buffer =
          checkedPointerCast<BufferedInputDataSource>(dataSource.get());
      buffer->load(stream);
    };

    return {
        std::move(columnChunkBuffers),
        std::move(columnChunkData),
        std::async(std::launch::deferred, syncFunction, dataSource, stream),
        std::nullopt};
  }

  // KvikIO dataSource: Impl borrowed from `fetch_byte_ranges_to_device_async()`
  // in `parquet_io_utils.cpp` in cuDF.
  std::vector<size_t> ioOffsets;
  std::vector<size_t> ioSizes;
  std::vector<uint8_t*> destinations;

  for (size_t chunk = 0; chunk < byteRanges.size();) {
    const auto ioOffset = static_cast<size_t>(byteRanges[chunk].offset());
    auto ioSize = static_cast<size_t>(byteRanges[chunk].size());
    size_t nextChunk = chunk + 1;
    while (nextChunk < byteRanges.size()) {
      const size_t nextOffset = byteRanges[nextChunk].offset();
      if (nextOffset != ioOffset + ioSize) {
        break;
      }
      ioSize += byteRanges[nextChunk].size();
      nextChunk++;
    }
    if (ioSize != 0) {
      ioOffsets.push_back(ioOffset);
      ioSizes.push_back(ioSize);
      destinations.push_back(
          const_cast<uint8_t*>(columnChunkData[chunk].data()));
    }
    chunk = nextChunk;
  }
  VELOX_CHECK_EQ(
      ioOffsets.size(),
      ioSizes.size(),
      "Number of IO offsets and sizes must be equal");
  VELOX_CHECK_EQ(
      ioSizes.size(),
      destinations.size(),
      "Number of IO sizes and destinations must be equal");

  auto iter = cuda::make_zip_iterator(
      ioOffsets.begin(), ioSizes.begin(), destinations.begin());

  std::vector<std::future<size_t>> deviceReadTasks;
  std::vector<std::future<size_t>> hostReadTasks;
  deviceReadTasks.reserve(ioOffsets.size());
  hostReadTasks.reserve(ioOffsets.size());

  // KvikIO's remote read uses internal worker streams. Synchronize preceding
  // consumer work before scheduling the batch; completion of each pread()
  // includes its H2D copy.
  stream.synchronize();

  {
    std::lock_guard<std::mutex> lock(ioBatchMutex());

    std::for_each(iter, iter + ioOffsets.size(), [&](const auto& tuple) {
      const auto ioOffset = cuda::std::get<0>(tuple);
      const auto ioSize = cuda::std::get<1>(tuple);
      const auto dest = cuda::std::get<2>(tuple);

      if (dataSource->supports_device_read() and
          dataSource->is_device_read_preferred(ioSize)) {
        deviceReadTasks.emplace_back(
            dataSource->device_read_async(ioOffset, ioSize, dest, stream));
      } else {
        // TODO(mh): We can't yet guarantee (without a safe thread pool) that
        // all `cudaMemcpyAsync`s will be launched by the time we release the
        // mutex. That said, this is a rare usecase as host-buffer data should
        // prefer using a `BufferedInputDataSource` datasource.
        hostReadTasks.emplace_back(
            std::async(
                std::launch::async,
                [dataSource, ioOffset, ioSize, dest, stream]() {
                  auto hostBuffer = dataSource->host_read(ioOffset, ioSize);
                  CUDF_CUDA_TRY(cudaMemcpyAsync(
                      dest,
                      hostBuffer->data(),
                      hostBuffer->size(),
                      cudaMemcpyHostToDevice,
                      stream.value()));
                  return ioSize;
                }));
      }
    });
  }

  auto syncFunction = [](decltype(hostReadTasks)&& hostReadTasks,
                         decltype(deviceReadTasks)&& deviceReadTasks) {
    for (auto& task : hostReadTasks) {
      task.get();
    }
    for (auto& task : deviceReadTasks) {
      task.get();
    }
  };

  return {
      std::move(columnChunkBuffers),
      std::move(columnChunkData),
      std::async(
          std::launch::deferred,
          std::move(syncFunction),
          std::move(hostReadTasks),
          std::move(deviceReadTasks)),
      std::nullopt};
}

} // namespace facebook::velox::cudf_velox::connector::hive
