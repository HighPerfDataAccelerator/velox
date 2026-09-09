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
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include <glog/logging.h>
#include <malloc.h>
#include <rmm/cuda_stream_view.hpp>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include "cuda_runtime.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"

namespace facebook::velox::ucx_exchange {

namespace {
void accountFreedHostBytesAndTrim(uint64_t bytes) {
  constexpr uint64_t kTrimInterval = 64ULL * 1024 * 1024;
  static std::atomic<uint64_t> freedSinceTrim{0};
  if (bytes == 0) {
    return;
  }
  const auto accumulated =
      freedSinceTrim.fetch_add(bytes, std::memory_order_acq_rel) + bytes;
  if (accumulated >= kTrimInterval) {
    const auto claimed = freedSinceTrim.exchange(0, std::memory_order_acq_rel);
    if (claimed >= kTrimInterval) {
      malloc_trim(0);
    }
  }
}

void retireRequest(
    std::shared_ptr<ucxx::Request>& current,
    std::vector<std::shared_ptr<ucxx::Request>>& inFlight) {
  inFlight.erase(
      std::remove_if(
          inFlight.begin(),
          inFlight.end(),
          [](const auto& request) {
            return request == nullptr || request->isCompleted();
          }),
      inFlight.end());
  if (current != nullptr && !current->isCompleted()) {
    inFlight.push_back(std::move(current));
  } else {
    current.reset();
  }
}

const folly::F14FastMap<UcxExchangeServer::ServerState, std::string_view>&
serverStateNames() {
  static const folly::
      F14FastMap<UcxExchangeServer::ServerState, std::string_view>
          kNames = {
              {UcxExchangeServer::ServerState::Created, "Created"},
              {UcxExchangeServer::ServerState::ReadyToTransfer,
               "ReadyToTransfer"},
              {UcxExchangeServer::ServerState::DataRequestReady,
               "DataRequestReady"},
              {UcxExchangeServer::ServerState::WaitingForDataFromQueue,
               "WaitingForDataFromQueue"},
              {UcxExchangeServer::ServerState::DataReady, "DataReady"},
              {UcxExchangeServer::ServerState::WaitingForHostStage,
               "WaitingForHostStage"},
              {UcxExchangeServer::ServerState::WaitingForSendComplete,
               "WaitingForSendComplete"},
              {UcxExchangeServer::ServerState::WaitingForIntraNodeRetrieve,
               "WaitingForIntraNodeRetrieve"},
              {UcxExchangeServer::ServerState::Done, "Done"},
          };
  return kNames;
}

bool intraNodeProducerPollRequeueEnabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("GLUTEN_UCX_INTRANODE_PRODUCER_POLL_REQUEUE");
    return value != nullptr && value[0] != '\0' &&
        !(value[0] == '0' && value[1] == '\0');
  }();
  return enabled;
}

bool reuseControlEndpointForBulk() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("GLUTEN_UCX_REUSE_CONTROL_ENDPOINT_FOR_BULK");
    return value != nullptr && value[0] != '\0' &&
        !(value[0] == '0' && value[1] == '\0');
  }();
  return enabled;
}

int64_t intraNodeProducerPollRequeueLimit() {
  static const int64_t limit = [] {
    const char* value =
        std::getenv("GLUTEN_UCX_INTRANODE_PRODUCER_POLL_REQUEUE_LIMIT");
    if (value == nullptr || value[0] == '\0') {
      return int64_t{-1};
    }
    try {
      return static_cast<int64_t>(std::stoll(value));
    } catch (...) {
      return int64_t{-1};
    }
  }();
  return limit;
}

int64_t maxInFlightSendHostBytes() {
  static const int64_t limit = [] {
    if (const char* value =
            std::getenv("GLUTEN_UCX_MAX_INFLIGHT_SEND_HOST_BYTES")) {
      try {
        const auto parsed = static_cast<int64_t>(std::stoll(value));
        if (parsed > 0) {
          return parsed;
        }
      } catch (...) {
      }
    }
    return static_cast<int64_t>(2) * 1024 * 1024 * 1024;
  }();
  return limit;
}

std::atomic<int64_t> inFlightSendHostBytes{0};

bool pinnedHostStagingEnabled() {
  static const bool enabled = [] {
    const auto* value = std::getenv("GLUTEN_UCX_PINNED_HOST_STAGING");
    return value != nullptr && std::string_view(value) == "1";
  }();
  return enabled;
}

class PinnedSendBufferPool {
 public:
  static PinnedSendBufferPool& instance() {
    static PinnedSendBufferPool pool;
    return pool;
  }

  std::pair<uint8_t*, size_t> acquire(size_t requested) {
    const auto capacity = sizeClass(requested);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto& buffers = buffers_[capacity];
      if (!buffers.empty()) {
        auto* data = buffers.back();
        buffers.pop_back();
        cachedBytes_ -= capacity;
        return {data, capacity};
      }
    }
    void* allocation = nullptr;
    if (cudaHostAlloc(&allocation, capacity, cudaHostAllocPortable) !=
        cudaSuccess) {
      cudaGetLastError();
      return {nullptr, 0};
    }
    return {static_cast<uint8_t*>(allocation), capacity};
  }

  void release(uint8_t* data, size_t capacity) {
    bool cache = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (capacity <= maxCachedBytes_ &&
          cachedBytes_ <= maxCachedBytes_ - capacity) {
        buffers_[capacity].push_back(data);
        cachedBytes_ += capacity;
        cache = true;
      }
    }
    if (!cache) {
      cudaFreeHost(data);
    }
  }

  ~PinnedSendBufferPool() {
    for (auto& [capacity, buffers] : buffers_) {
      for (auto* data : buffers) {
        cudaFreeHost(data);
      }
    }
  }

 private:
  static size_t sizeClass(size_t requested) {
    constexpr size_t kMinimumClass = 64UL << 10;
    size_t capacity = kMinimumClass;
    while (capacity < requested &&
           capacity <= std::numeric_limits<size_t>::max() / 2) {
      capacity *= 2;
    }
    return capacity < requested ? requested : capacity;
  }

  static size_t maxCachedBytes() {
    constexpr size_t kDefault = 512ULL << 20;
    const auto* value = std::getenv("GLUTEN_UCX_PINNED_SEND_POOL_MAX_BYTES");
    if (value == nullptr || value[0] == '\0') {
      return kDefault;
    }
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' ? static_cast<size_t>(parsed)
                                        : kDefault;
  }

  PinnedSendBufferPool() : maxCachedBytes_(maxCachedBytes()) {}

  std::mutex mutex_;
  std::unordered_map<size_t, std::vector<uint8_t*>> buffers_;
  size_t cachedBytes_{0};
  const size_t maxCachedBytes_;
};

class PinnedSendBuffer {
 public:
  explicit PinnedSendBuffer(size_t size) {
    std::tie(data_, capacity_) = PinnedSendBufferPool::instance().acquire(size);
  }

  ~PinnedSendBuffer() {
    if (data_ != nullptr) {
      PinnedSendBufferPool::instance().release(data_, capacity_);
    }
  }

  PinnedSendBuffer(const PinnedSendBuffer&) = delete;
  PinnedSendBuffer& operator=(const PinnedSendBuffer&) = delete;

  uint8_t* data() const {
    return data_;
  }

 private:
  uint8_t* data_{nullptr};
  size_t capacity_{0};
};

bool tryReserveSendHostBytes(int64_t bytes) {
  VELOX_CHECK_GE(bytes, 0);
  auto current = inFlightSendHostBytes.load(std::memory_order_relaxed);
  while (true) {
    // Permit one oversized transfer when no other host staging is active. A
    // single packed table cannot be split by the current wire protocol, and
    // rejecting it forever would deadlock the exchange.
    if (current > 0 && current + bytes > maxInFlightSendHostBytes()) {
      return false;
    }
    if (inFlightSendHostBytes.compare_exchange_weak(
            current,
            current + bytes,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return true;
    }
  }
}

void releaseSendHostBytes(int64_t bytes) {
  if (bytes <= 0) {
    return;
  }
  const auto previous =
      inFlightSendHostBytes.fetch_sub(bytes, std::memory_order_acq_rel);
  VELOX_CHECK_GE(previous, bytes);
}

} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeServer,
    ServerState,
    serverStateNames)

// Context wrappers for UCXX tagSend callbackData. These decouple the
// ucxx::Request lifetime (which must survive for UCP wireup replay) from
// the buffer lifetime (which should be freed promptly after DMA completes).
//
// The Request holds a shared_ptr to the context via callbackData. The
// context holds a shared_ptr to the actual buffer. When the send completion
// callback fires, it moves the buffer out of the context, releasing the GPU
// (or CPU) memory. The context remains alive as an empty shell for the
// lifetime of the Request, which is safe and costs negligible memory.
struct MetaSendContext {
  std::shared_ptr<uint8_t> metadata;
};

struct DataSendContext {
  std::shared_ptr<cudf::packed_columns> data;
  // The UCX build used by Gluten MPP may not include CUDA memory-type
  // transports.  In that case handing an rmm device pointer to tagSend makes
  // the shared-memory transport memcpy from an inaccessible address.  Keep a
  // host staging buffer alive with the request and let UCX move host memory.
  std::shared_ptr<std::vector<uint8_t>> hostData;
  std::shared_ptr<PinnedSendBuffer> pinnedHostData;
  cudaEvent_t stageEvent{nullptr};
  int64_t reservedHostBytes{0};
  std::chrono::time_point<std::chrono::high_resolution_clock> stageStart;

  ~DataSendContext() {
    destroyStageEvent(/*synchronize=*/true);
    releaseHostReservation();
  }

  void destroyStageEvent(bool synchronize = false) {
    if (stageEvent == nullptr) {
      return;
    }
    // Destructors must not throw. On cancellation, wait before member
    // destruction returns the pinned buffer to the pool.
    if (synchronize) {
      cudaEventSynchronize(stageEvent);
    }
    cudaEventDestroy(stageEvent);
    stageEvent = nullptr;
  }

  bool reserveHostBytes(int64_t bytes) {
    VELOX_CHECK_EQ(reservedHostBytes, 0);
    if (!tryReserveSendHostBytes(bytes)) {
      return false;
    }
    reservedHostBytes = bytes;
    return true;
  }

  void releaseHostReservation() {
    if (reservedHostBytes > 0) {
      releaseSendHostBytes(reservedHostBytes);
      reservedHostBytes = 0;
    }
  }
};

void UcxExchangeServer::setState(ServerState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private
UcxExchangeServer::UcxExchangeServer(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> controlEndpointRef,
    uint64_t remoteWorkerId,
    std::string remoteDataHost,
    uint16_t remoteDataPort,
    const PartitionKey& key,
    bool isIntraNodeTransfer)
    : CommElement(communicator, std::move(controlEndpointRef)),
      partitionKey_(key),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      isIntraNodeTransfer_(isIntraNodeTransfer),
      remoteWorkerId_(remoteWorkerId),
      remoteDataHost_(std::move(remoteDataHost)),
      remoteDataPort_(remoteDataPort),
      queueMgr_(UcxOutputQueueManager::getInstanceRef()) {
  setState(ServerState::Created);

  if (isIntraNodeTransfer_) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " Detected same-node source (intra-node transfer) for "
            << partitionKey_.toString();
  }
}

// static
std::shared_ptr<UcxExchangeServer> UcxExchangeServer::create(
    const std::shared_ptr<Communicator> communicator,
    std::shared_ptr<EndpointRef> controlEndpointRef,
    uint64_t remoteWorkerId,
    std::string remoteDataHost,
    uint16_t remoteDataPort,
    const PartitionKey& key,
    bool isIntraNodeTransfer) {
  auto ptr = std::shared_ptr<UcxExchangeServer>(new UcxExchangeServer(
      communicator,
      std::move(controlEndpointRef),
      remoteWorkerId,
      std::move(remoteDataHost),
      remoteDataPort,
      key,
      isIntraNodeTransfer));
  return ptr;
}

void UcxExchangeServer::process() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  switch (state_) {
    case ServerState::Created: {
      if (!isIntraNodeTransfer_ && !dataEndpointRef_) {
        if (reuseControlEndpointForBulk()) {
          // UCX endpoints are bidirectional. The accepted control endpoint is
          // already connected to this source and can carry tagged GPU buffers
          // as well as the small handshake response. Reusing it removes the
          // second listener connection and its multi-second wire-up bubble.
          dataEndpointRef_ = endpointRef_;
          LOG(INFO) << "Reusing UCX control endpoint for bulk data: task="
                    << partitionKey_.toString()
                    << " peer=" << endpointRef_->getPeerAddress();
          setState(ServerState::ReadyToTransfer);
          wakeCommunicator();
          break;
        }
        auto communicator = tryCommunicator();
        if (!communicator) {
          close();
          return;
        }
        try {
          dataEndpointRef_ = communicator->getOrCreateDataEndpoint(
              remoteWorkerId_, remoteDataHost_, remoteDataPort_);
          dataEndpointRef_->addCommElem(getSelfPtr());
        } catch (const std::exception& e) {
          LOG(ERROR) << "Failed to create deferred UCX bulk-data endpoint: "
                     << "task=" << partitionKey_.toString()
                     << " remoteWorkerId=" << remoteWorkerId_
                     << " error=" << e.what();
          if (endpointRef_) {
            communicator->deferEndpointCleanup(endpointRef_);
          }
          close();
          return;
        }
      }
      setState(ServerState::ReadyToTransfer);
      wakeCommunicator();
      break;
    }
    case ServerState::ReadyToTransfer:
      // Count-only / rendezvous push (Presto-style): no consumer credit
      // request. Go straight to dequeue + send; the data tagSend blocks at
      // rendezvous until the source posts its matching tagRecv
      // (getMetadata/getData), which is the sole flow-control mechanism.
      setState(ServerState::DataRequestReady);
      wakeCommunicator();
      break;
    case ServerState::DataRequestReady:
      setState(ServerState::WaitingForDataFromQueue);
      // Register the callback with the destination queue to get data.
      // If the queue doesn't exist yet, getData will create an empty
      // queue and the callback will be triggered once the corresponding
      // source task has initialized the queue and added data to it.
      // Use weak_ptr to prevent use-after-free if close() is called during
      // callback
      {
        std::weak_ptr<UcxExchangeServer> weakQueue = weak_from_this();
        queueMgr_->getData(
            partitionKey_.taskId,
            partitionKey_.destination,
            // Unbounded per-fetch cap; rendezvous + queue-occupancy
            // backpressure are the flow control (no byte-credit).
            std::numeric_limits<uint64_t>::max(),
            static_cast<int64_t>(sequenceNumber_),
            [weakQueue](
                std::shared_ptr<cudf::packed_columns> data,
                int64_t sequence,
                std::vector<int64_t> remainingBytes) {
              auto self = weakQueue.lock();
              if (!self) {
                return; // Object was destroyed, safe to ignore
              }
              // Check if close() was called - avoid processing if we're
              // shutting down
              if (self->closed_.load(std::memory_order_acquire)) {
                VLOG(3) << "@" << self->partitionKey_.taskId
                        << " getData callback called after close, ignoring";
                return;
              }
              if (sequence != static_cast<int64_t>(self->sequenceNumber_)) {
                // The destination queue has already advanced beyond this
                // duplicate/retried server.  Close only this stale server.
                // In particular, do not deleteResults(): another active
                // server owns the already-advanced queue and may still need
                // its remaining pages.
                LOG(WARNING)
                    << "Closing stale UCX exchange server for task="
                    << self->partitionKey_.taskId
                    << " destination=" << self->partitionKey_.destination
                    << " requestedSequence=" << self->sequenceNumber_
                    << " acknowledgedSequence=" << sequence;
                self->skipQueueDeleteOnClose_.store(
                    true, std::memory_order_release);
                self->setState(ServerState::Done);
                self->wakeCommunicator();
                return;
              }
              // This upcall may be called from another thread than the
              // communicator thread. It is called
              // when data on the queue becomes available.
              VLOG(3) << "@" << self->partitionKey_.taskId
                      << " Found data for client: "
                      << self->partitionKey_.toString()
                      << " sequence=" << sequence;
              std::lock_guard<std::recursive_mutex> lock(self->dataMutex_);
              VELOX_CHECK(
                  self->dataPtr_ == nullptr,
                  "Data pointer exists: Illegal state!");
              self->dataPtr_ = std::move(data);
              self->setState(ServerState::DataReady);
              self->wakeCommunicator();
            });
      }
      wakeCommunicator();
      break;
    case ServerState::WaitingForDataFromQueue:
      // Waiting for data is handled by an upcall from the data queue. Nothing
      // to do
      break;
    case ServerState::DataReady:
      sendData();
      break;
    case ServerState::WaitingForHostStage: {
      std::shared_ptr<DataSendContext> dataCtx;
      {
        std::lock_guard<std::recursive_mutex> lock(dataMutex_);
        dataCtx = pendingDataSend_;
      }
      VELOX_CHECK_NOT_NULL(dataCtx);
      VELOX_CHECK_NOT_NULL(dataCtx->stageEvent);
      const auto eventStatus = cudaEventQuery(dataCtx->stageEvent);
      if (eventStatus == cudaErrorNotReady) {
        // Requeue at the tail so other UCX work progresses while the D2H copy
        // runs. The communicator already busy-progresses active exchanges.
        wakeCommunicator();
        break;
      }
      CUDF_CUDA_TRY(eventStatus);
      dataCtx->destroyStageEvent();
      {
        std::lock_guard<std::recursive_mutex> lock(dataMutex_);
        VELOX_CHECK(pendingDataSend_ == dataCtx);
        pendingDataSend_.reset();
      }
      postDataSend(dataCtx);
      break;
    }
    case ServerState::WaitingForSendComplete:
      // Waiting for send complete is handled by an upcall from UCXX. Nothing to
      // do
      break;
    case ServerState::WaitingForIntraNodeRetrieve:
      // Intra-node transfer: the registry re-enqueues us when the source has
      // retrieved the data. Do a non-blocking check only for that wakeup (or a
      // defensive spurious work item); do not self-requeue and spin.
      if (intraNodeRetrieveFuture_.valid()) {
        auto status =
            intraNodeRetrieveFuture_.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
          intraNodeRetrieveFuture_.get(); // Clear the future
          intraNodePollCount_ = 0;
          onIntraNodeRetrieveComplete();
        } else if (
            intraNodeProducerPollRequeueEnabled() &&
            (intraNodeProducerPollRequeueLimit() < 0 ||
             intraNodePollCount_ <
                 static_cast<uint32_t>(intraNodeProducerPollRequeueLimit()))) {
          ++intraNodePollCount_;
          if (intraNodePollCount_ % 100 == 0) {
            VLOG(2) << "[INTRA] [ExSrv " << partitionKey_.toString()
                    << " seq=" << sequenceNumber_
                    << "] still waiting for source retrieval, polls="
                    << intraNodePollCount_;
          }
          wakeCommunicator();
        }
      }
      break;
    case ServerState::Done:
      close();
      // In control-endpoint reuse mode both members reference the same
      // EndpointRef and this server was registered on it only once.
      if (dataEndpointRef_ == endpointRef_) {
        dataEndpointRef_.reset();
      }
      if (endpointRef_) {
        endpointRef_->removeCommElem(getSelfPtr());
        endpointRef_ = nullptr;
      }
      if (dataEndpointRef_) {
        dataEndpointRef_->removeCommElem(getSelfPtr());
        dataEndpointRef_ = nullptr;
      }
      break;
  };
}

void UcxExchangeServer::close() {
  // Use memory_order_acq_rel to ensure proper synchronization with callbacks
  // that check closed_ with memory_order_acquire.
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(
          expected, desired, std::memory_order_acq_rel)) {
    return; // already closed.
  }
  VLOG(2) << "[UCX-SERVER-CLOSE] task=" << partitionKey_.taskId
          << " key=" << partitionKey_.toString() << " peer="
          << (endpointRef_ ? endpointRef_->getPeerAddress() : "(unknown)")
          << " state=" << toName(getState()) << " seq=" << sequenceNumber_
          << " hasMetaRequest=" << (metaRequest_ != nullptr)
          << " hasDataRequest=" << (dataRequest_ != nullptr)
          << " hasDataPtr=" << (dataPtr_ != nullptr);

  if (queueMgr_ && !skipQueueDeleteOnClose_.load(std::memory_order_acquire)) {
    queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
  }

  // Cancel any outstanding requests. With weak_ptr callbacks, the callbacks
  // will safely no-op if we're destroyed before they complete.
  if (metaRequest_ && !metaRequest_->isCompleted()) {
    metaRequest_->cancel();
  }
  if (dataRequest_ && !dataRequest_->isCompleted()) {
    dataRequest_->cancel();
  }

  // Move all requests to the Communicator's deferred list so the GPU
  // buffers they reference (via their arg shared_ptr) stay alive until
  // UCX has fully processed any in-flight operations.
  auto communicator = communicator_.lock();
  if (communicator) {
    if (metaRequest_) {
      communicator->deferRequestCleanup(std::move(metaRequest_));
    }
    if (dataRequest_) {
      communicator->deferRequestCleanup(std::move(dataRequest_));
    }
    for (auto& req : completedRequests_) {
      communicator->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }

  if (communicator) {
    communicator->unregister(getSelfPtr());
  }
}

std::string UcxExchangeServer::toString() {
  std::stringstream out;
  out << "[ExSrv " << partitionKey_.toString() << " - " << sequenceNumber_
      << "]";
  return out.str();
}

// ------ private methods ---------

std::shared_ptr<UcxExchangeServer> UcxExchangeServer::getSelfPtr() {
  return shared_from_this();
}

void UcxExchangeServer::wakeCommunicator() {
  if (auto communicator = tryCommunicator()) {
    communicator->addToWorkQueue(getSelfPtr());
  }
}

void UcxExchangeServer::sendData() {
  auto communicator = tryCommunicator();
  if (!communicator) {
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);

  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrv "
          << partitionKey_.toString() << " seq=" << sequenceNumber_
          << "] sendData hasData=" << (dataPtr_ != nullptr)
          << (dataPtr_ && dataPtr_->gpu_data
                  ? " size=" + std::to_string(dataPtr_->gpu_data->size())
                  : "");

  if (isIntraNodeTransfer_) {
    // INTRA-NODE TRANSFER PATH: Use registry for all communication, no UCXX
    // needed
    sendStart_ = std::chrono::high_resolution_clock::now();

    if (dataPtr_) {
      bytes_ = dataPtr_->gpu_data->size();

      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing data for sequence "
              << sequenceNumber_ << " of size " << bytes_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      const auto stream = dataPtr_->gpu_data->stream();
      // The consumer tags uniquely owned pages with this stream so downstream
      // reads and stream-ordered async frees remain ordered with the buffer.
      // dataPtr_ is already a shared_ptr, pass directly to share ownership.
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key,
              dataPtr_,
              stream,
              /*atEnd=*/false,
              makeIntraNodeRetrieveWakeup());
      dataPtr_.reset();
      intraNodeAtEndPublished_ = false;

      // Go dormant until the source retrieves the entry and the registry wakeup
      // re-enqueues this server.
      setState(ServerState::WaitingForIntraNodeRetrieve);
      if (intraNodeProducerPollRequeueEnabled()) {
        wakeCommunicator();
      }
    } else {
      // Data pointer is null, so no more data will be coming.
      // Publish atEnd marker to registry
      VLOG(3) << "@" << partitionKey_.taskId
              << " Intra-node transfer: publishing atEnd for sequence "
              << sequenceNumber_;

      IntraNodeTransferKey key{
          partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};
      intraNodeRetrieveFuture_ =
          IntraNodeTransferRegistry::getInstance()->publish(
              key,
              nullptr,
              rmm::cuda_stream_default,
              /*atEnd=*/true,
              makeIntraNodeRetrieveWakeup());
      intraNodeAtEndPublished_ = true;

      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);

      // Wait for source to acknowledge atEnd before finishing. The registry
      // wakeup re-enqueues this server when that happens.
      setState(ServerState::WaitingForIntraNodeRetrieve);
      if (intraNodeProducerPollRequeueEnabled()) {
        wakeCommunicator();
      }
    }
  } else {
    // REMOTE EXCHANGE PATH: Use UCXX for metadata and data transfer
    const bool useHostStaging = !communicator->hasCudaTransport();
    std::shared_ptr<DataSendContext> dataCtx;
    if (dataPtr_) {
      const auto hostBytes = static_cast<int64_t>(dataPtr_->gpu_data->size());
      dataCtx = std::make_shared<DataSendContext>();
      if (useHostStaging && !dataCtx->reserveHostBytes(hostBytes)) {
        // Keep dataPtr_ and state=DataReady.  Completed UCX callbacks release
        // process-wide credit; requeueing lets this server retry without
        // dequeuing or staging another packed table.
        wakeCommunicator();
        return;
      }
    }
    std::shared_ptr<MetadataMsg> metadataMsg = std::make_shared<MetadataMsg>();

    if (dataPtr_) {
      // Copy metadata (not move) because in broadcast mode, the same
      // packed_columns may be shared across multiple destination queues.
      // Metadata is small (CPU-side), so copying is negligible.
      metadataMsg->cudfMetadata =
          std::make_unique<std::vector<uint8_t>>(*dataPtr_->metadata);
      metadataMsg->dataSizeBytes = dataPtr_->gpu_data->size();
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = false;
    } else {
      VLOG(3) << "@" << partitionKey_.taskId << " Final exchange for "
              << partitionKey_.toString();
      metadataMsg->cudfMetadata = nullptr;
      metadataMsg->dataSizeBytes = 0;
      metadataMsg->remainingBytes = {};
      metadataMsg->atEnd = true;
    }

    auto [serializedMetadata, serMetaSize] = metadataMsg->serialize();

    // send metadata.
    uint64_t metadataTag =
        getMetadataTag(this->partitionKeyHash_, this->sequenceNumber_);
    // Use weak_ptr to prevent use-after-free if close() is called during
    // callback
    std::weak_ptr<UcxExchangeServer> weakMeta = weak_from_this();
    retireRequest(metaRequest_, completedRequests_);

    // Wrap the serialized metadata in a context so the callback can release
    // it after the send completes, while the Request (and context shell)
    // stays alive for UCP wireup replay.
    auto metaCtx = std::make_shared<MetaSendContext>();
    metaCtx->metadata = serializedMetadata;

    VELOX_CHECK_NOT_NULL(dataEndpointRef_);
    metaRequest_ = dataEndpointRef_->endpoint_->tagSend(
        metaCtx->metadata.get(),
        serMetaSize,
        ucxx::Tag{metadataTag},
        false,
        [tid = partitionKey_.toString(), metadataTag, weakMeta](
            ucs_status_t status, std::shared_ptr<void> arg) {
          // Release the metadata buffer from the context. The context
          // shell stays alive with the Request; only the payload is freed.
          auto ctx = std::static_pointer_cast<MetaSendContext>(arg);
          auto metaHolder = std::move(ctx->metadata); // release CPU buffer

          auto self = weakMeta.lock();
          if (!self) {
            return; // Object was destroyed, safe to ignore
          }
          // Check if close() was called
          if (self->closed_.load(std::memory_order_acquire)) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata send callback called after close, ignoring";
            return;
          }
          if (status == UCS_OK) {
            VLOG(3) << "@" << self->partitionKey_.taskId
                    << " metadata successfully sent to " << tid
                    << " with tag: " << std::hex << metadataTag;
          } else {
            VLOG(0) << "[UCX-SERVER-METADATA-SEND-ERROR] task="
                    << self->partitionKey_.taskId << " key=" << tid
                    << " seq=" << self->sequenceNumber_ << " tag=" << std::hex
                    << metadataTag << std::dec
                    << " status=" << ucs_status_string(status);
            self->setState(ServerState::Done);
            self->wakeCommunicator();
          }
        },
        metaCtx);

    // send the data chunk (if any)
    if (dataPtr_) {
      sendStart_ = std::chrono::high_resolution_clock::now();
      bytes_ = dataPtr_->gpu_data->size();

      VLOG(3) << "@" << partitionKey_.taskId
              << " Sending rmm::buffer: " << std::hex
              << dataPtr_->gpu_data.get()
              << " pointing to device memory: " << std::hex
              << dataPtr_->gpu_data->data() << std::dec << " to task "
              << partitionKey_.toString() << ":" << this->sequenceNumber_
              << std::dec << " of size " << bytes_;

      // Wrap the GPU data buffer in a context so the callback can release
      // it after the DMA completes, while the Request (and context shell)
      // stays alive for UCP wireup replay.
      dataCtx->data = dataPtr_;
      if (useHostStaging) {
        const auto producerStream = dataCtx->data->gpu_data->stream();
        if (pinnedHostStagingEnabled()) {
          dataCtx->pinnedHostData = std::make_shared<PinnedSendBuffer>(bytes_);
        }
        if (dataCtx->pinnedHostData != nullptr &&
            dataCtx->pinnedHostData->data() != nullptr) {
          // The packed buffer was produced on producerStream. Queueing the
          // D2H copy on that same stream preserves producer ordering without
          // blocking the communicator thread. An event lets the communicator
          // poll completion without executing application code in a CUDA host
          // callback.
          dataCtx->stageStart = std::chrono::high_resolution_clock::now();
          const auto eventStatus = cudaEventCreateWithFlags(
              &dataCtx->stageEvent, cudaEventDisableTiming);
          if (eventStatus != cudaSuccess) {
            cudaGetLastError();
            LOG(WARNING) << "@" << partitionKey_.taskId
                         << " failed to create async D2H completion event: "
                         << cudaGetErrorString(eventStatus)
                         << "; falling back to synchronous pinned staging";
            CUDF_CUDA_TRY(cudaStreamSynchronize(producerStream.value()));
            CUDF_CUDA_TRY(cudaMemcpy(
                dataCtx->pinnedHostData->data(),
                dataCtx->data->gpu_data->data(),
                bytes_,
                cudaMemcpyDeviceToHost));
            postDataSend(dataCtx);
            return;
          }
          CUDF_CUDA_TRY(cudaMemcpyAsync(
              dataCtx->pinnedHostData->data(),
              dataCtx->data->gpu_data->data(),
              bytes_,
              cudaMemcpyDeviceToHost,
              producerStream.value()));
          CUDF_CUDA_TRY(
              cudaEventRecord(dataCtx->stageEvent, producerStream.value()));
          pendingDataSend_ = dataCtx;
          setState(ServerState::WaitingForHostStage);
          wakeCommunicator();
          return;
        } else {
          dataCtx->pinnedHostData.reset();
          dataCtx->hostData = std::make_shared<std::vector<uint8_t>>(bytes_);
          CUDF_CUDA_TRY(cudaStreamSynchronize(producerStream.value()));
          CUDF_CUDA_TRY(cudaMemcpy(
              dataCtx->hostData->data(),
              dataCtx->data->gpu_data->data(),
              bytes_,
              cudaMemcpyDeviceToHost));
        }
      }
      postDataSend(dataCtx);
    } else {
      // Data pointer is null, so no more data will be coming.
      VLOG(3) << "@" << partitionKey_.taskId
              << " Finished transferring partition for task "
              << partitionKey_.toString();
      queueMgr_->deleteResults(partitionKey_.taskId, partitionKey_.destination);
      setState(ServerState::Done);
      wakeCommunicator();
    }
  }
}

void UcxExchangeServer::postDataSend(
    const std::shared_ptr<DataSendContext>& dataCtx) {
  std::lock_guard<std::recursive_mutex> lock(dataMutex_);
  if (closed_.load(std::memory_order_acquire)) {
    return;
  }
  VELOX_CHECK_NOT_NULL(dataCtx);
  VELOX_CHECK_NOT_NULL(dataCtx->data);
  VELOX_CHECK_NOT_NULL(dataEndpointRef_);

  void* sendBuffer = dataCtx->data->gpu_data->data();
  const char* path = "direct-device";
  if (dataCtx->pinnedHostData != nullptr) {
    VELOX_CHECK_NOT_NULL(dataCtx->pinnedHostData->data());
    sendBuffer = dataCtx->pinnedHostData->data();
    path = "async-pinned-host-staged";
    const auto stageMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - dataCtx->stageStart)
            .count();
    VLOG(2) << "@" << partitionKey_.taskId << " async D2H staged " << bytes_
            << " bytes in " << stageMicros << " us";
  } else if (dataCtx->hostData != nullptr) {
    sendBuffer = dataCtx->hostData->data();
    path = "pageable-host-staged";
  }

  VLOG(2) << "@" << partitionKey_.taskId << " posting " << path << " send for "
          << bytes_ << " bytes";

  setState(ServerState::WaitingForSendComplete);
  const uint64_t dataTag =
      getDataTag(this->partitionKeyHash_, this->sequenceNumber_);
  std::weak_ptr<UcxExchangeServer> weakData = weak_from_this();
  retireRequest(dataRequest_, completedRequests_);
  dataRequest_ = dataEndpointRef_->endpoint_->tagSend(
      sendBuffer,
      static_cast<size_t>(bytes_),
      ucxx::Tag{dataTag},
      false,
      [weakData](ucs_status_t status, std::shared_ptr<void> arg) {
        // Release payload buffers promptly after UCX completes. The empty
        // callback context remains retained by the Request for replay safety.
        auto ctx = std::static_pointer_cast<DataSendContext>(arg);
        auto dataHolder = std::move(ctx->data);
        auto hostDataHolder = std::move(ctx->hostData);
        auto pinnedHostDataHolder = std::move(ctx->pinnedHostData);
        const bool usedPageableHostData = hostDataHolder != nullptr;
        const auto releasedHostBytes = ctx->reservedHostBytes;
        ctx->releaseHostReservation();
        hostDataHolder.reset();
        pinnedHostDataHolder.reset();
        if (usedPageableHostData) {
          // Return large pageable arenas to the OS. Pinned buffers instead go
          // to their separately bounded exchange pool.
          accountFreedHostBytesAndTrim(releasedHostBytes);
        }

        if (auto self = weakData.lock()) {
          self->sendComplete(status, arg);
        }
      },
      dataCtx);
}

void UcxExchangeServer::sendComplete(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(2) << "[UCX-SERVER-SEND-COMPLETE-AFTER-CLOSE] task="
            << partitionKey_.taskId << " key=" << partitionKey_.toString()
            << " seq=" << sequenceNumber_
            << " status=" << ucs_status_string(status);
    return;
  }
  if (status == UCS_OK) {
    std::lock_guard<std::recursive_mutex> lock(dataMutex_);
    VELOX_CHECK_NOT_NULL(dataPtr_, "dataPtr_ is null");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - sendStart_;
    auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    auto throughput = bytes_ / micros;

    VLOG(3) << "@" << partitionKey_.taskId << " duration: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                   .count()
            << " ms ";
    VLOG(3) << "@" << partitionKey_.taskId << " throughput: " << throughput
            << " MByte/s";

    this->sequenceNumber_++;
    dataPtr_.reset(); // release memory.
    VLOG(3) << "@" << partitionKey_.taskId
            << " Releasing dataPtr_ in sendComplete.";
    setState(ServerState::ReadyToTransfer);
  } else {
    VLOG(0) << "[UCX-SERVER-DATA-SEND-ERROR] task=" << partitionKey_.taskId
            << " key=" << partitionKey_.toString() << " seq=" << sequenceNumber_
            << " bytes=" << bytes_ << " status=" << ucs_status_string(status);
    setState(ServerState::Done);
  }
  wakeCommunicator();
}

std::function<void()> UcxExchangeServer::makeIntraNodeRetrieveWakeup() {
  std::weak_ptr<CommElement> weakSelf = getSelfPtr();
  auto weakCommunicator = communicator_;
  return [weakSelf, weakCommunicator]() {
    if (auto server = weakSelf.lock(); server) {
      auto communicator = weakCommunicator.lock();
      if (!communicator) {
        return;
      }
      communicator->addToWorkQueue(server);
    }
  };
}

void UcxExchangeServer::onIntraNodeRetrieveComplete() {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << "@" << partitionKey_.taskId
            << " onIntraNodeRetrieveComplete called after close, ignoring";
    return;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - sendStart_;
  auto micros =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  auto throughput = (micros > 0) ? (bytes_ / micros) : 0;

  VLOG(3)
      << "@" << partitionKey_.taskId << " Intra-node transfer duration: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
      << " ms ";
  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer throughput: " << throughput << " MByte/s";

  VLOG(3) << "@" << partitionKey_.taskId
          << " Intra-node transfer complete for sequence " << sequenceNumber_;

  if (intraNodeAtEndPublished_) {
    // This was the final atEnd marker, we're done
    VLOG(3) << "@" << partitionKey_.taskId
            << " Intra-node transfer: atEnd acknowledged, finishing";
    setState(ServerState::Done);
  } else {
    // More data may be coming, continue transfer loop
    this->sequenceNumber_++;
    setState(ServerState::ReadyToTransfer);
  }
  wakeCommunicator();
}

} // namespace facebook::velox::ucx_exchange
