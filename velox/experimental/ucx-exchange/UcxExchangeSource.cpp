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

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cudf/contiguous_split.hpp>
#include <folly/String.h>
#include <folly/Uri.h>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/ucx-exchange/IntraNodeTransferRegistry.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"

using namespace facebook::velox::exec;
namespace facebook::velox::ucx_exchange {

namespace {
std::shared_ptr<uint8_t> allocateReplaySafeHostReceiveBuffer(uint64_t bytes) {
  // UCXX retains completed requests for wireup replay. The raw receive pointer
  // therefore cannot come from the reusable pinned pool: once the consumer
  // releases that lease, replay could overwrite an unrelated later packet.
  // Keep the transport image pageable and request-owned; UcxExchange uses a
  // separate short-lived pinned bounce when it performs the deferred H2D.
  return std::shared_ptr<uint8_t>(
      new uint8_t[bytes], std::default_delete<uint8_t[]>());
}

void retireRequest(
    std::shared_ptr<ucxx::Request>& current,
    std::vector<std::shared_ptr<ucxx::Request>>& inFlight) {
  if (current != nullptr) {
    inFlight.push_back(std::move(current));
  }
  // Keep a short replay-safety window, including each request's unique receive
  // buffer. Reusing one buffer while retaining old UCXX requests allows a
  // wireup replay to overwrite a later packet before its guarded callback is
  // invoked. Four alternating metadata/data requests retain two full packet
  // generations without growing for the lifetime of a long exchange.
  constexpr size_t kRetainedRequestWindow = 4;
  while (inFlight.size() > kRetainedRequestWindow) {
    const auto completed = std::find_if(
        inFlight.begin(), inFlight.end(), [](const auto& request) {
          return request == nullptr || request->isCompleted();
        });
    if (completed == inFlight.end()) {
      break;
    }
    inFlight.erase(completed);
  }
}

const folly::F14FastMap<UcxExchangeSource::ReceiverState, std::string_view>&
receiverStateNames() {
  static const folly::F14FastMap<
      UcxExchangeSource::ReceiverState,
      std::string_view>
      kNames = {
          {UcxExchangeSource::ReceiverState::Created, "Created"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeComplete,
           "WaitingForHandshakeComplete"},
          {UcxExchangeSource::ReceiverState::WaitingForHandshakeResponse,
           "WaitingForHandshakeResponse"},
          {UcxExchangeSource::ReceiverState::ReadyToReceive, "ReadyToReceive"},
          {UcxExchangeSource::ReceiverState::WaitingForMetadata,
           "WaitingForMetadata"},
          {UcxExchangeSource::ReceiverState::WaitingForReceiveCredit,
           "WaitingForReceiveCredit"},
          {UcxExchangeSource::ReceiverState::WaitingForData, "WaitingForData"},
          {UcxExchangeSource::ReceiverState::WaitingForIntraNodeData,
           "WaitingForIntraNodeData"},
          {UcxExchangeSource::ReceiverState::Done, "Done"},
      };
  return kNames;
}

int64_t maxInFlightRecvHostBytes() {
  static const int64_t limit = [] {
    if (const char* value =
            std::getenv("GLUTEN_UCX_MAX_INFLIGHT_RECV_HOST_BYTES")) {
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

std::atomic<int64_t> inFlightRecvHostBytes{0};

rmm::mr::statistics_resource_adaptor& receiveDeviceMemoryResource() {
  // Allocate UCX receive pages from the same async resource as cuDF operators.
  // A separate cuda_memory_resource cannot reuse blocks cached by the async
  // pool and can therefore fail cudaMalloc while scan/aggregation has ample
  // reusable memory in that pool. Keep a dedicated statistics adaptor so
  // receive credit continues to account for packed pages after ownership moves
  // out of UcxExchangeSource.
  static rmm::mr::statistics_resource_adaptor resource{
      cuda::mr::any_resource<cuda::mr::device_accessible>{
          rmm::mr::get_current_device_resource_ref()}};
  return resource;
}

int64_t maxInFlightRecvDeviceBytes() {
  static const int64_t limit = [] {
    if (const char* value =
            std::getenv("GLUTEN_UCX_MAX_INFLIGHT_RECV_DEVICE_BYTES")) {
      try {
        const auto parsed = static_cast<int64_t>(std::stoll(value));
        if (parsed > 0) {
          return parsed;
        }
      } catch (...) {
      }
    }
    // A query-wide credit pool needs dependency-aware wakeups. A blind global
    // cap can deadlock a DAG when an edge holding credit is downstream of an
    // edge waiting for credit. Keep this diagnostic mechanism opt-in until
    // the coordinator owns that dependency-aware arbitration.
    return static_cast<int64_t>(0);
  }();
  return limit;
}

uint64_t recvConsumerProgressBytes() {
  static const uint64_t reserve = []() -> uint64_t {
    constexpr uint64_t kDefault = uint64_t{1} << 30;
    if (const char* value =
            std::getenv("GLUTEN_UCX_RECV_CONSUMER_PROGRESS_BYTES")) {
      try {
        const auto parsed = std::stoull(value);
        if (parsed > 0) {
          return parsed;
        }
      } catch (...) {
      }
    }
    return kDefault;
  }();
  return reserve;
}

std::size_t recvAdmissionMinHeadroom() {
  const auto base = facebook::velox::cudf_velox::CudfConfig::getInstance()
                        .deviceMemoryMinHeadroomBytes;
  const auto progress = recvConsumerProgressBytes();
  const auto maximum = std::numeric_limits<std::size_t>::max();
  return progress > maximum || base > maximum - progress
      ? maximum
      : base + progress;
}

constexpr auto kRecvWorkspaceProgressAge = std::chrono::seconds(2);

std::mutex recvWorkspaceProgressMutex;
std::unordered_map<int, uint64_t> recvWorkspaceProgressBytesByDevice;

void releaseRecvWorkspaceProgressBytes(int device, uint64_t bytes) {
  std::lock_guard<std::mutex> lock(recvWorkspaceProgressMutex);
  const auto it = recvWorkspaceProgressBytesByDevice.find(device);
  VELOX_CHECK(it != recvWorkspaceProgressBytesByDevice.end());
  VELOX_CHECK_GE(it->second, bytes);
  it->second -= bytes;
  if (it->second == 0) {
    recvWorkspaceProgressBytesByDevice.erase(it);
  }
}

class RecvWorkspaceProgressLease {
 public:
  RecvWorkspaceProgressLease(int device, uint64_t bytes)
      : device_{device}, bytes_{bytes} {}

  ~RecvWorkspaceProgressLease() {
    releaseRecvWorkspaceProgressBytes(device_, bytes_);
  }

 private:
  const int device_;
  const uint64_t bytes_;
};

std::shared_ptr<void> tryAcquireRecvWorkspaceProgressLease(uint64_t bytes) {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
    return nullptr;
  }
  const auto limit = recvConsumerProgressBytes();
  if (bytes == 0 || bytes > limit) {
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(recvWorkspaceProgressMutex);
    auto& current = recvWorkspaceProgressBytesByDevice[device];
    if (current > limit - bytes) {
      if (current == 0) {
        recvWorkspaceProgressBytesByDevice.erase(device);
      }
      return nullptr;
    }
    current += bytes;
  }
  try {
    return std::make_shared<RecvWorkspaceProgressLease>(device, bytes);
  } catch (...) {
    releaseRecvWorkspaceProgressBytes(device, bytes);
    throw;
  }
}

bool hasRecvDeviceCredit(int64_t bytes) {
  const auto limit = maxInFlightRecvDeviceBytes();
  if (limit <= 0) {
    return true;
  }
  const auto current = receiveDeviceMemoryResource().get_bytes_counter().value;
  // Packed tables cannot be split. Permit one oversized receive only when no
  // other receive page is live, matching the host-credit behavior.
  return current == 0 || current + bytes <= limit;
}

std::atomic<int64_t> lastReportedReceiveDevicePeakBucket{0};

bool tryReserveRecvHostBytes(int64_t bytes) {
  VELOX_CHECK_GE(bytes, 0);
  auto current = inFlightRecvHostBytes.load(std::memory_order_relaxed);
  while (true) {
    // The current protocol cannot split one packed table. Permit one
    // oversized receive when the process has no other host receive buffer.
    if (current > 0 && current + bytes > maxInFlightRecvHostBytes()) {
      return false;
    }
    if (inFlightRecvHostBytes.compare_exchange_weak(
            current,
            current + bytes,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return true;
    }
  }
}

void releaseRecvHostBytes(int64_t bytes) {
  if (bytes <= 0) {
    return;
  }
  const auto previous =
      inFlightRecvHostBytes.fetch_sub(bytes, std::memory_order_acq_rel);
  VELOX_CHECK_GE(previous, bytes);
}

} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(
    UcxExchangeSource,
    ReceiverState,
    receiverStateNames)

int64_t UcxExchangeSource::maxInFlightRecvBytes() {
  // Read once. See header for rationale: recv buffers are off the operator
  // pool, so an unbounded byte footprint scales O(#peers) and exhausts the GPU
  // at 4 peers. This cap makes the producer's tagSend block at rendezvous,
  // leaving the async operator pool headroom. Deadlock-safe: the count-based
  // resume path (UcxExchangeClient::next) drains both count and bytes.
  static const int64_t kBytes = [] {
    if (const char* env = std::getenv("GLUTEN_UCX_MAX_INFLIGHT_RECV_BYTES")) {
      try {
        const int64_t v = std::stoll(env);
        if (v > 0) {
          return v;
        }
      } catch (...) {
      }
    }
    return static_cast<int64_t>(8) * 1024 * 1024 * 1024; // 8 GiB default
  }();
  return kBytes;
}

void UcxExchangeSource::setState(ReceiverState newState) {
  auto oldState = state_.exchange(newState, std::memory_order_seq_cst);
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(oldState) << " -> " << toName(newState);
}

// This constructor is private.
UcxExchangeSource::UcxExchangeSource(
    const std::shared_ptr<Communicator> communicator,
    std::string_view taskId,
    std::string_view host,
    uint16_t port,
    const PartitionKey& partitionKey,
    const std::shared_ptr<UcxExchangeQueue> queue)
    : CommElement(communicator),
      host_(host),
      port_(port),
      taskId_(taskId),
      partitionKey_(partitionKey),
      partitionKeyHash_(fnv1a_32(partitionKey_.toString())),
      queue_(std::move(queue)) {
  setState(ReceiverState::Created);
}

/*static*/
std::shared_ptr<UcxExchangeSource> UcxExchangeSource::create(
    std::string_view taskId,
    std::string_view url,
    const std::shared_ptr<UcxExchangeQueue>& queue) {
  folly::Uri uri(url);
  // Note that there is no distinct schema for the UCXX exchange.
  // The approach is to ignore the schema and not check for HTTP or HTTPS.
  // FIXME: Can't use the HTTP port as this conflicts with Prestissimo!
  // For the time being, there's an ugly hack that just increases the port by 3.
  const std::string host = uri.host();
  int port = uri.port() + 3;
  std::shared_ptr<Communicator> communicator = Communicator::getInstance();
  auto key = extractTaskAndDestinationId(uri.path());
  auto source = std::shared_ptr<UcxExchangeSource>(
      new UcxExchangeSource(communicator, taskId, host, port, key, queue));
  // register the exchange source with the communicator. This makes sure that
  // "progress" is called.
  communicator->registerCommElement(source);
  VLOG(3) << source->toString()
          << " creating UcxExchangeSource for url: " << url;
  return source;
}

void UcxExchangeSource::process() {
  auto communicator = tryCommunicator();
  if (!communicator) {
    deliverEndMarker();
    return;
  }
  if (closed_) {
    // Driver thread called closed
    cleanUp();
    return;
  }

  switch (state_) {
    case ReceiverState::Created: {
      // Get the endpoint.
      HostPort hp{host_, port_};
      std::shared_ptr<UcxExchangeSource> selfPtr = getSelfPtr();
      auto epRef = communicator->assocEndpointRef(selfPtr, hp);
      if (epRef) {
        setEndpoint(epRef);
        setStateIf(
            ReceiverState::Created, ReceiverState::WaitingForHandshakeComplete);
        sendHandshake();
      } else {
        // connection failed.
        VLOG(0) << toString() << " Failed to connect to " << host_ << ":"
                << std::to_string(port_);
        deliverEndMarker();
        setState(ReceiverState::Done);
      }
      wakeCommunicator();
    } break;
    case ReceiverState::WaitingForHandshakeComplete:
      // Waiting for handshake send completion is handled by callback.
      break;
    case ReceiverState::WaitingForHandshakeResponse:
      // Waiting for HandshakeResponse is handled by callback.
      break;
    case ReceiverState::ReadyToReceive: {
      // Backpressure: don't post the next receive if the consumer queue is
      // overloaded. The source goes dormant (not in work queue) and will be
      // woken by UcxExchangeClient::next() calling resumeFromBackpressure()
      // when the queue drains below the low water mark.
      //
      // This creates natural backpressure: the server's tagSend for data
      // will block at rendezvous until we post a matching tagRecv. For
      // intra-node: the server's publish future won't resolve until we poll.
      UcxExchangeQueue::BackpressureStats stats;
      // Backpressure on BOTH item count and aggregate bytes. The byte cap is
      // what bounds the off-pool receive-buffer footprint that otherwise scales
      // O(#peers) and OOMs/deadlocks the GPU at 4 peers (the count cap alone
      // let a few large chunks fill the device before pausing).
      bool shouldPause = false;
      bool newlyBackpressured = false;
      {
        // Serialize observing the queue and publishing the dormant flag with
        // dequeue. A consumer either drains first and makes shouldPause false,
        // or drains after the flag is armed and resumes this source.
        std::lock_guard<std::mutex> lock(queue_->mutex());
        shouldPause = queue_->shouldPauseReceiveLocked(
            kBackpressureHighWaterMark, maxInFlightRecvBytes(), &stats);
        if (shouldPause) {
          newlyBackpressured =
              !backpressureActive_.exchange(true, std::memory_order_acq_rel);
        }
      }
      if (shouldPause) {
        if (newlyBackpressured) {
          VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
                  << "] pausing, queueSize=" << stats.queueSize
                  << " (high=" << kBackpressureHighWaterMark
                  << "), queueBytes=" << stats.queuedBytes
                  << ", pendingReceiveBytes=" << stats.pendingReceiveBytes
                  << " (cap=" << maxInFlightRecvBytes() << ")";
        }
        // Go dormant — do NOT re-enqueue into work queue.
        // UcxExchangeClient::next() will call resumeFromBackpressure().
        break;
      }

      // Count-only backpressure (Presto-style): post the next receive directly.
      // The server's tagSend blocks at rendezvous until we post the matching
      // tagRecv, so no explicit byte-credit request is needed. Intra-node waits
      // on the registry; remote waits for UCX metadata/data tags.
      if (isIntraNodeTransfer_) {
        setStateIf(
            ReceiverState::ReadyToReceive,
            ReceiverState::WaitingForIntraNodeData);
        waitForIntraNodeData();
      } else {
        setStateIf(
            ReceiverState::ReadyToReceive, ReceiverState::WaitingForMetadata);
        getMetadata();
      }
    } break;
    case ReceiverState::WaitingForMetadata:
      // Waiting for metadata is handled by an upcall from UCXX. Nothing to do
      break;
    case ReceiverState::WaitingForReceiveCredit:
      tryStartDataReceive(
          pendingReceive_, ReceiverState::WaitingForReceiveCredit);
      break;
    case ReceiverState::WaitingForData:
      // Waiting for data is handled by an upcall from UCXX. Nothing to do.
      break;
    case ReceiverState::WaitingForIntraNodeData:
      // Poll for intra-node transfer data
      waitForIntraNodeData();
      break;
    case ReceiverState::Done:
      // We need to call clean-up in this thread to remove any state
      cleanUp();
      break;
  }
}

void UcxExchangeSource::cleanUp() {
  releaseReceiveReservation();
  pendingReceive_.reset();

  uint32_t value = static_cast<uint32_t>(getState());
  if (value != static_cast<uint32_t>(ReceiverState::Done)) {
    // Unexpected cleanup
    VLOG(3) << toString()
            << " In UcxExchangeSource::cleanUp state == " << value;
  }

  // Cancel any outstanding request. With weak_ptr callbacks, the callback
  // will safely no-op if we're destroyed before it completes.
  if (request_ && !request_->isCompleted()) {
    // The Task has failed and we may need to cancel outstanding requests
    request_->cancel();
  }

  // Move all requests to the Communicator's deferred list so the GPU
  // buffers they reference (via their arg shared_ptr) stay alive until
  // UCX has fully processed any in-flight operations.
  auto communicator = communicator_.lock();
  if (communicator) {
    if (request_) {
      communicator->deferRequestCleanup(std::move(request_));
    }
    for (auto& req : completedRequests_) {
      communicator->deferRequestCleanup(std::move(req));
    }
    completedRequests_.clear();
  }

  if (endpointRef_) {
    endpointRef_->removeCommElem(getSelfPtr());
    endpointRef_ = nullptr;
  }
  if (communicator) {
    communicator->unregister(getSelfPtr());
  }
}

void UcxExchangeSource::close() {
  // This is called by the driver thread so we need to be careful to
  // indicate to the process thread that we are closing and
  // let it do the actual cleaning up.

  // Use memory_order_acq_rel to ensure proper synchronization with callbacks
  // that check closed_ with memory_order_acquire.
  bool expected = false;
  bool desired = true;
  if (!closed_.compare_exchange_strong(
          expected, desired, std::memory_order_acq_rel)) {
    return; // already closed.
  }

  VLOG(2) << "[UCX-SOURCE-CLOSE] " << toString()
          << " state=" << toName(getState()) << " seq=" << sequenceNumber_
          << " hasRequest=" << (request_ != nullptr) << " atEnd=" << atEnd_;

  if (!atEnd_) {
    sendCancel();
  }

  // Guarantee the end marker is delivered before transitioning to Done.
  deliverEndMarker();

  // Let the Communicator progress thread do the actual clean-up.
  setState(ReceiverState::Done);
  wakeCommunicator();
}

void UcxExchangeSource::closeWithError(std::string error) {
  if (!closed_.load(std::memory_order_acquire) &&
      getState() != ReceiverState::Done && !atEnd_) {
    queue_->setError(std::move(error));
  }
  close();
}

void UcxExchangeSource::resumeFromBackpressure() {
  bool expected = true;
  if (backpressureActive_.compare_exchange_strong(
          expected, false, std::memory_order_acq_rel)) {
    VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
            << "] resumed by consumer, queueSize=" << queue_->size();
    wakeCommunicator();
  }
}

folly::F14FastMap<std::string, int64_t> UcxExchangeSource::stats() const {
  VELOX_UNREACHABLE();
}

folly::F14FastMap<std::string, RuntimeMetric> UcxExchangeSource::metrics()
    const {
  folly::F14FastMap<std::string, RuntimeMetric> map;

  // these metrics will be aggregated over all exchange sources of the same
  // exchange client.
  map["ucxExchangeSource.numPackedColumns"] = metrics_.numPackedColumns_;
  map["ucxExchangeSource.totalBytes"] = metrics_.totalBytes_;
  map["ucxExchangeSource.hostStagedBytes"] = metrics_.hostStagedBytes_;
  map["ucxExchangeSource.asyncHostCopyBytes"] =
      metrics_.asyncHostCopyBytes_;
  map["ucxExchangeSource.asyncHostCopyThrottledBytes"] =
      metrics_.asyncHostCopyThrottledBytes_;
  map["ucxExchangeSource.hostCopySyncNanos"] =
      metrics_.hostCopySyncNanos_;
  map["ucxExchangeSource.rttPerRequest"] = metrics_.rttPerRequest_;
  return map;
}

// private methods ---
PartitionKey UcxExchangeSource::extractTaskAndDestinationId(
    std::string_view path) {
  // The URL path has the form: /v1/task/<taskId>/results/<destinationId>"
  std::vector<folly::StringPiece> components;
  folly::split('/', path, components, true);

  VELOX_CHECK_EQ(components[0], "v1");
  VELOX_CHECK_EQ(components[1], "task");
  VELOX_CHECK_EQ(components[3], "results");

  uint32_t destinationId;
  try {
    destinationId = static_cast<uint32_t>(std::stoul(components[4].str()));
  } catch (const std::exception& e) {
    VELOX_UNSUPPORTED("Illegal destination in task URL: {}", path);
  }

  return PartitionKey{components[2].str(), destinationId};
}

std::shared_ptr<UcxExchangeSource> UcxExchangeSource::getSelfPtr() {
  std::shared_ptr<UcxExchangeSource> ptr;
  try {
    ptr = shared_from_this();
  } catch (std::bad_weak_ptr& exp) {
    ptr = nullptr;
  }
  return ptr;
}

void UcxExchangeSource::wakeCommunicator() {
  if (auto communicator = tryCommunicator()) {
    communicator->addToWorkQueue(getSelfPtr());
  }
}

void UcxExchangeSource::enqueue(
    PackedTableWithStreamPtr data,
    int64_t reservedReceiveBytes) {
  std::vector<velox::ContinuePromise> queuePromises;
  {
    std::lock_guard<std::mutex> l(queue_->mutex());

    queue_->enqueueLocked(std::move(data), queuePromises, reservedReceiveBytes);
  }
  // wake up consumers of the UcxExchangeQueue
  for (auto& promise : queuePromises) {
    promise.setValue();
  }
}

void UcxExchangeSource::deliverEndMarker() {
  if (!registered_.load(std::memory_order_acquire)) {
    // Never registered with queue -- don't deliver end marker to avoid
    // spurious numCompleted_ increments.
    return;
  }
  bool expected = false;
  if (!endMarkerDelivered_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    // Already delivered by another thread/path.
    return;
  }
  VLOG(3) << toString() << " delivering end-of-stream marker to queue";
  enqueue(nullptr);
}

void UcxExchangeSource::releaseReceiveReservation() {
  if (reservedReceiveBytes_ > 0) {
    queue_->releaseReservedReceive(reservedReceiveBytes_);
    reservedReceiveBytes_ = 0;
  }
  if (reservedGlobalHostReceiveBytes_ > 0) {
    releaseRecvHostBytes(reservedGlobalHostReceiveBytes_);
    reservedGlobalHostReceiveBytes_ = 0;
  }
}

void UcxExchangeSource::setEndpoint(std::shared_ptr<EndpointRef> endpointRef) {
  endpointRef_ = std::move(endpointRef);
}

void UcxExchangeSource::sendHandshake() {
  auto communicator = tryCommunicator();
  if (!communicator) {
    deliverEndMarker();
    return;
  }
  handshakeRequestBuffer_ = std::make_shared<HandshakeMsg>();
  auto& handshakeReq = handshakeRequestBuffer_;
  handshakeReq->destination = partitionKey_.destination;
  // Use sizeof(...) - 1 and explicitly null-terminate to prevent buffer
  // overread if taskId is longer than the destination buffer.
  strncpy(
      handshakeReq->taskId,
      partitionKey_.taskId.c_str(),
      sizeof(handshakeReq->taskId) - 1);
  handshakeReq->taskId[sizeof(handshakeReq->taskId) - 1] = '\0';
  handshakeReq->workerId = communicator->getWorkerId();

  VLOG(2) << "[UCX-SOURCE-HANDSHAKE-SEND] localTask=" << taskId_
          << " remoteTask=" << partitionKey_.taskId
          << " destination=" << partitionKey_.destination << " peer=" << host_
          << ":" << port_ << " workerId=" << handshakeReq->workerId;

  // Create the handshake which will register client's existence with the server
  ucxx::AmReceiverCallbackInfo info(
      communicator->kAmCallbackOwner, communicator->kAmCallbackId);
  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  // Pass handshakeReq as the callback arg to keep the send buffer alive until
  // the async amSend completes. UCXX stores it as shared_ptr<void> but the
  // type-erased deleter still calls ~HandshakeMsg correctly.
  request_ = endpointRef_->endpoint_->amSend(
      handshakeReq.get(),
      sizeof(*handshakeReq),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onHandshake(status, arg);
        }
      },
      handshakeReq);
}

void UcxExchangeSource::sendCancel() {
  auto communicator = communicator_.lock();
  if (!communicator || communicator->isShuttingDown() || !endpointRef_) {
    return;
  }

  auto cancel = std::make_shared<HandshakeMsg>();
  cancel->destination =
      partitionKey_.destination | kCancelDestinationFlag;
  strncpy(
      cancel->taskId,
      partitionKey_.taskId.c_str(),
      sizeof(cancel->taskId) - 1);
  cancel->taskId[sizeof(cancel->taskId) - 1] = '\0';
  cancel->workerId = communicator->getWorkerId();
  VLOG(1) << "[UCX-CANCEL] send task=" << partitionKey_.taskId
          << " destination=" << partitionKey_.destination;

  ucxx::AmReceiverCallbackInfo info(
      communicator->kAmCallbackOwner, communicator->kAmCallbackId);
  auto cancelRequest = endpointRef_->endpoint_->amSend(
      cancel.get(),
      sizeof(*cancel),
      UCS_MEMORY_TYPE_HOST,
      info,
      false,
      [key = partitionKey_](ucs_status_t status, std::shared_ptr<void>) {
        if (status != UCS_OK) {
          VLOG(1) << "[UCX-CANCEL] failed task=" << key.taskId
                  << " destination=" << key.destination
                  << " status=" << ucs_status_string(status);
        }
      },
      cancel);
  communicator->deferRequestCleanup(std::move(cancelRequest));
}

void UcxExchangeSource::onHandshake(
    ucs_status_t status,
    std::shared_ptr<void> /*arg*/) {
  // arg holds the HandshakeMsg that was sent — it is unused here because this
  // is a send completion callback (the outgoing data has already been
  // transmitted). The parameter exists only because UCXX uses it as a lifetime
  // handle; letting it go out of scope releases the send buffer.

  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onHandshake called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeComplete) {
    VLOG(2) << toString() << " onHandshake called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to send handshake to host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
  } else {
    VLOG(3) << toString() << "+ onHandshake " << ucs_status_string(status)
            << " peer=" << host_ << ":" << port_;
    // Now wait for the HandshakeResponse from the server
    setStateIf(
        ReceiverState::WaitingForHandshakeComplete,
        ReceiverState::WaitingForHandshakeResponse);
    receiveHandshakeResponse();
  }
}

void UcxExchangeSource::getMetadata() {
  // Use kMaxMetaBufSize to support tables with many columns.
  // The sender allocates exact size needed; receiver pre-allocates max. A
  // Each retained UCXX request needs a distinct receive address. An old
  // request can be replayed during endpoint wireup; sharing this allocation
  // with the current request would let the transport overwrite new metadata.
  auto metadataReq =
      std::make_shared<std::vector<uint8_t>>(kMaxMetaBufSize);
  const auto expectedSequence = sequenceNumber_;
  uint64_t metadataTag = getMetadataTag(partitionKeyHash_, expectedSequence);

  VLOG(3) << toString()
          << " waiting for metadata for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << metadataTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  request_ = endpointRef_->endpoint_->tagRecv(
      reinterpret_cast<void*>(metadataReq->data()),
      kMaxMetaBufSize,
      ucxx::Tag{metadataTag},
      ucxx::TagMaskFull,
      false,
      [weak, expectedSequence](
          ucs_status_t status, std::shared_ptr<void> arg) {
        auto metadata = std::static_pointer_cast<std::vector<uint8_t>>(arg);
        if (auto self = weak.lock()) {
          self->onMetadata(status, metadata, expectedSequence);
        }
      },
      metadataReq);
}

void UcxExchangeSource::onMetadata(
    ucs_status_t status,
    std::shared_ptr<void> arg,
    uint32_t expectedSequence) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(2) << "[UCX-SOURCE-METADATA-AFTER-CLOSE] " << toString()
            << " seq=" << sequenceNumber_
            << " status=" << ucs_status_string(status);
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay. State alone is
  // insufficient because an old metadata callback may arrive while a later
  // sequence is also WaitingForMetadata and uses the same receive buffer.
  if (expectedSequence != sequenceNumber_) {
    VLOG(1) << toString() << " ignoring replayed metadata callback for seq="
            << expectedSequence << ", currentSeq=" << sequenceNumber_;
    return;
  }
  if (getState() != ReceiverState::WaitingForMetadata) {
    VLOG(2) << toString() << " onMetadata called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onMetadata " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive metadata from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
  } else {
    VELOX_CHECK_NOT_NULL(arg, "Didn't get metadata");

    // arg contains the actual serialized metadata, deserialize the metadata
    std::shared_ptr<std::vector<uint8_t>> metadataMsg =
        std::static_pointer_cast<std::vector<uint8_t>>(arg);

    if (exchangeVariableWidthValidationEnabled()) {
      uint32_t serializedBytes = 0;
      std::memcpy(
          &serializedBytes,
          metadataMsg->data() + sizeof(kMagicNumber),
          sizeof(serializedBytes));
      VELOX_CHECK_LE(serializedBytes, metadataMsg->size());
      LOG(WARNING) << "UCX receiver metadata key=" << partitionKey_.toString()
                   << " sequence=" << expectedSequence
                   << " bytes=" << serializedBytes << " fingerprint=0x"
                   << std::hex
                   << diagnosticBufferFingerprint(
                          metadataMsg->data(), serializedBytes)
                   << std::dec;
    }

    auto ptr = std::make_shared<DataAndMetadata>();

    ptr->metadata =
        std::move(MetadataMsg::deserializeMetadataMsg(metadataMsg->data()));

    VLOG(3) << toString()
            << " Datasize bytes == " << ptr->metadata.dataSizeBytes;

    if (ptr->metadata.atEnd) {
      // It seems that all data has been transferred
      atEnd_ = true;
      // enqueue a nullpointer to mark the end for this source.
      VLOG(3) << "There is no more data to transfer for " << toString();
      deliverEndMarker();
      setStateIf(ReceiverState::WaitingForMetadata, ReceiverState::Done);
      wakeCommunicator();
      // jump out of this function.
      return;
    }

    pendingReceive_ = ptr;
    tryStartDataReceive(pendingReceive_, ReceiverState::WaitingForMetadata);
  }
}

bool UcxExchangeSource::tryStartDataReceive(
    const std::shared_ptr<DataAndMetadata>& ptr,
    ReceiverState expectedState) {
  if (ptr == nullptr) {
    return false;
  }
  auto communicator = tryCommunicator();
  if (!communicator) {
    return false;
  }

  UcxExchangeQueue::BackpressureStats stats;
  if (!queue_->tryReserveReceive(
          ptr->metadata.dataSizeBytes, maxInFlightRecvBytes(), &stats)) {
    // Publish the dormant receiver state before arming the consumer wake. If
    // the consumer drained before this transition, the reservation recheck
    // below observes that credit. If it drains after backpressureActive_ is
    // armed, resumeFromBackpressure() observes a receiver that is already in
    // WaitingForReceiveCredit and schedules a useful communicator pass.
    if (!setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit)) {
      return false;
    }
    expectedState = ReceiverState::WaitingForReceiveCredit;
    if (!backpressureActive_.exchange(true, std::memory_order_acq_rel)) {
      VLOG(1) << "[BACKPRESSURE] [ExSrc " << toString()
              << "] waiting for receive credit, requestedBytes="
              << ptr->metadata.dataSizeBytes
              << ", queueBytes=" << stats.queuedBytes
              << ", pendingReceiveBytes=" << stats.pendingReceiveBytes
              << " (cap=" << maxInFlightRecvBytes() << ")";
    }

    // Close the other half of the condition-variable handshake: a consumer
    // that drained immediately before backpressureActive_ was armed could not
    // wake this source. Recheck after arming; on failure the active flag stays
    // published for the next dequeue, and on success this pass owns credit.
    if (!queue_->tryReserveReceive(
            ptr->metadata.dataSizeBytes, maxInFlightRecvBytes(), &stats)) {
      return false;
    }
    backpressureActive_.store(false, std::memory_order_release);
  }
  reservedReceiveBytes_ = ptr->metadata.dataSizeBytes;

  const bool useHostStaging = shouldHostStageDeviceTransfer(
      communicator->hasCudaTransport(), ptr->metadata.dataSizeBytes);
  if (useHostStaging && !tryReserveRecvHostBytes(ptr->metadata.dataSizeBytes)) {
    queue_->releaseReservedReceive(reservedReceiveBytes_);
    reservedReceiveBytes_ = 0;
    if (getState() == expectedState) {
      setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
    }
    // Global credit is shared across otherwise unrelated ExchangeQueues, so
    // a dequeue from this queue cannot wake us. Retry from the communicator
    // work loop; a completed receive releases this transient credit before
    // publishing the bounce page to its independently bounded queue.
    wakeCommunicator();
    return false;
  }
  if (useHostStaging) {
    reservedGlobalHostReceiveBytes_ = ptr->metadata.dataSizeBytes;
  }

  if (!useHostStaging && !hasRecvDeviceCredit(ptr->metadata.dataSizeBytes)) {
    releaseReceiveReservation();
    if (getState() == expectedState) {
      setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
    }
    // A different ExchangeQueue may release the process-wide credit. Retry
    // from the communicator loop instead of waiting on this queue's promise.
    wakeCommunicator();
    return false;
  }

  // Exchange receive buffers are allocated by the communicator thread, not a
  // Velox Operator, but they compete with operator kernels for the same async
  // CUDA pool. Reserve their not-yet-allocated bytes in the shared workspace
  // domain so four executors cannot all admit against the same physical
  // headroom snapshot. The reservation is released only after the allocation
  // stream is synchronized; from then on cudaMemGetInfo/async-pool usage makes
  // the live buffer visible to subsequent admission decisions. In addition to
  // the steady-state watermark, preserve one minimum consumer workspace. If
  // receives consume that last GiB, the Filter/TopN input which must dequeue
  // them cannot run and both sides wait forever despite a valid byte cap.
  if (!useHostStaging) {
    const auto now = std::chrono::steady_clock::now();
    const bool useProgressHeadroom =
        receiveWorkspaceBlockedSince_.has_value() &&
        now - receiveWorkspaceBlockedSince_.value() >=
            kRecvWorkspaceProgressAge;
    const auto receiveMinHeadroom = useProgressHeadroom
        ? facebook::velox::cudf_velox::CudfConfig::getInstance()
              .deviceMemoryMinHeadroomBytes
        : recvAdmissionMinHeadroom();
    auto receiveProgressLease = useProgressHeadroom
        ? tryAcquireRecvWorkspaceProgressLease(ptr->metadata.dataSizeBytes)
        : nullptr;
    if (useProgressHeadroom && receiveProgressLease == nullptr) {
      releaseReceiveReservation();
      if (getState() == expectedState) {
        setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
      }
      wakeCommunicator();
      return false;
    }
    auto receiveWorkspace =
        facebook::velox::cudf_velox::
            tryAcquireBackgroundDeviceMemoryWorkspace(
                ptr->metadata.dataSizeBytes,
                receiveMinHeadroom,
                facebook::velox::cudf_velox::DeviceMemoryWorkspacePriority::
                    kInput);
    if (!receiveWorkspace.has_value()) {
      if (!receiveWorkspaceBlockedSince_.has_value()) {
        receiveWorkspaceBlockedSince_ = now;
      }
      releaseReceiveReservation();
      if (getState() == expectedState) {
        setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
      }
      wakeCommunicator();
      return false;
    }
    if (useProgressHeadroom && !receiveWorkspaceProgressLogged_) {
      LOG(WARNING) << "CUDF_UCX_RECV_PROGRESS_ADMITTED task=" << taskId_
                   << " remoteTask=" << partitionKey_.taskId
                   << " destination=" << partitionKey_.destination
                   << " bytes=" << ptr->metadata.dataSizeBytes
                   << " progressLimitBytes=" << recvConsumerProgressBytes()
                   << " minHeadroomBytes=" << receiveMinHeadroom;
      receiveWorkspaceProgressLogged_ = true;
    }
    receiveWorkspaceBlockedSince_.reset();
    ptr->receiveWorkspaceProgressLease = std::move(receiveProgressLease);
  }

  // REMOTE EXCHANGE PATH: Allocate buffer and receive via UCXX.
  auto stream =
      facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
  ptr->stream = stream;

  // UCX writes receive buffers from its progress thread, outside CUDA stream
  // ordering. Allocate from the shared async pool, then synchronize this
  // allocation stream before exposing the pointer to UCX. Receive completion
  // transfers ownership to a packed table that retains the same stream, so
  // downstream work and eventual deallocation remain stream ordered.
  auto& recvMemoryResource = receiveDeviceMemoryResource();
  const auto allocateReceiveBuffer = [&]() {
    ptr->dataBuf = std::make_unique<rmm::device_buffer>(
        ptr->metadata.dataSizeBytes,
        stream,
        cuda::mr::any_resource<cuda::mr::device_accessible>{
            recvMemoryResource});
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  };
  if (!useHostStaging) {
    try {
      allocateReceiveBuffer();
    } catch (const rmm::bad_alloc& firstError) {
      // On the allocation-failure slow path, wait for stream-ordered frees and
      // let the shared async pool reclaim completed blocks, then retry once.
      cudaGetLastError();
      const auto syncStatus = cudaDeviceSynchronize();
      if (syncStatus == cudaSuccess) {
        try {
          allocateReceiveBuffer();
          LOG(WARNING)
              << toString()
              << " recovered UCX receive allocation after device synchronize: "
              << ptr->metadata.dataSizeBytes
              << " bytes; first error: " << firstError.what();
        } catch (const rmm::bad_alloc& retryError) {
          VLOG(0) << toString() << " *** RMM failed to allocate "
                  << ptr->metadata.dataSizeBytes
                  << " bytes after device synchronize retry: "
                  << retryError.what();
        }
      } else {
        VLOG(0) << toString()
                << " *** cudaDeviceSynchronize failed while recovering "
                   "receive allocation: "
                << cudaGetErrorString(syncStatus);
      }
    }
  }
  if (useHostStaging || ptr->dataBuf != nullptr) {
    receiveWorkspaceProgressLogged_ = false;
    if (facebook::velox::cudf_velox::deviceMemoryDiagnosticsEnabled()) {
      constexpr int64_t kReportStep = 512LL << 20;
      const auto bytes = recvMemoryResource.get_bytes_counter();
      const auto peakBucket = bytes.peak / kReportStep;
      auto reported =
          lastReportedReceiveDevicePeakBucket.load(std::memory_order_relaxed);
      bool crossedPeakBucket = false;
      while (peakBucket > reported) {
        if (lastReportedReceiveDevicePeakBucket.compare_exchange_weak(
                reported,
                peakBucket,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
          crossedPeakBucket = true;
          break;
        }
      }
      if (!useHostStaging &&
          (crossedPeakBucket ||
           ptr->metadata.dataSizeBytes >= static_cast<uint64_t>(64) << 20)) {
        facebook::velox::cudf_velox::logDeviceMemorySnapshot(
            fmt::format(
                "operator=UcxExchangeSource state=receive.allocate "
                "task={} remoteTask={} destination={} allocationBytes={} "
                "ucxRecvCurrentBytes={} ucxRecvPeakBytes={} "
                "ucxRecvTotalBytes={} queueBytes={} pendingReceiveBytes={} cap={}",
                taskId_,
                partitionKey_.taskId,
                partitionKey_.destination,
                ptr->metadata.dataSizeBytes,
                bytes.value,
                bytes.peak,
                bytes.total,
                stats.queuedBytes,
                stats.pendingReceiveBytes,
                maxInFlightRecvBytes()));
      }
    }
  } else {
    // The pending metadata is retried in place. Do not leave the process-wide
    // progress lane attached to it after a recoverable allocation failure.
    ptr->receiveWorkspaceProgressLease.reset();
    releaseReceiveReservation();
    if (getState() == expectedState) {
      setStateIf(expectedState, ReceiverState::WaitingForReceiveCredit);
    }
    LOG(WARNING) << toString()
                 << " deferring UCX receive after recoverable GPU allocation "
                    "pressure: "
                 << ptr->metadata.dataSizeBytes << " bytes";
    wakeCommunicator();
    return false;
  }

  VLOG(3) << toString() << " Allocated " << ptr->metadata.dataSizeBytes
          << (useHostStaging ? " bytes of deferred host receive memory"
                             : " bytes of device memory");

  // CUDA-aware transports can receive directly into the final device buffer.
  // Only stage through host memory when the active UCX context has confirmed
  // that no CUDA transport is available; sm/tcp cannot write through a device
  // pointer safely in that configuration.
  void* receiveBuffer =
      useHostStaging ? nullptr : ptr->dataBuf->data();
  if (useHostStaging) {
    const auto receiveSize = static_cast<size_t>(ptr->metadata.dataSizeBytes);
    // Keep this allocation attached to the retained UCXX request for the
    // bounded replay window. onData publishes a shared alias, never moves the
    // request's owner away from the raw pointer registered with UCX.
    ptr->hostData = allocateReplaySafeHostReceiveBuffer(receiveSize);
    ptr->hostDataPinned = false;
    receiveBuffer = ptr->hostData.get();
  }
  VLOG(2) << toString() << " posting "
          << (useHostStaging
                  ? (ptr->hostDataPinned ? "pinned-host-staged"
                                         : "pageable-host-staged")
                  : "direct-device")
          << " receive for " << ptr->metadata.dataSizeBytes << " bytes";

  const auto expectedSequence = sequenceNumber_;
  uint64_t dataTag = getDataTag(partitionKeyHash_, expectedSequence);
  VLOG(3) << toString() << " waiting for data for chunk: " << sequenceNumber_
          << " using tag: " << std::hex << dataTag << std::dec;

  if (!setStateIf(expectedState, ReceiverState::WaitingForData)) {
    releaseReceiveReservation();
    VLOG(1) << toString() << " tryStartDataReceive Invalid previous state ";
    return false;
  }

  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  request_ = endpointRef_->endpoint_->tagRecv(
      receiveBuffer,
      ptr->metadata.dataSizeBytes,
      ucxx::Tag{dataTag},
      ucxx::TagMaskFull,
      false,
      [weak, expectedSequence](
          ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onData(status, arg, expectedSequence);
        }
      },
      ptr);
  pendingReceive_.reset();
  return true;
}

void UcxExchangeSource::onData(
    ucs_status_t status,
    std::shared_ptr<void> arg,
    uint32_t expectedSequence) {
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(2) << "[UCX-SOURCE-DATA-AFTER-CLOSE] " << toString()
            << " seq=" << sequenceNumber_
            << " status=" << ucs_status_string(status);
    releaseReceiveReservation();
    deliverEndMarker();
    return;
  }
  if (expectedSequence != sequenceNumber_) {
    VLOG(1) << toString() << " ignoring replayed data callback for seq="
            << expectedSequence << ", currentSeq=" << sequenceNumber_;
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForData) {
    VLOG(2) << toString() << " onData called in state " << toName(getState())
            << ", ignoring (possible UCXX replay)";
    return;
  }
  VLOG(3) << toString() << " + onData " << ucs_status_string(status);

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive data from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << "[UCX-SOURCE-DATA-ERROR] " << toString()
            << " seq=" << sequenceNumber_ << " state=" << toName(getState())
            << " error=" << errorMsg;
    releaseReceiveReservation();
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
  } else {
    VLOG(3) << toString() << "+ onData " << ucs_status_string(status)
            << " got chunk: " << sequenceNumber_;

    this->sequenceNumber_++;

    std::shared_ptr<DataAndMetadata> ptr =
        std::static_pointer_cast<DataAndMetadata>(arg);

    if (ptr->hostData != nullptr) {
      metrics_.hostStagedBytes_.addValue(ptr->metadata.dataSizeBytes);
      if (exchangeVariableWidthValidationEnabled()) {
        LOG(WARNING) << "UCX receiver staged key="
                     << partitionKey_.toString()
                     << " sequence=" << expectedSequence
                     << " bytes=" << ptr->metadata.dataSizeBytes
                     << " fingerprint=0x" << std::hex
                     << diagnosticBufferFingerprint(
                            ptr->hostData.get(),
                            ptr->metadata.dataSizeBytes)
                     << std::dec;
      }
      auto data = std::make_unique<PackedTableWithStream>(
          std::move(ptr->metadata.cudfMetadata),
          ptr->hostData,
          ptr->metadata.dataSizeBytes,
          ptr->hostDataPinned,
          ptr->stream,
          nullptr);
      // The process-wide host credit bounds concurrent UCX receive DMA, not
      // completed bounce pages. Holding it until H2D creates a cross-edge
      // dependency inversion: a downstream queue can consume all global
      // credit while the upstream queue needed to unblock it cannot receive.
      // Completed pages remain bounded by UcxExchangeQueue's per-edge byte
      // credit (512 MiB in Job 144), so release the transient global credit
      // immediately before publishing the page.
      releaseRecvHostBytes(reservedGlobalHostReceiveBytes_);
      reservedGlobalHostReceiveBytes_ = 0;
      metrics_.numPackedColumns_.addValue(1);
      metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);
      const int64_t reservedReceiveBytes = reservedReceiveBytes_;
      enqueue(std::move(data), reservedReceiveBytes);
      reservedReceiveBytes_ = 0;
      setStateIf(
          ReceiverState::WaitingForData, ReceiverState::ReadyToReceive);
      wakeCommunicator();
      return;
    }
    metrics_.numPackedColumns_.addValue(1);
    metrics_.totalBytes_.addValue(ptr->metadata.dataSizeBytes);

    // Create packed_columns from the received metadata and data buffer
    cudf::packed_columns packedCols(
        std::move(ptr->metadata.cudfMetadata), std::move(ptr->dataBuf));

    // Unpack to get the table_view and create a packed_table
    cudf::table_view tableView = cudf::unpack(packedCols);
    if (exchangeVariableWidthValidationEnabled()) {
      LOG(WARNING) << "UCX receiver validating remoteTask="
                   << partitionKey_.taskId
                   << " destination=" << partitionKey_.destination
                   << " sequence=" << expectedSequence
                   << " rows=" << tableView.num_rows()
                   << " bytes=" << ptr->metadata.dataSizeBytes;
      LOG(WARNING) << "UCX receiver page remoteTask=" << partitionKey_.taskId
                   << " destination=" << partitionKey_.destination
                   << " sequence=" << expectedSequence
                   << " rows=" << tableView.num_rows()
                   << " bytes=" << ptr->metadata.dataSizeBytes
                   << " layout=" << cudf_velox::validateVariableWidthTableLayout(
                          tableView, ptr->stream);
    }
    auto packedTable = std::make_unique<cudf::packed_table>(
        cudf::packed_table{tableView, std::move(packedCols)});

    // Bundle the packed_table with the stream that was used for allocation
    auto data = std::make_unique<PackedTableWithStream>(
        std::move(packedTable),
        ptr->stream,
        std::move(ptr->receiveWorkspaceProgressLease));

    const int64_t reservedReceiveBytes = reservedReceiveBytes_;
    enqueue(std::move(data), reservedReceiveBytes);
    reservedReceiveBytes_ = 0;
    setStateIf(ReceiverState::WaitingForData, ReceiverState::ReadyToReceive);
  }
  wakeCommunicator();
}

void UcxExchangeSource::receiveHandshakeResponse() {
  auto responseBuffer = std::make_shared<HandshakeResponse>();
  uint64_t responseTag = getHandshakeResponseTag(partitionKeyHash_);

  VLOG(3) << toString()
          << " waiting for HandshakeResponse with tag: " << std::hex
          << responseTag << std::dec;

  // Use weak_ptr to prevent use-after-free if close() is called during callback
  std::weak_ptr<UcxExchangeSource> weak = weak_from_this();
  retireRequest(request_, completedRequests_);
  request_ = endpointRef_->endpoint_->tagRecv(
      responseBuffer.get(),
      sizeof(*responseBuffer),
      ucxx::Tag{responseTag},
      ucxx::TagMaskFull,
      false,
      [weak](ucs_status_t status, std::shared_ptr<void> arg) {
        if (auto self = weak.lock()) {
          self->onHandshakeResponse(status, arg);
        }
      },
      responseBuffer);
}

void UcxExchangeSource::onHandshakeResponse(
    ucs_status_t status,
    std::shared_ptr<void> arg) {
  // The peer has now consumed the AM payload and returned a response, so the
  // retained handshake buffer can finally be released.
  handshakeRequestBuffer_.reset();
  // Check if close() was called - avoid processing if we're shutting down
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " onHandshakeResponse called after close, ignoring";
    deliverEndMarker();
    return;
  }
  // Guard against replayed callbacks from UCP wireup replay.
  if (getState() != ReceiverState::WaitingForHandshakeResponse) {
    VLOG(2) << toString() << " onHandshakeResponse called in state "
            << toName(getState()) << ", ignoring (possible UCXX replay)";
    return;
  }

  if (status != UCS_OK) {
    std::string errorMsg = fmt::format(
        "Failed to receive HandshakeResponse from host {}:{}, task {}: {}",
        host_,
        port_,
        partitionKey_.toString(),
        ucs_status_string(status));
    VLOG(0) << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
    return;
  }

  std::shared_ptr<HandshakeResponse> response =
      std::static_pointer_cast<HandshakeResponse>(arg);

  isIntraNodeTransfer_ = response->isIntraNodeTransfer;

  VLOG(2) << "[UCX-SOURCE-HANDSHAKE-RESPONSE] localTask=" << taskId_
          << " remoteTask=" << partitionKey_.taskId
          << " destination=" << partitionKey_.destination << " peer=" << host_
          << ":" << port_ << " isIntraNodeTransfer=" << isIntraNodeTransfer_;

  setStateIf(
      ReceiverState::WaitingForHandshakeResponse,
      ReceiverState::ReadyToReceive);
  wakeCommunicator();
}

void UcxExchangeSource::waitForIntraNodeData() {
  // Check if close() was called
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString()
            << " waitForIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  IntraNodeTransferKey key{
      partitionKey_.taskId, partitionKey_.destination, sequenceNumber_};

  auto result = IntraNodeTransferRegistry::getInstance()->poll(key);

  if (!result.has_value()) {
    // Event-driven wait: re-queuing here would busy-spin the single-threaded
    // Communicator and can starve the same-process producer's publish() (which
    // needs that same thread), livelocking the intra-node transfer. Instead
    // register a one-shot wakeup and go dormant; publish()/cancelTask()
    // re-enqueues this source exactly once when data is available. The weak_ptr
    // keeps the wakeup safe if the source is destroyed first. Capture the
    // Communicator weakly as well so a dormant waiter cannot extend the
    // singleton lifetime during executor shutdown.
    auto self = getSelfPtr();
    std::weak_ptr<CommElement> weakSelf = self;
    auto weakCommunicator = communicator_;
    const bool readyNow =
        IntraNodeTransferRegistry::getInstance()->registerWaiter(
            key, [weakSelf, weakCommunicator]() {
              if (auto source = weakSelf.lock(); source) {
                auto communicator = weakCommunicator.lock();
                if (!communicator) {
                  return;
                }
                communicator->addToWorkQueue(source);
              }
            });
    if (readyNow) {
      // Data landed (or the task was cancelled) between poll and register —
      // re-poll once instead of going dormant.
      wakeCommunicator();
    }
    return;
  }

  intraNodePollCount_ = 0;
  onIntraNodeData(std::move(result.value()));
}

void UcxExchangeSource::onIntraNodeData(IntraNodeTransferResult result) {
  // Check if close() was called
  if (closed_.load(std::memory_order_acquire)) {
    VLOG(3) << toString() << " onIntraNodeData called after close, ignoring";
    deliverEndMarker();
    return;
  }

  if (result.atEnd) {
    // End of stream
    atEnd_ = true;
    VLOG(3) << toString() << " Intra-node transfer: end of stream";
    deliverEndMarker();
    setState(ReceiverState::Done);

    wakeCommunicator();
    return;
  }

  if (!result.data && !result.isHostBacked()) {
    // Error - should not happen if atEnd is false
    std::string errorMsg = fmt::format(
        "Intra-node transfer data is null for task {}, dest {}, seq {}",
        partitionKey_.taskId,
        partitionKey_.destination,
        sequenceNumber_);
    VLOG(0) << toString() << " " << errorMsg;
    queue_->setError(errorMsg);
    deliverEndMarker();
    setState(ReceiverState::Done);
    wakeCommunicator();
    return;
  }

  const auto dataBytes = result.isHostBacked()
      ? result.hostDataSize
      : result.data->gpu_data->size();
  VLOG(3) << toString()
          << " Intra-node transfer: received data for seq=" << sequenceNumber_
          << " size=" << dataBytes
          << " hostBacked=" << result.isHostBacked();

  metrics_.numPackedColumns_.addValue(1);
  metrics_.totalBytes_.addValue(dataBytes);
  if (result.isHostBacked()) {
    VELOX_CHECK_NOT_NULL(result.hostMetadata);
    metrics_.hostStagedBytes_.addValue(dataBytes);
    auto stream =
        facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
    auto tableWithStream = std::make_unique<PackedTableWithStream>(
        std::move(result.hostMetadata),
        std::move(result.hostData),
        result.hostDataSize,
        result.hostDataPinned,
        stream,
        nullptr);
    enqueue(std::move(tableWithStream));

    this->sequenceNumber_++;
    setStateIf(
        ReceiverState::WaitingForIntraNodeData, ReceiverState::ReadyToReceive);
    wakeCommunicator();
    return;
  }

  auto data = std::move(result.data);
  // Make the consumer page independently owned. Moving a uniquely referenced
  // producer allocation into the consumer looked safe, but under sustained
  // HASH exchange its stream-ordered allocation was recycled while the page
  // was still queued: Job 144 consistently observed valid pages 1..84 and a
  // negative STRING offset in page 85 before concatenate touched it. A
  // consumer-stream clone is an explicit ownership boundary for both HASH and
  // BROADCAST pages. Join the producer stream and finish the bounded D2D copy
  // before releasing the registry's producer allocation.
  auto stream =
      facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
  std::vector<rmm::cuda_stream_view> producerStreams{result.stream};
  cudf::detail::join_streams(producerStreams, stream);
  auto consumerData = std::make_unique<rmm::device_buffer>(
      data->gpu_data->data(), data->gpu_data->size(), stream);
  stream.synchronize();
  cudf::packed_columns packedCols(
      std::make_unique<std::vector<uint8_t>>(*data->metadata),
      std::move(consumerData));

  // Unpack to get the table_view and create a packed_table
  cudf::table_view tableView = cudf::unpack(packedCols);
  auto packedTable = std::make_unique<cudf::packed_table>(
      cudf::packed_table{tableView, std::move(packedCols)});

  auto tableWithStream =
      std::make_unique<PackedTableWithStream>(std::move(packedTable), stream);

  enqueue(std::move(tableWithStream));

  this->sequenceNumber_++;
  setStateIf(
      ReceiverState::WaitingForIntraNodeData, ReceiverState::ReadyToReceive);
  wakeCommunicator();
}

bool UcxExchangeSource::setStateIf(
    UcxExchangeSource::ReceiverState expected,
    UcxExchangeSource::ReceiverState desired) {
  ReceiverState exp = expected;
  // since spurious failures can happen even if state_ == expected, we need
  // to do this in a loop.
  while (!state_.compare_exchange_strong(
      exp, desired, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    if (exp != expected) {
      // no spurious failure, state isn't what we've expected.
      return false;
    }
    // spurious failure.
    exp = expected; // reset for the next try
  }
  VLOG(2) << (isIntraNodeTransfer_ ? "[INTRA]" : "[REMOTE]") << " [ExSrc "
          << toString() << " seq=" << sequenceNumber_ << "] "
          << toName(expected) << " -> " << toName(desired);
  return true;
}

} // namespace facebook::velox::ucx_exchange
