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
#include <fmt/format.h>
#include <cstring>

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/ucx-exchange/Acceptor.h"
#include "velox/experimental/ucx-exchange/Communicator.h"
#include "velox/experimental/ucx-exchange/EndpointRef.h"
#include "velox/experimental/ucx-exchange/UcxExchangeProtocol.h"
#include "velox/experimental/ucx-exchange/UcxExchangeServer.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"

namespace facebook::velox::ucx_exchange {
namespace {

// Active-message callbacks run on the UCX progress thread and must not throw or
// call closeBlocking(). Defer endpoint teardown so the peer's outstanding
// HandshakeResponse receive completes with an error instead of waiting forever.
void rejectHandshake(ucp_ep_h ep, const std::string& reason) noexcept {
  LOG(ERROR) << "Rejecting UCX exchange handshake: " << reason;
  try {
    auto communicator = Communicator::tryGetInstance();
    if (!communicator || communicator->isShuttingDown()) {
      return;
    }
    auto epRef = communicator->findEndpointRefByHandle(ep);
    if (!epRef) {
      LOG(ERROR) << "Cannot close rejected UCX handshake endpoint: endpoint "
                    "reference is unavailable";
      return;
    }
    communicator->deferEndpointCleanup(std::move(epRef));
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to defer rejected UCX handshake cleanup: "
               << e.what();
  } catch (...) {
    LOG(ERROR)
        << "Failed to defer rejected UCX handshake cleanup: unknown error";
  }
}

} // namespace

/*static*/
void Acceptor::cStyleAMCallback(
    std::shared_ptr<ucxx::Request> request,
    ucp_ep_h ep) {
  try {
    VELOX_CHECK_NOT_NULL(request, "AMCallback called with nullptr request!");
    VELOX_CHECK(
        request->isCompleted(), "AMCallback called with incomplete request!");
    if (request->getStatus() != UCS_OK) {
      rejectHandshake(
          ep,
          fmt::format(
              "active-message receive failed: {}",
              ucs_status_string(request->getStatus())));
      return;
    }
    auto communicator = Communicator::tryGetInstance();
    if (!communicator || communicator->isShuttingDown()) {
      return;
    }
    auto buffer =
        std::dynamic_pointer_cast<ucxx::Buffer>(request->getRecvBuffer());
    VELOX_CHECK_NOT_NULL(buffer, "AMCallback: failed to get receive buffer.");
    // Validate buffer size BEFORE casting to prevent reading past buffer
    // bounds.
    VELOX_CHECK_GE(
        buffer->getSize(),
        sizeof(HandshakeMsg),
        "AMCallback: received buffer size ({}) is smaller than HandshakeMsg ({}). "
        "Possible protocol mismatch or truncated message.",
        buffer->getSize(),
        sizeof(HandshakeMsg));
    // Copy out of UCXX-owned storage before parsing.  This also avoids relying
    // on the alignment of the active-message receive buffer.
    HandshakeMsg handshake{};
    std::memcpy(&handshake, buffer->data(), sizeof(handshake));
    const auto taskIdEnd = static_cast<const char*>(
        std::memchr(handshake.taskId, '\0', sizeof(handshake.taskId)));
    VELOX_CHECK_NOT_NULL(taskIdEnd, "Handshake task id is not NUL terminated");
    VELOX_CHECK_GT(taskIdEnd, handshake.taskId, "Handshake task id is empty");
    VELOX_CHECK_LT(
        handshake.destination,
        65536,
        "Handshake destination is invalid: {}",
        handshake.destination);

    // Create a exchangeServer based on the information received in the initial
    // handshake.
    auto epRef = communicator->findEndpointRefByHandle(ep);
    VELOX_CHECK_NOT_NULL(epRef, "Could not find endpoint reference");
    const std::string peerAddress = epRef->getPeerAddress();

    const PartitionKey key = {handshake.taskId, handshake.destination};

    // Determine if this is an intra-process transfer by comparing the source's
    // workerId with our Communicator's workerId. A match means both source and
    // server are in the same Communicator singleton (same process), so
    // IntraNodeTransferRegistry (in-process std::promise/future) can be used.
    //
    // Previous approach used IP comparison (getLocalIpAddresses), which fails
    // when multiple Docker containers share the same host IP address.
    const bool sameWorker = handshake.workerId == communicator->getWorkerId();
    bool isIntraNodeTransfer =
        cudf_velox::CudfConfig::getInstance().intraNodeExchange && sameWorker;

    // Disable intra-node until the task is initialized. Broadcast is safe after
    // initialization because the intra-node source clones shared pages.
    if (isIntraNodeTransfer) {
      auto queueMgr = UcxOutputQueueManager::getInstanceRef();
      const bool canUseIntraNode = queueMgr->canUseIntraNode(key.taskId);
      VLOG(2) << "[UCX-ACCEPTOR-INTRA-CHECK] task=" << key.taskId
              << " destination=" << key.destination << " peer=" << peerAddress
              << " sourceWorkerId=" << handshake.workerId
              << " localWorkerId=" << communicator->getWorkerId()
              << " sameWorker=" << sameWorker
              << " canUseIntraNode=" << canUseIntraNode
              << " queue=" << queueMgr->describeQueueForIntraNode(key.taskId);
      if (!canUseIntraNode) {
        VLOG(2) << "[ACCEPTOR] Disabling intra-node for task " << key.taskId
                << " (not initialized or broadcast)";
        isIntraNodeTransfer = false;
      }
    } else {
      VLOG(2) << "[UCX-ACCEPTOR-REMOTE] task=" << key.taskId
              << " destination=" << key.destination << " peer=" << peerAddress
              << " sourceWorkerId=" << handshake.workerId
              << " localWorkerId=" << communicator->getWorkerId()
              << " sameWorker=" << sameWorker << " intraNodeEnabled="
              << cudf_velox::CudfConfig::getInstance().intraNodeExchange;
    }

    auto exchangeServer = UcxExchangeServer::create(
        communicator, epRef, key, isIntraNodeTransfer);

    // Add this exchangeServer to the endpoint reference.
    epRef->addCommElem(exchangeServer);

    // Register exchangeServer with communicator.
    communicator->registerCommElement(exchangeServer);
    VLOG(2) << "[ACCEPTOR] new server: " << exchangeServer->toString()
            << " peer=" << peerAddress
            << " isIntraNodeTransfer=" << isIntraNodeTransfer;

    // Send HandshakeResponse back to the source to inform about intra-node
    // transfer. This allows the source to bypass UCXX for all subsequent data
    // transfers.
    auto response = std::make_shared<HandshakeResponse>();
    response->isIntraNodeTransfer = exchangeServer->isIntraNodeTransfer();

    uint32_t keyHash = fnv1a_32(key.toString());
    uint64_t responseTag = getHandshakeResponseTag(keyHash);

    VLOG(3) << "Sending HandshakeResponse to " << key.toString()
            << " peer=" << peerAddress
            << " isIntraNodeTransfer=" << response->isIntraNodeTransfer
            << " tag=" << std::hex << responseTag;

    // Fire-and-forget: we don't need to track this request completion
    epRef->endpoint_->tagSend(
        response.get(),
        sizeof(*response),
        ucxx::Tag{responseTag},
        false,
        [response, keyStr = key.toString(), peerAddress](
            ucs_status_t status, std::shared_ptr<void> arg) {
          if (status == UCS_OK) {
            VLOG(3) << "HandshakeResponse sent successfully to " << keyStr
                    << " peer=" << peerAddress;
          } else {
            VLOG(0) << "Failed to send HandshakeResponse to " << keyStr << ": "
                    << ucs_status_string(status) << " peer=" << peerAddress;
          }
        },
        response);
  } catch (const std::exception& e) {
    rejectHandshake(ep, e.what());
  } catch (...) {
    rejectHandshake(ep, "unknown active-message callback failure");
  }
}

// Add endpoint reference to ucp_cp -> epRef map.
void Acceptor::registerEndpointRef(std::shared_ptr<EndpointRef> endpointRef) {
  auto epHandle = endpointRef->endpoint_->getHandle();
  auto res = handleToEndpointRef_.insert(std::pair{epHandle, endpointRef});
  VELOX_CHECK(res.second, "Endpoint handle already exists!");
}
} // namespace facebook::velox::ucx_exchange
