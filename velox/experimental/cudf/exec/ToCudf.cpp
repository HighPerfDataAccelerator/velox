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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/CudfTopNRowNumber.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/JitExpression.h"
#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"

#include "folly/Conv.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Values.h"

#include <cudf/detail/nvtx/ranges.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <dlfcn.h>
#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

static const std::string kCudfAdapterName = "cuDF";
DEFINE_bool(velox_cudf_enabled, true, "Enable cuDF-Velox acceleration");
DEFINE_string(velox_cudf_memory_resource, "async", "Memory resource for cuDF");
DEFINE_bool(velox_cudf_debug, false, "Enable debug printing");
DEFINE_bool(velox_ucx_exchange, true, "Enable UCX exchange");

using namespace facebook::velox::ucx_exchange;

namespace facebook::velox::cudf_velox {

namespace {

template <class... Deriveds, class Base>
bool isAnyOf(const Base* p) {
  return ((dynamic_cast<const Deriveds*>(p) != nullptr) || ...);
}

bool envFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  std::string normalized(value);
  std::transform(
      normalized.begin(),
      normalized.end(),
      normalized.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return normalized != "0" && normalized != "false" && normalized != "off" &&
      normalized != "no";
}

// Per-thread GPU operator-attribution context. The velox operators set this
// (through ScopedGpuMemoryOperatorContext, which dlsym-resolves the extern "C"
// setter defined at the bottom of this file), and the memory tracker reads it
// when recording each allocation. Read the thread-local directly here to avoid
// a reader-side dlsym (the setter still goes through dlsym so the header stays
// usable from a standalone Velox build).
thread_local std::string tlsGpuOperatorContext;

std::string currentGpuOperatorContext() {
  if (tlsGpuOperatorContext.empty()) {
    return "unknown";
  }
  return tlsGpuOperatorContext;
}

class OutputMemoryResourceTracker final
    : public rmm::mr::device_memory_resource {
 public:
  explicit OutputMemoryResourceTracker(
      std::shared_ptr<rmm::mr::device_memory_resource> upstream,
      std::string label)
      : upstream_(std::move(upstream)), label_(std::move(label)) {}

  void dumpDiagnostics(const std::string& prefix) {
    struct ContextSummary {
      std::size_t bytes{0};
      std::size_t count{0};
    };
    std::vector<std::pair<std::string, ContextSummary>> summaries;
    std::size_t activeBytes = 0;
    std::size_t activeCount = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::unordered_map<std::string, ContextSummary> byContext;
      activeCount = activeAllocations_.size();
      for (const auto& entry : activeAllocations_) {
        activeBytes += entry.second.bytes;
        auto& summary = byContext[entry.second.context];
        summary.bytes += entry.second.bytes;
        ++summary.count;
      }
      summaries.reserve(byContext.size());
      for (const auto& entry : byContext) {
        summaries.emplace_back(entry.first, entry.second);
      }
    }

    std::sort(
        summaries.begin(),
        summaries.end(),
        [](const auto& left, const auto& right) {
          if (left.second.bytes != right.second.bytes) {
            return left.second.bytes > right.second.bytes;
          }
          return left.second.count > right.second.count;
        });

    LOG(ERROR) << prefix << " " << label_
               << " activeOutputBytes=" << activeBytes
               << " activeOutputMiB=" << (activeBytes / 1048576.0)
               << " activeOutputCount=" << activeCount
               << " upstreamType=" << typeid(*upstream_).name();
    const auto limit = std::min<std::size_t>(summaries.size(), 30);
    for (std::size_t i = 0; i < limit; ++i) {
      LOG(ERROR) << prefix << " " << label_ << " activeContext rank=" << i
                 << " bytes=" << summaries[i].second.bytes
                 << " MiB=" << (summaries[i].second.bytes / 1048576.0)
                 << " count=" << summaries[i].second.count
                 << " context=" << summaries[i].first;
    }
  }

 private:
  struct ActiveAllocation {
    std::size_t bytes;
    std::string context;
  };

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override {
    try {
      void* ptr = upstream_->allocate(stream, bytes);
      recordAllocation(ptr, bytes);
      return ptr;
    } catch (const std::exception& e) {
      dumpFailure(bytes, e.what());
      throw;
    } catch (...) {
      dumpFailure(bytes, "unknown exception");
      throw;
    }
  }

  void do_deallocate(
      void* ptr,
      std::size_t bytes,
      rmm::cuda_stream_view stream) noexcept override {
    upstream_->deallocate(stream, ptr, bytes);
    std::lock_guard<std::mutex> lock(mutex_);
    activeAllocations_.erase(ptr);
  }

  bool do_is_equal(
      rmm::mr::device_memory_resource const& other) const noexcept override {
    if (&other == this) {
      return true;
    }
    if (auto* wrapped =
            dynamic_cast<const OutputMemoryResourceTracker*>(&other)) {
      return upstream_->is_equal(*wrapped->upstream_);
    }
    return upstream_->is_equal(other);
  }

  void recordAllocation(void* ptr, std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    activeAllocations_[ptr] =
        ActiveAllocation{bytes, currentGpuOperatorContext()};
  }

  void dumpFailure(std::size_t requestedBytes, const char* error) {
    std::size_t freeMem = 0;
    std::size_t totalMem = 0;
    const auto memErr = cudaMemGetInfo(&freeMem, &totalMem);

    LOG(ERROR) << label_ << " OOM requestedBytes=" << requestedBytes
               << " requestedMiB=" << (requestedBytes / 1048576.0)
               << " error=" << (error == nullptr ? "unknown" : error)
               << " currentContext=" << currentGpuOperatorContext()
               << " cudaMemGetInfoStatus="
               << (memErr == cudaSuccess ? "ok" : cudaGetErrorString(memErr))
               << " cudaFreeBytes=" << freeMem
               << " cudaFreeMiB=" << (freeMem / 1048576.0)
               << " cudaTotalBytes=" << totalMem
               << " cudaTotalMiB=" << (totalMem / 1048576.0)
               << " upstreamType=" << typeid(*upstream_).name();
    dumpDiagnostics(label_ + " OOM");
  }

  std::shared_ptr<rmm::mr::device_memory_resource> upstream_;
  std::string label_;
  std::mutex mutex_;
  std::unordered_map<void*, ActiveAllocation> activeAllocations_;
};

std::mutex& outputMemoryTrackerRegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<std::weak_ptr<OutputMemoryResourceTracker>>&
outputMemoryTrackerRegistry() {
  static std::vector<std::weak_ptr<OutputMemoryResourceTracker>> trackers;
  return trackers;
}

void registerOutputMemoryTracker(
    const std::shared_ptr<OutputMemoryResourceTracker>& tracker) {
  std::lock_guard<std::mutex> lock(outputMemoryTrackerRegistryMutex());
  auto& trackers = outputMemoryTrackerRegistry();
  trackers.erase(
      std::remove_if(
          trackers.begin(),
          trackers.end(),
          [](const auto& weak) { return weak.expired(); }),
      trackers.end());
  trackers.push_back(tracker);
}

void dumpRegisteredOutputMemoryTrackers(const std::string& prefix) {
  std::vector<std::shared_ptr<OutputMemoryResourceTracker>> trackers;
  {
    std::lock_guard<std::mutex> lock(outputMemoryTrackerRegistryMutex());
    auto& registry = outputMemoryTrackerRegistry();
    for (auto it = registry.begin(); it != registry.end();) {
      if (auto tracker = it->lock()) {
        trackers.push_back(std::move(tracker));
        ++it;
      } else {
        it = registry.erase(it);
      }
    }
  }
  if (trackers.empty()) {
    LOG(ERROR) << prefix << " cudfMemoryResources activeTrackerCount=0";
    return;
  }
  LOG(ERROR) << prefix
             << " cudfMemoryResources activeTrackerCount=" << trackers.size();
  for (auto& tracker : trackers) {
    tracker->dumpDiagnostics(prefix);
  }
}

std::shared_ptr<rmm::mr::device_memory_resource> maybeWrapSharedMemoryResource(
    std::shared_ptr<rmm::mr::device_memory_resource> resource) {
  if (!envFlagEnabled("GLUTEN_GPU_MEMORY_OOM_DUMP")) {
    return resource;
  }
  LOG(INFO)
      << "CudfSharedPoolMemoryResource OOM dump wrapper enabled upstreamType="
      << typeid(*resource).name();
  auto tracker = std::make_shared<OutputMemoryResourceTracker>(
      std::move(resource), "CudfSharedPoolMemoryResource");
  registerOutputMemoryTracker(tracker);
  return tracker;
}

std::shared_ptr<rmm::mr::device_memory_resource> maybeWrapOutputMemoryResource(
    std::shared_ptr<rmm::mr::device_memory_resource> resource) {
  if (!envFlagEnabled("GLUTEN_GPU_MEMORY_OOM_DUMP")) {
    return resource;
  }
  LOG(INFO) << "CudfOutputMemoryResource OOM dump wrapper enabled upstreamType="
            << typeid(*resource).name();
  auto tracker = std::make_shared<OutputMemoryResourceTracker>(
      std::move(resource), "CudfOutputMemoryResource");
  registerOutputMemoryTracker(tracker);
  return tracker;
}

} // namespace

extern "C" void glutenCudfDumpMemoryResources(const char* prefix) {
  dumpRegisteredOutputMemoryTrackers(
      prefix == nullptr ? std::string("GpuMemoryTracker")
                        : std::string(prefix));
}

// GPU memory operator-attribution context, exported so the velox-side
// ScopedGpuMemoryOperatorContext (which dlsym-resolves these to stay usable in
// a standalone Velox build) can tag each allocation with its operator. Without
// these definitions the dump reports context=unavailable/unknown. Marked
// default-visibility so dlsym(RTLD_DEFAULT, ...) finds them inside
// libgluten.so.
extern "C" __attribute__((visibility("default"))) const char*
glutenGpuMemoryTrackerCurrentOperatorContext() {
  return tlsGpuOperatorContext.empty() ? nullptr
                                       : tlsGpuOperatorContext.c_str();
}

extern "C" __attribute__((visibility("default"))) void
glutenGpuMemoryTrackerSetCurrentOperatorContext(const char* context) {
  tlsGpuOperatorContext = (context == nullptr) ? std::string() : context;
}

extern "C" __attribute__((visibility("default"))) void
glutenGpuMemoryTrackerClearCurrentOperatorContext() {
  tlsGpuOperatorContext.clear();
}

core::PlanNodePtr CompileState::getPlanNode(const core::PlanNodeId& id) const {
  auto& nodes = driverFactory_.planNodes;
  auto it = std::find_if(nodes.cbegin(), nodes.cend(), [&id](const auto& node) {
    return node->id() == id;
  });
  if (it != nodes.end()) {
    return *it;
  }
  VELOX_CHECK(driverFactory_.consumerNode->id() == id);
  return driverFactory_.consumerNode;
}

bool CompileState::compile(bool allowCpuFallback) {
  auto operators = driver_.operators();

  // Cache debug flag to avoid repeated getInstance() calls
  const bool debugEnabled = CudfConfig::getInstance().debugEnabled;

  // Cache "before" operator descriptions so we can print before/after together.
  std::vector<std::pair<int32_t, std::string>> beforeOperators;
  if (debugEnabled) {
    for (const auto& op : operators) {
      beforeOperators.emplace_back(op->operatorId(), op->toString());
    }
  }

  bool replacementsMade = false;
  auto ctx = driver_.driverCtx();

  // Helper to check if planNodeId is valid (some operators like CallbackSink
  // have "N/A")
  auto isValidPlanNodeId = [](const core::PlanNodeId& id) {
    return !id.empty() && id != "N/A";
  };

  // Use adapter registry for GPU Operator Replacement
  auto& registry = OperatorAdapterRegistry::getInstance();

  // Cached operator properties including adapter pointer.
  struct OperatorProperties : OperatorAdapter::Properties {
    const OperatorAdapter* adapter = nullptr;
  };

  auto getOperatorProperties =
      [&registry, this, &isValidPlanNodeId, ctx](const exec::Operator* op) {
        OperatorProperties props;
        auto adapter = registry.findAdapter(op);
        props.adapter = adapter;
        if (adapter && isValidPlanNodeId(op->planNodeId())) {
          static_cast<OperatorAdapter::Properties&>(props) =
              adapter->properties(op, getPlanNode(op->planNodeId()), ctx);
        }
        if (isAnyOf<CudfOperator>(op)) {
          // CudfOperator is always fully GPU compatible
          // (runs on GPU, accepts GPU input, produces GPU output).
          props.canRunOnGPU = true;
          props.acceptsGpuInput = true;
          props.producesGpuOutput = true;
        }
        return props;
      };

  // caching operator properties
  std::vector<OperatorProperties> opProps(operators.size());
  std::transform(
      operators.begin(),
      operators.end(),
      opProps.begin(),
      getOperatorProperties);

  int32_t operatorsOffset = 0;
  for (int32_t operatorIndex = 0; operatorIndex < operators.size();
       ++operatorIndex) {
    std::vector<std::unique_ptr<exec::Operator>> replaceOp;

    exec::Operator* oper = operators[operatorIndex];
    auto replacingOperatorIndex = operatorIndex + operatorsOffset;
    VELOX_CHECK(oper);
    const auto& thisOpProps =
        opProps[operatorIndex]; // cached operator properties

    const bool previousOperatorIsNotGpu =
        operatorIndex > 0 and !opProps[operatorIndex - 1].producesGpuOutput;
    const bool nextOperatorIsNotGpu = (operatorIndex < operators.size() - 1) and
        !opProps[operatorIndex + 1].acceptsGpuInput;
    const bool isLastOperatorOfTask =
        driverFactory_.outputDriver and operatorIndex == operators.size() - 1;

    auto id = oper->operatorId();

    // Cache planNode for this operator (avoid multiple lookups)
    core::PlanNodePtr planNode = nullptr;
    if (isValidPlanNodeId(oper->planNodeId())) {
      planNode = getPlanNode(oper->planNodeId());
    }

    if (previousOperatorIsNotGpu and thisOpProps.acceptsGpuInput and planNode) {
      replaceOp.push_back(
          std::make_unique<CudfFromVelox>(
              id, planNode->outputType(), ctx, planNode->id() + "-from-velox"));
    }
    if (not replaceOp.empty()) {
      // from-velox only, because need to inserted before current operator.
      operatorsOffset += replaceOp.size();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_,
          replacingOperatorIndex,
          replacingOperatorIndex,
          std::move(replaceOp));
      replacingOperatorIndex = operatorIndex + operatorsOffset;
      replaceOp.clear();
      replacementsMade = true;
    }

    // Use adapter registry to handle operator replacement
    auto keepOperator = 1; // Default: keep original
    const auto& adapter = thisOpProps.adapter;
    bool isPureCpuOperator = true;

    if (adapter) {
      keepOperator = adapter->keepOperator();
      if (keepOperator == 0) {
        if (planNode && thisOpProps.canRunOnGPU) {
          auto replacements =
              adapter->createReplacements(oper, planNode, ctx, id);
          for (auto& r : replacements) {
            replaceOp.push_back(std::move(r));
          }
          isPureCpuOperator = false;
        } else {
          // This is the CPU fallback case.
          isPureCpuOperator = true;
        }
      } else {
        // adapter is present and keepOperator is 1, so this is GPU compatible
        // operator. so this CPU operators is allowed even if fallback is
        // disabled.
        isPureCpuOperator = false;
      }
    } else {
      // special case for CudfOperator
      if (isAnyOf<CudfOperator>(oper)) {
        isPureCpuOperator = false;
      } else {
        // CPU operator without adapter
        isPureCpuOperator = true;
      }
    }

    if (thisOpProps.producesGpuOutput and
        (nextOperatorIsNotGpu or isLastOperatorOfTask) and planNode) {
      replaceOp.push_back(
          std::make_unique<CudfToVelox>(
              id, planNode->outputType(), ctx, planNode->id() + "-to-velox"));
    }

    if (debugEnabled) {
      VLOG(1) << "Operator: ID " << oper->operatorId() << ": "
              << oper->toString() << ", keepOperator = " << keepOperator
              << ", isPureCpuOperator = " << isPureCpuOperator
              << ", replaceOp.size() = " << replaceOp.size()
              << ", previousOperatorIsNotGpu = " << previousOperatorIsNotGpu
              << ", nextOperatorIsNotGpu = " << nextOperatorIsNotGpu
              << ", isLastOperatorOfTask = " << isLastOperatorOfTask
              << ", canRunOnGPU[" << operatorIndex
              << "] = " << thisOpProps.canRunOnGPU << ", acceptsGpuInput["
              << operatorIndex << "] = " << thisOpProps.acceptsGpuInput
              << ", producesGpuOutput[" << operatorIndex
              << "] = " << thisOpProps.producesGpuOutput
              << ", planNode = " << bool(planNode);
    }
    if (!allowCpuFallback) {
      // condition is if GPU replacement success or if CPU operators itself is
      // GPU compatible. or if specific CPU operator is allowed even when
      // fallback is disabled.
      VELOX_CHECK(!isPureCpuOperator, "Replacement with cuDF operator failed");
    } else if (isPureCpuOperator) {
      LOG(WARNING)
          << "Replacement with cuDF operator failed. Falling back to CPU execution";
      LOG(WARNING) << "Replacement Failed Operator: " << oper->toString();
      auto planNode = getPlanNode(oper->planNodeId());
      LOG(WARNING) << "Replacement Failed PlanNode: "
                   << planNode->toString(true, false);
    }

    if (not replaceOp.empty()) {
      // ReplaceOp, to-velox.
      operatorsOffset += replaceOp.size() - 1 + keepOperator;
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_,
          replacingOperatorIndex + keepOperator,
          replacingOperatorIndex + 1,
          std::move(replaceOp));
      replacementsMade = true;
    }
  }

  if (debugEnabled) {
    // Print before/after together for easy comparison.
    LOG(INFO) << "Operators " << "before adapting for cuDF"
              << ": count [" << beforeOperators.size() << "]";
    for (const auto& [id, desc] : beforeOperators) {
      LOG(INFO) << "  Operator: ID " << id << ": " << desc;
    }
    LOG(INFO) << "allowCpuFallback = " << allowCpuFallback;

    operators = driver_.operators();
    LOG(INFO) << "Operators " << "after adapting for cuDF"
              << ": count [" << operators.size() << "]";
    for (const auto& op : operators) {
      LOG(INFO) << "  Operator: ID " << op->operatorId() << ": "
                << op->toString();
    }
  }

  VLOG(3) << "- CompileState::compile";
  return replacementsMade;
}

struct CudfDriverAdapter {
  CudfDriverAdapter(bool allowCpuFallback)
      : allowCpuFallback_{allowCpuFallback} {}

  // Call operator needed by DriverAdapter
  bool operator()(const exec::DriverFactory& factory, exec::Driver& driver) {
    if (!driver.driverCtx()->queryConfig().get<bool>(
            CudfConfig::kCudfEnabled, CudfConfig::getInstance().enabled) &&
        allowCpuFallback_) {
      return false;
    }
    auto state = CompileState(factory, driver);
    auto res = state.compile(allowCpuFallback_);
    return res;
  }

 private:
  bool allowCpuFallback_;
};

static bool isCudfRegistered = false;

bool cudfIsRegistered() {
  return isCudfRegistered;
}

void registerCudf() {
  if (cudfIsRegistered()) {
    return;
  }

  // Register operator adapters
  registerAllOperatorAdapters();

  auto prefix = CudfConfig::getInstance().functionNamePrefix;
  registerBuiltinFunctions(prefix);
  registerStepAwareBuiltinAggregationFunctions(prefix);

  CUDF_FUNC_RANGE();
  cudaFree(nullptr); // Initialize CUDA context at startup

  const std::string mrMode = CudfConfig::getInstance().memoryResource;
  auto mr = cudf_velox::createMemoryResource(
      mrMode, CudfConfig::getInstance().memoryPercent);
  mr = maybeWrapSharedMemoryResource(mr);
  cudf::set_current_device_resource(mr.get());
  // rmm::device_buffer and other ref-based allocations default to
  // rmm::mr::get_current_device_resource_ref(), which is a SEPARATE slot from
  // the pointer-based current resource set by cudf::set_current_device_resource
  // above. Without also setting the ref-based slot, device_buffer allocations
  // (UCX receive buffers, operator outputs) bypass this pool and fall back to
  // the default non-pool cuda_memory_resource, growing device memory unbounded
  // across a sustained session until the device OOMs. Point the ref-based
  // current resource at the same pool so all allocations share its budget.
  cudf::set_current_device_resource_ref(mr.get());
  mr_ = mr;

  const auto& outputMrMode = CudfConfig::getInstance().outputMemoryResource;
  if (!outputMrMode.empty() && outputMrMode != mrMode) {
    output_mr_ = cudf_velox::createMemoryResource(
        outputMrMode, CudfConfig::getInstance().memoryPercent);
    output_mr_ = maybeWrapSharedMemoryResource(output_mr_);
  } else {
    output_mr_ = mr_;
  }
  output_mr_ = maybeWrapOutputMemoryResource(output_mr_);

  exec::Operator::registerOperator(
      std::make_unique<CudfHashJoinBridgeTranslator>());
  CudfDriverAdapter cda{CudfConfig::getInstance().allowCpuFallback};
  exec::DriverAdapter cudfAdapter{kCudfAdapterName, {}, cda};
  exec::DriverFactory::registerAdapter(cudfAdapter);

  if (CudfConfig::getInstance().astExpressionEnabled) {
    registerAstEvaluator(CudfConfig::getInstance().astExpressionPriority);
  }

  if (CudfConfig::getInstance().jitExpressionEnabled) {
    registerJitEvaluator(CudfConfig::getInstance().jitExpressionPriority);
  }

  isCudfRegistered = true;
}

void unregisterCudf() {
  output_mr_ = nullptr;
  mr_ = nullptr;
  exec::DriverFactory::adapters.erase(
      std::remove_if(
          exec::DriverFactory::adapters.begin(),
          exec::DriverFactory::adapters.end(),
          [](const exec::DriverAdapter& adapter) {
            return adapter.label == kCudfAdapterName;
          }),
      exec::DriverFactory::adapters.end());

  isCudfRegistered = false;
}

CudfConfig& CudfConfig::getInstance() {
  static CudfConfig instance;
  return instance;
}

void CudfConfig::initialize(
    std::unordered_map<std::string, std::string>&& config) {
  if (config.find(kCudfEnabled) != config.end()) {
    enabled = folly::to<bool>(config[kCudfEnabled]);
  }
  if (config.find(kCudfDebugEnabled) != config.end()) {
    debugEnabled = folly::to<bool>(config[kCudfDebugEnabled]);
  }
  if (config.find(kCudfMemoryResource) != config.end()) {
    memoryResource = config[kCudfMemoryResource];
  }
  if (config.find(kCudfMemoryPercent) != config.end()) {
    memoryPercent = folly::to<int32_t>(config[kCudfMemoryPercent]);
  }
  if (config.find(kCudfOutputMr) != config.end()) {
    outputMemoryResource = config[kCudfOutputMr];
  }
  if (config.find(kCudfBatchSizeMinThreshold) != config.end()) {
    batchSizeMinThreshold =
        folly::to<int32_t>(config[kCudfBatchSizeMinThreshold]);
  }
  if (config.find(kCudfBatchSizeMaxThreshold) != config.end()) {
    batchSizeMaxThreshold =
        folly::to<int32_t>(config[kCudfBatchSizeMaxThreshold]);
  }
  if (config.find(kCudfConcatOptimizationEnabled) != config.end()) {
    concatOptimizationEnabled =
        folly::to<bool>(config[kCudfConcatOptimizationEnabled]);
  }
  if (config.find(kCudfFunctionNamePrefix) != config.end()) {
    functionNamePrefix = config[kCudfFunctionNamePrefix];
  }
  if (config.find(kCudfAstExpressionEnabled) != config.end()) {
    astExpressionEnabled = folly::to<bool>(config[kCudfAstExpressionEnabled]);
  }
  if (config.find(kCudfJitExpressionEnabled) != config.end()) {
    jitExpressionEnabled = folly::to<bool>(config[kCudfJitExpressionEnabled]);
  }
  if (config.find(kCudfAstExpressionPriority) != config.end()) {
    astExpressionPriority =
        folly::to<int32_t>(config[kCudfAstExpressionPriority]);
  }
  if (config.find(kCudfAllowCpuFallback) != config.end()) {
    allowCpuFallback = folly::to<bool>(config[kCudfAllowCpuFallback]);
  }
  if (config.find(kCudfLogFallback) != config.end()) {
    logFallback = folly::to<bool>(config[kCudfLogFallback]);
  }

  if (config.find(kCudfTopNBatchSize) != config.end()) {
    topNBatchSize = folly::to<int32_t>(config[kCudfTopNBatchSize]);
  }
  if (config.find(kCudfFunctionEngine) != config.end()) {
    functionEngine = config[kCudfFunctionEngine];
  }

  if (config.find(kUcxExchange) != config.end()) {
    exchange = folly::to<bool>(config[kUcxExchange]);
  }
  if (config.find(kUcxIntraNodeExchange) != config.end()) {
    intraNodeExchange = folly::to<bool>(config[kUcxIntraNodeExchange]);
  }
  if (config.find(kUcxxErrorHandling) != config.end()) {
    ucxxErrorHandling = folly::to<bool>(config[kUcxxErrorHandling]);
  }
  if (config.find(kUcxxBlockingPolling) != config.end()) {
    ucxxBlockingPolling = folly::to<bool>(config[kUcxxBlockingPolling]);
  }
  if (config.find(kUcxExchangeLogLevel) != config.end()) {
    exchangeLogLevel = folly::to<int32_t>(config[kUcxExchangeLogLevel]);
  }
}

} // namespace facebook::velox::cudf_velox
