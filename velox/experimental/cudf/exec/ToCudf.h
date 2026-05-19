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

#include "velox/experimental/ucx-exchange/UcxExchangeClient.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace facebook::velox::cudf_velox {

struct TaskPipelineKey {
  std::string taskId;
  int pipelineId;

  TaskPipelineKey(const std::string& taskId, int pipelineId)
      : taskId(taskId), pipelineId(pipelineId) {}

  bool operator==(const TaskPipelineKey& other) const {
    return taskId == other.taskId && pipelineId == other.pipelineId;
  }

  struct Hash {
    std::size_t operator()(const TaskPipelineKey& key) const {
      const auto taskHash = std::hash<std::string>{}(key.taskId);
      const auto pipelineHash = std::hash<int>{}(key.pipelineId);
      return taskHash ^ (pipelineHash << 1);
    }
  };
};

using UcxExchangeClientMap = std::unordered_map<
    TaskPipelineKey,
    std::weak_ptr<ucx_exchange::UcxExchangeClient>,
    TaskPipelineKey::Hash>;

UcxExchangeClientMap& getUcxExchangeClientMap();

class CompileState {
 public:
  CompileState(const exec::DriverFactory& driverFactory, exec::Driver& driver)
      : driverFactory_(driverFactory), driver_(driver) {}

  exec::Driver& driver() {
    return driver_;
  }

  // Get plan node by id lookup.
  core::PlanNodePtr getPlanNode(const core::PlanNodeId& id) const;

  // Replaces sequences of Operators in the Driver given at construction with
  // cuDF equivalents. Returns true if the Driver was changed.
  bool compile(bool allow_cpu_fallback);

  const exec::DriverFactory& driverFactory_;
  exec::Driver& driver_;
};

/// Registers adapter to add cuDF operators to Drivers.
void registerCudf();
void unregisterCudf();

/// Returns true if cuDF is registered.
bool cudfIsRegistered();

} // namespace facebook::velox::cudf_velox
