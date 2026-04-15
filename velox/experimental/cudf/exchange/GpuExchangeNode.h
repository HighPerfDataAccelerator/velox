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

#include "velox/core/PlanNode.h"

namespace facebook::velox::cudf_velox {

/// Plan node for a GPU exchange operator. This is a leaf node (no sources)
/// that receives data from an ExchangeClient. Similar to core::ExchangeNode
/// but designed for GPU data paths where pages contain CudfVectors.
class GpuExchangeNode : public core::PlanNode {
 public:
  GpuExchangeNode(const core::PlanNodeId& id, RowTypePtr outputType)
      : PlanNode(id), outputType_(std::move(outputType)) {}

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    static const std::vector<core::PlanNodePtr> kEmpty;
    return kEmpty;
  }

  bool requiresExchangeClient() const override {
    return true;
  }

  bool requiresSplits() const override {
    return true;
  }

  std::string_view name() const override {
    return "GpuExchange";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}

  const RowTypePtr outputType_;
};

} // namespace facebook::velox::cudf_velox
