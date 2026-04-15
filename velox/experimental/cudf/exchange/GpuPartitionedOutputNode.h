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

/// Plan node for GPU-accelerated partitioned output. Similar to
/// core::PartitionedOutputNode but creates a GpuPartitionedOutput operator
/// that partitions CudfVector data on GPU and publishes GpuSerializedPages
/// to OutputBufferManager without D2H copy.
class GpuPartitionedOutputNode : public core::PlanNode {
 public:
  enum class Kind {
    kPartitioned,
    kBroadcast,
    kArbitrary,
  };

  GpuPartitionedOutputNode(
      const core::PlanNodeId& id,
      Kind kind,
      const std::vector<core::TypedExprPtr>& keys,
      int numPartitions,
      core::PartitionFunctionSpecPtr partitionFunctionSpec,
      RowTypePtr outputType,
      core::PlanNodePtr source)
      : PlanNode(id),
        kind_(kind),
        keys_(keys),
        numPartitions_(numPartitions),
        partitionFunctionSpec_(std::move(partitionFunctionSpec)),
        outputType_(std::move(outputType)),
        sources_{std::move(source)} {
    VELOX_USER_CHECK_GT(numPartitions_, 0, "numPartitions must be positive");
    if (kind_ != Kind::kPartitioned) {
      VELOX_USER_CHECK(
          keys_.empty(),
          "Non-partitioned output must not have partition keys");
    }
  }

  /// Factory for broadcast output.
  static std::shared_ptr<GpuPartitionedOutputNode> broadcast(
      const core::PlanNodeId& id,
      int numPartitions,
      RowTypePtr outputType,
      core::PlanNodePtr source) {
    return std::make_shared<GpuPartitionedOutputNode>(
        id,
        Kind::kBroadcast,
        std::vector<core::TypedExprPtr>{},
        numPartitions,
        nullptr,
        std::move(outputType),
        std::move(source));
  }

  /// Factory for arbitrary output.
  static std::shared_ptr<GpuPartitionedOutputNode> arbitrary(
      const core::PlanNodeId& id,
      RowTypePtr outputType,
      core::PlanNodePtr source) {
    return std::make_shared<GpuPartitionedOutputNode>(
        id,
        Kind::kArbitrary,
        std::vector<core::TypedExprPtr>{},
        1,
        nullptr,
        std::move(outputType),
        std::move(source));
  }

  const RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }

  const RowTypePtr& inputType() const {
    return sources_[0]->outputType();
  }

  const std::vector<core::TypedExprPtr>& keys() const {
    return keys_;
  }

  int numPartitions() const {
    return numPartitions_;
  }

  Kind kind() const {
    return kind_;
  }

  bool isPartitioned() const {
    return kind_ == Kind::kPartitioned;
  }

  bool isBroadcast() const {
    return kind_ == Kind::kBroadcast;
  }

  bool isArbitrary() const {
    return kind_ == Kind::kArbitrary;
  }

  const core::PartitionFunctionSpecPtr& partitionFunctionSpec() const {
    return partitionFunctionSpec_;
  }

  std::string_view name() const override {
    return "GpuPartitionedOutput";
  }

 private:
  void addDetails(std::stringstream& stream) const override {
    stream << "Kind: ";
    switch (kind_) {
      case Kind::kPartitioned:
        stream << "PARTITIONED";
        break;
      case Kind::kBroadcast:
        stream << "BROADCAST";
        break;
      case Kind::kArbitrary:
        stream << "ARBITRARY";
        break;
    }
    stream << ", numPartitions: " << numPartitions_;
  }

  const Kind kind_;
  const std::vector<core::TypedExprPtr> keys_;
  const int numPartitions_;
  const core::PartitionFunctionSpecPtr partitionFunctionSpec_;
  const RowTypePtr outputType_;
  const std::vector<core::PlanNodePtr> sources_;
};

} // namespace facebook::velox::cudf_velox
