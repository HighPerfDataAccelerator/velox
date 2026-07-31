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

namespace facebook::velox::ucx_exchange {

/// Partitions a producer once, then publishes each packed partition to multiple
/// independent UCX destination ranges. Non-empty key channels select hash
/// partitioning; empty key channels select round-robin partitioning. The
/// physical output node has basePartitions * fanout destinations, while only
/// basePartitions are materialized.
class FanoutPartitionFunctionSpec final : public core::PartitionFunctionSpec {
 public:
  FanoutPartitionFunctionSpec(
      RowTypePtr inputType,
      std::vector<column_index_t> keyChannels,
      int32_t basePartitions,
      int32_t fanout);

  std::unique_ptr<core::PartitionFunction> create(
      int numPartitions,
      bool localExchange) const override;

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static core::PartitionFunctionSpecPtr deserialize(
      const folly::dynamic& obj,
      void* context);

  const std::vector<column_index_t>& keyChannels() const {
    return keyChannels_;
  }

  int32_t basePartitions() const {
    return basePartitions_;
  }

  int32_t fanout() const {
    return fanout_;
  }

 private:
  const RowTypePtr inputType_;
  const std::vector<column_index_t> keyChannels_;
  const int32_t basePartitions_;
  const int32_t fanout_;
};

} // namespace facebook::velox::ucx_exchange
