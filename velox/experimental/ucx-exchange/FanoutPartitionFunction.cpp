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
#include "velox/experimental/ucx-exchange/FanoutPartitionFunction.h"

#include <sstream>

namespace facebook::velox::ucx_exchange {
namespace {

class UcxOnlyFanoutPartitionFunction final : public core::PartitionFunction {
 public:
  std::optional<uint32_t> partition(const RowVector&, std::vector<uint32_t>&)
      override {
    VELOX_FAIL(
        "FANOUT is supported only by UcxPartitionedOutput; refusing to "
        "materialize duplicate partitions on the CPU path");
  }
};

} // namespace

FanoutPartitionFunctionSpec::FanoutPartitionFunctionSpec(
    RowTypePtr inputType,
    std::vector<column_index_t> keyChannels,
    int32_t basePartitions,
    int32_t fanout)
    : inputType_(std::move(inputType)),
      keyChannels_(std::move(keyChannels)),
      basePartitions_(basePartitions),
      fanout_(fanout) {
  VELOX_CHECK_GT(basePartitions_, 0);
  VELOX_CHECK_GT(fanout_, 1);
}

std::unique_ptr<core::PartitionFunction> FanoutPartitionFunctionSpec::create(
    int,
    bool) const {
  return std::make_unique<UcxOnlyFanoutPartitionFunction>();
}

std::string FanoutPartitionFunctionSpec::toString() const {
  std::ostringstream keys;
  for (size_t i = 0; i < keyChannels_.size(); ++i) {
    if (i > 0) {
      keys << ", ";
    }
    keys << inputType_->nameOf(keyChannels_[i]);
  }
  return keyChannels_.empty()
      ? fmt::format(
            "FANOUT_ROUND_ROBIN(partitions={}, fanout={})",
            basePartitions_,
            fanout_)
      : fmt::format(
            "FANOUT_HASH({}; partitions={}, fanout={})",
            keys.str(),
            basePartitions_,
            fanout_);
}

folly::dynamic FanoutPartitionFunctionSpec::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "FanoutPartitionFunctionSpec";
  obj["inputType"] = inputType_->serialize();
  obj["keyChannels"] = ISerializable::serialize(keyChannels_);
  obj["basePartitions"] = basePartitions_;
  obj["fanout"] = fanout_;
  return obj;
}

core::PartitionFunctionSpecPtr FanoutPartitionFunctionSpec::deserialize(
    const folly::dynamic& obj,
    void* context) {
  return std::make_shared<FanoutPartitionFunctionSpec>(
      ISerializable::deserialize<RowType>(obj["inputType"]),
      ISerializable::deserialize<std::vector<column_index_t>>(
          obj["keyChannels"], context),
      obj["basePartitions"].asInt(),
      obj["fanout"].asInt());
}

} // namespace facebook::velox::ucx_exchange
