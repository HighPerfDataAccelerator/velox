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
#include "velox/experimental/ucx-exchange/RangePartitionFunction.h"

#include <sstream>

namespace facebook::velox::ucx_exchange {
namespace {

class UcxOnlyRangePartitionFunction final : public core::PartitionFunction {
 public:
  std::optional<uint32_t> partition(
      const RowVector&,
      std::vector<uint32_t>&) override {
    VELOX_FAIL(
        "RANGE_PID is supported only by UcxPartitionedOutput; refusing to "
        "silently substitute hash or round-robin partitioning");
  }
};

} // namespace

std::unique_ptr<core::PartitionFunction> RangePartitionFunctionSpec::create(
    int,
    bool) const {
  return std::make_unique<UcxOnlyRangePartitionFunction>();
}

std::string RangePartitionFunctionSpec::toString() const {
  std::ostringstream keys;
  for (size_t i = 0; i < keyChannels_.size(); ++i) {
    if (i > 0) {
      keys << ", ";
    }
    keys << inputType_->nameOf(keyChannels_[i]);
  }
  return fmt::format("RANGE_PID({})", keys.str());
}

folly::dynamic RangePartitionFunctionSpec::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "RangePartitionFunctionSpec";
  obj["inputType"] = inputType_->serialize();
  obj["keyChannels"] = ISerializable::serialize(keyChannels_);
  obj["boundsJson"] = boundsJson_;
  return obj;
}

core::PartitionFunctionSpecPtr RangePartitionFunctionSpec::deserialize(
    const folly::dynamic& obj,
    void* context) {
  return std::make_shared<RangePartitionFunctionSpec>(
      ISerializable::deserialize<RowType>(obj["inputType"]),
      ISerializable::deserialize<std::vector<column_index_t>>(
          obj["keyChannels"], context),
      obj["boundsJson"].asString());
}

} // namespace facebook::velox::ucx_exchange
