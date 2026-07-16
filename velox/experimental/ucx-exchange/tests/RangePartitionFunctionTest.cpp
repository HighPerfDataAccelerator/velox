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

#include <gtest/gtest.h>

#include "velox/experimental/ucx-exchange/RangePartitionFunction.h"

namespace facebook::velox::ucx_exchange {
namespace {

constexpr auto kBounds = R"json({
  "version": 1,
  "keys": [{"sparkType":"int","ascending":true,"nullsFirst":true}],
  "bounds": [[{"isNull":false,"value":10}]]
})json";

TEST(RangePartitionFunctionSpecTest, explicitPidIdentityAndSerde) {
  auto rowType = ROW({"k", "payload"}, {INTEGER(), VARCHAR()});
  auto spec = std::make_shared<RangePartitionFunctionSpec>(
      rowType, std::vector<column_index_t>{0}, kBounds);

  EXPECT_EQ(spec->toString(), "RANGE_PID(k)");
  EXPECT_EQ(spec->keyChannels(), std::vector<column_index_t>({0}));
  EXPECT_EQ(spec->boundsJson(), kBounds);

  auto copy =
      RangePartitionFunctionSpec::deserialize(spec->serialize(), nullptr);
  EXPECT_EQ(copy->toString(), "RANGE_PID(k)");
}

TEST(RangePartitionFunctionSpecTest, ordinaryCpuPathFailsInsteadOfHashing) {
  auto spec = std::make_shared<RangePartitionFunctionSpec>(
      ROW({"k"}, {INTEGER()}), std::vector<column_index_t>{0}, kBounds);
  auto function = spec->create(2, false);
  std::vector<uint32_t> partitions;
  auto input = RowVector(nullptr, ROW({"k"}, {INTEGER()}), nullptr, 0, {});

  EXPECT_THROW(function->partition(input, partitions), VeloxRuntimeError);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
