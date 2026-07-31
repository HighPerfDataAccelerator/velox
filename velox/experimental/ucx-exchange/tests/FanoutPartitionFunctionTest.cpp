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

#include <gtest/gtest.h>

namespace facebook::velox::ucx_exchange {
namespace {

TEST(FanoutPartitionFunctionSpecTest, contractAndSerde) {
  auto rowType = ROW({"key", "value"}, {BIGINT(), VARCHAR()});
  auto spec = std::make_shared<FanoutPartitionFunctionSpec>(
      rowType, std::vector<column_index_t>{0}, 8, 3);

  EXPECT_EQ(spec->basePartitions(), 8);
  EXPECT_EQ(spec->fanout(), 3);
  EXPECT_EQ(spec->keyChannels(), std::vector<column_index_t>{0});
  EXPECT_EQ(spec->toString(), "FANOUT_HASH(key; partitions=8, fanout=3)");

  auto restored =
      FanoutPartitionFunctionSpec::deserialize(spec->serialize(), nullptr);
  auto* fanout =
      dynamic_cast<const FanoutPartitionFunctionSpec*>(restored.get());
  ASSERT_NE(fanout, nullptr);
  EXPECT_EQ(fanout->basePartitions(), 8);
  EXPECT_EQ(fanout->fanout(), 3);
  EXPECT_EQ(fanout->keyChannels(), std::vector<column_index_t>{0});
}

TEST(FanoutPartitionFunctionSpecTest, ordinaryCpuPathFails) {
  auto spec = std::make_shared<FanoutPartitionFunctionSpec>(
      ROW({"key"}, {BIGINT()}), std::vector<column_index_t>{0}, 4, 2);
  auto function = spec->create(8, false);
  std::vector<uint32_t> partitions;
  auto input = RowVector(nullptr, ROW({"key"}, {BIGINT()}), nullptr, 0, {});
  EXPECT_THROW(function->partition(input, partitions), VeloxRuntimeError);
}

TEST(FanoutPartitionFunctionSpecTest, roundRobinContractAndSerde) {
  auto rowType = ROW({"value"}, {VARCHAR()});
  auto spec = std::make_shared<FanoutPartitionFunctionSpec>(
      rowType, std::vector<column_index_t>{}, 4, 16);

  EXPECT_TRUE(spec->keyChannels().empty());
  EXPECT_EQ(spec->toString(), "FANOUT_ROUND_ROBIN(partitions=4, fanout=16)");

  auto restored =
      FanoutPartitionFunctionSpec::deserialize(spec->serialize(), nullptr);
  auto* fanout =
      dynamic_cast<const FanoutPartitionFunctionSpec*>(restored.get());
  ASSERT_NE(fanout, nullptr);
  EXPECT_TRUE(fanout->keyChannels().empty());
  EXPECT_EQ(fanout->basePartitions(), 4);
  EXPECT_EQ(fanout->fanout(), 16);
}

} // namespace
} // namespace facebook::velox::ucx_exchange
