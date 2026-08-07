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

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

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

TEST(RangePartitionFunctionSpecTest, repeatedBoundsSplitOnlyEqualKeys) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  const auto makeColumn = [&](const std::vector<int32_t>& values) {
    auto column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        values.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    EXPECT_EQ(
        cudaMemcpyAsync(
            column->mutable_view().data<int32_t>(),
            values.data(),
            values.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream.value()),
        cudaSuccess);
    return column;
  };
  auto boundaries = makeColumn({10, 10, 10, 20});
  auto keys = makeColumn({9, 10, 10, 10, 10, 10, 11, 20, 20, 21});
  const cudf::table_view boundaryTable{{boundaries->view()}};
  const cudf::table_view keyTable{{keys->view()}};
  const std::vector<cudf::order> orders{cudf::order::ASCENDING};
  const std::vector<cudf::null_order> nullOrders{cudf::null_order::BEFORE};

  auto ids = makeRangePartitionIds(
      boundaryTable,
      keyTable,
      orders,
      nullOrders,
      true,
      0,
      stream,
      mr);
  std::vector<int32_t> actual(ids->size());
  EXPECT_EQ(
      cudaMemcpyAsync(
          actual.data(),
          ids->view().data<int32_t>(),
          actual.size() * sizeof(int32_t),
          cudaMemcpyDeviceToHost,
          stream.value()),
      cudaSuccess);
  stream.synchronize();
  EXPECT_EQ(actual, std::vector<int32_t>({0, 1, 2, 0, 1, 2, 3, 3, 3, 4}));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
