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

#include "velox/experimental/cudf/exec/CudfHashJoinTable.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox::test {
namespace {

template <typename T>
std::unique_ptr<cudf::column> column(
    std::vector<std::optional<T>> const& values,
    cuda::stream_ref stream) {
  auto mr = cudf::get_current_device_resource_ref();
  auto result = cudf::make_numeric_column(
      cudf::data_type{cudf::type_to_id<T>()},
      values.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  std::vector<T> data;
  std::vector<cudf::bitmask_type> mask(
      cudf::bitmask_allocation_size_bytes(values.size()) /
          sizeof(cudf::bitmask_type),
      0);
  cudf::size_type nulls = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    data.push_back(values[i].value_or(T{}));
    if (values[i]) {
      mask[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++nulls;
    }
  }
  if (!data.empty()) {
    EXPECT_EQ(
        cudaMemcpyAsync(
            result->mutable_view().template data<T>(),
            data.data(),
            data.size() * sizeof(T),
            cudaMemcpyHostToDevice,
            stream.get()),
        cudaSuccess);
  }
  if (nulls) {
    result->set_null_mask(
        rmm::device_buffer(
            mask.data(), mask.size() * sizeof(mask[0]), stream, mr),
        nulls);
  }
  stream.sync();
  return result;
}

using Pairs = std::vector<std::pair<cudf::size_type, cudf::size_type>>;
Pairs probe(
    CudfHashJoinTable const& join,
    cudf::table_view keys,
    cuda::stream_ref stream) {
  auto indices =
      join.innerJoin(keys, stream, cudf::get_current_device_resource_ref());
  std::vector<cudf::size_type> left(indices.first->size());
  std::vector<cudf::size_type> right(indices.second->size());
  EXPECT_EQ(left.size(), right.size());
  if (!left.empty()) {
    EXPECT_EQ(
        cudaMemcpyAsync(
            left.data(),
            indices.first->data(),
            left.size() * sizeof(left[0]),
            cudaMemcpyDeviceToHost,
            stream.get()),
        cudaSuccess);
    EXPECT_EQ(
        cudaMemcpyAsync(
            right.data(),
            indices.second->data(),
            right.size() * sizeof(right[0]),
            cudaMemcpyDeviceToHost,
            stream.get()),
        cudaSuccess);
  }
  stream.sync();
  Pairs result;
  for (size_t i = 0; i < left.size(); ++i) {
    result.emplace_back(left[i], right[i]);
  }
  std::sort(result.begin(), result.end());
  return result;
}

TEST(CudfHashJoinTableTest, uniqueAndDisabled) {
  rmm::cuda_stream stream;
  auto build = column<int64_t>({3, 1, 2}, stream);
  auto keys = column<int64_t>({2, 2, 4, std::nullopt, 1}, stream);
  for (bool enabled : {false, true}) {
    CudfHashJoinTable join(
        cudf::table_view{{build->view()}},
        enabled,
        .5,
        stream,
        cudf::get_current_device_resource_ref());
    EXPECT_EQ(join.isDistinct(), enabled);
    EXPECT_EQ(
        probe(join, cudf::table_view{{keys->view()}}, stream),
        (Pairs{{0, 2}, {1, 2}, {4, 1}}));
  }
}

TEST(CudfHashJoinTableTest, duplicateAndNullFallback) {
  rmm::cuda_stream stream;
  auto keys = column<int32_t>({1, 2, std::nullopt, 4}, stream);
  for (auto const& values : std::vector<std::vector<std::optional<int32_t>>>{
           {1, 2, 2}, {1, std::nullopt, 2}}) {
    auto build = column<int32_t>(values, stream);
    CudfHashJoinTable join(
        cudf::table_view{{build->view()}},
        true,
        .5,
        stream,
        cudf::get_current_device_resource_ref());
    EXPECT_FALSE(join.isDistinct());
    Pairs expected =
        values[1] ? Pairs{{0, 0}, {1, 1}, {1, 2}} : Pairs{{0, 0}, {1, 2}};
    EXPECT_EQ(probe(join, cudf::table_view{{keys->view()}}, stream), expected);
  }
}

TEST(CudfHashJoinTableTest, compositeKeys) {
  rmm::cuda_stream stream;
  auto first = column<int64_t>({1, 1, 2}, stream);
  auto second = column<int32_t>({1, 2, 1}, stream);
  auto probeFirst = column<int64_t>({1, 2, 1, 3}, stream);
  auto probeSecond = column<int32_t>({2, 2, 1, 1}, stream);
  CudfHashJoinTable join(
      cudf::table_view{{first->view(), second->view()}},
      true,
      .5,
      stream,
      cudf::get_current_device_resource_ref());
  EXPECT_TRUE(join.isDistinct());
  EXPECT_EQ(
      probe(
          join,
          cudf::table_view{{probeFirst->view(), probeSecond->view()}},
          stream),
      (Pairs{{0, 1}, {2, 0}}));
  auto duplicate = column<int32_t>({1, 1, 2}, stream);
  CudfHashJoinTable fallback(
      cudf::table_view{{first->view(), duplicate->view()}},
      true,
      .5,
      stream,
      cudf::get_current_device_resource_ref());
  EXPECT_FALSE(fallback.isDistinct());
  EXPECT_EQ(
      probe(
          fallback,
          cudf::table_view{{probeFirst->view(), probeSecond->view()}},
          stream),
      (Pairs{{1, 2}, {2, 0}, {2, 1}}));
}

TEST(CudfHashJoinTableTest, floatingPointAndEmptyFallback) {
  rmm::cuda_stream stream;
  auto build = column<double>({1., 2.}, stream);
  CudfHashJoinTable floating(
      cudf::table_view{{build->view()}},
      true,
      .5,
      stream,
      cudf::get_current_device_resource_ref());
  EXPECT_FALSE(floating.isDistinct());
  EXPECT_EQ(
      probe(floating, cudf::table_view{{build->view()}}, stream),
      (Pairs{{0, 0}, {1, 1}}));
  auto empty = column<int64_t>({}, stream);
  CudfHashJoinTable emptyJoin(
      cudf::table_view{{empty->view()}},
      true,
      .5,
      stream,
      cudf::get_current_device_resource_ref());
  EXPECT_FALSE(emptyJoin.isDistinct());
  EXPECT_TRUE(
      probe(emptyJoin, cudf::table_view{{empty->view()}}, stream).empty());
}

TEST(CudfHashJoinTableTest, repeatedKeysAcrossChunks) {
  rmm::cuda_stream stream;
  auto first = column<int64_t>({1, 2}, stream);
  auto second = column<int64_t>({2, 3}, stream);
  auto keys = column<int64_t>({2, 3}, stream);
  CudfHashJoinTable a(
      cudf::table_view{{first->view()}},
      true,
      .5,
      stream,
      cudf::get_current_device_resource_ref());
  CudfHashJoinTable b(
      cudf::table_view{{second->view()}},
      true,
      .5,
      stream,
      cudf::get_current_device_resource_ref());
  EXPECT_TRUE(a.isDistinct());
  EXPECT_TRUE(b.isDistinct());
  EXPECT_EQ(
      probe(a, cudf::table_view{{keys->view()}}, stream), (Pairs{{0, 1}}));
  EXPECT_EQ(
      probe(b, cudf::table_view{{keys->view()}}, stream),
      (Pairs{{0, 0}, {1, 1}}));
}

} // namespace
} // namespace facebook::velox::cudf_velox::test
