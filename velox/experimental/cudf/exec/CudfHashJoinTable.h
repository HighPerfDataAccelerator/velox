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

#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/hash_join.hpp>

#include <memory>

namespace facebook::velox::cudf_velox {

// Owns one build chunk's index. The caller retains the underlying key columns.
class CudfHashJoinTable {
 public:
  CudfHashJoinTable(
      cudf::table_view keys,
      bool tryDistinct,
      double loadFactor,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr);

  using JoinIndices = std::pair<
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;

  JoinIndices innerJoin(
      cudf::table_view probe,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref mr) const;

  bool isDistinct() const {
    return distinct_ != nullptr;
  }

 private:
  std::unique_ptr<cudf::hash_join> general_;
  std::unique_ptr<cudf::distinct_hash_join> distinct_;
};

} // namespace facebook::velox::cudf_velox
