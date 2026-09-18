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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfHashJoinTable.h"

#include <cudf/reduction/distinct_count.hpp>
#include <cudf/version_config.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {

CudfHashJoinTable::CudfHashJoinTable(
    cudf::table_view keys,
    bool tryDistinct,
    double loadFactor,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) {
#if CUDF_VERSION_MAJOR > 26 || \
    (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 8)
  const bool eligible = tryDistinct && keys.num_columns() > 0 &&
      keys.num_rows() > 0 &&
      std::all_of(keys.begin(), keys.end(), [](auto const& key) {
                          if (key.has_nulls()) {
                            return false;
                          }
                          switch (key.type().id()) {
                            case cudf::type_id::INT8:
                            case cudf::type_id::INT16:
                            case cudf::type_id::INT32:
                            case cudf::type_id::INT64:
                              return true;
                            default:
                              return false;
                          }
                        });
  // Count whole key rows, not individual columns. The temporary index is
  // released before constructing the retained index.
  if (eligible &&
      cudf::distinct_count(keys, cudf::null_equality::UNEQUAL, stream) ==
          keys.num_rows()) {
    distinct_ = std::make_unique<cudf::distinct_hash_join>(
        keys, cudf::null_equality::UNEQUAL, loadFactor, stream, mr);
    return;
  }
  general_ = std::make_unique<cudf::hash_join>(
      keys,
      cudf::nullable_join::YES,
      cudf::null_equality::UNEQUAL,
      loadFactor,
      stream,
      mr);
#else
  general_ = std::make_unique<cudf::hash_join>(
      keys, cudf::null_equality::UNEQUAL, stream, mr);
#endif
}

CudfHashJoinTable::JoinIndices CudfHashJoinTable::innerJoin(
    cudf::table_view probe,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const {
  if (distinct_) {
    return distinct_->inner_join(probe, stream, mr);
  }
  return general_->inner_join(probe, std::nullopt, stream, mr);
}

} // namespace facebook::velox::cudf_velox
