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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <string_view>

namespace facebook::velox::cudf_velox {

// Implements concat(strings, literal, cast(integers as varchar)) without
// materializing the intermediate integer strings or a repeated literal
// column. Null inputs have the same empty-string replacement semantics as the
// existing Spark concat evaluator.
std::unique_ptr<cudf::column> concatStringLiteralIntegral(
    const cudf::column_view& strings,
    std::string_view literal,
    const cudf::column_view& integers,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

} // namespace facebook::velox::cudf_velox
