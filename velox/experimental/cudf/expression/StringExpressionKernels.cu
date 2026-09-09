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

#include "velox/experimental/cudf/expression/StringExpressionKernels.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>

namespace facebook::velox::cudf_velox {
namespace {

template <typename T>
__device__ cuda::std::make_unsigned_t<T> unsignedMagnitude(T value) {
  using U = cuda::std::make_unsigned_t<T>;
  const auto bits = static_cast<U>(value);
  return value < 0 ? U{0} - bits : bits;
}

template <typename T>
__device__ cudf::size_type formattedIntegerSize(T value) {
  auto magnitude = unsignedMagnitude(value);
  cudf::size_type digits = 1;
  while (magnitude >= 10) {
    magnitude /= 10;
    ++digits;
  }
  return digits + (value < 0 ? 1 : 0);
}

template <typename T>
__device__ void writeInteger(T value, char* output, cudf::size_type size) {
  auto magnitude = unsignedMagnitude(value);
  auto cursor = size;
  do {
    output[--cursor] = static_cast<char>('0' + magnitude % 10);
    magnitude /= 10;
  } while (magnitude != 0);
  if (value < 0) {
    output[0] = '-';
  }
}

template <typename T>
struct ConcatStringLiteralIntegralFunctor {
  cudf::column_device_view strings;
  cudf::string_view literal;
  cudf::column_device_view integers;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets{};

  __device__ void operator()(cudf::size_type index) const {
    const bool stringValid = !strings.is_null(index);
    const bool integerValid = !integers.is_null(index);
    const auto stringValue = stringValid
        ? strings.element<cudf::string_view>(index)
        : cudf::string_view{};
    const auto integerValue = integerValid ? integers.element<T>(index) : T{0};
    const auto integerBytes =
        integerValid ? formattedIntegerSize(integerValue) : 0;
    const auto outputBytes =
        stringValue.size_bytes() + literal.size_bytes() + integerBytes;

    if (d_chars == nullptr) {
      d_sizes[index] = outputBytes;
      return;
    }

    auto* output = d_chars + d_offsets[index];
    for (cudf::size_type i = 0; i < stringValue.size_bytes(); ++i) {
      output[i] = stringValue.data()[i];
    }
    output += stringValue.size_bytes();
    for (cudf::size_type i = 0; i < literal.size_bytes(); ++i) {
      output[i] = literal.data()[i];
    }
    output += literal.size_bytes();
    if (integerValid) {
      writeInteger(integerValue, output, integerBytes);
    }
  }
};

struct DispatchConcatStringLiteralIntegral {
  const cudf::column_view& strings;
  cudf::string_view literal;
  const cudf::column_view& integers;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
    requires(cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, bool>)
  std::unique_ptr<cudf::column> operator()() const {
    auto stringsDevice = cudf::column_device_view::create(strings, stream);
    auto integersDevice = cudf::column_device_view::create(integers, stream);
    auto [offsets, chars] = cudf::strings::detail::make_strings_children(
        ConcatStringLiteralIntegralFunctor<T>{
            *stringsDevice, literal, *integersDevice},
        strings.size(),
        stream,
        mr);
    return cudf::make_strings_column(
        strings.size(),
        std::move(offsets),
        chars.release(),
        0,
        rmm::device_buffer{});
  }

  template <typename T>
    requires(!cuda::std::is_integral_v<T> || cuda::std::is_same_v<T, bool>)
  std::unique_ptr<cudf::column> operator()() const {
    CUDF_FAIL("concat integral-cast fusion requires an integer column");
  }
};

} // namespace

std::unique_ptr<cudf::column> concatStringLiteralIntegral(
    const cudf::column_view& strings,
    std::string_view literal,
    const cudf::column_view& integers,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(
      strings.type().id() == cudf::type_id::STRING,
      "concat integral-cast fusion requires a strings column");
  CUDF_EXPECTS(
      strings.size() == integers.size(),
      "concat integral-cast fusion requires equal-sized inputs");
  if (strings.is_empty()) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  cudf::string_scalar literalScalar(literal, true, stream, mr);
  return cudf::type_dispatcher(
      integers.type(),
      DispatchConcatStringLiteralIntegral{
          strings, literalScalar.value(stream), integers, stream, mr});
}

} // namespace facebook::velox::cudf_velox
