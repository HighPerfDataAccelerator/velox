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
#include "velox/experimental/cudf/expression/DecimalExpressionKernels.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>

namespace facebook::velox::cudf_velox {
namespace {

template <typename InT, typename OutT>
__global__ void decimalDivideKernel(
    const InT* lhs,
    const InT* rhs,
    OutT* out,
    cudf::bitmask_type* validMask,
    int32_t numRows,
    __int128_t scale) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numRows) {
    return;
  }
  __int128_t numerator = static_cast<__int128_t>(lhs[idx]);
  __int128_t denom = static_cast<__int128_t>(rhs[idx]);
  if (denom == 0) {
    out[idx] = OutT{0};
    if (validMask) {
      atomicAnd(
          &validMask[idx / 32],
          ~(cudf::bitmask_type{1} << (idx % 32)));
    }
    return;
  }
  int sign = 1;
  if (numerator < 0) {
    numerator = -numerator;
    sign = -sign;
  }
  if (denom < 0) {
    denom = -denom;
    sign = -sign;
  }
  __int128_t scaled = numerator * scale;
  __int128_t quotient = scaled / denom;
  __int128_t remainder = scaled % denom;
  if (remainder * 2 >= denom) {
    ++quotient;
  }
  if (sign < 0) {
    quotient = -quotient;
  }
  out[idx] = static_cast<OutT>(quotient);
}

inline __int128_t pow10Int128(int32_t exp) {
  __int128_t value = 1;
  for (int32_t i = 0; i < exp; ++i) {
    value *= 10;
  }
  return value;
}

template <typename InT, typename OutT>
void launchDivideKernel(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::mutable_column_view out,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  if (lhs.size() == 0) {
    return;
  }
  int32_t blockSize = 256;
  int32_t gridSize = (lhs.size() + blockSize - 1) / blockSize;
  auto scale = pow10Int128(aRescale);
  decimalDivideKernel<<<gridSize, blockSize, 0, stream.value()>>>(
      lhs.data<InT>(),
      rhs.data<InT>(),
      out.data<OutT>(),
      out.null_mask(),
      lhs.size(),
      scale);
  CUDF_CUDA_TRY(cudaGetLastError());
}

} // namespace

std::unique_ptr<cudf::column> decimalDivide(
    const cudf::column_view& lhs,
    const cudf::column_view& rhs,
    cudf::data_type outputType,
    int32_t aRescale,
    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Decimal divide requires equal sizes");
  CUDF_EXPECTS(
      aRescale >= 0, "Decimal divide requires non-negative rescale factor");

  auto lhsTypeId = lhs.type().id();
  auto rhsTypeId = rhs.type().id();

  std::unique_ptr<cudf::column> lhsCast;
  std::unique_ptr<cudf::column> rhsCast;
  cudf::column_view lhsAligned = lhs;
  cudf::column_view rhsAligned = rhs;

  auto isIntegral = [](cudf::type_id id) {
    return id == cudf::type_id::INT8 || id == cudf::type_id::INT16 ||
        id == cudf::type_id::INT32 || id == cudf::type_id::INT64;
  };

  if (isIntegral(lhsTypeId) || isIntegral(rhsTypeId) ||
      lhsTypeId != rhsTypeId) {
    auto targetId = cudf::type_id::DECIMAL128;

    if (isIntegral(lhsTypeId)) {
      auto decType = cudf::data_type{
          cudf::type_id::DECIMAL128, numeric::scale_type{0}};
      lhsCast = cudf::cast(lhs, decType, stream);
      lhsAligned = lhsCast->view();
    } else if (lhsTypeId == cudf::type_id::DECIMAL64) {
      auto castType =
          cudf::data_type{targetId, lhs.type().scale()};
      lhsCast = cudf::cast(lhs, castType, stream);
      lhsAligned = lhsCast->view();
    }

    if (isIntegral(rhsTypeId)) {
      auto decType = cudf::data_type{
          cudf::type_id::DECIMAL128, numeric::scale_type{0}};
      rhsCast = cudf::cast(rhs, decType, stream);
      rhsAligned = rhsCast->view();
    } else if (rhsTypeId == cudf::type_id::DECIMAL64) {
      auto castType =
          cudf::data_type{targetId, rhs.type().scale()};
      rhsCast = cudf::cast(rhs, castType, stream);
      rhsAligned = rhsCast->view();
    }

    lhsTypeId = lhsAligned.type().id();
    rhsTypeId = rhsAligned.type().id();
  }
  CUDF_EXPECTS(
      lhsTypeId == rhsTypeId,
      "Decimal divide: type alignment failed");

  auto [nullMask, nullCount] =
      cudf::bitmask_and(
          cudf::table_view({lhsAligned, rhsAligned}), stream);
  auto out = cudf::make_fixed_width_column(
      outputType, lhs.size(), std::move(nullMask), nullCount,
      stream);

  if (lhsTypeId == cudf::type_id::DECIMAL64) {
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernel<int64_t, int64_t>(
          lhsAligned, rhsAligned, out->mutable_view(),
          aRescale, stream);
    } else {
      CUDF_EXPECTS(
          outputType.id() == cudf::type_id::DECIMAL128,
          "Unexpected output type for decimal divide");
      launchDivideKernel<int64_t, __int128_t>(
          lhsAligned, rhsAligned, out->mutable_view(),
          aRescale, stream);
    }
  } else {
    CUDF_EXPECTS(
        lhsTypeId == cudf::type_id::DECIMAL128,
        "Unsupported input type for decimal divide");
    if (outputType.id() == cudf::type_id::DECIMAL64) {
      launchDivideKernel<__int128_t, int64_t>(
          lhsAligned, rhsAligned, out->mutable_view(),
          aRescale, stream);
    } else {
      CUDF_EXPECTS(
          outputType.id() == cudf::type_id::DECIMAL128,
          "Unexpected output type for decimal divide");
      launchDivideKernel<__int128_t, __int128_t>(
          lhsAligned, rhsAligned, out->mutable_view(),
          aRescale, stream);
    }
  }

  if (out->nullable()) {
    out->set_null_count(cudf::null_count(
        out->view().null_mask(), 0, out->size(), stream));
  }
  return out;
}

} // namespace facebook::velox::cudf_velox
