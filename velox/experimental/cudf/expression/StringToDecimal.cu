/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

// Adapted from NVIDIA/spark-rapids-jni cast_string.cu at commit
// 80d402b9e2ac23aedd07cb49b2ea3d8ab929ad54.
#include "velox/experimental/cudf/expression/StringToDecimal.h"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cuda/std/algorithm>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

namespace facebook::velox::cudf_velox {
namespace {

constexpr int32_t kThreadsPerBlock = 256;

__host__ __device__ constexpr bool isWhitespace(char value) {
  const auto c = static_cast<unsigned char>(value);
  return c <= 0x1f || c == ' ';
}

template <typename T>
__device__ bool willOverflow(T value, bool adding) {
  if constexpr (cuda::std::is_signed_v<T>) {
    if (!adding) {
      return value < cuda::std::numeric_limits<T>::min() / 10;
    }
  }
  return value > cuda::std::numeric_limits<T>::max() / 10;
}

template <typename T>
__device__ bool willOverflow(T lhs, T rhs, bool adding) {
  if constexpr (cuda::std::is_signed_v<T>) {
    if (!adding) {
      return lhs < cuda::std::numeric_limits<T>::min() + rhs;
    }
  }
  return lhs > cuda::std::numeric_limits<T>::max() - rhs;
}

template <typename T>
__device__ cuda::std::pair<bool, T>
appendDigit(bool first, T value, T digit, bool positive) {
  if (!first) {
    if (willOverflow(value, positive)) {
      return {false, value};
    }
    value *= 10;
  }
  if (willOverflow(value, digit, positive)) {
    return {false, value};
  }
  return {true, positive ? value + digit : value - digit};
}

template <typename T>
__device__ cuda::std::optional<cuda::std::tuple<bool, int32_t, int32_t>>
validateAndLocateDecimal(const char* chars, int32_t length, bool strip) {
  enum class State {
    kDigits,
    kExponent,
    kDecimalPoint,
    kExponentOrSign,
    kExponentSign,
    kTrailingWhitespace,
    kInvalid,
  };

  T exponent = 0;
  int32_t index = 0;
  bool positive = true;
  bool exponentPositive = true;
  int32_t decimalLocation = -1;

  if (length == 0) {
    return cuda::std::nullopt;
  }
  if (strip) {
    while (index < length && isWhitespace(chars[index])) {
      ++index;
    }
  }
  if (index == length) {
    return cuda::std::nullopt;
  }
  if (chars[index] == '-') {
    positive = false;
    ++index;
  } else if (chars[index] == '+') {
    ++index;
  }
  if (index == length) {
    return cuda::std::nullopt;
  }

  const auto firstDigit = index;
  int32_t lastDigit = length;
  auto state = State::kDigits;
  bool sawMantissaDigit = false;
  bool sawExponentDigit = false;

  for (int32_t i = index; i < length; ++i) {
    const auto value = chars[i];
    const auto relativeIndex = i - index;
    const auto previous = state;

    switch (state) {
      case State::kTrailingWhitespace:
        if (!isWhitespace(value)) {
          state = State::kInvalid;
        }
        break;
      case State::kDecimalPoint:
      case State::kDigits:
        if (value >= '0' && value <= '9') {
          state = State::kDigits;
          sawMantissaDigit = true;
        } else if (value == '.' && decimalLocation == -1) {
          decimalLocation = relativeIndex;
          state = State::kDecimalPoint;
        } else if ((value == 'e' || value == 'E') && sawMantissaDigit) {
          state = State::kExponentOrSign;
        } else if (strip && isWhitespace(value) && relativeIndex != 0) {
          state = State::kTrailingWhitespace;
        } else {
          state = State::kInvalid;
        }
        break;
      case State::kExponentOrSign:
        if (value == '+') {
          state = State::kExponentSign;
        } else if (value == '-') {
          exponentPositive = false;
          state = State::kExponentSign;
        } else if (value >= '0' && value <= '9') {
          state = State::kExponent;
          sawExponentDigit = true;
        } else {
          state = State::kInvalid;
        }
        break;
      case State::kExponentSign:
      case State::kExponent:
        if (value >= '0' && value <= '9') {
          state = State::kExponent;
          sawExponentDigit = true;
        } else if (strip && isWhitespace(value) && sawExponentDigit) {
          state = State::kTrailingWhitespace;
        } else {
          state = State::kInvalid;
        }
        break;
      case State::kInvalid:
        break;
    }

    if (state == State::kInvalid) {
      return cuda::std::nullopt;
    }
    if (previous == State::kDigits && state != State::kDigits &&
        state != State::kDecimalPoint) {
      lastDigit = i;
    }
    if (state == State::kExponent) {
      const T digit = value - '0';
      auto [success, newValue] =
          appendDigit(exponent == 0, exponent, digit, exponentPositive);
      if (!success) {
        return cuda::std::nullopt;
      }
      exponent = newValue;
    }
  }

  if (!sawMantissaDigit || state == State::kExponentOrSign ||
      state == State::kExponentSign) {
    return cuda::std::nullopt;
  }
  if (decimalLocation < 0) {
    decimalLocation = lastDigit - firstDigit;
  }
  decimalLocation += exponent;
  return cuda::std::tuple{positive, decimalLocation, firstDigit};
}

template <typename T>
CUDF_KERNEL void stringToDecimalKernel(
    T* output,
    cudf::bitmask_type* validity,
    const char* chars,
    const cudf::size_type* offsets,
    const cudf::bitmask_type* inputNullMask,
    cudf::size_type size,
    int32_t scale,
    int32_t precision,
    bool stripWhitespace) {
  auto block = cooperative_groups::this_thread_block();
  auto warp =
      cooperative_groups::tiled_partition<cudf::detail::warp_size>(block);
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= size) {
    return;
  }

  const auto rowStart = offsets[row];
  const auto length = offsets[row + 1] - rowStart;
  bool valid =
      (inputNullMask == nullptr || cudf::bit_is_set(inputNullMask, row)) &&
      length > 0;

  auto countSignificantDigits =
      [](const char* input, int32_t inputLength, int32_t numDigits) {
        int32_t count = 0;
        int32_t digitsFound = 0;
        for (int32_t i = 0; i < inputLength && digitsFound < numDigits; ++i) {
          if (input[i] == 'e' || input[i] == 'E') {
            break;
          }
          if (input[i] != '.') {
            ++digitsFound;
            if (count != 0 || input[i] != '0') {
              ++count;
            }
          }
        }
        return count;
      };

  const auto validated = valid
      ? validateAndLocateDecimal<T>(chars + rowStart, length, stripWhitespace)
      : cuda::std::nullopt;
  valid = validated.has_value();

  if (valid) {
    auto [positive, decimalLocation, firstDigit] = *validated;
    const auto maxDigitsBeforeDecimal = precision + scale;
    const auto significantDigitsBeforeDecimalInString = countSignificantDigits(
        chars + rowStart + firstDigit, length - firstDigit, decimalLocation);
    const auto lastDigit = decimalLocation - scale;

    int32_t preciseDigits = 0;
    int32_t totalDigits = 0;
    T value = 0;
    bool foundSignificantDigit = false;
    int32_t roundingDigits = 0;

    if (lastDigit >= 0) {
      for (int32_t i = firstDigit; i < length && valid; ++i) {
        const auto c = chars[rowStart + i];
        if (c == '.') {
          continue;
        }
        if (c < '0' || c > '9') {
          break;
        }

        const T digit = c - '0';
        if (preciseDigits + 1 > precision || totalDigits + 1 > lastDigit) {
          if (digit >= 5) {
            const auto previousValue = value;
            if (willOverflow(value, static_cast<T>(1), positive)) {
              valid = false;
              break;
            }
            value += positive ? 1 : -1;

            auto countDigits = [](T number) {
              int32_t count = 0;
              while (number != 0) {
                ++count;
                number /= 10;
              }
              return count;
            };
            if (previousValue != 0 &&
                countDigits(value) > countDigits(previousValue)) {
              ++totalDigits;
              ++preciseDigits;
              ++decimalLocation;
              ++roundingDigits;
            }
          }
          break;
        }

        ++totalDigits;
        if (foundSignificantDigit || totalDigits > decimalLocation ||
            digit != 0) {
          foundSignificantDigit = true;
          ++preciseDigits;
        }
        auto [success, newValue] =
            appendDigit(i == firstDigit, value, digit, positive);
        if (!success) {
          valid = false;
          break;
        }
        value = newValue;
      }
    }

    const auto significantPrecedingZeros =
        decimalLocation < 0 ? -decimalLocation : 0;
    const auto zerosToDecimal = cuda::std::max(
        0,
        scale > 0 ? decimalLocation - totalDigits - scale
                  : decimalLocation - totalDigits);
    const auto significantDigitsBeforeDecimal =
        significantDigitsBeforeDecimalInString + zerosToDecimal +
        roundingDigits;
    const auto leadingZeros = totalDigits - preciseDigits;
    if (maxDigitsBeforeDecimal < decimalLocation - leadingZeros) {
      valid = false;
    }

    for (int32_t i = 0; i < zerosToDecimal && valid; ++i) {
      if (willOverflow(value, positive)) {
        valid = false;
        break;
      }
      value *= 10;
      ++preciseDigits;
    }

    const auto digitsAfterDecimal = preciseDigits -
        significantDigitsBeforeDecimal + significantPrecedingZeros;
    const auto digitsNeededAfterDecimal =
        cuda::std::min(precision - significantDigitsBeforeDecimal, -scale);
    for (int32_t i = digitsAfterDecimal; i < digitsNeededAfterDecimal && valid;
         ++i) {
      if (willOverflow(value, positive)) {
        valid = false;
        break;
      }
      value *= 10;
    }
    if (valid) {
      output[row] = value;
    }
  }

  const auto validityWord = warp.ballot(static_cast<int>(valid));
  if (warp.thread_rank() == 0) {
    validity[warp.meta_group_rank() + blockIdx.x * warp.meta_group_size()] =
        validityWord;
  }
}

template <typename Decimal>
std::unique_ptr<cudf::column> launchStringToDecimal(
    const cudf::strings_column_view& input,
    cudf::data_type outputType,
    int32_t precision,
    bool stripWhitespace,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  using Storage = cudf::device_storage_type_t<Decimal>;
  rmm::device_uvector<Storage> data(input.size(), stream, mr);
  const auto words = cudf::bitmask_allocation_size_bytes(input.size()) /
      sizeof(cudf::bitmask_type);
  rmm::device_uvector<cudf::bitmask_type> nullMask(words, stream, mr);
  const dim3 blocks((input.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const dim3 threads(kThreadsPerBlock);
  stringToDecimalKernel<<<blocks, threads, 0, stream.value()>>>(
      data.data(),
      nullMask.data(),
      input.chars_begin(stream),
      input.offsets().data<cudf::size_type>(),
      input.null_mask(),
      input.size(),
      outputType.scale(),
      precision,
      stripWhitespace);
  CUDF_CUDA_TRY(cudaGetLastError());
  const auto nullCount =
      cudf::null_count(nullMask.data(), 0, input.size(), stream);
  return std::make_unique<cudf::column>(
      outputType, input.size(), data.release(), nullMask.release(), nullCount);
}

} // namespace

std::unique_ptr<cudf::column> tryCastStringToDecimal(
    const cudf::strings_column_view& input,
    cudf::data_type outputType,
    int32_t precision,
    bool stripWhitespace,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (input.is_empty()) {
    return std::make_unique<cudf::column>(
        outputType, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
  }
  switch (outputType.id()) {
    case cudf::type_id::DECIMAL64:
      return launchStringToDecimal<numeric::decimal64>(
          input, outputType, precision, stripWhitespace, stream, mr);
    case cudf::type_id::DECIMAL128:
      return launchStringToDecimal<numeric::decimal128>(
          input, outputType, precision, stripWhitespace, stream, mr);
    default:
      CUDF_FAIL("String-to-decimal cast requires DECIMAL64 or DECIMAL128");
  }
}

} // namespace facebook::velox::cudf_velox
