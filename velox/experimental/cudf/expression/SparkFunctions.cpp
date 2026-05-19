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

#include "velox/experimental/cudf/expression/AstUtils.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"

#include "velox/common/base/BloomFilter.h"
#include "velox/expression/ConstantExpr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/BaseVector.h"

#include <cuda_runtime.h>
#include <cudf/binaryop.hpp>
#include <cudf/hashing.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <folly/hash/Hash.h>
#include <rmm/device_buffer.hpp>

namespace facebook::velox::cudf_velox {
namespace {

// Spark date_add function implementation.
// For the presto date_add, the first value is unit string,
// may need to get the function with prefix, if the prefix is "", it is Spark
// function.
class DateAddFunction : public CudfFunction {
 public:
  DateAddFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "date_add function expects exactly 2 inputs");
    VELOX_CHECK(
        expr->inputs()[0]->type()->isDate(),
        "First argument to date_add must be a date");
    VELOX_CHECK_NULL(
        std::dynamic_pointer_cast<velox::exec::ConstantExpr>(
            expr->inputs()[0]));
    // The date_add second argument could be int8_t, int16_t, int32_t.
    value_ = makeScalarFromConstantExpr(
        expr->inputs()[1], cudf::type_id::DURATION_DAYS);
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    auto inputCol = asView(inputColumns[0]);
    return cudf::binary_operation(
        inputCol,
        *value_,
        cudf::binary_operator::ADD,
        cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
        stream,
        mr);
  }

 private:
  std::unique_ptr<cudf::scalar> value_;
};

class HashFunction : public CudfFunction {
 public:
  HashFunction(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_GE(expr->inputs().size(), 2, "hash expects at least 2 inputs");
    auto seedExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(seedExpr, "hash seed must be a constant");
    int32_t seedValue =
        seedExpr->value()->as<SimpleVector<int32_t>>()->valueAt(0);
    VELOX_CHECK_GE(seedValue, 0);
    seedValue_ = seedValue;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty());
    auto inputTableView = convertToTableView(inputColumns);
    return cudf::hashing::murmurhash3_x86_32(
        inputTableView, seedValue_, stream, mr);
  }

 private:
  static cudf::table_view convertToTableView(
      std::vector<ColumnOrView>& inputColumns) {
    std::vector<cudf::column_view> columns;
    columns.reserve(inputColumns.size());

    for (auto& col : inputColumns) {
      columns.push_back(asView(col));
    }

    return cudf::table_view(columns);
  }

  uint32_t seedValue_;
};

class XxHash64Function : public CudfFunction {
 public:
  explicit XxHash64Function(const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_GE(
        expr->inputs().size(), 2, "xxhash64 expects at least 2 inputs");
    auto seedExpr = std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(seedExpr, "xxhash64 seed must be a constant");
    seedValue_ = static_cast<uint64_t>(
        seedExpr->value()->as<SimpleVector<int64_t>>()->valueAt(0));
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK(!inputColumns.empty());
    std::vector<cudf::column_view> columns;
    columns.reserve(inputColumns.size());
    for (auto& col : inputColumns) {
      columns.push_back(asView(col));
    }
    return cudf::hashing::xxhash_64(
        cudf::table_view(columns), seedValue_, stream, mr);
  }

 private:
  uint64_t seedValue_;
};

class MightContainFunction : public CudfFunction {
 public:
  explicit MightContainFunction(
      const std::shared_ptr<velox::exec::Expr>& expr) {
    using velox::exec::ConstantExpr;
    VELOX_CHECK_EQ(
        expr->inputs().size(), 2, "might_contain expects exactly 2 inputs");
    auto bloomExpr =
        std::dynamic_pointer_cast<ConstantExpr>(expr->inputs()[0]);
    VELOX_CHECK_NOT_NULL(
        bloomExpr, "might_contain bloom filter must be a constant");
    auto bloomValue = bloomExpr->value();
    if (bloomValue->isNullAt(0)) {
      hasFilter_ = false;
      return;
    }
    auto serialized = bloomValue->as<SimpleVector<StringView>>()->valueAt(0);
    serialized_.assign(serialized.data(), serialized.size());
    hasFilter_ = true;
  }

  ColumnOrView eval(
      std::vector<ColumnOrView>& inputColumns,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const override {
    VELOX_CHECK_EQ(
        inputColumns.size(),
        1,
        "might_contain receives 1 column input; bloom filter is literal");
    auto inputView = asView(inputColumns[0]);
    const auto numRows = inputView.size();

    std::vector<int64_t> hostKeys(numRows);
    if (numRows > 0) {
      copyKeysToHost(inputView, hostKeys, stream);
    }

    std::vector<cudf::bitmask_type> hostNulls;
    const cudf::bitmask_type* deviceNulls = inputView.null_mask();
    if (deviceNulls != nullptr) {
      const auto words = cudf::bitmask_allocation_size_bytes(numRows) /
          sizeof(cudf::bitmask_type);
      hostNulls.resize(words);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostNulls.data(),
          deviceNulls,
          words * sizeof(cudf::bitmask_type),
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    stream.synchronize();

    std::vector<uint8_t> hostResult(numRows, 0);
    for (cudf::size_type i = 0; i < numRows; ++i) {
      if (!hasFilter_) {
        continue;
      }
      if (!hostNulls.empty()) {
        const bool valid = (hostNulls[i / 32] >> (i % 32)) & 1u;
        if (!valid) {
          continue;
        }
      }
      const uint64_t hashed = folly::hasher<int64_t>()(hostKeys[i]);
      hostResult[i] =
          velox::BloomFilter<>::mayContain(serialized_.c_str(), hashed) ? 1u
                                                                        : 0u;
    }

    rmm::device_buffer data(numRows * sizeof(uint8_t), stream, mr);
    if (numRows > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          data.data(),
          hostResult.data(),
          numRows * sizeof(uint8_t),
          cudaMemcpyHostToDevice,
          stream.value()));
    }

    rmm::device_buffer nullMask{};
    cudf::size_type nullCount = 0;
    if (deviceNulls != nullptr) {
      nullMask = rmm::device_buffer(
          deviceNulls,
          cudf::bitmask_allocation_size_bytes(numRows),
          stream,
          mr);
      nullCount = inputView.null_count();
    }

    return std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::BOOL8},
        numRows,
        std::move(data),
        std::move(nullMask),
        nullCount);
  }

 private:
  template <typename T>
  static void copyKeysAsInt64(
      const cudf::column_view& inputView,
      std::vector<int64_t>& hostKeys,
      rmm::cuda_stream_view stream) {
    std::vector<T> tmp(inputView.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        tmp.data(),
        inputView.data<T>(),
        inputView.size() * sizeof(T),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    for (cudf::size_type i = 0; i < inputView.size(); ++i) {
      hostKeys[i] = static_cast<int64_t>(tmp[i]);
    }
  }

  static void copyKeysToHost(
      const cudf::column_view& inputView,
      std::vector<int64_t>& hostKeys,
      rmm::cuda_stream_view stream) {
    const auto inputType = inputView.type().id();
    switch (inputType) {
      case cudf::type_id::INT64:
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            hostKeys.data(),
            inputView.data<int64_t>(),
            inputView.size() * sizeof(int64_t),
            cudaMemcpyDeviceToHost,
            stream.value()));
        break;
      case cudf::type_id::INT32:
        copyKeysAsInt64<int32_t>(inputView, hostKeys, stream);
        break;
      case cudf::type_id::INT16:
        copyKeysAsInt64<int16_t>(inputView, hostKeys, stream);
        break;
      case cudf::type_id::INT8:
        copyKeysAsInt64<int8_t>(inputView, hostKeys, stream);
        break;
      case cudf::type_id::UINT64:
        copyKeysAsInt64<uint64_t>(inputView, hostKeys, stream);
        break;
      case cudf::type_id::UINT32:
        copyKeysAsInt64<uint32_t>(inputView, hostKeys, stream);
        break;
      case cudf::type_id::UINT16:
        copyKeysAsInt64<uint16_t>(inputView, hostKeys, stream);
        break;
      case cudf::type_id::UINT8:
        copyKeysAsInt64<uint8_t>(inputView, hostKeys, stream);
        break;
      default:
        VELOX_FAIL(
            "might_contain hash input must be an integer type; saw cudf type_id={}",
            static_cast<int>(inputType));
    }
  }

  std::string serialized_;
  bool hasFilter_{false};
};

} // namespace

void registerSparkFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  registerCudfFunction(
      prefix + "hash_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<HashFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("integer")
           .argumentType("any")
           .build()});

  registerCudfFunction(
      prefix + "xxhash64_with_seed",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<XxHash64Function>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("bigint")
           .constantArgumentType("bigint")
           .argumentType("any")
           .variableArity("any")
           .build()});

  registerCudfFunction(
      prefix + "date_add",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<DateAddFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("date")
           .argumentType("date")
           .constantArgumentType("integer")
           .build()});

  registerCudfFunction(
      prefix + "might_contain",
      [](const std::string&, const std::shared_ptr<velox::exec::Expr>& expr) {
        return std::make_shared<MightContainFunction>(expr);
      },
      {FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("tinyint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("smallint")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("integer")
           .build(),
       FunctionSignatureBuilder()
           .returnType("boolean")
           .constantArgumentType("varbinary")
           .argumentType("bigint")
           .build()});
}

} // namespace facebook::velox::cudf_velox
