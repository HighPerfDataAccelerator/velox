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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/CudfReduce.h"
#include "velox/experimental/cudf/exec/DecimalAggregationHostOps.h"
#include "velox/experimental/cudf/exec/DecimalAggregationState.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/common/base/BloomFilter.h"
#include "velox/core/Expressions.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/type/Type.h"
#include "velox/vector/SimpleVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <folly/hash/Hash.h>

#include <algorithm>
#include <string>

namespace {

using namespace facebook::velox;
using facebook::velox::cudf_velox::castDecimal64InputToDecimal128;
using facebook::velox::cudf_velox::CountInputKind;
using facebook::velox::cudf_velox::finalizeDecimalAverage;
using facebook::velox::cudf_velox::get_output_mr;
using facebook::velox::cudf_velox::get_temp_mr;
using facebook::velox::cudf_velox::ReduceAggregator;
using facebook::velox::cudf_velox::ResolvedAggregateInfo;
using facebook::velox::cudf_velox::serializeDecimalPartialOrIntermediateState;
using facebook::velox::cudf_velox::validateIntermediateColumnType;

#define DEFINE_SIMPLE_REDUCE_AGGREGATOR(Name, name)                            \
  struct Reduce##Name##Aggregator : ReduceAggregator {                         \
    Reduce##Name##Aggregator(                                                  \
        core::AggregationNode::Step step,                                      \
        uint32_t inputIndex,                                                   \
        VectorPtr constant,                                                    \
        const TypePtr& resultType)                                             \
        : ReduceAggregator(step, inputIndex, constant, resultType) {}          \
                                                                               \
    std::unique_ptr<cudf::column> doReduce(                                    \
        cudf::table_view const& input,                                         \
        TypePtr const& outputType,                                             \
        vector_size_t /* inputRowCount */,                                     \
        rmm::cuda_stream_view stream,                                          \
        rmm::device_async_resource_ref mr) override {                          \
      auto const aggRequest =                                                  \
          cudf::make_##name##_aggregation<cudf::reduce_aggregation>();         \
      auto const cudfOutputType = cudf_velox::veloxToCudfDataType(outputType); \
      auto const resultScalar = cudf::reduce(                                  \
          input.column(inputIndex),                                            \
          *aggRequest,                                                         \
          cudfOutputType,                                                      \
          stream,                                                              \
          get_temp_mr());                                                      \
      return cudf::make_column_from_scalar(*resultScalar, 1, stream, mr);      \
    }                                                                          \
  };

DEFINE_SIMPLE_REDUCE_AGGREGATOR(Sum, sum)

std::unique_ptr<cudf::column> reduceMinMaxWithInputType(
    cudf::column_view inputCol,
    const cudf::reduce_aggregation& aggRequest,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const resultScalar = cudf::reduce(
      inputCol, aggRequest, inputCol.type(), stream, get_temp_mr());
  auto resultCol = cudf::make_column_from_scalar(*resultScalar, 1, stream, mr);
  auto const cudfOutputType = cudf_velox::veloxToCudfDataType(outputType);
  if (resultCol->type() != cudfOutputType) {
    resultCol = cudf::cast(resultCol->view(), cudfOutputType, stream, mr);
  }
  return resultCol;
}

#define DEFINE_MIN_MAX_REDUCE_AGGREGATOR(Name, name)                      \
  struct Reduce##Name##Aggregator : ReduceAggregator {                    \
    Reduce##Name##Aggregator(                                             \
        core::AggregationNode::Step step,                                 \
        uint32_t inputIndex,                                              \
        VectorPtr constant,                                               \
        const TypePtr& resultType)                                        \
        : ReduceAggregator(step, inputIndex, constant, resultType) {}     \
                                                                          \
    std::unique_ptr<cudf::column> doReduce(                               \
        cudf::table_view const& input,                                    \
        TypePtr const& outputType,                                        \
        vector_size_t /* inputRowCount */,                                \
        rmm::cuda_stream_view stream,                                     \
        rmm::device_async_resource_ref mr) override {                     \
      auto const aggRequest =                                             \
          cudf::make_##name##_aggregation<cudf::reduce_aggregation>();    \
      return reduceMinMaxWithInputType(                                   \
          input.column(inputIndex), *aggRequest, outputType, stream, mr); \
    }                                                                     \
  };

DEFINE_MIN_MAX_REDUCE_AGGREGATOR(Min, min)
DEFINE_MIN_MAX_REDUCE_AGGREGATOR(Max, max)

struct ReduceCountAggregator : ReduceAggregator {
  ReduceCountAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      CountInputKind inputKind,
      const TypePtr& resultType)
      : ReduceAggregator(step, inputIndex, nullptr, resultType),
        inputKind_(inputKind) {}

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      vector_size_t inputRowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    if (exec::isRawInput(step)) {
      int64_t count;
      switch (inputKind_) {
        case CountInputKind::kNullConstant:
          count = 0;
          break;
        case CountInputKind::kCountAll:
          count = input.num_columns() > 0 ? input.num_rows() : inputRowCount;
          break;
        case CountInputKind::kColumn: {
          VELOX_CHECK_GT(
              input.num_columns(),
              0,
              "count(column) requires at least one input column");
          auto inputCol = input.column(inputIndex);
          count = inputCol.size() - inputCol.null_count();
          break;
        }
        default:
          VELOX_UNREACHABLE();
      }

      auto resultScalar =
          cudf::numeric_scalar<int64_t>(count, true, stream, get_temp_mr());

      return cudf::make_column_from_scalar(resultScalar, 1, stream, mr);
    } else {
      // For non-raw input (intermediate/final), use sum aggregation
      auto const aggRequest =
          cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto const cudfOutputType = cudf::data_type(cudf::type_id::INT64);
      auto const resultScalar = cudf::reduce(
          input.column(inputIndex),
          *aggRequest,
          cudfOutputType,
          stream,
          get_temp_mr());
      resultScalar->set_valid_async(true, stream);
      return cudf::make_column_from_scalar(*resultScalar, 1, stream, mr);
    }
  }

 private:
  CountInputKind inputKind_;
};

struct ReduceMeanAggregator : ReduceAggregator {
  ReduceMeanAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : ReduceAggregator(step, inputIndex, constant, resultType) {}

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      vector_size_t /* inputRowCount */,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    switch (step) {
      case core::AggregationNode::Step::kSingle: {
        auto const aggRequest =
            cudf::make_mean_aggregation<cudf::reduce_aggregation>();
        auto const cudfOutputType = cudf_velox::veloxToCudfDataType(outputType);
        auto const resultScalar = cudf::reduce(
            input.column(inputIndex),
            *aggRequest,
            cudfOutputType,
            stream,
            get_temp_mr());
        return cudf::make_column_from_scalar(*resultScalar, 1, stream, mr);
      }
      case core::AggregationNode::Step::kPartial: {
        VELOX_CHECK(outputType->isRow());
        auto const& rowType = outputType->asRow();
        auto const sumType = rowType.childAt(0);
        auto const countType = rowType.childAt(1);
        auto const cudfSumType = cudf_velox::veloxToCudfDataType(sumType);
        auto const cudfCountType = cudf_velox::veloxToCudfDataType(countType);

        // sum
        auto const aggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sumResultScalar = cudf::reduce(
            input.column(inputIndex),
            *aggRequest,
            cudfSumType,
            stream,
            get_temp_mr());
        auto sumCol =
            cudf::make_column_from_scalar(*sumResultScalar, 1, stream, mr);

        // libcudf doesn't have a count agg for reduce. What we want is to
        // count the number of valid rows.
        auto countCol = cudf::make_column_from_scalar(
            cudf::numeric_scalar<int64_t>(
                input.column(inputIndex).size() -
                    input.column(inputIndex).null_count(),
                true,
                stream,
                get_temp_mr()),
            1,
            stream,
            mr);

        // Assemble into struct as expected by velox.
        auto children = std::vector<std::unique_ptr<cudf::column>>();
        children.push_back(std::move(sumCol));
        children.push_back(std::move(countCol));
        return std::make_unique<cudf::column>(
            cudf::data_type(cudf::type_id::STRUCT),
            1,
            rmm::device_buffer{},
            rmm::device_buffer{},
            0,
            std::move(children));
      }
      case core::AggregationNode::Step::kFinal: {
        // Input column has two children: sum and count
        auto const sumCol = input.column(inputIndex).child(0);
        auto const countCol = input.column(inputIndex).child(1);

        // sum the sums
        auto const sumAggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const sumResultScalar = cudf::reduce(
            sumCol, *sumAggRequest, sumCol.type(), stream, get_temp_mr());
        auto sumResultCol =
            cudf::make_column_from_scalar(*sumResultScalar, 1, stream, mr);

        // sum the counts
        auto const countAggRequest =
            cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        auto const countResultScalar = cudf::reduce(
            countCol, *countAggRequest, countCol.type(), stream, get_temp_mr());

        // divide the sums by the counts
        auto const cudfOutputType = cudf_velox::veloxToCudfDataType(outputType);
        return cudf::binary_operation(
            *sumResultCol,
            *countResultScalar,
            cudf::binary_operator::DIV,
            cudfOutputType,
            stream,
            mr);
      }
      default:
        VELOX_NYI("Unsupported aggregation step for mean");
    }
  }
};

// Materializes reduced sum/count scalars into 1-row columns.
cudf_velox::DecimalSumStateColumns makeSumCountColumns(
    cudf::scalar const& sumScalar,
    cudf::scalar const& countScalar,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  cudf_velox::DecimalSumStateColumns cols;
  cols.sum = cudf::make_column_from_scalar(sumScalar, 1, stream, mr);
  cols.count = cudf::make_column_from_scalar(countScalar, 1, stream, mr);
  return cols;
}

std::unique_ptr<cudf::column> partialDecimalSumCountToSerializedString(
    cudf::column_view inputCol,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::column> castedInput;
  inputCol = castDecimal64InputToDecimal128(inputCol, castedInput, stream);
  auto const sumAgg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto sumScalar =
      cudf::reduce(inputCol, *sumAgg, inputCol.type(), stream, get_temp_mr());
  auto countAgg = cudf::make_count_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::EXCLUDE);
  auto countScalar = cudf::reduce(
      inputCol,
      *countAgg,
      cudf::data_type{cudf::type_id::INT64},
      stream,
      get_temp_mr());
  auto cols = makeSumCountColumns(*sumScalar, *countScalar, stream, mr);
  return serializeDecimalPartialOrIntermediateState(
      std::move(cols.sum), std::move(cols.count), stream, mr);
}

// Decodes serialized decimal SUM state, sums the per-row partial sums and
// counts, and returns them as 1-row columns. Shared by the intermediate and
// final reduce steps before re-serializing or finalizing. The merged columns
// are consumed by the caller and never leave the operator, so they come from
// the temporary memory resource.
cudf_velox::DecimalSumStateColumns mergeSerializedDecimalSumState(
    cudf::column_view inputCol,
    int32_t scale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const sumAgg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto sumAndCount =
      cudf_velox::deserializeDecimalSumState(inputCol, scale, stream);
  auto sumScalar = cudf::reduce(
      sumAndCount.sum->view(),
      *sumAgg,
      sumAndCount.sum->view().type(),
      stream,
      mr);
  auto countScalar = cudf::reduce(
      sumAndCount.count->view(),
      *sumAgg,
      cudf::data_type{cudf::type_id::INT64},
      stream,
      mr);
  return makeSumCountColumns(*sumScalar, *countScalar, stream, mr);
}

std::unique_ptr<cudf::column> intermediateDecimalMergeSerializedString(
    cudf::column_view inputCol,
    int32_t scale,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto merged = mergeSerializedDecimalSumState(inputCol, scale, stream, mr);
  return serializeDecimalPartialOrIntermediateState(
      std::move(merged.sum), std::move(merged.count), stream, mr);
}

std::unique_ptr<cudf::column> finalDecimalAvgFromSerializedString(
    cudf::column_view inputCol,
    int32_t scale,
    TypePtr const& resultType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto merged = mergeSerializedDecimalSumState(inputCol, scale, stream, mr);
  return finalizeDecimalAverage(
      std::move(merged.sum), std::move(merged.count), resultType, stream, mr);
}

std::unique_ptr<cudf::column> singleDecimalAvgFromRawColumn(
    cudf::column_view inputCol,
    TypePtr const& resultType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::unique_ptr<cudf::column> castedInput;
  inputCol = castDecimal64InputToDecimal128(inputCol, castedInput, stream);
  auto const sumAgg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto sumScalar =
      cudf::reduce(inputCol, *sumAgg, inputCol.type(), stream, get_temp_mr());
  auto countAgg = cudf::make_count_aggregation<cudf::reduce_aggregation>(
      cudf::null_policy::EXCLUDE);
  auto countScalar = cudf::reduce(
      inputCol,
      *countAgg,
      cudf::data_type{cudf::type_id::INT64},
      stream,
      get_temp_mr());
  auto cols = makeSumCountColumns(*sumScalar, *countScalar, stream, mr);
  return finalizeDecimalAverage(
      std::move(cols.sum), std::move(cols.count), resultType, stream, mr);
}

std::unique_ptr<cudf::column> singleOrRawDecimalSumWithCast(
    cudf::column_view inputCol,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto const sumAgg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto const cudfOutType = cudf_velox::veloxToCudfDataType(outputType);
  std::unique_ptr<cudf::column> castedInput;
  if (outputType->isDecimal() && inputCol.type() != cudfOutType) {
    castedInput = cudf::cast(inputCol, cudfOutType, stream, get_temp_mr());
    inputCol = castedInput->view();
  }
  auto const resultScalar =
      cudf::reduce(inputCol, *sumAgg, cudfOutType, stream, get_temp_mr());
  return cudf::make_column_from_scalar(*resultScalar, 1, stream, mr);
}

std::unique_ptr<cudf::column> reduceIntermediateDecimalFromSerializedColumn(
    cudf::column_view inputCol,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  validateIntermediateColumnType(inputCol);
  // outputType here could be DECIMAL or VARBINARY
  auto scale = outputType->isDecimal()
      ? getDecimalPrecisionScale(*outputType).second
      : 0;
  return intermediateDecimalMergeSerializedString(inputCol, scale, stream, mr);
}

std::unique_ptr<cudf::column> reduceFinalDecimalSumFromSerializedColumn(
    cudf::column_view inputCol,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  validateIntermediateColumnType(inputCol);
  auto scale = getDecimalPrecisionScale(*outputType).second;
  auto sumAndCount =
      cudf_velox::deserializeDecimalSumState(inputCol, scale, stream);
  return singleOrRawDecimalSumWithCast(
      sumAndCount.sum->view(), outputType, stream, mr);
}

std::unique_ptr<cudf::column> reduceFinalDecimalAvgFromSerializedColumn(
    cudf::column_view inputCol,
    TypePtr const& outputType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  validateIntermediateColumnType(inputCol);
  auto scale = getDecimalPrecisionScale(*outputType).second;
  return finalDecimalAvgFromSerializedString(
      inputCol, scale, outputType, stream, mr);
}

// Decimal SUM and AVG use dedicated aggregators rather than cudf::reduce's
// built-in sum/mean: partial/intermediate state is VARBINARY-encoded sum+count
// (see DecimalAggregationState), and the final divide needs decimal half-up
// rounding.
struct ReduceDecimalSumAggregator : ReduceAggregator {
  ReduceDecimalSumAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : ReduceAggregator(step, inputIndex, constant, resultType) {}

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      vector_size_t /* inputRowCount */,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    cudf::column_view inputCol = input.column(inputIndex);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return singleOrRawDecimalSumWithCast(inputCol, outputType, stream, mr);
      case core::AggregationNode::Step::kPartial:
        return partialDecimalSumCountToSerializedString(inputCol, stream, mr);
      case core::AggregationNode::Step::kIntermediate:
        return reduceIntermediateDecimalFromSerializedColumn(
            inputCol, outputType, stream, mr);
      case core::AggregationNode::Step::kFinal:
        return reduceFinalDecimalSumFromSerializedColumn(
            inputCol, outputType, stream, mr);
      default:
        VELOX_NYI("Unsupported aggregation step for decimal sum reduce");
    }
  }
};

struct ReduceDecimalAvgAggregator : ReduceAggregator {
  ReduceDecimalAvgAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType)
      : ReduceAggregator(step, inputIndex, constant, resultType) {}

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      vector_size_t /* inputRowCount */,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    cudf::column_view inputCol = input.column(inputIndex);
    switch (step) {
      case core::AggregationNode::Step::kSingle:
        return singleDecimalAvgFromRawColumn(inputCol, resultType, stream, mr);
      case core::AggregationNode::Step::kPartial:
        return partialDecimalSumCountToSerializedString(inputCol, stream, mr);
      case core::AggregationNode::Step::kIntermediate:
        return reduceIntermediateDecimalFromSerializedColumn(
            inputCol, outputType, stream, mr);
      case core::AggregationNode::Step::kFinal:
        VELOX_CHECK(
            *outputType == *resultType, "outputType/resultType mismatch");
        return reduceFinalDecimalAvgFromSerializedColumn(
            inputCol, outputType, stream, mr);
      default:
        VELOX_NYI("Unsupported aggregation step for decimal avg reduce");
    }
  }
};

struct ApproxDistinctAggregator : ReduceAggregator {
  static constexpr cudf::null_policy kNullPolicy = cudf::null_policy::EXCLUDE;
  static constexpr cudf::nan_policy kNanPolicy = cudf::nan_policy::NAN_IS_VALID;

  ApproxDistinctAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType,
      std::int32_t precision = 11) // Default 11 matches Velox's 2.3% standard
                                   // error (2^11 = 2048 buckets)
      : ReduceAggregator{step, inputIndex, constant, resultType},
        precision_{precision} {
    VELOX_CHECK(
        constant == nullptr,
        "ApproxDistinctAggregator does not support constant input");
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& outputType,
      vector_size_t /* inputRowCount */,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    if (exec::isRawInput(step)) {
      return doPartialReduce(input, stream, mr);
    } else if (step == core::AggregationNode::Step::kIntermediate) {
      return doIntermediateReduce(input, stream, mr);
    } else {
      return doFinalReduce(input, stream, mr);
    }
  }

 private:
  std::unique_ptr<cudf::column> makeSketchColumn(
      cuda::std::span<cuda::std::byte const> sketch_bytes,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto sketch_size = static_cast<cudf::size_type>(sketch_bytes.size());

    cudf::size_type offsets[2] = {0, sketch_size};
    rmm::device_buffer offsets_device{2 * sizeof(cudf::size_type), stream, mr};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        offsets_device.data(),
        offsets,
        2 * sizeof(cudf::size_type),
        cudaMemcpyHostToDevice,
        stream.value()));

    rmm::device_buffer chars_buffer{sketch_bytes.size(), stream, mr};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        chars_buffer.data(),
        sketch_bytes.data(),
        sketch_bytes.size(),
        cudaMemcpyDeviceToDevice,
        stream.value()));

    // Sync stream before stack-allocated offsets goes out of scope
    stream.synchronize();

    auto offsets_column = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        2,
        std::move(offsets_device),
        rmm::device_buffer{},
        0);

    return cudf::make_strings_column(
        1,
        std::move(offsets_column),
        std::move(chars_buffer),
        0,
        rmm::device_buffer{});
  }

  template <typename Func>
  auto mergeSketchesAndApply(
      cudf::column_view const& sketch_column,
      Func&& func,
      rmm::cuda_stream_view stream) {
    auto strings_col = cudf::strings_column_view(sketch_column);
    auto offsets_col = strings_col.offsets();
    auto chars_ptr = strings_col.chars_begin(stream);

    auto num_offsets = sketch_column.size() + 1;
    std::vector<cudf::size_type> host_offsets(num_offsets);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        host_offsets.data(),
        offsets_col.begin<cudf::size_type>(),
        num_offsets * sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize(); // Need host_offsets before proceeding

    cudf::size_type first_offset = host_offsets[0];
    cudf::size_type first_size = host_offsets[1] - first_offset;

    // Copy to mutable aligned buffer - cudf::approx_distinct_count requires
    // non-const span and proper alignment for int32 registers
    rmm::device_buffer aligned_sketch{
        static_cast<std::size_t>(first_size), stream};
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        aligned_sketch.data(),
        chars_ptr + first_offset,
        static_cast<std::size_t>(first_size),
        cudaMemcpyDeviceToDevice,
        stream.value()));

    cudf::approx_distinct_count merged_sketch(
        cuda::std::span<cuda::std::byte>(
            static_cast<cuda::std::byte*>(aligned_sketch.data()), first_size),
        precision_,
        kNullPolicy,
        kNanPolicy);

    for (cudf::size_type i = 1; i < sketch_column.size(); ++i) {
      cudf::size_type start_offset = host_offsets[i];
      cudf::size_type end_offset = host_offsets[i + 1];
      cudf::size_type size = end_offset - start_offset;

      if (size > 0) {
        rmm::device_buffer temp_sketch{static_cast<std::size_t>(size), stream};
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            temp_sketch.data(),
            chars_ptr + start_offset,
            size,
            cudaMemcpyDeviceToDevice,
            stream.value()));

        merged_sketch.merge(
            cuda::std::span<cuda::std::byte>(
                static_cast<cuda::std::byte*>(temp_sketch.data()), size),
            stream);
      }
    }

    return func(merged_sketch);
  }

  std::unique_ptr<cudf::column> doPartialReduce(
      cudf::table_view const& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto inputTable = cudf::table_view({input.column(inputIndex)});

    cudf::approx_distinct_count sketch{
        inputTable, precision_, kNullPolicy, kNanPolicy, stream};

    return makeSketchColumn(sketch.sketch(), stream, mr);
  }

  std::unique_ptr<cudf::column> doIntermediateReduce(
      cudf::table_view const& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto sketch_column = input.column(inputIndex);

    if (sketch_column.size() == 0) {
      return makeSketchColumn({}, stream, mr);
    }

    return mergeSketchesAndApply(
        sketch_column,
        [this, stream, mr](cudf::approx_distinct_count& sketch) {
          return makeSketchColumn(sketch.sketch(), stream, mr);
        },
        stream);
  }

  std::unique_ptr<cudf::column> doFinalReduce(
      cudf::table_view const& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    auto sketch_column = input.column(inputIndex);

    if (sketch_column.size() == 0) {
      return cudf::make_column_from_scalar(
          cudf::numeric_scalar<int64_t>(0, true, stream, get_temp_mr()),
          1,
          stream,
          mr);
    }

    return mergeSketchesAndApply(
        sketch_column,
        [stream, mr](cudf::approx_distinct_count& sketch) {
          std::size_t estimate = sketch.estimate(stream);
          return cudf::make_column_from_scalar(
              cudf::numeric_scalar<int64_t>(
                  static_cast<int64_t>(estimate), true, stream, get_temp_mr()),
              1,
              stream,
              mr);
        },
        stream);
  }

  std::int32_t precision_;
};

int64_t constantInt64Value(const core::TypedExprPtr& expr) {
  VELOX_CHECK(
      expr->isConstantKind(), "bloom_filter_agg extra arg must be constant");
  const auto* constant = expr->asUnchecked<core::ConstantTypedExpr>();
  VELOX_USER_CHECK(
      !constant->isNull(), "bloom_filter_agg extra argument cannot be null");

  auto readFromVector = [](const VectorPtr& vec) -> int64_t {
    switch (vec->typeKind()) {
      case TypeKind::TINYINT:
        return vec->as<SimpleVector<int8_t>>()->valueAt(0);
      case TypeKind::SMALLINT:
        return vec->as<SimpleVector<int16_t>>()->valueAt(0);
      case TypeKind::INTEGER:
        return vec->as<SimpleVector<int32_t>>()->valueAt(0);
      case TypeKind::BIGINT:
        return vec->as<SimpleVector<int64_t>>()->valueAt(0);
      default:
        VELOX_FAIL(
            "bloom_filter_agg extra argument must be integer, got {}",
            vec->type()->toString());
    }
  };

  if (constant->hasValueVector()) {
    return readFromVector(constant->valueVector());
  }
  switch (constant->type()->kind()) {
    case TypeKind::TINYINT:
      return constant->value().value<int8_t>();
    case TypeKind::SMALLINT:
      return constant->value().value<int16_t>();
    case TypeKind::INTEGER:
      return constant->value().value<int32_t>();
    case TypeKind::BIGINT:
      return constant->value().value<int64_t>();
    default:
      VELOX_FAIL(
          "bloom_filter_agg extra argument must be integer, got {}",
          constant->type()->toString());
  }
}

std::unique_ptr<cudf::column> makeHostBytesStringColumn(
    const std::string& data,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  const auto size = static_cast<cudf::size_type>(data.size());
  cudf::size_type offsets[2] = {0, size};
  rmm::device_buffer offsetsDevice{2 * sizeof(cudf::size_type), stream, mr};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      offsetsDevice.data(),
      offsets,
      2 * sizeof(cudf::size_type),
      cudaMemcpyHostToDevice,
      stream.value()));

  rmm::device_buffer chars{data.size(), stream, mr};
  if (!data.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        chars.data(),
        data.data(),
        data.size(),
        cudaMemcpyHostToDevice,
        stream.value()));
  }
  stream.synchronize();

  auto offsetsColumn = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      2,
      std::move(offsetsDevice),
      rmm::device_buffer{},
      0);
  return cudf::make_strings_column(
      1, std::move(offsetsColumn), std::move(chars), 0, rmm::device_buffer{});
}

std::unique_ptr<cudf::column> makeSerializedBloomColumn(
    const BloomFilter<>& bloom,
    bool initialized,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  if (!initialized) {
    cudf::string_scalar nullScalar("", false, stream, mr);
    return cudf::make_column_from_scalar(nullScalar, 1, stream, mr);
  }
  std::string data;
  data.resize(bloom.serializedSize());
  bloom.serialize(data.data());
  return makeHostBytesStringColumn(data, stream, mr);
}

struct BloomFilterAggAggregator : ReduceAggregator {
  BloomFilterAggAggregator(
      core::AggregationNode::Step step,
      uint32_t inputIndex,
      VectorPtr constant,
      const TypePtr& resultType,
      std::vector<core::TypedExprPtr> extraInputs,
      const core::QueryConfig* queryConfig)
      : ReduceAggregator(step, inputIndex, std::move(constant), resultType),
        extraInputs_(std::move(extraInputs)) {
    initCapacity(queryConfig);
  }

  std::unique_ptr<cudf::column> doReduce(
      cudf::table_view const& input,
      TypePtr const& /* outputType */,
      vector_size_t inputRowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) override {
    if (exec::isRawInput(step)) {
      return reduceRaw(input, inputRowCount, stream, mr);
    }
    return reduceMerge(input, stream, mr);
  }

 private:
  void initCapacity(const core::QueryConfig* queryConfig) {
    int64_t defaultExpectedNumItems = 1'000'000;
    int64_t defaultNumBits = 8'388'608;
    int64_t maxNumBits = 67'108'864;
    int64_t maxNumItems = 4'000'000;
    if (queryConfig != nullptr) {
      functions::sparksql::SparkQueryConfig spark{*queryConfig};
      defaultExpectedNumItems = spark.bloomFilterExpectedNumItems();
      defaultNumBits = spark.bloomFilterNumBits();
      maxNumBits = spark.bloomFilterMaxNumBits();
      maxNumItems = spark.bloomFilterMaxNumItems();
    }

    int64_t estimatedNumItems;
    int64_t numBits;
    if (extraInputs_.size() >= 2) {
      estimatedNumItems = constantInt64Value(extraInputs_[0]);
      numBits = constantInt64Value(extraInputs_[1]);
    } else if (extraInputs_.size() == 1) {
      estimatedNumItems = constantInt64Value(extraInputs_[0]);
      numBits = BloomFilter<>::optimalNumOfBits(estimatedNumItems, maxNumItems);
    } else {
      estimatedNumItems = defaultExpectedNumItems;
      numBits = defaultNumBits;
    }
    VELOX_USER_CHECK_GT(
        estimatedNumItems, 0, "estimatedNumItems must be positive");
    VELOX_USER_CHECK_GT(numBits, 0, "numBits must be positive");
    capacity_ = static_cast<int32_t>(std::min(numBits, maxNumBits) / 16);
  }

  std::unique_ptr<cudf::column> reduceRaw(
      cudf::table_view const& input,
      vector_size_t inputRowCount,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    BloomFilter<> bloom;
    bool initialized = false;

    auto insertValue = [&](int64_t value) {
      if (!initialized) {
        bloom.reset(capacity_);
        initialized = true;
      }
      bloom.insert(folly::hasher<int64_t>()(value));
    };

    if (constant != nullptr) {
      VELOX_USER_CHECK(
          !constant->isNullAt(0),
          "First argument of bloom_filter_agg cannot be null");
      VELOX_CHECK_EQ(constant->typeKind(), TypeKind::BIGINT);
      if (inputRowCount > 0) {
        insertValue(constant->as<SimpleVector<int64_t>>()->valueAt(0));
      }
      return makeSerializedBloomColumn(bloom, initialized, stream, mr);
    }

    VELOX_CHECK_GT(input.num_columns(), inputIndex);
    auto inputCol = input.column(inputIndex);
    const auto numRows = inputCol.size();
    if (numRows == 0) {
      return makeSerializedBloomColumn(bloom, false, stream, mr);
    }

    VELOX_CHECK(
        inputCol.type().id() == cudf::type_id::INT64 ||
            inputCol.type().id() == cudf::type_id::UINT64,
        "bloom_filter_agg raw input must be BIGINT");

    std::vector<int64_t> hostValues(static_cast<size_t>(numRows));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostValues.data(),
        inputCol.data<int64_t>(),
        static_cast<size_t>(numRows) * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value()));

    std::vector<cudf::bitmask_type> hostNulls;
    if (inputCol.nullable() && inputCol.null_mask() != nullptr) {
      const auto bytes = cudf::bitmask_allocation_size_bytes(numRows);
      hostNulls.resize(bytes / sizeof(cudf::bitmask_type));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostNulls.data(),
          inputCol.null_mask(),
          bytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    stream.synchronize();

    for (cudf::size_type i = 0; i < numRows; ++i) {
      if (!hostNulls.empty() && !cudf::bit_is_set(hostNulls.data(), i)) {
        VELOX_USER_FAIL("First argument of bloom_filter_agg cannot be null");
      }
      insertValue(hostValues[static_cast<size_t>(i)]);
    }
    return makeSerializedBloomColumn(bloom, initialized, stream, mr);
  }

  std::unique_ptr<cudf::column> reduceMerge(
      cudf::table_view const& input,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) {
    VELOX_CHECK_GT(input.num_columns(), inputIndex);
    auto sketchColumn = input.column(inputIndex);
    BloomFilter<> bloom;
    bool initialized = false;
    if (sketchColumn.size() == 0) {
      return makeSerializedBloomColumn(bloom, false, stream, mr);
    }

    auto stringsCol = cudf::strings_column_view(sketchColumn);
    auto offsetsCol = stringsCol.offsets();
    auto charsPtr = stringsCol.chars_begin(stream);
    const auto numOffsets = sketchColumn.size() + 1;
    std::vector<cudf::size_type> hostOffsets(static_cast<size_t>(numOffsets));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostOffsets.data(),
        offsetsCol.begin<cudf::size_type>(),
        static_cast<size_t>(numOffsets) * sizeof(cudf::size_type),
        cudaMemcpyDeviceToHost,
        stream.value()));

    std::vector<cudf::bitmask_type> hostNulls;
    if (sketchColumn.nullable() && sketchColumn.null_mask() != nullptr) {
      const auto bytes =
          cudf::bitmask_allocation_size_bytes(sketchColumn.size());
      hostNulls.resize(bytes / sizeof(cudf::bitmask_type));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostNulls.data(),
          sketchColumn.null_mask(),
          bytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    stream.synchronize();

    const auto charsSize =
        static_cast<size_t>(hostOffsets.back() - hostOffsets.front());
    std::vector<char> hostChars(charsSize);
    if (charsSize > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          hostChars.data(),
          charsPtr + hostOffsets.front(),
          charsSize,
          cudaMemcpyDeviceToHost,
          stream.value()));
      stream.synchronize();
    }

    const auto base = hostOffsets.front();
    for (cudf::size_type i = 0; i < sketchColumn.size(); ++i) {
      if (!hostNulls.empty() && !cudf::bit_is_set(hostNulls.data(), i)) {
        continue;
      }
      const auto start = hostOffsets[static_cast<size_t>(i)] - base;
      const auto end = hostOffsets[static_cast<size_t>(i) + 1] - base;
      if (end <= start) {
        continue;
      }
      bloom.merge(hostChars.data() + start);
      initialized = true;
    }
    return makeSerializedBloomColumn(bloom, initialized, stream, mr);
  }

  std::vector<core::TypedExprPtr> extraInputs_;
  int32_t capacity_{0};
};

std::unique_ptr<ReduceAggregator> createReduceAggregator(
    const ResolvedAggregateInfo& p,
    const core::QueryConfig* queryConfig) {
  auto const& kind = p.kind;
  auto prefix = cudf_velox::CudfConfig::getInstance().functionNamePrefix;
  if (kind.rfind(prefix + "sum", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<ReduceDecimalSumAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType);
    }
    return std::make_unique<ReduceSumAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "count", 0) == 0) {
    VELOX_CHECK(p.countInputKind.has_value());
    return std::make_unique<ReduceCountAggregator>(
        p.companionStep, p.inputIndex, *p.countInputKind, p.resultType);
  } else if (kind.rfind(prefix + "min", 0) == 0) {
    return std::make_unique<ReduceMinAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "max", 0) == 0) {
    return std::make_unique<ReduceMaxAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "avg", 0) == 0) {
    if (p.isDecimalAggregate) {
      return std::make_unique<ReduceDecimalAvgAggregator>(
          p.companionStep, p.inputIndex, p.constant, p.resultType);
    }
    return std::make_unique<ReduceMeanAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "approx_distinct", 0) == 0) {
    return std::make_unique<ApproxDistinctAggregator>(
        p.companionStep, p.inputIndex, p.constant, p.resultType);
  } else if (kind.rfind(prefix + "bloom_filter_agg", 0) == 0) {
    return std::make_unique<BloomFilterAggAggregator>(
        p.companionStep,
        p.inputIndex,
        p.constant,
        p.resultType,
        p.extraInputs,
        queryConfig);
  } else {
    VELOX_NYI("Reduce aggregation not yet supported, kind: {}", kind);
  }
}

} // namespace

namespace facebook::velox::cudf_velox {

std::vector<std::unique_ptr<ReduceAggregator>> toReduceAggregators(
    core::AggregationNode const& aggregationNode,
    core::AggregationNode::Step step,
    TypePtr const& outputType,
    std::vector<VectorPtr> const& constants,
    const core::QueryConfig* queryConfig) {
  auto params =
      resolveAggregateInfos(aggregationNode, step, outputType, constants);

  std::vector<std::unique_ptr<ReduceAggregator>> aggregators;
  aggregators.reserve(params.size());
  for (const auto& p : params) {
    aggregators.push_back(createReduceAggregator(p, queryConfig));
  }
  return aggregators;
}

bool canReduceAggregationBeEvaluatedByCudf(
    const core::CallTypedExpr& call,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    core::QueryCtx* queryCtx) {
  return canAggregationBeEvaluatedByRegistry(
      getReduceAggregationRegistry(), call, step, rawInputTypes, queryCtx);
}

bool canReduceBeEvaluatedByCudf(
    const core::AggregationNode& aggregationNode,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  const core::PlanNode* sourceNode = aggregationNode.sources().empty()
      ? nullptr
      : aggregationNode.sources()[0].get();

  // Get the aggregation step from the node
  auto step = aggregationNode.step();

  // Check supported aggregation functions using reduce registry
  for (const auto& aggregate : aggregationNode.aggregates()) {
    // Use step-aware validation that handles partial/final/intermediate steps
    if (!canReduceAggregationBeEvaluatedByCudf(
            *aggregate.call, step, aggregate.rawInputTypes, queryCtx)) {
      return false;
    }

    // `distinct` aggregations are not supported, in testing fails with "De-dup
    // before aggregation is not yet supported"
    if (aggregate.distinct) {
      return false;
    }

    // `mask` is NOT supported (in testing do not appear to be applied and
    // return incorrect results )
    if (aggregate.mask) {
      return false;
    }

    if (isCountFunctionName(aggregate.call->name())) {
      continue;
    }

    // Check input expressions can be evaluated by cuDF, expand the input first.
    for (const auto& input : aggregate.call->inputs()) {
      auto expandedInput = expandFieldReference(input, sourceNode);
      if (!canExprRunOnGpu(expandedInput, queryCtx, pool)) {
        return false;
      }
    }
  }

  return true;
}

CudfReduce::CudfReduce(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<core::AggregationNode const> const& aggregationNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          aggregationNode->outputType(),
          aggregationNode->id(),
          std::string{"CudfReduce"} +
              std::string{
                  core::AggregationNode::toName(aggregationNode->step())},
          nvtx3::rgb{34, 139, 34}, // Forest Green
          NvtxMethodFlag::kAddInput | NvtxMethodFlag::kGetOutput,
          std::nullopt,
          aggregationNode),
      aggregationNode_(aggregationNode),
      isPartialOutput_(
          exec::isPartialOutput(aggregationNode->step()) &&
          !hasFinalAggs(aggregationNode->aggregates())) {}

void CudfReduce::initialize() {
  Operator::initialize();

  inputType_ = aggregationNode_->sources()[0]->outputType();

  numAggregates_ = aggregationNode_->aggregates().size();
  const auto inputRowSchema = asRowType(inputType_);
  std::vector<column_index_t> emptyKeys;
  auto aggregationInput = buildAggregationInputChannels(
      *aggregationNode_, *operatorCtx_, inputRowSchema, emptyKeys);
  aggregationInputChannels_ = std::move(aggregationInput.channels);
  precomputedInputEvaluators_ = createAggregationInputEvaluators(
      aggregationInput.precomputedInputs, *operatorCtx_, inputRowSchema);
  aggregators_ = toReduceAggregators(
      *aggregationNode_,
      aggregationNode_->step(),
      outputType_,
      aggregationInput.constants,
      &operatorCtx_->driverCtx()->queryConfig());

  aggregationNode_.reset();
}

void CudfReduce::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  numInputRows_ += input->size();

  auto cudfInput = std::dynamic_pointer_cast<cudf_velox::CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput);

  inputs_.push_back(std::move(cudfInput));
}

CudfVectorPtr CudfReduce::doGlobalAggregation(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  std::vector<std::unique_ptr<cudf::column>> resultColumns;
  resultColumns.reserve(aggregators_.size());
  for (auto i = 0; i < aggregators_.size(); i++) {
    resultColumns.push_back(
        aggregators_[i]->doReduce(
            tableView, outputType_->childAt(i), numInputRows_, stream, mr));
  }

  return std::make_shared<cudf_velox::CudfVector>(
      pool(),
      outputType_,
      1,
      std::make_unique<cudf::table>(std::move(resultColumns)),
      stream);
}

RowVectorPtr CudfReduce::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  if (!isPartialOutput_ && !noMoreInput_) {
    // Final aggregation has to wait for all batches to arrive so we cannot
    // return any results here.
    return nullptr;
  }

  if (inputs_.empty() && !noMoreInput_) {
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();

  auto tbl = getConcatenatedTable(
      std::move(inputs_), inputType_, stream, get_temp_mr());

  // Release input data after synchronizing.
  stream.synchronize();
  inputs_.clear();

  if (noMoreInput_) {
    finished_ = true;
  }

  VELOX_CHECK_NOT_NULL(tbl);

  auto preparedInput = prepareAggregationInput(
      tbl->view(),
      tbl->num_rows(),
      precomputedInputEvaluators_,
      stream,
      get_temp_mr());
  auto tableView = preparedInput.tableView.num_columns() == 0
      ? preparedInput.tableView
      : preparedInput.tableView.select(
            aggregationInputChannels_.begin(), aggregationInputChannels_.end());
  auto output = doGlobalAggregation(tableView, stream, get_output_mr());
  if (isPartialOutput_ && !noMoreInput_) {
    numInputRows_ = 0;
  }
  return output;
}

void CudfReduce::doNoMoreInput() {
  Operator::noMoreInput();
  if (isPartialOutput_ && inputs_.empty()) {
    finished_ = true;
  }
}

bool CudfReduce::isFinished() {
  return finished_;
}

} // namespace facebook::velox::cudf_velox
