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

#include "velox/experimental/cudf/exec/AggregationRegistry.h"
#include "velox/experimental/cudf/exec/SparkAggregateFunctions.h"

#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::cudf_velox {

void registerSparkAggregateFunctions(const std::string& prefix) {
  using exec::FunctionSignatureBuilder;

  unregisterAggregateFunctions();
  registerCommonAggregationFunctions(getGroupbyAggregationRegistry(), prefix);
  registerCommonAggregationFunctions(getReduceAggregationRegistry(), prefix);
  registerGroupbyOnlyAggregationFunctions(
      getGroupbyAggregationRegistry(), prefix);
  registerReduceOnlyAggregationFunctions(
      getReduceAggregationRegistry(), prefix);

  // Spark: SUM(REAL) -> DOUBLE, AVG(REAL) -> DOUBLE
  appendGroupbyAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  appendGroupbyAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kPartial,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "sum",
      core::AggregationNode::Step::kPartial,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  // SUM final/intermediate: DOUBLE->DOUBLE already registered.

  appendGroupbyAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  appendReduceAggregationFunctionForStep(
      prefix + "avg",
      core::AggregationNode::Step::kSingle,
      FunctionSignatureBuilder()
          .returnType("double")
          .argumentType("real")
          .build());
  // AVG final: row(DOUBLE,BIGINT)->DOUBLE already registered.

  auto collectionRawSignature = FunctionSignatureBuilder()
                                    .typeVariable("T")
                                    .returnType("array(T)")
                                    .argumentType("T")
                                    .build();
  auto collectionMergeSignature = FunctionSignatureBuilder()
                                      .typeVariable("T")
                                      .returnType("array(T)")
                                      .argumentType("array(T)")
                                      .build();
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_set",
      core::AggregationNode::Step::kSingle,
      collectionRawSignature);
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_set",
      core::AggregationNode::Step::kPartial,
      collectionRawSignature);
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_set",
      core::AggregationNode::Step::kIntermediate,
      collectionMergeSignature);
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_set",
      core::AggregationNode::Step::kFinal,
      collectionMergeSignature);

  // Spark collect_list ignores null input values and preserves duplicates.
  // The Velox-cuDF adapter implements it only for grouped aggregation, so do
  // not advertise these signatures in the reduce registry.
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_list",
      core::AggregationNode::Step::kSingle,
      collectionRawSignature);
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_list",
      core::AggregationNode::Step::kPartial,
      collectionRawSignature);
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_list",
      core::AggregationNode::Step::kIntermediate,
      collectionMergeSignature);
  appendGroupbyAggregationFunctionForStep(
      prefix + "collect_list",
      core::AggregationNode::Step::kFinal,
      collectionMergeSignature);

  // Spark runtime bloom filters are uncorrelated scalar subqueries, so the
  // native path is a global CudfReduce. Do not advertise these signatures in
  // the groupby registry: grouped bloom_filter_agg is not implemented.
  auto bloomRawOneArg = FunctionSignatureBuilder()
                            .returnType("varbinary")
                            .argumentType("bigint")
                            .build();
  auto bloomRawTwoArg = FunctionSignatureBuilder()
                            .returnType("varbinary")
                            .argumentType("bigint")
                            .constantArgumentType("bigint")
                            .build();
  auto bloomRawThreeArg = FunctionSignatureBuilder()
                              .returnType("varbinary")
                              .argumentType("bigint")
                              .constantArgumentType("bigint")
                              .constantArgumentType("bigint")
                              .build();
  auto bloomMerge = FunctionSignatureBuilder()
                        .returnType("varbinary")
                        .argumentType("varbinary")
                        .build();
  for (auto step : {
           core::AggregationNode::Step::kPartial,
           core::AggregationNode::Step::kSingle,
       }) {
    appendReduceAggregationFunctionForStep(
        prefix + "bloom_filter_agg", step, bloomRawOneArg);
    appendReduceAggregationFunctionForStep(
        prefix + "bloom_filter_agg", step, bloomRawTwoArg);
    appendReduceAggregationFunctionForStep(
        prefix + "bloom_filter_agg", step, bloomRawThreeArg);
  }
  appendReduceAggregationFunctionForStep(
      prefix + "bloom_filter_agg",
      core::AggregationNode::Step::kIntermediate,
      bloomMerge);
  appendReduceAggregationFunctionForStep(
      prefix + "bloom_filter_agg",
      core::AggregationNode::Step::kFinal,
      bloomMerge);
}

} // namespace facebook::velox::cudf_velox
