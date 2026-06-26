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

#include "velox/experimental/cudf/exec/CudfWindow.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/OperatorUtils.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <algorithm>

namespace facebook::velox::cudf_velox {
namespace {

bool isSupportedScalarWindowType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
      return true;
    default:
      return false;
  }
}

bool isFieldAccessExpr(const core::TypedExprPtr& expr) {
  return std::dynamic_pointer_cast<const core::FieldAccessTypedExpr>(expr) !=
      nullptr;
}

bool isDefaultRowNumberFrame(const core::WindowNode::Frame& frame) {
  return frame.startType ==
      core::WindowNode::BoundType::kUnboundedPreceding &&
      frame.endType == core::WindowNode::BoundType::kCurrentRow &&
      frame.startValue == nullptr && frame.endValue == nullptr;
}

bool isFullPartitionRowsFrame(const core::WindowNode::Frame& frame) {
  return frame.type == core::WindowNode::WindowType::kRows &&
      frame.startType == core::WindowNode::BoundType::kUnboundedPreceding &&
      frame.endType == core::WindowNode::BoundType::kUnboundedFollowing &&
      frame.startValue == nullptr && frame.endValue == nullptr;
}

bool isSupportedRankLikeFunction(const core::WindowNode::Function& function) {
  const auto& name = function.functionCall->name();
  if ((name != "row_number" && name != "rank") || function.ignoreNulls ||
      !function.functionCall->inputs().empty() ||
      !isDefaultRowNumberFrame(function.frame)) {
    return false;
  }

  const auto resultKind = function.functionCall->type()->kind();
  return resultKind == TypeKind::INTEGER || resultKind == TypeKind::BIGINT;
}

bool isSupportedSumInputType(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
      return true;
    default:
      return false;
  }
}

bool isSupportedFullPartitionSumFunction(
    const core::WindowNode::Function& function) {
  if (function.functionCall->name() != "sum" || function.ignoreNulls ||
      function.functionCall->inputs().size() != 1 ||
      !isFullPartitionRowsFrame(function.frame)) {
    return false;
  }

  const auto& input = function.functionCall->inputs()[0];
  return isFieldAccessExpr(input) && isSupportedSumInputType(input->type());
}

std::vector<cudf::column_view> selectColumns(
    cudf::table_view const& table,
    const std::vector<cudf::size_type>& channels) {
  std::vector<cudf::column_view> columns;
  columns.reserve(channels.size());
  for (const auto channel : channels) {
    columns.push_back(table.column(channel));
  }
  return columns;
}

} // namespace

bool isSupportedCudfWindowNode(
    const std::shared_ptr<const core::WindowNode>& node) {
  if (node == nullptr || node->windowFunctions().empty()) {
    return false;
  }

  bool hasRowNumber = false;
  bool hasFullPartitionSum = false;
  for (const auto& function : node->windowFunctions()) {
    if (isSupportedRankLikeFunction(function)) {
      hasRowNumber = true;
    } else if (isSupportedFullPartitionSumFunction(function)) {
      hasFullPartitionSum = true;
    } else {
      return false;
    }
  }

  if (hasRowNumber && node->sortingKeys().empty()) {
    return false;
  }

  if (hasRowNumber && hasFullPartitionSum) {
    return false;
  }

  for (const auto& key : node->partitionKeys()) {
    if (!isFieldAccessExpr(key) || !isSupportedScalarWindowType(key->type())) {
      return false;
    }
  }

  for (const auto& key : node->sortingKeys()) {
    if (!isFieldAccessExpr(key) || !isSupportedScalarWindowType(key->type())) {
      return false;
    }
  }

  return true;
}

CudfWindow::CudfWindow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowNode>& windowNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          windowNode->outputType(),
          windowNode->id(),
          "CudfWindow",
          nvtx3::rgb{123, 104, 238},
          NvtxMethodFlag::kAll,
          std::nullopt,
          windowNode),
      windowNode_(windowNode),
      inputType_(windowNode->sources()[0]->outputType()) {
  for (const auto& key : windowNode->partitionKeys()) {
    const auto channel = exec::exprToChannel(key.get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel, "Window partition key must be a column");
    partitionKeyChannels_.push_back(static_cast<cudf::size_type>(channel));
  }

  const auto& sortingKeys = windowNode->sortingKeys();
  const auto& sortingOrders = windowNode->sortingOrders();
  for (size_t i = 0; i < sortingKeys.size(); ++i) {
    const auto channel = exec::exprToChannel(sortingKeys[i].get(), inputType_);
    VELOX_CHECK(
        channel != kConstantChannel, "Window sorting key must be a column");
    sortKeyChannels_.push_back(static_cast<cudf::size_type>(channel));

    const auto& order = sortingOrders[i];
    sortOrders_.push_back(
        order.isAscending() ? cudf::order::ASCENDING
                            : cudf::order::DESCENDING);
    sortNullOrders_.push_back(
        (order.isNullsFirst() ^ !order.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfWindow::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }

  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfInput, "Expected CudfVector input");
  inputs_.push_back(std::move(cudfInput));
}

void CudfWindow::doNoMoreInput() {
  Operator::noMoreInput();
  if (inputs_.empty()) {
    finished_ = true;
  }
}

RowVectorPtr CudfWindow::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  if (inputs_.empty()) {
    finished_ = true;
    return nullptr;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  auto input = getConcatenatedTable(std::move(inputs_), inputType_, stream, mr);
  inputs_.clear();

  auto output = computeOutputTable(std::move(input), stream, mr);
  finished_ = true;
  if (output->num_rows() == 0) {
    return nullptr;
  }

  return std::make_shared<CudfVector>(
      pool(), outputType_, output->num_rows(), std::move(output), stream);
}

std::unique_ptr<cudf::column> CudfWindow::computeRowNumberColumn(
    cudf::table_view const& sortedInput,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  const auto valuesColumn = sortedInput.column(sortKeyChannels_[0]);
  const auto order = sortOrders_[0];
  const auto nullOrder = sortNullOrders_[0];

  std::unique_ptr<cudf::column> rowNumber;
  if (partitionKeyChannels_.empty()) {
    auto aggregation = cudf::make_rank_aggregation<cudf::scan_aggregation>(
        cudf::rank_method::FIRST,
        order,
        cudf::null_policy::INCLUDE,
        nullOrder);
    rowNumber = cudf::scan(
        valuesColumn,
        *aggregation,
        cudf::scan_type::INCLUSIVE,
        cudf::null_policy::INCLUDE,
        stream,
        mr);
  } else {
    auto partitionColumns = selectColumns(sortedInput, partitionKeyChannels_);
    std::vector<cudf::groupby::scan_request> requests(1);
    requests[0].values = valuesColumn;
    requests[0].aggregations.push_back(
        cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::FIRST,
            order,
            cudf::null_policy::INCLUDE,
            nullOrder));

    cudf::groupby::groupby grouper(
        cudf::table_view(partitionColumns),
        cudf::null_policy::INCLUDE,
        cudf::sorted::YES,
        std::vector<cudf::order>(
            partitionKeyChannels_.size(), cudf::order::ASCENDING),
        std::vector<cudf::null_order>(
            partitionKeyChannels_.size(), cudf::null_order::BEFORE));

    auto scanResult = grouper.scan(requests, stream, mr);
    VELOX_CHECK_EQ(scanResult.second.size(), 1);
    VELOX_CHECK_EQ(scanResult.second[0].results.size(), 1);
    rowNumber = std::move(scanResult.second[0].results[0]);
  }

  const auto expectedCudfType = veloxToCudfDataType(expectedType);
  if (rowNumber->type() != expectedCudfType) {
    rowNumber = cudf::cast(rowNumber->view(), expectedCudfType, stream, mr);
  }
  return rowNumber;
}

std::unique_ptr<cudf::column> CudfWindow::computeRankColumn(
    cudf::table_view const& sortedInput,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  auto rowNumber = computeRowNumberColumn(sortedInput, expectedType, stream, mr);

  std::vector<cudf::size_type> rankKeyChannels = partitionKeyChannels_;
  rankKeyChannels.insert(
      rankKeyChannels.end(), sortKeyChannels_.begin(), sortKeyChannels_.end());
  VELOX_CHECK(!rankKeyChannels.empty(), "Window rank requires sort keys");

  auto rankKeyColumns = selectColumns(sortedInput, rankKeyChannels);
  cudf::groupby::groupby grouper(
      cudf::table_view(rankKeyColumns), cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = rowNumber->view();
  requests[0].aggregations.push_back(
      cudf::make_min_aggregation<cudf::groupby_aggregation>());

  auto [groupKeys, results] = grouper.aggregate(requests, stream, mr);
  VELOX_CHECK_EQ(results.size(), 1);
  VELOX_CHECK_EQ(results[0].results.size(), 1);
  auto minRankByPeerGroup = std::move(results[0].results[0]);

  cudf::hash_join lookup(
      groupKeys->view(),
      cudf::nullable_join::YES,
      cudf::null_equality::EQUAL,
      0.5,
      stream);

  auto [leftJoinIndices, rightJoinIndices] = lookup.left_join(
      cudf::table_view(rankKeyColumns),
      static_cast<std::size_t>(sortedInput.num_rows()),
      stream,
      mr);

  auto leftIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*leftJoinIndices}};
  auto rightIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

  auto gatheredRanks = cudf::gather(
      cudf::table_view{{minRankByPeerGroup->view()}},
      rightIndicesCol,
      cudf::out_of_bounds_policy::NULLIFY,
      stream,
      mr);
  auto gatheredColumns = gatheredRanks->release();
  VELOX_CHECK_EQ(gatheredColumns.size(), 1);

  auto target = cudf::allocate_like(
      gatheredColumns[0]->view(),
      sortedInput.num_rows(),
      cudf::mask_allocation_policy::RETAIN,
      stream,
      mr);
  auto scattered = cudf::scatter(
      cudf::table_view{{gatheredColumns[0]->view()}},
      leftIndicesCol,
      cudf::table_view{{target->view()}},
      stream,
      mr);
  return std::move(scattered->release()[0]);
}

std::unique_ptr<cudf::column> CudfWindow::computeFullPartitionSumColumn(
    cudf::table_view const& input,
    const core::WindowNode::Function& function,
    const TypePtr& expectedType,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  VELOX_CHECK(!partitionKeyChannels_.empty());

  const auto valueChannel =
      exec::exprToChannel(function.functionCall->inputs()[0].get(), inputType_);
  VELOX_CHECK(
      valueChannel != kConstantChannel,
      "Window sum input must be a column");

  auto partitionColumns = selectColumns(input, partitionKeyChannels_);

  cudf::groupby::groupby grouper(
      cudf::table_view(partitionColumns), cudf::null_policy::INCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = input.column(valueChannel);
  requests[0].aggregations.push_back(
      cudf::make_sum_aggregation<cudf::groupby_aggregation>());

  auto [groupKeys, results] = grouper.aggregate(requests, stream, mr);
  VELOX_CHECK_EQ(results.size(), 1);
  VELOX_CHECK_EQ(results[0].results.size(), 1);

  auto sumByGroup = std::move(results[0].results[0]);
  const auto expectedCudfType = veloxToCudfDataType(expectedType);
  if (sumByGroup->type() != expectedCudfType) {
    sumByGroup = cudf::cast(sumByGroup->view(), expectedCudfType, stream, mr);
  }

  cudf::hash_join lookup(
      groupKeys->view(),
      cudf::nullable_join::YES,
      cudf::null_equality::EQUAL,
      0.5,
      stream);

  auto [leftJoinIndices, rightJoinIndices] = lookup.left_join(
      cudf::table_view(partitionColumns),
      static_cast<std::size_t>(input.num_rows()),
      stream,
      mr);

  auto leftIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*leftJoinIndices}};
  auto rightIndicesCol = cudf::column_view{
      cudf::device_span<cudf::size_type const>{*rightJoinIndices}};

  auto gatheredSums = cudf::gather(
      cudf::table_view{{sumByGroup->view()}},
      rightIndicesCol,
      cudf::out_of_bounds_policy::NULLIFY,
      stream,
      mr);
  auto gatheredColumns = gatheredSums->release();
  VELOX_CHECK_EQ(gatheredColumns.size(), 1);

  auto target = cudf::allocate_like(
      gatheredColumns[0]->view(),
      input.num_rows(),
      cudf::mask_allocation_policy::RETAIN,
      stream,
      mr);
  auto scattered = cudf::scatter(
      cudf::table_view{{gatheredColumns[0]->view()}},
      leftIndicesCol,
      cudf::table_view{{target->view()}},
      stream,
      mr);
  return std::move(scattered->release()[0]);
}

std::unique_ptr<cudf::table> CudfWindow::computeOutputTable(
    std::unique_ptr<cudf::table> input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const {
  const auto inputSize = inputType_->size();
  const auto numFunctions = windowNode_->windowFunctions().size();

  if (input->num_rows() == 0) {
    auto columns = input->release();
    for (size_t i = 0; i < numFunctions; ++i) {
      columns.push_back(
          cudf::make_empty_column(
              veloxToCudfDataType(outputType_->childAt(inputSize + i))));
    }
    return std::make_unique<cudf::table>(std::move(columns));
  }

  const auto inputView = input->view();
  const auto hasFullPartitionSum = std::all_of(
      windowNode_->windowFunctions().begin(),
      windowNode_->windowFunctions().end(),
      isSupportedFullPartitionSumFunction);
  if (hasFullPartitionSum) {
    std::vector<std::unique_ptr<cudf::column>> resultColumns;
    resultColumns.reserve(numFunctions);
    for (size_t i = 0; i < numFunctions; ++i) {
      resultColumns.push_back(computeFullPartitionSumColumn(
          inputView,
          windowNode_->windowFunctions()[i],
          outputType_->childAt(inputSize + i),
          stream,
          mr));
    }

    auto columns = input->release();
    for (auto& column : resultColumns) {
      columns.push_back(std::move(column));
    }
    return std::make_unique<cudf::table>(std::move(columns));
  }

  std::vector<cudf::column_view> sortKeyViews;
  std::vector<cudf::order> columnOrders;
  std::vector<cudf::null_order> nullOrders;

  for (const auto channel : partitionKeyChannels_) {
    sortKeyViews.push_back(inputView.column(channel));
    columnOrders.push_back(cudf::order::ASCENDING);
    nullOrders.push_back(cudf::null_order::BEFORE);
  }
  for (size_t i = 0; i < sortKeyChannels_.size(); ++i) {
    sortKeyViews.push_back(inputView.column(sortKeyChannels_[i]));
    columnOrders.push_back(sortOrders_[i]);
    nullOrders.push_back(sortNullOrders_[i]);
  }

  std::unique_ptr<cudf::column> sortedOrder;
  std::unique_ptr<cudf::table> sortedTable;
  cudf::table_view sortedView;

  if (windowNode_->inputsSorted()) {
    sortedView = inputView;
  } else {
    sortedOrder = cudf::stable_sorted_order(
        cudf::table_view(sortKeyViews), columnOrders, nullOrders, stream, mr);
    sortedTable = cudf::gather(
        inputView,
        sortedOrder->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        cudf::negative_index_policy::NOT_ALLOWED,
        stream,
        mr);
    sortedView = sortedTable->view();
  }

  std::vector<std::unique_ptr<cudf::column>> resultColumns;
  resultColumns.reserve(numFunctions);
  for (size_t i = 0; i < numFunctions; ++i) {
    const auto& function = windowNode_->windowFunctions()[i];
    const auto& functionName = function.functionCall->name();
    if (functionName == "rank") {
      resultColumns.push_back(computeRankColumn(
          sortedView, outputType_->childAt(inputSize + i), stream, mr));
    } else {
      resultColumns.push_back(computeRowNumberColumn(
          sortedView, outputType_->childAt(inputSize + i), stream, mr));
    }
  }

  if (sortedOrder) {
    for (auto& column : resultColumns) {
      auto target = cudf::allocate_like(
          column->view(),
          column->size(),
          cudf::mask_allocation_policy::RETAIN,
          stream,
          mr);
      auto scattered = cudf::scatter(
          cudf::table_view{{column->view()}},
          sortedOrder->view(),
          cudf::table_view{{target->view()}},
          stream,
          mr);
      column = std::move(scattered->release()[0]);
    }
  }

  auto columns = input->release();
  for (auto& column : resultColumns) {
    columns.push_back(std::move(column));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // namespace facebook::velox::cudf_velox
