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

#include "velox/experimental/cudf/exec/CudfShufflePartition.h"
#include "velox/experimental/cudf/exec/GpuGuard.h"

#include <cudf/binaryop.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>

namespace facebook::velox::cudf_velox {

CudfShufflePartition::CudfShufflePartition(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId,
    int32_t numPartitions)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "CudfShufflePartition"),
      NvtxHelper(
          nvtx3::rgb{0, 191, 255}, // Deep sky blue
          operatorId,
          fmt::format("[{}]", planNodeId)),
      numPartitions_(numPartitions) {
  VELOX_CHECK_GT(numPartitions_, 0, "numPartitions must be > 0");
}

void CudfShufflePartition::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  GpuGuard gpuGuard;
  VELOX_CHECK_NULL(output_, "Previous output not consumed");

  auto cudfVec = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVec, "Input must be a CudfVector");

  auto tableView = cudfVec->getTableView();
  auto stream = cudfVec->stream();
  VELOX_CHECK_GT(
      tableView.num_columns(),
      0,
      "Input table must have at least one column (hash)");

  // First column is the hash value produced by the exchange hash expression.
  auto firstCol = tableView.column(0);

  gpuTimer_.start(stream);

  // PID = hash % numPartitions  (on GPU)
  auto numPartScalar = cudf::numeric_scalar<int32_t>(numPartitions_, true, stream);
  auto pidCol = cudf::binary_operation(
      firstCol,
      numPartScalar,
      cudf::binary_operator::PYMOD,
      cudf::data_type{cudf::type_id::INT32},
      stream);

  // Build table with PID replacing hash as column 0.
  std::vector<cudf::column_view> cols;
  cols.reserve(tableView.num_columns());
  cols.push_back(pidCol->view());
  for (cudf::size_type i = 1; i < tableView.num_columns(); ++i) {
    cols.push_back(tableView.column(i));
  }
  cudf::table_view fullTable(cols);

  // cudf::partition reorders rows so that rows with the same PID are
  // contiguous.  offsets[i] is the start index of partition i.
  auto [partitioned, offsets] = cudf::partition(
      fullTable,
      pidCol->view(),
      static_cast<cudf::size_type>(numPartitions_),
      stream);
  VELOX_CHECK_EQ(
      offsets.size(),
      static_cast<size_t>(numPartitions_) + 1,
      "cudf::partition must return numPartitions+1 offsets");

  gpuTimer_.stop(stream);

  // Sync before locals go out of scope — cudf::partition reads from pidCol
  // and input tableView asynchronously. When addInput() returns, pidCol is
  // destroyed and cudfVec may be the last reference to the input CudfVector.
  // RMM pool deallocate is immediate, so device buffers would be freed while
  // the async partition kernel is still reading.
  stream.synchronize();
  output_ = std::make_shared<CudfVector>(
      pool(),
      outputType_,
      partitioned->num_rows(),
      std::move(partitioned),
      stream);
}

RowVectorPtr CudfShufflePartition::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (!output_) {
    finished_ = noMoreInput_;
    return nullptr;
  }
  auto result = std::move(output_);
  finished_ = noMoreInput_;
  return result;
}

} // namespace facebook::velox::cudf_velox
