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

#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

namespace facebook::velox::cudf_velox {

/// Pipeline operator that performs cudf::partition() on GPU before CudfToVelox
/// D2H.  The first column of the input CudfVector is the hash value; this
/// operator computes PID = hash % numPartitions, replaces the first column with
/// the PID, and reorders rows by partition via cudf::partition().  The output is
/// a CudfVector whose rows are grouped by partition (first col = sorted PID).
///
/// Inserted by ToCudf::CompileState when the query config key
/// "cudf.shuffle_num_partitions" is > 0 and the operator is the last in the
/// output pipeline (i.e. feeding a shuffle exchange).
class CudfShufflePartition : public exec::Operator, public NvtxHelper {
 public:
  static constexpr const char* kShuffleNumPartitions =
      "cudf.shuffle_num_partitions";

  CudfShufflePartition(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId,
      int32_t numPartitions);

  std::string toString() const override {
    return fmt::format("CudfShufflePartition({})", numPartitions_);
  }

  bool needsInput() const override {
    return !finished_ && output_ == nullptr;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_ && output_ == nullptr;
  }

 private:
  const int32_t numPartitions_;
  CudfVectorPtr output_;
  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
