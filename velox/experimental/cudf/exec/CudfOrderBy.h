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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <string>

namespace facebook::velox::cudf_velox {

class CudfOrderBy : public CudfOperatorBase {
 public:
  CudfOrderBy(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      const std::shared_ptr<const core::OrderByNode>& orderByNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  void spillSortedRun();
  void initializeSortedRunReaders();
  std::unique_ptr<cudf::table> mergeNextSortedBatch(
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr);
  void cleanupSpillFiles();

  CudfVectorPtr outputTable_;
  std::shared_ptr<const core::OrderByNode> orderByNode_;
  std::vector<CudfVectorPtr> inputs_;
  std::vector<cudf::size_type> sortKeys_;
  std::vector<cudf::order> columnOrder_;
  std::vector<cudf::null_order> nullOrder_;
  uint64_t bufferedBytes_{0};
  struct SortedRun {
    std::string path;
    std::unique_ptr<cudf::io::chunked_parquet_reader> reader;
  };
  std::vector<SortedRun> sortedRuns_;
  std::string spillDirectory_;
  uint64_t spillFileSequence_{0};
  std::unique_ptr<cudf::table> mergeCarry_;
  bool readersInitialized_{false};
  bool mergeFinished_{false};
  bool spilled_{false};
  bool finished_{false};
};

} // namespace facebook::velox::cudf_velox
