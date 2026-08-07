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

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <deque>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudfFromVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kGpuBatchSizeRows =
      "velox.cudf.gpu_batch_size_rows";
  static constexpr const char* kMaxBatchBytes =
      "velox.cudf.from_velox.max_batch_bytes";

  CudfFromVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  bool needsInput() const override {
    return !finished_ &&
        (inputs_.empty() ||
         (currentOutputSize_ < preferredBatchRows() &&
          currentOutputBytes_ < maxBatchBytes()));
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
  void doClose() override;

 private:
  vector_size_t preferredBatchRows() const;
  uint64_t maxBatchBytes() const;
  const std::optional<std::string> timestampTimeZone_;
  std::vector<RowVectorPtr> inputs_;
  std::size_t currentOutputSize_ = 0;
  uint64_t currentOutputBytes_ = 0;
  bool finished_ = false;
};

class CudfToVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kPassthroughMode =
      "velox.cudf.to_velox.passthrough_mode";
  // Hard byte target for one GPU-to-host conversion. Arrow's regular
  // StringArray uses signed 32-bit offsets, so converting an arbitrarily
  // large CudfVector before slicing can fail before a CPU-side slice is even
  // possible. MPP sets this to its GPU target batch bytes.
  static constexpr const char* kMaxBatchBytes =
      "velox.cudf.to_velox.max_batch_bytes";

  CudfToVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  bool needsInput() const override {
    // Keep at most one upstream CudfVector live. In particular, don't let
    // downstream backpressure turn this conversion boundary into an
    // unbounded device/host queue.
    return !finished_ && !cudfBuffer_ && inputs_.empty();
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
  void doClose() override;

 private:
  bool isPassthroughMode() const;
  uint64_t maxBatchBytes() const;
  vector_size_t nextBatchRows() const;
  RowVectorPtr convertNextSliceToVelox();
  std::deque<CudfVectorPtr> inputs_;
  // Keep the device input and drain byte-bounded table views from it. Slicing
  // after conversion is too late for Arrow string buffers larger than 2 GiB
  // and retains the complete wide RowVector on host.
  CudfVectorPtr cudfBuffer_;
  vector_size_t cudfOffset_{0};
  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
