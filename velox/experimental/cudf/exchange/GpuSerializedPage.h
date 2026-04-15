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

#include "velox/common/base/Exceptions.h"
#include "velox/exec/SerializedPage.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

/// A SerializedPageBase subclass that wraps a CudfVector GPU pointer WITHOUT
/// serialization or D2H copy. GPU pages are unwrapped directly by the GPU
/// exchange consumer rather than deserialized through ByteInputStream.
class GpuSerializedPage : public exec::SerializedPageBase {
 public:
  explicit GpuSerializedPage(std::shared_ptr<CudfVector> cudfVector)
      : cudfVector_(std::move(cudfVector)) {
    VELOX_CHECK_NOT_NULL(
        cudfVector_, "GpuSerializedPage requires a non-null CudfVector");
  }

  ~GpuSerializedPage() override = default;

  /// Returns the GPU memory size of the underlying CudfVector.
  uint64_t size() const override {
    return cudfVector_->estimateFlatSize();
  }

  /// Returns the number of rows in the CudfVector.
  std::optional<int64_t> numRows() const override {
    return static_cast<int64_t>(cudfVector_->size());
  }

  /// GPU pages are not deserialized via ByteInputStream. Consumers should
  /// use cudfVector() to unwrap the GPU data directly.
  std::unique_ptr<ByteInputStream> prepareStreamForDeserialize() override {
    VELOX_UNSUPPORTED(
        "GpuSerializedPage does not support CPU deserialization. "
        "Use cudfVector() to access GPU data directly.");
  }

  /// No CPU IOBuf representation exists for GPU data.
  std::unique_ptr<folly::IOBuf> getIOBuf() const override {
    VELOX_UNSUPPORTED(
        "GpuSerializedPage does not have a CPU IOBuf representation. "
        "Use cudfVector() to access GPU data directly.");
  }

  /// Returns the underlying CudfVector.
  const std::shared_ptr<CudfVector>& cudfVector() const {
    return cudfVector_;
  }

 private:
  std::shared_ptr<CudfVector> cudfVector_;
};

} // namespace facebook::velox::cudf_velox
