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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/merge.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>

#include <malloc.h>
#include <unistd.h>

#include <atomic>
#include <filesystem>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kSortedRunBytes = 3ULL << 30;
constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
std::atomic<uint64_t> orderBySpillDirectorySequence{0};

std::unique_ptr<cudf::table> copyTableSlice(
    cudf::table_view input,
    cudf::size_type begin,
    cudf::size_type end,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  VELOX_CHECK_LE(begin, end);
  auto slices = cudf::slice(input, {begin, end}, stream);
  VELOX_CHECK_EQ(slices.size(), 1);
  return std::make_unique<cudf::table>(slices.front(), stream, mr);
}

cudf::size_type firstSearchPosition(
    cudf::column_view positions,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_EQ(positions.size(), 1);
  cudf::size_type result{0};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      &result,
      positions.data<cudf::size_type>(),
      sizeof(result),
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();
  return result;
}

} // namespace

CudfOrderBy::CudfOrderBy(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::OrderByNode>& orderByNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          orderByNode->outputType(),
          orderByNode->id(),
          "CudfOrderBy",
          nvtx3::rgb{64, 224, 208}, // Turquoise
          NvtxMethodFlag::kAll,
          std::nullopt,
          orderByNode),
      orderByNode_(orderByNode) {
  sortKeys_.reserve(orderByNode->sortingKeys().size());
  columnOrder_.reserve(orderByNode->sortingKeys().size());
  nullOrder_.reserve(orderByNode->sortingKeys().size());
  for (int i = 0; i < orderByNode->sortingKeys().size(); ++i) {
    const auto channel =
        exec::exprToChannel(orderByNode->sortingKeys()[i].get(), outputType_);
    VELOX_CHECK(
        channel != kConstantChannel,
        "OrderBy doesn't allow constant sorting keys");
    sortKeys_.push_back(channel);
    auto const& sortingOrder = orderByNode->sortingOrders()[i];
    columnOrder_.push_back(
        sortingOrder.isAscending() ? cudf::order::ASCENDING
                                   : cudf::order::DESCENDING);
    nullOrder_.push_back(
        (sortingOrder.isNullsFirst() ^ !sortingOrder.isAscending())
            ? cudf::null_order::BEFORE
            : cudf::null_order::AFTER);
  }
}

void CudfOrderBy::doAddInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput);
    bufferedBytes_ += cudfInput->estimateFlatSize();
    inputs_.push_back(std::move(cudfInput));
    if (bufferedBytes_ >= kSortedRunBytes) {
      spillSortedRun();
    }
  }
}

void CudfOrderBy::doNoMoreInput() {
  Operator::noMoreInput();

  if (spilled_ && !inputs_.empty()) {
    spillSortedRun();
  }
  if (spilled_) {
    initializeSortedRunReaders();
    return;
  }

  if (inputs_.empty()) {
    finished_ = true;
    return;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  // Using the output memory resource to allow spilling to CPU memory.
  auto tbl = getConcatenatedTable(
      std::exchange(inputs_, {}), outputType_, stream, get_output_mr());
  bufferedBytes_ = 0;

  VELOX_CHECK_NOT_NULL(tbl);

  auto keys = tbl->view().select(sortKeys_);
  auto values = tbl->view();
  auto result = cudf::sort_by_key(
      values, keys, columnOrder_, nullOrder_, stream, get_output_mr());
  auto const size = result->num_rows();
  outputTable_ = std::make_shared<CudfVector>(
      pool(), outputType_, size, std::move(result), stream);
}

RowVectorPtr CudfOrderBy::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (spilled_) {
    auto stream = cudfGlobalStreamPool().get_stream();
    auto result = mergeNextSortedBatch(stream, get_output_mr());
    if (!result || result->num_rows() == 0) {
      finished_ = true;
      cleanupSpillFiles();
      return nullptr;
    }
    return std::make_shared<CudfVector>(
        pool(), outputType_, result->num_rows(), std::move(result), stream);
  }
  finished_ = true;
  return std::exchange(outputTable_, nullptr);
}

void CudfOrderBy::spillSortedRun() {
  if (inputs_.empty()) {
    return;
  }
  namespace fs = std::filesystem;
  if (!spilled_) {
    const auto sequence = orderBySpillDirectorySequence.fetch_add(1);
    spillDirectory_ = (fs::temp_directory_path() /
                       fmt::format(
                           "velox-cudf-orderby-spill-{}-{}",
                           static_cast<int64_t>(::getpid()),
                           sequence))
                          .string();
    fs::create_directories(spillDirectory_);
    spilled_ = true;
  }

  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  auto input =
      getConcatenatedTable(std::exchange(inputs_, {}), outputType_, stream, mr);
  bufferedBytes_ = 0;
  auto sorted = cudf::sort_by_key(
      input->view(),
      input->view().select(sortKeys_),
      columnOrder_,
      nullOrder_,
      stream,
      mr);
  auto path = fmt::format(
      "{}/run-{:06}.parquet", spillDirectory_, spillFileSequence_++);
  auto options = cudf::io::parquet_writer_options::builder(
                     cudf::io::sink_info{path}, sorted->view())
                     .build();
  cudf::io::write_parquet(options, stream);
  sortedRuns_.push_back({std::move(path), nullptr});
  ::malloc_trim(0);
}

void CudfOrderBy::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  auto stream = cudfGlobalStreamPool().get_stream();
  auto mr = get_output_mr();
  for (auto& run : sortedRuns_) {
    auto options = cudf::io::parquet_reader_options::builder(
                       cudf::io::source_info{run.path})
                       .build();
    run.reader = std::make_unique<cudf::io::chunked_parquet_reader>(
        kMergeChunkBytes, 0, options, stream, mr);
  }
  readersInitialized_ = true;
}

std::unique_ptr<cudf::table> CudfOrderBy::mergeNextSortedBatch(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  while (!mergeFinished_) {
    std::vector<std::unique_ptr<cudf::table>> chunks;
    std::vector<cudf::table_view> mergeViews;
    std::vector<cudf::table_view> boundaryRows;
    if (mergeCarry_ && mergeCarry_->num_rows() > 0) {
      mergeViews.push_back(mergeCarry_->view());
    }
    for (auto& run : sortedRuns_) {
      if (!run.reader || !run.reader->has_next()) {
        continue;
      }
      auto chunk = run.reader->read_chunk();
      if (chunk.tbl->num_rows() == 0) {
        continue;
      }
      chunks.push_back(std::move(chunk.tbl));
      mergeViews.push_back(chunks.back()->view());
      if (run.reader->has_next()) {
        auto last = cudf::slice(
            chunks.back()->view(),
            {chunks.back()->num_rows() - 1, chunks.back()->num_rows()},
            stream);
        boundaryRows.push_back(last.front());
      }
    }
    if (mergeViews.empty()) {
      mergeFinished_ = true;
      return std::exchange(mergeCarry_, nullptr);
    }

    std::unique_ptr<cudf::table> merged = mergeViews.size() == 1
        ? std::make_unique<cudf::table>(mergeViews.front(), stream, mr)
        : cudf::merge(
              mergeViews, sortKeys_, columnOrder_, nullOrder_, stream, mr);
    mergeCarry_.reset();
    if (boundaryRows.empty()) {
      mergeFinished_ = true;
      return merged;
    }

    auto boundaryCandidates = cudf::concatenate(boundaryRows, stream, mr);
    auto sortedBoundaries = cudf::sort_by_key(
        boundaryCandidates->view(),
        boundaryCandidates->view().select(sortKeys_),
        columnOrder_,
        nullOrder_,
        stream,
        mr);
    auto boundary = cudf::slice(sortedBoundaries->view(), {0, 1}, stream);
    auto positions = cudf::upper_bound(
        merged->view().select(sortKeys_),
        boundary.front().select(sortKeys_),
        columnOrder_,
        nullOrder_,
        stream,
        mr);
    const auto safeEnd = firstSearchPosition(positions->view(), stream);
    mergeCarry_ =
        copyTableSlice(merged->view(), safeEnd, merged->num_rows(), stream, mr);
    if (safeEnd > 0) {
      return copyTableSlice(merged->view(), 0, safeEnd, stream, mr);
    }
  }
  return nullptr;
}

void CudfOrderBy::cleanupSpillFiles() {
  sortedRuns_.clear();
  mergeCarry_.reset();
  if (spillDirectory_.empty()) {
    return;
  }
  std::error_code error;
  std::filesystem::remove_all(spillDirectory_, error);
  spillDirectory_.clear();
  ::malloc_trim(0);
}

void CudfOrderBy::doClose() {
  Operator::close();
  // Release stored inputs
  // Release cudf memory resources
  inputs_.clear();
  outputTable_.reset();
  cleanupSpillFiles();
}
} // namespace facebook::velox::cudf_velox
