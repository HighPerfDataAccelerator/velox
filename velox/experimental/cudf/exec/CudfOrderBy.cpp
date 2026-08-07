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
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/Utilities.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/merge.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>

#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <utility>

namespace facebook::velox::cudf_velox {
namespace {

constexpr uint64_t kMergeChunkBytes = 32ULL << 20;
constexpr uint64_t kHostPackChunkBytes = 32ULL << 20;
constexpr size_t kFinalMergeRuns = 2;
constexpr uint64_t kRawPackedMagic = 0x314b434150464455ULL; // "UDFPACK1"
constexpr uint32_t kRawPackedVersion = 1;

struct RawPackedHeader {
  uint64_t magic{kRawPackedMagic};
  uint32_t version{kRawPackedVersion};
  int32_t rows{0};
  uint64_t dataBytes{0};
};

std::atomic<uint64_t> orderBySpillDirectorySequence{0};
std::atomic<uint64_t> testingSortedRunBytes{0};
std::atomic<size_t> testingMergeFanIn{0};
std::atomic<uint64_t> testingHostSpillBytes{0};
std::atomic<uint64_t> mergeChunkBytes{kMergeChunkBytes};
std::atomic<uint64_t> testingOutputChunkBytes{0};
std::atomic<cudf::size_type> testingMaxOutputRows{0};

// Read-only diagnostics used to assert the external-sort bounds in GPU tests.
// They never participate in operator scheduling or correctness.
std::atomic<uint64_t> observedMaxActiveRuns{0};
std::atomic<uint64_t> observedSourceChunks{0};
std::atomic<uint64_t> observedMergeOutputBatches{0};
std::atomic<uint64_t> observedEmittedChunks{0};
std::atomic<uint64_t> observedSpillCleanups{0};
// Shared by every CudfOrderBy in one executor process. This prevents four
// concurrent RANGE buckets from each treating the complete configured host
// allowance as private memory.
std::atomic<uint64_t> orderByHostSpillBytesInUse{0};

bool reserveHostSpillBytes(uint64_t bytes, uint64_t limit) {
  if (bytes == 0) {
    return true;
  }
  if (limit == 0 || bytes > limit) {
    return false;
  }
  auto current = orderByHostSpillBytesInUse.load(std::memory_order_relaxed);
  while (current <= limit - bytes) {
    if (orderByHostSpillBytesInUse.compare_exchange_weak(
            current,
            current + bytes,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

void releaseHostSpillBytes(uint64_t bytes) noexcept {
  if (bytes > 0) {
    orderByHostSpillBytesInUse.fetch_sub(bytes, std::memory_order_acq_rel);
  }
}

bool isSpillSafeType(const TypePtr& type) {
  if (type == nullptr || type->providesCustomComparison()) {
    return false;
  }

  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
      return true;
    case TypeKind::INTEGER:
      // DATE is supported, but interval year-month has not been validated
      // through the internal Parquet representation.
      return !type->isIntervalYearMonth();
    case TypeKind::BIGINT:
      // Plain BIGINT and short DECIMAL are supported. Interval day-time has
      // not been validated through the internal Parquet representation.
      return !type->isIntervalDayTime();
    case TypeKind::HUGEINT:
      return type->isDecimal();
    case TypeKind::TIMESTAMP:
      switch (CudfConfig::getInstance().timestampUnit) {
        case cudf::type_id::TIMESTAMP_MILLISECONDS:
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
        case cudf::type_id::TIMESTAMP_NANOSECONDS:
          return true;
        default:
          // Parquet changes TIMESTAMP_SECONDS to milliseconds unless the
          // reader is given an explicit target, making spill data-dependent.
          return false;
      }
    case TypeKind::ARRAY:
      return type->size() == 1 && isSpillSafeType(type->childAt(0));
    case TypeKind::ROW:
      if (type->size() == 0) {
        return false;
      }
      for (uint32_t i = 0; i < type->size(); ++i) {
        if (!isSpillSafeType(type->childAt(i))) {
          return false;
        }
      }
      return true;
    case TypeKind::MAP:
      return type->size() == 2 && isSpillSafeType(type->childAt(0)) &&
          isSpillSafeType(type->childAt(1));
    case TypeKind::VARBINARY:
    case TypeKind::UNKNOWN:
    case TypeKind::FUNCTION:
    case TypeKind::OPAQUE:
    case TypeKind::INVALID:
      return false;
  }
  return false;
}

bool isSupportedSortKeyType(const TypePtr& type) {
  if (!isSpillSafeType(type) || !type->isOrderable()) {
    return false;
  }

  // Nested cuDF sort semantics have not been validated against Velox. Nested
  // values may still be carried as payload when every leaf is spill-safe.
  return type->kind() != TypeKind::ARRAY && type->kind() != TypeKind::MAP &&
      type->kind() != TypeKind::ROW;
}

void updateAtomicMax(std::atomic<uint64_t>& target, uint64_t value) {
  auto current = target.load();
  while (current < value && !target.compare_exchange_weak(current, value)) {
  }
}

void resetTestingStats() {
  observedMaxActiveRuns.store(0);
  observedSourceChunks.store(0);
  observedMergeOutputBatches.store(0);
  observedEmittedChunks.store(0);
  observedSpillCleanups.store(0);
}

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

CudfOrderBy::HostPackedChunk::~HostPackedChunk() {
  reset();
}

CudfOrderBy::HostPackedChunk::HostPackedChunk(
    HostPackedChunk&& other) noexcept
    : metadata(std::move(other.metadata)),
      data(std::move(other.data)),
      dataBytes(std::exchange(other.dataBytes, 0)),
      rows(std::exchange(other.rows, 0)) {}

CudfOrderBy::HostPackedChunk& CudfOrderBy::HostPackedChunk::operator=(
    HostPackedChunk&& other) noexcept {
  if (this != &other) {
    reset();
    metadata = std::move(other.metadata);
    data = std::move(other.data);
    dataBytes = std::exchange(other.dataBytes, 0);
    rows = std::exchange(other.rows, 0);
  }
  return *this;
}

void CudfOrderBy::HostPackedChunk::reset() noexcept {
  metadata.reset();
  data.reset();
  releaseHostSpillBytes(std::exchange(dataBytes, 0));
  rows = 0;
}

bool CudfOrderBy::isSupported(
    const RowTypePtr& outputType,
    const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys) {
  if (outputType == nullptr || sortingKeys.empty() ||
      !isSpillSafeType(outputType)) {
    return false;
  }

  return std::all_of(
      sortingKeys.begin(), sortingKeys.end(), [](const auto& key) {
        return key != nullptr && isSupportedSortKeyType(key->type());
      });
}

bool CudfOrderBy::isSupported(
    const std::shared_ptr<const core::OrderByNode>& orderByNode) {
  return orderByNode != nullptr &&
      isSupported(orderByNode->outputType(), orderByNode->sortingKeys());
}

void CudfOrderBy::testingSetMemoryLimits(
    uint64_t runBytes,
    uint64_t chunkBytes,
    uint64_t outputBytes,
    cudf::size_type outputRows,
    size_t fanIn,
    uint64_t hostBytes) {
  VELOX_CHECK_GT(runBytes, 0);
  VELOX_CHECK_GT(chunkBytes, 0);
  VELOX_CHECK_GT(outputBytes, 0);
  VELOX_CHECK_GT(outputRows, 0);
  VELOX_CHECK_GE(fanIn, 2);
  testingSortedRunBytes.store(runBytes);
  testingMergeFanIn.store(fanIn);
  testingHostSpillBytes.store(hostBytes);
  mergeChunkBytes.store(chunkBytes);
  testingOutputChunkBytes.store(outputBytes);
  testingMaxOutputRows.store(outputRows);
  resetTestingStats();
}

void CudfOrderBy::testingResetMemoryLimits() {
  testingSortedRunBytes.store(0);
  testingMergeFanIn.store(0);
  testingHostSpillBytes.store(0);
  mergeChunkBytes.store(kMergeChunkBytes);
  testingOutputChunkBytes.store(0);
  testingMaxOutputRows.store(0);
  resetTestingStats();
}

uint64_t CudfOrderBy::testingMaxActiveRuns() {
  return observedMaxActiveRuns.load();
}

uint64_t CudfOrderBy::testingSourceChunks() {
  return observedSourceChunks.load();
}

uint64_t CudfOrderBy::testingMergeOutputBatches() {
  return observedMergeOutputBatches.load();
}

uint64_t CudfOrderBy::testingEmittedChunks() {
  return observedEmittedChunks.load();
}

uint64_t CudfOrderBy::testingSpillCleanups() {
  return observedSpillCleanups.load();
}

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
      orderByNode_(orderByNode),
      stateStream_(cudfGlobalStreamPool().get_stream()),
      sortedRunBytes_(
          testingSortedRunBytes.load() > 0
              ? testingSortedRunBytes.load()
              : CudfConfig::getInstance().orderBySortedRunBytes),
      hostSpillBytes_(
          testingHostSpillBytes.load() > 0
              ? testingHostSpillBytes.load()
              : CudfConfig::getInstance().orderByHostSpillBytes),
      mergeFanIn_(
          testingMergeFanIn.load() > 0
              ? testingMergeFanIn.load()
              : static_cast<size_t>(
                    CudfConfig::getInstance().orderByMergeFanIn)),
      outputChunkBytes_(
          testingOutputChunkBytes.load() > 0
              ? testingOutputChunkBytes.load()
              : CudfConfig::getInstance().orderByOutputChunkBytes),
      maxOutputRows_(
          testingMaxOutputRows.load() > 0
              ? testingMaxOutputRows.load()
              : static_cast<cudf::size_type>(
                    CudfConfig::getInstance().orderByMaxOutputRows)) {
  VELOX_CHECK(
      isSupported(orderByNode),
      "CudfOrderBy received an unsupported external-spill schema or sorting "
      "key type");
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
  if (input->size() == 0) {
    return;
  }

  try {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudfInput, "Expected CudfVector input");

    const auto inputStream = cudfInput->stream();
    if (inputStream.value() != stateStream_.value()) {
      std::vector<rmm::cuda_stream_view> inputStreams{inputStream};
      cudf::detail::join_streams(inputStreams, stateStream_);
    }
    // A packed-table backing buffer can retain a different deallocation stream
    // even when the vector's logical stream already equals stateStream_.
    VELOX_CHECK(
        cudfInput->rebindStream(stateStream_),
        "CudfOrderBy cannot rebind its input to the state stream");

    bufferedBytes_ += cudfInput->estimateFlatSize();
    inputs_.push_back(std::move(cudfInput));
    if (bufferedBytes_ >= sortedRunBytes_) {
      spillSortedRun();
    }
  } catch (...) {
    cleanupSpillStateAfterFailure("addInput");
    throw;
  }
}

void CudfOrderBy::doNoMoreInput() {
  Operator::noMoreInput();

  try {
    if (spilled_ && !inputs_.empty()) {
      spillSortedRun();
    }
    if (spilled_) {
      prepareSpilledOutput();
      return;
    }

    if (inputs_.empty()) {
      finished_ = true;
      return;
    }

    auto input = getConcatenatedTable(
        std::exchange(inputs_, {}), outputType_, stateStream_, get_output_mr());
    bufferedBytes_ = 0;
    VELOX_CHECK_NOT_NULL(input);

    auto sorted = cudf::sort_by_key(
        input->view(),
        input->view().select(sortKeys_),
        columnOrder_,
        nullOrder_,
        stateStream_,
        get_output_mr());
    setPendingOutput(std::move(sorted));
  } catch (...) {
    cleanupSpillStateAfterFailure("noMoreInput");
    throw;
  }
}

RowVectorPtr CudfOrderBy::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  try {
    if (auto output = takePendingOutput()) {
      return output;
    }

    if (!spilled_) {
      finished_ = true;
      return nullptr;
    }

    prepareSpilledOutput();
    auto merged = mergeNextSortedBatch();
    if (merged && merged->num_rows() > 0) {
      setPendingOutput(std::move(merged));
      return takePendingOutput();
    }

    finished_ = true;
    cleanupSpillFiles();
    return nullptr;
  } catch (...) {
    cleanupSpillStateAfterFailure("getOutput");
    throw;
  }
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

  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=sortRun.concatenate.begin "
          "bufferedBytes={} bufferedInputs={} existingRuns={} "
          "sortedRunBytes={} mergeFanIn={}",
          orderByNode_->id(),
          bufferedBytes_,
          inputs_.size(),
          sortedRuns_.size(),
          sortedRunBytes_,
          mergeFanIn_));
  auto input = getConcatenatedTable(
      std::exchange(inputs_, {}), outputType_, stateStream_, get_output_mr());
  bufferedBytes_ = 0;
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=sortRun.concatenate.end rows={}",
          orderByNode_->id(),
          input->num_rows()));
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=sortRun.sort.begin rows={}",
          orderByNode_->id(),
          input->num_rows()));
  auto sorted = cudf::sort_by_key(
      input->view(),
      input->view().select(sortKeys_),
      columnOrder_,
      nullOrder_,
      stateStream_,
      get_output_mr());
  // The sorted output no longer references the concatenated input. Release it
  // before allocating the bounded pack staging buffer.
  input.reset();
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=sortRun.sort.end rows={}",
          orderByNode_->id(),
          sorted->num_rows()));

  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=sortRun.store.begin rows={} "
          "existingRuns={} hostBytesInUse={} hostBytesLimit={}",
          orderByNode_->id(),
          sorted->num_rows(),
          sortedRuns_.size(),
          orderByHostSpillBytesInUse.load(std::memory_order_relaxed),
          hostSpillBytes_));

  SortedRun run;
  const auto storedInHost = appendHostPackedChunk(sorted->view(), run);
  if (!storedInHost) {
    appendRawPackedChunk(sorted->view(), run);
  }
  sortedRuns_.push_back(std::move(run));
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=sortRun.store.end rows={} runs={} "
          "tier={} hostBytesInUse={} path={}",
          orderByNode_->id(),
          sorted->num_rows(),
          sortedRuns_.size(),
          storedInHost ? "host-packed" : "raw-packed-disk",
          orderByHostSpillBytesInUse.load(std::memory_order_relaxed),
          sortedRuns_.back().path));
  ::malloc_trim(0);
}

bool CudfOrderBy::appendHostPackedChunk(
    cudf::table_view table,
    SortedRun& run) {
  VELOX_CHECK_GT(table.num_rows(), 0);
  auto packer = cudf::chunked_pack::create(
      table, kHostPackChunkBytes, stateStream_, get_output_mr());
  const auto bytes = packer->get_total_contiguous_size();
  if (!reserveHostSpillBytes(bytes, hostSpillBytes_)) {
    return false;
  }

  HostPackedChunk chunk;
  chunk.dataBytes = bytes;
  chunk.rows = table.num_rows();
  try {
    // The following D2H loop overwrites every byte. Array value-initialization
    // would first memset hundreds of MiB per run and double host writes on the
    // external-sort critical path.
    chunk.data = std::unique_ptr<uint8_t[]>(new uint8_t[bytes]);
  } catch (const std::bad_alloc&) {
    chunk.reset();
    return false;
  }

  rmm::device_buffer staging(
      kHostPackChunkBytes, stateStream_, get_output_mr());
  uint64_t offset = 0;
  while (packer->has_next()) {
    const auto copied = packer->next(cudf::device_span<uint8_t>{
        static_cast<uint8_t*>(staging.data()), staging.size()});
    VELOX_CHECK_LE(offset + copied, bytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        chunk.data.get() + offset,
        staging.data(),
        copied,
        cudaMemcpyDeviceToHost,
        stateStream_.value()));
    // One reusable bounded device buffer backs all chunks. Complete the D2H
    // transfer before the next pack iteration overwrites it.
    stateStream_.synchronize();
    offset += copied;
  }
  VELOX_CHECK_EQ(offset, bytes);
  chunk.metadata = packer->build_metadata();
  run.hostChunks.push_back(std::move(chunk));
  return true;
}

void CudfOrderBy::appendRawPackedChunk(
    cudf::table_view table,
    SortedRun& run) {
  VELOX_CHECK_GT(table.num_rows(), 0);
  if (run.path.empty()) {
    run.path = fmt::format(
        "{}/run-{:06}.cudfpack", spillDirectory_, spillFileSequence_++);
  }

  auto packer = cudf::chunked_pack::create(
      table, kHostPackChunkBytes, stateStream_, get_output_mr());
  RawPackedHeader header;
  header.rows = table.num_rows();
  header.dataBytes = packer->get_total_contiguous_size();

  std::ofstream output(run.path, std::ios::binary | std::ios::app);
  VELOX_CHECK(output.good(), "Cannot open raw packed run {}", run.path);
  output.write(
      reinterpret_cast<const char*>(&header), sizeof(RawPackedHeader));

  rmm::device_buffer deviceStaging(
      kHostPackChunkBytes, stateStream_, get_output_mr());
  auto hostStaging =
      std::unique_ptr<uint8_t[]>(new uint8_t[kHostPackChunkBytes]);
  uint64_t copiedBytes = 0;
  while (packer->has_next()) {
    const auto copied = packer->next(cudf::device_span<uint8_t>{
        static_cast<uint8_t*>(deviceStaging.data()), deviceStaging.size()});
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostStaging.get(),
        deviceStaging.data(),
        copied,
        cudaMemcpyDeviceToHost,
        stateStream_.value()));
    stateStream_.synchronize();
    output.write(reinterpret_cast<const char*>(hostStaging.get()), copied);
    copiedBytes += copied;
  }
  VELOX_CHECK_EQ(copiedBytes, header.dataBytes);

  auto metadata = packer->build_metadata();
  const uint64_t metadataBytes = metadata->size();
  output.write(
      reinterpret_cast<const char*>(&metadataBytes), sizeof(metadataBytes));
  output.write(
      reinterpret_cast<const char*>(metadata->data()), metadataBytes);
  output.flush();
  VELOX_CHECK(output.good(), "Failed writing raw packed run {}", run.path);
  run.diskBytes +=
      sizeof(RawPackedHeader) + header.dataBytes + sizeof(metadataBytes) +
      metadataBytes;
}

bool CudfOrderBy::loadRawPackedChunk(
    SortedRun& run,
    rmm::cuda_stream_view stream,
    MergeStats& stats) {
  if (run.path.empty() || run.diskReadOffset >= run.diskBytes) {
    return false;
  }

  std::ifstream input(run.path, std::ios::binary);
  VELOX_CHECK(input.good(), "Cannot open raw packed run {}", run.path);
  input.seekg(run.diskReadOffset);

  RawPackedHeader header;
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  VELOX_CHECK(
      input.good() && header.magic == kRawPackedMagic &&
          header.version == kRawPackedVersion && header.rows > 0,
      "Invalid raw packed run header path={} offset={}",
      run.path,
      run.diskReadOffset);

  auto gpuData = std::make_unique<rmm::device_buffer>(
      header.dataBytes, stream, get_output_mr());
  auto hostStaging =
      std::unique_ptr<uint8_t[]>(new uint8_t[kHostPackChunkBytes]);
  uint64_t offset = 0;
  while (offset < header.dataBytes) {
    const auto bytes =
        std::min<uint64_t>(kHostPackChunkBytes, header.dataBytes - offset);
    input.read(reinterpret_cast<char*>(hostStaging.get()), bytes);
    VELOX_CHECK(input.good(), "Truncated raw packed data {}", run.path);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        static_cast<uint8_t*>(gpuData->data()) + offset,
        hostStaging.get(),
        bytes,
        cudaMemcpyHostToDevice,
        stream.value()));
    stream.synchronize();
    offset += bytes;
  }

  uint64_t metadataBytes = 0;
  input.read(reinterpret_cast<char*>(&metadataBytes), sizeof(metadataBytes));
  VELOX_CHECK(
      input.good() && metadataBytes > 0,
      "Invalid raw packed metadata size path={}",
      run.path);
  auto metadata = std::make_unique<std::vector<uint8_t>>(metadataBytes);
  input.read(reinterpret_cast<char*>(metadata->data()), metadataBytes);
  VELOX_CHECK(input.good(), "Truncated raw packed metadata {}", run.path);

  run.diskReadOffset = static_cast<uint64_t>(input.tellg());
  cudf::packed_columns columns{std::move(metadata), std::move(gpuData)};
  auto view = cudf::unpack(columns);
  run.packedChunk = std::make_unique<cudf::packed_table>(
      cudf::packed_table{view, std::move(columns)});
  run.chunkBytes = header.dataBytes;

  ++stats.sourceChunks;
  observedSourceChunks.fetch_add(1);
  stats.sourceRows += header.rows;
  stats.sourceBytes += header.dataBytes;
  return true;
}

void CudfOrderBy::resetPausedChunk(SortedRun& run) {
  run.chunk.reset();
  run.packedChunk.reset();
  run.chunkOffset = 0;
  run.chunkBytes = 0;
}

cudf::table_view CudfOrderBy::pausedChunkView(const SortedRun& run) const {
  if (run.chunk) {
    return run.chunk->view();
  }
  VELOX_CHECK_NOT_NULL(run.packedChunk);
  return run.packedChunk->table;
}

bool CudfOrderBy::runHasMoreInput(const SortedRun& run) const {
  if ((run.chunk || run.packedChunk) &&
      run.chunkOffset < pausedChunkView(run).num_rows()) {
    return true;
  }
  return run.nextHostChunk < run.hostChunks.size() ||
      run.diskReadOffset < run.diskBytes;
}

uint64_t CudfOrderBy::measureTableBytes(
    std::unique_ptr<cudf::table>& table,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_NOT_NULL(table);
  auto vector = std::make_shared<CudfVector>(
      pool(), outputType_, table->num_rows(), std::move(table), stream);
  const auto bytes = vector->estimateFlatSize();
  table = vector->release();
  return bytes;
}

bool CudfOrderBy::loadPausedChunk(
    SortedRun& run,
    rmm::cuda_stream_view stream,
    MergeStats& stats) {
  VELOX_CHECK(stream.value() == stateStream_.value());

  if ((run.chunk || run.packedChunk) &&
      run.chunkOffset < pausedChunkView(run).num_rows()) {
    return true;
  }
  resetPausedChunk(run);

  if (run.nextHostChunk < run.hostChunks.size()) {
    auto host = std::move(run.hostChunks[run.nextHostChunk++]);
    auto gpuData = std::make_unique<rmm::device_buffer>(
        host.dataBytes, stream, get_output_mr());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        gpuData->data(),
        host.data.get(),
        host.dataBytes,
        cudaMemcpyHostToDevice,
        stream.value()));
    stream.synchronize();

    cudf::packed_columns columns{
        std::move(host.metadata), std::move(gpuData)};
    auto view = cudf::unpack(columns);
    run.packedChunk = std::make_unique<cudf::packed_table>(
        cudf::packed_table{view, std::move(columns)});
    run.chunkBytes = host.dataBytes;
    const auto rows = pausedChunkView(run).num_rows();
    host.reset();

    ++stats.sourceChunks;
    observedSourceChunks.fetch_add(1);
    stats.sourceRows += rows;
    stats.sourceBytes += run.chunkBytes;
    return rows > 0;
  }

  return loadRawPackedChunk(run, stream, stats);
}

std::unique_ptr<cudf::table> CudfOrderBy::mergeNextPausedBatch(
    std::vector<SortedRun*>& runs,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr,
    bool& finished,
    MergeStats& stats) {
  VELOX_CHECK(stream.value() == stateStream_.value());
  if (finished) {
    return nullptr;
  }

  std::vector<SortedRun*> activeRuns;
  std::vector<cudf::table_view> remainingViews;
  activeRuns.reserve(runs.size());
  remainingViews.reserve(runs.size());
  uint64_t residentRows{0};
  uint64_t residentBytes{0};
  for (auto* run : runs) {
    VELOX_CHECK_NOT_NULL(run);
    if (!loadPausedChunk(*run, stream, stats)) {
      continue;
    }
    auto slices = cudf::slice(
        pausedChunkView(*run),
        {run->chunkOffset, pausedChunkView(*run).num_rows()},
        stream);
    VELOX_CHECK_EQ(slices.size(), 1);
    activeRuns.push_back(run);
    remainingViews.push_back(slices.front());
    residentRows += slices.front().num_rows();
    // Count the complete owning chunk, which safely overestimates a suffix.
    residentBytes += run->chunkBytes;
  }

  stats.maxActiveRuns =
      std::max<uint64_t>(stats.maxActiveRuns, activeRuns.size());
  stats.maxResidentRows = std::max(stats.maxResidentRows, residentRows);
  stats.maxResidentBytes = std::max(stats.maxResidentBytes, residentBytes);
  updateAtomicMax(observedMaxActiveRuns, activeRuns.size());
  VELOX_CHECK_LE(
      activeRuns.size(),
      mergeFanIn_,
      "CudfOrderBy opened more readers than its configured merge fan-in");

  if (activeRuns.empty()) {
    finished = true;
    return nullptr;
  }

  std::vector<cudf::table_view> safeViews;
  std::vector<cudf::size_type> consumed(activeRuns.size(), 0);
  if (activeRuns.size() == 1) {
    safeViews.push_back(remainingViews.front());
    consumed.front() = remainingViews.front().num_rows();
  } else {
    // The smallest current tail is a safe global boundary: unread rows in
    // every sorted run compare after (or equal to) it. Keep each unconsumed
    // suffix in its owning reader chunk instead of copying a growing carry.
    std::vector<cudf::table_view> boundaryRows;
    boundaryRows.reserve(remainingViews.size());
    for (const auto& view : remainingViews) {
      auto last =
          cudf::slice(view, {view.num_rows() - 1, view.num_rows()}, stream);
      boundaryRows.push_back(last.front());
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

    for (size_t index = 0; index < remainingViews.size(); ++index) {
      auto positions = cudf::upper_bound(
          remainingViews[index].select(sortKeys_),
          boundary.front().select(sortKeys_),
          columnOrder_,
          nullOrder_,
          stream,
          mr);
      consumed[index] = firstSearchPosition(positions->view(), stream);
      if (consumed[index] == 0) {
        continue;
      }
      auto safe =
          cudf::slice(remainingViews[index], {0, consumed[index]}, stream);
      safeViews.push_back(safe.front());
    }
  }

  VELOX_CHECK(!safeViews.empty(), "Paused OrderBy merge made no progress");
  auto output = safeViews.size() == 1
      ? std::make_unique<cudf::table>(safeViews.front(), stream, mr)
      : cudf::merge(safeViews, sortKeys_, columnOrder_, nullOrder_, stream, mr);

  for (size_t index = 0; index < activeRuns.size(); ++index) {
    activeRuns[index]->chunkOffset += consumed[index];
  }

  ++stats.outputBatches;
  observedMergeOutputBatches.fetch_add(1);
  stats.outputRows += output->num_rows();
  const auto batchBytes = measureTableBytes(output, stream);
  stats.outputBytes += batchBytes;
  stats.maxOutputBytes = std::max(stats.maxOutputBytes, batchBytes);

  finished = true;
  for (auto* run : runs) {
    if (runHasMoreInput(*run)) {
      finished = false;
      break;
    }
  }
  return output;
}

void CudfOrderBy::compactSortedRunsForMerge() {
  MergeStats compactionStats;
  while (sortedRuns_.size() > kFinalMergeRuns) {
    const auto inputRunCount = sortedRuns_.size();
    std::vector<SortedRun> nextLevel;
    nextLevel.reserve((sortedRuns_.size() + mergeFanIn_ - 1) / mergeFanIn_);
    std::vector<std::string> obsoletePaths;
    MergeStats levelStats;

    for (size_t begin = 0; begin < sortedRuns_.size(); begin += mergeFanIn_) {
      const auto end = std::min(sortedRuns_.size(), begin + mergeFanIn_);
      if (end - begin == 1) {
        nextLevel.push_back(std::move(sortedRuns_[begin]));
        continue;
      }

      std::vector<SortedRun*> runs;
      runs.reserve(end - begin);
      for (size_t index = begin; index < end; ++index) {
        auto& run = sortedRuns_[index];
        resetPausedChunk(run);
        runs.push_back(&run);
      }

      SortedRun outputRun;
      bool groupFinished{false};
      while (!groupFinished) {
        auto merged = mergeNextPausedBatch(
            runs, stateStream_, get_output_mr(), groupFinished, levelStats);
        if (merged && merged->num_rows() > 0) {
          // Preserve batch order within this logical run. Host is the first
          // tier; raw lossless packed cuDF bytes are the final disk tier.
          if (!appendHostPackedChunk(merged->view(), outputRun)) {
            appendRawPackedChunk(merged->view(), outputRun);
          }
        }
      }

      for (size_t index = begin; index < end; ++index) {
        auto& run = sortedRuns_[index];
        resetPausedChunk(run);
        if (!run.path.empty()) {
          obsoletePaths.push_back(run.path);
        }
      }
      nextLevel.push_back(std::move(outputRun));
    }

    // Complete I/O and stream-ordered frees before deleting the previous level.
    stateStream_.synchronize();
    for (const auto& path : obsoletePaths) {
      std::error_code error;
      std::filesystem::remove(path, error);
      if (error) {
        LOG(WARNING) << "CudfOrderBy failed to remove compacted run path="
                     << path << " error=" << error.message();
      }
    }
    sortedRuns_ = std::move(nextLevel);

    compactionStats.sourceChunks += levelStats.sourceChunks;
    compactionStats.sourceRows += levelStats.sourceRows;
    compactionStats.sourceBytes += levelStats.sourceBytes;
    compactionStats.outputBatches += levelStats.outputBatches;
    compactionStats.outputRows += levelStats.outputRows;
    compactionStats.outputBytes += levelStats.outputBytes;
    compactionStats.maxResidentRows =
        std::max(compactionStats.maxResidentRows, levelStats.maxResidentRows);
    compactionStats.maxResidentBytes =
        std::max(compactionStats.maxResidentBytes, levelStats.maxResidentBytes);
    compactionStats.maxOutputBytes =
        std::max(compactionStats.maxOutputBytes, levelStats.maxOutputBytes);
    compactionStats.maxActiveRuns =
        std::max(compactionStats.maxActiveRuns, levelStats.maxActiveRuns);

    logDeviceMemorySnapshot(
        fmt::format(
            "operator=CudfOrderBy node={} state=compaction.level.end "
            "inputRuns={} outputRuns={} sourceChunks={} outputBatches={} "
            "maxResidentBytes={} maxOutputBytes={} maxActiveRuns={}",
            orderByNode_->id(),
            inputRunCount,
            sortedRuns_.size(),
            levelStats.sourceChunks,
            levelStats.outputBatches,
            levelStats.maxResidentBytes,
            levelStats.maxOutputBytes,
            levelStats.maxActiveRuns));
  }

  stateStream_.synchronize();
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=compaction.end runs={} "
          "sourceChunks={} outputBatches={} maxResidentBytes={} "
          "maxOutputBytes={} maxActiveRuns={}",
          orderByNode_->id(),
          sortedRuns_.size(),
          compactionStats.sourceChunks,
          compactionStats.outputBatches,
          compactionStats.maxResidentBytes,
          compactionStats.maxOutputBytes,
          compactionStats.maxActiveRuns));
}

void CudfOrderBy::initializeSortedRunReaders() {
  if (readersInitialized_) {
    return;
  }
  VELOX_CHECK_LE(sortedRuns_.size(), kFinalMergeRuns);
  for (auto& run : sortedRuns_) {
    resetPausedChunk(run);
    run.diskReadOffset = 0;
  }
  readersInitialized_ = true;
  logDeviceMemorySnapshot(
      fmt::format(
          "operator=CudfOrderBy node={} state=output.merge.begin runs={} "
          "chunkReadLimit={} hostBytesInUse={}",
          orderByNode_->id(),
          sortedRuns_.size(),
          mergeChunkBytes.load(),
          orderByHostSpillBytesInUse.load(std::memory_order_relaxed)));
}

void CudfOrderBy::prepareSpilledOutput() {
  if (readersInitialized_) {
    return;
  }
  compactSortedRunsForMerge();
  initializeSortedRunReaders();
}

std::unique_ptr<cudf::table> CudfOrderBy::mergeNextSortedBatch() {
  if (mergeFinished_) {
    return nullptr;
  }
  std::vector<SortedRun*> runs;
  runs.reserve(sortedRuns_.size());
  for (auto& run : sortedRuns_) {
    runs.push_back(&run);
  }
  auto result = mergeNextPausedBatch(
      runs, stateStream_, get_output_mr(), mergeFinished_, outputMergeStats_);
  if (mergeFinished_) {
    logDeviceMemorySnapshot(
        fmt::format(
            "operator=CudfOrderBy node={} state=output.merge.end runs={} "
            "sourceChunks={} sourceRows={} sourceBytes={} outputBatches={} "
            "outputRows={} outputBytes={} maxResidentRows={} "
            "maxResidentBytes={} maxOutputBytes={} maxActiveRuns={}",
            orderByNode_->id(),
            sortedRuns_.size(),
            outputMergeStats_.sourceChunks,
            outputMergeStats_.sourceRows,
            outputMergeStats_.sourceBytes,
            outputMergeStats_.outputBatches,
            outputMergeStats_.outputRows,
            outputMergeStats_.outputBytes,
            outputMergeStats_.maxResidentRows,
            outputMergeStats_.maxResidentBytes,
            outputMergeStats_.maxOutputBytes,
            outputMergeStats_.maxActiveRuns));
  }
  return result;
}

void CudfOrderBy::setPendingOutput(std::unique_ptr<cudf::table> output) {
  VELOX_CHECK(!pendingOutput_);
  if (!output || output->num_rows() == 0) {
    return;
  }
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = measureTableBytes(output, stateStream_);
  pendingOutput_ = std::move(output);
}

CudfVectorPtr CudfOrderBy::takePendingOutput() {
  if (!pendingOutput_) {
    return nullptr;
  }

  const auto totalRows = pendingOutput_->num_rows();
  VELOX_CHECK_LT(pendingOutputOffset_, totalRows);
  const auto remainingRows = totalRows - pendingOutputOffset_;
  const auto byteLimit = outputChunkBytes_;
  cudf::size_type targetRows = std::min(remainingRows, maxOutputRows_);
  if (pendingOutputBytes_ > byteLimit) {
    const auto proportionalRows =
        static_cast<cudf::size_type>(std::max<uint64_t>(
            1,
            static_cast<uint64_t>(totalRows) * byteLimit /
                pendingOutputBytes_));
    targetRows = std::min(targetRows, proportionalRows);
  }

  // Avoid copying an already materialized in-memory sort result when it fits
  // the configured output bounds. Spilled/oversized results keep the bounded
  // slicing path below.
  if (pendingOutputOffset_ == 0 && targetRows == totalRows &&
      pendingOutputBytes_ <= byteLimit) {
    auto output = std::make_shared<CudfVector>(
        pool(),
        outputType_,
        totalRows,
        std::move(pendingOutput_),
        stateStream_);
    pendingOutputOffset_ = 0;
    pendingOutputBytes_ = 0;
    observedEmittedChunks.fetch_add(1);
    return output;
  }

  while (true) {
    auto chunk = copyTableSlice(
        pendingOutput_->view(),
        pendingOutputOffset_,
        pendingOutputOffset_ + targetRows,
        stateStream_,
        get_output_mr());
    auto output = std::make_shared<CudfVector>(
        pool(), outputType_, targetRows, std::move(chunk), stateStream_);
    const auto actualBytes = output->estimateFlatSize();
    if (actualBytes <= byteLimit || targetRows == 1) {
      if (actualBytes > byteLimit) {
        LOG(WARNING) << "CudfOrderBy node=" << orderByNode_->id()
                     << " emitted one oversized row bytes=" << actualBytes
                     << " byteLimit=" << byteLimit;
      }
      pendingOutputOffset_ += targetRows;
      observedEmittedChunks.fetch_add(1);
      if (pendingOutputOffset_ == totalRows) {
        pendingOutput_.reset();
        pendingOutputOffset_ = 0;
        pendingOutputBytes_ = 0;
      }
      return output;
    }

    const auto proportionalRows =
        static_cast<cudf::size_type>(std::max<uint64_t>(
            1, static_cast<uint64_t>(targetRows) * byteLimit / actualBytes));
    targetRows = std::min<cudf::size_type>(targetRows - 1, proportionalRows);
  }
}

void CudfOrderBy::cleanupSpillFiles() {
  // Finish reader/writer work before destroying owners, then wait for their
  // stream-ordered frees before removing spill files.
  stateStream_.synchronize();
  sortedRuns_.clear();
  pendingOutput_.reset();
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = 0;
  stateStream_.synchronize();

  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    if (error) {
      LOG(WARNING) << "CudfOrderBy failed to remove spill directory path="
                   << spillDirectory_ << " error=" << error.message();
    } else {
      spillDirectory_.clear();
      observedSpillCleanups.fetch_add(1);
    }
  }
  ::malloc_trim(0);
}

void CudfOrderBy::cleanupSpillStateAfterFailure(
    std::string_view context) noexcept {
  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfOrderBy " << context
                 << " pre-destruction cleanup failed: " << error.what();
  }

  inputs_.clear();
  sortedRuns_.clear();
  pendingOutput_.reset();
  pendingOutputOffset_ = 0;
  pendingOutputBytes_ = 0;

  try {
    stateStream_.synchronize();
  } catch (const std::exception& error) {
    LOG(WARNING) << "CudfOrderBy " << context
                 << " post-destruction cleanup failed: " << error.what();
  }

  if (!spillDirectory_.empty()) {
    std::error_code error;
    std::filesystem::remove_all(spillDirectory_, error);
    if (error) {
      LOG(WARNING) << "CudfOrderBy " << context
                   << " failed to remove spill directory path="
                   << spillDirectory_ << " error=" << error.message();
    } else {
      spillDirectory_.clear();
      observedSpillCleanups.fetch_add(1);
    }
  }
}

void CudfOrderBy::doClose() {
  // close() also runs during task failure; cleanup preserves the original
  // exception while draining and destroying all state-stream owners.
  cleanupSpillStateAfterFailure("close");
  Operator::close();
}

} // namespace facebook::velox::cudf_velox
