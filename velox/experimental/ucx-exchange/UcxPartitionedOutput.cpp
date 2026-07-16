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
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include <fmt/format.h>
#include <folly/json.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/RangePartitionFunction.h"

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/search.hpp>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::ucx_exchange {

namespace {
int64_t targetRowsPerUcxChunk(const core::QueryConfig& queryConfig) {
  if (const char* value =
          std::getenv("GLUTEN_UCX_PARTITIONED_OUTPUT_BATCH_ROWS")) {
    try {
      const auto parsed = static_cast<int64_t>(std::stoll(value));
      if (parsed > 0) {
        return parsed;
      }
    } catch (...) {
    }
  }
  return queryConfig.ucxPartitionedOutputBatchRows();
}

RowVectorPtr buildRangeBoundaryVector(
    const std::string& boundsJson,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& keyChannels,
    memory::MemoryPool* pool,
    std::vector<cudf::order>& orders,
    std::vector<cudf::null_order>& nullOrders) {
  const auto descriptor = folly::parseJson(boundsJson);
  VELOX_CHECK_EQ(
      descriptor["version"].asInt(),
      1,
      "Unsupported RANGE_PID descriptor version");
  const auto& keys = descriptor["keys"];
  const auto& bounds = descriptor["bounds"];
  VELOX_CHECK(keys.isArray(), "RANGE_PID keys must be an array");
  VELOX_CHECK(bounds.isArray(), "RANGE_PID bounds must be an array");
  VELOX_CHECK_EQ(
      keys.size(),
      keyChannels.size(),
      "RANGE_PID key metadata/channel mismatch");

  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(keyChannels.size());
  types.reserve(keyChannels.size());
  orders.clear();
  nullOrders.clear();
  for (size_t key = 0; key < keyChannels.size(); ++key) {
    const auto channel = keyChannels[key];
    VELOX_CHECK_LT(
        channel, inputType->size(), "RANGE_PID key channel out of bounds");
    names.push_back(inputType->nameOf(channel));
    types.push_back(inputType->childAt(channel));
    orders.push_back(
        keys[key]["ascending"].asBool() ? cudf::order::ASCENDING
                                        : cudf::order::DESCENDING);
    nullOrders.push_back(
        keys[key]["nullsFirst"].asBool() ? cudf::null_order::BEFORE
                                         : cudf::null_order::AFTER);
  }

  auto boundaryType = ROW(std::move(names), std::move(types));
  auto vector = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(boundaryType, bounds.size(), pool));
  VELOX_CHECK_NOT_NULL(vector);

  for (size_t row = 0; row < bounds.size(); ++row) {
    VELOX_CHECK(bounds[row].isArray(), "RANGE_PID boundary must be an array");
    VELOX_CHECK_EQ(
        bounds[row].size(),
        keyChannels.size(),
        "RANGE_PID boundary width mismatch");
    for (size_t key = 0; key < keyChannels.size(); ++key) {
      const auto& encoded = bounds[row][key];
      auto child = vector->childAt(key);
      if (encoded["isNull"].asBool()) {
        child->setNull(row, true);
        continue;
      }

      const auto& value = encoded["value"];
      const auto& type = boundaryType->childAt(key);
      if (type->isDate()) {
        child->asFlatVector<int32_t>()->set(row, value.asInt());
      } else if (type->isTimestamp()) {
        child->asFlatVector<Timestamp>()->set(
            row, Timestamp::fromMicros(value.asInt()));
      } else {
        switch (type->kind()) {
          case TypeKind::BOOLEAN:
            child->asFlatVector<bool>()->set(row, value.asBool());
            break;
          case TypeKind::TINYINT:
            child->asFlatVector<int8_t>()->set(row, value.asInt());
            break;
          case TypeKind::SMALLINT:
            child->asFlatVector<int16_t>()->set(row, value.asInt());
            break;
          case TypeKind::INTEGER:
            child->asFlatVector<int32_t>()->set(row, value.asInt());
            break;
          case TypeKind::BIGINT:
            child->asFlatVector<int64_t>()->set(row, value.asInt());
            break;
          case TypeKind::REAL:
            child->asFlatVector<float>()->set(
                row, static_cast<float>(value.asDouble()));
            break;
          case TypeKind::DOUBLE:
            child->asFlatVector<double>()->set(row, value.asDouble());
            break;
          case TypeKind::VARCHAR: {
            const auto stringValue = value.asString();
            child->asFlatVector<StringView>()->set(
                row, StringView(stringValue));
            break;
          }
          default:
            VELOX_UNSUPPORTED(
                "RANGE_PID D1 does not support native key type {}",
                type->toString());
        }
      }
    }
  }
  return vector;
}

void normalizePartitionOffsets(
    std::vector<cudf::size_type>& offsets,
    size_t numPartitions) {
  VELOX_CHECK(
      offsets.size() == numPartitions || offsets.size() == numPartitions + 1,
      "Unexpected libcudf partition offset count {} for {} partitions",
      offsets.size(),
      numPartitions);
  VELOX_CHECK_EQ(offsets.front(), 0);
  offsets.erase(offsets.begin());
  if (offsets.size() == numPartitions) {
    offsets.pop_back();
  }
}

uint64_t targetBytesPerUcxChunk(const core::QueryConfig& queryConfig) {
  if (const char* value =
          std::getenv("GLUTEN_UCX_PARTITIONED_OUTPUT_BATCH_BYTES")) {
    try {
      return std::stoull(value);
    } catch (...) {
    }
  }
  return queryConfig.ucxPartitionedOutputBatchBytes();
}

cudf::size_type rowsPerUcxChunk(
    cudf::size_type rows,
    uint64_t bytes,
    int64_t targetRows,
    uint64_t targetBytes) {
  auto rowsPerChunk = rows;
  if (targetRows > 0) {
    rowsPerChunk = std::min<cudf::size_type>(rowsPerChunk, targetRows);
  }
  if (targetBytes > 0 && bytes > targetBytes && rows > 0) {
    const auto byteLimitedRows = std::max<uint64_t>(
        1, static_cast<uint64_t>(rows) * targetBytes / bytes);
    rowsPerChunk = std::min<cudf::size_type>(
        rowsPerChunk,
        static_cast<cudf::size_type>(std::min<uint64_t>(
            byteLimitedRows,
            std::numeric_limits<cudf::size_type>::max())));
  }
  return std::max<cudf::size_type>(1, rowsPerChunk);
}
} // namespace

// Computes a mapping from names in n2 to names in n1
// and returns that mapping in remap.
// Names in n2 must occurs in n1.
static void getRemapping(
    const RowTypePtr& inputType,
    const RowTypePtr& outputType,
    std::vector<uint32_t>& remap) {
  remap.clear();
  remap.reserve(outputType->size());
  std::unordered_map<std::string, size_t> nextOccurrence;
  for (uint32_t out = 0; out < outputType->size(); ++out) {
    const auto& name = outputType->nameOf(out);
    std::vector<uint32_t> matches;
    for (uint32_t in = 0; in < inputType->size(); ++in) {
      if (inputType->nameOf(in) == name &&
          inputType->childAt(in)->equivalent(*outputType->childAt(out))) {
        matches.push_back(in);
      }
    }
    VELOX_CHECK(
        !matches.empty(),
        "UCX output field {}:{} has no name-and-type match in input {}",
        name,
        outputType->childAt(out)->toString(),
        inputType->toString());
    auto& occurrence = nextOccurrence[name];
    const auto selected = matches[std::min(occurrence, matches.size() - 1)];
    ++occurrence;
    remap.push_back(selected);
  }
}

UcxPartitionedOutput::UcxPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "cudfPartitionedOutput"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queueManager_(UcxOutputQueueManager::getInstanceRef()),
      numPartitions_(planNode->numPartitions()),
      pipelineId_(ctx->pipelineId),
      driverId_(ctx->driverId),
      targetRowsPerChunk_(targetRowsPerUcxChunk(ctx->queryConfig())),
      targetBytesPerChunk_(targetBytesPerUcxChunk(ctx->queryConfig())) {
  this->initPartitionKeys(planNode);
  auto sources = planNode->sources();
  std::vector<std::string> inNames, outNames;
  inNames.reserve(planNode->inputType()->size());
  for (int i = 0; i < planNode->inputType()->size(); ++i) {
    inNames.push_back(planNode->inputType()->nameOf(i));
  }
  outNames.reserve(planNode->outputType()->size());
  for (int i = 0; i < planNode->outputType()->size(); ++i) {
    outNames.push_back(planNode->outputType()->nameOf(i));
  }
  if (inNames != outNames) {
    getRemapping(planNode->inputType(), planNode->outputType(), remap_);
  }
}

void UcxPartitionedOutput::addInput(RowVectorPtr input) {
  CudaAllocationTraceScope allocationTrace(
      fmt::format("UcxPartitionedOutput task={} method=addInput", taskId()));
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " addInput";
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "Input must be a CudfVector");
  VELOX_CHECK(
      !future_.valid() || future_.hasValue(),
      "addInput with outstanding future!");

  // Record stats per-input (before buffering).
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }

  pendingRows_ += cudfVector->getTableView().num_rows();
  pendingBytes_ += cudfVector->estimateFlatSize();
  pendingInputs_.push_back(std::move(cudfVector));

  if ((targetRowsPerChunk_ <= 0 && targetBytesPerChunk_ == 0) ||
      (targetRowsPerChunk_ > 0 && pendingRows_ >= targetRowsPerChunk_) ||
      (targetBytesPerChunk_ > 0 && pendingBytes_ >= targetBytesPerChunk_)) {
    flushPending();
  }
}

void UcxPartitionedOutput::flushPending() {
  CudaAllocationTraceScope allocationTrace(
      fmt::format(
          "UcxPartitionedOutput task={} method=flushPending", taskId()));
  if (pendingInputs_.empty()) {
    return;
  }

  try {
    const auto pendingBytes = pendingBytes_;
    cudf::table_view tableView;
    rmm::cuda_stream_view stream = pendingInputs_.back()->stream();
    // Keeps the merged table alive while tableView references it.
    std::unique_ptr<cudf::table> mergedTable;

    if (pendingInputs_.size() == 1) {
      // Fast path: use the single input's view directly (no GPU alloc).
      auto& cv = pendingInputs_[0];
      stream = cv->stream();
      tableView = remap_.empty()
          ? cv->getTableView()
          : cv->getTableView().select(remap_.begin(), remap_.end());
    } else {
      // Collect (remapped) table views.
      std::vector<cudf::table_view> views;
      std::vector<rmm::cuda_stream_view> inputStreams;
      views.reserve(pendingInputs_.size());
      inputStreams.reserve(pendingInputs_.size());
      for (auto& v : pendingInputs_) {
        inputStreams.push_back(v->stream());
        views.push_back(
            remap_.empty()
                ? v->getTableView()
                : v->getTableView().select(remap_.begin(), remap_.end()));
      }

      cudf::detail::join_streams(inputStreams, stream);
      mergedTable = cudf::concatenate(
          views, stream, cudf::get_current_device_resource_ref());

      orderCudfVectorDeallocationsAfterStream(
          pendingInputs_, inputStreams, stream);

      // Free input GPU memory before partitioning (peak = 2x -> 1x).
      pendingInputs_.clear();

      tableView = mergedTable->view();
    }

    // Partition + enqueue (identical to previous addInput logic).
    auto queueManager = sharedQueueManager();
    if (numPartitions_ > 1) {
      if (!rangeBoundsJson_.empty()) {
        rangePartition(tableView, stream);
      } else if (partitionKeyIndices_.size() > 0 || spec_ == "gather") {
        hashPartition(tableView, stream);
      } else {
        equalPartition(tableView, stream);
      }
    } else {
      const auto tableRows = tableView.num_rows();
      const auto rowsPerChunk = rowsPerUcxChunk(
          tableRows,
          pendingBytes,
          targetRowsPerChunk_,
          targetBytesPerChunk_);
      // SINGLE/gather exchanges must obey the same chunk bound as HASH and
      // RANGE.  Packing the whole input here bypassed splitAndEnqueue and let
      // a global sort/gather create tens-of-GiB host-staging transfers.
      for (cudf::size_type start = 0; start < tableRows;
           start += rowsPerChunk) {
        const auto end =
            std::min<cudf::size_type>(tableRows, start + rowsPerChunk);
        auto slicedTables = cudf::slice(tableView, {start, end});
        VELOX_CHECK_EQ(slicedTables.size(), 1);
        auto packedCols = cudf::pack(
            slicedTables[0], stream, cudf::get_current_device_resource_ref());
        stream.synchronize();
        auto packedColsPtr = std::make_unique<cudf::packed_columns>(
            std::move(packedCols.metadata), std::move(packedCols.gpu_data));
        queueManager->enqueue(
            this->taskId(),
            0,
            std::move(packedColsPtr),
            slicedTables[0].num_rows());
      }
    }

    // Check backpressure after enqueue.
    auto blocked = queueManager->checkBlocked(this->taskId(), &future_);
    if (blocked) {
      VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
              << " is blocked, can no longer write to output!";
    }
    blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                              : exec::BlockingReason::kNotBlocked;

    pendingInputs_.clear();
    pendingRows_ = 0;
    pendingBytes_ = 0;

  } catch (const rmm::bad_alloc& e) {
    VLOG(1)
        << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
        << " caught memory alloc error, removing all memory in output queues";
    pendingInputs_.clear();
    pendingRows_ = 0;
    pendingBytes_ = 0;
    for (int i = 0; i < numPartitions_; i++) {
      sharedQueueManager()->deleteResults(this->taskId(), i);
    }
    throw;
  }
}

exec::BlockingReason UcxPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr UcxPartitionedOutput::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_) {
    return nullptr;
  }
  if (noMoreInput_) {
    flushPending(); // drain any remaining buffered inputs
    sharedQueueManager()->noMoreData(this->taskId());
    finished_ = true;
  }
  return nullptr;
}

bool UcxPartitionedOutput::isFinished() {
  return finished_;
}

std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
UcxPartitionedOutput::sharedQueueManager() {
  auto shared_queueManager = queueManager_.lock();
  VELOX_CHECK_NOT_NULL(
      shared_queueManager, "OutputQueueManager was already destructed");
  return shared_queueManager;
}

void UcxPartitionedOutput::initPartitionKeys(
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode) {
  // Following Logic copied direcly from CudLocalPartition (!)

  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  spec_ = planNode->partitionFunctionSpec().toString();

  if (auto* rangeFunctionSpec = dynamic_cast<const RangePartitionFunctionSpec*>(
          &planNode->partitionFunctionSpec())) {
    partitionKeyIndices_ = rangeFunctionSpec->keyChannels();
    rangeBoundsJson_ = rangeFunctionSpec->boundsJson();
    VELOX_CHECK(
        !partitionKeyIndices_.empty() && !rangeBoundsJson_.empty(),
        "RANGE_PID requires both keys and Spark boundaries");
    return;
  }

  // Only parse keys if it's a hash function
  if (spec_.find("HASH(") != std::string::npos) {
    // Extract keys between HASH( and )
    size_t start = spec_.find("HASH(") + 5;
    size_t end = spec_.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec_.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
  }
}

void UcxPartitionedOutput::hashPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Hashing and partitioning into " << numPartitions_ << " chunks";

  // Use cudf hash partitioning
  std::vector<cudf::size_type> partitionKeyIndices;
  for (const auto& idx : partitionKeyIndices_) {
    partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
  }

  auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
      tableView,
      partitionKeyIndices,
      numPartitions_,
      cudf::hash_id::HASH_MURMUR3,
      cudf::DEFAULT_HASH_SEED,
      stream);

  VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
  VELOX_CHECK_EQ(partitionOffsets[0], 0);

  // Erase first element since it's always 0 and we don't need it.
  partitionOffsets.erase(partitionOffsets.begin());
  partitionOffsets.pop_back();

  splitAndEnqueue(partitionedTable->view(), partitionOffsets, stream);
}

void UcxPartitionedOutput::rangePartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK(!rangeBoundsJson_.empty(), "RANGE_PID descriptor is missing");
  VELOX_CHECK(!partitionKeyIndices_.empty(), "RANGE_PID keys are missing");

  if (!rangeBoundaries_) {
    auto boundaryVector = buildRangeBoundaryVector(
        rangeBoundsJson_,
        outputType_,
        partitionKeyIndices_,
        pool(),
        rangeOrders_,
        rangeNullOrders_);
    rangeBoundaries_ = cudf_velox::with_arrow::toCudfTable(
        boundaryVector,
        pool(),
        stream,
        cudf::get_current_device_resource_ref());
    VELOX_CHECK_LT(
        rangeBoundaries_->num_rows(),
        numPartitions_,
        "RANGE_PID boundary count must be smaller than requested partitions");
  }

  std::vector<cudf::size_type> rangeKeyIndices;
  rangeKeyIndices.reserve(partitionKeyIndices_.size());
  for (const auto index : partitionKeyIndices_) {
    rangeKeyIndices.push_back(static_cast<cudf::size_type>(index));
  }
  const auto keyTable = tableView.select(rangeKeyIndices);
  auto partitionIds = cudf::lower_bound(
      rangeBoundaries_->view(),
      keyTable,
      rangeOrders_,
      rangeNullOrders_,
      stream,
      cudf::get_current_device_resource_ref());
  VELOX_CHECK(
      partitionIds->size() == tableView.num_rows(),
      "RANGE_PID must produce exactly one id per input row");
  VELOX_CHECK(
      partitionIds->type().id() == cudf::type_id::INT32,
      "RANGE_PID must produce an INT32 partition map");

  // libcudf::partition groups by the explicit INT32 map. No hash function is
  // involved; the returned table is routed directly to destination queues.
  auto [partitionedTable, partitionOffsets] = cudf::partition(
      tableView,
      partitionIds->view(),
      numPartitions_,
      stream,
      cudf::get_current_device_resource_ref());
  normalizePartitionOffsets(partitionOffsets, numPartitions_);
  splitAndEnqueue(partitionedTable->view(), partitionOffsets, stream);
}

void UcxPartitionedOutput::equalPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Splitting into " << numPartitions_ << " chunks";
  std::vector<cudf::size_type> offsets;
  cudf::size_type size = tableView.num_rows();
  for (int i = 1; i < numPartitions_; ++i) {
    cudf::size_type idx = size * i / numPartitions_;
    offsets.push_back(idx);
  }
  splitAndEnqueue(tableView, offsets, stream);
}

void UcxPartitionedOutput::splitAndEnqueue(
    cudf::table_view tableView,
    std::vector<cudf::size_type> offsets,
    rmm::cuda_stream_view stream) {
  auto contiguousTables = cudf::contiguous_split(
      tableView, offsets, stream, cudf::get_current_device_resource_ref());

  // Synchronize the stream to ensure CUDA operations complete before enqueuing.
  // UCXX/UCX is not stream-aware, so without syncing, data could be sent before
  // the GPU kernels have finished writing to the buffers.
  stream.synchronize();

  VELOX_CHECK_EQ(
      offsets.size() + 1, numPartitions_, "mismatch in numPartitions_");
  auto queueManager = sharedQueueManager();
  for (int i = 0; i < numPartitions_; ++i) {
    auto const& partitionTable = contiguousTables[i];
    const auto partitionRows = partitionTable.table.num_rows();
    if (partitionRows == 0) {
      // Skip empty partitions.
      continue;
    }

    const auto partitionBytes = partitionTable.data.gpu_data->size();
    const auto rowsPerChunk = rowsPerUcxChunk(
        partitionRows,
        partitionBytes,
        targetRowsPerChunk_,
        targetBytesPerChunk_);
    if (rowsPerChunk < partitionRows) {
      VLOG(2) << "UcxPartitionedOutput chunking task=" << taskId()
              << " destination=" << i << " rows=" << partitionRows
              << " bytes=" << partitionBytes
              << " rowsPerChunk=" << rowsPerChunk
              << " targetRowsPerChunk=" << targetRowsPerChunk_
              << " targetBytesPerChunk=" << targetBytesPerChunk_;
      for (cudf::size_type start = 0; start < partitionRows;
           start += rowsPerChunk) {
        const auto end =
            std::min<cudf::size_type>(partitionRows, start + rowsPerChunk);
        auto slicedTables = cudf::slice(
            partitionTable.table,
            {static_cast<cudf::size_type>(start),
             static_cast<cudf::size_type>(end)},
            stream);
        VELOX_CHECK_EQ(slicedTables.size(), 1);
        auto packedCols = cudf::pack(slicedTables[0], stream);
        stream.synchronize();
        auto packedColsPtr = std::make_unique<cudf::packed_columns>(
            std::move(packedCols.metadata), std::move(packedCols.gpu_data));
        queueManager->enqueue(
            this->taskId(),
            i,
            std::move(packedColsPtr),
            slicedTables[0].num_rows());
      }
      continue;
    }

    auto packedColsPtr = std::make_unique<cudf::packed_columns>(
        std::move(contiguousTables[i].data.metadata),
        std::move(contiguousTables[i].data.gpu_data));

    // enqueue partition data on Ucx Output Buffer
    queueManager->enqueue(
        this->taskId(),
        i,
        std::move(packedColsPtr),
        partitionTable.table.num_rows());
  }
}

} // namespace facebook::velox::ucx_exchange
