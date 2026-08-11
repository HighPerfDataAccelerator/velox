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

#include "velox/experimental/cudf/exec/CudfPackedMicroBucket.h"
#include "velox/experimental/cudf/exec/CudfPackedRestore.h"
#include "velox/experimental/cudf/exec/CudfPackedSpill.h"

#include "velox/common/testutil/TempDirectoryPath.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>
#include <lz4.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

CudfPackedMicroBucketDescriptor microBucket(uint64_t id, uint64_t bytes) {
  auto metadata = std::make_shared<uint8_t>(0);
  auto storage = std::make_shared<uint8_t>(0);
  CudfPackedMicroBucketDescriptor bucket;
  bucket.id = id;
  bucket.hashPrefix = id;
  bucket.hashBitCount = 8;
  bucket.rows = bytes;
  bucket.restoreBytes = bytes;
  bucket.storedBytes = bytes;
  bucket.extents.push_back(
      CudfPackedMicroBucketExtent{
          CudfPackedMicroBucketExtent::Tier::kHost,
          std::move(metadata),
          std::move(storage),
          0,
          bytes,
          bytes});
  return bucket;
}

TEST(CudfPackedSpillTest, microBucketBestFitPlanDoesNotMovePayloads) {
  std::vector<CudfPackedMicroBucketDescriptor> buckets;
  buckets.push_back(microBucket(0, 8));
  buckets.push_back(microBucket(1, 7));
  buckets.push_back(microBucket(2, 5));
  buckets.push_back(microBucket(3, 4));
  const auto* firstOwner = buckets[0].extents[0].storageOwner.get();

  const auto waves = planCudfPackedMicroBucketRestoreWaves(buckets, 12);
  ASSERT_EQ(waves.size(), 2);
  EXPECT_EQ(waves[0].restoreBytes, 12);
  EXPECT_EQ(waves[0].admissionBytes, 12);
  EXPECT_EQ(waves[1].restoreBytes, 12);
  EXPECT_FALSE(waves[0].oversized);
  EXPECT_FALSE(waves[1].oversized);
  // Planning only references descriptor indices; opaque payload ownership and
  // bytes remain untouched on host.
  EXPECT_EQ(buckets[0].extents[0].storageOwner.get(), firstOwner);
}

TEST(CudfPackedSpillTest, microBucketPlanAccountsWorkspaceAndSkew) {
  std::vector<CudfPackedMicroBucketDescriptor> buckets;
  buckets.push_back(microBucket(0, 80));
  buckets.push_back(microBucket(1, 50));
  buckets.push_back(microBucket(2, 10));
  CudfPackedMicroBucketPlanOptions options;
  options.fixedReserveBytes = 20;
  options.perBucketReserveBytes = 10;
  options.maxBucketsPerWave = 2;

  const auto waves =
      planCudfPackedMicroBucketRestoreWaves(buckets, 100, options);
  ASSERT_EQ(waves.size(), 2);
  EXPECT_FALSE(waves[0].oversized);
  EXPECT_EQ(waves[0].restoreBytes, 80);
  EXPECT_EQ(waves[0].admissionBytes, 80);
  EXPECT_TRUE(waves[1].oversized);
  EXPECT_EQ(waves[1].restoreBytes, 80);
  EXPECT_EQ(waves[1].admissionBytes, 90);
  ASSERT_EQ(waves[1].bucketIndices.size(), 1);
  EXPECT_EQ(waves[1].bucketIndices.front(), 0);
}

TEST(CudfPackedSpillTest, microBucketStreamingPlanPreservesExtents) {
  auto bucket = microBucket(0, 40);
  bucket.extents.push_back(bucket.extents.front());
  bucket.extents.push_back(bucket.extents.front());
  bucket.restoreBytes = 120;
  bucket.storedBytes = 120;
  CudfPackedMicroBucketPlanOptions options;
  options.fixedReserveBytes = 10;
  options.perBucketReserveBytes = 10;

  const auto slices = planCudfPackedMicroBucketStreamingRestore(
      std::vector<CudfPackedMicroBucketDescriptor>{bucket}, 100, options);
  ASSERT_EQ(slices.size(), 2);
  EXPECT_EQ(slices[0].extentBegin, 0);
  EXPECT_EQ(slices[0].extentEnd, 2);
  EXPECT_EQ(slices[0].restoreBytes, 80);
  EXPECT_EQ(slices[0].admissionBytes, 90);
  EXPECT_TRUE(slices[0].first);
  EXPECT_FALSE(slices[0].last);
  EXPECT_EQ(slices[1].extentBegin, 2);
  EXPECT_EQ(slices[1].extentEnd, 3);
  EXPECT_TRUE(slices[1].last);
}

TEST(CudfPackedSpillTest, microBucketDescriptorRejectsBadAccounting) {
  auto bucket = microBucket(3, 16);
  bucket.restoreBytes = 15;
  EXPECT_THROW(validateCudfPackedMicroBucket(bucket), std::invalid_argument);

  auto missingStorage = microBucket(4, 0);
  missingStorage.extents.front().storedBytes = 1;
  missingStorage.storedBytes = 1;
  missingStorage.extents.front().storageOwner.reset();
  EXPECT_THROW(
      validateCudfPackedMicroBucket(missingStorage), std::invalid_argument);
}

TEST(CudfPackedSpillTest, microBucketAlignmentRequiresSameHashIdentity) {
  auto build = microBucket(3, 16);
  build.partitioning = CudfPackedPartitioningIdentity{1, 42, 0x1234, 7};
  auto probe = microBucket(3, 8);
  probe.partitioning = build.partitioning;
  EXPECT_NO_THROW(validateCompatibleCudfPackedMicroBuckets(build, probe));

  probe.partitioning.hashSeed++;
  EXPECT_THROW(
      validateCompatibleCudfPackedMicroBuckets(build, probe),
      std::invalid_argument);
  probe.partitioning = build.partitioning;
  probe.hashPrefix++;
  EXPECT_THROW(
      validateCompatibleCudfPackedMicroBuckets(build, probe),
      std::invalid_argument);
}

TEST(CudfPackedSpillTest, microBucketHashBitsFollowObservedBytes) {
  constexpr uint64_t kGiB = uint64_t{1} << 30;
  constexpr uint64_t kMiB = uint64_t{1} << 20;
  EXPECT_EQ(
      chooseCudfPackedMicroBucketHashBits(594 * kGiB, 512 * kMiB, 7, 12), 11);
  EXPECT_EQ(
      chooseCudfPackedMicroBucketHashBits(1 * kGiB, 512 * kMiB, 7, 12), 7);
  EXPECT_EQ(chooseCudfPackedMicroBucketHashBits(8 * kGiB, 1 * kMiB, 7, 12), 12);
}

TEST(CudfPackedSpillTest, asyncOffsetLocalRoundTripAndCleanup) {
  auto directory = common::testutil::TempDirectoryPath::create();
  const auto path =
      (std::filesystem::path(directory->getPath()) / "packed.bin").string();
  uint64_t accountedBytes = 0;
  auto file = std::make_shared<CudfPackedSpillFile>(
      path, nullptr, [&](uint64_t bytes) { accountedBytes += bytes; });

  auto first = std::shared_ptr<uint8_t>(
      new uint8_t[4]{1, 2, 3, 4}, std::default_delete<uint8_t[]>());
  auto second = std::shared_ptr<uint8_t>(
      new uint8_t[3]{5, 6, 7}, std::default_delete<uint8_t[]>());
  auto [firstOffset, firstWrite] = file->appendAsync(first, 4);
  auto [secondOffset, secondWrite] = file->appendAsync(second, 3);
  EXPECT_EQ(firstOffset, 0);
  EXPECT_EQ(secondOffset, 4);
  firstWrite.get();
  secondWrite.get();
  EXPECT_EQ(accountedBytes, 7);

  uint8_t restored[7]{};
  file->read(0, sizeof(restored), restored);
  const uint8_t expected[]{1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(std::memcmp(restored, expected, sizeof(expected)), 0);

  file.reset();
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(CudfPackedSpillTest, sharedHostBudgetReservation) {
  const auto initial = currentCudfPackedHostMemoryReservedBytes();
  auto first = tryReserveCudfPackedHostMemory(32, initial + 64);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(currentCudfPackedHostMemoryReservedBytes(), initial + 32);

  auto exceedsBudget = tryReserveCudfPackedHostMemory(40, initial + 64);
  EXPECT_EQ(exceedsBudget, nullptr);
  first.reset();
  EXPECT_EQ(currentCudfPackedHostMemoryReservedBytes(), initial);

  auto afterRelease = tryReserveCudfPackedHostMemory(40, initial + 64);
  ASSERT_NE(afterRelease, nullptr);
  afterRelease.reset();
  EXPECT_EQ(currentCudfPackedHostMemoryReservedBytes(), initial);
}

TEST(CudfPackedSpillTest, asynchronousCompressionRoundTrip) {
  auto directory = common::testutil::TempDirectoryPath::create();
  const auto path =
      (std::filesystem::path(directory->getPath()) / "compressed.bin").string();
  auto file = std::make_shared<CudfPackedSpillFile>(path, nullptr, nullptr);
  constexpr uint64_t kBytes = 1 << 20;
  auto source = std::shared_ptr<uint8_t>(
      new uint8_t[kBytes], std::default_delete<uint8_t[]>());
  std::memset(source.get(), 0x5a, kBytes);

  const auto result = file->appendCompressedAsync(source, kBytes).get();
  EXPECT_TRUE(result.compressed);
  EXPECT_LT(result.storedBytes, kBytes);
  std::vector<uint8_t> stored(result.storedBytes);
  file->read(result.fileOffset, result.storedBytes, stored.data());
  std::vector<uint8_t> restored(kBytes);
  ASSERT_EQ(
      LZ4_decompress_safe(
          reinterpret_cast<const char*>(stored.data()),
          reinterpret_cast<char*>(restored.data()),
          static_cast<int>(stored.size()),
          static_cast<int>(restored.size())),
      kBytes);
  EXPECT_EQ(std::memcmp(restored.data(), source.get(), kBytes), 0);
}

TEST(CudfPackedSpillTest, asynchronousRawResultRoundTrip) {
  auto directory = common::testutil::TempDirectoryPath::create();
  const auto path =
      (std::filesystem::path(directory->getPath()) / "raw-result.bin").string();
  auto file = std::make_shared<CudfPackedSpillFile>(path, nullptr, nullptr);
  constexpr uint64_t kBytes = 1 << 20;
  auto source = std::shared_ptr<uint8_t>(
      new uint8_t[kBytes], std::default_delete<uint8_t[]>());
  std::memset(source.get(), 0x5a, kBytes);

  const auto result = file->appendCompressedAsync(source, kBytes, false).get();
  EXPECT_FALSE(result.compressed);
  EXPECT_EQ(result.storedBytes, kBytes);
  EXPECT_EQ(result.compressionMicros, 0);
  std::vector<uint8_t> restored(kBytes);
  file->read(result.fileOffset, result.storedBytes, restored.data());
  EXPECT_EQ(std::memcmp(restored.data(), source.get(), kBytes), 0);
}

TEST(CudfPackedSpillTest, bulkRestoreUsesOneDeviceAllocation) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  const std::vector<std::vector<int32_t>> expected{{1, 2, 3, 4}, {10, 20, 30}};
  std::vector<CudfPackedHostRestoreChunk> chunks;
  for (const auto& values : expected) {
    auto column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        values.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    ASSERT_EQ(
        cudaMemcpyAsync(
            column->mutable_view().data<int32_t>(),
            values.data(),
            values.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream.value()),
        cudaSuccess);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    cudf::table table(std::move(columns));
    auto packed = cudf::pack(table.view(), stream, mr);
    const auto bytes = packed.gpu_data->size();
    auto hostData = std::shared_ptr<uint8_t>(
        new uint8_t[bytes], std::default_delete<uint8_t[]>());
    ASSERT_EQ(
        cudaMemcpyAsync(
            hostData.get(),
            packed.gpu_data->data(),
            bytes,
            cudaMemcpyDeviceToHost,
            stream.value()),
        cudaSuccess);
    stream.synchronize();
    chunks.push_back(
        CudfPackedHostRestoreChunk{
            std::move(packed.metadata), std::move(hostData), bytes});
  }

  auto restored =
      bulkRestoreCudfPackedHostChunks(std::move(chunks), stream, mr);
  ASSERT_NE(restored.deviceBuffer(), nullptr);
  ASSERT_EQ(restored.tables().size(), expected.size());
  const auto* allocationBegin =
      static_cast<const uint8_t*>(restored.deviceBuffer()->data());
  const auto* allocationEnd = allocationBegin + restored.deviceBuffer()->size();
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& table = restored.tables()[i];
    ASSERT_EQ(table.num_columns(), 1);
    ASSERT_EQ(table.num_rows(), expected[i].size());
    const auto* columnData =
        reinterpret_cast<const uint8_t*>(table.column(0).data<int32_t>());
    EXPECT_GE(columnData, allocationBegin);
    EXPECT_LT(columnData, allocationEnd);
    std::vector<int32_t> actual(expected[i].size());
    ASSERT_EQ(
        cudaMemcpy(
            actual.data(),
            columnData,
            actual.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(actual, expected[i]);
  }
}

TEST(CudfPackedSpillTest, bulkRestoreUsesPinnedBounceForLargePageableWave) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  constexpr size_t kChunks = 5;
  constexpr size_t kRowsPerChunk = 1 << 20;
  std::vector<CudfPackedHostRestoreChunk> chunks;
  chunks.reserve(kChunks);

  for (size_t chunk = 0; chunk < kChunks; ++chunk) {
    std::vector<int32_t> values(kRowsPerChunk, chunk + 1);
    auto column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        values.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().data<int32_t>(),
        values.data(),
        values.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.value()));
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    cudf::table table(std::move(columns));
    auto packed = cudf::pack(table.view(), stream, mr);
    const auto bytes = packed.gpu_data->size();
    auto hostData = std::shared_ptr<uint8_t>(
        new uint8_t[bytes], std::default_delete<uint8_t[]>());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostData.get(),
        packed.gpu_data->data(),
        bytes,
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    CudfPackedHostRestoreChunk restoreChunk;
    restoreChunk.metadata = std::move(packed.metadata);
    restoreChunk.dataBytes = bytes;
    restoreChunk.materializeIntoPinned = [hostData = std::move(hostData),
                                          bytes](uint8_t* destination) {
      std::memcpy(destination, hostData.get(), bytes);
    };
    chunks.push_back(std::move(restoreChunk));
  }

  auto restored = bulkRestoreCudfPackedHostChunks(
      std::move(chunks),
      stream,
      mr,
      CudfBulkPackedRestoreOptions{.pinnedHostThreads = 8});
  ASSERT_EQ(restored.tables().size(), kChunks);
  EXPECT_GT(restored.stats().pinnedBounceBytes, 16ULL << 20);
  EXPECT_EQ(restored.stats().pageableDirectBytes, 0);
  EXPECT_EQ(restored.stats().pinnedBounceCopies, 1);
  EXPECT_EQ(restored.stats().pinnedHostThreadLimit, 8);
  EXPECT_EQ(restored.stats().parallelHostStageGroups, 1);
  EXPECT_EQ(restored.stats().parallelHostStageChunks, kChunks);
  for (size_t chunk = 0; chunk < kChunks; ++chunk) {
    const auto& table = restored.tables()[chunk];
    ASSERT_EQ(table.num_rows(), kRowsPerChunk);
    int32_t first = 0;
    int32_t last = 0;
    CUDF_CUDA_TRY(cudaMemcpy(
        &first,
        table.column(0).data<int32_t>(),
        sizeof(first),
        cudaMemcpyDeviceToHost));
    CUDF_CUDA_TRY(cudaMemcpy(
        &last,
        table.column(0).data<int32_t>() + kRowsPerChunk - 1,
        sizeof(last),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(first, chunk + 1);
    EXPECT_EQ(last, chunk + 1);
  }
}

TEST(CudfPackedSpillTest, singleWaveUsesLastAvailablePinnedBounceSlot) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  constexpr size_t kRows = 5ULL << 20;
  std::vector<int32_t> values(kRows, 815);
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      values.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      column->mutable_view().data<int32_t>(),
      values.data(),
      values.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  cudf::table table(std::move(columns));
  auto packed = cudf::pack(table.view(), stream, mr);
  const auto bytes = packed.gpu_data->size();
  ASSERT_GT(bytes, 16ULL << 20);
  ASSERT_LT(bytes, 128ULL << 20);
  auto hostData = std::shared_ptr<uint8_t>(
      new uint8_t[bytes], std::default_delete<uint8_t[]>());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostData.get(),
      packed.gpu_data->data(),
      bytes,
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  // Occupy the complete process-wide pool, then release exactly one slot.
  // The single-wave restore must use that slot; the old all-or-nothing
  // two-slab acquisition fell back to pageable H2D here.
  std::vector<std::shared_ptr<uint8_t>> occupied;
  while (auto slot = acquireCudfPackedPinnedBuffer(128ULL << 20)) {
    occupied.push_back(std::move(slot));
    ASSERT_LE(occupied.size(), 16);
  }
  ASSERT_FALSE(occupied.empty());
  occupied.pop_back();

  CudfPackedHostRestoreChunk chunk;
  chunk.metadata = std::move(packed.metadata);
  chunk.dataBytes = bytes;
  chunk.materializeIntoPinned = [hostData = std::move(hostData),
                                 bytes](uint8_t* destination) {
    std::memcpy(destination, hostData.get(), bytes);
  };
  std::vector<CudfPackedHostRestoreChunk> chunks;
  chunks.push_back(std::move(chunk));
  auto restored =
      bulkRestoreCudfPackedHostChunks(std::move(chunks), stream, mr);
  EXPECT_EQ(restored.stats().pinnedBounceBytes, bytes);
  EXPECT_EQ(restored.stats().pageableDirectBytes, 0);
  EXPECT_EQ(restored.stats().pinnedBounceCopies, 1);
  ASSERT_EQ(restored.tables().size(), 1);
  ASSERT_EQ(restored.tables()[0].num_rows(), kRows);
  int32_t first = 0;
  int32_t last = 0;
  CUDF_CUDA_TRY(cudaMemcpy(
      &first,
      restored.tables()[0].column(0).data<int32_t>(),
      sizeof(first),
      cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(
      &last,
      restored.tables()[0].column(0).data<int32_t>() + kRows - 1,
      sizeof(last),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(first, 815);
  EXPECT_EQ(last, 815);
}

TEST(CudfPackedSpillTest, bulkRestoreBouncesResidentPageableWave) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  constexpr size_t kChunks = 5;
  constexpr size_t kRowsPerChunk = 1 << 20;
  std::vector<CudfPackedHostRestoreChunk> chunks;
  chunks.reserve(kChunks);

  for (size_t chunk = 0; chunk < kChunks; ++chunk) {
    std::vector<int32_t> values(kRowsPerChunk, chunk + 101);
    auto column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT32},
        values.size(),
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().data<int32_t>(),
        values.data(),
        values.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.value()));
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(column));
    cudf::table table(std::move(columns));
    auto packed = cudf::pack(table.view(), stream, mr);
    const auto bytes = packed.gpu_data->size();
    auto hostData = std::shared_ptr<uint8_t>(
        new uint8_t[bytes], std::default_delete<uint8_t[]>());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        hostData.get(),
        packed.gpu_data->data(),
        bytes,
        cudaMemcpyDeviceToHost,
        stream.value()));
    stream.synchronize();
    CudfPackedHostRestoreChunk restoreChunk;
    restoreChunk.metadata = std::move(packed.metadata);
    restoreChunk.data = std::move(hostData);
    restoreChunk.dataBytes = bytes;
    restoreChunk.stageResidentPageableThroughPinned = true;
    chunks.push_back(std::move(restoreChunk));
  }

  auto restored =
      bulkRestoreCudfPackedHostChunks(std::move(chunks), stream, mr);
  ASSERT_EQ(restored.tables().size(), kChunks);
  EXPECT_GT(restored.stats().pinnedBounceBytes, 16ULL << 20);
  EXPECT_EQ(
      restored.stats().residentPageableBounceBytes,
      restored.stats().pinnedBounceBytes);
  EXPECT_EQ(restored.stats().pageableDirectBytes, 0);
  for (size_t chunk = 0; chunk < kChunks; ++chunk) {
    const auto& table = restored.tables()[chunk];
    ASSERT_EQ(table.num_rows(), kRowsPerChunk);
    int32_t first = 0;
    CUDF_CUDA_TRY(cudaMemcpy(
        &first,
        table.column(0).data<int32_t>(),
        sizeof(first),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(first, chunk + 101);
  }
}

TEST(CudfPackedSpillTest, bulkRestoreMatchesJob144BucketShape) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  constexpr size_t kChunks = 68;
  constexpr size_t kRowsPerChunk = (9ULL << 20) / sizeof(int32_t);
  std::vector<int32_t> values(kRowsPerChunk, 144);
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      values.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      column->mutable_view().data<int32_t>(),
      values.data(),
      values.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  cudf::table table(std::move(columns));
  auto packed = cudf::pack(table.view(), stream, mr);
  const auto bytes = packed.gpu_data->size();
  auto hostData = std::shared_ptr<uint8_t>(
      new uint8_t[bytes], std::default_delete<uint8_t[]>());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostData.get(),
      packed.gpu_data->data(),
      bytes,
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  const auto* configuredPinnedSource =
      std::getenv("CUDF_PACKED_RESTORE_TEST_SOURCE_PINNED");
  const bool sourcePinned = configuredPinnedSource != nullptr &&
      std::string_view(configuredPinnedSource) == "1";
  const auto* configuredMaterializer =
      std::getenv("CUDF_PACKED_RESTORE_TEST_MATERIALIZER");
  const bool useMaterializer = configuredMaterializer != nullptr &&
      std::string_view(configuredMaterializer) == "1";
  std::shared_ptr<uint8_t> restoreData = hostData;
  if (sourcePinned) {
    restoreData = acquireCudfPackedPinnedBuffer(bytes);
    ASSERT_NE(restoreData, nullptr);
    std::memcpy(restoreData.get(), hostData.get(), bytes);
  }
  std::vector<CudfPackedHostRestoreChunk> chunks;
  chunks.reserve(kChunks);
  for (size_t chunk = 0; chunk < kChunks; ++chunk) {
    if (useMaterializer) {
      CudfPackedHostRestoreChunk restoreChunk;
      restoreChunk.metadata =
          std::make_unique<std::vector<uint8_t>>(*packed.metadata);
      restoreChunk.dataBytes = bytes;
      restoreChunk.materializeIntoPinned = [restoreData,
                                            bytes](uint8_t* destination) {
        std::memcpy(destination, restoreData.get(), bytes);
      };
      chunks.push_back(std::move(restoreChunk));
    } else {
      chunks.push_back(
          CudfPackedHostRestoreChunk{
              std::make_unique<std::vector<uint8_t>>(*packed.metadata),
              restoreData,
              bytes});
    }
  }

  const auto* configuredBounce =
      std::getenv("CUDF_PACKED_RESTORE_PINNED_BOUNCE_BYTES");
  const bool bounceDisabled =
      configuredBounce != nullptr && std::string_view(configuredBounce) == "0";
  if (!bounceDisabled) {
    // Job 144 has already used this process-wide pool for D2H spill before
    // finalize begins. Exclude first-use cudaHostAlloc cost from this
    // steady-state restore comparison.
    auto first = acquireCudfPackedPinnedBuffer(128ULL << 20);
    auto second = acquireCudfPackedPinnedBuffer(128ULL << 20);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
  }
  const auto start = std::chrono::steady_clock::now();
  auto restored =
      bulkRestoreCudfPackedHostChunks(std::move(chunks), stream, mr);
  const auto elapsedMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  const auto totalBytes = bytes * kChunks;
  if (useMaterializer && !bounceDisabled) {
    EXPECT_EQ(restored.stats().pinnedBounceBytes, totalBytes);
    EXPECT_EQ(restored.stats().pageableDirectBytes, 0);
  } else {
    EXPECT_EQ(restored.stats().pinnedBounceBytes, 0);
    EXPECT_EQ(restored.stats().pageableDirectBytes, totalBytes);
  }
  ASSERT_EQ(restored.tables().size(), kChunks);
  for (const auto index : {size_t{0}, kChunks - 1}) {
    const auto& restoredTable = restored.tables()[index];
    ASSERT_EQ(restoredTable.num_rows(), kRowsPerChunk);
    int32_t first = 0;
    CUDF_CUDA_TRY(cudaMemcpy(
        &first,
        restoredTable.column(0).data<int32_t>(),
        sizeof(first),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(first, 144);
  }
  std::cout << "Job144-shaped bulk restore bytes=" << totalBytes
            << " elapsedUs=" << elapsedMicros
            << " sourcePinned=" << sourcePinned
            << " materializer=" << useMaterializer
            << " pinnedBounceBytes=" << restored.stats().pinnedBounceBytes
            << " pageableDirectBytes=" << restored.stats().pageableDirectBytes
            << " pinnedBounceCopies=" << restored.stats().pinnedBounceCopies
            << " hostStageUs=" << restored.stats().hostStageMicros
            << " reuseWaitUs=" << restored.stats().bounceReuseWaitMicros
            << " copySyncUs=" << restored.stats().copyStreamSynchronizeMicros
            << " parallelHostStageGroups="
            << restored.stats().parallelHostStageGroups
            << " parallelHostStageChunks="
            << restored.stats().parallelHostStageChunks << std::endl;
}

TEST(CudfPackedSpillTest, DISABLED_job144GraceBuildRestoreBenchmark) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  size_t numChunks = 34;
  if (const auto* value = std::getenv("CUDF_PACKED_RESTORE_BENCH_CHUNKS")) {
    char* end = nullptr;
    const auto requested = std::strtoull(value, &end, 10);
    if (end != value && *end == '\0' && requested > 0 && requested <= 68) {
      numChunks = requested;
    }
  }
  constexpr size_t kRowsPerChunk = (30ULL << 20) / sizeof(int32_t);
  std::vector<int32_t> values(kRowsPerChunk, 144);
  auto column = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32},
      values.size(),
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      column->mutable_view().data<int32_t>(),
      values.data(),
      values.size() * sizeof(int32_t),
      cudaMemcpyHostToDevice,
      stream.value()));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  cudf::table source(std::move(columns));
  auto packed = cudf::pack(source.view(), stream, mr);
  const auto bytes = packed.gpu_data->size();
  auto hostData = std::shared_ptr<uint8_t>(
      new uint8_t[bytes], std::default_delete<uint8_t[]>());
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      hostData.get(),
      packed.gpu_data->data(),
      bytes,
      cudaMemcpyDeviceToHost,
      stream.value()));
  stream.synchronize();

  // Exclude first-use cudaHostAlloc cost. These are the same two bounded
  // 128-MiB slots available to production bulk restore.
  auto first = acquireCudfPackedPinnedBuffer(128ULL << 20);
  auto second = acquireCudfPackedPinnedBuffer(128ULL << 20);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  first.reset();
  second.reset();

  const auto* legacyValue = std::getenv("CUDF_PACKED_RESTORE_BENCH_LEGACY");
  const bool legacy =
      legacyValue != nullptr && std::string_view(legacyValue) == "1";
  const auto* residentValue = std::getenv("CUDF_PACKED_RESTORE_BENCH_RESIDENT");
  const bool resident =
      residentValue != nullptr && std::string_view(residentValue) == "1";
  const auto* prePinnedValue =
      std::getenv("CUDF_PACKED_RESTORE_BENCH_PREPINNED");
  const bool prePinned =
      prePinnedValue != nullptr && std::string_view(prePinnedValue) == "1";
  const auto* residentBounceValue =
      std::getenv("CUDF_PACKED_RESTORE_BENCH_RESIDENT_BOUNCE");
  const bool residentBounce = residentBounceValue != nullptr &&
      std::string_view(residentBounceValue) == "1";
  std::shared_ptr<uint8_t> directData = hostData;
  if (prePinned) {
    directData = acquireCudfPackedPinnedBuffer(bytes);
    ASSERT_NE(directData, nullptr);
    std::memcpy(directData.get(), hostData.get(), bytes);
  }
  const auto start = std::chrono::steady_clock::now();
  std::unique_ptr<cudf::table> combined;
  CudfBulkPackedRestore bulkRestored;
  CudfBulkPackedRestoreStats bulkStats;
  if (legacy) {
    std::vector<std::unique_ptr<cudf::packed_table>> restored;
    std::vector<cudf::table_view> views;
    restored.reserve(numChunks);
    views.reserve(numChunks);
    for (size_t chunk = 0; chunk < numChunks; ++chunk) {
      std::shared_ptr<uint8_t> pinned;
      const uint8_t* sourceData = directData.get();
      if (!resident) {
        pinned = acquireCudfPackedPinnedBuffer(bytes);
        ASSERT_NE(pinned, nullptr);
        std::memcpy(pinned.get(), hostData.get(), bytes);
        sourceData = pinned.get();
      }
      auto gpuData = std::make_unique<rmm::device_buffer>(bytes, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          gpuData->data(),
          sourceData,
          bytes,
          cudaMemcpyHostToDevice,
          stream.value()));
      stream.synchronize();
      cudf::packed_columns restoredColumns{
          std::make_unique<std::vector<uint8_t>>(*packed.metadata),
          std::move(gpuData)};
      auto view = cudf::unpack(restoredColumns);
      restored.push_back(
          std::make_unique<cudf::packed_table>(
              cudf::packed_table{view, std::move(restoredColumns)}));
      views.push_back(restored.back()->table);
    }
    combined = cudf::concatenate(views, stream, mr);
    stream.synchronize();
  } else {
    std::vector<CudfPackedHostRestoreChunk> chunks;
    chunks.reserve(numChunks);
    for (size_t chunk = 0; chunk < numChunks; ++chunk) {
      CudfPackedHostRestoreChunk restoreChunk;
      restoreChunk.metadata =
          std::make_unique<std::vector<uint8_t>>(*packed.metadata);
      restoreChunk.dataBytes = bytes;
      if (resident) {
        restoreChunk.data = directData;
        restoreChunk.stageResidentPageableThroughPinned = residentBounce;
      } else {
        restoreChunk.materializeIntoPinned = [hostData,
                                              bytes](uint8_t* destination) {
          std::memcpy(destination, hostData.get(), bytes);
        };
      }
      chunks.push_back(std::move(restoreChunk));
    }
    bulkRestored =
        bulkRestoreCudfPackedHostChunks(std::move(chunks), stream, mr);
    bulkStats = bulkRestored.stats();
    combined = cudf::concatenate(bulkRestored.tables(), stream, mr);
    stream.synchronize();
  }
  const auto elapsedMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  ASSERT_NE(combined, nullptr);
  ASSERT_EQ(combined->num_rows(), numChunks * kRowsPerChunk);
  int32_t firstValue = 0;
  int32_t lastValue = 0;
  CUDF_CUDA_TRY(cudaMemcpy(
      &firstValue,
      combined->view().column(0).data<int32_t>(),
      sizeof(firstValue),
      cudaMemcpyDeviceToHost));
  CUDF_CUDA_TRY(cudaMemcpy(
      &lastValue,
      combined->view().column(0).data<int32_t>() + combined->num_rows() - 1,
      sizeof(lastValue),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(firstValue, 144);
  EXPECT_EQ(lastValue, 144);
  std::cout << "Job144 Grace build restore benchmark mode="
            << (legacy ? "legacy" : "bulk") << " resident=" << resident
            << " prePinned=" << prePinned
            << " residentBounce=" << residentBounce << " chunks=" << numChunks
            << " bytes=" << bytes * numChunks << " elapsedUs=" << elapsedMicros
            << " pinnedBounceBytes=" << bulkStats.pinnedBounceBytes
            << " residentPageableBounceBytes="
            << bulkStats.residentPageableBounceBytes
            << " pageableDirectBytes=" << bulkStats.pageableDirectBytes
            << " pinnedBounceCopies=" << bulkStats.pinnedBounceCopies
            << " hostStageUs=" << bulkStats.hostStageMicros
            << " reuseWaitUs=" << bulkStats.bounceReuseWaitMicros
            << " copySyncUs=" << bulkStats.copyStreamSynchronizeMicros
            << std::endl;
}

TEST(CudfPackedSpillTest, DISABLED_graceBuildHostStorageBenchmark) {
  const auto stream = cudf::get_default_stream();
  const auto mr = cudf::get_current_device_resource_ref();
  constexpr size_t kChunks = 8;
  constexpr size_t kChunkBytes = 32ULL << 20;
  constexpr size_t kTotalBytes = kChunks * kChunkBytes;
  rmm::device_buffer gpuData(kTotalBytes, stream, mr);
  CUDF_CUDA_TRY(
      cudaMemsetAsync(gpuData.data(), 0x5a, kTotalBytes, stream.value()));
  stream.synchronize();

  // Exclude first-use pinned allocation; the production pool is warm after
  // the first Grace input slice.
  auto warmPinned = acquireCudfPackedPinnedBuffer(kTotalBytes);
  ASSERT_NE(warmPinned, nullptr);
  warmPinned.reset();
  const auto* directValue =
      std::getenv("CUDF_HASH_JOIN_BUILD_STORAGE_BENCH_DIRECT");
  const bool direct =
      directValue != nullptr && std::string_view(directValue) == "1";
  const bool shared =
      directValue != nullptr && std::string_view(directValue) == "shared";
  const size_t parallelWorkers = directValue == nullptr
      ? 0
      : (std::string_view(directValue) == "parallel2"
             ? 2
             : (std::string_view(directValue) == "parallel4"
                    ? 4
                    : (std::string_view(directValue) == "parallel" ? 8 : 0)));

  std::vector<std::shared_ptr<uint8_t>> pageable(kChunks);
  const auto start = std::chrono::steady_clock::now();
  if (direct) {
    for (auto& chunk : pageable) {
      chunk = std::shared_ptr<uint8_t>(
          new uint8_t[kChunkBytes], std::default_delete<uint8_t[]>());
    }
    for (size_t chunk = 0; chunk < kChunks; ++chunk) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          pageable[chunk].get(),
          static_cast<const uint8_t*>(gpuData.data()) + chunk * kChunkBytes,
          kChunkBytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    stream.synchronize();
  } else {
    auto pinned = acquireCudfPackedPinnedBuffer(kTotalBytes);
    ASSERT_NE(pinned, nullptr);
    for (size_t chunk = 0; chunk < kChunks; ++chunk) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          pinned.get() + chunk * kChunkBytes,
          static_cast<const uint8_t*>(gpuData.data()) + chunk * kChunkBytes,
          kChunkBytes,
          cudaMemcpyDeviceToHost,
          stream.value()));
    }
    stream.synchronize();
    if (shared) {
      auto allocation = std::shared_ptr<uint8_t>(
          new uint8_t[kTotalBytes], std::default_delete<uint8_t[]>());
      std::memcpy(allocation.get(), pinned.get(), kTotalBytes);
      for (size_t chunk = 0; chunk < kChunks; ++chunk) {
        pageable[chunk] = std::shared_ptr<uint8_t>(
            allocation, allocation.get() + chunk * kChunkBytes);
      }
    } else if (parallelWorkers > 0) {
      for (auto& chunk : pageable) {
        chunk = std::shared_ptr<uint8_t>(
            new uint8_t[kChunkBytes], std::default_delete<uint8_t[]>());
      }
      std::vector<std::future<void>> copies;
      copies.reserve(parallelWorkers);
      for (size_t worker = 0; worker < parallelWorkers; ++worker) {
        copies.push_back(std::async(std::launch::async, [&, pinned, worker]() {
          for (size_t chunk = worker; chunk < kChunks;
               chunk += parallelWorkers) {
            std::memcpy(
                pageable[chunk].get(),
                pinned.get() + chunk * kChunkBytes,
                kChunkBytes);
          }
        }));
      }
      for (auto& copy : copies) {
        copy.get();
      }
    } else {
      for (size_t chunk = 0; chunk < kChunks; ++chunk) {
        pageable[chunk] = std::shared_ptr<uint8_t>(
            new uint8_t[kChunkBytes], std::default_delete<uint8_t[]>());
        std::memcpy(
            pageable[chunk].get(),
            pinned.get() + chunk * kChunkBytes,
            kChunkBytes);
      }
    }
  }
  const auto elapsedMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count();
  ASSERT_EQ(pageable.front().get()[0], 0x5a);
  ASSERT_EQ(pageable.back().get()[kChunkBytes - 1], 0x5a);
  std::cout << "Grace build host storage benchmark mode="
            << (direct ? "direct-pageable"
                       : (shared ? "shared-demote"
                                 : (parallelWorkers > 0 ? "parallel-demote"
                                                        : "pinned-demote")))
            << " workers=" << parallelWorkers << " chunks=" << kChunks
            << " bytes=" << kTotalBytes << " elapsedUs=" << elapsedMicros
            << std::endl;
}

} // namespace
} // namespace facebook::velox::cudf_velox
