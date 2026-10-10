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
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/exec/tests/utils/OperatorTestBase.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <folly/ScopeGuard.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <future>
#include <set>
#include <string>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::cudf_velox {
namespace {

constexpr int64_t kAllocSize = 8 << 20; // 8MB.

class GpuResourcesTest : public OperatorTestBase {
 protected:
  // The memory resource configuration has to be in place before registerCudf()
  // creates the resources, so each test registers cuDF itself.
  void SetUp() override {
    OperatorTestBase::SetUp();
    auto& config = CudfConfig::getInstance();
    savedMemoryResource_ = config.memoryResource;
    savedOutputMemoryResource_ = config.outputMemoryResource;
  }

  void TearDown() override {
    unregisterCudf();
    auto& config = CudfConfig::getInstance();
    config.memoryResource = savedMemoryResource_;
    config.outputMemoryResource = savedOutputMemoryResource_;
    OperatorTestBase::TearDown();
  }

  // Asserts that an allocation from each of 'resources' raises the live byte
  // count, and that freeing them returns it to the value it started at.
  void assertTracksAllocations(
      const std::vector<rmm::device_async_resource_ref>& resources) {
    const auto baseline = cudfAllocatedBytes();
    ASSERT_GE(baseline, 0);

    auto stream = cudfGlobalStreamPool().get_stream();
    {
      std::vector<rmm::device_buffer> buffers;
      buffers.reserve(resources.size());
      for (const auto& resource : resources) {
        buffers.emplace_back(kAllocSize, stream, resource);
      }
      // A resource may round the request up, hence the inequality.
      EXPECT_GE(
          cudfAllocatedBytes(),
          baseline + static_cast<int64_t>(resources.size()) * kAllocSize);
    }
    EXPECT_EQ(cudfAllocatedBytes(), baseline);
  }

  std::string savedMemoryResource_;
  std::string savedOutputMemoryResource_;
};

TEST_F(GpuResourcesTest, allocatedBytesWithoutCudf) {
  {
    SCOPED_TRACE("Before registration");
    ASSERT_FALSE(cudfIsRegistered());
    EXPECT_EQ(cudfAllocatedBytes(), -1);
  }

  registerCudf();
  ASSERT_GE(cudfAllocatedBytes(), 0);
  unregisterCudf();

  {
    SCOPED_TRACE("After unregistration");
    EXPECT_EQ(cudfAllocatedBytes(), -1);
  }
}

TEST_F(GpuResourcesTest, allocatedBytesTracksLiveAllocations) {
  registerCudf();
  ASSERT_TRUE(cudfIsRegistered());

  assertTracksAllocations({cudf::get_current_device_resource_ref()});
}

TEST_F(GpuResourcesTest, allocatedBytesIncludesSeparateOutputResource) {
  auto& config = CudfConfig::getInstance();
  config.memoryResource = "async";
  config.outputMemoryResource = "cuda";
  registerCudf();

  assertTracksAllocations(
      {cudf::get_current_device_resource_ref(), get_output_mr()});
}

} // namespace
} // namespace facebook::velox::cudf_velox

namespace facebook::velox::cudf_velox::test {
namespace {
std::set<cudaStream_t> handles(const std::vector<cuda::stream_ref>& streams) {
  std::set<cudaStream_t> result;
  for (auto stream : streams) {
    result.insert(stream.get());
  }
  return result;
}
} // namespace

TEST(GpuResourcesTest, boundedNonBlockingPool) {
  auto& pool = cudfGlobalStreamPool();
  EXPECT_TRUE(pool.get_streams(0).empty());
  const auto streams = pool.get_streams(65);
  EXPECT_EQ(streams.size(), 65);
  const auto unique = handles(streams);
  EXPECT_EQ(unique.size(), 32);
  for (auto stream : unique) {
    unsigned flags = 0;
    ASSERT_EQ(cudaStreamGetFlags(stream, &flags), cudaSuccess);
    EXPECT_EQ(flags, cudaStreamNonBlocking);
  }
  for (size_t i = 32; i < streams.size(); ++i) {
    EXPECT_EQ(streams[i].get(), streams[i - 32].get());
  }
  std::vector<cuda::stream_ref> directStreams;
  directStreams.reserve(65);
  for (size_t i = 0; i < 65; ++i) {
    directStreams.emplace_back(pool.get_stream());
  }
  EXPECT_EQ(handles(directStreams), unique);
  for (size_t i = 32; i < directStreams.size(); ++i) {
    EXPECT_EQ(directStreams[i].get(), directStreams[i - 32].get());
  }
}

TEST(GpuResourcesTest, concurrentBatchesAndThreadExit) {
  int device;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  auto* pool = &cudfGlobalStreamPool();
  const auto expected = handles(pool->get_streams(32));
  std::promise<void> start;
  auto ready = start.get_future().share();
  std::vector<std::future<cudaStream_t>> workers;
  constexpr int kNumWorkers = 16;
  workers.reserve(kNumWorkers);
  {
    // Release waiting workers before destroying their futures if launch throws.
    SCOPE_EXIT {
      start.set_value();
    };
    for (int i = 0; i < kNumWorkers; ++i) {
      workers.push_back(std::async(std::launch::async, [&, ready] {
        EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
        ready.wait();
        auto& shared = cudfGlobalStreamPool();
        EXPECT_EQ(&shared, pool);
        for (int j = 0; j < 100; ++j) {
          EXPECT_EQ(handles(shared.get_streams(32)), expected);
          EXPECT_EQ(expected.count(shared.get_stream().get()), 1);
        }
        return shared.get_stream().get();
      }));
    }
  }
  for (auto& worker : workers) {
    const auto retained = worker.get();
    // The worker has exited, but its stream must still support GPU work.
    void* data;
    ASSERT_EQ(cudaMallocAsync(&data, sizeof(int), retained), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(data, 0, sizeof(int), retained), cudaSuccess);
    int result = -1;
    ASSERT_EQ(
        cudaMemcpyAsync(
            &result, data, sizeof(int), cudaMemcpyDeviceToHost, retained),
        cudaSuccess);
    ASSERT_EQ(cudaFreeAsync(data, retained), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(retained), cudaSuccess);
    EXPECT_EQ(result, 0);
  }
}

TEST(GpuResourcesTest, deviceIsolation) {
  int count;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices";
  }
  int original;
  ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);
  auto* first = &cudfGlobalStreamPool();
  const auto firstStreams = handles(first->get_streams(32));
  const int secondDevice = (original + 1) % count;
  {
    SCOPE_EXIT {
      EXPECT_EQ(cudaSetDevice(original), cudaSuccess);
    };
    ASSERT_EQ(cudaSetDevice(secondDevice), cudaSuccess);
    auto* second = &cudfGlobalStreamPool();
    EXPECT_NE(first, second);
    for (auto stream : second->get_streams(32)) {
      int device;
      ASSERT_EQ(cudaStreamGetDevice(stream.get(), &device), cudaSuccess);
      EXPECT_EQ(device, secondDevice);
    }
  }
  EXPECT_EQ(&cudfGlobalStreamPool(), first);
  EXPECT_EQ(handles(first->get_streams(32)), firstStreams);
}

} // namespace facebook::velox::cudf_velox::test
