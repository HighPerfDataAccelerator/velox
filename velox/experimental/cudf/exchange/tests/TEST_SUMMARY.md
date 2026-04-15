# GPU Exchange Component Test Suite

This directory contains comprehensive unit and integration tests for the GPU exchange components in Velox's experimental cuDF integration.

## Test Files

### 1. GpuSerializedPageTest.cc

Unit tests for the `GpuSerializedPage` class, which wraps a GPU-resident CudfVector as a SerializedPageBase.

#### Test Cases:

- **ConstructFromCudfVector**: Verifies basic construction and that `size()` and `numRows()` return correct values from the underlying CudfVector.

- **NullInputRejects**: Ensures that passing a null CudfVector triggers a VELOX_CHECK failure with appropriate error message.

- **CudfVectorAccessor**: Confirms that `cudfVector()` returns the exact same shared_ptr that was used during construction.

- **MoveSemantics**: Tests that move constructor properly transfers ownership and that the moved-from object is left in a valid state.

- **SharedOwnership**: Verifies that multiple GpuSerializedPage instances can safely share the same underlying CudfVector, and that reference counting works correctly.

- **ZeroRowPage**: Handles edge case of an empty CudfVector (0 bytes, 0 rows).

- **PrepareStreamThrows**: Confirms that `prepareStreamForDeserialize()` throws VELOX_UNSUPPORTED, since GPU memory cannot be streamed directly to a ByteInputStream.

- **GetIOBufThrows**: Confirms that `getIOBuf()` throws VELOX_UNSUPPORTED, since GPU memory cannot be directly converted to IOBuf.

- **LargePageSize**: Tests that very large pages (100MB+) are handled correctly without overflow or memory issues.

- **MultipleAccessorsAreSafe**: Verifies that calling `cudfVector()` repeatedly is safe and consistent.

### 2. GpuMultiFragmentTest.cc

Integration tests for multi-fragment GPU exchange pipelines, following the Velox pattern from `MultiFragmentTest.cpp`.

#### Test Cases:

- **TwoFragmentHashPartition**: 
  - Fragment0: Values → GpuPartitionedOutput (hash partitioning to 3 partitions)
  - Fragment1: GpuExchange → Aggregation
  - Verifies: Correct hash-based distribution and aggregation results

- **TwoFragmentBroadcast**:
  - Fragment0: Values → GpuPartitionedOutput (broadcast)
  - Fragment1: GpuExchange (broadcast) on 2 consumer tasks
  - Verifies: All rows received by all consumers

- **ThreeFragmentChain**:
  - F0 → exchange → F1 → exchange → F2
  - Verifies: End-to-end correctness through chained exchanges

- **PipelineOverlap**:
  - Two fragments with concurrent execution
  - Verifies: Both tasks start within reasonable time window (overlapping pipeline)

- **LargeData**:
  - 150K rows through 2-fragment hash-partitioned pipeline
  - Verifies: Correctness at scale without data loss or corruption

- **EmptyPartitions**:
  - Hash partitioning where some output partitions receive no rows
  - Verifies: Correct handling of empty partitions and sparse data distribution

- **BackpressureThroughput**:
  - Large dataset through exchange with constrained buffer sizes
  - Verifies: Producer blocks when buffer full, resumes when consumer drains

## Component Interfaces Being Tested

### GpuSerializedPage
```cpp
class GpuSerializedPage : public SerializedPageBase {
  explicit GpuSerializedPage(std::shared_ptr<CudfVector> cudfVector);
  std::shared_ptr<CudfVector> cudfVector() const;
  uint64_t size() const override;
  std::optional<int64_t> numRows() const override;
  std::unique_ptr<ByteInputStream> prepareStreamForDeserialize() override;
  std::unique_ptr<folly::IOBuf> getIOBuf() const override;
};
```

### GpuPartitionedOutput
- Velox Operator that partitions GPU data and enqueues GpuSerializedPages to OutputBufferManager
- Supports hash, broadcast, and arbitrary partitioning schemes

### GpuExchange
- Velox SourceOperator that reads GpuSerializedPages from ExchangeClient
- Returns GPU-resident data (CudfVector) to downstream operators

### LocalGpuExchangeSource
- ExchangeSource factory for "gpu-local://" scheme
- Reads GPU pages from OutputBufferManager without CPU-side serialization

## Test Design Principles

1. **Unit vs Integration**: GpuSerializedPageTest focuses on single-component behavior; GpuMultiFragmentTest validates full pipelines.

2. **Error Handling**: Tests verify both success paths and error conditions (null inputs, unsupported operations).

3. **Edge Cases**: Tests include zero rows, large data, and sparse distributions.

4. **Concurrency**: Multi-fragment tests exercise concurrent Task execution and data flow.

5. **Memory Safety**: Tests verify shared ownership semantics and proper reference counting.

6. **GPU-First Design**: Tests confirm that data stays GPU-resident where possible (GpuSerializedPage used, not PrestoSerializedPage).

## Building and Running

### Build
```bash
cd /home/ferdinandx/mpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make velox_cudf_exchange_test
```

### Run All Tests
```bash
./velox/experimental/cudf/exchange/tests/velox_cudf_exchange_test
```

### Run Specific Test
```bash
./velox/experimental/cudf/exchange/tests/velox_cudf_exchange_test --gtest_filter="GpuSerializedPageTest.ConstructFromCudfVector"
```

### Run with Verbosity
```bash
./velox/experimental/cudf/exchange/tests/velox_cudf_exchange_test --gtest_filter="GpuMultiFragmentTest.*" -v
```

## Dependencies

- Velox core (exec, serializers, memory management)
- CUDA/cuDF libraries (for actual GPU operations when available)
- Google Test framework
- Folly utility library

## Notes

- Tests may not fully compile until source components (GpuSerializedPage.h, GpuExchange.h, etc.) are implemented
- Tests are written based on specified interfaces and will adapt as implementation details finalize
- Mock CudfVector used in unit tests; integration tests use actual Velox vector types
- Test framework follows Velox conventions (HiveConnectorTestBase, PlanBuilder, etc.)

## Future Enhancements

- Performance benchmarks for exchange throughput
- Memory pressure and spill scenarios
- Multi-GPU exchange testing
- Stress tests with high concurrency
- Error recovery and fault injection tests
