# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(velox_cudf_exchange_test
  Main.cpp
  GpuSerializedPageTest.cc
  GpuMultiFragmentTest.cc
  GpuLockStub.cc
)

target_link_libraries(
  velox_cudf_exchange_test
  velox_cudf_exchange
  velox_cudf_vector
  velox_cudf_exec
  velox_exec
  velox_exec_test_lib
  velox_test_util
  velox_dwio_dwrf_reader
  velox_hive_connector
  gtest
)

add_test(
  NAME velox_cudf_exchange_test
  COMMAND velox_cudf_exchange_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(
  velox_cudf_exchange_test
  PROPERTIES LABELS cuda_driver TIMEOUT 3000
)
