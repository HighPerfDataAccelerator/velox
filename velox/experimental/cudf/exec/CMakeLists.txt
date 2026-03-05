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

add_library(
  velox_cudf_exec
  CudfAssignUniqueId.cpp
  CudfConversion.cpp
  CudfFilterProject.cpp
  CudfHashAggregation.cpp
  CudfHashJoin.cpp
  CudfLimit.cpp
  CudfLocalPartition.cpp
  CudfOrderBy.cpp
  CudfShufflePartition.cpp
  CudfTopN.cpp
  DebugUtil.cpp
  OperatorAdapters.cpp
  PinnedArrowInterop.cpp
  ToCudf.cpp
  Utilities.cpp
  VeloxCudfInterop.cpp
)

target_link_libraries(
  velox_cudf_exec
  PUBLIC cudf::cudf
  PRIVATE
    arrow
    velox_arrow_bridge
    velox_exception
    velox_common_base
    velox_cudf_vector
    velox_exec
    velox_cudf_hive_connector
    velox_cudf_expression
)

# PinnedArrowInterop.cpp uses nanoarrow headers directly (cudf's private dep).
# nanoarrow is already linked into libcudf.so; we only need the headers + namespace define.
if(DEFINED nanoarrow_SOURCE_DIR)
  target_include_directories(velox_cudf_exec PRIVATE
    "${nanoarrow_SOURCE_DIR}/src"
    "${nanoarrow_BINARY_DIR}/src"
  )
else()
  # Fallback: find nanoarrow headers from the system or cudf install.
  find_path(NANOARROW_INCLUDE_DIR nanoarrow/nanoarrow.h
    HINTS ${CUDAToolkit_LIBRARY_DIR}/../include
          ${CMAKE_INSTALL_PREFIX}/include)
  if(NANOARROW_INCLUDE_DIR)
    target_include_directories(velox_cudf_exec PRIVATE "${NANOARROW_INCLUDE_DIR}")
  else()
    # Last resort: fetch nanoarrow headers (header-only, not linked).
    include(FetchContent)
    FetchContent_Declare(
      nanoarrow
      GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
      GIT_TAG apache-arrow-nanoarrow-0.7.0
      GIT_SHALLOW TRUE)
    FetchContent_GetProperties(nanoarrow)
    if(NOT nanoarrow_POPULATED)
      FetchContent_MakeAvailable(nanoarrow)
    endif()
    target_include_directories(velox_cudf_exec PRIVATE
      "${nanoarrow_SOURCE_DIR}/src"
      "${nanoarrow_BINARY_DIR}/src")
  endif()
endif()
target_compile_definitions(velox_cudf_exec PRIVATE NANOARROW_NAMESPACE=cudf)

target_compile_options(velox_cudf_exec PRIVATE -Wno-missing-field-initializers)
