# These ownership/backpressure regressions can also run without CUDA or the
# Velox dependency stack: cmake -S <this directory> -B <build directory>.
cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(VeloxCudfHostPipelineTests LANGUAGES CXX)
  enable_testing()
endif()

find_package(Threads REQUIRED)
get_filename_component(VELOX_HOST_TEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

foreach(test IN ITEMS BudgetedPageableCache BoundedBatchWriter)
  set(target "velox_cudf_host_${test}_test")
  add_executable(${target} "${test}Test.cpp")
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_include_directories(${target} PRIVATE "${VELOX_HOST_TEST_ROOT}")
  target_link_libraries(${target} PRIVATE Threads::Threads)
  if(DEFINED CUDF_TESTS_OUTPUT_DIR)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_TESTS_OUTPUT_DIR}")
  endif()
  # Keep assertions enabled even when the main tree is built in release mode.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -UNDEBUG)
  endif()
  add_test(NAME ${target} COMMAND ${target})
  set_tests_properties(${target} PROPERTIES TIMEOUT 60)
endforeach()
