#!/usr/bin/env bash

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

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

(($# == 3)) ||
  die "Usage: $0 GLUTEN_DIR RESULT_DIR CUDF_VERSION_INFO"

gluten_dir=$1
result_dir=$2
cudf_version_info=$3
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
velox_home="$(cd -- "${script_dir}/../.." && pwd -P)"
cuda_arch=${CUDA_ARCH:-75-real}
num_threads=${NUM_THREADS:-4}

[[ -d ${gluten_dir} ]] || die "Spark-Gluten directory is missing: ${gluten_dir}"
gluten_dir="$(cd -- "${gluten_dir}" && pwd -P)"
build_entrypoint="${gluten_dir}/dev/builddeps-veloxbe.sh"
[[ -x ${build_entrypoint} ]] ||
  die "Spark-Gluten build entrypoint is not executable: ${build_entrypoint}"

[[ -r ${cudf_version_info} ]] ||
  die "SYSTEM-cuDF version info is unreadable: ${cudf_version_info}"
cudf_version_info="$(
  cd -- "$(dirname -- "${cudf_version_info}")" && pwd -P
)/$(basename -- "${cudf_version_info}")"

[[ ${cuda_arch} =~ ^[0-9]+-real$ ]] ||
  die "CUDA_ARCH must be a real architecture such as 75-real"
[[ ${num_threads} =~ ^[1-9][0-9]*$ ]] ||
  die "NUM_THREADS must be a positive integer"

mkdir -p "${result_dir}"
result_dir="$(cd -- "${result_dir}" && pwd -P)"
junit_path="${result_dir}/velox-targeted.xml"
rm -f -- "${junit_path}"

export GLUTEN_DIR="${gluten_dir}"
export VELOX_HOME="${velox_home}"
export CUDA_ARCH="${cuda_arch}"
export NUM_THREADS="${num_threads}"
export TARGETS="velox libcurl_shared velox_cudf_config_test"

"${build_entrypoint}" \
  --enable_gpu=ON \
  --enable_hdfs=ON \
  --enable_s3=OFF \
  "--cuda_arch=${CUDA_ARCH}" \
  --cudf_source=SYSTEM \
  "--cudf_version_info=${cudf_version_info}" \
  --cudf_compatibility_check=ON \
  --rebuild_if_mismatch=OFF \
  --run_setup_script=OFF \
  --build_arrow=OFF \
  --build_tests=OFF \
  --build_examples=OFF \
  --build_benchmarks=OFF \
  "--velox_home=${VELOX_HOME}" \
  --build_velox_tests=ON \
  --build_velox_benchmarks=OFF \
  "--num_threads=${NUM_THREADS}" \
  build_velox

test_binary="${VELOX_HOME}/_build/release/velox/experimental/cudf/tests/velox_cudf_config_test"
[[ -x ${test_binary} ]] || die "Velox cuDF test binary is missing: ${test_binary}"

if [[ -f /opt/rh/gcc-toolset-14/enable ]]; then
  # shellcheck source=/dev/null
  source /opt/rh/gcc-toolset-14/enable
fi
export LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export UCX_MODULE_DIR=/usr/local/lib/ucx
unset GTEST_FILTER GTEST_OUTPUT GTEST_REPEAT \
  GTEST_SHARD_INDEX GTEST_TOTAL_SHARDS TESTBRIDGE_TEST_ONLY

"${test_binary}" "--gtest_output=xml:${junit_path}"

[[ -s ${junit_path} ]] || die "GTest did not produce JUnit XML: ${junit_path}"
if ! grep -q 'status="run"' "${junit_path}"; then
  echo "ERROR: GTest selected no tests: ${junit_path}" >&2
  exit 1
fi
