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

usage() {
  cat <<'EOF'
Usage: run-cudf-premerge.sh \
  --gluten-dir DIR \
  --result-dir DIR \
  --cudf-version-info FILE \
  [--cuda-arch ARCH] \
  [--jobs N]

Builds the Velox cuDF configuration test with Spark-Gluten's SYSTEM-cuDF
adapter, then runs the complete test binary and writes JUnit XML.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
velox_home="$(cd -- "${script_dir}/../.." && pwd)"
gluten_dir=
result_dir=
cudf_version_info=
cuda_arch=75-real
num_threads=4

while (($# > 0)); do
  case "$1" in
  --gluten-dir)
    (($# >= 2)) || die "--gluten-dir requires a value"
    gluten_dir=$2
    shift 2
    ;;
  --result-dir)
    (($# >= 2)) || die "--result-dir requires a value"
    result_dir=$2
    shift 2
    ;;
  --cudf-version-info)
    (($# >= 2)) || die "--cudf-version-info requires a value"
    cudf_version_info=$2
    shift 2
    ;;
  --cuda-arch)
    (($# >= 2)) || die "--cuda-arch requires a value"
    cuda_arch=$2
    shift 2
    ;;
  --jobs)
    (($# >= 2)) || die "--jobs requires a value"
    num_threads=$2
    shift 2
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    die "unknown argument: $1"
    ;;
  esac
done

[[ -n ${gluten_dir} ]] || die "--gluten-dir is required"
[[ -n ${result_dir} ]] || die "--result-dir is required"
[[ -n ${cudf_version_info} ]] || die "--cudf-version-info is required"
[[ ${num_threads} =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"

[[ -d ${gluten_dir} ]] || die "Spark-Gluten directory is missing: ${gluten_dir}"
gluten_dir="$(cd -- "${gluten_dir}" && pwd -P)"
[[ -f ${cudf_version_info} && -r ${cudf_version_info} ]] ||
  die "SYSTEM-cuDF version info is missing or unreadable: ${cudf_version_info}"
cudf_version_info="$(
  cd -- "$(dirname -- "${cudf_version_info}")" && pwd -P
)/$(basename -- "${cudf_version_info}")"

if [[ ${cuda_arch} =~ ^[0-9]+$ ]]; then
  cuda_arch="${cuda_arch}-real"
fi
[[ ${cuda_arch} =~ ^([0-9]+)-real$ ]] ||
  die "--cuda-arch must be a numeric real architecture such as 75-real"
gpu_arch=${BASH_REMATCH[1]}

build_entrypoint="${gluten_dir}/dev/builddeps-veloxbe.sh"
[[ -x ${build_entrypoint} ]] ||
  die "Spark-Gluten build entrypoint is not executable: ${build_entrypoint}"

command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v tee >/dev/null 2>&1 || die "tee is required"

case ";${CUDA_ARCHITECTURES:-};" in
*";${gpu_arch};"*) ;;
*) die "builder image does not include SM${gpu_arch}" ;;
esac

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
gpu_capability="$(
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits |
    sed -n '1p' |
    tr -d '[:space:].'
)"
[[ ${gpu_capability} == "${gpu_arch}" ]] ||
  die "GPU compute capability ${gpu_capability:-unknown} does not match SM${gpu_arch}"

mkdir -p "${result_dir}"
result_dir="$(cd -- "${result_dir}" && pwd)"
junit_path="${result_dir}/velox-targeted.xml"
log_path="${result_dir}/velox-targeted.log"
rm -f -- "${junit_path}" "${log_path}"

export GLUTEN_DIR="${gluten_dir}"
export VELOX_HOME="${velox_home}"
export NUM_THREADS="${num_threads}"
export CUDA_ARCH="${cuda_arch}"
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
runtime_library_path=/usr/local/lib64:/usr/local/lib
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  runtime_library_path="${runtime_library_path}:${LD_LIBRARY_PATH}"
fi
export LD_LIBRARY_PATH="${runtime_library_path}"
export UCX_MODULE_DIR=/usr/local/lib/ucx
unset GTEST_FILTER GTEST_OUTPUT GTEST_REPEAT \
  GTEST_SHARD_INDEX GTEST_TOTAL_SHARDS TESTBRIDGE_TEST_ONLY

test_status=0
"${test_binary}" \
  "--gtest_output=xml:${junit_path}" \
  2>&1 | tee "${log_path}" || test_status=$?

[[ -s ${junit_path} ]] || die "GTest did not produce JUnit XML: ${junit_path}"
python3 - "${junit_path}" <<'PY'
import sys
import xml.etree.ElementTree as ET

junit_path = sys.argv[1]
test_cases = ET.parse(junit_path).getroot().findall(".//testcase")
executed = [
    case
    for case in test_cases
    if case.get("status") != "notrun" and case.get("result") != "suppressed"
]
if not executed:
    raise SystemExit(f"ERROR: GTest selected no tests: {junit_path}")
print(f"Verified {len(executed)} executed GTest case(s) in {junit_path}")
PY

if ((test_status != 0)); then
  exit "${test_status}"
fi
