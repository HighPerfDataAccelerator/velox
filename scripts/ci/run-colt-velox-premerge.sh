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

require_env() {
  local name="$1"
  [[ -n ${!name:-} ]] || die "${name} is required"
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

require_env COLT_ROOT
require_env VELOX_REPO
require_env VELOX_SHA
require_env GLUTEN_REPO
require_env GLUTEN_REF
require_env PREBUILD_IMAGES
require_env VELOX_TARGETS
require_env VELOX_TIERS
require_env CUDA_ARCH
require_env COLT_WORKSPACE
require_env COLT_RUN_DIR

COLT_RESULT_FILE="${COLT_RESULT_FILE:-${COLT_RUN_DIR}/result.json}"
COLT_GPU_DEVICES="${COLT_GPU_DEVICES:-0}"
COLT_PARALLEL="${COLT_PARALLEL:-4}"
COLT_MEM_LIMIT="${COLT_MEM_LIMIT:-20G}"
FAIL_ON_UNSTABLE="${FAIL_ON_UNSTABLE:-true}"

[[ -x "${COLT_ROOT}/colt" ]] || die "Colt executable not found: ${COLT_ROOT}/colt"
[[ ${VELOX_SHA} =~ ^[0-9a-fA-F]{40}$ ]] ||
  die "VELOX_SHA must be a full 40-character commit SHA"
[[ ${COLT_PARALLEL} =~ ^[1-9][0-9]*$ ]] ||
  die "COLT_PARALLEL must be a positive integer"

git -C "${VELOX_REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
  die "VELOX_REPO is not a Git worktree: ${VELOX_REPO}"
resolved_velox_sha="$(
  git -C "${VELOX_REPO}" rev-parse --verify "${VELOX_SHA}^{commit}" 2>/dev/null
)" || die "VELOX_SHA is not available in VELOX_REPO: ${VELOX_SHA}"
normalized_velox_sha="$(printf '%s' "${VELOX_SHA}" | tr '[:upper:]' '[:lower:]')"
[[ ${resolved_velox_sha} == "${normalized_velox_sha}" ]] ||
  die "VELOX_SHA resolved to an unexpected commit: ${resolved_velox_sha}"

declare -a targets=()
while IFS= read -r raw_target; do
  target="$(trim "${raw_target%%#*}")"
  [[ -z ${target} ]] && continue
  [[ ${target} =~ ^[^:[:space:]]+:[^[:space:].]+\.[^[:space:]]+$ ]] ||
    die "invalid Velox target '${target}'; expected BINARY:Suite.test"
  targets+=("${target}")
done <<<"${VELOX_TARGETS}"
((${#targets[@]} > 0)) || die "VELOX_TARGETS contains no targets"

declare -a tiers=()
while IFS= read -r tier; do
  [[ -z ${tier} ]] && continue
  tiers+=("${tier}")
done < <(printf '%s\n' "${VELOX_TIERS}" | tr ',[:space:]' '\n')
((${#tiers[@]} > 0)) || die "VELOX_TIERS contains no tiers"

declare -a prebuild_images=()
while IFS= read -r raw_image; do
  image="$(trim "${raw_image%%#*}")"
  [[ -z ${image} ]] && continue
  [[ ${image} != *[[:space:]]* ]] || die "invalid prebuild image: ${image}"
  prebuild_images+=("${image}")
done <<<"${PREBUILD_IMAGES}"
((${#prebuild_images[@]} > 0)) || die "PREBUILD_IMAGES contains no images"

if [[ -e ${COLT_RUN_DIR} && ! -d ${COLT_RUN_DIR} ]]; then
  die "COLT_RUN_DIR exists and is not a directory: ${COLT_RUN_DIR}"
fi
if [[ -d ${COLT_RUN_DIR} ]] &&
  [[ -n "$(find "${COLT_RUN_DIR}" -mindepth 1 -print -quit)" ]]; then
  die "COLT_RUN_DIR must be empty at startup: ${COLT_RUN_DIR}"
fi
[[ ! -e ${COLT_RESULT_FILE} ]] ||
  die "COLT_RESULT_FILE already exists: ${COLT_RESULT_FILE}"
mkdir -p "${COLT_WORKSPACE}" "${COLT_RUN_DIR}"

declare -a command=(
  "${COLT_ROOT}/colt"
  workflow
  verify
  --mode=run
  --scope=velox
  --worktree-mode=snapshot
  --build-mode=clean
  "--velox-repo=${VELOX_REPO}"
  "--velox-sha=${VELOX_SHA}"
  "--gluten-repo=${GLUTEN_REPO}"
  "--gluten-ref=${GLUTEN_REF}"
  --require-prebuild-match
  "--cuda-arch=${CUDA_ARCH}"
  "--workspace=${COLT_WORKSPACE}"
  "--run-dir=${COLT_RUN_DIR}"
  "--result-file=${COLT_RESULT_FILE}"
  --backend=docker
  "--gpu-devices=${COLT_GPU_DEVICES}"
  "--parallel=${COLT_PARALLEL}"
  "--mem-limit=${COLT_MEM_LIMIT}"
)

for tier in "${tiers[@]}"; do
  command+=("--tier=${tier}")
done
for target in "${targets[@]}"; do
  command+=("--target=${target}")
done
for image in "${prebuild_images[@]}"; do
  echo "Pulling Colt prebuild candidate: ${image}"
  if ! docker pull "${image}"; then
    echo "WARNING: unable to pull prebuild candidate: ${image}" >&2
  fi
  command+=("--prebuild-image=${image}")
done
if [[ ${FAIL_ON_UNSTABLE} == true ]]; then
  command+=(--fail-on-unstable)
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'

PYTHONUNBUFFERED=1 "${command[@]}"

required_outputs=(
  "${COLT_RUN_DIR}/workflow-manifest.json"
  "${COLT_RUN_DIR}/workflow-lock.json"
  "${COLT_RUN_DIR}/junit.xml"
  "${COLT_RUN_DIR}/workflow-summary.txt"
  "${COLT_RUN_DIR}/workflow-summary.html"
  "${COLT_RESULT_FILE}"
)
for output in "${required_outputs[@]}"; do
  [[ -s ${output} ]] || {
    echo "ERROR: Colt succeeded without required artifact: ${output}" >&2
    exit 1
  }
done
