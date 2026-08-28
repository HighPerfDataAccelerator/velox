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

#include <cstddef>
#include <cstdint>

// Spark-Gluten supplies strong definitions when Velox is part of libgluten.
// Standalone Velox tests need a fallback that reports the bridge unavailable.
extern "C" __attribute__((weak)) bool glutenCrtS3RangeReaderAvailable() {
  return false;
}

extern "C" __attribute__((weak)) uint64_t glutenCrtS3ObjectSize(const char*) {
  return 0;
}

extern "C" __attribute__((weak)) uint64_t glutenCrtS3ReadRanges(
    const char*,
    uint8_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    size_t) {
  return 0;
}
