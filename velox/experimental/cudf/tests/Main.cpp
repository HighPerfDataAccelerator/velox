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
#include "velox/common/memory/Memory.h"
#include "velox/common/process/ThreadDebugInfo.h"

#include <folly/Unit.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

// Gluten supplies these CRT S3 bridge symbols in production. Most standalone
// cuDF tests do not exercise S3, but their common connector link still needs a
// default implementation. S3-specific tests can override these weak stubs.
extern "C" bool __attribute__((weak)) glutenCrtS3RangeReaderAvailable() {
  return false;
}

extern "C" uint64_t __attribute__((weak)) glutenCrtS3ObjectSize(const char*) {
  return 0;
}

extern "C" uint64_t __attribute__((weak)) glutenCrtS3ReadRanges(
    const char*,
    uint8_t*,
    const uint64_t*,
    const uint64_t*,
    const uint64_t*,
    size_t) {
  return 0;
}

// This main is needed for some tests on linux.
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Signal handler required for ThreadDebugInfoTest
  facebook::velox::process::addDefaultFatalSignalHandler();
  folly::Init init(&argc, &argv, false);
  // Match the regular Velox exec test main. Without an initialized global
  // memory manager, fixtures that create vectors fail before SetUp() on their
  // first small allocation, so no cuDF operator code is exercised.
  facebook::velox::memory::MemoryManager::initialize({});
  return RUN_ALL_TESTS();
}
