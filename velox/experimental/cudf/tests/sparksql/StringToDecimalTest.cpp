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

#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/SparkFunctions.h"
#include "velox/experimental/cudf/tests/CudfFunctionBaseTest.h"

#include "velox/functions/sparksql/registration/Register.h"
#include "velox/parse/TypeResolver.h"

namespace facebook::velox::cudf_velox {
namespace {

class StringToDecimalTest : public CudfFunctionBaseTest {
 protected:
  static void SetUpTestCase() {
    parse::registerTypeResolver();
    functions::sparksql::registerFunctions("");
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
    registerCudf();
    registerSparkFunctions("");
  }

  static void TearDownTestCase() {
    unregisterFunctions();
    unregisterCudf();
  }

  void assertMatchesCpu(
      const std::string& expression,
      std::vector<std::optional<StringView>> values) {
    auto input =
        makeRowVector({makeNullableFlatVector<StringView>(std::move(values))});
    assertExpressionMatchesCpu(expression, input, input->rowType());
  }
};

TEST_F(StringToDecimalTest, shortDecimal) {
  assertMatchesCpu(
      "try_cast(c0 as decimal(12, 2))",
      {"9999999999.99",
       "15",
       "1.5",
       "-1.5",
       "1.556",
       "1.554",
       "0000.123",
       ".123",
       "9.",
       "3E2",
       "-3E+2",
       "3E-2",
       "3.5E-2",
       "31.523e-2",
       " -3E+2 ",
       "not-a-number",
       "",
       std::nullopt});
}

TEST_F(StringToDecimalTest, longDecimal) {
  assertMatchesCpu(
      "try_cast(c0 as decimal(38, 0))",
      {"99999999999999999999999999999999999999",
       "-99999999999999999999999999999999999999",
       "100000000000000000000000000000000000000",
       "1.5",
       "-1.5",
       "1.23e37",
       "1.23e67",
       " 1.23 ",
       "1. 23",
       std::nullopt});
}

TEST_F(StringToDecimalTest, longDecimalWithScale) {
  assertMatchesCpu(
      "try_cast(c0 as decimal(20, 4))",
      {"112345612.23e-6",
       "112345662.23e-6",
       "1.23e-6",
       "1.26e-3",
       "1.23456781e3",
       "1.23456789e3",
       "1.23456789123451789123456789e9",
       "1.23456789123456789123456789e9",
       std::nullopt});

  assertMatchesCpu(
      "try_cast(c0 as decimal(38, 38))",
      {"0.999999999999999999999999999999999999992",
       "0.999999999999999999999999999999999999996",
       "111111111111111111.23",
       std::nullopt});
}

TEST_F(StringToDecimalTest, invalidValuesAndPrecisionOverflow) {
  assertMatchesCpu(
      "try_cast(c0 as decimal(5, 0))",
      {"0",
       "80",
       "81",
       "99999",
       "100000",
       "-100000",
       "+",
       ".",
       "9e",
       "-3E+",
       "-3E+2.1",
       "   ",
       std::nullopt});
}

TEST_F(StringToDecimalTest, validationPredicate) {
  assertMatchesCpu(
      "not(rlike(c0, '^[0-9]+$')) OR "
      "isnull(try_cast(c0 as decimal(38, 0))) OR "
      "decimal_greaterthan("
      "try_cast(c0 as decimal(38, 0)), "
      "cast(80 as decimal(38, 0)))",
      {"0",
       "80",
       "81",
       "99999999999999999999999999999999999999",
       "100000000000000000000000000000000000000",
       "-1",
       "not-a-number",
       std::nullopt});
}

} // namespace
} // namespace facebook::velox::cudf_velox
