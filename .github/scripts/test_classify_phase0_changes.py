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

import unittest

from classify_phase0_changes import classify_paths


class ClassifyPhase0ChangesTest(unittest.TestCase):
    def test_docs_only_does_not_require_native(self):
        result = classify_paths(["README.md", "velox/docs/develop/index.rst"])
        self.assertEqual("docs-only", result["classification"])
        self.assertFalse(result["native_required"])

    def test_workflow_only_does_not_require_native(self):
        result = classify_paths(
            [".github/workflows/preliminary_checks.yml", ".yamllint"]
        )
        self.assertEqual("workflow-only", result["classification"])
        self.assertFalse(result["native_required"])

    def test_cudf_change_is_spark_mpp_native(self):
        result = classify_paths(["velox/experimental/cudf/exec/CudfGroupby.cpp"])
        self.assertEqual("spark-mpp-cudf", result["classification"])
        self.assertTrue(result["native_required"])
        self.assertTrue(result["spark_mpp"])

    def test_ucx_change_is_spark_mpp_native(self):
        result = classify_paths(
            ["velox/experimental/ucx-exchange/UcxPartitionedOutput.cpp"]
        )
        self.assertEqual("spark-mpp-ucx", result["classification"])
        self.assertTrue(result["native_required"])
        self.assertTrue(result["spark_mpp"])

    def test_build_system_change_requires_native(self):
        result = classify_paths(["CMakeLists.txt", "scripts/setup-ubuntu.sh"])
        self.assertEqual("dependency-build", result["classification"])
        self.assertTrue(result["native_required"])

    def test_regular_velox_source_is_native_cpu(self):
        result = classify_paths(["velox/exec/Task.cpp"])
        self.assertEqual("native-cpu", result["classification"])
        self.assertTrue(result["native_required"])
        self.assertFalse(result["spark_mpp"])

    def test_mixed_categories_have_stable_order(self):
        result = classify_paths(
            [
                "CMakeLists.txt",
                "README.md",
                "velox/experimental/ucx-exchange/UcxExchange.cpp",
                "velox/experimental/cudf/exec/CudfLimit.cpp",
            ]
        )
        self.assertEqual(
            "docs-only,spark-mpp-cudf,spark-mpp-ucx,dependency-build",
            result["classification"],
        )
        self.assertTrue(result["native_required"])

    def test_empty_change_set_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "No changed files"):
            classify_paths([])


if __name__ == "__main__":
    unittest.main()
