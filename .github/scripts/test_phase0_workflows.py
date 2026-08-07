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

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE0_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "preliminary_checks.yml"


class Phase0WorkflowContractTest(unittest.TestCase):
    def setUp(self):
        self.contents = PHASE0_WORKFLOW.read_text(encoding="utf-8")

    def test_phase0_targets_dev_and_has_stable_gate(self):
        self.assertRegex(
            self.contents,
            r"(?ms)^  pull_request:\n    branches:\n      - dev\n    types:",
        )
        self.assertIn("name: Velox Phase 0 / Gate", self.contents)
        self.assertIn("    if: always()", self.contents)
        self.assertNotIn("uses: ./.github/workflows/", self.contents)
        self.assertEqual(5, self.contents.count("timeout-minutes: 5"))
        self.assertEqual(5, self.contents.count("runs-on: 4-core-ubuntu"))

    def test_phase0_runs_its_unit_and_contract_tests(self):
        self.assertIn("name: Velox Phase 0 / Unit and Contract Tests", self.contents)
        self.assertIn("python3 -m unittest discover", self.contents)
        self.assertIn("-p 'test_*.py'", self.contents)
        self.assertIn("--diff-filter=ACMRD", self.contents)
        self.assertIn("- self-tests", self.contents)

    def test_phase0_classification_is_exposed_to_followup_work(self):
        self.assertIn("name: Velox Phase 0 / Change Classification", self.contents)
        self.assertIn(
            "python3 .github/scripts/classify_phase0_changes.py", self.contents
        )
        for output in ("classification", "native_required", "spark_mpp"):
            self.assertIn(
                f"{output}: ${{{{ steps.classify.outputs.{output} }}}}",
                self.contents,
            )

    def test_phase0_uses_immutable_actions_without_write_credentials(self):
        action_refs = re.findall(r"uses: [^@\s]+@([^\s]+)", self.contents)
        self.assertTrue(action_refs)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in action_refs))
        self.assertEqual(3, self.contents.count("persist-credentials: false"))
        self.assertIn("permissions:\n  contents: read", self.contents)


if __name__ == "__main__":
    unittest.main()
