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
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


class Phase0WorkflowContractTest(unittest.TestCase):
    def test_only_lightweight_workflows_have_pull_request_triggers(self):
        pull_request_workflows = []
        for workflow in WORKFLOW_DIR.glob("*.yml"):
            contents = workflow.read_text(encoding="utf-8")
            if re.search(r"^  pull_request:\s*$", contents, re.MULTILINE):
                pull_request_workflows.append(workflow.name)

        self.assertEqual(
            ["breeze.yml", "docs.yml", "preliminary_checks.yml"],
            sorted(pull_request_workflows),
        )

    def test_phase0_targets_dev_and_has_stable_gate(self):
        contents = (WORKFLOW_DIR / "preliminary_checks.yml").read_text(encoding="utf-8")

        self.assertIn("      - dev", contents)
        self.assertIn("name: Velox Phase 0 / Gate", contents)
        self.assertNotIn("uses: ./.github/workflows/blossom-ci.yml", contents)
        self.assertEqual(5, contents.count("timeout-minutes: 5"))

    def test_phase0_runs_its_unit_and_contract_tests(self):
        contents = (WORKFLOW_DIR / "preliminary_checks.yml").read_text(encoding="utf-8")

        self.assertIn("name: Velox Phase 0 / Unit and Contract Tests", contents)
        self.assertIn("python3 -m unittest discover", contents)
        self.assertIn("--diff-filter=ACMRD", contents)
        self.assertIn("- self-tests", contents)

    def test_blossom_retains_the_real_internal_ci_entrypoints(self):
        contents = (WORKFLOW_DIR / "blossom-ci.yml").read_text(encoding="utf-8")

        self.assertIn("issue_comment:", contents)
        self.assertIn("workflow_dispatch:", contents)
        self.assertIn("runs-on: blossom", contents)
        self.assertIn("OPERATION: AUTH", contents)
        self.assertIn("OPERATION: START-CI-JOB", contents)
        self.assertIn("REPO_KEY_DATA: ${{ secrets.BLOSSOM_KEY }}", contents)
        self.assertIn("pull-requests: read", contents)
        self.assertIn("name: Blossom runner smoke", contents)
        self.assertIn("command -v blossom-ci", contents)
        self.assertIn(
            "actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd", contents
        )
        self.assertNotIn("NVIDIA/spark-rapids-common/checkout@", contents)
        self.assertNotIn("@main", contents)
        self.assertNotIn("actions/setup-java@v5", contents)
        self.assertNotIn("PASS BY", contents)

    def test_documentation_push_is_limited_to_dev(self):
        contents = (WORKFLOW_DIR / "docs.yml").read_text(encoding="utf-8")

        self.assertRegex(
            contents,
            r"(?ms)^  push:\n    branches:\n      - dev\n    paths:",
        )

    def test_documentation_check_is_available_for_relevant_fork_prs(self):
        contents = (WORKFLOW_DIR / "docs.yml").read_text(encoding="utf-8")

        self.assertRegex(
            contents,
            r"(?ms)^  pull_request:\n    branches:\n      - dev\n    paths:",
        )
        self.assertIn("HighPerfDataAccelerator/velox", contents)
        self.assertIn("'ubuntu-latest' || '16-core-ubuntu'", contents)
        self.assertIn("timeout-minutes: 5", contents)
        self.assertIn("persist-credentials: false", contents)
        self.assertIn("contents: read", contents)

    def test_breeze_cpu_check_is_available_for_relevant_fork_prs(self):
        contents = (WORKFLOW_DIR / "breeze.yml").read_text(encoding="utf-8")
        cpu_job, gpu_job = contents.split("  ubuntu-gpu-relwithdebinfo:", maxsplit=1)

        self.assertRegex(
            contents,
            r"(?ms)^  pull_request:\n    branches:\n      - dev\n    paths:",
        )
        self.assertIn("timeout-minutes: 5", cpu_job)
        self.assertIn("HighPerfDataAccelerator/velox", cpu_job)
        self.assertIn("github.event_name == 'pull_request'", cpu_job)
        self.assertIn("github.event_name == 'workflow_dispatch'", cpu_job)
        self.assertNotIn("HighPerfDataAccelerator/velox", gpu_job)

    def test_retained_heavy_workflows_have_non_pr_entrypoints(self):
        retained_workflows = {
            "benchmark.yml",
            "breeze.yml",
            "build-metrics.yml",
            "build_pyvelox.yml",
            "dependency-graph.yml",
            "docker.yml",
            "docs.yml",
            "linux-build.yml",
            "macos.yml",
            "scheduled.yml",
            "ubuntu-bundled-deps.yml",
        }

        for name in retained_workflows:
            with self.subTest(workflow=name):
                contents = (WORKFLOW_DIR / name).read_text(encoding="utf-8")
                self.assertRegex(
                    contents,
                    r"(?m)^  (workflow_dispatch|schedule|push):",
                )


if __name__ == "__main__":
    unittest.main()
