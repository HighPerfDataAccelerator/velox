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
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "blossom-ci.yml"


def job(contents: str, name: str, next_name: Optional[str] = None) -> str:
    start = contents.index(f"  {name}:\n")
    if next_name is None:
        return contents[start:]
    end = contents.index(f"  {next_name}:\n", start)
    return contents[start:end]


class BlossomWorkflowContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contents = WORKFLOW.read_text(encoding="utf-8")

    def test_only_default_branch_controlled_and_manual_events_are_enabled(self):
        self.assertRegex(self.contents, r"(?m)^  issue_comment:\n")
        self.assertRegex(self.contents, r"(?m)^  workflow_dispatch:\n")
        self.assertNotRegex(
            self.contents,
            r"(?m)^  pull_request(?:_target)?:",
        )

        authorization = job(self.contents, "Authorization", "Vulnerability-scan")
        self.assertIn("github.event_name == 'issue_comment'", authorization)
        self.assertIn("github.event.issue.pull_request", authorization)
        self.assertIn("github.event.comment.body == 'build'", authorization)

        vulnerability_scan = job(
            self.contents,
            "Vulnerability-scan",
            "Job-trigger",
        )
        job_trigger = job(self.contents, "Job-trigger", "Runner-smoke")
        self.assertIn("needs: [Authorization]", vulnerability_scan)
        self.assertIn("needs: [Vulnerability-scan]", job_trigger)

    def test_actions_are_immutable_and_checkout_does_not_persist_credentials(self):
        actions = re.findall(r"(?m)^\s+-?\s*uses:\s*([^@\s]+)@([^\s#]+)", self.contents)
        expected_actions = {
            "NVIDIA/blossom-action": "2b0c950b993808dc31f80a1ccc32615902e39032",
            "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
            "actions/setup-java": "03ad4de0992f5dab5e18fcb136590ce7c4a0ac95",
        }
        self.assertEqual(len(expected_actions), len(actions))
        self.assertEqual(
            expected_actions,
            dict(actions),
        )
        for action, ref in actions:
            with self.subTest(action=action):
                self.assertRegex(ref, r"^[0-9a-f]{40}$")

        self.assertNotIn("NVIDIA/spark-rapids-common/checkout@", self.contents)
        self.assertIn("persist-credentials: false", self.contents)

    def test_token_permissions_are_read_only(self):
        permissions = self.contents.split("\npermissions:\n", maxsplit=1)[1].split(
            "\njobs:\n",
            maxsplit=1,
        )[0]
        self.assertEqual("  contents: read\n  pull-requests: read\n", permissions)
        self.assertNotRegex(self.contents, r"(?m)^\s+[^#\n]+:\s*write\s*$")

    def test_runner_smoke_is_manual_isolated_and_side_effect_free(self):
        runner_smoke = job(self.contents, "Runner-smoke", "Upload-Log")
        upload_log = job(self.contents, "Upload-Log")

        self.assertIn("inputs.mode == 'runner-smoke'", runner_smoke)
        self.assertIn("command -v blossom-ci", runner_smoke)
        self.assertNotIn("secrets.", runner_smoke)
        self.assertNotIn("OPERATION:", runner_smoke)
        self.assertNotIn("uses:", runner_smoke)

        self.assertIn("inputs.mode == 'post-processing'", upload_log)
        self.assertIn("OPERATION: POST-PROCESSING", upload_log)
        self.assertNotIn("inputs.mode == 'runner-smoke'", upload_log)

    def test_real_internal_entrypoints_remain_authorization_gated(self):
        self.assertIn("OPERATION: AUTH", self.contents)
        self.assertIn("OPERATION: START-CI-JOB", self.contents)
        self.assertIn("OPERATION: POST-PROCESSING", self.contents)
        self.assertIn("REPO_KEY_DATA: ${{ secrets.BLOSSOM_KEY }}", self.contents)
        self.assertNotIn("PASS BY", self.contents)


if __name__ == "__main__":
    unittest.main()
