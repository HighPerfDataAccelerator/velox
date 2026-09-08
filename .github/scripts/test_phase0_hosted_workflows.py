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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


class Phase0HostedWorkflowContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.breeze = (WORKFLOW_DIR / "breeze.yml").read_text(encoding="utf-8")
        cls.docs = (WORKFLOW_DIR / "docs.yml").read_text(encoding="utf-8")

    def test_breeze_has_path_selected_dev_pull_request_entrypoint(self):
        self.assertIn("  workflow_dispatch: {}", self.breeze)
        self.assertRegex(
            self.breeze,
            r"(?ms)^  pull_request:\n    branches:\n      - dev\n    paths:\n"
            r"(?:      - .+\n)+  push:",
        )
        self.assertIn("      - velox/experimental/breeze/**", self.breeze)
        self.assertIn("      - .github/workflows/breeze.yml", self.breeze)

    def test_breeze_exposes_only_cpu_lane_to_hpa(self):
        cpu_job, gpu_job = self.breeze.split("  ubuntu-gpu-relwithdebinfo:", maxsplit=1)

        self.assertIn("runs-on: ubuntu-22.04", cpu_job)
        self.assertIn("timeout-minutes: 5", cpu_job)
        self.assertIn("github.repository == 'facebookincubator/velox'", cpu_job)
        self.assertIn("github.repository == 'HighPerfDataAccelerator/velox'", cpu_job)
        self.assertIn("github.event_name == 'pull_request'", cpu_job)
        self.assertIn("github.event_name == 'workflow_dispatch'", cpu_job)
        self.assertIn("runs-on: 4-core-ubuntu-gpu-t4", gpu_job)
        self.assertIn("github.repository == 'facebookincubator/velox'", gpu_job)
        self.assertNotIn("HighPerfDataAccelerator/velox", gpu_job)

    def test_breeze_uses_read_only_checkout(self):
        self.assertIn("permissions:\n  contents: read", self.breeze)
        self.assertEqual(2, self.breeze.count("persist-credentials: false"))
        self.assertNotIn("persist-credentials: true", self.breeze)

    def test_docs_has_path_selected_dev_entrypoints(self):
        self.assertIn("  workflow_dispatch: {}", self.docs)
        for event in ("pull_request", "push"):
            with self.subTest(event=event):
                self.assertRegex(
                    self.docs,
                    rf"(?ms)^  {event}:\n    branches:\n      - dev\n    paths:\n"
                    r"      - velox/docs/\*\*\n"
                    r"      - \.github/workflows/docs\.yml\n",
                )

    def test_docs_uses_hpa_hosted_runner_with_bounded_read_only_job(self):
        self.assertIn("permissions:\n  contents: read", self.docs)
        self.assertIn(
            "runs-on: ${{ github.repository == 'HighPerfDataAccelerator/velox' "
            "&& 'ubuntu-latest' || '16-core-ubuntu' }}",
            self.docs,
        )
        self.assertRegex(
            self.docs,
            r"(?ms)^  build_docs:\n(?:    .+\n){2}    timeout-minutes: 5$",
        )
        self.assertIn("persist-credentials: false", self.docs)
        self.assertNotIn("persist-credentials: true", self.docs)

    def test_docs_never_publishes_from_hpa(self):
        self.assertIn(
            "github.event_name == 'push' && "
            "github.repository == 'facebookincubator/velox'",
            self.docs,
        )


if __name__ == "__main__":
    unittest.main()
