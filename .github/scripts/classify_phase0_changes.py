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

"""Classify pull-request changes for the lightweight Velox Phase 0 gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

CATEGORY_ORDER = (
    "docs-only",
    "workflow-only",
    "native-cpu",
    "spark-mpp-cudf",
    "spark-mpp-ucx",
    "dependency-build",
)

NON_NATIVE_CATEGORIES = {"docs-only", "workflow-only"}


def _is_documentation(path: str) -> bool:
    return (
        path.endswith(".md")
        or path.startswith(("docs/", "velox/docs/", "website/"))
        or path
        in {
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "README.md",
            "SECURITY.md",
        }
    )


def _is_workflow(path: str) -> bool:
    return path.startswith(".github/") or path in {
        ".pre-commit-config.yaml",
        ".yamllint",
        ".yamlfmt",
    }


def _is_dependency_build(path: str) -> bool:
    return (
        path == "CMakeLists.txt"
        or path == "Makefile"
        or path.startswith(("CMake/", "third_party/"))
        or path.startswith("scripts/setup-")
        or path
        in {
            "scripts/setup-common.sh",
            "scripts/setup-helper-functions.sh",
            "scripts/setup-versions.sh",
        }
    )


def category_for_path(path: str) -> str:
    """Return the most specific Phase 0 category for one repository path."""
    if _is_documentation(path):
        return "docs-only"
    if _is_workflow(path):
        return "workflow-only"
    if path.startswith("velox/experimental/cudf/"):
        return "spark-mpp-cudf"
    if path.startswith("velox/experimental/ucx-exchange/"):
        return "spark-mpp-ucx"
    if _is_dependency_build(path):
        return "dependency-build"
    return "native-cpu"


def classify_paths(paths: Iterable[str]) -> dict[str, object]:
    """Classify paths and return stable, machine-readable gate metadata."""
    normalized_paths = sorted(
        {path.strip().removeprefix("./") for path in paths if path.strip()}
    )
    if not normalized_paths:
        raise ValueError("No changed files were provided")

    categories = {category_for_path(path) for path in normalized_paths}
    ordered_categories = [
        category for category in CATEGORY_ORDER if category in categories
    ]
    native_required = not categories.issubset(NON_NATIVE_CATEGORIES)

    return {
        "classification": ",".join(ordered_categories),
        "native_required": native_required,
        "spark_mpp": bool(categories & {"spark-mpp-cudf", "spark-mpp-ucx"}),
        "changed_file_count": len(normalized_paths),
        "changed_files": normalized_paths,
    }


def _write_github_outputs(result: dict[str, object], output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as output:
        output.write(f"classification={result['classification']}\n")
        output.write(f"native_required={str(result['native_required']).lower()}\n")
        output.write(f"spark_mpp={str(result['spark_mpp']).lower()}\n")
        output.write(f"changed_file_count={result['changed_file_count']}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--changed-files",
        required=True,
        type=Path,
        help="File containing one repository-relative changed path per line.",
    )
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Optional GitHub Actions output file.",
    )
    args = parser.parse_args()

    result = classify_paths(args.changed_files.read_text(encoding="utf-8").splitlines())
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.github_output:
        _write_github_outputs(result, args.github_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
