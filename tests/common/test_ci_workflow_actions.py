"""Regression tests for CI workflow action compatibility and helper scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PR_CHANGE_FILTER_PATH = ROOT / "scripts" / "ci" / "pr_change_filter.py"
DEPENDENCY_REVIEW_PATH = ROOT / "scripts" / "ci" / "dependency_review.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pr_change_filter = _load_module(PR_CHANGE_FILTER_PATH, "pr_change_filter")
dependency_review = _load_module(DEPENDENCY_REVIEW_PATH, "dependency_review")


def test_workflows_use_node24_capable_action_versions() -> None:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )

    assert "actions/checkout@v4" not in workflow_text
    assert "actions/setup-python@v5" not in workflow_text
    assert "actions/upload-artifact@v4" not in workflow_text
    assert "actions/download-artifact@v4" not in workflow_text
    assert "dorny/paths-filter@v3" not in workflow_text
    assert "actions/dependency-review-action@v4" not in workflow_text
    assert "github/codeql-action/init@v3" not in workflow_text
    assert "github/codeql-action/analyze@v3" not in workflow_text


def test_pr_change_filter_classifies_docs_only_changes() -> None:
    outputs = pr_change_filter.classify_changed_files(["docs/compatibility.md"])

    assert outputs == {
        "docs": True,
        "all_platforms": False,
        "airflow": False,
        "prefect": False,
        "dagster": False,
        "mage_kestra": False,
        "dbt": False,
        "docs_only": True,
    }


def test_pr_change_filter_classifies_platform_and_package_changes() -> None:
    outputs = pr_change_filter.classify_changed_files(
        [".github/workflows/security.yml", "packages/dbt/macros/truthound_check.sql"]
    )

    assert outputs["all_platforms"] is True
    assert outputs["dbt"] is True
    assert outputs["docs_only"] is False


def test_pr_change_filter_collects_previous_filenames_for_renames() -> None:
    paths = pr_change_filter.collect_changed_paths(
        [
            {"filename": "packages/dbt/new_name.sql", "previous_filename": "docs/old_name.md"},
            {"filename": "README.md"},
        ]
    )

    assert paths == ["README.md", "docs/old_name.md", "packages/dbt/new_name.sql"]


def test_dependency_review_only_flags_vulnerable_added_packages() -> None:
    changes = [
        {
            "change_type": "removed",
            "name": "safe-old",
            "version": "1.0.0",
            "manifest": "pyproject.toml",
            "vulnerabilities": [{"advisory_ghsa_id": "GHSA-old", "severity": "high"}],
        },
        {
            "change_type": "added",
            "name": "safe-new",
            "version": "2.0.0",
            "manifest": "pyproject.toml",
            "vulnerabilities": [],
        },
        {
            "change_type": "added",
            "name": "risky-new",
            "version": "3.0.0",
            "manifest": "requirements.txt",
            "vulnerabilities": [{"advisory_ghsa_id": "GHSA-risk", "severity": "critical"}],
        },
    ]

    findings = dependency_review.find_vulnerable_additions(changes)

    assert findings == [changes[2]]
    assert "risky-new 3.0.0 via requirements.txt: GHSA-risk (critical)" in dependency_review.format_findings(findings)
