"""Tests for the CI support matrix export helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "ci" / "export_support_matrix.py"


spec = importlib.util.spec_from_file_location("ci_support_matrix", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
ci_support_matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_support_matrix)


def test_airflow_pr_matrix_uses_primary_only() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "airflow", "pr")

    assert payload["python_version"] == "3.12"
    assert payload["version_matrix"] == {"version": ["3.1.8"]}


def test_dbt_main_matrix_includes_all_dispatch_targets() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "dbt", "main")

    assert payload["run_execution"] is True
    assert payload["execution_target"] == "postgres"
    targets = [entry["target"] for entry in payload["compile_matrix"]["include"]]
    assert targets == ["postgres", "snowflake", "bigquery", "redshift", "databricks"]


def test_docs_support_matrix_block_is_in_sync() -> None:
    data = ci_support_matrix.load_support_matrix()
    text = (ROOT / "docs" / "compatibility.md").read_text(encoding="utf-8")

    current = ci_support_matrix.extract_generated_block(text)

    assert current == ci_support_matrix.render_generated_support_block(data)
