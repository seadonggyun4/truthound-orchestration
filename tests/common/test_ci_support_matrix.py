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

    assert payload["host_matrix"] == {
        "include": [
            {
                "label": "primary",
                "version": "3.2.0",
                "python_version": "3.12",
                "constraints": [],
                "constraint_urls": [],
            }
        ]
    }


def test_prefect_main_matrix_pairs_minimum_host_with_python_311() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "prefect", "main")

    assert payload["host_matrix"] == {
        "include": [
            {
                "label": "min",
                "version": "2.14.0",
                "python_version": "3.11",
                "constraints": [
                    "anyio<4.0.0",
                    "griffe<1.0.0",
                    "pendulum<3.0.0",
                    "starlette<0.28.0",
                ],
                "constraint_urls": [],
            },
            {
                "label": "primary",
                "version": "3.6.22",
                "python_version": "3.12",
                "constraints": [],
                "constraint_urls": [],
            },
        ]
    }


def test_dagster_min_matrix_applies_legacy_pendulum_constraint() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "dagster", "main")

    assert payload["host_matrix"]["include"][0] == {
        "label": "min",
        "version": "1.5.0",
        "python_version": "3.11",
        "constraints": ["pendulum<3.0.0"],
        "constraint_urls": [],
    }


def test_dbt_main_matrix_includes_all_dispatch_targets() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "dbt", "main")

    assert payload["run_execution"] is True
    assert payload["execution_target"] == "postgres"
    targets = [entry["target"] for entry in payload["compile_matrix"]["include"]]
    assert targets == ["postgres", "snowflake", "bigquery", "redshift", "databricks"]


def test_mage_and_kestra_payloads_export_independent_python_lanes() -> None:
    data = ci_support_matrix.load_support_matrix()

    mage_payload = ci_support_matrix.build_workflow_payload(data, "mage", "main")
    kestra_payload = ci_support_matrix.build_workflow_payload(data, "kestra", "main")

    assert mage_payload == {
        "python_version": "3.12",
        "truthound_range": ">=3.0,<4.0",
    }
    assert kestra_payload == {
        "python_version": "3.12",
        "truthound_range": ">=3.0,<4.0",
    }


def test_security_audit_inputs_are_support_matrix_driven() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_security_audit_inputs(data)

    assert payload["python_version"] == "3.12"
    blocking = {entry["label"]: entry for entry in payload["blocking_matrix"]["include"]}
    canary = {entry["label"]: entry for entry in payload["canary_matrix"]["include"]}

    assert list(blocking) == [
        "base",
        "airflow",
        "prefect",
        "dagster",
        "dbt",
        "kestra",
        "opentelemetry",
    ]
    assert blocking["base"] == {
        "label": "base",
        "extra": "",
        "host_requirements": ["truthound==3.0.0"],
        "constraints": [],
        "ignore_vulns": [],
        "constraint_urls": [],
    }
    assert blocking["airflow"]["host_requirements"] == [
        "truthound==3.0.0",
        "apache-airflow==3.2.0",
    ]
    assert blocking["airflow"]["constraints"] == [
        "cryptography>=46.0.5",
        "pyjwt>=2.12.0",
    ]
    assert blocking["airflow"]["ignore_vulns"] == ["CVE-2025-62727"]
    assert blocking["airflow"]["constraint_urls"] == []
    assert blocking["prefect"]["host_requirements"] == [
        "truthound==3.0.0",
        "prefect==3.6.22",
        "starlette==0.49.1",
    ]
    assert blocking["prefect"]["constraints"] == [
        "cryptography>=46.0.5",
        "protobuf>=5.29.6",
        "pyjwt>=2.12.0",
        "starlette==0.49.1",
    ]
    assert blocking["prefect"]["ignore_vulns"] == []
    assert blocking["dbt"]["host_requirements"] == [
        "truthound==3.0.0",
        "dbt-core==1.10.20",
        "dbt-postgres==1.10.0",
    ]
    assert blocking["dbt"]["ignore_vulns"] == []
    assert blocking["kestra"]["host_requirements"] == [
        "truthound==3.0.0",
        "kestra==1.3.0",
    ]
    assert blocking["kestra"]["ignore_vulns"] == []
    assert blocking["opentelemetry"]["host_requirements"] == [
        "truthound==3.0.0",
        "opentelemetry-api==1.40.0",
        "opentelemetry-sdk==1.40.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.40.0",
        "opentelemetry-exporter-otlp-proto-http==1.40.0",
    ]
    assert blocking["opentelemetry"]["constraints"] == [
        "cryptography>=46.0.5",
        "googleapis-common-protos>=1.57,<2",
        "protobuf>=5.29.6,<6",
    ]
    assert blocking["opentelemetry"]["ignore_vulns"] == []
    assert canary == {
        "all": {
            "label": "all",
            "extra": "all",
            "host_requirements": ["truthound==3.0.0"],
            "constraints": [],
            "ignore_vulns": [],
            "constraint_urls": [],
        }
    }


def test_docs_support_matrix_block_is_in_sync() -> None:
    data = ci_support_matrix.load_support_matrix()
    text = (ROOT / "docs" / "compatibility.md").read_text(encoding="utf-8")

    current = ci_support_matrix.extract_generated_block(text)

    assert current == ci_support_matrix.render_generated_support_block(data)
