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
                "version": "3.3.2",
                "python_version": "3.12",
                "constraints": data["security"]["surfaces"]["airflow"]["constraints"],
                "constraint_urls": [],
            }
        ]
    }


def test_prefect_main_matrix_is_primary_only_for_slim_main() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "prefect", "main")

    assert payload["host_matrix"] == {
        "include": [
            {
                "label": "primary",
                "version": "3.6.29",
                "python_version": "3.12",
                "constraints": ["fastapi<0.130.0"],
                "constraint_urls": [],
            },
        ]
    }


def test_dagster_release_min_matrix_applies_legacy_pendulum_constraint() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "dagster", "release")

    assert payload["host_matrix"]["include"][0] == {
        "label": "min",
        "version": "1.5.0",
        "python_version": "3.11",
        "constraints": ["pendulum<3.0.0"],
        "constraint_urls": [],
    }


def test_dbt_main_matrix_is_compile_smoke_only() -> None:
    data = ci_support_matrix.load_support_matrix()

    payload = ci_support_matrix.build_workflow_payload(data, "dbt", "main")

    assert payload["run_execution"] is False
    assert payload["execution_target"] == "postgres"
    targets = [entry["target"] for entry in payload["compile_matrix"]["include"]]
    assert targets == ["postgres"]


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
    assert payload["advisory_matrix"]["include"] == []
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
        "truthound==3.3.0",
        "apache-airflow==3.3.2",
    ]
    assert blocking["airflow"]["constraints"] == [
        "cryptography>=46.0.5",
        "certifi>=2026.10.19",
        "dnspython>=2.9.0",
        "opentelemetry-api>=1.45.1",
        "pyjwt>=2.15.0",
        "urllib3>=2.8.0",
    ]
    assert blocking["airflow"]["ignore_vulns"] == []
    assert blocking["airflow"]["constraint_urls"] == []
    assert blocking["prefect"]["host_requirements"] == [
        "truthound==3.0.0",
        "prefect==3.6.29",
        "starlette==1.3.1",
    ]
    assert blocking["prefect"]["constraints"] == [
        "cryptography>=46.0.5",
        "protobuf>=5.29.6",
        "pyjwt>=2.12.0",
        "starlette==1.3.1",
    ]
    assert blocking["prefect"]["ignore_vulns"] == []
    assert blocking["dbt"]["host_requirements"] == [
        "truthound==3.3.0",
        "dbt-core==1.11.16",
        "dbt-postgres==1.10.0",
    ]
    assert blocking["dbt"]["ignore_vulns"] == []
    assert blocking["kestra"]["host_requirements"] == [
        "truthound==3.0.0",
        "kestra==1.3.0",
    ]
    assert blocking["kestra"]["ignore_vulns"] == []
    assert blocking["opentelemetry"]["host_requirements"] == [
        "truthound==3.3.0",
        "opentelemetry-api==1.45.1",
        "opentelemetry-sdk==1.45.1",
        "opentelemetry-exporter-otlp-proto-grpc==1.45.1",
        "opentelemetry-exporter-otlp-proto-http==1.45.1",
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


def test_minimum_airflow_constraints_match_the_security_baseline() -> None:
    data = ci_support_matrix.load_support_matrix()
    lane = ci_support_matrix.build_workflow_payload(data, "airflow", "release")["host_matrix"][
        "include"
    ][0]
    assert not lane.get("test_constraints")
    assert lane["version"] == "3.3.2"
    assert lane["python_version"] == "3.11"
    assert lane["constraint_urls"] == []
    assert lane["constraints"] == data["security"]["surfaces"]["airflow"]["constraints"]


def test_prefect_minimum_includes_server_dependencies() -> None:
    data = ci_support_matrix.load_support_matrix()
    lane = ci_support_matrix.build_workflow_payload(data, "prefect", "release")["host_matrix"][
        "include"
    ][0]
    assert "fastapi<0.100.0" in lane["constraints"]
    assert "cachetools>=5.0.0" in lane["constraints"]


def test_dbt_core_minimum_allows_security_patched_sqlparse() -> None:
    data = ci_support_matrix.load_support_matrix()
    assert data["platforms"]["dbt"]["core_range"] == "dbt-core>=1.11.14,<1.12.0"
    payload = ci_support_matrix.build_workflow_payload(data, "dbt", "release")
    assert payload["execution_matrix"] == {
        "include": [{"core_version": "1.11.14"}, {"core_version": "1.11.16"}]
    }


def test_published_extras_reject_vulnerable_dependency_versions() -> None:
    """A clean audit must also apply to installations through package extras."""
    import tomllib

    from packaging.requirements import Requirement

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    floors = [
        ("dbt-core", "1.10.20", "1.11.14", ("dbt", "all")),
        ("sqlparse", "0.5.5", "0.6.0", ("dbt", "all")),
        ("opentelemetry-api", "1.45.0", "1.45.1", ("dbt", "opentelemetry", "all")),
    ]
    for name, vulnerable, fixed, surfaces in floors:
        for surface in surfaces:
            requirements = {req.name: req for req in map(Requirement, extras[surface])}
            assert vulnerable not in requirements[name].specifier, (surface, name)
            assert fixed in requirements[name].specifier, (surface, name)


def test_dbt_sql_package_enforces_security_supported_core() -> None:
    import yaml
    from packaging.specifiers import SpecifierSet

    project = yaml.safe_load((ROOT / "packages/dbt/dbt_project.yml").read_text())
    supported = SpecifierSet(",".join(project["require-dbt-version"]))
    assert "1.10.20" not in supported
    assert "1.11.13" not in supported
    assert "1.11.14" in supported
    assert "1.11.16" in supported


def test_redshift_compile_lane_allows_the_fixed_sql_parser() -> None:
    from packaging.requirements import Requirement

    data = ci_support_matrix.load_support_matrix()
    payload = ci_support_matrix.build_workflow_payload(data, "dbt", "release")
    lane = next(row for row in payload["compile_matrix"]["include"] if row["target"] == "redshift")
    requirement = Requirement(lane["adapter"])
    assert "1.10.2" not in requirement.specifier
    assert "1.11.0" not in requirement.specifier
    assert "1.11.1" in requirement.specifier


def test_airflow_security_and_primary_host_stay_aligned() -> None:
    """Do not silently audit a different host than the default runtime lane."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    data = ci_support_matrix.load_support_matrix()
    primary = data["platforms"]["airflow"]["primary"]
    surface = data["security"]["surfaces"]["airflow"]
    requirements = {req.name: req for req in map(Requirement, surface["host_requirements"])}
    assert Version(primary) >= Version("3.3.2")
    assert str(requirements["apache-airflow"].specifier) == f"=={primary}"
    assert str(requirements["truthound"].specifier) == "==3.3.0"
    assert "airflow" in data["security"]["blocking_surfaces"]
    assert surface["ignore_vulns"] == []


def test_airflow_package_surfaces_reject_hosts_below_security_floor() -> None:
    import tomllib

    from packaging.requirements import Requirement

    root_metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    provider = tomllib.loads((ROOT / "packages/airflow/pyproject.toml").read_text())
    extras = root_metadata["project"]["optional-dependencies"]
    for dependencies in (extras["airflow"], extras["all"], provider["project"]["dependencies"]):
        requirement = next(
            req for req in map(Requirement, dependencies) if req.name == "apache-airflow"
        )
        for unsupported in ("2.6.0", "3.2.0", "3.3.0", "3.3.1"):
            assert unsupported not in requirement.specifier
        assert "3.3.2" in requirement.specifier


def test_airflow_release_tests_security_floor_on_both_python_versions() -> None:
    data = ci_support_matrix.load_support_matrix()
    lanes = ci_support_matrix.build_workflow_payload(data, "airflow", "release")["host_matrix"][
        "include"
    ]
    assert {(lane["version"], lane["python_version"]) for lane in lanes} == {
        ("3.3.2", "3.11"),
        ("3.3.2", "3.12"),
    }
    for lane in lanes:
        assert not lane.get("test_constraints")
        assert lane["constraint_urls"] == []
        assert lane["constraints"] == data["security"]["surfaces"]["airflow"]["constraints"]


def test_airflow_security_transitives_are_enforced_by_packages_and_ci() -> None:
    import tomllib

    from packaging.requirements import Requirement

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    provider = tomllib.loads((ROOT / "packages/airflow/pyproject.toml").read_text())
    extras = metadata["project"]["optional-dependencies"]
    floors = {
        "certifi": ("2026.7.22", "2026.10.19"),
        "dnspython": ("2.8.0", "2.9.0"),
        "opentelemetry-api": ("1.44.0", "1.45.1"),
        "pyjwt": ("2.14.0", "2.15.0"),
        "urllib3": ("2.7.0", "2.8.0"),
    }
    data = ci_support_matrix.load_support_matrix()
    surface = data["security"]["surfaces"]["airflow"]
    sources = [
        extras["airflow"],
        extras["all"],
        provider["project"]["dependencies"],
        surface["constraints"],
    ]
    for lane in data["platforms"]["airflow"]["ci"].values():
        assert lane["constraint_urls"] == []
        assert lane["constraints"] == surface["constraints"]
        sources.append(lane["constraints"])
    for dependencies in sources:
        requirements = {req.name.lower(): req for req in map(Requirement, dependencies)}
        for name, (vulnerable, fixed) in floors.items():
            assert vulnerable not in requirements[name].specifier
            assert fixed in requirements[name].specifier
