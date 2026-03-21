"""Tests for the dbt first-party suite runner."""

from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "packages" / "dbt" / "scripts" / "run_first_party_suite.py"


spec = importlib.util.spec_from_file_location("dbt_first_party_suite", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
dbt_first_party_suite = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dbt_first_party_suite
spec.loader.exec_module(dbt_first_party_suite)


def test_build_commands_include_explicit_project_and_profiles_dirs() -> None:
    project_dir = Path("/tmp/truthound-dbt-project")
    profiles_dir = Path("/tmp/truthound-dbt-profiles")

    commands = dbt_first_party_suite.build_commands(
        "compile",
        "postgres",
        macro_smoke=False,
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        dbt_executable="dbt",
    )

    assert [command.name for command in commands] == ["deps", "compile"]
    assert commands[0].argv == (
        "dbt",
        "deps",
        "--target",
        "postgres",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(profiles_dir),
    )
    assert commands[1].argv == (
        "dbt",
        "parse",
        "--target",
        "postgres",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(profiles_dir),
    )


def test_execute_macro_smoke_targets_reference_existing_columns() -> None:
    project_dir = Path("/tmp/truthound-dbt-project")
    profiles_dir = Path("/tmp/truthound-dbt-profiles")

    commands = dbt_first_party_suite.build_commands(
        "execute",
        "postgres",
        macro_smoke=True,
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        dbt_executable="dbt",
    )

    assert [command.name for command in commands] == [
        "deps",
        "seed",
        "run",
        "test",
        "run_truthound_check",
        "run_truthound_summary",
    ]

    check_args = json.loads(commands[-2].argv[-1])
    summary_args = json.loads(commands[-1].argv[-1])

    assert check_args == {
        "model_name": "test_model_valid",
        "rules": [{"column": "id", "check": "not_null"}],
        "options": {"limit": 50},
    }
    assert summary_args == {
        "model_name": "test_reference_model",
        "rules": [{"column": "name", "check": "not_null"}],
        "options": {"limit": 50},
    }


def test_validate_dbt_configuration_passes_for_checked_in_project() -> None:
    project_dir = ROOT / "packages" / "dbt" / "integration_tests"

    profile_name = dbt_first_party_suite.validate_dbt_configuration(
        project_dir,
        project_dir,
        "postgres",
    )

    assert profile_name == "integration_tests"


def test_validate_dbt_configuration_fails_on_profile_mismatch(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    profiles_dir = tmp_path / "profiles"
    project_dir.mkdir()
    profiles_dir.mkdir()
    (project_dir / "dbt_project.yml").write_text(
        "name: broken\nprofile: missing_profile\n",
        encoding="utf-8",
    )
    (profiles_dir / "profiles.yml").write_text(
        "integration_tests:\n  target: postgres\n  outputs:\n    postgres:\n      type: postgres\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing_profile"):
        dbt_first_party_suite.validate_dbt_configuration(
            project_dir,
            profiles_dir,
            "postgres",
        )


def test_generic_test_definitions_live_in_macro_path() -> None:
    generic_tests = ROOT / "packages" / "dbt" / "macros" / "truthound_generic_tests.sql"
    deprecated_location = (
        ROOT / "packages" / "dbt" / "tests" / "generic" / "test_truthound_check.sql"
    )

    text = generic_tests.read_text(encoding="utf-8")

    assert "{% test truthound_check(" in text
    assert "{% test truthound_not_null(" in text
    assert not deprecated_location.exists()


def test_integration_schema_uses_package_qualified_generic_tests() -> None:
    schema_path = ROOT / "packages" / "dbt" / "integration_tests" / "tests" / "schema.yml"
    schema_text = schema_path.read_text(encoding="utf-8")

    assert "truthound.truthound_check" in schema_text
    assert "truthound.truthound_not_null" in schema_text
    assert "truthound.truthound_range" in schema_text


def test_valid_seed_urls_match_project_url_pattern() -> None:
    project_text = (ROOT / "packages" / "dbt" / "dbt_project.yml").read_text(encoding="utf-8")
    match = re.search(r"^\s+url:\s+'([^']+)'", project_text, re.MULTILINE)
    assert match is not None
    url_pattern = re.compile(match.group(1))

    seed_path = ROOT / "packages" / "dbt" / "integration_tests" / "seeds" / "test_valid_data.csv"
    with seed_path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        invalid_urls = [row["url"] for row in rows if not url_pattern.match(row["url"])]

    assert invalid_urls == []
