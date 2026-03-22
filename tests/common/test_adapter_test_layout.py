"""Regression tests for adapter test ownership and workflow boundaries."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _package_test_files(adapter: str) -> list[str]:
    return sorted(path.name for path in (ROOT / "packages" / adapter / "tests").glob("test_*.py"))


def _root_test_files(adapter: str) -> list[str]:
    return sorted(path.name for path in (ROOT / "tests" / adapter).glob("test_*.py"))


def test_adapter_native_suites_live_package_local() -> None:
    assert len(_package_test_files("dbt")) >= 6
    assert len(_package_test_files("mage")) >= 9
    assert len(_package_test_files("kestra")) >= 7


def test_root_adapter_suites_are_contract_only() -> None:
    assert _root_test_files("dbt") == ["test_first_party_suite.py"]
    assert _root_test_files("mage") == []
    assert _root_test_files("kestra") == []
