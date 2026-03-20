"""Tests for shared runtime contracts and preflight behavior."""

from __future__ import annotations

from dataclasses import dataclass

from common.engines import (
    EngineCreationRequest,
    build_compatibility_report,
    normalize_runtime_context,
    resolve_data_source,
    run_preflight,
)
from common.runtime import DataSourceKind


@dataclass
class _FakeEngine:
    engine_name: str = "truthound"
    engine_version: str = "3.1.0"


def test_resolve_data_source_local_path_is_zero_config() -> None:
    source = resolve_data_source(data_path="data/sample.csv")

    assert source is not None
    assert source.kind == DataSourceKind.LOCAL_PATH
    assert source.requires_connection is False
    assert source.format_hint == "csv"


def test_resolve_data_source_sql_requires_connection() -> None:
    source = resolve_data_source(sql="SELECT * FROM users")

    assert source is not None
    assert source.kind == DataSourceKind.SQL
    assert source.requires_connection is True


def test_build_compatibility_report_flags_missing_connection_for_sql() -> None:
    runtime_context = normalize_runtime_context(platform="airflow")
    source = resolve_data_source(sql="SELECT * FROM users")

    report = build_compatibility_report(
        _FakeEngine(),
        runtime_context=runtime_context,
        resolved_source=source,
    )

    assert report.compatible is False
    assert any(check.name == "source_resolution" for check in report.failures)


def test_run_preflight_includes_truthound_safe_auto_check(monkeypatch) -> None:
    monkeypatch.setattr("common.engines.resolver.create_engine", lambda *args, **kwargs: _FakeEngine())
    runtime_context = normalize_runtime_context(platform="prefect")

    report = run_preflight(
        EngineCreationRequest(engine_name="truthound", runtime_context=runtime_context),
        data_path="dataset.parquet",
    )

    assert report.compatible is True
    assert report.resolved_source is not None
    assert report.resolved_source.kind == DataSourceKind.LOCAL_PATH
    assert any(check.name == "zero_config" for check in report.compatibility.checks)
