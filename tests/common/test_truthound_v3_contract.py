"""Truthound 3.x contract and orchestration boundary tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from truthound.core.results import CheckResult as CoreCheckResult

from common.base import CheckResult, CheckStatus
from common.engines import TruthoundEngine, TruthoundEngineConfig, create_engine
from common.engines.truthound_mapping import map_validation_run_result
from common.engines.version import check_version_compatibility


truthound = pytest.importorskip("truthound")
TruthoundContext = pytest.importorskip("truthound.context").TruthoundContext
TruthoundContextConfig = pytest.importorskip("truthound.context").TruthoundContextConfig
ValidationRunResult = pytest.importorskip("truthound.core.results").ValidationRunResult


@pytest.mark.parametrize("round_trip", [False, True])
def test_summary_only_failure_survives_mapping(round_trip: bool) -> None:
    """Redacting evidence must not open a downstream quality gate."""
    run = ValidationRunResult(
        suite_name="redacted", source="synthetic", row_count=3, column_count=1,
        checks=(CoreCheckResult(name="required", success=False, issue_count=1),),
    )
    if round_trip:
        run = ValidationRunResult.from_dict(run.to_dict())
    assert not run.success
    result = map_validation_run_result(run, engine_name="truthound", execution_time_ms=0)
    assert result.status == CheckStatus.FAILED
    assert result.failed_count == 1
    assert result.passed_count == 0
    assert not result.is_success
    assert not CheckResult.from_dict(result.to_dict()).is_success


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, None],
            "email": ["a@test.com", "invalid", "c@test.com"],
        }
    )


def test_truthound_check_returns_validation_run_result(tmp_path: Path) -> None:
    context = TruthoundContext(
        root_dir=tmp_path / "truthound-context",
        config=TruthoundContextConfig(
            persist_runs=False,
            persist_docs=False,
            auto_create_workspace=False,
        ),
    )

    run_result = truthound.check(_sample_frame(), context=context, auto_schema=True)

    assert isinstance(run_result, ValidationRunResult)
    assert run_result.checks


def test_orchestration_maps_validation_run_result(tmp_path: Path) -> None:
    context = TruthoundContext(
        root_dir=tmp_path / "truthound-context",
        config=TruthoundContextConfig(
            persist_runs=False,
            persist_docs=False,
            auto_create_workspace=False,
        ),
    )

    engine = TruthoundEngine()
    result = engine.check(_sample_frame(), context=context, auto_schema=True)

    assert result.status.name == "FAILED"
    assert result.failed_count >= 1
    assert result.metadata["engine"] == "truthound"
    assert result.metadata["row_count"] == 3
    assert result.failures


def test_truthound_check_without_rules_uses_auto_schema_by_default(tmp_path: Path) -> None:
    context = TruthoundContext(
        root_dir=tmp_path / "truthound-context",
        config=TruthoundContextConfig(
            persist_runs=False,
            persist_docs=False,
            auto_create_workspace=False,
        ),
    )

    engine = TruthoundEngine()
    result = engine.check(_sample_frame(), context=context)

    assert result.metadata["engine"] == "truthound"
    assert result.metadata["row_count"] == 3


def test_truthound_engine_ignores_platform_only_kwargs(tmp_path: Path) -> None:
    context = TruthoundContext(
        root_dir=tmp_path / "truthound-context",
        config=TruthoundContextConfig(
            persist_runs=False,
            persist_docs=False,
            auto_create_workspace=False,
        ),
    )

    engine = TruthoundEngine()

    check_result = engine.check(_sample_frame(), context=context, timeout=30)
    profile_result = engine.profile(
        _sample_frame(),
        context=context,
        include_statistics=True,
        include_patterns=False,
        include_distributions=True,
        timeout=30,
    )
    learn_result = engine.learn(
        _sample_frame(),
        context=context,
        strictness="strict",
        timeout=30,
    )

    assert check_result.metadata["row_count"] == 3
    assert profile_result.row_count == 3
    assert learn_result.columns_analyzed >= 1


def test_default_ephemeral_mode_has_no_workspace_side_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    engine = TruthoundEngine()
    result = engine.check(_sample_frame(), auto_schema=True)

    assert result.metadata["engine"] == "truthound"
    assert not (tmp_path / ".truthound").exists()


def test_truthound_engine_version_is_3x_compatible() -> None:
    engine = TruthoundEngine()
    compatibility = check_version_compatibility(engine, ">=3.0.0,<4.0.0")

    assert compatibility.compatible is True


def test_create_engine_uses_shared_truthound_resolver() -> None:
    engine = create_engine(
        "truthound",
        parallel=True,
        max_workers=2,
        context_mode="project",
        auto_start=False,
        auto_stop=False,
    )

    assert isinstance(engine, TruthoundEngine)
    assert isinstance(engine.config, TruthoundEngineConfig)
    assert engine.config.parallel is True
    assert engine.config.max_workers == 2
    assert engine.config.context_mode == "project"
