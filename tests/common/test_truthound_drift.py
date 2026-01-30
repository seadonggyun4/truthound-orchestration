"""Tests for TruthoundEngine drift detection (TASK 1-1)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.base import (
    ColumnDrift,
    DriftMethod,
    DriftResult,
    DriftStatus,
    Severity,
)
from common.engines.truthound import TruthoundEngine, TruthoundEngineConfig
from common.exceptions import ValidationExecutionError


# =============================================================================
# Fixtures
# =============================================================================


class FakeDriftReport:
    """Fake Truthound drift report for testing."""

    def __init__(self, columns: list[dict[str, Any]]) -> None:
        self._columns = columns

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self._columns}


@pytest.fixture
def engine() -> TruthoundEngine:
    """Create a TruthoundEngine with mocked truthound module."""
    e = TruthoundEngine.__new__(TruthoundEngine)
    e._config = TruthoundEngineConfig()
    e._truthound = MagicMock()
    e._truthound.__version__ = "1.0.0"
    e._version = "1.0.0"
    from common.engines.lifecycle import EngineStateTracker

    e._state_tracker = EngineStateTracker("truthound")
    e._lock = __import__("threading").RLock()
    e._schema_cache = {}
    return e


# =============================================================================
# DriftResult Conversion Tests
# =============================================================================


class TestConvertDriftResult:
    """Tests for _convert_drift_result."""

    def test_no_drift(self, engine: TruthoundEngine) -> None:
        report = FakeDriftReport(
            columns=[
                {
                    "column": "age",
                    "method": "ks",
                    "statistic": 0.05,
                    "p_value": 0.8,
                    "threshold": 0.1,
                    "is_drifted": False,
                    "severity": "info",
                },
            ]
        )
        import time

        result = engine._convert_drift_result(report, time.perf_counter(), "ks")

        assert result.status == DriftStatus.NO_DRIFT
        assert result.total_columns == 1
        assert result.drifted_count == 0
        assert not result.is_drifted
        assert result.drift_rate == 0.0
        assert len(result.drifted_columns) == 1
        assert result.drifted_columns[0].column == "age"
        assert result.drifted_columns[0].method == DriftMethod.KS
        assert not result.drifted_columns[0].is_drifted

    def test_drift_detected(self, engine: TruthoundEngine) -> None:
        report = FakeDriftReport(
            columns=[
                {
                    "column": "age",
                    "method": "psi",
                    "statistic": 0.25,
                    "p_value": 0.01,
                    "threshold": 0.1,
                    "is_drifted": True,
                    "severity": "high",
                },
                {
                    "column": "name",
                    "method": "psi",
                    "statistic": 0.02,
                    "p_value": 0.9,
                    "threshold": 0.1,
                    "is_drifted": False,
                    "severity": "info",
                },
            ]
        )
        import time

        result = engine._convert_drift_result(report, time.perf_counter(), "psi")

        assert result.status == DriftStatus.DRIFT_DETECTED
        assert result.total_columns == 2
        assert result.drifted_count == 1
        assert result.is_drifted
        assert result.drift_rate == 50.0
        assert result.method == DriftMethod.PSI

    def test_unknown_method_falls_back_to_auto(self, engine: TruthoundEngine) -> None:
        report = FakeDriftReport(columns=[])
        import time

        result = engine._convert_drift_result(report, time.perf_counter(), "unknown")

        assert result.method == DriftMethod.AUTO

    def test_severity_mapping(self, engine: TruthoundEngine) -> None:
        report = FakeDriftReport(
            columns=[
                {
                    "column": "c1",
                    "method": "ks",
                    "statistic": 0.9,
                    "p_value": 0.001,
                    "threshold": 0.05,
                    "is_drifted": True,
                    "severity": "critical",
                },
            ]
        )
        import time

        result = engine._convert_drift_result(report, time.perf_counter(), "ks")
        assert result.drifted_columns[0].severity == Severity.CRITICAL


# =============================================================================
# detect_drift Integration Tests
# =============================================================================


class TestDetectDrift:
    """Tests for detect_drift method."""

    def test_detect_drift_calls_compare(self, engine: TruthoundEngine) -> None:
        fake_report = FakeDriftReport(
            columns=[
                {
                    "column": "x",
                    "method": "ks",
                    "statistic": 0.5,
                    "p_value": 0.01,
                    "threshold": 0.05,
                    "is_drifted": True,
                    "severity": "high",
                },
            ]
        )
        engine._truthound.compare.return_value = fake_report

        result = engine.detect_drift("baseline", "current", method="ks")

        engine._truthound.compare.assert_called_once()
        call_kwargs = engine._truthound.compare.call_args
        assert call_kwargs[0] == ("baseline", "current")
        assert call_kwargs[1]["method"] == "ks"
        assert isinstance(result, DriftResult)
        assert result.status == DriftStatus.DRIFT_DETECTED

    def test_detect_drift_uses_config_defaults(self, engine: TruthoundEngine) -> None:
        engine._config = TruthoundEngineConfig(
            default_drift_method="psi", default_drift_threshold=0.2
        )
        fake_report = FakeDriftReport(columns=[])
        engine._truthound.compare.return_value = fake_report

        engine.detect_drift("b", "c")

        call_kwargs = engine._truthound.compare.call_args[1]
        assert call_kwargs["method"] == "psi"
        assert call_kwargs["threshold"] == 0.2

    def test_detect_drift_with_columns(self, engine: TruthoundEngine) -> None:
        fake_report = FakeDriftReport(columns=[])
        engine._truthound.compare.return_value = fake_report

        engine.detect_drift("b", "c", columns=["a", "b"])

        call_kwargs = engine._truthound.compare.call_args[1]
        assert call_kwargs["columns"] == ["a", "b"]

    def test_detect_drift_exception(self, engine: TruthoundEngine) -> None:
        engine._truthound.compare.side_effect = RuntimeError("fail")

        with pytest.raises(ValidationExecutionError, match="drift detection failed"):
            engine.detect_drift("b", "c")


# =============================================================================
# DriftResult Serialization Tests
# =============================================================================


class TestDriftResultSerialization:
    """Tests for DriftResult to_dict/from_dict roundtrip."""

    def test_roundtrip(self, engine: TruthoundEngine) -> None:
        report = FakeDriftReport(
            columns=[
                {
                    "column": "age",
                    "method": "ks",
                    "statistic": 0.15,
                    "p_value": 0.03,
                    "threshold": 0.05,
                    "is_drifted": True,
                    "severity": "high",
                    "baseline_stats": {"mean": 30.0},
                    "current_stats": {"mean": 45.0},
                },
            ]
        )
        import time

        result = engine._convert_drift_result(report, time.perf_counter(), "ks")
        d = result.to_dict()
        restored = DriftResult.from_dict(d)

        assert restored.status == result.status
        assert restored.drifted_count == result.drifted_count
        assert restored.total_columns == result.total_columns
        assert len(restored.drifted_columns) == 1
        assert restored.drifted_columns[0].column == "age"


# =============================================================================
# Config Tests
# =============================================================================


class TestTruthoundEngineConfigDrift:
    """Tests for drift-related config fields."""

    def test_default_drift_config(self) -> None:
        config = TruthoundEngineConfig()
        assert config.default_drift_method == "auto"
        assert config.default_drift_threshold is None

    def test_with_drift_defaults(self) -> None:
        config = TruthoundEngineConfig().with_drift_defaults("psi", 0.15)
        assert config.default_drift_method == "psi"
        assert config.default_drift_threshold == 0.15

    def test_invalid_drift_method(self) -> None:
        with pytest.raises(ValueError, match="default_drift_method"):
            TruthoundEngineConfig(default_drift_method="invalid")

    @pytest.mark.parametrize(
        "method",
        ["ks", "psi", "chi2", "kl", "js", "wasserstein", "auto"],
    )
    def test_all_valid_methods(self, method: str) -> None:
        config = TruthoundEngineConfig(default_drift_method=method)
        assert config.default_drift_method == method
