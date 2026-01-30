"""Tests for TruthoundEngine anomaly detection (TASK 1-2)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from common.base import AnomalyResult, AnomalyScore, AnomalyStatus
from common.engines.truthound import TruthoundEngine, TruthoundEngineConfig
from common.exceptions import ValidationExecutionError


# =============================================================================
# Fixtures
# =============================================================================


class FakeAnomalyReport:
    """Fake Truthound anomaly report for testing."""

    def __init__(
        self,
        columns: list[dict[str, Any]],
        anomalous_row_count: int = 0,
        total_row_count: int = 100,
    ) -> None:
        self._columns = columns
        self._anomalous = anomalous_row_count
        self._total = total_row_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self._columns,
            "anomalous_row_count": self._anomalous,
            "total_row_count": self._total,
        }


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
# AnomalyResult Conversion Tests
# =============================================================================


class TestConvertAnomalyResult:
    """Tests for _convert_anomaly_result."""

    def test_no_anomalies(self, engine: TruthoundEngine) -> None:
        report = FakeAnomalyReport(
            columns=[
                {
                    "column": "value",
                    "score": 0.1,
                    "threshold": 0.5,
                    "is_anomaly": False,
                },
            ],
            anomalous_row_count=0,
            total_row_count=100,
        )
        import time

        result = engine._convert_anomaly_result(
            report, time.perf_counter(), "isolation_forest", range(100)
        )

        assert result.status == AnomalyStatus.NORMAL
        assert not result.has_anomalies
        assert result.anomaly_rate == 0.0
        assert len(result.anomalies) == 1
        assert result.anomalies[0].detector == "isolation_forest"

    def test_anomalies_detected(self, engine: TruthoundEngine) -> None:
        report = FakeAnomalyReport(
            columns=[
                {
                    "column": "value",
                    "score": 0.9,
                    "threshold": 0.5,
                    "is_anomaly": True,
                },
            ],
            anomalous_row_count=5,
            total_row_count=100,
        )
        import time

        result = engine._convert_anomaly_result(
            report, time.perf_counter(), "z_score", range(100)
        )

        assert result.status == AnomalyStatus.ANOMALY_DETECTED
        assert result.has_anomalies
        assert result.anomaly_rate == 5.0
        assert result.detector == "z_score"


# =============================================================================
# detect_anomalies Integration Tests
# =============================================================================


class TestDetectAnomalies:
    """Tests for detect_anomalies method."""

    def _setup_detector(self, engine: TruthoundEngine) -> MagicMock:
        detector_cls = MagicMock()
        detector_instance = MagicMock()
        detector_cls.return_value = detector_instance
        detector_instance.predict.return_value = FakeAnomalyReport(
            columns=[
                {
                    "column": "x",
                    "score": 0.8,
                    "threshold": 0.5,
                    "is_anomaly": True,
                },
            ],
            anomalous_row_count=3,
            total_row_count=50,
        )
        engine._truthound.ml = MagicMock()
        engine._truthound.ml.IsolationForestDetector = detector_cls
        engine._truthound.ml.ZScoreDetector = MagicMock()
        engine._truthound.ml.LocalOutlierFactor = MagicMock()
        engine._truthound.ml.EnsembleDetector = MagicMock()
        return detector_instance

    def test_detect_anomalies_basic(self, engine: TruthoundEngine) -> None:
        det = self._setup_detector(engine)

        result = engine.detect_anomalies("data")

        assert isinstance(result, AnomalyResult)
        assert result.status == AnomalyStatus.ANOMALY_DETECTED
        det.fit.assert_called_once()
        det.predict.assert_called_once()

    def test_detect_anomalies_uses_config_defaults(self, engine: TruthoundEngine) -> None:
        engine._config = TruthoundEngineConfig(
            default_anomaly_detector="z_score",
            default_contamination=0.1,
        )
        det = self._setup_detector(engine)
        engine._truthound.ml.ZScoreDetector = MagicMock(return_value=det)

        engine.detect_anomalies("data")

        engine._truthound.ml.ZScoreDetector.assert_called_once_with(contamination=0.1)

    def test_detect_anomalies_with_columns(self, engine: TruthoundEngine) -> None:
        det = self._setup_detector(engine)

        engine.detect_anomalies("data", columns=["a", "b"])

        det.fit.assert_called_once_with("data", columns=["a", "b"])

    def test_detect_anomalies_unsupported_detector(self, engine: TruthoundEngine) -> None:
        with pytest.raises(ValidationExecutionError, match="Unsupported anomaly detector"):
            engine.detect_anomalies("data", detector="nonexistent")

    def test_detect_anomalies_no_ml_module(self, engine: TruthoundEngine) -> None:
        engine._truthound.ml = None
        delattr(engine._truthound, "ml")

        with pytest.raises(ValidationExecutionError, match="ml module not available"):
            engine.detect_anomalies("data")


# =============================================================================
# Serialization Tests
# =============================================================================


class TestAnomalyResultSerialization:
    """Tests for AnomalyResult to_dict/from_dict roundtrip."""

    def test_roundtrip(self, engine: TruthoundEngine) -> None:
        report = FakeAnomalyReport(
            columns=[
                {
                    "column": "x",
                    "score": 0.9,
                    "threshold": 0.5,
                    "is_anomaly": True,
                },
            ],
            anomalous_row_count=10,
            total_row_count=200,
        )
        import time

        result = engine._convert_anomaly_result(
            report, time.perf_counter(), "lof", range(200)
        )
        d = result.to_dict()
        restored = AnomalyResult.from_dict(d)

        assert restored.status == result.status
        assert restored.anomalous_row_count == result.anomalous_row_count
        assert restored.total_row_count == result.total_row_count
        assert len(restored.anomalies) == 1


# =============================================================================
# Config Tests
# =============================================================================


class TestTruthoundEngineConfigAnomaly:
    """Tests for anomaly-related config fields."""

    def test_defaults(self) -> None:
        config = TruthoundEngineConfig()
        assert config.default_anomaly_detector == "isolation_forest"
        assert config.default_contamination == 0.05

    def test_with_anomaly_defaults(self) -> None:
        config = TruthoundEngineConfig().with_anomaly_defaults("z_score", 0.1)
        assert config.default_anomaly_detector == "z_score"
        assert config.default_contamination == 0.1

    def test_invalid_detector(self) -> None:
        with pytest.raises(ValueError, match="default_anomaly_detector"):
            TruthoundEngineConfig(default_anomaly_detector="invalid")

    def test_invalid_contamination_low(self) -> None:
        with pytest.raises(ValueError, match="default_contamination"):
            TruthoundEngineConfig(default_contamination=0.0)

    def test_invalid_contamination_high(self) -> None:
        with pytest.raises(ValueError, match="default_contamination"):
            TruthoundEngineConfig(default_contamination=0.5)
