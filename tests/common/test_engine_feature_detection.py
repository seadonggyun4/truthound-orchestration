"""Tests for engine feature detection utilities (TASK 1-5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from common.engines.base import (
    AnomalyDetectionEngine,
    DriftDetectionEngine,
    StreamingEngine,
    supports_anomaly,
    supports_drift,
    supports_streaming,
)
from common.engines.truthound import TruthoundEngine, TruthoundEngineConfig


# =============================================================================
# Helper: create engine without starting truthound
# =============================================================================


def _make_truthound_engine() -> TruthoundEngine:
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
# TruthoundEngine Protocol checks
# =============================================================================


class TestTruthoundFeatureDetection:
    """TruthoundEngine implements all extension Protocols."""

    def test_supports_drift(self) -> None:
        engine = _make_truthound_engine()
        assert supports_drift(engine) is True
        assert isinstance(engine, DriftDetectionEngine)

    def test_supports_anomaly(self) -> None:
        engine = _make_truthound_engine()
        assert supports_anomaly(engine) is True
        assert isinstance(engine, AnomalyDetectionEngine)

    def test_supports_streaming(self) -> None:
        engine = _make_truthound_engine()
        assert supports_streaming(engine) is True
        assert isinstance(engine, StreamingEngine)


# =============================================================================
# GE/Pandera do NOT implement extension Protocols
# =============================================================================


class TestGEFeatureDetection:
    """GreatExpectationsAdapter does not support drift/anomaly/streaming."""

    def test_ge_no_drift(self) -> None:
        from common.engines.great_expectations import GreatExpectationsAdapter

        engine = GreatExpectationsAdapter.__new__(GreatExpectationsAdapter)
        assert supports_drift(engine) is False
        assert not isinstance(engine, DriftDetectionEngine)

    def test_ge_no_anomaly(self) -> None:
        from common.engines.great_expectations import GreatExpectationsAdapter

        engine = GreatExpectationsAdapter.__new__(GreatExpectationsAdapter)
        assert supports_anomaly(engine) is False

    def test_ge_no_streaming(self) -> None:
        from common.engines.great_expectations import GreatExpectationsAdapter

        engine = GreatExpectationsAdapter.__new__(GreatExpectationsAdapter)
        assert supports_streaming(engine) is False

    def test_ge_capabilities_explicit(self) -> None:
        from common.engines.great_expectations import GreatExpectationsAdapter

        engine = GreatExpectationsAdapter.__new__(GreatExpectationsAdapter)
        caps = engine._get_capabilities()
        assert caps.supports_drift is False
        assert caps.supports_anomaly is False


class TestPanderaFeatureDetection:
    """PanderaAdapter does not support drift/anomaly/streaming."""

    def test_pandera_no_drift(self) -> None:
        from common.engines.pandera import PanderaAdapter

        engine = PanderaAdapter.__new__(PanderaAdapter)
        assert supports_drift(engine) is False

    def test_pandera_no_anomaly(self) -> None:
        from common.engines.pandera import PanderaAdapter

        engine = PanderaAdapter.__new__(PanderaAdapter)
        assert supports_anomaly(engine) is False

    def test_pandera_no_streaming(self) -> None:
        from common.engines.pandera import PanderaAdapter

        engine = PanderaAdapter.__new__(PanderaAdapter)
        assert supports_streaming(engine) is False

    def test_pandera_capabilities_explicit(self) -> None:
        from common.engines.pandera import PanderaAdapter

        engine = PanderaAdapter.__new__(PanderaAdapter)
        caps = engine._get_capabilities()
        assert caps.supports_drift is False
        assert caps.supports_anomaly is False
