"""Tests for TruthoundEngine capabilities update (TASK 1-4)."""

from __future__ import annotations

from common.base import DriftMethod
from common.engines.truthound import TruthoundEngine, TruthoundEngineConfig


class TestTruthoundCapabilities:
    """Tests for updated EngineCapabilities."""

    def test_supports_drift(self) -> None:
        engine = TruthoundEngine.__new__(TruthoundEngine)
        engine._config = TruthoundEngineConfig()
        caps = engine._get_capabilities()
        assert caps.supports_drift is True

    def test_supports_anomaly(self) -> None:
        engine = TruthoundEngine.__new__(TruthoundEngine)
        engine._config = TruthoundEngineConfig()
        caps = engine._get_capabilities()
        assert caps.supports_anomaly is True

    def test_supports_streaming(self) -> None:
        engine = TruthoundEngine.__new__(TruthoundEngine)
        engine._config = TruthoundEngineConfig()
        caps = engine._get_capabilities()
        assert caps.supports_streaming is True

    def test_drift_methods_listed(self) -> None:
        engine = TruthoundEngine.__new__(TruthoundEngine)
        engine._config = TruthoundEngineConfig()
        caps = engine._get_capabilities()

        expected_methods = {m.value for m in DriftMethod}
        actual_methods = set(caps.supported_drift_methods)
        assert actual_methods == expected_methods

    def test_anomaly_detectors_listed(self) -> None:
        engine = TruthoundEngine.__new__(TruthoundEngine)
        engine._config = TruthoundEngineConfig()
        caps = engine._get_capabilities()

        expected = {"isolation_forest", "z_score", "lof", "ensemble"}
        actual = set(caps.supported_anomaly_detectors)
        assert actual == expected

    def test_existing_capabilities_preserved(self) -> None:
        engine = TruthoundEngine.__new__(TruthoundEngine)
        engine._config = TruthoundEngineConfig()
        caps = engine._get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is True
        assert caps.supports_async is False
        assert "polars" in caps.supported_data_types
