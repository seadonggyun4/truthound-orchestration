"""Tests for extension Protocols (TASK 0-2)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Sequence

import pytest

from common.base import (
    AnomalyResult,
    AnomalyStatus,
    CheckResult,
    CheckStatus,
    DriftMethod,
    DriftResult,
    DriftStatus,
)
from common.engines.base import (
    AnomalyDetectionEngine,
    AsyncStreamingEngine,
    DataQualityEngine,
    DriftDetectionEngine,
    EngineCapabilities,
    StreamingEngine,
    supports_anomaly,
    supports_drift,
    supports_streaming,
)


# =============================================================================
# Stub Engines for Protocol Testing
# =============================================================================


class BasicEngine:
    """Engine that only implements DataQualityEngine."""

    @property
    def engine_name(self) -> str:
        return "basic"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def check(self, data: Any, rules: Sequence[Any], **kwargs: Any) -> CheckResult:
        return CheckResult(status=CheckStatus.PASSED)

    def profile(self, data: Any, **kwargs: Any) -> Any:
        return None

    def learn(self, data: Any, **kwargs: Any) -> Any:
        return None


class FullEngine(BasicEngine):
    """Engine that implements all extension protocols."""

    def detect_drift(
        self,
        baseline: Any,
        current: Any,
        *,
        method: str = "auto",
        columns: Sequence[str] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> DriftResult:
        return DriftResult(status=DriftStatus.NO_DRIFT, total_columns=1)

    def detect_anomalies(
        self,
        data: Any,
        *,
        detector: str = "isolation_forest",
        columns: Sequence[str] | None = None,
        contamination: float = 0.05,
        **kwargs: Any,
    ) -> AnomalyResult:
        return AnomalyResult(status=AnomalyStatus.NORMAL, total_row_count=1)

    def check_stream(
        self,
        stream: Any,
        *,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> Iterator[CheckResult]:
        yield CheckResult(status=CheckStatus.PASSED)


# =============================================================================
# Protocol isinstance Tests
# =============================================================================


class TestProtocolIsinstance:
    def test_basic_engine_is_data_quality_engine(self) -> None:
        assert isinstance(BasicEngine(), DataQualityEngine)

    def test_basic_engine_not_drift(self) -> None:
        assert not isinstance(BasicEngine(), DriftDetectionEngine)

    def test_basic_engine_not_anomaly(self) -> None:
        assert not isinstance(BasicEngine(), AnomalyDetectionEngine)

    def test_basic_engine_not_streaming(self) -> None:
        assert not isinstance(BasicEngine(), StreamingEngine)

    def test_full_engine_is_drift(self) -> None:
        assert isinstance(FullEngine(), DriftDetectionEngine)

    def test_full_engine_is_anomaly(self) -> None:
        assert isinstance(FullEngine(), AnomalyDetectionEngine)

    def test_full_engine_is_streaming(self) -> None:
        assert isinstance(FullEngine(), StreamingEngine)

    def test_full_engine_is_data_quality(self) -> None:
        assert isinstance(FullEngine(), DataQualityEngine)


# =============================================================================
# Feature Detection Utility Tests
# =============================================================================


class TestFeatureDetection:
    def test_supports_drift_basic(self) -> None:
        assert supports_drift(BasicEngine()) is False

    def test_supports_drift_full(self) -> None:
        assert supports_drift(FullEngine()) is True

    def test_supports_anomaly_basic(self) -> None:
        assert supports_anomaly(BasicEngine()) is False

    def test_supports_anomaly_full(self) -> None:
        assert supports_anomaly(FullEngine()) is True

    def test_supports_streaming_basic(self) -> None:
        assert supports_streaming(BasicEngine()) is False

    def test_supports_streaming_full(self) -> None:
        assert supports_streaming(FullEngine()) is True


# =============================================================================
# EngineCapabilities Extension Tests
# =============================================================================


class TestEngineCapabilitiesExtensions:
    def test_default_drift_false(self) -> None:
        caps = EngineCapabilities()
        assert caps.supports_drift is False
        assert caps.supports_anomaly is False
        assert caps.supported_drift_methods == ()
        assert caps.supported_anomaly_detectors == ()

    def test_drift_enabled(self) -> None:
        caps = EngineCapabilities(
            supports_drift=True,
            supported_drift_methods=("ks", "psi", "auto"),
        )
        assert caps.supports_drift is True
        assert "ks" in caps.supported_drift_methods

    def test_anomaly_enabled(self) -> None:
        caps = EngineCapabilities(
            supports_anomaly=True,
            supported_anomaly_detectors=("isolation_forest", "z_score"),
        )
        assert caps.supports_anomaly is True
        assert "isolation_forest" in caps.supported_anomaly_detectors


# =============================================================================
# Protocol Method Invocation Tests
# =============================================================================


class TestProtocolInvocation:
    def test_detect_drift_returns_result(self) -> None:
        engine = FullEngine()
        result = engine.detect_drift(None, None)
        assert isinstance(result, DriftResult)
        assert result.status == DriftStatus.NO_DRIFT

    def test_detect_anomalies_returns_result(self) -> None:
        engine = FullEngine()
        result = engine.detect_anomalies(None)
        assert isinstance(result, AnomalyResult)
        assert result.status == AnomalyStatus.NORMAL

    def test_check_stream_yields_results(self) -> None:
        engine = FullEngine()
        results = list(engine.check_stream(iter([])))
        assert len(results) == 1
        assert results[0].status == CheckStatus.PASSED
