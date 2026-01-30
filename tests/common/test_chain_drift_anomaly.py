"""Tests for EngineChain drift/anomaly support (TASK 4-1)."""

from __future__ import annotations

import pytest

from common.base import DriftStatus, AnomalyStatus
from common.engines.chain import (
    AllEnginesFailedError,
    EngineChain,
    ConditionalEngineChain,
    NoEngineSelectedError,
    SelectorEngineChain,
)
from common.testing import (
    MockDataQualityEngine,
    MockDriftDetectionEngine,
    MockAnomalyDetectionEngine,
    MockFullEngine,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def drift_engine() -> MockDriftDetectionEngine:
    return MockDriftDetectionEngine(should_detect_drift=False)


@pytest.fixture()
def anomaly_engine() -> MockAnomalyDetectionEngine:
    return MockAnomalyDetectionEngine(should_detect_anomaly=False)


@pytest.fixture()
def extended_engine() -> MockFullEngine:
    return MockFullEngine()


@pytest.fixture()
def plain_engine() -> MockDataQualityEngine:
    return MockDataQualityEngine()


# =============================================================================
# EngineChain.detect_drift Tests
# =============================================================================


class TestEngineChainDetectDrift:
    def test_detect_drift_with_compatible_engine(
        self, extended_engine: MockFullEngine
    ) -> None:
        chain = EngineChain([extended_engine])
        result = chain.detect_drift(baseline="b", current="c")
        assert result.status == DriftStatus.NO_DRIFT

    def test_detect_drift_filters_incompatible_engines(
        self, plain_engine: MockDataQualityEngine, extended_engine: MockFullEngine
    ) -> None:
        chain = EngineChain([plain_engine, extended_engine])
        result = chain.detect_drift(baseline="b", current="c")
        assert result.status == DriftStatus.NO_DRIFT

    def test_detect_drift_no_compatible_engine_raises(
        self, plain_engine: MockDataQualityEngine
    ) -> None:
        chain = EngineChain([plain_engine])
        with pytest.raises(NoEngineSelectedError):
            chain.detect_drift(baseline="b", current="c")

    def test_detect_drift_fallback(self, extended_engine: MockFullEngine) -> None:
        """If first drift engine fails, fallback to next."""
        failing = MockFullEngine(name="mock_full_failing")

        # Make first engine raise on detect_drift
        def raise_on_drift(*a: object, **kw: object) -> None:
            raise RuntimeError("Engine failure")

        failing.detect_drift = raise_on_drift  # type: ignore[assignment]

        chain = EngineChain([failing, extended_engine])
        result = chain.detect_drift(baseline="b", current="c")
        assert result.status == DriftStatus.NO_DRIFT


# =============================================================================
# EngineChain.detect_anomalies Tests
# =============================================================================


class TestEngineChainDetectAnomalies:
    def test_detect_anomalies_with_compatible_engine(
        self, extended_engine: MockFullEngine
    ) -> None:
        chain = EngineChain([extended_engine])
        result = chain.detect_anomalies(data="data")
        assert result.status == AnomalyStatus.NORMAL

    def test_detect_anomalies_filters_incompatible(
        self, plain_engine: MockDataQualityEngine, extended_engine: MockFullEngine
    ) -> None:
        chain = EngineChain([plain_engine, extended_engine])
        result = chain.detect_anomalies(data="data")
        assert result.status == AnomalyStatus.NORMAL

    def test_detect_anomalies_no_compatible_engine_raises(
        self, plain_engine: MockDataQualityEngine
    ) -> None:
        chain = EngineChain([plain_engine])
        with pytest.raises(NoEngineSelectedError):
            chain.detect_anomalies(data="data")


# =============================================================================
# ConditionalEngineChain Tests
# =============================================================================


class TestConditionalChainDriftAnomaly:
    def test_conditional_detect_drift(
        self, extended_engine: MockFullEngine
    ) -> None:
        chain = ConditionalEngineChain()
        chain.add_route(lambda data, rules: True, extended_engine)
        result = chain.detect_drift(baseline="b", current="c")
        assert result.status == DriftStatus.NO_DRIFT

    def test_conditional_detect_anomalies(
        self, extended_engine: MockFullEngine
    ) -> None:
        chain = ConditionalEngineChain()
        chain.add_route(lambda data, rules: True, extended_engine)
        result = chain.detect_anomalies(data="data")
        assert result.status == AnomalyStatus.NORMAL


# =============================================================================
# SelectorEngineChain Tests
# =============================================================================


class TestSelectorChainDriftAnomaly:
    def test_selector_detect_drift(
        self, extended_engine: MockFullEngine
    ) -> None:
        class AlwaysFirst:
            def select_engine(self, data, rules, engines, context):
                return engines[0] if engines else None

        chain = SelectorEngineChain(
            engines=[extended_engine],
            selector=AlwaysFirst(),
        )
        result = chain.detect_drift(baseline="b", current="c")
        assert result.status == DriftStatus.NO_DRIFT

    def test_selector_detect_anomalies(
        self, extended_engine: MockFullEngine
    ) -> None:
        class AlwaysFirst:
            def select_engine(self, data, rules, engines, context):
                return engines[0] if engines else None

        chain = SelectorEngineChain(
            engines=[extended_engine],
            selector=AlwaysFirst(),
        )
        result = chain.detect_anomalies(data="data")
        assert result.status == AnomalyStatus.NORMAL
