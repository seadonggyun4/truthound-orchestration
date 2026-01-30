"""Tests for TASK 2-1: Mock engine extensions (drift, anomaly, streaming)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from common.base import (
    AnomalyResult,
    AnomalyScore,
    AnomalyStatus,
    CheckResult,
    CheckStatus,
    ColumnDrift,
    DriftMethod,
    DriftResult,
    DriftStatus,
)
from common.engines.base import (
    AnomalyDetectionEngine,
    AsyncStreamingEngine,
    DriftDetectionEngine,
    StreamingEngine,
)
from common.testing import (
    AsyncMockAnomalyDetectionEngine,
    AsyncMockDriftDetectionEngine,
    AsyncMockFullEngine,
    AsyncMockStreamingEngine,
    MockAnomalyDetectionEngine,
    MockDriftDetectionEngine,
    MockFullEngine,
    MockStreamingEngine,
    assert_anomaly_result,
    assert_drift_result,
    create_async_mock_full_engine,
    create_mock_anomaly_result,
    create_mock_anomaly_score,
    create_mock_column_drift,
    create_mock_drift_result,
    create_mock_full_engine,
)


# =============================================================================
# MockDriftDetectionEngine
# =============================================================================


class TestMockDriftDetectionEngine:
    def test_protocol_conformance(self) -> None:
        engine = MockDriftDetectionEngine()
        assert isinstance(engine, DriftDetectionEngine)

    def test_no_drift_by_default(self) -> None:
        engine = MockDriftDetectionEngine()
        result = engine.detect_drift(baseline=[], current=[])
        assert result.status == DriftStatus.NO_DRIFT

    def test_drift_detected(self) -> None:
        engine = MockDriftDetectionEngine(should_detect_drift=True)
        result = engine.detect_drift(baseline=[], current=[])
        assert result.status == DriftStatus.DRIFT_DETECTED

    def test_custom_result(self) -> None:
        custom = create_mock_drift_result(is_drifted=True)
        engine = MockDriftDetectionEngine(drift_result=custom)
        result = engine.detect_drift(baseline=[], current=[])
        assert result is custom

    def test_raises_on_error(self) -> None:
        engine = MockDriftDetectionEngine()
        engine.configure(raise_error=ValueError("boom"))
        with pytest.raises(ValueError, match="boom"):
            engine.detect_drift(baseline=[], current=[])

    def test_call_count(self) -> None:
        engine = MockDriftDetectionEngine()
        engine.detect_drift(baseline="b", current="c", method="ks")
        assert engine.call_count == 1

    def test_get_calls(self) -> None:
        engine = MockDriftDetectionEngine()
        engine.detect_drift(baseline="b", current="c", method="psi", threshold=0.1)
        calls = engine.get_calls()
        assert len(calls) == 1
        baseline, current, kwargs = calls[0]
        assert baseline == "b"
        assert current == "c"
        assert kwargs["method"] == "psi"
        assert kwargs["threshold"] == 0.1


# =============================================================================
# MockAnomalyDetectionEngine
# =============================================================================


class TestMockAnomalyDetectionEngine:
    def test_protocol_conformance(self) -> None:
        engine = MockAnomalyDetectionEngine()
        assert isinstance(engine, AnomalyDetectionEngine)

    def test_normal_by_default(self) -> None:
        engine = MockAnomalyDetectionEngine()
        result = engine.detect_anomalies(data=[])
        assert result.status == AnomalyStatus.NORMAL

    def test_anomaly_detected(self) -> None:
        engine = MockAnomalyDetectionEngine(should_detect_anomaly=True)
        result = engine.detect_anomalies(data=[])
        assert result.status == AnomalyStatus.ANOMALY_DETECTED

    def test_raises_on_error(self) -> None:
        engine = MockAnomalyDetectionEngine()
        engine.configure(raise_error=RuntimeError("fail"))
        with pytest.raises(RuntimeError, match="fail"):
            engine.detect_anomalies(data=[])

    def test_call_count(self) -> None:
        engine = MockAnomalyDetectionEngine()
        engine.detect_anomalies(data="d", detector="zscore")
        assert engine.call_count == 1
        calls = engine.get_calls()
        assert calls[0][1]["detector"] == "zscore"


# =============================================================================
# MockStreamingEngine
# =============================================================================


class TestMockStreamingEngine:
    def test_protocol_conformance(self) -> None:
        engine = MockStreamingEngine()
        assert isinstance(engine, StreamingEngine)

    def test_yields_results(self) -> None:
        batch = [
            CheckResult(status=CheckStatus.PASSED),
            CheckResult(status=CheckStatus.FAILED),
        ]
        engine = MockStreamingEngine(batch_results=batch)
        results = list(engine.check_stream(stream=[]))
        assert len(results) == 2
        assert results[0].status == CheckStatus.PASSED
        assert results[1].status == CheckStatus.FAILED

    def test_empty_by_default(self) -> None:
        engine = MockStreamingEngine()
        results = list(engine.check_stream(stream=[]))
        assert len(results) == 0

    def test_raises_on_error(self) -> None:
        engine = MockStreamingEngine()
        engine.configure(raise_error=IOError("stream fail"))
        with pytest.raises(IOError, match="stream fail"):
            list(engine.check_stream(stream=[]))

    def test_call_count(self) -> None:
        engine = MockStreamingEngine()
        list(engine.check_stream(stream="s", batch_size=500))
        assert engine.call_count == 1


# =============================================================================
# MockFullEngine
# =============================================================================


class TestMockFullEngine:
    def test_has_all_protocols(self) -> None:
        engine = MockFullEngine()
        assert isinstance(engine, DriftDetectionEngine)
        assert isinstance(engine, AnomalyDetectionEngine)
        assert isinstance(engine, StreamingEngine)

    def test_check_works(self) -> None:
        engine = MockFullEngine()
        result = engine.check(data=[], rules=[])
        assert isinstance(result, CheckResult)

    def test_detect_drift(self) -> None:
        engine = MockFullEngine()
        engine.configure_drift(should_detect_drift=True)
        result = engine.detect_drift(baseline=[], current=[])
        assert result.status == DriftStatus.DRIFT_DETECTED

    def test_detect_anomalies(self) -> None:
        engine = MockFullEngine()
        engine.configure_anomaly(should_detect_anomaly=True)
        result = engine.detect_anomalies(data=[])
        assert result.status == AnomalyStatus.ANOMALY_DETECTED

    def test_check_stream(self) -> None:
        batch = [CheckResult(status=CheckStatus.PASSED)]
        engine = MockFullEngine()
        engine.configure_streaming(batch_results=batch)
        results = list(engine.check_stream(stream=[]))
        assert len(results) == 1

    def test_factory_function(self) -> None:
        engine = create_mock_full_engine(
            check_success=False,
            should_detect_drift=True,
            should_detect_anomaly=True,
        )
        assert engine.check(data=[], rules=[]).status == CheckStatus.FAILED
        assert engine.detect_drift([], []).status == DriftStatus.DRIFT_DETECTED
        assert engine.detect_anomalies([]).status == AnomalyStatus.ANOMALY_DETECTED

    def test_reset(self) -> None:
        engine = MockFullEngine()
        engine.detect_drift([], [])
        engine.detect_anomalies([])
        engine.reset()
        assert engine.drift_call_count == 0
        assert engine.anomaly_call_count == 0


# =============================================================================
# Async Variants
# =============================================================================


class TestAsyncMockDriftDetectionEngine:
    def test_detect_drift(self) -> None:
        engine = AsyncMockDriftDetectionEngine()
        result = asyncio.get_event_loop().run_until_complete(
            engine.detect_drift(baseline=[], current=[])
        )
        assert result.status == DriftStatus.NO_DRIFT

    def test_drift_detected(self) -> None:
        engine = AsyncMockDriftDetectionEngine(should_detect_drift=True)
        result = asyncio.get_event_loop().run_until_complete(
            engine.detect_drift(baseline=[], current=[])
        )
        assert result.status == DriftStatus.DRIFT_DETECTED


class TestAsyncMockAnomalyDetectionEngine:
    def test_detect_anomalies(self) -> None:
        engine = AsyncMockAnomalyDetectionEngine()
        result = asyncio.get_event_loop().run_until_complete(
            engine.detect_anomalies(data=[])
        )
        assert result.status == AnomalyStatus.NORMAL


class TestAsyncMockStreamingEngine:
    def test_check_stream(self) -> None:
        batch = [CheckResult(status=CheckStatus.PASSED), CheckResult(status=CheckStatus.PASSED)]
        engine = AsyncMockStreamingEngine(batch_results=batch)

        async def collect() -> list[CheckResult]:
            aiter = await engine.check_stream(stream=[])
            return [r async for r in aiter]

        results = asyncio.get_event_loop().run_until_complete(collect())
        assert len(results) == 2

    def test_protocol_conformance(self) -> None:
        engine = AsyncMockStreamingEngine()
        assert isinstance(engine, AsyncStreamingEngine)


class TestAsyncMockFullEngine:
    def test_all_async_methods(self) -> None:
        engine = AsyncMockFullEngine()
        engine.configure_drift(should_detect_drift=True)
        engine.configure_anomaly(should_detect_anomaly=True)
        batch = [CheckResult(status=CheckStatus.PASSED)]
        engine.configure_streaming(batch_results=batch)

        loop = asyncio.get_event_loop()

        check = loop.run_until_complete(engine.check(data=[], rules=[]))
        assert isinstance(check, CheckResult)

        drift = loop.run_until_complete(engine.detect_drift([], []))
        assert drift.status == DriftStatus.DRIFT_DETECTED

        anomaly = loop.run_until_complete(engine.detect_anomalies([]))
        assert anomaly.status == AnomalyStatus.ANOMALY_DETECTED

        async def collect() -> list[CheckResult]:
            aiter = await engine.check_stream(stream=[])
            return [r async for r in aiter]

        stream = loop.run_until_complete(collect())
        assert len(stream) == 1

    def test_factory(self) -> None:
        engine = create_async_mock_full_engine(should_detect_drift=True)
        result = asyncio.get_event_loop().run_until_complete(
            engine.detect_drift([], [])
        )
        assert result.status == DriftStatus.DRIFT_DETECTED


# =============================================================================
# Factory Functions
# =============================================================================


class TestFactoryFunctions:
    def test_create_mock_drift_result_no_drift(self) -> None:
        result = create_mock_drift_result(is_drifted=False)
        assert result.status == DriftStatus.NO_DRIFT

    def test_create_mock_drift_result_drifted(self) -> None:
        result = create_mock_drift_result(is_drifted=True)
        assert result.status == DriftStatus.DRIFT_DETECTED
        assert result.drifted_count >= 1

    def test_create_mock_drift_result_custom_columns(self) -> None:
        cols = [
            create_mock_column_drift(column="a", is_drifted=True),
            create_mock_column_drift(column="b", is_drifted=False),
        ]
        result = create_mock_drift_result(is_drifted=True, drifted_columns=cols)
        assert result.drifted_count == 1
        assert len(result.drifted_columns) == 2

    def test_create_mock_anomaly_result_normal(self) -> None:
        result = create_mock_anomaly_result(has_anomalies=False)
        assert result.status == AnomalyStatus.NORMAL

    def test_create_mock_anomaly_result_detected(self) -> None:
        result = create_mock_anomaly_result(has_anomalies=True)
        assert result.status == AnomalyStatus.ANOMALY_DETECTED
        assert len(result.anomalies) >= 1

    def test_create_mock_column_drift(self) -> None:
        cd = create_mock_column_drift(column="age", is_drifted=True)
        assert isinstance(cd, ColumnDrift)
        assert cd.column == "age"
        assert cd.is_drifted is True

    def test_create_mock_anomaly_score(self) -> None:
        score = create_mock_anomaly_score(column="price", is_anomaly=True)
        assert isinstance(score, AnomalyScore)
        assert score.column == "price"
        assert score.is_anomaly is True


# =============================================================================
# Assertion Helpers
# =============================================================================


class TestAssertionHelpers:
    def test_assert_drift_result_passes(self) -> None:
        result = create_mock_drift_result(is_drifted=True)
        assert_drift_result(result, expected_status=DriftStatus.DRIFT_DETECTED)

    def test_assert_drift_result_fails(self) -> None:
        result = create_mock_drift_result(is_drifted=False)
        with pytest.raises(AssertionError):
            assert_drift_result(result, expected_status=DriftStatus.DRIFT_DETECTED)

    def test_assert_anomaly_result_passes(self) -> None:
        result = create_mock_anomaly_result(has_anomalies=False)
        assert_anomaly_result(result, expected_status=AnomalyStatus.NORMAL)

    def test_assert_anomaly_result_fails(self) -> None:
        result = create_mock_anomaly_result(has_anomalies=False)
        with pytest.raises(AssertionError):
            assert_anomaly_result(result, expected_status=AnomalyStatus.ANOMALY_DETECTED)
