"""Tests for BatchExecutor drift/anomaly support (TASK 4-2)."""

from __future__ import annotations

import pytest

from common.base import AnomalyStatus, DriftStatus
from common.engines.batch import (
    AsyncBatchExecutor,
    BatchConfig,
    BatchExecutor,
    BatchOperationError,
)
from common.testing import (
    AsyncMockFullEngine,
    MockDataQualityEngine,
    MockFullEngine,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def extended_engine() -> MockFullEngine:
    return MockFullEngine()


@pytest.fixture()
def plain_engine() -> MockDataQualityEngine:
    return MockDataQualityEngine()


@pytest.fixture()
def batch_config() -> BatchConfig:
    return BatchConfig(batch_size=10, max_workers=2, fail_fast=False, continue_on_error=True)


# =============================================================================
# BatchExecutor.drift_batch Tests
# =============================================================================


class TestBatchExecutorDriftBatch:
    def test_drift_batch_basic(
        self, extended_engine: MockFullEngine, batch_config: BatchConfig
    ) -> None:
        executor = BatchExecutor(extended_engine, default_config=batch_config)
        result = executor.drift_batch(
            baseline_datasets=["b1", "b2"],
            current_datasets=["c1", "c2"],
        )
        assert result.total_batches == 2
        assert result.successful_batches == 2
        assert result.failed_batches == 0

    def test_drift_batch_unsupported_engine_raises(
        self, plain_engine: MockDataQualityEngine, batch_config: BatchConfig
    ) -> None:
        executor = BatchExecutor(plain_engine, default_config=batch_config)
        with pytest.raises(BatchOperationError, match="does not support drift"):
            executor.drift_batch(
                baseline_datasets=["b1"],
                current_datasets=["c1"],
            )

    def test_drift_batch_length_mismatch_raises(
        self, extended_engine: MockFullEngine, batch_config: BatchConfig
    ) -> None:
        executor = BatchExecutor(extended_engine, default_config=batch_config)
        with pytest.raises(ValueError, match="same length"):
            executor.drift_batch(
                baseline_datasets=["b1", "b2"],
                current_datasets=["c1"],
            )


# =============================================================================
# BatchExecutor.drift_chunked Tests
# =============================================================================


class TestBatchExecutorDriftChunked:
    def test_drift_chunked_unsupported_engine_raises(
        self, plain_engine: MockDataQualityEngine, batch_config: BatchConfig
    ) -> None:
        executor = BatchExecutor(plain_engine, default_config=batch_config)
        with pytest.raises(BatchOperationError, match="does not support drift"):
            executor.drift_chunked(baseline="b", current="c")


# =============================================================================
# BatchExecutor.anomaly_batch Tests
# =============================================================================


class TestBatchExecutorAnomalyBatch:
    def test_anomaly_batch_basic(
        self, extended_engine: MockFullEngine, batch_config: BatchConfig
    ) -> None:
        executor = BatchExecutor(extended_engine, default_config=batch_config)
        result = executor.anomaly_batch(datasets=["d1", "d2", "d3"])
        assert result.total_batches == 3
        assert result.successful_batches == 3

    def test_anomaly_batch_unsupported_engine_raises(
        self, plain_engine: MockDataQualityEngine, batch_config: BatchConfig
    ) -> None:
        executor = BatchExecutor(plain_engine, default_config=batch_config)
        with pytest.raises(BatchOperationError, match="does not support anomaly"):
            executor.anomaly_batch(datasets=["d1"])


# =============================================================================
# BatchExecutor.anomaly_chunked Tests
# =============================================================================


class TestBatchExecutorAnomalyChunked:
    def test_anomaly_chunked_unsupported_engine_raises(
        self, plain_engine: MockDataQualityEngine, batch_config: BatchConfig
    ) -> None:
        executor = BatchExecutor(plain_engine, default_config=batch_config)
        with pytest.raises(BatchOperationError, match="does not support anomaly"):
            executor.anomaly_chunked(data="data")


# =============================================================================
# AsyncBatchExecutor Drift/Anomaly Tests
# =============================================================================


class TestAsyncBatchExecutorDriftAnomaly:
    @pytest.mark.asyncio()
    async def test_drift_batch_unsupported_raises(self) -> None:
        from common.testing import AsyncMockDataQualityEngine

        engine = AsyncMockDataQualityEngine()
        executor = AsyncBatchExecutor(engine)
        with pytest.raises(BatchOperationError, match="does not support drift"):
            await executor.drift_batch(
                baseline_datasets=["b1"],
                current_datasets=["c1"],
            )

    @pytest.mark.asyncio()
    async def test_anomaly_batch_unsupported_raises(self) -> None:
        from common.testing import AsyncMockDataQualityEngine

        engine = AsyncMockDataQualityEngine()
        executor = AsyncBatchExecutor(engine)
        with pytest.raises(BatchOperationError, match="does not support anomaly"):
            await executor.anomaly_batch(datasets=["d1"])

    @pytest.mark.asyncio()
    async def test_drift_batch_with_extended_engine(self) -> None:
        engine = AsyncMockFullEngine()
        executor = AsyncBatchExecutor(engine)
        result = await executor.drift_batch(
            baseline_datasets=["b1", "b2"],
            current_datasets=["c1", "c2"],
        )
        assert result.total_batches == 2
        assert result.successful_batches == 2

    @pytest.mark.asyncio()
    async def test_anomaly_batch_with_extended_engine(self) -> None:
        engine = AsyncMockFullEngine()
        executor = AsyncBatchExecutor(engine)
        result = await executor.anomaly_batch(datasets=["d1", "d2"])
        assert result.total_batches == 2
        assert result.successful_batches == 2

    @pytest.mark.asyncio()
    async def test_drift_batch_length_mismatch_raises(self) -> None:
        engine = AsyncMockFullEngine()
        executor = AsyncBatchExecutor(engine)
        with pytest.raises(ValueError, match="same length"):
            await executor.drift_batch(
                baseline_datasets=["b1", "b2"],
                current_datasets=["c1"],
            )
