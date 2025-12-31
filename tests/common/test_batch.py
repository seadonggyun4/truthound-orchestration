"""Tests for Batch Operations module.

This module contains comprehensive tests for the batch processing system
including BatchExecutor, AsyncBatchExecutor, chunkers, aggregators, and hooks.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.base import (
    CheckResult,
    CheckStatus,
    LearnResult,
    LearnStatus,
    LearnedRule,
    ProfileResult,
    ProfileStatus,
    ValidationFailure,
    Severity,
    ColumnProfile,
)
from common.engines.batch import (
    # Config
    BatchConfig,
    DEFAULT_BATCH_CONFIG,
    PARALLEL_BATCH_CONFIG,
    SEQUENTIAL_BATCH_CONFIG,
    FAIL_FAST_BATCH_CONFIG,
    LARGE_DATA_BATCH_CONFIG,
    # Enums
    ExecutionStrategy,
    AggregationStrategy,
    ChunkingStrategy,
    # Results
    BatchResult,
    BatchItemResult,
    # Executors
    BatchExecutor,
    AsyncBatchExecutor,
    create_batch_executor,
    create_async_batch_executor,
    # Chunkers
    RowCountChunker,
    PolarsChunker,
    DatasetListChunker,
    # Aggregators
    CheckResultAggregator,
    ProfileResultAggregator,
    LearnResultAggregator,
    # Hooks
    BatchHook,
    LoggingBatchHook,
    MetricsBatchHook,
    CompositeBatchHook,
    # Exceptions
    BatchOperationError,
    BatchExecutionError,
    ChunkingError,
    AggregationError,
)
from common.testing import MockDataQualityEngine, AsyncMockDataQualityEngine


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_engine():
    """Create a mock data quality engine."""
    engine = MockDataQualityEngine()
    engine.configure_check(success=True)
    engine.configure_profile(success=True)
    engine.configure_learn(success=True)
    return engine


@pytest.fixture
def failing_mock_engine():
    """Create a mock engine that fails."""
    engine = MockDataQualityEngine()
    engine.configure_check(success=False, raise_error=ValueError("Mock failure"))
    engine.configure_profile(success=False, raise_error=ValueError("Mock failure"))
    engine.configure_learn(success=False, raise_error=ValueError("Mock failure"))
    return engine


@pytest.fixture
def async_mock_engine():
    """Create an async mock data quality engine."""
    engine = AsyncMockDataQualityEngine()
    engine.configure_check(success=True)
    engine.configure_profile(success=True)
    engine.configure_learn(success=True)
    return engine


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return list(range(100))


@pytest.fixture
def sample_datasets():
    """Create multiple sample datasets."""
    return [list(range(10)), list(range(20)), list(range(30))]


@pytest.fixture
def sample_check_results():
    """Create sample CheckResults for testing aggregation."""
    return [
        CheckResult(
            status=CheckStatus.PASSED,
            passed_count=5,
            failed_count=0,
            execution_time_ms=10.0,
        ),
        CheckResult(
            status=CheckStatus.FAILED,
            passed_count=3,
            failed_count=2,
            failures=(
                ValidationFailure(
                    rule_name="not_null",
                    column="id",
                    message="Found null values",
                    severity=Severity.ERROR,
                ),
            ),
            execution_time_ms=15.0,
        ),
        CheckResult(
            status=CheckStatus.WARNING,
            passed_count=4,
            failed_count=0,
            warning_count=1,
            execution_time_ms=12.0,
        ),
    ]


# =============================================================================
# BatchConfig Tests
# =============================================================================


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchConfig()
        assert config.batch_size == 10000
        assert config.max_workers == 4
        assert config.execution_strategy == ExecutionStrategy.ADAPTIVE
        assert config.aggregation_strategy == AggregationStrategy.MERGE
        assert config.chunking_strategy == ChunkingStrategy.ROW_COUNT
        assert config.fail_fast is False
        assert config.continue_on_error is True

    def test_with_batch_size(self):
        """Test with_batch_size builder method."""
        config = BatchConfig()
        new_config = config.with_batch_size(5000)
        assert new_config.batch_size == 5000
        assert config.batch_size == 10000  # Original unchanged

    def test_with_max_workers(self):
        """Test with_max_workers builder method."""
        config = BatchConfig()
        new_config = config.with_max_workers(8)
        assert new_config.max_workers == 8

    def test_with_execution_strategy(self):
        """Test with_execution_strategy builder method."""
        config = BatchConfig()
        new_config = config.with_execution_strategy(ExecutionStrategy.PARALLEL)
        assert new_config.execution_strategy == ExecutionStrategy.PARALLEL

    def test_with_aggregation_strategy(self):
        """Test with_aggregation_strategy builder method."""
        config = BatchConfig()
        new_config = config.with_aggregation_strategy(AggregationStrategy.WORST)
        assert new_config.aggregation_strategy == AggregationStrategy.WORST

    def test_with_fail_fast(self):
        """Test with_fail_fast builder method."""
        config = BatchConfig()
        new_config = config.with_fail_fast(True)
        assert new_config.fail_fast is True

    def test_with_timeouts(self):
        """Test with_timeouts builder method."""
        config = BatchConfig()
        new_config = config.with_timeouts(per_batch=30.0, total=300.0)
        assert new_config.timeout_per_batch_seconds == 30.0
        assert new_config.total_timeout_seconds == 300.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = BatchConfig(batch_size=5000, max_workers=8)
        data = config.to_dict()
        assert data["batch_size"] == 5000
        assert data["max_workers"] == 8
        assert data["execution_strategy"] == "ADAPTIVE"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "batch_size": 5000,
            "max_workers": 8,
            "execution_strategy": "PARALLEL",
        }
        config = BatchConfig.from_dict(data)
        assert config.batch_size == 5000
        assert config.max_workers == 8
        assert config.execution_strategy == ExecutionStrategy.PARALLEL

    def test_preset_configs(self):
        """Test preset configurations."""
        assert DEFAULT_BATCH_CONFIG.batch_size == 10000
        assert PARALLEL_BATCH_CONFIG.execution_strategy == ExecutionStrategy.PARALLEL
        assert SEQUENTIAL_BATCH_CONFIG.execution_strategy == ExecutionStrategy.SEQUENTIAL
        assert FAIL_FAST_BATCH_CONFIG.fail_fast is True
        assert LARGE_DATA_BATCH_CONFIG.batch_size == 50000


# =============================================================================
# BatchResult Tests
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult and BatchItemResult."""

    def test_batch_item_result_success(self):
        """Test successful BatchItemResult."""
        result = CheckResult(status=CheckStatus.PASSED, passed_count=5)
        item = BatchItemResult(index=0, result=result, execution_time_ms=10.0)

        assert item.is_success is True
        assert item.is_failure is False
        assert item.index == 0
        assert item.execution_time_ms == 10.0

    def test_batch_item_result_failure(self):
        """Test failed BatchItemResult."""
        error = ValueError("Test error")
        item = BatchItemResult(index=1, error=error, execution_time_ms=5.0)

        assert item.is_success is False
        assert item.is_failure is True
        assert item.error is error

    def test_batch_result_complete_success(self):
        """Test BatchResult with all successes."""
        items = tuple(
            BatchItemResult(
                index=i,
                result=CheckResult(status=CheckStatus.PASSED),
            )
            for i in range(3)
        )
        result = BatchResult(
            batch_results=items,
            total_batches=3,
            successful_batches=3,
            failed_batches=0,
        )

        assert result.is_complete_success is True
        assert result.is_partial_success is False
        assert result.is_complete_failure is False
        assert result.success_rate == 100.0

    def test_batch_result_partial_success(self):
        """Test BatchResult with mixed results."""
        items = (
            BatchItemResult(index=0, result=CheckResult(status=CheckStatus.PASSED)),
            BatchItemResult(index=1, error=ValueError("error")),
        )
        result = BatchResult(
            batch_results=items,
            total_batches=2,
            successful_batches=1,
            failed_batches=1,
        )

        assert result.is_complete_success is False
        assert result.is_partial_success is True
        assert result.success_rate == 50.0

    def test_batch_result_get_failures(self):
        """Test getting failed results."""
        items = (
            BatchItemResult(index=0, result=CheckResult(status=CheckStatus.PASSED)),
            BatchItemResult(index=1, error=ValueError("error")),
            BatchItemResult(index=2, error=ValueError("error2")),
        )
        result = BatchResult(batch_results=items, total_batches=3)

        failures = result.get_failures()
        assert len(failures) == 2
        assert all(f.is_failure for f in failures)

    def test_batch_result_to_dict(self):
        """Test serialization to dictionary."""
        result = BatchResult(
            total_batches=3,
            successful_batches=2,
            failed_batches=1,
            total_execution_time_ms=100.0,
        )
        data = result.to_dict()

        assert data["total_batches"] == 3
        assert data["successful_batches"] == 2
        assert data["is_complete_success"] is False
        assert data["success_rate"] == pytest.approx(66.67, rel=0.01)


# =============================================================================
# Chunker Tests
# =============================================================================


class TestChunkers:
    """Tests for data chunkers."""

    def test_row_count_chunker(self):
        """Test RowCountChunker."""
        chunker = RowCountChunker()
        data = list(range(100))
        config = BatchConfig(batch_size=30)

        chunks = list(chunker.chunk(data, config))
        assert len(chunks) == 4  # 30 + 30 + 30 + 10
        assert len(chunks[0]) == 30
        assert len(chunks[-1]) == 10

    def test_row_count_chunker_estimate(self):
        """Test chunk estimation."""
        chunker = RowCountChunker()
        data = list(range(100))
        config = BatchConfig(batch_size=30)

        estimate = chunker.estimate_chunks(data, config)
        assert estimate == 4

    def test_dataset_list_chunker(self):
        """Test DatasetListChunker."""
        chunker = DatasetListChunker()
        datasets = [[1, 2], [3, 4, 5], [6]]
        config = BatchConfig()

        chunks = list(chunker.chunk(datasets, config))
        assert len(chunks) == 3
        assert chunks[0] == [1, 2]
        assert chunks[1] == [3, 4, 5]

    def test_polars_chunker_fallback(self):
        """Test PolarsChunker fallback for non-Polars data."""
        chunker = PolarsChunker()
        data = list(range(50))
        config = BatchConfig(batch_size=20)

        chunks = list(chunker.chunk(data, config))
        assert len(chunks) == 3

    @pytest.mark.skipif(
        True,  # Skip if polars not installed
        reason="Polars not installed"
    )
    def test_polars_chunker_with_dataframe(self):
        """Test PolarsChunker with actual Polars DataFrame."""
        import polars as pl

        chunker = PolarsChunker()
        df = pl.DataFrame({"a": range(100), "b": range(100)})
        config = BatchConfig(batch_size=30)

        chunks = list(chunker.chunk(df, config))
        assert len(chunks) == 4
        assert chunks[0].height == 30


# =============================================================================
# Aggregator Tests
# =============================================================================


class TestCheckResultAggregator:
    """Tests for CheckResultAggregator."""

    def test_merge_aggregation(self, sample_check_results):
        """Test MERGE aggregation strategy."""
        aggregator = CheckResultAggregator()
        config = BatchConfig(aggregation_strategy=AggregationStrategy.MERGE)

        result = aggregator.aggregate(sample_check_results, config)

        assert result.status == CheckStatus.FAILED  # Worst status
        assert result.passed_count == 12  # 5 + 3 + 4
        assert result.failed_count == 2
        assert result.warning_count == 1
        assert len(result.failures) == 1
        assert result.execution_time_ms == 37.0  # 10 + 15 + 12

    def test_worst_aggregation(self, sample_check_results):
        """Test WORST aggregation strategy."""
        aggregator = CheckResultAggregator()
        config = BatchConfig(aggregation_strategy=AggregationStrategy.WORST)

        result = aggregator.aggregate(sample_check_results, config)
        assert result.status == CheckStatus.FAILED

    def test_best_aggregation(self, sample_check_results):
        """Test BEST aggregation strategy."""
        aggregator = CheckResultAggregator()
        config = BatchConfig(aggregation_strategy=AggregationStrategy.BEST)

        result = aggregator.aggregate(sample_check_results, config)
        assert result.status == CheckStatus.PASSED

    def test_first_failure_aggregation(self, sample_check_results):
        """Test FIRST_FAILURE aggregation strategy."""
        aggregator = CheckResultAggregator()
        config = BatchConfig(aggregation_strategy=AggregationStrategy.FIRST_FAILURE)

        result = aggregator.aggregate(sample_check_results, config)
        # Second result is the first failure
        assert result.status == CheckStatus.FAILED
        assert result.failed_count == 2

    def test_empty_results(self):
        """Test aggregation of empty results."""
        aggregator = CheckResultAggregator()
        config = BatchConfig()

        result = aggregator.aggregate([], config)
        assert result.status == CheckStatus.SKIPPED


class TestProfileResultAggregator:
    """Tests for ProfileResultAggregator."""

    def test_merge_profile_results(self):
        """Test merging profile results."""
        results = [
            ProfileResult(
                status=ProfileStatus.COMPLETED,
                row_count=100,
                columns=(ColumnProfile(column_name="a", dtype="int"),),
                execution_time_ms=10.0,
            ),
            ProfileResult(
                status=ProfileStatus.COMPLETED,
                row_count=200,
                columns=(ColumnProfile(column_name="b", dtype="str"),),
                execution_time_ms=15.0,
            ),
        ]

        aggregator = ProfileResultAggregator()
        config = BatchConfig()
        result = aggregator.aggregate(results, config)

        assert result.status == ProfileStatus.COMPLETED
        assert result.row_count == 300
        assert len(result.columns) == 2
        assert result.execution_time_ms == 25.0


class TestLearnResultAggregator:
    """Tests for LearnResultAggregator."""

    def test_merge_learn_results(self):
        """Test merging learn results."""
        results = [
            LearnResult(
                status=LearnStatus.COMPLETED,
                rules=(
                    LearnedRule(rule_type="not_null", column="a", confidence=0.9),
                ),
                execution_time_ms=10.0,
            ),
            LearnResult(
                status=LearnStatus.COMPLETED,
                rules=(
                    LearnedRule(rule_type="not_null", column="a", confidence=0.95),  # Higher confidence
                    LearnedRule(rule_type="unique", column="b", confidence=0.8),
                ),
                execution_time_ms=15.0,
            ),
        ]

        aggregator = LearnResultAggregator()
        config = BatchConfig()
        result = aggregator.aggregate(results, config)

        assert result.status == LearnStatus.COMPLETED
        assert len(result.rules) == 2
        # Should keep higher confidence rule for column "a"
        a_rule = next(r for r in result.rules if r.column == "a")
        assert a_rule.confidence == 0.95


# =============================================================================
# Hook Tests
# =============================================================================


class TestBatchHooks:
    """Tests for batch operation hooks."""

    def test_logging_hook(self, caplog):
        """Test LoggingBatchHook."""
        import logging

        hook = LoggingBatchHook()

        hook.on_batch_start(0, 3, {})
        hook.on_batch_complete(0, None, 10.0)
        hook.on_batch_error(1, ValueError("test"), 5.0)
        hook.on_all_complete(3, 2, 1, 100.0)

    def test_metrics_hook(self):
        """Test MetricsBatchHook."""
        hook = MetricsBatchHook()

        hook.on_batch_start(0, 3, {})
        hook.on_batch_complete(0, None, 10.0)
        hook.on_batch_complete(1, None, 15.0)
        hook.on_batch_error(2, ValueError("test"), 5.0)
        hook.on_all_complete(3, 2, 1, 100.0)

        assert hook.successful_count == 2
        assert hook.failed_count == 1
        assert hook.average_batch_time_ms == 10.0  # (10 + 15 + 5) / 3
        assert hook.total_time_ms == 100.0

    def test_metrics_hook_reset(self):
        """Test MetricsBatchHook reset."""
        hook = MetricsBatchHook()
        hook.on_batch_complete(0, None, 10.0)
        assert hook.successful_count == 1

        hook.reset()
        assert hook.successful_count == 0
        assert hook.failed_count == 0

    def test_composite_hook(self):
        """Test CompositeBatchHook."""
        hook1 = MagicMock(spec=BatchHook)
        hook2 = MagicMock(spec=BatchHook)
        composite = CompositeBatchHook([hook1, hook2])

        composite.on_batch_start(0, 3, {"key": "value"})

        hook1.on_batch_start.assert_called_once_with(0, 3, {"key": "value"})
        hook2.on_batch_start.assert_called_once_with(0, 3, {"key": "value"})


# =============================================================================
# BatchExecutor Tests
# =============================================================================


class TestBatchExecutor:
    """Tests for synchronous BatchExecutor."""

    def test_check_batch_sequential(self, mock_engine, sample_datasets):
        """Test check_batch with sequential execution."""
        executor = BatchExecutor(mock_engine)
        config = BatchConfig(execution_strategy=ExecutionStrategy.SEQUENTIAL)

        result = executor.check_batch(sample_datasets, config=config)

        assert result.total_batches == 3
        assert result.successful_batches == 3
        assert result.failed_batches == 0
        assert result.is_complete_success is True

    def test_check_batch_parallel(self, mock_engine, sample_datasets):
        """Test check_batch with parallel execution."""
        executor = BatchExecutor(mock_engine)
        config = BatchConfig(
            execution_strategy=ExecutionStrategy.PARALLEL,
            max_workers=2,
        )

        result = executor.check_batch(sample_datasets, config=config)

        assert result.total_batches == 3
        assert result.successful_batches == 3
        assert result.is_complete_success is True

    def test_check_chunked(self, mock_engine, sample_data):
        """Test check_chunked for large data."""
        executor = BatchExecutor(mock_engine)
        config = BatchConfig(batch_size=30)

        result = executor.check_chunked(sample_data, config=config)

        assert result.total_batches == 4  # 100 / 30 = 4 chunks
        assert result.is_complete_success is True

    def test_profile_batch(self, mock_engine, sample_datasets):
        """Test profile_batch."""
        executor = BatchExecutor(mock_engine)

        result = executor.profile_batch(sample_datasets)

        assert result.total_batches == 3
        assert result.successful_batches == 3

    def test_learn_batch(self, mock_engine, sample_datasets):
        """Test learn_batch."""
        executor = BatchExecutor(mock_engine)

        result = executor.learn_batch(sample_datasets)

        assert result.total_batches == 3
        assert result.successful_batches == 3

    def test_empty_datasets(self, mock_engine):
        """Test with empty dataset list."""
        executor = BatchExecutor(mock_engine)

        result = executor.check_batch([])

        assert result.total_batches == 0
        assert result.is_complete_success is True

    def test_with_hooks(self, mock_engine, sample_datasets):
        """Test executor with hooks."""
        metrics_hook = MetricsBatchHook()
        executor = BatchExecutor(mock_engine, hooks=[metrics_hook])

        executor.check_batch(sample_datasets)

        assert metrics_hook.successful_count == 3

    def test_fail_fast(self, failing_mock_engine, sample_datasets):
        """Test fail_fast behavior."""
        executor = BatchExecutor(failing_mock_engine)
        config = FAIL_FAST_BATCH_CONFIG

        result = executor.check_batch(sample_datasets, config=config)

        # Should stop after first failure
        assert result.failed_batches >= 1
        # With fail_fast in sequential mode, we stop early
        if config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            assert result.total_batches >= 1

    def test_continue_on_error(self, failing_mock_engine, sample_datasets):
        """Test continue_on_error behavior."""
        executor = BatchExecutor(failing_mock_engine)
        config = BatchConfig(continue_on_error=True)

        result = executor.check_batch(sample_datasets, config=config)

        # Should process all batches despite errors
        assert result.total_batches == 3

    def test_create_batch_executor(self, mock_engine):
        """Test create_batch_executor convenience function."""
        executor = create_batch_executor(
            mock_engine,
            enable_logging=True,
            enable_metrics=True,
        )

        assert len(executor._hooks) == 2

    def test_add_hook(self, mock_engine):
        """Test adding hooks dynamically."""
        executor = BatchExecutor(mock_engine)
        hook = MetricsBatchHook()

        executor.add_hook(hook)

        assert hook in executor._hooks


# =============================================================================
# AsyncBatchExecutor Tests
# =============================================================================


class TestAsyncBatchExecutor:
    """Tests for asynchronous AsyncBatchExecutor."""

    @pytest.mark.asyncio
    async def test_check_batch_sequential(self, async_mock_engine, sample_datasets):
        """Test async check_batch with sequential execution."""
        executor = AsyncBatchExecutor(async_mock_engine)
        config = BatchConfig(execution_strategy=ExecutionStrategy.SEQUENTIAL)

        result = await executor.check_batch(sample_datasets, config=config)

        assert result.total_batches == 3
        assert result.successful_batches == 3
        assert result.is_complete_success is True

    @pytest.mark.asyncio
    async def test_check_batch_parallel(self, async_mock_engine, sample_datasets):
        """Test async check_batch with parallel execution."""
        executor = AsyncBatchExecutor(async_mock_engine)
        config = BatchConfig(
            execution_strategy=ExecutionStrategy.PARALLEL,
            max_workers=2,
        )

        result = await executor.check_batch(sample_datasets, config=config)

        assert result.total_batches == 3
        assert result.successful_batches == 3

    @pytest.mark.asyncio
    async def test_check_chunked(self, async_mock_engine, sample_data):
        """Test async check_chunked for large data."""
        executor = AsyncBatchExecutor(async_mock_engine)
        config = BatchConfig(batch_size=30)

        result = await executor.check_chunked(sample_data, config=config)

        assert result.total_batches == 4
        assert result.is_complete_success is True

    @pytest.mark.asyncio
    async def test_profile_batch(self, async_mock_engine, sample_datasets):
        """Test async profile_batch."""
        executor = AsyncBatchExecutor(async_mock_engine)

        result = await executor.profile_batch(sample_datasets)

        assert result.total_batches == 3
        assert result.successful_batches == 3

    @pytest.mark.asyncio
    async def test_learn_batch(self, async_mock_engine, sample_datasets):
        """Test async learn_batch."""
        executor = AsyncBatchExecutor(async_mock_engine)

        result = await executor.learn_batch(sample_datasets)

        assert result.total_batches == 3
        assert result.successful_batches == 3

    @pytest.mark.asyncio
    async def test_empty_datasets(self, async_mock_engine):
        """Test with empty dataset list."""
        executor = AsyncBatchExecutor(async_mock_engine)

        result = await executor.check_batch([])

        assert result.total_batches == 0

    @pytest.mark.asyncio
    async def test_with_hooks(self, async_mock_engine, sample_datasets):
        """Test async executor with hooks."""
        metrics_hook = MetricsBatchHook()
        executor = AsyncBatchExecutor(async_mock_engine, hooks=[metrics_hook])

        await executor.check_batch(sample_datasets)

        assert metrics_hook.successful_count == 3

    @pytest.mark.asyncio
    async def test_create_async_batch_executor(self, async_mock_engine):
        """Test create_async_batch_executor convenience function."""
        executor = create_async_batch_executor(
            async_mock_engine,
            enable_logging=True,
            enable_metrics=True,
        )

        assert len(executor._hooks) == 2


# =============================================================================
# Exception Tests
# =============================================================================


class TestBatchExceptions:
    """Tests for batch operation exceptions."""

    def test_batch_operation_error(self):
        """Test BatchOperationError."""
        error = BatchOperationError(
            "Batch failed",
            batch_index=2,
            failed_count=1,
            total_count=5,
        )

        assert error.batch_index == 2
        assert error.failed_count == 1
        assert error.total_count == 5
        assert "batch_index" in error.details

    def test_batch_execution_error(self):
        """Test BatchExecutionError."""
        cause = ValueError("Original error")
        error = BatchExecutionError(
            "Execution failed",
            batch_index=1,
            cause=cause,
        )

        assert error.batch_index == 1
        assert error.cause is cause

    def test_chunking_error(self):
        """Test ChunkingError."""
        error = ChunkingError("Failed to chunk data")
        assert isinstance(error, BatchOperationError)

    def test_aggregation_error(self):
        """Test AggregationError."""
        error = AggregationError("Failed to aggregate results")
        assert isinstance(error, BatchOperationError)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBatchIntegration:
    """Integration tests for batch operations."""

    def test_full_batch_workflow(self, mock_engine, sample_datasets):
        """Test complete batch workflow with all components."""
        # Create executor with hooks
        metrics_hook = MetricsBatchHook()
        executor = BatchExecutor(
            mock_engine,
            default_config=BatchConfig(
                execution_strategy=ExecutionStrategy.PARALLEL,
                max_workers=2,
            ),
            hooks=[metrics_hook],
        )

        # Execute batch check
        result = executor.check_batch(sample_datasets)

        # Verify results
        assert result.is_complete_success is True
        assert result.aggregated_result is not None
        assert metrics_hook.successful_count == 3

    def test_batch_result_serialization(self, mock_engine, sample_datasets):
        """Test that batch results can be serialized."""
        executor = BatchExecutor(mock_engine)
        result = executor.check_batch(sample_datasets)

        # Should be serializable to dict
        data = result.to_dict()
        assert isinstance(data, dict)
        assert "total_batches" in data
        assert "batch_results" in data

    @pytest.mark.asyncio
    async def test_async_batch_workflow(self, async_mock_engine, sample_datasets):
        """Test complete async batch workflow."""
        metrics_hook = MetricsBatchHook()
        executor = AsyncBatchExecutor(
            async_mock_engine,
            hooks=[metrics_hook],
        )

        result = await executor.check_batch(sample_datasets)

        assert result.is_complete_success is True
        assert metrics_hook.successful_count == 3
