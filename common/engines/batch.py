"""Batch Operations for Data Quality Engines.

This module provides batch processing capabilities for data quality operations,
enabling efficient processing of large datasets through chunking, parallel
execution, and result aggregation.

Key Features:
    - Chunked data processing for memory efficiency
    - Parallel and sequential execution strategies
    - Configurable batch sizes and concurrency
    - Progress tracking and callbacks
    - Result aggregation with multiple strategies
    - Both sync and async support

Design Principles:
    1. Strategy Pattern: Pluggable chunking and aggregation strategies
    2. Protocol-based: Works with any DataQualityEngine implementation
    3. Immutable Config: All configuration objects are frozen dataclasses
    4. Observable: Hook system for progress tracking and monitoring

Example:
    >>> from common.engines import TruthoundEngine
    >>> from common.engines.batch import BatchExecutor, BatchConfig
    >>>
    >>> engine = TruthoundEngine()
    >>> executor = BatchExecutor(engine)
    >>> results = executor.check_batch(
    ...     datasets=[df1, df2, df3],
    ...     config=BatchConfig(parallel=True, max_workers=4),
    ... )
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

from common.base import (
    AnomalyResult,
    AnomalyStatus,
    CheckResult,
    CheckStatus,
    DriftResult,
    DriftStatus,
    LearnResult,
    LearnStatus,
    ProfileResult,
    ProfileStatus,
    ValidationFailure,
)
from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from common.engines.base import AsyncDataQualityEngine, DataQualityEngine


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
TResult = TypeVar("TResult", CheckResult, ProfileResult, LearnResult)
TData = TypeVar("TData")


# =============================================================================
# Exceptions
# =============================================================================


class BatchOperationError(TruthoundIntegrationError):
    """Base exception for batch operation errors.

    Attributes:
        batch_index: Index of the batch that failed (if applicable).
        failed_count: Number of failed batch operations.
        total_count: Total number of batch operations.
    """

    def __init__(
        self,
        message: str,
        *,
        batch_index: int | None = None,
        failed_count: int = 0,
        total_count: int = 0,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        details = details or {}
        if batch_index is not None:
            details["batch_index"] = batch_index
        details["failed_count"] = failed_count
        details["total_count"] = total_count
        super().__init__(message, details=details, cause=cause)
        self.batch_index = batch_index
        self.failed_count = failed_count
        self.total_count = total_count


class BatchExecutionError(BatchOperationError):
    """Exception raised when batch execution fails."""

    pass


class ChunkingError(BatchOperationError):
    """Exception raised when data chunking fails."""

    pass


class AggregationError(BatchOperationError):
    """Exception raised when result aggregation fails."""

    pass


# =============================================================================
# Enums
# =============================================================================


class ExecutionStrategy(Enum):
    """Strategy for executing batch operations.

    Attributes:
        SEQUENTIAL: Execute batches one at a time.
        PARALLEL: Execute batches in parallel using threads.
        ADAPTIVE: Automatically choose based on batch count and size.
    """

    SEQUENTIAL = auto()
    PARALLEL = auto()
    ADAPTIVE = auto()


class AggregationStrategy(Enum):
    """Strategy for aggregating batch results.

    Attributes:
        MERGE: Merge all results into a single result.
        WORST: Use the worst status from all results.
        BEST: Use the best status from all results.
        MAJORITY: Use the most common status.
        FIRST_FAILURE: Stop on first failure and return it.
        ALL: Return all individual results without aggregation.
    """

    MERGE = auto()
    WORST = auto()
    BEST = auto()
    MAJORITY = auto()
    FIRST_FAILURE = auto()
    ALL = auto()


class ChunkingStrategy(Enum):
    """Strategy for chunking data.

    Attributes:
        ROW_COUNT: Chunk by number of rows.
        BYTE_SIZE: Chunk by approximate byte size.
        COLUMN_GROUPS: Chunk by column groups.
        CUSTOM: Use a custom chunking function.
    """

    ROW_COUNT = auto()
    BYTE_SIZE = auto()
    COLUMN_GROUPS = auto()
    CUSTOM = auto()


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class BatchConfig:
    """Configuration for batch operations.

    Attributes:
        batch_size: Number of rows per batch (for ROW_COUNT chunking).
        max_workers: Maximum number of parallel workers.
        execution_strategy: How to execute batches.
        aggregation_strategy: How to aggregate results.
        chunking_strategy: How to chunk data.
        fail_fast: Stop on first failure.
        continue_on_error: Continue processing even if some batches fail.
        timeout_per_batch_seconds: Timeout for each batch operation.
        total_timeout_seconds: Total timeout for all batch operations.
        collect_partial_results: Collect results even on failure.
        extra: Additional configuration options.
    """

    batch_size: int = 10000
    max_workers: int = 4
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MERGE
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.ROW_COUNT
    fail_fast: bool = False
    continue_on_error: bool = True
    timeout_per_batch_seconds: float | None = None
    total_timeout_seconds: float | None = None
    collect_partial_results: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def with_batch_size(self, size: int) -> BatchConfig:
        """Create a new config with specified batch size."""
        return BatchConfig(
            batch_size=size,
            max_workers=self.max_workers,
            execution_strategy=self.execution_strategy,
            aggregation_strategy=self.aggregation_strategy,
            chunking_strategy=self.chunking_strategy,
            fail_fast=self.fail_fast,
            continue_on_error=self.continue_on_error,
            timeout_per_batch_seconds=self.timeout_per_batch_seconds,
            total_timeout_seconds=self.total_timeout_seconds,
            collect_partial_results=self.collect_partial_results,
            extra=self.extra,
        )

    def with_max_workers(self, workers: int) -> BatchConfig:
        """Create a new config with specified max workers."""
        return BatchConfig(
            batch_size=self.batch_size,
            max_workers=workers,
            execution_strategy=self.execution_strategy,
            aggregation_strategy=self.aggregation_strategy,
            chunking_strategy=self.chunking_strategy,
            fail_fast=self.fail_fast,
            continue_on_error=self.continue_on_error,
            timeout_per_batch_seconds=self.timeout_per_batch_seconds,
            total_timeout_seconds=self.total_timeout_seconds,
            collect_partial_results=self.collect_partial_results,
            extra=self.extra,
        )

    def with_execution_strategy(self, strategy: ExecutionStrategy) -> BatchConfig:
        """Create a new config with specified execution strategy."""
        return BatchConfig(
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            execution_strategy=strategy,
            aggregation_strategy=self.aggregation_strategy,
            chunking_strategy=self.chunking_strategy,
            fail_fast=self.fail_fast,
            continue_on_error=self.continue_on_error,
            timeout_per_batch_seconds=self.timeout_per_batch_seconds,
            total_timeout_seconds=self.total_timeout_seconds,
            collect_partial_results=self.collect_partial_results,
            extra=self.extra,
        )

    def with_aggregation_strategy(self, strategy: AggregationStrategy) -> BatchConfig:
        """Create a new config with specified aggregation strategy."""
        return BatchConfig(
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            execution_strategy=self.execution_strategy,
            aggregation_strategy=strategy,
            chunking_strategy=self.chunking_strategy,
            fail_fast=self.fail_fast,
            continue_on_error=self.continue_on_error,
            timeout_per_batch_seconds=self.timeout_per_batch_seconds,
            total_timeout_seconds=self.total_timeout_seconds,
            collect_partial_results=self.collect_partial_results,
            extra=self.extra,
        )

    def with_fail_fast(self, fail_fast: bool = True) -> BatchConfig:
        """Create a new config with fail-fast behavior."""
        return BatchConfig(
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            execution_strategy=self.execution_strategy,
            aggregation_strategy=self.aggregation_strategy,
            chunking_strategy=self.chunking_strategy,
            fail_fast=fail_fast,
            continue_on_error=self.continue_on_error,
            timeout_per_batch_seconds=self.timeout_per_batch_seconds,
            total_timeout_seconds=self.total_timeout_seconds,
            collect_partial_results=self.collect_partial_results,
            extra=self.extra,
        )

    def with_timeouts(
        self,
        per_batch: float | None = None,
        total: float | None = None,
    ) -> BatchConfig:
        """Create a new config with specified timeouts."""
        return BatchConfig(
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            execution_strategy=self.execution_strategy,
            aggregation_strategy=self.aggregation_strategy,
            chunking_strategy=self.chunking_strategy,
            fail_fast=self.fail_fast,
            continue_on_error=self.continue_on_error,
            timeout_per_batch_seconds=per_batch,
            total_timeout_seconds=total,
            collect_partial_results=self.collect_partial_results,
            extra=self.extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "execution_strategy": self.execution_strategy.name,
            "aggregation_strategy": self.aggregation_strategy.name,
            "chunking_strategy": self.chunking_strategy.name,
            "fail_fast": self.fail_fast,
            "continue_on_error": self.continue_on_error,
            "timeout_per_batch_seconds": self.timeout_per_batch_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "collect_partial_results": self.collect_partial_results,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchConfig:
        """Create BatchConfig from a dictionary."""
        return cls(
            batch_size=data.get("batch_size", 10000),
            max_workers=data.get("max_workers", 4),
            execution_strategy=ExecutionStrategy[
                data.get("execution_strategy", "ADAPTIVE")
            ],
            aggregation_strategy=AggregationStrategy[
                data.get("aggregation_strategy", "MERGE")
            ],
            chunking_strategy=ChunkingStrategy[
                data.get("chunking_strategy", "ROW_COUNT")
            ],
            fail_fast=data.get("fail_fast", False),
            continue_on_error=data.get("continue_on_error", True),
            timeout_per_batch_seconds=data.get("timeout_per_batch_seconds"),
            total_timeout_seconds=data.get("total_timeout_seconds"),
            collect_partial_results=data.get("collect_partial_results", True),
            extra=data.get("extra", {}),
        )


# Preset Configurations
DEFAULT_BATCH_CONFIG = BatchConfig()

PARALLEL_BATCH_CONFIG = BatchConfig(
    execution_strategy=ExecutionStrategy.PARALLEL,
    max_workers=8,
)

SEQUENTIAL_BATCH_CONFIG = BatchConfig(
    execution_strategy=ExecutionStrategy.SEQUENTIAL,
)

FAIL_FAST_BATCH_CONFIG = BatchConfig(
    fail_fast=True,
    continue_on_error=False,
    aggregation_strategy=AggregationStrategy.FIRST_FAILURE,
)

LARGE_DATA_BATCH_CONFIG = BatchConfig(
    batch_size=50000,
    max_workers=4,
    execution_strategy=ExecutionStrategy.PARALLEL,
)


# =============================================================================
# Batch Result Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class BatchItemResult(Generic[TResult]):
    """Result of a single batch operation.

    Attributes:
        index: Index of this batch in the sequence.
        result: The operation result (if successful).
        error: The error (if failed).
        execution_time_ms: Time taken for this batch.
        metadata: Additional metadata about this batch.
    """

    index: int
    result: TResult | None = None
    error: Exception | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if this batch succeeded."""
        return self.error is None and self.result is not None

    @property
    def is_failure(self) -> bool:
        """Check if this batch failed."""
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "result": self.result.to_dict() if self.result else None,
            "error": str(self.error) if self.error else None,
            "is_success": self.is_success,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class BatchResult(Generic[TResult]):
    """Aggregated result of batch operations.

    Attributes:
        aggregated_result: The combined result (based on aggregation strategy).
        batch_results: Individual results for each batch.
        total_batches: Total number of batches processed.
        successful_batches: Number of successful batches.
        failed_batches: Number of failed batches.
        total_execution_time_ms: Total execution time.
        metadata: Additional metadata about the batch operation.
    """

    aggregated_result: TResult | None = None
    batch_results: tuple[BatchItemResult[TResult], ...] = ()
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    total_execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete_success(self) -> bool:
        """Check if all batches succeeded."""
        return self.failed_batches == 0 and self.successful_batches == self.total_batches

    @property
    def is_partial_success(self) -> bool:
        """Check if some batches succeeded."""
        return self.successful_batches > 0 and self.failed_batches > 0

    @property
    def is_complete_failure(self) -> bool:
        """Check if all batches failed."""
        return self.failed_batches == self.total_batches

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_batches == 0:
            return 100.0
        return (self.successful_batches / self.total_batches) * 100

    def get_failures(self) -> tuple[BatchItemResult[TResult], ...]:
        """Get all failed batch results."""
        return tuple(r for r in self.batch_results if r.is_failure)

    def get_successes(self) -> tuple[BatchItemResult[TResult], ...]:
        """Get all successful batch results."""
        return tuple(r for r in self.batch_results if r.is_success)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "aggregated_result": (
                self.aggregated_result.to_dict() if self.aggregated_result else None
            ),
            "batch_results": [r.to_dict() for r in self.batch_results],
            "total_batches": self.total_batches,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "total_execution_time_ms": self.total_execution_time_ms,
            "is_complete_success": self.is_complete_success,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class DataChunker(Protocol[TData]):
    """Protocol for data chunking implementations.

    Implementations should split data into smaller chunks for batch processing.
    """

    def chunk(
        self,
        data: TData,
        config: BatchConfig,
    ) -> Iterator[TData]:
        """Split data into chunks.

        Args:
            data: Data to chunk.
            config: Batch configuration.

        Yields:
            Data chunks.
        """
        ...

    def estimate_chunks(
        self,
        data: TData,
        config: BatchConfig,
    ) -> int:
        """Estimate the number of chunks.

        Args:
            data: Data to chunk.
            config: Batch configuration.

        Returns:
            Estimated number of chunks.
        """
        ...


@runtime_checkable
class ResultAggregator(Protocol[TResult]):
    """Protocol for result aggregation implementations.

    Implementations should combine multiple results into a single result.
    """

    def aggregate(
        self,
        results: Sequence[TResult],
        config: BatchConfig,
    ) -> TResult:
        """Aggregate multiple results.

        Args:
            results: Results to aggregate.
            config: Batch configuration.

        Returns:
            Aggregated result.
        """
        ...


@runtime_checkable
class BatchHook(Protocol):
    """Protocol for batch operation lifecycle hooks.

    Implementations can observe and react to batch operation events.
    """

    def on_batch_start(
        self,
        batch_index: int,
        total_batches: int,
        metadata: dict[str, Any],
    ) -> None:
        """Called when a batch starts processing."""
        ...

    def on_batch_complete(
        self,
        batch_index: int,
        result: Any,
        execution_time_ms: float,
    ) -> None:
        """Called when a batch completes successfully."""
        ...

    def on_batch_error(
        self,
        batch_index: int,
        error: Exception,
        execution_time_ms: float,
    ) -> None:
        """Called when a batch fails."""
        ...

    def on_all_complete(
        self,
        total_batches: int,
        successful: int,
        failed: int,
        total_time_ms: float,
    ) -> None:
        """Called when all batches complete."""
        ...


# =============================================================================
# Data Chunkers
# =============================================================================


class RowCountChunker:
    """Chunker that splits data by row count.

    Works with any data type that supports len() and slicing.
    """

    def chunk(
        self,
        data: Any,
        config: BatchConfig,
    ) -> Iterator[Any]:
        """Split data into chunks by row count.

        Args:
            data: Data to chunk (DataFrame or sequence).
            config: Batch configuration.

        Yields:
            Data chunks.
        """
        total_rows = len(data)
        batch_size = config.batch_size

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            yield data[start:end]

    def estimate_chunks(
        self,
        data: Any,
        config: BatchConfig,
    ) -> int:
        """Estimate number of chunks."""
        total_rows = len(data)
        batch_size = config.batch_size
        return (total_rows + batch_size - 1) // batch_size


class PolarsChunker:
    """Optimized chunker for Polars DataFrames."""

    def chunk(
        self,
        data: Any,
        config: BatchConfig,
    ) -> Iterator[Any]:
        """Split Polars DataFrame into chunks.

        Args:
            data: Polars DataFrame to chunk.
            config: Batch configuration.

        Yields:
            DataFrame chunks.
        """
        try:
            import polars as pl

            if not isinstance(data, pl.DataFrame):
                # Fallback to row-based chunking
                chunker = RowCountChunker()
                yield from chunker.chunk(data, config)
                return

            total_rows = data.height
            batch_size = config.batch_size

            for start in range(0, total_rows, batch_size):
                yield data.slice(start, batch_size)

        except ImportError:
            # Polars not available, fallback
            chunker = RowCountChunker()
            yield from chunker.chunk(data, config)

    def estimate_chunks(
        self,
        data: Any,
        config: BatchConfig,
    ) -> int:
        """Estimate number of chunks."""
        try:
            import polars as pl

            if isinstance(data, pl.DataFrame):
                total_rows = data.height
            else:
                total_rows = len(data)
        except ImportError:
            total_rows = len(data)

        batch_size = config.batch_size
        return (total_rows + batch_size - 1) // batch_size


class DatasetListChunker:
    """Chunker for lists of datasets (no further splitting)."""

    def chunk(
        self,
        data: Sequence[Any],
        config: BatchConfig,
    ) -> Iterator[Any]:
        """Yield each dataset as a separate chunk.

        Args:
            data: Sequence of datasets.
            config: Batch configuration (unused).

        Yields:
            Individual datasets.
        """
        yield from data

    def estimate_chunks(
        self,
        data: Sequence[Any],
        config: BatchConfig,
    ) -> int:
        """Return count of datasets."""
        return len(data)


# =============================================================================
# Result Aggregators
# =============================================================================


class CheckResultAggregator:
    """Aggregator for CheckResult objects."""

    def aggregate(
        self,
        results: Sequence[CheckResult],
        config: BatchConfig,
    ) -> CheckResult:
        """Aggregate multiple CheckResults.

        Args:
            results: CheckResults to aggregate.
            config: Batch configuration.

        Returns:
            Aggregated CheckResult.
        """
        if not results:
            return CheckResult(status=CheckStatus.SKIPPED)

        strategy = config.aggregation_strategy

        if strategy == AggregationStrategy.FIRST_FAILURE:
            for result in results:
                if result.status == CheckStatus.FAILED:
                    return result
            return results[-1]

        if strategy == AggregationStrategy.WORST:
            return self._aggregate_worst(results)

        if strategy == AggregationStrategy.BEST:
            return self._aggregate_best(results)

        if strategy == AggregationStrategy.MAJORITY:
            return self._aggregate_majority(results)

        # Default: MERGE
        return self._aggregate_merge(results)

    def _aggregate_merge(self, results: Sequence[CheckResult]) -> CheckResult:
        """Merge all results into one."""
        total_passed = sum(r.passed_count for r in results)
        total_failed = sum(r.failed_count for r in results)
        total_warning = sum(r.warning_count for r in results)
        total_skipped = sum(r.skipped_count for r in results)
        total_time = sum(r.execution_time_ms for r in results)

        all_failures: list[ValidationFailure] = []
        for result in results:
            all_failures.extend(result.failures)

        # Determine overall status
        if total_failed > 0:
            status = CheckStatus.FAILED
        elif total_warning > 0:
            status = CheckStatus.WARNING
        elif total_passed > 0:
            status = CheckStatus.PASSED
        else:
            status = CheckStatus.SKIPPED

        return CheckResult(
            status=status,
            passed_count=total_passed,
            failed_count=total_failed,
            warning_count=total_warning,
            skipped_count=total_skipped,
            failures=tuple(all_failures),
            execution_time_ms=total_time,
            metadata={
                "batch_count": len(results),
                "aggregation_strategy": "MERGE",
            },
        )

    def _aggregate_worst(self, results: Sequence[CheckResult]) -> CheckResult:
        """Return result with worst status."""
        status_priority = {
            CheckStatus.ERROR: 0,
            CheckStatus.FAILED: 1,
            CheckStatus.WARNING: 2,
            CheckStatus.SKIPPED: 3,
            CheckStatus.PASSED: 4,
        }
        worst = min(results, key=lambda r: status_priority.get(r.status, 5))
        return worst

    def _aggregate_best(self, results: Sequence[CheckResult]) -> CheckResult:
        """Return result with best status."""
        status_priority = {
            CheckStatus.PASSED: 0,
            CheckStatus.WARNING: 1,
            CheckStatus.SKIPPED: 2,
            CheckStatus.FAILED: 3,
            CheckStatus.ERROR: 4,
        }
        best = min(results, key=lambda r: status_priority.get(r.status, 5))
        return best

    def _aggregate_majority(self, results: Sequence[CheckResult]) -> CheckResult:
        """Return result with most common status."""
        from collections import Counter

        status_counts = Counter(r.status for r in results)
        most_common_status = status_counts.most_common(1)[0][0]

        # Return first result with that status
        for result in results:
            if result.status == most_common_status:
                return result
        return results[0]


class ProfileResultAggregator:
    """Aggregator for ProfileResult objects."""

    def aggregate(
        self,
        results: Sequence[ProfileResult],
        config: BatchConfig,
    ) -> ProfileResult:
        """Aggregate multiple ProfileResults.

        Args:
            results: ProfileResults to aggregate.
            config: Batch configuration.

        Returns:
            Aggregated ProfileResult.
        """
        if not results:
            return ProfileResult(status=ProfileStatus.COMPLETED)

        # Merge column profiles
        all_columns = []
        for result in results:
            all_columns.extend(result.columns)

        # Determine status
        statuses = [r.status for r in results]
        if ProfileStatus.FAILED in statuses:
            status = ProfileStatus.FAILED
        elif ProfileStatus.PARTIAL in statuses:
            status = ProfileStatus.PARTIAL
        else:
            status = ProfileStatus.COMPLETED

        total_rows = sum(r.row_count for r in results)
        total_time = sum(r.execution_time_ms for r in results)

        return ProfileResult(
            status=status,
            row_count=total_rows,
            column_count=len(all_columns),
            columns=tuple(all_columns),
            execution_time_ms=total_time,
            metadata={
                "batch_count": len(results),
                "aggregation_strategy": config.aggregation_strategy.name,
            },
        )


class LearnResultAggregator:
    """Aggregator for LearnResult objects."""

    def aggregate(
        self,
        results: Sequence[LearnResult],
        config: BatchConfig,
    ) -> LearnResult:
        """Aggregate multiple LearnResults.

        Args:
            results: LearnResults to aggregate.
            config: Batch configuration.

        Returns:
            Aggregated LearnResult.
        """
        if not results:
            return LearnResult(status=LearnStatus.COMPLETED)

        # Merge learned rules (deduplicate by rule_type + column)
        seen_rules: dict[tuple[str, str], Any] = {}
        for result in results:
            for rule in result.rules:
                key = (rule.rule_type, rule.column)
                if key not in seen_rules or rule.confidence > seen_rules[key].confidence:
                    seen_rules[key] = rule

        # Determine status
        statuses = [r.status for r in results]
        if LearnStatus.FAILED in statuses:
            status = LearnStatus.FAILED
        elif LearnStatus.PARTIAL in statuses:
            status = LearnStatus.PARTIAL
        else:
            status = LearnStatus.COMPLETED

        total_time = sum(r.execution_time_ms for r in results)
        total_columns = sum(r.columns_analyzed for r in results)

        return LearnResult(
            status=status,
            rules=tuple(seen_rules.values()),
            columns_analyzed=total_columns,
            execution_time_ms=total_time,
            metadata={
                "batch_count": len(results),
                "aggregation_strategy": config.aggregation_strategy.name,
            },
        )


class DriftResultBatchAggregator:
    """Aggregator for DriftResult objects in batch context.

    Merges drift results from multiple chunks/datasets, deduplicating
    column drifts and selecting the worst overall status.
    """

    def aggregate(
        self,
        results: Sequence[DriftResult],
        config: BatchConfig,
    ) -> DriftResult:
        """Aggregate multiple DriftResults.

        Args:
            results: DriftResults to aggregate.
            config: Batch configuration.

        Returns:
            Aggregated DriftResult.
        """
        if not results:
            from common.base import DriftMethod

            return DriftResult(
                status=DriftStatus.NO_DRIFT,
                drifted_columns=(),
                total_columns=0,
                drifted_count=0,
                method=DriftMethod.AUTO,
            )

        # Merge column drifts, keeping the higher statistic per column
        seen: dict[str, Any] = {}
        for result in results:
            for col in result.drifted_columns:
                key = col.column
                if key not in seen or col.statistic > seen[key].statistic:
                    seen[key] = col

        all_columns = tuple(seen.values())
        drifted_count = sum(1 for c in all_columns if c.is_drifted)
        total_columns = max(r.total_columns for r in results)

        # Worst status wins
        status_priority = {
            DriftStatus.ERROR: 0,
            DriftStatus.DRIFT_DETECTED: 1,
            DriftStatus.WARNING: 2,
            DriftStatus.NO_DRIFT: 3,
        }
        worst = min(results, key=lambda r: status_priority.get(r.status, 3))

        total_time = sum(r.execution_time_ms for r in results)

        from common.base import DriftMethod

        # Use most common method
        methods = [r.method for r in results]
        method = max(set(methods), key=methods.count) if methods else DriftMethod.AUTO

        return DriftResult(
            status=worst.status,
            drifted_columns=all_columns,
            total_columns=total_columns,
            drifted_count=drifted_count,
            method=method,
            execution_time_ms=total_time,
            metadata={
                "batch_count": len(results),
                "aggregation_strategy": config.aggregation_strategy.name,
            },
        )


class AnomalyResultBatchAggregator:
    """Aggregator for AnomalyResult objects in batch context.

    Merges anomaly results from multiple chunks/datasets, deduplicating
    anomaly scores and selecting the worst overall status.
    """

    def aggregate(
        self,
        results: Sequence[AnomalyResult],
        config: BatchConfig,
    ) -> AnomalyResult:
        """Aggregate multiple AnomalyResults.

        Args:
            results: AnomalyResults to aggregate.
            config: Batch configuration.

        Returns:
            Aggregated AnomalyResult.
        """
        if not results:
            return AnomalyResult(
                status=AnomalyStatus.NORMAL,
                anomalies=(),
                anomalous_row_count=0,
                total_row_count=0,
                detector="merged",
            )

        # Merge anomaly scores, keeping highest score per column
        seen: dict[str, Any] = {}
        for result in results:
            for anomaly in result.anomalies:
                key = anomaly.column
                if key not in seen or anomaly.score > seen[key].score:
                    seen[key] = anomaly

        all_anomalies = tuple(seen.values())
        total_rows = sum(r.total_row_count for r in results)
        anomalous_rows = sum(r.anomalous_row_count for r in results)

        # Worst status wins
        status_priority = {
            AnomalyStatus.ERROR: 0,
            AnomalyStatus.ANOMALY_DETECTED: 1,
            AnomalyStatus.WARNING: 2,
            AnomalyStatus.NORMAL: 3,
        }
        worst = min(results, key=lambda r: status_priority.get(r.status, 3))

        total_time = sum(r.execution_time_ms for r in results)

        return AnomalyResult(
            status=worst.status,
            anomalies=all_anomalies,
            anomalous_row_count=anomalous_rows,
            total_row_count=total_rows,
            detector="merged",
            execution_time_ms=total_time,
            metadata={
                "batch_count": len(results),
                "aggregation_strategy": config.aggregation_strategy.name,
            },
        )


# =============================================================================
# Batch Hooks
# =============================================================================


class LoggingBatchHook:
    """Hook that logs batch operation events."""

    def __init__(self, logger: Any | None = None) -> None:
        """Initialize with optional logger.

        Args:
            logger: Logger instance. If None, uses standard logging.
        """
        if logger is None:
            import logging

            logger = logging.getLogger(__name__)
        self._logger = logger

    def on_batch_start(
        self,
        batch_index: int,
        total_batches: int,
        metadata: dict[str, Any],
    ) -> None:
        """Log batch start."""
        self._logger.info(
            f"Starting batch {batch_index + 1}/{total_batches}",
            extra={"batch_index": batch_index, **metadata},
        )

    def on_batch_complete(
        self,
        batch_index: int,
        result: Any,
        execution_time_ms: float,
    ) -> None:
        """Log batch completion."""
        self._logger.info(
            f"Completed batch {batch_index + 1} in {execution_time_ms:.2f}ms",
            extra={"batch_index": batch_index, "execution_time_ms": execution_time_ms},
        )

    def on_batch_error(
        self,
        batch_index: int,
        error: Exception,
        execution_time_ms: float,
    ) -> None:
        """Log batch error."""
        self._logger.error(
            f"Batch {batch_index + 1} failed: {error}",
            extra={"batch_index": batch_index, "error": str(error)},
            exc_info=True,
        )

    def on_all_complete(
        self,
        total_batches: int,
        successful: int,
        failed: int,
        total_time_ms: float,
    ) -> None:
        """Log overall completion."""
        self._logger.info(
            f"Batch operation complete: {successful}/{total_batches} successful, "
            f"{failed} failed, total time {total_time_ms:.2f}ms",
        )


class MetricsBatchHook:
    """Hook that collects batch operation metrics."""

    def __init__(self) -> None:
        """Initialize metrics storage."""
        self._lock = threading.Lock()
        self._batch_times: list[float] = []
        self._successful_count = 0
        self._failed_count = 0
        self._total_time_ms = 0.0

    def on_batch_start(
        self,
        batch_index: int,
        total_batches: int,
        metadata: dict[str, Any],
    ) -> None:
        """Record batch start (no-op for metrics)."""
        pass

    def on_batch_complete(
        self,
        batch_index: int,
        result: Any,
        execution_time_ms: float,
    ) -> None:
        """Record successful batch."""
        with self._lock:
            self._batch_times.append(execution_time_ms)
            self._successful_count += 1

    def on_batch_error(
        self,
        batch_index: int,
        error: Exception,
        execution_time_ms: float,
    ) -> None:
        """Record failed batch."""
        with self._lock:
            self._batch_times.append(execution_time_ms)
            self._failed_count += 1

    def on_all_complete(
        self,
        total_batches: int,
        successful: int,
        failed: int,
        total_time_ms: float,
    ) -> None:
        """Record overall completion."""
        with self._lock:
            self._total_time_ms = total_time_ms

    @property
    def successful_count(self) -> int:
        """Get count of successful batches."""
        with self._lock:
            return self._successful_count

    @property
    def failed_count(self) -> int:
        """Get count of failed batches."""
        with self._lock:
            return self._failed_count

    @property
    def average_batch_time_ms(self) -> float:
        """Get average batch execution time."""
        with self._lock:
            if not self._batch_times:
                return 0.0
            return sum(self._batch_times) / len(self._batch_times)

    @property
    def total_time_ms(self) -> float:
        """Get total execution time."""
        with self._lock:
            return self._total_time_ms

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._batch_times.clear()
            self._successful_count = 0
            self._failed_count = 0
            self._total_time_ms = 0.0


class CompositeBatchHook:
    """Hook that delegates to multiple hooks."""

    def __init__(self, hooks: Sequence[BatchHook]) -> None:
        """Initialize with hooks.

        Args:
            hooks: Hooks to delegate to.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: BatchHook) -> None:
        """Add a hook."""
        self._hooks.append(hook)

    def on_batch_start(
        self,
        batch_index: int,
        total_batches: int,
        metadata: dict[str, Any],
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            hook.on_batch_start(batch_index, total_batches, metadata)

    def on_batch_complete(
        self,
        batch_index: int,
        result: Any,
        execution_time_ms: float,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            hook.on_batch_complete(batch_index, result, execution_time_ms)

    def on_batch_error(
        self,
        batch_index: int,
        error: Exception,
        execution_time_ms: float,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            hook.on_batch_error(batch_index, error, execution_time_ms)

    def on_all_complete(
        self,
        total_batches: int,
        successful: int,
        failed: int,
        total_time_ms: float,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            hook.on_all_complete(total_batches, successful, failed, total_time_ms)


# =============================================================================
# Batch Executor
# =============================================================================


class BatchExecutor:
    """Executor for batch data quality operations.

    Provides check_batch(), profile_batch(), and learn_batch() methods
    for processing multiple datasets or large datasets in chunks.

    Example:
        >>> engine = TruthoundEngine()
        >>> executor = BatchExecutor(engine)
        >>>
        >>> # Process multiple datasets
        >>> results = executor.check_batch(
        ...     datasets=[df1, df2, df3],
        ...     config=BatchConfig(parallel=True),
        ... )
        >>>
        >>> # Process large dataset in chunks
        >>> result = executor.check_chunked(
        ...     data=large_df,
        ...     rules=[{"type": "not_null", "column": "id"}],
        ...     config=BatchConfig(batch_size=10000),
        ... )
    """

    def __init__(
        self,
        engine: DataQualityEngine,
        *,
        default_config: BatchConfig | None = None,
        hooks: Sequence[BatchHook] | None = None,
    ) -> None:
        """Initialize BatchExecutor.

        Args:
            engine: Data quality engine to use.
            default_config: Default batch configuration.
            hooks: Batch operation hooks.
        """
        self._engine = engine
        self._default_config = default_config or DEFAULT_BATCH_CONFIG
        self._hooks = list(hooks) if hooks else []
        self._check_aggregator = CheckResultAggregator()
        self._profile_aggregator = ProfileResultAggregator()
        self._learn_aggregator = LearnResultAggregator()
        self._drift_aggregator: ResultAggregator[Any] | None = None
        self._anomaly_aggregator: ResultAggregator[Any] | None = None
        self._chunker = PolarsChunker()

    def _get_drift_aggregator(self) -> ResultAggregator[Any]:
        """Lazily initialize and return the drift result aggregator."""
        if self._drift_aggregator is None:
            self._drift_aggregator = DriftResultBatchAggregator()
        return self._drift_aggregator

    def _get_anomaly_aggregator(self) -> ResultAggregator[Any]:
        """Lazily initialize and return the anomaly result aggregator."""
        if self._anomaly_aggregator is None:
            self._anomaly_aggregator = AnomalyResultBatchAggregator()
        return self._anomaly_aggregator

    @property
    def engine(self) -> DataQualityEngine:
        """Return the underlying engine."""
        return self._engine

    def add_hook(self, hook: BatchHook) -> None:
        """Add a batch operation hook."""
        self._hooks.append(hook)

    def _notify_batch_start(
        self,
        batch_index: int,
        total_batches: int,
        metadata: dict[str, Any],
    ) -> None:
        """Notify hooks of batch start."""
        for hook in self._hooks:
            hook.on_batch_start(batch_index, total_batches, metadata)

    def _notify_batch_complete(
        self,
        batch_index: int,
        result: Any,
        execution_time_ms: float,
    ) -> None:
        """Notify hooks of batch completion."""
        for hook in self._hooks:
            hook.on_batch_complete(batch_index, result, execution_time_ms)

    def _notify_batch_error(
        self,
        batch_index: int,
        error: Exception,
        execution_time_ms: float,
    ) -> None:
        """Notify hooks of batch error."""
        for hook in self._hooks:
            hook.on_batch_error(batch_index, error, execution_time_ms)

    def _notify_all_complete(
        self,
        total_batches: int,
        successful: int,
        failed: int,
        total_time_ms: float,
    ) -> None:
        """Notify hooks of overall completion."""
        for hook in self._hooks:
            hook.on_all_complete(total_batches, successful, failed, total_time_ms)

    def _should_use_parallel(
        self,
        batch_count: int,
        config: BatchConfig,
    ) -> bool:
        """Determine if parallel execution should be used."""
        if config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            return False
        if config.execution_strategy == ExecutionStrategy.PARALLEL:
            return True
        # ADAPTIVE: use parallel for 2+ batches
        return batch_count >= 2

    def check_batch(
        self,
        datasets: Sequence[Any],
        rules: Sequence[Mapping[str, Any]] | None = None,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[CheckResult]:
        """Execute check operation on multiple datasets.

        Args:
            datasets: Sequence of datasets to check.
            rules: Validation rules to apply.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.check().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config
        rules = rules or []

        def check_one(data: Any) -> CheckResult:
            return self._engine.check(data, list(rules), **kwargs)

        return self._execute_batch(
            items=list(datasets),
            operation=check_one,
            aggregator=self._check_aggregator,
            config=config,
            operation_name="check",
        )

    def check_chunked(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] | None = None,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[CheckResult]:
        """Execute check operation on a large dataset by chunking.

        Args:
            data: Large dataset to chunk and check.
            rules: Validation rules to apply.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.check().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config
        rules = rules or []
        chunks = list(self._chunker.chunk(data, config))

        def check_one(chunk: Any) -> CheckResult:
            return self._engine.check(chunk, list(rules), **kwargs)

        return self._execute_batch(
            items=chunks,
            operation=check_one,
            aggregator=self._check_aggregator,
            config=config,
            operation_name="check_chunked",
        )

    def profile_batch(
        self,
        datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[ProfileResult]:
        """Execute profile operation on multiple datasets.

        Args:
            datasets: Sequence of datasets to profile.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.profile().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config

        def profile_one(data: Any) -> ProfileResult:
            return self._engine.profile(data, **kwargs)

        return self._execute_batch(
            items=list(datasets),
            operation=profile_one,
            aggregator=self._profile_aggregator,
            config=config,
            operation_name="profile",
        )

    def profile_chunked(
        self,
        data: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[ProfileResult]:
        """Execute profile operation on a large dataset by chunking.

        Args:
            data: Large dataset to chunk and profile.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.profile().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config
        chunks = list(self._chunker.chunk(data, config))

        def profile_one(chunk: Any) -> ProfileResult:
            return self._engine.profile(chunk, **kwargs)

        return self._execute_batch(
            items=chunks,
            operation=profile_one,
            aggregator=self._profile_aggregator,
            config=config,
            operation_name="profile_chunked",
        )

    def learn_batch(
        self,
        datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[LearnResult]:
        """Execute learn operation on multiple datasets.

        Args:
            datasets: Sequence of datasets to learn from.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.learn().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config

        def learn_one(data: Any) -> LearnResult:
            return self._engine.learn(data, **kwargs)

        return self._execute_batch(
            items=list(datasets),
            operation=learn_one,
            aggregator=self._learn_aggregator,
            config=config,
            operation_name="learn",
        )

    def learn_chunked(
        self,
        data: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[LearnResult]:
        """Execute learn operation on a large dataset by chunking.

        Args:
            data: Large dataset to chunk and learn from.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.learn().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config
        chunks = list(self._chunker.chunk(data, config))

        def learn_one(chunk: Any) -> LearnResult:
            return self._engine.learn(chunk, **kwargs)

        return self._execute_batch(
            items=chunks,
            operation=learn_one,
            aggregator=self._learn_aggregator,
            config=config,
            operation_name="learn_chunked",
        )

    def drift_batch(
        self,
        baseline_datasets: Sequence[Any],
        current_datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[DriftResult]:
        """Execute drift detection on multiple dataset pairs.

        Each baseline/current pair is processed as a batch item.
        The engine must implement DriftDetectionEngine.

        Args:
            baseline_datasets: Sequence of baseline datasets.
            current_datasets: Sequence of current datasets (same length).
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_drift().

        Returns:
            BatchResult containing individual and aggregated drift results.

        Raises:
            BatchOperationError: If engine doesn't support drift detection.
            ValueError: If baseline and current datasets have different lengths.
        """
        from common.engines.base import supports_drift

        if not supports_drift(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support drift detection"
            )
        if len(baseline_datasets) != len(current_datasets):
            msg = (
                f"baseline_datasets ({len(baseline_datasets)}) and "
                f"current_datasets ({len(current_datasets)}) must have the same length"
            )
            raise ValueError(msg)

        config = config or self._default_config
        pairs = list(zip(baseline_datasets, current_datasets))

        def drift_one(pair: tuple[Any, Any]) -> DriftResult:
            return self._engine.detect_drift(pair[0], pair[1], **kwargs)

        return self._execute_batch(
            items=pairs,
            operation=drift_one,
            aggregator=self._get_drift_aggregator(),
            config=config,
            operation_name="drift",
        )

    def drift_chunked(
        self,
        baseline: Any,
        current: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[DriftResult]:
        """Execute drift detection on large datasets by chunking.

        Both baseline and current are chunked in parallel, and drift is
        detected per chunk pair.

        Args:
            baseline: Large baseline dataset.
            current: Large current dataset.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_drift().

        Returns:
            BatchResult containing individual and aggregated drift results.

        Raises:
            BatchOperationError: If engine doesn't support drift detection.
        """
        from common.engines.base import supports_drift

        if not supports_drift(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support drift detection"
            )

        config = config or self._default_config
        baseline_chunks = list(self._chunker.chunk(baseline, config))
        current_chunks = list(self._chunker.chunk(current, config))

        # Align chunks: use min length
        chunk_count = min(len(baseline_chunks), len(current_chunks))
        pairs = [(baseline_chunks[i], current_chunks[i]) for i in range(chunk_count)]

        def drift_one(pair: tuple[Any, Any]) -> DriftResult:
            return self._engine.detect_drift(pair[0], pair[1], **kwargs)

        return self._execute_batch(
            items=pairs,
            operation=drift_one,
            aggregator=self._get_drift_aggregator(),
            config=config,
            operation_name="drift_chunked",
        )

    def anomaly_batch(
        self,
        datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[AnomalyResult]:
        """Execute anomaly detection on multiple datasets.

        The engine must implement AnomalyDetectionEngine.

        Args:
            datasets: Sequence of datasets to check for anomalies.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_anomalies().

        Returns:
            BatchResult containing individual and aggregated anomaly results.

        Raises:
            BatchOperationError: If engine doesn't support anomaly detection.
        """
        from common.engines.base import supports_anomaly

        if not supports_anomaly(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support anomaly detection"
            )

        config = config or self._default_config

        def anomaly_one(data: Any) -> AnomalyResult:
            return self._engine.detect_anomalies(data, **kwargs)

        return self._execute_batch(
            items=list(datasets),
            operation=anomaly_one,
            aggregator=self._get_anomaly_aggregator(),
            config=config,
            operation_name="anomaly",
        )

    def anomaly_chunked(
        self,
        data: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[AnomalyResult]:
        """Execute anomaly detection on a large dataset by chunking.

        Args:
            data: Large dataset to chunk and check for anomalies.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_anomalies().

        Returns:
            BatchResult containing individual and aggregated anomaly results.

        Raises:
            BatchOperationError: If engine doesn't support anomaly detection.
        """
        from common.engines.base import supports_anomaly

        if not supports_anomaly(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support anomaly detection"
            )

        config = config or self._default_config
        chunks = list(self._chunker.chunk(data, config))

        def anomaly_one(chunk: Any) -> AnomalyResult:
            return self._engine.detect_anomalies(chunk, **kwargs)

        return self._execute_batch(
            items=chunks,
            operation=anomaly_one,
            aggregator=self._get_anomaly_aggregator(),
            config=config,
            operation_name="anomaly_chunked",
        )

    def _execute_batch(
        self,
        items: list[Any],
        operation: Callable[[Any], TResult],
        aggregator: ResultAggregator[TResult],
        config: BatchConfig,
        operation_name: str,
    ) -> BatchResult[TResult]:
        """Execute batch operation.

        Args:
            items: Items to process.
            operation: Operation to apply to each item.
            aggregator: Result aggregator.
            config: Batch configuration.
            operation_name: Name of the operation for logging.

        Returns:
            BatchResult with all results.
        """
        total_batches = len(items)
        if total_batches == 0:
            return BatchResult(
                total_batches=0,
                successful_batches=0,
                failed_batches=0,
            )

        start_time = time.perf_counter()
        use_parallel = self._should_use_parallel(total_batches, config)

        if use_parallel:
            batch_results = self._execute_parallel(
                items=items,
                operation=operation,
                config=config,
            )
        else:
            batch_results = self._execute_sequential(
                items=items,
                operation=operation,
                config=config,
            )

        total_time_ms = (time.perf_counter() - start_time) * 1000

        # Count successes and failures
        successful = sum(1 for r in batch_results if r.is_success)
        failed = sum(1 for r in batch_results if r.is_failure)

        # Notify completion
        self._notify_all_complete(total_batches, successful, failed, total_time_ms)

        # Aggregate results
        successful_results = [r.result for r in batch_results if r.result is not None]
        aggregated = (
            aggregator.aggregate(successful_results, config) if successful_results else None
        )

        return BatchResult(
            aggregated_result=aggregated,
            batch_results=tuple(batch_results),
            total_batches=total_batches,
            successful_batches=successful,
            failed_batches=failed,
            total_execution_time_ms=total_time_ms,
            metadata={
                "operation": operation_name,
                "execution_strategy": "parallel" if use_parallel else "sequential",
                "config": config.to_dict(),
            },
        )

    def _execute_sequential(
        self,
        items: list[Any],
        operation: Callable[[Any], TResult],
        config: BatchConfig,
    ) -> list[BatchItemResult[TResult]]:
        """Execute batches sequentially."""
        results: list[BatchItemResult[TResult]] = []
        total_batches = len(items)

        for index, item in enumerate(items):
            self._notify_batch_start(index, total_batches, {})
            start_time = time.perf_counter()

            try:
                result = operation(item)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_complete(index, result, execution_time_ms)
                results.append(
                    BatchItemResult(
                        index=index,
                        result=result,
                        execution_time_ms=execution_time_ms,
                    )
                )
            except Exception as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_error(index, e, execution_time_ms)

                if config.fail_fast:
                    results.append(
                        BatchItemResult(
                            index=index,
                            error=e,
                            execution_time_ms=execution_time_ms,
                        )
                    )
                    break

                if config.continue_on_error:
                    results.append(
                        BatchItemResult(
                            index=index,
                            error=e,
                            execution_time_ms=execution_time_ms,
                        )
                    )
                else:
                    raise BatchExecutionError(
                        f"Batch {index} failed: {e}",
                        batch_index=index,
                        cause=e,
                    ) from e

        return results

    def _execute_parallel(
        self,
        items: list[Any],
        operation: Callable[[Any], TResult],
        config: BatchConfig,
    ) -> list[BatchItemResult[TResult]]:
        """Execute batches in parallel."""
        results: list[BatchItemResult[TResult]] = [None] * len(items)  # type: ignore
        total_batches = len(items)
        failed = False
        failed_lock = threading.Lock()

        def process_batch(index: int, item: Any) -> BatchItemResult[TResult]:
            nonlocal failed
            with failed_lock:
                if failed and config.fail_fast:
                    return BatchItemResult(
                        index=index,
                        error=BatchExecutionError("Skipped due to fail_fast"),
                        execution_time_ms=0.0,
                    )

            self._notify_batch_start(index, total_batches, {})
            start_time = time.perf_counter()

            try:
                result = operation(item)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_complete(index, result, execution_time_ms)
                return BatchItemResult(
                    index=index,
                    result=result,
                    execution_time_ms=execution_time_ms,
                )
            except Exception as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_error(index, e, execution_time_ms)

                with failed_lock:
                    failed = True

                return BatchItemResult(
                    index=index,
                    error=e,
                    execution_time_ms=execution_time_ms,
                )

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(process_batch, i, item): i
                for i, item in enumerate(items)
            }

            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()

                if config.fail_fast:
                    with failed_lock:
                        if failed:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break

        # Fill in skipped results
        for i, r in enumerate(results):
            if r is None:
                results[i] = BatchItemResult(
                    index=i,
                    error=BatchExecutionError("Skipped"),
                    execution_time_ms=0.0,
                )

        return results


# =============================================================================
# Async Batch Executor
# =============================================================================


class AsyncBatchExecutor:
    """Async executor for batch data quality operations.

    Provides async versions of check_batch(), profile_batch(), and learn_batch()
    for use in async contexts.

    Example:
        >>> async_engine = SyncEngineAsyncAdapter(TruthoundEngine())
        >>> executor = AsyncBatchExecutor(async_engine)
        >>>
        >>> results = await executor.check_batch(
        ...     datasets=[df1, df2, df3],
        ...     config=BatchConfig(parallel=True),
        ... )
    """

    def __init__(
        self,
        engine: AsyncDataQualityEngine,
        *,
        default_config: BatchConfig | None = None,
        hooks: Sequence[BatchHook] | None = None,
    ) -> None:
        """Initialize AsyncBatchExecutor.

        Args:
            engine: Async data quality engine to use.
            default_config: Default batch configuration.
            hooks: Batch operation hooks.
        """
        self._engine = engine
        self._default_config = default_config or DEFAULT_BATCH_CONFIG
        self._hooks = list(hooks) if hooks else []
        self._check_aggregator = CheckResultAggregator()
        self._profile_aggregator = ProfileResultAggregator()
        self._learn_aggregator = LearnResultAggregator()
        self._drift_aggregator: ResultAggregator[Any] | None = None
        self._anomaly_aggregator: ResultAggregator[Any] | None = None
        self._chunker = PolarsChunker()

    def _get_drift_aggregator(self) -> ResultAggregator[Any]:
        """Lazily initialize and return drift result aggregator."""
        if self._drift_aggregator is None:
            self._drift_aggregator = DriftResultBatchAggregator()
        return self._drift_aggregator

    def _get_anomaly_aggregator(self) -> ResultAggregator[Any]:
        """Lazily initialize and return anomaly result aggregator."""
        if self._anomaly_aggregator is None:
            self._anomaly_aggregator = AnomalyResultBatchAggregator()
        return self._anomaly_aggregator

    @property
    def engine(self) -> AsyncDataQualityEngine:
        """Return the underlying engine."""
        return self._engine

    def add_hook(self, hook: BatchHook) -> None:
        """Add a batch operation hook."""
        self._hooks.append(hook)

    def _notify_batch_start(
        self,
        batch_index: int,
        total_batches: int,
        metadata: dict[str, Any],
    ) -> None:
        """Notify hooks of batch start."""
        for hook in self._hooks:
            hook.on_batch_start(batch_index, total_batches, metadata)

    def _notify_batch_complete(
        self,
        batch_index: int,
        result: Any,
        execution_time_ms: float,
    ) -> None:
        """Notify hooks of batch completion."""
        for hook in self._hooks:
            hook.on_batch_complete(batch_index, result, execution_time_ms)

    def _notify_batch_error(
        self,
        batch_index: int,
        error: Exception,
        execution_time_ms: float,
    ) -> None:
        """Notify hooks of batch error."""
        for hook in self._hooks:
            hook.on_batch_error(batch_index, error, execution_time_ms)

    def _notify_all_complete(
        self,
        total_batches: int,
        successful: int,
        failed: int,
        total_time_ms: float,
    ) -> None:
        """Notify hooks of overall completion."""
        for hook in self._hooks:
            hook.on_all_complete(total_batches, successful, failed, total_time_ms)

    def _should_use_parallel(
        self,
        batch_count: int,
        config: BatchConfig,
    ) -> bool:
        """Determine if parallel execution should be used."""
        if config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            return False
        if config.execution_strategy == ExecutionStrategy.PARALLEL:
            return True
        return batch_count >= 2

    async def check_batch(
        self,
        datasets: Sequence[Any],
        rules: Sequence[Mapping[str, Any]] | None = None,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[CheckResult]:
        """Execute check operation on multiple datasets asynchronously.

        Args:
            datasets: Sequence of datasets to check.
            rules: Validation rules to apply.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.check().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config
        rules = rules or []

        async def check_one(data: Any) -> CheckResult:
            return await self._engine.check(data, list(rules), **kwargs)

        return await self._execute_batch(
            items=list(datasets),
            operation=check_one,
            aggregator=self._check_aggregator,
            config=config,
            operation_name="check",
        )

    async def check_chunked(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] | None = None,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[CheckResult]:
        """Execute check operation on a large dataset by chunking asynchronously.

        Args:
            data: Large dataset to chunk and check.
            rules: Validation rules to apply.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.check().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config
        rules = rules or []
        chunks = list(self._chunker.chunk(data, config))

        async def check_one(chunk: Any) -> CheckResult:
            return await self._engine.check(chunk, list(rules), **kwargs)

        return await self._execute_batch(
            items=chunks,
            operation=check_one,
            aggregator=self._check_aggregator,
            config=config,
            operation_name="check_chunked",
        )

    async def profile_batch(
        self,
        datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[ProfileResult]:
        """Execute profile operation on multiple datasets asynchronously.

        Args:
            datasets: Sequence of datasets to profile.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.profile().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config

        async def profile_one(data: Any) -> ProfileResult:
            return await self._engine.profile(data, **kwargs)

        return await self._execute_batch(
            items=list(datasets),
            operation=profile_one,
            aggregator=self._profile_aggregator,
            config=config,
            operation_name="profile",
        )

    async def profile_chunked(
        self,
        data: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[ProfileResult]:
        """Execute profile operation on a large dataset by chunking asynchronously."""
        config = config or self._default_config
        chunks = list(self._chunker.chunk(data, config))

        async def profile_one(chunk: Any) -> ProfileResult:
            return await self._engine.profile(chunk, **kwargs)

        return await self._execute_batch(
            items=chunks,
            operation=profile_one,
            aggregator=self._profile_aggregator,
            config=config,
            operation_name="profile_chunked",
        )

    async def learn_batch(
        self,
        datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[LearnResult]:
        """Execute learn operation on multiple datasets asynchronously.

        Args:
            datasets: Sequence of datasets to learn from.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.learn().

        Returns:
            BatchResult containing individual and aggregated results.
        """
        config = config or self._default_config

        async def learn_one(data: Any) -> LearnResult:
            return await self._engine.learn(data, **kwargs)

        return await self._execute_batch(
            items=list(datasets),
            operation=learn_one,
            aggregator=self._learn_aggregator,
            config=config,
            operation_name="learn",
        )

    async def learn_chunked(
        self,
        data: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[LearnResult]:
        """Execute learn operation on a large dataset by chunking asynchronously."""
        config = config or self._default_config
        chunks = list(self._chunker.chunk(data, config))

        async def learn_one(chunk: Any) -> LearnResult:
            return await self._engine.learn(chunk, **kwargs)

        return await self._execute_batch(
            items=chunks,
            operation=learn_one,
            aggregator=self._learn_aggregator,
            config=config,
            operation_name="learn_chunked",
        )

    async def drift_batch(
        self,
        baseline_datasets: Sequence[Any],
        current_datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[DriftResult]:
        """Execute drift detection on multiple dataset pairs asynchronously.

        Each baseline/current pair is processed as a batch item.
        The engine must implement DriftDetectionEngine.

        Args:
            baseline_datasets: Sequence of baseline datasets.
            current_datasets: Sequence of current datasets (same length).
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_drift().

        Returns:
            BatchResult containing individual and aggregated drift results.

        Raises:
            BatchOperationError: If engine doesn't support drift detection.
            ValueError: If baseline and current datasets have different lengths.
        """
        from common.engines.base import supports_drift

        if not supports_drift(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support drift detection"
            )
        if len(baseline_datasets) != len(current_datasets):
            msg = (
                f"baseline_datasets ({len(baseline_datasets)}) and "
                f"current_datasets ({len(current_datasets)}) must have the same length"
            )
            raise ValueError(msg)

        config = config or self._default_config
        pairs = list(zip(baseline_datasets, current_datasets))

        async def drift_one(pair: tuple[Any, Any]) -> DriftResult:
            return await self._engine.detect_drift(pair[0], pair[1], **kwargs)

        return await self._execute_batch(
            items=pairs,
            operation=drift_one,
            aggregator=self._get_drift_aggregator(),
            config=config,
            operation_name="drift",
        )

    async def drift_chunked(
        self,
        baseline: Any,
        current: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[DriftResult]:
        """Execute drift detection on large datasets by chunking asynchronously.

        Both baseline and current are chunked in parallel, and drift is
        detected per chunk pair.

        Args:
            baseline: Large baseline dataset.
            current: Large current dataset.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_drift().

        Returns:
            BatchResult containing individual and aggregated drift results.

        Raises:
            BatchOperationError: If engine doesn't support drift detection.
        """
        from common.engines.base import supports_drift

        if not supports_drift(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support drift detection"
            )

        config = config or self._default_config
        baseline_chunks = list(self._chunker.chunk(baseline, config))
        current_chunks = list(self._chunker.chunk(current, config))

        # Align chunks: use min length
        chunk_count = min(len(baseline_chunks), len(current_chunks))
        pairs = [(baseline_chunks[i], current_chunks[i]) for i in range(chunk_count)]

        async def drift_one(pair: tuple[Any, Any]) -> DriftResult:
            return await self._engine.detect_drift(pair[0], pair[1], **kwargs)

        return await self._execute_batch(
            items=pairs,
            operation=drift_one,
            aggregator=self._get_drift_aggregator(),
            config=config,
            operation_name="drift_chunked",
        )

    async def anomaly_batch(
        self,
        datasets: Sequence[Any],
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[AnomalyResult]:
        """Execute anomaly detection on multiple datasets asynchronously.

        The engine must implement AnomalyDetectionEngine.

        Args:
            datasets: Sequence of datasets to check for anomalies.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_anomalies().

        Returns:
            BatchResult containing individual and aggregated anomaly results.

        Raises:
            BatchOperationError: If engine doesn't support anomaly detection.
        """
        from common.engines.base import supports_anomaly

        if not supports_anomaly(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support anomaly detection"
            )

        config = config or self._default_config

        async def anomaly_one(data: Any) -> AnomalyResult:
            return await self._engine.detect_anomalies(data, **kwargs)

        return await self._execute_batch(
            items=list(datasets),
            operation=anomaly_one,
            aggregator=self._get_anomaly_aggregator(),
            config=config,
            operation_name="anomaly",
        )

    async def anomaly_chunked(
        self,
        data: Any,
        *,
        config: BatchConfig | None = None,
        **kwargs: Any,
    ) -> BatchResult[AnomalyResult]:
        """Execute anomaly detection on a large dataset by chunking asynchronously.

        Args:
            data: Large dataset to chunk and check for anomalies.
            config: Batch configuration.
            **kwargs: Additional arguments passed to engine.detect_anomalies().

        Returns:
            BatchResult containing individual and aggregated anomaly results.

        Raises:
            BatchOperationError: If engine doesn't support anomaly detection.
        """
        from common.engines.base import supports_anomaly

        if not supports_anomaly(self._engine):
            raise BatchOperationError(
                f"Engine '{self._engine.engine_name}' does not support anomaly detection"
            )

        config = config or self._default_config
        chunks = list(self._chunker.chunk(data, config))

        async def anomaly_one(chunk: Any) -> AnomalyResult:
            return await self._engine.detect_anomalies(chunk, **kwargs)

        return await self._execute_batch(
            items=chunks,
            operation=anomaly_one,
            aggregator=self._get_anomaly_aggregator(),
            config=config,
            operation_name="anomaly_chunked",
        )

    async def _execute_batch(
        self,
        items: list[Any],
        operation: Callable[[Any], Any],
        aggregator: ResultAggregator[TResult],
        config: BatchConfig,
        operation_name: str,
    ) -> BatchResult[TResult]:
        """Execute batch operation asynchronously."""
        total_batches = len(items)
        if total_batches == 0:
            return BatchResult(
                total_batches=0,
                successful_batches=0,
                failed_batches=0,
            )

        start_time = time.perf_counter()
        use_parallel = self._should_use_parallel(total_batches, config)

        if use_parallel:
            batch_results = await self._execute_parallel(
                items=items,
                operation=operation,
                config=config,
            )
        else:
            batch_results = await self._execute_sequential(
                items=items,
                operation=operation,
                config=config,
            )

        total_time_ms = (time.perf_counter() - start_time) * 1000

        successful = sum(1 for r in batch_results if r.is_success)
        failed = sum(1 for r in batch_results if r.is_failure)

        self._notify_all_complete(total_batches, successful, failed, total_time_ms)

        successful_results = [r.result for r in batch_results if r.result is not None]
        aggregated = (
            aggregator.aggregate(successful_results, config) if successful_results else None
        )

        return BatchResult(
            aggregated_result=aggregated,
            batch_results=tuple(batch_results),
            total_batches=total_batches,
            successful_batches=successful,
            failed_batches=failed,
            total_execution_time_ms=total_time_ms,
            metadata={
                "operation": operation_name,
                "execution_strategy": "parallel" if use_parallel else "sequential",
                "config": config.to_dict(),
            },
        )

    async def _execute_sequential(
        self,
        items: list[Any],
        operation: Callable[[Any], Any],
        config: BatchConfig,
    ) -> list[BatchItemResult[TResult]]:
        """Execute batches sequentially."""
        results: list[BatchItemResult[TResult]] = []
        total_batches = len(items)

        for index, item in enumerate(items):
            self._notify_batch_start(index, total_batches, {})
            start_time = time.perf_counter()

            try:
                result = await operation(item)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_complete(index, result, execution_time_ms)
                results.append(
                    BatchItemResult(
                        index=index,
                        result=result,
                        execution_time_ms=execution_time_ms,
                    )
                )
            except Exception as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_error(index, e, execution_time_ms)

                if config.fail_fast:
                    results.append(
                        BatchItemResult(
                            index=index,
                            error=e,
                            execution_time_ms=execution_time_ms,
                        )
                    )
                    break

                if config.continue_on_error:
                    results.append(
                        BatchItemResult(
                            index=index,
                            error=e,
                            execution_time_ms=execution_time_ms,
                        )
                    )
                else:
                    raise BatchExecutionError(
                        f"Batch {index} failed: {e}",
                        batch_index=index,
                        cause=e,
                    ) from e

        return results

    async def _execute_parallel(
        self,
        items: list[Any],
        operation: Callable[[Any], Any],
        config: BatchConfig,
    ) -> list[BatchItemResult[TResult]]:
        """Execute batches in parallel using asyncio."""
        total_batches = len(items)

        async def process_batch(index: int, item: Any) -> BatchItemResult[TResult]:
            self._notify_batch_start(index, total_batches, {})
            start_time = time.perf_counter()

            try:
                result = await operation(item)
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_complete(index, result, execution_time_ms)
                return BatchItemResult(
                    index=index,
                    result=result,
                    execution_time_ms=execution_time_ms,
                )
            except Exception as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                self._notify_batch_error(index, e, execution_time_ms)
                return BatchItemResult(
                    index=index,
                    error=e,
                    execution_time_ms=execution_time_ms,
                )

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(config.max_workers)

        async def limited_process(index: int, item: Any) -> BatchItemResult[TResult]:
            async with semaphore:
                return await process_batch(index, item)

        tasks = [limited_process(i, item) for i, item in enumerate(items)]
        results = await asyncio.gather(*tasks)

        return list(results)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_batch_executor(
    engine: DataQualityEngine,
    *,
    config: BatchConfig | None = None,
    enable_logging: bool = False,
    enable_metrics: bool = False,
) -> BatchExecutor:
    """Create a BatchExecutor with optional hooks.

    Args:
        engine: Data quality engine to use.
        config: Default batch configuration.
        enable_logging: Enable logging hook.
        enable_metrics: Enable metrics hook.

    Returns:
        Configured BatchExecutor.
    """
    hooks: list[BatchHook] = []
    if enable_logging:
        hooks.append(LoggingBatchHook())
    if enable_metrics:
        hooks.append(MetricsBatchHook())

    return BatchExecutor(engine, default_config=config, hooks=hooks)


def create_async_batch_executor(
    engine: AsyncDataQualityEngine,
    *,
    config: BatchConfig | None = None,
    enable_logging: bool = False,
    enable_metrics: bool = False,
) -> AsyncBatchExecutor:
    """Create an AsyncBatchExecutor with optional hooks.

    Args:
        engine: Async data quality engine to use.
        config: Default batch configuration.
        enable_logging: Enable logging hook.
        enable_metrics: Enable metrics hook.

    Returns:
        Configured AsyncBatchExecutor.
    """
    hooks: list[BatchHook] = []
    if enable_logging:
        hooks.append(LoggingBatchHook())
    if enable_metrics:
        hooks.append(MetricsBatchHook())

    return AsyncBatchExecutor(engine, default_config=config, hooks=hooks)
