"""Result Aggregation module for multi-engine data quality operations.

This module provides a comprehensive system for aggregating results from
multiple data quality engines or batch operations. It supports various
aggregation strategies for CheckResult, ProfileResult, and LearnResult types.

Key Features:
    - Multiple aggregation strategies (MERGE, WORST, BEST, MAJORITY, etc.)
    - Protocol-based extensibility for custom aggregators
    - Multi-engine result comparison and combination
    - Weighted aggregation support
    - Hook system for observability
    - Thread-safe registry for aggregator management

Quick Start:
    >>> from common.engines import (
    ...     MultiEngineAggregator,
    ...     AggregationConfig,
    ...     CheckResultAggregator,
    ... )
    >>> aggregator = MultiEngineAggregator()
    >>> combined = aggregator.aggregate_check_results({
    ...     "truthound": result1,
    ...     "ge": result2,
    ...     "pandera": result3,
    ... })

Weighted Aggregation:
    >>> config = AggregationConfig(
    ...     strategy=ResultAggregationStrategy.WEIGHTED,
    ...     weights={"truthound": 2.0, "ge": 1.0, "pandera": 1.0},
    ... )
    >>> aggregator = MultiEngineAggregator(config=config)
    >>> combined = aggregator.aggregate_check_results(engine_results)

Custom Aggregation:
    >>> class CustomAggregator(BaseResultAggregator[CheckResult]):
    ...     def aggregate(self, results, config):
    ...         # Custom logic
    ...         ...
    >>> register_aggregator("custom", CustomAggregator())
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)


if TYPE_CHECKING:
    pass


# =============================================================================
# Type Variables
# =============================================================================

TResult = TypeVar("TResult")
TResult_co = TypeVar("TResult_co", covariant=True)


# =============================================================================
# Enums
# =============================================================================


class ResultAggregationStrategy(Enum):
    """Strategy for aggregating multiple results.

    Attributes:
        MERGE: Combine all results into a single merged result.
        WORST: Return the result with the worst status.
        BEST: Return the result with the best status.
        MAJORITY: Return based on the most common status.
        FIRST_FAILURE: Return the first failed result, or last if all pass.
        ALL: Keep all individual results without aggregation.
        WEIGHTED: Use weighted scoring based on engine weights.
        CONSENSUS: Require consensus across engines for each check.
        STRICT_ALL: All must pass for overall pass.
        LENIENT_ANY: Any pass results in overall pass.
    """

    MERGE = auto()
    WORST = auto()
    BEST = auto()
    MAJORITY = auto()
    FIRST_FAILURE = auto()
    ALL = auto()
    WEIGHTED = auto()
    CONSENSUS = auto()
    STRICT_ALL = auto()
    LENIENT_ANY = auto()


class ConflictResolution(Enum):
    """Strategy for resolving conflicts between engine results.

    Attributes:
        PREFER_FAILURE: In case of conflict, prefer the failure result.
        PREFER_SUCCESS: In case of conflict, prefer the success result.
        PREFER_PRIMARY: Prefer the primary/first engine's result.
        WEIGHTED_VOTE: Use weighted voting to resolve conflicts.
        RAISE_ERROR: Raise an error when conflict is detected.
    """

    PREFER_FAILURE = auto()
    PREFER_SUCCESS = auto()
    PREFER_PRIMARY = auto()
    WEIGHTED_VOTE = auto()
    RAISE_ERROR = auto()


class StatusPriority(Enum):
    """Priority ordering for result statuses.

    Lower numeric value = higher priority (worse status).
    """

    ERROR = 0
    FAILED = 1
    WARNING = 2
    SKIPPED = 3
    PASSED = 4

    @classmethod
    def from_check_status(cls, status: Any) -> StatusPriority:
        """Convert CheckStatus to StatusPriority."""
        name = status.name if hasattr(status, "name") else str(status)
        return cls[name] if name in cls.__members__ else cls.PASSED

    @classmethod
    def from_drift_status(cls, status: Any) -> StatusPriority:
        """Convert DriftStatus to StatusPriority."""
        name = status.name if hasattr(status, "name") else str(status)
        mapping = {
            "ERROR": cls.ERROR,
            "DRIFT_DETECTED": cls.FAILED,
            "WARNING": cls.WARNING,
            "NO_DRIFT": cls.PASSED,
        }
        return mapping.get(name, cls.PASSED)

    @classmethod
    def from_anomaly_status(cls, status: Any) -> StatusPriority:
        """Convert AnomalyStatus to StatusPriority."""
        name = status.name if hasattr(status, "name") else str(status)
        mapping = {
            "ERROR": cls.ERROR,
            "ANOMALY_DETECTED": cls.FAILED,
            "WARNING": cls.WARNING,
            "NORMAL": cls.PASSED,
        }
        return mapping.get(name, cls.PASSED)

    @classmethod
    def from_profile_status(cls, status: Any) -> StatusPriority:
        """Convert ProfileStatus to StatusPriority."""
        name = status.name if hasattr(status, "name") else str(status)
        mapping = {
            "COMPLETED": cls.PASSED,
            "PARTIAL": cls.WARNING,
            "FAILED": cls.FAILED,
        }
        return mapping.get(name, cls.PASSED)

    @classmethod
    def from_learn_status(cls, status: Any) -> StatusPriority:
        """Convert LearnStatus to StatusPriority."""
        name = status.name if hasattr(status, "name") else str(status)
        mapping = {
            "COMPLETED": cls.PASSED,
            "PARTIAL": cls.WARNING,
            "FAILED": cls.FAILED,
        }
        return mapping.get(name, cls.PASSED)


# =============================================================================
# Exceptions
# =============================================================================


class AggregationError(Exception):
    """Base exception for aggregation operations."""

    pass


class NoResultsError(AggregationError):
    """Raised when no results are provided for aggregation."""

    def __init__(self, message: str = "No results provided for aggregation") -> None:
        super().__init__(message)


class ConflictError(AggregationError):
    """Raised when conflicting results cannot be resolved."""

    def __init__(
        self,
        message: str,
        conflicts: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.conflicts = conflicts or {}


class AggregatorNotFoundError(AggregationError):
    """Raised when requested aggregator is not found in registry."""

    def __init__(self, aggregator_name: str) -> None:
        super().__init__(f"Aggregator '{aggregator_name}' not found in registry")
        self.aggregator_name = aggregator_name


class InvalidWeightError(AggregationError):
    """Raised when invalid weights are provided."""

    def __init__(self, message: str, weights: Mapping[str, float] | None = None) -> None:
        super().__init__(message)
        self.weights = weights or {}


# =============================================================================
# Configuration
# =============================================================================


def _utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class AggregationConfig:
    """Configuration for result aggregation.

    Attributes:
        strategy: The aggregation strategy to use.
        conflict_resolution: How to resolve conflicts between results.
        weights: Engine weights for weighted strategies (engine_name -> weight).
        primary_engine: Primary engine for PREFER_PRIMARY conflict resolution.
        include_metadata: Whether to include detailed metadata in aggregated result.
        preserve_individual_results: Keep individual results in metadata.
        consensus_threshold: Minimum ratio for consensus (0.0 to 1.0).
        fail_on_conflict: Raise error on unresolvable conflicts.
        merge_failures: Whether to merge all failures or keep separate.
        deduplicate_failures: Remove duplicate failures when merging.
        extra: Additional configuration parameters.

    Example:
        >>> config = AggregationConfig(
        ...     strategy=ResultAggregationStrategy.WEIGHTED,
        ...     weights={"truthound": 2.0, "ge": 1.0},
        ... )
    """

    strategy: ResultAggregationStrategy = ResultAggregationStrategy.MERGE
    conflict_resolution: ConflictResolution = ConflictResolution.PREFER_FAILURE
    weights: Mapping[str, float] = field(default_factory=dict)
    primary_engine: str | None = None
    include_metadata: bool = True
    preserve_individual_results: bool = False
    consensus_threshold: float = 0.5
    fail_on_conflict: bool = False
    merge_failures: bool = True
    deduplicate_failures: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.consensus_threshold <= 1.0:
            msg = f"consensus_threshold must be between 0.0 and 1.0, got {self.consensus_threshold}"
            raise ValueError(msg)

        if self.strategy == ResultAggregationStrategy.WEIGHTED and not self.weights:
            msg = "weights must be provided for WEIGHTED strategy"
            raise ValueError(msg)

    def with_strategy(self, strategy: ResultAggregationStrategy) -> AggregationConfig:
        """Create new config with different strategy."""
        return AggregationConfig(
            strategy=strategy,
            conflict_resolution=self.conflict_resolution,
            weights=self.weights,
            primary_engine=self.primary_engine,
            include_metadata=self.include_metadata,
            preserve_individual_results=self.preserve_individual_results,
            consensus_threshold=self.consensus_threshold,
            fail_on_conflict=self.fail_on_conflict,
            merge_failures=self.merge_failures,
            deduplicate_failures=self.deduplicate_failures,
            extra=self.extra,
        )

    def with_weights(self, weights: Mapping[str, float]) -> AggregationConfig:
        """Create new config with different weights."""
        return AggregationConfig(
            strategy=self.strategy,
            conflict_resolution=self.conflict_resolution,
            weights=weights,
            primary_engine=self.primary_engine,
            include_metadata=self.include_metadata,
            preserve_individual_results=self.preserve_individual_results,
            consensus_threshold=self.consensus_threshold,
            fail_on_conflict=self.fail_on_conflict,
            merge_failures=self.merge_failures,
            deduplicate_failures=self.deduplicate_failures,
            extra=self.extra,
        )

    def with_conflict_resolution(
        self, resolution: ConflictResolution
    ) -> AggregationConfig:
        """Create new config with different conflict resolution."""
        return AggregationConfig(
            strategy=self.strategy,
            conflict_resolution=resolution,
            weights=self.weights,
            primary_engine=self.primary_engine,
            include_metadata=self.include_metadata,
            preserve_individual_results=self.preserve_individual_results,
            consensus_threshold=self.consensus_threshold,
            fail_on_conflict=self.fail_on_conflict,
            merge_failures=self.merge_failures,
            deduplicate_failures=self.deduplicate_failures,
            extra=self.extra,
        )

    def with_primary_engine(self, engine: str) -> AggregationConfig:
        """Create new config with different primary engine."""
        return AggregationConfig(
            strategy=self.strategy,
            conflict_resolution=self.conflict_resolution,
            weights=self.weights,
            primary_engine=engine,
            include_metadata=self.include_metadata,
            preserve_individual_results=self.preserve_individual_results,
            consensus_threshold=self.consensus_threshold,
            fail_on_conflict=self.fail_on_conflict,
            merge_failures=self.merge_failures,
            deduplicate_failures=self.deduplicate_failures,
            extra=self.extra,
        )

    def with_consensus_threshold(self, threshold: float) -> AggregationConfig:
        """Create new config with different consensus threshold."""
        return AggregationConfig(
            strategy=self.strategy,
            conflict_resolution=self.conflict_resolution,
            weights=self.weights,
            primary_engine=self.primary_engine,
            include_metadata=self.include_metadata,
            preserve_individual_results=self.preserve_individual_results,
            consensus_threshold=threshold,
            fail_on_conflict=self.fail_on_conflict,
            merge_failures=self.merge_failures,
            deduplicate_failures=self.deduplicate_failures,
            extra=self.extra,
        )

    def with_preserve_individual_results(self, preserve: bool) -> AggregationConfig:
        """Create new config with different preserve_individual_results setting."""
        return AggregationConfig(
            strategy=self.strategy,
            conflict_resolution=self.conflict_resolution,
            weights=self.weights,
            primary_engine=self.primary_engine,
            include_metadata=self.include_metadata,
            preserve_individual_results=preserve,
            consensus_threshold=self.consensus_threshold,
            fail_on_conflict=self.fail_on_conflict,
            merge_failures=self.merge_failures,
            deduplicate_failures=self.deduplicate_failures,
            extra=self.extra,
        )

    def with_extra(self, **kwargs: Any) -> AggregationConfig:
        """Create new config with additional extra parameters."""
        new_extra = dict(self.extra)
        new_extra.update(kwargs)
        return AggregationConfig(
            strategy=self.strategy,
            conflict_resolution=self.conflict_resolution,
            weights=self.weights,
            primary_engine=self.primary_engine,
            include_metadata=self.include_metadata,
            preserve_individual_results=self.preserve_individual_results,
            consensus_threshold=self.consensus_threshold,
            fail_on_conflict=self.fail_on_conflict,
            merge_failures=self.merge_failures,
            deduplicate_failures=self.deduplicate_failures,
            extra=new_extra,
        )


# Preset configurations
DEFAULT_AGGREGATION_CONFIG = AggregationConfig()

STRICT_AGGREGATION_CONFIG = AggregationConfig(
    strategy=ResultAggregationStrategy.STRICT_ALL,
    conflict_resolution=ConflictResolution.PREFER_FAILURE,
    fail_on_conflict=True,
)

LENIENT_AGGREGATION_CONFIG = AggregationConfig(
    strategy=ResultAggregationStrategy.LENIENT_ANY,
    conflict_resolution=ConflictResolution.PREFER_SUCCESS,
)

CONSENSUS_AGGREGATION_CONFIG = AggregationConfig(
    strategy=ResultAggregationStrategy.CONSENSUS,
    consensus_threshold=0.6,
)

WORST_CASE_AGGREGATION_CONFIG = AggregationConfig(
    strategy=ResultAggregationStrategy.WORST,
    conflict_resolution=ConflictResolution.PREFER_FAILURE,
)


# =============================================================================
# Result Data Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class EngineResultEntry(Generic[TResult]):
    """Container for an engine's result with metadata.

    Attributes:
        engine_name: Name of the engine that produced the result.
        result: The result object.
        weight: Weight for weighted aggregation (default 1.0).
        timestamp: When the result was produced.
        metadata: Additional metadata.
    """

    engine_name: str
    result: TResult
    weight: float = 1.0
    timestamp: str = field(default_factory=_utc_now_iso)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AggregatedResult(Generic[TResult]):
    """Container for an aggregated result with provenance.

    Attributes:
        result: The aggregated result.
        strategy: Strategy used for aggregation.
        source_engines: Names of engines that contributed.
        source_count: Number of source results.
        conflict_count: Number of conflicts encountered.
        aggregation_time_ms: Time taken for aggregation.
        individual_results: Original results if preserved.
        metadata: Additional metadata including provenance.
    """

    result: TResult
    strategy: ResultAggregationStrategy
    source_engines: tuple[str, ...]
    source_count: int
    conflict_count: int = 0
    aggregation_time_ms: float = 0.0
    individual_results: Mapping[str, TResult] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_conflicts(self) -> bool:
        """Check if conflicts were encountered during aggregation."""
        return self.conflict_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "result": self.result.to_dict() if hasattr(self.result, "to_dict") else self.result,
            "strategy": self.strategy.name,
            "source_engines": list(self.source_engines),
            "source_count": self.source_count,
            "conflict_count": self.conflict_count,
            "aggregation_time_ms": self.aggregation_time_ms,
            "individual_results": (
                {
                    k: v.to_dict() if hasattr(v, "to_dict") else v
                    for k, v in self.individual_results.items()
                }
                if self.individual_results
                else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Result of comparing multiple engine results.

    Attributes:
        agreement_ratio: Ratio of engines that agree (0.0 to 1.0).
        unanimous: Whether all engines agree.
        majority_status: The status held by majority of engines.
        engine_statuses: Status from each engine.
        discrepancies: Details of any discrepancies found.
        metadata: Additional comparison metadata.
    """

    agreement_ratio: float
    unanimous: bool
    majority_status: str
    engine_statuses: Mapping[str, str]
    discrepancies: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_discrepancies(self) -> bool:
        """Check if any discrepancies exist."""
        return len(self.discrepancies) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agreement_ratio": self.agreement_ratio,
            "unanimous": self.unanimous,
            "majority_status": self.majority_status,
            "engine_statuses": dict(self.engine_statuses),
            "discrepancies": [dict(d) for d in self.discrepancies],
            "metadata": dict(self.metadata),
        }


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class ResultAggregator(Protocol[TResult_co]):
    """Protocol for result aggregation implementations.

    Implementations should combine multiple results into a single result
    based on the configured aggregation strategy.
    """

    def aggregate(
        self,
        results: Sequence[TResult_co],
        config: AggregationConfig,
    ) -> TResult_co:
        """Aggregate multiple results.

        Args:
            results: Results to aggregate.
            config: Aggregation configuration.

        Returns:
            Aggregated result.

        Raises:
            NoResultsError: If no results are provided.
            AggregationError: If aggregation fails.
        """
        ...


@runtime_checkable
class EngineResultAggregator(Protocol[TResult_co]):
    """Protocol for aggregating results from multiple engines.

    This protocol extends basic aggregation with engine-aware features
    like weighted aggregation and conflict resolution.
    """

    def aggregate_engine_results(
        self,
        engine_results: Mapping[str, TResult_co],
        config: AggregationConfig,
    ) -> AggregatedResult[TResult_co]:
        """Aggregate results from multiple engines.

        Args:
            engine_results: Mapping of engine name to result.
            config: Aggregation configuration.

        Returns:
            Aggregated result with provenance.

        Raises:
            NoResultsError: If no results are provided.
            ConflictError: If conflicts cannot be resolved.
            AggregationError: If aggregation fails.
        """
        ...


@runtime_checkable
class ResultComparator(Protocol[TResult_co]):
    """Protocol for comparing results from multiple engines."""

    def compare(
        self,
        engine_results: Mapping[str, TResult_co],
    ) -> ComparisonResult:
        """Compare results from multiple engines.

        Args:
            engine_results: Mapping of engine name to result.

        Returns:
            Comparison result with agreement metrics.
        """
        ...


@runtime_checkable
class AggregationHook(Protocol):
    """Protocol for aggregation lifecycle hooks.

    Implementations can observe and react to aggregation events.
    """

    def on_aggregation_start(
        self,
        result_count: int,
        strategy: ResultAggregationStrategy,
        metadata: Mapping[str, Any],
    ) -> None:
        """Called when aggregation starts."""
        ...

    def on_aggregation_complete(
        self,
        result: Any,
        source_count: int,
        conflict_count: int,
        duration_ms: float,
    ) -> None:
        """Called when aggregation completes successfully."""
        ...

    def on_conflict_detected(
        self,
        conflict_type: str,
        engines: Sequence[str],
        details: Mapping[str, Any],
    ) -> None:
        """Called when a conflict is detected."""
        ...

    def on_aggregation_error(
        self,
        error: Exception,
        result_count: int,
    ) -> None:
        """Called when aggregation fails."""
        ...


# =============================================================================
# Base Aggregator Implementation
# =============================================================================


class BaseResultAggregator(ABC, Generic[TResult]):
    """Abstract base class for result aggregators.

    Provides common functionality and template method pattern for aggregation.
    Subclasses must implement the `_aggregate_merge` method at minimum.
    """

    def __init__(
        self,
        hooks: Sequence[AggregationHook] | None = None,
    ) -> None:
        """Initialize aggregator.

        Args:
            hooks: Optional hooks for observability.
        """
        self._hooks = list(hooks) if hooks else []

    def add_hook(self, hook: AggregationHook) -> None:
        """Add a hook to the aggregator."""
        self._hooks.append(hook)

    def remove_hook(self, hook: AggregationHook) -> None:
        """Remove a hook from the aggregator."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def aggregate(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Aggregate multiple results based on strategy.

        Args:
            results: Results to aggregate.
            config: Aggregation configuration.

        Returns:
            Aggregated result.

        Raises:
            NoResultsError: If no results are provided.
            AggregationError: If aggregation fails.
        """
        if not results:
            raise NoResultsError()

        start_time = datetime.now(timezone.utc)
        self._notify_start(len(results), config.strategy, {})

        try:
            match config.strategy:
                case ResultAggregationStrategy.MERGE:
                    result = self._aggregate_merge(results, config)
                case ResultAggregationStrategy.WORST:
                    result = self._aggregate_worst(results, config)
                case ResultAggregationStrategy.BEST:
                    result = self._aggregate_best(results, config)
                case ResultAggregationStrategy.MAJORITY:
                    result = self._aggregate_majority(results, config)
                case ResultAggregationStrategy.FIRST_FAILURE:
                    result = self._aggregate_first_failure(results, config)
                case ResultAggregationStrategy.STRICT_ALL:
                    result = self._aggregate_strict_all(results, config)
                case ResultAggregationStrategy.LENIENT_ANY:
                    result = self._aggregate_lenient_any(results, config)
                case ResultAggregationStrategy.ALL:
                    # Return the last result for ALL strategy
                    result = results[-1]
                case _:
                    result = self._aggregate_merge(results, config)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._notify_complete(result, len(results), 0, duration)
            return result

        except Exception as e:
            self._notify_error(e, len(results))
            raise

    @abstractmethod
    def _aggregate_merge(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Merge all results into one.

        Must be implemented by subclasses.
        """
        ...

    def _aggregate_worst(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Return result with worst status.

        Default implementation uses status priority. Override for custom behavior.
        """
        return min(results, key=self._get_status_priority)

    def _aggregate_best(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Return result with best status.

        Default implementation uses status priority. Override for custom behavior.
        """
        return max(results, key=self._get_status_priority)

    def _aggregate_majority(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Return result with most common status.

        Default implementation counts statuses and returns first match.
        """
        status_counts = Counter(self._get_status_name(r) for r in results)
        most_common_status = status_counts.most_common(1)[0][0]

        for result in results:
            if self._get_status_name(result) == most_common_status:
                return result
        return results[0]

    def _aggregate_first_failure(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Return first failed result, or last if all pass."""
        for result in results:
            if self._is_failure(result):
                return result
        return results[-1]

    def _aggregate_strict_all(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """All must pass for overall pass.

        Returns merged result with FAILED status if any failed.
        """
        # Check if any result failed
        has_failure = any(self._is_failure(r) for r in results)

        merged = self._aggregate_merge(results, config)

        if has_failure:
            return self._create_failed_result(merged, results, config)
        return merged

    def _aggregate_lenient_any(
        self,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Any pass results in overall pass.

        Returns merged result with PASSED status if any passed.
        """
        has_success = any(self._is_success(r) for r in results)

        merged = self._aggregate_merge(results, config)

        if has_success:
            return self._create_passed_result(merged, results, config)
        return merged

    def _get_status_priority(self, result: TResult) -> int:
        """Get numeric priority for a result's status (lower = worse)."""
        if hasattr(result, "status"):
            return StatusPriority.from_check_status(result.status).value
        return StatusPriority.PASSED.value

    def _get_status_name(self, result: TResult) -> str:
        """Get the status name from a result."""
        if hasattr(result, "status"):
            status = result.status
            return status.name if hasattr(status, "name") else str(status)
        return "UNKNOWN"

    def _is_failure(self, result: TResult) -> bool:
        """Check if a result represents a failure."""
        status_name = self._get_status_name(result)
        return status_name in ("FAILED", "ERROR")

    def _is_success(self, result: TResult) -> bool:
        """Check if a result represents success."""
        return self._get_status_name(result) == "PASSED"

    def _create_failed_result(
        self,
        merged: TResult,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Create a result with failed status from merged data.

        Override in subclasses for type-specific behavior.
        """
        return merged

    def _create_passed_result(
        self,
        merged: TResult,
        results: Sequence[TResult],
        config: AggregationConfig,
    ) -> TResult:
        """Create a result with passed status from merged data.

        Override in subclasses for type-specific behavior.
        """
        return merged

    def _notify_start(
        self,
        count: int,
        strategy: ResultAggregationStrategy,
        metadata: Mapping[str, Any],
    ) -> None:
        """Notify hooks of aggregation start."""
        for hook in self._hooks:
            try:
                hook.on_aggregation_start(count, strategy, metadata)
            except Exception:
                pass  # Hooks should not break aggregation

    def _notify_complete(
        self,
        result: TResult,
        source_count: int,
        conflict_count: int,
        duration_ms: float,
    ) -> None:
        """Notify hooks of aggregation completion."""
        for hook in self._hooks:
            try:
                hook.on_aggregation_complete(result, source_count, conflict_count, duration_ms)
            except Exception:
                pass

    def _notify_conflict(
        self,
        conflict_type: str,
        engines: Sequence[str],
        details: Mapping[str, Any],
    ) -> None:
        """Notify hooks of a conflict."""
        for hook in self._hooks:
            try:
                hook.on_conflict_detected(conflict_type, engines, details)
            except Exception:
                pass

    def _notify_error(self, error: Exception, count: int) -> None:
        """Notify hooks of an error."""
        for hook in self._hooks:
            try:
                hook.on_aggregation_error(error, count)
            except Exception:
                pass


# =============================================================================
# CheckResult Aggregator
# =============================================================================


class CheckResultMergeAggregator(BaseResultAggregator["CheckResult"]):
    """Aggregator for CheckResult objects.

    Supports all aggregation strategies with special handling for:
    - Merging failures from multiple results
    - Deduplicating failures
    - Computing aggregate statistics
    - Weighted status determination
    """

    def _aggregate_merge(
        self,
        results: Sequence[Any],  # CheckResult
        config: AggregationConfig,
    ) -> Any:  # CheckResult
        """Merge all CheckResults into one."""
        from common.base import CheckResult, CheckStatus, ValidationFailure

        total_passed = sum(r.passed_count for r in results)
        total_failed = sum(r.failed_count for r in results)
        total_warning = sum(r.warning_count for r in results)
        total_skipped = sum(r.skipped_count for r in results)
        total_time = sum(r.execution_time_ms for r in results)

        # Collect all failures
        all_failures: list[ValidationFailure] = []
        seen_failures: set[tuple[str, str | None, str]] = set()

        for result in results:
            for failure in result.failures:
                if config.deduplicate_failures:
                    key = (failure.rule_name, failure.column, failure.message)
                    if key in seen_failures:
                        continue
                    seen_failures.add(key)
                all_failures.append(failure)

        # Determine overall status
        if total_failed > 0:
            status = CheckStatus.FAILED
        elif total_warning > 0:
            status = CheckStatus.WARNING
        elif total_passed > 0:
            status = CheckStatus.PASSED
        else:
            status = CheckStatus.SKIPPED

        # Build metadata
        metadata: dict[str, Any] = {
            "aggregation_strategy": config.strategy.name,
            "source_count": len(results),
        }
        if config.include_metadata:
            metadata["source_timestamps"] = [r.timestamp for r in results]

        if config.preserve_individual_results:
            metadata["individual_results"] = [r.to_dict() for r in results]

        return CheckResult(
            status=status,
            passed_count=total_passed,
            failed_count=total_failed,
            warning_count=total_warning,
            skipped_count=total_skipped,
            failures=tuple(all_failures),
            execution_time_ms=total_time,
            metadata=metadata,
        )

    def _create_failed_result(
        self,
        merged: Any,  # CheckResult
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:  # CheckResult
        """Create a failed result from merged data."""
        from common.base import CheckResult, CheckStatus

        return CheckResult(
            status=CheckStatus.FAILED,
            passed_count=merged.passed_count,
            failed_count=merged.failed_count,
            warning_count=merged.warning_count,
            skipped_count=merged.skipped_count,
            failures=merged.failures,
            execution_time_ms=merged.execution_time_ms,
            metadata={
                **merged.metadata,
                "aggregation_override": "STRICT_ALL",
            },
        )

    def _create_passed_result(
        self,
        merged: Any,  # CheckResult
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:  # CheckResult
        """Create a passed result from merged data."""
        from common.base import CheckResult, CheckStatus

        return CheckResult(
            status=CheckStatus.PASSED,
            passed_count=merged.passed_count,
            failed_count=merged.failed_count,
            warning_count=merged.warning_count,
            skipped_count=merged.skipped_count,
            failures=merged.failures,
            execution_time_ms=merged.execution_time_ms,
            metadata={
                **merged.metadata,
                "aggregation_override": "LENIENT_ANY",
            },
        )


class CheckResultWeightedAggregator(BaseResultAggregator["CheckResult"]):
    """Weighted aggregator for CheckResult objects.

    Uses engine weights to compute a weighted score and determine
    the final aggregated status.
    """

    def __init__(
        self,
        hooks: Sequence[AggregationHook] | None = None,
    ) -> None:
        super().__init__(hooks)
        self._merge_aggregator = CheckResultMergeAggregator(hooks)

    def aggregate(
        self,
        results: Sequence[Any],  # CheckResult
        config: AggregationConfig,
    ) -> Any:  # CheckResult
        """Aggregate with weighted scoring."""
        if config.strategy != ResultAggregationStrategy.WEIGHTED:
            return self._merge_aggregator.aggregate(results, config)
        return super().aggregate(results, config)

    def aggregate_with_weights(
        self,
        engine_results: Mapping[str, Any],  # CheckResult
        config: AggregationConfig,
    ) -> Any:  # CheckResult
        """Aggregate engine results using weights.

        Args:
            engine_results: Mapping of engine name to CheckResult.
            config: Aggregation configuration with weights.

        Returns:
            Weighted aggregated CheckResult.
        """
        from common.base import CheckResult, CheckStatus

        if not engine_results:
            raise NoResultsError()

        # Calculate weighted status score
        total_weight = 0.0
        weighted_score = 0.0
        status_scores = {
            "PASSED": 4.0,
            "WARNING": 2.0,
            "SKIPPED": 1.0,
            "FAILED": 0.0,
            "ERROR": 0.0,
        }

        for engine_name, result in engine_results.items():
            weight = config.weights.get(engine_name, 1.0)
            total_weight += weight

            status = result.status
            status_name = status.name if hasattr(status, "name") else str(status)
            score = status_scores.get(status_name, 1.0)
            weighted_score += weight * score

        # Normalize and determine status
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 2.0  # Default to midpoint

        # Map score to status
        if normalized_score >= 3.5:
            final_status = CheckStatus.PASSED
        elif normalized_score >= 1.5:
            final_status = CheckStatus.WARNING
        elif normalized_score >= 0.5:
            final_status = CheckStatus.SKIPPED
        else:
            final_status = CheckStatus.FAILED

        # Merge counts and failures
        results_list = list(engine_results.values())
        merged = self._merge_aggregator._aggregate_merge(results_list, config)

        return CheckResult(
            status=final_status,
            passed_count=merged.passed_count,
            failed_count=merged.failed_count,
            warning_count=merged.warning_count,
            skipped_count=merged.skipped_count,
            failures=merged.failures,
            execution_time_ms=merged.execution_time_ms,
            metadata={
                **merged.metadata,
                "weighted_score": normalized_score,
                "total_weight": total_weight,
                "engine_weights": dict(config.weights),
            },
        )

    def _aggregate_merge(
        self,
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:
        """Fall back to merge aggregator."""
        return self._merge_aggregator._aggregate_merge(results, config)


# =============================================================================
# ProfileResult Aggregator
# =============================================================================


class ProfileResultAggregator(BaseResultAggregator["ProfileResult"]):
    """Aggregator for ProfileResult objects.

    Merges column profiles from multiple results, handling:
    - Column profile merging with statistical aggregation
    - Correlation matrix combination
    - Row/column count aggregation
    """

    def _aggregate_merge(
        self,
        results: Sequence[Any],  # ProfileResult
        config: AggregationConfig,
    ) -> Any:  # ProfileResult
        """Merge all ProfileResults into one."""
        from common.base import ColumnProfile, ProfileResult, ProfileStatus

        # Collect and merge column profiles
        column_profiles_map: dict[str, list[Any]] = {}
        for result in results:
            for col in result.columns:
                if col.column_name not in column_profiles_map:
                    column_profiles_map[col.column_name] = []
                column_profiles_map[col.column_name].append(col)

        # Merge profiles for each column
        merged_columns: list[ColumnProfile] = []
        for col_name, profiles in column_profiles_map.items():
            merged_col = self._merge_column_profiles(col_name, profiles)
            merged_columns.append(merged_col)

        # Determine status
        statuses = [r.status for r in results]
        if ProfileStatus.FAILED in statuses:
            status = ProfileStatus.FAILED
        elif ProfileStatus.PARTIAL in statuses:
            status = ProfileStatus.PARTIAL
        else:
            status = ProfileStatus.COMPLETED

        # Aggregate counts
        total_rows = sum(r.row_count for r in results)
        total_time = sum(r.execution_time_ms for r in results)

        # Merge correlations (take first non-None)
        correlations = None
        for result in results:
            if result.correlations:
                correlations = result.correlations
                break

        # Build metadata
        metadata: dict[str, Any] = {
            "aggregation_strategy": config.strategy.name,
            "source_count": len(results),
        }
        if config.preserve_individual_results:
            metadata["individual_results"] = [r.to_dict() for r in results]

        return ProfileResult(
            status=status,
            row_count=total_rows,
            column_count=len(merged_columns),
            columns=tuple(merged_columns),
            correlations=correlations,
            execution_time_ms=total_time,
            metadata=metadata,
        )

    def _merge_column_profiles(
        self,
        col_name: str,
        profiles: Sequence[Any],  # ColumnProfile
    ) -> Any:  # ColumnProfile
        """Merge multiple profiles for the same column."""
        from common.base import ColumnProfile

        if len(profiles) == 1:
            return profiles[0]

        # Take first profile as base
        base = profiles[0]

        # Aggregate statistics
        total_null_count = sum(p.null_count for p in profiles)
        total_unique = max(p.unique_count for p in profiles)

        # Calculate weighted averages where applicable
        means = [p.mean for p in profiles if p.mean is not None]
        avg_mean = sum(means) / len(means) if means else None

        stds = [p.std for p in profiles if p.std is not None]
        avg_std = sum(stds) / len(stds) if stds else None

        # Take min/max across all profiles
        min_vals = [p.min_value for p in profiles if p.min_value is not None]
        min_val = min(min_vals) if min_vals else None

        max_vals = [p.max_value for p in profiles if p.max_value is not None]
        max_val = max(max_vals) if max_vals else None

        # Merge metadata
        merged_metadata = {}
        for p in profiles:
            merged_metadata.update(p.metadata)

        return ColumnProfile(
            column_name=col_name,
            dtype=base.dtype,
            null_count=total_null_count,
            null_percentage=base.null_percentage,  # Would need total rows to recalculate
            unique_count=total_unique,
            unique_percentage=base.unique_percentage,
            min_value=min_val,
            max_value=max_val,
            mean=avg_mean,
            std=avg_std,
            histogram=base.histogram,
            metadata=merged_metadata,
        )

    def _get_status_priority(self, result: Any) -> int:
        """Get priority for ProfileResult status."""
        if hasattr(result, "status"):
            return StatusPriority.from_profile_status(result.status).value
        return StatusPriority.PASSED.value


# =============================================================================
# LearnResult Aggregator
# =============================================================================


class LearnResultAggregator(BaseResultAggregator["LearnResult"]):
    """Aggregator for LearnResult objects.

    Merges learned rules from multiple results, handling:
    - Rule deduplication
    - Confidence score combination
    - Column analysis count aggregation
    """

    def _aggregate_merge(
        self,
        results: Sequence[Any],  # LearnResult
        config: AggregationConfig,
    ) -> Any:  # LearnResult
        """Merge all LearnResults into one."""
        from common.base import LearnedRule, LearnResult, LearnStatus

        # Collect and merge rules by (rule_type, column) key
        rules_map: dict[tuple[str, str], list[Any]] = {}
        for result in results:
            for rule in result.rules:
                key = (rule.rule_type, rule.column)
                if key not in rules_map:
                    rules_map[key] = []
                rules_map[key].append(rule)

        # Merge rules
        merged_rules: list[LearnedRule] = []
        for (rule_type, column), rule_list in rules_map.items():
            merged_rule = self._merge_learned_rules(rule_type, column, rule_list)
            merged_rules.append(merged_rule)

        # Determine status
        statuses = [r.status for r in results]
        if LearnStatus.FAILED in statuses:
            status = LearnStatus.FAILED
        elif LearnStatus.PARTIAL in statuses:
            status = LearnStatus.PARTIAL
        else:
            status = LearnStatus.COMPLETED

        # Aggregate counts
        total_columns = max((r.columns_analyzed for r in results), default=0)
        total_time = sum(r.execution_time_ms for r in results)

        # Build metadata
        metadata: dict[str, Any] = {
            "aggregation_strategy": config.strategy.name,
            "source_count": len(results),
        }
        if config.preserve_individual_results:
            metadata["individual_results"] = [r.to_dict() for r in results]

        return LearnResult(
            status=status,
            rules=tuple(merged_rules),
            columns_analyzed=total_columns,
            execution_time_ms=total_time,
            metadata=metadata,
        )

    def _merge_learned_rules(
        self,
        rule_type: str,
        column: str,
        rules: Sequence[Any],  # LearnedRule
    ) -> Any:  # LearnedRule
        """Merge multiple learned rules for the same type/column."""
        from common.base import LearnedRule

        if len(rules) == 1:
            return rules[0]

        # Use weighted average for confidence
        total_samples = sum(r.sample_size for r in rules)
        if total_samples > 0:
            weighted_confidence = sum(
                r.confidence * r.sample_size for r in rules
            ) / total_samples
        else:
            weighted_confidence = sum(r.confidence for r in rules) / len(rules)

        # Merge parameters (take first non-empty, then merge)
        merged_params: dict[str, Any] = {}
        for rule in rules:
            for key, value in rule.parameters.items():
                if key not in merged_params:
                    merged_params[key] = value

        # Merge metadata
        merged_metadata: dict[str, Any] = {
            "merged_from": len(rules),
        }
        for rule in rules:
            merged_metadata.update(rule.metadata)

        return LearnedRule(
            rule_type=rule_type,
            column=column,
            parameters=merged_params,
            confidence=weighted_confidence,
            sample_size=total_samples,
            metadata=merged_metadata,
        )

    def _get_status_priority(self, result: Any) -> int:
        """Get priority for LearnResult status."""
        if hasattr(result, "status"):
            return StatusPriority.from_learn_status(result.status).value
        return StatusPriority.PASSED.value


# =============================================================================
# DriftResult Aggregator
# =============================================================================


class DriftResultAggregator(BaseResultAggregator["DriftResult"]):
    """Aggregator for DriftResult objects.

    Supports WORST (most critical drift state) and MERGE (all column results
    combined) strategies for multi-engine drift detection results.
    """

    def _aggregate_merge(
        self,
        results: Sequence[Any],  # DriftResult
        config: AggregationConfig,
    ) -> Any:  # DriftResult
        """Merge all DriftResults into one."""
        from common.base import DriftMethod, DriftResult, DriftStatus

        # Merge all column drifts, deduplicating by column name
        all_column_drifts: list[Any] = []
        seen_columns: set[str] = set()

        for result in results:
            for col_drift in result.drifted_columns:
                if config.deduplicate_failures:
                    if col_drift.column in seen_columns:
                        continue
                    seen_columns.add(col_drift.column)
                all_column_drifts.append(col_drift)

        total_columns = max(r.total_columns for r in results)
        drifted_count = sum(1 for cd in all_column_drifts if cd.is_drifted)
        total_time = sum(r.execution_time_ms for r in results)

        # Determine status: worst across all results
        if any(r.status == DriftStatus.ERROR for r in results):
            status = DriftStatus.ERROR
        elif any(r.status == DriftStatus.DRIFT_DETECTED for r in results):
            status = DriftStatus.DRIFT_DETECTED
        elif any(r.status == DriftStatus.WARNING for r in results):
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.NO_DRIFT

        # Use the most common method, or AUTO
        methods = [r.method for r in results]
        method = max(set(methods), key=methods.count) if methods else DriftMethod.AUTO

        metadata: dict[str, Any] = {
            "aggregation_strategy": config.strategy.name,
            "source_count": len(results),
        }
        if config.include_metadata:
            metadata["source_timestamps"] = [r.timestamp for r in results]
        if config.preserve_individual_results:
            metadata["individual_results"] = [r.to_dict() for r in results]

        return DriftResult(
            status=status,
            drifted_columns=tuple(all_column_drifts),
            total_columns=total_columns,
            drifted_count=drifted_count,
            method=method,
            execution_time_ms=total_time,
            metadata=metadata,
        )

    def _get_status_priority(self, result: Any) -> int:
        """Get priority for DriftResult status."""
        if hasattr(result, "status"):
            return StatusPriority.from_drift_status(result.status).value
        return StatusPriority.PASSED.value

    def _is_failure(self, result: Any) -> bool:
        """Check if drift result represents a failure (drift detected or error)."""
        status_name = self._get_status_name(result)
        return status_name in ("DRIFT_DETECTED", "ERROR")

    def _is_success(self, result: Any) -> bool:
        """Check if drift result represents success (no drift)."""
        return self._get_status_name(result) == "NO_DRIFT"

    def _create_failed_result(
        self,
        merged: Any,
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:
        """Create a failed drift result from merged data."""
        from common.base import DriftResult, DriftStatus

        return DriftResult(
            status=DriftStatus.DRIFT_DETECTED,
            drifted_columns=merged.drifted_columns,
            total_columns=merged.total_columns,
            drifted_count=merged.drifted_count,
            method=merged.method,
            execution_time_ms=merged.execution_time_ms,
            metadata={**merged.metadata, "aggregation_override": "STRICT_ALL"},
        )

    def _create_passed_result(
        self,
        merged: Any,
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:
        """Create a passed drift result from merged data."""
        from common.base import DriftResult, DriftStatus

        return DriftResult(
            status=DriftStatus.NO_DRIFT,
            drifted_columns=merged.drifted_columns,
            total_columns=merged.total_columns,
            drifted_count=merged.drifted_count,
            method=merged.method,
            execution_time_ms=merged.execution_time_ms,
            metadata={**merged.metadata, "aggregation_override": "LENIENT_ANY"},
        )


# =============================================================================
# AnomalyResult Aggregator
# =============================================================================


class AnomalyResultAggregator(BaseResultAggregator["AnomalyResult"]):
    """Aggregator for AnomalyResult objects.

    Supports MERGE (all anomaly scores combined) and CONSENSUS (only anomalies
    agreed upon by multiple engines) strategies.
    """

    def _aggregate_merge(
        self,
        results: Sequence[Any],  # AnomalyResult
        config: AggregationConfig,
    ) -> Any:  # AnomalyResult
        """Merge all AnomalyResults into one."""
        from common.base import AnomalyResult, AnomalyStatus

        # Merge all anomaly scores, deduplicating by column
        all_anomalies: list[Any] = []
        seen_columns: set[str] = set()

        for result in results:
            for anomaly in result.anomalies:
                if config.deduplicate_failures:
                    if anomaly.column in seen_columns:
                        continue
                    seen_columns.add(anomaly.column)
                all_anomalies.append(anomaly)

        total_rows = max(r.total_row_count for r in results) if results else 0
        anomalous_rows = max(r.anomalous_row_count for r in results) if results else 0
        total_time = sum(r.execution_time_ms for r in results)

        # Determine status: worst across all results
        if any(r.status == AnomalyStatus.ERROR for r in results):
            status = AnomalyStatus.ERROR
        elif any(r.status == AnomalyStatus.ANOMALY_DETECTED for r in results):
            status = AnomalyStatus.ANOMALY_DETECTED
        elif any(r.status == AnomalyStatus.WARNING for r in results):
            status = AnomalyStatus.WARNING
        else:
            status = AnomalyStatus.NORMAL

        # Use the most common detector
        detectors = [r.detector for r in results]
        detector = max(set(detectors), key=detectors.count) if detectors else "isolation_forest"

        metadata: dict[str, Any] = {
            "aggregation_strategy": config.strategy.name,
            "source_count": len(results),
        }
        if config.include_metadata:
            metadata["source_timestamps"] = [r.timestamp for r in results]
        if config.preserve_individual_results:
            metadata["individual_results"] = [r.to_dict() for r in results]

        return AnomalyResult(
            status=status,
            anomalies=tuple(all_anomalies),
            anomalous_row_count=anomalous_rows,
            total_row_count=total_rows,
            detector=detector,
            execution_time_ms=total_time,
            metadata=metadata,
        )

    def _aggregate_consensus(
        self,
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:
        """Only include anomalies that multiple engines agree on.

        A column is considered anomalous only if at least `consensus_threshold`
        fraction of engines flag it.
        """
        from common.base import AnomalyResult, AnomalyStatus

        threshold = config.consensus_threshold
        engine_count = len(results)

        # Count how many engines flag each column as anomalous
        column_votes: dict[str, int] = {}
        column_scores: dict[str, list[Any]] = {}

        for result in results:
            for anomaly in result.anomalies:
                col = anomaly.column
                if col not in column_votes:
                    column_votes[col] = 0
                    column_scores[col] = []
                if anomaly.is_anomaly:
                    column_votes[col] += 1
                column_scores[col].append(anomaly)

        # Keep only columns meeting consensus threshold
        consensus_anomalies: list[Any] = []
        for col, votes in column_votes.items():
            ratio = votes / engine_count if engine_count > 0 else 0.0
            if ratio >= threshold:
                # Use the first score entry for this column
                consensus_anomalies.append(column_scores[col][0])

        total_rows = max(r.total_row_count for r in results) if results else 0
        anomalous_count = sum(1 for a in consensus_anomalies if a.is_anomaly)

        if anomalous_count > 0:
            status = AnomalyStatus.ANOMALY_DETECTED
        else:
            status = AnomalyStatus.NORMAL

        detectors = [r.detector for r in results]
        detector = max(set(detectors), key=detectors.count) if detectors else "isolation_forest"

        return AnomalyResult(
            status=status,
            anomalies=tuple(consensus_anomalies),
            anomalous_row_count=anomalous_count,
            total_row_count=total_rows,
            detector=detector,
            execution_time_ms=sum(r.execution_time_ms for r in results),
            metadata={
                "aggregation_strategy": "CONSENSUS",
                "consensus_threshold": threshold,
                "source_count": engine_count,
            },
        )

    def aggregate(
        self,
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:
        """Aggregate with consensus strategy support."""
        if config.strategy == ResultAggregationStrategy.CONSENSUS:
            if not results:
                raise NoResultsError()
            return self._aggregate_consensus(results, config)
        return super().aggregate(results, config)

    def _get_status_priority(self, result: Any) -> int:
        """Get priority for AnomalyResult status."""
        if hasattr(result, "status"):
            return StatusPriority.from_anomaly_status(result.status).value
        return StatusPriority.PASSED.value

    def _is_failure(self, result: Any) -> bool:
        """Check if anomaly result represents a failure."""
        status_name = self._get_status_name(result)
        return status_name in ("ANOMALY_DETECTED", "ERROR")

    def _is_success(self, result: Any) -> bool:
        """Check if anomaly result represents success (normal)."""
        return self._get_status_name(result) == "NORMAL"

    def _create_failed_result(
        self,
        merged: Any,
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:
        """Create a failed anomaly result."""
        from common.base import AnomalyResult, AnomalyStatus

        return AnomalyResult(
            status=AnomalyStatus.ANOMALY_DETECTED,
            anomalies=merged.anomalies,
            anomalous_row_count=merged.anomalous_row_count,
            total_row_count=merged.total_row_count,
            detector=merged.detector,
            execution_time_ms=merged.execution_time_ms,
            metadata={**merged.metadata, "aggregation_override": "STRICT_ALL"},
        )

    def _create_passed_result(
        self,
        merged: Any,
        results: Sequence[Any],
        config: AggregationConfig,
    ) -> Any:
        """Create a passed anomaly result."""
        from common.base import AnomalyResult, AnomalyStatus

        return AnomalyResult(
            status=AnomalyStatus.NORMAL,
            anomalies=merged.anomalies,
            anomalous_row_count=merged.anomalous_row_count,
            total_row_count=merged.total_row_count,
            detector=merged.detector,
            execution_time_ms=merged.execution_time_ms,
            metadata={**merged.metadata, "aggregation_override": "LENIENT_ANY"},
        )


# =============================================================================
# Multi-Engine Aggregator
# =============================================================================


class MultiEngineAggregator:
    """Aggregator for combining results from multiple data quality engines.

    This is the main entry point for multi-engine result aggregation.
    It provides methods for aggregating CheckResult, ProfileResult, and
    LearnResult objects from different engines.

    Example:
        >>> aggregator = MultiEngineAggregator()
        >>> combined = aggregator.aggregate_check_results({
        ...     "truthound": result1,
        ...     "ge": result2,
        ... })

    With weights:
        >>> config = AggregationConfig(
        ...     strategy=ResultAggregationStrategy.WEIGHTED,
        ...     weights={"truthound": 2.0, "ge": 1.0},
        ... )
        >>> aggregator = MultiEngineAggregator(config=config)
        >>> combined = aggregator.aggregate_check_results(engine_results)
    """

    def __init__(
        self,
        config: AggregationConfig | None = None,
        hooks: Sequence[AggregationHook] | None = None,
    ) -> None:
        """Initialize multi-engine aggregator.

        Args:
            config: Default aggregation configuration.
            hooks: Hooks for observability.
        """
        self._config = config or DEFAULT_AGGREGATION_CONFIG
        self._hooks = list(hooks) if hooks else []

        # Initialize type-specific aggregators
        self._check_aggregator = CheckResultMergeAggregator(hooks)
        self._check_weighted_aggregator = CheckResultWeightedAggregator(hooks)
        self._profile_aggregator = ProfileResultAggregator(hooks)
        self._learn_aggregator = LearnResultAggregator(hooks)
        self._drift_aggregator = DriftResultAggregator(hooks)
        self._anomaly_aggregator = AnomalyResultAggregator(hooks)

    @property
    def config(self) -> AggregationConfig:
        """Get current configuration."""
        return self._config

    def with_config(self, config: AggregationConfig) -> MultiEngineAggregator:
        """Create new aggregator with different configuration."""
        return MultiEngineAggregator(config=config, hooks=self._hooks)

    def aggregate_check_results(
        self,
        engine_results: Mapping[str, Any],  # CheckResult
        config: AggregationConfig | None = None,
    ) -> AggregatedResult[Any]:  # AggregatedResult[CheckResult]
        """Aggregate CheckResults from multiple engines.

        Args:
            engine_results: Mapping of engine name to CheckResult.
            config: Optional override configuration.

        Returns:
            Aggregated result with provenance.
        """
        cfg = config or self._config
        start_time = datetime.now(timezone.utc)

        if not engine_results:
            raise NoResultsError("No engine results provided")

        # For weighted strategy, use weighted aggregator
        if cfg.strategy == ResultAggregationStrategy.WEIGHTED:
            result = self._check_weighted_aggregator.aggregate_with_weights(
                engine_results, cfg
            )
        else:
            results_list = list(engine_results.values())
            result = self._check_aggregator.aggregate(results_list, cfg)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        individual = dict(engine_results) if cfg.preserve_individual_results else None

        return AggregatedResult(
            result=result,
            strategy=cfg.strategy,
            source_engines=tuple(engine_results.keys()),
            source_count=len(engine_results),
            aggregation_time_ms=duration,
            individual_results=individual,
            metadata={"config": cfg.strategy.name},
        )

    def aggregate_profile_results(
        self,
        engine_results: Mapping[str, Any],  # ProfileResult
        config: AggregationConfig | None = None,
    ) -> AggregatedResult[Any]:  # AggregatedResult[ProfileResult]
        """Aggregate ProfileResults from multiple engines.

        Args:
            engine_results: Mapping of engine name to ProfileResult.
            config: Optional override configuration.

        Returns:
            Aggregated result with provenance.
        """
        cfg = config or self._config
        start_time = datetime.now(timezone.utc)

        if not engine_results:
            raise NoResultsError("No engine results provided")

        results_list = list(engine_results.values())
        result = self._profile_aggregator.aggregate(results_list, cfg)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        individual = dict(engine_results) if cfg.preserve_individual_results else None

        return AggregatedResult(
            result=result,
            strategy=cfg.strategy,
            source_engines=tuple(engine_results.keys()),
            source_count=len(engine_results),
            aggregation_time_ms=duration,
            individual_results=individual,
            metadata={"config": cfg.strategy.name},
        )

    def aggregate_learn_results(
        self,
        engine_results: Mapping[str, Any],  # LearnResult
        config: AggregationConfig | None = None,
    ) -> AggregatedResult[Any]:  # AggregatedResult[LearnResult]
        """Aggregate LearnResults from multiple engines.

        Args:
            engine_results: Mapping of engine name to LearnResult.
            config: Optional override configuration.

        Returns:
            Aggregated result with provenance.
        """
        cfg = config or self._config
        start_time = datetime.now(timezone.utc)

        if not engine_results:
            raise NoResultsError("No engine results provided")

        results_list = list(engine_results.values())
        result = self._learn_aggregator.aggregate(results_list, cfg)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        individual = dict(engine_results) if cfg.preserve_individual_results else None

        return AggregatedResult(
            result=result,
            strategy=cfg.strategy,
            source_engines=tuple(engine_results.keys()),
            source_count=len(engine_results),
            aggregation_time_ms=duration,
            individual_results=individual,
            metadata={"config": cfg.strategy.name},
        )

    def aggregate_drift_results(
        self,
        engine_results: Mapping[str, Any],  # DriftResult
        config: AggregationConfig | None = None,
    ) -> AggregatedResult[Any]:  # AggregatedResult[DriftResult]
        """Aggregate DriftResults from multiple engines.

        Args:
            engine_results: Mapping of engine name to DriftResult.
            config: Optional override configuration.

        Returns:
            Aggregated result with provenance.
        """
        cfg = config or self._config
        start_time = datetime.now(timezone.utc)

        if not engine_results:
            raise NoResultsError("No engine results provided")

        results_list = list(engine_results.values())
        result = self._drift_aggregator.aggregate(results_list, cfg)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        individual = dict(engine_results) if cfg.preserve_individual_results else None

        return AggregatedResult(
            result=result,
            strategy=cfg.strategy,
            source_engines=tuple(engine_results.keys()),
            source_count=len(engine_results),
            aggregation_time_ms=duration,
            individual_results=individual,
            metadata={"config": cfg.strategy.name},
        )

    def aggregate_anomaly_results(
        self,
        engine_results: Mapping[str, Any],  # AnomalyResult
        config: AggregationConfig | None = None,
    ) -> AggregatedResult[Any]:  # AggregatedResult[AnomalyResult]
        """Aggregate AnomalyResults from multiple engines.

        Args:
            engine_results: Mapping of engine name to AnomalyResult.
            config: Optional override configuration.

        Returns:
            Aggregated result with provenance.
        """
        cfg = config or self._config
        start_time = datetime.now(timezone.utc)

        if not engine_results:
            raise NoResultsError("No engine results provided")

        results_list = list(engine_results.values())
        result = self._anomaly_aggregator.aggregate(results_list, cfg)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        individual = dict(engine_results) if cfg.preserve_individual_results else None

        return AggregatedResult(
            result=result,
            strategy=cfg.strategy,
            source_engines=tuple(engine_results.keys()),
            source_count=len(engine_results),
            aggregation_time_ms=duration,
            individual_results=individual,
            metadata={"config": cfg.strategy.name},
        )

    def compare_check_results(
        self,
        engine_results: Mapping[str, Any],  # CheckResult
    ) -> ComparisonResult:
        """Compare CheckResults from multiple engines.

        Args:
            engine_results: Mapping of engine name to CheckResult.

        Returns:
            Comparison result with agreement metrics.
        """
        if not engine_results:
            raise NoResultsError("No engine results provided")

        # Extract statuses
        engine_statuses: dict[str, str] = {}
        for name, result in engine_results.items():
            status = result.status
            engine_statuses[name] = status.name if hasattr(status, "name") else str(status)

        # Calculate agreement
        status_counts = Counter(engine_statuses.values())
        total = len(engine_statuses)
        most_common_status, most_common_count = status_counts.most_common(1)[0]

        agreement_ratio = most_common_count / total if total > 0 else 0.0
        unanimous = len(status_counts) == 1

        # Find discrepancies
        discrepancies: list[dict[str, Any]] = []
        if not unanimous:
            for name, status in engine_statuses.items():
                if status != most_common_status:
                    discrepancies.append({
                        "engine": name,
                        "status": status,
                        "expected": most_common_status,
                    })

        return ComparisonResult(
            agreement_ratio=agreement_ratio,
            unanimous=unanimous,
            majority_status=most_common_status,
            engine_statuses=engine_statuses,
            discrepancies=tuple(discrepancies),
            metadata={"total_engines": total},
        )


# =============================================================================
# Hooks
# =============================================================================


class BaseAggregationHook:
    """Base implementation of AggregationHook with no-op methods."""

    def on_aggregation_start(
        self,
        result_count: int,
        strategy: ResultAggregationStrategy,
        metadata: Mapping[str, Any],
    ) -> None:
        """Called when aggregation starts."""
        pass

    def on_aggregation_complete(
        self,
        result: Any,
        source_count: int,
        conflict_count: int,
        duration_ms: float,
    ) -> None:
        """Called when aggregation completes."""
        pass

    def on_conflict_detected(
        self,
        conflict_type: str,
        engines: Sequence[str],
        details: Mapping[str, Any],
    ) -> None:
        """Called when a conflict is detected."""
        pass

    def on_aggregation_error(
        self,
        error: Exception,
        result_count: int,
    ) -> None:
        """Called when aggregation fails."""
        pass


class LoggingAggregationHook(BaseAggregationHook):
    """Hook that logs aggregation events."""

    def __init__(self, logger_name: str = "aggregation") -> None:
        import logging

        self._logger = logging.getLogger(logger_name)

    def on_aggregation_start(
        self,
        result_count: int,
        strategy: ResultAggregationStrategy,
        metadata: Mapping[str, Any],
    ) -> None:
        self._logger.info(
            "Aggregation started: count=%d, strategy=%s",
            result_count,
            strategy.name,
        )

    def on_aggregation_complete(
        self,
        result: Any,
        source_count: int,
        conflict_count: int,
        duration_ms: float,
    ) -> None:
        self._logger.info(
            "Aggregation complete: sources=%d, conflicts=%d, duration=%.2fms",
            source_count,
            conflict_count,
            duration_ms,
        )

    def on_conflict_detected(
        self,
        conflict_type: str,
        engines: Sequence[str],
        details: Mapping[str, Any],
    ) -> None:
        self._logger.warning(
            "Conflict detected: type=%s, engines=%s",
            conflict_type,
            list(engines),
        )

    def on_aggregation_error(
        self,
        error: Exception,
        result_count: int,
    ) -> None:
        self._logger.error(
            "Aggregation failed: error=%s, result_count=%d",
            str(error),
            result_count,
        )


class MetricsAggregationHook(BaseAggregationHook):
    """Hook that collects aggregation metrics."""

    def __init__(self) -> None:
        self._aggregation_count = 0
        self._total_results_aggregated = 0
        self._total_conflicts = 0
        self._total_errors = 0
        self._total_duration_ms = 0.0
        self._strategy_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def aggregation_count(self) -> int:
        """Total number of aggregations performed."""
        return self._aggregation_count

    @property
    def average_duration_ms(self) -> float:
        """Average aggregation duration in milliseconds."""
        if self._aggregation_count == 0:
            return 0.0
        return self._total_duration_ms / self._aggregation_count

    @property
    def total_conflicts(self) -> int:
        """Total number of conflicts detected."""
        return self._total_conflicts

    @property
    def error_rate(self) -> float:
        """Error rate (errors / total aggregations)."""
        total = self._aggregation_count + self._total_errors
        if total == 0:
            return 0.0
        return self._total_errors / total

    def get_stats(self) -> dict[str, Any]:
        """Get all collected statistics."""
        with self._lock:
            return {
                "aggregation_count": self._aggregation_count,
                "total_results_aggregated": self._total_results_aggregated,
                "total_conflicts": self._total_conflicts,
                "total_errors": self._total_errors,
                "average_duration_ms": self.average_duration_ms,
                "error_rate": self.error_rate,
                "strategy_counts": dict(self._strategy_counts),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._aggregation_count = 0
            self._total_results_aggregated = 0
            self._total_conflicts = 0
            self._total_errors = 0
            self._total_duration_ms = 0.0
            self._strategy_counts.clear()

    def on_aggregation_start(
        self,
        result_count: int,
        strategy: ResultAggregationStrategy,
        metadata: Mapping[str, Any],
    ) -> None:
        with self._lock:
            strategy_name = strategy.name
            self._strategy_counts[strategy_name] = (
                self._strategy_counts.get(strategy_name, 0) + 1
            )

    def on_aggregation_complete(
        self,
        result: Any,
        source_count: int,
        conflict_count: int,
        duration_ms: float,
    ) -> None:
        with self._lock:
            self._aggregation_count += 1
            self._total_results_aggregated += source_count
            self._total_conflicts += conflict_count
            self._total_duration_ms += duration_ms

    def on_conflict_detected(
        self,
        conflict_type: str,
        engines: Sequence[str],
        details: Mapping[str, Any],
    ) -> None:
        # Conflicts are counted in on_aggregation_complete
        pass

    def on_aggregation_error(
        self,
        error: Exception,
        result_count: int,
    ) -> None:
        with self._lock:
            self._total_errors += 1


class CompositeAggregationHook(BaseAggregationHook):
    """Hook that delegates to multiple hooks."""

    def __init__(self, hooks: Sequence[AggregationHook] | None = None) -> None:
        self._hooks = list(hooks) if hooks else []

    def add_hook(self, hook: AggregationHook) -> None:
        """Add a hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: AggregationHook) -> None:
        """Remove a hook."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_aggregation_start(
        self,
        result_count: int,
        strategy: ResultAggregationStrategy,
        metadata: Mapping[str, Any],
    ) -> None:
        for hook in self._hooks:
            try:
                hook.on_aggregation_start(result_count, strategy, metadata)
            except Exception:
                pass

    def on_aggregation_complete(
        self,
        result: Any,
        source_count: int,
        conflict_count: int,
        duration_ms: float,
    ) -> None:
        for hook in self._hooks:
            try:
                hook.on_aggregation_complete(result, source_count, conflict_count, duration_ms)
            except Exception:
                pass

    def on_conflict_detected(
        self,
        conflict_type: str,
        engines: Sequence[str],
        details: Mapping[str, Any],
    ) -> None:
        for hook in self._hooks:
            try:
                hook.on_conflict_detected(conflict_type, engines, details)
            except Exception:
                pass

    def on_aggregation_error(
        self,
        error: Exception,
        result_count: int,
    ) -> None:
        for hook in self._hooks:
            try:
                hook.on_aggregation_error(error, result_count)
            except Exception:
                pass


# =============================================================================
# Registry
# =============================================================================


class AggregatorRegistry:
    """Thread-safe registry for result aggregators.

    Provides centralized management of aggregator instances with
    lazy initialization and thread-safe access.

    Example:
        >>> registry = AggregatorRegistry()
        >>> registry.register("custom", CustomAggregator())
        >>> aggregator = registry.get("custom")
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._aggregators: dict[str, BaseResultAggregator[Any]] = {}
        self._lock = threading.RLock()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default aggregators."""
        self._aggregators["check"] = CheckResultMergeAggregator()
        self._aggregators["check_weighted"] = CheckResultWeightedAggregator()
        self._aggregators["profile"] = ProfileResultAggregator()
        self._aggregators["learn"] = LearnResultAggregator()
        self._aggregators["drift"] = DriftResultAggregator()
        self._aggregators["anomaly"] = AnomalyResultAggregator()

    def register(
        self,
        name: str,
        aggregator: BaseResultAggregator[Any],
        allow_override: bool = False,
    ) -> None:
        """Register an aggregator.

        Args:
            name: Name for the aggregator.
            aggregator: Aggregator instance.
            allow_override: Whether to allow overriding existing registration.

        Raises:
            ValueError: If name exists and allow_override is False.
        """
        with self._lock:
            if name in self._aggregators and not allow_override:
                msg = f"Aggregator '{name}' already registered"
                raise ValueError(msg)
            self._aggregators[name] = aggregator

    def unregister(self, name: str) -> BaseResultAggregator[Any] | None:
        """Unregister an aggregator.

        Args:
            name: Name of the aggregator.

        Returns:
            The unregistered aggregator, or None if not found.
        """
        with self._lock:
            return self._aggregators.pop(name, None)

    def get(self, name: str) -> BaseResultAggregator[Any]:
        """Get an aggregator by name.

        Args:
            name: Name of the aggregator.

        Returns:
            The aggregator instance.

        Raises:
            AggregatorNotFoundError: If aggregator not found.
        """
        with self._lock:
            if name not in self._aggregators:
                raise AggregatorNotFoundError(name)
            return self._aggregators[name]

    def get_or_none(self, name: str) -> BaseResultAggregator[Any] | None:
        """Get an aggregator by name, returning None if not found."""
        with self._lock:
            return self._aggregators.get(name)

    def has(self, name: str) -> bool:
        """Check if an aggregator is registered."""
        with self._lock:
            return name in self._aggregators

    def list(self) -> list[str]:
        """List all registered aggregator names."""
        with self._lock:
            return list(self._aggregators.keys())

    def clear(self) -> None:
        """Clear all registered aggregators and re-register defaults."""
        with self._lock:
            self._aggregators.clear()
            self._register_defaults()


# Global registry instance
_global_registry: AggregatorRegistry | None = None
_registry_lock = threading.Lock()


def get_aggregator_registry() -> AggregatorRegistry:
    """Get the global aggregator registry.

    Returns:
        The global AggregatorRegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = AggregatorRegistry()
    return _global_registry


def get_aggregator(name: str) -> BaseResultAggregator[Any]:
    """Get an aggregator from the global registry.

    Args:
        name: Name of the aggregator.

    Returns:
        The aggregator instance.
    """
    return get_aggregator_registry().get(name)


def register_aggregator(
    name: str,
    aggregator: BaseResultAggregator[Any],
    allow_override: bool = False,
) -> None:
    """Register an aggregator in the global registry.

    Args:
        name: Name for the aggregator.
        aggregator: Aggregator instance.
        allow_override: Whether to allow overriding existing registration.
    """
    get_aggregator_registry().register(name, aggregator, allow_override)


def list_aggregators() -> list[str]:
    """List all registered aggregator names.

    Returns:
        List of aggregator names.
    """
    return get_aggregator_registry().list()


# =============================================================================
# Convenience Functions
# =============================================================================


def aggregate_check_results(
    engine_results: Mapping[str, Any],
    config: AggregationConfig | None = None,
) -> AggregatedResult[Any]:
    """Aggregate CheckResults from multiple engines.

    Convenience function that creates a MultiEngineAggregator and aggregates.

    Args:
        engine_results: Mapping of engine name to CheckResult.
        config: Optional aggregation configuration.

    Returns:
        Aggregated result with provenance.

    Example:
        >>> combined = aggregate_check_results({
        ...     "truthound": result1,
        ...     "ge": result2,
        ... })
    """
    aggregator = MultiEngineAggregator(config=config)
    return aggregator.aggregate_check_results(engine_results)


def aggregate_profile_results(
    engine_results: Mapping[str, Any],
    config: AggregationConfig | None = None,
) -> AggregatedResult[Any]:
    """Aggregate ProfileResults from multiple engines.

    Args:
        engine_results: Mapping of engine name to ProfileResult.
        config: Optional aggregation configuration.

    Returns:
        Aggregated result with provenance.
    """
    aggregator = MultiEngineAggregator(config=config)
    return aggregator.aggregate_profile_results(engine_results)


def aggregate_learn_results(
    engine_results: Mapping[str, Any],
    config: AggregationConfig | None = None,
) -> AggregatedResult[Any]:
    """Aggregate LearnResults from multiple engines.

    Args:
        engine_results: Mapping of engine name to LearnResult.
        config: Optional aggregation configuration.

    Returns:
        Aggregated result with provenance.
    """
    aggregator = MultiEngineAggregator(config=config)
    return aggregator.aggregate_learn_results(engine_results)


def compare_check_results(
    engine_results: Mapping[str, Any],
) -> ComparisonResult:
    """Compare CheckResults from multiple engines.

    Args:
        engine_results: Mapping of engine name to CheckResult.

    Returns:
        Comparison result with agreement metrics.
    """
    aggregator = MultiEngineAggregator()
    return aggregator.compare_check_results(engine_results)


def aggregate_drift_results(
    engine_results: Mapping[str, Any],
    config: AggregationConfig | None = None,
) -> AggregatedResult[Any]:
    """Aggregate DriftResults from multiple engines.

    Convenience function that creates a MultiEngineAggregator and aggregates.

    Args:
        engine_results: Mapping of engine name to DriftResult.
        config: Optional aggregation configuration.

    Returns:
        Aggregated result with provenance.

    Example:
        >>> combined = aggregate_drift_results({
        ...     "truthound": drift_result1,
        ...     "ge": drift_result2,
        ... })
    """
    aggregator = MultiEngineAggregator(config=config)
    return aggregator.aggregate_drift_results(engine_results)


def aggregate_anomaly_results(
    engine_results: Mapping[str, Any],
    config: AggregationConfig | None = None,
) -> AggregatedResult[Any]:
    """Aggregate AnomalyResults from multiple engines.

    Convenience function that creates a MultiEngineAggregator and aggregates.

    Args:
        engine_results: Mapping of engine name to AnomalyResult.
        config: Optional aggregation configuration.

    Returns:
        Aggregated result with provenance.

    Example:
        >>> combined = aggregate_anomaly_results({
        ...     "truthound": anomaly_result1,
        ...     "ge": anomaly_result2,
        ... })
    """
    aggregator = MultiEngineAggregator(config=config)
    return aggregator.aggregate_anomaly_results(engine_results)


def create_multi_engine_aggregator(
    strategy: ResultAggregationStrategy = ResultAggregationStrategy.MERGE,
    weights: Mapping[str, float] | None = None,
    hooks: Sequence[AggregationHook] | None = None,
) -> MultiEngineAggregator:
    """Create a configured MultiEngineAggregator.

    Args:
        strategy: Aggregation strategy to use.
        weights: Optional engine weights for weighted strategies.
        hooks: Optional hooks for observability.

    Returns:
        Configured MultiEngineAggregator instance.
    """
    config = AggregationConfig(
        strategy=strategy,
        weights=weights or {},
    )
    return MultiEngineAggregator(config=config, hooks=hooks)
