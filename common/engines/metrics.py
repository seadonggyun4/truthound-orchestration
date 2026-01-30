"""Engine Metrics Integration for Truthound Orchestration.

This module provides enterprise-grade metrics collection and observability
for data quality engine operations. It integrates with the existing metrics
infrastructure (Counter, Gauge, Histogram) and tracing system to provide:

- Automatic latency tracking for check/profile/learn operations
- Error rate monitoring per engine
- Data quality score tracking
- Distributed tracing integration
- Health check metrics bridge

Design Principles:
    1. Protocol-based: Easy to extend with custom collectors
    2. Non-invasive: Transparent wrapper pattern preserves engine behavior
    3. Composable: Multiple hooks can be combined
    4. Thread-safe: All operations are thread-safe
    5. Minimal overhead: Hooks are isolated from engine execution

Quick Start:
    >>> from common.engines.metrics import InstrumentedEngine, MetricsEngineHook
    >>> from common.engines import TruthoundEngine
    >>> engine = InstrumentedEngine(
    ...     TruthoundEngine(),
    ...     hooks=[MetricsEngineHook()],
    ... )
    >>> result = engine.check(data, auto_schema=True)

Using Decorators:
    >>> from common.engines.metrics import engine_instrumented
    >>> @engine_instrumented()
    >>> class MyEngine(EngineInfoMixin):
    ...     ...

Custom Hooks:
    >>> class MyHook(EngineMetricsHook):
    ...     def on_check_end(self, engine_name, result, duration_ms, context):
    ...         send_to_datadog(engine_name, result, duration_ms)

Available Hooks:
    - MetricsEngineHook: Prometheus-style metrics collection
    - LoggingEngineHook: Structured logging for engine operations
    - TracingEngineHook: Distributed tracing with span creation
    - HealthMetricsHook: Health check integration
    - CompositeEngineHook: Combine multiple hooks
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    runtime_checkable,
)

from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from common.base import (
        AnomalyResult,
        CheckResult,
        DriftResult,
        LearnResult,
        ProfileResult,
    )
    from common.engines.base import (
        AnomalyDetectionEngine,
        DataQualityEngine,
        DriftDetectionEngine,
        StreamingEngine,
    )
    from common.health import HealthCheckResult
    from common.metrics import (
        Counter,
        Gauge,
        Histogram,
        MetricsRegistry,
        Span,
        TracingRegistry,
    )


# =============================================================================
# Exceptions
# =============================================================================


class EngineMetricsError(TruthoundIntegrationError):
    """Base exception for engine metrics errors.

    Attributes:
        engine_name: Name of the engine.
        operation: Operation that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        engine_name: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize engine metrics error.

        Args:
            message: Human-readable error description.
            engine_name: Name of the engine.
            operation: Operation that failed.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if engine_name:
            details["engine_name"] = engine_name
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, cause=cause)
        self.engine_name = engine_name
        self.operation = operation


# =============================================================================
# Enums
# =============================================================================


class EngineOperation(Enum):
    """Types of engine operations for metrics tracking.

    Attributes:
        CHECK: Validation check operation.
        PROFILE: Data profiling operation.
        LEARN: Rule learning operation.
        HEALTH_CHECK: Health check operation.
    """

    CHECK = auto()
    PROFILE = auto()
    LEARN = auto()
    HEALTH_CHECK = auto()
    DRIFT = auto()
    ANOMALY = auto()
    STREAM_CHECK = auto()


class OperationStatus(Enum):
    """Status of an engine operation for metrics labeling.

    Attributes:
        SUCCESS: Operation completed successfully.
        FAILURE: Operation returned a failure result.
        ERROR: Operation raised an exception.
    """

    SUCCESS = auto()
    FAILURE = auto()
    ERROR = auto()


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class EngineMetricsConfig:
    """Configuration for engine metrics collection.

    Immutable configuration object for engine metrics operations.
    Use builder methods to create modified copies.

    Attributes:
        enabled: Whether metrics collection is enabled.
        prefix: Prefix for all metric names.
        include_data_size: Whether to record data size metrics.
        include_result_counts: Whether to record result count metrics.
        include_tracing: Whether to create spans for operations.
        histogram_buckets: Custom histogram bucket boundaries for latency.
        default_labels: Default labels applied to all metrics.

    Example:
        >>> config = EngineMetricsConfig(
        ...     prefix="truthound",
        ...     include_tracing=True,
        ... )
        >>> disabled_config = config.with_enabled(False)
    """

    enabled: bool = True
    prefix: str = "engine"
    include_data_size: bool = True
    include_result_counts: bool = True
    include_tracing: bool = True
    histogram_buckets: tuple[float, ...] = (
        0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0
    )
    default_labels: dict[str, str] = field(default_factory=dict)

    def with_enabled(self, enabled: bool) -> EngineMetricsConfig:
        """Create config with new enabled state.

        Args:
            enabled: New enabled state.

        Returns:
            New EngineMetricsConfig with updated value.
        """
        return EngineMetricsConfig(
            enabled=enabled,
            prefix=self.prefix,
            include_data_size=self.include_data_size,
            include_result_counts=self.include_result_counts,
            include_tracing=self.include_tracing,
            histogram_buckets=self.histogram_buckets,
            default_labels=self.default_labels,
        )

    def with_prefix(self, prefix: str) -> EngineMetricsConfig:
        """Create config with new prefix.

        Args:
            prefix: New metric name prefix.

        Returns:
            New EngineMetricsConfig with updated value.
        """
        return EngineMetricsConfig(
            enabled=self.enabled,
            prefix=prefix,
            include_data_size=self.include_data_size,
            include_result_counts=self.include_result_counts,
            include_tracing=self.include_tracing,
            histogram_buckets=self.histogram_buckets,
            default_labels=self.default_labels,
        )

    def with_default_labels(self, **labels: str) -> EngineMetricsConfig:
        """Create config with additional default labels.

        Args:
            **labels: Additional default labels.

        Returns:
            New EngineMetricsConfig with merged labels.
        """
        return EngineMetricsConfig(
            enabled=self.enabled,
            prefix=self.prefix,
            include_data_size=self.include_data_size,
            include_result_counts=self.include_result_counts,
            include_tracing=self.include_tracing,
            histogram_buckets=self.histogram_buckets,
            default_labels={**self.default_labels, **labels},
        )

    def with_histogram_buckets(self, *buckets: float) -> EngineMetricsConfig:
        """Create config with custom histogram buckets.

        Args:
            *buckets: Bucket boundaries.

        Returns:
            New EngineMetricsConfig with updated buckets.
        """
        return EngineMetricsConfig(
            enabled=self.enabled,
            prefix=self.prefix,
            include_data_size=self.include_data_size,
            include_result_counts=self.include_result_counts,
            include_tracing=self.include_tracing,
            histogram_buckets=tuple(sorted(buckets)),
            default_labels=self.default_labels,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "include_data_size": self.include_data_size,
            "include_result_counts": self.include_result_counts,
            "include_tracing": self.include_tracing,
            "histogram_buckets": list(self.histogram_buckets),
            "default_labels": self.default_labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create EngineMetricsConfig from dictionary.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New EngineMetricsConfig instance.
        """
        default_buckets = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        return cls(
            enabled=data.get("enabled", True),
            prefix=data.get("prefix", "engine"),
            include_data_size=data.get("include_data_size", True),
            include_result_counts=data.get("include_result_counts", True),
            include_tracing=data.get("include_tracing", True),
            histogram_buckets=tuple(data.get("histogram_buckets", default_buckets)),
            default_labels=data.get("default_labels", {}),
        )


# Default configurations
DEFAULT_ENGINE_METRICS_CONFIG = EngineMetricsConfig()

DISABLED_ENGINE_METRICS_CONFIG = EngineMetricsConfig(enabled=False)

MINIMAL_ENGINE_METRICS_CONFIG = EngineMetricsConfig(
    include_data_size=False,
    include_result_counts=False,
    include_tracing=False,
)

FULL_ENGINE_METRICS_CONFIG = EngineMetricsConfig(
    include_data_size=True,
    include_result_counts=True,
    include_tracing=True,
    histogram_buckets=(
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0
    ),
)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class EngineMetricsHook(Protocol):
    """Protocol for engine metrics hooks.

    Implement this to receive notifications about engine operation events.
    All methods should be exception-safe and non-blocking.

    Methods are called at specific points in the operation lifecycle:
    - on_*_start: Called before the operation begins
    - on_*_end: Called after the operation completes (success or failure result)
    - on_error: Called when an operation raises an exception

    Context Dictionary:
        The context dict can include:
        - data_type: Type name of the input data
        - data_size: Size/length of input data (if available)
        - rules_count: Number of rules (for check operations)
        - kwargs: Original kwargs passed to the operation

    Example:
        >>> class MyHook:
        ...     def on_check_start(self, engine_name, data_size, context):
        ...         print(f"Starting check on {engine_name}")
        ...
        ...     def on_check_end(self, engine_name, result, duration_ms, context):
        ...         print(f"Check completed: {result.status.name}")
    """

    @abstractmethod
    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when check operation starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of input data (row count if available).
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when check operation completes.

        Args:
            engine_name: Name of the engine.
            result: Check result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when profile operation starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of input data.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when profile operation completes.

        Args:
            engine_name: Name of the engine.
            result: Profile result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when learn operation starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of input data.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when learn operation completes.

        Args:
            engine_name: Name of the engine.
            result: Learn result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_drift_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when drift detection starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of input data (row count if available).
            context: Additional context including method, columns, threshold.
        """
        ...

    @abstractmethod
    def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when drift detection completes.

        Args:
            engine_name: Name of the engine.
            result: Drift detection result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when anomaly detection starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of input data.
            context: Additional context including detector, columns, contamination.
        """
        ...

    @abstractmethod
    def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when anomaly detection completes.

        Args:
            engine_name: Name of the engine.
            result: Anomaly detection result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_stream_check_start(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Called when streaming check starts.

        Args:
            engine_name: Name of the engine.
            context: Additional context including batch_size.
        """
        ...

    @abstractmethod
    def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when streaming check completes.

        Args:
            engine_name: Name of the engine.
            batch_count: Number of batches processed.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when any operation fails with an exception.

        Args:
            engine_name: Name of the engine.
            operation: Type of operation that failed.
            exception: The exception that was raised.
            duration_ms: Duration until failure in milliseconds.
            context: Additional context.
        """
        ...


@runtime_checkable
class AsyncEngineMetricsHook(Protocol):
    """Protocol for async engine metrics hooks.

    Async version of EngineMetricsHook for engines that support
    asynchronous operations.
    """

    @abstractmethod
    async def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when async check operation starts."""
        ...

    @abstractmethod
    async def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when async check operation completes."""
        ...

    @abstractmethod
    async def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when async profile operation starts."""
        ...

    @abstractmethod
    async def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when async profile operation completes."""
        ...

    @abstractmethod
    async def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when async learn operation starts."""
        ...

    @abstractmethod
    async def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when async learn operation completes."""
        ...

    @abstractmethod
    async def on_drift_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when async drift detection starts."""
        ...

    @abstractmethod
    async def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when async drift detection completes."""
        ...

    @abstractmethod
    async def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when async anomaly detection starts."""
        ...

    @abstractmethod
    async def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when async anomaly detection completes."""
        ...

    @abstractmethod
    async def on_stream_check_start(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Called when async streaming check starts."""
        ...

    @abstractmethod
    async def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when async streaming check completes."""
        ...

    @abstractmethod
    async def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when any async operation fails."""
        ...


# =============================================================================
# Base Hook Implementation
# =============================================================================


class BaseEngineMetricsHook:
    """Base implementation for engine metrics hooks.

    Provides no-op implementations for all hook methods.
    Subclass and override only the methods you need.
    """

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_drift_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_stream_check_start(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass


class AsyncBaseEngineMetricsHook:
    """Base implementation for async engine metrics hooks.

    Provides no-op implementations for all async hook methods.
    """

    async def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_drift_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_stream_check_start(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass

    async def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """No-op implementation."""
        pass


# =============================================================================
# Metrics Hook Implementation
# =============================================================================


class MetricsEngineHook(BaseEngineMetricsHook):
    """Collects Prometheus-style metrics for engine operations.

    Tracks the following metrics:
    - engine_check_duration_seconds: Histogram of check operation durations
    - engine_profile_duration_seconds: Histogram of profile operation durations
    - engine_learn_duration_seconds: Histogram of learn operation durations
    - engine_operations_total: Counter of total operations by type and status
    - engine_errors_total: Counter of errors by type and error class
    - engine_check_passed_total: Counter of passed validations
    - engine_check_failed_total: Counter of failed validations
    - engine_data_rows_processed: Counter of rows processed

    All metrics include engine name as a label.

    Example:
        >>> hook = MetricsEngineHook()
        >>> engine = InstrumentedEngine(TruthoundEngine(), hooks=[hook])
        >>> result = engine.check(data)
        >>> # Metrics are automatically recorded
    """

    def __init__(
        self,
        config: EngineMetricsConfig | None = None,
        registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize metrics hook.

        Args:
            config: Metrics configuration.
            registry: Metrics registry (uses global if None).
        """
        from common.metrics import get_metrics_registry

        self._config = config or DEFAULT_ENGINE_METRICS_CONFIG
        self._registry = registry or get_metrics_registry()
        self._lock = threading.Lock()

        # Initialize metrics lazily
        self._check_duration: Histogram | None = None
        self._profile_duration: Histogram | None = None
        self._learn_duration: Histogram | None = None
        self._drift_duration: Histogram | None = None
        self._anomaly_duration: Histogram | None = None
        self._stream_check_duration: Histogram | None = None
        self._operations_total: Counter | None = None
        self._errors_total: Counter | None = None
        self._passed_total: Counter | None = None
        self._failed_total: Counter | None = None
        self._drift_detected_total: Counter | None = None
        self._anomaly_detected_total: Counter | None = None
        self._stream_batches_total: Counter | None = None
        self._rows_processed: Counter | None = None
        self._active_operations: Gauge | None = None

    def _get_metric_name(self, name: str) -> str:
        """Get prefixed metric name."""
        if self._config.prefix:
            return f"{self._config.prefix}_{name}"
        return name

    def _ensure_metrics_initialized(self) -> None:
        """Initialize metrics if not already done."""
        if self._check_duration is not None:
            return

        with self._lock:
            if self._check_duration is not None:
                return


            # Duration histograms
            self._check_duration = self._registry.histogram(
                self._get_metric_name("check_duration_seconds"),
                "Duration of check operations in seconds",
                buckets=self._config.histogram_buckets,
            )
            self._profile_duration = self._registry.histogram(
                self._get_metric_name("profile_duration_seconds"),
                "Duration of profile operations in seconds",
                buckets=self._config.histogram_buckets,
            )
            self._learn_duration = self._registry.histogram(
                self._get_metric_name("learn_duration_seconds"),
                "Duration of learn operations in seconds",
                buckets=self._config.histogram_buckets,
            )
            self._drift_duration = self._registry.histogram(
                self._get_metric_name("drift_duration_seconds"),
                "Duration of drift detection operations in seconds",
                buckets=self._config.histogram_buckets,
            )
            self._anomaly_duration = self._registry.histogram(
                self._get_metric_name("anomaly_duration_seconds"),
                "Duration of anomaly detection operations in seconds",
                buckets=self._config.histogram_buckets,
            )
            self._stream_check_duration = self._registry.histogram(
                self._get_metric_name("stream_check_duration_seconds"),
                "Duration of streaming check operations in seconds",
                buckets=self._config.histogram_buckets,
            )

            # Operation counters
            self._operations_total = self._registry.counter(
                self._get_metric_name("operations_total"),
                "Total number of engine operations",
            )
            self._errors_total = self._registry.counter(
                self._get_metric_name("errors_total"),
                "Total number of engine operation errors",
            )

            # Result counters
            self._passed_total = self._registry.counter(
                self._get_metric_name("check_passed_total"),
                "Total number of passed validations",
            )
            self._failed_total = self._registry.counter(
                self._get_metric_name("check_failed_total"),
                "Total number of failed validations",
            )

            # Drift/Anomaly/Stream counters
            self._drift_detected_total = self._registry.counter(
                self._get_metric_name("drift_detected_total"),
                "Total number of drift detections",
            )
            self._anomaly_detected_total = self._registry.counter(
                self._get_metric_name("anomaly_detected_total"),
                "Total number of anomaly detections",
            )
            self._stream_batches_total = self._registry.counter(
                self._get_metric_name("stream_batches_total"),
                "Total number of stream batches processed",
            )

            # Data processing
            self._rows_processed = self._registry.counter(
                self._get_metric_name("data_rows_processed_total"),
                "Total number of data rows processed",
            )

            # Active operations gauge
            self._active_operations = self._registry.gauge(
                self._get_metric_name("active_operations"),
                "Number of currently active operations",
            )

    def _merge_labels(self, **labels: str) -> dict[str, str]:
        """Merge labels with default labels."""
        return {**self._config.default_labels, **labels}

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record check start metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()
        labels = self._merge_labels(engine=engine_name, operation="check")

        if self._active_operations:
            self._active_operations.inc(labels=labels)

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record check completion metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()

        # Determine status
        status = "success" if result.is_success else "failure"
        labels = self._merge_labels(
            engine=engine_name,
            operation="check",
            status=status,
        )

        # Duration
        if self._check_duration:
            self._check_duration.observe(duration_ms / 1000.0, labels=labels)

        # Operation count
        if self._operations_total:
            self._operations_total.inc(labels=labels)

        # Active operations
        check_labels = self._merge_labels(engine=engine_name, operation="check")
        if self._active_operations:
            self._active_operations.dec(labels=check_labels)

        # Result counts
        if self._config.include_result_counts:
            engine_labels = self._merge_labels(engine=engine_name)
            if self._passed_total and result.passed_count > 0:
                self._passed_total.inc(result.passed_count, labels=engine_labels)
            if self._failed_total and result.failed_count > 0:
                self._failed_total.inc(result.failed_count, labels=engine_labels)

        # Data size
        if self._config.include_data_size:
            data_size = context.get("data_size")
            if data_size and self._rows_processed:
                self._rows_processed.inc(
                    data_size,
                    labels=self._merge_labels(engine=engine_name, operation="check"),
                )

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record profile start metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()
        labels = self._merge_labels(engine=engine_name, operation="profile")

        if self._active_operations:
            self._active_operations.inc(labels=labels)

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record profile completion metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()

        status = "success" if result.is_success else "failure"
        labels = self._merge_labels(
            engine=engine_name,
            operation="profile",
            status=status,
        )

        # Duration
        if self._profile_duration:
            self._profile_duration.observe(duration_ms / 1000.0, labels=labels)

        # Operation count
        if self._operations_total:
            self._operations_total.inc(labels=labels)

        # Active operations
        profile_labels = self._merge_labels(engine=engine_name, operation="profile")
        if self._active_operations:
            self._active_operations.dec(labels=profile_labels)

        # Data size
        if self._config.include_data_size and result.row_count and self._rows_processed:
            self._rows_processed.inc(
                result.row_count,
                labels=self._merge_labels(engine=engine_name, operation="profile"),
            )

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record learn start metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()
        labels = self._merge_labels(engine=engine_name, operation="learn")

        if self._active_operations:
            self._active_operations.inc(labels=labels)

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record learn completion metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()

        status = "success" if result.is_success else "failure"
        labels = self._merge_labels(
            engine=engine_name,
            operation="learn",
            status=status,
        )

        # Duration
        if self._learn_duration:
            self._learn_duration.observe(duration_ms / 1000.0, labels=labels)

        # Operation count
        if self._operations_total:
            self._operations_total.inc(labels=labels)

        # Active operations
        learn_labels = self._merge_labels(engine=engine_name, operation="learn")
        if self._active_operations:
            self._active_operations.dec(labels=learn_labels)

    def on_drift_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record drift detection start metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()
        labels = self._merge_labels(engine=engine_name, operation="drift")

        if self._active_operations:
            self._active_operations.inc(labels=labels)

    def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record drift detection completion metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()

        status = "drift_detected" if result.is_drifted else "no_drift"
        labels = self._merge_labels(
            engine=engine_name,
            operation="drift",
            status=status,
        )

        if self._drift_duration:
            self._drift_duration.observe(duration_ms / 1000.0, labels=labels)

        if self._operations_total:
            self._operations_total.inc(labels=labels)

        drift_labels = self._merge_labels(engine=engine_name, operation="drift")
        if self._active_operations:
            self._active_operations.dec(labels=drift_labels)

        if self._config.include_result_counts and result.is_drifted:
            engine_labels = self._merge_labels(engine=engine_name)
            if self._drift_detected_total:
                self._drift_detected_total.inc(
                    result.drifted_count, labels=engine_labels
                )

    def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record anomaly detection start metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()
        labels = self._merge_labels(engine=engine_name, operation="anomaly")

        if self._active_operations:
            self._active_operations.inc(labels=labels)

    def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record anomaly detection completion metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()

        status = "anomaly_detected" if result.has_anomalies else "normal"
        labels = self._merge_labels(
            engine=engine_name,
            operation="anomaly",
            status=status,
        )

        if self._anomaly_duration:
            self._anomaly_duration.observe(duration_ms / 1000.0, labels=labels)

        if self._operations_total:
            self._operations_total.inc(labels=labels)

        anomaly_labels = self._merge_labels(engine=engine_name, operation="anomaly")
        if self._active_operations:
            self._active_operations.dec(labels=anomaly_labels)

        if self._config.include_result_counts and result.has_anomalies:
            engine_labels = self._merge_labels(engine=engine_name)
            if self._anomaly_detected_total:
                self._anomaly_detected_total.inc(
                    result.anomalous_row_count, labels=engine_labels
                )

    def on_stream_check_start(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record streaming check start metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()
        labels = self._merge_labels(engine=engine_name, operation="stream_check")

        if self._active_operations:
            self._active_operations.inc(labels=labels)

    def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record streaming check completion metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()

        labels = self._merge_labels(
            engine=engine_name,
            operation="stream_check",
            status="success",
        )

        if self._stream_check_duration:
            self._stream_check_duration.observe(duration_ms / 1000.0, labels=labels)

        if self._operations_total:
            self._operations_total.inc(labels=labels)

        stream_labels = self._merge_labels(engine=engine_name, operation="stream_check")
        if self._active_operations:
            self._active_operations.dec(labels=stream_labels)

        if self._config.include_result_counts and self._stream_batches_total:
            engine_labels = self._merge_labels(engine=engine_name)
            self._stream_batches_total.inc(batch_count, labels=engine_labels)

    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record error metrics."""
        if not self._config.enabled:
            return

        self._ensure_metrics_initialized()

        op_name = operation.name.lower()
        labels = self._merge_labels(
            engine=engine_name,
            operation=op_name,
            status="error",
        )

        # Operation count with error status
        if self._operations_total:
            self._operations_total.inc(labels=labels)

        # Error counter with error type
        error_labels = self._merge_labels(
            engine=engine_name,
            operation=op_name,
            error_type=type(exception).__name__,
        )
        if self._errors_total:
            self._errors_total.inc(labels=error_labels)

        # Active operations
        active_labels = self._merge_labels(engine=engine_name, operation=op_name)
        if self._active_operations:
            self._active_operations.dec(labels=active_labels)


# =============================================================================
# Logging Hook Implementation
# =============================================================================


class LoggingEngineHook(BaseEngineMetricsHook):
    """Structured logging for engine operations.

    Logs start, end, and error events with structured context.
    Uses the common.logging infrastructure for consistent formatting.

    Log Levels:
    - DEBUG: Operation start events
    - INFO: Operation end events (success)
    - WARNING: Operation end events (failure result, not exception)
    - ERROR: Operation error events (exception raised)

    Example:
        >>> hook = LoggingEngineHook()
        >>> engine = InstrumentedEngine(TruthoundEngine(), hooks=[hook])
        >>> result = engine.check(data)
        >>> # Logs: INFO "Check completed" engine="truthound" status="PASSED" ...
    """

    def __init__(
        self,
        logger_name: str | None = None,
        log_start: bool = True,
        log_end: bool = True,
    ) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.engines.metrics).
            log_start: Whether to log start events.
            log_end: Whether to log end events.
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.engines.metrics")
        self._log_start = log_start
        self._log_end = log_end

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Log check start."""
        if not self._log_start:
            return

        self._logger.debug(
            "Check starting",
            engine=engine_name,
            operation="check",
            data_size=data_size,
            **{k: v for k, v in context.items() if k not in ("kwargs",)},
        )

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log check completion."""
        if not self._log_end:
            return

        log_method = self._logger.info if result.is_success else self._logger.warning
        log_method(
            "Check completed",
            engine=engine_name,
            operation="check",
            status=result.status.name,
            passed_count=result.passed_count,
            failed_count=result.failed_count,
            warning_count=result.warning_count,
            duration_ms=round(duration_ms, 2),
            pass_rate=round(result.pass_rate, 2),
        )

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Log profile start."""
        if not self._log_start:
            return

        self._logger.debug(
            "Profile starting",
            engine=engine_name,
            operation="profile",
            data_size=data_size,
        )

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log profile completion."""
        if not self._log_end:
            return

        log_method = self._logger.info if result.is_success else self._logger.warning
        log_method(
            "Profile completed",
            engine=engine_name,
            operation="profile",
            status=result.status.name,
            row_count=result.row_count,
            column_count=result.column_count,
            duration_ms=round(duration_ms, 2),
        )

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Log learn start."""
        if not self._log_start:
            return

        self._logger.debug(
            "Learn starting",
            engine=engine_name,
            operation="learn",
            data_size=data_size,
        )

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log learn completion."""
        if not self._log_end:
            return

        log_method = self._logger.info if result.is_success else self._logger.warning
        log_method(
            "Learn completed",
            engine=engine_name,
            operation="learn",
            status=result.status.name,
            rules_learned=len(result.rules),
            columns_analyzed=result.columns_analyzed,
            duration_ms=round(duration_ms, 2),
        )

    def on_drift_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Log drift detection start."""
        if not self._log_start:
            return

        self._logger.debug(
            "Drift detection starting",
            engine=engine_name,
            operation="drift",
            data_size=data_size,
            method=context.get("method"),
        )

    def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log drift detection completion."""
        if not self._log_end:
            return

        log_method = self._logger.warning if result.is_drifted else self._logger.info
        log_method(
            "Drift detection completed",
            engine=engine_name,
            operation="drift",
            status=result.status.name,
            is_drifted=result.is_drifted,
            drifted_columns=result.drifted_count,
            total_columns=result.total_columns,
            drift_rate=round(result.drift_rate, 4),
            duration_ms=round(duration_ms, 2),
        )

    def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Log anomaly detection start."""
        if not self._log_start:
            return

        self._logger.debug(
            "Anomaly detection starting",
            engine=engine_name,
            operation="anomaly",
            data_size=data_size,
            detector=context.get("detector"),
        )

    def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log anomaly detection completion."""
        if not self._log_end:
            return

        log_method = self._logger.warning if result.has_anomalies else self._logger.info
        log_method(
            "Anomaly detection completed",
            engine=engine_name,
            operation="anomaly",
            status=result.status.name,
            has_anomalies=result.has_anomalies,
            anomalous_rows=result.anomalous_row_count,
            total_rows=result.total_row_count,
            anomaly_rate=round(result.anomaly_rate, 4),
            duration_ms=round(duration_ms, 2),
        )

    def on_stream_check_start(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Log streaming check start."""
        if not self._log_start:
            return

        self._logger.debug(
            "Streaming check starting",
            engine=engine_name,
            operation="stream_check",
            batch_size=context.get("batch_size"),
        )

    def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log streaming check completion."""
        if not self._log_end:
            return

        self._logger.info(
            "Streaming check completed",
            engine=engine_name,
            operation="stream_check",
            batch_count=batch_count,
            duration_ms=round(duration_ms, 2),
        )

    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Log operation error."""
        self._logger.error(
            "Engine operation failed",
            engine=engine_name,
            operation=operation.name.lower(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            duration_ms=round(duration_ms, 2),
        )


# =============================================================================
# Tracing Hook Implementation
# =============================================================================


class TracingEngineHook(BaseEngineMetricsHook):
    """Distributed tracing for engine operations.

    Creates spans for each engine operation with attributes for:
    - Engine name and version
    - Operation type
    - Data size
    - Result status
    - Error information

    Spans are automatically linked to parent spans if available.

    Example:
        >>> hook = TracingEngineHook()
        >>> engine = InstrumentedEngine(TruthoundEngine(), hooks=[hook])
        >>> result = engine.check(data)
        >>> # Creates span: "engine.check" with attributes
    """

    def __init__(
        self,
        config: EngineMetricsConfig | None = None,
        registry: TracingRegistry | None = None,
    ) -> None:
        """Initialize tracing hook.

        Args:
            config: Metrics configuration.
            registry: Tracing registry (uses global if None).
        """
        from common.metrics import get_tracing_registry

        self._config = config or DEFAULT_ENGINE_METRICS_CONFIG
        self._registry = registry or get_tracing_registry()
        self._active_spans: dict[str, Span] = {}
        self._lock = threading.Lock()

    def _get_span_name(self, operation: str) -> str:
        """Get span name for operation."""
        if self._config.prefix:
            return f"{self._config.prefix}.{operation}"
        return f"engine.{operation}"

    def _get_span_key(self, engine_name: str, operation: str) -> str:
        """Get unique key for tracking active spans."""
        thread_id = threading.current_thread().ident
        return f"{engine_name}:{operation}:{thread_id}"

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Start check span."""
        if not self._config.include_tracing:
            return

        from common.metrics import SpanKind

        span = self._registry.start_span(
            self._get_span_name("check"),
            kind=SpanKind.INTERNAL,
            attributes={
                "engine.name": engine_name,
                "engine.operation": "check",
                **({"data.size": data_size} if data_size else {}),
            },
        )

        key = self._get_span_key(engine_name, "check")
        with self._lock:
            self._active_spans[key] = span

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """End check span."""
        if not self._config.include_tracing:
            return

        key = self._get_span_key(engine_name, "check")
        with self._lock:
            span = self._active_spans.pop(key, None)

        if span:
            from common.metrics import SpanStatus

            span.set_attribute("check.status", result.status.name)
            span.set_attribute("check.passed_count", result.passed_count)
            span.set_attribute("check.failed_count", result.failed_count)
            span.set_attribute("check.pass_rate", round(result.pass_rate, 2))
            span.set_attribute("duration_ms", round(duration_ms, 2))

            status = SpanStatus.OK if result.is_success else SpanStatus.ERROR
            span.set_status(status)
            span.end()

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Start profile span."""
        if not self._config.include_tracing:
            return

        from common.metrics import SpanKind

        span = self._registry.start_span(
            self._get_span_name("profile"),
            kind=SpanKind.INTERNAL,
            attributes={
                "engine.name": engine_name,
                "engine.operation": "profile",
                **({"data.size": data_size} if data_size else {}),
            },
        )

        key = self._get_span_key(engine_name, "profile")
        with self._lock:
            self._active_spans[key] = span

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """End profile span."""
        if not self._config.include_tracing:
            return

        key = self._get_span_key(engine_name, "profile")
        with self._lock:
            span = self._active_spans.pop(key, None)

        if span:
            from common.metrics import SpanStatus

            span.set_attribute("profile.status", result.status.name)
            span.set_attribute("profile.row_count", result.row_count)
            span.set_attribute("profile.column_count", result.column_count)
            span.set_attribute("duration_ms", round(duration_ms, 2))

            status = SpanStatus.OK if result.is_success else SpanStatus.ERROR
            span.set_status(status)
            span.end()

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Start learn span."""
        if not self._config.include_tracing:
            return

        from common.metrics import SpanKind

        span = self._registry.start_span(
            self._get_span_name("learn"),
            kind=SpanKind.INTERNAL,
            attributes={
                "engine.name": engine_name,
                "engine.operation": "learn",
                **({"data.size": data_size} if data_size else {}),
            },
        )

        key = self._get_span_key(engine_name, "learn")
        with self._lock:
            self._active_spans[key] = span

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """End learn span."""
        if not self._config.include_tracing:
            return

        key = self._get_span_key(engine_name, "learn")
        with self._lock:
            span = self._active_spans.pop(key, None)

        if span:
            from common.metrics import SpanStatus

            span.set_attribute("learn.status", result.status.name)
            span.set_attribute("learn.rules_count", len(result.rules))
            span.set_attribute("learn.columns_analyzed", result.columns_analyzed)
            span.set_attribute("duration_ms", round(duration_ms, 2))

            status = SpanStatus.OK if result.is_success else SpanStatus.ERROR
            span.set_status(status)
            span.end()

    def on_drift_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Start drift detection span."""
        if not self._config.include_tracing:
            return

        from common.metrics import SpanKind

        span = self._registry.start_span(
            self._get_span_name("drift"),
            kind=SpanKind.INTERNAL,
            attributes={
                "engine.name": engine_name,
                "engine.operation": "drift",
                **({"data.size": data_size} if data_size else {}),
                **({"drift.method": context["method"]} if "method" in context else {}),
            },
        )

        key = self._get_span_key(engine_name, "drift")
        with self._lock:
            self._active_spans[key] = span

    def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """End drift detection span."""
        if not self._config.include_tracing:
            return

        key = self._get_span_key(engine_name, "drift")
        with self._lock:
            span = self._active_spans.pop(key, None)

        if span:
            from common.metrics import SpanStatus

            span.set_attribute("drift.status", result.status.name)
            span.set_attribute("drift.is_drifted", result.is_drifted)
            span.set_attribute("drift.drifted_count", result.drifted_count)
            span.set_attribute("drift.total_columns", result.total_columns)
            span.set_attribute("drift.drift_rate", round(result.drift_rate, 4))
            span.set_attribute("duration_ms", round(duration_ms, 2))

            status = SpanStatus.OK if not result.is_drifted else SpanStatus.ERROR
            span.set_status(status)
            span.end()

    def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Start anomaly detection span."""
        if not self._config.include_tracing:
            return

        from common.metrics import SpanKind

        span = self._registry.start_span(
            self._get_span_name("anomaly"),
            kind=SpanKind.INTERNAL,
            attributes={
                "engine.name": engine_name,
                "engine.operation": "anomaly",
                **({"data.size": data_size} if data_size else {}),
                **({"anomaly.detector": context["detector"]} if "detector" in context else {}),
            },
        )

        key = self._get_span_key(engine_name, "anomaly")
        with self._lock:
            self._active_spans[key] = span

    def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """End anomaly detection span."""
        if not self._config.include_tracing:
            return

        key = self._get_span_key(engine_name, "anomaly")
        with self._lock:
            span = self._active_spans.pop(key, None)

        if span:
            from common.metrics import SpanStatus

            span.set_attribute("anomaly.status", result.status.name)
            span.set_attribute("anomaly.has_anomalies", result.has_anomalies)
            span.set_attribute("anomaly.anomalous_rows", result.anomalous_row_count)
            span.set_attribute("anomaly.total_rows", result.total_row_count)
            span.set_attribute("anomaly.anomaly_rate", round(result.anomaly_rate, 4))
            span.set_attribute("duration_ms", round(duration_ms, 2))

            status = SpanStatus.OK if not result.has_anomalies else SpanStatus.ERROR
            span.set_status(status)
            span.end()

    def on_stream_check_start(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Start streaming check span."""
        if not self._config.include_tracing:
            return

        from common.metrics import SpanKind

        span = self._registry.start_span(
            self._get_span_name("stream_check"),
            kind=SpanKind.INTERNAL,
            attributes={
                "engine.name": engine_name,
                "engine.operation": "stream_check",
                **({"stream.batch_size": context["batch_size"]} if "batch_size" in context else {}),
            },
        )

        key = self._get_span_key(engine_name, "stream_check")
        with self._lock:
            self._active_spans[key] = span

    def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """End streaming check span."""
        if not self._config.include_tracing:
            return

        key = self._get_span_key(engine_name, "stream_check")
        with self._lock:
            span = self._active_spans.pop(key, None)

        if span:
            from common.metrics import SpanStatus

            span.set_attribute("stream.batch_count", batch_count)
            span.set_attribute("duration_ms", round(duration_ms, 2))

            span.set_status(SpanStatus.OK)
            span.end()

    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """End span with error."""
        if not self._config.include_tracing:
            return

        op_name = operation.name.lower()
        key = self._get_span_key(engine_name, op_name)
        with self._lock:
            span = self._active_spans.pop(key, None)

        if span:
            span.record_exception(exception)
            span.set_attribute("duration_ms", round(duration_ms, 2))
            span.end()


# =============================================================================
# Composite Hook
# =============================================================================


class CompositeEngineHook(BaseEngineMetricsHook):
    """Combines multiple engine hooks.

    Calls all hooks in order, suppressing exceptions from individual hooks
    to prevent one hook from breaking others.

    Example:
        >>> composite = CompositeEngineHook([
        ...     MetricsEngineHook(),
        ...     LoggingEngineHook(),
        ...     TracingEngineHook(),
        ... ])
        >>> engine = InstrumentedEngine(TruthoundEngine(), hooks=[composite])
    """

    def __init__(self, hooks: Sequence[EngineMetricsHook] | None = None) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks: list[EngineMetricsHook] = list(hooks or [])
        self._lock = threading.Lock()

    def add_hook(self, hook: EngineMetricsHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        with self._lock:
            self._hooks.append(hook)

    def remove_hook(self, hook: EngineMetricsHook) -> bool:
        """Remove a hook.

        Args:
            hook: Hook to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            try:
                self._hooks.remove(hook)
                return True
            except ValueError:
                return False

    @property
    def hooks(self) -> list[EngineMetricsHook]:
        """Get list of hooks."""
        with self._lock:
            return list(self._hooks)

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_check_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_check_start(engine_name, data_size, context)

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_check_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_check_end(engine_name, result, duration_ms, context)

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_profile_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_profile_start(engine_name, data_size, context)

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_profile_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_profile_end(engine_name, result, duration_ms, context)

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_learn_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_learn_start(engine_name, data_size, context)

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_learn_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_learn_end(engine_name, result, duration_ms, context)

    def on_drift_start(
        self,
        engine_name: str,
        baseline_size: int | None,
        current_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_drift_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_drift_start(engine_name, baseline_size, current_size, context)

    def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_drift_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_drift_end(engine_name, result, duration_ms, context)

    def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_anomaly_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_anomaly_start(engine_name, data_size, context)

    def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_anomaly_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_anomaly_end(engine_name, result, duration_ms, context)

    def on_stream_check_start(
        self,
        engine_name: str,
        batch_size: int,
        context: dict[str, Any],
    ) -> None:
        """Call on_stream_check_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_stream_check_start(engine_name, batch_size, context)

    def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_stream_check_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_stream_check_end(engine_name, batch_count, duration_ms, context)

    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_error on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.on_error(engine_name, operation, exception, duration_ms, context)


class AsyncCompositeEngineHook(AsyncBaseEngineMetricsHook):
    """Async version of CompositeEngineHook."""

    def __init__(self, hooks: Sequence[AsyncEngineMetricsHook] | None = None) -> None:
        """Initialize async composite hook.

        Args:
            hooks: List of async hooks to call.
        """
        self._hooks: list[AsyncEngineMetricsHook] = list(hooks or [])
        self._lock = threading.Lock()

    def add_hook(self, hook: AsyncEngineMetricsHook) -> None:
        """Add a hook."""
        with self._lock:
            self._hooks.append(hook)

    @property
    def hooks(self) -> list[AsyncEngineMetricsHook]:
        """Get list of hooks."""
        with self._lock:
            return list(self._hooks)

    async def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_check_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_check_start(engine_name, data_size, context)

    async def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_check_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_check_end(engine_name, result, duration_ms, context)

    async def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_profile_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_profile_start(engine_name, data_size, context)

    async def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_profile_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_profile_end(engine_name, result, duration_ms, context)

    async def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_learn_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_learn_start(engine_name, data_size, context)

    async def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_learn_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_learn_end(engine_name, result, duration_ms, context)

    async def on_drift_start(
        self,
        engine_name: str,
        baseline_size: int | None,
        current_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_drift_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_drift_start(engine_name, baseline_size, current_size, context)

    async def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_drift_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_drift_end(engine_name, result, duration_ms, context)

    async def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call on_anomaly_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_anomaly_start(engine_name, data_size, context)

    async def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_anomaly_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_anomaly_end(engine_name, result, duration_ms, context)

    async def on_stream_check_start(
        self,
        engine_name: str,
        batch_size: int,
        context: dict[str, Any],
    ) -> None:
        """Call on_stream_check_start on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_stream_check_start(engine_name, batch_size, context)

    async def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_stream_check_end on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_stream_check_end(engine_name, batch_count, duration_ms, context)

    async def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call on_error on all hooks."""
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                await hook.on_error(engine_name, operation, exception, duration_ms, context)


# =============================================================================
# Sync to Async Adapter
# =============================================================================


class SyncToAsyncEngineHookAdapter(AsyncBaseEngineMetricsHook):
    """Adapter to use sync hooks in async contexts.

    Wraps a sync EngineMetricsHook to be used as AsyncEngineMetricsHook.
    Runs sync methods in the default executor.

    Example:
        >>> sync_hook = MetricsEngineHook()
        >>> async_hook = SyncToAsyncEngineHookAdapter(sync_hook)
    """

    def __init__(self, sync_hook: EngineMetricsHook) -> None:
        """Initialize adapter.

        Args:
            sync_hook: Sync hook to wrap.
        """
        self._sync_hook = sync_hook

    async def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_check_start,
            engine_name,
            data_size,
            context,
        )

    async def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_check_end,
            engine_name,
            result,
            duration_ms,
            context,
        )

    async def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_profile_start,
            engine_name,
            data_size,
            context,
        )

    async def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_profile_end,
            engine_name,
            result,
            duration_ms,
            context,
        )

    async def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_learn_start,
            engine_name,
            data_size,
            context,
        )

    async def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_learn_end,
            engine_name,
            result,
            duration_ms,
            context,
        )

    async def on_drift_start(
        self,
        engine_name: str,
        baseline_size: int | None,
        current_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_drift_start,
            engine_name,
            baseline_size,
            current_size,
            context,
        )

    async def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_drift_end,
            engine_name,
            result,
            duration_ms,
            context,
        )

    async def on_anomaly_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_anomaly_start,
            engine_name,
            data_size,
            context,
        )

    async def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_anomaly_end,
            engine_name,
            result,
            duration_ms,
            context,
        )

    async def on_stream_check_start(
        self,
        engine_name: str,
        batch_size: int,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_stream_check_start,
            engine_name,
            batch_size,
            context,
        )

    async def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_stream_check_end,
            engine_name,
            batch_count,
            duration_ms,
            context,
        )

    async def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Call sync hook in executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._sync_hook.on_error,
            engine_name,
            operation,
            exception,
            duration_ms,
            context,
        )


# =============================================================================
# Instrumented Engine Wrapper
# =============================================================================


def _get_data_size(data: Any) -> int | None:
    """Attempt to get size of data."""
    if hasattr(data, "__len__"):
        try:
            return len(data)
        except Exception:
            pass
    if hasattr(data, "shape"):
        try:
            return data.shape[0]
        except Exception:
            pass
    if hasattr(data, "height"):
        try:
            return data.height
        except Exception:
            pass
    return None


class InstrumentedEngine:
    """Transparent wrapper that adds metrics to any DataQualityEngine.

    Wraps an existing engine and automatically collects metrics for all
    operations without modifying the engine's behavior.

    All original engine methods and properties are proxied through.

    Example:
        >>> from common.engines import TruthoundEngine
        >>> from common.engines.metrics import InstrumentedEngine, MetricsEngineHook
        >>>
        >>> base_engine = TruthoundEngine()
        >>> engine = InstrumentedEngine(
        ...     base_engine,
        ...     hooks=[MetricsEngineHook(), LoggingEngineHook()],
        ... )
        >>>
        >>> # Use like a normal engine
        >>> result = engine.check(data, auto_schema=True)

    With Context Manager:
        >>> with InstrumentedEngine(TruthoundEngine()) as engine:
        ...     result = engine.check(data)
    """

    def __init__(
        self,
        engine: DataQualityEngine,
        hooks: Sequence[EngineMetricsHook] | None = None,
        config: EngineMetricsConfig | None = None,
    ) -> None:
        """Initialize instrumented engine.

        Args:
            engine: Engine to wrap.
            hooks: Metrics hooks to use.
            config: Metrics configuration.
        """
        self._engine = engine
        self._config = config or DEFAULT_ENGINE_METRICS_CONFIG
        self._hooks: list[EngineMetricsHook] = list(hooks or [])

        # Create composite hook for efficient calling
        if len(self._hooks) > 1:
            self._composite_hook: EngineMetricsHook = CompositeEngineHook(self._hooks)
        elif len(self._hooks) == 1:
            self._composite_hook = self._hooks[0]
        else:
            self._composite_hook = BaseEngineMetricsHook()

    @property
    def engine(self) -> DataQualityEngine:
        """Get the wrapped engine."""
        return self._engine

    @property
    def engine_name(self) -> str:
        """Proxy engine_name property."""
        return self._engine.engine_name

    @property
    def engine_version(self) -> str:
        """Proxy engine_version property."""
        return self._engine.engine_version

    def add_hook(self, hook: EngineMetricsHook) -> None:
        """Add a metrics hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)
        # Rebuild composite
        if len(self._hooks) > 1:
            self._composite_hook = CompositeEngineHook(self._hooks)
        else:
            self._composite_hook = self._hooks[0]

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> CheckResult:
        """Execute check with metrics collection.

        Args:
            data: Data to validate.
            rules: Validation rules.
            **kwargs: Engine-specific parameters.

        Returns:
            CheckResult from the wrapped engine.
        """
        if not self._config.enabled:
            return self._engine.check(data, rules, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
            "rules_count": len(rules),
        }

        # Notify start
        with contextlib.suppress(Exception):
            self._composite_hook.on_check_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = self._engine.check(data, rules, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                self._composite_hook.on_check_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.CHECK,
                    e,
                    duration_ms,
                    context,
                )

            raise

    def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Execute profile with metrics collection.

        Args:
            data: Data to profile.
            **kwargs: Engine-specific parameters.

        Returns:
            ProfileResult from the wrapped engine.
        """
        if not self._config.enabled:
            return self._engine.profile(data, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
        }

        # Notify start
        with contextlib.suppress(Exception):
            self._composite_hook.on_profile_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = self._engine.profile(data, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                self._composite_hook.on_profile_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.PROFILE,
                    e,
                    duration_ms,
                    context,
                )

            raise

    def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Execute learn with metrics collection.

        Args:
            data: Data to learn from.
            **kwargs: Engine-specific parameters.

        Returns:
            LearnResult from the wrapped engine.
        """
        if not self._config.enabled:
            return self._engine.learn(data, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
        }

        # Notify start
        with contextlib.suppress(Exception):
            self._composite_hook.on_learn_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = self._engine.learn(data, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                self._composite_hook.on_learn_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.LEARN,
                    e,
                    duration_ms,
                    context,
                )

            raise

    def detect_drift(
        self,
        baseline: Any,
        current: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute drift detection with metrics collection.

        Args:
            baseline: Baseline data for comparison.
            current: Current data to check for drift.
            **kwargs: Engine-specific parameters.

        Returns:
            DriftResult from the wrapped engine.
        """
        if not self._config.enabled:
            return self._engine.detect_drift(baseline, current, **kwargs)

        data_size = _get_data_size(current)
        context: dict[str, Any] = {
            "data_type": type(current).__name__,
            "data_size": data_size,
            "method": kwargs.get("method"),
        }

        baseline_size = _get_data_size(baseline)

        # Notify start
        with contextlib.suppress(Exception):
            self._composite_hook.on_drift_start(
                self.engine_name, baseline_size, data_size, context
            )

        start_time = time.perf_counter()
        try:
            result = self._engine.detect_drift(baseline, current, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                self._composite_hook.on_drift_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.DRIFT,
                    e,
                    duration_ms,
                    context,
                )

            raise

    def detect_anomalies(
        self,
        data: Any,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Execute anomaly detection with metrics collection.

        Args:
            data: Data to check for anomalies.
            **kwargs: Engine-specific parameters.

        Returns:
            AnomalyResult from the wrapped engine.
        """
        if not self._config.enabled:
            return self._engine.detect_anomalies(data, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
            "detector": kwargs.get("detector"),
        }

        # Notify start
        with contextlib.suppress(Exception):
            self._composite_hook.on_anomaly_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = self._engine.detect_anomalies(data, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                self._composite_hook.on_anomaly_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.ANOMALY,
                    e,
                    duration_ms,
                    context,
                )

            raise

    def check_stream(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> Iterator[CheckResult]:
        """Execute streaming check with metrics collection.

        Args:
            data: Data to validate in streaming fashion.
            rules: Validation rules.
            **kwargs: Engine-specific parameters.

        Yields:
            CheckResult for each batch from the wrapped engine.
        """
        if not self._config.enabled:
            yield from self._engine.check_stream(data, rules, **kwargs)
            return

        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "batch_size": kwargs.get("batch_size"),
        }

        batch_size = kwargs.get("batch_size", 1000)

        # Notify start
        with contextlib.suppress(Exception):
            self._composite_hook.on_stream_check_start(
                self.engine_name, batch_size, context
            )

        start_time = time.perf_counter()
        batch_count = 0
        try:
            for batch_result in self._engine.check_stream(data, rules, **kwargs):
                batch_count += 1
                yield batch_result

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                self._composite_hook.on_stream_check_end(
                    self.engine_name, batch_count, duration_ms, context
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.STREAM_CHECK,
                    e,
                    duration_ms,
                    context,
                )

            raise

    # Proxy lifecycle methods if available
    def start(self) -> None:
        """Proxy start method if available."""
        if hasattr(self._engine, "start"):
            self._engine.start()

    def stop(self) -> None:
        """Proxy stop method if available."""
        if hasattr(self._engine, "stop"):
            self._engine.stop()

    def health_check(self) -> HealthCheckResult:
        """Proxy health_check method if available."""
        if hasattr(self._engine, "health_check"):
            return self._engine.health_check()
        raise NotImplementedError("Wrapped engine does not support health_check")

    def get_info(self) -> Any:
        """Proxy get_info method if available."""
        if hasattr(self._engine, "get_info"):
            return self._engine.get_info()
        raise NotImplementedError("Wrapped engine does not support get_info")

    def get_capabilities(self) -> Any:
        """Proxy get_capabilities method if available."""
        if hasattr(self._engine, "get_capabilities"):
            return self._engine.get_capabilities()
        raise NotImplementedError("Wrapped engine does not support get_capabilities")

    def __enter__(self) -> InstrumentedEngine:
        """Enter context manager."""
        if hasattr(self._engine, "__enter__"):
            self._engine.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if hasattr(self._engine, "__exit__"):
            self._engine.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attributes to the wrapped engine."""
        return getattr(self._engine, name)


# =============================================================================
# Async Instrumented Engine
# =============================================================================


class AsyncInstrumentedEngine:
    """Async version of InstrumentedEngine for async engines.

    Wraps an async engine and automatically collects metrics for all
    async operations.

    Example:
        >>> from common.engines import SyncEngineAsyncAdapter, TruthoundEngine
        >>> from common.engines.metrics import AsyncInstrumentedEngine
        >>>
        >>> base_engine = SyncEngineAsyncAdapter(TruthoundEngine())
        >>> engine = AsyncInstrumentedEngine(base_engine, hooks=[...])
        >>>
        >>> async with engine:
        ...     result = await engine.check(data)
    """

    def __init__(
        self,
        engine: Any,  # AsyncDataQualityEngine
        hooks: Sequence[AsyncEngineMetricsHook] | None = None,
        config: EngineMetricsConfig | None = None,
    ) -> None:
        """Initialize async instrumented engine.

        Args:
            engine: Async engine to wrap.
            hooks: Async metrics hooks to use.
            config: Metrics configuration.
        """
        self._engine = engine
        self._config = config or DEFAULT_ENGINE_METRICS_CONFIG
        self._hooks: list[AsyncEngineMetricsHook] = list(hooks or [])

        # Create composite hook
        if len(self._hooks) > 1:
            self._composite_hook: AsyncEngineMetricsHook = AsyncCompositeEngineHook(self._hooks)
        elif len(self._hooks) == 1:
            self._composite_hook = self._hooks[0]
        else:
            self._composite_hook = AsyncBaseEngineMetricsHook()

    @property
    def engine(self) -> Any:
        """Get the wrapped engine."""
        return self._engine

    @property
    def engine_name(self) -> str:
        """Proxy engine_name property."""
        return self._engine.engine_name

    @property
    def engine_version(self) -> str:
        """Proxy engine_version property."""
        return self._engine.engine_version

    async def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> CheckResult:
        """Execute async check with metrics collection."""
        if not self._config.enabled:
            return await self._engine.check(data, rules, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
            "rules_count": len(rules),
        }

        # Notify start
        with contextlib.suppress(Exception):
            await self._composite_hook.on_check_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = await self._engine.check(data, rules, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                await self._composite_hook.on_check_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                await self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.CHECK,
                    e,
                    duration_ms,
                    context,
                )

            raise

    async def profile(
        self,
        data: Any,
        **kwargs: Any,
    ) -> ProfileResult:
        """Execute async profile with metrics collection."""
        if not self._config.enabled:
            return await self._engine.profile(data, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
        }

        # Notify start
        with contextlib.suppress(Exception):
            await self._composite_hook.on_profile_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = await self._engine.profile(data, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                await self._composite_hook.on_profile_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                await self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.PROFILE,
                    e,
                    duration_ms,
                    context,
                )

            raise

    async def learn(
        self,
        data: Any,
        **kwargs: Any,
    ) -> LearnResult:
        """Execute async learn with metrics collection."""
        if not self._config.enabled:
            return await self._engine.learn(data, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
        }

        # Notify start
        with contextlib.suppress(Exception):
            await self._composite_hook.on_learn_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = await self._engine.learn(data, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify end
            with contextlib.suppress(Exception):
                await self._composite_hook.on_learn_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Notify error
            with contextlib.suppress(Exception):
                await self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.LEARN,
                    e,
                    duration_ms,
                    context,
                )

            raise

    async def detect_drift(
        self,
        baseline: Any,
        current: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute async drift detection with metrics collection."""
        if not self._config.enabled:
            return await self._engine.detect_drift(baseline, current, **kwargs)

        data_size = _get_data_size(current)
        baseline_size = _get_data_size(baseline)
        context: dict[str, Any] = {
            "data_type": type(current).__name__,
            "data_size": data_size,
            "method": kwargs.get("method"),
        }

        with contextlib.suppress(Exception):
            await self._composite_hook.on_drift_start(
                self.engine_name, baseline_size, data_size, context
            )

        start_time = time.perf_counter()
        try:
            result = await self._engine.detect_drift(baseline, current, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            with contextlib.suppress(Exception):
                await self._composite_hook.on_drift_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            with contextlib.suppress(Exception):
                await self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.DRIFT,
                    e,
                    duration_ms,
                    context,
                )

            raise

    async def detect_anomalies(
        self,
        data: Any,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Execute async anomaly detection with metrics collection."""
        if not self._config.enabled:
            return await self._engine.detect_anomalies(data, **kwargs)

        data_size = _get_data_size(data)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "data_size": data_size,
            "detector": kwargs.get("detector"),
        }

        with contextlib.suppress(Exception):
            await self._composite_hook.on_anomaly_start(self.engine_name, data_size, context)

        start_time = time.perf_counter()
        try:
            result = await self._engine.detect_anomalies(data, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            with contextlib.suppress(Exception):
                await self._composite_hook.on_anomaly_end(
                    self.engine_name, result, duration_ms, context
                )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            with contextlib.suppress(Exception):
                await self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.ANOMALY,
                    e,
                    duration_ms,
                    context,
                )

            raise

    async def check_stream(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> Any:
        """Execute async streaming check with metrics collection."""
        if not self._config.enabled:
            async for batch_result in self._engine.check_stream(data, rules, **kwargs):
                yield batch_result
            return

        batch_size = kwargs.get("batch_size", 1000)
        context: dict[str, Any] = {
            "data_type": type(data).__name__,
            "batch_size": batch_size,
        }

        with contextlib.suppress(Exception):
            await self._composite_hook.on_stream_check_start(
                self.engine_name, batch_size, context
            )

        start_time = time.perf_counter()
        batch_count = 0
        try:
            async for batch_result in self._engine.check_stream(data, rules, **kwargs):
                batch_count += 1
                yield batch_result

            duration_ms = (time.perf_counter() - start_time) * 1000

            with contextlib.suppress(Exception):
                await self._composite_hook.on_stream_check_end(
                    self.engine_name, batch_count, duration_ms, context
                )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            with contextlib.suppress(Exception):
                await self._composite_hook.on_error(
                    self.engine_name,
                    EngineOperation.STREAM_CHECK,
                    e,
                    duration_ms,
                    context,
                )

            raise

    async def __aenter__(self) -> AsyncInstrumentedEngine:
        """Enter async context manager."""
        if hasattr(self._engine, "__aenter__"):
            await self._engine.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        if hasattr(self._engine, "__aexit__"):
            await self._engine.__aexit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        """Proxy any other attributes to the wrapped engine."""
        return getattr(self._engine, name)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_instrumented_engine(
    engine: DataQualityEngine,
    *,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    enable_tracing: bool = False,
    config: EngineMetricsConfig | None = None,
) -> InstrumentedEngine:
    """Create an instrumented engine with common hooks.

    Factory function for creating an InstrumentedEngine with a
    combination of commonly-used hooks.

    Args:
        engine: Engine to wrap.
        enable_metrics: Whether to add MetricsEngineHook.
        enable_logging: Whether to add LoggingEngineHook.
        enable_tracing: Whether to add TracingEngineHook.
        config: Metrics configuration.

    Returns:
        InstrumentedEngine with configured hooks.

    Example:
        >>> engine = create_instrumented_engine(
        ...     TruthoundEngine(),
        ...     enable_metrics=True,
        ...     enable_logging=True,
        ... )
    """
    hooks: list[EngineMetricsHook] = []
    cfg = config or DEFAULT_ENGINE_METRICS_CONFIG

    if enable_metrics:
        hooks.append(MetricsEngineHook(config=cfg))
    if enable_logging:
        hooks.append(LoggingEngineHook())
    if enable_tracing:
        hooks.append(TracingEngineHook(config=cfg))

    return InstrumentedEngine(engine, hooks=hooks, config=cfg)


def create_async_instrumented_engine(
    engine: Any,  # AsyncDataQualityEngine
    *,
    enable_metrics: bool = True,
    enable_logging: bool = True,
    enable_tracing: bool = False,
    config: EngineMetricsConfig | None = None,
) -> AsyncInstrumentedEngine:
    """Create an async instrumented engine with common hooks.

    Factory function for creating an AsyncInstrumentedEngine with
    async-compatible hooks.

    Args:
        engine: Async engine to wrap.
        enable_metrics: Whether to add metrics hooks.
        enable_logging: Whether to add logging hooks.
        enable_tracing: Whether to add tracing hooks.
        config: Metrics configuration.

    Returns:
        AsyncInstrumentedEngine with configured hooks.
    """
    hooks: list[AsyncEngineMetricsHook] = []
    cfg = config or DEFAULT_ENGINE_METRICS_CONFIG

    if enable_metrics:
        hooks.append(SyncToAsyncEngineHookAdapter(MetricsEngineHook(config=cfg)))
    if enable_logging:
        hooks.append(SyncToAsyncEngineHookAdapter(LoggingEngineHook()))
    if enable_tracing:
        hooks.append(SyncToAsyncEngineHookAdapter(TracingEngineHook(config=cfg)))

    return AsyncInstrumentedEngine(engine, hooks=hooks, config=cfg)


# =============================================================================
# Statistics Collector
# =============================================================================


@dataclass
class EngineOperationStats:
    """Statistics for engine operations.

    Provides a snapshot of operation statistics collected by hooks.

    Attributes:
        total_operations: Total number of operations.
        successful_operations: Number of successful operations.
        failed_operations: Number of failed operations (failure result).
        error_operations: Number of error operations (exception).
        total_duration_ms: Total duration in milliseconds.
        total_rows_processed: Total rows processed.
        operation_counts: Counts by operation type.
        error_counts: Counts by error type.
    """

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    error_operations: int = 0
    total_duration_ms: float = 0.0
    total_rows_processed: int = 0
    operation_counts: dict[str, int] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)

    @property
    def average_duration_ms(self) -> float:
        """Calculate average operation duration."""
        if self.total_operations == 0:
            return 0.0
        return self.total_duration_ms / self.total_operations

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_operations == 0:
            return 100.0
        return (self.successful_operations / self.total_operations) * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        if self.total_operations == 0:
            return 0.0
        return (self.error_operations / self.total_operations) * 100

    @property
    def check_count(self) -> int:
        """Get total check operations count."""
        return self.operation_counts.get("check", 0)

    @property
    def profile_count(self) -> int:
        """Get total profile operations count."""
        return self.operation_counts.get("profile", 0)

    @property
    def learn_count(self) -> int:
        """Get total learn operations count."""
        return self.operation_counts.get("learn", 0)

    @property
    def drift_count(self) -> int:
        """Get total drift detection operations count."""
        return self.operation_counts.get("drift", 0)

    @property
    def anomaly_count(self) -> int:
        """Get total anomaly detection operations count."""
        return self.operation_counts.get("anomaly", 0)

    @property
    def stream_check_count(self) -> int:
        """Get total streaming check operations count."""
        return self.operation_counts.get("stream_check", 0)

    @property
    def check_success_rate(self) -> float:
        """Calculate check operation success rate as a percentage."""
        check_total = self.check_count
        if check_total == 0:
            return 100.0
        check_errors = self.operation_counts.get("check_errors", 0)
        return ((check_total - check_errors) / check_total) * 100

    @property
    def drift_success_rate(self) -> float:
        """Calculate drift detection success rate (no drift = success)."""
        drift_total = self.drift_count
        if drift_total == 0:
            return 100.0
        drift_errors = self.operation_counts.get("drift_errors", 0)
        return ((drift_total - drift_errors) / drift_total) * 100

    @property
    def anomaly_success_rate(self) -> float:
        """Calculate anomaly detection success rate (no anomaly = success)."""
        anomaly_total = self.anomaly_count
        if anomaly_total == 0:
            return 100.0
        anomaly_errors = self.operation_counts.get("anomaly_errors", 0)
        return ((anomaly_total - anomaly_errors) / anomaly_total) * 100


class StatsCollectorHook(BaseEngineMetricsHook):
    """Hook that collects operation statistics in memory.

    Useful for testing and debugging. Not recommended for production
    due to unbounded memory growth.

    Example:
        >>> hook = StatsCollectorHook()
        >>> engine = InstrumentedEngine(TruthoundEngine(), hooks=[hook])
        >>> result = engine.check(data)
        >>> print(hook.stats.success_rate)
    """

    def __init__(self) -> None:
        """Initialize stats collector."""
        self._lock = threading.Lock()
        self._stats = EngineOperationStats()

    @property
    def stats(self) -> EngineOperationStats:
        """Get current statistics."""
        with self._lock:
            return EngineOperationStats(
                total_operations=self._stats.total_operations,
                successful_operations=self._stats.successful_operations,
                failed_operations=self._stats.failed_operations,
                error_operations=self._stats.error_operations,
                total_duration_ms=self._stats.total_duration_ms,
                total_rows_processed=self._stats.total_rows_processed,
                operation_counts=dict(self._stats.operation_counts),
                error_counts=dict(self._stats.error_counts),
            )

    def reset(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = EngineOperationStats()

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record check statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_duration_ms += duration_ms
            self._stats.operation_counts["check"] = (
                self._stats.operation_counts.get("check", 0) + 1
            )

            if result.is_success:
                self._stats.successful_operations += 1
            else:
                self._stats.failed_operations += 1

            data_size = context.get("data_size", 0)
            if data_size:
                self._stats.total_rows_processed += data_size

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record profile statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_duration_ms += duration_ms
            self._stats.operation_counts["profile"] = (
                self._stats.operation_counts.get("profile", 0) + 1
            )

            if result.is_success:
                self._stats.successful_operations += 1
            else:
                self._stats.failed_operations += 1

            if result.row_count:
                self._stats.total_rows_processed += result.row_count

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record learn statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_duration_ms += duration_ms
            self._stats.operation_counts["learn"] = (
                self._stats.operation_counts.get("learn", 0) + 1
            )

            if result.is_success:
                self._stats.successful_operations += 1
            else:
                self._stats.failed_operations += 1

    def on_drift_end(
        self,
        engine_name: str,
        result: DriftResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record drift detection statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_duration_ms += duration_ms
            self._stats.operation_counts["drift"] = (
                self._stats.operation_counts.get("drift", 0) + 1
            )

            if not result.is_drifted:
                self._stats.successful_operations += 1
            else:
                self._stats.failed_operations += 1

    def on_anomaly_end(
        self,
        engine_name: str,
        result: AnomalyResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record anomaly detection statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_duration_ms += duration_ms
            self._stats.operation_counts["anomaly"] = (
                self._stats.operation_counts.get("anomaly", 0) + 1
            )

            if not result.has_anomalies:
                self._stats.successful_operations += 1
            else:
                self._stats.failed_operations += 1

    def on_stream_check_end(
        self,
        engine_name: str,
        batch_count: int,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record streaming check statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_duration_ms += duration_ms
            self._stats.operation_counts["stream_check"] = (
                self._stats.operation_counts.get("stream_check", 0) + 1
            )
            self._stats.successful_operations += 1

    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record error statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.error_operations += 1
            self._stats.total_duration_ms += duration_ms

            op_name = operation.name.lower()
            self._stats.operation_counts[op_name] = (
                self._stats.operation_counts.get(op_name, 0) + 1
            )

            error_type = type(exception).__name__
            self._stats.error_counts[error_type] = (
                self._stats.error_counts.get(error_type, 0) + 1
            )
