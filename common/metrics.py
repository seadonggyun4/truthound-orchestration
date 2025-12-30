"""Metrics and Tracing utilities for Truthound Integrations.

This module provides a flexible, extensible metrics and distributed tracing system
designed for observability in distributed data quality workflows. It supports:

- Multiple metric types (Counter, Gauge, Histogram, Summary)
- Distributed tracing with span context propagation
- Configurable exporters (Console, JSON, Prometheus, OTLP)
- Platform-specific adapters (Airflow, Dagster, Prefect)
- Pre/post hooks for custom metric collection
- Async and sync function support
- Labels/tags for metric dimensions

Design Principles:
    1. Protocol-based: Easy to extend with custom collectors and exporters
    2. Immutable Config: Thread-safe configuration using frozen dataclass
    3. Observable: Hook system for monitoring metric events
    4. Composable: Works well with logging, retry, and other patterns
    5. Backend Agnostic: Pluggable exporters for different observability backends

Example:
    >>> from common.metrics import Counter, Histogram, trace
    >>> requests = Counter("requests_total", "Total request count")
    >>> requests.inc()
    >>> requests.inc(labels={"method": "POST", "endpoint": "/api/check"})

    >>> latency = Histogram("request_duration_seconds", "Request duration")
    >>> with latency.time():
    ...     process_request()

    >>> @trace(name="validate_data")
    ... def validate(data):
    ...     return truthound.check(data)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import random
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    from collections.abc import Callable, Generator, Iterator, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class MetricsError(TruthoundIntegrationError):
    """Base exception for metrics errors.

    Attributes:
        metric_name: Name of the metric.
        metric_type: Type of the metric.
    """

    def __init__(
        self,
        message: str,
        *,
        metric_name: str | None = None,
        metric_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize metrics error.

        Args:
            message: Human-readable error description.
            metric_name: Name of the metric.
            metric_type: Type of the metric.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if metric_name:
            details["metric_name"] = metric_name
        if metric_type:
            details["metric_type"] = metric_type
        super().__init__(message, details=details, cause=cause)
        self.metric_name = metric_name
        self.metric_type = metric_type


class TracingError(TruthoundIntegrationError):
    """Base exception for tracing errors.

    Attributes:
        span_name: Name of the span.
        trace_id: Trace identifier.
    """

    def __init__(
        self,
        message: str,
        *,
        span_name: str | None = None,
        trace_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize tracing error.

        Args:
            message: Human-readable error description.
            span_name: Name of the span.
            trace_id: Trace identifier.
            details: Optional dictionary with additional error context.
            cause: Optional original exception that caused this error.
        """
        details = details or {}
        if span_name:
            details["span_name"] = span_name
        if trace_id:
            details["trace_id"] = trace_id
        super().__init__(message, details=details, cause=cause)
        self.span_name = span_name
        self.trace_id = trace_id


# =============================================================================
# Enums
# =============================================================================


class MetricType(Enum):
    """Types of metrics.

    Attributes:
        COUNTER: Monotonically increasing counter.
        GAUGE: Value that can go up or down.
        HISTOGRAM: Distribution of values in configurable buckets.
        SUMMARY: Statistical summary with quantiles.
    """

    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


class SpanKind(Enum):
    """Kind of span in distributed tracing.

    Attributes:
        INTERNAL: Internal operation.
        SERVER: Server-side of a synchronous request.
        CLIENT: Client-side of a synchronous request.
        PRODUCER: Producer of an asynchronous message.
        CONSUMER: Consumer of an asynchronous message.
    """

    INTERNAL = auto()
    SERVER = auto()
    CLIENT = auto()
    PRODUCER = auto()
    CONSUMER = auto()


class SpanStatus(Enum):
    """Status of a span.

    Attributes:
        UNSET: Status not set.
        OK: Operation completed successfully.
        ERROR: Operation failed.
    """

    UNSET = auto()
    OK = auto()
    ERROR = auto()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class MetricExporter(Protocol):
    """Protocol for metric exporters.

    Implement this protocol to export metrics to different backends.
    """

    @abstractmethod
    def export(self, metrics: Sequence[MetricData]) -> None:
        """Export metrics to backend.

        Args:
            metrics: Sequence of metric data to export.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter and flush any pending metrics."""
        ...


@runtime_checkable
class SpanExporter(Protocol):
    """Protocol for span exporters.

    Implement this protocol to export spans to different backends.
    """

    @abstractmethod
    def export(self, spans: Sequence[SpanData]) -> None:
        """Export spans to backend.

        Args:
            spans: Sequence of span data to export.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter and flush any pending spans."""
        ...


@runtime_checkable
class MetricsHook(Protocol):
    """Protocol for metrics event hooks.

    Implement this to receive notifications about metric events.
    """

    @abstractmethod
    def on_record(
        self,
        metric_name: str,
        metric_type: MetricType,
        value: float,
        labels: dict[str, str],
        context: dict[str, Any],
    ) -> None:
        """Called when a metric value is recorded.

        Args:
            metric_name: Name of the metric.
            metric_type: Type of the metric.
            value: Recorded value.
            labels: Metric labels.
            context: Additional context.
        """
        ...


@runtime_checkable
class TracingHook(Protocol):
    """Protocol for tracing event hooks.

    Implement this to receive notifications about span events.
    """

    @abstractmethod
    def on_span_start(
        self,
        span: Span,
        context: dict[str, Any],
    ) -> None:
        """Called when a span starts.

        Args:
            span: The span that started.
            context: Additional context.
        """
        ...

    @abstractmethod
    def on_span_end(
        self,
        span: Span,
        context: dict[str, Any],
    ) -> None:
        """Called when a span ends.

        Args:
            span: The span that ended.
            context: Additional context.
        """
        ...


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Configuration for metrics collection.

    Immutable configuration object for metrics operations.
    Use builder methods to create modified copies.

    Attributes:
        enabled: Whether metrics collection is enabled.
        prefix: Prefix for all metric names.
        default_labels: Default labels applied to all metrics.
        histogram_buckets: Default histogram bucket boundaries.
        summary_quantiles: Default summary quantiles.
        export_interval_seconds: Interval for exporting metrics.
        max_export_batch_size: Maximum batch size for export.

    Example:
        >>> config = MetricsConfig(
        ...     prefix="truthound",
        ...     default_labels={"service": "orchestration"},
        ... )
        >>> disabled_config = config.with_enabled(False)
    """

    enabled: bool = True
    prefix: str = ""
    default_labels: dict[str, str] = field(default_factory=dict)
    histogram_buckets: tuple[float, ...] = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    )
    summary_quantiles: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)
    export_interval_seconds: float = 60.0
    max_export_batch_size: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.export_interval_seconds < 0:
            raise ValueError("export_interval_seconds must be non-negative")
        if self.max_export_batch_size < 1:
            raise ValueError("max_export_batch_size must be at least 1")

    def with_enabled(self, enabled: bool) -> MetricsConfig:
        """Create config with new enabled state.

        Args:
            enabled: New enabled state.

        Returns:
            New MetricsConfig with updated value.
        """
        return MetricsConfig(
            enabled=enabled,
            prefix=self.prefix,
            default_labels=self.default_labels,
            histogram_buckets=self.histogram_buckets,
            summary_quantiles=self.summary_quantiles,
            export_interval_seconds=self.export_interval_seconds,
            max_export_batch_size=self.max_export_batch_size,
        )

    def with_prefix(self, prefix: str) -> MetricsConfig:
        """Create config with new prefix.

        Args:
            prefix: New metric name prefix.

        Returns:
            New MetricsConfig with updated value.
        """
        return MetricsConfig(
            enabled=self.enabled,
            prefix=prefix,
            default_labels=self.default_labels,
            histogram_buckets=self.histogram_buckets,
            summary_quantiles=self.summary_quantiles,
            export_interval_seconds=self.export_interval_seconds,
            max_export_batch_size=self.max_export_batch_size,
        )

    def with_default_labels(self, **labels: str) -> MetricsConfig:
        """Create config with additional default labels.

        Args:
            **labels: Additional default labels.

        Returns:
            New MetricsConfig with merged labels.
        """
        return MetricsConfig(
            enabled=self.enabled,
            prefix=self.prefix,
            default_labels={**self.default_labels, **labels},
            histogram_buckets=self.histogram_buckets,
            summary_quantiles=self.summary_quantiles,
            export_interval_seconds=self.export_interval_seconds,
            max_export_batch_size=self.max_export_batch_size,
        )

    def with_histogram_buckets(self, *buckets: float) -> MetricsConfig:
        """Create config with custom histogram buckets.

        Args:
            *buckets: Bucket boundaries.

        Returns:
            New MetricsConfig with updated buckets.
        """
        return MetricsConfig(
            enabled=self.enabled,
            prefix=self.prefix,
            default_labels=self.default_labels,
            histogram_buckets=tuple(sorted(buckets)),
            summary_quantiles=self.summary_quantiles,
            export_interval_seconds=self.export_interval_seconds,
            max_export_batch_size=self.max_export_batch_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "prefix": self.prefix,
            "default_labels": self.default_labels,
            "histogram_buckets": list(self.histogram_buckets),
            "summary_quantiles": list(self.summary_quantiles),
            "export_interval_seconds": self.export_interval_seconds,
            "max_export_batch_size": self.max_export_batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create MetricsConfig from dictionary.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New MetricsConfig instance.
        """
        return cls(
            enabled=data.get("enabled", True),
            prefix=data.get("prefix", ""),
            default_labels=data.get("default_labels", {}),
            histogram_buckets=tuple(data.get(
                "histogram_buckets",
                (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )),
            summary_quantiles=tuple(data.get(
                "summary_quantiles",
                (0.5, 0.9, 0.95, 0.99),
            )),
            export_interval_seconds=data.get("export_interval_seconds", 60.0),
            max_export_batch_size=data.get("max_export_batch_size", 1000),
        )


@dataclass(frozen=True, slots=True)
class TracingConfig:
    """Configuration for distributed tracing.

    Immutable configuration object for tracing operations.
    Use builder methods to create modified copies.

    Attributes:
        enabled: Whether tracing is enabled.
        service_name: Name of the service for spans.
        sample_rate: Sampling rate (0.0 to 1.0).
        max_attributes: Maximum number of attributes per span.
        max_events: Maximum number of events per span.
        propagate_context: Whether to propagate context across boundaries.

    Example:
        >>> config = TracingConfig(
        ...     service_name="truthound-orchestration",
        ...     sample_rate=0.1,
        ... )
    """

    enabled: bool = True
    service_name: str = "truthound"
    sample_rate: float = 1.0
    max_attributes: int = 128
    max_events: int = 128
    propagate_context: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        if self.max_attributes < 1:
            raise ValueError("max_attributes must be at least 1")
        if self.max_events < 1:
            raise ValueError("max_events must be at least 1")

    def with_enabled(self, enabled: bool) -> TracingConfig:
        """Create config with new enabled state.

        Args:
            enabled: New enabled state.

        Returns:
            New TracingConfig with updated value.
        """
        return TracingConfig(
            enabled=enabled,
            service_name=self.service_name,
            sample_rate=self.sample_rate,
            max_attributes=self.max_attributes,
            max_events=self.max_events,
            propagate_context=self.propagate_context,
        )

    def with_service_name(self, service_name: str) -> TracingConfig:
        """Create config with new service name.

        Args:
            service_name: New service name.

        Returns:
            New TracingConfig with updated value.
        """
        return TracingConfig(
            enabled=self.enabled,
            service_name=service_name,
            sample_rate=self.sample_rate,
            max_attributes=self.max_attributes,
            max_events=self.max_events,
            propagate_context=self.propagate_context,
        )

    def with_sample_rate(self, sample_rate: float) -> TracingConfig:
        """Create config with new sample rate.

        Args:
            sample_rate: New sample rate (0.0 to 1.0).

        Returns:
            New TracingConfig with updated value.
        """
        return TracingConfig(
            enabled=self.enabled,
            service_name=self.service_name,
            sample_rate=sample_rate,
            max_attributes=self.max_attributes,
            max_events=self.max_events,
            propagate_context=self.propagate_context,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "service_name": self.service_name,
            "sample_rate": self.sample_rate,
            "max_attributes": self.max_attributes,
            "max_events": self.max_events,
            "propagate_context": self.propagate_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create TracingConfig from dictionary.

        Args:
            data: Dictionary with configuration data.

        Returns:
            New TracingConfig instance.
        """
        return cls(
            enabled=data.get("enabled", True),
            service_name=data.get("service_name", "truthound"),
            sample_rate=data.get("sample_rate", 1.0),
            max_attributes=data.get("max_attributes", 128),
            max_events=data.get("max_events", 128),
            propagate_context=data.get("propagate_context", True),
        )


# Default configurations
DEFAULT_METRICS_CONFIG = MetricsConfig()
DEFAULT_TRACING_CONFIG = TracingConfig()

DISABLED_METRICS_CONFIG = MetricsConfig(enabled=False)
DISABLED_TRACING_CONFIG = TracingConfig(enabled=False)

LOW_SAMPLE_TRACING_CONFIG = TracingConfig(sample_rate=0.1)
HIGH_CARDINALITY_METRICS_CONFIG = MetricsConfig(
    histogram_buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
)


# =============================================================================
# Data Types
# =============================================================================


def _utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


def _generate_id(length: int = 16) -> str:
    """Generate a random hex ID.

    Args:
        length: Number of bytes (result will be 2x in hex chars).

    Returns:
        Random hex string.
    """
    return "".join(f"{random.randint(0, 255):02x}" for _ in range(length))


def _generate_trace_id() -> str:
    """Generate a 128-bit trace ID."""
    return _generate_id(16)


def _generate_span_id() -> str:
    """Generate a 64-bit span ID."""
    return _generate_id(8)


@dataclass(frozen=True, slots=True)
class MetricData:
    """Data container for a metric observation.

    Attributes:
        name: Metric name.
        metric_type: Type of metric.
        value: Recorded value.
        labels: Metric labels/tags.
        timestamp: ISO format timestamp.
        unit: Optional unit of measurement.
        description: Optional metric description.
    """

    name: str
    metric_type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now_iso)
    unit: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.metric_type.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create MetricData from dictionary."""
        return cls(
            name=data["name"],
            metric_type=MetricType[data["type"]],
            value=data["value"],
            labels=data.get("labels", {}),
            timestamp=data.get("timestamp", _utc_now_iso()),
            unit=data.get("unit", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True, slots=True)
class SpanEvent:
    """An event within a span.

    Attributes:
        name: Event name.
        timestamp: ISO format timestamp.
        attributes: Event attributes.
    """

    name: str
    timestamp: str = field(default_factory=_utc_now_iso)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        }


@dataclass(frozen=True, slots=True)
class SpanData:
    """Data container for a span.

    Attributes:
        name: Span name.
        trace_id: Trace identifier.
        span_id: Span identifier.
        parent_span_id: Parent span identifier (if any).
        kind: Kind of span.
        status: Span status.
        start_time: Start timestamp.
        end_time: End timestamp (if ended).
        duration_ms: Duration in milliseconds.
        attributes: Span attributes.
        events: Events within the span.
        service_name: Service name.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time: str = field(default_factory=_utc_now_iso)
    end_time: str | None = None
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: tuple[SpanEvent, ...] = ()
    service_name: str = "truthound"
    status_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.name,
            "status": self.status.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "service_name": self.service_name,
            "status_message": self.status_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create SpanData from dictionary."""
        events = tuple(
            SpanEvent(
                name=e["name"],
                timestamp=e.get("timestamp", _utc_now_iso()),
                attributes=e.get("attributes", {}),
            )
            for e in data.get("events", [])
        )
        return cls(
            name=data["name"],
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            kind=SpanKind[data.get("kind", "INTERNAL")],
            status=SpanStatus[data.get("status", "UNSET")],
            start_time=data.get("start_time", _utc_now_iso()),
            end_time=data.get("end_time"),
            duration_ms=data.get("duration_ms", 0.0),
            attributes=data.get("attributes", {}),
            events=events,
            service_name=data.get("service_name", "truthound"),
            status_message=data.get("status_message", ""),
        )


# =============================================================================
# Metric Types
# =============================================================================


class Counter:
    """A counter metric that can only increase.

    Counters are used for counting events, requests, errors, etc.

    Example:
        >>> counter = Counter("requests_total", "Total requests")
        >>> counter.inc()
        >>> counter.inc(5)
        >>> counter.inc(labels={"method": "POST"})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Sequence[str] | None = None,
        registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize counter.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Expected label keys.
            registry: Registry to register with.
        """
        self._name = name
        self._description = description
        self._unit = unit
        self._label_keys = tuple(labels or ())
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()
        self._registry = registry

        if registry:
            registry.register_metric(self)

    @property
    def name(self) -> str:
        """Get metric name."""
        return self._name

    @property
    def metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.COUNTER

    @property
    def description(self) -> str:
        """Get metric description."""
        return self._description

    def inc(
        self,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment the counter.

        Args:
            value: Amount to increment (must be positive).
            labels: Optional labels for this observation.

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Counter increment must be non-negative")

        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_tuple] = self._values.get(label_tuple, 0.0) + value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current counter value.

        Args:
            labels: Labels to get value for.

        Returns:
            Current counter value.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values.get(label_tuple, 0.0)

    def collect(self) -> list[MetricData]:
        """Collect all metric data.

        Returns:
            List of MetricData for each label combination.
        """
        with self._lock:
            return [
                MetricData(
                    name=self._name,
                    metric_type=MetricType.COUNTER,
                    value=value,
                    labels=dict(labels),
                    unit=self._unit,
                    description=self._description,
                )
                for labels, value in self._values.items()
            ]

    def reset(self) -> None:
        """Reset all counter values (use with caution)."""
        with self._lock:
            self._values.clear()


class Gauge:
    """A gauge metric that can go up or down.

    Gauges are used for values that can increase or decrease,
    like current memory usage, active connections, temperature, etc.

    Example:
        >>> gauge = Gauge("memory_usage_bytes", "Current memory usage")
        >>> gauge.set(1024000)
        >>> gauge.inc(100)
        >>> gauge.dec(50)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Sequence[str] | None = None,
        registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize gauge.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Expected label keys.
            registry: Registry to register with.
        """
        self._name = name
        self._description = description
        self._unit = unit
        self._label_keys = tuple(labels or ())
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()
        self._registry = registry

        if registry:
            registry.register_metric(self)

    @property
    def name(self) -> str:
        """Get metric name."""
        return self._name

    @property
    def metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.GAUGE

    @property
    def description(self) -> str:
        """Get metric description."""
        return self._description

    def set(
        self,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set the gauge value.

        Args:
            value: Value to set.
            labels: Optional labels for this observation.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_tuple] = value

    def inc(
        self,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment the gauge.

        Args:
            value: Amount to increment.
            labels: Optional labels.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_tuple] = self._values.get(label_tuple, 0.0) + value

    def dec(
        self,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Decrement the gauge.

        Args:
            value: Amount to decrement.
            labels: Optional labels.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            self._values[label_tuple] = self._values.get(label_tuple, 0.0) - value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value.

        Args:
            labels: Labels to get value for.

        Returns:
            Current gauge value.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._values.get(label_tuple, 0.0)

    @contextlib.contextmanager
    def track_inprogress(
        self,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to track in-progress operations.

        Increments on entry, decrements on exit.

        Args:
            labels: Optional labels.

        Yields:
            None
        """
        self.inc(labels=labels)
        try:
            yield
        finally:
            self.dec(labels=labels)

    def collect(self) -> list[MetricData]:
        """Collect all metric data.

        Returns:
            List of MetricData for each label combination.
        """
        with self._lock:
            return [
                MetricData(
                    name=self._name,
                    metric_type=MetricType.GAUGE,
                    value=value,
                    labels=dict(labels),
                    unit=self._unit,
                    description=self._description,
                )
                for labels, value in self._values.items()
            ]

    def reset(self) -> None:
        """Reset all gauge values."""
        with self._lock:
            self._values.clear()


class Histogram:
    """A histogram metric for measuring distributions.

    Histograms sample observations and count them in configurable buckets.
    They also provide sum and count of all observations.

    Example:
        >>> histogram = Histogram(
        ...     "request_duration_seconds",
        ...     "Request duration",
        ...     buckets=(0.1, 0.5, 1.0, 5.0),
        ... )
        >>> histogram.observe(0.42)
        >>> with histogram.time():
        ...     process_request()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize histogram.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            buckets: Bucket boundaries.
            labels: Expected label keys.
            registry: Registry to register with.
        """
        self._name = name
        self._description = description
        self._unit = unit
        self._buckets = tuple(sorted(
            buckets or (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        ))
        self._label_keys = tuple(labels or ())
        self._bucket_counts: dict[tuple[tuple[str, str], ...], dict[float, int]] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = {}
        self._counts: dict[tuple[tuple[str, str], ...], int] = {}
        self._lock = threading.Lock()
        self._registry = registry

        if registry:
            registry.register_metric(self)

    @property
    def name(self) -> str:
        """Get metric name."""
        return self._name

    @property
    def metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.HISTOGRAM

    @property
    def description(self) -> str:
        """Get metric description."""
        return self._description

    @property
    def buckets(self) -> tuple[float, ...]:
        """Get bucket boundaries."""
        return self._buckets

    def observe(
        self,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Observe a value.

        Args:
            value: Value to observe.
            labels: Optional labels.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            # Initialize if needed
            if label_tuple not in self._bucket_counts:
                self._bucket_counts[label_tuple] = dict.fromkeys(self._buckets, 0)
                self._sums[label_tuple] = 0.0
                self._counts[label_tuple] = 0

            # Update buckets
            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[label_tuple][bucket] += 1

            # Update sum and count
            self._sums[label_tuple] += value
            self._counts[label_tuple] += 1

    @contextlib.contextmanager
    def time(
        self,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to time an operation.

        Args:
            labels: Optional labels.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, labels=labels)

    def get_sample_count(self, labels: dict[str, str] | None = None) -> int:
        """Get total observation count.

        Args:
            labels: Labels to get count for.

        Returns:
            Total count.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._counts.get(label_tuple, 0)

    def get_sample_sum(self, labels: dict[str, str] | None = None) -> float:
        """Get sum of all observations.

        Args:
            labels: Labels to get sum for.

        Returns:
            Total sum.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            return self._sums.get(label_tuple, 0.0)

    def collect(self) -> list[MetricData]:
        """Collect all metric data.

        Returns:
            List of MetricData for each bucket and label combination.
        """
        result: list[MetricData] = []
        with self._lock:
            for labels, bucket_counts in self._bucket_counts.items():
                label_dict = dict(labels)
                # Add bucket metrics
                for bucket, count in bucket_counts.items():
                    result.append(MetricData(
                        name=f"{self._name}_bucket",
                        metric_type=MetricType.HISTOGRAM,
                        value=count,
                        labels={**label_dict, "le": str(bucket)},
                        unit=self._unit,
                        description=self._description,
                    ))
                # Add sum
                result.append(MetricData(
                    name=f"{self._name}_sum",
                    metric_type=MetricType.HISTOGRAM,
                    value=self._sums.get(labels, 0.0),
                    labels=label_dict,
                    unit=self._unit,
                    description=self._description,
                ))
                # Add count
                result.append(MetricData(
                    name=f"{self._name}_count",
                    metric_type=MetricType.HISTOGRAM,
                    value=self._counts.get(labels, 0),
                    labels=label_dict,
                    unit=self._unit,
                    description=self._description,
                ))
        return result

    def reset(self) -> None:
        """Reset all histogram values."""
        with self._lock:
            self._bucket_counts.clear()
            self._sums.clear()
            self._counts.clear()


class Summary:
    """A summary metric for calculating quantiles.

    Summaries calculate streaming quantiles over a sliding time window.

    Example:
        >>> summary = Summary(
        ...     "request_size_bytes",
        ...     "Request size",
        ...     quantiles=(0.5, 0.9, 0.99),
        ... )
        >>> summary.observe(1024)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        quantiles: Sequence[float] | None = None,
        max_samples: int = 1000,
        labels: Sequence[str] | None = None,
        registry: MetricsRegistry | None = None,
    ) -> None:
        """Initialize summary.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            quantiles: Quantiles to calculate.
            max_samples: Maximum samples to keep for quantile calculation.
            labels: Expected label keys.
            registry: Registry to register with.
        """
        self._name = name
        self._description = description
        self._unit = unit
        self._quantiles = tuple(sorted(quantiles or (0.5, 0.9, 0.95, 0.99)))
        self._max_samples = max_samples
        self._label_keys = tuple(labels or ())
        self._samples: dict[tuple[tuple[str, str], ...], list[float]] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = {}
        self._counts: dict[tuple[tuple[str, str], ...], int] = {}
        self._lock = threading.Lock()
        self._registry = registry

        if registry:
            registry.register_metric(self)

    @property
    def name(self) -> str:
        """Get metric name."""
        return self._name

    @property
    def metric_type(self) -> MetricType:
        """Get metric type."""
        return MetricType.SUMMARY

    @property
    def description(self) -> str:
        """Get metric description."""
        return self._description

    def observe(
        self,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Observe a value.

        Args:
            value: Value to observe.
            labels: Optional labels.
        """
        label_tuple = tuple(sorted((labels or {}).items()))
        with self._lock:
            if label_tuple not in self._samples:
                self._samples[label_tuple] = []
                self._sums[label_tuple] = 0.0
                self._counts[label_tuple] = 0

            # Add sample (with rotation if needed)
            samples = self._samples[label_tuple]
            if len(samples) >= self._max_samples:
                samples.pop(0)
            samples.append(value)

            self._sums[label_tuple] += value
            self._counts[label_tuple] += 1

    @contextlib.contextmanager
    def time(
        self,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to time an operation.

        Args:
            labels: Optional labels.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, labels=labels)

    def _calculate_quantile(self, samples: list[float], q: float) -> float:
        """Calculate quantile from samples.

        Args:
            samples: Sorted list of samples.
            q: Quantile (0.0 to 1.0).

        Returns:
            Quantile value.
        """
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        idx = int(q * (len(sorted_samples) - 1))
        return sorted_samples[idx]

    def collect(self) -> list[MetricData]:
        """Collect all metric data.

        Returns:
            List of MetricData for each quantile and label combination.
        """
        result: list[MetricData] = []
        with self._lock:
            for labels, samples in self._samples.items():
                label_dict = dict(labels)
                # Add quantile metrics
                result.extend(
                    MetricData(
                        name=self._name,
                        metric_type=MetricType.SUMMARY,
                        value=self._calculate_quantile(samples, q),
                        labels={**label_dict, "quantile": str(q)},
                        unit=self._unit,
                        description=self._description,
                    )
                    for q in self._quantiles
                )
                # Add sum
                result.append(MetricData(
                    name=f"{self._name}_sum",
                    metric_type=MetricType.SUMMARY,
                    value=self._sums.get(labels, 0.0),
                    labels=label_dict,
                    unit=self._unit,
                    description=self._description,
                ))
                # Add count
                result.append(MetricData(
                    name=f"{self._name}_count",
                    metric_type=MetricType.SUMMARY,
                    value=self._counts.get(labels, 0),
                    labels=label_dict,
                    unit=self._unit,
                    description=self._description,
                ))
        return result

    def reset(self) -> None:
        """Reset all summary values."""
        with self._lock:
            self._samples.clear()
            self._sums.clear()
            self._counts.clear()


# =============================================================================
# Span Implementation
# =============================================================================


class Span:
    """A span representing a unit of work in distributed tracing.

    Spans track the execution of operations and can have attributes,
    events, and links to other spans.

    Example:
        >>> with Span("process_data") as span:
        ...     span.set_attribute("rows", 1000)
        ...     result = process(data)
        ...     span.add_event("processing_complete")
    """

    def __init__(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        config: TracingConfig | None = None,
    ) -> None:
        """Initialize span.

        Args:
            name: Span name.
            kind: Kind of span.
            parent: Parent span for nested traces.
            attributes: Initial span attributes.
            trace_id: Trace ID (generated if not provided).
            config: Tracing configuration.
        """
        self._name = name
        self._kind = kind
        self._config = config or DEFAULT_TRACING_CONFIG

        # Generate or inherit IDs
        if parent:
            self._trace_id = parent.trace_id
            self._parent_span_id = parent.span_id
        else:
            self._trace_id = trace_id or _generate_trace_id()
            self._parent_span_id = None

        self._span_id = _generate_span_id()
        self._attributes: dict[str, Any] = dict(attributes or {})
        self._events: list[SpanEvent] = []
        self._status = SpanStatus.UNSET
        self._status_message = ""
        self._start_time = _utc_now_iso()
        self._start_perf = time.perf_counter()
        self._end_time: str | None = None
        self._duration_ms: float = 0.0
        self._ended = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get span name."""
        return self._name

    @property
    def trace_id(self) -> str:
        """Get trace ID."""
        return self._trace_id

    @property
    def span_id(self) -> str:
        """Get span ID."""
        return self._span_id

    @property
    def parent_span_id(self) -> str | None:
        """Get parent span ID."""
        return self._parent_span_id

    @property
    def is_recording(self) -> bool:
        """Check if span is still recording."""
        return not self._ended

    @property
    def kind(self) -> SpanKind:
        """Get span kind."""
        return self._kind

    @property
    def status(self) -> SpanStatus:
        """Get span status."""
        return self._status

    @property
    def attributes(self) -> dict[str, Any]:
        """Get span attributes."""
        with self._lock:
            return dict(self._attributes)

    def set_attribute(self, key: str, value: Any) -> Span:
        """Set a span attribute.

        Args:
            key: Attribute key.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        if self._ended:
            return self

        with self._lock:
            if len(self._attributes) < self._config.max_attributes:
                self._attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> Span:
        """Set multiple span attributes.

        Args:
            attributes: Dictionary of attributes.

        Returns:
            Self for chaining.
        """
        for key, value in attributes.items():
            self.set_attribute(key, value)
        return self

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Event attributes.

        Returns:
            Self for chaining.
        """
        if self._ended:
            return self

        with self._lock:
            if len(self._events) < self._config.max_events:
                self._events.append(SpanEvent(
                    name=name,
                    attributes=attributes or {},
                ))
        return self

    def set_status(
        self,
        status: SpanStatus,
        message: str = "",
    ) -> Span:
        """Set the span status.

        Args:
            status: Status to set.
            message: Optional status message.

        Returns:
            Self for chaining.
        """
        if self._ended:
            return self

        with self._lock:
            self._status = status
            self._status_message = message
        return self

    def record_exception(
        self,
        exception: Exception,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Record an exception.

        Args:
            exception: Exception to record.
            attributes: Additional attributes.

        Returns:
            Self for chaining.
        """
        exc_attrs = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            **(attributes or {}),
        }
        self.add_event("exception", exc_attrs)
        self.set_status(SpanStatus.ERROR, str(exception))
        return self

    def end(self) -> None:
        """End the span."""
        if self._ended:
            return

        with self._lock:
            self._ended = True
            self._end_time = _utc_now_iso()
            self._duration_ms = (time.perf_counter() - self._start_perf) * 1000

    def to_span_data(self) -> SpanData:
        """Convert to SpanData.

        Returns:
            SpanData representation.
        """
        with self._lock:
            return SpanData(
                name=self._name,
                trace_id=self._trace_id,
                span_id=self._span_id,
                parent_span_id=self._parent_span_id,
                kind=self._kind,
                status=self._status,
                start_time=self._start_time,
                end_time=self._end_time,
                duration_ms=self._duration_ms,
                attributes=dict(self._attributes),
                events=tuple(self._events),
                service_name=self._config.service_name,
                status_message=self._status_message,
            )

    def __enter__(self) -> Span:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if exc_val is not None and isinstance(exc_val, Exception):
            self.record_exception(exc_val)
        elif self._status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()


# =============================================================================
# Exporters
# =============================================================================


class ConsoleMetricExporter:
    """Exporter that prints metrics to console."""

    def __init__(self, pretty: bool = True) -> None:
        """Initialize exporter.

        Args:
            pretty: Whether to pretty-print output.
        """
        self._pretty = pretty
        self._lock = threading.Lock()

    def export(self, metrics: Sequence[MetricData]) -> None:
        """Export metrics to console.

        Args:
            metrics: Metrics to export.
        """
        with self._lock:
            for metric in metrics:
                if self._pretty:
                    labels_str = ", ".join(
                        f"{k}={v}" for k, v in metric.labels.items()
                    )
                    print(f"[METRIC] {metric.name}{{{labels_str}}} = {metric.value}")
                else:
                    print(json.dumps(metric.to_dict()))

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class ConsoleSpanExporter:
    """Exporter that prints spans to console."""

    def __init__(self, pretty: bool = True) -> None:
        """Initialize exporter.

        Args:
            pretty: Whether to pretty-print output.
        """
        self._pretty = pretty
        self._lock = threading.Lock()

    def export(self, spans: Sequence[SpanData]) -> None:
        """Export spans to console.

        Args:
            spans: Spans to export.
        """
        with self._lock:
            for span in spans:
                if self._pretty:
                    status_str = span.status.name
                    print(
                        f"[SPAN] {span.name} "
                        f"trace={span.trace_id[:8]}... "
                        f"span={span.span_id} "
                        f"duration={span.duration_ms:.2f}ms "
                        f"status={status_str}"
                    )
                else:
                    print(json.dumps(span.to_dict()))

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class InMemoryMetricExporter:
    """Exporter that stores metrics in memory for testing."""

    def __init__(self) -> None:
        """Initialize exporter."""
        self._metrics: list[MetricData] = []
        self._lock = threading.Lock()

    @property
    def metrics(self) -> list[MetricData]:
        """Get all exported metrics."""
        with self._lock:
            return list(self._metrics)

    def export(self, metrics: Sequence[MetricData]) -> None:
        """Export metrics to memory.

        Args:
            metrics: Metrics to export.
        """
        with self._lock:
            self._metrics.extend(metrics)

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass

    def clear(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self._metrics.clear()


class InMemorySpanExporter:
    """Exporter that stores spans in memory for testing."""

    def __init__(self) -> None:
        """Initialize exporter."""
        self._spans: list[SpanData] = []
        self._lock = threading.Lock()

    @property
    def spans(self) -> list[SpanData]:
        """Get all exported spans."""
        with self._lock:
            return list(self._spans)

    def export(self, spans: Sequence[SpanData]) -> None:
        """Export spans to memory.

        Args:
            spans: Spans to export.
        """
        with self._lock:
            self._spans.extend(spans)

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass

    def clear(self) -> None:
        """Clear all stored spans."""
        with self._lock:
            self._spans.clear()


class CompositeMetricExporter:
    """Exporter that delegates to multiple exporters."""

    def __init__(self, exporters: Sequence[MetricExporter]) -> None:
        """Initialize composite exporter.

        Args:
            exporters: List of exporters to delegate to.
        """
        self._exporters = list(exporters)

    def add_exporter(self, exporter: MetricExporter) -> None:
        """Add an exporter.

        Args:
            exporter: Exporter to add.
        """
        self._exporters.append(exporter)

    def export(self, metrics: Sequence[MetricData]) -> None:
        """Export metrics to all exporters.

        Args:
            metrics: Metrics to export.
        """
        for exporter in self._exporters:
            with contextlib.suppress(Exception):
                exporter.export(metrics)

    def shutdown(self) -> None:
        """Shutdown all exporters."""
        for exporter in self._exporters:
            with contextlib.suppress(Exception):
                exporter.shutdown()


class CompositeSpanExporter:
    """Exporter that delegates to multiple span exporters."""

    def __init__(self, exporters: Sequence[SpanExporter]) -> None:
        """Initialize composite exporter.

        Args:
            exporters: List of exporters to delegate to.
        """
        self._exporters = list(exporters)

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add an exporter.

        Args:
            exporter: Exporter to add.
        """
        self._exporters.append(exporter)

    def export(self, spans: Sequence[SpanData]) -> None:
        """Export spans to all exporters.

        Args:
            spans: Spans to export.
        """
        for exporter in self._exporters:
            with contextlib.suppress(Exception):
                exporter.export(spans)

    def shutdown(self) -> None:
        """Shutdown all exporters."""
        for exporter in self._exporters:
            with contextlib.suppress(Exception):
                exporter.shutdown()


# =============================================================================
# Hooks
# =============================================================================


class LoggingMetricsHook:
    """Hook that logs metric events."""

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.metrics).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.metrics")

    def on_record(
        self,
        metric_name: str,
        metric_type: MetricType,
        value: float,
        labels: dict[str, str],
        context: dict[str, Any],
    ) -> None:
        """Log metric recording.

        Args:
            metric_name: Name of the metric.
            metric_type: Type of the metric.
            value: Recorded value.
            labels: Metric labels.
            context: Additional context.
        """
        self._logger.debug(
            "Metric recorded",
            metric_name=metric_name,
            metric_type=metric_type.name,
            value=value,
            labels=labels,
            **context,
        )


class LoggingTracingHook:
    """Hook that logs tracing events."""

    def __init__(self, logger_name: str | None = None) -> None:
        """Initialize logging hook.

        Args:
            logger_name: Logger name (default: common.tracing).
        """
        from common.logging import get_logger

        self._logger = get_logger(logger_name or "common.tracing")

    def on_span_start(
        self,
        span: Span,
        context: dict[str, Any],
    ) -> None:
        """Log span start.

        Args:
            span: The span that started.
            context: Additional context.
        """
        self._logger.debug(
            "Span started",
            span_name=span.name,
            trace_id=span.trace_id,
            span_id=span.span_id,
            **context,
        )

    def on_span_end(
        self,
        span: Span,
        context: dict[str, Any],
    ) -> None:
        """Log span end.

        Args:
            span: The span that ended.
            context: Additional context.
        """
        span_data = span.to_span_data()
        log_method = self._logger.info
        if span_data.status == SpanStatus.ERROR:
            log_method = self._logger.error

        log_method(
            "Span ended",
            span_name=span.name,
            trace_id=span.trace_id,
            span_id=span.span_id,
            duration_ms=span_data.duration_ms,
            status=span_data.status.name,
            **context,
        )


class CompositeMetricsHook:
    """Combine multiple metrics hooks."""

    def __init__(self, hooks: Sequence[MetricsHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: MetricsHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def on_record(
        self,
        metric_name: str,
        metric_type: MetricType,
        value: float,
        labels: dict[str, str],
        context: dict[str, Any],
    ) -> None:
        """Call on_record on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_record(metric_name, metric_type, value, labels, context)


class CompositeTracingHook:
    """Combine multiple tracing hooks."""

    def __init__(self, hooks: Sequence[TracingHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to call.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: TracingHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def on_span_start(self, span: Span, context: dict[str, Any]) -> None:
        """Call on_span_start on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_span_start(span, context)

    def on_span_end(self, span: Span, context: dict[str, Any]) -> None:
        """Call on_span_end on all hooks."""
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_span_end(span, context)


# =============================================================================
# Registry
# =============================================================================


class MetricsRegistry:
    """Registry for managing metrics.

    Provides a central location to create, retrieve, and collect metrics.

    Example:
        >>> registry = MetricsRegistry()
        >>> counter = registry.counter("requests_total", "Total requests")
        >>> counter.inc()
        >>> all_metrics = registry.collect()
    """

    def __init__(
        self,
        config: MetricsConfig | None = None,
        exporters: Sequence[MetricExporter] | None = None,
        hooks: Sequence[MetricsHook] | None = None,
    ) -> None:
        """Initialize registry.

        Args:
            config: Metrics configuration.
            exporters: Metric exporters.
            hooks: Metrics event hooks.
        """
        self._config = config or DEFAULT_METRICS_CONFIG
        self._metrics: dict[str, Counter | Gauge | Histogram | Summary] = {}
        self._lock = threading.Lock()
        self._exporter: MetricExporter | None = None
        if exporters:
            self._exporter = CompositeMetricExporter(list(exporters))
        self._hook: MetricsHook | None = None
        if hooks:
            self._hook = CompositeMetricsHook(list(hooks))

    @property
    def config(self) -> MetricsConfig:
        """Get metrics configuration."""
        return self._config

    def _prefixed_name(self, name: str) -> str:
        """Get prefixed metric name.

        Args:
            name: Base metric name.

        Returns:
            Prefixed name.
        """
        if self._config.prefix:
            return f"{self._config.prefix}_{name}"
        return name

    def register_metric(
        self,
        metric: Counter | Gauge | Histogram | Summary,
    ) -> None:
        """Register a metric.

        Args:
            metric: Metric to register.
        """
        with self._lock:
            self._metrics[metric.name] = metric

    def counter(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Sequence[str] | None = None,
    ) -> Counter:
        """Create or get a counter.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Expected label keys.

        Returns:
            Counter instance.
        """
        prefixed_name = self._prefixed_name(name)
        with self._lock:
            if prefixed_name in self._metrics:
                existing = self._metrics[prefixed_name]
                if isinstance(existing, Counter):
                    return existing
                raise MetricsError(
                    f"Metric {prefixed_name} already exists with different type",
                    metric_name=prefixed_name,
                    metric_type=existing.metric_type.name,
                )

            counter = Counter(
                name=prefixed_name,
                description=description,
                unit=unit,
                labels=labels,
            )
            self._metrics[prefixed_name] = counter
            return counter

    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: Sequence[str] | None = None,
    ) -> Gauge:
        """Create or get a gauge.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            labels: Expected label keys.

        Returns:
            Gauge instance.
        """
        prefixed_name = self._prefixed_name(name)
        with self._lock:
            if prefixed_name in self._metrics:
                existing = self._metrics[prefixed_name]
                if isinstance(existing, Gauge):
                    return existing
                raise MetricsError(
                    f"Metric {prefixed_name} already exists with different type",
                    metric_name=prefixed_name,
                    metric_type=existing.metric_type.name,
                )

            gauge = Gauge(
                name=prefixed_name,
                description=description,
                unit=unit,
                labels=labels,
            )
            self._metrics[prefixed_name] = gauge
            return gauge

    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
    ) -> Histogram:
        """Create or get a histogram.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            buckets: Bucket boundaries.
            labels: Expected label keys.

        Returns:
            Histogram instance.
        """
        prefixed_name = self._prefixed_name(name)
        with self._lock:
            if prefixed_name in self._metrics:
                existing = self._metrics[prefixed_name]
                if isinstance(existing, Histogram):
                    return existing
                raise MetricsError(
                    f"Metric {prefixed_name} already exists with different type",
                    metric_name=prefixed_name,
                    metric_type=existing.metric_type.name,
                )

            histogram = Histogram(
                name=prefixed_name,
                description=description,
                unit=unit,
                buckets=buckets or self._config.histogram_buckets,
                labels=labels,
            )
            self._metrics[prefixed_name] = histogram
            return histogram

    def summary(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        quantiles: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
    ) -> Summary:
        """Create or get a summary.

        Args:
            name: Metric name.
            description: Human-readable description.
            unit: Unit of measurement.
            quantiles: Quantiles to calculate.
            labels: Expected label keys.

        Returns:
            Summary instance.
        """
        prefixed_name = self._prefixed_name(name)
        with self._lock:
            if prefixed_name in self._metrics:
                existing = self._metrics[prefixed_name]
                if isinstance(existing, Summary):
                    return existing
                raise MetricsError(
                    f"Metric {prefixed_name} already exists with different type",
                    metric_name=prefixed_name,
                    metric_type=existing.metric_type.name,
                )

            summary = Summary(
                name=prefixed_name,
                description=description,
                unit=unit,
                quantiles=quantiles or self._config.summary_quantiles,
                labels=labels,
            )
            self._metrics[prefixed_name] = summary
            return summary

    def collect(self) -> list[MetricData]:
        """Collect all metrics.

        Returns:
            List of all metric data.
        """
        result: list[MetricData] = []
        with self._lock:
            for metric in self._metrics.values():
                result.extend(metric.collect())
        return result

    def export(self) -> None:
        """Export all metrics to configured exporters."""
        if self._exporter is None:
            return

        metrics = self.collect()
        if metrics:
            self._exporter.export(metrics)

    def get(self, name: str) -> Counter | Gauge | Histogram | Summary | None:
        """Get a metric by name.

        Args:
            name: Metric name.

        Returns:
            Metric if found, None otherwise.
        """
        prefixed_name = self._prefixed_name(name)
        with self._lock:
            return self._metrics.get(prefixed_name)

    def remove(self, name: str) -> bool:
        """Remove a metric.

        Args:
            name: Metric name.

        Returns:
            True if removed, False if not found.
        """
        prefixed_name = self._prefixed_name(name)
        with self._lock:
            if prefixed_name in self._metrics:
                del self._metrics[prefixed_name]
                return True
            return False

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()

    @property
    def names(self) -> list[str]:
        """Get all metric names."""
        with self._lock:
            return list(self._metrics.keys())

    def shutdown(self) -> None:
        """Shutdown registry and exporters."""
        if self._exporter:
            self._exporter.shutdown()


class TracingRegistry:
    """Registry for managing traces.

    Provides a central location to create spans and manage trace context.

    Example:
        >>> registry = TracingRegistry()
        >>> with registry.start_span("process") as span:
        ...     span.set_attribute("key", "value")
    """

    def __init__(
        self,
        config: TracingConfig | None = None,
        exporters: Sequence[SpanExporter] | None = None,
        hooks: Sequence[TracingHook] | None = None,
    ) -> None:
        """Initialize registry.

        Args:
            config: Tracing configuration.
            exporters: Span exporters.
            hooks: Tracing event hooks.
        """
        self._config = config or DEFAULT_TRACING_CONFIG
        self._exporter: SpanExporter | None = None
        if exporters:
            self._exporter = CompositeSpanExporter(list(exporters))
        self._hook: TracingHook | None = None
        if hooks:
            self._hook = CompositeTracingHook(list(hooks))
        self._current_span: threading.local = threading.local()
        self._pending_spans: list[SpanData] = []
        self._lock = threading.Lock()

    @property
    def config(self) -> TracingConfig:
        """Get tracing configuration."""
        return self._config

    def get_current_span(self) -> Span | None:
        """Get the current active span.

        Returns:
            Current span or None.
        """
        return getattr(self._current_span, "span", None)

    def _set_current_span(self, span: Span | None) -> None:
        """Set the current active span.

        Args:
            span: Span to set as current.
        """
        self._current_span.span = span

    def _should_sample(self) -> bool:
        """Determine if a trace should be sampled.

        Returns:
            True if trace should be sampled.
        """
        if not self._config.enabled:
            return False
        return random.random() < self._config.sample_rate

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        parent: Span | None = None,
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name.
            kind: Kind of span.
            attributes: Initial attributes.
            parent: Parent span (uses current if None).

        Returns:
            New Span instance.
        """
        if parent is None:
            parent = self.get_current_span()

        span = Span(
            name=name,
            kind=kind,
            parent=parent,
            attributes=attributes,
            config=self._config,
        )

        if self._hook:
            self._hook.on_span_start(span, {})

        return span

    @contextlib.contextmanager
    def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Context manager for creating a span.

        Args:
            name: Span name.
            kind: Kind of span.
            attributes: Initial attributes.

        Yields:
            The created span.
        """
        if not self._should_sample():
            # Return a no-op span when not sampling
            yield Span(name, config=DISABLED_TRACING_CONFIG)
            return

        span = self.start_span(name, kind, attributes)
        previous_span = self.get_current_span()
        self._set_current_span(span)

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
            span.end()

            if self._hook:
                self._hook.on_span_end(span, {})

            self._set_current_span(previous_span)

            # Queue span for export
            with self._lock:
                self._pending_spans.append(span.to_span_data())

    def export(self) -> None:
        """Export pending spans."""
        if self._exporter is None:
            return

        with self._lock:
            spans = list(self._pending_spans)
            self._pending_spans.clear()

        if spans:
            self._exporter.export(spans)

    def shutdown(self) -> None:
        """Shutdown registry and export pending spans."""
        self.export()
        if self._exporter:
            self._exporter.shutdown()


# Global registries
_default_metrics_registry = MetricsRegistry()
_default_tracing_registry = TracingRegistry()


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry.

    Returns:
        The global MetricsRegistry instance.
    """
    return _default_metrics_registry


def get_tracing_registry() -> TracingRegistry:
    """Get the global tracing registry.

    Returns:
        The global TracingRegistry instance.
    """
    return _default_tracing_registry


def configure_metrics(
    config: MetricsConfig | None = None,
    exporters: Sequence[MetricExporter] | None = None,
    hooks: Sequence[MetricsHook] | None = None,
) -> MetricsRegistry:
    """Configure the global metrics registry.

    Args:
        config: Metrics configuration.
        exporters: Metric exporters.
        hooks: Metrics event hooks.

    Returns:
        Configured MetricsRegistry.
    """
    global _default_metrics_registry
    _default_metrics_registry = MetricsRegistry(
        config=config,
        exporters=exporters,
        hooks=hooks,
    )
    return _default_metrics_registry


def configure_tracing(
    config: TracingConfig | None = None,
    exporters: Sequence[SpanExporter] | None = None,
    hooks: Sequence[TracingHook] | None = None,
) -> TracingRegistry:
    """Configure the global tracing registry.

    Args:
        config: Tracing configuration.
        exporters: Span exporters.
        hooks: Tracing event hooks.

    Returns:
        Configured TracingRegistry.
    """
    global _default_tracing_registry
    _default_tracing_registry = TracingRegistry(
        config=config,
        exporters=exporters,
        hooks=hooks,
    )
    return _default_tracing_registry


# =============================================================================
# Convenience Functions
# =============================================================================


def counter(
    name: str,
    description: str = "",
    unit: str = "",
    labels: Sequence[str] | None = None,
) -> Counter:
    """Create or get a counter from the global registry.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
        labels: Expected label keys.

    Returns:
        Counter instance.
    """
    return _default_metrics_registry.counter(name, description, unit, labels)


def gauge(
    name: str,
    description: str = "",
    unit: str = "",
    labels: Sequence[str] | None = None,
) -> Gauge:
    """Create or get a gauge from the global registry.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
        labels: Expected label keys.

    Returns:
        Gauge instance.
    """
    return _default_metrics_registry.gauge(name, description, unit, labels)


def histogram(
    name: str,
    description: str = "",
    unit: str = "",
    buckets: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> Histogram:
    """Create or get a histogram from the global registry.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
        buckets: Bucket boundaries.
        labels: Expected label keys.

    Returns:
        Histogram instance.
    """
    return _default_metrics_registry.histogram(name, description, unit, buckets, labels)


def summary(
    name: str,
    description: str = "",
    unit: str = "",
    quantiles: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
) -> Summary:
    """Create or get a summary from the global registry.

    Args:
        name: Metric name.
        description: Human-readable description.
        unit: Unit of measurement.
        quantiles: Quantiles to calculate.
        labels: Expected label keys.

    Returns:
        Summary instance.
    """
    return _default_metrics_registry.summary(name, description, unit, quantiles, labels)


# =============================================================================
# Decorators
# =============================================================================


def timed(
    name: str | None = None,
    description: str = "",
    labels: dict[str, str] | None = None,
    use_histogram: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to time function execution.

    Creates a histogram or summary to track execution time.

    Args:
        name: Metric name (defaults to function name).
        description: Metric description.
        labels: Static labels to apply.
        use_histogram: Use histogram (True) or summary (False).

    Returns:
        Decorator function.

    Example:
        >>> @timed("process_duration_seconds", labels={"component": "validator"})
        ... def process_data(data):
        ...     return validate(data)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        metric_name = name or f"{func.__name__}_duration_seconds"
        static_labels = labels or {}

        if use_histogram:
            metric = histogram(metric_name, description)
        else:
            metric = summary(metric_name, description)

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    metric.observe(duration, labels=static_labels)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with metric.time(labels=static_labels):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def counted(
    name: str | None = None,
    description: str = "",
    labels: dict[str, str] | None = None,
    count_exceptions: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to count function calls.

    Creates a counter to track function invocations.

    Args:
        name: Metric name (defaults to function name).
        description: Metric description.
        labels: Static labels to apply.
        count_exceptions: Whether to count exceptions.

    Returns:
        Decorator function.

    Example:
        >>> @counted("api_calls_total", labels={"endpoint": "/check"})
        ... def check_data(data):
        ...     return validate(data)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        metric_name = name or f"{func.__name__}_total"
        metric = counter(metric_name, description)
        static_labels = labels or {}

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = await func(*args, **kwargs)
                    metric.inc(labels={**static_labels, "status": "success"})
                    return result
                except Exception:
                    if count_exceptions:
                        metric.inc(labels={**static_labels, "status": "error"})
                    raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                metric.inc(labels={**static_labels, "status": "success"})
                return result
            except Exception:
                if count_exceptions:
                    metric.inc(labels={**static_labels, "status": "error"})
                raise

        return sync_wrapper

    return decorator


def trace(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace function execution.

    Creates a span for the function execution.

    Args:
        name: Span name (defaults to function name).
        kind: Kind of span.
        attributes: Initial span attributes.

    Returns:
        Decorator function.

    Example:
        >>> @trace(name="validate_data", attributes={"component": "validator"})
        ... def validate(data):
        ...     return truthound.check(data)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or func.__name__
        static_attrs = attributes or {}

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _default_tracing_registry.trace(
                    span_name, kind, static_attrs
                ) as span:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _default_tracing_registry.trace(
                span_name, kind, static_attrs
            ) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator


def instrumented(
    metric_name: str | None = None,
    span_name: str | None = None,
    labels: dict[str, str] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator combining metrics and tracing.

    Creates both a histogram for timing and a span for tracing.

    Args:
        metric_name: Metric name (defaults to function name).
        span_name: Span name (defaults to function name).
        labels: Metric labels.
        attributes: Span attributes.

    Returns:
        Decorator function.

    Example:
        >>> @instrumented(labels={"component": "validator"})
        ... def validate(data):
        ...     return truthound.check(data)
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        m_name = metric_name or f"{func.__name__}_duration_seconds"
        s_name = span_name or func.__name__
        static_labels = labels or {}
        static_attrs = attributes or {}

        hist = histogram(m_name, f"Duration of {func.__name__}")

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _default_tracing_registry.trace(s_name, attributes=static_attrs) as span:
                    span.set_attribute("function.name", func.__name__)
                    start = time.perf_counter()
                    try:
                        result = await func(*args, **kwargs)
                        hist.observe(
                            time.perf_counter() - start,
                            labels={**static_labels, "status": "success"},
                        )
                        return result
                    except Exception:
                        hist.observe(
                            time.perf_counter() - start,
                            labels={**static_labels, "status": "error"},
                        )
                        raise

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with _default_tracing_registry.trace(s_name, attributes=static_attrs) as span:
                span.set_attribute("function.name", func.__name__)
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    hist.observe(
                        time.perf_counter() - start,
                        labels={**static_labels, "status": "success"},
                    )
                    return result
                except Exception:
                    hist.observe(
                        time.perf_counter() - start,
                        labels={**static_labels, "status": "error"},
                    )
                    raise

        return sync_wrapper

    return decorator


# =============================================================================
# Context Propagation
# =============================================================================


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Context for propagating trace information.

    Used for passing trace context across service boundaries.

    Attributes:
        trace_id: Trace identifier.
        span_id: Current span identifier.
        sampled: Whether the trace is sampled.
        trace_state: Additional trace state.
    """

    trace_id: str
    span_id: str
    sampled: bool = True
    trace_state: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "sampled": self.sampled,
            "trace_state": self.trace_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create TraceContext from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            sampled=data.get("sampled", True),
            trace_state=data.get("trace_state", {}),
        )

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers (W3C Trace Context format).

        Returns:
            Dictionary of header name to value.
        """
        sampled_flag = "01" if self.sampled else "00"
        traceparent = f"00-{self.trace_id}-{self.span_id}-{sampled_flag}"
        headers = {"traceparent": traceparent}

        if self.trace_state:
            tracestate = ",".join(f"{k}={v}" for k, v in self.trace_state.items())
            headers["tracestate"] = tracestate

        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> Self | None:
        """Parse from HTTP headers (W3C Trace Context format).

        Args:
            headers: Dictionary of header name to value.

        Returns:
            TraceContext if valid headers found, None otherwise.
        """
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        parts = traceparent.split("-")
        if len(parts) != 4:
            return None

        version, trace_id, span_id, flags = parts
        if version != "00":
            return None

        sampled = flags == "01"

        trace_state: dict[str, str] = {}
        tracestate = headers.get("tracestate")
        if tracestate:
            for item in tracestate.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    trace_state[key.strip()] = value.strip()

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            sampled=sampled,
            trace_state=trace_state,
        )

    @classmethod
    def from_span(cls, span: Span) -> Self:
        """Create TraceContext from a Span.

        Args:
            span: Span to extract context from.

        Returns:
            TraceContext instance.
        """
        return cls(
            trace_id=span.trace_id,
            span_id=span.span_id,
            sampled=True,
        )


def inject_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject current trace context into headers.

    Args:
        headers: Headers dictionary to inject into.

    Returns:
        Headers with trace context added.
    """
    current_span = _default_tracing_registry.get_current_span()
    if current_span:
        ctx = TraceContext.from_span(current_span)
        headers.update(ctx.to_headers())
    return headers


def extract_context(headers: dict[str, str]) -> TraceContext | None:
    """Extract trace context from headers.

    Args:
        headers: Headers to extract from.

    Returns:
        TraceContext if found, None otherwise.
    """
    return TraceContext.from_headers(headers)
