"""Testing utilities for OpenTelemetry integration.

This module provides mock objects and test helpers for testing
code that uses OpenTelemetry instrumentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from common.opentelemetry.config import OTelConfig, SamplingConfig
from common.opentelemetry.types import SamplingDecision

__all__ = [
    # Mock providers
    "MockMeterProvider",
    "MockTracerProvider",
    "MockMeter",
    "MockTracer",
    "MockSpan",
    # Mock instruments
    "MockCounter",
    "MockGauge",
    "MockHistogram",
    # Mock hook
    "MockOTelEngineHook",
    # Test helpers
    "create_test_config",
    "create_disabled_config",
    "collect_spans",
    "collect_metrics",
    "reset_test_state",
]


@dataclass
class MockMetricData:
    """Mock metric data point."""

    name: str
    value: float
    attributes: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MockSpanData:
    """Mock span data."""

    name: str
    span_id: str
    trace_id: str
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None


class MockCounter:
    """Mock OpenTelemetry Counter."""

    def __init__(self, name: str, description: str = "", unit: str = "1") -> None:
        """Initialize MockCounter."""
        self.name = name
        self.description = description
        self.unit = unit
        self.values: list[MockMetricData] = []

    def add(self, amount: float, attributes: dict[str, Any] | None = None) -> None:
        """Add a value to the counter."""
        self.values.append(MockMetricData(
            name=self.name,
            value=amount,
            attributes=attributes or {},
        ))

    @property
    def total(self) -> float:
        """Get total count."""
        return sum(v.value for v in self.values)


class MockGauge:
    """Mock OpenTelemetry Gauge (UpDownCounter)."""

    def __init__(self, name: str, description: str = "", unit: str = "1") -> None:
        """Initialize MockGauge."""
        self.name = name
        self.description = description
        self.unit = unit
        self.values: list[MockMetricData] = []

    def add(self, amount: float, attributes: dict[str, Any] | None = None) -> None:
        """Add a value to the gauge."""
        self.values.append(MockMetricData(
            name=self.name,
            value=amount,
            attributes=attributes or {},
        ))

    @property
    def current(self) -> float:
        """Get current value (sum of all additions)."""
        return sum(v.value for v in self.values)


class MockHistogram:
    """Mock OpenTelemetry Histogram."""

    def __init__(self, name: str, description: str = "", unit: str = "1") -> None:
        """Initialize MockHistogram."""
        self.name = name
        self.description = description
        self.unit = unit
        self.values: list[MockMetricData] = []

    def record(self, value: float, attributes: dict[str, Any] | None = None) -> None:
        """Record a value to the histogram."""
        self.values.append(MockMetricData(
            name=self.name,
            value=value,
            attributes=attributes or {},
        ))

    @property
    def count(self) -> int:
        """Get number of recorded values."""
        return len(self.values)

    @property
    def sum(self) -> float:
        """Get sum of recorded values."""
        return sum(v.value for v in self.values)


class MockMeter:
    """Mock OpenTelemetry Meter."""

    def __init__(self, name: str, version: str | None = None) -> None:
        """Initialize MockMeter."""
        self.name = name
        self.version = version
        self.counters: dict[str, MockCounter] = {}
        self.gauges: dict[str, MockGauge] = {}
        self.histograms: dict[str, MockHistogram] = {}

    def create_counter(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
    ) -> MockCounter:
        """Create a counter."""
        counter = MockCounter(name, description, unit)
        self.counters[name] = counter
        return counter

    def create_up_down_counter(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
    ) -> MockGauge:
        """Create an up-down counter (gauge)."""
        gauge = MockGauge(name, description, unit)
        self.gauges[name] = gauge
        return gauge

    def create_histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "1",
    ) -> MockHistogram:
        """Create a histogram."""
        histogram = MockHistogram(name, description, unit)
        self.histograms[name] = histogram
        return histogram


class MockMeterProvider:
    """Mock OpenTelemetry MeterProvider."""

    def __init__(self) -> None:
        """Initialize MockMeterProvider."""
        self.meters: dict[str, MockMeter] = {}
        self._is_shutdown = False

    def get_meter(
        self,
        name: str,
        version: str | None = None,
        schema_url: str | None = None,
    ) -> MockMeter:
        """Get or create a meter."""
        key = f"{name}:{version or ''}"
        if key not in self.meters:
            self.meters[key] = MockMeter(name, version)
        return self.meters[key]

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush (no-op for mock)."""
        return True

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown provider."""
        self._is_shutdown = True
        return True


class MockSpan:
    """Mock OpenTelemetry Span."""

    _id_counter = 0

    def __init__(
        self,
        name: str,
        parent: "MockSpan | None" = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MockSpan."""
        MockSpan._id_counter += 1
        self.name = name
        self.span_id = f"span_{MockSpan._id_counter:016x}"
        self.trace_id = parent.trace_id if parent else f"trace_{MockSpan._id_counter:032x}"
        self.parent_id = parent.span_id if parent else None
        self.attributes: dict[str, Any] = attributes or {}
        self.events: list[dict[str, Any]] = []
        self.status = "UNSET"
        self.status_description: str | None = None
        self.start_time = time.time()
        self.end_time: float | None = None
        self._is_recording = True
        self._exception: Exception | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute."""
        if self._is_recording:
            self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple attributes."""
        if self._is_recording:
            self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Add an event."""
        if self._is_recording:
            self.events.append({
                "name": name,
                "attributes": attributes or {},
                "timestamp": timestamp or time.time(),
            })

    def record_exception(self, exception: Exception) -> None:
        """Record an exception."""
        if self._is_recording:
            self._exception = exception
            self.add_event(
                "exception",
                {
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )

    def set_status(self, status: Any, description: str | None = None) -> None:
        """Set span status."""
        if hasattr(status, "status_code"):
            # Real Status object
            self.status = str(status.status_code.name) if hasattr(status.status_code, "name") else "OK"
            self.status_description = status.description
        elif hasattr(status, "value"):
            self.status = status.value
        else:
            self.status = str(status)
        if description:
            self.status_description = description

    def end(self, end_time: float | None = None) -> None:
        """End the span."""
        self.end_time = end_time or time.time()
        self._is_recording = False

    def is_recording(self) -> bool:
        """Check if span is recording."""
        return self._is_recording

    def get_span_context(self) -> "MockSpanContext":
        """Get span context."""
        return MockSpanContext(self.trace_id, self.span_id)


@dataclass
class MockSpanContext:
    """Mock OpenTelemetry SpanContext."""

    trace_id: str
    span_id: str
    is_valid: bool = True
    is_remote: bool = False
    trace_flags: "MockTraceFlags" = field(default_factory=lambda: MockTraceFlags())


@dataclass
class MockTraceFlags:
    """Mock OpenTelemetry TraceFlags."""

    sampled: bool = True


class MockTracer:
    """Mock OpenTelemetry Tracer."""

    def __init__(
        self,
        name: str,
        version: str | None = None,
    ) -> None:
        """Initialize MockTracer."""
        self.name = name
        self.version = version
        self.spans: list[MockSpan] = []
        self._current_span: MockSpan | None = None

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: dict[str, Any] | None = None,
        links: Sequence[Any] | None = None,
        start_time: float | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> MockSpan:
        """Start a new span."""
        span = MockSpan(name, parent=self._current_span, attributes=attributes)
        self.spans.append(span)
        return span

    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: dict[str, Any] | None = None,
        links: Sequence[Any] | None = None,
        start_time: float | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ):
        """Start a span and set as current (context manager)."""
        return MockSpanContextManager(self, name, attributes)


class MockSpanContextManager:
    """Context manager for mock spans."""

    def __init__(
        self,
        tracer: MockTracer,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Initialize context manager."""
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._span: MockSpan | None = None
        self._previous_span: MockSpan | None = None

    def __enter__(self) -> MockSpan:
        """Enter context."""
        self._previous_span = self._tracer._current_span
        self._span = self._tracer.start_span(self._name, attributes=self._attributes)
        self._tracer._current_span = self._span
        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        if self._span:
            if exc_val:
                self._span.record_exception(exc_val)
                self._span.set_status("ERROR", str(exc_val))
            self._span.end()
        self._tracer._current_span = self._previous_span


class MockTracerProvider:
    """Mock OpenTelemetry TracerProvider."""

    def __init__(self) -> None:
        """Initialize MockTracerProvider."""
        self.tracers: dict[str, MockTracer] = {}
        self._is_shutdown = False

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
    ) -> MockTracer:
        """Get or create a tracer."""
        key = f"{instrumenting_module_name}:{instrumenting_library_version or ''}"
        if key not in self.tracers:
            self.tracers[key] = MockTracer(instrumenting_module_name, instrumenting_library_version)
        return self.tracers[key]

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush (no-op for mock)."""
        return True

    def shutdown(self) -> bool:
        """Shutdown provider."""
        self._is_shutdown = True
        return True

    def add_span_processor(self, processor: Any) -> None:
        """Add span processor (no-op for mock)."""
        pass


class MockOTelEngineHook:
    """Mock OpenTelemetry engine hook for testing.

    Tracks all hook calls for verification in tests.
    """

    def __init__(self) -> None:
        """Initialize MockOTelEngineHook."""
        self.calls: list[dict[str, Any]] = []
        self.check_starts: list[dict[str, Any]] = []
        self.check_ends: list[dict[str, Any]] = []
        self.profile_starts: list[dict[str, Any]] = []
        self.profile_ends: list[dict[str, Any]] = []
        self.learn_starts: list[dict[str, Any]] = []
        self.learn_ends: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record check start."""
        data = {
            "engine_name": engine_name,
            "data_size": data_size,
            "context": context,
            "timestamp": time.time(),
        }
        self.calls.append({"type": "check_start", **data})
        self.check_starts.append(data)

    def on_check_end(
        self,
        engine_name: str,
        result: Any,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record check end."""
        data = {
            "engine_name": engine_name,
            "result": result,
            "duration_ms": duration_ms,
            "context": context,
            "timestamp": time.time(),
        }
        self.calls.append({"type": "check_end", **data})
        self.check_ends.append(data)

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record profile start."""
        data = {
            "engine_name": engine_name,
            "data_size": data_size,
            "context": context,
            "timestamp": time.time(),
        }
        self.calls.append({"type": "profile_start", **data})
        self.profile_starts.append(data)

    def on_profile_end(
        self,
        engine_name: str,
        result: Any,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record profile end."""
        data = {
            "engine_name": engine_name,
            "result": result,
            "duration_ms": duration_ms,
            "context": context,
            "timestamp": time.time(),
        }
        self.calls.append({"type": "profile_end", **data})
        self.profile_ends.append(data)

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Record learn start."""
        data = {
            "engine_name": engine_name,
            "data_size": data_size,
            "context": context,
            "timestamp": time.time(),
        }
        self.calls.append({"type": "learn_start", **data})
        self.learn_starts.append(data)

    def on_learn_end(
        self,
        engine_name: str,
        result: Any,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record learn end."""
        data = {
            "engine_name": engine_name,
            "result": result,
            "duration_ms": duration_ms,
            "context": context,
            "timestamp": time.time(),
        }
        self.calls.append({"type": "learn_end", **data})
        self.learn_ends.append(data)

    def on_error(
        self,
        engine_name: str,
        operation: Any,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record error."""
        data = {
            "engine_name": engine_name,
            "operation": operation,
            "exception": exception,
            "duration_ms": duration_ms,
            "context": context,
            "timestamp": time.time(),
        }
        self.calls.append({"type": "error", **data})
        self.errors.append(data)

    def reset(self) -> None:
        """Reset all recorded calls."""
        self.calls.clear()
        self.check_starts.clear()
        self.check_ends.clear()
        self.profile_starts.clear()
        self.profile_ends.clear()
        self.learn_starts.clear()
        self.learn_ends.clear()
        self.errors.clear()


def create_test_config() -> OTelConfig:
    """Create a configuration for testing.

    Returns configuration with in-memory exporters.
    """
    from common.opentelemetry.types import ExporterType

    return OTelConfig(
        enabled=True,
        metrics_exporter=ExporterType.MEMORY,
        traces_exporter=ExporterType.MEMORY,
        sampling=SamplingConfig(sample_rate=1.0),
    )


def create_disabled_config() -> OTelConfig:
    """Create a disabled configuration.

    Returns configuration with all features disabled.
    """
    from common.opentelemetry.config import DISABLED_OTEL_CONFIG
    return DISABLED_OTEL_CONFIG


def collect_spans(tracer: MockTracer | MockTracerProvider) -> list[MockSpan]:
    """Collect all spans from a mock tracer.

    Args:
        tracer: MockTracer or MockTracerProvider.

    Returns:
        List of all recorded spans.
    """
    if isinstance(tracer, MockTracerProvider):
        spans = []
        for t in tracer.tracers.values():
            spans.extend(t.spans)
        return spans
    return tracer.spans


def collect_metrics(meter: MockMeter | MockMeterProvider) -> dict[str, list[MockMetricData]]:
    """Collect all metrics from a mock meter.

    Args:
        meter: MockMeter or MockMeterProvider.

    Returns:
        Dictionary mapping metric names to recorded values.
    """
    result: dict[str, list[MockMetricData]] = {}

    if isinstance(meter, MockMeterProvider):
        for m in meter.meters.values():
            for name, counter in m.counters.items():
                result[name] = counter.values
            for name, gauge in m.gauges.items():
                result[name] = gauge.values
            for name, histogram in m.histograms.items():
                result[name] = histogram.values
    else:
        for name, counter in meter.counters.items():
            result[name] = counter.values
        for name, gauge in meter.gauges.items():
            result[name] = gauge.values
        for name, histogram in meter.histograms.items():
            result[name] = histogram.values

    return result


def reset_test_state() -> None:
    """Reset global test state.

    Call this in test teardown to ensure clean state.
    """
    MockSpan._id_counter = 0
