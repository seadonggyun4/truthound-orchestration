"""Tests for the metrics and tracing module."""

from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

# Check if pytest-asyncio is available
HAS_PYTEST_ASYNCIO = importlib.util.find_spec("pytest_asyncio") is not None

asyncio_test = pytest.mark.skipif(
    not HAS_PYTEST_ASYNCIO,
    reason="pytest-asyncio not installed"
)

from common.metrics import (
    # Exceptions
    MetricsError,
    TracingError,
    # Enums
    MetricType,
    SpanKind,
    SpanStatus,
    # Config
    MetricsConfig,
    TracingConfig,
    DEFAULT_METRICS_CONFIG,
    DEFAULT_TRACING_CONFIG,
    DISABLED_METRICS_CONFIG,
    DISABLED_TRACING_CONFIG,
    # Data types
    MetricData,
    SpanData,
    SpanEvent,
    TraceContext,
    # Metric types
    Counter,
    Gauge,
    Histogram,
    Summary,
    # Span
    Span,
    # Exporters
    ConsoleMetricExporter,
    ConsoleSpanExporter,
    InMemoryMetricExporter,
    InMemorySpanExporter,
    CompositeMetricExporter,
    CompositeSpanExporter,
    # Hooks
    LoggingMetricsHook,
    LoggingTracingHook,
    CompositeMetricsHook,
    CompositeTracingHook,
    # Registry
    MetricsRegistry,
    TracingRegistry,
    get_metrics_registry,
    get_tracing_registry,
    configure_metrics,
    configure_tracing,
    # Convenience functions
    counter,
    gauge,
    histogram,
    summary,
    # Decorators
    timed,
    counted,
    trace,
    instrumented,
    # Context propagation
    inject_context,
    extract_context,
)


# =============================================================================
# MetricsConfig Tests
# =============================================================================


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.prefix == ""
        assert config.default_labels == {}
        assert config.export_interval_seconds == 60.0
        assert config.max_export_batch_size == 1000

    def test_with_enabled(self) -> None:
        """Test with_enabled builder method."""
        config = MetricsConfig()
        disabled = config.with_enabled(False)
        assert disabled.enabled is False
        assert config.enabled is True  # Original unchanged

    def test_with_prefix(self) -> None:
        """Test with_prefix builder method."""
        config = MetricsConfig()
        prefixed = config.with_prefix("myapp")
        assert prefixed.prefix == "myapp"
        assert config.prefix == ""

    def test_with_default_labels(self) -> None:
        """Test with_default_labels builder method."""
        config = MetricsConfig(default_labels={"env": "prod"})
        updated = config.with_default_labels(service="api")
        assert updated.default_labels == {"env": "prod", "service": "api"}

    def test_with_histogram_buckets(self) -> None:
        """Test with_histogram_buckets builder method."""
        config = MetricsConfig()
        custom = config.with_histogram_buckets(0.1, 1.0, 10.0)
        assert custom.histogram_buckets == (0.1, 1.0, 10.0)

    def test_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValueError, match="export_interval_seconds"):
            MetricsConfig(export_interval_seconds=-1)

        with pytest.raises(ValueError, match="max_export_batch_size"):
            MetricsConfig(max_export_batch_size=0)

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        config = MetricsConfig(prefix="test", enabled=False)
        data = config.to_dict()
        assert data["prefix"] == "test"
        assert data["enabled"] is False

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {"prefix": "test", "enabled": False}
        config = MetricsConfig.from_dict(data)
        assert config.prefix == "test"
        assert config.enabled is False


# =============================================================================
# TracingConfig Tests
# =============================================================================


class TestTracingConfig:
    """Tests for TracingConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TracingConfig()
        assert config.enabled is True
        assert config.service_name == "truthound"
        assert config.sample_rate == 1.0
        assert config.propagate_context is True

    def test_with_sample_rate(self) -> None:
        """Test with_sample_rate builder method."""
        config = TracingConfig()
        sampled = config.with_sample_rate(0.1)
        assert sampled.sample_rate == 0.1
        assert config.sample_rate == 1.0

    def test_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValueError, match="sample_rate"):
            TracingConfig(sample_rate=-0.1)

        with pytest.raises(ValueError, match="sample_rate"):
            TracingConfig(sample_rate=1.5)


# =============================================================================
# Counter Tests
# =============================================================================


class TestCounter:
    """Tests for Counter metric."""

    def test_basic_increment(self) -> None:
        """Test basic counter increment."""
        counter = Counter("test_counter", "Test counter")
        assert counter.get() == 0.0

        counter.inc()
        assert counter.get() == 1.0

        counter.inc(5)
        assert counter.get() == 6.0

    def test_increment_with_labels(self) -> None:
        """Test counter increment with labels."""
        counter = Counter("requests", "Request count")

        counter.inc(labels={"method": "GET"})
        counter.inc(labels={"method": "POST"})
        counter.inc(2, labels={"method": "GET"})

        assert counter.get(labels={"method": "GET"}) == 3.0
        assert counter.get(labels={"method": "POST"}) == 1.0

    def test_negative_increment_raises(self) -> None:
        """Test that negative increment raises ValueError."""
        counter = Counter("test", "Test")
        with pytest.raises(ValueError, match="non-negative"):
            counter.inc(-1)

    def test_collect(self) -> None:
        """Test metric collection."""
        counter = Counter("test", "Test")
        counter.inc(labels={"a": "1"})
        counter.inc(2, labels={"a": "2"})

        metrics = counter.collect()
        assert len(metrics) == 2
        assert all(m.metric_type == MetricType.COUNTER for m in metrics)

    def test_reset(self) -> None:
        """Test counter reset."""
        counter = Counter("test", "Test")
        counter.inc(10)
        counter.reset()
        assert counter.get() == 0.0


# =============================================================================
# Gauge Tests
# =============================================================================


class TestGauge:
    """Tests for Gauge metric."""

    def test_set_value(self) -> None:
        """Test setting gauge value."""
        gauge = Gauge("temperature", "Current temperature")

        gauge.set(25.5)
        assert gauge.get() == 25.5

        gauge.set(30.0)
        assert gauge.get() == 30.0

    def test_inc_dec(self) -> None:
        """Test increment and decrement."""
        gauge = Gauge("connections", "Active connections")

        gauge.inc()
        assert gauge.get() == 1.0

        gauge.inc(5)
        assert gauge.get() == 6.0

        gauge.dec(2)
        assert gauge.get() == 4.0

    def test_track_inprogress(self) -> None:
        """Test in-progress tracking context manager."""
        gauge = Gauge("active_requests", "Active requests")

        with gauge.track_inprogress():
            assert gauge.get() == 1.0
        assert gauge.get() == 0.0

    def test_track_inprogress_with_labels(self) -> None:
        """Test in-progress tracking with labels."""
        gauge = Gauge("active", "Active")
        labels = {"endpoint": "/api"}

        with gauge.track_inprogress(labels=labels):
            assert gauge.get(labels=labels) == 1.0
        assert gauge.get(labels=labels) == 0.0


# =============================================================================
# Histogram Tests
# =============================================================================


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self) -> None:
        """Test observing values."""
        hist = Histogram("duration", "Duration", buckets=(0.1, 0.5, 1.0))

        hist.observe(0.05)
        hist.observe(0.3)
        hist.observe(0.8)
        hist.observe(1.5)

        assert hist.get_sample_count() == 4
        assert hist.get_sample_sum() == pytest.approx(2.65, rel=1e-3)

    def test_time_context_manager(self) -> None:
        """Test timing context manager."""
        hist = Histogram("test_duration", "Test", buckets=(0.1, 1.0))

        with hist.time():
            time.sleep(0.01)

        assert hist.get_sample_count() == 1
        assert hist.get_sample_sum() >= 0.01

    def test_buckets(self) -> None:
        """Test bucket counting."""
        hist = Histogram("test", "Test", buckets=(1.0, 5.0, 10.0))

        hist.observe(0.5)  # <= 1, 5, 10
        hist.observe(3.0)  # <= 5, 10
        hist.observe(7.0)  # <= 10
        hist.observe(15.0)  # > all buckets

        metrics = hist.collect()
        bucket_metrics = [m for m in metrics if "_bucket" in m.name]

        # Check that we have bucket metrics
        assert len(bucket_metrics) == 3

    def test_collect(self) -> None:
        """Test metric collection."""
        hist = Histogram("test", "Test", buckets=(1.0,))
        hist.observe(0.5)

        metrics = hist.collect()
        # Should have: 1 bucket + sum + count
        assert len(metrics) == 3
        names = [m.name for m in metrics]
        assert "test_bucket" in names
        assert "test_sum" in names
        assert "test_count" in names


# =============================================================================
# Summary Tests
# =============================================================================


class TestSummary:
    """Tests for Summary metric."""

    def test_observe(self) -> None:
        """Test observing values."""
        summary = Summary("response_size", "Response size")

        for i in range(100):
            summary.observe(i)

        metrics = summary.collect()
        # Check we have quantile, sum, and count metrics
        quantile_metrics = [m for m in metrics if "quantile" in m.labels]
        assert len(quantile_metrics) == 4  # Default quantiles

    def test_time_context_manager(self) -> None:
        """Test timing context manager."""
        summary = Summary("test_duration", "Test")

        with summary.time():
            time.sleep(0.01)

        metrics = summary.collect()
        sum_metric = next(m for m in metrics if "_sum" in m.name)
        assert sum_metric.value >= 0.01

    def test_max_samples(self) -> None:
        """Test sample limiting."""
        summary = Summary("test", "Test", max_samples=10)

        for i in range(100):
            summary.observe(i)

        # Should still work but only keep last 10 samples
        metrics = summary.collect()
        count_metric = next(m for m in metrics if "_count" in m.name)
        assert count_metric.value == 100  # Total count tracked


# =============================================================================
# Span Tests
# =============================================================================


class TestSpan:
    """Tests for Span."""

    def test_basic_span(self) -> None:
        """Test basic span creation and usage."""
        with Span("test_operation") as span:
            span.set_attribute("key", "value")

        data = span.to_span_data()
        assert data.name == "test_operation"
        assert data.status == SpanStatus.OK
        assert data.attributes["key"] == "value"
        assert data.duration_ms > 0

    def test_nested_spans(self) -> None:
        """Test nested span creation."""
        with Span("parent") as parent:
            with Span("child", parent=parent) as child:
                pass

        parent_data = parent.to_span_data()
        child_data = child.to_span_data()

        assert child_data.trace_id == parent_data.trace_id
        assert child_data.parent_span_id == parent_data.span_id

    def test_span_events(self) -> None:
        """Test adding events to span."""
        with Span("test") as span:
            span.add_event("step1", {"detail": "starting"})
            span.add_event("step2")

        data = span.to_span_data()
        assert len(data.events) == 2
        assert data.events[0].name == "step1"

    def test_record_exception(self) -> None:
        """Test exception recording."""
        with Span("test") as span:
            try:
                raise ValueError("test error")
            except ValueError as e:
                span.record_exception(e)

        data = span.to_span_data()
        assert data.status == SpanStatus.ERROR
        assert len(data.events) == 1
        assert data.events[0].name == "exception"

    def test_span_context_manager_exception(self) -> None:
        """Test span handles exceptions in context manager."""
        with pytest.raises(RuntimeError):
            with Span("test") as span:
                raise RuntimeError("test")

        data = span.to_span_data()
        assert data.status == SpanStatus.ERROR


# =============================================================================
# Exporter Tests
# =============================================================================


class TestExporters:
    """Tests for exporters."""

    def test_in_memory_metric_exporter(self) -> None:
        """Test InMemoryMetricExporter."""
        exporter = InMemoryMetricExporter()

        metric = MetricData(
            name="test",
            metric_type=MetricType.COUNTER,
            value=1.0,
        )
        exporter.export([metric])

        assert len(exporter.metrics) == 1
        assert exporter.metrics[0].name == "test"

        exporter.clear()
        assert len(exporter.metrics) == 0

    def test_in_memory_span_exporter(self) -> None:
        """Test InMemorySpanExporter."""
        exporter = InMemorySpanExporter()

        span = SpanData(
            name="test",
            trace_id="abc123",
            span_id="def456",
        )
        exporter.export([span])

        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "test"

        exporter.clear()
        assert len(exporter.spans) == 0

    def test_composite_metric_exporter(self) -> None:
        """Test CompositeMetricExporter."""
        exporter1 = InMemoryMetricExporter()
        exporter2 = InMemoryMetricExporter()
        composite = CompositeMetricExporter([exporter1, exporter2])

        metric = MetricData(
            name="test",
            metric_type=MetricType.GAUGE,
            value=42.0,
        )
        composite.export([metric])

        assert len(exporter1.metrics) == 1
        assert len(exporter2.metrics) == 1

    def test_composite_span_exporter(self) -> None:
        """Test CompositeSpanExporter."""
        exporter1 = InMemorySpanExporter()
        exporter2 = InMemorySpanExporter()
        composite = CompositeSpanExporter([exporter1, exporter2])

        span = SpanData(
            name="test",
            trace_id="abc123",
            span_id="def456",
        )
        composite.export([span])

        assert len(exporter1.spans) == 1
        assert len(exporter2.spans) == 1


# =============================================================================
# Registry Tests
# =============================================================================


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_create_counter(self) -> None:
        """Test counter creation through registry."""
        registry = MetricsRegistry()
        c = registry.counter("test_counter", "Test")

        c.inc()
        assert c.get() == 1.0

    def test_create_gauge(self) -> None:
        """Test gauge creation through registry."""
        registry = MetricsRegistry()
        g = registry.gauge("test_gauge", "Test")

        g.set(42.0)
        assert g.get() == 42.0

    def test_create_histogram(self) -> None:
        """Test histogram creation through registry."""
        registry = MetricsRegistry()
        h = registry.histogram("test_hist", "Test")

        h.observe(0.5)
        assert h.get_sample_count() == 1

    def test_create_summary(self) -> None:
        """Test summary creation through registry."""
        registry = MetricsRegistry()
        s = registry.summary("test_summary", "Test")

        s.observe(100)
        metrics = s.collect()
        assert len(metrics) > 0

    def test_prefix(self) -> None:
        """Test metric name prefixing."""
        config = MetricsConfig(prefix="myapp")
        registry = MetricsRegistry(config=config)

        c = registry.counter("requests", "Requests")
        assert c.name == "myapp_requests"

    def test_collect(self) -> None:
        """Test collecting all metrics."""
        registry = MetricsRegistry()
        c = registry.counter("c", "Counter")
        g = registry.gauge("g", "Gauge")

        c.inc(5)
        g.set(10)

        metrics = registry.collect()
        assert len(metrics) == 2

    def test_export(self) -> None:
        """Test exporting metrics."""
        exporter = InMemoryMetricExporter()
        registry = MetricsRegistry(exporters=[exporter])

        c = registry.counter("test", "Test")
        c.inc()

        registry.export()
        assert len(exporter.metrics) == 1

    def test_duplicate_metric_same_type(self) -> None:
        """Test getting existing metric of same type."""
        registry = MetricsRegistry()
        c1 = registry.counter("test", "Test")
        c2 = registry.counter("test", "Test")

        assert c1 is c2

    def test_duplicate_metric_different_type(self) -> None:
        """Test error when creating metric with different type."""
        registry = MetricsRegistry()
        registry.counter("test", "Test")

        with pytest.raises(MetricsError):
            registry.gauge("test", "Test")


class TestTracingRegistry:
    """Tests for TracingRegistry."""

    def test_start_span(self) -> None:
        """Test starting a span."""
        registry = TracingRegistry()
        span = registry.start_span("test")

        assert span.name == "test"
        assert span.is_recording

    def test_trace_context_manager(self) -> None:
        """Test trace context manager."""
        exporter = InMemorySpanExporter()
        registry = TracingRegistry(exporters=[exporter])

        with registry.trace("test_operation") as span:
            span.set_attribute("key", "value")

        registry.export()
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "test_operation"

    def test_nested_traces(self) -> None:
        """Test nested trace context managers."""
        exporter = InMemorySpanExporter()
        registry = TracingRegistry(exporters=[exporter])

        with registry.trace("parent") as parent_span:
            with registry.trace("child") as child_span:
                pass

        registry.export()
        assert len(exporter.spans) == 2

        parent = next(s for s in exporter.spans if s.name == "parent")
        child = next(s for s in exporter.spans if s.name == "child")

        assert child.trace_id == parent.trace_id
        assert child.parent_span_id == parent.span_id

    def test_sampling(self) -> None:
        """Test trace sampling."""
        config = TracingConfig(sample_rate=0.0)  # Never sample
        exporter = InMemorySpanExporter()
        registry = TracingRegistry(config=config, exporters=[exporter])

        with registry.trace("test"):
            pass

        registry.export()
        # Should not export when not sampled
        assert len(exporter.spans) == 0


# =============================================================================
# Decorator Tests
# =============================================================================


class TestDecorators:
    """Tests for metric and tracing decorators."""

    def test_timed_decorator(self) -> None:
        """Test @timed decorator."""
        registry = MetricsRegistry()

        @timed("test_func_duration")
        def test_func() -> str:
            time.sleep(0.01)
            return "done"

        # Reconfigure to use our registry
        configure_metrics()

        result = test_func()
        assert result == "done"

    def test_counted_decorator(self) -> None:
        """Test @counted decorator."""
        configure_metrics()

        @counted("test_calls")
        def test_func() -> str:
            return "done"

        result = test_func()
        assert result == "done"

    def test_counted_with_exception(self) -> None:
        """Test @counted decorator with exception."""
        configure_metrics()

        @counted("error_calls")
        def failing_func() -> None:
            raise RuntimeError("test")

        with pytest.raises(RuntimeError):
            failing_func()

    def test_trace_decorator(self) -> None:
        """Test @trace decorator."""
        exporter = InMemorySpanExporter()
        configure_tracing(exporters=[exporter])

        @trace(name="traced_operation")
        def traced_func() -> str:
            return "done"

        result = traced_func()
        assert result == "done"

        get_tracing_registry().export()
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "traced_operation"

    def test_instrumented_decorator(self) -> None:
        """Test @instrumented decorator combining metrics and tracing."""
        metric_exporter = InMemoryMetricExporter()
        span_exporter = InMemorySpanExporter()

        configure_metrics(exporters=[metric_exporter])
        configure_tracing(exporters=[span_exporter])

        @instrumented()
        def instrumented_func() -> str:
            return "done"

        result = instrumented_func()
        assert result == "done"

        get_metrics_registry().export()
        get_tracing_registry().export()

        assert len(span_exporter.spans) == 1

    @asyncio_test
    @pytest.mark.asyncio
    async def test_timed_async(self) -> None:
        """Test @timed decorator with async function."""
        configure_metrics()

        @timed("async_duration")
        async def async_func() -> str:
            await asyncio.sleep(0.01)
            return "done"

        result = await async_func()
        assert result == "done"

    @asyncio_test
    @pytest.mark.asyncio
    async def test_trace_async(self) -> None:
        """Test @trace decorator with async function."""
        exporter = InMemorySpanExporter()
        configure_tracing(exporters=[exporter])

        @trace(name="async_traced")
        async def async_traced() -> str:
            return "done"

        result = await async_traced()
        assert result == "done"

        get_tracing_registry().export()
        assert len(exporter.spans) == 1


# =============================================================================
# Context Propagation Tests
# =============================================================================


class TestContextPropagation:
    """Tests for trace context propagation."""

    def test_trace_context_to_headers(self) -> None:
        """Test converting TraceContext to headers."""
        ctx = TraceContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            sampled=True,
        )

        headers = ctx.to_headers()
        assert "traceparent" in headers
        assert "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01" == headers["traceparent"]

    def test_trace_context_from_headers(self) -> None:
        """Test parsing TraceContext from headers."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }

        ctx = TraceContext.from_headers(headers)
        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.sampled is True

    def test_trace_context_from_invalid_headers(self) -> None:
        """Test parsing invalid headers returns None."""
        assert TraceContext.from_headers({}) is None
        assert TraceContext.from_headers({"traceparent": "invalid"}) is None

    def test_inject_context(self) -> None:
        """Test injecting context into headers."""
        configure_tracing()
        registry = get_tracing_registry()

        with registry.trace("test"):
            headers: dict[str, str] = {}
            inject_context(headers)
            assert "traceparent" in headers

    def test_extract_context(self) -> None:
        """Test extracting context from headers."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }

        ctx = extract_context(headers)
        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"


# =============================================================================
# Data Type Tests
# =============================================================================


class TestMetricData:
    """Tests for MetricData."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        data = MetricData(
            name="test",
            metric_type=MetricType.COUNTER,
            value=42.0,
            labels={"key": "value"},
        )

        d = data.to_dict()
        assert d["name"] == "test"
        assert d["type"] == "COUNTER"
        assert d["value"] == 42.0
        assert d["labels"] == {"key": "value"}

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "name": "test",
            "type": "GAUGE",
            "value": 10.0,
            "labels": {"a": "b"},
        }

        data = MetricData.from_dict(d)
        assert data.name == "test"
        assert data.metric_type == MetricType.GAUGE
        assert data.value == 10.0


class TestSpanData:
    """Tests for SpanData."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        span = SpanData(
            name="test",
            trace_id="abc123",
            span_id="def456",
            status=SpanStatus.OK,
        )

        d = span.to_dict()
        assert d["name"] == "test"
        assert d["trace_id"] == "abc123"
        assert d["status"] == "OK"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "name": "test",
            "trace_id": "abc123",
            "span_id": "def456",
            "status": "ERROR",
        }

        span = SpanData.from_dict(d)
        assert span.name == "test"
        assert span.status == SpanStatus.ERROR


# =============================================================================
# Hook Tests
# =============================================================================


class TestHooks:
    """Tests for hooks."""

    def test_composite_metrics_hook(self) -> None:
        """Test CompositeMetricsHook."""
        mock1 = MagicMock()
        mock2 = MagicMock()

        # Add required methods
        mock1.on_record = MagicMock()
        mock2.on_record = MagicMock()

        hook = CompositeMetricsHook([mock1, mock2])
        hook.on_record("test", MetricType.COUNTER, 1.0, {}, {})

        mock1.on_record.assert_called_once()
        mock2.on_record.assert_called_once()

    def test_composite_tracing_hook(self) -> None:
        """Test CompositeTracingHook."""
        mock1 = MagicMock()
        mock2 = MagicMock()

        mock1.on_span_start = MagicMock()
        mock1.on_span_end = MagicMock()
        mock2.on_span_start = MagicMock()
        mock2.on_span_end = MagicMock()

        hook = CompositeTracingHook([mock1, mock2])

        span = Span("test")
        hook.on_span_start(span, {})
        hook.on_span_end(span, {})

        mock1.on_span_start.assert_called_once()
        mock2.on_span_end.assert_called_once()


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_metrics_registry(self) -> None:
        """Test getting global metrics registry."""
        registry = get_metrics_registry()
        assert isinstance(registry, MetricsRegistry)

    def test_get_tracing_registry(self) -> None:
        """Test getting global tracing registry."""
        registry = get_tracing_registry()
        assert isinstance(registry, TracingRegistry)

    def test_convenience_counter(self) -> None:
        """Test convenience counter function."""
        configure_metrics()
        c = counter("global_counter", "Global")
        c.inc()
        assert c.get() == 1.0

    def test_convenience_gauge(self) -> None:
        """Test convenience gauge function."""
        configure_metrics()
        g = gauge("global_gauge", "Global")
        g.set(42)
        assert g.get() == 42.0

    def test_convenience_histogram(self) -> None:
        """Test convenience histogram function."""
        configure_metrics()
        h = histogram("global_hist", "Global")
        h.observe(0.5)
        assert h.get_sample_count() == 1

    def test_convenience_summary(self) -> None:
        """Test convenience summary function."""
        configure_metrics()
        s = summary("global_summary", "Global")
        s.observe(100)
        metrics = s.collect()
        assert len(metrics) > 0
