"""Tests for OpenTelemetry testing utilities."""

import pytest

from common.opentelemetry.testing import (
    MockCounter,
    MockGauge,
    MockHistogram,
    MockMeter,
    MockMeterProvider,
    MockOTelEngineHook,
    MockSpan,
    MockTracer,
    MockTracerProvider,
    collect_metrics,
    collect_spans,
    create_disabled_config,
    create_test_config,
    reset_test_state,
)
from common.opentelemetry.types import ExporterType


class TestMockCounter:
    """Tests for MockCounter."""

    def test_creation(self):
        """Test creating a counter."""
        counter = MockCounter("requests_total", "Total requests")
        assert counter.name == "requests_total"
        assert counter.description == "Total requests"
        assert counter.total == 0

    def test_add(self):
        """Test adding values."""
        counter = MockCounter("requests_total")
        counter.add(1)
        counter.add(5)
        assert counter.total == 6
        assert len(counter.values) == 2

    def test_add_with_attributes(self):
        """Test adding values with attributes."""
        counter = MockCounter("requests_total")
        counter.add(1, {"method": "GET"})
        counter.add(2, {"method": "POST"})
        assert counter.total == 3
        assert counter.values[0].attributes["method"] == "GET"
        assert counter.values[1].attributes["method"] == "POST"


class TestMockGauge:
    """Tests for MockGauge."""

    def test_creation(self):
        """Test creating a gauge."""
        gauge = MockGauge("active_connections")
        assert gauge.name == "active_connections"
        assert gauge.current == 0

    def test_add(self):
        """Test adding values."""
        gauge = MockGauge("active_connections")
        gauge.add(10)
        gauge.add(-3)
        assert gauge.current == 7


class TestMockHistogram:
    """Tests for MockHistogram."""

    def test_creation(self):
        """Test creating a histogram."""
        histogram = MockHistogram("request_duration", unit="s")
        assert histogram.name == "request_duration"
        assert histogram.unit == "s"
        assert histogram.count == 0

    def test_record(self):
        """Test recording values."""
        histogram = MockHistogram("request_duration")
        histogram.record(0.1)
        histogram.record(0.5)
        histogram.record(1.0)
        assert histogram.count == 3
        assert histogram.sum == 1.6


class TestMockMeter:
    """Tests for MockMeter."""

    def test_creation(self):
        """Test creating a meter."""
        meter = MockMeter("my-meter", "1.0.0")
        assert meter.name == "my-meter"
        assert meter.version == "1.0.0"

    def test_create_counter(self):
        """Test creating a counter."""
        meter = MockMeter("test")
        counter = meter.create_counter("requests_total", "Total requests")
        assert isinstance(counter, MockCounter)
        assert meter.counters["requests_total"] is counter

    def test_create_up_down_counter(self):
        """Test creating an up-down counter (gauge)."""
        meter = MockMeter("test")
        gauge = meter.create_up_down_counter("active_connections")
        assert isinstance(gauge, MockGauge)
        assert meter.gauges["active_connections"] is gauge

    def test_create_histogram(self):
        """Test creating a histogram."""
        meter = MockMeter("test")
        histogram = meter.create_histogram("request_duration")
        assert isinstance(histogram, MockHistogram)
        assert meter.histograms["request_duration"] is histogram


class TestMockMeterProvider:
    """Tests for MockMeterProvider."""

    def test_creation(self):
        """Test creating a meter provider."""
        provider = MockMeterProvider()
        assert len(provider.meters) == 0

    def test_get_meter(self):
        """Test getting a meter."""
        provider = MockMeterProvider()
        meter = provider.get_meter("my-meter", "1.0.0")
        assert isinstance(meter, MockMeter)
        assert meter.name == "my-meter"

    def test_get_meter_caching(self):
        """Test meter caching."""
        provider = MockMeterProvider()
        meter1 = provider.get_meter("my-meter", "1.0.0")
        meter2 = provider.get_meter("my-meter", "1.0.0")
        assert meter1 is meter2

    def test_shutdown(self):
        """Test shutdown."""
        provider = MockMeterProvider()
        assert provider.shutdown() is True
        assert provider._is_shutdown is True


class TestMockSpan:
    """Tests for MockSpan."""

    def test_creation(self):
        """Test creating a span."""
        span = MockSpan("test-operation")
        assert span.name == "test-operation"
        assert span.span_id is not None
        assert span.trace_id is not None
        assert span.parent_id is None

    def test_parent_child(self):
        """Test parent-child relationship."""
        parent = MockSpan("parent")
        child = MockSpan("child", parent=parent)
        assert child.parent_id == parent.span_id
        assert child.trace_id == parent.trace_id

    def test_set_attribute(self):
        """Test setting attributes."""
        span = MockSpan("test")
        span.set_attribute("key", "value")
        span.set_attribute("number", 42)
        assert span.attributes["key"] == "value"
        assert span.attributes["number"] == 42

    def test_add_event(self):
        """Test adding events."""
        span = MockSpan("test")
        span.add_event("my-event", {"detail": "info"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "my-event"
        assert span.events[0]["attributes"]["detail"] == "info"

    def test_record_exception(self):
        """Test recording exceptions."""
        span = MockSpan("test")
        exc = ValueError("test error")
        span.record_exception(exc)
        assert span._exception is exc
        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"

    def test_end(self):
        """Test ending span."""
        span = MockSpan("test")
        assert span.is_recording() is True
        span.end()
        assert span.is_recording() is False
        assert span.end_time is not None


class TestMockTracer:
    """Tests for MockTracer."""

    def test_creation(self):
        """Test creating a tracer."""
        tracer = MockTracer("my-tracer", "1.0.0")
        assert tracer.name == "my-tracer"
        assert tracer.version == "1.0.0"

    def test_start_span(self):
        """Test starting a span."""
        tracer = MockTracer("test")
        span = tracer.start_span("my-operation")
        assert isinstance(span, MockSpan)
        assert span in tracer.spans

    def test_start_as_current_span(self):
        """Test context manager."""
        tracer = MockTracer("test")
        with tracer.start_as_current_span("my-operation") as span:
            assert isinstance(span, MockSpan)
            assert span.is_recording()
        assert not span.is_recording()


class TestMockTracerProvider:
    """Tests for MockTracerProvider."""

    def test_creation(self):
        """Test creating a tracer provider."""
        provider = MockTracerProvider()
        assert len(provider.tracers) == 0

    def test_get_tracer(self):
        """Test getting a tracer."""
        provider = MockTracerProvider()
        tracer = provider.get_tracer("my-tracer", "1.0.0")
        assert isinstance(tracer, MockTracer)

    def test_get_tracer_caching(self):
        """Test tracer caching."""
        provider = MockTracerProvider()
        tracer1 = provider.get_tracer("my-tracer", "1.0.0")
        tracer2 = provider.get_tracer("my-tracer", "1.0.0")
        assert tracer1 is tracer2

    def test_shutdown(self):
        """Test shutdown."""
        provider = MockTracerProvider()
        assert provider.shutdown() is True
        assert provider._is_shutdown is True


class TestMockOTelEngineHook:
    """Tests for MockOTelEngineHook."""

    def test_creation(self):
        """Test creating the hook."""
        hook = MockOTelEngineHook()
        assert len(hook.calls) == 0

    def test_on_check_start(self):
        """Test check start recording."""
        hook = MockOTelEngineHook()
        hook.on_check_start("truthound", 1000, {"key": "value"})
        assert len(hook.calls) == 1
        assert hook.calls[0]["type"] == "check_start"
        assert len(hook.check_starts) == 1

    def test_on_check_end(self):
        """Test check end recording."""
        hook = MockOTelEngineHook()
        hook.on_check_end("truthound", None, 123.4, {})
        assert len(hook.calls) == 1
        assert hook.calls[0]["type"] == "check_end"
        assert hook.calls[0]["duration_ms"] == 123.4

    def test_on_error(self):
        """Test error recording."""
        hook = MockOTelEngineHook()
        exc = ValueError("test error")
        hook.on_error("truthound", "check", exc, 50.0, {})
        assert len(hook.errors) == 1
        assert hook.errors[0]["exception"] is exc

    def test_reset(self):
        """Test resetting recorded calls."""
        hook = MockOTelEngineHook()
        hook.on_check_start("truthound", 1000, {})
        hook.on_check_end("truthound", None, 100.0, {})
        assert len(hook.calls) == 2
        hook.reset()
        assert len(hook.calls) == 0
        assert len(hook.check_starts) == 0
        assert len(hook.check_ends) == 0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_test_config(self):
        """Test creating test configuration."""
        config = create_test_config()
        assert config.enabled is True
        assert config.metrics_exporter == ExporterType.MEMORY
        assert config.traces_exporter == ExporterType.MEMORY
        assert config.sampling.sample_rate == 1.0

    def test_create_disabled_config(self):
        """Test creating disabled configuration."""
        config = create_disabled_config()
        assert config.enabled is False

    def test_collect_spans(self):
        """Test collecting spans from tracer."""
        tracer = MockTracer("test")
        tracer.start_span("span1")
        tracer.start_span("span2")
        spans = collect_spans(tracer)
        assert len(spans) == 2

    def test_collect_spans_from_provider(self):
        """Test collecting spans from provider."""
        provider = MockTracerProvider()
        tracer = provider.get_tracer("test")
        tracer.start_span("span1")
        tracer.start_span("span2")
        spans = collect_spans(provider)
        assert len(spans) == 2

    def test_collect_metrics(self):
        """Test collecting metrics from meter."""
        meter = MockMeter("test")
        counter = meter.create_counter("requests")
        counter.add(5)
        histogram = meter.create_histogram("latency")
        histogram.record(0.1)
        metrics = collect_metrics(meter)
        assert "requests" in metrics
        assert "latency" in metrics

    def test_reset_test_state(self):
        """Test resetting test state."""
        # Create some spans to increment counter
        span1 = MockSpan("test1")
        span2 = MockSpan("test2")
        reset_test_state()
        span3 = MockSpan("test3")
        # After reset, span IDs should start from 1 again (hex format)
        assert span3.span_id == "span_0000000000000001"
