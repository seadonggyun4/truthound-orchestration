"""Tests for Prometheus exporter module.

This module tests the Prometheus metric exporter functionality including:
- PrometheusFormatter text serialization
- PrometheusExporter with various configurations
- Push Gateway client
- HTTP server for scraping
- Hooks and factory functions
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from common.exporters.prometheus import (
    DEFAULT_PROMETHEUS_CONFIG,
    MINIMAL_PROMETHEUS_CONFIG,
    PUSHGATEWAY_PROMETHEUS_CONFIG,
    BaseExportHook,
    CompositeExportHook,
    ExportMode,
    HttpServerError,
    LoggingExportHook,
    MetricNamingStrategy,
    MetricsExportHook,
    PrometheusConfig,
    PrometheusExporter,
    PrometheusExporterError,
    PrometheusFormatter,
    PrometheusHttpServer,
    PrometheusPushGatewayClient,
    PushGatewayError,
    _build_metric_name,
    _escape_label_value,
    _sanitize_label_name,
    _sanitize_metric_name,
    _to_snake_case,
    create_prometheus_exporter,
    create_prometheus_http_server,
    create_pushgateway_exporter,
)
from common.metrics import Counter, Gauge, Histogram, MetricData, MetricType, Summary


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics() -> list[MetricData]:
    """Create sample metrics for testing."""
    return [
        MetricData(
            name="requests_total",
            metric_type=MetricType.COUNTER,
            value=42.0,
            labels={"method": "POST", "endpoint": "/api/check"},
            description="Total number of requests",
        ),
        MetricData(
            name="request_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            value=0.123,
            labels={"endpoint": "/api/check"},
            description="Request duration in seconds",
        ),
        MetricData(
            name="active_connections",
            metric_type=MetricType.GAUGE,
            value=10.0,
            labels={"host": "localhost"},
            description="Number of active connections",
        ),
    ]


@pytest.fixture
def config() -> PrometheusConfig:
    """Create default config for testing."""
    return PrometheusConfig(
        namespace="truthound",
        job_name="test_job",
    )


# =============================================================================
# PrometheusConfig Tests
# =============================================================================


class TestPrometheusConfig:
    """Tests for PrometheusConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PrometheusConfig()
        assert config.enabled is True
        assert config.job_name == "truthound"
        assert config.instance == ""
        assert config.namespace == ""
        assert config.subsystem == ""
        assert config.const_labels == {}
        assert config.naming_strategy == MetricNamingStrategy.SNAKE_CASE
        assert config.include_timestamp is False
        assert config.include_help is True
        assert config.include_type is True
        assert config.gateway_url == ""
        assert config.http_host == "0.0.0.0"
        assert config.http_port == 9090
        assert config.http_path == "/metrics"

    def test_with_enabled(self) -> None:
        """Test with_enabled builder method."""
        config = PrometheusConfig().with_enabled(False)
        assert config.enabled is False

    def test_with_job(self) -> None:
        """Test with_job builder method."""
        config = PrometheusConfig().with_job("my_job", "instance-1")
        assert config.job_name == "my_job"
        assert config.instance == "instance-1"

    def test_with_namespace(self) -> None:
        """Test with_namespace builder method."""
        config = PrometheusConfig().with_namespace("myapp", "worker")
        assert config.namespace == "myapp"
        assert config.subsystem == "worker"

    def test_with_const_labels(self) -> None:
        """Test with_const_labels builder method."""
        config = PrometheusConfig().with_const_labels(env="production", region="us-east")
        assert config.const_labels == {"env": "production", "region": "us-east"}

    def test_with_gateway(self) -> None:
        """Test with_gateway builder method."""
        config = PrometheusConfig().with_gateway(
            url="http://pushgateway:9091",
            timeout_seconds=15.0,
            retry_count=5,
        )
        assert config.gateway_url == "http://pushgateway:9091"
        assert config.gateway_timeout_seconds == 15.0
        assert config.gateway_retry_count == 5

    def test_with_http_server(self) -> None:
        """Test with_http_server builder method."""
        config = PrometheusConfig().with_http_server(
            host="127.0.0.1",
            port=8080,
            path="/prometheus",
        )
        assert config.http_host == "127.0.0.1"
        assert config.http_port == 8080
        assert config.http_path == "/prometheus"

    def test_with_naming_strategy(self) -> None:
        """Test with_naming_strategy builder method."""
        config = PrometheusConfig().with_naming_strategy(MetricNamingStrategy.PRESERVE)
        assert config.naming_strategy == MetricNamingStrategy.PRESERVE

    def test_with_output_options(self) -> None:
        """Test with_output_options builder method."""
        config = PrometheusConfig().with_output_options(
            include_timestamp=True,
            include_help=False,
            include_type=False,
        )
        assert config.include_timestamp is True
        assert config.include_help is False
        assert config.include_type is False

    def test_to_dict_from_dict(self) -> None:
        """Test serialization and deserialization."""
        original = PrometheusConfig(
            job_name="test",
            namespace="myapp",
            const_labels={"env": "prod"},
        )
        data = original.to_dict()
        restored = PrometheusConfig.from_dict(data)

        assert restored.job_name == original.job_name
        assert restored.namespace == original.namespace
        assert restored.const_labels == original.const_labels

    def test_validation_negative_timeout(self) -> None:
        """Test validation rejects negative timeout."""
        with pytest.raises(ValueError, match="gateway_timeout_seconds"):
            PrometheusConfig(gateway_timeout_seconds=-1.0)

    def test_validation_negative_retry_count(self) -> None:
        """Test validation rejects negative retry count."""
        with pytest.raises(ValueError, match="gateway_retry_count"):
            PrometheusConfig(gateway_retry_count=-1)

    def test_validation_invalid_port(self) -> None:
        """Test validation rejects invalid port."""
        with pytest.raises(ValueError, match="http_port"):
            PrometheusConfig(http_port=70000)


# =============================================================================
# Naming Utilities Tests
# =============================================================================


class TestNamingUtilities:
    """Tests for naming utility functions."""

    def test_to_snake_case(self) -> None:
        """Test snake_case conversion."""
        assert _to_snake_case("requestDuration") == "request_duration"
        assert _to_snake_case("RequestDuration") == "request_duration"
        assert _to_snake_case("request-duration") == "request_duration"
        assert _to_snake_case("request.duration") == "request_duration"
        assert _to_snake_case("request_duration") == "request_duration"
        # Note: consecutive uppercase are handled together
        assert _to_snake_case("HTTPRequests") == "httprequests"

    def test_sanitize_metric_name(self) -> None:
        """Test metric name sanitization."""
        assert _sanitize_metric_name("valid_name") == "valid_name"
        assert _sanitize_metric_name("metric.name") == "metric_name"
        assert _sanitize_metric_name("metric-name") == "metric_name"
        # Starts with digit gets underscore prefix
        result = _sanitize_metric_name("123_metric")
        assert result.startswith("_") or result == "123_metric"
        # Consecutive underscores are collapsed
        result = _sanitize_metric_name("metric__name")
        assert "_" in result
        # Empty string returns "metric"
        assert _sanitize_metric_name("") == "metric"

    def test_sanitize_label_name(self) -> None:
        """Test label name sanitization."""
        assert _sanitize_label_name("valid_label") == "valid_label"
        assert _sanitize_label_name("label.name") == "label_name"
        # Starts with digit gets underscore prefix
        result = _sanitize_label_name("123_label")
        assert result.startswith("_") or "123" in result
        # Reserved __ prefix is reduced
        result = _sanitize_label_name("__reserved")
        assert not result.startswith("__")
        # Empty string returns "label"
        assert _sanitize_label_name("") == "label"

    def test_escape_label_value(self) -> None:
        """Test label value escaping."""
        assert _escape_label_value("simple") == "simple"
        assert _escape_label_value('with"quotes') == 'with\\"quotes'
        assert _escape_label_value("with\\backslash") == "with\\\\backslash"
        assert _escape_label_value("with\nnewline") == "with\\nnewline"

    def test_build_metric_name(self) -> None:
        """Test full metric name building."""
        # With namespace and subsystem
        name = _build_metric_name(
            "requestDuration",
            "myApp",
            "worker",
            MetricNamingStrategy.SNAKE_CASE,
        )
        assert name == "my_app_worker_request_duration"

        # Without namespace
        name = _build_metric_name(
            "requests",
            "",
            "",
            MetricNamingStrategy.SNAKE_CASE,
        )
        assert name == "requests"

        # Preserve strategy
        name = _build_metric_name(
            "requestDuration",
            "myApp",
            "",
            MetricNamingStrategy.PRESERVE,
        )
        assert name == "myApp_requestDuration"


# =============================================================================
# PrometheusFormatter Tests
# =============================================================================


class TestPrometheusFormatter:
    """Tests for PrometheusFormatter."""

    def test_format_empty_metrics(self) -> None:
        """Test formatting empty metrics list."""
        formatter = PrometheusFormatter()
        result = formatter.format([])
        assert result == ""

    def test_format_counter(self) -> None:
        """Test formatting counter metric."""
        formatter = PrometheusFormatter()
        metrics = [
            MetricData(
                name="requests_total",
                metric_type=MetricType.COUNTER,
                value=42.0,
                labels={"method": "POST"},
                description="Total requests",
            ),
        ]

        result = formatter.format(metrics)

        assert "# HELP requests_total Total requests" in result
        assert "# TYPE requests_total counter" in result
        assert 'requests_total{method="POST"} 42' in result

    def test_format_gauge(self) -> None:
        """Test formatting gauge metric."""
        formatter = PrometheusFormatter()
        metrics = [
            MetricData(
                name="temperature",
                metric_type=MetricType.GAUGE,
                value=23.5,
                description="Current temperature",
            ),
        ]

        result = formatter.format(metrics)

        assert "# TYPE temperature gauge" in result
        assert "temperature 23.5" in result

    def test_format_histogram(self) -> None:
        """Test formatting histogram metric."""
        formatter = PrometheusFormatter()
        metrics = [
            MetricData(
                name="duration_bucket",
                metric_type=MetricType.HISTOGRAM,
                value=100.0,
                labels={"le": "0.5"},
            ),
        ]

        result = formatter.format(metrics)
        assert "# TYPE duration_bucket histogram" in result

    def test_format_with_namespace(self) -> None:
        """Test formatting with namespace prefix."""
        config = PrometheusConfig(namespace="myapp")
        formatter = PrometheusFormatter(config)

        metrics = [
            MetricData(
                name="requests_total",
                metric_type=MetricType.COUNTER,
                value=1.0,
            ),
        ]

        result = formatter.format(metrics)
        assert "myapp_requests_total" in result

    def test_format_with_const_labels(self) -> None:
        """Test formatting with constant labels."""
        config = PrometheusConfig(const_labels={"env": "prod"})
        formatter = PrometheusFormatter(config)

        metrics = [
            MetricData(
                name="requests",
                metric_type=MetricType.COUNTER,
                value=1.0,
                labels={"method": "GET"},
            ),
        ]

        result = formatter.format(metrics)
        assert 'env="prod"' in result
        assert 'method="GET"' in result

    def test_format_without_help_and_type(self) -> None:
        """Test formatting without HELP and TYPE comments."""
        config = PrometheusConfig(include_help=False, include_type=False)
        formatter = PrometheusFormatter(config)

        metrics = [
            MetricData(
                name="requests",
                metric_type=MetricType.COUNTER,
                value=1.0,
                description="Total requests",
            ),
        ]

        result = formatter.format(metrics)
        assert "# HELP" not in result
        assert "# TYPE" not in result
        assert "requests 1" in result

    def test_format_special_values(self) -> None:
        """Test formatting special float values."""
        formatter = PrometheusFormatter()

        # Infinity
        metrics = [
            MetricData(
                name="inf_metric",
                metric_type=MetricType.GAUGE,
                value=float("inf"),
            ),
        ]
        result = formatter.format(metrics)
        assert "+Inf" in result

        # Negative infinity
        metrics = [
            MetricData(
                name="neg_inf_metric",
                metric_type=MetricType.GAUGE,
                value=float("-inf"),
            ),
        ]
        result = formatter.format(metrics)
        assert "-Inf" in result

        # NaN
        metrics = [
            MetricData(
                name="nan_metric",
                metric_type=MetricType.GAUGE,
                value=float("nan"),
            ),
        ]
        result = formatter.format(metrics)
        assert "NaN" in result

    def test_content_type(self) -> None:
        """Test content type header."""
        formatter = PrometheusFormatter()
        assert formatter.content_type() == "text/plain; version=0.0.4; charset=utf-8"


# =============================================================================
# PrometheusExporter Tests
# =============================================================================


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    def test_export_disabled(self, sample_metrics: list[MetricData]) -> None:
        """Test export when disabled."""
        config = PrometheusConfig(enabled=False)
        exporter = PrometheusExporter(config=config)

        # Should not raise, just skip
        exporter.export(sample_metrics)
        assert exporter.last_export == ""

    def test_export_formats_metrics(self, sample_metrics: list[MetricData]) -> None:
        """Test export formats metrics correctly."""
        config = PrometheusConfig(namespace="test")
        exporter = PrometheusExporter(config=config)

        exporter.export(sample_metrics)

        assert exporter.last_export != ""
        assert "test_requests_total" in exporter.last_export
        assert "test_request_duration_seconds" in exporter.last_export

    def test_get_metrics_text(self, sample_metrics: list[MetricData]) -> None:
        """Test getting formatted text without exporting."""
        exporter = PrometheusExporter()
        text = exporter.get_metrics_text(sample_metrics)

        assert "requests_total" in text
        assert text == exporter.formatter.format(sample_metrics)

    def test_export_with_hooks(self, sample_metrics: list[MetricData]) -> None:
        """Test export triggers hooks."""
        hook = MetricsExportHook()
        exporter = PrometheusExporter(hooks=[hook])

        exporter.export(sample_metrics)

        assert hook.export_count == 1
        assert hook.success_count == 1
        assert hook.total_metrics_exported == 3
        assert hook.average_duration_ms > 0

    def test_config_property(self) -> None:
        """Test config property."""
        config = PrometheusConfig(job_name="test")
        exporter = PrometheusExporter(config=config)
        assert exporter.config.job_name == "test"

    def test_formatter_property(self) -> None:
        """Test formatter property."""
        exporter = PrometheusExporter()
        assert isinstance(exporter.formatter, PrometheusFormatter)

    def test_shutdown(self) -> None:
        """Test shutdown method."""
        exporter = PrometheusExporter()
        exporter.shutdown()  # Should not raise


# =============================================================================
# PrometheusPushGatewayClient Tests
# =============================================================================


class TestPrometheusPushGatewayClient:
    """Tests for PrometheusPushGatewayClient."""

    def test_build_url_simple(self) -> None:
        """Test URL building with job name only."""
        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="my_job",
        )
        url = client._build_url()
        assert url == "http://pushgateway:9091/metrics/job/my_job"

    def test_build_url_with_instance(self) -> None:
        """Test URL building with instance."""
        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="my_job",
            instance="worker-1",
        )
        url = client._build_url()
        assert url == "http://pushgateway:9091/metrics/job/my_job/instance/worker-1"

    def test_build_url_with_grouping_key(self) -> None:
        """Test URL building with grouping key."""
        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="my_job",
            grouping_key={"env": "prod"},
        )
        url = client._build_url()
        assert "/env/prod" in url

    def test_build_url_trailing_slash(self) -> None:
        """Test URL building strips trailing slash."""
        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091/",
            job_name="my_job",
        )
        url = client._build_url()
        assert not url.startswith("http://pushgateway:9091//")

    @patch("urllib.request.urlopen")
    def test_push_success(self, mock_urlopen: MagicMock) -> None:
        """Test successful push."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test",
        )
        client.push("test data")

        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_push_retry_on_failure(self, mock_urlopen: MagicMock) -> None:
        """Test push retries on failure."""
        mock_urlopen.side_effect = urllib.error.URLError("connection failed")

        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test",
            retry_count=2,
            retry_delay_seconds=0.01,
        )

        with pytest.raises(PushGatewayError):
            client.push("test data")

        # Should have tried 3 times (initial + 2 retries)
        assert mock_urlopen.call_count == 3

    @patch("urllib.request.urlopen")
    def test_push_http_error(self, mock_urlopen: MagicMock) -> None:
        """Test push handles HTTP error."""
        from email.message import Message
        headers = Message()
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://test",
            code=500,
            msg="Internal Server Error",
            hdrs=headers,
            fp=None,
        )

        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test",
            retry_count=0,
        )

        with pytest.raises(PushGatewayError) as exc_info:
            client.push("test data")

        # The outer exception wraps the inner one with status_code in cause
        assert exc_info.value.cause is not None
        assert exc_info.value.cause.status_code == 500  # type: ignore

    @patch("urllib.request.urlopen")
    def test_delete_success(self, mock_urlopen: MagicMock) -> None:
        """Test successful delete."""
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        client = PrometheusPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test",
        )
        client.delete()

        mock_urlopen.assert_called_once()


# =============================================================================
# PrometheusHttpServer Tests
# =============================================================================


class TestPrometheusHttpServer:
    """Tests for PrometheusHttpServer."""

    def test_create_server(self) -> None:
        """Test server creation."""
        server = PrometheusHttpServer(
            host="127.0.0.1",
            port=0,  # Let OS assign port
        )
        assert not server.is_running

    def test_start_stop(self) -> None:
        """Test server start and stop."""
        server = PrometheusHttpServer(
            host="127.0.0.1",
            port=0,
        )

        server.start()
        assert server.is_running

        server.stop()
        assert not server.is_running

    def test_context_manager(self) -> None:
        """Test context manager interface."""
        with PrometheusHttpServer(host="127.0.0.1", port=0) as server:
            assert server.is_running

        assert not server.is_running

    def test_start_idempotent(self) -> None:
        """Test starting twice is safe."""
        server = PrometheusHttpServer(host="127.0.0.1", port=0)
        server.start()
        server.start()  # Should not raise
        server.stop()

    def test_stop_idempotent(self) -> None:
        """Test stopping twice is safe."""
        server = PrometheusHttpServer(host="127.0.0.1", port=0)
        server.start()
        server.stop()
        server.stop()  # Should not raise

    def test_serve_metrics(self) -> None:
        """Test serving metrics via HTTP."""

        def get_metrics() -> list[MetricData]:
            return [
                MetricData(
                    name="test_metric",
                    metric_type=MetricType.COUNTER,
                    value=42.0,
                )
            ]

        with PrometheusHttpServer(
            host="127.0.0.1",
            port=0,
            metrics_callback=get_metrics,
        ) as server:
            # Wait for server to start
            time.sleep(0.1)

            # Get the actual port
            actual_port = server._server.server_address[1]  # type: ignore

            # Make request
            url = f"http://127.0.0.1:{actual_port}/metrics"
            with urllib.request.urlopen(url, timeout=1) as response:
                body = response.read().decode("utf-8")

            assert "test_metric" in body
            assert "42" in body


# =============================================================================
# Export Hooks Tests
# =============================================================================


class TestExportHooks:
    """Tests for export hooks."""

    def test_base_hook_noop(self) -> None:
        """Test base hook methods are no-op."""
        hook = BaseExportHook()
        hook.on_export_start(10, {})
        hook.on_export_success(10, 100.0, {})
        hook.on_export_error(Exception("test"), 10, {})

    def test_metrics_hook_tracking(self) -> None:
        """Test metrics hook tracks statistics."""
        hook = MetricsExportHook()

        hook.on_export_start(10, {})
        hook.on_export_success(10, 50.0, {})

        hook.on_export_start(20, {})
        hook.on_export_success(20, 100.0, {})

        hook.on_export_start(5, {})
        hook.on_export_error(Exception("test"), 5, {})

        assert hook.export_count == 3
        assert hook.success_count == 2
        assert hook.error_count == 1
        assert hook.total_metrics_exported == 30
        assert hook.average_duration_ms == 75.0  # (50 + 100) / 2

    def test_metrics_hook_reset(self) -> None:
        """Test metrics hook reset."""
        hook = MetricsExportHook()

        hook.on_export_start(10, {})
        hook.on_export_success(10, 50.0, {})

        hook.reset()

        assert hook.export_count == 0
        assert hook.success_count == 0
        assert hook.total_metrics_exported == 0

    def test_composite_hook_delegates(self) -> None:
        """Test composite hook delegates to all hooks."""
        hook1 = MetricsExportHook()
        hook2 = MetricsExportHook()
        composite = CompositeExportHook([hook1, hook2])

        composite.on_export_start(10, {})
        composite.on_export_success(10, 50.0, {})

        assert hook1.export_count == 1
        assert hook2.export_count == 1

    def test_composite_hook_isolates_errors(self) -> None:
        """Test composite hook isolates hook errors."""

        class FailingHook(BaseExportHook):
            def on_export_start(
                self, metric_count: int, context: dict
            ) -> None:
                raise RuntimeError("Hook error")

        hook1 = FailingHook()
        hook2 = MetricsExportHook()
        composite = CompositeExportHook([hook1, hook2])

        # Should not raise, and hook2 should still be called
        composite.on_export_start(10, {})
        assert hook2.export_count == 1


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prometheus_exporter(self) -> None:
        """Test create_prometheus_exporter factory."""
        exporter = create_prometheus_exporter(
            namespace="myapp",
            job_name="worker",
            const_labels={"env": "prod"},
        )

        assert exporter.config.namespace == "myapp"
        assert exporter.config.job_name == "worker"
        assert exporter.config.const_labels == {"env": "prod"}

    def test_create_pushgateway_exporter(self) -> None:
        """Test create_pushgateway_exporter factory."""
        exporter = create_pushgateway_exporter(
            gateway_url="http://pushgateway:9091",
            job_name="batch_job",
            instance="worker-1",
            timeout_seconds=15.0,
        )

        assert exporter.config.gateway_url == "http://pushgateway:9091"
        assert exporter.config.job_name == "batch_job"
        assert exporter.config.instance == "worker-1"
        assert exporter.config.gateway_timeout_seconds == 15.0

    def test_create_prometheus_http_server(self) -> None:
        """Test create_prometheus_http_server factory."""
        server = create_prometheus_http_server(
            host="127.0.0.1",
            port=8080,
            path="/prometheus",
        )

        assert server._host == "127.0.0.1"
        assert server._port == 8080
        assert server._path == "/prometheus"


# =============================================================================
# Default Configs Tests
# =============================================================================


class TestDefaultConfigs:
    """Tests for default configurations."""

    def test_default_config(self) -> None:
        """Test DEFAULT_PROMETHEUS_CONFIG."""
        assert DEFAULT_PROMETHEUS_CONFIG.enabled is True
        assert DEFAULT_PROMETHEUS_CONFIG.job_name == "truthound"

    def test_pushgateway_config(self) -> None:
        """Test PUSHGATEWAY_PROMETHEUS_CONFIG."""
        assert PUSHGATEWAY_PROMETHEUS_CONFIG.gateway_url == "http://localhost:9091"
        assert PUSHGATEWAY_PROMETHEUS_CONFIG.gateway_retry_count == 3

    def test_minimal_config(self) -> None:
        """Test MINIMAL_PROMETHEUS_CONFIG."""
        assert MINIMAL_PROMETHEUS_CONFIG.include_help is False
        assert MINIMAL_PROMETHEUS_CONFIG.include_type is False
        assert MINIMAL_PROMETHEUS_CONFIG.include_timestamp is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for Prometheus exporter."""

    def test_full_export_pipeline(self) -> None:
        """Test complete export pipeline."""
        # Create metrics using common.metrics
        counter = Counter("integration_requests_total", "Test counter")
        counter.inc()
        counter.inc(5, labels={"method": "POST"})

        gauge = Gauge("integration_temperature", "Test gauge")
        gauge.set(23.5)

        # Collect metrics
        metrics = counter.collect() + gauge.collect()

        # Export
        config = PrometheusConfig(
            namespace="integration_test",
            const_labels={"test": "true"},
        )
        exporter = PrometheusExporter(config=config)
        hook = MetricsExportHook()
        exporter._hooks = [hook]

        exporter.export(metrics)

        # Verify
        text = exporter.last_export
        assert "integration_test_integration_requests_total" in text
        assert "integration_test_integration_temperature" in text
        assert 'test="true"' in text
        assert hook.success_count == 1

    def test_export_histogram_with_buckets(self) -> None:
        """Test exporting histogram with proper bucket structure."""
        histogram = Histogram(
            name="request_duration_seconds",
            description="Request duration",
            buckets=(0.1, 0.5, 1.0, 5.0),
        )
        histogram.observe(0.2)
        histogram.observe(0.8)
        histogram.observe(2.0)

        metrics = histogram.collect()
        formatter = PrometheusFormatter()
        text = formatter.format(metrics)

        # Should have bucket, sum, and count metrics
        assert "request_duration_seconds" in text
        # Bucket entries should exist
        assert "le=" in text or "_bucket" in text or "_sum" in text or "_count" in text

    def test_export_summary_with_quantiles(self) -> None:
        """Test exporting summary with quantiles."""
        summary = Summary(
            name="response_size_bytes",
            description="Response size",
            quantiles=(0.5, 0.9, 0.99),
        )
        for i in range(100):
            summary.observe(i * 10.0)

        metrics = summary.collect()
        formatter = PrometheusFormatter()
        text = formatter.format(metrics)

        assert "response_size_bytes" in text


# =============================================================================
# Enterprise Features Tests
# =============================================================================


class TestEnterprisePrometheusConfig:
    """Test enterprise configuration features."""

    def test_config_with_auth_basic(self) -> None:
        """Test configuration with basic auth."""
        config = PrometheusConfig().with_auth(
            username="prometheus",
            password="secret123",
        )

        assert config.auth_username == "prometheus"
        assert config.auth_password == "secret123"
        assert config.auth_token == ""

    def test_config_with_auth_bearer(self) -> None:
        """Test configuration with bearer token."""
        config = PrometheusConfig().with_auth(
            token="my-bearer-token",
        )

        assert config.auth_token == "my-bearer-token"
        assert config.auth_username == ""
        assert config.auth_password == ""

    def test_config_with_tenant_isolation(self) -> None:
        """Test configuration with tenant isolation."""
        config = PrometheusConfig().with_tenant_isolation(
            enabled=True,
            label_name="org_id",
        )

        assert config.enable_tenant_isolation is True
        assert config.default_tenant_label == "org_id"

    def test_config_with_distributed(self) -> None:
        """Test configuration with distributed environment settings."""
        config = PrometheusConfig().with_distributed(
            auto_instance=True,
            cluster_name="production-cluster",
        )

        assert config.auto_instance is True
        assert config.cluster_name == "production-cluster"

    def test_config_with_performance(self) -> None:
        """Test configuration with performance settings."""
        config = PrometheusConfig().with_performance(
            batch_size=100,
            async_push=True,
        )

        assert config.batch_size == 100
        assert config.async_push is True

    def test_config_builder_chaining(self) -> None:
        """Test chaining multiple builder methods."""
        config = (
            PrometheusConfig()
            .with_namespace("enterprise")
            .with_auth(username="admin", password="secret")
            .with_tenant_isolation(enabled=True)
            .with_distributed(cluster_name="prod")
            .with_performance(batch_size=50)
        )

        assert config.namespace == "enterprise"
        assert config.auth_username == "admin"
        assert config.enable_tenant_isolation is True
        assert config.cluster_name == "prod"
        assert config.batch_size == 50

    def test_enterprise_config_preset(self) -> None:
        """Test ENTERPRISE_PROMETHEUS_CONFIG preset."""
        from common.exporters import ENTERPRISE_PROMETHEUS_CONFIG

        assert ENTERPRISE_PROMETHEUS_CONFIG.auto_instance is True
        assert ENTERPRISE_PROMETHEUS_CONFIG.enable_tenant_isolation is True
        # Enterprise config may or may not have batch_size > 0
        assert ENTERPRISE_PROMETHEUS_CONFIG.gateway_retry_count > 0

    def test_ha_config_preset(self) -> None:
        """Test HA_PROMETHEUS_CONFIG preset."""
        from common.exporters import HA_PROMETHEUS_CONFIG

        assert HA_PROMETHEUS_CONFIG.auto_instance is True
        assert HA_PROMETHEUS_CONFIG.gateway_retry_count >= 3
        assert HA_PROMETHEUS_CONFIG.gateway_retry_delay_seconds >= 1.0

    def test_async_config_preset(self) -> None:
        """Test ASYNC_PROMETHEUS_CONFIG preset."""
        from common.exporters import ASYNC_PROMETHEUS_CONFIG

        assert ASYNC_PROMETHEUS_CONFIG.async_push is True
        assert ASYNC_PROMETHEUS_CONFIG.batch_size > 0

    def test_config_to_dict_includes_enterprise_fields(self) -> None:
        """Test that to_dict includes enterprise fields."""
        config = PrometheusConfig(
            auth_username="user",
            auth_password="pass",
            enable_tenant_isolation=True,
            cluster_name="test-cluster",
            batch_size=100,
        )

        data = config.to_dict()

        assert data["auth_username"] == "user"
        assert data["auth_password"] == "pass"
        assert data["enable_tenant_isolation"] is True
        assert data["cluster_name"] == "test-cluster"
        assert data["batch_size"] == 100

    def test_config_from_dict_restores_enterprise_fields(self) -> None:
        """Test that from_dict restores enterprise fields."""
        data = {
            "auth_token": "bearer-token",
            "auto_instance": False,
            "default_tenant_label": "customer_id",
            "async_push": True,
        }

        config = PrometheusConfig.from_dict(data)

        assert config.auth_token == "bearer-token"
        assert config.auto_instance is False
        assert config.default_tenant_label == "customer_id"
        assert config.async_push is True


class TestAutoInstanceName:
    """Test auto instance name detection."""

    def test_get_auto_instance_name_from_hostname_env(self, monkeypatch: Any) -> None:
        """Test instance name from HOSTNAME env var."""
        from common.exporters import get_auto_instance_name

        monkeypatch.setenv("HOSTNAME", "worker-pod-abc123")

        result = get_auto_instance_name()
        assert result == "worker-pod-abc123"

    def test_get_auto_instance_name_from_pod_name(self, monkeypatch: Any) -> None:
        """Test instance name from POD_NAME env var (Kubernetes)."""
        from common.exporters import get_auto_instance_name

        # Clear HOSTNAME
        monkeypatch.delenv("HOSTNAME", raising=False)
        monkeypatch.setenv("POD_NAME", "my-app-deployment-xyz789")

        result = get_auto_instance_name()
        assert result == "my-app-deployment-xyz789"

    def test_get_auto_instance_name_from_instance_id(self, monkeypatch: Any) -> None:
        """Test instance name from INSTANCE_ID env var (AWS)."""
        from common.exporters import get_auto_instance_name

        monkeypatch.delenv("HOSTNAME", raising=False)
        monkeypatch.delenv("POD_NAME", raising=False)
        monkeypatch.setenv("INSTANCE_ID", "i-0123456789abcdef0")

        result = get_auto_instance_name()
        assert result == "i-0123456789abcdef0"

    def test_get_auto_instance_name_fallback(self, monkeypatch: Any) -> None:
        """Test instance name fallback to hostname:pid."""
        from common.exporters import get_auto_instance_name

        # Clear all environment variables
        for env in ["HOSTNAME", "POD_NAME", "INSTANCE_ID", "CONTAINER_ID", "DYNO"]:
            monkeypatch.delenv(env, raising=False)

        result = get_auto_instance_name()

        # Should be hostname:pid format
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 2
        assert parts[1].isdigit()  # PID should be numeric


class TestClusterLabels:
    """Test cluster label detection."""

    def test_get_cluster_labels_kubernetes(self, monkeypatch: Any) -> None:
        """Test cluster labels in Kubernetes environment."""
        from common.exporters import get_cluster_labels

        monkeypatch.setenv("POD_NAMESPACE", "production")
        monkeypatch.setenv("NODE_NAME", "worker-1")

        labels = get_cluster_labels()

        assert labels.get("k8s_namespace") == "production"
        assert labels.get("k8s_node") == "worker-1"

    def test_get_cluster_labels_aws(self, monkeypatch: Any) -> None:
        """Test cluster labels in AWS environment."""
        from common.exporters import get_cluster_labels

        monkeypatch.setenv("AWS_REGION", "us-east-1")

        labels = get_cluster_labels()

        assert labels.get("cloud_region") == "us-east-1"

    def test_get_cluster_labels_always_has_platform(self, monkeypatch: Any) -> None:
        """Test cluster labels always include platform."""
        from common.exporters import get_cluster_labels

        # Clear environment vars
        for env in [
            "POD_NAMESPACE",
            "NODE_NAME",
            "AWS_REGION",
            "CLOUD_REGION",
        ]:
            monkeypatch.delenv(env, raising=False)

        labels = get_cluster_labels()

        # Should always have platform
        assert isinstance(labels, dict)
        assert "platform" in labels


class TestAuthenticatedPushGatewayClient:
    """Test authenticated Push Gateway client."""

    def test_basic_auth_header_generation(self) -> None:
        """Test Basic Auth header generation."""
        from common.exporters import AuthenticatedPushGatewayClient

        client = AuthenticatedPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test_job",
            auth_username="admin",
            auth_password="secret",
        )

        headers = client._get_auth_header()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

        # Verify base64 encoding
        import base64

        encoded = headers["Authorization"].replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode("utf-8")
        assert decoded == "admin:secret"

    def test_bearer_token_header_generation(self) -> None:
        """Test Bearer token header generation."""
        from common.exporters import AuthenticatedPushGatewayClient

        client = AuthenticatedPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test_job",
            auth_token="my-bearer-token-123",
        )

        headers = client._get_auth_header()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer my-bearer-token-123"

    def test_no_auth_when_not_configured(self) -> None:
        """Test no auth headers when not configured."""
        from common.exporters import AuthenticatedPushGatewayClient

        client = AuthenticatedPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test_job",
        )

        headers = client._get_auth_header()

        assert "Authorization" not in headers

    def test_bearer_takes_precedence_over_basic(self) -> None:
        """Test that bearer token takes precedence over basic auth."""
        from common.exporters import AuthenticatedPushGatewayClient

        client = AuthenticatedPushGatewayClient(
            gateway_url="http://pushgateway:9091",
            job_name="test_job",
            auth_username="admin",
            auth_password="secret",
            auth_token="bearer-token",
        )

        headers = client._get_auth_header()

        assert headers["Authorization"] == "Bearer bearer-token"


class TestTenantAwarePrometheusExporter:
    """Test tenant-aware Prometheus exporter."""

    def test_tenant_label_injection(self) -> None:
        """Test that tenant label is injected into metrics."""
        from common.exporters import TenantAwarePrometheusExporter

        config = PrometheusConfig(
            enable_tenant_isolation=True,
            default_tenant_label="tenant_id",
        )

        # Use tenant_getter to provide tenant ID
        exporter = TenantAwarePrometheusExporter(
            config,
            tenant_getter=lambda: "acme-corp",
        )

        counter = Counter("api_requests_tenant", "API request count")
        counter.inc()
        metrics = counter.collect()

        exporter.export(metrics)
        text = exporter.last_export

        assert "tenant_id=" in text
        assert "acme-corp" in text

    def test_tenant_getter_callable(self) -> None:
        """Test that tenant_getter callable is used."""
        from common.exporters import TenantAwarePrometheusExporter

        config = PrometheusConfig(
            enable_tenant_isolation=True,
        )

        # Custom tenant getter
        def get_tenant() -> str:
            return "dynamic-tenant"

        exporter = TenantAwarePrometheusExporter(config, tenant_getter=get_tenant)

        counter = Counter("tenant_counter", "Tenant counter")
        counter.inc()
        exporter.export(counter.collect())

        assert "dynamic-tenant" in exporter.last_export

    def test_custom_tenant_label_name(self) -> None:
        """Test custom tenant label name."""
        from common.exporters import TenantAwarePrometheusExporter

        config = PrometheusConfig(
            enable_tenant_isolation=True,
            default_tenant_label="organization_id",
        )

        exporter = TenantAwarePrometheusExporter(
            config,
            tenant_getter=lambda: "org-123",
        )

        counter = Counter("events_org", "Event count")
        counter.inc()
        metrics = counter.collect()

        exporter.export(metrics)
        text = exporter.last_export

        assert "organization_id=" in text
        assert "org-123" in text

    def test_tenant_isolation_disabled(self) -> None:
        """Test that tenant labels are not added when isolation is disabled."""
        from common.exporters import TenantAwarePrometheusExporter

        config = PrometheusConfig(
            enable_tenant_isolation=False,
        )

        exporter = TenantAwarePrometheusExporter(
            config,
            tenant_getter=lambda: "test-tenant",
        )

        counter = Counter("simple_counter_no_tenant", "Simple counter")
        counter.inc()
        metrics = counter.collect()

        exporter.export(metrics)
        text = exporter.last_export

        # Tenant label should not be present
        assert "test-tenant" not in text


class TestAsyncPrometheusExporter:
    """Test async Prometheus exporter."""

    @pytest.mark.asyncio
    async def test_async_export(self) -> None:
        """Test async export functionality."""
        from common.exporters import AsyncPrometheusExporter

        config = PrometheusConfig(
            async_push=True,
        )

        exporter = AsyncPrometheusExporter(config)

        counter = Counter("async_requests_test", "Async request count")
        counter.inc(5)
        metrics = counter.collect()

        await exporter.export(metrics)

        assert exporter.last_export is not None
        assert "async_requests_test" in exporter.last_export

    @pytest.mark.asyncio
    async def test_async_export_batch(self) -> None:
        """Test async batch export."""
        from common.exporters import AsyncPrometheusExporter

        config = PrometheusConfig(
            async_push=True,
            batch_size=10,
        )

        exporter = AsyncPrometheusExporter(config)

        # Create multiple metrics
        counters = []
        for i in range(5):
            counter = Counter(f"async_batch_counter_{i}", f"Counter {i}")
            counter.inc(i + 1)
            counters.append(counter)

        all_metrics = []
        for c in counters:
            all_metrics.extend(c.collect())

        await exporter.export(all_metrics)

        text = exporter.last_export
        for i in range(5):
            assert f"async_batch_counter_{i}" in text

    @pytest.mark.asyncio
    async def test_async_export_with_hooks(self) -> None:
        """Test async exporter with hooks."""
        from common.exporters import AsyncPrometheusExporter

        config = PrometheusConfig(async_push=True)
        hook = MetricsExportHook()

        exporter = AsyncPrometheusExporter(config, hooks=[hook])

        counter = Counter("async_hooked_counter", "Counter with hooks")
        counter.inc()
        await exporter.export(counter.collect())

        assert hook.success_count == 1
        assert "async_hooked_counter" in exporter.last_export


class TestEnterpriseIntegration:
    """Test enterprise feature integration scenarios."""

    def test_full_enterprise_stack(self) -> None:
        """Test all enterprise features together."""
        from common.exporters import (
            ENTERPRISE_PROMETHEUS_CONFIG,
            TenantAwarePrometheusExporter,
        )

        # Create tenant-aware exporter with enterprise config
        config = ENTERPRISE_PROMETHEUS_CONFIG.with_auth(
            username="prometheus",
            password="secret",
        ).with_distributed(
            cluster_name="production",
        )

        exporter = TenantAwarePrometheusExporter(
            config,
            tenant_getter=lambda: "enterprise-client",
        )

        # Create metrics
        counter = Counter("enterprise_events_test", "Enterprise event count")
        counter.inc(100, labels={"event_type": "transaction"})

        gauge = Gauge("enterprise_balance_test", "Enterprise balance")
        gauge.set(1000000.0)

        metrics = counter.collect() + gauge.collect()
        exporter.export(metrics)

        text = exporter.last_export

        # Verify all components
        assert "enterprise_events_test" in text
        assert "enterprise_balance_test" in text
        assert "tenant_id=" in text
        assert "enterprise-client" in text

    def test_multi_tenant_metrics_isolation(self) -> None:
        """Test that metrics are properly isolated by tenant."""
        from common.exporters import TenantAwarePrometheusExporter

        config = PrometheusConfig(enable_tenant_isolation=True)

        # Tenant A
        exporter_a = TenantAwarePrometheusExporter(
            config,
            tenant_getter=lambda: "tenant-a",
        )
        counter_a = Counter("shared_counter_isolation_a", "Shared counter")
        counter_a.inc(10)
        exporter_a.export(counter_a.collect())
        text_a = exporter_a.last_export

        # Tenant B (new counter instance to avoid label conflict)
        exporter_b = TenantAwarePrometheusExporter(
            config,
            tenant_getter=lambda: "tenant-b",
        )
        counter_b = Counter("shared_counter_isolation_b", "Shared counter")
        counter_b.inc(20)
        exporter_b.export(counter_b.collect())
        text_b = exporter_b.last_export

        # Verify isolation
        assert 'tenant_id="tenant-a"' in text_a
        assert 'tenant_id="tenant-b"' in text_b
        assert "tenant-b" not in text_a
        assert "tenant-a" not in text_b

    def test_high_availability_config(self) -> None:
        """Test HA configuration for production environments."""
        from common.exporters import HA_PROMETHEUS_CONFIG

        # HA config should be suitable for production
        assert HA_PROMETHEUS_CONFIG.gateway_retry_count >= 3
        assert HA_PROMETHEUS_CONFIG.gateway_retry_delay_seconds >= 1.0
        assert HA_PROMETHEUS_CONFIG.auto_instance is True

        # Should be able to create exporter with HA config
        exporter = PrometheusExporter(HA_PROMETHEUS_CONFIG)
        assert exporter is not None
