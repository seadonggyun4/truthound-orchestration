"""Tests for OpenTelemetry configuration."""

import pytest

from common.opentelemetry.config import (
    DEFAULT_OTEL_CONFIG,
    DEVELOPMENT_OTEL_CONFIG,
    DISABLED_OTEL_CONFIG,
    PRODUCTION_OTEL_CONFIG,
    TESTING_OTEL_CONFIG,
    BatchConfig,
    OTelConfig,
    OTLPExporterConfig,
    ResourceConfig,
    SamplingConfig,
)
from common.opentelemetry.types import ExporterType, OTLPCompression, OTLPProtocol


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ResourceConfig()
        assert config.service_name == "truthound-orchestration"
        assert config.service_version == "0.1.0"
        assert config.service_namespace == "data-quality"
        assert config.deployment_environment == "development"
        assert config.service_instance_id is None
        assert config.additional_attributes == {}

    def test_with_service_name(self):
        """Test updating service name."""
        config = ResourceConfig()
        new_config = config.with_service_name("my-service")
        assert new_config.service_name == "my-service"
        assert config.service_name == "truthound-orchestration"  # Immutable

    def test_with_service_version(self):
        """Test updating service version."""
        config = ResourceConfig()
        new_config = config.with_service_version("2.0.0")
        assert new_config.service_version == "2.0.0"

    def test_with_environment(self):
        """Test updating deployment environment."""
        config = ResourceConfig()
        new_config = config.with_environment("production")
        assert new_config.deployment_environment == "production"

    def test_with_instance_id(self):
        """Test updating instance ID."""
        config = ResourceConfig()
        new_config = config.with_instance_id("instance-001")
        assert new_config.service_instance_id == "instance-001"

    def test_with_attributes(self):
        """Test adding custom attributes."""
        config = ResourceConfig()
        new_config = config.with_attributes(custom_key="custom_value", another="value")
        assert new_config.additional_attributes["custom_key"] == "custom_value"
        assert new_config.additional_attributes["another"] == "value"

    def test_to_attributes(self):
        """Test conversion to attribute dictionary."""
        config = ResourceConfig(
            service_name="test-service",
            service_version="1.0.0",
            service_namespace="test-ns",
            deployment_environment="testing",
            service_instance_id="test-001",
            additional_attributes={"custom": "value"},
        )
        attrs = config.to_attributes()

        assert attrs["service.name"] == "test-service"
        assert attrs["service.version"] == "1.0.0"
        assert attrs["service.namespace"] == "test-ns"
        assert attrs["deployment.environment"] == "testing"
        assert attrs["service.instance.id"] == "test-001"
        assert attrs["custom"] == "value"


class TestOTLPExporterConfig:
    """Tests for OTLPExporterConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OTLPExporterConfig()
        assert config.endpoint == "http://localhost:4317"
        assert config.protocol == OTLPProtocol.GRPC
        assert config.compression == OTLPCompression.GZIP
        assert config.headers == {}
        assert config.timeout_seconds == 10.0
        assert config.insecure is False
        assert config.certificate_file is None

    def test_with_endpoint(self):
        """Test updating endpoint."""
        config = OTLPExporterConfig()
        new_config = config.with_endpoint("http://collector:4317")
        assert new_config.endpoint == "http://collector:4317"

    def test_with_protocol(self):
        """Test updating protocol."""
        config = OTLPExporterConfig()
        new_config = config.with_protocol(OTLPProtocol.HTTP_PROTOBUF)
        assert new_config.protocol == OTLPProtocol.HTTP_PROTOBUF

    def test_with_compression(self):
        """Test updating compression."""
        config = OTLPExporterConfig()
        new_config = config.with_compression(OTLPCompression.NONE)
        assert new_config.compression == OTLPCompression.NONE

    def test_with_headers(self):
        """Test adding headers."""
        config = OTLPExporterConfig()
        new_config = config.with_headers(Authorization="Bearer token")
        assert new_config.headers["Authorization"] == "Bearer token"

    def test_with_timeout(self):
        """Test updating timeout."""
        config = OTLPExporterConfig()
        new_config = config.with_timeout(30.0)
        assert new_config.timeout_seconds == 30.0

    def test_with_tls(self):
        """Test updating TLS settings."""
        config = OTLPExporterConfig()
        new_config = config.with_tls(insecure=True, certificate_file="/path/to/cert")
        assert new_config.insecure is True
        assert new_config.certificate_file == "/path/to/cert"


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SamplingConfig()
        assert config.enabled is True
        assert config.sample_rate == 1.0
        assert config.parent_based is True
        assert config.error_sample_rate == 1.0
        assert config.slow_operation_threshold_ms == 1000.0
        assert config.slow_operation_sample_rate == 1.0

    def test_with_sample_rate(self):
        """Test updating sample rate."""
        config = SamplingConfig()
        new_config = config.with_sample_rate(0.5)
        assert new_config.sample_rate == 0.5

    def test_with_sample_rate_invalid(self):
        """Test validation of sample rate."""
        config = SamplingConfig()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            config.with_sample_rate(1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            config.with_sample_rate(-0.1)

    def test_with_error_sampling(self):
        """Test updating error sampling rate."""
        config = SamplingConfig()
        new_config = config.with_error_sampling(0.8)
        assert new_config.error_sample_rate == 0.8

    def test_with_slow_operation_sampling(self):
        """Test updating slow operation sampling."""
        config = SamplingConfig()
        new_config = config.with_slow_operation_sampling(500.0, 0.9)
        assert new_config.slow_operation_threshold_ms == 500.0
        assert new_config.slow_operation_sample_rate == 0.9


class TestBatchConfig:
    """Tests for BatchConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BatchConfig()
        assert config.max_queue_size == 2048
        assert config.max_export_batch_size == 512
        assert config.export_timeout_seconds == 30.0
        assert config.schedule_delay_seconds == 5.0

    def test_with_queue_size(self):
        """Test updating queue size."""
        config = BatchConfig()
        new_config = config.with_queue_size(4096)
        assert new_config.max_queue_size == 4096

    def test_with_batch_size(self):
        """Test updating batch size."""
        config = BatchConfig()
        new_config = config.with_batch_size(1024)
        assert new_config.max_export_batch_size == 1024

    def test_with_export_timeout(self):
        """Test updating export timeout."""
        config = BatchConfig()
        new_config = config.with_export_timeout(60.0)
        assert new_config.export_timeout_seconds == 60.0

    def test_with_schedule_delay(self):
        """Test updating schedule delay."""
        config = BatchConfig()
        new_config = config.with_schedule_delay(10.0)
        assert new_config.schedule_delay_seconds == 10.0


class TestOTelConfig:
    """Tests for OTelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OTelConfig()
        assert config.enabled is True
        assert config.metrics_enabled is True
        assert config.tracing_enabled is True
        assert config.bridge_enabled is True
        assert config.metrics_exporter == ExporterType.OTLP
        assert config.traces_exporter == ExporterType.OTLP
        assert config.propagators == ("tracecontext", "baggage")

    def test_with_enabled(self):
        """Test updating enabled status."""
        config = OTelConfig()
        new_config = config.with_enabled(False)
        assert new_config.enabled is False

    def test_with_metrics_enabled(self):
        """Test updating metrics enabled status."""
        config = OTelConfig()
        new_config = config.with_metrics_enabled(False)
        assert new_config.metrics_enabled is False

    def test_with_tracing_enabled(self):
        """Test updating tracing enabled status."""
        config = OTelConfig()
        new_config = config.with_tracing_enabled(False)
        assert new_config.tracing_enabled is False

    def test_with_bridge_enabled(self):
        """Test updating bridge enabled status."""
        config = OTelConfig()
        new_config = config.with_bridge_enabled(False)
        assert new_config.bridge_enabled is False

    def test_with_exporters(self):
        """Test updating exporter types."""
        config = OTelConfig()
        new_config = config.with_exporters(
            metrics=ExporterType.CONSOLE,
            traces=ExporterType.MEMORY,
        )
        assert new_config.metrics_exporter == ExporterType.CONSOLE
        assert new_config.traces_exporter == ExporterType.MEMORY

    def test_with_resource(self):
        """Test updating resource configuration."""
        config = OTelConfig()
        resource = ResourceConfig(service_name="custom-service")
        new_config = config.with_resource(resource)
        assert new_config.resource.service_name == "custom-service"

    def test_with_service_name(self):
        """Test convenience method for updating service name."""
        config = OTelConfig()
        new_config = config.with_service_name("my-service")
        assert new_config.resource.service_name == "my-service"

    def test_with_endpoint(self):
        """Test convenience method for updating endpoint."""
        config = OTelConfig()
        new_config = config.with_endpoint("http://collector:4317")
        assert new_config.otlp.endpoint == "http://collector:4317"

    def test_with_sample_rate(self):
        """Test convenience method for updating sample rate."""
        config = OTelConfig()
        new_config = config.with_sample_rate(0.5)
        assert new_config.sampling.sample_rate == 0.5

    def test_with_propagators(self):
        """Test updating propagators."""
        config = OTelConfig()
        new_config = config.with_propagators("tracecontext", "b3")
        assert new_config.propagators == ("tracecontext", "b3")

    def test_chaining(self):
        """Test chaining builder methods."""
        config = (
            OTelConfig()
            .with_enabled(True)
            .with_service_name("my-service")
            .with_endpoint("http://collector:4317")
            .with_sample_rate(0.5)
            .with_exporters(metrics=ExporterType.CONSOLE)
        )
        assert config.enabled is True
        assert config.resource.service_name == "my-service"
        assert config.otlp.endpoint == "http://collector:4317"
        assert config.sampling.sample_rate == 0.5
        assert config.metrics_exporter == ExporterType.CONSOLE


class TestPresetConfigurations:
    """Tests for preset configurations."""

    def test_default_config(self):
        """Test default configuration preset."""
        assert DEFAULT_OTEL_CONFIG.enabled is True
        assert DEFAULT_OTEL_CONFIG.metrics_enabled is True
        assert DEFAULT_OTEL_CONFIG.tracing_enabled is True

    def test_development_config(self):
        """Test development configuration preset."""
        assert DEVELOPMENT_OTEL_CONFIG.enabled is True
        assert DEVELOPMENT_OTEL_CONFIG.metrics_exporter == ExporterType.CONSOLE
        assert DEVELOPMENT_OTEL_CONFIG.traces_exporter == ExporterType.CONSOLE
        assert DEVELOPMENT_OTEL_CONFIG.resource.deployment_environment == "development"
        assert DEVELOPMENT_OTEL_CONFIG.sampling.sample_rate == 1.0

    def test_production_config(self):
        """Test production configuration preset."""
        assert PRODUCTION_OTEL_CONFIG.enabled is True
        assert PRODUCTION_OTEL_CONFIG.metrics_exporter == ExporterType.OTLP
        assert PRODUCTION_OTEL_CONFIG.traces_exporter == ExporterType.OTLP
        assert PRODUCTION_OTEL_CONFIG.resource.deployment_environment == "production"
        assert PRODUCTION_OTEL_CONFIG.sampling.sample_rate == 0.1  # 10% sampling
        assert PRODUCTION_OTEL_CONFIG.sampling.error_sample_rate == 1.0

    def test_testing_config(self):
        """Test testing configuration preset."""
        assert TESTING_OTEL_CONFIG.enabled is True
        assert TESTING_OTEL_CONFIG.metrics_exporter == ExporterType.MEMORY
        assert TESTING_OTEL_CONFIG.traces_exporter == ExporterType.MEMORY
        assert TESTING_OTEL_CONFIG.resource.deployment_environment == "testing"

    def test_disabled_config(self):
        """Test disabled configuration preset."""
        assert DISABLED_OTEL_CONFIG.enabled is False
        assert DISABLED_OTEL_CONFIG.metrics_enabled is False
        assert DISABLED_OTEL_CONFIG.tracing_enabled is False
        assert DISABLED_OTEL_CONFIG.bridge_enabled is False
        assert DISABLED_OTEL_CONFIG.metrics_exporter == ExporterType.NONE
        assert DISABLED_OTEL_CONFIG.traces_exporter == ExporterType.NONE
