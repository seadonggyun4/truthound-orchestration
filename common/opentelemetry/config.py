"""OpenTelemetry configuration for truthound-orchestration.

This module provides immutable configuration classes for OpenTelemetry
integration with builder pattern support for fluent configuration.
"""

from dataclasses import dataclass, field
from typing import Any

from common.opentelemetry.types import (
    ExporterType,
    OTLPCompression,
    OTLPProtocol,
    ResourceAttributes,
)

__all__ = [
    # Configuration classes
    "OTelConfig",
    "OTLPExporterConfig",
    "ResourceConfig",
    "SamplingConfig",
    "BatchConfig",
    # Preset configurations
    "DEFAULT_OTEL_CONFIG",
    "DEVELOPMENT_OTEL_CONFIG",
    "PRODUCTION_OTEL_CONFIG",
    "TESTING_OTEL_CONFIG",
    "DISABLED_OTEL_CONFIG",
]


@dataclass(frozen=True)
class ResourceConfig:
    """Configuration for OpenTelemetry Resource.

    A Resource represents the entity producing telemetry data.
    This configuration defines the attributes that identify your service.
    """

    service_name: str = "truthound-orchestration"
    """Name of the service producing telemetry."""

    service_version: str = "0.1.0"
    """Version of the service."""

    service_namespace: str = "data-quality"
    """Namespace grouping related services."""

    deployment_environment: str = "development"
    """Deployment environment (development, staging, production)."""

    service_instance_id: str | None = None
    """Unique identifier of the service instance."""

    additional_attributes: ResourceAttributes = field(default_factory=dict)
    """Additional resource attributes."""

    def with_service_name(self, name: str) -> "ResourceConfig":
        """Create a new config with updated service name."""
        return ResourceConfig(
            service_name=name,
            service_version=self.service_version,
            service_namespace=self.service_namespace,
            deployment_environment=self.deployment_environment,
            service_instance_id=self.service_instance_id,
            additional_attributes=self.additional_attributes,
        )

    def with_service_version(self, version: str) -> "ResourceConfig":
        """Create a new config with updated service version."""
        return ResourceConfig(
            service_name=self.service_name,
            service_version=version,
            service_namespace=self.service_namespace,
            deployment_environment=self.deployment_environment,
            service_instance_id=self.service_instance_id,
            additional_attributes=self.additional_attributes,
        )

    def with_environment(self, environment: str) -> "ResourceConfig":
        """Create a new config with updated deployment environment."""
        return ResourceConfig(
            service_name=self.service_name,
            service_version=self.service_version,
            service_namespace=self.service_namespace,
            deployment_environment=environment,
            service_instance_id=self.service_instance_id,
            additional_attributes=self.additional_attributes,
        )

    def with_instance_id(self, instance_id: str) -> "ResourceConfig":
        """Create a new config with updated instance ID."""
        return ResourceConfig(
            service_name=self.service_name,
            service_version=self.service_version,
            service_namespace=self.service_namespace,
            deployment_environment=self.deployment_environment,
            service_instance_id=instance_id,
            additional_attributes=self.additional_attributes,
        )

    def with_attributes(self, **attributes: Any) -> "ResourceConfig":
        """Create a new config with additional attributes."""
        merged = dict(self.additional_attributes)
        merged.update(attributes)
        return ResourceConfig(
            service_name=self.service_name,
            service_version=self.service_version,
            service_namespace=self.service_namespace,
            deployment_environment=self.deployment_environment,
            service_instance_id=self.service_instance_id,
            additional_attributes=merged,
        )

    def to_attributes(self) -> dict[str, Any]:
        """Convert to OpenTelemetry resource attributes dictionary."""
        attrs: dict[str, Any] = {
            "service.name": self.service_name,
            "service.version": self.service_version,
            "service.namespace": self.service_namespace,
            "deployment.environment": self.deployment_environment,
        }
        if self.service_instance_id:
            attrs["service.instance.id"] = self.service_instance_id
        attrs.update(self.additional_attributes)
        return attrs


@dataclass(frozen=True)
class OTLPExporterConfig:
    """Configuration for OTLP exporter.

    Defines how telemetry data is sent to an OpenTelemetry collector
    or backend service.
    """

    endpoint: str = "http://localhost:4317"
    """OTLP collector endpoint URL."""

    protocol: OTLPProtocol = OTLPProtocol.GRPC
    """Transport protocol to use."""

    compression: OTLPCompression = OTLPCompression.GZIP
    """Compression algorithm for payloads."""

    headers: dict[str, str] = field(default_factory=dict)
    """Additional headers to include in requests."""

    timeout_seconds: float = 10.0
    """Request timeout in seconds."""

    insecure: bool = False
    """Whether to use insecure connection (no TLS)."""

    certificate_file: str | None = None
    """Path to CA certificate file for TLS verification."""

    def with_endpoint(self, endpoint: str) -> "OTLPExporterConfig":
        """Create a new config with updated endpoint."""
        return OTLPExporterConfig(
            endpoint=endpoint,
            protocol=self.protocol,
            compression=self.compression,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
            insecure=self.insecure,
            certificate_file=self.certificate_file,
        )

    def with_protocol(self, protocol: OTLPProtocol) -> "OTLPExporterConfig":
        """Create a new config with updated protocol."""
        return OTLPExporterConfig(
            endpoint=self.endpoint,
            protocol=protocol,
            compression=self.compression,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
            insecure=self.insecure,
            certificate_file=self.certificate_file,
        )

    def with_compression(self, compression: OTLPCompression) -> "OTLPExporterConfig":
        """Create a new config with updated compression."""
        return OTLPExporterConfig(
            endpoint=self.endpoint,
            protocol=self.protocol,
            compression=compression,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
            insecure=self.insecure,
            certificate_file=self.certificate_file,
        )

    def with_headers(self, **headers: str) -> "OTLPExporterConfig":
        """Create a new config with additional headers."""
        merged = dict(self.headers)
        merged.update(headers)
        return OTLPExporterConfig(
            endpoint=self.endpoint,
            protocol=self.protocol,
            compression=self.compression,
            headers=merged,
            timeout_seconds=self.timeout_seconds,
            insecure=self.insecure,
            certificate_file=self.certificate_file,
        )

    def with_timeout(self, timeout_seconds: float) -> "OTLPExporterConfig":
        """Create a new config with updated timeout."""
        return OTLPExporterConfig(
            endpoint=self.endpoint,
            protocol=self.protocol,
            compression=self.compression,
            headers=self.headers,
            timeout_seconds=timeout_seconds,
            insecure=self.insecure,
            certificate_file=self.certificate_file,
        )

    def with_tls(
        self,
        insecure: bool = False,
        certificate_file: str | None = None,
    ) -> "OTLPExporterConfig":
        """Create a new config with updated TLS settings."""
        return OTLPExporterConfig(
            endpoint=self.endpoint,
            protocol=self.protocol,
            compression=self.compression,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
            insecure=insecure,
            certificate_file=certificate_file,
        )


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for trace sampling.

    Controls which traces are recorded and exported to reduce
    overhead in high-throughput scenarios.
    """

    enabled: bool = True
    """Whether sampling is enabled."""

    sample_rate: float = 1.0
    """Base sampling rate (0.0 to 1.0). 1.0 = sample everything."""

    parent_based: bool = True
    """Whether to respect parent sampling decisions."""

    error_sample_rate: float = 1.0
    """Sampling rate for error spans (typically higher than base rate)."""

    slow_operation_threshold_ms: float = 1000.0
    """Threshold in ms above which operations are always sampled."""

    slow_operation_sample_rate: float = 1.0
    """Sampling rate for operations exceeding slow threshold."""

    def with_sample_rate(self, rate: float) -> "SamplingConfig":
        """Create a new config with updated sample rate."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {rate}")
        return SamplingConfig(
            enabled=self.enabled,
            sample_rate=rate,
            parent_based=self.parent_based,
            error_sample_rate=self.error_sample_rate,
            slow_operation_threshold_ms=self.slow_operation_threshold_ms,
            slow_operation_sample_rate=self.slow_operation_sample_rate,
        )

    def with_error_sampling(self, rate: float) -> "SamplingConfig":
        """Create a new config with updated error sampling rate."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Error sample rate must be between 0.0 and 1.0, got {rate}")
        return SamplingConfig(
            enabled=self.enabled,
            sample_rate=self.sample_rate,
            parent_based=self.parent_based,
            error_sample_rate=rate,
            slow_operation_threshold_ms=self.slow_operation_threshold_ms,
            slow_operation_sample_rate=self.slow_operation_sample_rate,
        )

    def with_slow_operation_sampling(
        self,
        threshold_ms: float,
        rate: float = 1.0,
    ) -> "SamplingConfig":
        """Create a new config with updated slow operation sampling."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Slow operation sample rate must be between 0.0 and 1.0, got {rate}")
        return SamplingConfig(
            enabled=self.enabled,
            sample_rate=self.sample_rate,
            parent_based=self.parent_based,
            error_sample_rate=self.error_sample_rate,
            slow_operation_threshold_ms=threshold_ms,
            slow_operation_sample_rate=rate,
        )


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for batching exported telemetry.

    Controls how telemetry data is batched before export
    to optimize network usage and reduce overhead.
    """

    max_queue_size: int = 2048
    """Maximum number of items to queue before dropping."""

    max_export_batch_size: int = 512
    """Maximum number of items per export batch."""

    export_timeout_seconds: float = 30.0
    """Maximum time to wait for export to complete."""

    schedule_delay_seconds: float = 5.0
    """Delay between batch exports."""

    def with_queue_size(self, size: int) -> "BatchConfig":
        """Create a new config with updated queue size."""
        return BatchConfig(
            max_queue_size=size,
            max_export_batch_size=self.max_export_batch_size,
            export_timeout_seconds=self.export_timeout_seconds,
            schedule_delay_seconds=self.schedule_delay_seconds,
        )

    def with_batch_size(self, size: int) -> "BatchConfig":
        """Create a new config with updated batch size."""
        return BatchConfig(
            max_queue_size=self.max_queue_size,
            max_export_batch_size=size,
            export_timeout_seconds=self.export_timeout_seconds,
            schedule_delay_seconds=self.schedule_delay_seconds,
        )

    def with_export_timeout(self, timeout_seconds: float) -> "BatchConfig":
        """Create a new config with updated export timeout."""
        return BatchConfig(
            max_queue_size=self.max_queue_size,
            max_export_batch_size=self.max_export_batch_size,
            export_timeout_seconds=timeout_seconds,
            schedule_delay_seconds=self.schedule_delay_seconds,
        )

    def with_schedule_delay(self, delay_seconds: float) -> "BatchConfig":
        """Create a new config with updated schedule delay."""
        return BatchConfig(
            max_queue_size=self.max_queue_size,
            max_export_batch_size=self.max_export_batch_size,
            export_timeout_seconds=self.export_timeout_seconds,
            schedule_delay_seconds=delay_seconds,
        )


@dataclass(frozen=True)
class OTelConfig:
    """Main OpenTelemetry configuration.

    This is the top-level configuration class that aggregates all
    OpenTelemetry settings. Use builder methods to customize.

    Example:
        config = OTelConfig().with_enabled(True).with_service_name("my-service")
    """

    enabled: bool = True
    """Whether OpenTelemetry integration is enabled."""

    metrics_enabled: bool = True
    """Whether metrics export is enabled."""

    tracing_enabled: bool = True
    """Whether tracing is enabled."""

    bridge_enabled: bool = True
    """Whether to bridge existing metrics/tracing to OTel."""

    metrics_exporter: ExporterType = ExporterType.OTLP
    """Exporter type for metrics."""

    traces_exporter: ExporterType = ExporterType.OTLP
    """Exporter type for traces."""

    resource: ResourceConfig = field(default_factory=ResourceConfig)
    """Resource configuration."""

    otlp: OTLPExporterConfig = field(default_factory=OTLPExporterConfig)
    """OTLP exporter configuration."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    """Trace sampling configuration."""

    batch: BatchConfig = field(default_factory=BatchConfig)
    """Batch export configuration."""

    propagators: tuple[str, ...] = ("tracecontext", "baggage")
    """Context propagators to use."""

    def with_enabled(self, enabled: bool) -> "OTelConfig":
        """Create a new config with updated enabled status."""
        return OTelConfig(
            enabled=enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_metrics_enabled(self, enabled: bool) -> "OTelConfig":
        """Create a new config with updated metrics enabled status."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_tracing_enabled(self, enabled: bool) -> "OTelConfig":
        """Create a new config with updated tracing enabled status."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_bridge_enabled(self, enabled: bool) -> "OTelConfig":
        """Create a new config with updated bridge enabled status."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_exporters(
        self,
        metrics: ExporterType | None = None,
        traces: ExporterType | None = None,
    ) -> "OTelConfig":
        """Create a new config with updated exporter types."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=metrics if metrics is not None else self.metrics_exporter,
            traces_exporter=traces if traces is not None else self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_resource(self, resource: ResourceConfig) -> "OTelConfig":
        """Create a new config with updated resource configuration."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_service_name(self, name: str) -> "OTelConfig":
        """Create a new config with updated service name."""
        return self.with_resource(self.resource.with_service_name(name))

    def with_otlp(self, otlp: OTLPExporterConfig) -> "OTelConfig":
        """Create a new config with updated OTLP configuration."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_endpoint(self, endpoint: str) -> "OTelConfig":
        """Create a new config with updated OTLP endpoint."""
        return self.with_otlp(self.otlp.with_endpoint(endpoint))

    def with_sampling(self, sampling: SamplingConfig) -> "OTelConfig":
        """Create a new config with updated sampling configuration."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=sampling,
            batch=self.batch,
            propagators=self.propagators,
        )

    def with_sample_rate(self, rate: float) -> "OTelConfig":
        """Create a new config with updated sample rate."""
        return self.with_sampling(self.sampling.with_sample_rate(rate))

    def with_batch(self, batch: BatchConfig) -> "OTelConfig":
        """Create a new config with updated batch configuration."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=batch,
            propagators=self.propagators,
        )

    def with_propagators(self, *propagators: str) -> "OTelConfig":
        """Create a new config with updated propagators."""
        return OTelConfig(
            enabled=self.enabled,
            metrics_enabled=self.metrics_enabled,
            tracing_enabled=self.tracing_enabled,
            bridge_enabled=self.bridge_enabled,
            metrics_exporter=self.metrics_exporter,
            traces_exporter=self.traces_exporter,
            resource=self.resource,
            otlp=self.otlp,
            sampling=self.sampling,
            batch=self.batch,
            propagators=propagators,
        )


# Preset configurations
DEFAULT_OTEL_CONFIG = OTelConfig()
"""Default OpenTelemetry configuration with sensible defaults."""

DEVELOPMENT_OTEL_CONFIG = OTelConfig(
    enabled=True,
    metrics_exporter=ExporterType.CONSOLE,
    traces_exporter=ExporterType.CONSOLE,
    resource=ResourceConfig(deployment_environment="development"),
    sampling=SamplingConfig(sample_rate=1.0),  # Sample everything in dev
)
"""Development configuration with console exporters for debugging."""

PRODUCTION_OTEL_CONFIG = OTelConfig(
    enabled=True,
    metrics_exporter=ExporterType.OTLP,
    traces_exporter=ExporterType.OTLP,
    resource=ResourceConfig(deployment_environment="production"),
    otlp=OTLPExporterConfig(
        compression=OTLPCompression.GZIP,
        timeout_seconds=30.0,
    ),
    sampling=SamplingConfig(
        sample_rate=0.1,  # Sample 10% in production
        error_sample_rate=1.0,  # Always sample errors
        slow_operation_threshold_ms=500.0,
        slow_operation_sample_rate=1.0,
    ),
    batch=BatchConfig(
        max_queue_size=4096,
        max_export_batch_size=1024,
    ),
)
"""Production configuration with OTLP export and reduced sampling."""

TESTING_OTEL_CONFIG = OTelConfig(
    enabled=True,
    metrics_exporter=ExporterType.MEMORY,
    traces_exporter=ExporterType.MEMORY,
    resource=ResourceConfig(deployment_environment="testing"),
    sampling=SamplingConfig(sample_rate=1.0),
)
"""Testing configuration with in-memory exporters."""

DISABLED_OTEL_CONFIG = OTelConfig(
    enabled=False,
    metrics_enabled=False,
    tracing_enabled=False,
    bridge_enabled=False,
    metrics_exporter=ExporterType.NONE,
    traces_exporter=ExporterType.NONE,
)
"""Disabled configuration for environments without telemetry."""
