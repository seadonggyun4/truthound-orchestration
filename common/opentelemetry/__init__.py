"""OpenTelemetry integration for truthound-orchestration.

This module provides OpenTelemetry integration for data quality operations,
including metrics export, distributed tracing, and context propagation.

Features:
- Bridge layer for connecting internal telemetry to OpenTelemetry
- Configurable MeterProvider and TracerProvider wrappers
- Data quality-specific semantic conventions
- Custom samplers for intelligent sampling
- Engine metrics hook for automatic instrumentation

Installation:
    pip install truthound-orchestration[opentelemetry]

Basic Usage:
    from common.opentelemetry import (
        OTelConfig,
        configure_opentelemetry,
        create_otel_engine_hook,
    )

    # Configure OpenTelemetry
    config = OTelConfig().with_service_name("my-service")
    configure_opentelemetry(config)

    # Create engine hook for automatic instrumentation
    hook = create_otel_engine_hook()
    instrumented = InstrumentedEngine(engine, hooks=[hook])

Example with explicit providers:
    from common.opentelemetry import (
        OTelConfig,
        create_meter_provider,
        create_tracer_provider,
    )

    config = OTelConfig().with_endpoint("http://collector:4317")

    meter_provider = create_meter_provider(config)
    tracer_provider = create_tracer_provider(config)

    # Use providers...

    meter_provider.shutdown()
    tracer_provider.shutdown()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Core configuration and types
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
from common.opentelemetry.exceptions import (
    OTelBridgeError,
    OTelConfigurationError,
    OTelContextError,
    OTelError,
    OTelExporterError,
    OTelNotInstalledError,
    OTelProviderError,
    OTelSamplingError,
)
from common.opentelemetry.types import (
    Attributes,
    AttributeValue,
    ContextCarrier,
    ExporterType,
    MetricAttributes,
    OTLPCompression,
    OTLPProtocol,
    ResourceAttributes,
    SamplingDecision,
    SpanAttributes,
)

# Semantic conventions
from common.opentelemetry.semantic import (
    CheckAttributes,
    CheckStatus,
    DataQualityAttributes,
    DatasetAttributes,
    EngineAttributes,
    LearnAttributes,
    OperationStatus,
    OperationType,
    ProfileAttributes,
    RuleAttributes,
    Severity,
    create_check_attributes,
    create_engine_attributes,
    create_learn_attributes,
    create_profile_attributes,
)

if TYPE_CHECKING:
    from common.opentelemetry.bridge import (
        ContextBridge,
        MetricBridge,
        MetricBridgeExporter,
        SpanBridge,
        SpanBridgeExporter,
    )
    from common.opentelemetry.integration import (
        AlwaysOffSampler,
        AlwaysOnSampler,
        DataQualitySampler,
        OTelEngineMetricsHook,
        ParentBasedSampler,
        RatioBasedSampler,
    )
    from common.opentelemetry.providers import (
        InMemorySpanExporter,
        OTelMeterProvider,
        OTelTracerProvider,
        ResourceFactory,
    )

logger = logging.getLogger(__name__)

__all__ = [
    # Configuration
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
    # Types
    "OTLPProtocol",
    "OTLPCompression",
    "ExporterType",
    "SamplingDecision",
    "Attributes",
    "AttributeValue",
    "ResourceAttributes",
    "SpanAttributes",
    "MetricAttributes",
    "ContextCarrier",
    # Exceptions
    "OTelError",
    "OTelConfigurationError",
    "OTelExporterError",
    "OTelProviderError",
    "OTelBridgeError",
    "OTelSamplingError",
    "OTelContextError",
    "OTelNotInstalledError",
    # Semantic conventions
    "DataQualityAttributes",
    "EngineAttributes",
    "CheckAttributes",
    "ProfileAttributes",
    "LearnAttributes",
    "RuleAttributes",
    "DatasetAttributes",
    "OperationType",
    "OperationStatus",
    "CheckStatus",
    "Severity",
    "create_engine_attributes",
    "create_check_attributes",
    "create_profile_attributes",
    "create_learn_attributes",
    # Lazy-loaded functions
    "configure_opentelemetry",
    "create_meter_provider",
    "create_tracer_provider",
    "create_resource",
    "create_metric_bridge",
    "create_span_bridge",
    "create_otel_engine_hook",
    "create_sampler",
    "inject_context",
    "extract_context",
]


def _lazy_import_providers():
    """Lazily import provider modules."""
    from common.opentelemetry.providers import (
        OTelMeterProvider,
        OTelTracerProvider,
        create_meter_provider,
        create_resource,
        create_tracer_provider,
        get_default_resource,
        set_global_meter_provider,
        set_global_tracer_provider,
    )
    return {
        "OTelMeterProvider": OTelMeterProvider,
        "OTelTracerProvider": OTelTracerProvider,
        "create_meter_provider": create_meter_provider,
        "create_tracer_provider": create_tracer_provider,
        "create_resource": create_resource,
        "get_default_resource": get_default_resource,
        "set_global_meter_provider": set_global_meter_provider,
        "set_global_tracer_provider": set_global_tracer_provider,
    }


def _lazy_import_bridge():
    """Lazily import bridge modules."""
    from common.opentelemetry.bridge import (
        ContextBridge,
        MetricBridge,
        MetricBridgeExporter,
        SpanBridge,
        SpanBridgeExporter,
        context_from_internal,
        context_to_internal,
        create_metric_bridge,
        create_span_bridge,
        extract_context,
        inject_context,
    )
    return {
        "MetricBridgeExporter": MetricBridgeExporter,
        "MetricBridge": MetricBridge,
        "SpanBridgeExporter": SpanBridgeExporter,
        "SpanBridge": SpanBridge,
        "ContextBridge": ContextBridge,
        "create_metric_bridge": create_metric_bridge,
        "create_span_bridge": create_span_bridge,
        "inject_context": inject_context,
        "extract_context": extract_context,
        "context_from_internal": context_from_internal,
        "context_to_internal": context_to_internal,
    }


def _lazy_import_integration():
    """Lazily import integration modules."""
    from common.opentelemetry.integration import (
        AlwaysOffSampler,
        AlwaysOnSampler,
        DataQualitySampler,
        OTelEngineMetricsHook,
        ParentBasedSampler,
        RatioBasedSampler,
        create_otel_engine_hook,
        create_sampler,
    )
    return {
        "OTelEngineMetricsHook": OTelEngineMetricsHook,
        "create_otel_engine_hook": create_otel_engine_hook,
        "DataQualitySampler": DataQualitySampler,
        "AlwaysOnSampler": AlwaysOnSampler,
        "AlwaysOffSampler": AlwaysOffSampler,
        "RatioBasedSampler": RatioBasedSampler,
        "ParentBasedSampler": ParentBasedSampler,
        "create_sampler": create_sampler,
    }


# Global state for configured providers
_meter_provider: "OTelMeterProvider | None" = None
_tracer_provider: "OTelTracerProvider | None" = None
_is_configured = False


def configure_opentelemetry(
    config: OTelConfig | None = None,
    set_global: bool = True,
) -> tuple["OTelMeterProvider", "OTelTracerProvider"]:
    """Configure OpenTelemetry with the given configuration.

    This is the main entry point for configuring OpenTelemetry
    in your application.

    Args:
        config: OpenTelemetry configuration. Defaults to DEFAULT_OTEL_CONFIG.
        set_global: Whether to set providers as global.

    Returns:
        Tuple of (meter_provider, tracer_provider).

    Example:
        configure_opentelemetry(
            OTelConfig()
            .with_service_name("my-service")
            .with_endpoint("http://collector:4317")
        )
    """
    global _meter_provider, _tracer_provider, _is_configured

    providers = _lazy_import_providers()
    config = config or DEFAULT_OTEL_CONFIG

    if not config.enabled:
        logger.info("OpenTelemetry is disabled by configuration")
        return None, None

    try:
        # Create providers
        _meter_provider = providers["create_meter_provider"](config)
        _tracer_provider = providers["create_tracer_provider"](config)

        # Set as global if requested
        if set_global:
            if _meter_provider and _meter_provider.provider:
                providers["set_global_meter_provider"](_meter_provider)
            if _tracer_provider and _tracer_provider.provider:
                providers["set_global_tracer_provider"](_tracer_provider)

        _is_configured = True
        logger.info(
            f"OpenTelemetry configured: "
            f"service={config.resource.service_name}, "
            f"metrics={config.metrics_exporter.value}, "
            f"traces={config.traces_exporter.value}"
        )

        return _meter_provider, _tracer_provider
    except OTelNotInstalledError:
        logger.warning(
            "OpenTelemetry SDK is not installed. "
            "Install with: pip install truthound-orchestration[opentelemetry]"
        )
        return None, None
    except Exception as e:
        logger.error(f"Failed to configure OpenTelemetry: {e}")
        raise


def shutdown_opentelemetry() -> None:
    """Shutdown OpenTelemetry providers.

    Call this during application shutdown to ensure
    all telemetry data is exported.
    """
    global _meter_provider, _tracer_provider, _is_configured

    if _meter_provider:
        _meter_provider.shutdown()
        _meter_provider = None

    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None

    _is_configured = False
    logger.info("OpenTelemetry providers shutdown")


def is_configured() -> bool:
    """Check if OpenTelemetry is configured.

    Returns:
        True if OpenTelemetry has been configured.
    """
    return _is_configured


# Convenience functions that wrap the lazy imports


def create_meter_provider(
    config: OTelConfig | None = None,
    exporter=None,
) -> "OTelMeterProvider":
    """Create a configured OTelMeterProvider.

    Args:
        config: OpenTelemetry configuration.
        exporter: Custom metric exporter.

    Returns:
        Configured OTelMeterProvider.
    """
    providers = _lazy_import_providers()
    return providers["create_meter_provider"](config, exporter)


def create_tracer_provider(
    config: OTelConfig | None = None,
    exporter=None,
) -> "OTelTracerProvider":
    """Create a configured OTelTracerProvider.

    Args:
        config: OpenTelemetry configuration.
        exporter: Custom span exporter.

    Returns:
        Configured OTelTracerProvider.
    """
    providers = _lazy_import_providers()
    return providers["create_tracer_provider"](config, exporter)


def create_resource(config: ResourceConfig | None = None):
    """Create an OpenTelemetry Resource from configuration.

    Args:
        config: Resource configuration.

    Returns:
        OpenTelemetry Resource instance.
    """
    providers = _lazy_import_providers()
    return providers["create_resource"](config)


def create_metric_bridge(
    meter_provider=None,
    meter_name: str = "truthound.orchestration",
    prefix: str = "dq",
    default_labels: dict[str, str] | None = None,
) -> "MetricBridgeExporter":
    """Create a metric bridge exporter.

    Args:
        meter_provider: OpenTelemetry MeterProvider.
        meter_name: Name for the meter.
        prefix: Prefix for metric names.
        default_labels: Default labels for all metrics.

    Returns:
        Configured MetricBridgeExporter.
    """
    bridge = _lazy_import_bridge()
    return bridge["create_metric_bridge"](
        meter_provider, meter_name, prefix, default_labels
    )


def create_span_bridge(
    tracer_provider=None,
    tracer_name: str = "truthound.orchestration",
    default_attributes: dict[str, str] | None = None,
) -> "SpanBridgeExporter":
    """Create a span bridge exporter.

    Args:
        tracer_provider: OpenTelemetry TracerProvider.
        tracer_name: Name for the tracer.
        default_attributes: Default attributes for all spans.

    Returns:
        Configured SpanBridgeExporter.
    """
    bridge = _lazy_import_bridge()
    return bridge["create_span_bridge"](
        tracer_provider, tracer_name, default_attributes
    )


def create_otel_engine_hook(
    config: OTelConfig | None = None,
    meter_name: str = "truthound.orchestration.engine",
    tracer_name: str = "truthound.orchestration.engine",
    prefix: str = "dq_engine",
) -> "OTelEngineMetricsHook":
    """Create an OpenTelemetry engine metrics hook.

    Args:
        config: OpenTelemetry configuration.
        meter_name: Name for the meter.
        tracer_name: Name for the tracer.
        prefix: Prefix for metric names.

    Returns:
        Configured OTelEngineMetricsHook.
    """
    integration = _lazy_import_integration()
    return integration["create_otel_engine_hook"](
        config, meter_name, tracer_name, prefix
    )


def create_sampler(
    config: SamplingConfig | None = None,
    operation_rates: dict[str, float] | None = None,
) -> "DataQualitySampler":
    """Create a data quality sampler.

    Args:
        config: Sampling configuration.
        operation_rates: Per-operation sampling rates.

    Returns:
        Configured DataQualitySampler.
    """
    integration = _lazy_import_integration()
    return integration["create_sampler"](config, operation_rates)


def inject_context(
    carrier: ContextCarrier,
    propagators: tuple[str, ...] = ("tracecontext", "baggage"),
) -> None:
    """Inject current context into a carrier.

    Args:
        carrier: Dictionary to inject context into.
        propagators: Propagators to use.
    """
    bridge = _lazy_import_bridge()
    bridge["inject_context"](carrier, propagators)


def extract_context(
    carrier: ContextCarrier,
    propagators: tuple[str, ...] = ("tracecontext", "baggage"),
):
    """Extract context from a carrier.

    Args:
        carrier: Dictionary to extract context from.
        propagators: Propagators to use.

    Returns:
        Extracted OpenTelemetry context.
    """
    bridge = _lazy_import_bridge()
    return bridge["extract_context"](carrier, propagators)
