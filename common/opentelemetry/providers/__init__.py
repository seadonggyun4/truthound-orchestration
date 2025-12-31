"""OpenTelemetry providers for truthound-orchestration.

This module provides configured MeterProvider and TracerProvider
implementations for metrics and tracing.
"""

from common.opentelemetry.providers.meter import (
    OTelMeterProvider,
    create_meter_provider,
    get_global_meter_provider,
    set_global_meter_provider,
)
from common.opentelemetry.providers.resource import (
    ResourceFactory,
    create_resource,
    get_default_resource,
    merge_resources,
)
from common.opentelemetry.providers.tracer import (
    InMemorySpanExporter,
    OTelTracerProvider,
    create_tracer_provider,
    get_global_tracer_provider,
    set_global_tracer_provider,
)

__all__ = [
    # Resource
    "create_resource",
    "get_default_resource",
    "merge_resources",
    "ResourceFactory",
    # Meter provider
    "OTelMeterProvider",
    "create_meter_provider",
    "get_global_meter_provider",
    "set_global_meter_provider",
    # Tracer provider
    "OTelTracerProvider",
    "create_tracer_provider",
    "get_global_tracer_provider",
    "set_global_tracer_provider",
    "InMemorySpanExporter",
]
