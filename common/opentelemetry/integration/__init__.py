"""OpenTelemetry integration with data quality engines.

This module provides hooks and samplers for integrating
OpenTelemetry with the data quality engine system.
"""

from common.opentelemetry.integration.engine_hook import (
    OTelEngineMetricsHook,
    create_otel_engine_hook,
)
from common.opentelemetry.integration.sampling import (
    AlwaysOffSampler,
    AlwaysOnSampler,
    DataQualitySampler,
    ParentBasedSampler,
    RatioBasedSampler,
    create_sampler,
)

__all__ = [
    # Engine hook
    "OTelEngineMetricsHook",
    "create_otel_engine_hook",
    # Samplers
    "DataQualitySampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "RatioBasedSampler",
    "ParentBasedSampler",
    "create_sampler",
]
