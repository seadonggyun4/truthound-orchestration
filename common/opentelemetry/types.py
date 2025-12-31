"""OpenTelemetry type definitions for truthound-orchestration.

This module defines enums and type aliases used throughout the OpenTelemetry
integration layer.
"""

from enum import Enum
from typing import Any, Mapping, Sequence

__all__ = [
    # Enums
    "OTLPProtocol",
    "OTLPCompression",
    "ExporterType",
    "SamplingDecision",
    # Type aliases
    "Attributes",
    "AttributeValue",
    "ResourceAttributes",
    "SpanAttributes",
    "MetricAttributes",
]


class OTLPProtocol(Enum):
    """OTLP transport protocol options.

    Defines the available transport protocols for sending telemetry data
    to an OpenTelemetry collector or backend.
    """

    GRPC = "grpc"
    """gRPC protocol - recommended for high-throughput scenarios."""

    HTTP_PROTOBUF = "http/protobuf"
    """HTTP with Protocol Buffers encoding - good balance of compatibility and efficiency."""

    HTTP_JSON = "http/json"
    """HTTP with JSON encoding - most compatible but less efficient."""


class OTLPCompression(Enum):
    """OTLP compression options.

    Defines compression algorithms for reducing telemetry payload size.
    """

    NONE = "none"
    """No compression - lowest CPU overhead but highest bandwidth."""

    GZIP = "gzip"
    """GZIP compression - good compression ratio with moderate CPU overhead."""


class ExporterType(Enum):
    """Types of telemetry exporters.

    Used to specify which exporter to use for metrics and traces.
    """

    OTLP = "otlp"
    """OTLP exporter - sends data to OpenTelemetry collector."""

    CONSOLE = "console"
    """Console exporter - prints to stdout for debugging."""

    MEMORY = "memory"
    """In-memory exporter - stores data in memory for testing."""

    NONE = "none"
    """No-op exporter - discards all data."""


class SamplingDecision(Enum):
    """Sampling decision for traces.

    Determines whether a span should be recorded and/or sampled.
    """

    DROP = "drop"
    """Drop the span entirely - not recorded or sampled."""

    RECORD_ONLY = "record_only"
    """Record the span locally but don't export."""

    RECORD_AND_SAMPLE = "record_and_sample"
    """Record and export the span."""


# Type aliases for OpenTelemetry attributes
AttributeValue = str | int | float | bool | Sequence[str] | Sequence[int] | Sequence[float] | Sequence[bool]
"""Valid attribute value types per OpenTelemetry specification."""

Attributes = Mapping[str, AttributeValue]
"""Generic attributes mapping."""

ResourceAttributes = Mapping[str, AttributeValue]
"""Resource attributes identifying the entity producing telemetry."""

SpanAttributes = Mapping[str, AttributeValue]
"""Span attributes describing the operation."""

MetricAttributes = Mapping[str, AttributeValue]
"""Metric attributes (labels/dimensions)."""

# Context types
ContextCarrier = dict[str, Any]
"""Carrier for context propagation (typically HTTP headers)."""
