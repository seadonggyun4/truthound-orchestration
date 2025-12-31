"""Bridge layer for connecting internal telemetry to OpenTelemetry.

This module provides bridges that implement internal protocols
and forward telemetry data to OpenTelemetry.
"""

from common.opentelemetry.bridge.context import (
    ContextBridge,
    context_from_internal,
    context_to_internal,
    extract_context,
    get_current_context,
    inject_context,
    set_current_context,
)
from common.opentelemetry.bridge.metrics import (
    MetricBridge,
    MetricBridgeExporter,
    create_metric_bridge,
)
from common.opentelemetry.bridge.tracing import (
    SpanBridge,
    SpanBridgeExporter,
    create_span_bridge,
)

__all__ = [
    # Metrics bridge
    "MetricBridgeExporter",
    "MetricBridge",
    "create_metric_bridge",
    # Tracing bridge
    "SpanBridgeExporter",
    "SpanBridge",
    "create_span_bridge",
    # Context bridge
    "ContextBridge",
    "inject_context",
    "extract_context",
    "get_current_context",
    "set_current_context",
    "context_from_internal",
    "context_to_internal",
]
