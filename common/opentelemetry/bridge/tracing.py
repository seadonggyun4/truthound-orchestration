"""Bridge for converting internal spans to OpenTelemetry format.

This module provides the SpanBridgeExporter that implements the
internal SpanExporter protocol and bridges spans to OpenTelemetry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping

from common.opentelemetry.exceptions import OTelBridgeError, OTelNotInstalledError

if TYPE_CHECKING:
    from opentelemetry.trace import Span as OTelSpan
    from opentelemetry.trace import Tracer, TracerProvider

__all__ = [
    "SpanBridgeExporter",
    "SpanBridge",
    "create_span_bridge",
]

logger = logging.getLogger(__name__)


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.trace  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="tracing bridge") from e


class SpanBridge:
    """Bridge for converting internal spans to OpenTelemetry spans.

    This class handles the translation of internal span data to
    OpenTelemetry span format, maintaining proper parent-child relationships.
    """

    def __init__(
        self,
        tracer: Tracer,
        default_attributes: dict[str, Any] | None = None,
    ) -> None:
        """Initialize SpanBridge.

        Args:
            tracer: OpenTelemetry Tracer instance.
            default_attributes: Default attributes to add to all spans.
        """
        self._tracer = tracer
        self._default_attributes = default_attributes or {}
        self._active_spans: dict[str, OTelSpan] = {}

    def _merge_attributes(self, attributes: Mapping[str, Any] | None) -> dict[str, Any]:
        """Merge default attributes with provided attributes."""
        merged = dict(self._default_attributes)
        if attributes:
            merged.update(attributes)
        return merged

    def start_span(
        self,
        name: str,
        span_id: str | None = None,
        parent_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        kind: str = "INTERNAL",
    ) -> OTelSpan:
        """Start a new span.

        Args:
            name: Span name.
            span_id: Internal span ID for tracking.
            parent_id: Parent span ID.
            attributes: Span attributes.
            kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER).

        Returns:
            OpenTelemetry Span instance.
        """
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind

        # Map kind string to SpanKind
        kind_map = {
            "INTERNAL": SpanKind.INTERNAL,
            "SERVER": SpanKind.SERVER,
            "CLIENT": SpanKind.CLIENT,
            "PRODUCER": SpanKind.PRODUCER,
            "CONSUMER": SpanKind.CONSUMER,
        }
        otel_kind = kind_map.get(kind.upper(), SpanKind.INTERNAL)

        # Get parent context if parent_id is provided
        context = None
        if parent_id and parent_id in self._active_spans:
            parent_span = self._active_spans[parent_id]
            context = trace.set_span_in_context(parent_span)

        # Start the span
        span = self._tracer.start_span(
            name=name,
            context=context,
            kind=otel_kind,
            attributes=self._merge_attributes(attributes),
        )

        # Track the span if we have an ID
        if span_id:
            self._active_spans[span_id] = span

        return span

    def end_span(
        self,
        span: OTelSpan,
        span_id: str | None = None,
        status: str = "OK",
        exception: Exception | None = None,
    ) -> None:
        """End a span.

        Args:
            span: OpenTelemetry Span instance.
            span_id: Internal span ID for cleanup.
            status: Span status (OK, ERROR).
            exception: Exception to record if status is ERROR.
        """
        from opentelemetry.trace import Status, StatusCode

        # Set status
        if exception or status.upper() == "ERROR":
            span.set_status(Status(StatusCode.ERROR, str(exception) if exception else None))
            if exception:
                span.record_exception(exception)
        else:
            span.set_status(Status(StatusCode.OK))

        # End the span
        span.end()

        # Clean up tracking
        if span_id and span_id in self._active_spans:
            del self._active_spans[span_id]

    def add_event(
        self,
        span: OTelSpan,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        """Add an event to a span.

        Args:
            span: OpenTelemetry Span instance.
            name: Event name.
            attributes: Event attributes.
        """
        span.add_event(name, attributes=dict(attributes) if attributes else None)

    def set_attribute(
        self,
        span: OTelSpan,
        key: str,
        value: Any,
    ) -> None:
        """Set a span attribute.

        Args:
            span: OpenTelemetry Span instance.
            key: Attribute key.
            value: Attribute value.
        """
        span.set_attribute(key, value)


class SpanBridgeExporter:
    """Exporter that bridges internal spans to OpenTelemetry.

    This class implements the internal SpanExporter protocol and
    forwards spans to OpenTelemetry.

    Example:
        from opentelemetry.sdk.trace import TracerProvider
        provider = TracerProvider()
        exporter = SpanBridgeExporter(provider)

        # Use with internal tracing registry
        registry.add_exporter(exporter)
    """

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        tracer_name: str = "truthound.orchestration",
        default_attributes: dict[str, Any] | None = None,
    ) -> None:
        """Initialize SpanBridgeExporter.

        Args:
            tracer_provider: OpenTelemetry TracerProvider. If None, uses global.
            tracer_name: Name for the tracer.
            default_attributes: Default attributes for all spans.
        """
        _check_otel_installed()

        from opentelemetry.trace import get_tracer_provider

        self._provider = tracer_provider or get_tracer_provider()
        self._tracer = self._provider.get_tracer(tracer_name)
        self._bridge = SpanBridge(
            tracer=self._tracer,
            default_attributes=default_attributes,
        )
        self._exported_count = 0

    @property
    def exported_count(self) -> int:
        """Number of spans exported."""
        return self._exported_count

    def export(self, spans: list[Any]) -> None:
        """Export internal spans to OpenTelemetry.

        Converts internal SpanData objects to OpenTelemetry format.

        Args:
            spans: List of internal SpanData objects.
        """
        for span_data in spans:
            try:
                self._export_span(span_data)
                self._exported_count += 1
            except Exception as e:
                logger.warning(f"Failed to export span: {e}")

    def _export_span(self, span_data: Any) -> None:
        """Export a single span.

        Args:
            span_data: Internal SpanData object.
        """
        name = getattr(span_data, "name", "unknown")
        span_id = getattr(span_data, "span_id", None)
        parent_id = getattr(span_data, "parent_id", None)
        attributes = getattr(span_data, "attributes", None)
        status = getattr(span_data, "status", "OK")
        events = getattr(span_data, "events", [])
        kind = getattr(span_data, "kind", "INTERNAL")

        # Handle kind enum
        if hasattr(kind, "value"):
            kind = kind.value
        kind = str(kind).upper()

        try:
            # Start the span
            span = self._bridge.start_span(
                name=name,
                span_id=span_id,
                parent_id=parent_id,
                attributes=dict(attributes) if attributes else None,
                kind=kind,
            )

            # Add events
            for event in events:
                event_name = getattr(event, "name", str(event))
                event_attrs = getattr(event, "attributes", None)
                self._bridge.add_event(span, event_name, event_attrs)

            # Handle status
            status_str = status
            if hasattr(status, "value"):
                status_str = status.value
            status_str = str(status_str).upper()

            # End the span
            self._bridge.end_span(
                span=span,
                span_id=span_id,
                status=status_str,
            )
        except Exception as e:
            raise OTelBridgeError(
                message=f"Failed to convert span '{name}' to OpenTelemetry format",
                bridge_type="tracing",
                source_type="internal_span",
                target_type="otel_span",
                original_error=e,
            ) from e

    def flush(self) -> None:
        """Flush any buffered spans.

        Note: OpenTelemetry SDK handles flushing via the processor.
        This method exists for compatibility with the internal protocol.
        """
        # OpenTelemetry SDK handles flushing via the span processor
        pass

    def shutdown(self) -> None:
        """Shutdown the exporter.

        Note: The TracerProvider should be shutdown separately.
        """
        # TracerProvider should be shutdown by the caller
        pass


def create_span_bridge(
    tracer_provider: TracerProvider | None = None,
    tracer_name: str = "truthound.orchestration",
    default_attributes: dict[str, Any] | None = None,
) -> SpanBridgeExporter:
    """Create a span bridge exporter.

    Factory function for creating a SpanBridgeExporter with
    convenient defaults.

    Args:
        tracer_provider: OpenTelemetry TracerProvider. If None, uses global.
        tracer_name: Name for the tracer.
        default_attributes: Default attributes for all spans.

    Returns:
        Configured SpanBridgeExporter instance.

    Example:
        exporter = create_span_bridge()
        registry.add_exporter(exporter)
    """
    return SpanBridgeExporter(
        tracer_provider=tracer_provider,
        tracer_name=tracer_name,
        default_attributes=default_attributes,
    )
