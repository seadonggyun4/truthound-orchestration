"""Bridge for context propagation between internal and OpenTelemetry contexts.

This module provides utilities for converting between the internal TraceContext
and OpenTelemetry context, enabling seamless integration with both systems.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from common.opentelemetry.exceptions import OTelContextError, OTelNotInstalledError
from common.opentelemetry.types import ContextCarrier

if TYPE_CHECKING:
    from opentelemetry.context import Context

__all__ = [
    "ContextBridge",
    "inject_context",
    "extract_context",
    "get_current_context",
    "set_current_context",
    "context_from_internal",
    "context_to_internal",
]

logger = logging.getLogger(__name__)


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.context  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="context bridge") from e


class ContextBridge:
    """Bridge for converting between internal and OpenTelemetry contexts.

    This class provides bidirectional conversion between the internal
    TraceContext format (W3C Trace Context compatible) and OpenTelemetry's
    native context representation.

    Example:
        bridge = ContextBridge()

        # Inject context into headers
        headers = {}
        bridge.inject(headers)

        # Extract context from headers
        context = bridge.extract(headers)
    """

    def __init__(
        self,
        propagators: tuple[str, ...] = ("tracecontext", "baggage"),
    ) -> None:
        """Initialize ContextBridge.

        Args:
            propagators: List of propagator names to use.
                Defaults to W3C Trace Context and Baggage.
        """
        _check_otel_installed()

        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.propagators.composite import CompositePropagator

        self._propagators = propagators
        self._propagator = self._create_composite_propagator()

    def _create_composite_propagator(self) -> Any:
        """Create a composite propagator from configured propagators."""
        from opentelemetry.propagators.composite import CompositePropagator

        propagators = []
        for name in self._propagators:
            propagator = self._get_propagator(name)
            if propagator:
                propagators.append(propagator)

        return CompositePropagator(propagators)

    def _get_propagator(self, name: str) -> Any:
        """Get a propagator by name.

        Args:
            name: Propagator name (tracecontext, baggage, b3, etc.)

        Returns:
            Propagator instance or None if not available.
        """
        try:
            if name == "tracecontext":
                from opentelemetry.trace.propagation.tracecontext import (
                    TraceContextTextMapPropagator,
                )
                return TraceContextTextMapPropagator()
            elif name == "baggage":
                from opentelemetry.baggage.propagation import W3CBaggagePropagator
                return W3CBaggagePropagator()
            elif name == "b3":
                try:
                    from opentelemetry.propagators.b3 import B3MultiFormat
                    return B3MultiFormat()
                except ImportError:
                    logger.debug("B3 propagator not installed")
                    return None
            elif name == "b3multi":
                try:
                    from opentelemetry.propagators.b3 import B3MultiFormat
                    return B3MultiFormat()
                except ImportError:
                    logger.debug("B3 propagator not installed")
                    return None
            elif name == "jaeger":
                try:
                    from opentelemetry.propagators.jaeger import JaegerPropagator
                    return JaegerPropagator()
                except ImportError:
                    logger.debug("Jaeger propagator not installed")
                    return None
            else:
                logger.warning(f"Unknown propagator: {name}")
                return None
        except Exception as e:
            logger.warning(f"Failed to create propagator '{name}': {e}")
            return None

    def inject(
        self,
        carrier: ContextCarrier,
        context: Context | None = None,
    ) -> None:
        """Inject context into a carrier (e.g., HTTP headers).

        Args:
            carrier: Dictionary to inject context into.
            context: OpenTelemetry context to inject. If None, uses current.
        """
        try:
            from opentelemetry import context as otel_context

            ctx = context or otel_context.get_current()
            self._propagator.inject(carrier, context=ctx)
        except Exception as e:
            raise OTelContextError(
                message="Failed to inject context into carrier",
                operation="inject",
                propagator=str(self._propagators),
                original_error=e,
            ) from e

    def extract(
        self,
        carrier: ContextCarrier,
    ) -> Context:
        """Extract context from a carrier (e.g., HTTP headers).

        Args:
            carrier: Dictionary to extract context from.

        Returns:
            Extracted OpenTelemetry context.
        """
        try:
            return self._propagator.extract(carrier)
        except Exception as e:
            raise OTelContextError(
                message="Failed to extract context from carrier",
                operation="extract",
                propagator=str(self._propagators),
                original_error=e,
            ) from e


def inject_context(
    carrier: ContextCarrier,
    propagators: tuple[str, ...] = ("tracecontext", "baggage"),
) -> None:
    """Inject current context into a carrier.

    Convenience function for injecting the current OpenTelemetry context
    into HTTP headers or similar carrier dictionaries.

    Args:
        carrier: Dictionary to inject context into.
        propagators: Propagators to use.

    Example:
        headers = {}
        inject_context(headers)
        # headers now contains traceparent, baggage, etc.
    """
    bridge = ContextBridge(propagators=propagators)
    bridge.inject(carrier)


def extract_context(
    carrier: ContextCarrier,
    propagators: tuple[str, ...] = ("tracecontext", "baggage"),
) -> Context:
    """Extract context from a carrier.

    Convenience function for extracting OpenTelemetry context
    from HTTP headers or similar carrier dictionaries.

    Args:
        carrier: Dictionary to extract context from.
        propagators: Propagators to use.

    Returns:
        Extracted OpenTelemetry context.

    Example:
        context = extract_context(request.headers)
        with use_context(context):
            # Operations use the extracted context
            ...
    """
    bridge = ContextBridge(propagators=propagators)
    return bridge.extract(carrier)


def get_current_context() -> Context:
    """Get the current OpenTelemetry context.

    Returns:
        Current OpenTelemetry context.
    """
    _check_otel_installed()
    from opentelemetry import context as otel_context
    return otel_context.get_current()


def set_current_context(context: Context) -> Any:
    """Set the current OpenTelemetry context.

    Args:
        context: OpenTelemetry context to set as current.

    Returns:
        Token for restoring the previous context.
    """
    _check_otel_installed()
    from opentelemetry import context as otel_context
    return otel_context.attach(context)


def context_from_internal(internal_context: Any) -> Context | None:
    """Convert internal TraceContext to OpenTelemetry context.

    Args:
        internal_context: Internal TraceContext object with trace_id,
            span_id, and sampled attributes.

    Returns:
        OpenTelemetry context or None if conversion fails.
    """
    _check_otel_installed()

    try:
        from opentelemetry import trace
        from opentelemetry.trace import SpanContext, TraceFlags

        # Extract trace context fields
        trace_id = getattr(internal_context, "trace_id", None)
        span_id = getattr(internal_context, "span_id", None)
        sampled = getattr(internal_context, "sampled", True)

        if not trace_id or not span_id:
            return None

        # Convert hex strings to integers
        if isinstance(trace_id, str):
            trace_id = int(trace_id, 16)
        if isinstance(span_id, str):
            span_id = int(span_id, 16)

        # Create trace flags
        flags = TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT

        # Create span context
        span_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=True,
            trace_flags=flags,
        )

        # Create a non-recording span with the context
        span = trace.NonRecordingSpan(span_context)
        return trace.set_span_in_context(span)
    except Exception as e:
        logger.warning(f"Failed to convert internal context to OTel: {e}")
        return None


def context_to_internal(context: Context | None = None) -> dict[str, Any]:
    """Convert OpenTelemetry context to internal TraceContext format.

    Args:
        context: OpenTelemetry context. If None, uses current context.

    Returns:
        Dictionary with trace_id, span_id, sampled fields.
    """
    _check_otel_installed()

    try:
        from opentelemetry import context as otel_context
        from opentelemetry import trace

        ctx = context or otel_context.get_current()
        span = trace.get_current_span(ctx)
        span_context = span.get_span_context()

        if not span_context.is_valid:
            return {}

        return {
            "trace_id": format(span_context.trace_id, "032x"),
            "span_id": format(span_context.span_id, "016x"),
            "sampled": bool(span_context.trace_flags.sampled),
        }
    except Exception as e:
        logger.warning(f"Failed to convert OTel context to internal: {e}")
        return {}
