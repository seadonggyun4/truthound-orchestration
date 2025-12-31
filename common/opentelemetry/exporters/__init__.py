"""OpenTelemetry exporters for truthound-orchestration.

This module provides various exporter implementations for sending
telemetry data to different backends.

Note: The actual OTLP, Console, and Memory exporters are provided
by the OpenTelemetry SDK. This module provides convenient wrappers
and factory functions for creating them with our configuration.
"""

from common.opentelemetry.config import OTelConfig, OTLPExporterConfig
from common.opentelemetry.exceptions import OTelNotInstalledError
from common.opentelemetry.types import ExporterType, OTLPCompression, OTLPProtocol

__all__ = [
    "create_metric_exporter",
    "create_span_exporter",
    "ExporterType",
    "OTLPProtocol",
    "OTLPCompression",
]


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.sdk  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="exporters") from e


def create_metric_exporter(
    exporter_type: ExporterType = ExporterType.OTLP,
    config: OTLPExporterConfig | None = None,
):
    """Create a metric exporter based on type.

    Args:
        exporter_type: Type of exporter to create.
        config: OTLP configuration (only used for OTLP type).

    Returns:
        Configured metric exporter.

    Example:
        exporter = create_metric_exporter(ExporterType.CONSOLE)
    """
    _check_otel_installed()

    config = config or OTLPExporterConfig()

    if exporter_type == ExporterType.CONSOLE:
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
        return ConsoleMetricExporter()

    elif exporter_type == ExporterType.MEMORY:
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        return InMemoryMetricReader()

    elif exporter_type == ExporterType.OTLP:
        if config.protocol == OTLPProtocol.GRPC:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )
            except ImportError as e:
                raise OTelNotInstalledError(
                    feature="OTLP gRPC metric exporter"
                ) from e

            compression = None
            if config.compression == OTLPCompression.GZIP:
                from grpc import Compression
                compression = Compression.Gzip

            return OTLPMetricExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
                headers=tuple(config.headers.items()) if config.headers else None,
                timeout=int(config.timeout_seconds),
                compression=compression,
            )
        else:
            try:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )
            except ImportError as e:
                raise OTelNotInstalledError(
                    feature="OTLP HTTP metric exporter"
                ) from e

            compression = None
            if config.compression == OTLPCompression.GZIP:
                compression = "gzip"

            return OTLPMetricExporter(
                endpoint=config.endpoint,
                headers=config.headers,
                timeout=int(config.timeout_seconds),
                compression=compression,
            )

    elif exporter_type == ExporterType.NONE:
        # Return None for no-op
        return None

    else:
        raise ValueError(f"Unknown exporter type: {exporter_type}")


def create_span_exporter(
    exporter_type: ExporterType = ExporterType.OTLP,
    config: OTLPExporterConfig | None = None,
):
    """Create a span exporter based on type.

    Args:
        exporter_type: Type of exporter to create.
        config: OTLP configuration (only used for OTLP type).

    Returns:
        Configured span exporter.

    Example:
        exporter = create_span_exporter(ExporterType.CONSOLE)
    """
    _check_otel_installed()

    config = config or OTLPExporterConfig()

    if exporter_type == ExporterType.CONSOLE:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        return ConsoleSpanExporter()

    elif exporter_type == ExporterType.MEMORY:
        from common.opentelemetry.providers.tracer import InMemorySpanExporter
        return InMemorySpanExporter()

    elif exporter_type == ExporterType.OTLP:
        if config.protocol == OTLPProtocol.GRPC:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError as e:
                raise OTelNotInstalledError(
                    feature="OTLP gRPC trace exporter"
                ) from e

            compression = None
            if config.compression == OTLPCompression.GZIP:
                from grpc import Compression
                compression = Compression.Gzip

            return OTLPSpanExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
                headers=tuple(config.headers.items()) if config.headers else None,
                timeout=int(config.timeout_seconds),
                compression=compression,
            )
        else:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError as e:
                raise OTelNotInstalledError(
                    feature="OTLP HTTP trace exporter"
                ) from e

            compression = None
            if config.compression == OTLPCompression.GZIP:
                compression = "gzip"

            return OTLPSpanExporter(
                endpoint=config.endpoint,
                headers=config.headers,
                timeout=int(config.timeout_seconds),
                compression=compression,
            )

    elif exporter_type == ExporterType.NONE:
        # Return None for no-op
        return None

    else:
        raise ValueError(f"Unknown exporter type: {exporter_type}")
