"""OpenTelemetry Tracer provider for truthound-orchestration.

This module provides a configurable TracerProvider that integrates
with the OpenTelemetry tracing SDK.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from common.opentelemetry.config import BatchConfig, OTelConfig, OTLPExporterConfig
from common.opentelemetry.exceptions import OTelNotInstalledError, OTelProviderError
from common.opentelemetry.providers.resource import create_resource
from common.opentelemetry.types import ExporterType, OTLPCompression, OTLPProtocol

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.trace import Tracer

__all__ = [
    "OTelTracerProvider",
    "create_tracer_provider",
    "get_global_tracer_provider",
    "set_global_tracer_provider",
]

logger = logging.getLogger(__name__)


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="tracer provider") from e


def _create_otlp_span_exporter(config: OTLPExporterConfig) -> SpanExporter:
    """Create an OTLP span exporter based on configuration.

    Args:
        config: OTLP exporter configuration.

    Returns:
        Configured OTLP span exporter.
    """
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
        # HTTP protocol
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


def _create_console_span_exporter() -> SpanExporter:
    """Create a console span exporter for debugging.

    Returns:
        Console span exporter.
    """
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter
    return ConsoleSpanExporter()


class InMemorySpanExporter:
    """In-memory span exporter for testing.

    Stores all exported spans in memory for inspection.
    """

    def __init__(self) -> None:
        """Initialize InMemorySpanExporter."""
        self._spans: list[Any] = []
        self._is_shutdown = False

    @property
    def spans(self) -> list[Any]:
        """Get all collected spans."""
        return list(self._spans)

    def clear(self) -> None:
        """Clear all collected spans."""
        self._spans.clear()

    def export(self, spans: Any) -> Any:
        """Export spans to memory.

        Args:
            spans: Spans to export.

        Returns:
            Export result.
        """
        from opentelemetry.sdk.trace.export import SpanExportResult

        if self._is_shutdown:
            return SpanExportResult.FAILURE

        self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._is_shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush (no-op for in-memory)."""
        return True


class OTelTracerProvider:
    """OpenTelemetry Tracer Provider wrapper with configuration support.

    This class wraps the OpenTelemetry TracerProvider and provides
    convenient configuration and lifecycle management.

    Example:
        config = OTelConfig().with_service_name("my-service")
        provider = OTelTracerProvider(config)

        tracer = provider.get_tracer("my-module")
        with tracer.start_as_current_span("operation"):
            # Do work
            pass

        provider.shutdown()
    """

    def __init__(
        self,
        config: OTelConfig | None = None,
        exporter: SpanExporter | None = None,
    ) -> None:
        """Initialize OTelTracerProvider.

        Args:
            config: OpenTelemetry configuration.
            exporter: Custom span exporter. If None, creates based on config.
        """
        _check_otel_installed()

        self._config = config or OTelConfig()
        self._provider: TracerProvider | None = None
        self._processor: Any = None
        self._exporter = exporter
        self._memory_exporter: InMemorySpanExporter | None = None
        self._is_shutdown = False

        if self._config.enabled and self._config.tracing_enabled:
            self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Initialize the underlying TracerProvider."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        try:
            # Create resource
            resource = create_resource(self._config.resource)

            # Create provider
            self._provider = TracerProvider(resource=resource)

            # Create or use provided exporter
            if self._exporter is None:
                if self._config.traces_exporter == ExporterType.OTLP:
                    self._exporter = _create_otlp_span_exporter(self._config.otlp)
                elif self._config.traces_exporter == ExporterType.CONSOLE:
                    self._exporter = _create_console_span_exporter()
                elif self._config.traces_exporter == ExporterType.MEMORY:
                    self._memory_exporter = InMemorySpanExporter()
                    self._exporter = self._memory_exporter
                elif self._config.traces_exporter == ExporterType.NONE:
                    # No-op: don't add any processor
                    return

            if self._exporter is not None:
                # Use BatchSpanProcessor for better performance
                # Use SimpleSpanProcessor for testing/debugging
                if self._config.traces_exporter == ExporterType.MEMORY:
                    # Use simple processor for testing
                    self._processor = SimpleSpanProcessor(self._exporter)
                else:
                    self._processor = BatchSpanProcessor(
                        self._exporter,
                        max_queue_size=self._config.batch.max_queue_size,
                        max_export_batch_size=self._config.batch.max_export_batch_size,
                        export_timeout_millis=int(
                            self._config.batch.export_timeout_seconds * 1000
                        ),
                        schedule_delay_millis=int(
                            self._config.batch.schedule_delay_seconds * 1000
                        ),
                    )

                self._provider.add_span_processor(self._processor)

            # Apply sampling if configured
            if not self._config.sampling.enabled:
                # Disable sampling
                pass  # TracerProvider uses AlwaysOnSampler by default
        except Exception as e:
            raise OTelProviderError(
                message=f"Failed to initialize TracerProvider: {e}",
                provider_type="tracer",
                original_error=e,
            ) from e

    @property
    def provider(self) -> TracerProvider | None:
        """Get the underlying TracerProvider."""
        return self._provider

    @property
    def is_enabled(self) -> bool:
        """Check if the provider is enabled."""
        return (
            self._config.enabled
            and self._config.tracing_enabled
            and self._provider is not None
        )

    def get_tracer(
        self,
        name: str,
        version: str | None = None,
        schema_url: str | None = None,
    ) -> Tracer:
        """Get a Tracer instance.

        Args:
            name: Name of the instrumentation scope.
            version: Version of the instrumentation scope.
            schema_url: Schema URL of the instrumentation scope.

        Returns:
            Tracer instance.
        """
        if self._provider is None:
            # Return a no-op tracer
            from opentelemetry.trace import NoOpTracer
            return NoOpTracer()

        return self._provider.get_tracer(
            instrumenting_module_name=name,
            instrumenting_library_version=version,
            schema_url=schema_url,
        )

    def get_collected_spans(self) -> list[Any]:
        """Get collected spans (for in-memory exporter only).

        Returns:
            List of collected spans.

        Raises:
            OTelProviderError: If not using in-memory exporter.
        """
        if self._config.traces_exporter != ExporterType.MEMORY:
            raise OTelProviderError(
                message="get_collected_spans only available with MEMORY exporter",
                provider_type="tracer",
            )

        if self._memory_exporter is None:
            return []

        return self._memory_exporter.spans

    def clear_collected_spans(self) -> None:
        """Clear collected spans (for in-memory exporter only)."""
        if self._memory_exporter is not None:
            self._memory_exporter.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all spans.

        Args:
            timeout_millis: Timeout for flush operation.

        Returns:
            True if successful.
        """
        if self._provider is None:
            return True

        try:
            return self._provider.force_flush(timeout_millis)
        except Exception as e:
            logger.warning(f"Failed to force flush spans: {e}")
            return False

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown the provider.

        Args:
            timeout_millis: Timeout for shutdown operation.

        Returns:
            True if successful.
        """
        if self._is_shutdown:
            return True

        self._is_shutdown = True

        if self._provider is None:
            return True

        try:
            return self._provider.shutdown()
        except Exception as e:
            logger.warning(f"Failed to shutdown TracerProvider: {e}")
            return False

    def __enter__(self) -> "OTelTracerProvider":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()


def create_tracer_provider(
    config: OTelConfig | None = None,
    exporter: SpanExporter | None = None,
) -> OTelTracerProvider:
    """Create a configured OTelTracerProvider.

    Factory function for creating a tracer provider with configuration.

    Args:
        config: OpenTelemetry configuration.
        exporter: Custom span exporter.

    Returns:
        Configured OTelTracerProvider.

    Example:
        provider = create_tracer_provider(
            config=OTelConfig().with_service_name("my-service")
        )
    """
    return OTelTracerProvider(config=config, exporter=exporter)


def get_global_tracer_provider() -> TracerProvider:
    """Get the global TracerProvider.

    Returns:
        Global TracerProvider instance.
    """
    _check_otel_installed()
    from opentelemetry.trace import get_tracer_provider
    return get_tracer_provider()


def set_global_tracer_provider(provider: OTelTracerProvider | TracerProvider) -> None:
    """Set the global TracerProvider.

    Args:
        provider: TracerProvider to set as global.
    """
    _check_otel_installed()
    from opentelemetry.trace import set_tracer_provider

    if isinstance(provider, OTelTracerProvider):
        if provider.provider is not None:
            set_tracer_provider(provider.provider)
    else:
        set_tracer_provider(provider)
