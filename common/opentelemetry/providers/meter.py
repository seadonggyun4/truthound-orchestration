"""OpenTelemetry Meter provider for truthound-orchestration.

This module provides a configurable MeterProvider that integrates
with the OpenTelemetry metrics SDK.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from common.opentelemetry.config import BatchConfig, OTelConfig, OTLPExporterConfig
from common.opentelemetry.exceptions import OTelNotInstalledError, OTelProviderError
from common.opentelemetry.providers.resource import create_resource
from common.opentelemetry.types import ExporterType, OTLPCompression, OTLPProtocol

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import MetricExporter

__all__ = [
    "OTelMeterProvider",
    "create_meter_provider",
    "get_global_meter_provider",
    "set_global_meter_provider",
]

logger = logging.getLogger(__name__)


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.sdk.metrics  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="meter provider") from e


def _create_otlp_metric_exporter(config: OTLPExporterConfig) -> MetricExporter:
    """Create an OTLP metric exporter based on configuration.

    Args:
        config: OTLP exporter configuration.

    Returns:
        Configured OTLP metric exporter.
    """
    # Determine which exporter to use based on protocol
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
        # HTTP protocol
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


def _create_console_metric_exporter() -> MetricExporter:
    """Create a console metric exporter for debugging.

    Returns:
        Console metric exporter.
    """
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
    return ConsoleMetricExporter()


def _create_in_memory_metric_exporter() -> MetricExporter:
    """Create an in-memory metric exporter for testing.

    Returns:
        In-memory metric exporter.
    """
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    # Note: InMemoryMetricReader is a reader, not exporter
    # For testing, we'll use a custom approach or the reader directly
    # This is handled specially in the provider creation
    raise NotImplementedError(
        "In-memory exporter should use InMemoryMetricReader directly"
    )


class OTelMeterProvider:
    """OpenTelemetry Meter Provider wrapper with configuration support.

    This class wraps the OpenTelemetry MeterProvider and provides
    convenient configuration and lifecycle management.

    Example:
        config = OTelConfig().with_service_name("my-service")
        provider = OTelMeterProvider(config)

        meter = provider.get_meter("my-module")
        counter = meter.create_counter("requests")
        counter.add(1)

        provider.shutdown()
    """

    def __init__(
        self,
        config: OTelConfig | None = None,
        exporter: MetricExporter | None = None,
    ) -> None:
        """Initialize OTelMeterProvider.

        Args:
            config: OpenTelemetry configuration.
            exporter: Custom metric exporter. If None, creates based on config.
        """
        _check_otel_installed()

        self._config = config or OTelConfig()
        self._provider: MeterProvider | None = None
        self._reader: Any = None
        self._exporter = exporter
        self._is_shutdown = False

        if self._config.enabled and self._config.metrics_enabled:
            self._initialize_provider()

    def _initialize_provider(self) -> None:
        """Initialize the underlying MeterProvider."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        try:
            # Create resource
            resource = create_resource(self._config.resource)

            # Create or use provided exporter
            if self._exporter is None:
                if self._config.metrics_exporter == ExporterType.OTLP:
                    self._exporter = _create_otlp_metric_exporter(self._config.otlp)
                elif self._config.metrics_exporter == ExporterType.CONSOLE:
                    self._exporter = _create_console_metric_exporter()
                elif self._config.metrics_exporter == ExporterType.MEMORY:
                    # Use InMemoryMetricReader for testing
                    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
                    self._reader = InMemoryMetricReader()
                    self._provider = MeterProvider(
                        resource=resource,
                        metric_readers=[self._reader],
                    )
                    return
                elif self._config.metrics_exporter == ExporterType.NONE:
                    # No-op: create provider without readers
                    self._provider = MeterProvider(resource=resource)
                    return

            if self._exporter is not None:
                # Create periodic reader with batch config
                self._reader = PeriodicExportingMetricReader(
                    exporter=self._exporter,
                    export_interval_millis=int(
                        self._config.batch.schedule_delay_seconds * 1000
                    ),
                    export_timeout_millis=int(
                        self._config.batch.export_timeout_seconds * 1000
                    ),
                )

                self._provider = MeterProvider(
                    resource=resource,
                    metric_readers=[self._reader],
                )
        except Exception as e:
            raise OTelProviderError(
                message=f"Failed to initialize MeterProvider: {e}",
                provider_type="meter",
                original_error=e,
            ) from e

    @property
    def provider(self) -> MeterProvider | None:
        """Get the underlying MeterProvider."""
        return self._provider

    @property
    def is_enabled(self) -> bool:
        """Check if the provider is enabled."""
        return (
            self._config.enabled
            and self._config.metrics_enabled
            and self._provider is not None
        )

    def get_meter(
        self,
        name: str,
        version: str | None = None,
        schema_url: str | None = None,
    ) -> Meter:
        """Get a Meter instance.

        Args:
            name: Name of the instrumentation scope.
            version: Version of the instrumentation scope.
            schema_url: Schema URL of the instrumentation scope.

        Returns:
            Meter instance.

        Raises:
            OTelProviderError: If provider is not initialized.
        """
        if self._provider is None:
            # Return a no-op meter
            from opentelemetry.metrics import NoOpMeter
            return NoOpMeter(name)

        return self._provider.get_meter(
            name=name,
            version=version,
            schema_url=schema_url,
        )

    def get_collected_metrics(self) -> list[Any]:
        """Get collected metrics (for in-memory exporter only).

        Returns:
            List of collected metric data.

        Raises:
            OTelProviderError: If not using in-memory exporter.
        """
        if self._config.metrics_exporter != ExporterType.MEMORY:
            raise OTelProviderError(
                message="get_collected_metrics only available with MEMORY exporter",
                provider_type="meter",
            )

        if self._reader is None:
            return []

        # For InMemoryMetricReader, we need to get the data
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        if isinstance(self._reader, InMemoryMetricReader):
            return list(self._reader.get_metrics_data().resource_metrics)
        return []

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all metrics.

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
            logger.warning(f"Failed to force flush metrics: {e}")
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
            return self._provider.shutdown(timeout_millis)
        except Exception as e:
            logger.warning(f"Failed to shutdown MeterProvider: {e}")
            return False

    def __enter__(self) -> "OTelMeterProvider":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.shutdown()


def create_meter_provider(
    config: OTelConfig | None = None,
    exporter: MetricExporter | None = None,
) -> OTelMeterProvider:
    """Create a configured OTelMeterProvider.

    Factory function for creating a meter provider with configuration.

    Args:
        config: OpenTelemetry configuration.
        exporter: Custom metric exporter.

    Returns:
        Configured OTelMeterProvider.

    Example:
        provider = create_meter_provider(
            config=OTelConfig().with_service_name("my-service")
        )
    """
    return OTelMeterProvider(config=config, exporter=exporter)


def get_global_meter_provider() -> MeterProvider:
    """Get the global MeterProvider.

    Returns:
        Global MeterProvider instance.
    """
    _check_otel_installed()
    from opentelemetry.metrics import get_meter_provider
    return get_meter_provider()


def set_global_meter_provider(provider: OTelMeterProvider | MeterProvider) -> None:
    """Set the global MeterProvider.

    Args:
        provider: MeterProvider to set as global.
    """
    _check_otel_installed()
    from opentelemetry.metrics import set_meter_provider

    if isinstance(provider, OTelMeterProvider):
        if provider.provider is not None:
            set_meter_provider(provider.provider)
    else:
        set_meter_provider(provider)
