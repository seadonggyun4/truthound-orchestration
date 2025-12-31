"""Bridge for converting internal metrics to OpenTelemetry format.

This module provides the MetricBridgeExporter that implements the
internal MetricExporter protocol and bridges metrics to OpenTelemetry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from common.opentelemetry.exceptions import OTelBridgeError, OTelNotInstalledError

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter, MeterProvider

__all__ = [
    "MetricBridgeExporter",
    "MetricBridge",
    "create_metric_bridge",
]

logger = logging.getLogger(__name__)


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.metrics  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="metrics bridge") from e


@runtime_checkable
class InternalMetricExporter(Protocol):
    """Protocol matching the internal MetricExporter interface."""

    def export(self, metrics: list[Any]) -> None:
        """Export metrics."""
        ...


class MetricBridge:
    """Bridge for converting internal metric types to OpenTelemetry instruments.

    This class maintains a mapping of internal metric names to OTel instruments
    and handles the conversion of metric data.
    """

    def __init__(
        self,
        meter: Meter,
        prefix: str = "",
        default_labels: dict[str, str] | None = None,
    ) -> None:
        """Initialize MetricBridge.

        Args:
            meter: OpenTelemetry Meter instance.
            prefix: Prefix for all metric names.
            default_labels: Default labels to add to all metrics.
        """
        self._meter = meter
        self._prefix = prefix
        self._default_labels = default_labels or {}
        self._counters: dict[str, Any] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    def _get_metric_name(self, name: str) -> str:
        """Get prefixed metric name."""
        if self._prefix:
            return f"{self._prefix}_{name}"
        return name

    def _merge_labels(self, labels: dict[str, str] | None) -> dict[str, str]:
        """Merge default labels with provided labels."""
        merged = dict(self._default_labels)
        if labels:
            merged.update(labels)
        return merged

    def record_counter(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """Record a counter value.

        Args:
            name: Counter name.
            value: Value to add (must be non-negative).
            labels: Additional labels.
            description: Metric description.
        """
        metric_name = self._get_metric_name(name)
        if metric_name not in self._counters:
            self._counters[metric_name] = self._meter.create_counter(
                name=metric_name,
                description=description,
            )
        counter = self._counters[metric_name]
        counter.add(value, attributes=self._merge_labels(labels))

    def record_gauge(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """Record a gauge value.

        Note: OpenTelemetry uses UpDownCounter for gauge-like behavior.

        Args:
            name: Gauge name.
            value: Current value.
            labels: Additional labels.
            description: Metric description.
        """
        metric_name = self._get_metric_name(name)
        if metric_name not in self._gauges:
            # Use observable gauge with callback for true gauge semantics
            # For simplicity, we use UpDownCounter here
            self._gauges[metric_name] = self._meter.create_up_down_counter(
                name=metric_name,
                description=description,
            )
        gauge = self._gauges[metric_name]
        # For gauge, we track the delta. In production, use observable gauge.
        gauge.add(value, attributes=self._merge_labels(labels))

    def record_histogram(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        """Record a histogram value.

        Args:
            name: Histogram name.
            value: Value to record.
            labels: Additional labels.
            description: Metric description.
        """
        metric_name = self._get_metric_name(name)
        if metric_name not in self._histograms:
            self._histograms[metric_name] = self._meter.create_histogram(
                name=metric_name,
                description=description,
            )
        histogram = self._histograms[metric_name]
        histogram.record(value, attributes=self._merge_labels(labels))


class MetricBridgeExporter:
    """Exporter that bridges internal metrics to OpenTelemetry.

    This class implements the internal MetricExporter protocol and
    forwards metrics to OpenTelemetry instruments.

    Example:
        from opentelemetry.sdk.metrics import MeterProvider
        provider = MeterProvider()
        exporter = MetricBridgeExporter(provider)

        # Use with internal metrics registry
        registry.add_exporter(exporter)
    """

    def __init__(
        self,
        meter_provider: MeterProvider | None = None,
        meter_name: str = "truthound.orchestration",
        prefix: str = "dq",
        default_labels: dict[str, str] | None = None,
    ) -> None:
        """Initialize MetricBridgeExporter.

        Args:
            meter_provider: OpenTelemetry MeterProvider. If None, uses global.
            meter_name: Name for the meter.
            prefix: Prefix for metric names.
            default_labels: Default labels for all metrics.
        """
        _check_otel_installed()

        from opentelemetry.metrics import get_meter_provider

        self._provider = meter_provider or get_meter_provider()
        self._meter = self._provider.get_meter(meter_name)
        self._bridge = MetricBridge(
            meter=self._meter,
            prefix=prefix,
            default_labels=default_labels,
        )
        self._exported_count = 0

    @property
    def exported_count(self) -> int:
        """Number of metrics exported."""
        return self._exported_count

    def export(self, metrics: list[Any]) -> None:
        """Export internal metrics to OpenTelemetry.

        Converts internal MetricData objects to OpenTelemetry format.

        Args:
            metrics: List of internal MetricData objects.
        """
        for metric in metrics:
            try:
                self._export_metric(metric)
                self._exported_count += 1
            except Exception as e:
                logger.warning(f"Failed to export metric: {e}")

    def _export_metric(self, metric: Any) -> None:
        """Export a single metric.

        Args:
            metric: Internal MetricData object.
        """
        # Handle different internal metric types
        metric_type = getattr(metric, "metric_type", None)
        name = getattr(metric, "name", "unknown")
        value = getattr(metric, "value", 0)
        labels = getattr(metric, "labels", None)
        description = getattr(metric, "description", "")

        # Convert labels to dict if needed
        if labels and not isinstance(labels, dict):
            labels = dict(labels)

        try:
            if metric_type is not None:
                # Use the metric type enum value
                type_name = metric_type.value if hasattr(metric_type, "value") else str(metric_type)
                type_name = type_name.upper()
            else:
                # Default to counter
                type_name = "COUNTER"

            if type_name == "COUNTER":
                self._bridge.record_counter(name, value, labels, description)
            elif type_name == "GAUGE":
                self._bridge.record_gauge(name, value, labels, description)
            elif type_name in ("HISTOGRAM", "SUMMARY"):
                self._bridge.record_histogram(name, value, labels, description)
            else:
                # Default to counter for unknown types
                self._bridge.record_counter(name, value, labels, description)
        except Exception as e:
            raise OTelBridgeError(
                message=f"Failed to convert metric '{name}' to OpenTelemetry format",
                bridge_type="metrics",
                source_type=str(metric_type) if metric_type else "unknown",
                target_type="otel_instrument",
                original_error=e,
            ) from e

    def flush(self) -> None:
        """Flush any buffered metrics.

        Note: OpenTelemetry SDK handles flushing automatically.
        This method exists for compatibility with the internal protocol.
        """
        # OpenTelemetry SDK handles flushing automatically
        pass

    def shutdown(self) -> None:
        """Shutdown the exporter.

        Note: The MeterProvider should be shutdown separately.
        """
        # MeterProvider should be shutdown by the caller
        pass


def create_metric_bridge(
    meter_provider: MeterProvider | None = None,
    meter_name: str = "truthound.orchestration",
    prefix: str = "dq",
    default_labels: dict[str, str] | None = None,
) -> MetricBridgeExporter:
    """Create a metric bridge exporter.

    Factory function for creating a MetricBridgeExporter with
    convenient defaults.

    Args:
        meter_provider: OpenTelemetry MeterProvider. If None, uses global.
        meter_name: Name for the meter.
        prefix: Prefix for metric names.
        default_labels: Default labels for all metrics.

    Returns:
        Configured MetricBridgeExporter instance.

    Example:
        exporter = create_metric_bridge(prefix="myapp")
        registry.add_exporter(exporter)
    """
    return MetricBridgeExporter(
        meter_provider=meter_provider,
        meter_name=meter_name,
        prefix=prefix,
        default_labels=default_labels,
    )
