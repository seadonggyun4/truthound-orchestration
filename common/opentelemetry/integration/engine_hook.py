"""OpenTelemetry integration hook for data quality engines.

This module provides OTelEngineMetricsHook that implements the
EngineMetricsHook protocol and exports telemetry to OpenTelemetry.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from common.opentelemetry.config import OTelConfig
from common.opentelemetry.exceptions import OTelNotInstalledError
from common.opentelemetry.semantic.attributes import (
    CheckStatus,
    DataQualityAttributes,
    OperationStatus,
    OperationType,
    create_check_attributes,
    create_engine_attributes,
    create_learn_attributes,
    create_profile_attributes,
)

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Span, Tracer

    from common.base import CheckResult, LearnResult, ProfileResult

__all__ = [
    "OTelEngineMetricsHook",
    "create_otel_engine_hook",
]

logger = logging.getLogger(__name__)


def _check_otel_installed() -> None:
    """Check if OpenTelemetry SDK is installed."""
    try:
        import opentelemetry.metrics  # noqa: F401
        import opentelemetry.trace  # noqa: F401
    except ImportError as e:
        raise OTelNotInstalledError(feature="engine hook") from e


class OTelEngineMetricsHook:
    """OpenTelemetry metrics hook for data quality engines.

    This hook implements the EngineMetricsHook protocol and exports
    metrics and traces to OpenTelemetry for each engine operation.

    Metrics:
        - dq_engine_operations_total: Counter of operations by type and status
        - dq_engine_operation_duration_seconds: Histogram of operation durations
        - dq_engine_check_passed_total: Counter of passed checks
        - dq_engine_check_failed_total: Counter of failed checks
        - dq_engine_data_rows_total: Counter of rows processed

    Traces:
        - Creates spans for each operation (check, profile, learn)
        - Adds attributes for operation details
        - Records exceptions

    Example:
        hook = OTelEngineMetricsHook()
        instrumented = InstrumentedEngine(engine, hooks=[hook])
        result = instrumented.check(data, rules)
    """

    def __init__(
        self,
        config: OTelConfig | None = None,
        meter_name: str = "truthound.orchestration.engine",
        tracer_name: str = "truthound.orchestration.engine",
        prefix: str = "dq_engine",
    ) -> None:
        """Initialize OTelEngineMetricsHook.

        Args:
            config: OpenTelemetry configuration.
            meter_name: Name for the meter.
            tracer_name: Name for the tracer.
            prefix: Prefix for metric names.
        """
        _check_otel_installed()

        self._config = config or OTelConfig()
        self._meter_name = meter_name
        self._tracer_name = tracer_name
        self._prefix = prefix

        self._meter: Meter | None = None
        self._tracer: Tracer | None = None

        # Metrics
        self._operations_counter: Counter | None = None
        self._duration_histogram: Histogram | None = None
        self._check_passed_counter: Counter | None = None
        self._check_failed_counter: Counter | None = None
        self._data_rows_counter: Counter | None = None
        self._rules_counter: Counter | None = None

        # Active spans tracking
        self._active_spans: dict[str, tuple[Span, float]] = {}

        if self._config.enabled:
            self._initialize_instruments()

    def _initialize_instruments(self) -> None:
        """Initialize OpenTelemetry instruments."""
        from opentelemetry.metrics import get_meter_provider
        from opentelemetry.trace import get_tracer_provider

        # Get meter and tracer
        self._meter = get_meter_provider().get_meter(self._meter_name)
        self._tracer = get_tracer_provider().get_tracer(self._tracer_name)

        # Create metrics
        self._operations_counter = self._meter.create_counter(
            name=f"{self._prefix}_operations_total",
            description="Total number of data quality operations",
            unit="1",
        )

        self._duration_histogram = self._meter.create_histogram(
            name=f"{self._prefix}_operation_duration_seconds",
            description="Duration of data quality operations",
            unit="s",
        )

        self._check_passed_counter = self._meter.create_counter(
            name=f"{self._prefix}_check_passed_total",
            description="Total number of passed checks",
            unit="1",
        )

        self._check_failed_counter = self._meter.create_counter(
            name=f"{self._prefix}_check_failed_total",
            description="Total number of failed checks",
            unit="1",
        )

        self._data_rows_counter = self._meter.create_counter(
            name=f"{self._prefix}_data_rows_total",
            description="Total number of data rows processed",
            unit="1",
        )

        self._rules_counter = self._meter.create_counter(
            name=f"{self._prefix}_rules_total",
            description="Total number of rules applied",
            unit="1",
        )

    def _get_span_key(self, engine_name: str, operation: str) -> str:
        """Generate a unique key for tracking spans."""
        return f"{engine_name}:{operation}:{id(self)}"

    def _start_span(
        self,
        engine_name: str,
        operation: OperationType,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Start a span for an operation."""
        if self._tracer is None:
            return

        span_name = f"dq.{operation.value}"
        attrs = create_engine_attributes(engine_name)
        attrs[DataQualityAttributes.OPERATION_TYPE] = operation.value
        attrs[DataQualityAttributes.OPERATION_STATUS] = OperationStatus.STARTED.value

        if attributes:
            attrs.update(attributes)

        span = self._tracer.start_span(name=span_name, attributes=attrs)
        span_key = self._get_span_key(engine_name, operation.value)
        self._active_spans[span_key] = (span, time.time())

    def _end_span(
        self,
        engine_name: str,
        operation: OperationType,
        success: bool = True,
        attributes: dict[str, Any] | None = None,
        exception: Exception | None = None,
    ) -> None:
        """End a span for an operation."""
        span_key = self._get_span_key(engine_name, operation.value)
        span_data = self._active_spans.pop(span_key, None)

        if span_data is None:
            return

        span, start_time = span_data
        duration_ms = (time.time() - start_time) * 1000

        # Add final attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        span.set_attribute(
            DataQualityAttributes.OPERATION_STATUS,
            OperationStatus.COMPLETED.value if success else OperationStatus.FAILED.value,
        )
        span.set_attribute(DataQualityAttributes.OPERATION_DURATION_MS, duration_ms)

        # Record exception if present
        if exception:
            span.record_exception(exception)
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR, str(exception)))
        elif not success:
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR))
        else:
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.OK))

        span.end()

    # EngineMetricsHook Protocol implementation

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when a check operation starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of the data (rows or bytes).
            context: Additional context.
        """
        attributes: dict[str, Any] = {}
        if data_size is not None:
            attributes[DataQualityAttributes.DATASET_ROW_COUNT] = data_size

        self._start_span(engine_name, OperationType.CHECK, attributes)

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when a check operation ends.

        Args:
            engine_name: Name of the engine.
            result: Check result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        # Determine status
        status_value = getattr(result, "status", None)
        if status_value is not None:
            if hasattr(status_value, "value"):
                status_str = status_value.value
            else:
                status_str = str(status_value)
        else:
            status_str = "unknown"

        passed_count = getattr(result, "passed_count", 0)
        failed_count = getattr(result, "failed_count", 0)
        rules_count = passed_count + failed_count

        # Map to CheckStatus
        check_status = CheckStatus.PASSED
        if status_str.upper() in ("FAILED", "ERROR"):
            check_status = CheckStatus.FAILED
        elif status_str.upper() == "WARNING":
            check_status = CheckStatus.WARNING

        success = check_status in (CheckStatus.PASSED, CheckStatus.WARNING)

        # Record metrics
        labels = {
            DataQualityAttributes.ENGINE_NAME: engine_name,
            DataQualityAttributes.OPERATION_TYPE: OperationType.CHECK.value,
            DataQualityAttributes.CHECK_STATUS: check_status.value,
        }

        if self._operations_counter:
            self._operations_counter.add(1, labels)

        if self._duration_histogram:
            self._duration_histogram.record(
                duration_ms / 1000.0,  # Convert to seconds
                labels,
            )

        if self._check_passed_counter and passed_count > 0:
            self._check_passed_counter.add(
                passed_count,
                {DataQualityAttributes.ENGINE_NAME: engine_name},
            )

        if self._check_failed_counter and failed_count > 0:
            self._check_failed_counter.add(
                failed_count,
                {DataQualityAttributes.ENGINE_NAME: engine_name},
            )

        if self._rules_counter and rules_count > 0:
            self._rules_counter.add(
                rules_count,
                {DataQualityAttributes.ENGINE_NAME: engine_name},
            )

        # End span
        check_attrs = create_check_attributes(
            status=check_status,
            passed_count=passed_count,
            failed_count=failed_count,
            rules_count=rules_count,
        )
        self._end_span(engine_name, OperationType.CHECK, success, check_attrs)

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when a profile operation starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of the data.
            context: Additional context.
        """
        attributes: dict[str, Any] = {}
        if data_size is not None:
            attributes[DataQualityAttributes.DATASET_ROW_COUNT] = data_size

        self._start_span(engine_name, OperationType.PROFILE, attributes)

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when a profile operation ends.

        Args:
            engine_name: Name of the engine.
            result: Profile result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        # Extract profile info
        row_count = getattr(result, "row_count", 0)
        column_count = len(getattr(result, "columns", []))

        # Record metrics
        labels = {
            DataQualityAttributes.ENGINE_NAME: engine_name,
            DataQualityAttributes.OPERATION_TYPE: OperationType.PROFILE.value,
        }

        if self._operations_counter:
            self._operations_counter.add(1, labels)

        if self._duration_histogram:
            self._duration_histogram.record(duration_ms / 1000.0, labels)

        if self._data_rows_counter and row_count > 0:
            self._data_rows_counter.add(
                row_count,
                {DataQualityAttributes.ENGINE_NAME: engine_name},
            )

        # End span
        profile_attrs = create_profile_attributes(
            row_count=row_count,
            column_count=column_count,
        )
        self._end_span(engine_name, OperationType.PROFILE, True, profile_attrs)

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        """Called when a learn operation starts.

        Args:
            engine_name: Name of the engine.
            data_size: Size of the data.
            context: Additional context.
        """
        attributes: dict[str, Any] = {}
        if data_size is not None:
            attributes[DataQualityAttributes.DATASET_ROW_COUNT] = data_size

        self._start_span(engine_name, OperationType.LEARN, attributes)

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when a learn operation ends.

        Args:
            engine_name: Name of the engine.
            result: Learn result.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        # Extract learn info
        rules = getattr(result, "rules", [])
        rules_generated = len(rules)

        # Calculate average confidence
        confidences = [
            getattr(rule, "confidence", 1.0)
            for rule in rules
            if hasattr(rule, "confidence")
        ]
        confidence_avg = sum(confidences) / len(confidences) if confidences else 1.0

        # Record metrics
        labels = {
            DataQualityAttributes.ENGINE_NAME: engine_name,
            DataQualityAttributes.OPERATION_TYPE: OperationType.LEARN.value,
        }

        if self._operations_counter:
            self._operations_counter.add(1, labels)

        if self._duration_histogram:
            self._duration_histogram.record(duration_ms / 1000.0, labels)

        if self._rules_counter and rules_generated > 0:
            self._rules_counter.add(
                rules_generated,
                {DataQualityAttributes.ENGINE_NAME: engine_name},
            )

        # End span
        learn_attrs = create_learn_attributes(
            rules_generated=rules_generated,
            confidence_avg=confidence_avg,
        )
        self._end_span(engine_name, OperationType.LEARN, True, learn_attrs)

    def on_error(
        self,
        engine_name: str,
        operation: Any,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Called when an operation fails with an error.

        Args:
            engine_name: Name of the engine.
            operation: Operation type.
            exception: The exception that occurred.
            duration_ms: Duration in milliseconds.
            context: Additional context.
        """
        # Determine operation type
        if hasattr(operation, "value"):
            op_value = operation.value.lower()
        else:
            op_value = str(operation).lower()

        op_type = OperationType.CHECK
        if "profile" in op_value:
            op_type = OperationType.PROFILE
        elif "learn" in op_value:
            op_type = OperationType.LEARN

        # Record metrics
        labels = {
            DataQualityAttributes.ENGINE_NAME: engine_name,
            DataQualityAttributes.OPERATION_TYPE: op_type.value,
            DataQualityAttributes.ERROR_TYPE: type(exception).__name__,
        }

        if self._operations_counter:
            self._operations_counter.add(1, labels)

        if self._duration_histogram:
            self._duration_histogram.record(duration_ms / 1000.0, labels)

        # End span with error
        error_attrs = {
            DataQualityAttributes.ERROR_TYPE: type(exception).__name__,
            DataQualityAttributes.ERROR_MESSAGE: str(exception),
        }
        self._end_span(engine_name, op_type, False, error_attrs, exception)


def create_otel_engine_hook(
    config: OTelConfig | None = None,
    meter_name: str = "truthound.orchestration.engine",
    tracer_name: str = "truthound.orchestration.engine",
    prefix: str = "dq_engine",
) -> OTelEngineMetricsHook:
    """Create an OpenTelemetry engine metrics hook.

    Factory function for creating an OTelEngineMetricsHook.

    Args:
        config: OpenTelemetry configuration.
        meter_name: Name for the meter.
        tracer_name: Name for the tracer.
        prefix: Prefix for metric names.

    Returns:
        Configured OTelEngineMetricsHook.

    Example:
        hook = create_otel_engine_hook()
        instrumented = InstrumentedEngine(engine, hooks=[hook])
    """
    return OTelEngineMetricsHook(
        config=config,
        meter_name=meter_name,
        tracer_name=tracer_name,
        prefix=prefix,
    )
