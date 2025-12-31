"""Callbacks for Data Quality SLA and Alert Handling.

This module provides callback implementations for handling SLA violations
and quality alerts in Apache Airflow data quality operations.

Example:
    >>> from truthound_airflow.sla import (
    ...     DataQualitySLACallback,
    ...     QualityAlertCallback,
    ...     CallbackChain,
    ... )
    >>>
    >>> sla_callback = DataQualitySLACallback(config=sla_config)
    >>> alert_callback = QualityAlertCallback(
    ...     on_failure=lambda v: send_slack(v),
    ... )
    >>>
    >>> chain = CallbackChain([sla_callback, alert_callback])
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from truthound_airflow.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)
from truthound_airflow.sla.monitor import SLAMonitor

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from collections.abc import Sequence


logger = logging.getLogger(__name__)


# =============================================================================
# Callback Protocol
# =============================================================================


@runtime_checkable
class DataQualityCallback(Protocol):
    """Protocol for data quality callbacks.

    Implement this protocol to create custom callbacks for
    handling data quality events.

    Example:
        >>> class CustomCallback:
        ...     def on_success(self, result, context):
        ...         print("Quality check passed!")
        ...
        ...     def on_failure(self, result, violations, context):
        ...         print(f"Quality check failed: {violations}")
        ...
        ...     def on_sla_violation(self, violations, context):
        ...         for v in violations:
        ...             alert(v.message)
    """

    def on_success(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> None:
        """Called when quality check succeeds.

        Args:
            result: The check result dictionary.
            context: Airflow execution context.
        """
        ...

    def on_failure(
        self,
        result: dict[str, Any],
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Called when quality check fails.

        Args:
            result: The check result dictionary.
            violations: List of SLA violations.
            context: Airflow execution context.
        """
        ...

    def on_sla_violation(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Called when SLA violations are detected.

        Args:
            violations: List of detected violations.
            context: Airflow execution context.
        """
        ...


# =============================================================================
# Base Callback
# =============================================================================


class BaseDataQualityCallback(ABC):
    """Abstract base class for data quality callbacks.

    Provides common functionality for callback implementations.

    Attributes:
        name: Optional callback name for logging.
        enabled: Whether the callback is enabled.
    """

    def __init__(
        self,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize callback.

        Args:
            name: Optional callback name.
            enabled: Whether callback is enabled.
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled

    def on_success(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> None:
        """Called when quality check succeeds.

        Override to implement success handling.

        Args:
            result: The check result dictionary.
            context: Airflow execution context.
        """
        pass

    def on_failure(
        self,
        result: dict[str, Any],
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Called when quality check fails.

        Override to implement failure handling.

        Args:
            result: The check result dictionary.
            violations: List of SLA violations.
            context: Airflow execution context.
        """
        pass

    def on_sla_violation(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Called when SLA violations are detected.

        Override to implement violation handling.

        Args:
            violations: List of detected violations.
            context: Airflow execution context.
        """
        pass


# =============================================================================
# SLA Callback
# =============================================================================


class DataQualitySLACallback(BaseDataQualityCallback):
    """Callback for SLA monitoring and violation handling.

    This callback integrates with the SLA monitor to detect
    and handle SLA violations during data quality operations.

    Parameters
    ----------
    config : SLAConfig
        SLA configuration to monitor against.

    on_violation : Callable | None
        Callback function for handling violations.

    raise_on_violation : bool
        Whether to raise exception on violation.

    Examples
    --------
    Basic usage:

    >>> callback = DataQualitySLACallback(
    ...     config=SLAConfig(max_failure_rate=0.05),
    ... )

    With custom violation handler:

    >>> def handle_violation(violations, context):
    ...     for v in violations:
    ...         send_alert(v.message)
    >>>
    >>> callback = DataQualitySLACallback(
    ...     config=SLAConfig(max_failure_rate=0.05),
    ...     on_violation=handle_violation,
    ... )

    With DAG integration:

    >>> with DAG(...) as dag:
    ...     check = DataQualityCheckOperator(
    ...         task_id="check",
    ...         rules=[...],
    ...         data_path="...",
    ...         on_success_callback=callback.on_success,
    ...         on_failure_callback=callback.on_failure,
    ...     )
    """

    def __init__(
        self,
        config: SLAConfig,
        on_violation: Callable[[list[SLAViolation], Context], None] | None = None,
        raise_on_violation: bool = False,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize SLA callback.

        Args:
            config: SLA configuration.
            on_violation: Custom violation handler.
            raise_on_violation: Whether to raise on violation.
            name: Optional callback name.
            enabled: Whether callback is enabled.
        """
        super().__init__(name=name, enabled=enabled)
        self.config = config
        self.monitor = SLAMonitor(config=config, name=name)
        self._on_violation = on_violation
        self.raise_on_violation = raise_on_violation

    def on_success(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> None:
        """Handle successful quality check.

        Args:
            result: The check result dictionary.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        metrics = self._extract_metrics(result, context)
        violations = self.monitor.check(metrics, context)

        if violations:
            self._handle_violations(violations, context)

        logger.info(
            f"[{self.name}] Quality check succeeded - "
            f"pass_rate={metrics.pass_rate:.2%}, "
            f"duration={metrics.execution_time_seconds:.2f}s"
        )

    def on_failure(
        self,
        result: dict[str, Any],
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Handle failed quality check.

        Args:
            result: The check result dictionary.
            violations: Any pre-detected violations.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        metrics = self._extract_metrics(result, context)
        sla_violations = self.monitor.check(metrics, context)

        all_violations = list(violations) + sla_violations

        if all_violations:
            self._handle_violations(all_violations, context)

        logger.warning(
            f"[{self.name}] Quality check failed - "
            f"failure_rate={metrics.failure_rate:.2%}, "
            f"violations={len(all_violations)}"
        )

    def on_sla_violation(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Handle SLA violations.

        Args:
            violations: List of detected violations.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        self._handle_violations(violations, context)

    def _extract_metrics(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> SLAMetrics:
        """Extract metrics from result and context."""
        task_id = context.get("task", {}).task_id if context.get("task") else None
        dag_id = context.get("dag", {}).dag_id if context.get("dag") else None
        run_id = context.get("run_id")

        return SLAMetrics.from_check_result(
            result,
            task_id=task_id,
            dag_id=dag_id,
            run_id=run_id,
        )

    def _handle_violations(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Handle detected violations."""
        for violation in violations:
            logger.warning(
                f"[{self.name}] SLA Violation: {violation.violation_type.value} - "
                f"{violation.message}"
            )

        if self._on_violation:
            self._on_violation(violations, context)

        if self.raise_on_violation:
            from airflow.exceptions import AirflowException

            messages = [v.message for v in violations]
            raise AirflowException(f"SLA violations detected: {'; '.join(messages)}")


# =============================================================================
# Alert Callback
# =============================================================================


@dataclass
class AlertHandlers:
    """Container for alert handler functions.

    Attributes:
        on_info: Handler for info-level alerts.
        on_warning: Handler for warning-level alerts.
        on_error: Handler for error-level alerts.
        on_critical: Handler for critical-level alerts.
    """

    on_info: Callable[[SLAViolation, Context], None] | None = None
    on_warning: Callable[[SLAViolation, Context], None] | None = None
    on_error: Callable[[SLAViolation, Context], None] | None = None
    on_critical: Callable[[SLAViolation, Context], None] | None = None

    def get_handler(
        self,
        level: AlertLevel,
    ) -> Callable[[SLAViolation, Context], None] | None:
        """Get handler for alert level."""
        handlers = {
            AlertLevel.INFO: self.on_info,
            AlertLevel.WARNING: self.on_warning,
            AlertLevel.ERROR: self.on_error,
            AlertLevel.CRITICAL: self.on_critical,
        }
        return handlers.get(level)


class QualityAlertCallback(BaseDataQualityCallback):
    """Callback for sending quality alerts.

    This callback routes alerts to appropriate handlers based
    on the alert level of each violation.

    Parameters
    ----------
    handlers : AlertHandlers | None
        Handlers for different alert levels.

    on_failure : Callable | None
        General failure handler (called for all failures).

    on_success : Callable | None
        Success handler.

    default_handler : Callable | None
        Default handler if no specific handler found.

    Examples
    --------
    Basic usage:

    >>> callback = QualityAlertCallback(
    ...     on_failure=lambda v, ctx: send_slack(v.message),
    ... )

    With level-specific handlers:

    >>> callback = QualityAlertCallback(
    ...     handlers=AlertHandlers(
    ...         on_warning=lambda v, ctx: log.warning(v.message),
    ...         on_error=lambda v, ctx: send_email(v.message),
    ...         on_critical=lambda v, ctx: page_oncall(v.message),
    ...     ),
    ... )
    """

    def __init__(
        self,
        handlers: AlertHandlers | None = None,
        on_failure: Callable[[SLAViolation, Context], None] | None = None,
        on_success_handler: Callable[[dict[str, Any], Context], None] | None = None,
        default_handler: Callable[[SLAViolation, Context], None] | None = None,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize alert callback.

        Args:
            handlers: Level-specific handlers.
            on_failure: General failure handler.
            on_success_handler: Success handler.
            default_handler: Default handler.
            name: Optional callback name.
            enabled: Whether callback is enabled.
        """
        super().__init__(name=name, enabled=enabled)
        self.handlers = handlers or AlertHandlers()
        self._on_failure = on_failure
        self._on_success_handler = on_success_handler
        self._default_handler = default_handler

    def on_success(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> None:
        """Handle successful quality check.

        Args:
            result: The check result dictionary.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        if self._on_success_handler:
            self._on_success_handler(result, context)

    def on_failure(
        self,
        result: dict[str, Any],
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Handle failed quality check.

        Args:
            result: The check result dictionary.
            violations: List of SLA violations.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        for violation in violations:
            self._route_alert(violation, context)

    def on_sla_violation(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Handle SLA violations.

        Args:
            violations: List of detected violations.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        for violation in violations:
            self._route_alert(violation, context)

    def _route_alert(
        self,
        violation: SLAViolation,
        context: Context,
    ) -> None:
        """Route alert to appropriate handler."""
        # Try level-specific handler
        handler = self.handlers.get_handler(violation.alert_level)
        if handler:
            handler(violation, context)
            return

        # Try general failure handler
        if self._on_failure:
            self._on_failure(violation, context)
            return

        # Try default handler
        if self._default_handler:
            self._default_handler(violation, context)
            return

        # Log if no handler
        logger.warning(
            f"[{self.name}] Unhandled alert: "
            f"{violation.alert_level.value} - {violation.message}"
        )


# =============================================================================
# Callback Chain
# =============================================================================


class CallbackChain(BaseDataQualityCallback):
    """Chain multiple callbacks together.

    The callback chain executes multiple callbacks in sequence,
    allowing composition of callback behaviors.

    Parameters
    ----------
    callbacks : Sequence[BaseDataQualityCallback]
        Callbacks to chain together.

    stop_on_error : bool
        Whether to stop chain on callback error.

    Examples
    --------
    >>> chain = CallbackChain([
    ...     DataQualitySLACallback(config=sla_config),
    ...     QualityAlertCallback(on_failure=send_slack),
    ...     LoggingCallback(),
    ... ])
    """

    def __init__(
        self,
        callbacks: Sequence[BaseDataQualityCallback],
        stop_on_error: bool = False,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize callback chain.

        Args:
            callbacks: Callbacks to chain.
            stop_on_error: Stop on callback error.
            name: Optional chain name.
            enabled: Whether chain is enabled.
        """
        super().__init__(name=name, enabled=enabled)
        self.callbacks = list(callbacks)
        self.stop_on_error = stop_on_error

    def on_success(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> None:
        """Execute success handlers for all callbacks.

        Args:
            result: The check result dictionary.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        for callback in self.callbacks:
            if not callback.enabled:
                continue

            try:
                callback.on_success(result, context)
            except Exception as e:
                logger.exception(
                    f"[{self.name}] Error in callback "
                    f"{callback.name}.on_success: {e}"
                )
                if self.stop_on_error:
                    raise

    def on_failure(
        self,
        result: dict[str, Any],
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Execute failure handlers for all callbacks.

        Args:
            result: The check result dictionary.
            violations: List of SLA violations.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        for callback in self.callbacks:
            if not callback.enabled:
                continue

            try:
                callback.on_failure(result, violations, context)
            except Exception as e:
                logger.exception(
                    f"[{self.name}] Error in callback "
                    f"{callback.name}.on_failure: {e}"
                )
                if self.stop_on_error:
                    raise

    def on_sla_violation(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Execute violation handlers for all callbacks.

        Args:
            violations: List of detected violations.
            context: Airflow execution context.
        """
        if not self.enabled:
            return

        for callback in self.callbacks:
            if not callback.enabled:
                continue

            try:
                callback.on_sla_violation(violations, context)
            except Exception as e:
                logger.exception(
                    f"[{self.name}] Error in callback "
                    f"{callback.name}.on_sla_violation: {e}"
                )
                if self.stop_on_error:
                    raise

    def add(self, callback: BaseDataQualityCallback) -> None:
        """Add callback to chain.

        Args:
            callback: Callback to add.
        """
        self.callbacks.append(callback)

    def remove(self, callback: BaseDataQualityCallback) -> bool:
        """Remove callback from chain.

        Args:
            callback: Callback to remove.

        Returns:
            bool: True if removed, False if not found.
        """
        try:
            self.callbacks.remove(callback)
            return True
        except ValueError:
            return False


# =============================================================================
# Logging Callback
# =============================================================================


class LoggingCallback(BaseDataQualityCallback):
    """Simple callback that logs all events.

    Useful for debugging and audit trails.

    Examples
    --------
    >>> callback = LoggingCallback(log_level=logging.INFO)
    """

    def __init__(
        self,
        log_level: int = logging.INFO,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize logging callback.

        Args:
            log_level: Logging level.
            name: Optional callback name.
            enabled: Whether callback is enabled.
        """
        super().__init__(name=name, enabled=enabled)
        self.log_level = log_level

    def on_success(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> None:
        """Log success event."""
        if not self.enabled:
            return

        task_id = context.get("task", {}).task_id if context.get("task") else "unknown"
        logger.log(
            self.log_level,
            f"[{self.name}] Quality check SUCCESS - "
            f"task={task_id}, "
            f"passed={result.get('passed_count', 0)}, "
            f"failed={result.get('failed_count', 0)}, "
            f"duration={result.get('execution_time_ms', 0):.2f}ms",
        )

    def on_failure(
        self,
        result: dict[str, Any],
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Log failure event."""
        if not self.enabled:
            return

        task_id = context.get("task", {}).task_id if context.get("task") else "unknown"
        logger.log(
            self.log_level,
            f"[{self.name}] Quality check FAILURE - "
            f"task={task_id}, "
            f"passed={result.get('passed_count', 0)}, "
            f"failed={result.get('failed_count', 0)}, "
            f"violations={len(violations)}",
        )

        for violation in violations:
            logger.log(
                self.log_level,
                f"[{self.name}]   - {violation.violation_type.value}: {violation.message}",
            )

    def on_sla_violation(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Log SLA violations."""
        if not self.enabled:
            return

        for violation in violations:
            logger.log(
                self.log_level,
                f"[{self.name}] SLA VIOLATION - "
                f"type={violation.violation_type.value}, "
                f"level={violation.alert_level.value}, "
                f"message={violation.message}",
            )


# =============================================================================
# Metrics Callback
# =============================================================================


class MetricsCallback(BaseDataQualityCallback):
    """Callback that collects metrics.

    Collects statistics about quality check executions
    for monitoring and observability.

    Attributes:
        total_success: Total successful checks.
        total_failure: Total failed checks.
        total_violations: Total SLA violations.
        success_rate: Overall success rate.

    Examples
    --------
    >>> callback = MetricsCallback()
    >>> # After some checks...
    >>> print(f"Success rate: {callback.success_rate:.2%}")
    """

    def __init__(
        self,
        name: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize metrics callback.

        Args:
            name: Optional callback name.
            enabled: Whether callback is enabled.
        """
        super().__init__(name=name, enabled=enabled)
        self._total_success = 0
        self._total_failure = 0
        self._total_violations = 0
        self._total_execution_time_ms = 0.0
        self._execution_count = 0

    @property
    def total_success(self) -> int:
        """Total successful checks."""
        return self._total_success

    @property
    def total_failure(self) -> int:
        """Total failed checks."""
        return self._total_failure

    @property
    def total_violations(self) -> int:
        """Total SLA violations."""
        return self._total_violations

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        total = self._total_success + self._total_failure
        if total == 0:
            return 1.0
        return self._total_success / total

    @property
    def average_execution_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        if self._execution_count == 0:
            return 0.0
        return self._total_execution_time_ms / self._execution_count

    def on_success(
        self,
        result: dict[str, Any],
        context: Context,
    ) -> None:
        """Record success metrics."""
        if not self.enabled:
            return

        self._total_success += 1
        self._execution_count += 1
        self._total_execution_time_ms += result.get("execution_time_ms", 0.0)

    def on_failure(
        self,
        result: dict[str, Any],
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Record failure metrics."""
        if not self.enabled:
            return

        self._total_failure += 1
        self._total_violations += len(violations)
        self._execution_count += 1
        self._total_execution_time_ms += result.get("execution_time_ms", 0.0)

    def on_sla_violation(
        self,
        violations: list[SLAViolation],
        context: Context,
    ) -> None:
        """Record violation metrics."""
        if not self.enabled:
            return

        self._total_violations += len(violations)

    def reset(self) -> None:
        """Reset all metrics."""
        self._total_success = 0
        self._total_failure = 0
        self._total_violations = 0
        self._total_execution_time_ms = 0.0
        self._execution_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics.

        Returns:
            dict[str, Any]: Statistics dictionary.
        """
        return {
            "total_success": self._total_success,
            "total_failure": self._total_failure,
            "total_violations": self._total_violations,
            "success_rate": self.success_rate,
            "execution_count": self._execution_count,
            "average_execution_time_ms": self.average_execution_time_ms,
        }
