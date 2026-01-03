"""SLA hooks for Kestra data quality integration.

This module provides hooks for responding to SLA events such as
violations, passes, and threshold changes.

Example:
    >>> from truthound_kestra.sla.hooks import (
    ...     LoggingSLAHook,
    ...     MetricsSLAHook,
    ...     CompositeSLAHook,
    ... )
    >>>
    >>> hooks = [LoggingSLAHook(), MetricsSLAHook()]
    >>> monitor = SLAMonitor(config, hooks=hooks)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from truthound_kestra.sla.config import AlertLevel, SLAViolation
from truthound_kestra.utils.helpers import get_logger

if TYPE_CHECKING:
    from truthound_kestra.sla.monitor import SLAEvaluationResult

__all__ = [
    # Protocols
    "SLAHookProtocol",
    # Base class
    "BaseSLAHook",
    # Hook implementations
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CallbackSLAHook",
    "CompositeSLAHook",
    "KestraNotificationHook",
]

logger = get_logger(__name__)


@runtime_checkable
class SLAHookProtocol(Protocol):
    """Protocol for SLA hooks."""

    def on_sla_pass(self, result: SLAEvaluationResult) -> None:
        """Called when SLA evaluation passes."""
        ...

    def on_sla_violation(self, result: SLAEvaluationResult) -> None:
        """Called when SLA evaluation fails."""
        ...


class BaseSLAHook(ABC):
    """Base class for SLA hooks.

    This class provides a common interface for all SLA hooks.
    Subclasses should override the abstract methods to implement
    specific behavior.
    """

    @abstractmethod
    def on_sla_pass(self, result: SLAEvaluationResult) -> None:
        """Called when SLA evaluation passes.

        Args:
            result: The evaluation result.
        """
        ...

    @abstractmethod
    def on_sla_violation(self, result: SLAEvaluationResult) -> None:
        """Called when SLA evaluation fails.

        Args:
            result: The evaluation result with violations.
        """
        ...

    def on_consecutive_failures(
        self,
        result: SLAEvaluationResult,
        count: int,
    ) -> None:
        """Called when consecutive failures threshold is reached.

        Args:
            result: The evaluation result.
            count: Number of consecutive failures.
        """
        pass


class LoggingSLAHook(BaseSLAHook):
    """Hook that logs SLA events.

    Example:
        >>> hook = LoggingSLAHook(log_level="warning")
        >>> monitor = SLAMonitor(config, hooks=[hook])
    """

    def __init__(
        self,
        log_level: str = "info",
        include_details: bool = True,
    ) -> None:
        """Initialize the hook.

        Args:
            log_level: Log level for pass events.
            include_details: Whether to include detailed violation info.
        """
        self._log_level = log_level
        self._include_details = include_details
        self._logger = get_logger(__name__)

    def on_sla_pass(self, result: SLAEvaluationResult) -> None:
        """Log SLA pass event."""
        msg = (
            f"SLA passed: pass_rate={result.metrics.pass_rate:.2%}, "
            f"execution_time={result.metrics.execution_time_seconds:.2f}s"
        )

        if self._log_level == "debug":
            self._logger.debug(msg)
        else:
            self._logger.info(msg)

    def on_sla_violation(self, result: SLAEvaluationResult) -> None:
        """Log SLA violation event."""
        msg = (
            f"SLA violated: {len(result.violations)} violation(s), "
            f"consecutive_failures={result.consecutive_failures}"
        )
        self._logger.warning(msg)

        if self._include_details:
            for v in result.violations:
                self._logger.warning(
                    f"  [{v.alert_level.value.upper()}] {v.violation_type.value}: {v.message}"
                )

    def on_consecutive_failures(
        self,
        result: SLAEvaluationResult,
        count: int,
    ) -> None:
        """Log consecutive failures."""
        self._logger.error(
            f"CRITICAL: {count} consecutive SLA failures "
            f"(threshold: {result.config.max_consecutive_failures})"
        )


@dataclass
class MetricsSLAHook(BaseSLAHook):
    """Hook that collects SLA metrics.

    This hook tracks statistics about SLA evaluations for
    monitoring and reporting purposes.

    Attributes:
        pass_count: Number of passed evaluations.
        violation_count: Number of failed evaluations.
        violations_by_type: Count of violations by type.
        violations_by_level: Count of violations by alert level.

    Example:
        >>> hook = MetricsSLAHook()
        >>> monitor = SLAMonitor(config, hooks=[hook])
        >>> # After evaluations...
        >>> print(f"Pass rate: {hook.pass_rate:.2%}")
    """

    pass_count: int = 0
    violation_count: int = 0
    violations_by_type: dict[str, int] = field(default_factory=dict)
    violations_by_level: dict[str, int] = field(default_factory=dict)
    total_violations: int = 0
    last_evaluation: datetime | None = None

    @property
    def total_count(self) -> int:
        """Get total evaluation count."""
        return self.pass_count + self.violation_count

    @property
    def pass_rate(self) -> float:
        """Get SLA pass rate."""
        if self.total_count == 0:
            return 1.0
        return self.pass_count / self.total_count

    def on_sla_pass(self, result: SLAEvaluationResult) -> None:
        """Record pass event."""
        self.pass_count += 1
        self.last_evaluation = datetime.now(timezone.utc)

    def on_sla_violation(self, result: SLAEvaluationResult) -> None:
        """Record violation event."""
        self.violation_count += 1
        self.last_evaluation = datetime.now(timezone.utc)

        for v in result.violations:
            # Track by type
            type_key = v.violation_type.value
            self.violations_by_type[type_key] = (
                self.violations_by_type.get(type_key, 0) + 1
            )

            # Track by level
            level_key = v.alert_level.value
            self.violations_by_level[level_key] = (
                self.violations_by_level.get(level_key, 0) + 1
            )

            self.total_violations += 1

    def reset(self) -> None:
        """Reset all metrics."""
        self.pass_count = 0
        self.violation_count = 0
        self.violations_by_type.clear()
        self.violations_by_level.clear()
        self.total_violations = 0
        self.last_evaluation = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "pass_count": self.pass_count,
            "violation_count": self.violation_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
            "violations_by_type": self.violations_by_type,
            "violations_by_level": self.violations_by_level,
            "total_violations": self.total_violations,
            "last_evaluation": (
                self.last_evaluation.isoformat()
                if self.last_evaluation
                else None
            ),
        }


class CallbackSLAHook(BaseSLAHook):
    """Hook that calls custom callback functions.

    Example:
        >>> def on_violation(result):
        ...     send_alert(result)
        >>>
        >>> hook = CallbackSLAHook(on_violation=on_violation)
        >>> monitor = SLAMonitor(config, hooks=[hook])
    """

    def __init__(
        self,
        on_pass: Callable[[SLAEvaluationResult], None] | None = None,
        on_violation: Callable[[SLAEvaluationResult], None] | None = None,
        on_consecutive: Callable[[SLAEvaluationResult, int], None] | None = None,
    ) -> None:
        """Initialize with callback functions.

        Args:
            on_pass: Callback for pass events.
            on_violation: Callback for violation events.
            on_consecutive: Callback for consecutive failures.
        """
        self._on_pass = on_pass
        self._on_violation = on_violation
        self._on_consecutive = on_consecutive

    def on_sla_pass(self, result: SLAEvaluationResult) -> None:
        """Call pass callback if defined."""
        if self._on_pass:
            self._on_pass(result)

    def on_sla_violation(self, result: SLAEvaluationResult) -> None:
        """Call violation callback if defined."""
        if self._on_violation:
            self._on_violation(result)

    def on_consecutive_failures(
        self,
        result: SLAEvaluationResult,
        count: int,
    ) -> None:
        """Call consecutive failures callback if defined."""
        if self._on_consecutive:
            self._on_consecutive(result, count)


class CompositeSLAHook(BaseSLAHook):
    """Hook that delegates to multiple hooks.

    Example:
        >>> hooks = CompositeSLAHook([
        ...     LoggingSLAHook(),
        ...     MetricsSLAHook(),
        ... ])
        >>> monitor = SLAMonitor(config, hooks=[hooks])
    """

    def __init__(self, hooks: list[BaseSLAHook]) -> None:
        """Initialize with list of hooks.

        Args:
            hooks: List of hooks to delegate to.
        """
        self._hooks = hooks

    def add_hook(self, hook: BaseSLAHook) -> None:
        """Add a hook to the composite."""
        self._hooks.append(hook)

    def remove_hook(self, hook: BaseSLAHook) -> bool:
        """Remove a hook from the composite."""
        if hook in self._hooks:
            self._hooks.remove(hook)
            return True
        return False

    def on_sla_pass(self, result: SLAEvaluationResult) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_sla_pass(result)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__} failed: {e}")

    def on_sla_violation(self, result: SLAEvaluationResult) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_sla_violation(result)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__} failed: {e}")

    def on_consecutive_failures(
        self,
        result: SLAEvaluationResult,
        count: int,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_consecutive_failures(result, count)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__} failed: {e}")


class KestraNotificationHook(BaseSLAHook):
    """Hook that sends notifications via Kestra.

    This hook uses Kestra's notification capabilities to send
    alerts when SLA violations occur.

    Example:
        >>> hook = KestraNotificationHook(
        ...     channel="slack",
        ...     min_alert_level=AlertLevel.ERROR
        ... )
        >>> monitor = SLAMonitor(config, hooks=[hook])
    """

    def __init__(
        self,
        channel: str = "slack",
        min_alert_level: AlertLevel = AlertLevel.WARNING,
        include_metrics: bool = True,
    ) -> None:
        """Initialize the hook.

        Args:
            channel: Notification channel (slack, email, webhook).
            min_alert_level: Minimum alert level to trigger notification.
            include_metrics: Whether to include metrics in notification.
        """
        self._channel = channel
        self._min_alert_level = min_alert_level
        self._include_metrics = include_metrics
        self._logger = get_logger(__name__)

    def on_sla_pass(self, result: SLAEvaluationResult) -> None:
        """No notification for pass events by default."""
        pass

    def on_sla_violation(self, result: SLAEvaluationResult) -> None:
        """Send notification for violations."""
        # Filter violations by alert level
        relevant_violations = [
            v for v in result.violations
            if v.alert_level >= self._min_alert_level
        ]

        if not relevant_violations:
            return

        # Build notification message
        message = self._build_message(result, relevant_violations)

        # Send notification
        self._send_notification(message, result.max_alert_level)

    def _build_message(
        self,
        result: SLAEvaluationResult,
        violations: list[SLAViolation],
    ) -> str:
        """Build notification message."""
        lines = [
            "🚨 SLA Violation Alert",
            "",
            f"**Flow:** {result.config.flow_id or 'N/A'}",
            f"**Task:** {result.config.task_id or 'N/A'}",
            f"**Violations:** {len(violations)}",
            "",
        ]

        for v in violations:
            level_emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🔴",
            }.get(v.alert_level, "❓")

            lines.append(f"{level_emoji} **{v.violation_type.value}**")
            lines.append(f"   {v.message}")

        if self._include_metrics:
            lines.extend([
                "",
                "**Metrics:**",
                f"- Pass Rate: {result.metrics.pass_rate:.1%}",
                f"- Execution Time: {result.metrics.execution_time_seconds:.2f}s",
                f"- Consecutive Failures: {result.consecutive_failures}",
            ])

        return "\n".join(lines)

    def _send_notification(
        self,
        message: str,
        alert_level: AlertLevel | None,
    ) -> None:
        """Send notification via Kestra."""
        try:
            from kestra import Kestra

            # Kestra notification output
            Kestra.outputs({
                "sla_alert": {
                    "channel": self._channel,
                    "message": message,
                    "level": alert_level.value if alert_level else "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            })

            self._logger.info(f"Sent SLA notification via {self._channel}")

        except ImportError:
            # Fallback: just log
            self._logger.warning(
                f"Kestra SDK not available, notification not sent:\n{message}"
            )
        except Exception as e:
            self._logger.error(f"Failed to send notification: {e}")

    def on_consecutive_failures(
        self,
        result: SLAEvaluationResult,
        count: int,
    ) -> None:
        """Send critical notification for consecutive failures."""
        message = (
            f"🔴 CRITICAL: {count} consecutive SLA failures\n\n"
            f"**Flow:** {result.config.flow_id or 'N/A'}\n"
            f"**Task:** {result.config.task_id or 'N/A'}\n"
            f"**Threshold:** {result.config.max_consecutive_failures}"
        )

        self._send_notification(message, AlertLevel.CRITICAL)
