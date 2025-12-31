"""SLA hooks for event handling.

This module provides hook implementations for handling SLA events,
following the Protocol pattern for extensibility.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from truthound_prefect.sla.config import SLAMetrics, SLAViolation


class SLAHook(ABC):
    """Abstract base class for SLA hooks.

    Implement this class to handle SLA events like checks,
    violations, and successes.
    """

    @abstractmethod
    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called after each SLA check.

        Args:
            metrics: The metrics that were checked.
            violations: List of violations (empty if none).
            context: Optional additional context.
        """
        ...

    @abstractmethod
    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called for each SLA violation.

        Args:
            violation: The violation that occurred.
            context: Optional additional context.
        """
        ...

    def on_success(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called when a check passes without violations.

        Args:
            metrics: The metrics that were checked.
            context: Optional additional context.
        """
        pass  # Optional override

    def on_consecutive_failure(
        self,
        count: int,
        threshold: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called when consecutive failure threshold is reached.

        Args:
            count: Current consecutive failure count.
            threshold: The configured threshold.
            context: Optional additional context.
        """
        pass  # Optional override


class LoggingSLAHook(SLAHook):
    """Hook that logs SLA events.

    Uses Python's logging module to log events at configurable levels.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        check_level: int = logging.DEBUG,
        violation_level: int = logging.ERROR,
        success_level: int = logging.INFO,
    ) -> None:
        """Initialize the logging hook.

        Args:
            logger: Optional logger instance.
            check_level: Log level for check events.
            violation_level: Log level for violation events.
            success_level: Log level for success events.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._check_level = check_level
        self._violation_level = violation_level
        self._success_level = success_level

    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log the check event."""
        violation_count = len(violations)
        self._logger.log(
            self._check_level,
            "SLA check completed: %d violations | pass_rate=%.2f%% | "
            "execution_time=%.2fms | flow=%s | task=%s",
            violation_count,
            metrics.pass_rate * 100,
            metrics.execution_time_ms,
            metrics.flow_name or "unknown",
            metrics.task_name or "unknown",
        )

    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log the violation."""
        self._logger.log(
            self._violation_level,
            "SLA VIOLATION [%s]: %s | threshold=%s | actual=%s | flow=%s",
            violation.violation_type.value,
            violation.message,
            violation.threshold,
            violation.actual,
            violation.flow_name or "unknown",
        )

    def on_success(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log the success."""
        self._logger.log(
            self._success_level,
            "SLA check passed: pass_rate=%.2f%% | flow=%s | task=%s",
            metrics.pass_rate * 100,
            metrics.flow_name or "unknown",
            metrics.task_name or "unknown",
        )

    def on_consecutive_failure(
        self,
        count: int,
        threshold: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log the consecutive failure."""
        self._logger.critical(
            "SLA ALERT: Consecutive failures (%d) reached threshold (%d)",
            count,
            threshold,
        )


@dataclass
class SLAHookStats:
    """Statistics collected by MetricsSLAHook."""

    check_count: int = 0
    violation_count: int = 0
    success_count: int = 0
    consecutive_failure_alerts: int = 0
    violations_by_type: dict[str, int] = field(default_factory=dict)
    last_check_time: datetime | None = None
    last_violation_time: datetime | None = None


class MetricsSLAHook(SLAHook):
    """Hook that collects SLA metrics.

    Tracks statistics about SLA checks and violations.
    """

    def __init__(self) -> None:
        """Initialize the metrics hook."""
        self._stats = SLAHookStats()

    @property
    def stats(self) -> SLAHookStats:
        """Get the current statistics."""
        return self._stats

    @property
    def check_count(self) -> int:
        """Get the total check count."""
        return self._stats.check_count

    @property
    def violation_count(self) -> int:
        """Get the total violation count."""
        return self._stats.violation_count

    @property
    def success_count(self) -> int:
        """Get the success count."""
        return self._stats.success_count

    @property
    def success_rate(self) -> float:
        """Get the success rate."""
        if self._stats.check_count == 0:
            return 1.0
        return self._stats.success_count / self._stats.check_count

    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record the check event."""
        self._stats.check_count += 1
        self._stats.last_check_time = datetime.now()

    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record the violation."""
        self._stats.violation_count += 1
        self._stats.last_violation_time = datetime.now()

        violation_type = violation.violation_type.value
        self._stats.violations_by_type[violation_type] = (
            self._stats.violations_by_type.get(violation_type, 0) + 1
        )

    def on_success(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record the success."""
        self._stats.success_count += 1

    def on_consecutive_failure(
        self,
        count: int,
        threshold: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record the consecutive failure alert."""
        self._stats.consecutive_failure_alerts += 1

    def reset(self) -> None:
        """Reset statistics."""
        self._stats = SLAHookStats()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of statistics."""
        return {
            "check_count": self._stats.check_count,
            "violation_count": self._stats.violation_count,
            "success_count": self._stats.success_count,
            "success_rate": self.success_rate,
            "consecutive_failure_alerts": self._stats.consecutive_failure_alerts,
            "violations_by_type": dict(self._stats.violations_by_type),
            "last_check_time": (
                self._stats.last_check_time.isoformat()
                if self._stats.last_check_time
                else None
            ),
            "last_violation_time": (
                self._stats.last_violation_time.isoformat()
                if self._stats.last_violation_time
                else None
            ),
        }


class CompositeSLAHook(SLAHook):
    """Hook that delegates to multiple hooks.

    Provides failure isolation - errors in one hook don't affect others.
    """

    def __init__(self, hooks: list[SLAHook]) -> None:
        """Initialize with a list of hooks.

        Args:
            hooks: List of hooks to delegate to.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: SLAHook) -> None:
        """Add a hook to the composite."""
        self._hooks.append(hook)

    def remove_hook(self, hook: SLAHook) -> bool:
        """Remove a hook from the composite."""
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False

    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_check(metrics, violations, context)
            except Exception:
                pass  # Isolate failures

    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_violation(violation, context)
            except Exception:
                pass  # Isolate failures

    def on_success(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_success(metrics, context)
            except Exception:
                pass  # Isolate failures

    def on_consecutive_failure(
        self,
        count: int,
        threshold: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_consecutive_failure(count, threshold, context)
            except Exception:
                pass  # Isolate failures


class CallbackSLAHook(SLAHook):
    """Hook that calls user-defined callbacks.

    Provides a simple way to handle SLA events without subclassing.
    """

    def __init__(
        self,
        on_check_callback: Any | None = None,
        on_violation_callback: Any | None = None,
        on_success_callback: Any | None = None,
        on_consecutive_failure_callback: Any | None = None,
    ) -> None:
        """Initialize with callbacks.

        Args:
            on_check_callback: Called on each check.
            on_violation_callback: Called on each violation.
            on_success_callback: Called on each success.
            on_consecutive_failure_callback: Called on consecutive failure.
        """
        self._on_check = on_check_callback
        self._on_violation = on_violation_callback
        self._on_success = on_success_callback
        self._on_consecutive_failure = on_consecutive_failure_callback

    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Call the check callback."""
        if self._on_check:
            self._on_check(metrics, violations, context)

    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Call the violation callback."""
        if self._on_violation:
            self._on_violation(violation, context)

    def on_success(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Call the success callback."""
        if self._on_success:
            self._on_success(metrics, context)

    def on_consecutive_failure(
        self,
        count: int,
        threshold: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Call the consecutive failure callback."""
        if self._on_consecutive_failure:
            self._on_consecutive_failure(count, threshold, context)


__all__ = [
    "SLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CompositeSLAHook",
    "CallbackSLAHook",
    "SLAHookStats",
]
