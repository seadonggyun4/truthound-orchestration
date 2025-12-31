"""SLA Hooks for Event Handling.

This module provides hook interfaces for SLA events such as
check completion and violation detection.

Example:
    >>> from truthound_dagster.sla import LoggingSLAHook, SLAMonitor
    >>>
    >>> hook = LoggingSLAHook()
    >>> monitor = SLAMonitor(config, hooks=[hook])
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from truthound_dagster.sla.config import SLAMetrics, SLAViolation


class SLAHook(ABC):
    """Abstract base class for SLA event hooks.

    Implement this interface to handle SLA events such as
    check completion and violation detection.

    Example:
        >>> class MySLAHook(SLAHook):
        ...     def on_check(self, metrics, violations):
        ...         print(f"Checked: {metrics.total_count} rules")
        ...
        ...     def on_violation(self, violation):
        ...         alert(violation.message)
    """

    @abstractmethod
    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called when SLA check is performed.

        Args:
            metrics: The checked metrics.
            violations: Detected violations.
            context: Optional context data.
        """
        ...

    @abstractmethod
    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called for each violation detected.

        Args:
            violation: The violation.
            context: Optional context data.
        """
        ...

    def on_success(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called when check passes without violations.

        Args:
            metrics: The checked metrics.
            context: Optional context data.
        """
        pass

    def on_consecutive_failure(
        self,
        count: int,
        threshold: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called when consecutive failures occur.

        Args:
            count: Current consecutive failure count.
            threshold: Configured threshold.
            context: Optional context data.
        """
        pass


class LoggingSLAHook(SLAHook):
    """SLA hook that logs events.

    This hook logs SLA events using Python's logging module.

    Parameters
    ----------
    logger : logging.Logger | None
        Logger to use. Uses module logger if None.

    level : int
        Log level for regular events.

    violation_level : int
        Log level for violations.

    Example:
        >>> hook = LoggingSLAHook()
        >>> monitor = SLAMonitor(config, hooks=[hook])
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        violation_level: int = logging.WARNING,
    ) -> None:
        """Initialize logging hook.

        Args:
            logger: Logger to use.
            level: Log level for regular events.
            violation_level: Log level for violations.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._level = level
        self._violation_level = violation_level

    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log SLA check event."""
        if violations:
            self._logger.log(
                self._violation_level,
                f"SLA check found {len(violations)} violations: "
                f"passed={metrics.passed_count}, failed={metrics.failed_count}, "
                f"rate={metrics.failure_rate:.2%}",
            )
        else:
            self._logger.log(
                self._level,
                f"SLA check passed: "
                f"passed={metrics.passed_count}, failed={metrics.failed_count}, "
                f"time={metrics.execution_time_ms:.2f}ms",
            )

    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log SLA violation."""
        self._logger.log(
            self._violation_level,
            f"SLA Violation [{violation.alert_level.value.upper()}]: "
            f"{violation.message}",
        )

    def on_success(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log SLA success."""
        self._logger.log(
            self._level,
            f"SLA check passed: {metrics.passed_count} rules, "
            f"time={metrics.execution_time_ms:.2f}ms",
        )

    def on_consecutive_failure(
        self,
        count: int,
        threshold: int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log consecutive failure."""
        if count >= threshold:
            self._logger.log(
                self._violation_level,
                f"SLA consecutive failure threshold reached: "
                f"{count}/{threshold}",
            )


class MetricsSLAHook(SLAHook):
    """SLA hook that collects metrics.

    This hook tracks statistics about SLA checks and violations.

    Example:
        >>> hook = MetricsSLAHook()
        >>> monitor = SLAMonitor(config, hooks=[hook])
        >>> # ... run checks ...
        >>> print(hook.get_stats())
    """

    def __init__(self) -> None:
        """Initialize metrics hook."""
        self._check_count = 0
        self._violation_count = 0
        self._success_count = 0
        self._total_execution_time_ms = 0.0
        self._violations_by_type: dict[str, int] = {}

    @property
    def check_count(self) -> int:
        """Total number of checks."""
        return self._check_count

    @property
    def violation_count(self) -> int:
        """Total number of violations."""
        return self._violation_count

    @property
    def success_count(self) -> int:
        """Number of successful checks."""
        return self._success_count

    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)."""
        if self._check_count == 0:
            return 1.0
        return self._success_count / self._check_count

    @property
    def average_execution_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        if self._check_count == 0:
            return 0.0
        return self._total_execution_time_ms / self._check_count

    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record check metrics."""
        self._check_count += 1
        self._total_execution_time_ms += metrics.execution_time_ms

        if not violations:
            self._success_count += 1

    def on_violation(
        self,
        violation: SLAViolation,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record violation."""
        self._violation_count += 1
        vtype = violation.violation_type.value
        self._violations_by_type[vtype] = self._violations_by_type.get(vtype, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Get collected statistics.

        Returns:
            dict[str, Any]: Statistics dictionary.
        """
        return {
            "check_count": self._check_count,
            "violation_count": self._violation_count,
            "success_count": self._success_count,
            "success_rate": self.success_rate,
            "average_execution_time_ms": self.average_execution_time_ms,
            "violations_by_type": dict(self._violations_by_type),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._check_count = 0
        self._violation_count = 0
        self._success_count = 0
        self._total_execution_time_ms = 0.0
        self._violations_by_type.clear()


class CompositeSLAHook(SLAHook):
    """SLA hook that delegates to multiple hooks.

    Use this to combine multiple hooks.

    Parameters
    ----------
    hooks : list[SLAHook]
        Hooks to delegate to.

    Example:
        >>> composite = CompositeSLAHook([
        ...     LoggingSLAHook(),
        ...     MetricsSLAHook(),
        ... ])
    """

    def __init__(self, hooks: list[SLAHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: Hooks to delegate to.
        """
        self._hooks = list(hooks)

    @property
    def hooks(self) -> list[SLAHook]:
        """Get list of hooks."""
        return list(self._hooks)

    def add_hook(self, hook: SLAHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: SLAHook) -> bool:
        """Remove a hook.

        Args:
            hook: Hook to remove.

        Returns:
            bool: True if removed.
        """
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
                pass  # Isolate hook failures

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
                pass

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
                pass

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
                pass
