"""SLA Hooks for Mage Data Quality Operations.

This module provides lifecycle hooks for SLA monitoring events
such as violation detection, metrics recording, and alerting.

Example:
    >>> hook = LoggingSLAHook()
    >>> hook.on_violation(violation)
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from truthound_mage.sla.config import (
        SLAConfig,
        SLAMetrics,
        SLAViolation,
    )


class BaseSLAHook(ABC):
    """Abstract base class for SLA hooks.

    SLA hooks are invoked during SLA monitoring lifecycle events
    such as check start, violation detection, and check completion.

    Subclasses must implement:
        - on_check_start: Called before SLA check
        - on_violation: Called when violation detected
        - on_check_complete: Called after SLA check

    Example:
        >>> class CustomHook(BaseSLAHook):
        ...     def on_violation(self, violation: SLAViolation) -> None:
        ...         send_alert(violation.message)
    """

    @abstractmethod
    def on_check_start(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
    ) -> None:
        """Called before SLA check.

        Args:
            config: SLA configuration being used.
            metrics: Metrics about to be checked.
        """
        ...

    @abstractmethod
    def on_violation(
        self,
        violation: SLAViolation,
    ) -> None:
        """Called when SLA violation is detected.

        Args:
            violation: The detected violation.
        """
        ...

    @abstractmethod
    def on_check_complete(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
    ) -> None:
        """Called after SLA check completes.

        Args:
            config: SLA configuration used.
            metrics: Metrics that were checked.
            violations: List of violations found.
        """
        ...


class LoggingSLAHook(BaseSLAHook):
    """SLA hook that logs events.

    This hook logs SLA check events and violations using
    the common logging infrastructure.

    Example:
        >>> hook = LoggingSLAHook()
        >>> monitor = SLAMonitor(config=config, hooks=[hook])
    """

    def __init__(self, log_level: str = "INFO") -> None:
        """Initialize logging hook.

        Args:
            log_level: Logging level for messages.
        """
        self.log_level = log_level

    def on_check_start(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
    ) -> None:
        """Log SLA check start."""
        from common import get_logger

        logger = get_logger(__name__)
        logger.debug(
            "SLA check started",
            block_uuid=metrics.block_uuid,
            pipeline_uuid=metrics.pipeline_uuid,
        )

    def on_violation(
        self,
        violation: SLAViolation,
    ) -> None:
        """Log SLA violation."""
        from common import get_logger

        logger = get_logger(__name__)

        if violation.alert_level.value == "critical":
            logger.error(
                "CRITICAL SLA violation detected",
                violation_type=violation.violation_type.value,
                message=violation.message,
                threshold=violation.threshold,
                actual=violation.actual,
                block_uuid=violation.block_uuid,
                pipeline_uuid=violation.pipeline_uuid,
            )
        elif violation.alert_level.value == "error":
            logger.error(
                "SLA violation detected",
                violation_type=violation.violation_type.value,
                message=violation.message,
                threshold=violation.threshold,
                actual=violation.actual,
            )
        elif violation.alert_level.value == "warning":
            logger.warning(
                "SLA warning",
                violation_type=violation.violation_type.value,
                message=violation.message,
            )
        else:
            logger.info(
                "SLA notification",
                violation_type=violation.violation_type.value,
                message=violation.message,
            )

    def on_check_complete(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
    ) -> None:
        """Log SLA check completion."""
        from common import get_logger

        logger = get_logger(__name__)

        if violations:
            logger.warning(
                "SLA check completed with violations",
                violation_count=len(violations),
                pass_rate=metrics.pass_rate,
                failure_rate=metrics.failure_rate,
            )
        else:
            logger.debug(
                "SLA check passed",
                pass_rate=metrics.pass_rate,
                execution_time_ms=metrics.execution_time_ms,
            )


class MetricsSLAHook(BaseSLAHook):
    """SLA hook that collects metrics.

    This hook collects statistics about SLA checks and violations
    for monitoring and reporting.

    Attributes:
        check_count: Total number of checks performed.
        violation_count: Total number of violations detected.
        violations_by_type: Violations grouped by type.

    Example:
        >>> hook = MetricsSLAHook()
        >>> # ... perform checks ...
        >>> print(hook.get_stats())
    """

    def __init__(self) -> None:
        """Initialize metrics hook."""
        self._lock = threading.Lock()
        self._check_count = 0
        self._violation_count = 0
        self._pass_count = 0
        self._violations_by_type: dict[str, int] = defaultdict(int)
        self._total_execution_time_ms: float = 0.0
        self._last_check_time: datetime | None = None

    @property
    def check_count(self) -> int:
        """Total number of checks performed."""
        with self._lock:
            return self._check_count

    @property
    def violation_count(self) -> int:
        """Total number of violations detected."""
        with self._lock:
            return self._violation_count

    @property
    def pass_count(self) -> int:
        """Total number of checks that passed."""
        with self._lock:
            return self._pass_count

    @property
    def violations_by_type(self) -> dict[str, int]:
        """Violations grouped by type."""
        with self._lock:
            return dict(self._violations_by_type)

    def on_check_start(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
    ) -> None:
        """Record check start."""
        with self._lock:
            self._last_check_time = datetime.now(timezone.utc)

    def on_violation(
        self,
        violation: SLAViolation,
    ) -> None:
        """Record violation."""
        with self._lock:
            self._violation_count += 1
            self._violations_by_type[violation.violation_type.value] += 1

    def on_check_complete(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
    ) -> None:
        """Record check completion."""
        with self._lock:
            self._check_count += 1
            self._total_execution_time_ms += metrics.execution_time_ms

            if not violations:
                self._pass_count += 1

    def get_stats(self) -> dict[str, Any]:
        """Get collected statistics.

        Returns:
            Dictionary with metrics statistics.
        """
        with self._lock:
            return {
                "check_count": self._check_count,
                "violation_count": self._violation_count,
                "pass_count": self._pass_count,
                "pass_rate": (
                    self._pass_count / self._check_count if self._check_count > 0 else 0.0
                ),
                "violations_by_type": dict(self._violations_by_type),
                "avg_execution_time_ms": (
                    self._total_execution_time_ms / self._check_count
                    if self._check_count > 0
                    else 0.0
                ),
                "last_check_time": (
                    self._last_check_time.isoformat() if self._last_check_time else None
                ),
            }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._check_count = 0
            self._violation_count = 0
            self._pass_count = 0
            self._violations_by_type.clear()
            self._total_execution_time_ms = 0.0
            self._last_check_time = None


class CompositeSLAHook(BaseSLAHook):
    """Composite hook that delegates to multiple hooks.

    This hook allows combining multiple SLA hooks into a single
    hook that invokes all of them.

    Example:
        >>> logging_hook = LoggingSLAHook()
        >>> metrics_hook = MetricsSLAHook()
        >>> composite = CompositeSLAHook([logging_hook, metrics_hook])
    """

    def __init__(self, hooks: Sequence[BaseSLAHook]) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to delegate to.
        """
        self._hooks = list(hooks)

    def add_hook(self, hook: BaseSLAHook) -> None:
        """Add a hook to the composite.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: BaseSLAHook) -> bool:
        """Remove a hook from the composite.

        Args:
            hook: Hook to remove.

        Returns:
            True if removed, False if not found.
        """
        try:
            self._hooks.remove(hook)
            return True
        except ValueError:
            return False

    def on_check_start(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_check_start(config, metrics)
            except Exception:
                pass  # Hooks should not break execution

    def on_violation(
        self,
        violation: SLAViolation,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_violation(violation)
            except Exception:
                pass

    def on_check_complete(
        self,
        config: SLAConfig,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_check_complete(config, metrics, violations)
            except Exception:
                pass
