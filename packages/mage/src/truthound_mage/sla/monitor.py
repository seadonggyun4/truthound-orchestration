"""SLA Monitoring for Mage Data Quality Operations.

This module provides SLA monitoring capabilities that evaluate metrics
against configured thresholds and detect violations.

Example:
    >>> monitor = SLAMonitor(config=PRODUCTION_SLA_CONFIG)
    >>> violations = monitor.check(metrics)
    >>> for v in violations:
    ...     print(f"{v.violation_type.value}: {v.message}")
"""

from __future__ import annotations

import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Sequence

from truthound_mage.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)

if TYPE_CHECKING:
    pass


class SLAMonitor:
    """Monitor for evaluating SLA compliance.

    This class checks metrics against configured SLA thresholds and
    tracks consecutive failures.

    Attributes:
        config: SLA configuration with thresholds.

    Example:
        >>> monitor = SLAMonitor(config=SLAConfig(
        ...     max_failure_rate=0.05,
        ...     min_pass_rate=0.95,
        ... ))
        >>> metrics = SLAMetrics(passed_count=95, failed_count=5)
        >>> violations = monitor.check(metrics)
    """

    def __init__(
        self,
        config: SLAConfig | None = None,
        history_size: int = 100,
    ) -> None:
        """Initialize SLA monitor.

        Args:
            config: SLA configuration. Uses default if None.
            history_size: Maximum number of metrics to retain in history.
        """
        self.config = config or SLAConfig()
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._history: deque[SLAMetrics] = deque(maxlen=history_size)

    @property
    def consecutive_failures(self) -> int:
        """Get current consecutive failure count (thread-safe)."""
        with self._lock:
            return self._consecutive_failures

    def check(self, metrics: SLAMetrics) -> list[SLAViolation]:
        """Check metrics against SLA thresholds.

        Args:
            metrics: Metrics to evaluate.

        Returns:
            List of SLA violations found.
        """
        if not self.config.enabled:
            return []

        violations: list[SLAViolation] = []

        # Add to history
        with self._lock:
            self._history.append(metrics)

        # Check failure rate
        if self.config.max_failure_rate is not None:
            if metrics.failure_rate > self.config.max_failure_rate:
                violations.append(
                    SLAViolation(
                        violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
                        message=(
                            f"Failure rate {metrics.failure_rate:.2%} exceeds "
                            f"threshold {self.config.max_failure_rate:.2%}"
                        ),
                        threshold=self.config.max_failure_rate,
                        actual=metrics.failure_rate,
                        alert_level=self.config.alert_level,
                        block_uuid=metrics.block_uuid,
                        pipeline_uuid=metrics.pipeline_uuid,
                    )
                )

        # Check pass rate
        if self.config.min_pass_rate is not None:
            if metrics.pass_rate < self.config.min_pass_rate:
                violations.append(
                    SLAViolation(
                        violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
                        message=(
                            f"Pass rate {metrics.pass_rate:.2%} below "
                            f"minimum {self.config.min_pass_rate:.2%}"
                        ),
                        threshold=self.config.min_pass_rate,
                        actual=metrics.pass_rate,
                        alert_level=self.config.alert_level,
                        block_uuid=metrics.block_uuid,
                        pipeline_uuid=metrics.pipeline_uuid,
                    )
                )

        # Check execution time
        if self.config.max_execution_time_seconds is not None:
            if metrics.execution_time_seconds > self.config.max_execution_time_seconds:
                violations.append(
                    SLAViolation(
                        violation_type=SLAViolationType.EXECUTION_TIME_EXCEEDED,
                        message=(
                            f"Execution time {metrics.execution_time_seconds:.1f}s exceeds "
                            f"limit {self.config.max_execution_time_seconds:.1f}s"
                        ),
                        threshold=self.config.max_execution_time_seconds,
                        actual=metrics.execution_time_seconds,
                        alert_level=self.config.alert_level,
                        block_uuid=metrics.block_uuid,
                        pipeline_uuid=metrics.pipeline_uuid,
                    )
                )

        # Check row count
        if metrics.row_count is not None:
            if self.config.min_row_count is not None:
                if metrics.row_count < self.config.min_row_count:
                    violations.append(
                        SLAViolation(
                            violation_type=SLAViolationType.ROW_COUNT_BELOW_MINIMUM,
                            message=(
                                f"Row count {metrics.row_count} below "
                                f"minimum {self.config.min_row_count}"
                            ),
                            threshold=self.config.min_row_count,
                            actual=metrics.row_count,
                            alert_level=self.config.alert_level,
                            block_uuid=metrics.block_uuid,
                            pipeline_uuid=metrics.pipeline_uuid,
                        )
                    )

            if self.config.max_row_count is not None:
                if metrics.row_count > self.config.max_row_count:
                    violations.append(
                        SLAViolation(
                            violation_type=SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM,
                            message=(
                                f"Row count {metrics.row_count} above "
                                f"maximum {self.config.max_row_count}"
                            ),
                            threshold=self.config.max_row_count,
                            actual=metrics.row_count,
                            alert_level=self.config.alert_level,
                            block_uuid=metrics.block_uuid,
                            pipeline_uuid=metrics.pipeline_uuid,
                        )
                    )

        # Track consecutive failures
        if violations:
            self.record_failure()
        else:
            self.reset_failures()

        # Check consecutive failures
        if self.config.max_consecutive_failures is not None:
            if self._consecutive_failures > self.config.max_consecutive_failures:
                violations.append(
                    SLAViolation(
                        violation_type=SLAViolationType.CONSECUTIVE_FAILURES,
                        message=(
                            f"Consecutive failures {self._consecutive_failures} exceeds "
                            f"limit {self.config.max_consecutive_failures}"
                        ),
                        threshold=self.config.max_consecutive_failures,
                        actual=self._consecutive_failures,
                        alert_level=AlertLevel.CRITICAL,  # Always critical
                        block_uuid=metrics.block_uuid,
                        pipeline_uuid=metrics.pipeline_uuid,
                    )
                )

        return violations

    def record_failure(self) -> None:
        """Record a failure for consecutive failure tracking."""
        with self._lock:
            self._consecutive_failures += 1

    def reset_failures(self) -> None:
        """Reset consecutive failure count."""
        with self._lock:
            self._consecutive_failures = 0

    def get_history(self) -> list[SLAMetrics]:
        """Get metrics history.

        Returns:
            List of historical metrics.
        """
        with self._lock:
            return list(self._history)

    def get_stats(self) -> dict[str, Any]:
        """Get monitor statistics.

        Returns:
            Dictionary with monitor statistics.
        """
        with self._lock:
            if not self._history:
                return {
                    "total_checks": 0,
                    "avg_pass_rate": 0.0,
                    "avg_failure_rate": 0.0,
                    "consecutive_failures": self._consecutive_failures,
                }

            total_checks = len(self._history)
            avg_pass_rate = sum(m.pass_rate for m in self._history) / total_checks
            avg_failure_rate = sum(m.failure_rate for m in self._history) / total_checks

            return {
                "total_checks": total_checks,
                "avg_pass_rate": avg_pass_rate,
                "avg_failure_rate": avg_failure_rate,
                "consecutive_failures": self._consecutive_failures,
            }


class SLARegistry:
    """Registry for managing multiple SLA monitors.

    This class provides a central registry for SLA monitors,
    organized by block/pipeline identifiers.

    Example:
        >>> registry = SLARegistry()
        >>> registry.register("my_block", SLAConfig(min_pass_rate=0.95))
        >>> violations = registry.check("my_block", metrics)
    """

    def __init__(self) -> None:
        """Initialize SLA registry."""
        self._monitors: dict[str, SLAMonitor] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        config: SLAConfig | None = None,
    ) -> SLAMonitor:
        """Register a new SLA monitor.

        Args:
            name: Monitor identifier.
            config: SLA configuration.

        Returns:
            The registered SLAMonitor.
        """
        with self._lock:
            if name not in self._monitors:
                self._monitors[name] = SLAMonitor(config=config)
            return self._monitors[name]

    def get(self, name: str) -> SLAMonitor | None:
        """Get a registered monitor by name.

        Args:
            name: Monitor identifier.

        Returns:
            SLAMonitor if found, None otherwise.
        """
        with self._lock:
            return self._monitors.get(name)

    def get_or_create(
        self,
        name: str,
        config: SLAConfig | None = None,
    ) -> SLAMonitor:
        """Get existing monitor or create new one.

        Args:
            name: Monitor identifier.
            config: SLA configuration for new monitor.

        Returns:
            Existing or new SLAMonitor.
        """
        with self._lock:
            if name not in self._monitors:
                self._monitors[name] = SLAMonitor(config=config)
            return self._monitors[name]

    def check(
        self,
        name: str,
        metrics: SLAMetrics,
    ) -> list[SLAViolation]:
        """Check metrics using named monitor.

        Args:
            name: Monitor identifier.
            metrics: Metrics to evaluate.

        Returns:
            List of violations, empty list if monitor not found.
        """
        monitor = self.get(name)
        if monitor is None:
            return []
        return monitor.check(metrics)

    def list_monitors(self) -> list[str]:
        """List all registered monitor names.

        Returns:
            List of monitor names.
        """
        with self._lock:
            return list(self._monitors.keys())

    def remove(self, name: str) -> bool:
        """Remove a monitor from registry.

        Args:
            name: Monitor identifier.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._monitors:
                del self._monitors[name]
                return True
            return False

    def reset_all(self) -> None:
        """Reset all monitors' consecutive failure counts."""
        with self._lock:
            for monitor in self._monitors.values():
                monitor.reset_failures()

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics from all monitors.

        Returns:
            Dictionary mapping monitor names to their statistics.
        """
        with self._lock:
            return {name: monitor.get_stats() for name, monitor in self._monitors.items()}


# Global registry instance
_global_registry: SLARegistry | None = None


def get_sla_registry() -> SLARegistry:
    """Get the global SLA registry.

    Returns:
        Global SLARegistry instance.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SLARegistry()
    return _global_registry


def reset_sla_registry() -> None:
    """Reset the global SLA registry."""
    global _global_registry
    _global_registry = None
