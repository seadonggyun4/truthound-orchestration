"""SLA Monitor for Data Quality Operations in Dagster.

This module provides SLA monitoring and violation detection for
data quality operations in Dagster pipelines.

Example:
    >>> from truthound_dagster.sla import SLAMonitor, SLAConfig
    >>>
    >>> monitor = SLAMonitor(
    ...     config=SLAConfig(max_failure_rate=0.05),
    ... )
    >>>
    >>> violations = monitor.check(metrics)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from truthound_dagster.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)

if TYPE_CHECKING:
    pass


class SLAMonitor:
    """Monitor for SLA compliance in data quality operations.

    The monitor evaluates metrics against SLA configuration and
    detects violations. It also tracks consecutive failures.

    Parameters
    ----------
    config : SLAConfig
        SLA configuration to evaluate against.

    name : str | None
        Optional name for the monitor.

    Attributes
    ----------
    config : SLAConfig
        The SLA configuration.

    consecutive_failures : int
        Current count of consecutive failures.

    Examples
    --------
    Basic usage:

    >>> monitor = SLAMonitor(
    ...     config=SLAConfig(max_failure_rate=0.05),
    ... )
    >>> violations = monitor.check(metrics)
    >>> if violations:
    ...     for v in violations:
    ...         print(f"Violation: {v.message}")

    With consecutive failure tracking:

    >>> monitor = SLAMonitor(
    ...     config=SLAConfig(max_consecutive_failures=3),
    ... )
    >>> monitor.record_failure()  # count: 1
    >>> monitor.record_failure()  # count: 2
    >>> monitor.record_failure()  # count: 3
    >>> violations = monitor.check_consecutive_failures()
    """

    def __init__(
        self,
        config: SLAConfig,
        name: str | None = None,
    ) -> None:
        """Initialize SLA monitor.

        Args:
            config: SLA configuration.
            name: Optional monitor name.
        """
        self.config = config
        self.name = name
        self._consecutive_failures = 0
        self._lock = threading.Lock()
        self._history: list[SLAMetrics] = []
        self._max_history = 100

    @property
    def consecutive_failures(self) -> int:
        """Current consecutive failure count."""
        with self._lock:
            return self._consecutive_failures

    def check(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> list[SLAViolation]:
        """Check metrics against SLA configuration.

        Args:
            metrics: Metrics to evaluate.
            context: Optional context for metadata.

        Returns:
            list[SLAViolation]: List of detected violations.
        """
        if not self.config.enabled:
            return []

        violations: list[SLAViolation] = []
        asset_key = metrics.asset_key or (context or {}).get("asset_key")
        run_id = metrics.run_id or (context or {}).get("run_id")

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
                        asset_key=asset_key,
                        run_id=run_id,
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
                        asset_key=asset_key,
                        run_id=run_id,
                    )
                )

        # Check execution time
        if self.config.max_execution_time_seconds is not None:
            if metrics.execution_time_seconds > self.config.max_execution_time_seconds:
                violations.append(
                    SLAViolation(
                        violation_type=SLAViolationType.EXECUTION_TIME_EXCEEDED,
                        message=(
                            f"Execution time {metrics.execution_time_seconds:.2f}s exceeds "
                            f"limit {self.config.max_execution_time_seconds:.2f}s"
                        ),
                        threshold=self.config.max_execution_time_seconds,
                        actual=metrics.execution_time_seconds,
                        alert_level=self.config.alert_level,
                        asset_key=asset_key,
                        run_id=run_id,
                    )
                )

        # Check row count bounds
        if metrics.row_count is not None:
            if (
                self.config.min_row_count is not None
                and metrics.row_count < self.config.min_row_count
            ):
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
                        asset_key=asset_key,
                        run_id=run_id,
                    )
                )

            if (
                self.config.max_row_count is not None
                and metrics.row_count > self.config.max_row_count
            ):
                violations.append(
                    SLAViolation(
                        violation_type=SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM,
                        message=(
                            f"Row count {metrics.row_count} exceeds "
                            f"maximum {self.config.max_row_count}"
                        ),
                        threshold=self.config.max_row_count,
                        actual=metrics.row_count,
                        alert_level=self.config.alert_level,
                        asset_key=asset_key,
                        run_id=run_id,
                    )
                )

        # Update consecutive failure tracking
        if metrics.failed_count > 0:
            self.record_failure()
            consecutive_violations = self.check_consecutive_failures(
                asset_key=asset_key,
                run_id=run_id,
            )
            violations.extend(consecutive_violations)
        else:
            self.record_success()

        # Store metrics in history
        self._add_to_history(metrics)

        return violations

    def record_failure(self) -> int:
        """Record a failure and increment consecutive count.

        Returns:
            int: Current consecutive failure count.
        """
        with self._lock:
            self._consecutive_failures += 1
            return self._consecutive_failures

    def record_success(self) -> None:
        """Record a success and reset consecutive failures."""
        with self._lock:
            self._consecutive_failures = 0

    def check_consecutive_failures(
        self,
        asset_key: str | None = None,
        run_id: str | None = None,
    ) -> list[SLAViolation]:
        """Check for consecutive failure violations.

        Args:
            asset_key: Dagster asset key.
            run_id: Dagster run ID.

        Returns:
            list[SLAViolation]: Violations if threshold exceeded.
        """
        if self.config.max_consecutive_failures is None:
            return []

        with self._lock:
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                return [
                    SLAViolation(
                        violation_type=SLAViolationType.CONSECUTIVE_FAILURES,
                        message=(
                            f"Consecutive failures {self._consecutive_failures} "
                            f"reached threshold {self.config.max_consecutive_failures}"
                        ),
                        threshold=self.config.max_consecutive_failures,
                        actual=self._consecutive_failures,
                        alert_level=AlertLevel.CRITICAL,
                        asset_key=asset_key,
                        run_id=run_id,
                    )
                ]
        return []

    def reset(self) -> None:
        """Reset monitor state."""
        with self._lock:
            self._consecutive_failures = 0
            self._history.clear()

    def _add_to_history(self, metrics: SLAMetrics) -> None:
        """Add metrics to history with size limit."""
        with self._lock:
            self._history.append(metrics)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def get_history(self) -> list[SLAMetrics]:
        """Get metrics history.

        Returns:
            list[SLAMetrics]: Copy of metrics history.
        """
        with self._lock:
            return list(self._history)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of monitor state.

        Returns:
            dict[str, Any]: Summary dictionary.
        """
        with self._lock:
            history = list(self._history)

        if not history:
            return {
                "name": self.name,
                "enabled": self.config.enabled,
                "consecutive_failures": self.consecutive_failures,
                "history_count": 0,
                "average_pass_rate": None,
                "average_failure_rate": None,
                "average_execution_time_ms": None,
            }

        total_passed = sum(m.passed_count for m in history)
        total_failed = sum(m.failed_count for m in history)
        total_count = total_passed + total_failed
        avg_pass_rate = total_passed / total_count if total_count > 0 else 1.0
        avg_failure_rate = total_failed / total_count if total_count > 0 else 0.0
        avg_execution_time = sum(m.execution_time_ms for m in history) / len(history)

        return {
            "name": self.name,
            "enabled": self.config.enabled,
            "consecutive_failures": self.consecutive_failures,
            "history_count": len(history),
            "average_pass_rate": avg_pass_rate,
            "average_failure_rate": avg_failure_rate,
            "average_execution_time_ms": avg_execution_time,
        }


class SLARegistry:
    """Registry for managing multiple SLA monitors.

    The registry provides centralized management of SLA monitors
    for different assets, jobs, or domains.

    Example:
        >>> registry = SLARegistry()
        >>> registry.register(
        ...     "users_check",
        ...     SLAConfig(max_failure_rate=0.05),
        ... )
        >>> monitor = registry.get("users_check")
        >>> violations = monitor.check(metrics)
    """

    def __init__(self) -> None:
        """Initialize SLA registry."""
        self._monitors: dict[str, SLAMonitor] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        config: SLAConfig,
        replace: bool = False,
    ) -> SLAMonitor:
        """Register an SLA monitor.

        Args:
            name: Unique name for the monitor.
            config: SLA configuration.
            replace: Whether to replace existing.

        Returns:
            SLAMonitor: The registered monitor.

        Raises:
            ValueError: If name exists and replace is False.
        """
        with self._lock:
            if name in self._monitors and not replace:
                msg = f"SLA monitor '{name}' already registered"
                raise ValueError(msg)

            monitor = SLAMonitor(config=config, name=name)
            self._monitors[name] = monitor
            return monitor

    def get(self, name: str) -> SLAMonitor | None:
        """Get a monitor by name.

        Args:
            name: Monitor name.

        Returns:
            SLAMonitor | None: The monitor or None.
        """
        with self._lock:
            return self._monitors.get(name)

    def get_or_create(
        self,
        name: str,
        config: SLAConfig | None = None,
    ) -> SLAMonitor:
        """Get or create a monitor.

        Args:
            name: Monitor name.
            config: Configuration if creating new.

        Returns:
            SLAMonitor: The monitor.
        """
        with self._lock:
            if name in self._monitors:
                return self._monitors[name]

            actual_config = config or SLAConfig()
            monitor = SLAMonitor(config=actual_config, name=name)
            self._monitors[name] = monitor
            return monitor

    def unregister(self, name: str) -> bool:
        """Unregister a monitor.

        Args:
            name: Monitor name.

        Returns:
            bool: True if removed, False if not found.
        """
        with self._lock:
            if name in self._monitors:
                del self._monitors[name]
                return True
            return False

    def list_names(self) -> list[str]:
        """List all registered monitor names.

        Returns:
            list[str]: List of names.
        """
        with self._lock:
            return list(self._monitors.keys())

    def get_all(self) -> dict[str, SLAMonitor]:
        """Get all monitors.

        Returns:
            dict[str, SLAMonitor]: Copy of monitors dict.
        """
        with self._lock:
            return dict(self._monitors)

    def check_all(
        self,
        metrics_by_name: dict[str, SLAMetrics],
    ) -> dict[str, list[SLAViolation]]:
        """Check metrics against all matching monitors.

        Args:
            metrics_by_name: Metrics keyed by monitor name.

        Returns:
            dict[str, list[SLAViolation]]: Violations by monitor.
        """
        results: dict[str, list[SLAViolation]] = {}

        with self._lock:
            monitors = dict(self._monitors)

        for name, metrics in metrics_by_name.items():
            monitor = monitors.get(name)
            if monitor:
                results[name] = monitor.check(metrics)

        return results

    def reset_all(self) -> None:
        """Reset all monitors."""
        with self._lock:
            for monitor in self._monitors.values():
                monitor.reset()

    def get_summary_all(self) -> dict[str, dict[str, Any]]:
        """Get summaries from all monitors.

        Returns:
            dict[str, dict[str, Any]]: Summaries by monitor name.
        """
        with self._lock:
            return {name: monitor.get_summary() for name, monitor in self._monitors.items()}


# Global registry singleton
_global_registry: SLARegistry | None = None
_registry_lock = threading.Lock()


def get_sla_registry() -> SLARegistry:
    """Get the global SLA registry.

    Returns:
        SLARegistry: The global registry.
    """
    global _global_registry
    with _registry_lock:
        if _global_registry is None:
            _global_registry = SLARegistry()
        return _global_registry


def reset_sla_registry() -> None:
    """Reset the global SLA registry."""
    global _global_registry
    with _registry_lock:
        _global_registry = None
