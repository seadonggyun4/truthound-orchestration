"""SLA monitoring implementation.

This module provides thread-safe SLA monitoring with configurable
thresholds and violation detection.
"""

from __future__ import annotations

import threading
from typing import Any

from truthound_prefect.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)


class SLAMonitor:
    """Thread-safe SLA monitor.

    Monitors metrics against configured thresholds and tracks
    consecutive failures.

    Attributes:
        config: The SLA configuration.
        name: Optional name for the monitor.

    Example:
        >>> config = SLAConfig(max_failure_rate=0.05)
        >>> monitor = SLAMonitor(config, name="users_table")
        >>> violations = monitor.check(metrics)
        >>> if violations:
        ...     print(f"SLA violated: {violations[0].message}")
    """

    def __init__(
        self,
        config: SLAConfig,
        name: str | None = None,
        hooks: list[Any] | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            config: SLA configuration.
            name: Optional name for identification.
            hooks: Optional list of SLAHook instances.
        """
        self._config = config
        self._name = name or "default"
        self._hooks = hooks or []
        self._lock = threading.Lock()

        # State
        self._consecutive_failures = 0
        self._total_checks = 0
        self._total_violations = 0
        self._history: list[SLAMetrics] = []
        self._max_history = 100

    @property
    def config(self) -> SLAConfig:
        """Get the current configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get the monitor name."""
        return self._name

    @property
    def consecutive_failures(self) -> int:
        """Get the current consecutive failure count."""
        with self._lock:
            return self._consecutive_failures

    @property
    def total_checks(self) -> int:
        """Get the total number of checks."""
        with self._lock:
            return self._total_checks

    @property
    def total_violations(self) -> int:
        """Get the total number of violations."""
        with self._lock:
            return self._total_violations

    def check(
        self,
        metrics: SLAMetrics,
        context: dict[str, Any] | None = None,
    ) -> list[SLAViolation]:
        """Check metrics against SLA configuration.

        Args:
            metrics: The metrics to check.
            context: Optional additional context.

        Returns:
            List of SLA violations (empty if no violations).
        """
        if not self._config.enabled:
            return []

        violations: list[SLAViolation] = []
        ctx = context or {}

        with self._lock:
            self._total_checks += 1
            self._history.append(metrics)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        # Check failure rate
        if self._config.max_failure_rate is not None:
            if metrics.failure_rate > self._config.max_failure_rate:
                violations.append(SLAViolation(
                    violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
                    message=(
                        f"Failure rate {metrics.failure_rate:.2%} exceeds "
                        f"maximum {self._config.max_failure_rate:.2%}"
                    ),
                    threshold=self._config.max_failure_rate,
                    actual=metrics.failure_rate,
                    alert_level=self._config.alert_level,
                    flow_name=metrics.flow_name,
                    task_name=metrics.task_name,
                    run_id=metrics.run_id,
                ))

        # Check pass rate
        if self._config.min_pass_rate is not None:
            if metrics.pass_rate < self._config.min_pass_rate:
                violations.append(SLAViolation(
                    violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
                    message=(
                        f"Pass rate {metrics.pass_rate:.2%} below "
                        f"minimum {self._config.min_pass_rate:.2%}"
                    ),
                    threshold=self._config.min_pass_rate,
                    actual=metrics.pass_rate,
                    alert_level=self._config.alert_level,
                    flow_name=metrics.flow_name,
                    task_name=metrics.task_name,
                    run_id=metrics.run_id,
                ))

        # Check execution time
        if self._config.max_execution_time_seconds is not None:
            if metrics.execution_time_seconds > self._config.max_execution_time_seconds:
                violations.append(SLAViolation(
                    violation_type=SLAViolationType.EXECUTION_TIME_EXCEEDED,
                    message=(
                        f"Execution time {metrics.execution_time_seconds:.2f}s exceeds "
                        f"maximum {self._config.max_execution_time_seconds:.2f}s"
                    ),
                    threshold=self._config.max_execution_time_seconds,
                    actual=metrics.execution_time_seconds,
                    alert_level=self._config.alert_level,
                    flow_name=metrics.flow_name,
                    task_name=metrics.task_name,
                    run_id=metrics.run_id,
                ))

        # Check row count bounds
        if metrics.row_count is not None:
            if self._config.min_row_count is not None:
                if metrics.row_count < self._config.min_row_count:
                    violations.append(SLAViolation(
                        violation_type=SLAViolationType.ROW_COUNT_BELOW_MINIMUM,
                        message=(
                            f"Row count {metrics.row_count:,} below "
                            f"minimum {self._config.min_row_count:,}"
                        ),
                        threshold=self._config.min_row_count,
                        actual=metrics.row_count,
                        alert_level=self._config.alert_level,
                        flow_name=metrics.flow_name,
                        task_name=metrics.task_name,
                        run_id=metrics.run_id,
                    ))

            if self._config.max_row_count is not None:
                if metrics.row_count > self._config.max_row_count:
                    violations.append(SLAViolation(
                        violation_type=SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM,
                        message=(
                            f"Row count {metrics.row_count:,} exceeds "
                            f"maximum {self._config.max_row_count:,}"
                        ),
                        threshold=self._config.max_row_count,
                        actual=metrics.row_count,
                        alert_level=self._config.alert_level,
                        flow_name=metrics.flow_name,
                        task_name=metrics.task_name,
                        run_id=metrics.run_id,
                    ))

        # Track consecutive failures
        if not metrics.is_success or violations:
            consecutive = self.record_failure()

            # Check consecutive failures threshold
            if self._config.max_consecutive_failures is not None:
                if consecutive >= self._config.max_consecutive_failures:
                    violations.append(SLAViolation(
                        violation_type=SLAViolationType.CONSECUTIVE_FAILURES,
                        message=(
                            f"Consecutive failures ({consecutive}) reached "
                            f"threshold ({self._config.max_consecutive_failures})"
                        ),
                        threshold=self._config.max_consecutive_failures,
                        actual=consecutive,
                        alert_level=AlertLevel.CRITICAL,
                        flow_name=metrics.flow_name,
                        task_name=metrics.task_name,
                        run_id=metrics.run_id,
                    ))

                    # Notify hooks about consecutive failures
                    for hook in self._hooks:
                        if hasattr(hook, "on_consecutive_failure"):
                            try:
                                hook.on_consecutive_failure(
                                    consecutive,
                                    self._config.max_consecutive_failures,
                                    context=ctx,
                                )
                            except Exception:
                                pass  # Ignore hook errors
        else:
            self.record_success()

        # Update violation count
        if violations:
            with self._lock:
                self._total_violations += len(violations)

        # Notify hooks
        for hook in self._hooks:
            try:
                hook.on_check(metrics, violations, context=ctx)
                if violations:
                    for violation in violations:
                        hook.on_violation(violation, context=ctx)
                else:
                    if hasattr(hook, "on_success"):
                        hook.on_success(metrics, context=ctx)
            except Exception:
                pass  # Ignore hook errors

        return violations

    def record_failure(self) -> int:
        """Record a failure and return the consecutive failure count."""
        with self._lock:
            self._consecutive_failures += 1
            return self._consecutive_failures

    def record_success(self) -> None:
        """Record a success and reset consecutive failures."""
        with self._lock:
            self._consecutive_failures = 0

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._consecutive_failures = 0
            self._total_checks = 0
            self._total_violations = 0
            self._history.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of monitor state."""
        with self._lock:
            history = list(self._history)

        avg_failure_rate = 0.0
        avg_execution_time = 0.0

        if history:
            avg_failure_rate = sum(m.failure_rate for m in history) / len(history)
            avg_execution_time = sum(m.execution_time_ms for m in history) / len(history)

        return {
            "name": self._name,
            "total_checks": self._total_checks,
            "total_violations": self._total_violations,
            "consecutive_failures": self._consecutive_failures,
            "average_failure_rate": avg_failure_rate,
            "average_execution_time_ms": avg_execution_time,
            "history_size": len(history),
            "config": self._config.to_dict(),
        }


class SLARegistry:
    """Registry for managing multiple SLA monitors.

    Thread-safe singleton registry for centralized SLA management.

    Example:
        >>> registry = SLARegistry()
        >>> registry.register("users", SLAConfig(max_failure_rate=0.05))
        >>> monitor = registry.get("users")
        >>> violations = monitor.check(metrics)
    """

    _instance: SLARegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> SLARegistry:
        """Create singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._monitors: dict[str, SLAMonitor] = {}
                cls._instance._registry_lock = threading.Lock()
            return cls._instance

    def register(
        self,
        name: str,
        config: SLAConfig,
        replace: bool = False,
        hooks: list[Any] | None = None,
    ) -> SLAMonitor:
        """Register a new SLA monitor.

        Args:
            name: Unique name for the monitor.
            config: SLA configuration.
            replace: Whether to replace existing monitor.
            hooks: Optional list of hooks.

        Returns:
            The registered monitor.

        Raises:
            ValueError: If name exists and replace is False.
        """
        with self._registry_lock:
            if name in self._monitors and not replace:
                raise ValueError(f"Monitor '{name}' already exists")

            monitor = SLAMonitor(config, name=name, hooks=hooks)
            self._monitors[name] = monitor
            return monitor

    def get(self, name: str) -> SLAMonitor | None:
        """Get a monitor by name.

        Args:
            name: Name of the monitor.

        Returns:
            The monitor or None if not found.
        """
        with self._registry_lock:
            return self._monitors.get(name)

    def get_or_create(
        self,
        name: str,
        config: SLAConfig | None = None,
        hooks: list[Any] | None = None,
    ) -> SLAMonitor:
        """Get or create a monitor.

        Args:
            name: Name of the monitor.
            config: Configuration (used if creating).
            hooks: Hooks (used if creating).

        Returns:
            The monitor.
        """
        with self._registry_lock:
            if name not in self._monitors:
                cfg = config or SLAConfig()
                self._monitors[name] = SLAMonitor(cfg, name=name, hooks=hooks)
            return self._monitors[name]

    def unregister(self, name: str) -> bool:
        """Unregister a monitor.

        Args:
            name: Name of the monitor.

        Returns:
            True if the monitor was removed.
        """
        with self._registry_lock:
            if name in self._monitors:
                del self._monitors[name]
                return True
            return False

    def list_names(self) -> list[str]:
        """List all registered monitor names."""
        with self._registry_lock:
            return list(self._monitors.keys())

    def check_all(
        self,
        metrics_by_name: dict[str, SLAMetrics],
        context: dict[str, Any] | None = None,
    ) -> dict[str, list[SLAViolation]]:
        """Check multiple monitors at once.

        Args:
            metrics_by_name: Mapping of monitor names to metrics.
            context: Optional shared context.

        Returns:
            Mapping of monitor names to violations.
        """
        results: dict[str, list[SLAViolation]] = {}

        with self._registry_lock:
            monitors = dict(self._monitors)

        for name, metrics in metrics_by_name.items():
            monitor = monitors.get(name)
            if monitor:
                results[name] = monitor.check(metrics, context)

        return results

    def get_summary_all(self) -> dict[str, dict[str, Any]]:
        """Get summaries for all monitors."""
        with self._registry_lock:
            return {
                name: monitor.get_summary()
                for name, monitor in self._monitors.items()
            }

    def reset_all(self) -> None:
        """Reset all monitors."""
        with self._registry_lock:
            for monitor in self._monitors.values():
                monitor.reset()

    def clear(self) -> None:
        """Clear all monitors."""
        with self._registry_lock:
            self._monitors.clear()


# Global registry access
_global_registry: SLARegistry | None = None


def get_sla_registry() -> SLARegistry:
    """Get the global SLA registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SLARegistry()
    return _global_registry


def reset_sla_registry() -> None:
    """Reset the global SLA registry."""
    global _global_registry
    if _global_registry is not None:
        _global_registry.clear()
    _global_registry = None
    # Reset singleton
    SLARegistry._instance = None


__all__ = [
    "SLAMonitor",
    "SLARegistry",
    "get_sla_registry",
    "reset_sla_registry",
]
