"""SLA monitor for Kestra data quality integration.

This module provides the SLA monitoring implementation for
tracking and evaluating SLA compliance in Kestra workflows.

Example:
    >>> from truthound_kestra.sla.monitor import (
    ...     SLAMonitor,
    ...     SLARegistry,
    ... )
    >>>
    >>> monitor = SLAMonitor(config=SLAConfig(...))
    >>> result = monitor.evaluate(metrics)
    >>> if result.violations:
    ...     print(f"SLA violated: {result.violations}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from truthound_kestra.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)
from truthound_kestra.utils.exceptions import SLAViolationError
from truthound_kestra.utils.helpers import format_percentage, get_logger

__all__ = [
    # Protocols
    "SLAEvaluatorProtocol",
    # Classes
    "SLAMonitor",
    "SLARegistry",
    "SLAEvaluationResult",
    # Functions
    "evaluate_sla",
    "get_sla_registry",
    "register_sla",
]

logger = get_logger(__name__)


@runtime_checkable
class SLAEvaluatorProtocol(Protocol):
    """Protocol for SLA evaluators."""

    def evaluate(self, metrics: SLAMetrics) -> list[SLAViolation]:
        """Evaluate metrics against SLA."""
        ...


@dataclass
class SLAEvaluationResult:
    """Result of SLA evaluation.

    Attributes:
        config: SLA configuration used.
        metrics: Metrics that were evaluated.
        violations: List of SLA violations found.
        evaluated_at: When the evaluation occurred.
        consecutive_failures: Current consecutive failure count.

    Example:
        >>> result = monitor.evaluate(metrics)
        >>> if result.is_compliant:
        ...     print("SLA met!")
        >>> else:
        ...     for v in result.violations:
        ...         print(f"Violation: {v.message}")
    """

    config: SLAConfig
    metrics: SLAMetrics
    violations: list[SLAViolation] = field(default_factory=list)
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    consecutive_failures: int = 0

    @property
    def is_compliant(self) -> bool:
        """Check if SLA is compliant (no violations)."""
        return len(self.violations) == 0

    @property
    def has_critical_violations(self) -> bool:
        """Check if there are critical violations."""
        return any(v.alert_level == AlertLevel.CRITICAL for v in self.violations)

    @property
    def max_alert_level(self) -> AlertLevel | None:
        """Get the highest alert level among violations."""
        if not self.violations:
            return None
        return max(v.alert_level for v in self.violations)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "violations": [v.to_dict() for v in self.violations],
            "evaluated_at": self.evaluated_at.isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "is_compliant": self.is_compliant,
            "violation_count": len(self.violations),
        }


class SLAMonitor:
    """Monitor for evaluating SLA compliance.

    This class evaluates metrics against SLA thresholds and
    tracks consecutive failures.

    Attributes:
        config: SLA configuration.
        consecutive_failures: Current consecutive failure count.
        history: History of evaluation results.

    Example:
        >>> monitor = SLAMonitor(config=SLAConfig(
        ...     max_failure_rate=0.05,
        ...     min_pass_rate=0.95
        ... ))
        >>> result = monitor.evaluate(metrics)
    """

    def __init__(
        self,
        config: SLAConfig,
        hooks: list[Any] | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            config: SLA configuration.
            hooks: Optional list of hooks to notify.
        """
        self._config = config
        self._hooks = hooks or []
        self._consecutive_failures = 0
        self._history: list[SLAEvaluationResult] = []

    @property
    def config(self) -> SLAConfig:
        """Get SLA configuration."""
        return self._config

    @property
    def consecutive_failures(self) -> int:
        """Get current consecutive failure count."""
        return self._consecutive_failures

    @property
    def history(self) -> list[SLAEvaluationResult]:
        """Get evaluation history."""
        return self._history.copy()

    def evaluate(
        self,
        metrics: SLAMetrics,
        raise_on_violation: bool = False,
    ) -> SLAEvaluationResult:
        """Evaluate metrics against SLA.

        Args:
            metrics: Metrics to evaluate.
            raise_on_violation: Whether to raise exception on violation.

        Returns:
            SLAEvaluationResult with any violations found.

        Raises:
            SLAViolationError: If raise_on_violation is True and violations found.
        """
        if not self._config.enabled:
            return SLAEvaluationResult(
                config=self._config,
                metrics=metrics,
                consecutive_failures=self._consecutive_failures,
            )

        violations = []

        # Check failure rate
        if self._config.max_failure_rate is not None:
            if metrics.failure_rate > self._config.max_failure_rate:
                violations.append(SLAViolation(
                    violation_type=SLAViolationType.FAILURE_RATE_EXCEEDED,
                    message=(
                        f"Failure rate {format_percentage(metrics.failure_rate)} "
                        f"exceeds maximum {format_percentage(self._config.max_failure_rate)}"
                    ),
                    threshold=self._config.max_failure_rate,
                    actual=metrics.failure_rate,
                    alert_level=self._config.alert_level,
                ))

        # Check pass rate
        if self._config.min_pass_rate is not None:
            if metrics.pass_rate < self._config.min_pass_rate:
                violations.append(SLAViolation(
                    violation_type=SLAViolationType.PASS_RATE_BELOW_MINIMUM,
                    message=(
                        f"Pass rate {format_percentage(metrics.pass_rate)} "
                        f"below minimum {format_percentage(self._config.min_pass_rate)}"
                    ),
                    threshold=self._config.min_pass_rate,
                    actual=metrics.pass_rate,
                    alert_level=self._config.alert_level,
                ))

        # Check execution time
        if self._config.max_execution_time_seconds is not None:
            if metrics.execution_time_seconds > self._config.max_execution_time_seconds:
                violations.append(SLAViolation(
                    violation_type=SLAViolationType.EXECUTION_TIME_EXCEEDED,
                    message=(
                        f"Execution time {metrics.execution_time_seconds:.2f}s "
                        f"exceeds maximum {self._config.max_execution_time_seconds:.2f}s"
                    ),
                    threshold=self._config.max_execution_time_seconds,
                    actual=metrics.execution_time_seconds,
                    alert_level=self._config.alert_level,
                ))

        # Check row count
        if metrics.row_count is not None:
            if self._config.min_row_count is not None:
                if metrics.row_count < self._config.min_row_count:
                    violations.append(SLAViolation(
                        violation_type=SLAViolationType.ROW_COUNT_BELOW_MINIMUM,
                        message=(
                            f"Row count {metrics.row_count:,} "
                            f"below minimum {self._config.min_row_count:,}"
                        ),
                        threshold=self._config.min_row_count,
                        actual=metrics.row_count,
                        alert_level=self._config.alert_level,
                    ))

            if self._config.max_row_count is not None:
                if metrics.row_count > self._config.max_row_count:
                    violations.append(SLAViolation(
                        violation_type=SLAViolationType.ROW_COUNT_ABOVE_MAXIMUM,
                        message=(
                            f"Row count {metrics.row_count:,} "
                            f"exceeds maximum {self._config.max_row_count:,}"
                        ),
                        threshold=self._config.max_row_count,
                        actual=metrics.row_count,
                        alert_level=self._config.alert_level,
                    ))

        # Update consecutive failures
        if violations:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        # Check consecutive failures
        if self._consecutive_failures >= self._config.max_consecutive_failures:
            violations.append(SLAViolation(
                violation_type=SLAViolationType.CONSECUTIVE_FAILURES,
                message=(
                    f"{self._consecutive_failures} consecutive failures "
                    f"(max: {self._config.max_consecutive_failures})"
                ),
                threshold=self._config.max_consecutive_failures,
                actual=self._consecutive_failures,
                alert_level=AlertLevel.CRITICAL,
            ))

        # Create result
        result = SLAEvaluationResult(
            config=self._config,
            metrics=metrics,
            violations=violations,
            consecutive_failures=self._consecutive_failures,
        )

        # Store in history
        self._history.append(result)

        # Notify hooks
        self._notify_hooks(result)

        # Log result
        self._log_result(result)

        # Raise if requested
        if raise_on_violation and violations:
            raise SLAViolationError(
                message=f"SLA violated: {len(violations)} violation(s)",
                violations=violations,
            )

        return result

    def reset_consecutive_failures(self) -> None:
        """Reset the consecutive failure counter."""
        self._consecutive_failures = 0

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._history.clear()

    def _notify_hooks(self, result: SLAEvaluationResult) -> None:
        """Notify hooks of evaluation result."""
        for hook in self._hooks:
            try:
                if result.is_compliant:
                    if hasattr(hook, "on_sla_pass"):
                        hook.on_sla_pass(result)
                else:
                    if hasattr(hook, "on_sla_violation"):
                        hook.on_sla_violation(result)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__} failed: {e}")

    def _log_result(self, result: SLAEvaluationResult) -> None:
        """Log evaluation result."""
        if result.is_compliant:
            logger.info(
                f"SLA compliant: pass_rate={format_percentage(result.metrics.pass_rate)}, "
                f"execution_time={result.metrics.execution_time_seconds:.2f}s"
            )
        else:
            logger.warning(
                f"SLA violated: {len(result.violations)} violation(s), "
                f"consecutive_failures={result.consecutive_failures}"
            )
            for v in result.violations:
                logger.warning(f"  - {v.violation_type.value}: {v.message}")


class SLARegistry:
    """Registry for managing multiple SLA monitors.

    This class provides a centralized registry for SLA configurations
    and monitors, allowing easy management across flows and tasks.

    Example:
        >>> registry = SLARegistry()
        >>> registry.register("users_check", SLAConfig(max_failure_rate=0.05))
        >>> result = registry.evaluate("users_check", metrics)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._monitors: dict[str, SLAMonitor] = {}

    def register(
        self,
        name: str,
        config: SLAConfig,
        hooks: list[Any] | None = None,
    ) -> SLAMonitor:
        """Register an SLA configuration.

        Args:
            name: Unique name for the SLA.
            config: SLA configuration.
            hooks: Optional hooks for the monitor.

        Returns:
            The created SLAMonitor.
        """
        monitor = SLAMonitor(config, hooks)
        self._monitors[name] = monitor
        return monitor

    def get(self, name: str) -> SLAMonitor | None:
        """Get a monitor by name.

        Args:
            name: Monitor name.

        Returns:
            SLAMonitor or None if not found.
        """
        return self._monitors.get(name)

    def evaluate(
        self,
        name: str,
        metrics: SLAMetrics,
        raise_on_violation: bool = False,
    ) -> SLAEvaluationResult | None:
        """Evaluate metrics against a registered SLA.

        Args:
            name: SLA name.
            metrics: Metrics to evaluate.
            raise_on_violation: Whether to raise on violation.

        Returns:
            SLAEvaluationResult or None if SLA not found.
        """
        monitor = self._monitors.get(name)
        if monitor is None:
            logger.warning(f"SLA '{name}' not found in registry")
            return None
        return monitor.evaluate(metrics, raise_on_violation)

    def list_names(self) -> list[str]:
        """List all registered SLA names."""
        return list(self._monitors.keys())

    def remove(self, name: str) -> bool:
        """Remove an SLA from the registry.

        Args:
            name: SLA name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._monitors:
            del self._monitors[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered SLAs."""
        self._monitors.clear()

    def get_all_results(self) -> dict[str, list[SLAEvaluationResult]]:
        """Get all evaluation results from all monitors.

        Returns:
            Dictionary mapping SLA names to their evaluation histories.
        """
        return {
            name: monitor.history
            for name, monitor in self._monitors.items()
        }


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


def register_sla(
    name: str,
    config: SLAConfig,
    hooks: list[Any] | None = None,
) -> SLAMonitor:
    """Register an SLA in the global registry.

    Args:
        name: Unique name for the SLA.
        config: SLA configuration.
        hooks: Optional hooks.

    Returns:
        The created SLAMonitor.
    """
    return get_sla_registry().register(name, config, hooks)


def evaluate_sla(
    name: str,
    metrics: SLAMetrics,
    raise_on_violation: bool = False,
) -> SLAEvaluationResult | None:
    """Evaluate metrics against a registered SLA.

    Args:
        name: SLA name.
        metrics: Metrics to evaluate.
        raise_on_violation: Whether to raise on violation.

    Returns:
        SLAEvaluationResult or None if SLA not found.
    """
    return get_sla_registry().evaluate(name, metrics, raise_on_violation)


def reset_sla_registry() -> None:
    """Reset the global SLA registry.

    This clears all registered SLAs and resets the registry
    to a fresh state. Useful for testing.
    """
    global _global_registry
    _global_registry = SLARegistry()
