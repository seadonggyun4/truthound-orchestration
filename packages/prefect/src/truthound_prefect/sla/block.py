"""SLA Block for Prefect integration.

This module provides a Prefect Block for SLA monitoring that can be
saved, loaded, and used in Prefect flows and tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prefect.blocks.core import Block
from pydantic import Field

from truthound_prefect.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
)
from truthound_prefect.sla.hooks import CompositeSLAHook, LoggingSLAHook, MetricsSLAHook
from truthound_prefect.sla.monitor import SLAMonitor
from truthound_prefect.utils.exceptions import SLAViolationError

if TYPE_CHECKING:
    pass


class SLABlock(Block):
    """Prefect Block for SLA monitoring.

    This block provides SLA monitoring capabilities that can be
    saved, loaded, and used across Prefect deployments.

    Example:
        >>> # Create and save the block
        >>> block = SLABlock(
        ...     max_failure_rate=0.05,
        ...     max_execution_time_seconds=300.0,
        ... )
        >>> await block.save("production-sla")
        >>>
        >>> # Load and use in a flow
        >>> block = await SLABlock.load("production-sla")
        >>> violations = block.check(metrics)
    """

    _block_type_name = "SLA Monitor"
    _block_type_slug = "sla-monitor"
    _logo_url = "https://example.com/logo.png"
    _documentation_url = "https://github.com/truthound/truthound-orchestration"

    # Block configuration fields
    max_failure_rate: float | None = Field(
        default=None,
        description="Maximum allowed failure rate (0.0 to 1.0)",
    )
    min_pass_rate: float | None = Field(
        default=None,
        description="Minimum required pass rate (0.0 to 1.0)",
    )
    max_execution_time_seconds: float | None = Field(
        default=None,
        description="Maximum allowed execution time in seconds",
    )
    min_row_count: int | None = Field(
        default=None,
        description="Minimum required row count",
    )
    max_row_count: int | None = Field(
        default=None,
        description="Maximum allowed row count",
    )
    max_consecutive_failures: int | None = Field(
        default=3,
        description="Maximum allowed consecutive failures",
    )
    alert_on_warning: bool = Field(
        default=False,
        description="Whether to alert on warnings",
    )
    alert_level: str = Field(
        default="error",
        description="Default alert level (debug, info, warning, error, critical)",
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable logging of SLA events",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection",
    )
    raise_on_violation: bool = Field(
        default=False,
        description="Raise exception on SLA violation",
    )

    _monitor: SLAMonitor | None = None
    _metrics_hook: MetricsSLAHook | None = None

    def _get_config(self) -> SLAConfig:
        """Build SLA config from block settings."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            min_row_count=self.min_row_count,
            max_row_count=self.max_row_count,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_on_warning=self.alert_on_warning,
            alert_level=AlertLevel(self.alert_level),
            enabled=True,
        )

    def _get_monitor(self) -> SLAMonitor:
        """Get or create the SLA monitor."""
        if self._monitor is None:
            config = self._get_config()
            hooks = []

            if self.enable_logging:
                hooks.append(LoggingSLAHook())

            if self.enable_metrics:
                self._metrics_hook = MetricsSLAHook()
                hooks.append(self._metrics_hook)

            composite_hook = CompositeSLAHook(hooks) if hooks else None
            self._monitor = SLAMonitor(
                config,
                name=self._block_document_name or "sla-block",
                hooks=[composite_hook] if composite_hook else None,
            )

        return self._monitor

    def check(
        self,
        metrics: SLAMetrics | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[SLAViolation]:
        """Check metrics against SLA configuration.

        Args:
            metrics: The metrics to check (SLAMetrics or dict).
            context: Optional additional context.

        Returns:
            List of SLA violations.

        Raises:
            SLAViolationError: If raise_on_violation is True and violations occur.
        """
        monitor = self._get_monitor()

        # Convert dict to SLAMetrics if needed
        if isinstance(metrics, dict):
            metrics = SLAMetrics.from_check_result(metrics)

        violations = monitor.check(metrics, context)

        if violations and self.raise_on_violation:
            raise SLAViolationError(
                message=f"SLA violated with {len(violations)} violation(s)",
                violations=violations,
            )

        return violations

    def check_result(
        self,
        result: dict[str, Any],
        flow_name: str | None = None,
        task_name: str | None = None,
        run_id: str | None = None,
    ) -> list[SLAViolation]:
        """Check a serialized check result against SLA.

        Convenience method for checking quality check results.

        Args:
            result: Serialized check result dictionary.
            flow_name: Name of the flow.
            task_name: Name of the task.
            run_id: Prefect run ID.

        Returns:
            List of SLA violations.
        """
        metrics = SLAMetrics.from_check_result(
            result,
            flow_name=flow_name,
            task_name=task_name,
            run_id=run_id,
        )
        return self.check(metrics)

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary of metrics or empty dict if metrics disabled.
        """
        if self._metrics_hook:
            return self._metrics_hook.get_summary()
        return {}

    def get_summary(self) -> dict[str, Any]:
        """Get monitor summary.

        Returns:
            Dictionary with monitor state and configuration.
        """
        monitor = self._get_monitor()
        return monitor.get_summary()

    def reset(self) -> None:
        """Reset monitor state."""
        if self._monitor:
            self._monitor.reset()
        if self._metrics_hook:
            self._metrics_hook.reset()


__all__ = [
    "SLABlock",
]
