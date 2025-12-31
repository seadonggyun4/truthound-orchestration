"""SLA Resource for Dagster Integration.

This module provides a Dagster resource for SLA monitoring that can
be injected into ops and assets.

Example:
    >>> from dagster import Definitions
    >>> from truthound_dagster.sla import SLAResource, SLAConfig
    >>>
    >>> defs = Definitions(
    ...     resources={
    ...         "sla": SLAResource(
    ...             default_config=SLAConfig(max_failure_rate=0.05),
    ...         ),
    ...     },
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dagster import ConfigurableResource, InitResourceContext

from truthound_dagster.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
)
from truthound_dagster.sla.monitor import SLAMonitor, SLARegistry

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class SLAResourceConfig:
    """Configuration for SLA resource.

    Attributes:
        default_config: Default SLA configuration.
        enabled: Whether SLA monitoring is enabled.
        alert_on_violation: Whether to raise on violations.
        log_violations: Whether to log violations.

    Example:
        >>> config = SLAResourceConfig(
        ...     default_config=SLAConfig(max_failure_rate=0.05),
        ...     alert_on_violation=True,
        ... )
    """

    default_config: SLAConfig = field(default_factory=SLAConfig)
    enabled: bool = True
    alert_on_violation: bool = False
    log_violations: bool = True

    def with_default_config(self, config: SLAConfig) -> SLAResourceConfig:
        """Return new config with updated default SLA config."""
        return SLAResourceConfig(
            default_config=config,
            enabled=self.enabled,
            alert_on_violation=self.alert_on_violation,
            log_violations=self.log_violations,
        )

    def with_enabled(self, enabled: bool) -> SLAResourceConfig:
        """Return new config with updated enabled status."""
        return SLAResourceConfig(
            default_config=self.default_config,
            enabled=enabled,
            alert_on_violation=self.alert_on_violation,
            log_violations=self.log_violations,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_config": self.default_config.to_dict(),
            "enabled": self.enabled,
            "alert_on_violation": self.alert_on_violation,
            "log_violations": self.log_violations,
        }


class SLAResource(ConfigurableResource):
    """Dagster resource for SLA monitoring.

    This resource provides SLA monitoring capabilities that can be
    used in ops and assets to track quality metrics and detect violations.

    Parameters
    ----------
    max_failure_rate : float | None
        Default maximum failure rate.

    min_pass_rate : float | None
        Default minimum pass rate.

    max_execution_time_seconds : float | None
        Default maximum execution time.

    max_consecutive_failures : int
        Default maximum consecutive failures.

    alert_level : str
        Default alert level (info, warning, error, critical).

    enabled : bool
        Whether SLA monitoring is enabled.

    Example:
        >>> from dagster import Definitions, asset
        >>> from truthound_dagster.sla import SLAResource
        >>>
        >>> @asset
        ... def users(context, sla: SLAResource):
        ...     data = load_users()
        ...     metrics = SLAMetrics(passed_count=10, failed_count=1)
        ...     violations = sla.check(metrics)
        ...     return data
        >>>
        >>> defs = Definitions(
        ...     assets=[users],
        ...     resources={"sla": SLAResource()},
        ... )
    """

    # Dagster configuration fields
    max_failure_rate: float | None = None
    min_pass_rate: float | None = None
    max_execution_time_seconds: float | None = None
    max_consecutive_failures: int = 3
    alert_level: str = "error"
    enabled: bool = True

    # Internal state
    _registry: SLARegistry | None = None

    def setup_for_execution(self, context: InitResourceContext) -> None:
        """Set up the resource for execution.

        Args:
            context: Dagster initialization context.
        """
        self._registry = SLARegistry()

    def teardown_after_execution(self, context: InitResourceContext) -> None:
        """Tear down the resource after execution.

        Args:
            context: Dagster initialization context.
        """
        self._registry = None

    @property
    def registry(self) -> SLARegistry:
        """Get the SLA registry.

        Returns:
            SLARegistry: The registry.

        Raises:
            RuntimeError: If resource not initialized.
        """
        if self._registry is None:
            self._registry = SLARegistry()
        return self._registry

    def _get_alert_level(self) -> AlertLevel:
        """Convert string alert level to enum."""
        level_map = {
            "info": AlertLevel.INFO,
            "warning": AlertLevel.WARNING,
            "error": AlertLevel.ERROR,
            "critical": AlertLevel.CRITICAL,
        }
        return level_map.get(self.alert_level.lower(), AlertLevel.ERROR)

    def _get_default_config(self) -> SLAConfig:
        """Build default SLA config from resource settings."""
        return SLAConfig(
            max_failure_rate=self.max_failure_rate,
            min_pass_rate=self.min_pass_rate,
            max_execution_time_seconds=self.max_execution_time_seconds,
            max_consecutive_failures=self.max_consecutive_failures,
            alert_level=self._get_alert_level(),
            enabled=self.enabled,
        )

    def get_monitor(
        self,
        name: str,
        config: SLAConfig | None = None,
    ) -> SLAMonitor:
        """Get or create an SLA monitor.

        Args:
            name: Monitor name.
            config: Optional custom configuration.

        Returns:
            SLAMonitor: The monitor.
        """
        actual_config = config or self._get_default_config()
        return self.registry.get_or_create(name, actual_config)

    def check(
        self,
        metrics: SLAMetrics,
        name: str | None = None,
        config: SLAConfig | None = None,
    ) -> list[SLAViolation]:
        """Check metrics against SLA.

        Args:
            metrics: Metrics to check.
            name: Optional monitor name for tracking.
            config: Optional custom configuration.

        Returns:
            list[SLAViolation]: Detected violations.
        """
        if not self.enabled:
            return []

        actual_config = config or self._get_default_config()
        monitor_name = name or "default"
        monitor = self.registry.get_or_create(monitor_name, actual_config)

        return monitor.check(metrics)

    def check_result(
        self,
        result: dict[str, Any],
        name: str | None = None,
        config: SLAConfig | None = None,
        asset_key: str | None = None,
        run_id: str | None = None,
    ) -> list[SLAViolation]:
        """Check a quality check result against SLA.

        Args:
            result: Quality check result dictionary.
            name: Optional monitor name.
            config: Optional custom configuration.
            asset_key: Dagster asset key.
            run_id: Dagster run ID.

        Returns:
            list[SLAViolation]: Detected violations.
        """
        metrics = SLAMetrics.from_check_result(
            result=result,
            asset_key=asset_key,
            run_id=run_id,
        )
        return self.check(metrics, name=name, config=config)

    def record_success(self, name: str = "default") -> None:
        """Record a successful operation.

        Args:
            name: Monitor name.
        """
        monitor = self.registry.get(name)
        if monitor:
            monitor.record_success()

    def record_failure(self, name: str = "default") -> int:
        """Record a failed operation.

        Args:
            name: Monitor name.

        Returns:
            int: Current consecutive failure count.
        """
        monitor = self.registry.get(name)
        if monitor:
            return monitor.record_failure()
        return 0

    def get_summary(self, name: str | None = None) -> dict[str, Any]:
        """Get monitor summary.

        Args:
            name: Monitor name. If None, returns all summaries.

        Returns:
            dict[str, Any]: Summary data.
        """
        if name is None:
            return self.registry.get_summary_all()

        monitor = self.registry.get(name)
        if monitor:
            return monitor.get_summary()
        return {}

    def reset(self, name: str | None = None) -> None:
        """Reset monitor state.

        Args:
            name: Monitor name. If None, resets all.
        """
        if name is None:
            self.registry.reset_all()
        else:
            monitor = self.registry.get(name)
            if monitor:
                monitor.reset()
