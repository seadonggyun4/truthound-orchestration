"""SLA Management for Data Quality Operations in Dagster.

This module provides SLA (Service Level Agreement) management and
monitoring for data quality operations in Dagster pipelines.

Components:
    - SLAConfig: Immutable SLA configuration
    - SLAMonitor: SLA tracking and violation detection
    - SLAResource: Dagster resource for SLA monitoring
    - DataQualitySensor: Sensor for SLA violation detection

Example:
    >>> from truthound_dagster.sla import (
    ...     SLAConfig,
    ...     SLAMonitor,
    ...     SLAResource,
    ... )
    >>>
    >>> sla_config = SLAConfig(
    ...     max_failure_rate=0.05,
    ...     min_pass_rate=0.95,
    ...     max_execution_time_seconds=300.0,
    ... )
    >>>
    >>> monitor = SLAMonitor(config=sla_config)
    >>> violations = monitor.check(metrics)
"""

from truthound_dagster.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
    # Presets
    DEFAULT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
)
from truthound_dagster.sla.monitor import (
    SLAMonitor,
    SLARegistry,
    get_sla_registry,
    reset_sla_registry,
)
from truthound_dagster.sla.resource import (
    SLAResource,
    SLAResourceConfig,
)
from truthound_dagster.sla.hooks import (
    SLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
)

__all__ = [
    # Configuration
    "AlertLevel",
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    "SLAViolationType",
    # Presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    # Monitoring
    "SLAMonitor",
    "SLARegistry",
    "get_sla_registry",
    "reset_sla_registry",
    # Resource
    "SLAResource",
    "SLAResourceConfig",
    # Hooks
    "SLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CompositeSLAHook",
]
