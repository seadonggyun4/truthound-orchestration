"""SLA monitoring for data quality operations.

This package provides SLA (Service Level Agreement) monitoring
capabilities for tracking quality metrics and detecting violations.

Components:
    - SLAConfig: Immutable configuration for SLA thresholds
    - SLAMetrics: Metrics container for SLA evaluation
    - SLAViolation: Violation details when thresholds are exceeded
    - SLAMonitor: Thread-safe monitor with violation detection
    - SLARegistry: Centralized registry for multiple monitors
    - SLABlock: Prefect Block for SLA monitoring
    - Hooks: Event handlers for logging, metrics, etc.

Example:
    >>> from truthound_prefect.sla import SLAConfig, SLAMonitor, SLAMetrics
    >>>
    >>> # Configure SLA
    >>> config = SLAConfig(
    ...     max_failure_rate=0.05,
    ...     max_execution_time_seconds=300.0,
    ... )
    >>>
    >>> # Create monitor
    >>> monitor = SLAMonitor(config, name="users_table")
    >>>
    >>> # Check metrics
    >>> metrics = SLAMetrics.from_check_result(check_result)
    >>> violations = monitor.check(metrics)
    >>> if violations:
    ...     print(f"SLA violated: {violations[0].message}")
"""

from truthound_prefect.sla.block import SLABlock
from truthound_prefect.sla.config import (
    DEFAULT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)
from truthound_prefect.sla.hooks import (
    CallbackSLAHook,
    CompositeSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    SLAHook,
    SLAHookStats,
)
from truthound_prefect.sla.monitor import (
    SLAMonitor,
    SLARegistry,
    get_sla_registry,
    reset_sla_registry,
)

__all__ = [
    # Enums
    "AlertLevel",
    "SLAViolationType",
    # Data types
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    # Config presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
    # Monitor
    "SLAMonitor",
    "SLARegistry",
    "get_sla_registry",
    "reset_sla_registry",
    # Block
    "SLABlock",
    # Hooks
    "SLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CompositeSLAHook",
    "CallbackSLAHook",
    "SLAHookStats",
]
