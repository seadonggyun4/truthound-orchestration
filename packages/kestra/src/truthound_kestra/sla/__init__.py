"""SLA monitoring modules for Kestra data quality integration.

This package provides SLA configuration, monitoring, and hook systems
for tracking data quality metrics against defined thresholds.

Modules:
    config: SLA configuration and thresholds.
    monitor: SLA monitoring and evaluation.
    hooks: Event hooks for SLA violations and passes.

Example:
    >>> from truthound_kestra.sla import (
    ...     SLAConfig,
    ...     SLAMonitor,
    ...     LoggingSLAHook,
    ... )
    >>>
    >>> config = SLAConfig(
    ...     min_pass_rate=0.95,
    ...     max_execution_time_seconds=300.0,
    ... )
    >>> hooks = [LoggingSLAHook()]
    >>> monitor = SLAMonitor(config, hooks=hooks)
    >>> result = monitor.evaluate(metrics)
"""

from truthound_kestra.sla.config import (
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
from truthound_kestra.sla.hooks import (
    BaseSLAHook,
    CallbackSLAHook,
    CompositeSLAHook,
    KestraNotificationHook,
    LoggingSLAHook,
    MetricsSLAHook,
    SLAHookProtocol,
)
from truthound_kestra.sla.monitor import (
    SLAEvaluationResult,
    SLAMonitor,
    SLARegistry,
    evaluate_sla,
    get_sla_registry,
    register_sla,
    reset_sla_registry,
)

__all__ = [
    # Enums
    "AlertLevel",
    "SLAViolationType",
    # Configuration
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    # Presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
    # Monitor
    "SLAMonitor",
    "SLAEvaluationResult",
    "SLARegistry",
    # Hooks - Protocol
    "SLAHookProtocol",
    # Hooks - Base
    "BaseSLAHook",
    # Hooks - Implementations
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CallbackSLAHook",
    "CompositeSLAHook",
    "KestraNotificationHook",
    # Functions
    "get_sla_registry",
    "reset_sla_registry",
    "register_sla",
    "evaluate_sla",
]
