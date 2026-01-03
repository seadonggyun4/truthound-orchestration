"""SLA monitoring for Mage data quality operations.

This module provides SLA monitoring capabilities for data quality
operations in Mage AI pipelines.

Components:
    - SLAConfig: Immutable configuration for SLA thresholds
    - SLAMetrics: Metrics captured for SLA evaluation
    - SLAViolation: Represents an SLA violation
    - SLAMonitor: Monitors metrics against SLA thresholds
    - SLAHooks: Lifecycle hooks for SLA events
"""

from truthound_mage.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
    DEFAULT_SLA_CONFIG,
    STRICT_SLA_CONFIG,
    LENIENT_SLA_CONFIG,
    PRODUCTION_SLA_CONFIG,
)

from truthound_mage.sla.monitor import (
    SLAMonitor,
    SLARegistry,
)

from truthound_mage.sla.hooks import (
    BaseSLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
)

__all__ = [
    # Config
    "AlertLevel",
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    "SLAViolationType",
    # Presets
    "DEFAULT_SLA_CONFIG",
    "STRICT_SLA_CONFIG",
    "LENIENT_SLA_CONFIG",
    "PRODUCTION_SLA_CONFIG",
    # Monitor
    "SLAMonitor",
    "SLARegistry",
    # Hooks
    "BaseSLAHook",
    "LoggingSLAHook",
    "MetricsSLAHook",
    "CompositeSLAHook",
]
