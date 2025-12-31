"""SLA and Callback System for Data Quality Operations.

This module provides SLA (Service Level Agreement) management and
callback mechanisms for data quality operations in Apache Airflow.

Components:
    - SLAConfig: Immutable SLA configuration
    - SLAMonitor: SLA tracking and violation detection
    - DataQualitySLACallback: Callback for SLA miss handling
    - QualityAlertCallback: Alert callback for quality failures

Example:
    >>> from truthound_airflow.sla import (
    ...     SLAConfig,
    ...     DataQualitySLACallback,
    ... )
    >>>
    >>> sla_config = SLAConfig(
    ...     max_failure_rate=0.05,
    ...     min_pass_rate=0.95,
    ...     max_execution_time_seconds=300.0,
    ... )
    >>>
    >>> callback = DataQualitySLACallback(config=sla_config)
"""

from truthound_airflow.sla.config import (
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
)
from truthound_airflow.sla.monitor import (
    SLAMonitor,
    SLARegistry,
)
from truthound_airflow.sla.callbacks import (
    BaseDataQualityCallback,
    DataQualitySLACallback,
    QualityAlertCallback,
    CallbackChain,
)

__all__ = [
    # Configuration
    "AlertLevel",
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    "SLAViolationType",
    # Monitoring
    "SLAMonitor",
    "SLARegistry",
    # Callbacks
    "BaseDataQualityCallback",
    "DataQualitySLACallback",
    "QualityAlertCallback",
    "CallbackChain",
]
