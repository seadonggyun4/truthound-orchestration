"""Apache Airflow Provider for Data Quality Operations.

truthound-airflow provides operators, sensors, hooks, and utilities
for data quality operations in Apache Airflow. The package is engine-agnostic,
supporting Truthound, Great Expectations, Pandera, and custom engines.

Key Components:
    - Operators: Execute data quality checks, profiling, and learning
    - Sensors: Wait for quality conditions to be met
    - Hooks: Load data and manage connections
    - SLA: Monitor and alert on SLA violations
    - Utils: Serialization and connection utilities

Example:
    >>> from airflow import DAG
    >>> from truthound_airflow import (
    ...     DataQualityCheckOperator,
    ...     DataQualitySensor,
    ... )
    >>>
    >>> with DAG("quality_pipeline", ...) as dag:
    ...     check = DataQualityCheckOperator(
    ...         task_id="check_quality",
    ...         rules=[{"column": "id", "type": "not_null"}],
    ...         data_path="s3://bucket/data.parquet",
    ...     )
"""

from truthound_airflow.version import __version__, __version_tuple__

# =============================================================================
# Operators
# =============================================================================

from truthound_airflow.operators import (
    # Base
    BaseDataQualityOperator,
    OperatorConfig,
    CheckOperatorConfig,
    ProfileOperatorConfig,
    LearnOperatorConfig,
    ResultHandler,
    # Check
    DataQualityCheckOperator,
    TruthoundCheckOperator,
    # Profile
    DataQualityProfileOperator,
    TruthoundProfileOperator,
    # Learn
    DataQualityLearnOperator,
    TruthoundLearnOperator,
)

# =============================================================================
# Sensors
# =============================================================================

from truthound_airflow.sensors import (
    DataQualitySensor,
    SensorConfig,
    TruthoundSensor,
)

# =============================================================================
# Hooks
# =============================================================================

from truthound_airflow.hooks import (
    DataQualityHook,
    DataLoader,
    DataWriter,
    ConnectionConfig,
    TruthoundHook,
)

# =============================================================================
# SLA
# =============================================================================

from truthound_airflow.sla import (
    # Config
    AlertLevel,
    SLAConfig,
    SLAMetrics,
    SLAViolation,
    SLAViolationType,
    # Monitor
    SLAMonitor,
    SLARegistry,
    # Callbacks
    BaseDataQualityCallback,
    DataQualitySLACallback,
    QualityAlertCallback,
    CallbackChain,
)

# =============================================================================
# Provider Info
# =============================================================================


def get_provider_info() -> dict:
    """Return Airflow Provider metadata.

    This function is registered as an entry point and called by
    Airflow to discover provider information.

    Returns:
        dict: Provider metadata.
    """
    return {
        "package-name": "truthound-airflow",
        "name": "Data Quality",
        "description": "Apache Airflow provider for data quality operations",
        "connection-types": [
            {
                "connection-type": "truthound",
                "hook-class-name": "truthound_airflow.hooks.base.DataQualityHook",
            }
        ],
        "versions": [__version__],
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__version_tuple__",
    # Provider
    "get_provider_info",
    # Operators - Base
    "BaseDataQualityOperator",
    "OperatorConfig",
    "CheckOperatorConfig",
    "ProfileOperatorConfig",
    "LearnOperatorConfig",
    "ResultHandler",
    # Operators - Check
    "DataQualityCheckOperator",
    "TruthoundCheckOperator",
    # Operators - Profile
    "DataQualityProfileOperator",
    "TruthoundProfileOperator",
    # Operators - Learn
    "DataQualityLearnOperator",
    "TruthoundLearnOperator",
    # Sensors
    "DataQualitySensor",
    "SensorConfig",
    "TruthoundSensor",
    # Hooks
    "DataQualityHook",
    "DataLoader",
    "DataWriter",
    "ConnectionConfig",
    "TruthoundHook",
    # SLA - Config
    "AlertLevel",
    "SLAConfig",
    "SLAMetrics",
    "SLAViolation",
    "SLAViolationType",
    # SLA - Monitor
    "SLAMonitor",
    "SLARegistry",
    # SLA - Callbacks
    "BaseDataQualityCallback",
    "DataQualitySLACallback",
    "QualityAlertCallback",
    "CallbackChain",
]
