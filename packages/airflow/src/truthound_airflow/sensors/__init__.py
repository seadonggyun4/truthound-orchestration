"""Data Quality Sensors for Apache Airflow.

This module provides sensors for waiting until data quality conditions
are met before proceeding with downstream tasks.

Available Sensors:
    - DataQualitySensor: Wait for quality conditions to be met

Legacy Aliases:
    - TruthoundSensor -> DataQualitySensor

Example:
    >>> from truthound_airflow.sensors import DataQualitySensor
    >>>
    >>> wait = DataQualitySensor(
    ...     task_id="wait_for_quality",
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     data_path="s3://bucket/data.parquet",
    ...     min_pass_rate=0.99,
    ... )
"""

from truthound_airflow.sensors.quality import (
    DataQualitySensor,
    SensorConfig,
    TruthoundSensor,
)

__all__ = [
    # Configuration
    "SensorConfig",
    # Engine-agnostic sensor
    "DataQualitySensor",
    # Legacy alias
    "TruthoundSensor",
]
