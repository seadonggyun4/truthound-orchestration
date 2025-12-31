"""Data Quality Hooks for Apache Airflow.

This module provides hooks for data loading, connection management,
and integration with external data sources in data quality operations.

Available Hooks:
    - DataQualityHook: Unified data loading and connection management

Protocols:
    - DataLoader: Protocol for custom data loading implementations
    - DataWriter: Protocol for custom data writing implementations

Configuration:
    - ConnectionConfig: Connection configuration dataclass

Legacy Aliases:
    - TruthoundHook -> DataQualityHook

Example:
    >>> from truthound_airflow.hooks import DataQualityHook
    >>>
    >>> hook = DataQualityHook(connection_id="my_s3")
    >>> data = hook.load_data("s3://bucket/data.parquet")
    >>>
    >>> # Or with SQL query
    >>> hook = DataQualityHook(connection_id="my_postgres")
    >>> data = hook.query("SELECT * FROM users WHERE active = true")
"""

from truthound_airflow.hooks.base import (
    ConnectionConfig,
    DataLoader,
    DataQualityHook,
    DataWriter,
    TruthoundHook,
)

__all__ = [
    # Protocols
    "DataLoader",
    "DataWriter",
    # Configuration
    "ConnectionConfig",
    # Hook
    "DataQualityHook",
    # Legacy alias
    "TruthoundHook",
]
