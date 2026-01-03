"""I/O configuration for Mage data quality operations.

This module provides integration with Mage's io_config.yaml for
configuring data sources and sinks for data quality operations.

Components:
    - IOConfig: Main I/O configuration class
    - DataSourceConfig: Data source configuration
    - DataSinkConfig: Data sink configuration
    - load_io_config: Load configuration from file
"""

from truthound_mage.io.config import (
    IOConfig,
    DataSourceConfig,
    DataSinkConfig,
    load_io_config,
    DEFAULT_IO_CONFIG,
)

__all__ = [
    "IOConfig",
    "DataSourceConfig",
    "DataSinkConfig",
    "load_io_config",
    "DEFAULT_IO_CONFIG",
]
