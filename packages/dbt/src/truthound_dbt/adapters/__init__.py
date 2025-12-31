"""SQL Adapter implementations for various database platforms.

This module provides protocol-based adapters for generating database-specific
SQL for data quality checks. Each adapter implements the SQLAdapter protocol,
ensuring consistent behavior across different database platforms.

Supported Databases:
    - PostgreSQL (default)
    - Snowflake
    - BigQuery
    - Redshift
    - Databricks

Architecture:
    The adapter system uses a registry pattern for extensibility:

    ```
    SQLAdapter (Protocol)
        ├── PostgresAdapter (default)
        ├── SnowflakeAdapter
        ├── BigQueryAdapter
        ├── RedshiftAdapter
        └── DatabricksAdapter
    ```

Usage:
    >>> from truthound_dbt.adapters import get_adapter, SQLAdapter
    >>>
    >>> # Get a registered adapter
    >>> adapter = get_adapter("snowflake")
    >>>
    >>> # Generate SQL for regex matching
    >>> sql = adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
    >>> # Returns: regexp_like(email, '^[\\w.-]+@[\\w.-]+$')
    >>>
    >>> # Generate SQL for unique check
    >>> sql = adapter.unique_check("users", "id")
    >>> # Returns Snowflake-optimized SQL with QUALIFY clause

Custom Adapter:
    >>> from truthound_dbt.adapters import SQLAdapter, register_adapter
    >>>
    >>> class MyCustomAdapter:
    ...     @property
    ...     def name(self) -> str:
    ...         return "custom"
    ...
    ...     def regex_match(self, column: str, pattern: str) -> str:
    ...         return f"{column} REGEXP '{pattern}'"
    ...
    ...     # ... implement other methods
    >>>
    >>> register_adapter("custom", MyCustomAdapter())
"""

from truthound_dbt.adapters.base import (
    # Protocol
    SQLAdapter,
    # Configuration
    AdapterConfig,
    DEFAULT_ADAPTER_CONFIG,
    # Registry
    AdapterRegistry,
    get_adapter,
    get_adapter_registry,
    register_adapter,
    list_adapters,
    reset_adapter_registry,
    # Exceptions
    AdapterError,
    AdapterNotFoundError,
    UnsupportedOperationError,
    SQLGenerationError,
)
from truthound_dbt.adapters.postgres import PostgresAdapter
from truthound_dbt.adapters.snowflake import SnowflakeAdapter
from truthound_dbt.adapters.bigquery import BigQueryAdapter
from truthound_dbt.adapters.redshift import RedshiftAdapter
from truthound_dbt.adapters.databricks import DatabricksAdapter

__all__ = [
    # Protocol
    "SQLAdapter",
    # Configuration
    "AdapterConfig",
    "DEFAULT_ADAPTER_CONFIG",
    # Implementations
    "PostgresAdapter",
    "SnowflakeAdapter",
    "BigQueryAdapter",
    "RedshiftAdapter",
    "DatabricksAdapter",
    # Registry
    "AdapterRegistry",
    "get_adapter",
    "get_adapter_registry",
    "register_adapter",
    "list_adapters",
    "reset_adapter_registry",
    # Exceptions
    "AdapterError",
    "AdapterNotFoundError",
    "UnsupportedOperationError",
    "SQLGenerationError",
]
