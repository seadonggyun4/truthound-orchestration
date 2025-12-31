"""Base adapter protocol and registry for SQL generation.

This module defines the core abstractions for database-specific SQL adapters.
All adapters implement the SQLAdapter protocol, ensuring consistent interfaces
across different database platforms.

The design follows these principles:
    1. Protocol-First: Uses structural typing for duck typing support
    2. Immutable Config: Thread-safe frozen dataclasses
    3. Registry Pattern: Centralized adapter management
    4. Extensibility: Easy to add new database adapters

Example:
    >>> from truthound_dbt.adapters.base import SQLAdapter, get_adapter
    >>>
    >>> adapter = get_adapter("snowflake")
    >>> sql = adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Exceptions
# =============================================================================


class AdapterError(Exception):
    """Base exception for adapter-related errors."""

    pass


class AdapterNotFoundError(AdapterError):
    """Raised when a requested adapter is not found in the registry."""

    def __init__(self, adapter_name: str) -> None:
        self.adapter_name = adapter_name
        super().__init__(f"Adapter not found: {adapter_name}")


class UnsupportedOperationError(AdapterError):
    """Raised when an adapter doesn't support a specific operation."""

    def __init__(self, adapter_name: str, operation: str) -> None:
        self.adapter_name = adapter_name
        self.operation = operation
        super().__init__(
            f"Adapter '{adapter_name}' does not support operation: {operation}"
        )


class SQLGenerationError(AdapterError):
    """Raised when SQL generation fails."""

    def __init__(self, message: str, adapter_name: str | None = None) -> None:
        self.adapter_name = adapter_name
        prefix = f"[{adapter_name}] " if adapter_name else ""
        super().__init__(f"{prefix}{message}")


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Configuration for SQL adapters.

    Attributes:
        quote_identifiers: Whether to quote table/column identifiers.
        identifier_quote_char: Character used for quoting identifiers.
        string_quote_char: Character used for quoting strings.
        null_safe_equals: Whether to use null-safe equality comparisons.
        case_sensitive: Whether identifiers are case-sensitive.
        escape_backslash: Whether to escape backslashes in strings.
    """

    quote_identifiers: bool = False
    identifier_quote_char: str = '"'
    string_quote_char: str = "'"
    null_safe_equals: bool = False
    case_sensitive: bool = True
    escape_backslash: bool = True

    def with_quote_identifiers(self, enabled: bool) -> AdapterConfig:
        """Return new config with updated quote_identifiers."""
        return AdapterConfig(
            quote_identifiers=enabled,
            identifier_quote_char=self.identifier_quote_char,
            string_quote_char=self.string_quote_char,
            null_safe_equals=self.null_safe_equals,
            case_sensitive=self.case_sensitive,
            escape_backslash=self.escape_backslash,
        )


DEFAULT_ADAPTER_CONFIG = AdapterConfig()


# =============================================================================
# SQLAdapter Protocol
# =============================================================================


@runtime_checkable
class SQLAdapter(Protocol):
    """Protocol for database-specific SQL generation.

    All database adapters must implement this protocol to provide
    consistent SQL generation across different database platforms.

    The protocol defines methods for:
        - String operations (length, regex, patterns)
        - Comparison operations (equality, ranges)
        - Temporal operations (current time, date parsing)
        - Set operations (in, not in)
        - Uniqueness checks
        - Null checks
        - Sampling and limiting

    Example:
        >>> class CustomAdapter:
        ...     @property
        ...     def name(self) -> str:
        ...         return "custom"
        ...
        ...     def regex_match(self, column: str, pattern: str) -> str:
        ...         return f"{column} REGEXP '{pattern}'"
    """

    @property
    def name(self) -> str:
        """Return the adapter name (e.g., 'postgres', 'snowflake')."""
        ...

    @property
    def config(self) -> AdapterConfig:
        """Return the adapter configuration."""
        ...

    # =========================================================================
    # String Operations
    # =========================================================================

    def length(self, column: str) -> str:
        """Generate SQL for string length.

        Args:
            column: Column name.

        Returns:
            SQL expression for string length.

        Example:
            >>> adapter.length("name")
            'length(name)'
        """
        ...

    def regex_match(self, column: str, pattern: str) -> str:
        """Generate SQL for regex pattern matching.

        Args:
            column: Column name.
            pattern: Regex pattern to match.

        Returns:
            SQL expression for regex matching (returns boolean).

        Example:
            >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
            "email ~ '^[\\w.-]+@[\\w.-]+$'"  # PostgreSQL
        """
        ...

    def concat(self, *parts: str) -> str:
        """Generate SQL for string concatenation.

        Args:
            *parts: Parts to concatenate (columns or literals).

        Returns:
            SQL expression for concatenation.

        Example:
            >>> adapter.concat("first_name", "' '", "last_name")
            "first_name || ' ' || last_name"  # PostgreSQL
        """
        ...

    def lower(self, column: str) -> str:
        """Generate SQL for lowercase conversion.

        Args:
            column: Column name.

        Returns:
            SQL expression for lowercase.

        Example:
            >>> adapter.lower("email")
            'lower(email)'
        """
        ...

    def upper(self, column: str) -> str:
        """Generate SQL for uppercase conversion.

        Args:
            column: Column name.

        Returns:
            SQL expression for uppercase.

        Example:
            >>> adapter.upper("name")
            'upper(name)'
        """
        ...

    def trim(self, column: str) -> str:
        """Generate SQL for trimming whitespace.

        Args:
            column: Column name.

        Returns:
            SQL expression for trimming.

        Example:
            >>> adapter.trim("name")
            'trim(name)'
        """
        ...

    # =========================================================================
    # Temporal Operations
    # =========================================================================

    def current_timestamp(self) -> str:
        """Generate SQL for current timestamp.

        Returns:
            SQL expression for current timestamp.

        Example:
            >>> adapter.current_timestamp()
            'current_timestamp'  # PostgreSQL
            'current_timestamp()'  # Snowflake
        """
        ...

    def current_date(self) -> str:
        """Generate SQL for current date.

        Returns:
            SQL expression for current date.

        Example:
            >>> adapter.current_date()
            'current_date'
        """
        ...

    def date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for parsing date from string.

        Args:
            column: Column name containing date string.
            format_str: Date format string (uses YYYY-MM-DD style).

        Returns:
            SQL expression for date parsing.

        Example:
            >>> adapter.date_parse("date_str", "YYYY-MM-DD")
            "to_date(date_str, 'YYYY-MM-DD')"
        """
        ...

    def try_date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for safe date parsing (returns NULL on failure).

        Args:
            column: Column name containing date string.
            format_str: Date format string.

        Returns:
            SQL expression for safe date parsing.

        Example:
            >>> adapter.try_date_parse("date_str", "YYYY-MM-DD")
            "try_to_date(date_str, 'YYYY-MM-DD')"  # Snowflake
        """
        ...

    # =========================================================================
    # Comparison Operations
    # =========================================================================

    def equals(self, column: str, value: Any) -> str:
        """Generate SQL for equality comparison.

        Args:
            column: Column name.
            value: Value to compare.

        Returns:
            SQL expression for equality.

        Example:
            >>> adapter.equals("status", "active")
            "status = 'active'"
        """
        ...

    def not_equals(self, column: str, value: Any) -> str:
        """Generate SQL for inequality comparison.

        Args:
            column: Column name.
            value: Value to compare.

        Returns:
            SQL expression for inequality.

        Example:
            >>> adapter.not_equals("status", "deleted")
            "status != 'deleted'"
        """
        ...

    def greater_than(self, column: str, value: Any) -> str:
        """Generate SQL for greater than comparison.

        Args:
            column: Column name.
            value: Value to compare.

        Returns:
            SQL expression for greater than.

        Example:
            >>> adapter.greater_than("age", 18)
            'age > 18'
        """
        ...

    def less_than(self, column: str, value: Any) -> str:
        """Generate SQL for less than comparison.

        Args:
            column: Column name.
            value: Value to compare.

        Returns:
            SQL expression for less than.

        Example:
            >>> adapter.less_than("age", 65)
            'age < 65'
        """
        ...

    def between(self, column: str, min_val: Any, max_val: Any) -> str:
        """Generate SQL for range check.

        Args:
            column: Column name.
            min_val: Minimum value (inclusive).
            max_val: Maximum value (inclusive).

        Returns:
            SQL expression for range check.

        Example:
            >>> adapter.between("age", 18, 65)
            'age between 18 and 65'
        """
        ...

    # =========================================================================
    # Set Operations
    # =========================================================================

    def in_set(self, column: str, values: list[Any]) -> str:
        """Generate SQL for set membership check.

        Args:
            column: Column name.
            values: List of values to check against.

        Returns:
            SQL expression for set membership.

        Example:
            >>> adapter.in_set("status", ["active", "pending"])
            "status in ('active', 'pending')"
        """
        ...

    def not_in_set(self, column: str, values: list[Any]) -> str:
        """Generate SQL for set non-membership check.

        Args:
            column: Column name.
            values: List of values to check against.

        Returns:
            SQL expression for set non-membership.

        Example:
            >>> adapter.not_in_set("status", ["deleted", "banned"])
            "status not in ('deleted', 'banned')"
        """
        ...

    # =========================================================================
    # Null Operations
    # =========================================================================

    def is_null(self, column: str) -> str:
        """Generate SQL for null check.

        Args:
            column: Column name.

        Returns:
            SQL expression for null check.

        Example:
            >>> adapter.is_null("email")
            'email is null'
        """
        ...

    def is_not_null(self, column: str) -> str:
        """Generate SQL for not null check.

        Args:
            column: Column name.

        Returns:
            SQL expression for not null check.

        Example:
            >>> adapter.is_not_null("email")
            'email is not null'
        """
        ...

    def coalesce(self, *columns: str) -> str:
        """Generate SQL for coalesce operation.

        Args:
            *columns: Columns to coalesce.

        Returns:
            SQL expression for coalesce.

        Example:
            >>> adapter.coalesce("nickname", "first_name", "'Unknown'")
            "coalesce(nickname, first_name, 'Unknown')"
        """
        ...

    # =========================================================================
    # Aggregation and Uniqueness
    # =========================================================================

    def unique_check_sql(self, model: str, column: str) -> str:
        """Generate SQL for uniqueness check that returns duplicate rows.

        Args:
            model: Table/model reference (e.g., "{{ ref('users') }}").
            column: Column to check for uniqueness.

        Returns:
            SQL query that selects rows with duplicate values.

        Example:
            >>> adapter.unique_check_sql("users", "email")
            '''
            select t.*
            from users t
            inner join (
                select email
                from users
                group by email
                having count(*) > 1
            ) duplicates
            on t.email = duplicates.email
            '''
        """
        ...

    def row_count(self, model: str) -> str:
        """Generate SQL for row count.

        Args:
            model: Table/model reference.

        Returns:
            SQL expression for row count.

        Example:
            >>> adapter.row_count("users")
            'select count(*) from users'
        """
        ...

    # =========================================================================
    # Sampling and Limiting
    # =========================================================================

    def limit_sample(self, n: int) -> str:
        """Generate SQL for random sampling.

        Args:
            n: Number of rows to sample.

        Returns:
            SQL clause for random sampling.

        Example:
            >>> adapter.limit_sample(1000)
            'order by random() limit 1000'  # PostgreSQL
            'sample (1000 rows)'  # Snowflake
        """
        ...

    def limit(self, n: int) -> str:
        """Generate SQL for limiting rows.

        Args:
            n: Number of rows to return.

        Returns:
            SQL clause for limiting.

        Example:
            >>> adapter.limit(100)
            'limit 100'
        """
        ...

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def quote_identifier(self, identifier: str) -> str:
        """Quote an identifier (table/column name).

        Args:
            identifier: Identifier to quote.

        Returns:
            Quoted identifier.

        Example:
            >>> adapter.quote_identifier("user")
            '"user"'
        """
        ...

    def quote_string(self, value: str) -> str:
        """Quote and escape a string value.

        Args:
            value: String value to quote.

        Returns:
            Quoted and escaped string.

        Example:
            >>> adapter.quote_string("it's a test")
            "'it''s a test'"
        """
        ...

    def format_value(self, value: Any) -> str:
        """Format a value for SQL.

        Args:
            value: Value to format.

        Returns:
            SQL-formatted value.

        Example:
            >>> adapter.format_value("active")
            "'active'"
            >>> adapter.format_value(42)
            '42'
            >>> adapter.format_value(None)
            'null'
        """
        ...


# =============================================================================
# Base Adapter Implementation
# =============================================================================


class BaseSQLAdapter(ABC):
    """Abstract base class providing common SQL adapter functionality.

    This class implements common operations that work across most databases,
    while leaving database-specific operations to concrete subclasses.

    Subclasses should override methods where database-specific syntax is needed.

    Attributes:
        _config: Adapter configuration.
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        """Initialize the adapter.

        Args:
            config: Adapter configuration. Uses default if not provided.
        """
        self._config = config or DEFAULT_ADAPTER_CONFIG

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter name."""
        ...

    @property
    def config(self) -> AdapterConfig:
        """Return the adapter configuration."""
        return self._config

    # =========================================================================
    # Common Implementations
    # =========================================================================

    def length(self, column: str) -> str:
        """Generate SQL for string length."""
        return f"length({column})"

    def concat(self, *parts: str) -> str:
        """Generate SQL for string concatenation using || operator."""
        return " || ".join(parts)

    def lower(self, column: str) -> str:
        """Generate SQL for lowercase conversion."""
        return f"lower({column})"

    def upper(self, column: str) -> str:
        """Generate SQL for uppercase conversion."""
        return f"upper({column})"

    def trim(self, column: str) -> str:
        """Generate SQL for trimming whitespace."""
        return f"trim({column})"

    def current_date(self) -> str:
        """Generate SQL for current date."""
        return "current_date"

    def equals(self, column: str, value: Any) -> str:
        """Generate SQL for equality comparison."""
        return f"{column} = {self.format_value(value)}"

    def not_equals(self, column: str, value: Any) -> str:
        """Generate SQL for inequality comparison."""
        return f"{column} != {self.format_value(value)}"

    def greater_than(self, column: str, value: Any) -> str:
        """Generate SQL for greater than comparison."""
        return f"{column} > {self.format_value(value)}"

    def less_than(self, column: str, value: Any) -> str:
        """Generate SQL for less than comparison."""
        return f"{column} < {self.format_value(value)}"

    def between(self, column: str, min_val: Any, max_val: Any) -> str:
        """Generate SQL for range check."""
        return f"{column} between {self.format_value(min_val)} and {self.format_value(max_val)}"

    def in_set(self, column: str, values: list[Any]) -> str:
        """Generate SQL for set membership check."""
        formatted = ", ".join(self.format_value(v) for v in values)
        return f"{column} in ({formatted})"

    def not_in_set(self, column: str, values: list[Any]) -> str:
        """Generate SQL for set non-membership check."""
        formatted = ", ".join(self.format_value(v) for v in values)
        return f"{column} not in ({formatted})"

    def is_null(self, column: str) -> str:
        """Generate SQL for null check."""
        return f"{column} is null"

    def is_not_null(self, column: str) -> str:
        """Generate SQL for not null check."""
        return f"{column} is not null"

    def coalesce(self, *columns: str) -> str:
        """Generate SQL for coalesce operation."""
        return f"coalesce({', '.join(columns)})"

    def row_count(self, model: str) -> str:
        """Generate SQL for row count."""
        return f"select count(*) from {model}"

    def limit(self, n: int) -> str:
        """Generate SQL for limiting rows."""
        return f"limit {n}"

    def quote_identifier(self, identifier: str) -> str:
        """Quote an identifier."""
        if not self._config.quote_identifiers:
            return identifier
        char = self._config.identifier_quote_char
        escaped = identifier.replace(char, char + char)
        return f"{char}{escaped}{char}"

    def quote_string(self, value: str) -> str:
        """Quote and escape a string value."""
        char = self._config.string_quote_char
        escaped = value.replace(char, char + char)
        if self._config.escape_backslash:
            escaped = escaped.replace("\\", "\\\\")
        return f"{char}{escaped}{char}"

    def format_value(self, value: Any) -> str:
        """Format a value for SQL."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return self.quote_string(value)
        return self.quote_string(str(value))

    # =========================================================================
    # Default implementations that subclasses typically override
    # =========================================================================

    def unique_check_sql(self, model: str, column: str) -> str:
        """Generate SQL for uniqueness check using subquery (default)."""
        return f"""select t.*
from {model} t
inner join (
    select {column}
    from {model}
    group by {column}
    having count(*) > 1
) duplicates
on t.{column} = duplicates.{column}"""

    def limit_sample(self, n: int) -> str:
        """Generate SQL for random sampling (default: order by random)."""
        return f"order by random() limit {n}"


# =============================================================================
# Adapter Registry
# =============================================================================


class AdapterRegistry:
    """Registry for managing SQL adapters.

    This registry provides centralized management of SQL adapters,
    supporting registration, retrieval, and listing of adapters.

    Thread-safe implementation using a lock for concurrent access.

    Example:
        >>> registry = AdapterRegistry()
        >>> registry.register("custom", CustomAdapter())
        >>> adapter = registry.get("custom")
    """

    _instance: AdapterRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> AdapterRegistry:
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._adapters = {}
                    cls._instance._default = "postgres"
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if self._initialized:
            return
        self._adapters: dict[str, SQLAdapter] = {}
        self._default: str = "postgres"
        self._initialized: bool = True

    def register(
        self,
        name: str,
        adapter: SQLAdapter,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an adapter.

        Args:
            name: Adapter name (lowercase).
            adapter: Adapter instance.
            overwrite: Whether to overwrite existing adapter.

        Raises:
            ValueError: If adapter already registered and overwrite is False.
        """
        name = name.lower()
        with self._lock:
            if name in self._adapters and not overwrite:
                msg = f"Adapter '{name}' already registered"
                raise ValueError(msg)
            self._adapters[name] = adapter

    def get(self, name: str | None = None) -> SQLAdapter:
        """Get an adapter by name.

        Args:
            name: Adapter name. Uses default if None.

        Returns:
            The requested adapter.

        Raises:
            AdapterNotFoundError: If adapter not found.
        """
        name = (name or self._default).lower()
        with self._lock:
            if name not in self._adapters:
                raise AdapterNotFoundError(name)
            return self._adapters[name]

    def list_adapters(self) -> list[str]:
        """List all registered adapter names.

        Returns:
            List of adapter names.
        """
        with self._lock:
            return list(self._adapters.keys())

    def set_default(self, name: str) -> None:
        """Set the default adapter.

        Args:
            name: Adapter name to set as default.

        Raises:
            AdapterNotFoundError: If adapter not found.
        """
        name = name.lower()
        with self._lock:
            if name not in self._adapters:
                raise AdapterNotFoundError(name)
            self._default = name

    def get_default(self) -> str:
        """Get the default adapter name.

        Returns:
            Default adapter name.
        """
        return self._default

    def reset(self) -> None:
        """Reset the registry to initial state."""
        with self._lock:
            self._adapters.clear()
            self._default = "postgres"
            self._initialized = False


# =============================================================================
# Global Registry Functions
# =============================================================================

_registry: AdapterRegistry | None = None
_registry_lock = threading.Lock()


def get_adapter_registry() -> AdapterRegistry:
    """Get the global adapter registry.

    Returns:
        The global AdapterRegistry instance.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = AdapterRegistry()
                _register_default_adapters(_registry)
    return _registry


def _register_default_adapters(registry: AdapterRegistry) -> None:
    """Register default adapters."""
    # Import here to avoid circular imports
    from truthound_dbt.adapters.postgres import PostgresAdapter
    from truthound_dbt.adapters.snowflake import SnowflakeAdapter
    from truthound_dbt.adapters.bigquery import BigQueryAdapter
    from truthound_dbt.adapters.redshift import RedshiftAdapter
    from truthound_dbt.adapters.databricks import DatabricksAdapter

    registry.register("postgres", PostgresAdapter())
    registry.register("postgresql", PostgresAdapter())
    registry.register("default", PostgresAdapter())
    registry.register("snowflake", SnowflakeAdapter())
    registry.register("bigquery", BigQueryAdapter())
    registry.register("redshift", RedshiftAdapter())
    registry.register("databricks", DatabricksAdapter())


def get_adapter(name: str | None = None) -> SQLAdapter:
    """Get an adapter by name from the global registry.

    Args:
        name: Adapter name. Uses default if None.

    Returns:
        The requested adapter.

    Raises:
        AdapterNotFoundError: If adapter not found.
    """
    return get_adapter_registry().get(name)


def register_adapter(
    name: str,
    adapter: SQLAdapter,
    *,
    overwrite: bool = False,
) -> None:
    """Register an adapter in the global registry.

    Args:
        name: Adapter name.
        adapter: Adapter instance.
        overwrite: Whether to overwrite existing adapter.
    """
    get_adapter_registry().register(name, adapter, overwrite=overwrite)


def list_adapters() -> list[str]:
    """List all registered adapters in the global registry.

    Returns:
        List of adapter names.
    """
    return get_adapter_registry().list_adapters()


def reset_adapter_registry() -> None:
    """Reset the global adapter registry."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.reset()
        _registry = None
