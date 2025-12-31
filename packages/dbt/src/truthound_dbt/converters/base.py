"""Base converter protocol and types for rule-to-SQL conversion.

This module defines the core abstractions for converting data quality rules
into SQL expressions. The design follows these principles:

1. Protocol-First: Uses structural typing for duck typing support
2. Adapter Integration: Works with SQLAdapters for database-specific SQL
3. Extensibility: Easy to add new rule types via handlers
4. Immutable Types: Thread-safe frozen dataclasses

Example:
    >>> from truthound_dbt.converters.base import RuleConverter, RuleSQL
    >>> from truthound_dbt.adapters import get_adapter
    >>>
    >>> class MyConverter:
    ...     def convert(self, rule, adapter):
    ...         return RuleSQL(
    ...             where_clause=f"{rule['column']} is null",
    ...             rule_type="not_null",
    ...             column=rule["column"],
    ...         )
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from truthound_dbt.adapters.base import SQLAdapter


# =============================================================================
# Exceptions
# =============================================================================


class ConversionError(Exception):
    """Base exception for rule conversion errors."""

    pass


class UnsupportedRuleError(ConversionError):
    """Raised when a rule type is not supported."""

    def __init__(self, rule_type: str, adapter_name: str | None = None) -> None:
        self.rule_type = rule_type
        self.adapter_name = adapter_name
        msg = f"Unsupported rule type: {rule_type}"
        if adapter_name:
            msg += f" for adapter: {adapter_name}"
        super().__init__(msg)


class InvalidRuleError(ConversionError):
    """Raised when a rule definition is invalid."""

    def __init__(self, rule_type: str, message: str) -> None:
        self.rule_type = rule_type
        self.message = message
        super().__init__(f"Invalid rule '{rule_type}': {message}")


# =============================================================================
# Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class RuleSQL:
    """SQL generated from a rule.

    This immutable type represents the SQL expression(s) generated from
    a single data quality rule.

    Attributes:
        where_clause: SQL WHERE clause that selects failing rows.
            When this clause matches rows, those rows are considered failures.
        rule_type: The type of rule that generated this SQL.
        column: The column being checked (if applicable).
        select_clause: Optional custom SELECT clause.
        from_clause: Optional FROM clause override.
        join_clause: Optional JOIN clause (for referential checks).
        metadata: Additional metadata about the rule.

    Example:
        >>> sql = RuleSQL(
        ...     where_clause="email is null",
        ...     rule_type="not_null",
        ...     column="email",
        ... )
    """

    where_clause: str
    rule_type: str
    column: str | None = None
    select_clause: str | None = None
    from_clause: str | None = None
    join_clause: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_where_clause(self, where_clause: str) -> RuleSQL:
        """Return new RuleSQL with updated where_clause."""
        return RuleSQL(
            where_clause=where_clause,
            rule_type=self.rule_type,
            column=self.column,
            select_clause=self.select_clause,
            from_clause=self.from_clause,
            join_clause=self.join_clause,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class ConversionContext:
    """Context for rule conversion.

    Provides additional information for rule conversion, including
    model reference and options.

    Attributes:
        model: The model/table reference (e.g., "{{ ref('users') }}").
        sample_size: Optional sample size for performance.
        fail_fast: Whether to stop at first failure.
        where_filter: Optional WHERE filter to apply to source data.
        tags: Tags to include in metadata.
    """

    model: str
    sample_size: int | None = None
    fail_fast: bool = False
    where_filter: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """Result of converting a list of rules.

    Attributes:
        rules_sql: List of RuleSQL for each converted rule.
        combined_sql: Combined SQL query selecting all failing rows.
        rule_count: Number of rules converted.
        conversion_errors: Any errors encountered during conversion.
        metadata: Additional metadata.
    """

    rules_sql: tuple[RuleSQL, ...]
    combined_sql: str
    rule_count: int
    conversion_errors: tuple[ConversionError, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        """Check if any conversion errors occurred."""
        return len(self.conversion_errors) > 0

    @property
    def errors(self) -> tuple[ConversionError, ...]:
        """Alias for conversion_errors for convenience."""
        return self.conversion_errors

    @property
    def warnings(self) -> list[str]:
        """Return any warnings (placeholder for future use)."""
        return self.metadata.get("warnings", [])

    @property
    def success_count(self) -> int:
        """Number of rules successfully converted."""
        return len(self.rules_sql)


# =============================================================================
# RuleConverter Protocol
# =============================================================================


@runtime_checkable
class RuleConverter(Protocol):
    """Protocol for rule-to-SQL converters.

    Converters transform data quality rule definitions into SQL expressions
    that can be executed against a database.

    The protocol defines two methods:
        - convert: Convert a single rule
        - convert_all: Convert multiple rules and combine them

    Example:
        >>> class MyConverter:
        ...     def convert(self, rule, adapter, context=None):
        ...         return RuleSQL(
        ...             where_clause=f"{rule['column']} is null",
        ...             rule_type="not_null",
        ...             column=rule.get("column"),
        ...         )
        ...
        ...     def convert_all(self, rules, adapter, context):
        ...         # ... implementation
    """

    def convert(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        """Convert a single rule to SQL.

        Args:
            rule: Rule definition dictionary.
            adapter: SQL adapter for database-specific syntax.
            context: Optional conversion context.

        Returns:
            RuleSQL containing the generated SQL.

        Raises:
            UnsupportedRuleError: If rule type is not supported.
            InvalidRuleError: If rule definition is invalid.
        """
        ...

    def convert_all(
        self,
        rules: list[dict[str, Any]],
        adapter: SQLAdapter,
        context: ConversionContext,
    ) -> ConversionResult:
        """Convert multiple rules and combine into a single query.

        Args:
            rules: List of rule definitions.
            adapter: SQL adapter for database-specific syntax.
            context: Conversion context with model and options.

        Returns:
            ConversionResult with all converted rules and combined SQL.
        """
        ...


# =============================================================================
# Registry
# =============================================================================


class RuleConverterRegistry:
    """Registry for managing rule converters.

    This registry provides centralized management of rule converters,
    supporting registration, retrieval, and listing.

    Thread-safe implementation using a lock for concurrent access.

    Example:
        >>> registry = RuleConverterRegistry()
        >>> registry.register("custom", CustomConverter())
        >>> converter = registry.get("custom")
    """

    _instance: RuleConverterRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> RuleConverterRegistry:
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._converters = {}
                    cls._instance._default = "standard"
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if self._initialized:
            return
        self._converters: dict[str, RuleConverter] = {}
        self._default: str = "standard"
        self._initialized: bool = True

    def register(
        self,
        name: str,
        converter: RuleConverter,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a converter.

        Args:
            name: Converter name.
            converter: Converter instance.
            overwrite: Whether to overwrite existing converter.

        Raises:
            ValueError: If converter already registered and overwrite is False.
        """
        with self._lock:
            if name in self._converters and not overwrite:
                msg = f"Converter '{name}' already registered"
                raise ValueError(msg)
            self._converters[name] = converter

    def get(self, name: str | None = None) -> RuleConverter:
        """Get a converter by name.

        Args:
            name: Converter name. Uses default if None.

        Returns:
            The requested converter.

        Raises:
            KeyError: If converter not found.
        """
        name = name or self._default
        with self._lock:
            if name not in self._converters:
                raise KeyError(f"Converter not found: {name}")
            return self._converters[name]

    def list_converters(self) -> list[str]:
        """List all registered converter names.

        Returns:
            List of converter names.
        """
        with self._lock:
            return list(self._converters.keys())

    def set_default(self, name: str) -> None:
        """Set the default converter.

        Args:
            name: Converter name to set as default.

        Raises:
            KeyError: If converter not found.
        """
        with self._lock:
            if name not in self._converters:
                raise KeyError(f"Converter not found: {name}")
            self._default = name

    def reset(self) -> None:
        """Reset the registry to initial state."""
        with self._lock:
            self._converters.clear()
            self._default = "standard"
            self._initialized = False


# =============================================================================
# Global Registry Functions
# =============================================================================

_registry: RuleConverterRegistry | None = None
_registry_lock = threading.Lock()


def get_converter_registry() -> RuleConverterRegistry:
    """Get the global converter registry.

    Returns:
        The global RuleConverterRegistry instance.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = RuleConverterRegistry()
                _register_default_converters(_registry)
    return _registry


def _register_default_converters(registry: RuleConverterRegistry) -> None:
    """Register default converters."""
    from truthound_dbt.converters.rules import StandardRuleConverter

    registry.register("standard", StandardRuleConverter())
    registry.register("default", StandardRuleConverter())


def get_converter(name: str | None = None) -> RuleConverter:
    """Get a converter by name from the global registry.

    Args:
        name: Converter name. Uses default if None.

    Returns:
        The requested converter.

    Raises:
        KeyError: If converter not found.
    """
    return get_converter_registry().get(name)


def register_converter(
    name: str,
    converter: RuleConverter,
    *,
    overwrite: bool = False,
) -> None:
    """Register a converter in the global registry.

    Args:
        name: Converter name.
        converter: Converter instance.
        overwrite: Whether to overwrite existing converter.
    """
    get_converter_registry().register(name, converter, overwrite=overwrite)


def list_converters() -> list[str]:
    """List all registered converters in the global registry.

    Returns:
        List of converter names.
    """
    return get_converter_registry().list_converters()


def reset_converter_registry() -> None:
    """Reset the global converter registry."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.reset()
        _registry = None
