"""Rule handlers and standard converter implementation.

This module provides the StandardRuleConverter and individual rule handlers
for converting data quality rules to SQL expressions.

Architecture:
    ```
    StandardRuleConverter
        │
        ├── RuleHandlerRegistry (maps rule types to handlers)
        │
        ├── NotNullHandler
        ├── UniqueHandler
        ├── InSetHandler
        ├── RangeHandler
        ├── RegexHandler
        ├── EmailFormatHandler
        ├── ...
        └── ExpressionHandler
    ```

Each handler is responsible for converting a specific rule type to SQL.
Handlers are registered in a registry, making it easy to add new rule types.

Example:
    >>> from truthound_dbt.converters import StandardRuleConverter
    >>> from truthound_dbt.adapters import get_adapter
    >>>
    >>> converter = StandardRuleConverter()
    >>> adapter = get_adapter("snowflake")
    >>>
    >>> result = converter.convert(
    ...     {"type": "not_null", "column": "email"},
    ...     adapter,
    ... )
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from truthound_dbt.converters.base import (
    ConversionContext,
    ConversionResult,
    InvalidRuleError,
    RuleSQL,
    UnsupportedRuleError,
)

if TYPE_CHECKING:
    from truthound_dbt.adapters.base import SQLAdapter


# =============================================================================
# Rule Handler Protocol
# =============================================================================


class RuleHandler(ABC):
    """Abstract base class for rule handlers.

    Each handler is responsible for converting a specific rule type to SQL.
    Handlers work with SQLAdapters to generate database-specific syntax.

    Subclasses must implement:
        - rule_types: List of rule types this handler supports
        - handle: Convert the rule to SQL

    Example:
        >>> class CustomHandler(RuleHandler):
        ...     @property
        ...     def rule_types(self):
        ...         return ["custom_check"]
        ...
        ...     def handle(self, rule, adapter, context):
        ...         return RuleSQL(
        ...             where_clause=f"custom_check({rule['column']})",
        ...             rule_type="custom_check",
        ...             column=rule["column"],
        ...         )
    """

    @property
    @abstractmethod
    def rule_types(self) -> list[str]:
        """Return the list of rule types this handler supports."""
        ...

    @abstractmethod
    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        """Convert a rule to SQL.

        Args:
            rule: Rule definition dictionary.
            adapter: SQL adapter for database-specific syntax.
            context: Optional conversion context.

        Returns:
            RuleSQL containing the generated SQL.

        Raises:
            InvalidRuleError: If rule definition is invalid.
        """
        ...

    def validate(self, rule: dict[str, Any]) -> None:
        """Validate a rule definition.

        Args:
            rule: Rule definition to validate.

        Raises:
            InvalidRuleError: If rule is invalid.
        """
        # Default implementation - subclasses can override
        pass

    def _get_rule_type(self, rule: dict[str, Any]) -> str:
        """Extract rule type from rule definition.

        Args:
            rule: Rule definition.

        Returns:
            Rule type string.
        """
        return rule.get("type", rule.get("check", "unknown"))

    def _get_column(self, rule: dict[str, Any]) -> str:
        """Extract column from rule definition.

        Args:
            rule: Rule definition.

        Returns:
            Column name.

        Raises:
            InvalidRuleError: If column is missing.
        """
        column = rule.get("column")
        if not column:
            raise InvalidRuleError(
                self._get_rule_type(rule),
                "Missing required 'column' field",
            )
        return column


# =============================================================================
# Individual Rule Handlers
# =============================================================================


class NotNullHandler(RuleHandler):
    """Handler for not_null and not_empty rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["not_null", "notnull", "not_empty", "notempty"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        rule_type = self._get_rule_type(rule)

        if rule_type in ("not_empty", "notempty"):
            # Check for both NULL and empty string
            where = f"{adapter.is_null(column)} or {adapter.trim(column)} = ''"
        else:
            # Just NULL check
            where = adapter.is_null(column)

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=column,
        )


class UniqueHandler(RuleHandler):
    """Handler for unique and unique_combination rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["unique", "unique_combination"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        rule_type = self._get_rule_type(rule)

        if rule_type == "unique_combination":
            columns = rule.get("columns", [])
            if not columns:
                raise InvalidRuleError(rule_type, "Missing required 'columns' field")
            column_list = ", ".join(columns)
            column = column_list
        else:
            column = self._get_column(rule)

        # For unique checks, we need a different approach
        # Return a placeholder that the generator will handle
        return RuleSQL(
            where_clause="__UNIQUE_CHECK__",  # Placeholder
            rule_type=rule_type,
            column=column,
            metadata={"requires_unique_subquery": True},
        )


class InSetHandler(RuleHandler):
    """Handler for in_set and not_in_set rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["in_set", "accepted_values", "not_in_set", "rejected_values"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        rule_type = self._get_rule_type(rule)
        values = rule.get("values", [])

        if not values:
            raise InvalidRuleError(rule_type, "Missing required 'values' field")

        if rule_type in ("in_set", "accepted_values"):
            # Failure: value is NOT in the set
            where = f"{adapter.is_not_null(column)} and not {adapter.in_set(column, values)}"
        else:
            # Failure: value IS in the set
            where = adapter.in_set(column, values)

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=column,
            metadata={"values": values},
        )


class RangeHandler(RuleHandler):
    """Handler for range, positive, negative, and comparison rules."""

    @property
    def rule_types(self) -> list[str]:
        return [
            "range",
            "in_range",
            "between",
            "positive",
            "negative",
            "non_negative",
            "non_positive",
            "greater_than",
            "gt",
            "less_than",
            "lt",
            "greater_than_or_equal",
            "gte",
            "less_than_or_equal",
            "lte",
        ]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        rule_type = self._get_rule_type(rule)

        conditions = [adapter.is_not_null(column)]

        if rule_type in ("range", "in_range", "between"):
            min_val = rule.get("min")
            max_val = rule.get("max")
            if min_val is not None:
                conditions.append(adapter.less_than(column, min_val))
            if max_val is not None:
                conditions.append(adapter.greater_than(column, max_val))
            # Combine with OR (either below min or above max is a failure)
            if len(conditions) > 1:
                where = f"{conditions[0]} and ({' or '.join(conditions[1:])})"
            else:
                where = conditions[0]
        elif rule_type == "positive":
            where = f"{conditions[0]} and {column} <= 0"
        elif rule_type == "negative":
            where = f"{conditions[0]} and {column} >= 0"
        elif rule_type == "non_negative":
            where = f"{conditions[0]} and {column} < 0"
        elif rule_type == "non_positive":
            where = f"{conditions[0]} and {column} > 0"
        elif rule_type in ("greater_than", "gt"):
            value = rule.get("value", rule.get("threshold"))
            if value is None:
                raise InvalidRuleError(rule_type, "Missing required 'value' field")
            where = f"{conditions[0]} and {column} <= {adapter.format_value(value)}"
        elif rule_type in ("less_than", "lt"):
            value = rule.get("value", rule.get("threshold"))
            if value is None:
                raise InvalidRuleError(rule_type, "Missing required 'value' field")
            where = f"{conditions[0]} and {column} >= {adapter.format_value(value)}"
        elif rule_type in ("greater_than_or_equal", "gte"):
            value = rule.get("value", rule.get("threshold"))
            if value is None:
                raise InvalidRuleError(rule_type, "Missing required 'value' field")
            where = f"{conditions[0]} and {column} < {adapter.format_value(value)}"
        elif rule_type in ("less_than_or_equal", "lte"):
            value = rule.get("value", rule.get("threshold"))
            if value is None:
                raise InvalidRuleError(rule_type, "Missing required 'value' field")
            where = f"{conditions[0]} and {column} > {adapter.format_value(value)}"
        else:
            where = conditions[0]

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=column,
        )


class LengthHandler(RuleHandler):
    """Handler for length rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["length", "min_length", "max_length"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        rule_type = self._get_rule_type(rule)

        conditions = [adapter.is_not_null(column)]

        if rule_type == "min_length":
            min_len = rule.get("min", rule.get("length"))
            if min_len is None:
                raise InvalidRuleError(rule_type, "Missing required 'min' or 'length' field")
            conditions.append(f"{adapter.length(column)} < {min_len}")
        elif rule_type == "max_length":
            max_len = rule.get("max", rule.get("length"))
            if max_len is None:
                raise InvalidRuleError(rule_type, "Missing required 'max' or 'length' field")
            conditions.append(f"{adapter.length(column)} > {max_len}")
        else:  # length
            min_len = rule.get("min")
            max_len = rule.get("max")
            length_conditions = []
            if min_len is not None:
                length_conditions.append(f"{adapter.length(column)} < {min_len}")
            if max_len is not None:
                length_conditions.append(f"{adapter.length(column)} > {max_len}")
            if length_conditions:
                conditions.append(f"({' or '.join(length_conditions)})")

        where = " and ".join(conditions)

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=column,
        )


class RegexHandler(RuleHandler):
    """Handler for regex/pattern rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["regex", "pattern", "matches"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        rule_type = self._get_rule_type(rule)
        pattern = rule.get("pattern")

        if not pattern:
            raise InvalidRuleError(rule_type, "Missing required 'pattern' field")

        where = f"{adapter.is_not_null(column)} and not {adapter.regex_match(column, pattern)}"

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=column,
            metadata={"pattern": pattern},
        )


class EmailFormatHandler(RuleHandler):
    """Handler for email format rules."""

    # Standard email regex pattern
    EMAIL_PATTERN: ClassVar[str] = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    @property
    def rule_types(self) -> list[str]:
        return ["email_format", "email"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        pattern = rule.get("pattern", self.EMAIL_PATTERN)

        where = f"{adapter.is_not_null(column)} and not {adapter.regex_match(column, pattern)}"

        return RuleSQL(
            where_clause=where,
            rule_type="email_format",
            column=column,
            metadata={"pattern": pattern},
        )


class UrlFormatHandler(RuleHandler):
    """Handler for URL format rules."""

    URL_PATTERN: ClassVar[str] = r"^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    @property
    def rule_types(self) -> list[str]:
        return ["url_format", "url"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        pattern = rule.get("pattern", self.URL_PATTERN)

        where = f"{adapter.is_not_null(column)} and not {adapter.regex_match(column, pattern)}"

        return RuleSQL(
            where_clause=where,
            rule_type="url_format",
            column=column,
            metadata={"pattern": pattern},
        )


class UuidFormatHandler(RuleHandler):
    """Handler for UUID format rules."""

    UUID_PATTERN: ClassVar[str] = (
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )

    @property
    def rule_types(self) -> list[str]:
        return ["uuid_format", "uuid"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        pattern = rule.get("pattern", self.UUID_PATTERN)

        where = f"{adapter.is_not_null(column)} and not {adapter.regex_match(column, pattern)}"

        return RuleSQL(
            where_clause=where,
            rule_type="uuid_format",
            column=column,
            metadata={"pattern": pattern},
        )


class PhoneFormatHandler(RuleHandler):
    """Handler for phone format rules."""

    # E.164 format
    PHONE_PATTERN: ClassVar[str] = r"^\+?[1-9]\d{1,14}$"

    @property
    def rule_types(self) -> list[str]:
        return ["phone_format", "phone"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        pattern = rule.get("pattern", self.PHONE_PATTERN)

        where = f"{adapter.is_not_null(column)} and not {adapter.regex_match(column, pattern)}"

        return RuleSQL(
            where_clause=where,
            rule_type="phone_format",
            column=column,
            metadata={"pattern": pattern},
        )


class Ipv4FormatHandler(RuleHandler):
    """Handler for IPv4 format rules."""

    IPV4_PATTERN: ClassVar[str] = (
        r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
    )

    @property
    def rule_types(self) -> list[str]:
        return ["ipv4_format", "ipv4"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        pattern = rule.get("pattern", self.IPV4_PATTERN)

        where = f"{adapter.is_not_null(column)} and not {adapter.regex_match(column, pattern)}"

        return RuleSQL(
            where_clause=where,
            rule_type="ipv4_format",
            column=column,
            metadata={"pattern": pattern},
        )


class DateFormatHandler(RuleHandler):
    """Handler for date format and temporal rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["date_format", "not_future", "not_past"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        rule_type = self._get_rule_type(rule)

        if rule_type == "not_future":
            where = (
                f"{adapter.is_not_null(column)} and "
                f"{column} > {adapter.current_timestamp()}"
            )
        elif rule_type == "not_past":
            where = (
                f"{adapter.is_not_null(column)} and "
                f"{column} < {adapter.current_timestamp()}"
            )
        else:  # date_format
            format_str = rule.get("format", "YYYY-MM-DD")
            # Use try_date_parse to check if parsing fails
            where = (
                f"{adapter.is_not_null(column)} and "
                f"{adapter.try_date_parse(column, format_str)} is null"
            )

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=column,
        )


class ReferentialIntegrityHandler(RuleHandler):
    """Handler for referential integrity (foreign key) rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["referential_integrity", "foreign_key", "relationships"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        column = self._get_column(rule)
        rule_type = self._get_rule_type(rule)

        # Get target table and field
        to_model = rule.get("to", rule.get("to_model"))
        to_field = rule.get("field", rule.get("to_column", column))

        if not to_model:
            raise InvalidRuleError(rule_type, "Missing required 'to' (target model) field")

        # The WHERE clause for a referential integrity check
        # finds rows that have a value not present in the reference table
        where = (
            f"{adapter.is_not_null(column)} and "
            f"{column} not in (select {to_field} from {to_model})"
        )

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=column,
            metadata={
                "to_model": to_model,
                "to_field": to_field,
            },
        )


class ExpressionHandler(RuleHandler):
    """Handler for custom expression rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["expression", "custom", "sql"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        rule_type = self._get_rule_type(rule)
        expression = rule.get("expression", rule.get("sql"))

        if not expression:
            raise InvalidRuleError(rule_type, "Missing required 'expression' field")

        # The expression should return TRUE for failing rows
        where = expression

        return RuleSQL(
            where_clause=where,
            rule_type=rule_type,
            column=rule.get("column"),
            metadata={"expression": expression},
        )


class RowCountHandler(RuleHandler):
    """Handler for row count range rules."""

    @property
    def rule_types(self) -> list[str]:
        return ["row_count_range", "row_count"]

    def handle(
        self,
        rule: dict[str, Any],
        adapter: SQLAdapter,
        context: ConversionContext | None = None,
    ) -> RuleSQL:
        rule_type = self._get_rule_type(rule)
        min_count = rule.get("min")
        max_count = rule.get("max")

        # This is a model-level check, not row-level
        # Return metadata indicating this needs special handling
        return RuleSQL(
            where_clause="__ROW_COUNT_CHECK__",  # Placeholder
            rule_type=rule_type,
            column=None,
            metadata={
                "requires_row_count_check": True,
                "min_count": min_count,
                "max_count": max_count,
            },
        )


# =============================================================================
# Rule Handler Registry
# =============================================================================


class RuleHandlerRegistry:
    """Registry for rule handlers.

    Maps rule types to their handlers for efficient lookup.

    Thread-safe implementation using a lock for concurrent access.
    """

    _instance: RuleHandlerRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> RuleHandlerRegistry:
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._handlers = {}
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if self._initialized:
            return
        self._handlers: dict[str, RuleHandler] = {}
        self._initialized: bool = True

    def register(self, handler: RuleHandler) -> None:
        """Register a handler for its rule types.

        Args:
            handler: Handler to register.
        """
        with self._lock:
            for rule_type in handler.rule_types:
                self._handlers[rule_type.lower()] = handler

    def get(self, rule_type: str) -> RuleHandler | None:
        """Get handler for a rule type.

        Args:
            rule_type: Rule type to look up.

        Returns:
            Handler if found, None otherwise.
        """
        with self._lock:
            return self._handlers.get(rule_type.lower())

    def list_handlers(self) -> list[str]:
        """List all registered rule types.

        Returns:
            List of rule types.
        """
        with self._lock:
            return list(self._handlers.keys())

    def reset(self) -> None:
        """Reset the registry."""
        with self._lock:
            self._handlers.clear()
            self._initialized = False


# Global handler registry
_handler_registry: RuleHandlerRegistry | None = None
_handler_lock = threading.Lock()


def get_handler_registry() -> RuleHandlerRegistry:
    """Get the global handler registry."""
    global _handler_registry
    if _handler_registry is None:
        with _handler_lock:
            if _handler_registry is None:
                _handler_registry = RuleHandlerRegistry()
                _register_default_handlers(_handler_registry)
    return _handler_registry


def _register_default_handlers(registry: RuleHandlerRegistry) -> None:
    """Register default handlers."""
    registry.register(NotNullHandler())
    registry.register(UniqueHandler())
    registry.register(InSetHandler())
    registry.register(RangeHandler())
    registry.register(LengthHandler())
    registry.register(RegexHandler())
    registry.register(EmailFormatHandler())
    registry.register(UrlFormatHandler())
    registry.register(UuidFormatHandler())
    registry.register(PhoneFormatHandler())
    registry.register(Ipv4FormatHandler())
    registry.register(DateFormatHandler())
    registry.register(ReferentialIntegrityHandler())
    registry.register(ExpressionHandler())
    registry.register(RowCountHandler())


def get_handler(rule_type: str) -> RuleHandler | None:
    """Get handler for a rule type."""
    return get_handler_registry().get(rule_type)


def register_handler(handler: RuleHandler) -> None:
    """Register a handler."""
    get_handler_registry().register(handler)


def list_handlers() -> list[str]:
    """List all registered rule types."""
    return get_handler_registry().list_handlers()


# =============================================================================
# Standard Rule Converter
# =============================================================================


class StandardRuleConverter:
    """Standard implementation of RuleConverter.

    Converts data quality rules to SQL using registered handlers.
    Combines multiple rules into a single query using UNION ALL.

    Example:
        >>> converter = StandardRuleConverter()
        >>> adapter = get_adapter("snowflake")
        >>>
        >>> result = converter.convert(
        ...     {"type": "not_null", "column": "email"},
        ...     adapter,
        ... )
        >>> print(result.where_clause)
        'email is null'
    """

    def __init__(self) -> None:
        """Initialize converter."""
        self._handler_registry = get_handler_registry()

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
        rule_type = rule.get("type", rule.get("check", "unknown"))
        handler = self._handler_registry.get(rule_type)

        if handler is None:
            raise UnsupportedRuleError(rule_type, adapter.name)

        return handler.handle(rule, adapter, context)

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
        rules_sql: list[RuleSQL] = []
        errors: list[UnsupportedRuleError | InvalidRuleError] = []

        # Convert each rule
        for rule in rules:
            try:
                rule_sql = self.convert(rule, adapter, context)
                rules_sql.append(rule_sql)
            except (UnsupportedRuleError, InvalidRuleError) as e:
                errors.append(e)

        # Generate combined SQL
        combined_sql = self._generate_combined_sql(
            rules_sql,
            rules,
            adapter,
            context,
        )

        return ConversionResult(
            rules_sql=tuple(rules_sql),
            combined_sql=combined_sql,
            rule_count=len(rules),
            conversion_errors=tuple(errors),
            metadata={
                "model": context.model,
                "adapter": adapter.name,
            },
        )

    def _generate_combined_sql(
        self,
        rules_sql: list[RuleSQL],
        rules: list[dict[str, Any]],
        adapter: SQLAdapter,
        context: ConversionContext,
    ) -> str:
        """Generate combined SQL from multiple rules.

        Args:
            rules_sql: List of RuleSQL objects.
            rules: Original rule definitions.
            adapter: SQL adapter.
            context: Conversion context.

        Returns:
            Combined SQL query.
        """
        if not rules_sql:
            return f"select * from {context.model} where 1=0"

        cte_parts = []
        union_parts = []

        # Source data CTE with optional sampling and filtering
        source_sql = f"select * from {context.model}"
        if context.where_filter:
            source_sql += f" where {context.where_filter}"
        if context.sample_size:
            source_sql += f" {adapter.limit_sample(context.sample_size)}"

        cte_parts.append(f"source_data as (\n    {source_sql}\n)")

        # Generate CTE for each rule
        for idx, (rule_sql, rule) in enumerate(zip(rules_sql, rules)):
            rule_type = rule.get("type", rule.get("check", "unknown"))
            column = rule.get("column", "_model_")

            # Handle special cases
            if rule_sql.metadata.get("requires_unique_subquery"):
                # Use adapter's unique check SQL
                if isinstance(column, str) and "," in column:
                    # Multi-column unique - use subquery approach
                    check_sql = f"""select t.*
    from source_data t
    inner join (
        select {column}
        from source_data
        group by {column}
        having count(*) > 1
    ) duplicates
    on {' and '.join(f't.{c.strip()} = duplicates.{c.strip()}' for c in column.split(','))}"""
                else:
                    # Single column - may use QUALIFY
                    unique_sql = adapter.unique_check_sql("source_data", column)
                    check_sql = unique_sql
            elif rule_sql.metadata.get("requires_row_count_check"):
                # Row count check is special - generates a different query
                min_c = rule_sql.metadata.get("min_count")
                max_c = rule_sql.metadata.get("max_count")
                conditions = []
                if min_c is not None:
                    conditions.append(f"cnt < {min_c}")
                if max_c is not None:
                    conditions.append(f"cnt > {max_c}")
                if conditions:
                    check_sql = f"""select 1 as _row_count_failure
    from (select count(*) as cnt from source_data) counts
    where {' or '.join(conditions)}"""
                else:
                    check_sql = "select 1 where 1=0"  # No condition, no failure
            else:
                check_sql = f"""select *
    from source_data
    where {rule_sql.where_clause}"""

            cte_parts.append(f"rule_{idx}_failures as (\n    {check_sql}\n)")

            # Union part with metadata
            union_parts.append(f"""select
        '{column}' as _truthound_column,
        '{rule_type}' as _truthound_check,
        t.*
    from rule_{idx}_failures t""")

        # Combine all parts
        ctes = ",\n\n".join(cte_parts)
        unions = "\n    union all\n".join(union_parts)

        combined = f"""with {ctes},

all_failures as (
    {unions}
)

select * from all_failures"""

        if context.fail_fast:
            combined += "\nlimit 1"

        return combined
