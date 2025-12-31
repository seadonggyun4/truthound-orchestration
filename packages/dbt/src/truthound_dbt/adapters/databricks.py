"""Databricks SQL Adapter.

This module provides Databricks-specific SQL generation for data quality checks.
Databricks uses Spark SQL syntax with some extensions.

Features:
    - RLIKE for regex matching
    - TRY_CAST for safe type conversions
    - QUALIFY clause for window function filtering
    - rand() for random sampling

Example:
    >>> from truthound_dbt.adapters import DatabricksAdapter
    >>>
    >>> adapter = DatabricksAdapter()
    >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
    "email rlike '^[\\w.-]+@[\\w.-]+$'"
"""

from __future__ import annotations

from typing import Any

from truthound_dbt.adapters.base import (
    AdapterConfig,
    BaseSQLAdapter,
    DEFAULT_ADAPTER_CONFIG,
)


class DatabricksAdapter(BaseSQLAdapter):
    """Databricks SQL adapter.

    Provides Databricks-specific SQL generation including:
        - RLIKE operator for regex matching (Spark SQL style)
        - TRY_CAST for safe type conversions
        - QUALIFY clause for efficient window function filtering
        - rand() for random sampling
        - CONCAT function for string concatenation

    Databricks SQL is based on Spark SQL with some extensions like
    the QUALIFY clause for filtering on window function results.
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        """Initialize Databricks adapter.

        Args:
            config: Adapter configuration.
        """
        # Databricks uses backticks for identifiers
        if config is None:
            config = AdapterConfig(
                quote_identifiers=False,
                identifier_quote_char="`",
                string_quote_char="'",
                escape_backslash=True,
            )
        super().__init__(config)

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "databricks"

    def regex_match(self, column: str, pattern: str) -> str:
        """Generate SQL for regex matching using RLIKE.

        Uses Spark SQL's RLIKE operator for Java regex matching.

        Args:
            column: Column name.
            pattern: Regex pattern.

        Returns:
            SQL expression using rlike.

        Example:
            >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
            "email rlike '^[\\w.-]+@[\\w.-]+$'"
        """
        escaped_pattern = pattern.replace("'", "\\'")
        return f"{column} rlike '{escaped_pattern}'"

    def current_timestamp(self) -> str:
        """Generate SQL for current timestamp.

        Returns:
            'current_timestamp()'.
        """
        return "current_timestamp()"

    def date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for parsing date from string.

        Uses Spark's TO_DATE function.

        Args:
            column: Column name.
            format_str: Date format string (uses Java SimpleDateFormat style).

        Returns:
            SQL expression using to_date.

        Example:
            >>> adapter.date_parse("date_str", "yyyy-MM-dd")
            "to_date(date_str, 'yyyy-MM-dd')"
        """
        # Convert common format to Java style
        java_format = self._convert_date_format(format_str)
        return f"to_date({column}, '{java_format}')"

    def try_date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for safe date parsing.

        Uses TRY_CAST combined with TO_DATE for safe parsing.

        Args:
            column: Column name.
            format_str: Date format string.

        Returns:
            SQL expression for safe date parsing.

        Example:
            >>> adapter.try_date_parse("date_str", "yyyy-MM-dd")
            "try_to_date(date_str, 'yyyy-MM-dd')"
        """
        java_format = self._convert_date_format(format_str)
        return f"try_to_date({column}, '{java_format}')"

    def _convert_date_format(self, format_str: str) -> str:
        """Convert common format to Java SimpleDateFormat style.

        Args:
            format_str: Format string in YYYY-MM-DD style.

        Returns:
            Format string in Java style.
        """
        return (
            format_str.replace("YYYY", "yyyy")
            .replace("MM", "MM")  # Already correct
            .replace("DD", "dd")
            .replace("HH", "HH")  # Already correct
            .replace("MI", "mm")
            .replace("SS", "ss")
        )

    def limit_sample(self, n: int) -> str:
        """Generate SQL for random sampling.

        Uses ORDER BY rand() which shuffles all rows.

        Args:
            n: Number of rows to sample.

        Returns:
            SQL clause for random sampling.

        Example:
            >>> adapter.limit_sample(1000)
            'order by rand() limit 1000'
        """
        return f"order by rand() limit {n}"

    def unique_check_sql(self, model: str, column: str) -> str:
        """Generate SQL for uniqueness check using QUALIFY.

        Databricks SQL supports QUALIFY for efficient window function filtering.

        Args:
            model: Table/model reference.
            column: Column to check.

        Returns:
            SQL query using QUALIFY clause.
        """
        return f"""select *
from {model}
qualify count(*) over (partition by {column}) > 1"""

    def concat(self, *parts: str) -> str:
        """Generate SQL for string concatenation.

        Uses Spark's CONCAT function.

        Args:
            *parts: Parts to concatenate.

        Returns:
            SQL expression using CONCAT.

        Example:
            >>> adapter.concat("first_name", "' '", "last_name")
            "concat(first_name, ' ', last_name)"
        """
        return f"concat({', '.join(parts)})"

    def length(self, column: str) -> str:
        """Generate SQL for string length.

        Uses Spark's LENGTH function.

        Args:
            column: Column name.

        Returns:
            SQL expression for length.
        """
        return f"length({column})"

    def quote_identifier(self, identifier: str) -> str:
        """Quote an identifier using backticks.

        Databricks uses backticks for quoting identifiers.

        Args:
            identifier: Identifier to quote.

        Returns:
            Backtick-quoted identifier.

        Example:
            >>> adapter.quote_identifier("user")
            '`user`'
        """
        if not self._config.quote_identifiers:
            return identifier
        escaped = identifier.replace("`", "``")
        return f"`{escaped}`"
