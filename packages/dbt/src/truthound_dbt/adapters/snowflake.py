"""Snowflake SQL Adapter.

This module provides Snowflake-specific SQL generation for data quality checks.
Snowflake has unique features like QUALIFY clause and REGEXP_LIKE function
that enable more efficient queries.

Features:
    - QUALIFY clause for efficient window function filtering
    - REGEXP_LIKE for regex matching
    - TRY_TO_DATE for safe date parsing
    - SAMPLE clause for efficient random sampling

Example:
    >>> from truthound_dbt.adapters import SnowflakeAdapter
    >>>
    >>> adapter = SnowflakeAdapter()
    >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
    "regexp_like(email, '^[\\w.-]+@[\\w.-]+$')"
"""

from __future__ import annotations

from typing import Any

from truthound_dbt.adapters.base import (
    AdapterConfig,
    BaseSQLAdapter,
    DEFAULT_ADAPTER_CONFIG,
)


class SnowflakeAdapter(BaseSQLAdapter):
    """Snowflake SQL adapter.

    Provides Snowflake-specific SQL generation including:
        - QUALIFY clause for efficient uniqueness checks
        - REGEXP_LIKE function for regex matching
        - TRY_TO_DATE for safe date parsing
        - SAMPLE clause for efficient sampling
        - current_timestamp() with parentheses

    Snowflake's QUALIFY clause allows filtering on window function results
    without a subquery, making uniqueness checks more efficient.
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        """Initialize Snowflake adapter.

        Args:
            config: Adapter configuration.
        """
        super().__init__(config or DEFAULT_ADAPTER_CONFIG)

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "snowflake"

    def regex_match(self, column: str, pattern: str) -> str:
        """Generate SQL for regex matching using REGEXP_LIKE.

        Args:
            column: Column name.
            pattern: Regex pattern.

        Returns:
            SQL expression using regexp_like.

        Example:
            >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
            "regexp_like(email, '^[\\w.-]+@[\\w.-]+$')"
        """
        escaped_pattern = pattern.replace("'", "''")
        return f"regexp_like({column}, '{escaped_pattern}')"

    def current_timestamp(self) -> str:
        """Generate SQL for current timestamp.

        Snowflake requires parentheses.

        Returns:
            'current_timestamp()'.
        """
        return "current_timestamp()"

    def date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for parsing date from string.

        Uses Snowflake's TO_DATE function.

        Args:
            column: Column name.
            format_str: Date format string.

        Returns:
            SQL expression using to_date.

        Example:
            >>> adapter.date_parse("date_str", "YYYY-MM-DD")
            "to_date(date_str, 'YYYY-MM-DD')"
        """
        return f"to_date({column}, '{format_str}')"

    def try_date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for safe date parsing.

        Uses Snowflake's TRY_TO_DATE function which returns NULL on failure.

        Args:
            column: Column name.
            format_str: Date format string.

        Returns:
            SQL expression using try_to_date.

        Example:
            >>> adapter.try_date_parse("date_str", "YYYY-MM-DD")
            "try_to_date(date_str, 'YYYY-MM-DD')"
        """
        return f"try_to_date({column}, '{format_str}')"

    def limit_sample(self, n: int) -> str:
        """Generate SQL for random sampling.

        Uses Snowflake's efficient SAMPLE clause.

        Args:
            n: Number of rows to sample.

        Returns:
            SQL clause using SAMPLE.

        Example:
            >>> adapter.limit_sample(1000)
            'sample (1000 rows)'
        """
        return f"sample ({n} rows)"

    def unique_check_sql(self, model: str, column: str) -> str:
        """Generate SQL for uniqueness check using QUALIFY.

        Uses Snowflake's QUALIFY clause for more efficient window function
        filtering without requiring a subquery.

        Args:
            model: Table/model reference.
            column: Column to check.

        Returns:
            SQL query using QUALIFY clause.

        Example:
            >>> adapter.unique_check_sql("users", "email")
            '''
            select *
            from users
            qualify count(*) over (partition by email) > 1
            '''
        """
        return f"""select *
from {model}
qualify count(*) over (partition by {column}) > 1"""

    def concat(self, *parts: str) -> str:
        """Generate SQL for string concatenation.

        Snowflake supports both || and CONCAT function.
        We use || for consistency with other adapters.

        Args:
            *parts: Parts to concatenate.

        Returns:
            SQL expression using ||.
        """
        return " || ".join(parts)
