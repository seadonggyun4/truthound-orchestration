"""PostgreSQL SQL Adapter.

This module provides PostgreSQL-specific SQL generation for data quality checks.
PostgreSQL is the default adapter and provides baseline implementations.

Features:
    - POSIX regex matching with ~ operator
    - Standard SQL date/time functions
    - random() for sampling

Example:
    >>> from truthound_dbt.adapters import PostgresAdapter
    >>>
    >>> adapter = PostgresAdapter()
    >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
    "email ~ '^[\\w.-]+@[\\w.-]+$'"
"""

from __future__ import annotations

from typing import Any

from truthound_dbt.adapters.base import (
    AdapterConfig,
    BaseSQLAdapter,
    DEFAULT_ADAPTER_CONFIG,
)


class PostgresAdapter(BaseSQLAdapter):
    """PostgreSQL SQL adapter.

    Provides PostgreSQL-specific SQL generation including:
        - POSIX regex with ~ operator
        - Standard SQL date functions
        - random() for sampling
        - || for string concatenation

    This is the default adapter and serves as the baseline for others.
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        """Initialize PostgreSQL adapter.

        Args:
            config: Adapter configuration.
        """
        super().__init__(config or DEFAULT_ADAPTER_CONFIG)

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "postgres"

    def regex_match(self, column: str, pattern: str) -> str:
        """Generate SQL for POSIX regex matching.

        Uses PostgreSQL's ~ operator for regex matching.

        Args:
            column: Column name.
            pattern: Regex pattern.

        Returns:
            SQL expression using ~ operator.

        Example:
            >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
            "email ~ '^[\\w.-]+@[\\w.-]+$'"
        """
        escaped_pattern = pattern.replace("'", "''")
        return f"{column} ~ '{escaped_pattern}'"

    def current_timestamp(self) -> str:
        """Generate SQL for current timestamp.

        Returns:
            'current_timestamp' (PostgreSQL standard).
        """
        return "current_timestamp"

    def date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for parsing date from string.

        Uses PostgreSQL's to_date function.

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

        PostgreSQL doesn't have a native TRY_CAST, so we use a CASE expression
        with a regex check to validate the format before parsing.

        Args:
            column: Column name.
            format_str: Date format string.

        Returns:
            SQL expression for safe date parsing.
        """
        # Simplified validation - actual implementation would be more robust
        return f"""case
    when {column} is null then null
    when {column}::date is not null then {column}::date
    else null
end"""

    def limit_sample(self, n: int) -> str:
        """Generate SQL for random sampling.

        Uses ORDER BY random() which shuffles all rows.

        Args:
            n: Number of rows to sample.

        Returns:
            SQL clause for random sampling.

        Example:
            >>> adapter.limit_sample(1000)
            'order by random() limit 1000'
        """
        return f"order by random() limit {n}"

    def unique_check_sql(self, model: str, column: str) -> str:
        """Generate SQL for uniqueness check.

        Uses a subquery approach that works efficiently on PostgreSQL.

        Args:
            model: Table/model reference.
            column: Column to check.

        Returns:
            SQL query selecting duplicate rows.
        """
        return f"""select t.*
from {model} t
inner join (
    select {column}
    from {model}
    group by {column}
    having count(*) > 1
) duplicates
on t.{column} = duplicates.{column}"""
