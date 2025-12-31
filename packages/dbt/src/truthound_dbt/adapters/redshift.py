"""Redshift SQL Adapter.

This module provides Amazon Redshift-specific SQL generation for data quality checks.
Redshift is PostgreSQL-compatible but has some differences in function availability.

Features:
    - POSIX regex matching with ~ operator
    - TO_DATE for date parsing
    - random() for sampling
    - No QUALIFY clause (uses subqueries)

Example:
    >>> from truthound_dbt.adapters import RedshiftAdapter
    >>>
    >>> adapter = RedshiftAdapter()
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


class RedshiftAdapter(BaseSQLAdapter):
    """Amazon Redshift SQL adapter.

    Provides Redshift-specific SQL generation including:
        - POSIX regex with ~ operator (PostgreSQL compatible)
        - TO_DATE for date parsing
        - random() for sampling
        - Subquery-based uniqueness checks (no QUALIFY)

    Redshift is based on PostgreSQL but doesn't support all PostgreSQL
    features. Notable differences:
        - No QUALIFY clause
        - Limited regex functionality compared to PostgreSQL
        - No TRY_CAST (uses NVL2 or CASE expressions)
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        """Initialize Redshift adapter.

        Args:
            config: Adapter configuration.
        """
        super().__init__(config or DEFAULT_ADAPTER_CONFIG)

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "redshift"

    def regex_match(self, column: str, pattern: str) -> str:
        """Generate SQL for POSIX regex matching.

        Uses PostgreSQL-compatible ~ operator.

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
            'getdate()' (Redshift-specific).
        """
        return "getdate()"

    def date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for parsing date from string.

        Uses Redshift's TO_DATE function.

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

        Redshift doesn't have TRY_TO_DATE, so we use a CASE expression
        with error handling.

        Args:
            column: Column name.
            format_str: Date format string.

        Returns:
            SQL expression for safe date parsing.
        """
        # Use CASE with a subquery to catch parsing errors
        return f"""case
    when {column} is null then null
    when regexp_count({column}, '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}$') = 1
        then to_date({column}, '{format_str}')
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
        """Generate SQL for uniqueness check using subquery.

        Redshift doesn't support QUALIFY, so we use a subquery approach.

        Args:
            model: Table/model reference.
            column: Column to check.

        Returns:
            SQL query using subquery for uniqueness check.
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

    def concat(self, *parts: str) -> str:
        """Generate SQL for string concatenation.

        Uses PostgreSQL-compatible || operator.

        Args:
            *parts: Parts to concatenate.

        Returns:
            SQL expression using ||.
        """
        return " || ".join(parts)

    def coalesce(self, *columns: str) -> str:
        """Generate SQL for coalesce operation.

        Redshift supports both COALESCE and NVL.
        We use COALESCE for consistency.

        Args:
            *columns: Columns to coalesce.

        Returns:
            SQL expression using COALESCE.
        """
        return f"coalesce({', '.join(columns)})"
