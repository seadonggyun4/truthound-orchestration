"""BigQuery SQL Adapter.

This module provides BigQuery-specific SQL generation for data quality checks.
BigQuery has unique features like QUALIFY clause and REGEXP_CONTAINS function.

Features:
    - QUALIFY clause for efficient window function filtering
    - REGEXP_CONTAINS with r'' raw string syntax
    - SAFE_CAST and SAFE.PARSE_DATE for safe type conversions
    - RAND() for random sampling

Example:
    >>> from truthound_dbt.adapters import BigQueryAdapter
    >>>
    >>> adapter = BigQueryAdapter()
    >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
    "regexp_contains(email, r'^[\\w.-]+@[\\w.-]+$')"
"""

from __future__ import annotations

from typing import Any

from truthound_dbt.adapters.base import (
    AdapterConfig,
    BaseSQLAdapter,
    DEFAULT_ADAPTER_CONFIG,
)


class BigQueryAdapter(BaseSQLAdapter):
    """BigQuery SQL adapter.

    Provides BigQuery-specific SQL generation including:
        - QUALIFY clause for efficient uniqueness checks
        - REGEXP_CONTAINS with raw string syntax (r'pattern')
        - SAFE_CAST and SAFE.PARSE_DATE for safe conversions
        - RAND() for random sampling
        - CONCAT() function for string concatenation
        - current_timestamp() with parentheses

    BigQuery uses RE2 regex engine which has some differences from POSIX regex.
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        """Initialize BigQuery adapter.

        Args:
            config: Adapter configuration.
        """
        # BigQuery uses backticks for identifiers
        if config is None:
            config = AdapterConfig(
                quote_identifiers=False,
                identifier_quote_char="`",
                string_quote_char="'",
                escape_backslash=False,  # BigQuery raw strings handle escaping
            )
        super().__init__(config)

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "bigquery"

    def regex_match(self, column: str, pattern: str) -> str:
        """Generate SQL for regex matching using REGEXP_CONTAINS.

        Uses BigQuery's raw string syntax (r'pattern') which doesn't require
        double escaping of backslashes.

        Args:
            column: Column name.
            pattern: Regex pattern.

        Returns:
            SQL expression using regexp_contains.

        Example:
            >>> adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
            "regexp_contains(email, r'^[\\w.-]+@[\\w.-]+$')"
        """
        # BigQuery uses r'' raw string syntax
        # Single quotes inside the pattern need to be escaped
        escaped_pattern = pattern.replace("'", "\\'")
        return f"regexp_contains({column}, r'{escaped_pattern}')"

    def current_timestamp(self) -> str:
        """Generate SQL for current timestamp.

        BigQuery requires parentheses.

        Returns:
            'current_timestamp()'.
        """
        return "current_timestamp()"

    def date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for parsing date from string.

        Uses BigQuery's PARSE_DATE function. Format uses strftime-style
        specifiers (%Y, %m, %d).

        Args:
            column: Column name.
            format_str: Date format string (YYYY-MM-DD style).

        Returns:
            SQL expression using parse_date.

        Example:
            >>> adapter.date_parse("date_str", "YYYY-MM-DD")
            "parse_date('%Y-%m-%d', date_str)"
        """
        # Convert YYYY-MM-DD style to strftime style
        bq_format = self._convert_date_format(format_str)
        return f"parse_date('{bq_format}', {column})"

    def try_date_parse(self, column: str, format_str: str) -> str:
        """Generate SQL for safe date parsing.

        Uses BigQuery's SAFE.PARSE_DATE which returns NULL on failure.

        Args:
            column: Column name.
            format_str: Date format string.

        Returns:
            SQL expression using safe.parse_date.

        Example:
            >>> adapter.try_date_parse("date_str", "YYYY-MM-DD")
            "safe.parse_date('%Y-%m-%d', date_str)"
        """
        bq_format = self._convert_date_format(format_str)
        return f"safe.parse_date('{bq_format}', {column})"

    def _convert_date_format(self, format_str: str) -> str:
        """Convert YYYY-MM-DD style format to BigQuery strftime style.

        Args:
            format_str: Format string in YYYY-MM-DD style.

        Returns:
            Format string in strftime style.
        """
        return (
            format_str.replace("YYYY", "%Y")
            .replace("MM", "%m")
            .replace("DD", "%d")
            .replace("HH", "%H")
            .replace("MI", "%M")
            .replace("SS", "%S")
        )

    def limit_sample(self, n: int) -> str:
        """Generate SQL for random sampling.

        Uses ORDER BY RAND() which shuffles all rows.

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

        Uses BigQuery's QUALIFY clause for efficient window function filtering.

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

        Uses BigQuery's CONCAT function.

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

        Uses BigQuery's LENGTH function.

        Args:
            column: Column name.

        Returns:
            SQL expression for length.
        """
        return f"length({column})"

    def quote_identifier(self, identifier: str) -> str:
        """Quote an identifier using backticks.

        BigQuery uses backticks for quoting identifiers.

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
        escaped = identifier.replace("`", "\\`")
        return f"`{escaped}`"
