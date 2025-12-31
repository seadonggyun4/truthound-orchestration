"""SQL Query Generator.

This module provides SQL query generation from data quality rules using
database-specific adapters.

Example:
    >>> from truthound_dbt.generators import SQLGenerator
    >>> from truthound_dbt.adapters import get_adapter
    >>>
    >>> adapter = get_adapter("snowflake")
    >>> generator = SQLGenerator(adapter)
    >>>
    >>> sql = generator.generate_check_sql(
    ...     model="ref('users')",
    ...     rules=[
    ...         {"type": "not_null", "column": "id"},
    ...         {"type": "unique", "column": "email"},
    ...     ],
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from truthound_dbt.adapters.base import SQLAdapter
    from truthound_dbt.converters.base import RuleConverter


# =============================================================================
# Exceptions
# =============================================================================


class SQLGenerationError(Exception):
    """Base exception for SQL generation errors."""

    pass


class InvalidRuleError(SQLGenerationError):
    """Raised when a rule is invalid or unsupported."""

    def __init__(self, rule_type: str, message: str) -> None:
        self.rule_type = rule_type
        super().__init__(f"Invalid rule '{rule_type}': {message}")


class AdapterNotConfiguredError(SQLGenerationError):
    """Raised when adapter is not configured."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class SQLGeneratorConfig:
    """Configuration for SQL generator.

    Attributes:
        include_comments: Include SQL comments with rule info.
        use_cte: Use Common Table Expressions for complex queries.
        indent_size: Number of spaces for indentation.
        max_line_length: Maximum line length before wrapping.
        include_metadata: Include metadata in generated SQL.
        fail_on_error: Raise exception on conversion errors.
        sample_size: Optional sample size for large tables.
        timeout_seconds: Query timeout in seconds.
    """

    include_comments: bool = True
    use_cte: bool = True
    indent_size: int = 4
    max_line_length: int = 120
    include_metadata: bool = True
    fail_on_error: bool = True
    sample_size: int | None = None
    timeout_seconds: float | None = None

    def with_comments(self, include: bool = True) -> SQLGeneratorConfig:
        """Return config with comment setting."""
        return SQLGeneratorConfig(
            include_comments=include,
            use_cte=self.use_cte,
            indent_size=self.indent_size,
            max_line_length=self.max_line_length,
            include_metadata=self.include_metadata,
            fail_on_error=self.fail_on_error,
            sample_size=self.sample_size,
            timeout_seconds=self.timeout_seconds,
        )

    def with_cte(self, use: bool = True) -> SQLGeneratorConfig:
        """Return config with CTE setting."""
        return SQLGeneratorConfig(
            include_comments=self.include_comments,
            use_cte=use,
            indent_size=self.indent_size,
            max_line_length=self.max_line_length,
            include_metadata=self.include_metadata,
            fail_on_error=self.fail_on_error,
            sample_size=self.sample_size,
            timeout_seconds=self.timeout_seconds,
        )

    def with_sample_size(self, size: int | None) -> SQLGeneratorConfig:
        """Return config with sample size."""
        return SQLGeneratorConfig(
            include_comments=self.include_comments,
            use_cte=self.use_cte,
            indent_size=self.indent_size,
            max_line_length=self.max_line_length,
            include_metadata=self.include_metadata,
            fail_on_error=self.fail_on_error,
            sample_size=size,
            timeout_seconds=self.timeout_seconds,
        )


DEFAULT_SQL_GENERATOR_CONFIG = SQLGeneratorConfig()


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class SQLFragment:
    """A fragment of SQL for a single rule.

    Attributes:
        sql: The SQL fragment.
        rule_type: Type of the rule.
        column: Column being checked (if applicable).
        description: Human-readable description.
    """

    sql: str
    rule_type: str
    column: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratedSQL:
    """Result of SQL generation.

    Attributes:
        sql: The complete SQL query.
        fragments: Individual SQL fragments per rule.
        model: The model/table reference.
        adapter_name: Name of the adapter used.
        rule_count: Number of rules included.
        generated_at: Timestamp of generation.
        metadata: Additional metadata.
    """

    sql: str
    fragments: tuple[SQLFragment, ...]
    model: str
    adapter_name: str
    rule_count: int
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sql": self.sql,
            "fragments": [
                {
                    "sql": f.sql,
                    "rule_type": f.rule_type,
                    "column": f.column,
                    "description": f.description,
                }
                for f in self.fragments
            ],
            "model": self.model,
            "adapter_name": self.adapter_name,
            "rule_count": self.rule_count,
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# SQLGenerator
# =============================================================================


class SQLGenerator:
    """Generates SQL queries from data quality rules.

    This generator uses a SQL adapter for database-specific syntax and
    a rule converter to transform rules into SQL expressions.

    Example:
        >>> from truthound_dbt.generators import SQLGenerator
        >>> from truthound_dbt.adapters import get_adapter
        >>>
        >>> adapter = get_adapter("snowflake")
        >>> generator = SQLGenerator(adapter)
        >>>
        >>> result = generator.generate_check_sql(
        ...     model="ref('users')",
        ...     rules=[
        ...         {"type": "not_null", "column": "id"},
        ...         {"type": "unique", "column": "email"},
        ...     ],
        ... )
        >>> print(result.sql)
    """

    def __init__(
        self,
        adapter: SQLAdapter,
        config: SQLGeneratorConfig | None = None,
        converter: RuleConverter | None = None,
    ) -> None:
        """Initialize SQL generator.

        Args:
            adapter: SQL adapter for database-specific syntax.
            config: Generator configuration.
            converter: Rule converter (uses default if not provided).
        """
        self._adapter = adapter
        self._config = config or DEFAULT_SQL_GENERATOR_CONFIG

        # Import converter lazily to avoid circular imports
        if converter is None:
            from truthound_dbt.converters import get_converter

            self._converter = get_converter("standard")
        else:
            self._converter = converter

    @property
    def adapter(self) -> SQLAdapter:
        """Return the SQL adapter."""
        return self._adapter

    @property
    def config(self) -> SQLGeneratorConfig:
        """Return the generator configuration."""
        return self._config

    def generate_check_sql(
        self,
        model: str,
        rules: Sequence[dict[str, Any]],
        *,
        fail_fast: bool = False,
        sample_size: int | None = None,
    ) -> GeneratedSQL:
        """Generate SQL for checking data quality rules.

        Args:
            model: Model/table reference (e.g., "ref('users')").
            rules: List of rule dictionaries.
            fail_fast: Generate query that fails on first violation.
            sample_size: Optional sample size (overrides config).

        Returns:
            GeneratedSQL containing the complete query and metadata.

        Raises:
            SQLGenerationError: If SQL generation fails.
            InvalidRuleError: If a rule is invalid.
        """
        if not rules:
            return GeneratedSQL(
                sql=f"select 1 as _no_rules from {model} where 1=0",
                fragments=(),
                model=model,
                adapter_name=self._adapter.name,
                rule_count=0,
            )

        # Convert rules to SQL
        from truthound_dbt.converters.base import ConversionContext

        context = ConversionContext(model=model)
        conversion_result = self._converter.convert_all(
            rules=list(rules),
            adapter=self._adapter,
            context=context,
        )

        if not conversion_result.rules_sql and self._config.fail_on_error:
            raise SQLGenerationError(
                f"No valid rules converted. Errors: {conversion_result.errors}"
            )

        # Build fragments
        fragments = tuple(
            SQLFragment(
                sql=rule_sql.where_clause,
                rule_type=rule_sql.rule_type,
                column=rule_sql.column,
                description=rule_sql.metadata.get("description"),
            )
            for rule_sql in conversion_result.rules_sql
        )

        # Generate combined SQL
        effective_sample = sample_size or self._config.sample_size
        sql = self._generate_combined_query(
            model=model,
            fragments=fragments,
            fail_fast=fail_fast,
            sample_size=effective_sample,
        )

        return GeneratedSQL(
            sql=sql,
            fragments=fragments,
            model=model,
            adapter_name=self._adapter.name,
            rule_count=len(fragments),
            metadata={
                "fail_fast": fail_fast,
                "sample_size": effective_sample,
                "errors": conversion_result.errors,
                "warnings": conversion_result.warnings,
            },
        )

    def generate_single_rule_sql(
        self,
        model: str,
        rule: dict[str, Any],
    ) -> GeneratedSQL:
        """Generate SQL for a single rule.

        Args:
            model: Model/table reference.
            rule: Rule dictionary.

        Returns:
            GeneratedSQL for the single rule.
        """
        return self.generate_check_sql(model, [rule])

    def generate_unique_check_sql(
        self,
        model: str,
        column: str,
    ) -> str:
        """Generate SQL for uniqueness check.

        Uses adapter-specific SQL (QUALIFY for Snowflake/BigQuery,
        subquery for others).

        Args:
            model: Model/table reference.
            column: Column to check for uniqueness.

        Returns:
            SQL query string.
        """
        return self._adapter.unique_check_sql(model, column)

    def generate_not_null_check_sql(
        self,
        model: str,
        column: str,
    ) -> str:
        """Generate SQL for not-null check.

        Args:
            model: Model/table reference.
            column: Column to check.

        Returns:
            SQL query string.
        """
        return f"select * from {model} where {column} is null"

    def generate_referential_check_sql(
        self,
        model: str,
        column: str,
        reference_model: str,
        reference_column: str,
    ) -> str:
        """Generate SQL for referential integrity check.

        Args:
            model: Source model/table reference.
            column: Source column.
            reference_model: Referenced model/table.
            reference_column: Referenced column.

        Returns:
            SQL query string.
        """
        return f"""select t.*
from {model} t
left join {reference_model} r on t.{column} = r.{reference_column}
where t.{column} is not null
  and r.{reference_column} is null"""

    def _generate_combined_query(
        self,
        model: str,
        fragments: tuple[SQLFragment, ...],
        fail_fast: bool = False,
        sample_size: int | None = None,
    ) -> str:
        """Generate combined SQL query from fragments.

        Args:
            model: Model/table reference.
            fragments: SQL fragments for each rule.
            fail_fast: Fail on first violation.
            sample_size: Optional sample size.

        Returns:
            Combined SQL query string.
        """
        indent = " " * self._config.indent_size

        if not fragments:
            return f"select 1 as _no_rules from {model} where 1=0"

        # Build header comment
        lines: list[str] = []
        if self._config.include_comments:
            lines.append("-- Data Quality Check Query")
            lines.append(f"-- Model: {model}")
            lines.append(f"-- Adapter: {self._adapter.name}")
            lines.append(f"-- Rules: {len(fragments)}")
            lines.append(f"-- Generated: {datetime.now().isoformat()}")
            lines.append("")

        if self._config.use_cte:
            # CTE-based approach for multiple rules
            lines.append("with")

            # Source CTE (with optional sampling)
            lines.append(f"{indent}source_data as (")
            lines.append(f"{indent}{indent}select *")
            lines.append(f"{indent}{indent}from {model}")
            if sample_size:
                sample_clause = self._adapter.limit_sample(sample_size)
                lines.append(f"{indent}{indent}{sample_clause}")
            lines.append(f"{indent}),")

            # Rule CTEs
            for i, frag in enumerate(fragments):
                rule_name = f"rule_{i + 1}_{frag.rule_type}"
                lines.append("")
                if self._config.include_comments and frag.description:
                    lines.append(f"{indent}-- {frag.description}")
                lines.append(f"{indent}{rule_name} as (")
                lines.append(f"{indent}{indent}select")
                lines.append(f"{indent}{indent}{indent}'{frag.rule_type}' as rule_type,")
                if frag.column:
                    lines.append(f"{indent}{indent}{indent}'{frag.column}' as column_name,")
                else:
                    lines.append(f"{indent}{indent}{indent}null as column_name,")
                lines.append(f"{indent}{indent}{indent}count(*) as violation_count")
                lines.append(f"{indent}{indent}from source_data")
                lines.append(f"{indent}{indent}where {frag.sql}")

                # Add comma unless it's the last CTE
                if i < len(fragments) - 1:
                    lines.append(f"{indent}),")
                else:
                    lines.append(f"{indent})")

            # Final SELECT combining all rules
            lines.append("")
            lines.append("select")
            lines.append(f"{indent}rule_type,")
            lines.append(f"{indent}column_name,")
            lines.append(f"{indent}violation_count,")
            lines.append(f"{indent}case when violation_count > 0 then 'FAILED' else 'PASSED' end as status")
            lines.append("from (")

            for i, frag in enumerate(fragments):
                rule_name = f"rule_{i + 1}_{frag.rule_type}"
                lines.append(f"{indent}select * from {rule_name}")
                if i < len(fragments) - 1:
                    lines.append(f"{indent}union all")

            lines.append(")")

            if fail_fast:
                lines.append("where violation_count > 0")
                lines.append("limit 1")
            else:
                lines.append("order by violation_count desc")

        else:
            # Non-CTE approach (single combined WHERE)
            conditions = [f.sql for f in fragments]
            combined_where = " or ".join(f"({c})" for c in conditions)

            lines.append("select *")
            lines.append(f"from {model}")
            if sample_size:
                sample_clause = self._adapter.limit_sample(sample_size)
                lines.append(sample_clause.split("limit")[0].strip())
            lines.append(f"where {combined_where}")

            if sample_size:
                lines.append(f"limit {sample_size}")
            elif fail_fast:
                lines.append("limit 1")

        return "\n".join(lines)
