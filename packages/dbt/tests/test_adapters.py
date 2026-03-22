"""Tests for SQL adapters."""

import pytest

from truthound_dbt.adapters import (
    PostgresAdapter,
    SnowflakeAdapter,
    BigQueryAdapter,
    RedshiftAdapter,
    DatabricksAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    reset_adapter_registry,
    AdapterNotFoundError,
)


class TestAdapterRegistry:
    """Tests for adapter registry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_adapter_registry()

    def test_list_adapters_returns_all_registered(self):
        """Test listing all adapters."""
        adapters = list_adapters()
        assert "postgres" in adapters
        assert "snowflake" in adapters
        assert "bigquery" in adapters
        assert "redshift" in adapters
        assert "databricks" in adapters

    def test_get_adapter_postgres(self):
        """Test getting PostgreSQL adapter."""
        adapter = get_adapter("postgres")
        assert adapter.name == "postgres"

    def test_get_adapter_snowflake(self):
        """Test getting Snowflake adapter."""
        adapter = get_adapter("snowflake")
        assert adapter.name == "snowflake"

    def test_get_adapter_bigquery(self):
        """Test getting BigQuery adapter."""
        adapter = get_adapter("bigquery")
        assert adapter.name == "bigquery"

    def test_get_adapter_default(self):
        """Test getting default adapter."""
        adapter = get_adapter("default")
        assert adapter.name == "postgres"

    def test_get_adapter_not_found_raises(self):
        """Test getting non-existent adapter raises error."""
        with pytest.raises(AdapterNotFoundError):
            get_adapter("nonexistent")


class TestPostgresAdapter:
    """Tests for PostgreSQL adapter."""

    def setup_method(self):
        """Create adapter instance."""
        self.adapter = PostgresAdapter()

    def test_name(self):
        """Test adapter name."""
        assert self.adapter.name == "postgres"

    def test_regex_match(self):
        """Test regex match SQL generation."""
        sql = self.adapter.regex_match("email", r"^[\w.-]+@[\w.-]+$")
        assert "~" in sql
        assert "email" in sql

    def test_regex_match_escapes_quotes(self):
        """Test regex match escapes single quotes."""
        sql = self.adapter.regex_match("col", "pattern'with'quotes")
        assert "''" in sql

    def test_current_timestamp(self):
        """Test current timestamp generation."""
        sql = self.adapter.current_timestamp()
        assert sql == "current_timestamp"

    def test_date_parse(self):
        """Test date parse SQL."""
        sql = self.adapter.date_parse("date_col", "YYYY-MM-DD")
        assert "to_date" in sql
        assert "YYYY-MM-DD" in sql

    def test_limit_sample(self):
        """Test random sampling SQL."""
        sql = self.adapter.limit_sample(100)
        assert "random()" in sql
        assert "limit 100" in sql

    def test_unique_check_sql(self):
        """Test unique check SQL generation."""
        sql = self.adapter.unique_check_sql("users", "email")
        assert "users" in sql
        assert "email" in sql
        assert "group by" in sql.lower()
        assert "having count" in sql.lower()


class TestSnowflakeAdapter:
    """Tests for Snowflake adapter."""

    def setup_method(self):
        """Create adapter instance."""
        self.adapter = SnowflakeAdapter()

    def test_name(self):
        """Test adapter name."""
        assert self.adapter.name == "snowflake"

    def test_regex_match_uses_regexp_like(self):
        """Test regex uses regexp_like function."""
        sql = self.adapter.regex_match("email", r"^[\w.-]+@[\w.-]+$")
        assert "regexp_like" in sql

    def test_unique_check_uses_qualify(self):
        """Test unique check uses QUALIFY clause."""
        sql = self.adapter.unique_check_sql("users", "email")
        assert "qualify" in sql.lower()
        assert "partition by" in sql.lower()

    def test_try_date_parse(self):
        """Test safe date parse SQL."""
        sql = self.adapter.try_date_parse("date_col", "YYYY-MM-DD")
        assert "try_to_date" in sql

    def test_limit_sample(self):
        """Test random sampling with SAMPLE."""
        sql = self.adapter.limit_sample(100)
        assert "sample" in sql.lower()


class TestBigQueryAdapter:
    """Tests for BigQuery adapter."""

    def setup_method(self):
        """Create adapter instance."""
        self.adapter = BigQueryAdapter()

    def test_name(self):
        """Test adapter name."""
        assert self.adapter.name == "bigquery"

    def test_regex_match_uses_regexp_contains(self):
        """Test regex uses REGEXP_CONTAINS."""
        sql = self.adapter.regex_match("email", r"^[\w.-]+@[\w.-]+$")
        assert "regexp_contains" in sql.lower()

    def test_regex_match_uses_raw_string(self):
        """Test regex uses raw string syntax."""
        sql = self.adapter.regex_match("email", r"pattern")
        assert "r'" in sql

    def test_unique_check_uses_qualify(self):
        """Test unique check uses QUALIFY clause."""
        sql = self.adapter.unique_check_sql("users", "email")
        assert "qualify" in sql.lower()


class TestRedshiftAdapter:
    """Tests for Redshift adapter."""

    def setup_method(self):
        """Create adapter instance."""
        self.adapter = RedshiftAdapter()

    def test_name(self):
        """Test adapter name."""
        assert self.adapter.name == "redshift"

    def test_regex_match_uses_tilde(self):
        """Test regex uses ~ operator (PostgreSQL-compatible)."""
        sql = self.adapter.regex_match("email", r"^[\w.-]+@[\w.-]+$")
        assert "~" in sql

    def test_unique_check_uses_subquery(self):
        """Test unique check uses subquery (no QUALIFY)."""
        sql = self.adapter.unique_check_sql("users", "email")
        assert "qualify" not in sql.lower()
        assert "inner join" in sql.lower()


class TestDatabricksAdapter:
    """Tests for Databricks adapter."""

    def setup_method(self):
        """Create adapter instance."""
        self.adapter = DatabricksAdapter()

    def test_name(self):
        """Test adapter name."""
        assert self.adapter.name == "databricks"

    def test_regex_match_uses_rlike(self):
        """Test regex uses RLIKE operator."""
        sql = self.adapter.regex_match("email", r"^[\w.-]+@[\w.-]+$")
        assert "rlike" in sql

    def test_unique_check_uses_qualify(self):
        """Test unique check uses QUALIFY clause."""
        sql = self.adapter.unique_check_sql("users", "email")
        assert "qualify" in sql.lower()

    def test_limit_sample(self):
        """Test random sampling with rand()."""
        sql = self.adapter.limit_sample(100)
        assert "rand()" in sql


class TestCrossAdapterConsistency:
    """Tests for cross-adapter behavior consistency."""

    @pytest.fixture
    def all_adapters(self):
        """Get all adapter instances."""
        return [
            PostgresAdapter(),
            SnowflakeAdapter(),
            BigQueryAdapter(),
            RedshiftAdapter(),
            DatabricksAdapter(),
        ]

    def test_all_adapters_have_name(self, all_adapters):
        """Test all adapters return a name."""
        for adapter in all_adapters:
            assert adapter.name
            assert isinstance(adapter.name, str)

    def test_all_adapters_generate_regex_match(self, all_adapters):
        """Test all adapters generate regex match SQL."""
        for adapter in all_adapters:
            sql = adapter.regex_match("col", "pattern")
            assert sql
            assert "col" in sql

    def test_all_adapters_generate_unique_check(self, all_adapters):
        """Test all adapters generate unique check SQL."""
        for adapter in all_adapters:
            sql = adapter.unique_check_sql("table", "column")
            assert sql
            assert "table" in sql
            assert "column" in sql

    def test_all_adapters_generate_limit_sample(self, all_adapters):
        """Test all adapters generate sample SQL."""
        for adapter in all_adapters:
            sql = adapter.limit_sample(100)
            assert sql
            assert "100" in sql
