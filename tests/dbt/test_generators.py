"""Tests for SQL and test generators."""

import pytest

from truthound_dbt.generators import (
    SQLGenerator,
    SQLGeneratorConfig,
    GeneratedSQL,
    TestGenerator,
    TestGeneratorConfig,
    GeneratedTest,
    SchemaGenerator,
    ModelSchema,
    ColumnSchema,
)
from truthound_dbt.testing import MockAdapter, create_sample_rules


class TestSQLGenerator:
    """Tests for SQLGenerator."""

    def setup_method(self):
        """Create generator and adapter."""
        self.adapter = MockAdapter()
        self.generator = SQLGenerator(self.adapter)

    def test_generate_check_sql_empty_rules(self):
        """Test generating SQL with no rules."""
        result = self.generator.generate_check_sql("users", [])
        assert isinstance(result, GeneratedSQL)
        assert result.rule_count == 0
        assert "1=0" in result.sql  # No results expected

    def test_generate_check_sql_single_rule(self):
        """Test generating SQL with single rule."""
        rules = [{"type": "not_null", "column": "id"}]
        result = self.generator.generate_check_sql("users", rules)

        assert isinstance(result, GeneratedSQL)
        assert result.rule_count == 1
        assert result.model == "users"
        assert result.adapter_name == "mock"
        assert len(result.fragments) == 1

    def test_generate_check_sql_multiple_rules(self):
        """Test generating SQL with multiple rules."""
        rules = create_sample_rules(rule_types=["not_null", "unique", "in_range"])
        result = self.generator.generate_check_sql("users", rules)

        assert result.rule_count == 3
        assert len(result.fragments) == 3

    def test_generate_check_sql_includes_cte(self):
        """Test SQL uses CTE by default."""
        rules = create_sample_rules(rule_types=["not_null", "unique"])
        result = self.generator.generate_check_sql("users", rules)

        assert "with" in result.sql.lower()
        assert "source_data" in result.sql.lower()

    def test_generate_check_sql_with_sample_size(self):
        """Test SQL with sample size."""
        rules = [{"type": "not_null", "column": "id"}]
        result = self.generator.generate_check_sql("users", rules, sample_size=1000)

        assert "1000" in result.sql
        assert result.metadata.get("sample_size") == 1000

    def test_generate_check_sql_fail_fast(self):
        """Test SQL with fail_fast option."""
        rules = create_sample_rules(rule_types=["not_null", "unique"])
        result = self.generator.generate_check_sql("users", rules, fail_fast=True)

        assert result.metadata.get("fail_fast") is True
        assert "limit 1" in result.sql.lower()

    def test_generate_unique_check_sql(self):
        """Test generating unique check SQL."""
        sql = self.generator.generate_unique_check_sql("users", "email")
        assert "users" in sql
        assert "email" in sql

    def test_generate_not_null_check_sql(self):
        """Test generating not null check SQL."""
        sql = self.generator.generate_not_null_check_sql("users", "id")
        assert "users" in sql
        assert "id" in sql
        assert "is null" in sql.lower()

    def test_generate_referential_check_sql(self):
        """Test generating referential check SQL."""
        sql = self.generator.generate_referential_check_sql(
            "orders", "user_id", "users", "id"
        )
        assert "orders" in sql
        assert "users" in sql
        assert "user_id" in sql
        assert "left join" in sql.lower()

    def test_generated_sql_to_dict(self):
        """Test GeneratedSQL to_dict method."""
        rules = [{"type": "not_null", "column": "id"}]
        result = self.generator.generate_check_sql("users", rules)

        data = result.to_dict()
        assert "sql" in data
        assert "fragments" in data
        assert "model" in data
        assert "adapter_name" in data


class TestSQLGeneratorConfig:
    """Tests for SQLGeneratorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SQLGeneratorConfig()
        assert config.include_comments is True
        assert config.use_cte is True

    def test_with_comments(self):
        """Test with_comments builder."""
        config = SQLGeneratorConfig().with_comments(False)
        assert config.include_comments is False

    def test_with_cte(self):
        """Test with_cte builder."""
        config = SQLGeneratorConfig().with_cte(False)
        assert config.use_cte is False

    def test_with_sample_size(self):
        """Test with_sample_size builder."""
        config = SQLGeneratorConfig().with_sample_size(1000)
        assert config.sample_size == 1000


class TestTestGenerator:
    """Tests for TestGenerator."""

    def setup_method(self):
        """Create generator."""
        self.generator = TestGenerator()

    def test_generate_tests_empty_rules(self):
        """Test generating with no rules."""
        result = self.generator.generate_tests("users", [])
        assert result.test_count == 0
        assert result.model_name == "users"

    def test_generate_tests_not_null(self):
        """Test generating not_null test."""
        rules = [{"type": "not_null", "column": "id"}]
        result = self.generator.generate_tests("users", rules)

        assert result.test_count == 1
        assert "id" in result.column_tests
        assert len(result.column_tests["id"]) == 1

    def test_generate_tests_unique(self):
        """Test generating unique test."""
        rules = [{"type": "unique", "column": "email"}]
        result = self.generator.generate_tests("users", rules)

        assert result.test_count == 1
        assert "email" in result.column_tests

    def test_generate_tests_multiple_columns(self):
        """Test generating tests for multiple columns."""
        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "id"},
            {"type": "not_null", "column": "email"},
        ]
        result = self.generator.generate_tests("users", rules)

        assert result.test_count == 3
        assert "id" in result.column_tests
        assert "email" in result.column_tests
        assert len(result.column_tests["id"]) == 2

    def test_generate_tests_yaml_content(self):
        """Test YAML content generation."""
        rules = [{"type": "not_null", "column": "id"}]
        result = self.generator.generate_tests("users", rules)

        assert "version: 2" in result.yaml_content
        assert "models:" in result.yaml_content
        assert "- name: users" in result.yaml_content

    def test_generate_tests_yaml_dict(self):
        """Test YAML dict structure."""
        rules = [{"type": "not_null", "column": "id"}]
        result = self.generator.generate_tests("users", rules)

        assert result.yaml_dict["version"] == 2
        assert len(result.yaml_dict["models"]) == 1
        assert result.yaml_dict["models"][0]["name"] == "users"

    def test_generate_tests_with_extra_tags(self):
        """Test generating tests with extra tags."""
        rules = [{"type": "not_null", "column": "id"}]
        result = self.generator.generate_tests(
            "users", rules, extra_tags=["critical"]
        )

        test_config = result.column_tests["id"][0]
        assert "critical" in test_config.tags

    def test_generate_column_tests(self):
        """Test generating tests for specific column."""
        rules = [
            {"type": "not_null"},
            {"type": "unique"},
        ]
        tests = self.generator.generate_column_tests("email", rules)

        assert len(tests) == 2
        assert all(t.column_name == "email" for t in tests)


class TestSchemaGenerator:
    """Tests for SchemaGenerator."""

    def setup_method(self):
        """Create generator."""
        self.generator = SchemaGenerator()

    def test_generate_schema_from_model_schemas(self):
        """Test generating schema from ModelSchema objects."""
        models = [
            ModelSchema(
                name="users",
                description="User data",
                columns=(
                    ColumnSchema("id", tests=("not_null", "unique")),
                    ColumnSchema("email", tests=("not_null",)),
                ),
            )
        ]
        result = self.generator.generate_schema(models)

        assert "version: 2" in result.yaml_content
        assert len(result.models) == 1
        assert result.models[0].name == "users"

    def test_generate_schema_from_dicts(self):
        """Test generating schema from dictionaries."""
        models = [
            {
                "name": "users",
                "description": "User data",
                "columns": [
                    {"name": "id", "tests": ["not_null", "unique"]},
                    {"name": "email", "tests": ["not_null"]},
                ],
            }
        ]
        result = self.generator.generate_schema(models)

        assert len(result.models) == 1
        assert result.models[0].name == "users"

    def test_generate_model_schema(self):
        """Test generating single model schema."""
        model = self.generator.generate_model_schema(
            name="users",
            description="User data",
            columns=[
                {"name": "id", "tests": ["not_null"]},
            ],
        )

        assert model.name == "users"
        assert model.description == "User data"
        assert len(model.columns) == 1

    def test_generate_column_schema(self):
        """Test generating column schema."""
        column = self.generator.generate_column_schema(
            name="email",
            description="Email address",
            tests=["not_null", "unique"],
        )

        assert column.name == "email"
        assert column.description == "Email address"
        assert len(column.tests) == 2

    def test_merge_schemas(self):
        """Test merging two schemas."""
        base = self.generator.generate_schema([
            ModelSchema(
                name="users",
                columns=(ColumnSchema("id", tests=("not_null",)),),
            )
        ])
        overlay = self.generator.generate_schema([
            ModelSchema(
                name="users",
                columns=(ColumnSchema("id", tests=("unique",)),),
            )
        ])

        merged = self.generator.merge_schemas(base, overlay)

        # Tests should be merged
        assert len(merged.models) == 1
        user_model = merged.models[0]
        id_col = next(c for c in user_model.columns if c.name == "id")
        assert "not_null" in id_col.tests
        assert "unique" in id_col.tests
