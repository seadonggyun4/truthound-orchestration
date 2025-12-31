"""SQL and Test Generators for dbt integration.

This module provides generators for creating SQL queries and dbt test
configurations from data quality rules.

Components:
    - SQLGenerator: Generates SQL queries from rules using adapters
    - TestGenerator: Generates dbt test YAML configurations
    - SchemaGenerator: Generates dbt schema.yml files

Example:
    >>> from truthound_dbt.generators import SQLGenerator, TestGenerator
    >>> from truthound_dbt.adapters import get_adapter
    >>>
    >>> # SQL Generation
    >>> adapter = get_adapter("snowflake")
    >>> sql_gen = SQLGenerator(adapter)
    >>> sql = sql_gen.generate_check_sql(
    ...     model="ref('users')",
    ...     rules=[{"type": "not_null", "column": "id"}],
    ... )
    >>>
    >>> # Test Generation
    >>> test_gen = TestGenerator()
    >>> yaml_config = test_gen.generate_schema_yml(
    ...     model_name="stg_users",
    ...     rules=[{"type": "not_null", "column": "id"}],
    ... )
"""

from truthound_dbt.generators.sql import (
    # Main generator
    SQLGenerator,
    # Configuration
    SQLGeneratorConfig,
    DEFAULT_SQL_GENERATOR_CONFIG,
    # Result types
    GeneratedSQL,
    SQLFragment,
    # Exceptions
    SQLGenerationError,
    InvalidRuleError,
    AdapterNotConfiguredError,
)

from truthound_dbt.generators.test import (
    # Main generator
    TestGenerator,
    # Configuration
    TestGeneratorConfig,
    DEFAULT_TEST_GENERATOR_CONFIG,
    # Result types
    GeneratedTest,
    TestConfig,
    ColumnTest,
    # Exceptions
    TestGenerationError,
)

from truthound_dbt.generators.schema import (
    # Main generator
    SchemaGenerator,
    # Configuration
    SchemaGeneratorConfig,
    DEFAULT_SCHEMA_GENERATOR_CONFIG,
    # Result types
    ModelSchema,
    ColumnSchema,
    SchemaYAML,
    # Exceptions
    SchemaGenerationError,
)

__all__ = [
    # SQL Generator
    "SQLGenerator",
    "SQLGeneratorConfig",
    "DEFAULT_SQL_GENERATOR_CONFIG",
    "GeneratedSQL",
    "SQLFragment",
    "SQLGenerationError",
    "InvalidRuleError",
    "AdapterNotConfiguredError",
    # Test Generator
    "TestGenerator",
    "TestGeneratorConfig",
    "DEFAULT_TEST_GENERATOR_CONFIG",
    "GeneratedTest",
    "TestConfig",
    "ColumnTest",
    "TestGenerationError",
    # Schema Generator
    "SchemaGenerator",
    "SchemaGeneratorConfig",
    "DEFAULT_SCHEMA_GENERATOR_CONFIG",
    "ModelSchema",
    "ColumnSchema",
    "SchemaYAML",
    "SchemaGenerationError",
]
