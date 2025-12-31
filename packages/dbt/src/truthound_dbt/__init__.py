"""Truthound dbt Integration Package.

This package provides Python-based utilities for dbt data quality integration.
It enables programmatic rule generation, manifest parsing, cross-adapter SQL
generation, and seamless integration with the Truthound orchestration framework.

Key Features:
    - Cross-Adapter SQL Generation: Generate database-specific SQL for data quality checks
    - Rule-to-SQL Conversion: Convert common rule format to dbt-compatible SQL
    - Manifest Parsing: Extract and analyze Truthound tests from dbt manifest.json
    - Test Generation: Programmatically create dbt test configurations
    - Result Integration: Convert dbt test results to common CheckResult format

Architecture:
    The package follows protocol-based architecture for maximum extensibility:

    ```
    truthound_dbt/
    ├── adapters/           # Database-specific SQL adapters (Protocol-based)
    │   ├── base.py         # SQLAdapter Protocol
    │   ├── postgres.py     # PostgreSQL adapter
    │   ├── snowflake.py    # Snowflake adapter
    │   ├── bigquery.py     # BigQuery adapter
    │   ├── redshift.py     # Redshift adapter
    │   └── databricks.py   # Databricks adapter
    ├── converters/         # Rule-to-SQL converters
    │   ├── base.py         # RuleConverter Protocol
    │   └── rules.py        # Rule implementations
    ├── generators/         # SQL and test generators
    │   ├── sql.py          # SQL query generator
    │   └── test.py         # dbt test YAML generator
    ├── parsers/            # Manifest and result parsers
    │   ├── manifest.py     # Enhanced manifest parser
    │   └── results.py      # Run results parser
    ├── hooks/              # Lifecycle hooks
    │   └── base.py         # Hook protocols and implementations
    └── testing/            # Test utilities
        └── mocks.py        # Mock objects for testing
    ```

Quick Start:
    >>> from truthound_dbt import SQLGenerator, get_adapter
    >>>
    >>> # Get adapter for your database
    >>> adapter = get_adapter("snowflake")
    >>>
    >>> # Generate SQL for rules
    >>> generator = SQLGenerator(adapter)
    >>> sql = generator.generate_check_sql(
    ...     model="ref('users')",
    ...     rules=[
    ...         {"type": "not_null", "column": "id"},
    ...         {"type": "email_format", "column": "email"},
    ...     ],
    ... )

Manifest Parsing:
    >>> from truthound_dbt import ManifestParser
    >>>
    >>> parser = ManifestParser("target/manifest.json")
    >>> tests = parser.get_truthound_tests()
    >>> report = parser.generate_report()
    >>> print(report.to_markdown())

Test Generation:
    >>> from truthound_dbt import TestGenerator
    >>>
    >>> generator = TestGenerator()
    >>> yaml_config = generator.generate_schema_yml(
    ...     model_name="stg_users",
    ...     rules=[
    ...         {"type": "not_null", "column": "id"},
    ...         {"type": "unique", "column": "email"},
    ...     ],
    ... )

Result Integration:
    >>> from truthound_dbt import RunResultsParser
    >>>
    >>> parser = RunResultsParser("target/run_results.json")
    >>> check_results = parser.to_check_results()

Public API:
    Adapters:
        - SQLAdapter: Protocol for database-specific SQL generation
        - PostgresAdapter, SnowflakeAdapter, BigQueryAdapter, RedshiftAdapter, DatabricksAdapter
        - AdapterRegistry, get_adapter, register_adapter, list_adapters

    Converters:
        - RuleConverter: Protocol for rule-to-SQL conversion
        - RuleConverterRegistry, get_converter, register_converter

    Generators:
        - SQLGenerator: Generate SQL from rules
        - TestGenerator: Generate dbt test YAML configurations

    Parsers:
        - ManifestParser: Parse dbt manifest.json
        - RunResultsParser: Parse dbt run_results.json

    Hooks:
        - DbtHook: Protocol for lifecycle events
        - LoggingDbtHook, MetricsDbtHook

    Testing:
        - MockAdapter, MockManifest, create_mock_test_result
"""

__version__ = "0.1.0"

# =============================================================================
# Adapters
# =============================================================================
from truthound_dbt.adapters import (
    # Protocol
    SQLAdapter,
    # Implementations
    PostgresAdapter,
    SnowflakeAdapter,
    BigQueryAdapter,
    RedshiftAdapter,
    DatabricksAdapter,
    # Registry
    AdapterRegistry,
    get_adapter,
    get_adapter_registry,
    register_adapter,
    list_adapters,
    # Exceptions
    AdapterError,
    AdapterNotFoundError,
    UnsupportedOperationError,
)

# =============================================================================
# Converters
# =============================================================================
from truthound_dbt.converters import (
    # Protocol
    RuleConverter,
    # Implementations
    StandardRuleConverter,
    # Registry
    RuleConverterRegistry,
    get_converter,
    get_converter_registry,
    register_converter,
    list_converters,
    # Types
    ConversionResult,
    RuleSQL,
    # Exceptions
    ConversionError,
    UnsupportedRuleError,
)

# =============================================================================
# Generators
# =============================================================================
from truthound_dbt.generators import (
    # SQL Generator
    SQLGenerator,
    SQLGeneratorConfig,
    GeneratedSQL,
    # Test Generator
    TestGenerator,
    TestGeneratorConfig,
    GeneratedTest,
    # Schema Generator
    SchemaGenerator,
    ModelSchema,
    ColumnSchema,
)

# =============================================================================
# Parsers
# =============================================================================
from truthound_dbt.parsers import (
    # Manifest Parser
    ManifestParser,
    TruthoundTest,
    TruthoundRule,
    TruthoundReport,
    # Run Results Parser
    RunResultsParser,
    TestResult,
    RunSummary,
    # Check Category
    CheckCategory,
)

# =============================================================================
# Hooks
# =============================================================================
from truthound_dbt.hooks import (
    # Protocol
    DbtHook,
    AsyncDbtHook,
    # Implementations
    LoggingDbtHook,
    MetricsDbtHook,
    CompositeDbtHook,
    # Events
    DbtEvent,
    TestStartEvent,
    TestEndEvent,
    ParseStartEvent,
    ParseEndEvent,
)

# =============================================================================
# Testing
# =============================================================================
from truthound_dbt.testing import (
    MockAdapter,
    MockManifest,
    MockRunResults,
    create_mock_test_result,
    create_mock_manifest,
    create_sample_rules,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Adapters - Protocol
    "SQLAdapter",
    # Adapters - Implementations
    "PostgresAdapter",
    "SnowflakeAdapter",
    "BigQueryAdapter",
    "RedshiftAdapter",
    "DatabricksAdapter",
    # Adapters - Registry
    "AdapterRegistry",
    "get_adapter",
    "get_adapter_registry",
    "register_adapter",
    "list_adapters",
    # Adapters - Exceptions
    "AdapterError",
    "AdapterNotFoundError",
    "UnsupportedOperationError",
    # Converters - Protocol
    "RuleConverter",
    # Converters - Implementations
    "StandardRuleConverter",
    # Converters - Registry
    "RuleConverterRegistry",
    "get_converter",
    "get_converter_registry",
    "register_converter",
    "list_converters",
    # Converters - Types
    "ConversionResult",
    "RuleSQL",
    # Converters - Exceptions
    "ConversionError",
    "UnsupportedRuleError",
    # Generators - SQL
    "SQLGenerator",
    "SQLGeneratorConfig",
    "GeneratedSQL",
    # Generators - Test
    "TestGenerator",
    "TestGeneratorConfig",
    "GeneratedTest",
    # Generators - Schema
    "SchemaGenerator",
    "ModelSchema",
    "ColumnSchema",
    # Parsers - Manifest
    "ManifestParser",
    "TruthoundTest",
    "TruthoundRule",
    "TruthoundReport",
    # Parsers - Run Results
    "RunResultsParser",
    "TestResult",
    "RunSummary",
    # Parsers - Enums
    "CheckCategory",
    # Hooks - Protocol
    "DbtHook",
    "AsyncDbtHook",
    # Hooks - Implementations
    "LoggingDbtHook",
    "MetricsDbtHook",
    "CompositeDbtHook",
    # Hooks - Events
    "DbtEvent",
    "TestStartEvent",
    "TestEndEvent",
    "ParseStartEvent",
    "ParseEndEvent",
    # Testing
    "MockAdapter",
    "MockManifest",
    "MockRunResults",
    "create_mock_test_result",
    "create_mock_manifest",
    "create_sample_rules",
]
