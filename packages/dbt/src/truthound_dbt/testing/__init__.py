"""Testing Utilities for truthound_dbt.

This module provides mock objects and utilities for testing dbt integration
code without requiring actual dbt installations or databases.

Components:
    - MockAdapter: Mock SQL adapter for testing
    - MockManifest: Mock manifest data generator
    - MockRunResults: Mock run results generator
    - create_mock_test_result: Factory for test results
    - create_mock_manifest: Factory for manifest data
    - create_sample_rules: Factory for sample rules

Example:
    >>> from truthound_dbt.testing import MockAdapter, create_sample_rules
    >>>
    >>> adapter = MockAdapter()
    >>> rules = create_sample_rules(rule_types=["not_null", "unique"])
    >>>
    >>> # Use in tests
    >>> from truthound_dbt.generators import SQLGenerator
    >>> generator = SQLGenerator(adapter)
    >>> result = generator.generate_check_sql("test_model", rules)
"""

from truthound_dbt.testing.mocks import (
    # Adapters
    MockAdapter,
    MockAdapterConfig,
    # Manifest
    MockManifest,
    MockManifestConfig,
    # Run Results
    MockRunResults,
    MockRunResultsConfig,
    # Factories
    create_mock_test_result,
    create_mock_manifest,
    create_mock_run_results,
    create_sample_rules,
    create_sample_models,
    # Fixtures
    sample_manifest_data,
    sample_run_results_data,
)

__all__ = [
    # Adapters
    "MockAdapter",
    "MockAdapterConfig",
    # Manifest
    "MockManifest",
    "MockManifestConfig",
    # Run Results
    "MockRunResults",
    "MockRunResultsConfig",
    # Factories
    "create_mock_test_result",
    "create_mock_manifest",
    "create_mock_run_results",
    "create_sample_rules",
    "create_sample_models",
    # Fixtures
    "sample_manifest_data",
    "sample_run_results_data",
]
