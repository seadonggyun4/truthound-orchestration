"""Mock Objects for Testing.

This module provides mock implementations of adapters, manifest data,
and run results for testing without external dependencies.

Example:
    >>> from truthound_dbt.testing import MockAdapter, create_sample_rules
    >>>
    >>> adapter = MockAdapter()
    >>> rules = create_sample_rules()
    >>>
    >>> # Test SQL generation
    >>> from truthound_dbt.converters import StandardRuleConverter
    >>> converter = StandardRuleConverter()
    >>> result = converter.convert_all(rules, adapter)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Iterator, Sequence

from truthound_dbt.adapters.base import (
    AdapterConfig,
    BaseSQLAdapter,
    DEFAULT_ADAPTER_CONFIG,
)


# =============================================================================
# Mock Adapter
# =============================================================================


@dataclass(frozen=True, slots=True)
class MockAdapterConfig:
    """Configuration for mock adapter.

    Attributes:
        name: Adapter name.
        regex_operator: Regex operator to use.
        random_function: Random function name.
        supports_qualify: Whether QUALIFY is supported.
        call_history: Whether to track calls.
    """

    name: str = "mock"
    regex_operator: str = "~"
    random_function: str = "random()"
    supports_qualify: bool = False
    call_history: bool = True


class MockAdapter(BaseSQLAdapter):
    """Mock SQL adapter for testing.

    This adapter records all method calls and returns configurable
    SQL fragments for testing purposes.

    Example:
        >>> adapter = MockAdapter()
        >>> sql = adapter.regex_match("email", r"^[\\w.-]+@[\\w.-]+$")
        >>> print(sql)
        email ~ '^[\\w.-]+@[\\w.-]+$'
        >>> print(adapter.call_history)
        [('regex_match', ('email', '^[\\w.-]+@[\\w.-]+$'))]
    """

    def __init__(
        self,
        config: AdapterConfig | None = None,
        mock_config: MockAdapterConfig | None = None,
    ) -> None:
        """Initialize mock adapter.

        Args:
            config: Base adapter configuration.
            mock_config: Mock-specific configuration.
        """
        super().__init__(config or DEFAULT_ADAPTER_CONFIG)
        self._mock_config = mock_config or MockAdapterConfig()
        self._call_history: list[tuple[str, tuple[Any, ...]]] = []

    @property
    def name(self) -> str:
        """Return adapter name."""
        return self._mock_config.name

    @property
    def call_history(self) -> list[tuple[str, tuple[Any, ...]]]:
        """Return call history."""
        return self._call_history

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()

    def _record_call(self, method: str, *args: Any) -> None:
        """Record a method call."""
        if self._mock_config.call_history:
            self._call_history.append((method, args))

    def regex_match(self, column: str, pattern: str) -> str:
        """Generate mock regex match SQL."""
        self._record_call("regex_match", column, pattern)
        escaped = pattern.replace("'", "''")
        return f"{column} {self._mock_config.regex_operator} '{escaped}'"

    def current_timestamp(self) -> str:
        """Return current timestamp function."""
        self._record_call("current_timestamp")
        return "current_timestamp"

    def date_parse(self, column: str, format_str: str) -> str:
        """Generate date parse SQL."""
        self._record_call("date_parse", column, format_str)
        return f"to_date({column}, '{format_str}')"

    def try_date_parse(self, column: str, format_str: str) -> str:
        """Generate safe date parse SQL."""
        self._record_call("try_date_parse", column, format_str)
        return f"try_to_date({column}, '{format_str}')"

    def limit_sample(self, n: int) -> str:
        """Generate sample SQL."""
        self._record_call("limit_sample", n)
        return f"order by {self._mock_config.random_function} limit {n}"

    def unique_check_sql(self, model: str, column: str) -> str:
        """Generate unique check SQL."""
        self._record_call("unique_check_sql", model, column)
        if self._mock_config.supports_qualify:
            return f"select * from {model} qualify count(*) over (partition by {column}) > 1"
        return f"""select t.*
from {model} t
inner join (
    select {column}
    from {model}
    group by {column}
    having count(*) > 1
) duplicates
on t.{column} = duplicates.{column}"""


# =============================================================================
# Mock Manifest
# =============================================================================


@dataclass(frozen=True, slots=True)
class MockManifestConfig:
    """Configuration for mock manifest.

    Attributes:
        project_name: dbt project name.
        dbt_version: dbt version string.
        include_truthound_tests: Include Truthound-prefixed tests.
        test_count: Number of tests to generate.
        model_count: Number of models to generate.
    """

    project_name: str = "test_project"
    dbt_version: str = "1.7.0"
    include_truthound_tests: bool = True
    test_count: int = 5
    model_count: int = 3


class MockManifest:
    """Mock manifest generator for testing.

    This class generates realistic manifest.json data for testing
    the ManifestParser without requiring actual dbt projects.

    Example:
        >>> manifest = MockManifest()
        >>> data = manifest.generate()
        >>> print(data["metadata"]["dbt_version"])
        1.7.0
    """

    def __init__(self, config: MockManifestConfig | None = None) -> None:
        """Initialize mock manifest.

        Args:
            config: Manifest configuration.
        """
        self._config = config or MockManifestConfig()

    @property
    def config(self) -> MockManifestConfig:
        """Return configuration."""
        return self._config

    def generate(self) -> dict[str, Any]:
        """Generate mock manifest data.

        Returns:
            Dictionary representing manifest.json content.
        """
        models = self._generate_models()
        tests = self._generate_tests(models)

        return {
            "metadata": {
                "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v11.json",
                "dbt_version": self._config.dbt_version,
                "project_name": self._config.project_name,
                "project_id": "test-project-id",
                "generated_at": datetime.now().isoformat(),
            },
            "nodes": {**models, **tests},
            "sources": {},
            "macros": {},
            "docs": {},
            "exposures": {},
            "selectors": {},
        }

    def write_to_file(self, path: Path) -> None:
        """Write manifest to a file.

        Args:
            path: Path to write manifest.json.
        """
        data = self.generate()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create_temp_file(self) -> Path:
        """Create a temporary manifest file.

        Returns:
            Path to the temporary file.
        """
        f = NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        data = self.generate()
        json.dump(data, f, indent=2)
        f.close()
        return Path(f.name)

    def _generate_models(self) -> dict[str, Any]:
        """Generate model nodes."""
        models = {}
        for i in range(self._config.model_count):
            name = f"model_{i + 1}"
            unique_id = f"model.{self._config.project_name}.{name}"
            models[unique_id] = {
                "resource_type": "model",
                "name": name,
                "unique_id": unique_id,
                "description": f"Test model {i + 1}",
                "columns": {
                    "id": {"name": "id", "description": "Primary key"},
                    "name": {"name": "name", "description": "Name field"},
                    "email": {"name": "email", "description": "Email address"},
                    "created_at": {"name": "created_at", "description": "Creation timestamp"},
                },
                "tags": ["test"],
                "config": {"materialized": "table"},
                "depends_on": {"nodes": [], "macros": []},
                "meta": {},
            }
        return models

    def _generate_tests(self, models: dict[str, Any]) -> dict[str, Any]:
        """Generate test nodes."""
        tests = {}
        test_types = ["not_null", "unique", "accepted_values"]

        if self._config.include_truthound_tests:
            test_types.extend([
                "truthound_email_format",
                "truthound_range_check",
            ])

        model_ids = list(models.keys())

        for i in range(self._config.test_count):
            model_id = model_ids[i % len(model_ids)]
            model_name = models[model_id]["name"]
            test_type = test_types[i % len(test_types)]
            column = "email" if "email" in test_type else "id"

            test_name = f"{test_type}_{model_name}_{column}"
            unique_id = f"test.{self._config.project_name}.{test_name}"

            tests[unique_id] = {
                "resource_type": "test",
                "name": test_name,
                "unique_id": unique_id,
                "column_name": column,
                "tags": ["data_quality"],
                "config": {"severity": "warn", "enabled": True},
                "depends_on": {"nodes": [model_id], "macros": []},
                "meta": {},
                "test_metadata": {"name": test_type},
            }

        return tests


# =============================================================================
# Mock Run Results
# =============================================================================


@dataclass(frozen=True, slots=True)
class MockRunResultsConfig:
    """Configuration for mock run results.

    Attributes:
        test_count: Number of test results.
        pass_rate: Percentage of tests that pass (0.0 to 1.0).
        include_timing: Include timing information.
        elapsed_time: Total elapsed time.
    """

    test_count: int = 5
    pass_rate: float = 0.8
    include_timing: bool = True
    elapsed_time: float = 10.5


class MockRunResults:
    """Mock run results generator for testing.

    This class generates realistic run_results.json data for testing
    the RunResultsParser.

    Example:
        >>> results = MockRunResults()
        >>> data = results.generate()
        >>> print(data["elapsed_time"])
        10.5
    """

    def __init__(self, config: MockRunResultsConfig | None = None) -> None:
        """Initialize mock run results.

        Args:
            config: Run results configuration.
        """
        self._config = config or MockRunResultsConfig()

    @property
    def config(self) -> MockRunResultsConfig:
        """Return configuration."""
        return self._config

    def generate(self) -> dict[str, Any]:
        """Generate mock run results data.

        Returns:
            Dictionary representing run_results.json content.
        """
        results = []
        pass_count = int(self._config.test_count * self._config.pass_rate)

        for i in range(self._config.test_count):
            passed = i < pass_count
            status = "pass" if passed else "fail"

            result = {
                "unique_id": f"test.test_project.test_{i + 1}",
                "status": status,
                "execution_time": 0.5 + (i * 0.1),
                "message": None if passed else f"Test failed: {i + 1} failures found",
                "failures": 0 if passed else i + 1,
                "adapter_response": {},
            }

            if self._config.include_timing:
                now = datetime.now()
                result["timing"] = [
                    {
                        "name": "compile",
                        "started_at": now.isoformat(),
                        "completed_at": now.isoformat(),
                    },
                    {
                        "name": "execute",
                        "started_at": now.isoformat(),
                        "completed_at": now.isoformat(),
                    },
                ]

            results.append(result)

        return {
            "metadata": {
                "dbt_schema_version": "https://schemas.getdbt.com/dbt/run-results/v5.json",
                "generated_at": datetime.now().isoformat(),
            },
            "results": results,
            "elapsed_time": self._config.elapsed_time,
        }

    def write_to_file(self, path: Path) -> None:
        """Write run results to a file.

        Args:
            path: Path to write run_results.json.
        """
        data = self.generate()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create_temp_file(self) -> Path:
        """Create a temporary run results file.

        Returns:
            Path to the temporary file.
        """
        f = NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        )
        data = self.generate()
        json.dump(data, f, indent=2)
        f.close()
        return Path(f.name)


# =============================================================================
# Factory Functions
# =============================================================================


def create_mock_test_result(
    *,
    unique_id: str = "test.project.test_1",
    passed: bool = True,
    execution_time: float = 0.5,
    failures: int = 0,
    message: str | None = None,
) -> dict[str, Any]:
    """Create a mock test result dictionary.

    Args:
        unique_id: Test unique identifier.
        passed: Whether test passed.
        execution_time: Execution time in seconds.
        failures: Number of failures.
        message: Result message.

    Returns:
        Dictionary representing a test result.
    """
    return {
        "unique_id": unique_id,
        "status": "pass" if passed else "fail",
        "execution_time": execution_time,
        "message": message,
        "failures": failures,
        "adapter_response": {},
    }


def create_mock_manifest(
    *,
    project_name: str = "test_project",
    models: Sequence[str] | None = None,
    tests: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a mock manifest dictionary.

    Args:
        project_name: dbt project name.
        models: List of model names.
        tests: List of test configurations.

    Returns:
        Dictionary representing manifest.json.
    """
    if models is None:
        models = ["stg_users", "stg_orders"]

    if tests is None:
        tests = [
            {"type": "not_null", "column": "id", "model": models[0]},
            {"type": "unique", "column": "id", "model": models[0]},
        ]

    nodes = {}

    # Add models
    for model_name in models:
        unique_id = f"model.{project_name}.{model_name}"
        nodes[unique_id] = {
            "resource_type": "model",
            "name": model_name,
            "unique_id": unique_id,
            "columns": {
                "id": {"name": "id"},
                "created_at": {"name": "created_at"},
            },
            "config": {},
            "depends_on": {"nodes": []},
        }

    # Add tests
    for i, test in enumerate(tests):
        test_name = f"{test['type']}_{test.get('model', models[0])}_{test.get('column', 'id')}"
        unique_id = f"test.{project_name}.{test_name}"
        model_id = f"model.{project_name}.{test.get('model', models[0])}"

        nodes[unique_id] = {
            "resource_type": "test",
            "name": test_name,
            "unique_id": unique_id,
            "column_name": test.get("column"),
            "config": {"severity": "warn"},
            "depends_on": {"nodes": [model_id]},
            "test_metadata": {"name": test["type"]},
        }

    return {
        "metadata": {
            "dbt_version": "1.7.0",
            "project_name": project_name,
            "generated_at": datetime.now().isoformat(),
        },
        "nodes": nodes,
    }


def create_mock_run_results(
    *,
    test_results: Sequence[dict[str, Any]] | None = None,
    elapsed_time: float = 5.0,
) -> dict[str, Any]:
    """Create a mock run results dictionary.

    Args:
        test_results: List of test result dictionaries.
        elapsed_time: Total elapsed time.

    Returns:
        Dictionary representing run_results.json.
    """
    if test_results is None:
        test_results = [
            create_mock_test_result(unique_id="test.project.test_1", passed=True),
            create_mock_test_result(unique_id="test.project.test_2", passed=False, failures=3),
        ]

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
        },
        "results": list(test_results),
        "elapsed_time": elapsed_time,
    }


def create_sample_rules(
    *,
    rule_types: Sequence[str] | None = None,
    model: str = "users",
) -> list[dict[str, Any]]:
    """Create sample rule dictionaries for testing.

    Args:
        rule_types: List of rule types to include.
        model: Model name.

    Returns:
        List of rule dictionaries.
    """
    if rule_types is None:
        rule_types = ["not_null", "unique", "in_set", "in_range", "regex"]

    rules: list[dict[str, Any]] = []

    rule_templates = {
        "not_null": {"type": "not_null", "column": "id"},
        "unique": {"type": "unique", "column": "email"},
        "in_set": {
            "type": "in_set",
            "column": "status",
            "values": ["active", "inactive", "pending"],
        },
        "in_range": {
            "type": "in_range",
            "column": "age",
            "min": 0,
            "max": 150,
        },
        "regex": {
            "type": "regex",
            "column": "email",
            "pattern": r"^[\w.-]+@[\w.-]+\.\w+$",
        },
        "email_format": {"type": "email_format", "column": "email"},
        "url_format": {"type": "url_format", "column": "website"},
        "uuid_format": {"type": "uuid_format", "column": "user_id"},
        "phone_format": {"type": "phone_format", "column": "phone"},
        "ipv4_format": {"type": "ipv4_format", "column": "ip_address"},
        "date_format": {
            "type": "date_format",
            "column": "created_at",
            "format": "YYYY-MM-DD",
        },
        "min_length": {"type": "min_length", "column": "name", "length": 2},
        "max_length": {"type": "max_length", "column": "name", "length": 100},
        "referential_integrity": {
            "type": "referential_integrity",
            "column": "order_id",
            "reference_model": "orders",
            "reference_column": "id",
        },
        "expression": {
            "type": "expression",
            "expression": "total >= 0",
        },
        "row_count": {"type": "row_count", "min": 1},
    }

    for rule_type in rule_types:
        if rule_type in rule_templates:
            rule = dict(rule_templates[rule_type])
            rules.append(rule)

    return rules


def create_sample_models(
    *,
    count: int = 3,
    with_columns: bool = True,
    with_tests: bool = True,
) -> list[dict[str, Any]]:
    """Create sample model dictionaries for testing.

    Args:
        count: Number of models to create.
        with_columns: Include column definitions.
        with_tests: Include test definitions.

    Returns:
        List of model dictionaries.
    """
    models = []

    for i in range(count):
        model = {
            "name": f"model_{i + 1}",
            "description": f"Sample model {i + 1}",
        }

        if with_columns:
            model["columns"] = [
                {"name": "id", "description": "Primary key"},
                {"name": "name", "description": "Name field"},
                {"name": "email", "description": "Email address"},
                {"name": "created_at", "description": "Creation timestamp"},
            ]

            if with_tests:
                model["columns"][0]["tests"] = ["not_null", "unique"]
                model["columns"][2]["tests"] = ["not_null"]

        models.append(model)

    return models


# =============================================================================
# Fixtures
# =============================================================================


def sample_manifest_data() -> dict[str, Any]:
    """Get sample manifest data for testing.

    Returns:
        Sample manifest dictionary.
    """
    return create_mock_manifest(
        project_name="sample_project",
        models=["stg_users", "stg_orders", "fct_sales"],
        tests=[
            {"type": "not_null", "column": "id", "model": "stg_users"},
            {"type": "unique", "column": "id", "model": "stg_users"},
            {"type": "not_null", "column": "user_id", "model": "stg_orders"},
            {"type": "truthound_email_format", "column": "email", "model": "stg_users"},
        ],
    )


def sample_run_results_data() -> dict[str, Any]:
    """Get sample run results data for testing.

    Returns:
        Sample run results dictionary.
    """
    return create_mock_run_results(
        test_results=[
            create_mock_test_result(
                unique_id="test.sample_project.not_null_stg_users_id",
                passed=True,
                execution_time=0.3,
            ),
            create_mock_test_result(
                unique_id="test.sample_project.unique_stg_users_id",
                passed=True,
                execution_time=0.5,
            ),
            create_mock_test_result(
                unique_id="test.sample_project.not_null_stg_orders_user_id",
                passed=False,
                failures=5,
                message="5 null values found",
            ),
        ],
        elapsed_time=3.5,
    )
