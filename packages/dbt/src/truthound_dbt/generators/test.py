"""dbt Test Configuration Generator.

This module provides generation of dbt test YAML configurations from
data quality rules.

Example:
    >>> from truthound_dbt.generators import TestGenerator
    >>>
    >>> generator = TestGenerator()
    >>> result = generator.generate_tests(
    ...     model_name="stg_users",
    ...     rules=[
    ...         {"type": "not_null", "column": "id"},
    ...         {"type": "unique", "column": "email"},
    ...     ],
    ... )
    >>> print(result.yaml_content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class TestGenerationError(Exception):
    """Base exception for test generation errors."""

    pass


# =============================================================================
# Enums
# =============================================================================


class TestType(str, Enum):
    """dbt test types."""

    BUILT_IN = "built_in"
    GENERIC = "generic"
    SINGULAR = "singular"
    TRUTHOUND = "truthound"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class TestGeneratorConfig:
    """Configuration for test generator.

    Attributes:
        use_truthound_tests: Use Truthound generic tests when available.
        include_severity: Include severity in test config.
        include_description: Include description in test config.
        include_tags: Include tags in test config.
        default_severity: Default severity for tests.
        default_tags: Default tags for all tests.
        indent_size: YAML indentation size.
    """

    use_truthound_tests: bool = True
    include_severity: bool = True
    include_description: bool = True
    include_tags: bool = True
    default_severity: str = "warn"
    default_tags: tuple[str, ...] = ("data_quality",)
    indent_size: int = 2

    def with_severity(self, severity: str) -> TestGeneratorConfig:
        """Return config with default severity."""
        return TestGeneratorConfig(
            use_truthound_tests=self.use_truthound_tests,
            include_severity=self.include_severity,
            include_description=self.include_description,
            include_tags=self.include_tags,
            default_severity=severity,
            default_tags=self.default_tags,
            indent_size=self.indent_size,
        )

    def with_tags(self, *tags: str) -> TestGeneratorConfig:
        """Return config with default tags."""
        return TestGeneratorConfig(
            use_truthound_tests=self.use_truthound_tests,
            include_severity=self.include_severity,
            include_description=self.include_description,
            include_tags=self.include_tags,
            default_severity=self.default_severity,
            default_tags=tags,
            indent_size=self.indent_size,
        )


DEFAULT_TEST_GENERATOR_CONFIG = TestGeneratorConfig()


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class ColumnTest:
    """A test configuration for a column.

    Attributes:
        column_name: Name of the column.
        test_name: Name of the test.
        test_type: Type of test (built_in, generic, etc).
        config: Test configuration parameters.
    """

    column_name: str
    test_name: str
    test_type: TestType = TestType.BUILT_IN
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML."""
        if not self.config:
            return self.test_name
        return {self.test_name: self.config}


@dataclass(frozen=True, slots=True)
class TestConfig:
    """Configuration for a model test.

    Attributes:
        test_name: Name of the test.
        test_type: Type of test.
        column_name: Column name (for column-level tests).
        config: Test configuration parameters.
        severity: Test severity.
        tags: Test tags.
        description: Test description.
    """

    test_name: str
    test_type: TestType
    column_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    severity: str | None = None
    tags: tuple[str, ...] = ()
    description: str | None = None

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML output."""
        result: dict[str, Any] = {}

        if self.config:
            result = {**self.config}

        if self.severity:
            result["severity"] = self.severity

        if self.tags:
            result["tags"] = list(self.tags)

        if self.description:
            result["description"] = self.description

        if not result:
            return self.test_name

        return {self.test_name: result}


@dataclass(frozen=True, slots=True)
class GeneratedTest:
    """Result of test generation.

    Attributes:
        yaml_content: The YAML content as string.
        yaml_dict: The YAML content as dictionary.
        model_name: Name of the model.
        test_count: Number of tests generated.
        column_tests: Tests organized by column.
        model_tests: Model-level tests.
        generated_at: Timestamp of generation.
    """

    yaml_content: str
    yaml_dict: dict[str, Any]
    model_name: str
    test_count: int
    column_tests: dict[str, list[TestConfig]]
    model_tests: list[TestConfig]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "yaml_content": self.yaml_content,
            "model_name": self.model_name,
            "test_count": self.test_count,
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# Rule to Test Mapping
# =============================================================================


# Mapping of rule types to dbt test configurations
RULE_TO_DBT_TEST: dict[str, dict[str, Any]] = {
    # Built-in dbt tests
    "not_null": {
        "test_name": "not_null",
        "test_type": TestType.BUILT_IN,
    },
    "unique": {
        "test_name": "unique",
        "test_type": TestType.BUILT_IN,
    },
    "accepted_values": {
        "test_name": "accepted_values",
        "test_type": TestType.BUILT_IN,
        "config_mapper": lambda rule: {"values": rule.get("values", [])},
    },
    "relationships": {
        "test_name": "relationships",
        "test_type": TestType.BUILT_IN,
        "config_mapper": lambda rule: {
            "to": rule.get("reference_model"),
            "field": rule.get("reference_column"),
        },
    },
    # Truthound custom tests
    "in_set": {
        "test_name": "truthound_accepted_values",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {"values": rule.get("values", [])},
        "fallback": {
            "test_name": "accepted_values",
            "test_type": TestType.BUILT_IN,
            "config_mapper": lambda rule: {"values": rule.get("values", [])},
        },
    },
    "in_range": {
        "test_name": "truthound_range_check",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {
            "min_value": rule.get("min"),
            "max_value": rule.get("max"),
        },
    },
    "regex": {
        "test_name": "truthound_regex_match",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {"pattern": rule.get("pattern", "")},
    },
    "email_format": {
        "test_name": "truthound_email_format",
        "test_type": TestType.TRUTHOUND,
    },
    "url_format": {
        "test_name": "truthound_url_format",
        "test_type": TestType.TRUTHOUND,
    },
    "uuid_format": {
        "test_name": "truthound_uuid_format",
        "test_type": TestType.TRUTHOUND,
    },
    "phone_format": {
        "test_name": "truthound_phone_format",
        "test_type": TestType.TRUTHOUND,
    },
    "ipv4_format": {
        "test_name": "truthound_ipv4_format",
        "test_type": TestType.TRUTHOUND,
    },
    "date_format": {
        "test_name": "truthound_date_format",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {"format": rule.get("format", "YYYY-MM-DD")},
    },
    "min_length": {
        "test_name": "truthound_min_length",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {"min_length": rule.get("min", rule.get("length", 0))},
    },
    "max_length": {
        "test_name": "truthound_max_length",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {"max_length": rule.get("max", rule.get("length", 0))},
    },
    "length_range": {
        "test_name": "truthound_length_range",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {
            "min_length": rule.get("min", 0),
            "max_length": rule.get("max", 0),
        },
    },
    "referential_integrity": {
        "test_name": "relationships",
        "test_type": TestType.BUILT_IN,
        "config_mapper": lambda rule: {
            "to": rule.get("reference_model"),
            "field": rule.get("reference_column"),
        },
    },
    "row_count": {
        "test_name": "truthound_row_count",
        "test_type": TestType.TRUTHOUND,
        "model_level": True,
        "config_mapper": lambda rule: {
            "min_rows": rule.get("min"),
            "max_rows": rule.get("max"),
        },
    },
    "expression": {
        "test_name": "truthound_expression",
        "test_type": TestType.TRUTHOUND,
        "config_mapper": lambda rule: {"expression": rule.get("expression", "")},
    },
}


# =============================================================================
# TestGenerator
# =============================================================================


class TestGenerator:
    """Generates dbt test YAML configurations from rules.

    This generator converts data quality rules into dbt test configurations
    that can be added to schema.yml files.

    Example:
        >>> generator = TestGenerator()
        >>> result = generator.generate_tests(
        ...     model_name="stg_users",
        ...     rules=[
        ...         {"type": "not_null", "column": "id"},
        ...         {"type": "unique", "column": "email"},
        ...     ],
        ... )
        >>> print(result.yaml_content)
    """

    def __init__(self, config: TestGeneratorConfig | None = None) -> None:
        """Initialize test generator.

        Args:
            config: Generator configuration.
        """
        self._config = config or DEFAULT_TEST_GENERATOR_CONFIG

    @property
    def config(self) -> TestGeneratorConfig:
        """Return the generator configuration."""
        return self._config

    def generate_tests(
        self,
        model_name: str,
        rules: Sequence[dict[str, Any]],
        *,
        extra_tags: Sequence[str] | None = None,
    ) -> GeneratedTest:
        """Generate dbt test configurations from rules.

        Args:
            model_name: Name of the dbt model.
            rules: List of rule dictionaries.
            extra_tags: Additional tags to apply to all tests.

        Returns:
            GeneratedTest containing YAML content and metadata.
        """
        column_tests: dict[str, list[TestConfig]] = {}
        model_tests: list[TestConfig] = []

        tags = list(self._config.default_tags)
        if extra_tags:
            tags.extend(extra_tags)

        for rule in rules:
            test_config = self._convert_rule_to_test(rule, tags)
            if test_config is None:
                continue

            column = rule.get("column")
            if column and not self._is_model_level_test(rule):
                if column not in column_tests:
                    column_tests[column] = []
                column_tests[column].append(test_config)
            else:
                model_tests.append(test_config)

        # Build YAML dict
        yaml_dict = self._build_yaml_dict(model_name, column_tests, model_tests)

        # Convert to YAML string
        yaml_content = self._dict_to_yaml(yaml_dict)

        return GeneratedTest(
            yaml_content=yaml_content,
            yaml_dict=yaml_dict,
            model_name=model_name,
            test_count=sum(len(tests) for tests in column_tests.values()) + len(model_tests),
            column_tests=column_tests,
            model_tests=model_tests,
        )

    def generate_column_tests(
        self,
        column_name: str,
        rules: Sequence[dict[str, Any]],
    ) -> list[ColumnTest]:
        """Generate tests for a specific column.

        Args:
            column_name: Name of the column.
            rules: Rules to apply to the column.

        Returns:
            List of ColumnTest configurations.
        """
        result: list[ColumnTest] = []

        for rule in rules:
            test_config = self._convert_rule_to_test(rule, [])
            if test_config is None:
                continue

            result.append(
                ColumnTest(
                    column_name=column_name,
                    test_name=test_config.test_name,
                    test_type=test_config.test_type,
                    config=test_config.config,
                )
            )

        return result

    def _convert_rule_to_test(
        self,
        rule: dict[str, Any],
        tags: list[str],
    ) -> TestConfig | None:
        """Convert a rule to test configuration.

        Args:
            rule: Rule dictionary.
            tags: Tags to apply.

        Returns:
            TestConfig or None if rule is not supported.
        """
        rule_type = rule.get("type", "")
        mapping = RULE_TO_DBT_TEST.get(rule_type)

        if mapping is None:
            # Unsupported rule type
            return None

        # Check if we should use Truthound tests
        test_type = mapping["test_type"]
        test_name = mapping["test_name"]

        if test_type == TestType.TRUTHOUND and not self._config.use_truthound_tests:
            # Fall back to built-in if available
            fallback = mapping.get("fallback")
            if fallback:
                test_name = fallback["test_name"]
                test_type = fallback["test_type"]
                mapping = fallback
            else:
                return None

        # Build config
        config: dict[str, Any] = {}
        config_mapper = mapping.get("config_mapper")
        if config_mapper:
            config = config_mapper(rule)

        # Get severity
        severity = rule.get("severity", self._config.default_severity)
        if not self._config.include_severity:
            severity = None

        # Get description
        description = rule.get("description")
        if not self._config.include_description:
            description = None

        # Get tags
        test_tags = tuple(tags) if self._config.include_tags else ()

        return TestConfig(
            test_name=test_name,
            test_type=test_type,
            column_name=rule.get("column"),
            config=config,
            severity=severity,
            tags=test_tags,
            description=description,
        )

    def _is_model_level_test(self, rule: dict[str, Any]) -> bool:
        """Check if rule is a model-level test."""
        rule_type = rule.get("type", "")
        mapping = RULE_TO_DBT_TEST.get(rule_type, {})
        return mapping.get("model_level", False)

    def _build_yaml_dict(
        self,
        model_name: str,
        column_tests: dict[str, list[TestConfig]],
        model_tests: list[TestConfig],
    ) -> dict[str, Any]:
        """Build the YAML dictionary structure.

        Args:
            model_name: Name of the model.
            column_tests: Tests organized by column.
            model_tests: Model-level tests.

        Returns:
            Dictionary suitable for YAML output.
        """
        model_config: dict[str, Any] = {"name": model_name}

        # Add column tests
        if column_tests:
            columns: list[dict[str, Any]] = []
            for column_name, tests in sorted(column_tests.items()):
                column_entry: dict[str, Any] = {"name": column_name}
                if tests:
                    column_entry["tests"] = [t.to_yaml_dict() for t in tests]
                columns.append(column_entry)
            model_config["columns"] = columns

        # Add model-level tests
        if model_tests:
            model_config["tests"] = [t.to_yaml_dict() for t in model_tests]

        return {
            "version": 2,
            "models": [model_config],
        }

    def _dict_to_yaml(self, data: dict[str, Any]) -> str:
        """Convert dictionary to YAML string.

        A simple YAML serializer to avoid external dependencies.

        Args:
            data: Dictionary to convert.

        Returns:
            YAML formatted string.
        """
        lines: list[str] = []
        self._write_yaml_value(lines, data, 0)
        return "\n".join(lines)

    def _write_yaml_value(
        self,
        lines: list[str],
        value: Any,
        indent: int,
    ) -> None:
        """Recursively write a value as YAML.

        Args:
            lines: List of output lines.
            value: Value to write.
            indent: Current indentation level.
        """
        prefix = " " * (indent * self._config.indent_size)

        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)) and v:
                    lines.append(f"{prefix}{k}:")
                    self._write_yaml_value(lines, v, indent + 1)
                else:
                    self._write_yaml_inline(lines, k, v, indent)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    # First key on same line as dash
                    first_key = next(iter(item.keys()), None)
                    if first_key is not None:
                        first_val = item[first_key]
                        if isinstance(first_val, (dict, list)) and first_val:
                            lines.append(f"{prefix}- {first_key}:")
                            self._write_yaml_value(lines, first_val, indent + 2)
                        else:
                            self._write_yaml_inline(lines, f"- {first_key}", first_val, indent)
                        # Remaining keys
                        for k, v in list(item.items())[1:]:
                            extra_prefix = " " * ((indent + 1) * self._config.indent_size)
                            if isinstance(v, (dict, list)) and v:
                                lines.append(f"{extra_prefix}{k}:")
                                self._write_yaml_value(lines, v, indent + 2)
                            else:
                                self._write_yaml_inline(lines, k, v, indent + 1)
                elif isinstance(item, str):
                    lines.append(f"{prefix}- {item}")
                else:
                    lines.append(f"{prefix}- {self._format_value(item)}")
        else:
            lines.append(f"{prefix}{self._format_value(value)}")

    def _write_yaml_inline(
        self,
        lines: list[str],
        key: str,
        value: Any,
        indent: int,
    ) -> None:
        """Write a simple key-value pair."""
        prefix = " " * (indent * self._config.indent_size)
        lines.append(f"{prefix}{key}: {self._format_value(value)}")

    def _format_value(self, value: Any) -> str:
        """Format a simple value for YAML."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            # Quote if contains special characters
            if any(c in value for c in ":#{}[]|>*&!%@\\"):
                return f'"{value}"'
            return value
        if isinstance(value, list):
            return "[" + ", ".join(self._format_value(v) for v in value) + "]"
        return str(value)
