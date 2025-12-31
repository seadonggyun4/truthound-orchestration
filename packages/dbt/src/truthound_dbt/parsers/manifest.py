"""dbt Manifest Parser.

This module provides type-safe parsing of dbt manifest.json files with
support for extracting Truthound tests and generating reports.

Example:
    >>> from truthound_dbt.parsers import ManifestParser
    >>>
    >>> parser = ManifestParser("target/manifest.json")
    >>> tests = parser.get_truthound_tests()
    >>> for test in tests:
    ...     print(f"{test.name}: {test.rule_type}")
    >>>
    >>> report = parser.generate_report()
    >>> print(report.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class ManifestParseError(Exception):
    """Base exception for manifest parsing errors."""

    pass


class ManifestNotFoundError(ManifestParseError):
    """Raised when manifest file is not found."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Manifest not found: {path}")


class InvalidManifestError(ManifestParseError):
    """Raised when manifest JSON is invalid."""

    pass


# =============================================================================
# Enums
# =============================================================================


class CheckCategory(str, Enum):
    """Data quality check categories."""

    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    TIMELINESS = "timeliness"
    CUSTOM = "custom"


# Rule type to category mapping
RULE_CATEGORY_MAP: dict[str, CheckCategory] = {
    "not_null": CheckCategory.COMPLETENESS,
    "unique": CheckCategory.UNIQUENESS,
    "accepted_values": CheckCategory.VALIDITY,
    "relationships": CheckCategory.CONSISTENCY,
    "in_set": CheckCategory.VALIDITY,
    "in_range": CheckCategory.VALIDITY,
    "regex": CheckCategory.VALIDITY,
    "email_format": CheckCategory.VALIDITY,
    "url_format": CheckCategory.VALIDITY,
    "uuid_format": CheckCategory.VALIDITY,
    "phone_format": CheckCategory.VALIDITY,
    "ipv4_format": CheckCategory.VALIDITY,
    "date_format": CheckCategory.VALIDITY,
    "min_length": CheckCategory.VALIDITY,
    "max_length": CheckCategory.VALIDITY,
    "referential_integrity": CheckCategory.CONSISTENCY,
    "row_count": CheckCategory.COMPLETENESS,
    "expression": CheckCategory.CUSTOM,
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ManifestParserConfig:
    """Configuration for manifest parser.

    Attributes:
        truthound_prefix: Prefix for Truthound tests.
        include_disabled: Include disabled tests.
        include_sources: Include source definitions.
        parse_test_metadata: Parse test metadata blocks.
        resolve_refs: Resolve ref() and source() references.
    """

    truthound_prefix: str = "truthound_"
    include_disabled: bool = False
    include_sources: bool = True
    parse_test_metadata: bool = True
    resolve_refs: bool = True

    def with_prefix(self, prefix: str) -> ManifestParserConfig:
        """Return config with custom prefix."""
        return ManifestParserConfig(
            truthound_prefix=prefix,
            include_disabled=self.include_disabled,
            include_sources=self.include_sources,
            parse_test_metadata=self.parse_test_metadata,
            resolve_refs=self.resolve_refs,
        )


DEFAULT_MANIFEST_PARSER_CONFIG = ManifestParserConfig()


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class TruthoundRule:
    """A Truthound data quality rule.

    Attributes:
        rule_type: Type of the rule (e.g., "not_null").
        column: Column name (if column-level).
        parameters: Rule parameters.
        category: Check category.
        severity: Rule severity.
        description: Rule description.
    """

    rule_type: str
    column: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    category: CheckCategory = CheckCategory.CUSTOM
    severity: str = "warn"
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"type": self.rule_type}
        if self.column:
            result["column"] = self.column
        if self.parameters:
            result.update(self.parameters)
        if self.severity != "warn":
            result["severity"] = self.severity
        if self.description:
            result["description"] = self.description
        return result


@dataclass(frozen=True, slots=True)
class TruthoundTest:
    """A Truthound test extracted from manifest.

    Attributes:
        name: Test name.
        unique_id: Unique identifier in manifest.
        model: Associated model name.
        column: Column name (if column-level).
        rule_type: Type of the rule.
        config: Test configuration.
        tags: Test tags.
        severity: Test severity.
        meta: Test metadata.
        depends_on: Dependencies.
    """

    name: str
    unique_id: str
    model: str
    column: str | None = None
    rule_type: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    severity: str = "warn"
    meta: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()

    @property
    def category(self) -> CheckCategory:
        """Get the check category for this test."""
        return RULE_CATEGORY_MAP.get(self.rule_type, CheckCategory.CUSTOM)

    def to_rule(self) -> TruthoundRule:
        """Convert to TruthoundRule."""
        return TruthoundRule(
            rule_type=self.rule_type,
            column=self.column,
            parameters=self.config,
            category=self.category,
            severity=self.severity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "unique_id": self.unique_id,
            "model": self.model,
            "column": self.column,
            "rule_type": self.rule_type,
            "config": self.config,
            "tags": list(self.tags),
            "severity": self.severity,
            "category": self.category.value,
        }


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    """Information about a column in a model.

    Attributes:
        name: Column name.
        description: Column description.
        data_type: Column data type.
        tests: Tests defined on the column.
        meta: Column metadata.
    """

    name: str
    description: str | None = None
    data_type: str | None = None
    tests: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Information about a dbt model.

    Attributes:
        name: Model name.
        unique_id: Unique identifier.
        description: Model description.
        columns: Column information.
        tests: Model-level tests.
        depends_on: Dependencies.
        tags: Model tags.
        meta: Model metadata.
    """

    name: str
    unique_id: str
    description: str | None = None
    columns: tuple[ColumnInfo, ...] = ()
    tests: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TestInfo:
    """Information about a dbt test.

    Attributes:
        name: Test name.
        unique_id: Unique identifier.
        test_type: Type of test (generic, singular).
        model: Associated model.
        column: Associated column.
        severity: Test severity.
        config: Test configuration.
        tags: Test tags.
    """

    name: str
    unique_id: str
    test_type: str
    model: str | None = None
    column: str | None = None
    severity: str = "warn"
    config: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


# =============================================================================
# Report Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class TestCoverage:
    """Test coverage information.

    Attributes:
        total_models: Total number of models.
        tested_models: Number of models with tests.
        total_columns: Total number of columns.
        tested_columns: Number of columns with tests.
        total_tests: Total number of tests.
        truthound_tests: Number of Truthound tests.
    """

    total_models: int = 0
    tested_models: int = 0
    total_columns: int = 0
    tested_columns: int = 0
    total_tests: int = 0
    truthound_tests: int = 0

    @property
    def model_coverage(self) -> float:
        """Model coverage percentage."""
        if self.total_models == 0:
            return 0.0
        return (self.tested_models / self.total_models) * 100

    @property
    def column_coverage(self) -> float:
        """Column coverage percentage."""
        if self.total_columns == 0:
            return 0.0
        return (self.tested_columns / self.total_columns) * 100


@dataclass(frozen=True, slots=True)
class RuleDistribution:
    """Distribution of rule types.

    Attributes:
        by_type: Count by rule type.
        by_category: Count by category.
        by_model: Count by model.
    """

    by_type: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    by_model: dict[str, int] = field(default_factory=dict)


@dataclass
class TruthoundReport:
    """Report on Truthound tests in a manifest.

    Attributes:
        tests: List of Truthound tests.
        coverage: Test coverage information.
        distribution: Rule distribution.
        models: Model information.
        generated_at: Report generation timestamp.
    """

    tests: list[TruthoundTest]
    coverage: TestCoverage
    distribution: RuleDistribution
    models: list[ModelInfo]
    generated_at: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Generate a Markdown report."""
        lines = [
            "# Truthound Test Report",
            "",
            f"Generated: {self.generated_at.isoformat()}",
            "",
            "## Coverage Summary",
            "",
            f"- **Model Coverage**: {self.coverage.model_coverage:.1f}% ({self.coverage.tested_models}/{self.coverage.total_models})",
            f"- **Column Coverage**: {self.coverage.column_coverage:.1f}% ({self.coverage.tested_columns}/{self.coverage.total_columns})",
            f"- **Total Tests**: {self.coverage.total_tests}",
            f"- **Truthound Tests**: {self.coverage.truthound_tests}",
            "",
            "## Rule Distribution by Type",
            "",
        ]

        for rule_type, count in sorted(
            self.distribution.by_type.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- `{rule_type}`: {count}")

        lines.extend([
            "",
            "## Rule Distribution by Category",
            "",
        ])

        for category, count in sorted(
            self.distribution.by_category.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- **{category}**: {count}")

        lines.extend([
            "",
            "## Tests by Model",
            "",
        ])

        for model, count in sorted(
            self.distribution.by_model.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- `{model}`: {count} tests")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tests": [t.to_dict() for t in self.tests],
            "coverage": {
                "total_models": self.coverage.total_models,
                "tested_models": self.coverage.tested_models,
                "model_coverage_pct": self.coverage.model_coverage,
                "total_columns": self.coverage.total_columns,
                "tested_columns": self.coverage.tested_columns,
                "column_coverage_pct": self.coverage.column_coverage,
                "total_tests": self.coverage.total_tests,
                "truthound_tests": self.coverage.truthound_tests,
            },
            "distribution": {
                "by_type": self.distribution.by_type,
                "by_category": self.distribution.by_category,
                "by_model": self.distribution.by_model,
            },
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# ManifestParser
# =============================================================================


class ManifestParser:
    """Parser for dbt manifest.json files.

    This parser extracts Truthound tests, model information, and generates
    coverage and distribution reports.

    Example:
        >>> parser = ManifestParser("target/manifest.json")
        >>> tests = parser.get_truthound_tests()
        >>> report = parser.generate_report()
        >>> print(report.to_markdown())
    """

    def __init__(
        self,
        manifest_path: str | Path,
        config: ManifestParserConfig | None = None,
    ) -> None:
        """Initialize manifest parser.

        Args:
            manifest_path: Path to manifest.json file.
            config: Parser configuration.

        Raises:
            ManifestNotFoundError: If manifest file not found.
            InvalidManifestError: If manifest JSON is invalid.
        """
        self._path = Path(manifest_path)
        self._config = config or DEFAULT_MANIFEST_PARSER_CONFIG
        self._manifest: dict[str, Any] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        """Return the manifest path."""
        return self._path

    @property
    def config(self) -> ManifestParserConfig:
        """Return the parser configuration."""
        return self._config

    def load(self) -> None:
        """Load and parse the manifest file.

        Raises:
            ManifestNotFoundError: If manifest file not found.
            InvalidManifestError: If manifest JSON is invalid.
        """
        if not self._path.exists():
            raise ManifestNotFoundError(self._path)

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidManifestError(f"Invalid JSON: {e}") from e

        self._loaded = True

    def _ensure_loaded(self) -> None:
        """Ensure manifest is loaded."""
        if not self._loaded:
            self.load()

    def get_truthound_tests(self) -> list[TruthoundTest]:
        """Get all Truthound tests from manifest.

        Returns:
            List of TruthoundTest objects.
        """
        self._ensure_loaded()
        tests: list[TruthoundTest] = []

        nodes = self._manifest.get("nodes", {})
        for unique_id, node in nodes.items():
            if node.get("resource_type") != "test":
                continue

            test_name = node.get("name", "")
            if not self._is_truthound_test(test_name, node):
                continue

            if not self._config.include_disabled and node.get("config", {}).get(
                "enabled"
            ) is False:
                continue

            test = self._parse_truthound_test(unique_id, node)
            if test:
                tests.append(test)

        return tests

    def get_all_tests(self) -> list[TestInfo]:
        """Get all tests from manifest.

        Returns:
            List of TestInfo objects.
        """
        self._ensure_loaded()
        tests: list[TestInfo] = []

        nodes = self._manifest.get("nodes", {})
        for unique_id, node in nodes.items():
            if node.get("resource_type") != "test":
                continue

            if not self._config.include_disabled and node.get("config", {}).get(
                "enabled"
            ) is False:
                continue

            test = self._parse_test_info(unique_id, node)
            if test:
                tests.append(test)

        return tests

    def get_models(self) -> list[ModelInfo]:
        """Get all models from manifest.

        Returns:
            List of ModelInfo objects.
        """
        self._ensure_loaded()
        models: list[ModelInfo] = []

        nodes = self._manifest.get("nodes", {})
        for unique_id, node in nodes.items():
            if node.get("resource_type") != "model":
                continue

            model = self._parse_model_info(unique_id, node)
            if model:
                models.append(model)

        return models

    def get_tests_for_model(self, model_name: str) -> list[TestInfo]:
        """Get all tests for a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            List of TestInfo objects.
        """
        all_tests = self.get_all_tests()
        return [t for t in all_tests if t.model == model_name]

    def get_tests_for_column(
        self, model_name: str, column_name: str
    ) -> list[TestInfo]:
        """Get all tests for a specific column.

        Args:
            model_name: Name of the model.
            column_name: Name of the column.

        Returns:
            List of TestInfo objects.
        """
        all_tests = self.get_all_tests()
        return [
            t
            for t in all_tests
            if t.model == model_name and t.column == column_name
        ]

    def generate_report(self) -> TruthoundReport:
        """Generate a comprehensive report.

        Returns:
            TruthoundReport with coverage and distribution data.
        """
        self._ensure_loaded()

        truthound_tests = self.get_truthound_tests()
        all_tests = self.get_all_tests()
        models = self.get_models()

        # Calculate coverage
        tested_models = set()
        tested_columns = set()
        total_columns = 0

        for model in models:
            total_columns += len(model.columns)

        for test in all_tests:
            if test.model:
                tested_models.add(test.model)
            if test.model and test.column:
                tested_columns.add((test.model, test.column))

        coverage = TestCoverage(
            total_models=len(models),
            tested_models=len(tested_models),
            total_columns=total_columns,
            tested_columns=len(tested_columns),
            total_tests=len(all_tests),
            truthound_tests=len(truthound_tests),
        )

        # Calculate distribution
        by_type: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_model: dict[str, int] = {}

        for test in truthound_tests:
            by_type[test.rule_type] = by_type.get(test.rule_type, 0) + 1
            by_category[test.category.value] = (
                by_category.get(test.category.value, 0) + 1
            )
            by_model[test.model] = by_model.get(test.model, 0) + 1

        distribution = RuleDistribution(
            by_type=by_type,
            by_category=by_category,
            by_model=by_model,
        )

        return TruthoundReport(
            tests=truthound_tests,
            coverage=coverage,
            distribution=distribution,
            models=models,
        )

    def _is_truthound_test(self, test_name: str, node: dict[str, Any]) -> bool:
        """Check if a test is a Truthound test."""
        # Check test name prefix
        if test_name.startswith(self._config.truthound_prefix):
            return True

        # Check test metadata for truthound marker
        meta = node.get("meta", {})
        if meta.get("truthound"):
            return True

        # Check config for truthound marker
        config = node.get("config", {})
        if config.get("truthound"):
            return True

        return False

    def _parse_truthound_test(
        self, unique_id: str, node: dict[str, Any]
    ) -> TruthoundTest | None:
        """Parse a Truthound test from manifest node."""
        test_name = node.get("name", "")
        config = node.get("config", {})
        meta = node.get("meta", {})

        # Extract model name from depends_on
        depends_on = node.get("depends_on", {}).get("nodes", [])
        model = ""
        for dep in depends_on:
            if dep.startswith("model."):
                model = dep.split(".")[-1]
                break

        # Extract column name
        column = node.get("column_name")

        # Extract rule type from test name
        rule_type = test_name
        if test_name.startswith(self._config.truthound_prefix):
            rule_type = test_name[len(self._config.truthound_prefix) :]

        # Built-in tests
        if test_name in ("not_null", "unique", "accepted_values", "relationships"):
            rule_type = test_name

        return TruthoundTest(
            name=test_name,
            unique_id=unique_id,
            model=model,
            column=column,
            rule_type=rule_type,
            config=config,
            tags=tuple(node.get("tags", [])),
            severity=config.get("severity", "warn"),
            meta=meta,
            depends_on=tuple(depends_on),
        )

    def _parse_test_info(
        self, unique_id: str, node: dict[str, Any]
    ) -> TestInfo | None:
        """Parse test info from manifest node."""
        test_name = node.get("name", "")
        config = node.get("config", {})

        # Extract model name
        depends_on = node.get("depends_on", {}).get("nodes", [])
        model = None
        for dep in depends_on:
            if dep.startswith("model."):
                model = dep.split(".")[-1]
                break

        return TestInfo(
            name=test_name,
            unique_id=unique_id,
            test_type=node.get("test_metadata", {}).get("name", "generic"),
            model=model,
            column=node.get("column_name"),
            severity=config.get("severity", "warn"),
            config=config,
            tags=tuple(node.get("tags", [])),
        )

    def _parse_model_info(
        self, unique_id: str, node: dict[str, Any]
    ) -> ModelInfo | None:
        """Parse model info from manifest node."""
        columns: list[ColumnInfo] = []
        for col_name, col_data in node.get("columns", {}).items():
            columns.append(
                ColumnInfo(
                    name=col_name,
                    description=col_data.get("description"),
                    data_type=col_data.get("data_type"),
                    meta=col_data.get("meta", {}),
                )
            )

        return ModelInfo(
            name=node.get("name", ""),
            unique_id=unique_id,
            description=node.get("description"),
            columns=tuple(columns),
            depends_on=tuple(node.get("depends_on", {}).get("nodes", [])),
            tags=tuple(node.get("tags", [])),
            meta=node.get("meta", {}),
        )
