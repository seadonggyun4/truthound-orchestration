"""Truthound dbt Manifest Parser.

This module provides utilities for parsing dbt manifest.json files to extract
Truthound test information, generate reports, and integrate with CI pipelines.

Features:
    - Parse manifest.json to extract Truthound tests
    - Generate coverage reports
    - Validate test configurations
    - Export to various formats (JSON, CSV, Markdown)

Usage:
    python manifest_parser.py parse target/manifest.json
    python manifest_parser.py report target/manifest.json --format markdown
    python manifest_parser.py coverage target/manifest.json --threshold 80

Example:
    >>> from manifest_parser import ManifestParser
    >>> parser = ManifestParser("target/manifest.json")
    >>> tests = parser.get_truthound_tests()
    >>> report = parser.generate_report()
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CheckCategory(Enum):
    """Categories of data quality checks."""

    COMPLETENESS = "completeness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    REFERENTIAL = "referential"
    CUSTOM = "custom"


# Mapping of check types to categories
CHECK_CATEGORY_MAP: dict[str, CheckCategory] = {
    # Completeness
    "not_null": CheckCategory.COMPLETENESS,
    "not_empty": CheckCategory.COMPLETENESS,
    # Uniqueness
    "unique": CheckCategory.UNIQUENESS,
    "unique_combination": CheckCategory.UNIQUENESS,
    # Validity - Set membership
    "in_set": CheckCategory.VALIDITY,
    "accepted_values": CheckCategory.VALIDITY,
    "not_in_set": CheckCategory.VALIDITY,
    "rejected_values": CheckCategory.VALIDITY,
    # Validity - Numeric
    "range": CheckCategory.VALIDITY,
    "between": CheckCategory.VALIDITY,
    "positive": CheckCategory.VALIDITY,
    "negative": CheckCategory.VALIDITY,
    "non_negative": CheckCategory.VALIDITY,
    "non_positive": CheckCategory.VALIDITY,
    "greater_than": CheckCategory.VALIDITY,
    "gt": CheckCategory.VALIDITY,
    "less_than": CheckCategory.VALIDITY,
    "lt": CheckCategory.VALIDITY,
    "greater_than_or_equal": CheckCategory.VALIDITY,
    "gte": CheckCategory.VALIDITY,
    "less_than_or_equal": CheckCategory.VALIDITY,
    "lte": CheckCategory.VALIDITY,
    "equal": CheckCategory.VALIDITY,
    "eq": CheckCategory.VALIDITY,
    "not_equal": CheckCategory.VALIDITY,
    "neq": CheckCategory.VALIDITY,
    # Validity - String
    "length": CheckCategory.VALIDITY,
    "regex": CheckCategory.VALIDITY,
    "pattern": CheckCategory.VALIDITY,
    "email_format": CheckCategory.VALIDITY,
    "email": CheckCategory.VALIDITY,
    "url_format": CheckCategory.VALIDITY,
    "url": CheckCategory.VALIDITY,
    "uuid_format": CheckCategory.VALIDITY,
    "uuid": CheckCategory.VALIDITY,
    "phone_format": CheckCategory.VALIDITY,
    "phone": CheckCategory.VALIDITY,
    "ipv4_format": CheckCategory.VALIDITY,
    "ipv4": CheckCategory.VALIDITY,
    # Timeliness
    "not_future": CheckCategory.TIMELINESS,
    "not_past": CheckCategory.TIMELINESS,
    "date_format": CheckCategory.TIMELINESS,
    # Referential
    "referential_integrity": CheckCategory.REFERENTIAL,
    "foreign_key": CheckCategory.REFERENTIAL,
    # Custom
    "expression": CheckCategory.CUSTOM,
    "custom": CheckCategory.CUSTOM,
    "row_count_range": CheckCategory.CUSTOM,
}


@dataclass(frozen=True)
class TruthoundRule:
    """A single Truthound rule definition."""

    check_type: str
    column: str | None
    category: CheckCategory
    parameters: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TruthoundRule:
        """Create a rule from a dictionary."""
        check_type = data.get("check", data.get("type", "unknown"))
        column = data.get("column")
        category = CHECK_CATEGORY_MAP.get(check_type, CheckCategory.CUSTOM)

        # Extract parameters (everything except check, column, message)
        params = {
            k: v
            for k, v in data.items()
            if k not in ("check", "type", "column", "message")
        }

        return cls(
            check_type=check_type,
            column=column,
            category=category,
            parameters=params,
            message=data.get("message"),
        )


@dataclass
class TruthoundTest:
    """Information about a Truthound test."""

    test_name: str
    model_name: str
    rules: list[TruthoundRule]
    severity: str
    tags: list[str]
    config: dict[str, Any]
    file_path: str | None = None

    @property
    def rule_count(self) -> int:
        """Get the number of rules in this test."""
        return len(self.rules)

    @property
    def columns_covered(self) -> set[str]:
        """Get the set of columns covered by rules."""
        return {r.column for r in self.rules if r.column}

    @property
    def check_types(self) -> set[str]:
        """Get the set of check types used."""
        return {r.check_type for r in self.rules}

    @property
    def categories(self) -> set[CheckCategory]:
        """Get the set of categories covered."""
        return {r.category for r in self.rules}


@dataclass
class TruthoundReport:
    """Report generated from manifest analysis."""

    total_tests: int
    total_rules: int
    models_covered: int
    columns_covered: int
    rules_by_category: dict[str, int]
    rules_by_check_type: dict[str, int]
    tests_by_model: dict[str, int]
    tests_by_severity: dict[str, int]
    tests: list[TruthoundTest]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "summary": {
                "total_tests": self.total_tests,
                "total_rules": self.total_rules,
                "models_covered": self.models_covered,
                "columns_covered": self.columns_covered,
            },
            "rules_by_category": self.rules_by_category,
            "rules_by_check_type": self.rules_by_check_type,
            "tests_by_model": self.tests_by_model,
            "tests_by_severity": self.tests_by_severity,
            "tests": [
                {
                    "test_name": t.test_name,
                    "model_name": t.model_name,
                    "rule_count": t.rule_count,
                    "severity": t.severity,
                    "tags": t.tags,
                    "columns": list(t.columns_covered),
                    "check_types": list(t.check_types),
                }
                for t in self.tests
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Convert tests to CSV string."""
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "model_name",
                "test_name",
                "rule_count",
                "severity",
                "columns",
                "check_types",
            ],
        )
        writer.writeheader()
        for test in self.tests:
            writer.writerow(
                {
                    "model_name": test.model_name,
                    "test_name": test.test_name,
                    "rule_count": test.rule_count,
                    "severity": test.severity,
                    "columns": ",".join(test.columns_covered),
                    "check_types": ",".join(test.check_types),
                }
            )
        return output.getvalue()

    def to_markdown(self) -> str:
        """Convert report to Markdown string."""
        lines = [
            "# Truthound Test Report",
            "",
            "## Summary",
            "",
            f"- **Total Tests:** {self.total_tests}",
            f"- **Total Rules:** {self.total_rules}",
            f"- **Models Covered:** {self.models_covered}",
            f"- **Columns Covered:** {self.columns_covered}",
            "",
            "## Rules by Category",
            "",
            "| Category | Count |",
            "|----------|-------|",
        ]
        for category, count in sorted(self.rules_by_category.items()):
            lines.append(f"| {category} | {count} |")

        lines.extend(
            [
                "",
                "## Rules by Check Type",
                "",
                "| Check Type | Count |",
                "|------------|-------|",
            ]
        )
        for check_type, count in sorted(
            self.rules_by_check_type.items(), key=lambda x: -x[1]
        ):
            lines.append(f"| {check_type} | {count} |")

        lines.extend(
            [
                "",
                "## Tests by Model",
                "",
                "| Model | Tests | Rules |",
                "|-------|-------|-------|",
            ]
        )
        model_rules: dict[str, int] = {}
        for test in self.tests:
            model_rules[test.model_name] = (
                model_rules.get(test.model_name, 0) + test.rule_count
            )
        for model, test_count in sorted(self.tests_by_model.items()):
            lines.append(f"| {model} | {test_count} | {model_rules.get(model, 0)} |")

        lines.extend(
            [
                "",
                "## Test Details",
                "",
            ]
        )
        for test in self.tests:
            lines.append(f"### {test.model_name}")
            lines.append("")
            lines.append(f"- **Test:** {test.test_name}")
            lines.append(f"- **Severity:** {test.severity}")
            lines.append(f"- **Rules:** {test.rule_count}")
            if test.tags:
                lines.append(f"- **Tags:** {', '.join(test.tags)}")
            lines.append("")
            lines.append("| Column | Check | Parameters |")
            lines.append("|--------|-------|------------|")
            for rule in test.rules:
                params = (
                    ", ".join(f"{k}={v}" for k, v in rule.parameters.items())
                    if rule.parameters
                    else "-"
                )
                lines.append(f"| {rule.column or '_model_'} | {rule.check_type} | {params} |")
            lines.append("")

        return "\n".join(lines)


class ManifestParser:
    """Parser for dbt manifest.json files."""

    def __init__(self, manifest_path: str | Path) -> None:
        """Initialize the parser.

        Args:
            manifest_path: Path to the manifest.json file
        """
        self.manifest_path = Path(manifest_path)
        self._manifest: dict[str, Any] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        """Load and return the manifest data."""
        if self._manifest is None:
            with open(self.manifest_path) as f:
                self._manifest = json.load(f)
        return self._manifest

    def get_truthound_tests(self) -> list[TruthoundTest]:
        """Extract all Truthound tests from the manifest.

        Returns:
            List of TruthoundTest objects
        """
        tests: list[TruthoundTest] = []

        for node_id, node in self.manifest.get("nodes", {}).items():
            if node.get("resource_type") != "test":
                continue

            test_metadata = node.get("test_metadata", {})
            test_name = test_metadata.get("name", "")

            # Check for Truthound tests
            if not test_name.startswith("truthound"):
                continue

            kwargs = test_metadata.get("kwargs", {})
            config = node.get("config", {})

            # Extract model name from refs
            refs = node.get("refs", [])
            model_name = refs[0].get("name", "unknown") if refs else "unknown"

            # Parse rules
            rules_data = kwargs.get("rules", [])

            # For single-column tests (truthound_not_null, etc.)
            if not rules_data and "column_name" in kwargs:
                check_type = test_name.replace("truthound_", "")
                rules_data = [{"check": check_type, "column": kwargs["column_name"]}]
                # Add additional params
                for key in ["min", "max", "values", "pattern", "to", "field"]:
                    if key in kwargs:
                        if key == "to":
                            rules_data[0]["to_model"] = kwargs[key]
                        elif key == "field":
                            rules_data[0]["to_column"] = kwargs[key]
                        else:
                            rules_data[0][key] = kwargs[key]

            rules = [TruthoundRule.from_dict(r) for r in rules_data]

            tests.append(
                TruthoundTest(
                    test_name=node.get("name", ""),
                    model_name=model_name,
                    rules=rules,
                    severity=config.get("severity", "error"),
                    tags=config.get("tags", []),
                    config=config,
                    file_path=node.get("original_file_path"),
                )
            )

        return tests

    def generate_report(self) -> TruthoundReport:
        """Generate a comprehensive report from the manifest.

        Returns:
            TruthoundReport object
        """
        tests = self.get_truthound_tests()

        # Aggregate statistics
        all_columns: set[str] = set()
        rules_by_category: dict[str, int] = {}
        rules_by_check_type: dict[str, int] = {}
        tests_by_model: dict[str, int] = {}
        tests_by_severity: dict[str, int] = {}

        for test in tests:
            # Columns
            all_columns.update(test.columns_covered)

            # By model
            tests_by_model[test.model_name] = tests_by_model.get(test.model_name, 0) + 1

            # By severity
            tests_by_severity[test.severity] = (
                tests_by_severity.get(test.severity, 0) + 1
            )

            # By rule
            for rule in test.rules:
                # Category
                cat = rule.category.value
                rules_by_category[cat] = rules_by_category.get(cat, 0) + 1

                # Check type
                rules_by_check_type[rule.check_type] = (
                    rules_by_check_type.get(rule.check_type, 0) + 1
                )

        total_rules = sum(t.rule_count for t in tests)

        return TruthoundReport(
            total_tests=len(tests),
            total_rules=total_rules,
            models_covered=len(tests_by_model),
            columns_covered=len(all_columns),
            rules_by_category=rules_by_category,
            rules_by_check_type=rules_by_check_type,
            tests_by_model=tests_by_model,
            tests_by_severity=tests_by_severity,
            tests=tests,
        )

    def check_coverage(
        self,
        threshold: float = 0.8,
        required_categories: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Check if coverage meets requirements.

        Args:
            threshold: Minimum coverage ratio (0.0 to 1.0)
            required_categories: Categories that must be present

        Returns:
            Tuple of (passed, message)
        """
        report = self.generate_report()

        # Get all models from manifest
        all_models = {
            node.get("name")
            for node in self.manifest.get("nodes", {}).values()
            if node.get("resource_type") == "model"
        }

        if not all_models:
            return False, "No models found in manifest"

        # Calculate coverage
        covered_models = set(report.tests_by_model.keys())
        coverage_ratio = len(covered_models) / len(all_models)

        messages = []

        if coverage_ratio < threshold:
            messages.append(
                f"Coverage {coverage_ratio:.1%} below threshold {threshold:.1%}"
            )
            uncovered = all_models - covered_models
            if uncovered:
                messages.append(f"Uncovered models: {', '.join(sorted(uncovered))}")

        # Check required categories
        if required_categories:
            present_categories = set(report.rules_by_category.keys())
            missing = set(required_categories) - present_categories
            if missing:
                messages.append(f"Missing required categories: {', '.join(missing)}")

        if messages:
            return False, "\n".join(messages)

        return True, f"Coverage {coverage_ratio:.1%} meets threshold {threshold:.1%}"


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse dbt manifest.json for Truthound test information"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse manifest and show tests")
    parse_parser.add_argument("manifest", help="Path to manifest.json")
    parse_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("manifest", help="Path to manifest.json")
    report_parser.add_argument(
        "--format",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format",
    )
    report_parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )

    # Coverage command
    coverage_parser = subparsers.add_parser("coverage", help="Check test coverage")
    coverage_parser.add_argument("manifest", help="Path to manifest.json")
    coverage_parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum coverage ratio (default: 0.8)",
    )
    coverage_parser.add_argument(
        "--require-category",
        action="append",
        dest="required_categories",
        help="Required categories (can specify multiple)",
    )

    args = parser.parse_args()

    try:
        manifest_parser = ManifestParser(args.manifest)

        if args.command == "parse":
            tests = manifest_parser.get_truthound_tests()
            if args.format == "json":
                output = json.dumps(
                    [
                        {
                            "test_name": t.test_name,
                            "model_name": t.model_name,
                            "rule_count": t.rule_count,
                            "severity": t.severity,
                            "rules": [
                                {
                                    "check_type": r.check_type,
                                    "column": r.column,
                                    "category": r.category.value,
                                }
                                for r in t.rules
                            ],
                        }
                        for t in tests
                    ],
                    indent=2,
                )
            else:
                lines = [f"Found {len(tests)} Truthound tests:", ""]
                for test in tests:
                    lines.append(f"  {test.model_name}:")
                    lines.append(f"    Test: {test.test_name}")
                    lines.append(f"    Rules: {test.rule_count}")
                    lines.append(f"    Severity: {test.severity}")
                    lines.append("")
                output = "\n".join(lines)
            print(output)

        elif args.command == "report":
            report = manifest_parser.generate_report()
            if args.format == "json":
                output = report.to_json()
            elif args.format == "csv":
                output = report.to_csv()
            else:
                output = report.to_markdown()

            if args.output:
                Path(args.output).write_text(output)
                print(f"Report written to {args.output}")
            else:
                print(output)

        elif args.command == "coverage":
            passed, message = manifest_parser.check_coverage(
                threshold=args.threshold,
                required_categories=args.required_categories,
            )
            print(message)
            return 0 if passed else 1

    except FileNotFoundError:
        print(f"Error: Manifest file not found: {args.manifest}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in manifest: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
