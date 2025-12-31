"""dbt Manifest and Results Parsers.

This module provides parsers for dbt manifest.json and run_results.json
files with type-safe data structures and reporting capabilities.

Components:
    - ManifestParser: Parse dbt manifest.json files
    - RunResultsParser: Parse dbt run_results.json files
    - TruthoundTest: Type-safe representation of Truthound tests
    - TruthoundReport: Reporting and analysis utilities

Example:
    >>> from truthound_dbt.parsers import ManifestParser, RunResultsParser
    >>>
    >>> # Parse manifest
    >>> parser = ManifestParser("target/manifest.json")
    >>> tests = parser.get_truthound_tests()
    >>> report = parser.generate_report()
    >>> print(report.to_markdown())
    >>>
    >>> # Parse run results
    >>> results = RunResultsParser("target/run_results.json")
    >>> check_results = results.to_check_results()
"""

from truthound_dbt.parsers.manifest import (
    # Main parser
    ManifestParser,
    # Configuration
    ManifestParserConfig,
    DEFAULT_MANIFEST_PARSER_CONFIG,
    # Data types
    TruthoundTest,
    TruthoundRule,
    ModelInfo,
    ColumnInfo,
    TestInfo,
    # Report types
    TruthoundReport,
    TestCoverage,
    RuleDistribution,
    # Enums
    CheckCategory,
    # Exceptions
    ManifestParseError,
    ManifestNotFoundError,
    InvalidManifestError,
)

from truthound_dbt.parsers.results import (
    # Main parser
    RunResultsParser,
    # Configuration
    RunResultsParserConfig,
    DEFAULT_RUN_RESULTS_PARSER_CONFIG,
    # Data types
    TestResult,
    RunSummary,
    TimingInfo,
    # Exceptions
    RunResultsParseError,
    RunResultsNotFoundError,
)

__all__ = [
    # Manifest Parser
    "ManifestParser",
    "ManifestParserConfig",
    "DEFAULT_MANIFEST_PARSER_CONFIG",
    # Manifest Data Types
    "TruthoundTest",
    "TruthoundRule",
    "ModelInfo",
    "ColumnInfo",
    "TestInfo",
    # Report Types
    "TruthoundReport",
    "TestCoverage",
    "RuleDistribution",
    # Enums
    "CheckCategory",
    # Manifest Exceptions
    "ManifestParseError",
    "ManifestNotFoundError",
    "InvalidManifestError",
    # Run Results Parser
    "RunResultsParser",
    "RunResultsParserConfig",
    "DEFAULT_RUN_RESULTS_PARSER_CONFIG",
    # Run Results Data Types
    "TestResult",
    "RunSummary",
    "TimingInfo",
    # Run Results Exceptions
    "RunResultsParseError",
    "RunResultsNotFoundError",
]
