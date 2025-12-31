"""dbt Run Results Parser.

This module provides parsing of dbt run_results.json files with support
for converting results to common CheckResult format.

Example:
    >>> from truthound_dbt.parsers import RunResultsParser
    >>>
    >>> parser = RunResultsParser("target/run_results.json")
    >>> summary = parser.get_summary()
    >>> check_results = parser.to_check_results()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


# =============================================================================
# Exceptions
# =============================================================================


class RunResultsParseError(Exception):
    """Base exception for run results parsing errors."""

    pass


class RunResultsNotFoundError(RunResultsParseError):
    """Raised when run_results.json is not found."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Run results not found: {path}")


# =============================================================================
# Enums
# =============================================================================


class TestStatus(str, Enum):
    """dbt test execution status."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"
    SKIPPED = "skipped"


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class RunResultsParserConfig:
    """Configuration for run results parser.

    Attributes:
        include_timing: Include timing information.
        include_compiled_sql: Include compiled SQL.
        parse_failures: Parse failure messages.
        map_to_common_types: Map to common result types.
    """

    include_timing: bool = True
    include_compiled_sql: bool = True
    parse_failures: bool = True
    map_to_common_types: bool = True

    def with_timing(self, include: bool = True) -> RunResultsParserConfig:
        """Return config with timing setting."""
        return RunResultsParserConfig(
            include_timing=include,
            include_compiled_sql=self.include_compiled_sql,
            parse_failures=self.parse_failures,
            map_to_common_types=self.map_to_common_types,
        )


DEFAULT_RUN_RESULTS_PARSER_CONFIG = RunResultsParserConfig()


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class TimingInfo:
    """Timing information for a test execution.

    Attributes:
        name: Timing stage name (e.g., "compile", "execute").
        started_at: Start timestamp.
        completed_at: Completion timestamp.
        duration_seconds: Duration in seconds.
    """

    name: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimingInfo:
        """Create from dictionary."""
        started = None
        completed = None
        duration = 0.0

        if data.get("started_at"):
            try:
                started = datetime.fromisoformat(
                    data["started_at"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        if data.get("completed_at"):
            try:
                completed = datetime.fromisoformat(
                    data["completed_at"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        if started and completed:
            duration = (completed - started).total_seconds()

        return cls(
            name=data.get("name", "unknown"),
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
        )


@dataclass(frozen=True, slots=True)
class TestResult:
    """Result of a dbt test execution.

    Attributes:
        unique_id: Unique identifier of the test.
        status: Test status (pass, fail, warn, error, skipped).
        execution_time: Total execution time in seconds.
        message: Result message.
        failures: Number of failures (for test assertions).
        timing: Timing information.
        compiled_sql: Compiled SQL (if available).
        adapter_response: Adapter-specific response.
    """

    unique_id: str
    status: TestStatus
    execution_time: float = 0.0
    message: str | None = None
    failures: int = 0
    timing: tuple[TimingInfo, ...] = ()
    compiled_sql: str | None = None
    adapter_response: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASS

    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status in (TestStatus.FAIL, TestStatus.ERROR)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "unique_id": self.unique_id,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "message": self.message,
            "failures": self.failures,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Summary of a dbt run.

    Attributes:
        elapsed_time: Total elapsed time.
        generated_at: Timestamp of results generation.
        total_tests: Total number of tests.
        passed_tests: Number of passed tests.
        failed_tests: Number of failed tests.
        warned_tests: Number of tests with warnings.
        errored_tests: Number of errored tests.
        skipped_tests: Number of skipped tests.
    """

    elapsed_time: float = 0.0
    generated_at: datetime | None = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    warned_tests: int = 0
    errored_tests: int = 0
    skipped_tests: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed_tests == 0 and self.errored_tests == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "elapsed_time": self.elapsed_time,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "warned_tests": self.warned_tests,
            "errored_tests": self.errored_tests,
            "skipped_tests": self.skipped_tests,
            "success_rate": self.success_rate,
            "all_passed": self.all_passed,
        }


# =============================================================================
# RunResultsParser
# =============================================================================


class RunResultsParser:
    """Parser for dbt run_results.json files.

    This parser extracts test results and provides methods for converting
    to common result types.

    Example:
        >>> parser = RunResultsParser("target/run_results.json")
        >>> summary = parser.get_summary()
        >>> print(f"Success rate: {summary.success_rate:.1f}%")
        >>>
        >>> for result in parser.get_test_results():
        ...     print(f"{result.unique_id}: {result.status.value}")
    """

    def __init__(
        self,
        results_path: str | Path,
        config: RunResultsParserConfig | None = None,
    ) -> None:
        """Initialize run results parser.

        Args:
            results_path: Path to run_results.json file.
            config: Parser configuration.

        Raises:
            RunResultsNotFoundError: If results file not found.
        """
        self._path = Path(results_path)
        self._config = config or DEFAULT_RUN_RESULTS_PARSER_CONFIG
        self._results: dict[str, Any] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        """Return the results file path."""
        return self._path

    @property
    def config(self) -> RunResultsParserConfig:
        """Return the parser configuration."""
        return self._config

    def load(self) -> None:
        """Load and parse the results file.

        Raises:
            RunResultsNotFoundError: If results file not found.
            RunResultsParseError: If JSON is invalid.
        """
        if not self._path.exists():
            raise RunResultsNotFoundError(self._path)

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._results = json.load(f)
        except json.JSONDecodeError as e:
            raise RunResultsParseError(f"Invalid JSON: {e}") from e

        self._loaded = True

    def _ensure_loaded(self) -> None:
        """Ensure results are loaded."""
        if not self._loaded:
            self.load()

    def get_test_results(self) -> list[TestResult]:
        """Get all test results.

        Returns:
            List of TestResult objects.
        """
        self._ensure_loaded()
        results: list[TestResult] = []

        for result in self._results.get("results", []):
            # Only include test resources
            unique_id = result.get("unique_id", "")
            if not unique_id.startswith("test."):
                continue

            test_result = self._parse_test_result(result)
            if test_result:
                results.append(test_result)

        return results

    def get_summary(self) -> RunSummary:
        """Get run summary.

        Returns:
            RunSummary with aggregated statistics.
        """
        self._ensure_loaded()
        test_results = self.get_test_results()

        passed = sum(1 for r in test_results if r.status == TestStatus.PASS)
        failed = sum(1 for r in test_results if r.status == TestStatus.FAIL)
        warned = sum(1 for r in test_results if r.status == TestStatus.WARN)
        errored = sum(1 for r in test_results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)

        generated_at = None
        if self._results.get("metadata", {}).get("generated_at"):
            try:
                ts = self._results["metadata"]["generated_at"]
                generated_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return RunSummary(
            elapsed_time=self._results.get("elapsed_time", 0.0),
            generated_at=generated_at,
            total_tests=len(test_results),
            passed_tests=passed,
            failed_tests=failed,
            warned_tests=warned,
            errored_tests=errored,
            skipped_tests=skipped,
        )

    def get_failed_tests(self) -> list[TestResult]:
        """Get failed test results.

        Returns:
            List of failed TestResult objects.
        """
        return [r for r in self.get_test_results() if r.failed]

    def get_passed_tests(self) -> list[TestResult]:
        """Get passed test results.

        Returns:
            List of passed TestResult objects.
        """
        return [r for r in self.get_test_results() if r.passed]

    def to_check_results(self) -> list[dict[str, Any]]:
        """Convert to common CheckResult format.

        Returns a list of dictionaries that can be used to create
        common.base.CheckResult objects.

        Returns:
            List of dictionaries in CheckResult format.
        """
        if not self._config.map_to_common_types:
            return []

        self._ensure_loaded()
        check_results: list[dict[str, Any]] = []

        for test_result in self.get_test_results():
            # Map dbt status to common CheckStatus
            status_map = {
                TestStatus.PASS: "PASSED",
                TestStatus.FAIL: "FAILED",
                TestStatus.WARN: "WARNING",
                TestStatus.ERROR: "ERROR",
                TestStatus.SKIPPED: "SKIPPED",
            }

            check_result = {
                "status": status_map.get(test_result.status, "ERROR"),
                "passed_count": 1 if test_result.passed else 0,
                "failed_count": test_result.failures if test_result.failures else (
                    0 if test_result.passed else 1
                ),
                "execution_time_ms": test_result.execution_time * 1000,
                "metadata": {
                    "unique_id": test_result.unique_id,
                    "message": test_result.message,
                    "source": "dbt",
                },
            }

            if not test_result.passed and test_result.message:
                check_result["failures"] = [
                    {
                        "rule_type": "dbt_test",
                        "column": None,
                        "message": test_result.message,
                        "row_index": None,
                    }
                ]

            check_results.append(check_result)

        return check_results

    def _parse_test_result(self, result: dict[str, Any]) -> TestResult | None:
        """Parse a test result from run_results entry."""
        unique_id = result.get("unique_id", "")
        status_str = result.get("status", "error").lower()

        try:
            status = TestStatus(status_str)
        except ValueError:
            status = TestStatus.ERROR

        # Parse timing
        timing: list[TimingInfo] = []
        if self._config.include_timing:
            for t in result.get("timing", []):
                timing.append(TimingInfo.from_dict(t))

        # Get compiled SQL
        compiled_sql = None
        if self._config.include_compiled_sql:
            compiled_sql = result.get("compiled_code")

        return TestResult(
            unique_id=unique_id,
            status=status,
            execution_time=result.get("execution_time", 0.0),
            message=result.get("message"),
            failures=result.get("failures", 0) or 0,
            timing=tuple(timing),
            compiled_sql=compiled_sql,
            adapter_response=result.get("adapter_response", {}),
        )
