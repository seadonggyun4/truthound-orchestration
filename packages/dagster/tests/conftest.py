"""Pytest configuration and fixtures for truthound-dagster tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest


# Mock CheckStatus enum
class MockCheckStatus(Enum):
    """Mock check status for testing."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


# Mock Severity enum
class MockSeverity(Enum):
    """Mock severity for testing."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class MockValidationFailure:
    """Mock validation failure for testing."""

    rule_name: str = "test_rule"
    column: str = "test_column"
    message: str = "Test failure message"
    severity: MockSeverity = MockSeverity.HIGH
    failed_count: int = 10
    total_count: int = 100

    @property
    def failure_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count


@dataclass
class MockCheckResult:
    """Mock check result for testing."""

    status: MockCheckStatus = MockCheckStatus.PASSED
    passed_count: int = 10
    failed_count: int = 0
    warning_count: int = 0
    skipped_count: int = 0
    failures: list[MockValidationFailure] = field(default_factory=list)
    execution_time_ms: float = 100.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == MockCheckStatus.PASSED

    @property
    def failure_rate(self) -> float:
        total = self.passed_count + self.failed_count
        if total == 0:
            return 0.0
        return self.failed_count / total


@dataclass
class MockColumnProfile:
    """Mock column profile for testing."""

    column_name: str
    dtype: str = "string"
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 100
    unique_percentage: float = 100.0
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass
class MockProfileResult:
    """Mock profile result for testing."""

    row_count: int = 1000
    column_count: int = 5
    columns: list[MockColumnProfile] = field(default_factory=list)
    execution_time_ms: float = 50.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockLearnedRule:
    """Mock learned rule for testing."""

    column: str
    rule_type: str
    confidence: float = 0.95
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockLearnResult:
    """Mock learn result for testing."""

    rules: list[MockLearnedRule] = field(default_factory=list)
    execution_time_ms: float = 200.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class MockDataQualityEngine:
    """Mock data quality engine for testing."""

    def __init__(
        self,
        should_fail: bool = False,
        check_result: MockCheckResult | None = None,
        profile_result: MockProfileResult | None = None,
        learn_result: MockLearnResult | None = None,
    ) -> None:
        self.should_fail = should_fail
        self._check_result = check_result
        self._profile_result = profile_result
        self._learn_result = learn_result
        self.call_history: list[dict[str, Any]] = []

    @property
    def engine_name(self) -> str:
        return "mock_engine"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def check(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> MockCheckResult:
        self.call_history.append({
            "method": "check",
            "data": data,
            "rules": rules,
            "kwargs": kwargs,
        })

        if self.should_fail:
            return MockCheckResult(
                status=MockCheckStatus.FAILED,
                passed_count=5,
                failed_count=5,
                failures=[MockValidationFailure()],
            )

        if self._check_result:
            return self._check_result

        return MockCheckResult()

    def profile(self, data: Any, **kwargs: Any) -> MockProfileResult:
        self.call_history.append({
            "method": "profile",
            "data": data,
            "kwargs": kwargs,
        })

        if self._profile_result:
            return self._profile_result

        return MockProfileResult(
            columns=[
                MockColumnProfile(column_name="id"),
                MockColumnProfile(column_name="name"),
            ]
        )

    def learn(self, data: Any, **kwargs: Any) -> MockLearnResult:
        self.call_history.append({
            "method": "learn",
            "data": data,
            "kwargs": kwargs,
        })

        if self._learn_result:
            return self._learn_result

        return MockLearnResult(
            rules=[
                MockLearnedRule(column="id", rule_type="not_null"),
                MockLearnedRule(column="name", rule_type="unique"),
            ]
        )


@pytest.fixture
def mock_engine() -> MockDataQualityEngine:
    """Create a mock engine for testing."""
    return MockDataQualityEngine()


@pytest.fixture
def failing_engine() -> MockDataQualityEngine:
    """Create a failing mock engine for testing."""
    return MockDataQualityEngine(should_fail=True)


@pytest.fixture
def mock_check_result() -> MockCheckResult:
    """Create a mock check result."""
    return MockCheckResult()


@pytest.fixture
def failed_check_result() -> MockCheckResult:
    """Create a failed check result."""
    return MockCheckResult(
        status=MockCheckStatus.FAILED,
        passed_count=8,
        failed_count=2,
        failures=[
            MockValidationFailure(
                rule_name="not_null",
                column="id",
                message="Found 10 null values",
            ),
            MockValidationFailure(
                rule_name="unique",
                column="email",
                message="Found 5 duplicate values",
            ),
        ],
    )


@pytest.fixture
def mock_profile_result() -> MockProfileResult:
    """Create a mock profile result."""
    return MockProfileResult(
        row_count=1000,
        column_count=3,
        columns=[
            MockColumnProfile(column_name="id", dtype="int64"),
            MockColumnProfile(column_name="name", dtype="string", null_count=10, null_percentage=1.0),
            MockColumnProfile(column_name="age", dtype="int64", min_value=0, max_value=100),
        ],
    )


@pytest.fixture
def mock_learn_result() -> MockLearnResult:
    """Create a mock learn result."""
    return MockLearnResult(
        rules=[
            MockLearnedRule(column="id", rule_type="not_null", confidence=1.0),
            MockLearnedRule(column="id", rule_type="unique", confidence=1.0),
            MockLearnedRule(column="age", rule_type="in_range", confidence=0.95, parameters={"min": 0, "max": 150}),
        ],
    )


@pytest.fixture
def sample_rules() -> list[dict[str, Any]]:
    """Create sample validation rules."""
    return [
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
        {"type": "in_range", "column": "age", "min": 0, "max": 150},
    ]


@pytest.fixture
def sample_data() -> dict[str, list[Any]]:
    """Create sample data for testing."""
    return {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "email": ["a@test.com", "b@test.com", "c@test.com", "d@test.com", "e@test.com"],
        "age": [25, 30, 35, 40, 45],
    }


@pytest.fixture
def mock_dagster_context() -> MagicMock:
    """Create a mock Dagster context."""
    context = MagicMock()
    context.log = MagicMock()
    context.resources = MagicMock()
    return context
