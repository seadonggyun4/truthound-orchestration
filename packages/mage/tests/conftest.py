"""Pytest fixtures for Mage package tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from common.base import CheckResult, CheckStatus, ProfileResult, LearnResult
from common.testing import MockDataQualityEngine


@pytest.fixture
def mock_engine() -> MockDataQualityEngine:
    """Create a mock data quality engine."""
    return MockDataQualityEngine(should_fail=False)


@pytest.fixture
def failing_mock_engine() -> MockDataQualityEngine:
    """Create a mock engine that produces failed results."""
    return MockDataQualityEngine(should_fail=True)


@pytest.fixture
def sample_check_result() -> CheckResult:
    """Create a sample successful CheckResult."""
    return CheckResult(
        status=CheckStatus.PASSED,
        passed_count=10,
        failed_count=0,
        metadata={"engine": "mock", "execution_time_ms": 100.0},
    )


@pytest.fixture
def sample_failed_check_result() -> CheckResult:
    """Create a sample failed CheckResult."""
    return CheckResult(
        status=CheckStatus.FAILED,
        passed_count=8,
        failed_count=2,
        failures=(
            {"column": "id", "rule_type": "not_null", "message": "Found null values"},
            {"column": "email", "rule_type": "unique", "message": "Duplicate values found"},
        ),
        metadata={"engine": "mock"},
    )


@pytest.fixture
def sample_profile_result() -> ProfileResult:
    """Create a sample ProfileResult."""
    return ProfileResult(
        row_count=1000,
        column_count=5,
        columns=(
            {"column_name": "id", "dtype": "int64", "null_count": 0},
            {"column_name": "name", "dtype": "string", "null_count": 5},
        ),
        metadata={"engine": "mock"},
    )


@pytest.fixture
def sample_learn_result() -> LearnResult:
    """Create a sample LearnResult."""
    return LearnResult(
        rules=(
            {"column": "id", "rule_type": "not_null", "confidence": 1.0},
            {"column": "id", "rule_type": "unique", "confidence": 1.0},
        ),
        metadata={"engine": "mock"},
    )


@pytest.fixture
def sample_data() -> dict[str, list[Any]]:
    """Create sample data for testing."""
    return {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "email": ["a@test.com", "b@test.com", "c@test.com", "d@test.com", "e@test.com"],
        "age": [25, 30, 35, 28, 42],
    }


@pytest.fixture
def sample_rules() -> tuple[dict[str, Any], ...]:
    """Create sample rules for testing."""
    return (
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
        {"type": "in_range", "column": "age", "min": 0, "max": 150},
    )
