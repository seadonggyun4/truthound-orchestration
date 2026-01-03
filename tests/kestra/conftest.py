"""Pytest fixtures for Kestra package tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from common.base import (
    CheckResult,
    CheckStatus,
    ProfileResult,
    ProfileStatus,
    LearnResult,
    LearnStatus,
    ColumnProfile,
    LearnedRule,
    ValidationFailure,
    Severity,
)
from common.testing import MockDataQualityEngine


@pytest.fixture
def mock_engine() -> MockDataQualityEngine:
    """Create a mock data quality engine."""
    engine = MockDataQualityEngine()
    engine.configure_check(success=True)
    engine.configure_profile(success=True)
    engine.configure_learn(success=True)
    return engine


@pytest.fixture
def failing_mock_engine() -> MockDataQualityEngine:
    """Create a mock engine that produces failed results."""
    engine = MockDataQualityEngine()
    engine.configure_check(success=False, failed_count=5)
    engine.configure_profile(success=False)
    engine.configure_learn(success=False)
    return engine


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
            ValidationFailure(
                rule_name="not_null",
                column="id",
                message="Found null values",
                severity=Severity.ERROR,
            ),
            ValidationFailure(
                rule_name="unique",
                column="email",
                message="Duplicate values found",
                severity=Severity.ERROR,
            ),
        ),
        metadata={"engine": "mock"},
    )


@pytest.fixture
def sample_profile_result() -> ProfileResult:
    """Create a sample ProfileResult."""
    return ProfileResult(
        status=ProfileStatus.COMPLETED,
        row_count=1000,
        column_count=5,
        columns=(
            ColumnProfile(column_name="id", dtype="int64", null_count=0),
            ColumnProfile(column_name="name", dtype="string", null_count=5),
        ),
        metadata={"engine": "mock"},
    )


@pytest.fixture
def sample_learn_result() -> LearnResult:
    """Create a sample LearnResult."""
    return LearnResult(
        status=LearnStatus.COMPLETED,
        rules=(
            LearnedRule(rule_type="not_null", column="id", confidence=1.0),
            LearnedRule(rule_type="unique", column="id", confidence=1.0),
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


@pytest.fixture
def sample_sla_metrics() -> dict[str, Any]:
    """Create sample SLA metrics for testing."""
    return {
        "pass_rate": 0.95,
        "execution_time_seconds": 120.0,
        "row_count": 10000,
        "failed_count": 500,
    }


@pytest.fixture
def sample_flow_config() -> dict[str, Any]:
    """Create sample flow configuration for testing."""
    return {
        "id": "test_flow",
        "namespace": "company.data",
        "description": "Test data quality flow",
    }
