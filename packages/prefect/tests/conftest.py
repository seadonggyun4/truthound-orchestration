"""Pytest configuration and fixtures for truthound-prefect tests."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_check_result() -> dict[str, Any]:
    """Sample check result for testing."""
    return {
        "status": "passed",
        "is_success": True,
        "passed_count": 5,
        "failed_count": 0,
        "failure_rate": 0.0,
        "failures": [],
        "execution_time_ms": 150.5,
        "timestamp": datetime.now().isoformat(),
        "metadata": {"row_count": 1000},
    }


@pytest.fixture
def sample_failed_check_result() -> dict[str, Any]:
    """Sample failed check result for testing."""
    return {
        "status": "failed",
        "is_success": False,
        "passed_count": 3,
        "failed_count": 2,
        "failure_rate": 0.4,
        "failures": [
            {
                "rule_name": "not_null",
                "column": "id",
                "message": "Found 10 null values",
                "severity": "error",
                "failed_count": 10,
                "total_count": 100,
            },
            {
                "rule_name": "unique",
                "column": "email",
                "message": "Found 5 duplicate values",
                "severity": "warning",
                "failed_count": 5,
                "total_count": 100,
            },
        ],
        "execution_time_ms": 250.0,
        "timestamp": datetime.now().isoformat(),
        "metadata": {"row_count": 100},
    }


@pytest.fixture
def sample_profile_result() -> dict[str, Any]:
    """Sample profile result for testing."""
    return {
        "row_count": 1000,
        "column_count": 5,
        "columns": [
            {
                "column_name": "id",
                "dtype": "int64",
                "null_count": 0,
                "null_percentage": 0.0,
                "unique_count": 1000,
                "unique_percentage": 1.0,
            },
            {
                "column_name": "name",
                "dtype": "str",
                "null_count": 10,
                "null_percentage": 0.01,
                "unique_count": 900,
                "unique_percentage": 0.9,
            },
        ],
        "execution_time_ms": 500.0,
        "timestamp": datetime.now().isoformat(),
        "metadata": {},
    }


@pytest.fixture
def sample_learn_result() -> dict[str, Any]:
    """Sample learn result for testing."""
    return {
        "rules": [
            {
                "rule_type": "not_null",
                "column": "id",
                "confidence": 1.0,
            },
            {
                "rule_type": "unique",
                "column": "id",
                "confidence": 1.0,
            },
            {
                "rule_type": "in_range",
                "column": "age",
                "confidence": 0.95,
                "parameters": {"min": 0, "max": 120},
            },
        ],
        "execution_time_ms": 800.0,
        "timestamp": datetime.now().isoformat(),
        "metadata": {},
    }


@pytest.fixture
def sample_rules() -> list[dict[str, Any]]:
    """Sample rules for testing."""
    return [
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
        {"type": "in_range", "column": "age", "min": 0, "max": 150},
    ]


@pytest.fixture
def mock_engine() -> MagicMock:
    """Mock data quality engine for testing."""
    engine = MagicMock()
    engine.engine_name = "mock_engine"
    engine.engine_version = "1.0.0"
    return engine


@pytest.fixture
def mock_block(sample_check_result: dict[str, Any]) -> MagicMock:
    """Mock DataQualityBlock for testing."""
    block = MagicMock()
    block.check.return_value = sample_check_result
    block.profile.return_value = {
        "row_count": 1000,
        "column_count": 5,
        "columns": [],
        "execution_time_ms": 100.0,
    }
    block.learn.return_value = {
        "rules": [],
        "execution_time_ms": 100.0,
    }
    return block
