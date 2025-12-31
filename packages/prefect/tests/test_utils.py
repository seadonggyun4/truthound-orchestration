"""Tests for truthound_prefect.utils module."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from truthound_prefect.utils.exceptions import (
    BlockError,
    ConfigurationError,
    DataQualityError,
    EngineError,
)
from truthound_prefect.utils.helpers import (
    calculate_timeout,
    format_count,
    format_duration,
    format_percentage,
    merge_results,
    parse_rules_from_string,
    summarize_check_result,
)
from truthound_prefect.utils.serialization import (
    ResultSerializer,
    serialize_result,
    to_prefect_artifact,
)
from truthound_prefect.utils.types import (
    DataQualityOutput,
    LearnOutput,
    OperationStatus,
    OperationType,
    ProfileOutput,
    QualityCheckMode,
    QualityCheckOutput,
)


class TestExceptions:
    """Tests for custom exceptions."""

    def test_data_quality_error(self) -> None:
        """Test DataQualityError creation and to_dict."""
        error = DataQualityError(
            message="Test error",
            result={"failed_count": 5},
            metadata={"key": "value"},
        )
        assert str(error) == "Test error"
        assert error.result == {"failed_count": 5}

        d = error.to_dict()
        assert d["error_type"] == "DataQualityError"
        assert d["message"] == "Test error"

    def test_configuration_error(self) -> None:
        """Test ConfigurationError with field info."""
        error = ConfigurationError(
            message="Invalid config",
            field="timeout",
            value=-1,
            reason="must be positive",
        )
        assert "timeout" in str(error)
        assert "-1" in str(error)
        assert "must be positive" in str(error)

    def test_engine_error(self) -> None:
        """Test EngineError with engine info."""
        original = ValueError("inner error")
        error = EngineError(
            message="Engine failed",
            engine_name="truthound",
            operation="check",
            original_error=original,
        )
        assert "truthound" in str(error)
        assert "check" in str(error)
        assert "inner error" in str(error)

    def test_block_error(self) -> None:
        """Test BlockError with block info."""
        error = BlockError(
            message="Block failed",
            block_name="my-block",
            block_type="DataQualityBlock",
            operation="check",
        )
        assert "my-block" in str(error)
        assert "DataQualityBlock" in str(error)


class TestTypes:
    """Tests for type definitions."""

    def test_quality_check_mode_enum(self) -> None:
        """Test QualityCheckMode enum values."""
        assert QualityCheckMode.BEFORE == "before"
        assert QualityCheckMode.AFTER == "after"
        assert QualityCheckMode.BOTH == "both"
        assert QualityCheckMode.NONE == "none"

    def test_operation_type_enum(self) -> None:
        """Test OperationType enum values."""
        assert OperationType.CHECK == "check"
        assert OperationType.PROFILE == "profile"
        assert OperationType.LEARN == "learn"

    def test_operation_status_enum(self) -> None:
        """Test OperationStatus enum values."""
        assert OperationStatus.SUCCESS == "success"
        assert OperationStatus.FAILURE == "failure"

    def test_data_quality_output(self) -> None:
        """Test DataQualityOutput creation and methods."""
        output = DataQualityOutput(
            data={"key": "value"},
            result={"status": "passed"},
            is_success=True,
        )
        assert output.is_success is True
        assert output.data == {"key": "value"}

        # Test with_metadata
        new_output = output.with_metadata(extra="info")
        assert new_output.metadata["extra"] == "info"

        # Test to_dict
        d = output.to_dict()
        assert "data_type" in d
        assert "is_success" in d

    def test_quality_check_output(self) -> None:
        """Test QualityCheckOutput creation."""
        output = QualityCheckOutput(
            data=[1, 2, 3],
            result={"status": "passed"},
            is_success=True,
            passed_count=5,
            failed_count=0,
        )
        assert output.passed_count == 5
        assert output.failure_rate == 0.0

    def test_quality_check_output_from_result(
        self,
        sample_check_result: dict[str, Any],
    ) -> None:
        """Test QualityCheckOutput.from_result factory."""
        output = QualityCheckOutput.from_result(
            data=[1, 2, 3],
            result=sample_check_result,
        )
        assert output.passed_count == 5
        assert output.is_success is True

    def test_profile_output(self) -> None:
        """Test ProfileOutput creation."""
        output = ProfileOutput(
            data=[1, 2, 3],
            row_count=100,
            column_count=5,
        )
        assert output.row_count == 100
        assert output.column_count == 5

    def test_learn_output(self) -> None:
        """Test LearnOutput creation."""
        output = LearnOutput(
            data=[1, 2, 3],
            rules_count=3,
        )
        assert output.rules_count == 3


class TestHelpers:
    """Tests for helper functions."""

    def test_format_duration(self) -> None:
        """Test duration formatting."""
        assert format_duration(500) == "500.00ms"
        assert format_duration(5000) == "5.00s"
        assert format_duration(120000) == "2.00min"
        assert format_duration(3600000) == "1.00h"

    def test_format_percentage(self) -> None:
        """Test percentage formatting."""
        assert format_percentage(0.5) == "50.00%"
        assert format_percentage(0.9523, decimals=1) == "95.2%"

    def test_format_count(self) -> None:
        """Test count formatting with thousands separators."""
        assert format_count(1234567) == "1,234,567"

    def test_summarize_check_result(
        self,
        sample_check_result: dict[str, Any],
    ) -> None:
        """Test check result summarization."""
        summary = summarize_check_result(sample_check_result)
        assert "PASSED" in summary
        assert "5" in summary

    def test_summarize_failed_check_result(
        self,
        sample_failed_check_result: dict[str, Any],
    ) -> None:
        """Test failed check result summarization."""
        summary = summarize_check_result(sample_failed_check_result)
        assert "FAILED" in summary
        assert "3" in summary
        assert "2" in summary

    def test_calculate_timeout(self) -> None:
        """Test dynamic timeout calculation."""
        # Base timeout
        assert calculate_timeout(60.0) == 60.0

        # With data size
        timeout = calculate_timeout(60.0, data_size=10000, scale_factor=0.01)
        assert timeout == 160.0

        # Respects bounds
        assert calculate_timeout(10.0, min_timeout=30.0) == 30.0
        assert calculate_timeout(5000.0, max_timeout=3600.0) == 3600.0

    def test_parse_rules_from_string(self) -> None:
        """Test JSON rules parsing."""
        rules_json = '[{"type": "not_null", "column": "id"}]'
        rules = parse_rules_from_string(rules_json)
        assert len(rules) == 1
        assert rules[0]["type"] == "not_null"

        # Empty string
        assert parse_rules_from_string("") == []
        assert parse_rules_from_string("  ") == []

        # Invalid JSON
        with pytest.raises(ValueError):
            parse_rules_from_string("not json")

    def test_merge_results(
        self,
        sample_check_result: dict[str, Any],
        sample_failed_check_result: dict[str, Any],
    ) -> None:
        """Test result merging."""
        results = [sample_check_result, sample_failed_check_result]

        # Combine strategy
        merged = merge_results(results, strategy="combine")
        assert merged["passed_count"] == 8  # 5 + 3
        assert merged["failed_count"] == 2  # 0 + 2
        assert merged["merged_from"] == 2

        # Worst strategy
        worst = merge_results(results, strategy="worst")
        assert worst["failed_count"] == 2

        # Best strategy
        best = merge_results(results, strategy="best")
        assert best["passed_count"] == 5


class TestSerialization:
    """Tests for serialization utilities."""

    def test_serialize_result_check(
        self,
        sample_check_result: dict[str, Any],
    ) -> None:
        """Test serializing a check result dict (already serialized)."""
        # Since we're passing a dict, it should return as-is or process it
        # The serialize_result function expects actual result objects
        # For now, test the ResultSerializer directly
        serializer = ResultSerializer()

        # Create a mock result object
        class MockCheckResult:
            status = type("Status", (), {"value": "passed"})()
            is_success = True
            passed_count = 5
            failed_count = 0
            failures = []
            execution_time_ms = 150.5
            timestamp = datetime.now()
            metadata = {}

        result = serializer.serialize_check_result(MockCheckResult())
        assert result["status"] == "passed"
        assert result["passed_count"] == 5

    def test_to_prefect_artifact_table(
        self,
        sample_check_result: dict[str, Any],
    ) -> None:
        """Test converting result to table artifact."""
        artifact = to_prefect_artifact(sample_check_result, "table")
        assert artifact["type"] == "table"
        assert "data" in artifact
        assert len(artifact["data"]) >= 3  # At least status, passed, failed

    def test_to_prefect_artifact_markdown(
        self,
        sample_check_result: dict[str, Any],
    ) -> None:
        """Test converting result to markdown artifact."""
        artifact = to_prefect_artifact(sample_check_result, "markdown")
        assert artifact["type"] == "markdown"
        assert "content" in artifact
        assert "PASSED" in artifact["content"] or "passed" in artifact["content"]
