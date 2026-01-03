"""Tests for Mage utility functions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from truthound_mage.utils.exceptions import (
    DataQualityBlockError,
    BlockConfigurationError,
    BlockExecutionError,
    DataLoadError,
    SLAViolationError,
)
from truthound_mage.utils.types import (
    BlockMetadata,
    DataQualityOutput,
)
from truthound_mage.utils.serialization import (
    serialize_result,
    deserialize_result,
    serialize_check_result,
    deserialize_check_result,
    to_json,
    from_json,
)
from truthound_mage.utils.helpers import (
    format_check_result,
    format_violations,
    create_block_metadata,
    get_data_size,
    validate_data_input,
    create_output,
    merge_rules,
)
from common.base import CheckResult, CheckStatus


class TestExceptions:
    """Tests for exception classes."""

    def test_base_exception(self) -> None:
        """Test base exception."""
        exc = DataQualityBlockError(
            "Test error",
            block_uuid="test",
            pipeline_uuid="pipeline",
        )
        assert exc.message == "Test error"
        assert exc.block_uuid == "test"
        assert "test" in str(exc)

    def test_base_exception_to_dict(self) -> None:
        """Test exception to_dict."""
        exc = DataQualityBlockError("Test", block_uuid="test")
        data = exc.to_dict()

        assert data["error_type"] == "DataQualityBlockError"
        assert data["message"] == "Test"

    def test_configuration_error(self) -> None:
        """Test configuration error."""
        exc = BlockConfigurationError(
            "Invalid config",
            field="timeout",
            value=-1,
        )
        assert exc.field == "timeout"
        assert exc.value == -1

    def test_execution_error(self) -> None:
        """Test execution error."""
        original = ValueError("Original error")
        exc = BlockExecutionError(
            "Execution failed",
            original_error=original,
            operation="check",
        )
        assert exc.original_error is original
        assert "ValueError" in str(exc)

    def test_data_load_error(self) -> None:
        """Test data load error."""
        exc = DataLoadError(
            "Load failed",
            source="s3://bucket/data",
        )
        assert exc.source == "s3://bucket/data"

    def test_sla_violation_error(self) -> None:
        """Test SLA violation error."""
        violations = [
            {"alert_level": "critical", "message": "Test"},
            {"alert_level": "warning", "message": "Test2"},
        ]
        exc = SLAViolationError(
            "SLA violated",
            violations=violations,
        )
        assert exc.violation_count == 2
        assert exc.critical_count == 1


class TestBlockMetadata:
    """Tests for BlockMetadata."""

    def test_creation(self) -> None:
        """Test metadata creation."""
        metadata = BlockMetadata(
            block_uuid="test",
            pipeline_uuid="pipeline",
            started_at=datetime.now(timezone.utc),
        )
        assert metadata.block_uuid == "test"
        assert metadata.pipeline_uuid == "pipeline"

    def test_with_completion(self) -> None:
        """Test completion builder."""
        started = datetime.now(timezone.utc)
        metadata = BlockMetadata(
            block_uuid="test",
            started_at=started,
        )
        completed = metadata.with_completion()

        assert completed.completed_at is not None
        assert completed.duration_ms is not None
        assert completed.duration_ms >= 0

    def test_with_engine(self) -> None:
        """Test engine builder."""
        metadata = BlockMetadata(block_uuid="test")
        new_metadata = metadata.with_engine("truthound", "1.0.0")

        assert new_metadata.engine_name == "truthound"
        assert new_metadata.engine_version == "1.0.0"

    def test_with_extra(self) -> None:
        """Test extra fields builder."""
        metadata = BlockMetadata(block_uuid="test")
        new_metadata = metadata.with_extra(custom_field="value")

        assert dict(new_metadata.extra)["custom_field"] == "value"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        metadata = BlockMetadata(
            block_uuid="test",
            operation="check",
            tags=frozenset({"prod"}),
        )
        data = metadata.to_dict()

        assert data["block_uuid"] == "test"
        assert data["operation"] == "check"
        assert "prod" in data["tags"]

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "block_uuid": "test",
            "pipeline_uuid": "pipeline",
            "operation": "check",
        }
        metadata = BlockMetadata.from_dict(data)

        assert metadata.block_uuid == "test"
        assert metadata.operation == "check"


class TestDataQualityOutput:
    """Tests for DataQualityOutput."""

    def test_creation(self, sample_check_result) -> None:
        """Test output creation."""
        metadata = BlockMetadata(block_uuid="test")
        output = DataQualityOutput(
            success=True,
            result=sample_check_result,
            metadata=metadata,
        )
        assert output.success is True

    def test_with_data(self, sample_check_result) -> None:
        """Test data builder."""
        metadata = BlockMetadata(block_uuid="test")
        output = DataQualityOutput(
            success=True,
            result=sample_check_result,
            metadata=metadata,
        )
        new_output = output.with_data({"key": "value"})

        assert new_output.data == {"key": "value"}

    def test_with_summary(self, sample_check_result) -> None:
        """Test summary builder."""
        metadata = BlockMetadata(block_uuid="test")
        output = DataQualityOutput(
            success=True,
            result=sample_check_result,
            metadata=metadata,
        )
        new_output = output.with_summary(passed=10, failed=2)

        assert new_output.summary_dict["passed"] == 10
        assert new_output.summary_dict["failed"] == 2

    def test_to_dict(self, sample_check_result) -> None:
        """Test dictionary conversion."""
        metadata = BlockMetadata(block_uuid="test")
        output = DataQualityOutput(
            success=True,
            result=sample_check_result,
            metadata=metadata,
            summary=(("key", "value"),),
        )
        data = output.to_dict()

        assert data["success"] is True
        assert data["summary"]["key"] == "value"


class TestSerialization:
    """Tests for serialization functions."""

    def test_serialize_check_result(self, sample_check_result) -> None:
        """Test CheckResult serialization."""
        data = serialize_check_result(sample_check_result)

        assert data["_type"] == "CheckResult"
        assert data["status"] == "PASSED"
        assert data["passed_count"] == 10

    def test_deserialize_check_result(self) -> None:
        """Test CheckResult deserialization."""
        data = {
            "status": "PASSED",
            "passed_count": 10,
            "failed_count": 0,
        }
        result = deserialize_check_result(data)

        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 10

    def test_serialize_result_auto_detect(self, sample_check_result) -> None:
        """Test auto-detection of result type."""
        data = serialize_result(sample_check_result)
        assert data["_type"] == "CheckResult"

    def test_deserialize_result_auto_detect(self) -> None:
        """Test auto-detection during deserialization."""
        data = {
            "_type": "CheckResult",
            "status": "PASSED",
            "passed_count": 5,
            "failed_count": 0,
        }
        result = deserialize_result(data)

        assert isinstance(result, CheckResult)

    def test_to_json_from_json(self, sample_check_result) -> None:
        """Test JSON conversion."""
        json_str = to_json(sample_check_result)
        result = from_json(json_str, "check")

        assert result.status == CheckStatus.PASSED


class TestHelpers:
    """Tests for helper functions."""

    def test_format_check_result_passed(self, sample_check_result) -> None:
        """Test formatting passed result."""
        output = format_check_result(sample_check_result)

        assert "PASSED" in output
        assert "100.0%" in output

    def test_format_check_result_failed(self, sample_failed_check_result) -> None:
        """Test formatting failed result."""
        output = format_check_result(sample_failed_check_result)

        assert "FAILED" in output
        assert "Failures:" in output

    def test_format_violations(self) -> None:
        """Test formatting violations."""
        violations = [
            {"alert_level": "critical", "message": "Test violation"},
        ]
        output = format_violations(violations)

        assert "SLA Violations: 1" in output
        assert "CRITICAL" in output

    def test_format_violations_empty(self) -> None:
        """Test formatting empty violations."""
        output = format_violations([])
        assert "No SLA violations" in output

    def test_create_block_metadata(self) -> None:
        """Test metadata creation helper."""
        metadata = create_block_metadata(
            "test",
            pipeline_uuid="pipeline",
            operation="check",
            engine_name="truthound",
        )

        assert metadata.block_uuid == "test"
        assert metadata.pipeline_uuid == "pipeline"
        assert metadata.started_at is not None

    def test_get_data_size_dict(self) -> None:
        """Test getting size of dict data."""
        data = {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
        rows, cols = get_data_size(data)

        assert rows == 3
        assert cols == 2

    def test_get_data_size_list(self) -> None:
        """Test getting size of list data."""
        data = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        rows, cols = get_data_size(data)

        assert rows == 2
        assert cols == 2

    def test_validate_data_input_valid(self) -> None:
        """Test validating valid data."""
        data = {"a": [1, 2, 3]}
        valid, error = validate_data_input(data)

        assert valid is True
        assert error is None

    def test_validate_data_input_empty(self) -> None:
        """Test validating empty data."""
        data = {"a": []}
        valid, error = validate_data_input(data, allow_empty=False)

        assert valid is False
        assert "empty" in error.lower()

    def test_validate_data_input_min_rows(self) -> None:
        """Test minimum rows validation."""
        data = {"a": [1, 2]}
        valid, error = validate_data_input(data, min_rows=5)

        assert valid is False
        assert "minimum" in error.lower()

    def test_validate_data_input_required_columns(self) -> None:
        """Test required columns validation."""
        data = {"a": [1, 2]}
        valid, error = validate_data_input(
            data,
            required_columns=["a", "b"],
        )

        assert valid is False
        assert "Missing" in error

    def test_create_output(self, sample_check_result) -> None:
        """Test output creation helper."""
        metadata = create_block_metadata("test")
        output = create_output(
            sample_check_result,
            metadata,
            passed=10,
            failed=0,
        )

        assert output.success is True
        assert output.summary_dict["passed"] == 10

    def test_merge_rules(self) -> None:
        """Test merging rule sets."""
        rules1 = [{"type": "not_null", "column": "id"}]
        rules2 = [{"type": "unique", "column": "id"}]

        merged = merge_rules(rules1, rules2)

        assert len(merged) == 2

    def test_merge_rules_deduplicate(self) -> None:
        """Test deduplication when merging."""
        rules1 = [{"type": "not_null", "column": "id"}]
        rules2 = [{"type": "not_null", "column": "id"}]  # Duplicate

        merged = merge_rules(rules1, rules2, deduplicate=True)

        assert len(merged) == 1
