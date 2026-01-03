"""Tests for Mage exception hierarchy and error handling."""

from __future__ import annotations

import pytest

from truthound_mage.utils.exceptions import (
    DataQualityBlockError,
    BlockConfigurationError,
    BlockExecutionError,
    DataLoadError,
    SLAViolationError,
)


class TestDataQualityBlockError:
    """Tests for base DataQualityBlockError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        error = DataQualityBlockError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.block_uuid is None
        assert error.pipeline_uuid is None
        assert error.details == {}

    def test_creation_with_all_fields(self) -> None:
        """Test exception creation with all fields."""
        error = DataQualityBlockError(
            "Test error",
            block_uuid="block_123",
            pipeline_uuid="pipeline_456",
            details={"key": "value"},
        )
        assert error.block_uuid == "block_123"
        assert error.pipeline_uuid == "pipeline_456"
        assert error.details == {"key": "value"}

    def test_str_with_block_uuid(self) -> None:
        """Test string representation with block_uuid."""
        error = DataQualityBlockError("Error", block_uuid="block_123")
        assert "block_uuid=block_123" in str(error)

    def test_str_with_pipeline_uuid(self) -> None:
        """Test string representation with pipeline_uuid."""
        error = DataQualityBlockError("Error", pipeline_uuid="pipe_123")
        assert "pipeline_uuid=pipe_123" in str(error)

    def test_str_with_all_fields(self) -> None:
        """Test string representation with all fields."""
        error = DataQualityBlockError(
            "Error message",
            block_uuid="block_123",
            pipeline_uuid="pipe_456",
        )
        result = str(error)
        assert "Error message" in result
        assert "block_uuid=block_123" in result
        assert "pipeline_uuid=pipe_456" in result

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        error = DataQualityBlockError(
            "Test error",
            block_uuid="block_123",
            pipeline_uuid="pipe_456",
            details={"extra": "info"},
        )
        d = error.to_dict()
        assert d["error_type"] == "DataQualityBlockError"
        assert d["message"] == "Test error"
        assert d["block_uuid"] == "block_123"
        assert d["pipeline_uuid"] == "pipe_456"
        assert d["details"] == {"extra": "info"}

    def test_is_exception(self) -> None:
        """Test that it's a proper Exception subclass."""
        error = DataQualityBlockError("Test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """Test that exception can be raised and caught."""
        with pytest.raises(DataQualityBlockError) as exc_info:
            raise DataQualityBlockError("Test error")
        assert exc_info.value.message == "Test error"


class TestBlockConfigurationError:
    """Tests for BlockConfigurationError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = BlockConfigurationError("Invalid config")
        assert error.message == "Invalid config"
        assert error.field is None
        assert error.value is None

    def test_creation_with_field_and_value(self) -> None:
        """Test creation with field and value."""
        error = BlockConfigurationError(
            "Invalid timeout",
            field="timeout_seconds",
            value=-5,
        )
        assert error.field == "timeout_seconds"
        assert error.value == -5

    def test_inheritance(self) -> None:
        """Test inheritance from DataQualityBlockError."""
        error = BlockConfigurationError("Error")
        assert isinstance(error, DataQualityBlockError)
        assert isinstance(error, Exception)

    def test_to_dict_includes_field_and_value(self) -> None:
        """Test to_dict includes field and value."""
        error = BlockConfigurationError(
            "Invalid",
            field="timeout",
            value=100,
            block_uuid="block_1",
        )
        d = error.to_dict()
        assert d["error_type"] == "BlockConfigurationError"
        assert d["field"] == "timeout"
        assert d["value"] == 100
        assert d["block_uuid"] == "block_1"

    def test_can_catch_as_base_error(self) -> None:
        """Test catching as base error type."""
        with pytest.raises(DataQualityBlockError):
            raise BlockConfigurationError("Config error")


class TestBlockExecutionError:
    """Tests for BlockExecutionError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = BlockExecutionError("Execution failed")
        assert error.message == "Execution failed"
        assert error.original_error is None
        assert error.operation is None

    def test_creation_with_original_error(self) -> None:
        """Test creation with original error."""
        original = ValueError("Invalid data")
        error = BlockExecutionError(
            "Check failed",
            original_error=original,
            operation="check",
        )
        assert error.original_error is original
        assert error.operation == "check"

    def test_str_with_original_error(self) -> None:
        """Test string representation includes original error."""
        original = RuntimeError("Engine crashed")
        error = BlockExecutionError("Failed", original_error=original)
        result = str(error)
        assert "RuntimeError" in result
        assert "Engine crashed" in result
        assert "caused_by" in result

    def test_str_without_original_error(self) -> None:
        """Test string representation without original error."""
        error = BlockExecutionError("Failed")
        result = str(error)
        assert "caused_by" not in result

    def test_to_dict_with_original_error(self) -> None:
        """Test to_dict includes original error info."""
        original = TypeError("Wrong type")
        error = BlockExecutionError(
            "Failed",
            original_error=original,
            operation="profile",
        )
        d = error.to_dict()
        assert d["operation"] == "profile"
        assert d["original_error"]["type"] == "TypeError"
        assert d["original_error"]["message"] == "Wrong type"

    def test_to_dict_without_original_error(self) -> None:
        """Test to_dict without original error."""
        error = BlockExecutionError("Failed", operation="learn")
        d = error.to_dict()
        assert d["operation"] == "learn"
        assert "original_error" not in d or d.get("original_error") is None


class TestDataLoadError:
    """Tests for DataLoadError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = DataLoadError("Failed to load data")
        assert error.message == "Failed to load data"
        assert error.source is None
        assert error.original_error is None

    def test_creation_with_source(self) -> None:
        """Test creation with source."""
        error = DataLoadError(
            "Load failed",
            source="s3://bucket/data.parquet",
        )
        assert error.source == "s3://bucket/data.parquet"

    def test_creation_with_original_error(self) -> None:
        """Test creation with original error."""
        original = FileNotFoundError("File not found")
        error = DataLoadError(
            "Load failed",
            source="/path/to/file.csv",
            original_error=original,
        )
        assert error.original_error is original

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields."""
        original = OSError("I/O error")
        error = DataLoadError(
            "Load failed",
            source="db://table",
            original_error=original,
            block_uuid="loader_block",
        )
        d = error.to_dict()
        assert d["source"] == "db://table"
        assert d["original_error"]["type"] == "OSError"
        assert d["block_uuid"] == "loader_block"

    def test_inheritance(self) -> None:
        """Test inheritance."""
        error = DataLoadError("Error")
        assert isinstance(error, DataQualityBlockError)


class TestSLAViolationError:
    """Tests for SLAViolationError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = SLAViolationError("SLA violated")
        assert error.message == "SLA violated"
        assert error.violations == []

    def test_creation_with_violations(self) -> None:
        """Test creation with violations list."""
        violations = [
            {"type": "pass_rate", "threshold": 0.95, "actual": 0.90},
            {"type": "execution_time", "threshold": 60, "actual": 120},
        ]
        error = SLAViolationError("SLA violated", violations=violations)
        assert len(error.violations) == 2

    def test_violation_count_property(self) -> None:
        """Test violation_count property."""
        violations = [
            {"type": "a"},
            {"type": "b"},
            {"type": "c"},
        ]
        error = SLAViolationError("Error", violations=violations)
        assert error.violation_count == 3

    def test_critical_count_property(self) -> None:
        """Test critical_count property."""
        violations = [
            {"type": "a", "alert_level": "critical"},
            {"type": "b", "alert_level": "warning"},
            {"type": "c", "alert_level": "critical"},
        ]
        error = SLAViolationError("Error", violations=violations)
        assert error.critical_count == 2

    def test_critical_count_no_critical(self) -> None:
        """Test critical_count with no critical violations."""
        violations = [
            {"type": "a", "alert_level": "warning"},
            {"type": "b", "alert_level": "info"},
        ]
        error = SLAViolationError("Error", violations=violations)
        assert error.critical_count == 0

    def test_to_dict_includes_violations(self) -> None:
        """Test to_dict includes violations."""
        violations = [{"type": "test", "alert_level": "critical"}]
        error = SLAViolationError("Error", violations=violations)
        d = error.to_dict()
        assert d["violations"] == violations
        assert d["violation_count"] == 1
        assert d["critical_count"] == 1


class TestExceptionChaining:
    """Tests for exception chaining scenarios."""

    def test_block_execution_wraps_engine_error(self) -> None:
        """Test BlockExecutionError wrapping engine error."""
        original = RuntimeError("Engine internal error")
        error = BlockExecutionError(
            "Data quality check failed",
            original_error=original,
            operation="check",
            block_uuid="check_block",
        )

        # Verify chain
        assert error.original_error is original
        assert "RuntimeError" in str(error)

        # Verify serialization preserves chain info
        d = error.to_dict()
        assert d["original_error"]["type"] == "RuntimeError"

    def test_data_load_wraps_io_error(self) -> None:
        """Test DataLoadError wrapping IO error."""
        original = PermissionError("Access denied")
        error = DataLoadError(
            "Cannot load input file",
            source="/protected/file.csv",
            original_error=original,
        )

        d = error.to_dict()
        assert d["source"] == "/protected/file.csv"
        assert d["original_error"]["type"] == "PermissionError"


class TestExceptionCatching:
    """Tests for exception catching patterns."""

    def test_catch_all_data_quality_errors(self) -> None:
        """Test catching all data quality errors with base class."""
        errors = [
            DataQualityBlockError("Base error"),
            BlockConfigurationError("Config error"),
            BlockExecutionError("Execution error"),
            DataLoadError("Load error"),
            SLAViolationError("SLA error"),
        ]

        for error in errors:
            with pytest.raises(DataQualityBlockError):
                raise error

    def test_catch_specific_error_types(self) -> None:
        """Test catching specific error types."""
        with pytest.raises(BlockConfigurationError):
            raise BlockConfigurationError("Config", field="x")

        with pytest.raises(BlockExecutionError):
            raise BlockExecutionError("Exec", operation="check")

        with pytest.raises(DataLoadError):
            raise DataLoadError("Load", source="s3://")

        with pytest.raises(SLAViolationError):
            raise SLAViolationError("SLA", violations=[])

    def test_exception_does_not_match_sibling(self) -> None:
        """Test that sibling exceptions don't catch each other."""
        # BlockConfigurationError should not catch BlockExecutionError
        with pytest.raises(BlockExecutionError):
            try:
                raise BlockExecutionError("Exec error")
            except BlockConfigurationError:
                pytest.fail("Should not catch sibling exception")
