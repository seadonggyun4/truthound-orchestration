"""Tests for Kestra exception hierarchy and error handling."""

from __future__ import annotations

from typing import Any

import pytest

from truthound_kestra.utils.exceptions import (
    DataQualityError,
    ConfigurationError,
    EngineError,
    ScriptError,
    FlowError,
    OutputError,
    SLAViolationError,
    SerializationError,
)


class TestDataQualityError:
    """Tests for base DataQualityError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        error = DataQualityError(message="Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.result is None
        assert error.metadata == {}

    def test_creation_with_result(self) -> None:
        """Test creation with result."""
        result = {"status": "FAILED", "failed_count": 5}
        error = DataQualityError(message="Check failed", result=result)
        assert error.result == result

    def test_creation_with_metadata(self) -> None:
        """Test creation with metadata."""
        error = DataQualityError(
            message="Error",
            metadata={"task_id": "test_task"},
        )
        assert error.metadata["task_id"] == "test_task"

    def test_str_with_result(self) -> None:
        """Test string representation includes result summary."""
        error = DataQualityError(
            message="Check failed",
            result={"status": "FAILED", "failed_count": 3},
        )
        result_str = str(error)
        assert "status=FAILED" in result_str
        assert "failed=3" in result_str

    def test_str_without_result(self) -> None:
        """Test string representation without result."""
        error = DataQualityError(message="Simple error")
        assert str(error) == "Simple error"

    def test_repr(self) -> None:
        """Test repr representation."""
        error = DataQualityError(
            message="Error",
            result={"x": 1},
            metadata={"y": 2},
        )
        repr_str = repr(error)
        assert "DataQualityError" in repr_str
        assert "message='Error'" in repr_str

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        error = DataQualityError(
            message="Test",
            result={"status": "OK"},
            metadata={"key": "value"},
        )
        d = error.to_dict()
        assert d["type"] == "DataQualityError"
        assert d["message"] == "Test"
        assert d["result"] == {"status": "OK"}
        assert d["metadata"] == {"key": "value"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "message": "Restored error",
            "result": {"status": "FAILED"},
            "metadata": {"restored": True},
        }
        error = DataQualityError.from_dict(data)
        assert error.message == "Restored error"
        assert error.result == {"status": "FAILED"}
        assert error.metadata["restored"] is True

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal data."""
        data: dict[str, Any] = {}
        error = DataQualityError.from_dict(data)
        assert error.message == "Unknown error"
        assert error.result is None
        assert error.metadata == {}

    def test_is_exception(self) -> None:
        """Test that it's a proper Exception."""
        error = DataQualityError(message="Test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """Test raising and catching."""
        with pytest.raises(DataQualityError) as exc_info:
            raise DataQualityError(message="Test")
        assert exc_info.value.message == "Test"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = ConfigurationError(message="Invalid config")
        assert error.message == "Invalid config"
        assert error.field is None
        assert error.value is None
        assert error.reason is None

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields."""
        error = ConfigurationError(
            message="Invalid timeout",
            field="timeout_seconds",
            value=-5,
            reason="Must be positive",
        )
        assert error.field == "timeout_seconds"
        assert error.value == -5
        assert error.reason == "Must be positive"

    def test_str_format(self) -> None:
        """Test string format includes all fields."""
        error = ConfigurationError(
            message="Error",
            field="test_field",
            value=123,
            reason="Invalid",
        )
        s = str(error)
        assert "Error" in s
        assert "field='test_field'" in s
        assert "value=123" in s
        assert "reason='Invalid'" in s

    def test_to_dict(self) -> None:
        """Test to_dict includes configuration-specific fields."""
        error = ConfigurationError(
            message="Config error",
            field="batch_size",
            value=0,
            reason="Must be > 0",
        )
        d = error.to_dict()
        assert d["type"] == "ConfigurationError"
        assert d["field"] == "batch_size"
        assert d["value"] == 0
        assert d["reason"] == "Must be > 0"

    def test_inheritance(self) -> None:
        """Test inheritance from DataQualityError."""
        error = ConfigurationError(message="Error")
        assert isinstance(error, DataQualityError)


class TestEngineError:
    """Tests for EngineError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = EngineError(message="Engine failed")
        assert error.engine_name is None
        assert error.operation is None
        assert error.original_error is None

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields."""
        original = ValueError("Bad data")
        error = EngineError(
            message="Check failed",
            engine_name="truthound",
            operation="check",
            original_error=original,
        )
        assert error.engine_name == "truthound"
        assert error.operation == "check"
        assert error.original_error is original

    def test_str_format(self) -> None:
        """Test string format."""
        error = EngineError(
            message="Failed",
            engine_name="ge",
            operation="profile",
            original_error=RuntimeError("Crash"),
        )
        s = str(error)
        assert "engine='ge'" in s
        assert "operation='profile'" in s
        assert "caused_by=RuntimeError" in s

    def test_to_dict_with_original_error(self) -> None:
        """Test to_dict with original error."""
        original = TypeError("Type mismatch")
        error = EngineError(
            message="Engine error",
            engine_name="pandera",
            operation="learn",
            original_error=original,
        )
        d = error.to_dict()
        assert d["engine_name"] == "pandera"
        assert d["operation"] == "learn"
        assert d["original_error"] == "Type mismatch"
        assert d["original_error_type"] == "TypeError"

    def test_to_dict_without_original_error(self) -> None:
        """Test to_dict without original error."""
        error = EngineError(message="Error")
        d = error.to_dict()
        assert d["original_error"] is None
        assert d["original_error_type"] is None


class TestScriptError:
    """Tests for ScriptError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = ScriptError(message="Script failed")
        assert error.script_name is None
        assert error.task_id is None
        assert error.execution_id is None

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields."""
        error = ScriptError(
            message="Failed to load data",
            script_name="check_quality_script",
            task_id="validate_users",
            execution_id="exec_123",
        )
        assert error.script_name == "check_quality_script"
        assert error.task_id == "validate_users"
        assert error.execution_id == "exec_123"

    def test_str_format(self) -> None:
        """Test string format."""
        error = ScriptError(
            message="Error",
            script_name="my_script",
            task_id="task_1",
            execution_id="exec_1",
        )
        s = str(error)
        assert "script='my_script'" in s
        assert "task_id='task_1'" in s
        assert "execution_id='exec_1'" in s

    def test_to_dict(self) -> None:
        """Test to_dict includes script-specific fields."""
        error = ScriptError(
            message="Script error",
            script_name="profile_script",
            task_id="profile_task",
            execution_id="abc123",
        )
        d = error.to_dict()
        assert d["type"] == "ScriptError"
        assert d["script_name"] == "profile_script"
        assert d["task_id"] == "profile_task"
        assert d["execution_id"] == "abc123"


class TestFlowError:
    """Tests for FlowError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = FlowError(message="Flow failed")
        assert error.flow_id is None
        assert error.namespace is None
        assert error.trigger is None

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields."""
        error = FlowError(
            message="Flow generation failed",
            flow_id="data_quality_pipeline",
            namespace="production",
            trigger="schedule",
        )
        assert error.flow_id == "data_quality_pipeline"
        assert error.namespace == "production"
        assert error.trigger == "schedule"

    def test_str_format(self) -> None:
        """Test string format."""
        error = FlowError(
            message="Error",
            flow_id="my_flow",
            namespace="dev",
            trigger="webhook",
        )
        s = str(error)
        assert "flow_id='my_flow'" in s
        assert "namespace='dev'" in s
        assert "trigger='webhook'" in s

    def test_to_dict(self) -> None:
        """Test to_dict."""
        error = FlowError(
            message="Flow error",
            flow_id="pipeline",
            namespace="staging",
        )
        d = error.to_dict()
        assert d["type"] == "FlowError"
        assert d["flow_id"] == "pipeline"
        assert d["namespace"] == "staging"


class TestOutputError:
    """Tests for OutputError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = OutputError(message="Output failed")
        assert error.output_name is None
        assert error.output_type is None

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields."""
        error = OutputError(
            message="Failed to serialize output",
            output_name="check_result",
            output_type="json",
        )
        assert error.output_name == "check_result"
        assert error.output_type == "json"

    def test_str_format(self) -> None:
        """Test string format."""
        error = OutputError(
            message="Error",
            output_name="result",
            output_type="file",
        )
        s = str(error)
        assert "output='result'" in s
        assert "type='file'" in s

    def test_to_dict(self) -> None:
        """Test to_dict."""
        error = OutputError(
            message="Output error",
            output_name="profile_output",
            output_type="yaml",
        )
        d = error.to_dict()
        assert d["type"] == "OutputError"
        assert d["output_name"] == "profile_output"
        assert d["output_type"] == "yaml"


class TestSLAViolationError:
    """Tests for SLAViolationError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = SLAViolationError(message="SLA violated")
        assert error.violations == []

    def test_creation_with_violations(self) -> None:
        """Test creation with violations."""
        # Using mock violations (simple dicts with to_dict)
        from dataclasses import dataclass

        @dataclass
        class MockViolation:
            type: str
            threshold: float
            actual: float

            def to_dict(self) -> dict[str, Any]:
                return {"type": self.type, "threshold": self.threshold, "actual": self.actual}

        violations = [
            MockViolation("pass_rate", 0.95, 0.90),
            MockViolation("exec_time", 60.0, 120.0),
        ]
        error = SLAViolationError(message="SLA exceeded", violations=violations)
        assert len(error.violations) == 2

    def test_str_with_violations(self) -> None:
        """Test string with violation count."""
        error = SLAViolationError(
            message="Violations found",
            violations=["v1", "v2", "v3"],  # type: ignore
        )
        s = str(error)
        assert "3 violation(s)" in s

    def test_str_without_violations(self) -> None:
        """Test string without violations."""
        error = SLAViolationError(message="No violations")
        assert str(error) == "No violations"

    def test_to_dict_with_dict_violations(self) -> None:
        """Test to_dict with dict violations."""
        violations = [
            {"type": "a", "value": 1},
            {"type": "b", "value": 2},
        ]
        # Create mock violations that don't have to_dict
        error = SLAViolationError(message="SLA error", violations=violations)  # type: ignore
        d = error.to_dict()
        assert d["violation_count"] == 2


class TestSerializationError:
    """Tests for SerializationError."""

    def test_basic_creation(self) -> None:
        """Test basic creation."""
        error = SerializationError(message="Serialization failed")
        assert error.format is None
        assert error.direction is None

    def test_creation_with_all_fields(self) -> None:
        """Test creation with all fields."""
        error = SerializationError(
            message="JSON encoding failed",
            format="json",
            direction="serialize",
        )
        assert error.format == "json"
        assert error.direction == "serialize"

    def test_str_format(self) -> None:
        """Test string format."""
        error = SerializationError(
            message="Error",
            format="yaml",
            direction="deserialize",
        )
        s = str(error)
        assert "format='yaml'" in s
        assert "direction='deserialize'" in s

    def test_to_dict(self) -> None:
        """Test to_dict."""
        error = SerializationError(
            message="Serialize error",
            format="msgpack",
            direction="serialize",
        )
        d = error.to_dict()
        assert d["type"] == "SerializationError"
        assert d["format"] == "msgpack"
        assert d["direction"] == "serialize"


class TestExceptionHierarchy:
    """Tests for exception hierarchy relationships."""

    def test_all_inherit_from_base(self) -> None:
        """Test all exceptions inherit from DataQualityError."""
        exceptions = [
            ConfigurationError(message="Config"),
            EngineError(message="Engine"),
            ScriptError(message="Script"),
            FlowError(message="Flow"),
            OutputError(message="Output"),
            SLAViolationError(message="SLA"),
            SerializationError(message="Serialization"),
        ]

        for exc in exceptions:
            assert isinstance(exc, DataQualityError)
            assert isinstance(exc, Exception)

    def test_catch_all_with_base_class(self) -> None:
        """Test catching all exceptions with base class."""
        exceptions = [
            ConfigurationError(message="Config"),
            EngineError(message="Engine"),
            ScriptError(message="Script"),
            FlowError(message="Flow"),
            OutputError(message="Output"),
            SLAViolationError(message="SLA"),
            SerializationError(message="Serialization"),
        ]

        for exc in exceptions:
            with pytest.raises(DataQualityError):
                raise exc

    def test_sibling_exceptions_not_caught(self) -> None:
        """Test that sibling exceptions don't catch each other."""
        with pytest.raises(EngineError):
            try:
                raise EngineError(message="Engine error")
            except ConfigurationError:
                pytest.fail("Should not catch sibling")


class TestExceptionChaining:
    """Tests for exception chaining patterns."""

    def test_engine_error_wraps_original(self) -> None:
        """Test EngineError wrapping original exception."""
        original = RuntimeError("Internal failure")
        error = EngineError(
            message="Engine check failed",
            engine_name="truthound",
            operation="check",
            original_error=original,
        )

        assert error.original_error is original
        d = error.to_dict()
        assert "RuntimeError" in d["original_error_type"]

    def test_exception_chain_preserves_info(self) -> None:
        """Test that exception chain preserves all info."""
        original = ValueError("Bad input")
        script_error = ScriptError(
            message="Script execution failed",
            script_name="check_script",
            task_id="task_1",
            execution_id="exec_123",
            metadata={"original_type": type(original).__name__},
        )

        d = script_error.to_dict()
        assert d["script_name"] == "check_script"
        assert d["metadata"]["original_type"] == "ValueError"
