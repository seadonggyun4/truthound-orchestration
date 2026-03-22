"""Tests for truthound_kestra.utils module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from truthound_kestra.utils import (
    # Exceptions
    DataQualityError,
    ConfigurationError,
    EngineError,
    ScriptError,
    FlowError,
    OutputError,
    SLAViolationError,
    SerializationError,
    # Types
    CheckStatus,
    Severity,
    OperationType,
    OutputFormat,
    DataSourceType,
    ValidationFailure,
    ColumnProfile,
    LearnedRule,
    ScriptOutput,
    ExecutionContext,
    # Serialization
    ResultSerializer,
    JsonSerializer,
    YamlSerializer,
    MarkdownSerializer,
    SerializerConfig,
    DEFAULT_SERIALIZER_CONFIG,
    COMPACT_SERIALIZER_CONFIG,
    FULL_SERIALIZER_CONFIG,
    serialize_result,
    deserialize_result,
    serialize_to_format,
    get_serializer,
    # Helpers
    Timer,
    timed,
    get_logger,
    format_duration,
    format_percentage,
    format_count,
    format_status_badge,
    detect_data_source_type,
    parse_uri,
    validate_rules,
    merge_rules,
)


class TestExceptions:
    """Tests for exception classes."""

    def test_data_quality_error_basic(self) -> None:
        """Test basic DataQualityError creation."""
        error = DataQualityError(message="Test error")
        assert "Test error" in str(error)
        assert error.message == "Test error"
        assert error.result is None
        assert error.metadata == {}

    def test_data_quality_error_with_details(self) -> None:
        """Test DataQualityError with result and metadata."""
        error = DataQualityError(
            message="Test error",
            result={"status": "failed"},
            metadata={"key": "value"},
        )
        assert error.result == {"status": "failed"}
        assert error.metadata == {"key": "value"}

    def test_data_quality_error_to_dict(self) -> None:
        """Test DataQualityError serialization."""
        error = DataQualityError(
            message="Test error",
            result={"status": "failed"},
            metadata={"key": "value"},
        )
        d = error.to_dict()
        assert d["message"] == "Test error"
        assert d["result"] == {"status": "failed"}
        assert d["metadata"] == {"key": "value"}
        assert d["type"] == "DataQualityError"

    def test_data_quality_error_from_dict(self) -> None:
        """Test DataQualityError deserialization."""
        data = {
            "message": "Test error",
            "result": {"status": "failed"},
            "metadata": {"key": "value"},
        }
        error = DataQualityError.from_dict(data)
        assert error.message == "Test error"
        assert error.result == {"status": "failed"}

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError(
            message="Invalid config",
            field="timeout",
            reason="Must be positive",
        )
        assert error.field == "timeout"
        d = error.to_dict()
        assert d["field"] == "timeout"

    def test_engine_error(self) -> None:
        """Test EngineError."""
        error = EngineError(
            message="Engine failed",
            engine_name="truthound",
            operation="check",
        )
        assert error.engine_name == "truthound"
        assert error.operation == "check"

    def test_script_error(self) -> None:
        """Test ScriptError."""
        error = ScriptError(
            message="Script failed",
            script_name="check_quality",
            task_id="validate_users",
        )
        assert error.script_name == "check_quality"
        assert error.task_id == "validate_users"

    def test_flow_error(self) -> None:
        """Test FlowError."""
        error = FlowError(
            message="Flow failed",
            flow_id="my_flow",
            namespace="production",
        )
        assert error.flow_id == "my_flow"
        assert error.namespace == "production"

    def test_sla_violation_error(self) -> None:
        """Test SLAViolationError."""
        error = SLAViolationError(
            message="SLA violated",
            violations=[],
        )
        assert error.violations == []
        d = error.to_dict()
        assert "violations" in d

    def test_exception_inheritance(self) -> None:
        """Test exception hierarchy."""
        assert issubclass(ConfigurationError, DataQualityError)
        assert issubclass(EngineError, DataQualityError)
        assert issubclass(ScriptError, DataQualityError)
        assert issubclass(FlowError, DataQualityError)
        assert issubclass(SLAViolationError, DataQualityError)
        assert issubclass(SerializationError, DataQualityError)


class TestTypes:
    """Tests for type classes and enums."""

    def test_check_status_enum(self) -> None:
        """Test CheckStatus enum values."""
        assert CheckStatus.PASSED.value == "passed"
        assert CheckStatus.FAILED.value == "failed"
        assert CheckStatus.WARNING.value == "warning"
        assert CheckStatus.SKIPPED.value == "skipped"
        assert CheckStatus.ERROR.value == "error"

    def test_check_status_is_success(self) -> None:
        """Test CheckStatus.is_success method."""
        assert CheckStatus.PASSED.is_success() is True
        assert CheckStatus.WARNING.is_success() is True
        assert CheckStatus.FAILED.is_success() is False
        assert CheckStatus.ERROR.is_success() is False

    def test_severity_enum(self) -> None:
        """Test Severity enum values."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_severity_comparison(self) -> None:
        """Test Severity comparison using __lt__."""
        # CRITICAL > HIGH > MEDIUM > LOW > INFO
        assert Severity.INFO < Severity.LOW
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL

    def test_operation_type_enum(self) -> None:
        """Test OperationType enum values."""
        assert OperationType.CHECK.value == "check"
        assert OperationType.PROFILE.value == "profile"
        assert OperationType.LEARN.value == "learn"

    def test_output_format_enum(self) -> None:
        """Test OutputFormat enum values."""
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.YAML.value == "yaml"
        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_data_source_type_enum(self) -> None:
        """Test DataSourceType enum values."""
        assert DataSourceType.FILE.value == "file"
        assert DataSourceType.URI.value == "uri"
        assert DataSourceType.INLINE.value == "inline"
        assert DataSourceType.OUTPUT.value == "output"
        assert DataSourceType.SECRET.value == "secret"

    def test_validation_failure_creation(self) -> None:
        """Test ValidationFailure creation."""
        failure = ValidationFailure(
            column="id",
            rule_type="not_null",
            message="Found null values",
            severity=Severity.HIGH,
            failed_count=5,
        )
        assert failure.column == "id"
        assert failure.rule_type == "not_null"
        assert failure.severity == Severity.HIGH
        assert failure.failed_count == 5

    def test_validation_failure_to_dict(self) -> None:
        """Test ValidationFailure serialization."""
        failure = ValidationFailure(
            column="id",
            rule_type="not_null",
            message="Found null values",
        )
        d = failure.to_dict()
        assert d["column"] == "id"
        assert d["rule_type"] == "not_null"
        assert "severity" in d

    def test_validation_failure_from_dict(self) -> None:
        """Test ValidationFailure deserialization."""
        data = {
            "column": "id",
            "rule_type": "not_null",
            "message": "Found null values",
            "severity": "high",
            "failed_count": 5,
        }
        failure = ValidationFailure.from_dict(data)
        assert failure.column == "id"
        assert failure.rule_type == "not_null"
        assert failure.severity == Severity.HIGH

    def test_column_profile_creation(self) -> None:
        """Test ColumnProfile creation."""
        profile = ColumnProfile(
            column_name="age",
            dtype="int64",
            null_count=0,
            null_percentage=0.0,
            unique_count=50,
            min_value=18,
            max_value=100,
        )
        assert profile.column_name == "age"
        assert profile.dtype == "int64"
        assert profile.min_value == 18
        assert profile.max_value == 100

    def test_column_profile_to_dict(self) -> None:
        """Test ColumnProfile serialization."""
        profile = ColumnProfile(
            column_name="age",
            dtype="int64",
        )
        d = profile.to_dict()
        assert d["column_name"] == "age"
        assert d["dtype"] == "int64"

    def test_learned_rule_creation(self) -> None:
        """Test LearnedRule creation."""
        rule = LearnedRule(
            column="email",
            rule_type="unique",
            confidence=0.99,
            parameters={"strict": True},
        )
        assert rule.column == "email"
        assert rule.rule_type == "unique"
        assert rule.confidence == 0.99

    def test_learned_rule_to_rule_dict(self) -> None:
        """Test LearnedRule to_rule_dict."""
        rule = LearnedRule(
            column="age",
            rule_type="in_range",
            parameters={"min": 0, "max": 150},
        )
        rule_dict = rule.to_rule_dict()
        assert rule_dict["type"] == "in_range"
        assert rule_dict["column"] == "age"
        assert rule_dict["min"] == 0
        assert rule_dict["max"] == 150

    def test_script_output_creation(self) -> None:
        """Test ScriptOutput creation."""
        output = ScriptOutput(
            status=CheckStatus.PASSED,
            operation=OperationType.CHECK,
            execution_time_ms=150.5,
            passed_count=10,
            failed_count=0,
        )
        assert output.status == CheckStatus.PASSED
        assert output.operation == OperationType.CHECK
        assert output.execution_time_ms == 150.5
        assert output.passed_count == 10

    def test_script_output_is_success(self) -> None:
        """Test ScriptOutput is_success property."""
        passed = ScriptOutput(status=CheckStatus.PASSED)
        failed = ScriptOutput(status=CheckStatus.FAILED)
        assert passed.is_success is True
        assert failed.is_success is False

    def test_script_output_pass_rate(self) -> None:
        """Test ScriptOutput pass_rate property."""
        output = ScriptOutput(
            status=CheckStatus.PASSED,
            passed_count=90,
            failed_count=10,
        )
        assert output.pass_rate == pytest.approx(0.9)

    def test_execution_context_creation(self) -> None:
        """Test ExecutionContext creation."""
        context = ExecutionContext(
            execution_id="abc123",
            flow_id="my_flow",
            namespace="production",
            task_id="check_task",
        )
        assert context.execution_id == "abc123"
        assert context.flow_id == "my_flow"
        assert context.namespace == "production"
        assert context.task_id == "check_task"

    def test_execution_context_to_dict(self) -> None:
        """Test ExecutionContext serialization."""
        context = ExecutionContext(
            execution_id="abc123",
            flow_id="my_flow",
        )
        d = context.to_dict()
        assert d["execution_id"] == "abc123"
        assert d["flow_id"] == "my_flow"


class TestSerialization:
    """Tests for serialization classes."""

    def test_serializer_config_creation(self) -> None:
        """Test SerializerConfig creation."""
        config = SerializerConfig(
            include_metadata=True,
            compact=False,
            max_failures=100,
        )
        assert config.include_metadata is True
        assert config.compact is False
        assert config.max_failures == 100

    def test_serializer_config_builder(self) -> None:
        """Test SerializerConfig builder pattern."""
        config = SerializerConfig()
        config = config.with_compact(True)
        config = config.with_max_failures(50)

        assert config.compact is True
        assert config.max_failures == 50

    def test_json_serializer(self) -> None:
        """Test JsonSerializer."""
        serializer = JsonSerializer()
        data = {"status": "passed", "count": 10}
        result = serializer.serialize(data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["status"] == "passed"
        assert parsed["count"] == 10

    def test_json_serializer_deserialize(self) -> None:
        """Test JsonSerializer deserialization."""
        serializer = JsonSerializer()
        json_str = '{"status": "passed", "count": 10}'
        result = serializer.deserialize(json_str)

        assert result["status"] == "passed"
        assert result["count"] == 10

    def test_yaml_serializer(self) -> None:
        """Test YamlSerializer."""
        serializer = YamlSerializer()
        data = {"status": "passed", "count": 10}
        result = serializer.serialize(data)

        assert isinstance(result, str)
        assert "status:" in result or "status :" in result

    def test_markdown_serializer(self) -> None:
        """Test MarkdownSerializer."""
        serializer = MarkdownSerializer()
        data = {
            "status": "passed",
            "passed_count": 10,
            "failed_count": 0,
        }
        result = serializer.serialize(data)

        assert isinstance(result, str)
        assert "passed" in result.lower()

    def test_result_serializer_check(self) -> None:
        """Test ResultSerializer serialize_check_result."""
        serializer = ResultSerializer()
        data = {"status": "passed", "passed_count": 10}
        result = serializer.serialize_check_result(data)

        assert isinstance(result, dict)
        assert result["status"] == "passed"
        assert result["passed_count"] == 10

    def test_result_serializer_with_config(self) -> None:
        """Test ResultSerializer with custom config."""
        config = SerializerConfig(include_metadata=False)
        serializer = ResultSerializer(config)
        data = {"status": "passed"}
        result = serializer.serialize_check_result(data)

        assert "_serializer" not in result

    def test_serialize_result_function(self) -> None:
        """Test serialize_result convenience function."""
        data = {"status": "passed", "count": 10}
        result = serialize_result(data)

        assert isinstance(result, dict)
        assert result["status"] == "passed"

    def test_deserialize_result_function(self) -> None:
        """Test deserialize_result convenience function."""
        data = {"status": "passed", "count": 10, "_serializer": "truthound_kestra"}
        result = deserialize_result(data)

        assert result["status"] == "passed"
        assert "_serializer" not in result

    def test_get_serializer_function(self) -> None:
        """Test get_serializer factory function."""
        json_ser = get_serializer(OutputFormat.JSON)
        yaml_ser = get_serializer(OutputFormat.YAML)
        md_ser = get_serializer(OutputFormat.MARKDOWN)

        assert isinstance(json_ser, JsonSerializer)
        assert isinstance(yaml_ser, YamlSerializer)
        assert isinstance(md_ser, MarkdownSerializer)

    def test_preset_configs(self) -> None:
        """Test preset serializer configurations."""
        assert DEFAULT_SERIALIZER_CONFIG.include_metadata is True
        assert COMPACT_SERIALIZER_CONFIG.compact is True
        assert FULL_SERIALIZER_CONFIG.max_failures == 0


class TestHelpers:
    """Tests for helper functions."""

    def test_timer_context_manager(self) -> None:
        """Test Timer context manager."""
        with Timer("test") as timer:
            # Simulate some work
            _ = sum(range(1000))

        assert timer.elapsed_ms > 0

    def test_timer_not_started(self) -> None:
        """Test Timer before starting."""
        timer = Timer()
        assert timer.elapsed_ms == 0.0

    def test_timer_elapsed_formatted(self) -> None:
        """Test Timer elapsed_formatted property."""
        with Timer("test") as timer:
            _ = sum(range(1000))

        formatted = timer.elapsed_formatted
        assert isinstance(formatted, str)

    def test_format_duration_milliseconds(self) -> None:
        """Test format_duration with milliseconds."""
        result = format_duration(100.0)
        assert "ms" in result

    def test_format_duration_seconds(self) -> None:
        """Test format_duration with seconds."""
        result = format_duration(1500.0)  # 1.5 seconds
        assert "s" in result

    def test_format_duration_minutes(self) -> None:
        """Test format_duration with minutes."""
        result = format_duration(90000.0)  # 1.5 minutes
        assert "m" in result

    def test_format_percentage(self) -> None:
        """Test format_percentage."""
        assert "95" in format_percentage(0.95)
        assert "100" in format_percentage(1.0)
        assert "%" in format_percentage(0.5)

    def test_format_count(self) -> None:
        """Test format_count."""
        assert "100" in format_count(100)
        assert "," in format_count(1000)  # Thousand separator

    def test_format_status_badge_passed(self) -> None:
        """Test format_status_badge for passed status."""
        result = format_status_badge(CheckStatus.PASSED)
        assert isinstance(result, str)
        assert "PASSED" in result

    def test_format_status_badge_failed(self) -> None:
        """Test format_status_badge for failed status."""
        result = format_status_badge(CheckStatus.FAILED)
        assert isinstance(result, str)
        assert "FAILED" in result

    def test_detect_data_source_type_file(self) -> None:
        """Test detect_data_source_type for local files."""
        assert detect_data_source_type("/path/to/file.csv") == DataSourceType.FILE
        assert detect_data_source_type("./data.parquet") == DataSourceType.FILE

    def test_detect_data_source_type_uri(self) -> None:
        """Test detect_data_source_type for URI schemes."""
        assert detect_data_source_type("s3://bucket/key") == DataSourceType.URI
        assert detect_data_source_type("gs://bucket/key") == DataSourceType.URI
        assert detect_data_source_type("https://example.com/data.csv") == DataSourceType.URI

    def test_detect_data_source_type_output(self) -> None:
        """Test detect_data_source_type for Kestra output reference."""
        assert detect_data_source_type("{{ outputs.task.uri }}") == DataSourceType.OUTPUT

    def test_detect_data_source_type_secret(self) -> None:
        """Test detect_data_source_type for secret reference."""
        assert detect_data_source_type("secret://my-secret") == DataSourceType.SECRET

    def test_detect_data_source_type_inline(self) -> None:
        """Test detect_data_source_type for inline data."""
        # Anything that doesn't match other patterns is INLINE
        assert detect_data_source_type("some_data") == DataSourceType.INLINE

    def test_parse_uri_s3(self) -> None:
        """Test parse_uri for S3 URIs."""
        result = parse_uri("s3://my-bucket/path/to/file.csv")
        assert result["scheme"] == "s3"
        assert result["bucket"] == "my-bucket"
        assert result["key"] == "path/to/file.csv"

    def test_parse_uri_https(self) -> None:
        """Test parse_uri for HTTPS URIs."""
        result = parse_uri("https://example.com/data/file.csv")
        assert result["scheme"] == "https"
        assert result["bucket"] == "example.com"

    def test_validate_rules_valid(self) -> None:
        """Test validate_rules with valid rules."""
        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
        ]
        errors = validate_rules(rules)
        assert errors == []

    def test_validate_rules_missing_type(self) -> None:
        """Test validate_rules with missing type."""
        rules = [{"column": "id"}]  # Missing type
        errors = validate_rules(rules)
        assert len(errors) > 0
        assert "type" in errors[0].lower()

    def test_validate_rules_missing_column(self) -> None:
        """Test validate_rules with missing column."""
        rules = [{"type": "not_null"}]  # Missing column
        errors = validate_rules(rules)
        assert len(errors) > 0
        assert "column" in errors[0].lower()

    def test_validate_rules_invalid_type(self) -> None:
        """Test validate_rules with non-dict rule."""
        rules = ["not_a_dict"]
        errors = validate_rules(rules)
        assert len(errors) > 0

    def test_merge_rules(self) -> None:
        """Test merge_rules function."""
        base_rules = [{"type": "not_null", "column": "id"}]
        override_rules = [{"type": "unique", "column": "email"}]

        result = merge_rules(base_rules, override_rules)
        assert len(result) == 2

    def test_merge_rules_deduplication(self) -> None:
        """Test merge_rules with duplicate rules."""
        base_rules = [{"type": "not_null", "column": "id"}]
        override_rules = [{"type": "not_null", "column": "id"}]  # Duplicate

        result = merge_rules(base_rules, override_rules, deduplicate=True)
        assert len(result) == 1

    def test_merge_rules_none_handling(self) -> None:
        """Test merge_rules with None values."""
        base_rules = [{"type": "not_null", "column": "id"}]
        result = merge_rules(base_rules, None)
        assert len(result) == 1

    def test_get_logger(self) -> None:
        """Test get_logger function."""
        logger = get_logger(__name__)
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_get_logger_default_name(self) -> None:
        """Test get_logger with default name."""
        logger = get_logger()
        assert logger is not None
        assert logger.name == "truthound_kestra"

    def test_timed_decorator(self) -> None:
        """Test timed decorator."""
        @timed("test_operation")
        def sample_function() -> int:
            return sum(range(100))

        result = sample_function()
        assert result == sum(range(100))
