"""Tests for utils module."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from truthound_dagster.utils.exceptions import (
    ConfigurationError,
    DataQualityError,
    EngineError,
    SLAViolationError,
)
from truthound_dagster.utils.helpers import (
    build_asset_key,
    filter_metadata,
    format_count,
    format_duration,
    format_percentage,
    format_timestamp,
    get_current_timestamp,
    merge_metadata,
    parse_timestamp,
    safe_get,
    truncate_string,
    validate_rule_format,
    validate_rules,
)
from truthound_dagster.utils.serialization import (
    ResultSerializer,
    deserialize_result,
    serialize_result,
    to_dagster_metadata,
    to_json_serializable,
)
from truthound_dagster.utils.types import (
    DataQualityOutput,
    LearnOutput,
    ProfileOutput,
    QualityCheckOutput,
)


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_milliseconds(self) -> None:
        assert format_duration(500) == "500.00ms"
        assert format_duration(1.5) == "1.50ms"

    def test_seconds(self) -> None:
        assert format_duration(1234.5) == "1.23s"
        assert format_duration(5000) == "5.00s"

    def test_minutes(self) -> None:
        assert format_duration(65000) == "1m 5.00s"
        assert format_duration(120000) == "2m 0.00s"


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_default_decimals(self) -> None:
        assert format_percentage(0.5) == "50.00%"
        assert format_percentage(0.9523) == "95.23%"

    def test_custom_decimals(self) -> None:
        assert format_percentage(0.5, decimals=0) == "50%"
        assert format_percentage(0.12345, decimals=3) == "12.345%"


class TestFormatCount:
    """Tests for format_count function."""

    def test_count_only(self) -> None:
        assert format_count(5) == "5"
        assert format_count(100) == "100"

    def test_with_total(self) -> None:
        assert format_count(5, 10) == "5/10 (50.00%)"
        assert format_count(0, 100) == "0/100 (0.00%)"

    def test_zero_total(self) -> None:
        assert format_count(0, 0) == "0/0"


class TestTimestampFunctions:
    """Tests for timestamp functions."""

    def test_get_current_timestamp(self) -> None:
        ts = get_current_timestamp()
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_format_timestamp(self) -> None:
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        formatted = format_timestamp(dt)
        assert "2024-01-15" in formatted
        assert "10:30:00" in formatted

    def test_parse_timestamp(self) -> None:
        s = "2024-01-15T10:30:00+00:00"
        dt = parse_timestamp(s)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15


class TestTruncateString:
    """Tests for truncate_string function."""

    def test_no_truncation_needed(self) -> None:
        assert truncate_string("Hello", max_length=10) == "Hello"

    def test_truncation(self) -> None:
        assert truncate_string("Hello World", max_length=8) == "Hello..."

    def test_custom_suffix(self) -> None:
        assert truncate_string("Hello World", max_length=10, suffix="…") == "Hello Wor…"


class TestMergeMetadata:
    """Tests for merge_metadata function."""

    def test_merge_multiple(self) -> None:
        result = merge_metadata({"a": 1}, {"b": 2}, {"c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_override(self) -> None:
        result = merge_metadata({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_none_handling(self) -> None:
        result = merge_metadata({"a": 1}, None, {"b": 2})
        assert result == {"a": 1, "b": 2}


class TestFilterMetadata:
    """Tests for filter_metadata function."""

    def test_include(self) -> None:
        result = filter_metadata({"a": 1, "b": 2, "c": 3}, include=["a", "b"])
        assert result == {"a": 1, "b": 2}

    def test_exclude(self) -> None:
        result = filter_metadata({"a": 1, "b": 2, "c": 3}, exclude=["c"])
        assert result == {"a": 1, "b": 2}


class TestSafeGet:
    """Tests for safe_get function."""

    def test_key_exists(self) -> None:
        assert safe_get({"a": 1}, "a") == 1

    def test_key_missing(self) -> None:
        assert safe_get({"a": 1}, "b", default=0) == 0

    def test_type_check(self) -> None:
        assert safe_get({"a": 1}, "a", expected_type=int) == 1
        assert safe_get({"a": "1"}, "a", default=0, expected_type=int) == 0


class TestBuildAssetKey:
    """Tests for build_asset_key function."""

    def test_build_key(self) -> None:
        assert build_asset_key("schema", "table") == ("schema", "table")

    def test_filter_empty(self) -> None:
        assert build_asset_key("schema", "", "table") == ("schema", "table")


class TestValidateRuleFormat:
    """Tests for validate_rule_format function."""

    def test_valid_rule(self) -> None:
        is_valid, error = validate_rule_format({"type": "not_null", "column": "id"})
        assert is_valid is True
        assert error is None

    def test_missing_type(self) -> None:
        is_valid, error = validate_rule_format({"column": "id"})
        assert is_valid is False
        assert "type" in error

    def test_missing_column(self) -> None:
        is_valid, error = validate_rule_format({"type": "not_null"})
        assert is_valid is False
        assert "column" in error


class TestValidateRules:
    """Tests for validate_rules function."""

    def test_all_valid(self) -> None:
        rules = [
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
        ]
        is_valid, errors = validate_rules(rules)
        assert is_valid is True
        assert len(errors) == 0

    def test_some_invalid(self) -> None:
        rules = [
            {"type": "not_null", "column": "id"},
            {"column": "email"},  # Missing type
        ]
        is_valid, errors = validate_rules(rules)
        assert is_valid is False
        assert len(errors) == 1


class TestExceptions:
    """Tests for exception classes."""

    def test_data_quality_error(self) -> None:
        error = DataQualityError("Test error")
        assert str(error) == "Test error"
        assert error.result is None

    def test_data_quality_error_with_result(self) -> None:
        result = {"status": "failed"}
        error = DataQualityError("Test error", result=result)
        assert error.result == result

    def test_configuration_error(self) -> None:
        error = ConfigurationError("Invalid config", field="timeout")
        assert "Invalid config" in str(error)
        assert error.field == "timeout"

    def test_engine_error(self) -> None:
        error = EngineError("Engine failed", engine_name="truthound")
        assert error.engine_name == "truthound"

    def test_sla_violation_error(self) -> None:
        violations = [{"type": "failure_rate", "message": "Too high"}]
        error = SLAViolationError("SLA violated", violations=violations)
        assert len(error.violations) == 1


class TestResultSerializer:
    """Tests for ResultSerializer class."""

    def test_serialize_check_result(self, mock_check_result) -> None:
        serializer = ResultSerializer()
        result = serializer.serialize_check_result(mock_check_result)

        assert result["type"] == "check"
        assert result["status"] == "passed"
        assert result["is_success"] is True
        assert result["passed_count"] == 10

    def test_serialize_profile_result(self, mock_profile_result) -> None:
        serializer = ResultSerializer()
        result = serializer.serialize_profile_result(mock_profile_result)

        assert result["type"] == "profile"
        assert result["row_count"] == 1000
        assert result["column_count"] == 3

    def test_serialize_learn_result(self, mock_learn_result) -> None:
        serializer = ResultSerializer()
        result = serializer.serialize_learn_result(mock_learn_result)

        assert result["type"] == "learn"
        assert result["rule_count"] == 3


class TestSerializeResult:
    """Tests for serialize_result function."""

    def test_check_result(self, mock_check_result) -> None:
        result = serialize_result(mock_check_result)
        assert result["type"] == "check"

    def test_profile_result(self, mock_profile_result) -> None:
        result = serialize_result(mock_profile_result)
        assert result["type"] == "profile"

    def test_learn_result(self, mock_learn_result) -> None:
        result = serialize_result(mock_learn_result)
        assert result["type"] == "learn"

    def test_dict_passthrough(self) -> None:
        data = {"type": "custom", "value": 42}
        result = serialize_result(data)
        assert result == data

    def test_unknown_type(self) -> None:
        with pytest.raises(TypeError):
            serialize_result("invalid")


class TestDeserializeResult:
    """Tests for deserialize_result function."""

    def test_check_result(self) -> None:
        data = {
            "type": "check",
            "status": "passed",
            "timestamp": "2024-01-15T10:30:00+00:00",
        }
        result = deserialize_result(data)
        assert isinstance(result["timestamp"], datetime)

    def test_other_types(self) -> None:
        data = {"type": "profile", "row_count": 100}
        result = deserialize_result(data)
        assert result == data


class TestToDagsterMetadata:
    """Tests for to_dagster_metadata function."""

    def test_check_metadata(self) -> None:
        data = {
            "type": "check",
            "status": "passed",
            "is_success": True,
            "passed_count": 10,
            "failed_count": 0,
            "failure_rate": 0.0,
            "execution_time_ms": 100.0,
        }
        metadata = to_dagster_metadata(data)

        assert metadata["status"] == "passed"
        assert metadata["is_success"] is True
        assert metadata["passed_count"] == 10

    def test_profile_metadata(self) -> None:
        data = {
            "type": "profile",
            "row_count": 1000,
            "column_count": 5,
            "execution_time_ms": 50.0,
        }
        metadata = to_dagster_metadata(data)

        assert metadata["row_count"] == 1000
        assert metadata["column_count"] == 5


class TestToJsonSerializable:
    """Tests for to_json_serializable function."""

    def test_primitives(self) -> None:
        assert to_json_serializable(None) is None
        assert to_json_serializable(42) == 42
        assert to_json_serializable(3.14) == 3.14
        assert to_json_serializable("hello") == "hello"
        assert to_json_serializable(True) is True

    def test_datetime(self) -> None:
        dt = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = to_json_serializable(dt)
        assert isinstance(result, str)
        assert "2024-01-15" in result

    def test_list(self) -> None:
        result = to_json_serializable([1, 2, 3])
        assert result == [1, 2, 3]

    def test_dict(self) -> None:
        result = to_json_serializable({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_set(self) -> None:
        result = to_json_serializable({1, 2, 3})
        assert set(result) == {1, 2, 3}


class TestDataQualityOutput:
    """Tests for DataQualityOutput type."""

    def test_creation(self) -> None:
        output = DataQualityOutput(data={"key": "value"})
        assert output.data == {"key": "value"}
        assert output.is_success is True

    def test_with_metadata(self) -> None:
        output = DataQualityOutput(data={}, metadata={"a": 1})
        new_output = output.with_metadata(b=2)
        assert new_output.metadata == {"a": 1, "b": 2}

    def test_to_dict(self) -> None:
        output = DataQualityOutput(data={"key": "value"}, is_success=True)
        result = output.to_dict()
        assert result["data"] == {"key": "value"}
        assert result["is_success"] is True


class TestQualityCheckOutput:
    """Tests for QualityCheckOutput type."""

    def test_from_result(self, mock_check_result) -> None:
        output = QualityCheckOutput.from_result(
            data={"id": [1, 2, 3]},
            result=mock_check_result,
        )
        assert output.passed is True
        assert output.failure_count == 0

    def test_is_success(self, mock_check_result) -> None:
        output = QualityCheckOutput(
            data={},
            result=mock_check_result,
            passed=True,
        )
        assert output.is_success is True

    def test_to_dagster_metadata(self, mock_check_result) -> None:
        output = QualityCheckOutput.from_result(data={}, result=mock_check_result)
        metadata = output.to_dagster_metadata()
        assert "status" in metadata
        assert "passed" in metadata


class TestProfileOutput:
    """Tests for ProfileOutput type."""

    def test_from_result(self, mock_profile_result) -> None:
        output = ProfileOutput.from_result(
            data={},
            result=mock_profile_result,
        )
        assert output.row_count == 1000
        assert output.column_count == 3

    def test_get_column_profile(self, mock_profile_result) -> None:
        output = ProfileOutput.from_result(data={}, result=mock_profile_result)
        profile = output.get_column_profile("id")
        assert profile is not None
        assert profile["column_name"] == "id"

    def test_get_column_profile_not_found(self, mock_profile_result) -> None:
        output = ProfileOutput.from_result(data={}, result=mock_profile_result)
        profile = output.get_column_profile("nonexistent")
        assert profile is None


class TestLearnOutput:
    """Tests for LearnOutput type."""

    def test_from_result(self, mock_learn_result) -> None:
        output = LearnOutput.from_result(
            data={},
            result=mock_learn_result,
        )
        assert output.rule_count == 3

    def test_get_rules_for_column(self, mock_learn_result) -> None:
        output = LearnOutput.from_result(data={}, result=mock_learn_result)
        rules = output.get_rules_for_column("id")
        assert len(rules) == 2

    def test_get_high_confidence_rules(self, mock_learn_result) -> None:
        output = LearnOutput.from_result(data={}, result=mock_learn_result)
        rules = output.get_high_confidence_rules(min_confidence=0.99)
        assert len(rules) == 2  # id not_null and unique have 1.0 confidence
