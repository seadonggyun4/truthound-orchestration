"""Tests for serialization utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest


class TestResultSerializer:
    """Tests for ResultSerializer class."""

    def test_serialize_check_result(self) -> None:
        """Test serializing a check result."""
        from truthound_airflow.utils.serialization import ResultSerializer

        serializer = ResultSerializer()

        # Create a mock result
        mock_result = type(
            "MockResult",
            (),
            {
                "status": type("Status", (), {"value": "passed"})(),
                "is_success": True,
                "passed_count": 10,
                "failed_count": 0,
                "warning_count": 0,
                "failures": [],
                "execution_time_ms": 100.0,
                "timestamp": datetime.now(timezone.utc),
            },
        )()

        result = serializer.serialize_check_result(mock_result)

        assert result["status"] == "passed"
        assert result["is_success"] is True
        assert result["passed_count"] == 10
        assert result["failed_count"] == 0
        assert result["_serializer"] == "truthound_airflow"
        assert result["_type"] == "check_result"

    def test_serialize_check_result_with_failures(self) -> None:
        """Test serializing a check result with failures."""
        from truthound_airflow.utils.serialization import ResultSerializer

        serializer = ResultSerializer()

        mock_failure = type(
            "MockFailure",
            (),
            {
                "rule_name": "not_null",
                "column": "id",
                "message": "Found null values",
                "severity": type("Severity", (), {"value": "error"})(),
                "failed_count": 5,
                "total_count": 100,
            },
        )()

        mock_result = type(
            "MockResult",
            (),
            {
                "status": type("Status", (), {"value": "failed"})(),
                "is_success": False,
                "passed_count": 95,
                "failed_count": 5,
                "warning_count": 0,
                "failures": [mock_failure],
                "execution_time_ms": 150.0,
                "timestamp": datetime.now(timezone.utc),
            },
        )()

        result = serializer.serialize_check_result(mock_result)

        assert result["is_success"] is False
        assert len(result["failures"]) == 1
        assert result["failures"][0]["column"] == "id"

    def test_deserialize_check_result(self) -> None:
        """Test deserializing a check result."""
        from truthound_airflow.utils.serialization import ResultSerializer

        serializer = ResultSerializer()

        data = {
            "status": "passed",
            "is_success": True,
            "passed_count": 10,
            "failed_count": 0,
            "warning_count": 0,
            "failures": [],
            "execution_time_ms": 100.0,
            "timestamp": "2024-01-01T00:00:00Z",
            "_serializer": "truthound_airflow",
            "_type": "check_result",
        }

        result = serializer.deserialize_check_result(data)

        assert result["status"] == "passed"
        assert result["is_success"] is True
        # Metadata should be removed
        assert "_serializer" not in result
        assert "_type" not in result


class TestSerializeResult:
    """Tests for serialize_result function."""

    def test_serialize_check_result_object(self) -> None:
        """Test serializing a check result object."""
        from truthound_airflow.utils.serialization import serialize_result

        # Create a mock result object
        mock_result = type(
            "MockResult",
            (),
            {
                "status": type("Status", (), {"value": "passed"})(),
                "is_success": True,
                "passed_count": 10,
                "failed_count": 0,
                "warning_count": 0,
                "skipped_count": 0,
                "failures": [],
                "execution_time_ms": 100.0,
                "timestamp": datetime.now(timezone.utc),
            },
        )()

        result = serialize_result(mock_result)

        assert result["status"] == "passed"
        assert result["_serializer"] == "truthound_airflow"

    def test_serialize_object_with_to_dict(self) -> None:
        """Test serializing an object with to_dict method."""
        from truthound_airflow.utils.serialization import serialize_result

        mock_obj = type(
            "MockObj",
            (),
            {
                "to_dict": lambda self: {"key": "value", "status": type("Status", (), {"value": "passed"})()},
                "status": type("Status", (), {"value": "passed"})(),
                "is_success": True,
                "passed_count": 0,
                "failed_count": 0,
            },
        )()

        result = serialize_result(mock_obj)

        assert result["key"] == "value"
        assert result["_serializer"] == "truthound_airflow"


class TestToXcomValue:
    """Tests for to_xcom_value function."""

    def test_primitive_types(self) -> None:
        """Test primitive types are unchanged."""
        from truthound_airflow.utils.serialization import to_xcom_value

        assert to_xcom_value(42) == 42
        assert to_xcom_value("hello") == "hello"
        assert to_xcom_value(3.14) == 3.14
        assert to_xcom_value(True) is True
        assert to_xcom_value(None) is None

    def test_datetime_converted(self) -> None:
        """Test datetime is converted to ISO format."""
        from truthound_airflow.utils.serialization import to_xcom_value

        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        result = to_xcom_value(dt)

        assert result == "2024-01-01T12:00:00+00:00"

    def test_list_recursively_converted(self) -> None:
        """Test list elements are recursively converted."""
        from truthound_airflow.utils.serialization import to_xcom_value

        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = [1, "hello", dt]

        result = to_xcom_value(data)

        assert result[0] == 1
        assert result[1] == "hello"
        assert result[2] == "2024-01-01T12:00:00+00:00"

    def test_dict_recursively_converted(self) -> None:
        """Test dict values are recursively converted."""
        from truthound_airflow.utils.serialization import to_xcom_value

        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        data = {"timestamp": dt, "count": 10}

        result = to_xcom_value(data)

        assert result["timestamp"] == "2024-01-01T12:00:00+00:00"
        assert result["count"] == 10

    def test_frozenset_converted_to_string(self) -> None:
        """Test frozenset is converted to string representation."""
        from truthound_airflow.utils.serialization import to_xcom_value

        data = frozenset(["a", "b", "c"])

        result = to_xcom_value(data)

        # frozenset is converted to string by str()
        assert isinstance(result, str)
        assert "frozenset" in result or "a" in result


class TestFromXcomValue:
    """Tests for from_xcom_value function."""

    def test_primitive_types(self) -> None:
        """Test primitive types are unchanged."""
        from truthound_airflow.utils.serialization import from_xcom_value

        assert from_xcom_value(42) == 42
        assert from_xcom_value("hello") == "hello"
        assert from_xcom_value(None) is None

    def test_check_result_type_hint(self) -> None:
        """Test check_result type hint processes data."""
        from truthound_airflow.utils.serialization import from_xcom_value

        data = {
            "status": "passed",
            "is_success": True,
            "passed_count": 10,
            "_type": "check_result",
            "_serializer": "truthound_airflow",
        }

        result = from_xcom_value(data)

        assert result["status"] == "passed"
        assert "_type" not in result  # Metadata removed

    def test_dict_type_hint(self) -> None:
        """Test dict type hint returns dict."""
        from truthound_airflow.utils.serialization import from_xcom_value

        data = {"key": "value"}

        result = from_xcom_value(data, type_hint="dict")

        assert result == data


class TestXcomCompatibility:
    """Tests for XCom compatibility requirements."""

    def test_result_is_json_serializable(self) -> None:
        """Test that serialized result is JSON serializable."""
        import json

        from truthound_airflow.utils.serialization import to_xcom_value

        data = {
            "status": "passed",
            "timestamp": datetime.now(timezone.utc),
            "values": [1, 2, 3],
            "nested": {"key": "value"},
        }

        result = to_xcom_value(data)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_roundtrip_conversion(self) -> None:
        """Test that data survives a roundtrip conversion."""
        from truthound_airflow.utils.serialization import from_xcom_value, to_xcom_value

        original = {
            "count": 10,
            "name": "test",
            "values": [1, 2, 3],
        }

        serialized = to_xcom_value(original)
        deserialized = from_xcom_value(serialized)

        assert deserialized == original
