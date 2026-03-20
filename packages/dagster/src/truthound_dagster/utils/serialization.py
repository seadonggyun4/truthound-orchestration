"""Serialization utilities for Dagster integration."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from common.serializers import detect_result_type, serialize_result_wire


class ResultSerializer:
    """Serializer for data quality results."""

    def serialize_check_result(self, result: Any) -> dict[str, Any]:
        return serialize_result(result)

    def serialize_profile_result(self, result: Any) -> dict[str, Any]:
        return serialize_result(result)

    def serialize_learn_result(self, result: Any) -> dict[str, Any]:
        return serialize_result(result)

    def deserialize_check_result(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        if isinstance(result.get("timestamp"), str):
            result["timestamp"] = datetime.fromisoformat(result["timestamp"])
        return result


_serializer = ResultSerializer()


def _enum_name(value: Any, *, default: str) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    if hasattr(value, "value"):
        return str(value.value).upper()
    if value is None:
        return default
    return str(value)


def _timestamp_value(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)


def _serialize_duck_typed_result(result: Any) -> dict[str, Any]:
    if hasattr(result, "failures"):
        failures = []
        for failure in getattr(result, "failures", []):
            failures.append(
                {
                    "rule_name": getattr(failure, "rule_name", "unknown"),
                    "column": getattr(failure, "column", None),
                    "message": getattr(failure, "message", ""),
                    "severity": _enum_name(getattr(failure, "severity", None), default="ERROR"),
                    "failed_count": getattr(failure, "failed_count", 0),
                    "total_count": getattr(failure, "total_count", 0),
                }
            )

        return {
            "type": "check",
            "result_type": "check",
            "status": _enum_name(getattr(result, "status", None), default="UNKNOWN"),
            "is_success": getattr(result, "is_success", False),
            "passed_count": getattr(result, "passed_count", 0),
            "failed_count": getattr(result, "failed_count", 0),
            "warning_count": getattr(result, "warning_count", 0),
            "skipped_count": getattr(result, "skipped_count", 0),
            "failure_rate": getattr(result, "failure_rate", 0.0),
            "failures": failures,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "timestamp": _timestamp_value(getattr(result, "timestamp", None)),
        }

    if hasattr(result, "columns") and hasattr(result, "row_count"):
        columns = []
        for column in getattr(result, "columns", []):
            columns.append(
                {
                    "column_name": getattr(column, "column_name", "unknown"),
                    "dtype": str(getattr(column, "dtype", "unknown")),
                    "null_count": getattr(column, "null_count", 0),
                    "null_percentage": getattr(column, "null_percentage", 0.0),
                    "unique_count": getattr(column, "unique_count", 0),
                    "unique_percentage": getattr(column, "unique_percentage", 0.0),
                }
            )

        return {
            "type": "profile",
            "result_type": "profile",
            "row_count": getattr(result, "row_count", 0),
            "column_count": getattr(result, "column_count", len(columns)),
            "columns": columns,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "timestamp": _timestamp_value(getattr(result, "timestamp", None)),
        }

    if hasattr(result, "rules"):
        rules = []
        for rule in getattr(result, "rules", []):
            rules.append(
                {
                    "column": getattr(rule, "column", None),
                    "rule_type": getattr(rule, "rule_type", "unknown"),
                    "confidence": getattr(rule, "confidence", 0.0),
                    "parameters": getattr(rule, "parameters", {}),
                }
            )

        return {
            "type": "learn",
            "result_type": "learn",
            "rule_count": len(rules),
            "rules": rules,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "timestamp": _timestamp_value(getattr(result, "timestamp", None)),
        }

    raise TypeError(f"Unknown result type: {type(result).__name__}")


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize any result type to dictionary."""

    if isinstance(result, dict):
        return dict(result)

    if hasattr(result, "to_dict"):
        payload = serialize_result_wire(result, include_result_type=True)
        payload["type"] = detect_result_type(result)
        return payload

    return _serialize_duck_typed_result(result)


def deserialize_result(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a result dictionary."""

    if (data.get("type") or data.get("result_type")) == "check":
        return _serializer.deserialize_check_result(data)
    return dict(data)


def to_dagster_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Convert shared wire data to compact Dagster metadata."""

    metadata: dict[str, Any] = {}
    result_type = result.get("type") or result.get("result_type") or detect_result_type(result)

    if result_type == "check":
        metadata["status"] = result.get("status", "UNKNOWN")
        metadata["is_success"] = result.get("is_success", False)
        metadata["passed_count"] = result.get("passed_count", 0)
        metadata["failed_count"] = result.get("failed_count", 0)
        metadata["failure_rate"] = result.get("failure_rate", 0.0)
        metadata["execution_time_ms"] = result.get("execution_time_ms", 0.0)
    elif result_type == "profile":
        metadata["row_count"] = result.get("row_count", 0)
        metadata["column_count"] = result.get("column_count", 0)
        metadata["execution_time_ms"] = result.get("execution_time_ms", 0.0)
    elif result_type == "learn":
        metadata["rule_count"] = len(result.get("rules", []))
        metadata["execution_time_ms"] = result.get("execution_time_ms", 0.0)

    return metadata


def to_json_serializable(obj: Any) -> Any:
    """Convert nested objects to JSON-serializable values."""

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    if isinstance(obj, set):
        return [to_json_serializable(item) for item in sorted(obj, key=repr)]
    if isinstance(obj, dict):
        return {
            str(key): to_json_serializable(value)
            for key, value in obj.items()
        }
    if hasattr(obj, "to_dict"):
        return to_json_serializable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return to_json_serializable(obj.__dict__)
    return str(obj)


__all__ = [
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "to_dagster_metadata",
    "to_json_serializable",
]
