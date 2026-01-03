"""Serialization utilities for Mage Data Quality blocks.

This module provides functions to serialize and deserialize
data quality results for storage, transmission, and logging.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, TypeVar

from common.base import CheckResult, CheckStatus, ProfileResult, LearnResult

T = TypeVar("T")


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize any result type to dictionary.

    Automatically detects the result type and calls the appropriate
    serialization function.

    Args:
        result: A CheckResult, ProfileResult, LearnResult, or dict.

    Returns:
        Dictionary representation of the result.

    Example:
        >>> result = engine.check(data, rules)
        >>> data = serialize_result(result)
        >>> json.dumps(data)
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, CheckResult):
        return serialize_check_result(result)
    if isinstance(result, ProfileResult):
        return serialize_profile_result(result)
    if isinstance(result, LearnResult):
        return serialize_learn_result(result)
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "__dict__"):
        return _serialize_object(result)
    return {"value": result}


def deserialize_result(
    data: dict[str, Any],
    result_type: str | None = None,
) -> Any:
    """Deserialize dictionary to result object.

    Args:
        data: Dictionary representation.
        result_type: Type hint ("check", "profile", "learn").

    Returns:
        Deserialized result object.

    Example:
        >>> data = {"_type": "CheckResult", "status": "PASSED", ...}
        >>> result = deserialize_result(data)
    """
    # Infer type from data if not provided
    if result_type is None:
        result_type = data.get("_type", "").lower()
        if not result_type:
            if "status" in data and "passed_count" in data:
                result_type = "check"
            elif "columns" in data:
                result_type = "profile"
            elif "rules" in data:
                result_type = "learn"

    if result_type in ("check", "checkresult"):
        return deserialize_check_result(data)
    if result_type in ("profile", "profileresult"):
        return deserialize_profile_result(data)
    if result_type in ("learn", "learnresult"):
        return deserialize_learn_result(data)

    return data


def serialize_check_result(result: CheckResult) -> dict[str, Any]:
    """Serialize CheckResult to dictionary.

    Args:
        result: CheckResult to serialize.

    Returns:
        Dictionary representation.

    Example:
        >>> result = CheckResult(status=CheckStatus.PASSED, ...)
        >>> data = serialize_check_result(result)
    """
    data: dict[str, Any] = {
        "_type": "CheckResult",
        "status": result.status.name if hasattr(result.status, "name") else str(result.status),
        "passed_count": result.passed_count,
        "failed_count": result.failed_count,
        "warning_count": getattr(result, "warning_count", 0),
        "skipped_count": getattr(result, "skipped_count", 0),
        "total_count": getattr(result, "total_count", result.passed_count + result.failed_count),
    }

    # Serialize failures
    if hasattr(result, "failures") and result.failures:
        data["failures"] = [
            _serialize_failure(f) for f in result.failures
        ]

    # Serialize metadata
    if hasattr(result, "metadata") and result.metadata:
        data["metadata"] = _serialize_metadata(result.metadata)

    # Serialize execution time
    if hasattr(result, "execution_time_ms"):
        data["execution_time_ms"] = result.execution_time_ms

    # Serialize timestamp
    if hasattr(result, "timestamp"):
        data["timestamp"] = (
            result.timestamp.isoformat()
            if isinstance(result.timestamp, datetime)
            else str(result.timestamp)
        )

    return data


def deserialize_check_result(data: dict[str, Any]) -> CheckResult:
    """Deserialize dictionary to CheckResult.

    Args:
        data: Dictionary representation.

    Returns:
        CheckResult instance.
    """
    status_str = data.get("status", "PASSED")
    try:
        status = CheckStatus[status_str] if isinstance(status_str, str) else status_str
    except KeyError:
        status = CheckStatus.PASSED

    # Build kwargs for CheckResult
    kwargs: dict[str, Any] = {
        "status": status,
        "passed_count": data.get("passed_count", 0),
        "failed_count": data.get("failed_count", 0),
    }

    # Add optional fields if present
    if "failures" in data:
        kwargs["failures"] = tuple(data["failures"])
    if "metadata" in data:
        kwargs["metadata"] = data["metadata"]
    if "execution_time_ms" in data:
        kwargs["execution_time_ms"] = data["execution_time_ms"]

    return CheckResult(**kwargs)


def serialize_profile_result(result: ProfileResult) -> dict[str, Any]:
    """Serialize ProfileResult to dictionary.

    Args:
        result: ProfileResult to serialize.

    Returns:
        Dictionary representation.
    """
    data: dict[str, Any] = {
        "_type": "ProfileResult",
        "row_count": getattr(result, "row_count", 0),
        "column_count": getattr(result, "column_count", 0),
    }

    # Serialize columns
    if hasattr(result, "columns") and result.columns:
        data["columns"] = [
            _serialize_column_profile(c) for c in result.columns
        ]

    # Serialize metadata
    if hasattr(result, "metadata") and result.metadata:
        data["metadata"] = _serialize_metadata(result.metadata)

    # Serialize execution time
    if hasattr(result, "execution_time_ms"):
        data["execution_time_ms"] = result.execution_time_ms

    return data


def deserialize_profile_result(data: dict[str, Any]) -> ProfileResult:
    """Deserialize dictionary to ProfileResult.

    Args:
        data: Dictionary representation.

    Returns:
        ProfileResult instance.
    """
    kwargs: dict[str, Any] = {
        "row_count": data.get("row_count", 0),
        "column_count": data.get("column_count", 0),
    }

    if "columns" in data:
        kwargs["columns"] = tuple(data["columns"])
    if "metadata" in data:
        kwargs["metadata"] = data["metadata"]

    return ProfileResult(**kwargs)


def serialize_learn_result(result: LearnResult) -> dict[str, Any]:
    """Serialize LearnResult to dictionary.

    Args:
        result: LearnResult to serialize.

    Returns:
        Dictionary representation.
    """
    data: dict[str, Any] = {
        "_type": "LearnResult",
    }

    # Serialize rules
    if hasattr(result, "rules") and result.rules:
        data["rules"] = [
            _serialize_learned_rule(r) for r in result.rules
        ]

    # Serialize schema
    if hasattr(result, "schema") and result.schema:
        data["schema"] = _serialize_schema(result.schema)

    # Serialize metadata
    if hasattr(result, "metadata") and result.metadata:
        data["metadata"] = _serialize_metadata(result.metadata)

    # Serialize execution time
    if hasattr(result, "execution_time_ms"):
        data["execution_time_ms"] = result.execution_time_ms

    return data


def deserialize_learn_result(data: dict[str, Any]) -> LearnResult:
    """Deserialize dictionary to LearnResult.

    Args:
        data: Dictionary representation.

    Returns:
        LearnResult instance.
    """
    kwargs: dict[str, Any] = {}

    if "rules" in data:
        kwargs["rules"] = tuple(data["rules"])
    if "schema" in data:
        kwargs["schema"] = data["schema"]
    if "metadata" in data:
        kwargs["metadata"] = data["metadata"]

    return LearnResult(**kwargs)


def _serialize_failure(failure: Any) -> dict[str, Any]:
    """Serialize a validation failure."""
    if isinstance(failure, dict):
        return failure
    if hasattr(failure, "to_dict"):
        return failure.to_dict()

    result: dict[str, Any] = {}
    for attr in ("column", "rule_type", "message", "severity", "row_count", "sample_values"):
        if hasattr(failure, attr):
            value = getattr(failure, attr)
            if hasattr(value, "name"):  # Enum
                value = value.name
            result[attr] = value
    return result


def _serialize_column_profile(column: Any) -> dict[str, Any]:
    """Serialize a column profile."""
    if isinstance(column, dict):
        return column
    if hasattr(column, "to_dict"):
        return column.to_dict()

    result: dict[str, Any] = {}
    for attr in (
        "column_name",
        "dtype",
        "null_count",
        "null_percentage",
        "unique_count",
        "unique_percentage",
        "min_value",
        "max_value",
        "mean",
        "std",
        "median",
    ):
        if hasattr(column, attr):
            result[attr] = getattr(column, attr)
    return result


def _serialize_learned_rule(rule: Any) -> dict[str, Any]:
    """Serialize a learned rule."""
    if isinstance(rule, dict):
        return rule
    if hasattr(rule, "to_dict"):
        return rule.to_dict()

    result: dict[str, Any] = {}
    for attr in ("column", "rule_type", "parameters", "confidence"):
        if hasattr(rule, attr):
            result[attr] = getattr(rule, attr)
    return result


def _serialize_schema(schema: Any) -> dict[str, Any]:
    """Serialize a schema object."""
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "to_dict"):
        return schema.to_dict()
    if hasattr(schema, "__dict__"):
        return schema.__dict__
    return {"value": str(schema)}


def _serialize_metadata(metadata: Any) -> dict[str, Any]:
    """Serialize metadata."""
    if isinstance(metadata, dict):
        result = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, "name"):  # Enum
                result[key] = value.name
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    if hasattr(metadata, "to_dict"):
        return metadata.to_dict()
    return {"value": metadata}


def _serialize_object(obj: Any) -> dict[str, Any]:
    """Generic object serialization."""
    result: dict[str, Any] = {"_type": type(obj).__name__}
    for key, value in obj.__dict__.items():
        if key.startswith("_"):
            continue
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        elif hasattr(value, "name"):  # Enum
            result[key] = value.name
        elif hasattr(value, "to_dict"):
            result[key] = value.to_dict()
        elif hasattr(value, "__dict__"):
            result[key] = _serialize_object(value)
        else:
            result[key] = value
    return result


def to_json(result: Any, **kwargs: Any) -> str:
    """Serialize result to JSON string.

    Args:
        result: Result to serialize.
        **kwargs: Additional arguments for json.dumps.

    Returns:
        JSON string.
    """
    return json.dumps(serialize_result(result), **kwargs)


def from_json(data: str, result_type: str | None = None) -> Any:
    """Deserialize result from JSON string.

    Args:
        data: JSON string.
        result_type: Type hint ("check", "profile", "learn").

    Returns:
        Deserialized result.
    """
    return deserialize_result(json.loads(data), result_type)
