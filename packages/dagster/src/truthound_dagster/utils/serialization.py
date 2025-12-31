"""Serialization Utilities for Dagster Integration.

This module provides functions for serializing data quality results
to formats compatible with Dagster's metadata and output systems.

Example:
    >>> from truthound_dagster.utils import serialize_result, to_dagster_metadata
    >>>
    >>> result = engine.check(data, rules)
    >>> serialized = serialize_result(result)
    >>> metadata = to_dagster_metadata(serialized)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from common.base import CheckResult, LearnResult, ProfileResult


class ResultSerializer:
    """Serializer for data quality results.

    This class provides methods for serializing different types of
    data quality results to dictionaries compatible with Dagster.

    Example:
        >>> serializer = ResultSerializer()
        >>> check_dict = serializer.serialize_check_result(result)
    """

    def serialize_check_result(self, result: CheckResult) -> dict[str, Any]:
        """Serialize CheckResult to dictionary.

        Args:
            result: Check result to serialize.

        Returns:
            dict[str, Any]: Serialized result.
        """
        return {
            "type": "check",
            "status": result.status.value,
            "is_success": result.is_success,
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "warning_count": result.warning_count,
            "skipped_count": getattr(result, "skipped_count", 0),
            "failure_rate": result.failure_rate,
            "failures": [
                {
                    "rule_name": f.rule_name,
                    "column": f.column,
                    "message": f.message,
                    "severity": f.severity.value,
                    "failed_count": f.failed_count,
                    "total_count": f.total_count,
                    "failure_rate": f.failure_rate,
                }
                for f in result.failures
            ],
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
        }

    def serialize_profile_result(self, result: ProfileResult) -> dict[str, Any]:
        """Serialize ProfileResult to dictionary.

        Args:
            result: Profile result to serialize.

        Returns:
            dict[str, Any]: Serialized result.
        """
        columns = []
        for col in result.columns:
            col_data = {
                "column_name": col.column_name,
                "dtype": str(col.dtype),
                "null_count": col.null_count,
                "null_percentage": col.null_percentage,
                "unique_count": col.unique_count,
                "unique_percentage": col.unique_percentage,
            }

            # Add numeric stats if available
            if hasattr(col, "min_value") and col.min_value is not None:
                col_data.update(
                    {
                        "min_value": col.min_value,
                        "max_value": col.max_value,
                        "mean": getattr(col, "mean", None),
                        "std": getattr(col, "std", None),
                    }
                )

            columns.append(col_data)

        return {
            "type": "profile",
            "row_count": result.row_count,
            "column_count": result.column_count,
            "columns": columns,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
        }

    def serialize_learn_result(self, result: LearnResult) -> dict[str, Any]:
        """Serialize LearnResult to dictionary.

        Args:
            result: Learn result to serialize.

        Returns:
            dict[str, Any]: Serialized result.
        """
        rules = []
        for rule in result.rules:
            rule_data = {
                "column": rule.column,
                "rule_type": rule.rule_type,
                "confidence": rule.confidence,
                "parameters": rule.parameters,
            }
            rules.append(rule_data)

        return {
            "type": "learn",
            "rule_count": len(result.rules),
            "rules": rules,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
        }

    def deserialize_check_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize check result dictionary.

        This method validates and normalizes serialized data.
        It does not create a CheckResult object, but returns
        a validated dictionary.

        Args:
            data: Serialized result data.

        Returns:
            dict[str, Any]: Validated data.
        """
        # Parse timestamp if string
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return data


# Module-level functions for convenience
_serializer = ResultSerializer()


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize any result type to dictionary.

    This function automatically detects the result type and
    uses the appropriate serialization method.

    Args:
        result: Result to serialize (CheckResult, ProfileResult, etc.)

    Returns:
        dict[str, Any]: Serialized result.

    Raises:
        TypeError: If result type is not recognized.
    """
    # Check for type by duck typing
    if hasattr(result, "failures"):
        return _serializer.serialize_check_result(result)
    elif hasattr(result, "columns"):
        return _serializer.serialize_profile_result(result)
    elif hasattr(result, "rules"):
        return _serializer.serialize_learn_result(result)
    elif isinstance(result, dict):
        return result
    else:
        msg = f"Unknown result type: {type(result).__name__}"
        raise TypeError(msg)


def deserialize_result(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize result dictionary.

    Args:
        data: Serialized result data.

    Returns:
        dict[str, Any]: Validated data.
    """
    result_type = data.get("type", "check")

    if result_type == "check":
        return _serializer.deserialize_check_result(data)
    else:
        # For profile and learn, just return the data
        return data


def to_dagster_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Convert result dictionary to Dagster metadata format.

    This function transforms a result dictionary into the format
    expected by Dagster's metadata system.

    Args:
        result: Result dictionary.

    Returns:
        dict[str, Any]: Dagster-compatible metadata.
    """
    metadata: dict[str, Any] = {}

    result_type = result.get("type", "check")

    if result_type == "check":
        metadata["status"] = result.get("status", "unknown")
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
        metadata["rule_count"] = result.get("rule_count", 0)
        metadata["execution_time_ms"] = result.get("execution_time_ms", 0.0)

    return metadata


def to_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format.

    This function recursively converts Python objects to
    JSON-serializable types.

    Args:
        obj: Object to convert.

    Returns:
        JSON-serializable value.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif hasattr(obj, "to_dict"):
        return to_json_serializable(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        return to_json_serializable(obj.__dict__)
    else:
        return str(obj)
