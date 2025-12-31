"""Serialization Utilities for Data Quality Results.

This module provides serialization and deserialization utilities
for data quality results, optimized for XCom storage.

Example:
    >>> from truthound_airflow.utils import serialize_result
    >>>
    >>> serialized = serialize_result(check_result)
    >>> context["ti"].xcom_push(key="result", value=serialized)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...


@runtime_checkable
class HasTimestamp(Protocol):
    """Protocol for objects with timestamp."""

    timestamp: datetime


# =============================================================================
# Result Serializer
# =============================================================================


class ResultSerializer:
    """Serializer for data quality results.

    Provides methods for converting result objects to XCom-compatible
    dictionaries and vice versa.

    Example:
        >>> serializer = ResultSerializer()
        >>> xcom_value = serializer.serialize_check_result(result)
        >>> restored = serializer.deserialize_check_result(xcom_value)
    """

    def serialize_check_result(self, result: Any) -> dict[str, Any]:
        """Serialize CheckResult to dictionary.

        Args:
            result: CheckResult object.

        Returns:
            dict[str, Any]: XCom-compatible dictionary.
        """
        if hasattr(result, "to_dict"):
            base = result.to_dict()
        else:
            base = self._serialize_check_result_attrs(result)

        # Ensure timestamp is ISO format
        if "timestamp" in base and isinstance(base["timestamp"], datetime):
            base["timestamp"] = base["timestamp"].isoformat()

        # Add serialization metadata
        base["_serializer"] = "truthound_airflow"
        base["_version"] = "1.0"
        base["_type"] = "check_result"

        return base

    def _serialize_check_result_attrs(self, result: Any) -> dict[str, Any]:
        """Serialize CheckResult by attribute access."""
        failures = []
        for f in getattr(result, "failures", []):
            failure_dict = {
                "rule_name": getattr(f, "rule_name", "unknown"),
                "column": getattr(f, "column", None),
                "message": getattr(f, "message", ""),
                "severity": getattr(f, "severity", {}).value
                if hasattr(getattr(f, "severity", None), "value")
                else "medium",
                "failed_count": getattr(f, "failed_count", 0),
                "total_count": getattr(f, "total_count", 0),
                "failure_rate": getattr(f, "failure_rate", 0.0),
            }
            failures.append(failure_dict)

        timestamp = getattr(result, "timestamp", None)
        if timestamp and isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        status = getattr(result, "status", None)
        if hasattr(status, "value"):
            status = status.value
        elif status is None:
            status = "unknown"

        return {
            "status": status,
            "is_success": getattr(result, "is_success", False),
            "passed_count": getattr(result, "passed_count", 0),
            "failed_count": getattr(result, "failed_count", 0),
            "warning_count": getattr(result, "warning_count", 0),
            "skipped_count": getattr(result, "skipped_count", 0),
            "failure_rate": getattr(result, "failure_rate", 0.0),
            "failures": failures,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "timestamp": timestamp,
        }

    def deserialize_check_result(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Deserialize dictionary to CheckResult-compatible dict.

        Args:
            data: Serialized dictionary.

        Returns:
            dict[str, Any]: Deserialized result dictionary.
        """
        result = dict(data)

        # Parse timestamp
        if "timestamp" in result and isinstance(result["timestamp"], str):
            result["timestamp"] = datetime.fromisoformat(result["timestamp"])

        # Remove serialization metadata for cleaner output
        result.pop("_serializer", None)
        result.pop("_version", None)
        result.pop("_type", None)

        return result

    def serialize_profile_result(self, result: Any) -> dict[str, Any]:
        """Serialize ProfileResult to dictionary.

        Args:
            result: ProfileResult object.

        Returns:
            dict[str, Any]: XCom-compatible dictionary.
        """
        if hasattr(result, "to_dict"):
            base = result.to_dict()
        else:
            base = self._serialize_profile_result_attrs(result)

        base["_serializer"] = "truthound_airflow"
        base["_version"] = "1.0"
        base["_type"] = "profile_result"

        return base

    def _serialize_profile_result_attrs(self, result: Any) -> dict[str, Any]:
        """Serialize ProfileResult by attribute access."""
        columns = []
        for col in getattr(result, "columns", []):
            col_dict = {
                "column_name": getattr(col, "column_name", "unknown"),
                "dtype": str(getattr(col, "dtype", "unknown")),
                "null_count": getattr(col, "null_count", 0),
                "null_percentage": getattr(col, "null_percentage", 0.0),
                "unique_count": getattr(col, "unique_count", 0),
                "unique_percentage": getattr(col, "unique_percentage", 0.0),
                "statistics": getattr(col, "statistics", {}),
                "patterns": getattr(col, "patterns", []),
                "distribution": getattr(col, "distribution", {}),
            }
            columns.append(col_dict)

        return {
            "row_count": getattr(result, "row_count", 0),
            "column_count": getattr(result, "column_count", 0),
            "columns": columns,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
        }

    def deserialize_profile_result(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Deserialize dictionary to ProfileResult-compatible dict.

        Args:
            data: Serialized dictionary.

        Returns:
            dict[str, Any]: Deserialized result dictionary.
        """
        result = dict(data)
        result.pop("_serializer", None)
        result.pop("_version", None)
        result.pop("_type", None)
        return result

    def serialize_learn_result(self, result: Any) -> dict[str, Any]:
        """Serialize LearnResult to dictionary.

        Args:
            result: LearnResult object.

        Returns:
            dict[str, Any]: XCom-compatible dictionary.
        """
        if hasattr(result, "to_dict"):
            base = result.to_dict()
        else:
            base = self._serialize_learn_result_attrs(result)

        base["_serializer"] = "truthound_airflow"
        base["_version"] = "1.0"
        base["_type"] = "learn_result"

        return base

    def _serialize_learn_result_attrs(self, result: Any) -> dict[str, Any]:
        """Serialize LearnResult by attribute access."""
        rules = []
        for rule in getattr(result, "rules", []):
            rule_dict = {
                "rule_type": getattr(rule, "rule_type", "unknown"),
                "column": getattr(rule, "column", None),
                "parameters": getattr(rule, "parameters", {}),
                "confidence": getattr(rule, "confidence", 0.0),
            }
            rules.append(rule_dict)

        return {
            "rules": rules,
            "strictness": getattr(result, "strictness", "moderate"),
            "row_count": getattr(result, "row_count", 0),
            "column_count": getattr(result, "column_count", 0),
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
        }

    def deserialize_learn_result(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Deserialize dictionary to LearnResult-compatible dict.

        Args:
            data: Serialized dictionary.

        Returns:
            dict[str, Any]: Deserialized result dictionary.
        """
        result = dict(data)
        result.pop("_serializer", None)
        result.pop("_version", None)
        result.pop("_type", None)
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

_default_serializer = ResultSerializer()


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize a check result for XCom storage.

    Args:
        result: CheckResult object.

    Returns:
        dict[str, Any]: XCom-compatible dictionary.
    """
    return _default_serializer.serialize_check_result(result)


def deserialize_result(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a check result from XCom.

    Args:
        data: Serialized dictionary.

    Returns:
        dict[str, Any]: Deserialized result dictionary.
    """
    return _default_serializer.deserialize_check_result(data)


def serialize_profile(result: Any) -> dict[str, Any]:
    """Serialize a profile result for XCom storage.

    Args:
        result: ProfileResult object.

    Returns:
        dict[str, Any]: XCom-compatible dictionary.
    """
    return _default_serializer.serialize_profile_result(result)


def deserialize_profile(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a profile result from XCom.

    Args:
        data: Serialized dictionary.

    Returns:
        dict[str, Any]: Deserialized result dictionary.
    """
    return _default_serializer.deserialize_profile_result(data)


def serialize_learn_result(result: Any) -> dict[str, Any]:
    """Serialize a learn result for XCom storage.

    Args:
        result: LearnResult object.

    Returns:
        dict[str, Any]: XCom-compatible dictionary.
    """
    return _default_serializer.serialize_learn_result(result)


def deserialize_learn_result(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a learn result from XCom.

    Args:
        data: Serialized dictionary.

    Returns:
        dict[str, Any]: Deserialized result dictionary.
    """
    return _default_serializer.deserialize_learn_result(data)


def to_xcom_value(obj: Any) -> Any:
    """Convert object to XCom-compatible value.

    Handles common types and objects with to_dict method.

    Args:
        obj: Object to convert.

    Returns:
        Any: XCom-compatible value.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, (list, tuple)):
        return [to_xcom_value(item) for item in obj]

    if isinstance(obj, dict):
        return {k: to_xcom_value(v) for k, v in obj.items()}

    if hasattr(obj, "to_dict"):
        return to_xcom_value(obj.to_dict())

    if hasattr(obj, "value"):  # Enum
        return obj.value

    # Last resort: string representation
    return str(obj)


def from_xcom_value(
    data: Any,
    type_hint: str | None = None,
) -> Any:
    """Convert XCom value back to appropriate type.

    Args:
        data: XCom value.
        type_hint: Optional type hint from _type field.

    Returns:
        Any: Converted value.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        # Check for type hint
        data_type = type_hint or data.get("_type")

        if data_type == "check_result":
            return deserialize_result(data)
        elif data_type == "profile_result":
            return deserialize_profile(data)
        elif data_type == "learn_result":
            return deserialize_learn_result(data)

        # Generic dict processing
        return {k: from_xcom_value(v) for k, v in data.items()}

    if isinstance(data, list):
        return [from_xcom_value(item) for item in data]

    return data
