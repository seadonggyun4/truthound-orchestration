"""Serialization helpers for Prefect integration.

Prefect-specific helpers are thin wrappers around the shared orchestration wire
format from ``common.serializers``.
"""

from __future__ import annotations

from typing import Any

from common.serializers import detect_result_type, serialize_result_wire


class ResultSerializer:
    """Serializer for data quality results."""

    @staticmethod
    def serialize_check_result(result: Any) -> dict[str, Any]:
        return serialize_result(result)

    @staticmethod
    def serialize_profile_result(result: Any) -> dict[str, Any]:
        return serialize_result(result)

    @staticmethod
    def serialize_learn_result(result: Any) -> dict[str, Any]:
        return serialize_result(result)


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
    if hasattr(result, "passed_count") and hasattr(result, "failed_count"):
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

        total = getattr(result, "passed_count", 0) + getattr(result, "failed_count", 0)
        failure_rate = getattr(result, "failure_rate", None)
        if failure_rate is None:
            failure_rate = (getattr(result, "failed_count", 0) / total * 100) if total else 0.0

        return {
            "type": "check",
            "result_type": "check",
            "status": _enum_name(getattr(result, "status", None), default="UNKNOWN"),
            "is_success": getattr(result, "is_success", False),
            "passed_count": getattr(result, "passed_count", 0),
            "failed_count": getattr(result, "failed_count", 0),
            "warning_count": getattr(result, "warning_count", 0),
            "skipped_count": getattr(result, "skipped_count", 0),
            "failure_rate": failure_rate,
            "failures": failures,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "timestamp": _timestamp_value(getattr(result, "timestamp", None)),
            "metadata": getattr(result, "metadata", {}),
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
            "metadata": getattr(result, "metadata", {}),
        }

    if hasattr(result, "rules"):
        rules = []
        for rule in getattr(result, "rules", []):
            rules.append(
                {
                    "rule_type": getattr(rule, "rule_type", "unknown"),
                    "column": getattr(rule, "column", None),
                    "confidence": getattr(rule, "confidence", 0.0),
                    "parameters": getattr(rule, "parameters", {}),
                }
            )

        return {
            "type": "learn",
            "result_type": "learn",
            "rules": rules,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "timestamp": _timestamp_value(getattr(result, "timestamp", None)),
            "metadata": getattr(result, "metadata", {}),
        }

    raise TypeError(f"Unknown result type: {type(result).__name__}")


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize any supported result object to the shared wire format."""

    if isinstance(result, dict):
        return dict(result)

    if hasattr(result, "to_dict"):
        payload = serialize_result_wire(result, include_result_type=True)
        payload["type"] = detect_result_type(result)
        return payload

    return _serialize_duck_typed_result(result)


def deserialize_result(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize serialized data for Prefect consumers."""

    return dict(data)


def to_prefect_artifact(result: dict[str, Any], artifact_type: str = "table") -> dict[str, Any]:
    """Convert a serialized result to Prefect artifact format."""

    if artifact_type == "table":
        return _to_table_artifact(result)
    if artifact_type == "markdown":
        return _to_markdown_artifact(result)
    return {"type": artifact_type, "data": result}


def _to_table_artifact(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    result_type = result.get("type") or result.get("result_type")

    if result_type == "check" or ("passed_count" in result and "failed_count" in result):
        rows.extend(
            [
                {"metric": "Status", "value": result.get("status", "UNKNOWN")},
                {"metric": "Passed", "value": result.get("passed_count", 0)},
                {"metric": "Failed", "value": result.get("failed_count", 0)},
                {"metric": "Failure Rate", "value": f"{result.get('failure_rate', 0):.2f}%"},
            ]
        )
    elif result_type == "profile" or ("row_count" in result and "column_count" in result):
        rows.extend(
            [
                {"metric": "Row Count", "value": result.get("row_count", 0)},
                {"metric": "Column Count", "value": result.get("column_count", 0)},
            ]
        )
    elif result_type == "learn" or "rules" in result:
        rows.append({"metric": "Rules Learned", "value": len(result.get("rules", []))})

    if "execution_time_ms" in result:
        rows.append({"metric": "Duration (ms)", "value": result.get("execution_time_ms", 0)})

    return {"type": "table", "data": rows}


def _to_markdown_artifact(result: dict[str, Any]) -> dict[str, Any]:
    lines = ["## Data Quality Result\n"]

    if "status" in result:
        status_emoji = "✅" if result.get("is_success", True) else "❌"
        lines.append(f"**Status:** {status_emoji} {result['status']}\n")

    if "passed_count" in result:
        lines.append(f"- **Passed:** {result['passed_count']}")
        lines.append(f"- **Failed:** {result['failed_count']}")
        lines.append(f"- **Failure Rate:** {result.get('failure_rate', 0):.2f}%\n")

    if "failures" in result and result["failures"]:
        lines.append("### Failures\n")
        for failure in result["failures"][:10]:
            lines.append(
                f"- **{failure.get('rule_name', 'Unknown')}** "
                f"({failure.get('column', 'N/A')}): {failure.get('message', '')}"
            )

    return {"type": "markdown", "content": "\n".join(lines)}


__all__ = [
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "to_prefect_artifact",
]
