"""Serialization utilities for converting between common/ types and Prefect-compatible formats.

This module handles the conversion of data quality results to dictionaries
that can be stored as Prefect artifacts, task results, and flow outputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from common.base import CheckResult, LearnResult, ProfileResult


class ResultSerializer:
    """Serializer for data quality results.

    Converts common/ result types to Prefect-compatible dictionaries.
    Supports CheckResult, ProfileResult, and LearnResult.
    """

    @staticmethod
    def serialize_check_result(result: CheckResult) -> dict[str, Any]:
        """Serialize a CheckResult to a dictionary.

        Args:
            result: The CheckResult to serialize.

        Returns:
            Dictionary representation suitable for Prefect storage.
        """
        failures = []
        for f in result.failures:
            failure_dict: dict[str, Any] = {
                "rule_name": f.rule_name,
                "column": f.column,
                "message": f.message,
                "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                "failed_count": f.failed_count,
                "total_count": f.total_count,
            }
            if hasattr(f, "metadata") and f.metadata:
                failure_dict["metadata"] = f.metadata
            failures.append(failure_dict)

        total = result.passed_count + result.failed_count
        failure_rate = result.failed_count / total if total > 0 else 0.0

        return {
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "is_success": result.is_success,
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "failure_rate": failure_rate,
            "failures": failures,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat() if hasattr(result, "timestamp") else datetime.now().isoformat(),
            "metadata": result.metadata if hasattr(result, "metadata") else {},
        }

    @staticmethod
    def serialize_profile_result(result: ProfileResult) -> dict[str, Any]:
        """Serialize a ProfileResult to a dictionary.

        Args:
            result: The ProfileResult to serialize.

        Returns:
            Dictionary representation suitable for Prefect storage.
        """
        columns = []
        for col in result.columns:
            col_dict: dict[str, Any] = {
                "column_name": col.column_name,
                "dtype": col.dtype,
                "null_count": col.null_count,
                "null_percentage": col.null_percentage,
                "unique_count": col.unique_count,
                "unique_percentage": col.unique_percentage,
            }
            # Add optional statistics if available
            if hasattr(col, "min_value") and col.min_value is not None:
                col_dict["min_value"] = col.min_value
            if hasattr(col, "max_value") and col.max_value is not None:
                col_dict["max_value"] = col.max_value
            if hasattr(col, "mean_value") and col.mean_value is not None:
                col_dict["mean_value"] = col.mean_value
            if hasattr(col, "std_value") and col.std_value is not None:
                col_dict["std_value"] = col.std_value
            columns.append(col_dict)

        return {
            "row_count": result.row_count,
            "column_count": result.column_count,
            "columns": columns,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat() if hasattr(result, "timestamp") else datetime.now().isoformat(),
            "metadata": result.metadata if hasattr(result, "metadata") else {},
        }

    @staticmethod
    def serialize_learn_result(result: LearnResult) -> dict[str, Any]:
        """Serialize a LearnResult to a dictionary.

        Args:
            result: The LearnResult to serialize.

        Returns:
            Dictionary representation suitable for Prefect storage.
        """
        rules = []
        for rule in result.rules:
            rule_dict: dict[str, Any] = {
                "rule_type": rule.rule_type,
                "column": rule.column,
                "confidence": rule.confidence,
            }
            if hasattr(rule, "parameters") and rule.parameters:
                rule_dict["parameters"] = rule.parameters
            if hasattr(rule, "metadata") and rule.metadata:
                rule_dict["metadata"] = rule.metadata
            rules.append(rule_dict)

        return {
            "rules": rules,
            "rules_count": len(rules),
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat() if hasattr(result, "timestamp") else datetime.now().isoformat(),
            "metadata": result.metadata if hasattr(result, "metadata") else {},
        }


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize any data quality result to a dictionary.

    Auto-detects the result type and uses the appropriate serializer.

    Args:
        result: A CheckResult, ProfileResult, or LearnResult.

    Returns:
        Dictionary representation of the result.

    Raises:
        TypeError: If the result type is not recognized.
    """
    serializer = ResultSerializer()

    # Duck-type detection based on attributes
    if hasattr(result, "passed_count") and hasattr(result, "failed_count"):
        return serializer.serialize_check_result(result)
    elif hasattr(result, "columns") and hasattr(result, "row_count"):
        return serializer.serialize_profile_result(result)
    elif hasattr(result, "rules"):
        return serializer.serialize_learn_result(result)
    else:
        raise TypeError(f"Unknown result type: {type(result).__name__}")


def deserialize_result(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a result dictionary.

    This function validates and normalizes the dictionary format.
    It does not reconstruct the original result objects.

    Args:
        data: Dictionary to deserialize.

    Returns:
        Normalized dictionary representation.
    """
    # Just return as-is for now, could add validation later
    return dict(data)


def to_prefect_artifact(result: dict[str, Any], artifact_type: str = "table") -> dict[str, Any]:
    """Convert a serialized result to Prefect artifact format.

    Args:
        result: Serialized result dictionary.
        artifact_type: Type of Prefect artifact (table, markdown, etc.).

    Returns:
        Prefect-compatible artifact dictionary.
    """
    if artifact_type == "table":
        return _to_table_artifact(result)
    elif artifact_type == "markdown":
        return _to_markdown_artifact(result)
    else:
        return {"type": artifact_type, "data": result}


def _to_table_artifact(result: dict[str, Any]) -> dict[str, Any]:
    """Convert to table artifact format."""
    rows = []

    # Check result
    if "passed_count" in result and "failed_count" in result:
        rows.extend([
            {"metric": "Status", "value": result.get("status", "unknown")},
            {"metric": "Passed", "value": result.get("passed_count", 0)},
            {"metric": "Failed", "value": result.get("failed_count", 0)},
            {"metric": "Failure Rate", "value": f"{result.get('failure_rate', 0):.2%}"},
        ])

    # Profile result
    elif "row_count" in result and "column_count" in result:
        rows.extend([
            {"metric": "Row Count", "value": result.get("row_count", 0)},
            {"metric": "Column Count", "value": result.get("column_count", 0)},
        ])

    # Learn result
    elif "rules" in result:
        rows.append({"metric": "Rules Learned", "value": len(result.get("rules", []))})

    # Common fields
    if "execution_time_ms" in result:
        rows.append({"metric": "Duration (ms)", "value": result.get("execution_time_ms", 0)})

    return {"type": "table", "data": rows}


def _to_markdown_artifact(result: dict[str, Any]) -> dict[str, Any]:
    """Convert to markdown artifact format."""
    lines = ["## Data Quality Result\n"]

    if "status" in result:
        status_emoji = "✅" if result.get("is_success", True) else "❌"
        lines.append(f"**Status:** {status_emoji} {result['status']}\n")

    if "passed_count" in result:
        lines.append(f"- **Passed:** {result['passed_count']}")
        lines.append(f"- **Failed:** {result['failed_count']}")
        lines.append(f"- **Failure Rate:** {result.get('failure_rate', 0):.2%}\n")

    if "failures" in result and result["failures"]:
        lines.append("### Failures\n")
        for f in result["failures"][:10]:  # Limit to 10
            lines.append(f"- **{f.get('rule_name', 'Unknown')}** ({f.get('column', 'N/A')}): {f.get('message', '')}")

    return {"type": "markdown", "content": "\n".join(lines)}


__all__ = [
    "ResultSerializer",
    "serialize_result",
    "deserialize_result",
    "to_prefect_artifact",
]
