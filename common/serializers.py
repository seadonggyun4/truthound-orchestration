"""Serialization utilities for Truthound Integrations.

This module provides a factory-based serialization system that supports
multiple output formats for different platforms. Each platform may have
specific requirements for how results are serialized (e.g., Airflow XCom
size limits, Dagster MetadataValue format).

Design Pattern: Factory Pattern with Strategy
    - SerializerFactory: Creates and manages serializer instances
    - ResultSerializer: Protocol for all serializers
    - Platform-specific implementations: Airflow, Dagster, Prefect

Example:
    >>> from common.serializers import SerializerFactory
    >>> factory = SerializerFactory()
    >>> serializer = factory.get("airflow_xcom")
    >>> serialized = serializer.serialize(result)
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Protocol, Self, TypeVar, runtime_checkable

from common.base import AnomalyResult, CheckResult, DriftResult, LearnResult, ProfileResult
from common.exceptions import DeserializeError, SerializationError, SerializeError


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T", CheckResult, ProfileResult, LearnResult, DriftResult, AnomalyResult)


# =============================================================================
# Serializer Protocol
# =============================================================================


@runtime_checkable
class ResultSerializer(Protocol[T]):
    """Protocol for result serializers.

    All serializers must implement serialize and deserialize methods
    for converting between result objects and serialized formats.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of the serialization format."""
        ...

    @abstractmethod
    def serialize(self, result: T) -> Any:
        """Serialize a result object.

        Args:
            result: Result object to serialize.

        Returns:
            Serialized representation.

        Raises:
            SerializeError: If serialization fails.
        """
        ...

    @abstractmethod
    def deserialize(self, data: Any, result_type: type[T]) -> T:
        """Deserialize data to a result object.

        Args:
            data: Serialized data.
            result_type: Type of result to create.

        Returns:
            Deserialized result object.

        Raises:
            DeserializeError: If deserialization fails.
        """
        ...


# =============================================================================
# JSON Serializer
# =============================================================================


class JSONSerializer:
    """Standard JSON serializer for results.

    Converts result objects to JSON strings. Suitable for storage
    in databases, files, or API responses.

    Attributes:
        indent: JSON indentation level (None for compact).
        sort_keys: Whether to sort dictionary keys.
    """

    def __init__(
        self,
        *,
        indent: int | None = None,
        sort_keys: bool = False,
    ) -> None:
        """Initialize the JSON serializer.

        Args:
            indent: JSON indentation level.
            sort_keys: Whether to sort keys.
        """
        self.indent = indent
        self.sort_keys = sort_keys

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "json"

    def serialize(
        self,
        result: CheckResult | ProfileResult | LearnResult | DriftResult | AnomalyResult,
    ) -> str:
        """Serialize a result to JSON string.

        Args:
            result: Result object to serialize.

        Returns:
            JSON string representation.

        Raises:
            SerializeError: If serialization fails.
        """
        try:
            return json.dumps(
                result.to_dict(),
                indent=self.indent,
                sort_keys=self.sort_keys,
                default=self._json_default,
            )
        except (TypeError, ValueError) as e:
            raise SerializeError(
                f"Failed to serialize result to JSON: {e}",
                target_type="json",
                cause=e,
            ) from e

    def deserialize(
        self,
        data: str,
        result_type: type[T],
    ) -> T:
        """Deserialize JSON string to result object.

        Args:
            data: JSON string.
            result_type: Type of result to create.

        Returns:
            Deserialized result object.

        Raises:
            DeserializeError: If deserialization fails.
        """
        try:
            parsed = json.loads(data)
            return result_type.from_dict(parsed)
        except json.JSONDecodeError as e:
            raise DeserializeError(
                f"Failed to parse JSON: {e}",
                target_type=result_type.__name__,
                cause=e,
            ) from e
        except (KeyError, TypeError, ValueError) as e:
            raise DeserializeError(
                f"Failed to deserialize to {result_type.__name__}: {e}",
                target_type=result_type.__name__,
                cause=e,
            ) from e

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """Default JSON encoder for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# Dict Serializer
# =============================================================================


class DictSerializer:
    """Dictionary serializer for results.

    Converts result objects to Python dictionaries. Suitable for
    in-memory processing and XCom storage.
    """

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "dict"

    def serialize(
        self,
        result: CheckResult | ProfileResult | LearnResult | DriftResult | AnomalyResult,
    ) -> dict[str, Any]:
        """Serialize a result to dictionary.

        Args:
            result: Result object to serialize.

        Returns:
            Dictionary representation.

        Raises:
            SerializeError: If serialization fails.
        """
        try:
            return result.to_dict()
        except Exception as e:
            raise SerializeError(
                f"Failed to serialize result to dict: {e}",
                target_type="dict",
                cause=e,
            ) from e

    def deserialize(
        self,
        data: dict[str, Any],
        result_type: type[T],
    ) -> T:
        """Deserialize dictionary to result object.

        Args:
            data: Dictionary data.
            result_type: Type of result to create.

        Returns:
            Deserialized result object.

        Raises:
            DeserializeError: If deserialization fails.
        """
        try:
            return result_type.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            raise DeserializeError(
                f"Failed to deserialize to {result_type.__name__}: {e}",
                target_type=result_type.__name__,
                cause=e,
            ) from e


# =============================================================================
# Airflow XCom Serializer
# =============================================================================


@dataclass(frozen=True, slots=True)
class AirflowXComConfig:
    """Configuration for Airflow XCom serializer.

    Attributes:
        max_failures: Maximum number of failures to include.
        include_sample_values: Whether to include sample values.
        max_sample_values: Maximum sample values per failure.
        include_metadata: Whether to include full metadata.
    """

    max_failures: int = 50
    include_sample_values: bool = True
    max_sample_values: int = 5
    include_metadata: bool = True


class AirflowXComSerializer:
    """Optimized serializer for Airflow XCom storage.

    Airflow XCom has size limitations. This serializer optimizes
    the result for XCom storage by:
    - Limiting the number of failures included
    - Truncating sample values
    - Optionally excluding metadata

    Attributes:
        config: Serializer configuration.
    """

    def __init__(self, config: AirflowXComConfig | None = None) -> None:
        """Initialize the Airflow XCom serializer.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AirflowXComConfig()

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "airflow_xcom"

    def serialize(
        self,
        result: CheckResult | ProfileResult | LearnResult | DriftResult | AnomalyResult,
    ) -> dict[str, Any]:
        """Serialize a result for XCom storage.

        Args:
            result: Result object to serialize.

        Returns:
            Optimized dictionary for XCom.

        Raises:
            SerializeError: If serialization fails.
        """
        try:
            data = result.to_dict()

            if isinstance(result, CheckResult):
                data = self._optimize_check_result(data, result)
            elif isinstance(result, DriftResult):
                data = self._optimize_drift_result(data, result)
            elif isinstance(result, AnomalyResult):
                data = self._optimize_anomaly_result(data, result)

            return data
        except Exception as e:
            raise SerializeError(
                f"Failed to serialize result for XCom: {e}",
                target_type="airflow_xcom",
                cause=e,
            ) from e

    def _optimize_check_result(
        self,
        data: dict[str, Any],
        result: CheckResult,
    ) -> dict[str, Any]:
        """Optimize CheckResult for XCom size limits.

        Args:
            data: Serialized result data.
            result: Original result object.

        Returns:
            Optimized dictionary.
        """
        # Limit failures
        failures = data.get("failures", [])
        if len(failures) > self.config.max_failures:
            data["failures"] = failures[: self.config.max_failures]
            data["failures_truncated"] = True
            data["total_failures"] = len(failures)

        # Optionally remove or limit sample values
        if not self.config.include_sample_values:
            for failure in data.get("failures", []):
                failure.pop("sample_values", None)
        elif self.config.max_sample_values > 0:
            for failure in data.get("failures", []):
                sample_values = failure.get("sample_values", [])
                if len(sample_values) > self.config.max_sample_values:
                    failure["sample_values"] = sample_values[: self.config.max_sample_values]

        # Optionally exclude metadata
        if not self.config.include_metadata:
            data.pop("metadata", None)
            for failure in data.get("failures", []):
                failure.pop("metadata", None)

        return data

    def _optimize_drift_result(
        self,
        data: dict[str, Any],
        result: DriftResult,
    ) -> dict[str, Any]:
        """Optimize DriftResult for XCom size limits.

        Args:
            data: Serialized result data.
            result: Original result object.

        Returns:
            Optimized dictionary.
        """
        columns = data.get("drifted_columns", [])
        if len(columns) > self.config.max_failures:
            data["drifted_columns"] = columns[: self.config.max_failures]
            data["columns_truncated"] = True
            data["total_drifted_columns_count"] = len(columns)

        if not self.config.include_metadata:
            data.pop("metadata", None)
            for col in data.get("drifted_columns", []):
                col.pop("metadata", None)

        return data

    def _optimize_anomaly_result(
        self,
        data: dict[str, Any],
        result: AnomalyResult,
    ) -> dict[str, Any]:
        """Optimize AnomalyResult for XCom size limits.

        Args:
            data: Serialized result data.
            result: Original result object.

        Returns:
            Optimized dictionary.
        """
        anomalies = data.get("anomalies", [])
        if len(anomalies) > self.config.max_failures:
            data["anomalies"] = anomalies[: self.config.max_failures]
            data["anomalies_truncated"] = True
            data["total_anomalies_count"] = len(anomalies)

        if not self.config.include_metadata:
            data.pop("metadata", None)
            for a in data.get("anomalies", []):
                a.pop("metadata", None)

        return data

    def deserialize(
        self,
        data: dict[str, Any],
        result_type: type[T],
    ) -> T:
        """Deserialize XCom data to result object.

        Args:
            data: XCom dictionary data.
            result_type: Type of result to create.

        Returns:
            Deserialized result object.

        Raises:
            DeserializeError: If deserialization fails.
        """
        try:
            return result_type.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            raise DeserializeError(
                f"Failed to deserialize XCom to {result_type.__name__}: {e}",
                target_type=result_type.__name__,
                cause=e,
            ) from e


# =============================================================================
# Dagster Output Serializer
# =============================================================================


class DagsterOutputSerializer:
    """Serializer for Dagster MetadataValue format.

    Converts results to a format suitable for Dagster output metadata,
    including structured data for the Dagster UI.
    """

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "dagster"

    def serialize(
        self,
        result: CheckResult | ProfileResult | LearnResult | DriftResult | AnomalyResult,
    ) -> dict[str, Any]:
        """Serialize a result for Dagster output.

        Args:
            result: Result object to serialize.

        Returns:
            Dictionary with Dagster-compatible metadata.

        Raises:
            SerializeError: If serialization fails.
        """
        try:
            data = result.to_dict()

            # Add Dagster-specific formatting
            metadata: dict[str, Any] = {
                "truthound_result": data,
            }

            if isinstance(result, CheckResult):
                metadata.update(self._format_check_metadata(result))
            elif isinstance(result, ProfileResult):
                metadata.update(self._format_profile_metadata(result))
            elif isinstance(result, DriftResult):
                metadata.update(self._format_drift_metadata(result))
            elif isinstance(result, AnomalyResult):
                metadata.update(self._format_anomaly_metadata(result))

            return metadata
        except Exception as e:
            raise SerializeError(
                f"Failed to serialize result for Dagster: {e}",
                target_type="dagster",
                cause=e,
            ) from e

    def _format_check_metadata(self, result: CheckResult) -> dict[str, Any]:
        """Format CheckResult metadata for Dagster UI.

        Args:
            result: CheckResult to format.

        Returns:
            Dagster-compatible metadata dictionary.
        """
        return {
            "status": {"raw_value": result.status.name, "type": "text"},
            "pass_rate": {"raw_value": result.pass_rate, "type": "float"},
            "passed_count": {"raw_value": result.passed_count, "type": "int"},
            "failed_count": {"raw_value": result.failed_count, "type": "int"},
            "execution_time_ms": {"raw_value": result.execution_time_ms, "type": "float"},
            "failure_summary": {
                "raw_value": self._create_failure_summary(result),
                "type": "md",
            },
        }

    def _format_profile_metadata(self, result: ProfileResult) -> dict[str, Any]:
        """Format ProfileResult metadata for Dagster UI.

        Args:
            result: ProfileResult to format.

        Returns:
            Dagster-compatible metadata dictionary.
        """
        return {
            "status": {"raw_value": result.status.name, "type": "text"},
            "row_count": {"raw_value": result.row_count, "type": "int"},
            "column_count": {"raw_value": result.column_count, "type": "int"},
            "execution_time_ms": {"raw_value": result.execution_time_ms, "type": "float"},
        }

    def _format_drift_metadata(self, result: DriftResult) -> dict[str, Any]:
        """Format DriftResult metadata for Dagster UI.

        Args:
            result: DriftResult to format.

        Returns:
            Dagster-compatible metadata dictionary.
        """
        return {
            "status": {"raw_value": result.status.name, "type": "text"},
            "drift_rate": {"raw_value": result.drift_rate, "type": "float"},
            "total_columns": {"raw_value": result.total_columns, "type": "int"},
            "drifted_count": {"raw_value": result.drifted_count, "type": "int"},
            "method": {"raw_value": result.method.value, "type": "text"},
            "execution_time_ms": {"raw_value": result.execution_time_ms, "type": "float"},
            "drift_summary": {
                "raw_value": self._create_drift_summary(result),
                "type": "md",
            },
        }

    def _format_anomaly_metadata(self, result: AnomalyResult) -> dict[str, Any]:
        """Format AnomalyResult metadata for Dagster UI.

        Args:
            result: AnomalyResult to format.

        Returns:
            Dagster-compatible metadata dictionary.
        """
        return {
            "status": {"raw_value": result.status.name, "type": "text"},
            "anomaly_rate": {"raw_value": result.anomaly_rate, "type": "float"},
            "anomalous_row_count": {"raw_value": result.anomalous_row_count, "type": "int"},
            "total_row_count": {"raw_value": result.total_row_count, "type": "int"},
            "detector": {"raw_value": result.detector, "type": "text"},
            "execution_time_ms": {"raw_value": result.execution_time_ms, "type": "float"},
            "anomaly_summary": {
                "raw_value": self._create_anomaly_summary(result),
                "type": "md",
            },
        }

    def _create_failure_summary(self, result: CheckResult) -> str:
        """Create markdown summary of failures.

        Args:
            result: CheckResult with failures.

        Returns:
            Markdown formatted summary.
        """
        if not result.failures:
            return "No failures"

        lines = ["| Rule | Column | Message |", "|------|--------|---------|"]
        for failure in result.failures[:10]:  # Limit to 10 for readability
            rule = failure.rule_name
            column = failure.column or "-"
            message = failure.message[:50] + "..." if len(failure.message) > 50 else failure.message
            lines.append(f"| {rule} | {column} | {message} |")

        if len(result.failures) > 10:
            lines.append(f"\n*...and {len(result.failures) - 10} more failures*")

        return "\n".join(lines)

    def _create_drift_summary(self, result: DriftResult) -> str:
        """Create markdown summary of drift results.

        Args:
            result: DriftResult to summarize.

        Returns:
            Markdown formatted summary.
        """
        if not result.drifted_columns:
            return "No drift detected"

        lines = [
            "| Column | Method | Statistic | P-Value | Drifted |",
            "|--------|--------|-----------|---------|---------|",
        ]
        for col in result.drifted_columns[:10]:
            lines.append(
                f"| {col.column} | {col.method} | {col.statistic:.4f} | "
                f"{col.p_value:.4f} | {'Yes' if col.is_drifted else 'No'} |"
            )

        if len(result.drifted_columns) > 10:
            lines.append(f"\n*...and {len(result.drifted_columns) - 10} more columns*")

        return "\n".join(lines)

    def _create_anomaly_summary(self, result: AnomalyResult) -> str:
        """Create markdown summary of anomaly results.

        Args:
            result: AnomalyResult to summarize.

        Returns:
            Markdown formatted summary.
        """
        if not result.anomalies:
            return "No anomalies detected"

        lines = [
            "| Column | Score | Threshold | Anomaly | Detector |",
            "|--------|-------|-----------|---------|----------|",
        ]
        for a in result.anomalies[:10]:
            lines.append(
                f"| {a.column} | {a.score:.4f} | {a.threshold:.4f} | "
                f"{'Yes' if a.is_anomaly else 'No'} | {a.detector} |"
            )

        if len(result.anomalies) > 10:
            lines.append(f"\n*...and {len(result.anomalies) - 10} more anomalies*")

        return "\n".join(lines)

    def deserialize(
        self,
        data: dict[str, Any],
        result_type: type[T],
    ) -> T:
        """Deserialize Dagster metadata to result object.

        Args:
            data: Dagster metadata dictionary.
            result_type: Type of result to create.

        Returns:
            Deserialized result object.

        Raises:
            DeserializeError: If deserialization fails.
        """
        try:
            # Extract the actual result from Dagster metadata
            result_data = data.get("truthound_result", data)
            return result_type.from_dict(result_data)
        except (KeyError, TypeError, ValueError) as e:
            raise DeserializeError(
                f"Failed to deserialize Dagster data to {result_type.__name__}: {e}",
                target_type=result_type.__name__,
                cause=e,
            ) from e


# =============================================================================
# Prefect Artifact Serializer
# =============================================================================


class PrefectArtifactSerializer:
    """Serializer for Prefect artifact format.

    Converts results to a format suitable for Prefect artifacts,
    including markdown reports for the Prefect UI.
    """

    @property
    def format_name(self) -> str:
        """Return the format name."""
        return "prefect"

    def serialize(
        self,
        result: CheckResult | ProfileResult | LearnResult | DriftResult | AnomalyResult,
    ) -> dict[str, Any]:
        """Serialize a result for Prefect artifact.

        Args:
            result: Result object to serialize.

        Returns:
            Dictionary with Prefect-compatible artifact data.

        Raises:
            SerializeError: If serialization fails.
        """
        try:
            data = result.to_dict()

            artifact: dict[str, Any] = {
                "type": "truthound_result",
                "data": data,
            }

            if isinstance(result, CheckResult):
                artifact["markdown"] = self._create_check_markdown(result)
            elif isinstance(result, ProfileResult):
                artifact["markdown"] = self._create_profile_markdown(result)
            elif isinstance(result, DriftResult):
                artifact["markdown"] = self._create_drift_markdown(result)
            elif isinstance(result, AnomalyResult):
                artifact["markdown"] = self._create_anomaly_markdown(result)

            return artifact
        except Exception as e:
            raise SerializeError(
                f"Failed to serialize result for Prefect: {e}",
                target_type="prefect",
                cause=e,
            ) from e

    def _create_check_markdown(self, result: CheckResult) -> str:
        """Create markdown report for CheckResult.

        Args:
            result: CheckResult to format.

        Returns:
            Markdown formatted report.
        """
        lines = [
            "# Truthound Validation Report",
            "",
            f"**Status:** {result.status.name}",
            f"**Pass Rate:** {result.pass_rate:.1f}%",
            "",
            "## Summary",
            "",
            f"- Passed: {result.passed_count}",
            f"- Failed: {result.failed_count}",
            f"- Warnings: {result.warning_count}",
            f"- Skipped: {result.skipped_count}",
            "",
            f"**Execution Time:** {result.execution_time_ms:.2f}ms",
            "",
        ]

        if result.failures:
            lines.extend([
                "## Failures",
                "",
                "| Severity | Rule | Column | Message |",
                "|----------|------|--------|---------|",
            ])
            for failure in result.failures[:20]:
                severity = failure.severity.name
                rule = failure.rule_name
                column = failure.column or "-"
                message = (
                    failure.message[:40] + "..."
                    if len(failure.message) > 40
                    else failure.message
                )
                lines.append(f"| {severity} | {rule} | {column} | {message} |")

            if len(result.failures) > 20:
                lines.append(f"\n*...and {len(result.failures) - 20} more failures*")

        return "\n".join(lines)

    def _create_profile_markdown(self, result: ProfileResult) -> str:
        """Create markdown report for ProfileResult.

        Args:
            result: ProfileResult to format.

        Returns:
            Markdown formatted report.
        """
        lines = [
            "# Truthound Profile Report",
            "",
            f"**Status:** {result.status.name}",
            f"**Rows:** {result.row_count:,}",
            f"**Columns:** {result.column_count}",
            "",
            "## Column Statistics",
            "",
            "| Column | Type | Nulls | Unique |",
            "|--------|------|-------|--------|",
        ]

        for col in result.columns[:30]:
            null_pct = f"{col.null_percentage:.1f}%"
            unique_pct = f"{col.unique_percentage:.1f}%"
            lines.append(f"| {col.column_name} | {col.dtype} | {null_pct} | {unique_pct} |")

        if len(result.columns) > 30:
            lines.append(f"\n*...and {len(result.columns) - 30} more columns*")

        lines.extend([
            "",
            f"**Execution Time:** {result.execution_time_ms:.2f}ms",
        ])

        return "\n".join(lines)

    def _create_drift_markdown(self, result: DriftResult) -> str:
        """Create markdown report for DriftResult.

        Args:
            result: DriftResult to format.

        Returns:
            Markdown formatted report.
        """
        lines = [
            "# Truthound Drift Detection Report",
            "",
            f"**Status:** {result.status.name}",
            f"**Drift Rate:** {result.drift_rate:.1f}%",
            f"**Method:** {result.method.value}",
            "",
            "## Summary",
            "",
            f"- Total Columns: {result.total_columns}",
            f"- Drifted Columns: {result.drifted_count}",
            "",
            f"**Execution Time:** {result.execution_time_ms:.2f}ms",
            "",
        ]

        if result.drifted_columns:
            lines.extend([
                "## Drifted Columns",
                "",
                "| Column | Method | Statistic | P-Value | Severity |",
                "|--------|--------|-----------|---------|----------|",
            ])
            for col in result.drifted_columns[:20]:
                severity = col.severity if hasattr(col, "severity") else "-"
                lines.append(
                    f"| {col.column} | {col.method} | {col.statistic:.4f} | "
                    f"{col.p_value:.4f} | {severity} |"
                )

            if len(result.drifted_columns) > 20:
                lines.append(f"\n*...and {len(result.drifted_columns) - 20} more columns*")

        return "\n".join(lines)

    def _create_anomaly_markdown(self, result: AnomalyResult) -> str:
        """Create markdown report for AnomalyResult.

        Args:
            result: AnomalyResult to format.

        Returns:
            Markdown formatted report.
        """
        lines = [
            "# Truthound Anomaly Detection Report",
            "",
            f"**Status:** {result.status.name}",
            f"**Anomaly Rate:** {result.anomaly_rate:.1f}%",
            f"**Detector:** {result.detector}",
            "",
            "## Summary",
            "",
            f"- Total Rows: {result.total_row_count:,}",
            f"- Anomalous Rows: {result.anomalous_row_count:,}",
            "",
            f"**Execution Time:** {result.execution_time_ms:.2f}ms",
            "",
        ]

        if result.anomalies:
            lines.extend([
                "## Anomaly Scores",
                "",
                "| Column | Score | Threshold | Anomaly | Detector |",
                "|--------|-------|-----------|---------|----------|",
            ])
            for a in result.anomalies[:20]:
                lines.append(
                    f"| {a.column} | {a.score:.4f} | {a.threshold:.4f} | "
                    f"{'Yes' if a.is_anomaly else 'No'} | {a.detector} |"
                )

            if len(result.anomalies) > 20:
                lines.append(f"\n*...and {len(result.anomalies) - 20} more anomalies*")

        return "\n".join(lines)

    def deserialize(
        self,
        data: dict[str, Any],
        result_type: type[T],
    ) -> T:
        """Deserialize Prefect artifact to result object.

        Args:
            data: Prefect artifact dictionary.
            result_type: Type of result to create.

        Returns:
            Deserialized result object.

        Raises:
            DeserializeError: If deserialization fails.
        """
        try:
            result_data = data.get("data", data)
            return result_type.from_dict(result_data)
        except (KeyError, TypeError, ValueError) as e:
            raise DeserializeError(
                f"Failed to deserialize Prefect artifact to {result_type.__name__}: {e}",
                target_type=result_type.__name__,
                cause=e,
            ) from e


# =============================================================================
# Serializer Factory
# =============================================================================


class SerializerFactory:
    """Factory for creating and managing serializers.

    Provides a centralized way to get serializers by name and
    register custom serializers.

    Example:
        >>> factory = SerializerFactory()
        >>> serializer = factory.get("json")
        >>> json_str = serializer.serialize(result)
        >>>
        >>> # Register custom serializer
        >>> factory.register("custom", CustomSerializer())
    """

    _default_serializers: ClassVar[dict[str, type]] = {
        "json": JSONSerializer,
        "dict": DictSerializer,
        "airflow_xcom": AirflowXComSerializer,
        "dagster": DagsterOutputSerializer,
        "prefect": PrefectArtifactSerializer,
    }

    def __init__(self) -> None:
        """Initialize the factory with default serializers."""
        self._serializers: dict[str, Any] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default serializers."""
        for name, serializer_class in self._default_serializers.items():
            self._serializers[name] = serializer_class()

    def get(self, name: str) -> Any:
        """Get a serializer by name.

        Args:
            name: Serializer name.

        Returns:
            Serializer instance.

        Raises:
            SerializationError: If serializer not found.
        """
        if name not in self._serializers:
            available = ", ".join(sorted(self._serializers.keys()))
            raise SerializationError(
                f"Unknown serializer: {name}. Available: {available}",
                target_type=name,
            )
        return self._serializers[name]

    def register(self, name: str, serializer: Any) -> None:
        """Register a custom serializer.

        Args:
            name: Name to register the serializer under.
            serializer: Serializer instance.
        """
        self._serializers[name] = serializer

    def unregister(self, name: str) -> bool:
        """Unregister a serializer.

        Args:
            name: Serializer name to unregister.

        Returns:
            True if serializer was removed, False if not found.
        """
        if name in self._serializers:
            del self._serializers[name]
            return True
        return False

    def list_serializers(self) -> list[str]:
        """List all registered serializer names.

        Returns:
            List of serializer names.
        """
        return list(self._serializers.keys())

    @classmethod
    def create_for_platform(cls, platform: str) -> Self:
        """Create a factory configured for a specific platform.

        Args:
            platform: Platform name (airflow, dagster, prefect).

        Returns:
            Factory instance with platform-appropriate default.
        """
        _ = platform  # Reserved for future platform-specific configurations
        factory = cls()
        # Set platform-specific defaults or configurations
        return factory


# =============================================================================
# Convenience Functions
# =============================================================================


# Global factory instance for convenience
_default_factory = SerializerFactory()


def serialize_result(
    result: CheckResult | ProfileResult | LearnResult | DriftResult | AnomalyResult,
    format: str = "dict",
) -> Any:
    """Serialize a result using the default factory.

    Args:
        result: Result object to serialize.
        format: Serialization format name.

    Returns:
        Serialized result.

    Raises:
        SerializationError: If serialization fails.
    """
    serializer = _default_factory.get(format)
    return serializer.serialize(result)


def deserialize_result(
    data: Any,
    result_type: type[T],
    format: str = "dict",
) -> T:
    """Deserialize data to a result using the default factory.

    Args:
        data: Serialized data.
        result_type: Type of result to create.
        format: Serialization format name.

    Returns:
        Deserialized result object.

    Raises:
        DeserializeError: If deserialization fails.
    """
    serializer = _default_factory.get(format)
    return serializer.deserialize(data, result_type)


def get_serializer(name: str) -> Any:
    """Get a serializer from the default factory.

    Args:
        name: Serializer name.

    Returns:
        Serializer instance.
    """
    return _default_factory.get(name)


def register_serializer(name: str, serializer: Any) -> None:
    """Register a serializer in the default factory.

    Args:
        name: Serializer name.
        serializer: Serializer instance.
    """
    _default_factory.register(name, serializer)
