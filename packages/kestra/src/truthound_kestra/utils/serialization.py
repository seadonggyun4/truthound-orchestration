"""Serialization utilities for Kestra data quality integration.

This module provides serializers for converting data quality results
to and from various formats suitable for Kestra task outputs.

Example:
    >>> from truthound_kestra.utils.serialization import (
    ...     ResultSerializer,
    ...     serialize_result,
    ...     deserialize_result,
    ... )
    >>>
    >>> serializer = ResultSerializer()
    >>> data = serializer.serialize_check_result(result)
    >>> restored = serializer.deserialize_check_result(data)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from truthound_kestra.utils.exceptions import SerializationError
from truthound_kestra.utils.types import (
    CheckStatus,
    ColumnProfile,
    LearnedRule,
    OperationType,
    OutputFormat,
    ScriptOutput,
    ValidationFailure,
)

if TYPE_CHECKING:
    pass

__all__ = [
    # Protocols
    "Serializable",
    "ResultSerializerProtocol",
    # Classes
    "ResultSerializer",
    "JsonSerializer",
    "YamlSerializer",
    "MarkdownSerializer",
    # Functions
    "serialize_result",
    "deserialize_result",
    "serialize_to_format",
    "get_serializer",
]

# Serializer version for compatibility tracking
_SERIALIZER_VERSION = "1.0"


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to dict."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...


@runtime_checkable
class ResultSerializerProtocol(Protocol):
    """Protocol for result serializers."""

    def serialize_check_result(self, result: Any) -> dict[str, Any]:
        """Serialize check result."""
        ...

    def deserialize_check_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize check result."""
        ...


@dataclass(frozen=True, slots=True)
class SerializerConfig:
    """Configuration for result serialization.

    Attributes:
        include_metadata: Whether to include serializer metadata.
        include_timestamps: Whether to include timestamps.
        timestamp_format: Format for timestamp serialization.
        compact: Whether to produce compact output.
        max_failures: Maximum number of failures to include (0 = all).
        max_sample_rows: Maximum sample row indices per failure.

    Example:
        >>> config = SerializerConfig(
        ...     include_metadata=True,
        ...     compact=False,
        ...     max_failures=100
        ... )
    """

    include_metadata: bool = True
    include_timestamps: bool = True
    timestamp_format: str = "iso"
    compact: bool = False
    max_failures: int = 0
    max_sample_rows: int = 10

    def with_compact(self, compact: bool) -> SerializerConfig:
        """Return new config with updated compact setting."""
        return SerializerConfig(
            include_metadata=self.include_metadata,
            include_timestamps=self.include_timestamps,
            timestamp_format=self.timestamp_format,
            compact=compact,
            max_failures=self.max_failures,
            max_sample_rows=self.max_sample_rows,
        )

    def with_max_failures(self, max_failures: int) -> SerializerConfig:
        """Return new config with updated max_failures."""
        return SerializerConfig(
            include_metadata=self.include_metadata,
            include_timestamps=self.include_timestamps,
            timestamp_format=self.timestamp_format,
            compact=self.compact,
            max_failures=max_failures,
            max_sample_rows=self.max_sample_rows,
        )


# Preset configurations
DEFAULT_SERIALIZER_CONFIG = SerializerConfig()
COMPACT_SERIALIZER_CONFIG = SerializerConfig(
    include_metadata=False,
    compact=True,
    max_failures=10,
    max_sample_rows=5,
)
FULL_SERIALIZER_CONFIG = SerializerConfig(
    include_metadata=True,
    include_timestamps=True,
    compact=False,
    max_failures=0,
    max_sample_rows=100,
)


class ResultSerializer:
    """Serializer for data quality results.

    This class provides methods to serialize and deserialize
    data quality results for Kestra task outputs.

    Attributes:
        config: Serialization configuration.

    Example:
        >>> serializer = ResultSerializer()
        >>> data = serializer.serialize_check_result(result)
        >>> restored = serializer.deserialize_check_result(data)
    """

    def __init__(self, config: SerializerConfig | None = None) -> None:
        """Initialize serializer with optional configuration.

        Args:
            config: Optional serialization configuration.
        """
        self._config = config or DEFAULT_SERIALIZER_CONFIG

    @property
    def config(self) -> SerializerConfig:
        """Get serializer configuration."""
        return self._config

    def serialize_check_result(self, result: Any) -> dict[str, Any]:
        """Serialize a check result to dictionary.

        Args:
            result: Check result to serialize. Can be:
                - dict with result data
                - Object with to_dict() method
                - ScriptOutput instance

        Returns:
            Dictionary containing serialized result.

        Raises:
            SerializationError: If serialization fails.
        """
        try:
            if isinstance(result, ScriptOutput):
                base = result.to_dict()
            elif isinstance(result, dict):
                base = dict(result)
            elif hasattr(result, "to_dict"):
                base = result.to_dict()
            else:
                base = self._serialize_check_result_attrs(result)

            # Truncate failures if configured
            if self._config.max_failures > 0 and "failures" in base:
                base["failures"] = base["failures"][: self._config.max_failures]
                base["failures_truncated"] = len(base.get("failures", [])) > self._config.max_failures

            # Truncate sample indices in failures
            if "failures" in base and self._config.max_sample_rows > 0:
                for failure in base.get("failures", []):
                    if "failed_indices" in failure:
                        failure["failed_indices"] = failure["failed_indices"][
                            : self._config.max_sample_rows
                        ]

            # Add metadata
            if self._config.include_metadata:
                base["_serializer"] = "truthound_kestra"
                base["_version"] = _SERIALIZER_VERSION
                base["_type"] = "check_result"

            # Normalize timestamps
            if self._config.include_timestamps:
                base = self._normalize_timestamps(base)

            return base

        except Exception as e:
            raise SerializationError(
                message=f"Failed to serialize check result: {e}",
                format="dict",
                direction="serialize",
            ) from e

    def deserialize_check_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize a check result from dictionary.

        Args:
            data: Dictionary containing serialized result.

        Returns:
            Deserialized result dictionary.

        Raises:
            SerializationError: If deserialization fails.
        """
        try:
            result = dict(data)

            # Parse timestamps
            result = self._parse_timestamps(result)

            # Remove metadata
            result.pop("_serializer", None)
            result.pop("_version", None)
            result.pop("_type", None)

            return result

        except Exception as e:
            raise SerializationError(
                message=f"Failed to deserialize check result: {e}",
                format="dict",
                direction="deserialize",
            ) from e

    def serialize_profile_result(self, result: Any) -> dict[str, Any]:
        """Serialize a profile result to dictionary.

        Args:
            result: Profile result to serialize.

        Returns:
            Dictionary containing serialized result.
        """
        try:
            if isinstance(result, ScriptOutput):
                base = result.to_dict()
            elif isinstance(result, dict):
                base = dict(result)
            elif hasattr(result, "to_dict"):
                base = result.to_dict()
            else:
                base = self._serialize_profile_result_attrs(result)

            if self._config.include_metadata:
                base["_serializer"] = "truthound_kestra"
                base["_version"] = _SERIALIZER_VERSION
                base["_type"] = "profile_result"

            if self._config.include_timestamps:
                base = self._normalize_timestamps(base)

            return base

        except Exception as e:
            raise SerializationError(
                message=f"Failed to serialize profile result: {e}",
                format="dict",
                direction="serialize",
            ) from e

    def serialize_learn_result(self, result: Any) -> dict[str, Any]:
        """Serialize a learn result to dictionary.

        Args:
            result: Learn result to serialize.

        Returns:
            Dictionary containing serialized result.
        """
        try:
            if isinstance(result, ScriptOutput):
                base = result.to_dict()
            elif isinstance(result, dict):
                base = dict(result)
            elif hasattr(result, "to_dict"):
                base = result.to_dict()
            else:
                base = self._serialize_learn_result_attrs(result)

            if self._config.include_metadata:
                base["_serializer"] = "truthound_kestra"
                base["_version"] = _SERIALIZER_VERSION
                base["_type"] = "learn_result"

            if self._config.include_timestamps:
                base = self._normalize_timestamps(base)

            return base

        except Exception as e:
            raise SerializationError(
                message=f"Failed to serialize learn result: {e}",
                format="dict",
                direction="serialize",
            ) from e

    def _serialize_check_result_attrs(self, result: Any) -> dict[str, Any]:
        """Serialize check result from object attributes."""
        return {
            "status": getattr(result, "status", "unknown"),
            "passed_count": getattr(result, "passed_count", 0),
            "failed_count": getattr(result, "failed_count", 0),
            "warning_count": getattr(result, "warning_count", 0),
            "total_rows": getattr(result, "total_rows", 0),
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "failures": self._serialize_failures(
                getattr(result, "failures", [])
            ),
            "metadata": getattr(result, "metadata", {}),
        }

    def _serialize_profile_result_attrs(self, result: Any) -> dict[str, Any]:
        """Serialize profile result from object attributes."""
        columns = getattr(result, "columns", [])
        return {
            "status": getattr(result, "status", "unknown"),
            "row_count": getattr(result, "row_count", 0),
            "column_count": len(columns),
            "columns": [
                c.to_dict() if hasattr(c, "to_dict") else dict(c)
                for c in columns
            ],
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "metadata": getattr(result, "metadata", {}),
        }

    def _serialize_learn_result_attrs(self, result: Any) -> dict[str, Any]:
        """Serialize learn result from object attributes."""
        rules = getattr(result, "rules", [])
        return {
            "status": getattr(result, "status", "unknown"),
            "rule_count": len(rules),
            "rules": [
                r.to_dict() if hasattr(r, "to_dict") else dict(r)
                for r in rules
            ],
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
            "metadata": getattr(result, "metadata", {}),
        }

    def _serialize_failures(self, failures: list[Any]) -> list[dict[str, Any]]:
        """Serialize list of failures."""
        result = []
        for f in failures:
            if hasattr(f, "to_dict"):
                result.append(f.to_dict())
            elif isinstance(f, dict):
                result.append(f)
            else:
                result.append({
                    "rule_type": getattr(f, "rule_type", "unknown"),
                    "column": getattr(f, "column", "unknown"),
                    "message": getattr(f, "message", str(f)),
                    "severity": getattr(f, "severity", "medium"),
                    "failed_count": getattr(f, "failed_count", 0),
                })
        return result

    def _normalize_timestamps(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert datetime objects to ISO format strings."""
        result = dict(data)
        for key, value in result.items():
            if isinstance(value, datetime):
                if self._config.timestamp_format == "iso":
                    result[key] = value.isoformat()
                elif self._config.timestamp_format == "unix":
                    result[key] = value.timestamp()
            elif isinstance(value, dict):
                result[key] = self._normalize_timestamps(value)
            elif isinstance(value, list):
                result[key] = [
                    self._normalize_timestamps(item) if isinstance(item, dict) else item
                    for item in value
                ]
        return result

    def _parse_timestamps(self, data: dict[str, Any]) -> dict[str, Any]:
        """Parse ISO format strings to datetime objects."""
        result = dict(data)
        timestamp_keys = {"timestamp", "trigger_date", "created_at", "updated_at"}

        for key, value in result.items():
            if key in timestamp_keys and isinstance(value, str):
                try:
                    result[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Keep as string if parsing fails
            elif isinstance(value, dict):
                result[key] = self._parse_timestamps(value)
            elif isinstance(value, list):
                result[key] = [
                    self._parse_timestamps(item) if isinstance(item, dict) else item
                    for item in value
                ]
        return result


class BaseFormatSerializer(ABC):
    """Base class for format-specific serializers."""

    def __init__(self, config: SerializerConfig | None = None) -> None:
        """Initialize with optional configuration."""
        self._config = config or DEFAULT_SERIALIZER_CONFIG
        self._result_serializer = ResultSerializer(config)

    @property
    @abstractmethod
    def format(self) -> OutputFormat:
        """Get the output format."""
        ...

    @abstractmethod
    def serialize(self, data: dict[str, Any]) -> str:
        """Serialize dictionary to format-specific string."""
        ...

    @abstractmethod
    def deserialize(self, content: str) -> dict[str, Any]:
        """Deserialize format-specific string to dictionary."""
        ...


class JsonSerializer(BaseFormatSerializer):
    """JSON format serializer."""

    @property
    def format(self) -> OutputFormat:
        """Get the output format."""
        return OutputFormat.JSON

    def serialize(self, data: dict[str, Any]) -> str:
        """Serialize dictionary to JSON string.

        Args:
            data: Dictionary to serialize.

        Returns:
            JSON string.

        Raises:
            SerializationError: If serialization fails.
        """
        try:
            indent = None if self._config.compact else 2
            return json.dumps(data, indent=indent, default=str, ensure_ascii=False)
        except Exception as e:
            raise SerializationError(
                message=f"Failed to serialize to JSON: {e}",
                format="json",
                direction="serialize",
            ) from e

    def deserialize(self, content: str) -> dict[str, Any]:
        """Deserialize JSON string to dictionary.

        Args:
            content: JSON string to deserialize.

        Returns:
            Dictionary.

        Raises:
            SerializationError: If deserialization fails.
        """
        try:
            return json.loads(content)
        except Exception as e:
            raise SerializationError(
                message=f"Failed to deserialize JSON: {e}",
                format="json",
                direction="deserialize",
            ) from e


class YamlSerializer(BaseFormatSerializer):
    """YAML format serializer."""

    @property
    def format(self) -> OutputFormat:
        """Get the output format."""
        return OutputFormat.YAML

    def serialize(self, data: dict[str, Any]) -> str:
        """Serialize dictionary to YAML string.

        Args:
            data: Dictionary to serialize.

        Returns:
            YAML string.

        Raises:
            SerializationError: If serialization fails.
        """
        try:
            import yaml

            return yaml.dump(
                data,
                default_flow_style=self._config.compact,
                allow_unicode=True,
                sort_keys=False,
            )
        except ImportError as e:
            raise SerializationError(
                message="PyYAML is required for YAML serialization",
                format="yaml",
                direction="serialize",
            ) from e
        except Exception as e:
            raise SerializationError(
                message=f"Failed to serialize to YAML: {e}",
                format="yaml",
                direction="serialize",
            ) from e

    def deserialize(self, content: str) -> dict[str, Any]:
        """Deserialize YAML string to dictionary.

        Args:
            content: YAML string to deserialize.

        Returns:
            Dictionary.

        Raises:
            SerializationError: If deserialization fails.
        """
        try:
            import yaml

            return yaml.safe_load(content)
        except ImportError as e:
            raise SerializationError(
                message="PyYAML is required for YAML deserialization",
                format="yaml",
                direction="deserialize",
            ) from e
        except Exception as e:
            raise SerializationError(
                message=f"Failed to deserialize YAML: {e}",
                format="yaml",
                direction="deserialize",
            ) from e


class MarkdownSerializer(BaseFormatSerializer):
    """Markdown format serializer for human-readable reports."""

    @property
    def format(self) -> OutputFormat:
        """Get the output format."""
        return OutputFormat.MARKDOWN

    def serialize(self, data: dict[str, Any]) -> str:
        """Serialize dictionary to Markdown report.

        Args:
            data: Dictionary to serialize.

        Returns:
            Markdown string.
        """
        lines = []
        result_type = data.get("_type", "result")

        # Header
        status = data.get("status", "unknown")
        status_emoji = self._get_status_emoji(status)
        lines.append(f"# Data Quality Report {status_emoji}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Status:** {status}")
        if "passed_count" in data:
            lines.append(f"- **Passed:** {data['passed_count']}")
        if "failed_count" in data:
            lines.append(f"- **Failed:** {data['failed_count']}")
        if "total_rows" in data:
            lines.append(f"- **Total Rows:** {data['total_rows']}")
        if "execution_time_ms" in data:
            lines.append(f"- **Execution Time:** {data['execution_time_ms']:.2f}ms")
        lines.append("")

        # Failures section (for check results)
        failures = data.get("failures", [])
        if failures:
            lines.append("## Validation Failures")
            lines.append("")
            lines.append("| Column | Rule | Message | Severity |")
            lines.append("|--------|------|---------|----------|")
            for failure in failures[: self._config.max_failures or len(failures)]:
                col = failure.get("column", "-")
                rule = failure.get("rule_type", "-")
                msg = failure.get("message", "-")
                sev = failure.get("severity", "-")
                lines.append(f"| {col} | {rule} | {msg} | {sev} |")
            lines.append("")

        # Columns section (for profile results)
        columns = data.get("columns", [])
        if columns:
            lines.append("## Column Profiles")
            lines.append("")
            lines.append("| Column | Type | Nulls | Unique |")
            lines.append("|--------|------|-------|--------|")
            for col in columns:
                name = col.get("column_name", "-")
                dtype = col.get("dtype", "-")
                null_pct = col.get("null_percentage", 0)
                unique_pct = col.get("unique_percentage", 0)
                lines.append(f"| {name} | {dtype} | {null_pct:.1f}% | {unique_pct:.1f}% |")
            lines.append("")

        # Rules section (for learn results)
        rules = data.get("rules", [])
        if rules:
            lines.append("## Learned Rules")
            lines.append("")
            lines.append("| Column | Rule Type | Confidence |")
            lines.append("|--------|-----------|------------|")
            for rule in rules:
                col = rule.get("column", "-")
                rule_type = rule.get("rule_type", "-")
                conf = rule.get("confidence", 0)
                lines.append(f"| {col} | {rule_type} | {conf:.2f} |")
            lines.append("")

        # Footer
        timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())
        lines.append("---")
        lines.append(f"*Generated at: {timestamp}*")

        return "\n".join(lines)

    def deserialize(self, content: str) -> dict[str, Any]:
        """Deserialize Markdown to dictionary (not fully supported).

        Note: Markdown deserialization is limited as it's primarily
        a presentation format.

        Args:
            content: Markdown string.

        Returns:
            Basic dictionary with content.
        """
        return {"_raw_markdown": content, "_type": "markdown"}

    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        emoji_map = {
            "passed": "✅",
            "failed": "❌",
            "warning": "⚠️",
            "skipped": "⏭️",
            "error": "💥",
        }
        return emoji_map.get(status.lower(), "❓")


# Module-level default serializer
_default_serializer = ResultSerializer()

# Serializer registry
_serializer_registry: dict[OutputFormat, type[BaseFormatSerializer]] = {
    OutputFormat.JSON: JsonSerializer,
    OutputFormat.YAML: YamlSerializer,
    OutputFormat.MARKDOWN: MarkdownSerializer,
}


def get_serializer(
    format: OutputFormat | str,
    config: SerializerConfig | None = None,
) -> BaseFormatSerializer:
    """Get a serializer for the specified format.

    Args:
        format: Output format (json, yaml, markdown, csv).
        config: Optional serializer configuration.

    Returns:
        Format-specific serializer.

    Raises:
        ValueError: If format is not supported.

    Example:
        >>> serializer = get_serializer("json")
        >>> output = serializer.serialize(data)
    """
    if isinstance(format, str):
        format = OutputFormat(format.lower())

    if format not in _serializer_registry:
        raise ValueError(f"Unsupported format: {format}")

    return _serializer_registry[format](config)


def serialize_result(
    result: Any,
    result_type: str = "check",
) -> dict[str, Any]:
    """Serialize a result using the default serializer.

    Args:
        result: Result to serialize.
        result_type: Type of result ('check', 'profile', 'learn').

    Returns:
        Serialized dictionary.

    Example:
        >>> data = serialize_result(check_result, "check")
    """
    if result_type == "check":
        return _default_serializer.serialize_check_result(result)
    elif result_type == "profile":
        return _default_serializer.serialize_profile_result(result)
    elif result_type == "learn":
        return _default_serializer.serialize_learn_result(result)
    else:
        return _default_serializer.serialize_check_result(result)


def deserialize_result(data: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a result using the default serializer.

    Args:
        data: Serialized result dictionary.

    Returns:
        Deserialized dictionary.

    Example:
        >>> result = deserialize_result(data)
    """
    return _default_serializer.deserialize_check_result(data)


def serialize_to_format(
    result: Any,
    format: OutputFormat | str = OutputFormat.JSON,
    config: SerializerConfig | None = None,
) -> str:
    """Serialize result to a specific format string.

    Args:
        result: Result to serialize.
        format: Output format.
        config: Optional serializer configuration.

    Returns:
        Formatted string.

    Example:
        >>> json_str = serialize_to_format(result, "json")
        >>> yaml_str = serialize_to_format(result, "yaml")
    """
    serializer = get_serializer(format, config)
    data = serialize_result(result)
    return serializer.serialize(data)
