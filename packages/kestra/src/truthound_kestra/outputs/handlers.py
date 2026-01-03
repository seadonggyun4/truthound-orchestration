"""Kestra output handlers for data quality results.

This module provides handlers for formatting and sending data quality
results to Kestra's output system.

Example:
    >>> from truthound_kestra.outputs.handlers import (
    ...     KestraOutputHandler,
    ...     send_check_result,
    ... )
    >>>
    >>> handler = KestraOutputHandler()
    >>> handler.send_check_result(result)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from truthound_kestra.utils.exceptions import OutputError
from truthound_kestra.utils.helpers import (
    format_duration,
    format_percentage,
    get_logger,
)
from truthound_kestra.utils.serialization import (
    MarkdownSerializer,
    ResultSerializer,
    serialize_result,
)
from truthound_kestra.utils.types import (
    CheckStatus,
    OperationType,
    OutputFormat,
    ScriptOutput,
)

__all__ = [
    # Protocols
    "OutputHandlerProtocol",
    # Classes
    "KestraOutputHandler",
    "FileOutputHandler",
    "MultiOutputHandler",
    # Config
    "OutputConfig",
    # Functions
    "send_check_result",
    "send_profile_result",
    "send_learn_result",
    "send_outputs",
]

logger = get_logger(__name__)


@runtime_checkable
class OutputHandlerProtocol(Protocol):
    """Protocol for output handlers."""

    def send(self, outputs: dict[str, Any]) -> None:
        """Send outputs."""
        ...


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Configuration for output handling.

    Attributes:
        format: Output format (json, yaml, markdown).
        include_summary: Whether to include a summary.
        include_failures: Whether to include detailed failures.
        max_failures: Maximum number of failures to include.
        include_metadata: Whether to include metadata.
        file_output: Whether to also write to file.
        file_path: Path for file output.

    Example:
        >>> config = OutputConfig(
        ...     format=OutputFormat.JSON,
        ...     include_summary=True,
        ...     max_failures=50
        ... )
    """

    format: OutputFormat = OutputFormat.JSON
    include_summary: bool = True
    include_failures: bool = True
    max_failures: int = 100
    include_metadata: bool = True
    file_output: bool = False
    file_path: str | None = None

    def with_format(self, format: OutputFormat | str) -> OutputConfig:
        """Return new config with updated format."""
        if isinstance(format, str):
            format = OutputFormat(format.lower())
        return OutputConfig(
            format=format,
            include_summary=self.include_summary,
            include_failures=self.include_failures,
            max_failures=self.max_failures,
            include_metadata=self.include_metadata,
            file_output=self.file_output,
            file_path=self.file_path,
        )

    def with_file_output(self, path: str) -> OutputConfig:
        """Return new config with file output enabled."""
        return OutputConfig(
            format=self.format,
            include_summary=self.include_summary,
            include_failures=self.include_failures,
            max_failures=self.max_failures,
            include_metadata=self.include_metadata,
            file_output=True,
            file_path=path,
        )


class BaseOutputHandler(ABC):
    """Base class for output handlers."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        """Initialize handler with optional configuration.

        Args:
            config: Optional output configuration.
        """
        self._config = config or OutputConfig()
        self._serializer = ResultSerializer()

    @property
    def config(self) -> OutputConfig:
        """Get output configuration."""
        return self._config

    @abstractmethod
    def send(self, outputs: dict[str, Any]) -> None:
        """Send outputs to the destination.

        Args:
            outputs: Dictionary of outputs to send.
        """
        ...

    def format_check_result(
        self,
        result: ScriptOutput | dict[str, Any],
    ) -> dict[str, Any]:
        """Format a check result for output.

        Args:
            result: Check result to format.

        Returns:
            Formatted output dictionary.
        """
        if isinstance(result, ScriptOutput):
            data = result.to_dict()
        else:
            data = dict(result)

        output: dict[str, Any] = {
            "status": data.get("status", "unknown"),
            "is_success": data.get("is_success", False),
            "operation": data.get("operation", "check"),
        }

        if self._config.include_summary:
            output["summary"] = self._build_summary(data)

        if self._config.include_failures and data.get("failures"):
            failures = data["failures"]
            if self._config.max_failures > 0:
                failures = failures[: self._config.max_failures]
            output["failures"] = failures
            output["failures_count"] = len(data.get("failures", []))

        if self._config.include_metadata:
            output["metadata"] = data.get("metadata", {})
            output["timestamp"] = data.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat()
            )

        return output

    def _build_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        """Build a summary from result data."""
        passed = data.get("passed_count", 0)
        failed = data.get("failed_count", 0)
        total = passed + failed
        pass_rate = passed / total if total > 0 else 1.0

        return {
            "passed_count": passed,
            "failed_count": failed,
            "warning_count": data.get("warning_count", 0),
            "total_count": total,
            "total_rows": data.get("total_rows", 0),
            "pass_rate": pass_rate,
            "pass_rate_formatted": format_percentage(pass_rate),
            "execution_time": format_duration(
                data.get("execution_time_ms", 0)
            ),
        }


class KestraOutputHandler(BaseOutputHandler):
    """Handler for sending outputs to Kestra.

    This handler formats results and sends them to Kestra's output
    system using the Kestra Python SDK.

    Example:
        >>> handler = KestraOutputHandler()
        >>> handler.send_check_result(result)
    """

    def send(self, outputs: dict[str, Any]) -> None:
        """Send outputs to Kestra.

        Args:
            outputs: Dictionary of outputs to send.
        """
        try:
            from kestra import Kestra

            Kestra.outputs(outputs)
            logger.debug(f"Sent {len(outputs)} outputs to Kestra")
        except ImportError:
            # Fallback: print as JSON for Kestra to parse
            print(f"::outputs:: {json.dumps(outputs, default=str)}")
            logger.debug(f"Printed {len(outputs)} outputs as JSON")
        except Exception as e:
            raise OutputError(
                message=f"Failed to send outputs to Kestra: {e}",
                output_name=", ".join(outputs.keys()),
                output_type="kestra",
            ) from e

    def send_check_result(
        self,
        result: ScriptOutput | dict[str, Any],
        output_name: str = "check_result",
    ) -> None:
        """Send a check result to Kestra.

        Args:
            result: Check result to send.
            output_name: Name for the output.
        """
        formatted = self.format_check_result(result)
        self.send({
            output_name: formatted,
            "status": formatted.get("status"),
            "is_success": formatted.get("is_success"),
        })

    def send_profile_result(
        self,
        result: ScriptOutput | dict[str, Any],
        output_name: str = "profile_result",
    ) -> None:
        """Send a profile result to Kestra.

        Args:
            result: Profile result to send.
            output_name: Name for the output.
        """
        if isinstance(result, ScriptOutput):
            data = result.to_dict()
        else:
            data = dict(result)

        output = {
            "status": data.get("status", "passed"),
            "columns": data.get("columns", []),
            "column_count": len(data.get("columns", [])),
            "row_count": data.get("total_rows", 0),
        }

        if self._config.include_metadata:
            output["metadata"] = data.get("metadata", {})
            output["timestamp"] = data.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat()
            )

        self.send({output_name: output})

    def send_learn_result(
        self,
        result: ScriptOutput | dict[str, Any],
        output_name: str = "learn_result",
    ) -> None:
        """Send a learn result to Kestra.

        Args:
            result: Learn result to send.
            output_name: Name for the output.
        """
        if isinstance(result, ScriptOutput):
            data = result.to_dict()
        else:
            data = dict(result)

        rules = data.get("rules", [])

        output = {
            "status": data.get("status", "passed"),
            "rules": rules,
            "rule_count": len(rules),
            "learned_rules": [
                {"type": r.get("rule_type"), "column": r.get("column"), **r.get("parameters", {})}
                for r in rules
            ],
        }

        if self._config.include_metadata:
            output["metadata"] = data.get("metadata", {})
            output["timestamp"] = data.get(
                "timestamp",
                datetime.now(timezone.utc).isoformat()
            )

        self.send({output_name: output, "learned_rules": output["learned_rules"]})

    def send_metric(
        self,
        name: str,
        value: float | int,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Send a metric to Kestra.

        Args:
            name: Metric name.
            value: Metric value.
            tags: Optional tags.
        """
        try:
            from kestra import Kestra

            Kestra.counter(name, value, tags or {})
        except ImportError:
            # Fallback
            metric = {"name": name, "value": value, "tags": tags or {}}
            print(f"::metric:: {json.dumps(metric)}")


class FileOutputHandler(BaseOutputHandler):
    """Handler for writing outputs to files.

    Example:
        >>> handler = FileOutputHandler(
        ...     config=OutputConfig(file_path="/tmp/result.json")
        ... )
        >>> handler.send_check_result(result)
    """

    def send(self, outputs: dict[str, Any]) -> None:
        """Write outputs to file.

        Args:
            outputs: Dictionary of outputs to write.
        """
        if not self._config.file_path:
            raise OutputError(
                message="File path not configured",
                output_type="file",
            )

        try:
            content: str

            if self._config.format == OutputFormat.JSON:
                content = json.dumps(outputs, indent=2, default=str)
            elif self._config.format == OutputFormat.YAML:
                import yaml
                content = yaml.dump(outputs, default_flow_style=False)
            elif self._config.format == OutputFormat.MARKDOWN:
                serializer = MarkdownSerializer()
                content = serializer.serialize(outputs)
            else:
                content = json.dumps(outputs, default=str)

            with open(self._config.file_path, "w") as f:
                f.write(content)

            logger.debug(f"Wrote outputs to {self._config.file_path}")

        except Exception as e:
            raise OutputError(
                message=f"Failed to write outputs to file: {e}",
                output_name=self._config.file_path,
                output_type="file",
            ) from e

    def send_check_result(
        self,
        result: ScriptOutput | dict[str, Any],
        output_name: str = "check_result",
    ) -> None:
        """Write a check result to file.

        Args:
            result: Check result to write.
            output_name: Name for the output (used as key).
        """
        formatted = self.format_check_result(result)
        self.send({output_name: formatted})


class MultiOutputHandler(BaseOutputHandler):
    """Handler that sends outputs to multiple destinations.

    Example:
        >>> kestra_handler = KestraOutputHandler()
        >>> file_handler = FileOutputHandler(
        ...     config=OutputConfig(file_path="/tmp/result.json")
        ... )
        >>> handler = MultiOutputHandler([kestra_handler, file_handler])
        >>> handler.send_check_result(result)
    """

    def __init__(
        self,
        handlers: list[BaseOutputHandler],
        config: OutputConfig | None = None,
    ) -> None:
        """Initialize with multiple handlers.

        Args:
            handlers: List of output handlers.
            config: Optional shared configuration.
        """
        super().__init__(config)
        self._handlers = handlers

    def send(self, outputs: dict[str, Any]) -> None:
        """Send outputs to all handlers.

        Args:
            outputs: Dictionary of outputs to send.
        """
        errors = []

        for handler in self._handlers:
            try:
                handler.send(outputs)
            except Exception as e:
                errors.append((handler, e))
                logger.warning(f"Handler {type(handler).__name__} failed: {e}")

        if errors and len(errors) == len(self._handlers):
            # All handlers failed
            raise OutputError(
                message=f"All output handlers failed: {[str(e) for _, e in errors]}",
                output_type="multi",
            )

    def send_check_result(
        self,
        result: ScriptOutput | dict[str, Any],
        output_name: str = "check_result",
    ) -> None:
        """Send a check result to all handlers.

        Args:
            result: Check result to send.
            output_name: Name for the output.
        """
        formatted = self.format_check_result(result)
        self.send({output_name: formatted})


# Module-level handler
_default_handler = KestraOutputHandler()


def send_outputs(outputs: dict[str, Any]) -> None:
    """Send outputs using the default handler.

    Args:
        outputs: Dictionary of outputs to send.

    Example:
        >>> send_outputs({"result": data, "status": "passed"})
    """
    _default_handler.send(outputs)


def send_check_result(
    result: ScriptOutput | dict[str, Any],
    output_name: str = "check_result",
) -> None:
    """Send a check result using the default handler.

    Args:
        result: Check result to send.
        output_name: Name for the output.

    Example:
        >>> send_check_result(script_output)
    """
    _default_handler.send_check_result(result, output_name)


def send_profile_result(
    result: ScriptOutput | dict[str, Any],
    output_name: str = "profile_result",
) -> None:
    """Send a profile result using the default handler.

    Args:
        result: Profile result to send.
        output_name: Name for the output.

    Example:
        >>> send_profile_result(script_output)
    """
    _default_handler.send_profile_result(result, output_name)


def send_learn_result(
    result: ScriptOutput | dict[str, Any],
    output_name: str = "learn_result",
) -> None:
    """Send a learn result using the default handler.

    Args:
        result: Learn result to send.
        output_name: Name for the output.

    Example:
        >>> send_learn_result(script_output)
    """
    _default_handler.send_learn_result(result, output_name)
