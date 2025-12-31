"""Message formatters.

This module provides pluggable message formatters for transforming
notification payloads into channel-specific formats.

Features:
    - Protocol-based formatter abstraction
    - Built-in formatters (text, markdown, JSON, Slack blocks)
    - Template-based formatting
    - Formatter registry for runtime lookup

Example:
    >>> from packages.enterprise.notifications.formatters import (
    ...     MarkdownFormatter,
    ...     get_formatter,
    ...     register_formatter,
    ... )
    >>>
    >>> formatter = get_formatter("markdown")
    >>> formatted = formatter.format(payload)
"""

from __future__ import annotations

import json
import re
import string
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from packages.enterprise.notifications.exceptions import NotificationFormatterError
from packages.enterprise.notifications.types import (
    NotificationLevel,
    NotificationPayload,
)


@runtime_checkable
class MessageFormatter(Protocol):
    """Protocol for message formatters.

    Formatters transform NotificationPayload objects into
    channel-specific message formats.
    """

    @property
    def name(self) -> str:
        """Get the formatter name."""
        ...

    def format(self, payload: NotificationPayload) -> str | dict[str, Any]:
        """Format a notification payload.

        Args:
            payload: Notification payload to format

        Returns:
            Formatted message (string or dict for structured formats)
        """
        ...


class BaseFormatter(ABC):
    """Abstract base class for formatters.

    Provides common functionality for all formatters.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize the formatter.

        Args:
            name: Formatter name (defaults to class name)
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get the formatter name."""
        return self._name

    @abstractmethod
    def format(self, payload: NotificationPayload) -> str | dict[str, Any]:
        """Format a notification payload."""
        ...

    def _get_level_indicator(self, level: NotificationLevel) -> str:
        """Get a text indicator for the notification level."""
        indicators = {
            NotificationLevel.DEBUG: "[DEBUG]",
            NotificationLevel.INFO: "[INFO]",
            NotificationLevel.WARNING: "[WARNING]",
            NotificationLevel.ERROR: "[ERROR]",
            NotificationLevel.CRITICAL: "[CRITICAL]",
        }
        return indicators.get(level, "[INFO]")


class TextFormatter(BaseFormatter):
    """Simple text formatter.

    Produces plain text output suitable for logs or simple notifications.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_context: bool = True,
        include_metadata: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize the text formatter.

        Args:
            include_timestamp: Include timestamp in output
            include_context: Include context data
            include_metadata: Include metadata
            name: Formatter name
        """
        super().__init__(name or "text")
        self._include_timestamp = include_timestamp
        self._include_context = include_context
        self._include_metadata = include_metadata

    def format(self, payload: NotificationPayload) -> str:
        """Format as plain text."""
        lines = []

        # Level indicator
        indicator = self._get_level_indicator(payload.level)

        # Timestamp
        if self._include_timestamp:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            lines.append(f"{indicator} {timestamp}")
        else:
            lines.append(indicator)

        # Title
        if payload.title:
            lines.append(f"Title: {payload.title}")

        # Message
        lines.append(f"Message: {payload.message}")

        # Context
        if self._include_context and payload.context:
            lines.append("Context:")
            for key, value in payload.context:
                lines.append(f"  {key}: {value}")

        # Metadata
        if self._include_metadata:
            if payload.metadata.source:
                lines.append(f"Source: {payload.metadata.source}")
            if payload.metadata.correlation_id:
                lines.append(f"Correlation ID: {payload.metadata.correlation_id}")

        return "\n".join(lines)


class MarkdownFormatter(BaseFormatter):
    """Markdown formatter.

    Produces Markdown-formatted output for rich text displays.
    """

    def __init__(
        self,
        include_context: bool = True,
        include_metadata: bool = True,
        name: str | None = None,
    ) -> None:
        """Initialize the markdown formatter.

        Args:
            include_context: Include context data
            include_metadata: Include metadata
            name: Formatter name
        """
        super().__init__(name or "markdown")
        self._include_context = include_context
        self._include_metadata = include_metadata

    def _get_level_emoji(self, level: NotificationLevel) -> str:
        """Get emoji for notification level."""
        emojis = {
            NotificationLevel.DEBUG: "🔍",
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }
        return emojis.get(level, "ℹ️")

    def format(self, payload: NotificationPayload) -> str:
        """Format as Markdown."""
        lines = []

        emoji = self._get_level_emoji(payload.level)

        # Header
        title = payload.title or f"{payload.level.value.upper()} Notification"
        lines.append(f"## {emoji} {title}")
        lines.append("")

        # Message
        lines.append(payload.message)
        lines.append("")

        # Context
        if self._include_context and payload.context:
            lines.append("### Details")
            lines.append("")
            lines.append("| Key | Value |")
            lines.append("|-----|-------|")
            for key, value in payload.context:
                lines.append(f"| `{key}` | {value} |")
            lines.append("")

        # Metadata
        if self._include_metadata:
            meta_parts = []
            if payload.metadata.source:
                meta_parts.append(f"**Source:** `{payload.metadata.source}`")
            if payload.metadata.correlation_id:
                meta_parts.append(f"**ID:** `{payload.metadata.correlation_id}`")
            meta_parts.append(f"**Level:** `{payload.level.value}`")
            meta_parts.append(
                f"**Time:** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`"
            )

            if meta_parts:
                lines.append("---")
                lines.append(" | ".join(meta_parts))

        return "\n".join(lines)


class SlackBlockFormatter(BaseFormatter):
    """Slack Block Kit formatter.

    Produces Slack Block Kit JSON for rich Slack messages.
    """

    LEVEL_COLORS = {
        NotificationLevel.DEBUG: "#808080",
        NotificationLevel.INFO: "#36a64f",
        NotificationLevel.WARNING: "#ffc107",
        NotificationLevel.ERROR: "#dc3545",
        NotificationLevel.CRITICAL: "#6f42c1",
    }

    LEVEL_EMOJIS = {
        NotificationLevel.DEBUG: ":mag:",
        NotificationLevel.INFO: ":information_source:",
        NotificationLevel.WARNING: ":warning:",
        NotificationLevel.ERROR: ":x:",
        NotificationLevel.CRITICAL: ":rotating_light:",
    }

    def __init__(self, name: str | None = None) -> None:
        """Initialize the Slack block formatter."""
        super().__init__(name or "slack_blocks")

    def format(self, payload: NotificationPayload) -> dict[str, Any]:
        """Format as Slack Block Kit JSON."""
        blocks = []

        emoji = self.LEVEL_EMOJIS.get(payload.level, ":bell:")
        color = self.LEVEL_COLORS.get(payload.level, "#808080")

        # Header
        title = payload.title or f"{payload.level.value.upper()} Notification"
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {title}",
                "emoji": True,
            },
        })

        # Message
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": payload.message,
            },
        })

        # Context fields
        if payload.context:
            fields = []
            for key, value in payload.context:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}",
                })

            for i in range(0, len(fields), 10):
                blocks.append({
                    "type": "section",
                    "fields": fields[i:i + 10],
                })

        # Divider
        blocks.append({"type": "divider"})

        # Footer
        footer_elements = [
            {
                "type": "mrkdwn",
                "text": f"Level: `{payload.level.value}`",
            },
            {
                "type": "mrkdwn",
                "text": f"Time: `{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}`",
            },
        ]

        if payload.metadata.source:
            footer_elements.insert(0, {
                "type": "mrkdwn",
                "text": f"Source: `{payload.metadata.source}`",
            })

        blocks.append({
            "type": "context",
            "elements": footer_elements,
        })

        return {
            "blocks": blocks,
            "attachments": [{"color": color, "fallback": payload.message}],
        }


class JsonFormatter(BaseFormatter):
    """JSON formatter.

    Produces JSON output for structured data systems.
    """

    def __init__(
        self,
        indent: int | None = None,
        include_all: bool = True,
        name: str | None = None,
    ) -> None:
        """Initialize the JSON formatter.

        Args:
            indent: JSON indentation level (None for compact)
            include_all: Include all payload data
            name: Formatter name
        """
        super().__init__(name or "json")
        self._indent = indent
        self._include_all = include_all

    def format(self, payload: NotificationPayload) -> str:
        """Format as JSON string."""
        if self._include_all:
            data = payload.to_dict()
        else:
            data = {
                "message": payload.message,
                "level": payload.level.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if payload.title:
                data["title"] = payload.title

        return json.dumps(data, indent=self._indent, default=str)


class TemplateFormatter(BaseFormatter):
    """Template-based formatter.

    Uses Python string templates for flexible formatting.

    Example:
        >>> formatter = TemplateFormatter(
        ...     template="${level}: ${message}",
        ... )
        >>> formatted = formatter.format(payload)
    """

    def __init__(
        self,
        template: str,
        name: str | None = None,
    ) -> None:
        """Initialize the template formatter.

        Args:
            template: Template string with ${var} placeholders
            name: Formatter name
        """
        super().__init__(name or "template")
        self._template = string.Template(template)

    def format(self, payload: NotificationPayload) -> str:
        """Format using template."""
        # Build substitution dict
        subs = {
            "message": payload.message,
            "level": payload.level.value,
            "level_upper": payload.level.value.upper(),
            "title": payload.title or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": payload.metadata.source or "",
            "correlation_id": payload.metadata.correlation_id or "",
        }

        # Add context items
        for key, value in payload.context:
            subs[f"ctx_{key}"] = str(value)

        try:
            return self._template.safe_substitute(subs)
        except Exception as e:
            raise NotificationFormatterError(
                message=f"Template formatting failed: {e}",
                formatter_name=self.name,
                template=self._template.template,
            ) from e


# =============================================================================
# Formatter Registry
# =============================================================================

_formatters: dict[str, MessageFormatter] = {}
_formatter_lock = threading.Lock()


def register_formatter(formatter: MessageFormatter) -> None:
    """Register a formatter in the global registry.

    Args:
        formatter: Formatter to register
    """
    with _formatter_lock:
        _formatters[formatter.name] = formatter


def get_formatter(name: str) -> MessageFormatter:
    """Get a formatter by name.

    Args:
        name: Formatter name

    Returns:
        Formatter instance

    Raises:
        NotificationFormatterError: If formatter not found
    """
    with _formatter_lock:
        formatter = _formatters.get(name)
        if formatter is None:
            raise NotificationFormatterError(
                message=f"Formatter '{name}' not found",
                formatter_name=name,
            )
        return formatter


def list_formatters() -> list[str]:
    """List all registered formatter names.

    Returns:
        List of formatter names
    """
    with _formatter_lock:
        return list(_formatters.keys())


def reset_formatters() -> None:
    """Reset formatter registry to defaults."""
    global _formatters
    with _formatter_lock:
        _formatters = {}
        _register_defaults()


def _register_defaults() -> None:
    """Register default formatters."""
    defaults = [
        TextFormatter(),
        MarkdownFormatter(),
        SlackBlockFormatter(),
        JsonFormatter(),
    ]
    for formatter in defaults:
        _formatters[formatter.name] = formatter


# Register defaults on module load
_register_defaults()
