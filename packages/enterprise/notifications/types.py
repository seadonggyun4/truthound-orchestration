"""Notification types and enums.

This module defines the core types, enums, and data containers used throughout
the notification system. All types are designed to be immutable and serializable.

Example:
    >>> from packages.enterprise.notifications.types import (
    ...     NotificationChannel,
    ...     NotificationLevel,
    ...     NotificationResult,
    ... )
    >>>
    >>> result = NotificationResult(
    ...     success=True,
    ...     channel=NotificationChannel.SLACK,
    ...     status=NotificationStatus.SENT,
    ...     message_id="msg_123",
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NotificationChannel(str, Enum):
    """Supported notification channels.

    Each channel represents a distinct delivery mechanism for notifications.
    Channels can be extended by implementing the NotificationHandler protocol.
    """

    SLACK = "slack"
    WEBHOOK = "webhook"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    TEAMS = "teams"
    DISCORD = "discord"
    SMS = "sms"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


class NotificationLevel(str, Enum):
    """Notification severity levels.

    Levels are ordered from least to most severe. Handlers may filter
    notifications based on minimum level thresholds.

    Attributes:
        DEBUG: Detailed debugging information
        INFO: General informational messages
        WARNING: Warning conditions that should be noted
        ERROR: Error conditions that require attention
        CRITICAL: Critical conditions requiring immediate action
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value

    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        priorities = {
            NotificationLevel.DEBUG: 0,
            NotificationLevel.INFO: 1,
            NotificationLevel.WARNING: 2,
            NotificationLevel.ERROR: 3,
            NotificationLevel.CRITICAL: 4,
        }
        return priorities[self]

    def __lt__(self, other: NotificationLevel) -> bool:
        if not isinstance(other, NotificationLevel):
            return NotImplemented
        return self.priority < other.priority

    def __le__(self, other: NotificationLevel) -> bool:
        if not isinstance(other, NotificationLevel):
            return NotImplemented
        return self.priority <= other.priority

    def __gt__(self, other: NotificationLevel) -> bool:
        if not isinstance(other, NotificationLevel):
            return NotImplemented
        return self.priority > other.priority

    def __ge__(self, other: NotificationLevel) -> bool:
        if not isinstance(other, NotificationLevel):
            return NotImplemented
        return self.priority >= other.priority


class NotificationStatus(str, Enum):
    """Status of a notification send attempt.

    Tracks the lifecycle of a notification from creation to delivery.
    """

    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    THROTTLED = "throttled"

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal (final) status."""
        return self in {
            NotificationStatus.SENT,
            NotificationStatus.DELIVERED,
            NotificationStatus.FAILED,
            NotificationStatus.SKIPPED,
        }

    @property
    def is_success(self) -> bool:
        """Check if this status indicates success."""
        return self in {
            NotificationStatus.SENT,
            NotificationStatus.DELIVERED,
        }


@dataclass(frozen=True, slots=True)
class NotificationMetadata:
    """Metadata associated with a notification.

    Contains contextual information about the notification source,
    timing, and any custom attributes.

    Attributes:
        source: The component that generated the notification
        correlation_id: ID to correlate related notifications
        trace_id: Distributed tracing ID
        span_id: Distributed tracing span ID
        tags: Custom tags for categorization
        attributes: Additional custom attributes
        created_at: When the notification was created
    """

    source: str | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    tags: frozenset[str] = field(default_factory=frozenset)
    attributes: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def with_source(self, source: str) -> NotificationMetadata:
        """Create a copy with a new source."""
        return NotificationMetadata(
            source=source,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tags=self.tags,
            attributes=self.attributes,
            created_at=self.created_at,
        )

    def with_correlation_id(self, correlation_id: str) -> NotificationMetadata:
        """Create a copy with a new correlation ID."""
        return NotificationMetadata(
            source=self.source,
            correlation_id=correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tags=self.tags,
            attributes=self.attributes,
            created_at=self.created_at,
        )

    def with_tags(self, *tags: str) -> NotificationMetadata:
        """Create a copy with additional tags."""
        return NotificationMetadata(
            source=self.source,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tags=self.tags | frozenset(tags),
            attributes=self.attributes,
            created_at=self.created_at,
        )

    def with_attribute(self, key: str, value: Any) -> NotificationMetadata:
        """Create a copy with an additional attribute."""
        return NotificationMetadata(
            source=self.source,
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            tags=self.tags,
            attributes=self.attributes + ((key, value),),
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tags": list(self.tags),
            "attributes": dict(self.attributes),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotificationMetadata:
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            source=data.get("source"),
            correlation_id=data.get("correlation_id"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            tags=frozenset(data.get("tags", [])),
            attributes=tuple(data.get("attributes", {}).items()),
            created_at=created_at,
        )


@dataclass(frozen=True, slots=True)
class NotificationPayload:
    """Payload for a notification message.

    Contains the message content, level, and any additional context
    needed for formatting and delivery.

    Attributes:
        message: The main notification message
        level: Severity level of the notification
        title: Optional title/subject
        context: Additional context data for formatting
        metadata: Notification metadata
    """

    message: str
    level: NotificationLevel = NotificationLevel.INFO
    title: str | None = None
    context: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    metadata: NotificationMetadata = field(default_factory=NotificationMetadata)

    def with_title(self, title: str) -> NotificationPayload:
        """Create a copy with a title."""
        return NotificationPayload(
            message=self.message,
            level=self.level,
            title=title,
            context=self.context,
            metadata=self.metadata,
        )

    def with_context(self, **kwargs: Any) -> NotificationPayload:
        """Create a copy with additional context."""
        return NotificationPayload(
            message=self.message,
            level=self.level,
            title=self.title,
            context=self.context + tuple(kwargs.items()),
            metadata=self.metadata,
        )

    def with_level(self, level: NotificationLevel) -> NotificationPayload:
        """Create a copy with a different level."""
        return NotificationPayload(
            message=self.message,
            level=level,
            title=self.title,
            context=self.context,
            metadata=self.metadata,
        )

    def with_metadata(self, metadata: NotificationMetadata) -> NotificationPayload:
        """Create a copy with different metadata."""
        return NotificationPayload(
            message=self.message,
            level=self.level,
            title=self.title,
            context=self.context,
            metadata=metadata,
        )

    @property
    def context_dict(self) -> dict[str, Any]:
        """Get context as a dictionary."""
        return dict(self.context)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "level": self.level.value,
            "title": self.title,
            "context": dict(self.context),
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotificationPayload:
        """Create from dictionary."""
        return cls(
            message=data["message"],
            level=NotificationLevel(data.get("level", "info")),
            title=data.get("title"),
            context=tuple(data.get("context", {}).items()),
            metadata=NotificationMetadata.from_dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class NotificationResult:
    """Result of a notification send attempt.

    Contains the outcome of sending a notification to a specific channel,
    including success status, timing, and any error information.

    Attributes:
        success: Whether the notification was sent successfully
        channel: The channel that was used
        status: The final status of the send attempt
        handler_name: Name of the handler that processed this
        message_id: External message ID if available
        timestamp: When the result was recorded
        duration_ms: Time taken to send in milliseconds
        retry_count: Number of retry attempts made
        error: Error message if failed
        error_type: Type of error if failed
        response_data: Raw response data from the channel
    """

    success: bool
    channel: NotificationChannel
    status: NotificationStatus
    handler_name: str | None = None
    message_id: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_ms: float = 0.0
    retry_count: int = 0
    error: str | None = None
    error_type: str | None = None
    response_data: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @classmethod
    def success_result(
        cls,
        channel: NotificationChannel,
        handler_name: str,
        *,
        message_id: str | None = None,
        duration_ms: float = 0.0,
        response_data: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """Create a successful result."""
        return cls(
            success=True,
            channel=channel,
            status=NotificationStatus.SENT,
            handler_name=handler_name,
            message_id=message_id,
            duration_ms=duration_ms,
            response_data=tuple((response_data or {}).items()),
        )

    @classmethod
    def failure_result(
        cls,
        channel: NotificationChannel,
        handler_name: str,
        error: str,
        *,
        error_type: str | None = None,
        duration_ms: float = 0.0,
        retry_count: int = 0,
    ) -> NotificationResult:
        """Create a failed result."""
        return cls(
            success=False,
            channel=channel,
            status=NotificationStatus.FAILED,
            handler_name=handler_name,
            error=error,
            error_type=error_type or "NotificationError",
            duration_ms=duration_ms,
            retry_count=retry_count,
        )

    @classmethod
    def skipped_result(
        cls,
        channel: NotificationChannel,
        handler_name: str,
        reason: str,
    ) -> NotificationResult:
        """Create a skipped result."""
        return cls(
            success=True,  # Skipped is not a failure
            channel=channel,
            status=NotificationStatus.SKIPPED,
            handler_name=handler_name,
            error=reason,
        )

    @property
    def response_dict(self) -> dict[str, Any]:
        """Get response data as a dictionary."""
        return dict(self.response_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "channel": self.channel.value,
            "status": self.status.value,
            "handler_name": self.handler_name,
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "error": self.error,
            "error_type": self.error_type,
            "response_data": dict(self.response_data),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotificationResult:
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            success=data["success"],
            channel=NotificationChannel(data["channel"]),
            status=NotificationStatus(data["status"]),
            handler_name=data.get("handler_name"),
            message_id=data.get("message_id"),
            timestamp=timestamp,
            duration_ms=data.get("duration_ms", 0.0),
            retry_count=data.get("retry_count", 0),
            error=data.get("error"),
            error_type=data.get("error_type"),
            response_data=tuple(data.get("response_data", {}).items()),
        )


@dataclass(frozen=True, slots=True)
class BatchNotificationResult:
    """Result of sending notifications to multiple channels.

    Aggregates results from multiple handlers for a single notification.

    Attributes:
        results: Individual results by handler name
        total_count: Total number of handlers attempted
        success_count: Number of successful sends
        failure_count: Number of failed sends
        skipped_count: Number of skipped sends
        duration_ms: Total time taken
    """

    results: tuple[tuple[str, NotificationResult], ...]
    total_count: int
    success_count: int
    failure_count: int
    skipped_count: int
    duration_ms: float = 0.0

    @property
    def all_success(self) -> bool:
        """Check if all notifications were successful."""
        return self.failure_count == 0

    @property
    def any_success(self) -> bool:
        """Check if any notification was successful."""
        return self.success_count > 0

    @property
    def results_dict(self) -> dict[str, NotificationResult]:
        """Get results as a dictionary."""
        return dict(self.results)

    def get_failures(self) -> list[tuple[str, NotificationResult]]:
        """Get list of failed results."""
        return [
            (name, result)
            for name, result in self.results
            if not result.success and result.status != NotificationStatus.SKIPPED
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": {name: result.to_dict() for name, result in self.results},
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_results(
        cls,
        results: dict[str, NotificationResult],
        duration_ms: float = 0.0,
    ) -> BatchNotificationResult:
        """Create from a dictionary of results."""
        success_count = sum(
            1 for r in results.values()
            if r.success and r.status != NotificationStatus.SKIPPED
        )
        failure_count = sum(
            1 for r in results.values()
            if not r.success
        )
        skipped_count = sum(
            1 for r in results.values()
            if r.status == NotificationStatus.SKIPPED
        )

        return cls(
            results=tuple(results.items()),
            total_count=len(results),
            success_count=success_count,
            failure_count=failure_count,
            skipped_count=skipped_count,
            duration_ms=duration_ms,
        )
