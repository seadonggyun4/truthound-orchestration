"""Notification exceptions.

This module defines the exception hierarchy for the notification system.
All exceptions inherit from NotificationError and include rich context
for debugging and error handling.

Exception Hierarchy:
    NotificationError (base)
    ├── NotificationConfigError (configuration issues)
    ├── NotificationSendError (send failures)
    │   ├── NotificationTimeoutError (timeouts)
    │   └── NotificationRetryExhaustedError (retry exhaustion)
    ├── NotificationHandlerNotFoundError (handler lookup failures)
    └── NotificationFormatterError (formatting issues)

Example:
    >>> try:
    ...     await handler.send(payload)
    ... except NotificationTimeoutError as e:
    ...     print(f"Timeout after {e.timeout_seconds}s for {e.handler_name}")
    ... except NotificationSendError as e:
    ...     print(f"Send failed: {e.message}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
)


@dataclass
class NotificationError(Exception):
    """Base exception for all notification errors.

    Attributes:
        message: Human-readable error message
        handler_name: Name of the handler that raised the error
        channel: Notification channel involved
        metadata: Additional error context
    """

    message: str
    handler_name: str | None = None
    channel: NotificationChannel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [self.message]
        if self.handler_name:
            parts.append(f"handler={self.handler_name}")
        if self.channel:
            parts.append(f"channel={self.channel.value}")
        return " | ".join(parts)

    def __post_init__(self) -> None:
        super().__init__(str(self))

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "handler_name": self.handler_name,
            "channel": self.channel.value if self.channel else None,
            "metadata": self.metadata,
        }


@dataclass
class NotificationConfigError(NotificationError):
    """Exception raised for configuration errors.

    Raised when notification configuration is invalid, missing required
    fields, or contains incompatible settings.

    Attributes:
        field_name: Name of the invalid configuration field
        field_value: The invalid value (if safe to include)
        reason: Explanation of why the value is invalid
    """

    field_name: str | None = None
    field_value: Any = None
    reason: str | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.field_name:
            parts.append(f"field={self.field_name}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        return " | ".join(parts)


@dataclass
class NotificationSendError(NotificationError):
    """Exception raised when sending a notification fails.

    This is the primary exception for send failures. More specific
    exceptions (timeout, retry exhausted) inherit from this.

    Attributes:
        original_error: The underlying exception that caused the failure
        response_status: HTTP status code if applicable
        response_body: Response body if applicable
        attempt_count: Number of attempts made
    """

    original_error: Exception | None = None
    response_status: int | None = None
    response_body: str | None = None
    attempt_count: int = 1

    def __str__(self) -> str:
        parts = [self.message]
        if self.handler_name:
            parts.append(f"handler={self.handler_name}")
        if self.response_status:
            parts.append(f"status={self.response_status}")
        if self.attempt_count > 1:
            parts.append(f"attempts={self.attempt_count}")
        return " | ".join(parts)


@dataclass
class NotificationTimeoutError(NotificationSendError):
    """Exception raised when a notification send times out.

    Attributes:
        timeout_seconds: The timeout value that was exceeded
        elapsed_seconds: Actual time elapsed before timeout
    """

    timeout_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        parts = [self.message]
        if self.handler_name:
            parts.append(f"handler={self.handler_name}")
        parts.append(f"timeout={self.timeout_seconds}s")
        if self.elapsed_seconds > 0:
            parts.append(f"elapsed={self.elapsed_seconds:.2f}s")
        return " | ".join(parts)


@dataclass
class NotificationRetryExhaustedError(NotificationSendError):
    """Exception raised when all retry attempts are exhausted.

    Attributes:
        max_retries: Maximum number of retries configured
        last_error: The error from the final attempt
        errors: List of errors from each attempt
    """

    max_retries: int = 0
    last_error: Exception | None = None
    errors: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        parts = [self.message]
        if self.handler_name:
            parts.append(f"handler={self.handler_name}")
        parts.append(f"max_retries={self.max_retries}")
        parts.append(f"attempt_count={self.attempt_count}")
        return " | ".join(parts)


@dataclass
class NotificationHandlerNotFoundError(NotificationError):
    """Exception raised when a handler cannot be found.

    Attributes:
        requested_name: The handler name that was requested
        available_handlers: List of available handler names
    """

    requested_name: str | None = None
    available_handlers: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        parts = [self.message]
        if self.requested_name:
            parts.append(f"requested={self.requested_name}")
        if self.available_handlers:
            parts.append(f"available={list(self.available_handlers)}")
        return " | ".join(parts)


@dataclass
class NotificationFormatterError(NotificationError):
    """Exception raised when message formatting fails.

    Attributes:
        formatter_name: Name of the formatter that failed
        template: Template string if applicable
        context_keys: Context keys that were available
    """

    formatter_name: str | None = None
    template: str | None = None
    context_keys: tuple[str, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        parts = [self.message]
        if self.formatter_name:
            parts.append(f"formatter={self.formatter_name}")
        if self.context_keys:
            parts.append(f"context_keys={list(self.context_keys)}")
        return " | ".join(parts)


@dataclass
class NotificationThrottledError(NotificationError):
    """Exception raised when notifications are being throttled.

    Attributes:
        throttle_key: The key being throttled
        retry_after_seconds: Suggested wait time before retry
        current_count: Current request count
        limit: Maximum allowed requests
    """

    throttle_key: str | None = None
    retry_after_seconds: float = 0.0
    current_count: int = 0
    limit: int = 0

    def __str__(self) -> str:
        parts = [self.message]
        if self.handler_name:
            parts.append(f"handler={self.handler_name}")
        if self.retry_after_seconds > 0:
            parts.append(f"retry_after={self.retry_after_seconds}s")
        if self.limit > 0:
            parts.append(f"count={self.current_count}/{self.limit}")
        return " | ".join(parts)


@dataclass
class NotificationFilteredError(NotificationError):
    """Exception raised when a notification is filtered/blocked.

    This is not necessarily an error condition - notifications may be
    intentionally filtered based on level, tags, or other criteria.

    Attributes:
        filter_reason: Why the notification was filtered
        filter_name: Name of the filter that blocked it
        notification_level: The level of the filtered notification
    """

    filter_reason: str | None = None
    filter_name: str | None = None
    notification_level: NotificationLevel | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.filter_name:
            parts.append(f"filter={self.filter_name}")
        if self.filter_reason:
            parts.append(f"reason={self.filter_reason}")
        return " | ".join(parts)
