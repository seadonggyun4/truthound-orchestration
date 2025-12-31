"""Notification hooks.

This module provides hooks for monitoring and extending notification behavior.
Hooks are invoked before and after notification sends, allowing for:
    - Logging
    - Metrics collection
    - Custom processing
    - Filtering

The hook system follows the observer pattern with failure isolation -
a failing hook will not prevent notification delivery.

Example:
    >>> from packages.enterprise.notifications.hooks import (
    ...     LoggingNotificationHook,
    ...     MetricsNotificationHook,
    ...     CompositeNotificationHook,
    ... )
    >>>
    >>> hooks = CompositeNotificationHook([
    ...     LoggingNotificationHook(),
    ...     MetricsNotificationHook(),
    ... ])
    >>>
    >>> handler = SlackNotificationHandler(
    ...     webhook_url="...",
    ...     hooks=[hooks],
    ... )
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable

from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationResult,
)


@runtime_checkable
class NotificationHook(Protocol):
    """Protocol for notification hooks.

    Hooks are invoked at various points in the notification lifecycle.
    All methods are optional - implement only what you need.
    """

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Called before sending a notification.

        Args:
            context: Send context containing:
                - handler_name: str
                - channel: NotificationChannel
                - payload: NotificationPayload
                - config: NotificationConfig
        """
        ...

    def on_after_send(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Called after sending a notification.

        Args:
            context: Send context (same as on_before_send, plus):
                - result: NotificationResult
                - duration_ms: float
                - error: Exception (if failed)
            success: Whether the send was successful
        """
        ...

    def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        """Called before a retry attempt.

        Args:
            context: Send context
            attempt: Current attempt number (1-based)
            error: Error from previous attempt
        """
        ...


class AsyncNotificationHook(Protocol):
    """Protocol for async notification hooks."""

    async def on_before_send(self, context: dict[str, Any]) -> None:
        """Called before sending a notification."""
        ...

    async def on_after_send(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Called after sending a notification."""
        ...

    async def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        """Called before a retry attempt."""
        ...


class BaseNotificationHook(ABC):
    """Abstract base class for notification hooks.

    Provides default no-op implementations for all hook methods.
    Subclasses can override only the methods they need.
    """

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Called before sending a notification."""
        pass

    def on_after_send(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Called after sending a notification."""
        pass

    def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        """Called before a retry attempt."""
        pass


class LoggingNotificationHook(BaseNotificationHook):
    """Hook that logs notification events.

    Provides structured logging for all notification lifecycle events.

    Attributes:
        logger: Logger instance to use
        log_level: Default log level for events
        log_payload: Whether to log payload contents
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        log_level: int = logging.INFO,
        log_payload: bool = False,
    ) -> None:
        """Initialize the logging hook.

        Args:
            logger: Logger to use (defaults to module logger)
            log_level: Log level for events
            log_payload: Whether to include payload in logs
        """
        self._logger = logger or logging.getLogger(__name__)
        self._log_level = log_level
        self._log_payload = log_payload

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Log before send event."""
        handler_name = context.get("handler_name", "unknown")
        channel = context.get("channel", "unknown")
        payload = context.get("payload")

        msg = f"Sending notification via {handler_name} ({channel})"
        extra: dict[str, Any] = {
            "handler_name": handler_name,
            "channel": str(channel),
        }

        if self._log_payload and payload:
            extra["level"] = payload.level.value
            extra["message_preview"] = payload.message[:100]

        self._logger.log(self._log_level, msg, extra=extra)

    def on_after_send(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Log after send event."""
        handler_name = context.get("handler_name", "unknown")
        channel = context.get("channel", "unknown")
        duration_ms = context.get("duration_ms", 0)
        result = context.get("result")

        if success:
            msg = f"Notification sent successfully via {handler_name}"
            level = self._log_level
        else:
            msg = f"Notification failed via {handler_name}"
            level = logging.ERROR

        extra: dict[str, Any] = {
            "handler_name": handler_name,
            "channel": str(channel),
            "duration_ms": f"{duration_ms:.2f}",
            "success": success,
        }

        if result and not success:
            extra["error"] = result.error

        self._logger.log(level, msg, extra=extra)

    def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        """Log retry event."""
        handler_name = context.get("handler_name", "unknown")

        self._logger.warning(
            f"Retrying notification via {handler_name} (attempt {attempt})",
            extra={
                "handler_name": handler_name,
                "attempt": attempt,
                "error": str(error),
            },
        )


@dataclass
class NotificationStats:
    """Statistics for notification sends.

    Thread-safe statistics collection for monitoring notification performance.
    """

    total_sends: int = 0
    successful_sends: int = 0
    failed_sends: int = 0
    skipped_sends: int = 0
    total_retries: int = 0
    total_duration_ms: float = 0.0
    last_send_time: datetime | None = None
    errors_by_type: dict[str, int] = field(default_factory=dict)
    sends_by_channel: dict[str, int] = field(default_factory=dict)
    sends_by_level: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_sends + self.failed_sends
        if total == 0:
            return 0.0
        return self.successful_sends / total

    @property
    def average_duration_ms(self) -> float:
        """Calculate average send duration."""
        if self.total_sends == 0:
            return 0.0
        return self.total_duration_ms / self.total_sends

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_sends": self.total_sends,
            "successful_sends": self.successful_sends,
            "failed_sends": self.failed_sends,
            "skipped_sends": self.skipped_sends,
            "total_retries": self.total_retries,
            "total_duration_ms": self.total_duration_ms,
            "success_rate": self.success_rate,
            "average_duration_ms": self.average_duration_ms,
            "last_send_time": self.last_send_time.isoformat() if self.last_send_time else None,
            "errors_by_type": self.errors_by_type,
            "sends_by_channel": self.sends_by_channel,
            "sends_by_level": self.sends_by_level,
        }


class MetricsNotificationHook(BaseNotificationHook):
    """Hook that collects notification metrics.

    Tracks statistics about notification sends including success rates,
    durations, and error types.

    Attributes:
        stats: Collected statistics
    """

    def __init__(self) -> None:
        """Initialize the metrics hook."""
        self._stats = NotificationStats()
        self._start_times: dict[int, float] = {}

    @property
    def stats(self) -> NotificationStats:
        """Get collected statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats = NotificationStats()
        self._start_times.clear()

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Record send start time."""
        context_id = id(context)
        self._start_times[context_id] = time.perf_counter()

        # Track by level
        payload = context.get("payload")
        if payload:
            level = payload.level.value
            self._stats.sends_by_level[level] = self._stats.sends_by_level.get(level, 0) + 1

    def on_after_send(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Record send result."""
        self._stats.total_sends += 1
        self._stats.last_send_time = datetime.now(timezone.utc)

        # Calculate duration
        context_id = id(context)
        start_time = self._start_times.pop(context_id, None)
        if start_time:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._stats.total_duration_ms += duration_ms

        # Track by channel
        channel = context.get("channel")
        if channel:
            channel_str = str(channel)
            self._stats.sends_by_channel[channel_str] = (
                self._stats.sends_by_channel.get(channel_str, 0) + 1
            )

        # Track result
        result = context.get("result")
        if result:
            from packages.enterprise.notifications.types import NotificationStatus

            if result.status == NotificationStatus.SKIPPED:
                self._stats.skipped_sends += 1
            elif success:
                self._stats.successful_sends += 1
            else:
                self._stats.failed_sends += 1
                if result.error_type:
                    self._stats.errors_by_type[result.error_type] = (
                        self._stats.errors_by_type.get(result.error_type, 0) + 1
                    )

    def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        """Record retry attempt."""
        self._stats.total_retries += 1


class CompositeNotificationHook(BaseNotificationHook):
    """Hook that delegates to multiple hooks.

    Provides failure isolation - if one hook fails, others still execute.

    Attributes:
        hooks: List of hooks to invoke
    """

    def __init__(self, hooks: list[NotificationHook] | None = None) -> None:
        """Initialize the composite hook.

        Args:
            hooks: List of hooks to delegate to
        """
        self._hooks: list[NotificationHook] = list(hooks) if hooks else []

    def add_hook(self, hook: NotificationHook) -> None:
        """Add a hook to the composite."""
        self._hooks.append(hook)

    def remove_hook(self, hook: NotificationHook) -> None:
        """Remove a hook from the composite."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Invoke all hooks' on_before_send."""
        for hook in self._hooks:
            try:
                hook.on_before_send(context)
            except Exception:
                # Isolate hook failures
                pass

    def on_after_send(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Invoke all hooks' on_after_send."""
        for hook in self._hooks:
            try:
                hook.on_after_send(context, success)
            except Exception:
                # Isolate hook failures
                pass

    def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        """Invoke all hooks' on_retry."""
        for hook in self._hooks:
            try:
                hook.on_retry(context, attempt, error)
            except Exception:
                # Isolate hook failures
                pass


class CallbackNotificationHook(BaseNotificationHook):
    """Hook that invokes callbacks for events.

    Allows for ad-hoc hook behavior without creating a new class.

    Example:
        >>> hook = CallbackNotificationHook(
        ...     on_before=lambda ctx: print(f"Sending to {ctx['channel']}"),
        ...     on_after=lambda ctx, ok: print(f"Done: {ok}"),
        ... )
    """

    def __init__(
        self,
        on_before: Callable[[dict[str, Any]], None] | None = None,
        on_after: Callable[[dict[str, Any], bool], None] | None = None,
        on_retry_callback: Callable[[dict[str, Any], int, Exception], None] | None = None,
    ) -> None:
        """Initialize the callback hook.

        Args:
            on_before: Callback for before send
            on_after: Callback for after send
            on_retry_callback: Callback for retry
        """
        self._on_before = on_before
        self._on_after = on_after
        self._on_retry_callback = on_retry_callback

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Invoke before callback."""
        if self._on_before:
            self._on_before(context)

    def on_after_send(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Invoke after callback."""
        if self._on_after:
            self._on_after(context, success)

    def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        """Invoke retry callback."""
        if self._on_retry_callback:
            self._on_retry_callback(context, attempt, error)


class FilteringNotificationHook(BaseNotificationHook):
    """Hook that filters notifications based on criteria.

    Can be used to suppress notifications based on level, tags,
    or custom predicates.

    Note: This hook can modify context to set a 'skip' flag that
    handlers can check.
    """

    def __init__(
        self,
        min_level: NotificationLevel | None = None,
        required_tags: set[str] | None = None,
        excluded_tags: set[str] | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        """Initialize the filtering hook.

        Args:
            min_level: Minimum level to allow
            required_tags: Tags that must be present
            excluded_tags: Tags that must not be present
            predicate: Custom filter function (returns True to allow)
        """
        self._min_level = min_level
        self._required_tags = required_tags or set()
        self._excluded_tags = excluded_tags or set()
        self._predicate = predicate

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Check if notification should be filtered."""
        payload = context.get("payload")
        if not payload:
            return

        should_skip = False

        # Check level
        if self._min_level and payload.level < self._min_level:
            should_skip = True

        # Check required tags
        if self._required_tags:
            payload_tags = set(payload.metadata.tags)
            if not self._required_tags.issubset(payload_tags):
                should_skip = True

        # Check excluded tags
        if self._excluded_tags:
            payload_tags = set(payload.metadata.tags)
            if self._excluded_tags.intersection(payload_tags):
                should_skip = True

        # Check custom predicate
        if self._predicate and not self._predicate(context):
            should_skip = True

        # Set skip flag in context
        context["skip"] = should_skip


class ThrottlingNotificationHook(BaseNotificationHook):
    """Hook that throttles notifications.

    Limits the rate of notifications to prevent flooding.
    Uses a sliding window algorithm.

    Attributes:
        max_per_window: Maximum notifications per window
        window_seconds: Time window size
    """

    def __init__(
        self,
        max_per_window: int = 10,
        window_seconds: float = 60.0,
    ) -> None:
        """Initialize the throttling hook.

        Args:
            max_per_window: Maximum notifications per window
            window_seconds: Window size in seconds
        """
        self._max_per_window = max_per_window
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []

    def on_before_send(self, context: dict[str, Any]) -> None:
        """Check if notification should be throttled."""
        now = time.time()
        cutoff = now - self._window_seconds

        # Remove old timestamps
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        # Check if over limit
        if len(self._timestamps) >= self._max_per_window:
            context["skip"] = True
            context["skip_reason"] = "throttled"
        else:
            self._timestamps.append(now)

    @property
    def current_count(self) -> int:
        """Get current count in window."""
        now = time.time()
        cutoff = now - self._window_seconds
        return len([t for t in self._timestamps if t > cutoff])

    def reset(self) -> None:
        """Reset throttle state."""
        self._timestamps.clear()
