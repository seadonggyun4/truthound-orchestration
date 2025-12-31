"""Base notification handler protocol and abstract classes.

This module defines the NotificationHandler protocol that all notification
handlers must implement. It provides both sync and async variants.

The protocol-based design allows for:
    - Duck typing compatibility
    - Easy testing with mocks
    - Extensibility without inheritance
    - Type checking support

Example:
    >>> from packages.enterprise.notifications.handlers.base import (
    ...     NotificationHandler,
    ...     AsyncNotificationHandler,
    ... )
    >>>
    >>> class MyHandler(AsyncNotificationHandler):
    ...     @property
    ...     def channel(self) -> NotificationChannel:
    ...         return NotificationChannel.CUSTOM
    ...
    ...     async def send(self, payload: NotificationPayload) -> NotificationResult:
    ...         # Custom implementation
    ...         return NotificationResult.success_result(...)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from packages.enterprise.notifications.config import NotificationConfig
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
    NotificationStatus,
)

if TYPE_CHECKING:
    from packages.enterprise.notifications.hooks import NotificationHook


@runtime_checkable
class NotificationHandler(Protocol):
    """Protocol for notification handlers.

    All notification handlers must implement this protocol. The protocol
    supports both sync and async implementations through the `send` method.

    Attributes:
        name: Unique identifier for this handler instance
        channel: The notification channel this handler uses
        config: Handler configuration
        enabled: Whether the handler is currently enabled
    """

    @property
    def name(self) -> str:
        """Get the handler's unique name."""
        ...

    @property
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        ...

    @property
    def config(self) -> NotificationConfig:
        """Get the handler configuration."""
        ...

    @property
    def enabled(self) -> bool:
        """Check if the handler is enabled."""
        ...

    async def send(
        self,
        payload: NotificationPayload,
    ) -> NotificationResult:
        """Send a notification.

        Args:
            payload: The notification payload to send

        Returns:
            Result of the send operation
        """
        ...

    def should_send(self, payload: NotificationPayload) -> bool:
        """Check if the notification should be sent.

        Args:
            payload: The notification payload to check

        Returns:
            True if the notification should be sent
        """
        ...


class BaseNotificationHandler(ABC):
    """Abstract base class for notification handlers.

    Provides common functionality for all handlers including:
        - Configuration management
        - Level filtering
        - Hook invocation
        - Error handling

    Subclasses must implement:
        - channel property
        - _do_send method

    Example:
        >>> class MyHandler(BaseNotificationHandler):
        ...     @property
        ...     def channel(self) -> NotificationChannel:
        ...         return NotificationChannel.CUSTOM
        ...
        ...     async def _do_send(self, payload, formatted_message):
        ...         # Implementation
        ...         return NotificationResult.success_result(...)
    """

    def __init__(
        self,
        name: str | None = None,
        config: NotificationConfig | None = None,
        hooks: list[NotificationHook] | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            name: Unique name for this handler instance
            config: Handler configuration
            hooks: List of hooks for event handling
        """
        self._name = name or self.__class__.__name__
        self._config = config or NotificationConfig()
        self._hooks = hooks or []
        self._enabled = self._config.enabled

    @property
    def name(self) -> str:
        """Get the handler's unique name."""
        return self._name

    @property
    @abstractmethod
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        ...

    @property
    def config(self) -> NotificationConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Check if the handler is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the handler."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the handler."""
        self._enabled = False

    def add_hook(self, hook: NotificationHook) -> None:
        """Add a hook to the handler."""
        self._hooks.append(hook)

    def remove_hook(self, hook: NotificationHook) -> None:
        """Remove a hook from the handler."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    def should_send(self, payload: NotificationPayload) -> bool:
        """Check if the notification should be sent.

        Evaluates:
            - Handler enabled state
            - Minimum level threshold
            - Config-based filtering

        Args:
            payload: The notification payload to check

        Returns:
            True if the notification should be sent
        """
        if not self._enabled:
            return False
        if not self._config.should_send(payload.level):
            return False
        return True

    async def send(
        self,
        payload: NotificationPayload,
    ) -> NotificationResult:
        """Send a notification.

        Handles:
            - Pre-send checks
            - Hook invocation (before/after)
            - Error handling
            - Result creation

        Args:
            payload: The notification payload to send

        Returns:
            Result of the send operation
        """
        import time

        # Check if we should send
        if not self.should_send(payload):
            return NotificationResult.skipped_result(
                channel=self.channel,
                handler_name=self.name,
                reason=f"Filtered: level={payload.level.value}, min_level={self._config.min_level.value}",
            )

        # Prepare context for hooks
        context: dict[str, Any] = {
            "handler_name": self.name,
            "channel": self.channel,
            "payload": payload,
            "config": self._config,
        }

        # Invoke before hooks
        await self._invoke_before_hooks(context)

        start_time = time.perf_counter()
        try:
            # Format and send
            formatted = await self._format_message(payload)
            result = await self._do_send(payload, formatted)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update result with duration
            result = NotificationResult(
                success=result.success,
                channel=result.channel,
                status=result.status,
                handler_name=self.name,
                message_id=result.message_id,
                timestamp=result.timestamp,
                duration_ms=duration_ms,
                retry_count=result.retry_count,
                error=result.error,
                error_type=result.error_type,
                response_data=result.response_data,
            )

            # Invoke after hooks
            context["result"] = result
            context["duration_ms"] = duration_ms
            await self._invoke_after_hooks(context, success=result.success)

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = NotificationResult.failure_result(
                channel=self.channel,
                handler_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
            )

            # Invoke after hooks with error
            context["result"] = result
            context["duration_ms"] = duration_ms
            context["error"] = e
            await self._invoke_after_hooks(context, success=False)

            return result

    @abstractmethod
    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Actually send the notification.

        Subclasses must implement this method to perform the actual send.

        Args:
            payload: The original notification payload
            formatted_message: The formatted message to send

        Returns:
            Result of the send operation
        """
        ...

    async def _format_message(
        self,
        payload: NotificationPayload,
    ) -> str | dict[str, Any]:
        """Format the notification message.

        Override this method to customize message formatting.

        Args:
            payload: The notification payload

        Returns:
            Formatted message (string or dict for structured messages)
        """
        return payload.message

    async def _invoke_before_hooks(self, context: dict[str, Any]) -> None:
        """Invoke before-send hooks."""
        for hook in self._hooks:
            try:
                if asyncio.iscoroutinefunction(hook.on_before_send):
                    await hook.on_before_send(context)
                else:
                    hook.on_before_send(context)
            except Exception:
                # Hooks should not affect send operation
                pass

    async def _invoke_after_hooks(
        self,
        context: dict[str, Any],
        success: bool,
    ) -> None:
        """Invoke after-send hooks."""
        for hook in self._hooks:
            try:
                if asyncio.iscoroutinefunction(hook.on_after_send):
                    await hook.on_after_send(context, success)
                else:
                    hook.on_after_send(context, success)
            except Exception:
                # Hooks should not affect send operation
                pass


class AsyncNotificationHandler(BaseNotificationHandler):
    """Base class for async notification handlers.

    Provides async-specific functionality. Subclasses should override
    `_do_send` with their async implementation.
    """

    pass


class SyncNotificationHandler(BaseNotificationHandler):
    """Base class for sync notification handlers.

    Wraps synchronous send operations to work with the async interface.
    Subclasses should override `_do_send_sync` for synchronous sends.
    """

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Wrap sync send in async."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._do_send_sync,
            payload,
            formatted_message,
        )

    def _do_send_sync(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Synchronous send implementation.

        Override this method for sync handlers.

        Args:
            payload: The original notification payload
            formatted_message: The formatted message to send

        Returns:
            Result of the send operation
        """
        raise NotImplementedError("Subclasses must implement _do_send_sync")
