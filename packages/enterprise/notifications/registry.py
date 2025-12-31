"""Notification registry and dispatcher.

This module provides centralized management for notification handlers,
including registration, lookup, and multi-channel dispatch.

Features:
    - Singleton registry pattern for global access
    - Named handler registration
    - Multi-channel notification dispatch
    - Parallel and sequential send modes
    - Hook integration
    - Result aggregation

Example:
    >>> from packages.enterprise.notifications.registry import (
    ...     get_notification_registry,
    ...     register_handler,
    ...     notify_all,
    ... )
    >>>
    >>> # Register handlers
    >>> register_handler("slack", SlackNotificationHandler(...))
    >>> register_handler("webhook", WebhookNotificationHandler(...))
    >>>
    >>> # Send to all handlers
    >>> results = await notify_all("Alert message", level=NotificationLevel.ERROR)
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from packages.enterprise.notifications.config import NotificationConfig
from packages.enterprise.notifications.exceptions import (
    NotificationError,
    NotificationHandlerNotFoundError,
)
from packages.enterprise.notifications.types import (
    BatchNotificationResult,
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
    NotificationStatus,
)

if TYPE_CHECKING:
    from packages.enterprise.notifications.handlers.base import NotificationHandler
    from packages.enterprise.notifications.hooks import NotificationHook


@dataclass
class DispatchConfig:
    """Configuration for notification dispatch.

    Attributes:
        parallel: Whether to send to handlers in parallel
        max_workers: Maximum concurrent sends (for parallel mode)
        stop_on_first_success: Stop after first successful send
        stop_on_first_failure: Stop after first failed send
        enabled_handlers: Only use these handlers (None = all)
        disabled_handlers: Skip these handlers
    """

    parallel: bool = True
    max_workers: int = 10
    stop_on_first_success: bool = False
    stop_on_first_failure: bool = False
    enabled_handlers: frozenset[str] | None = None
    disabled_handlers: frozenset[str] = field(default_factory=frozenset)

    def with_parallel(self, parallel: bool = True) -> DispatchConfig:
        """Create a copy with parallel mode setting."""
        return DispatchConfig(
            parallel=parallel,
            max_workers=self.max_workers,
            stop_on_first_success=self.stop_on_first_success,
            stop_on_first_failure=self.stop_on_first_failure,
            enabled_handlers=self.enabled_handlers,
            disabled_handlers=self.disabled_handlers,
        )

    def with_handlers(
        self,
        enabled: list[str] | None = None,
        disabled: list[str] | None = None,
    ) -> DispatchConfig:
        """Create a copy with handler filtering."""
        return DispatchConfig(
            parallel=self.parallel,
            max_workers=self.max_workers,
            stop_on_first_success=self.stop_on_first_success,
            stop_on_first_failure=self.stop_on_first_failure,
            enabled_handlers=frozenset(enabled) if enabled else self.enabled_handlers,
            disabled_handlers=frozenset(disabled) if disabled else self.disabled_handlers,
        )


DEFAULT_DISPATCH_CONFIG = DispatchConfig()
SEQUENTIAL_DISPATCH_CONFIG = DispatchConfig(parallel=False)
FAILOVER_DISPATCH_CONFIG = DispatchConfig(
    parallel=False,
    stop_on_first_success=True,
)


class NotificationDispatcher:
    """Dispatches notifications to multiple handlers.

    Handles the actual sending of notifications to one or more handlers,
    with support for parallel execution and result aggregation.

    Attributes:
        config: Dispatch configuration
        hooks: Global hooks applied to all handlers
    """

    def __init__(
        self,
        config: DispatchConfig | None = None,
        hooks: list[NotificationHook] | None = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            config: Dispatch configuration
            hooks: Global hooks for all handlers
        """
        self._config = config or DEFAULT_DISPATCH_CONFIG
        self._hooks = hooks or []

    @property
    def config(self) -> DispatchConfig:
        """Get the dispatch configuration."""
        return self._config

    async def dispatch(
        self,
        handlers: dict[str, NotificationHandler],
        payload: NotificationPayload,
    ) -> BatchNotificationResult:
        """Dispatch a notification to multiple handlers.

        Args:
            handlers: Dictionary of handler name -> handler
            payload: Notification payload to send

        Returns:
            Aggregated results from all handlers
        """
        import time

        start_time = time.perf_counter()

        # Filter handlers
        active_handlers = self._filter_handlers(handlers)

        if not active_handlers:
            return BatchNotificationResult(
                results=(),
                total_count=0,
                success_count=0,
                failure_count=0,
                skipped_count=0,
                duration_ms=0.0,
            )

        # Dispatch based on mode
        if self._config.parallel:
            results = await self._dispatch_parallel(active_handlers, payload)
        else:
            results = await self._dispatch_sequential(active_handlers, payload)

        duration_ms = (time.perf_counter() - start_time) * 1000
        return BatchNotificationResult.from_results(results, duration_ms)

    def _filter_handlers(
        self,
        handlers: dict[str, NotificationHandler],
    ) -> dict[str, NotificationHandler]:
        """Filter handlers based on configuration."""
        result = {}

        for name, handler in handlers.items():
            # Check enabled list
            if self._config.enabled_handlers is not None:
                if name not in self._config.enabled_handlers:
                    continue

            # Check disabled list
            if name in self._config.disabled_handlers:
                continue

            # Check handler enabled state
            if not handler.enabled:
                continue

            result[name] = handler

        return result

    async def _dispatch_parallel(
        self,
        handlers: dict[str, NotificationHandler],
        payload: NotificationPayload,
    ) -> dict[str, NotificationResult]:
        """Dispatch to handlers in parallel."""
        semaphore = asyncio.Semaphore(self._config.max_workers)

        async def send_with_semaphore(
            name: str,
            handler: NotificationHandler,
        ) -> tuple[str, NotificationResult]:
            async with semaphore:
                try:
                    result = await handler.send(payload)
                    return name, result
                except Exception as e:
                    return name, NotificationResult.failure_result(
                        channel=handler.channel,
                        handler_name=name,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        tasks = [
            send_with_semaphore(name, handler)
            for name, handler in handlers.items()
        ]

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for item in completed:
            if isinstance(item, Exception):
                # This shouldn't happen due to internal exception handling
                continue
            name, result = item
            results[name] = result

        return results

    async def _dispatch_sequential(
        self,
        handlers: dict[str, NotificationHandler],
        payload: NotificationPayload,
    ) -> dict[str, NotificationResult]:
        """Dispatch to handlers sequentially."""
        results = {}

        for name, handler in handlers.items():
            try:
                result = await handler.send(payload)
                results[name] = result

                # Check stop conditions
                if self._config.stop_on_first_success and result.success:
                    break
                if self._config.stop_on_first_failure and not result.success:
                    break

            except Exception as e:
                results[name] = NotificationResult.failure_result(
                    channel=handler.channel,
                    handler_name=name,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                if self._config.stop_on_first_failure:
                    break

        return results


class NotificationRegistry:
    """Central registry for notification handlers.

    Provides a singleton pattern for global handler management.
    Thread-safe for registration and lookup operations.

    Attributes:
        handlers: Registered handlers by name
        dispatcher: Notification dispatcher
    """

    _instance: NotificationRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> NotificationRegistry:
        """Create or return the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._handlers: dict[str, NotificationHandler] = {}
                instance._dispatcher = NotificationDispatcher()
                instance._hooks: list[NotificationHook] = []
                cls._instance = instance
            return cls._instance

    def register(
        self,
        name: str,
        handler: NotificationHandler,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a notification handler.

        Args:
            name: Unique name for the handler
            handler: Handler instance to register
            overwrite: Whether to overwrite existing handler

        Raises:
            ValueError: If name already exists and overwrite=False
        """
        with self._lock:
            if name in self._handlers and not overwrite:
                raise ValueError(
                    f"Handler '{name}' already registered. "
                    "Use overwrite=True to replace."
                )
            self._handlers[name] = handler

    def unregister(self, name: str) -> NotificationHandler | None:
        """Unregister a handler by name.

        Args:
            name: Handler name to remove

        Returns:
            The removed handler, or None if not found
        """
        with self._lock:
            return self._handlers.pop(name, None)

    def get(self, name: str) -> NotificationHandler | None:
        """Get a handler by name.

        Args:
            name: Handler name

        Returns:
            Handler instance, or None if not found
        """
        return self._handlers.get(name)

    def get_required(self, name: str) -> NotificationHandler:
        """Get a handler by name, raising if not found.

        Args:
            name: Handler name

        Returns:
            Handler instance

        Raises:
            NotificationHandlerNotFoundError: If handler not found
        """
        handler = self._handlers.get(name)
        if handler is None:
            raise NotificationHandlerNotFoundError(
                message=f"Handler '{name}' not found",
                requested_name=name,
                available_handlers=tuple(self._handlers.keys()),
            )
        return handler

    def list_handlers(self) -> list[str]:
        """List all registered handler names.

        Returns:
            List of handler names
        """
        return list(self._handlers.keys())

    def get_handlers_by_channel(
        self,
        channel: NotificationChannel,
    ) -> dict[str, NotificationHandler]:
        """Get all handlers for a specific channel.

        Args:
            channel: Channel to filter by

        Returns:
            Dictionary of matching handlers
        """
        return {
            name: handler
            for name, handler in self._handlers.items()
            if handler.channel == channel
        }

    def add_hook(self, hook: NotificationHook) -> None:
        """Add a global hook for all dispatches."""
        self._hooks.append(hook)

    def set_dispatcher(self, dispatcher: NotificationDispatcher) -> None:
        """Set the notification dispatcher."""
        self._dispatcher = dispatcher

    async def notify(
        self,
        name: str,
        message: str,
        *,
        level: NotificationLevel = NotificationLevel.INFO,
        title: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> NotificationResult:
        """Send a notification to a specific handler.

        Args:
            name: Handler name
            message: Notification message
            level: Notification level
            title: Optional title
            context: Additional context

        Returns:
            Send result
        """
        handler = self.get_required(name)

        payload = NotificationPayload(
            message=message,
            level=level,
            title=title,
            context=tuple((context or {}).items()),
        )

        return await handler.send(payload)

    async def notify_all(
        self,
        message: str,
        *,
        level: NotificationLevel = NotificationLevel.INFO,
        title: str | None = None,
        context: dict[str, Any] | None = None,
        dispatch_config: DispatchConfig | None = None,
    ) -> BatchNotificationResult:
        """Send a notification to all registered handlers.

        Args:
            message: Notification message
            level: Notification level
            title: Optional title
            context: Additional context
            dispatch_config: Override dispatch configuration

        Returns:
            Aggregated results from all handlers
        """
        if dispatch_config:
            dispatcher = NotificationDispatcher(config=dispatch_config)
        else:
            dispatcher = self._dispatcher

        payload = NotificationPayload(
            message=message,
            level=level,
            title=title,
            context=tuple((context or {}).items()),
        )

        return await dispatcher.dispatch(self._handlers, payload)

    async def notify_channel(
        self,
        channel: NotificationChannel,
        message: str,
        *,
        level: NotificationLevel = NotificationLevel.INFO,
        title: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> BatchNotificationResult:
        """Send a notification to all handlers of a specific channel.

        Args:
            channel: Target channel
            message: Notification message
            level: Notification level
            title: Optional title
            context: Additional context

        Returns:
            Aggregated results from matching handlers
        """
        handlers = self.get_handlers_by_channel(channel)

        payload = NotificationPayload(
            message=message,
            level=level,
            title=title,
            context=tuple((context or {}).items()),
        )

        return await self._dispatcher.dispatch(handlers, payload)

    def clear(self) -> None:
        """Clear all registered handlers."""
        with self._lock:
            self._handlers.clear()

    def reset(self) -> None:
        """Reset the registry to initial state."""
        with self._lock:
            self._handlers.clear()
            self._hooks.clear()
            self._dispatcher = NotificationDispatcher()


# =============================================================================
# Global Singleton Access
# =============================================================================

_registry: NotificationRegistry | None = None
_registry_lock = threading.Lock()


def get_notification_registry() -> NotificationRegistry:
    """Get the global notification registry.

    Returns:
        The singleton registry instance
    """
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = NotificationRegistry()
        return _registry


def reset_notification_registry() -> None:
    """Reset the global notification registry.

    Useful for testing.
    """
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.reset()
        _registry = None


# =============================================================================
# Convenience Functions
# =============================================================================


def register_handler(
    name: str,
    handler: NotificationHandler,
    *,
    overwrite: bool = False,
) -> None:
    """Register a handler in the global registry.

    Args:
        name: Handler name
        handler: Handler instance
        overwrite: Whether to overwrite existing
    """
    get_notification_registry().register(name, handler, overwrite=overwrite)


def unregister_handler(name: str) -> NotificationHandler | None:
    """Unregister a handler from the global registry.

    Args:
        name: Handler name to remove

    Returns:
        Removed handler, or None
    """
    return get_notification_registry().unregister(name)


async def notify(
    name: str,
    message: str,
    *,
    level: NotificationLevel = NotificationLevel.INFO,
    title: str | None = None,
    context: dict[str, Any] | None = None,
) -> NotificationResult:
    """Send a notification via a specific handler.

    Args:
        name: Handler name
        message: Notification message
        level: Notification level
        title: Optional title
        context: Additional context

    Returns:
        Send result
    """
    return await get_notification_registry().notify(
        name,
        message,
        level=level,
        title=title,
        context=context,
    )


async def notify_all(
    message: str,
    *,
    level: NotificationLevel = NotificationLevel.INFO,
    title: str | None = None,
    context: dict[str, Any] | None = None,
) -> BatchNotificationResult:
    """Send a notification to all registered handlers.

    Args:
        message: Notification message
        level: Notification level
        title: Optional title
        context: Additional context

    Returns:
        Aggregated results
    """
    return await get_notification_registry().notify_all(
        message,
        level=level,
        title=title,
        context=context,
    )
