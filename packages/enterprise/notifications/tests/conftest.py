"""Pytest fixtures for notification tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.enterprise.notifications.config import (
    NotificationConfig,
    SlackConfig,
    WebhookConfig,
)
from packages.enterprise.notifications.handlers.base import (
    AsyncNotificationHandler,
    NotificationHandler,
)
from packages.enterprise.notifications.hooks import (
    BaseNotificationHook,
    MetricsNotificationHook,
)
from packages.enterprise.notifications.registry import (
    NotificationRegistry,
    reset_notification_registry,
)
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
)


@pytest.fixture
def sample_payload() -> NotificationPayload:
    """Create a sample notification payload."""
    return NotificationPayload(
        message="Test notification message",
        level=NotificationLevel.INFO,
        title="Test Title",
        context=(("key1", "value1"), ("key2", 42)),
    )


@pytest.fixture
def error_payload() -> NotificationPayload:
    """Create an error notification payload."""
    return NotificationPayload(
        message="An error occurred",
        level=NotificationLevel.ERROR,
        title="Error Alert",
        context=(("error_code", "E001"), ("component", "validator")),
    )


@pytest.fixture
def critical_payload() -> NotificationPayload:
    """Create a critical notification payload."""
    return NotificationPayload(
        message="Critical system failure",
        level=NotificationLevel.CRITICAL,
        title="CRITICAL ALERT",
        context=(("system", "data_quality"), ("impact", "high")),
    )


@pytest.fixture
def default_config() -> NotificationConfig:
    """Create a default notification config."""
    return NotificationConfig()


@pytest.fixture
def slack_config() -> SlackConfig:
    """Create a sample Slack config."""
    return SlackConfig(
        webhook_url="https://hooks.slack.com/services/TEST/TEST/TEST",
        channel="#test-channel",
        username="Test Bot",
        icon_emoji=":robot:",
    )


@pytest.fixture
def webhook_config() -> WebhookConfig:
    """Create a sample webhook config."""
    return WebhookConfig(
        url="https://api.example.com/notify",
        method="POST",
    )


class MockNotificationHandler(AsyncNotificationHandler):
    """Mock notification handler for testing."""

    def __init__(
        self,
        name: str = "mock",
        channel: NotificationChannel = NotificationChannel.CUSTOM,
        should_succeed: bool = True,
        response_delay: float = 0.0,
        hooks: list | None = None,
    ) -> None:
        super().__init__(name=name, hooks=hooks)
        self._channel = channel
        self._should_succeed = should_succeed
        self._response_delay = response_delay
        self.send_calls: list[NotificationPayload] = []

    @property
    def channel(self) -> NotificationChannel:
        return self._channel

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        import asyncio

        self.send_calls.append(payload)

        if self._response_delay > 0:
            await asyncio.sleep(self._response_delay)

        if self._should_succeed:
            return NotificationResult.success_result(
                channel=self.channel,
                handler_name=self.name,
                message_id="mock_msg_123",
            )
        else:
            return NotificationResult.failure_result(
                channel=self.channel,
                handler_name=self.name,
                error="Mock failure",
                error_type="MockError",
            )


@pytest.fixture
def mock_handler() -> MockNotificationHandler:
    """Create a mock notification handler."""
    return MockNotificationHandler()


@pytest.fixture
def failing_mock_handler() -> MockNotificationHandler:
    """Create a failing mock notification handler."""
    return MockNotificationHandler(name="failing_mock", should_succeed=False)


@pytest.fixture
def metrics_hook() -> MetricsNotificationHook:
    """Create a metrics hook for testing."""
    return MetricsNotificationHook()


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    reset_notification_registry()
    yield
    reset_notification_registry()


class RecordingHook(BaseNotificationHook):
    """Hook that records all events for testing."""

    def __init__(self) -> None:
        self.before_events: list[dict[str, Any]] = []
        self.after_events: list[tuple[dict[str, Any], bool]] = []
        self.retry_events: list[tuple[dict[str, Any], int, Exception]] = []

    def on_before_send(self, context: dict[str, Any]) -> None:
        self.before_events.append(context.copy())

    def on_after_send(self, context: dict[str, Any], success: bool) -> None:
        self.after_events.append((context.copy(), success))

    def on_retry(
        self,
        context: dict[str, Any],
        attempt: int,
        error: Exception,
    ) -> None:
        self.retry_events.append((context.copy(), attempt, error))


@pytest.fixture
def recording_hook() -> RecordingHook:
    """Create a recording hook for testing."""
    return RecordingHook()
