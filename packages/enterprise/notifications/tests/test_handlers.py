"""Tests for notification handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.enterprise.notifications.config import (
    NotificationConfig,
    SlackConfig,
    WebhookConfig,
)
from packages.enterprise.notifications.handlers.slack import SlackNotificationHandler
from packages.enterprise.notifications.handlers.webhook import (
    WebhookNotificationHandler,
)
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationStatus,
)

from .conftest import MockNotificationHandler, RecordingHook


class TestBaseHandler:
    """Tests for base handler functionality."""

    @pytest.mark.asyncio
    async def test_handler_send(self, sample_payload: NotificationPayload) -> None:
        """Test basic send functionality."""
        handler = MockNotificationHandler()
        result = await handler.send(sample_payload)

        assert result.success
        assert result.channel == NotificationChannel.CUSTOM
        assert len(handler.send_calls) == 1

    @pytest.mark.asyncio
    async def test_handler_level_filtering(self) -> None:
        """Test that handlers filter by minimum level."""
        config = NotificationConfig(min_level=NotificationLevel.ERROR)
        handler = MockNotificationHandler()
        handler._config = config

        # Info level should be skipped
        info_payload = NotificationPayload(
            message="Info message",
            level=NotificationLevel.INFO,
        )
        result = await handler.send(info_payload)

        assert result.status == NotificationStatus.SKIPPED
        assert len(handler.send_calls) == 0

        # Error level should be sent
        error_payload = NotificationPayload(
            message="Error message",
            level=NotificationLevel.ERROR,
        )
        result = await handler.send(error_payload)

        assert result.success
        assert len(handler.send_calls) == 1

    @pytest.mark.asyncio
    async def test_handler_disabled(self, sample_payload: NotificationPayload) -> None:
        """Test that disabled handlers skip sends."""
        handler = MockNotificationHandler()
        handler.disable()

        result = await handler.send(sample_payload)

        assert result.status == NotificationStatus.SKIPPED
        assert len(handler.send_calls) == 0

    @pytest.mark.asyncio
    async def test_handler_hooks(
        self,
        sample_payload: NotificationPayload,
        recording_hook: RecordingHook,
    ) -> None:
        """Test that hooks are invoked."""
        handler = MockNotificationHandler(hooks=[recording_hook])

        await handler.send(sample_payload)

        assert len(recording_hook.before_events) == 1
        assert len(recording_hook.after_events) == 1
        assert recording_hook.after_events[0][1]  # success=True

    @pytest.mark.asyncio
    async def test_handler_failure(self, sample_payload: NotificationPayload) -> None:
        """Test handler failure handling."""
        handler = MockNotificationHandler(should_succeed=False)

        result = await handler.send(sample_payload)

        assert not result.success
        assert result.status == NotificationStatus.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_handler_duration_tracking(
        self,
        sample_payload: NotificationPayload,
    ) -> None:
        """Test that duration is tracked."""
        handler = MockNotificationHandler(response_delay=0.01)

        result = await handler.send(sample_payload)

        assert result.duration_ms > 0


class TestSlackHandler:
    """Tests for Slack notification handler."""

    def test_creation_with_config(self, slack_config: SlackConfig) -> None:
        """Test Slack handler creation with config."""
        handler = SlackNotificationHandler(config=slack_config)

        assert handler.channel == NotificationChannel.SLACK
        assert handler.slack_config.channel == "#test-channel"

    def test_creation_with_url(self) -> None:
        """Test Slack handler creation with URL."""
        handler = SlackNotificationHandler(
            webhook_url="https://hooks.slack.com/services/TEST"
        )

        assert handler.channel == NotificationChannel.SLACK

    def test_creation_requires_url_or_config(self) -> None:
        """Test that creation requires URL or config."""
        with pytest.raises(ValueError):
            SlackNotificationHandler()

    @pytest.mark.asyncio
    async def test_format_message(
        self,
        slack_config: SlackConfig,
        sample_payload: NotificationPayload,
    ) -> None:
        """Test message formatting."""
        handler = SlackNotificationHandler(config=slack_config)

        formatted = await handler._format_message(sample_payload)

        assert isinstance(formatted, dict)
        assert "blocks" in formatted
        assert "attachments" in formatted
        assert formatted["channel"] == "#test-channel"

    @pytest.mark.asyncio
    async def test_format_message_with_context(
        self,
        slack_config: SlackConfig,
    ) -> None:
        """Test message formatting with context."""
        handler = SlackNotificationHandler(config=slack_config)

        payload = NotificationPayload(
            message="Test message",
            level=NotificationLevel.WARNING,
            context=(("key1", "value1"), ("key2", 42)),
        )

        formatted = await handler._format_message(payload)

        # Should have context fields
        assert len(formatted["blocks"]) > 2

    def test_simple_message_format(self, slack_config: SlackConfig) -> None:
        """Test simple message formatting."""
        handler = SlackNotificationHandler(config=slack_config)

        message = handler.format_simple_message(
            "Simple message",
            NotificationLevel.ERROR,
        )

        assert "text" in message
        assert ":x:" in message["text"]  # Error emoji


class TestWebhookHandler:
    """Tests for webhook notification handler."""

    def test_creation_with_config(self, webhook_config: WebhookConfig) -> None:
        """Test webhook handler creation with config."""
        handler = WebhookNotificationHandler(config=webhook_config)

        assert handler.channel == NotificationChannel.WEBHOOK

    def test_creation_with_url(self) -> None:
        """Test webhook handler creation with URL."""
        handler = WebhookNotificationHandler(url="https://api.example.com/notify")

        assert handler.channel == NotificationChannel.WEBHOOK

    def test_creation_requires_url_or_config(self) -> None:
        """Test that creation requires URL or config."""
        with pytest.raises(ValueError):
            WebhookNotificationHandler()

    def test_build_headers_basic(self, webhook_config: WebhookConfig) -> None:
        """Test header building."""
        handler = WebhookNotificationHandler(config=webhook_config)

        headers = handler._build_headers()

        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_with_bearer(self) -> None:
        """Test header building with bearer token."""
        config = WebhookConfig(
            url="https://example.com"
        ).with_bearer_token("token123")

        handler = WebhookNotificationHandler(config=config)
        headers = handler._build_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer token123"

    def test_build_headers_with_basic_auth(self) -> None:
        """Test header building with basic auth."""
        config = WebhookConfig(
            url="https://example.com"
        ).with_basic_auth("user", "pass")

        handler = WebhookNotificationHandler(config=config)
        headers = handler._build_headers()

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")

    @pytest.mark.asyncio
    async def test_format_message(
        self,
        webhook_config: WebhookConfig,
        sample_payload: NotificationPayload,
    ) -> None:
        """Test message formatting."""
        handler = WebhookNotificationHandler(config=webhook_config)

        formatted = await handler._format_message(sample_payload)

        assert isinstance(formatted, dict)
        assert "message" in formatted
        assert "level" in formatted
        assert "timestamp" in formatted
        assert formatted["message"] == sample_payload.message

    @pytest.mark.asyncio
    async def test_format_message_with_metadata(
        self,
        sample_payload: NotificationPayload,
    ) -> None:
        """Test message formatting includes metadata."""
        config = WebhookConfig(url="https://example.com")
        handler = WebhookNotificationHandler(config=config)

        formatted = await handler._format_message(sample_payload)

        assert "metadata" in formatted
