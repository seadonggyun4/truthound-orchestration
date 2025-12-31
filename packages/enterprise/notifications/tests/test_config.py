"""Tests for notification configuration."""

import pytest

from packages.enterprise.notifications.config import (
    CRITICAL_NOTIFICATION_CONFIG,
    DEFAULT_NOTIFICATION_CONFIG,
    FAST_NOTIFICATION_CONFIG,
    LENIENT_NOTIFICATION_CONFIG,
    NotificationConfig,
    RetryConfig,
    SlackConfig,
    WebhookConfig,
)
from packages.enterprise.notifications.exceptions import NotificationConfigError
from packages.enterprise.notifications.types import NotificationLevel


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 30.0
        assert config.exponential_backoff
        assert config.jitter

    def test_validation_negative_retries(self) -> None:
        """Test validation rejects negative retries."""
        with pytest.raises(NotificationConfigError):
            RetryConfig(max_retries=-1)

    def test_validation_zero_delay(self) -> None:
        """Test validation rejects zero delay."""
        with pytest.raises(NotificationConfigError):
            RetryConfig(base_delay_seconds=0)

    def test_validation_max_less_than_base(self) -> None:
        """Test validation rejects max < base delay."""
        with pytest.raises(NotificationConfigError):
            RetryConfig(base_delay_seconds=10.0, max_delay_seconds=5.0)

    def test_builder_methods(self) -> None:
        """Test builder methods."""
        config = RetryConfig()

        with_retries = config.with_max_retries(5)
        assert with_retries.max_retries == 5
        assert config.max_retries == 3  # Original unchanged

        with_delays = config.with_delays(base_delay_seconds=2.0)
        assert with_delays.base_delay_seconds == 2.0

        with_backoff = config.with_exponential_backoff(False)
        assert not with_backoff.exponential_backoff

    def test_serialization(self) -> None:
        """Test serialization."""
        config = RetryConfig(max_retries=5, base_delay_seconds=2.0)

        data = config.to_dict()
        assert data["max_retries"] == 5

        restored = RetryConfig.from_dict(data)
        assert restored.max_retries == config.max_retries


class TestNotificationConfig:
    """Tests for NotificationConfig."""

    def test_default_config(self) -> None:
        """Test default notification configuration."""
        config = NotificationConfig()
        assert config.enabled
        assert config.timeout_seconds == 30.0
        assert config.min_level == NotificationLevel.INFO
        assert not config.async_send

    def test_validation_negative_timeout(self) -> None:
        """Test validation rejects negative timeout."""
        with pytest.raises(NotificationConfigError):
            NotificationConfig(timeout_seconds=-1)

    def test_builder_methods(self) -> None:
        """Test builder methods."""
        config = NotificationConfig()

        disabled = config.with_enabled(False)
        assert not disabled.enabled

        with_timeout = config.with_timeout(60.0)
        assert with_timeout.timeout_seconds == 60.0

        with_retry = config.with_retry(count=5, base_delay=2.0)
        assert with_retry.retry_config.max_retries == 5
        assert with_retry.retry_config.base_delay_seconds == 2.0

        with_level = config.with_min_level(NotificationLevel.ERROR)
        assert with_level.min_level == NotificationLevel.ERROR

        with_tags = config.with_tags("tag1", "tag2")
        assert "tag1" in with_tags.tags

        with_async = config.with_async_send(True)
        assert with_async.async_send

        with_formatter = config.with_formatter("markdown")
        assert with_formatter.formatter_name == "markdown"

        with_rate_limit = config.with_rate_limit(100, 60.0)
        assert with_rate_limit.rate_limit_requests == 100

    def test_should_send(self) -> None:
        """Test should_send method."""
        config = NotificationConfig(min_level=NotificationLevel.WARNING)

        assert not config.should_send(NotificationLevel.DEBUG)
        assert not config.should_send(NotificationLevel.INFO)
        assert config.should_send(NotificationLevel.WARNING)
        assert config.should_send(NotificationLevel.ERROR)
        assert config.should_send(NotificationLevel.CRITICAL)

    def test_should_send_disabled(self) -> None:
        """Test should_send when disabled."""
        config = NotificationConfig(enabled=False)
        assert not config.should_send(NotificationLevel.CRITICAL)

    def test_serialization(self) -> None:
        """Test serialization."""
        config = NotificationConfig(
            timeout_seconds=60.0,
            min_level=NotificationLevel.ERROR,
        )

        data = config.to_dict()
        assert data["timeout_seconds"] == 60.0
        assert data["min_level"] == "error"

        restored = NotificationConfig.from_dict(data)
        assert restored.timeout_seconds == config.timeout_seconds
        assert restored.min_level == config.min_level


class TestSlackConfig:
    """Tests for SlackConfig."""

    def test_creation(self) -> None:
        """Test Slack config creation."""
        config = SlackConfig(
            webhook_url="https://hooks.slack.com/services/TEST",
            channel="#test",
            username="Bot",
        )

        assert config.channel == "#test"
        assert config.username == "Bot"
        assert config.mrkdwn

    def test_validation_missing_url(self) -> None:
        """Test validation requires webhook_url."""
        with pytest.raises(NotificationConfigError):
            SlackConfig(webhook_url="")

    def test_validation_http_url(self) -> None:
        """Test validation rejects HTTP URLs."""
        with pytest.raises(NotificationConfigError):
            SlackConfig(webhook_url="http://hooks.slack.com/services/TEST")

    def test_builder_methods(self) -> None:
        """Test builder methods."""
        config = SlackConfig(
            webhook_url="https://hooks.slack.com/services/TEST"
        )

        with_channel = config.with_channel("#alerts")
        assert with_channel.channel == "#alerts"

        with_username = config.with_username("Alert Bot")
        assert with_username.username == "Alert Bot"

        with_icon = config.with_icon(emoji=":robot:")
        assert with_icon.icon_emoji == ":robot:"

    def test_to_dict_masks_url(self) -> None:
        """Test to_dict masks sensitive webhook URL."""
        config = SlackConfig(
            webhook_url="https://hooks.slack.com/services/SECRET"
        )

        data = config.to_dict()
        assert data["webhook_url"] == "***MASKED***"


class TestWebhookConfig:
    """Tests for WebhookConfig."""

    def test_creation(self) -> None:
        """Test webhook config creation."""
        config = WebhookConfig(
            url="https://api.example.com/notify",
            method="POST",
        )

        assert config.url == "https://api.example.com/notify"
        assert config.method == "POST"
        assert config.verify_ssl

    def test_validation_missing_url(self) -> None:
        """Test validation requires url."""
        with pytest.raises(NotificationConfigError):
            WebhookConfig(url="")

    def test_validation_invalid_method(self) -> None:
        """Test validation rejects invalid methods."""
        with pytest.raises(NotificationConfigError):
            WebhookConfig(url="https://example.com", method="GET")

    def test_validation_invalid_auth_type(self) -> None:
        """Test validation rejects invalid auth types."""
        with pytest.raises(NotificationConfigError):
            WebhookConfig(url="https://example.com", auth_type="oauth")

    def test_builder_methods(self) -> None:
        """Test builder methods."""
        config = WebhookConfig(url="https://example.com")

        with_header = config.with_header("X-Custom", "value")
        assert ("X-Custom", "value") in with_header.headers

        with_bearer = config.with_bearer_token("token123")
        assert with_bearer.auth_type == "bearer"
        assert with_bearer.auth_credentials == "token123"

        with_api_key = config.with_api_key("key123", "X-API-Key")
        assert with_api_key.auth_type == "api_key"

    def test_to_dict_masks_credentials(self) -> None:
        """Test to_dict masks sensitive credentials."""
        config = WebhookConfig(
            url="https://example.com"
        ).with_bearer_token("secret_token")

        data = config.to_dict()
        assert data["auth_credentials"] == "***MASKED***"


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_config(self) -> None:
        """Test default config preset."""
        assert DEFAULT_NOTIFICATION_CONFIG.enabled
        assert DEFAULT_NOTIFICATION_CONFIG.timeout_seconds == 30.0

    def test_critical_config(self) -> None:
        """Test critical config preset."""
        assert CRITICAL_NOTIFICATION_CONFIG.timeout_seconds < 30.0
        assert CRITICAL_NOTIFICATION_CONFIG.retry_config.max_retries > 3
        assert not CRITICAL_NOTIFICATION_CONFIG.async_send

    def test_lenient_config(self) -> None:
        """Test lenient config preset."""
        assert LENIENT_NOTIFICATION_CONFIG.timeout_seconds > 30.0
        assert LENIENT_NOTIFICATION_CONFIG.async_send
        assert LENIENT_NOTIFICATION_CONFIG.min_level == NotificationLevel.WARNING

    def test_fast_config(self) -> None:
        """Test fast config preset."""
        assert FAST_NOTIFICATION_CONFIG.timeout_seconds < 30.0
        assert FAST_NOTIFICATION_CONFIG.retry_config.max_retries == 0
        assert FAST_NOTIFICATION_CONFIG.async_send
