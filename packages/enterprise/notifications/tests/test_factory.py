"""Tests for factory functions."""

import pytest

from packages.enterprise.notifications.config import (
    CRITICAL_NOTIFICATION_CONFIG,
    DEFAULT_NOTIFICATION_CONFIG,
    NotificationConfig,
)
from packages.enterprise.notifications.exceptions import NotificationConfigError
from packages.enterprise.notifications.factory import (
    create_critical_slack_handler,
    create_critical_webhook_handler,
    create_handler_from_config,
    create_slack_handler,
    create_webhook_handler,
)
from packages.enterprise.notifications.handlers.slack import SlackNotificationHandler
from packages.enterprise.notifications.handlers.webhook import WebhookNotificationHandler
from packages.enterprise.notifications.hooks import LoggingNotificationHook
from packages.enterprise.notifications.types import NotificationChannel


class TestCreateSlackHandler:
    """Tests for create_slack_handler factory."""

    def test_minimal_creation(self) -> None:
        """Test creating Slack handler with minimal args."""
        handler = create_slack_handler(
            webhook_url="https://hooks.slack.com/services/xxx",
        )

        assert isinstance(handler, SlackNotificationHandler)
        assert handler.channel == NotificationChannel.SLACK
        assert handler.name == "slack"

    def test_with_all_options(self) -> None:
        """Test creating Slack handler with all options."""
        handler = create_slack_handler(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#alerts",
            username="Bot",
            icon_emoji=":robot:",
            name="custom_slack",
        )

        assert handler.name == "custom_slack"
        assert handler.slack_config.channel == "#alerts"
        assert handler.slack_config.username == "Bot"
        assert handler.slack_config.icon_emoji == ":robot:"

    def test_with_custom_config(self) -> None:
        """Test creating Slack handler with custom config."""
        config = NotificationConfig(timeout_seconds=60.0)
        handler = create_slack_handler(
            webhook_url="https://hooks.slack.com/services/xxx",
            config=config,
        )

        assert handler.config.timeout_seconds == 60.0

    def test_with_hooks(self) -> None:
        """Test creating Slack handler with hooks."""
        hooks = [LoggingNotificationHook()]
        handler = create_slack_handler(
            webhook_url="https://hooks.slack.com/services/xxx",
            hooks=hooks,
        )

        assert handler is not None


class TestCreateWebhookHandler:
    """Tests for create_webhook_handler factory."""

    def test_minimal_creation(self) -> None:
        """Test creating webhook handler with minimal args."""
        handler = create_webhook_handler(url="https://api.example.com/notify")

        assert isinstance(handler, WebhookNotificationHandler)
        assert handler.channel == NotificationChannel.WEBHOOK
        assert handler.name == "webhook"

    def test_with_bearer_token(self) -> None:
        """Test creating webhook handler with bearer token."""
        handler = create_webhook_handler(
            url="https://api.example.com/notify",
            auth_token="test-token",
        )

        assert handler.webhook_config.auth_type == "bearer"
        assert handler.webhook_config.auth_credentials == "test-token"

    def test_with_api_key(self) -> None:
        """Test creating webhook handler with API key."""
        handler = create_webhook_handler(
            url="https://api.example.com/notify",
            api_key="my-api-key",
            api_key_header="X-Custom-Key",
        )

        headers = handler.webhook_config.headers_dict
        assert "X-Custom-Key" in headers

    def test_with_basic_auth(self) -> None:
        """Test creating webhook handler with basic auth."""
        handler = create_webhook_handler(
            url="https://api.example.com/notify",
            basic_auth=("user", "pass"),
        )

        assert handler.webhook_config.auth_type == "basic"

    def test_with_custom_headers(self) -> None:
        """Test creating webhook handler with custom headers."""
        handler = create_webhook_handler(
            url="https://api.example.com/notify",
            headers={"X-Custom-Header": "value"},
        )

        headers = handler.webhook_config.headers_dict
        assert headers.get("X-Custom-Header") == "value"

    def test_with_ssl_verification(self) -> None:
        """Test creating webhook handler with SSL verification options."""
        handler = create_webhook_handler(
            url="https://api.example.com/notify",
            verify_ssl=False,
        )

        assert handler.webhook_config.verify_ssl is False

    def test_with_custom_method(self) -> None:
        """Test creating webhook handler with custom HTTP method."""
        handler = create_webhook_handler(
            url="https://api.example.com/notify",
            method="PUT",
        )

        assert handler.webhook_config.method == "PUT"


class TestCreateHandlerFromConfig:
    """Tests for create_handler_from_config factory."""

    def test_create_slack_from_config(self) -> None:
        """Test creating Slack handler from config dict."""
        config = {
            "webhook_url": "https://hooks.slack.com/services/xxx",
            "channel": "#alerts",
            "username": "Bot",
        }

        handler = create_handler_from_config("slack", config)

        assert isinstance(handler, SlackNotificationHandler)
        assert handler.slack_config.channel == "#alerts"

    def test_create_webhook_from_config(self) -> None:
        """Test creating webhook handler from config dict."""
        config = {
            "url": "https://api.example.com/notify",
            "method": "POST",
            "auth_token": "token",
        }

        handler = create_handler_from_config("webhook", config)

        assert isinstance(handler, WebhookNotificationHandler)
        assert handler.webhook_config.auth_type == "bearer"

    def test_create_with_base_config(self) -> None:
        """Test creating handler with base config in dict."""
        config = {
            "webhook_url": "https://hooks.slack.com/services/xxx",
            "timeout_seconds": 60.0,
            "min_level": "warning",
        }

        handler = create_handler_from_config("slack", config)

        assert handler.config.timeout_seconds == 60.0

    def test_create_with_nested_base_config(self) -> None:
        """Test creating handler with nested base_config."""
        config = {
            "webhook_url": "https://hooks.slack.com/services/xxx",
            "base_config": {
                "timeout_seconds": 45.0,
                "enabled": True,
            },
        }

        handler = create_handler_from_config("slack", config)

        assert handler.config.timeout_seconds == 45.0

    def test_unknown_handler_type_raises(self) -> None:
        """Test that unknown handler type raises error."""
        with pytest.raises(NotificationConfigError) as exc_info:
            create_handler_from_config("unknown", {})

        assert exc_info.value.field_name == "handler_type"
        assert exc_info.value.field_value == "unknown"

    def test_slack_missing_webhook_url_raises(self) -> None:
        """Test that missing webhook_url raises error."""
        with pytest.raises(NotificationConfigError) as exc_info:
            create_handler_from_config("slack", {})

        assert exc_info.value.field_name == "webhook_url"

    def test_webhook_missing_url_raises(self) -> None:
        """Test that missing url raises error."""
        with pytest.raises(NotificationConfigError) as exc_info:
            create_handler_from_config("webhook", {})

        assert exc_info.value.field_name == "url"

    def test_case_insensitive_type(self) -> None:
        """Test that handler type is case insensitive."""
        config = {
            "webhook_url": "https://hooks.slack.com/services/xxx",
        }

        handler = create_handler_from_config("SLACK", config)

        assert isinstance(handler, SlackNotificationHandler)

    def test_webhook_basic_auth_from_dict(self) -> None:
        """Test basic auth from dict format."""
        config = {
            "url": "https://api.example.com/notify",
            "basic_auth": {"username": "user", "password": "pass"},
        }

        handler = create_handler_from_config("webhook", config)

        assert handler.webhook_config.auth_type == "basic"

    def test_webhook_basic_auth_from_tuple(self) -> None:
        """Test basic auth from tuple format."""
        config = {
            "url": "https://api.example.com/notify",
            "basic_auth": ["user", "pass"],
        }

        handler = create_handler_from_config("webhook", config)

        assert handler.webhook_config.auth_type == "basic"

    def test_with_name_parameter(self) -> None:
        """Test creating handler with custom name."""
        config = {
            "webhook_url": "https://hooks.slack.com/services/xxx",
        }

        handler = create_handler_from_config("slack", config, name="custom_name")

        assert handler.name == "custom_name"

    def test_with_hooks_parameter(self) -> None:
        """Test creating handler with hooks."""
        config = {
            "webhook_url": "https://hooks.slack.com/services/xxx",
        }
        hooks = [LoggingNotificationHook()]

        handler = create_handler_from_config("slack", config, hooks=hooks)

        assert handler is not None


class TestCriticalHandlers:
    """Tests for critical alert handler factories."""

    def test_create_critical_slack_handler(self) -> None:
        """Test creating critical Slack handler."""
        handler = create_critical_slack_handler(
            webhook_url="https://hooks.slack.com/services/xxx",
        )

        assert isinstance(handler, SlackNotificationHandler)
        assert handler.name == "critical_slack"
        assert handler.config == CRITICAL_NOTIFICATION_CONFIG

    def test_create_critical_slack_with_channel(self) -> None:
        """Test creating critical Slack handler with channel."""
        handler = create_critical_slack_handler(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#critical-alerts",
        )

        assert handler.slack_config.channel == "#critical-alerts"

    def test_create_critical_slack_with_custom_name(self) -> None:
        """Test creating critical Slack handler with custom name."""
        handler = create_critical_slack_handler(
            webhook_url="https://hooks.slack.com/services/xxx",
            name="my_critical_slack",
        )

        assert handler.name == "my_critical_slack"

    def test_create_critical_webhook_handler(self) -> None:
        """Test creating critical webhook handler."""
        handler = create_critical_webhook_handler(
            url="https://api.example.com/critical",
        )

        assert isinstance(handler, WebhookNotificationHandler)
        assert handler.name == "critical_webhook"
        assert handler.config == CRITICAL_NOTIFICATION_CONFIG

    def test_create_critical_webhook_with_auth(self) -> None:
        """Test creating critical webhook handler with auth."""
        handler = create_critical_webhook_handler(
            url="https://api.example.com/critical",
            auth_token="secret-token",
        )

        assert handler.webhook_config.auth_type == "bearer"

    def test_create_critical_webhook_with_custom_name(self) -> None:
        """Test creating critical webhook handler with custom name."""
        handler = create_critical_webhook_handler(
            url="https://api.example.com/critical",
            name="my_critical_webhook",
        )

        assert handler.name == "my_critical_webhook"
