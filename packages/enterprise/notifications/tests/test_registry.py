"""Tests for notification registry and dispatcher."""

import pytest

from packages.enterprise.notifications.config import SlackConfig, WebhookConfig
from packages.enterprise.notifications.exceptions import NotificationHandlerNotFoundError
from packages.enterprise.notifications.handlers.slack import SlackNotificationHandler
from packages.enterprise.notifications.handlers.webhook import WebhookNotificationHandler
from packages.enterprise.notifications.registry import (
    DEFAULT_DISPATCH_CONFIG,
    FAILOVER_DISPATCH_CONFIG,
    SEQUENTIAL_DISPATCH_CONFIG,
    DispatchConfig,
    NotificationDispatcher,
    NotificationRegistry,
    get_notification_registry,
    register_handler,
    reset_notification_registry,
    unregister_handler,
)
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
)


class TestDispatchConfig:
    """Tests for DispatchConfig."""

    def test_default_config(self) -> None:
        """Test default dispatch configuration."""
        config = DispatchConfig()

        assert config.parallel is True
        assert config.max_workers == 10
        assert config.stop_on_first_success is False
        assert config.stop_on_first_failure is False
        assert config.enabled_handlers is None
        assert config.disabled_handlers == frozenset()

    def test_preset_configs(self) -> None:
        """Test preset configurations."""
        assert DEFAULT_DISPATCH_CONFIG.parallel is True
        assert SEQUENTIAL_DISPATCH_CONFIG.parallel is False
        assert FAILOVER_DISPATCH_CONFIG.parallel is False
        assert FAILOVER_DISPATCH_CONFIG.stop_on_first_success is True

    def test_with_parallel(self) -> None:
        """Test parallel mode modification."""
        config = DispatchConfig()
        new_config = config.with_parallel(False)

        assert config.parallel is True  # Original unchanged
        assert new_config.parallel is False

    def test_with_handlers(self) -> None:
        """Test handler filtering modification."""
        config = DispatchConfig()
        new_config = config.with_handlers(
            enabled=["slack", "webhook"],
            disabled=["email"],
        )

        assert config.enabled_handlers is None  # Original unchanged
        assert new_config.enabled_handlers == frozenset(["slack", "webhook"])
        assert new_config.disabled_handlers == frozenset(["email"])


class TestNotificationDispatcher:
    """Tests for NotificationDispatcher."""

    def test_initialization(self) -> None:
        """Test dispatcher initialization."""
        dispatcher = NotificationDispatcher()

        assert dispatcher.config == DEFAULT_DISPATCH_CONFIG

    def test_custom_config(self) -> None:
        """Test dispatcher with custom config."""
        config = DispatchConfig(parallel=False, max_workers=5)
        dispatcher = NotificationDispatcher(config=config)

        assert dispatcher.config.parallel is False
        assert dispatcher.config.max_workers == 5

    @pytest.mark.asyncio
    async def test_dispatch_empty_handlers(self) -> None:
        """Test dispatch with no handlers."""
        dispatcher = NotificationDispatcher()
        payload = NotificationPayload(message="Test")

        result = await dispatcher.dispatch({}, payload)

        assert result.total_count == 0
        assert result.success_count == 0

    @pytest.mark.asyncio
    async def test_filter_disabled_handlers(self) -> None:
        """Test that disabled handlers are filtered."""
        config = DispatchConfig(disabled_handlers=frozenset(["webhook"]))
        dispatcher = NotificationDispatcher(config=config)

        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        slack = SlackNotificationHandler(config=slack_config, name="slack")

        webhook_config = WebhookConfig(url="https://example.com/webhook")
        webhook = WebhookNotificationHandler(config=webhook_config, name="webhook")

        handlers = {"slack": slack, "webhook": webhook}
        filtered = dispatcher._filter_handlers(handlers)

        assert "slack" in filtered
        assert "webhook" not in filtered

    @pytest.mark.asyncio
    async def test_filter_enabled_handlers(self) -> None:
        """Test that only enabled handlers are used."""
        config = DispatchConfig(enabled_handlers=frozenset(["slack"]))
        dispatcher = NotificationDispatcher(config=config)

        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        slack = SlackNotificationHandler(config=slack_config, name="slack")

        webhook_config = WebhookConfig(url="https://example.com/webhook")
        webhook = WebhookNotificationHandler(config=webhook_config, name="webhook")

        handlers = {"slack": slack, "webhook": webhook}
        filtered = dispatcher._filter_handlers(handlers)

        assert "slack" in filtered
        assert "webhook" not in filtered


class TestNotificationRegistry:
    """Tests for NotificationRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> None:
        """Reset registry before and after each test."""
        reset_notification_registry()
        yield
        reset_notification_registry()

    def test_singleton_pattern(self) -> None:
        """Test that registry is a singleton."""
        registry1 = NotificationRegistry()
        registry2 = NotificationRegistry()

        assert registry1 is registry2

    def test_register_handler(self) -> None:
        """Test handler registration."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        registry.register("slack", handler)

        assert registry.get("slack") is handler
        assert "slack" in registry.list_handlers()

    def test_register_duplicate_raises(self) -> None:
        """Test that duplicate registration raises error."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler1 = SlackNotificationHandler(config=slack_config, name="slack1")
        handler2 = SlackNotificationHandler(config=slack_config, name="slack2")

        registry.register("slack", handler1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("slack", handler2)

    def test_register_overwrite(self) -> None:
        """Test overwriting existing registration."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler1 = SlackNotificationHandler(config=slack_config, name="slack1")
        handler2 = SlackNotificationHandler(config=slack_config, name="slack2")

        registry.register("slack", handler1)
        registry.register("slack", handler2, overwrite=True)

        assert registry.get("slack") is handler2

    def test_unregister_handler(self) -> None:
        """Test handler unregistration."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        registry.register("slack", handler)
        removed = registry.unregister("slack")

        assert removed is handler
        assert registry.get("slack") is None

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering non-existent handler."""
        registry = get_notification_registry()
        result = registry.unregister("nonexistent")

        assert result is None

    def test_get_required_found(self) -> None:
        """Test get_required with existing handler."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        registry.register("slack", handler)
        result = registry.get_required("slack")

        assert result is handler

    def test_get_required_not_found(self) -> None:
        """Test get_required with missing handler."""
        registry = get_notification_registry()

        with pytest.raises(NotificationHandlerNotFoundError) as exc_info:
            registry.get_required("nonexistent")

        assert exc_info.value.requested_name == "nonexistent"

    def test_list_handlers(self) -> None:
        """Test listing all handlers."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        slack = SlackNotificationHandler(config=slack_config, name="slack")

        webhook_config = WebhookConfig(url="https://example.com/webhook")
        webhook = WebhookNotificationHandler(config=webhook_config, name="webhook")

        registry.register("slack", slack)
        registry.register("webhook", webhook)

        handlers = registry.list_handlers()

        assert "slack" in handlers
        assert "webhook" in handlers

    def test_get_handlers_by_channel(self) -> None:
        """Test filtering handlers by channel."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        slack = SlackNotificationHandler(config=slack_config, name="slack")

        webhook_config = WebhookConfig(url="https://example.com/webhook")
        webhook = WebhookNotificationHandler(config=webhook_config, name="webhook")

        registry.register("slack", slack)
        registry.register("webhook", webhook)

        slack_handlers = registry.get_handlers_by_channel(NotificationChannel.SLACK)
        webhook_handlers = registry.get_handlers_by_channel(NotificationChannel.WEBHOOK)

        assert "slack" in slack_handlers
        assert "webhook" not in slack_handlers
        assert "webhook" in webhook_handlers
        assert "slack" not in webhook_handlers

    def test_clear(self) -> None:
        """Test clearing all handlers."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        registry.register("slack", handler)
        registry.clear()

        assert registry.list_handlers() == []

    def test_reset(self) -> None:
        """Test resetting registry to initial state."""
        registry = get_notification_registry()
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        registry.register("slack", handler)
        registry.reset()

        assert registry.list_handlers() == []


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> None:
        """Reset registry before and after each test."""
        reset_notification_registry()
        yield
        reset_notification_registry()

    def test_register_handler_function(self) -> None:
        """Test register_handler convenience function."""
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        register_handler("slack", handler)

        registry = get_notification_registry()
        assert registry.get("slack") is handler

    def test_unregister_handler_function(self) -> None:
        """Test unregister_handler convenience function."""
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        register_handler("slack", handler)
        removed = unregister_handler("slack")

        assert removed is handler
        registry = get_notification_registry()
        assert registry.get("slack") is None

    def test_reset_notification_registry(self) -> None:
        """Test registry reset function."""
        slack_config = SlackConfig(webhook_url="https://hooks.slack.com/test")
        handler = SlackNotificationHandler(config=slack_config, name="slack")

        register_handler("slack", handler)
        reset_notification_registry()

        registry = get_notification_registry()
        assert registry.list_handlers() == []
