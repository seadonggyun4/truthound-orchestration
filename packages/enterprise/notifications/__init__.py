"""Truthound Orchestration Notifications Module.

Multi-channel notification system for data quality alerts and SLA violations.
Supports Slack, Webhook, Email, and extensible to other channels.

Features:
    - Protocol-based handler abstraction for easy extensibility
    - Async-first design with sync compatibility
    - Built-in retry and circuit breaker integration
    - Flexible message formatting (text, Markdown, structured)
    - Hook system for logging, metrics, and custom processing
    - Central registry for multi-channel dispatch

Quick Start:
    >>> from packages.enterprise.notifications import (
    ...     SlackNotificationHandler,
    ...     WebhookNotificationHandler,
    ...     get_notification_registry,
    ...     NotificationLevel,
    ... )
    >>>
    >>> # Get global registry
    >>> registry = get_notification_registry()
    >>>
    >>> # Register handlers
    >>> registry.register("slack", SlackNotificationHandler(
    ...     webhook_url="https://hooks.slack.com/services/...",
    ... ))
    >>> registry.register("webhook", WebhookNotificationHandler(
    ...     url="https://api.example.com/notify",
    ... ))
    >>>
    >>> # Send to all channels
    >>> results = await registry.notify_all(
    ...     message="Data quality check failed!",
    ...     level=NotificationLevel.ERROR,
    ...     context={"check_id": "chk_123", "failure_count": 5},
    ... )

Extending with Custom Handlers:
    >>> from packages.enterprise.notifications import NotificationHandler, NotificationResult
    >>>
    >>> class MyCustomHandler(NotificationHandler):
    ...     async def send(self, message, level, context) -> NotificationResult:
    ...         # Custom implementation
    ...         ...
    ...         return NotificationResult(success=True, channel=self.channel)
"""

# =============================================================================
# Types and Enums
# =============================================================================
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationStatus,
    NotificationResult,
    NotificationPayload,
    NotificationMetadata,
)

# =============================================================================
# Exceptions
# =============================================================================
from packages.enterprise.notifications.exceptions import (
    NotificationError,
    NotificationSendError,
    NotificationTimeoutError,
    NotificationRetryExhaustedError,
    NotificationConfigError,
    NotificationHandlerNotFoundError,
    NotificationFormatterError,
)

# =============================================================================
# Configuration
# =============================================================================
from packages.enterprise.notifications.config import (
    NotificationConfig,
    SlackConfig,
    WebhookConfig,
    EmailConfig,
    EmailEncryption,
    EmailProvider,
    RetryConfig as NotificationRetryConfig,
    # Presets
    DEFAULT_NOTIFICATION_CONFIG,
    CRITICAL_NOTIFICATION_CONFIG,
    LENIENT_NOTIFICATION_CONFIG,
    FAST_NOTIFICATION_CONFIG,
)

# =============================================================================
# Handlers
# =============================================================================
from packages.enterprise.notifications.handlers.base import (
    NotificationHandler,
    AsyncNotificationHandler,
    SyncNotificationHandler,
)
from packages.enterprise.notifications.handlers.email import (
    EmailNotificationHandler,
)
from packages.enterprise.notifications.handlers.slack import (
    SlackNotificationHandler,
)
from packages.enterprise.notifications.handlers.webhook import (
    WebhookNotificationHandler,
)

# =============================================================================
# Formatters
# =============================================================================
from packages.enterprise.notifications.formatters import (
    MessageFormatter,
    TextFormatter,
    MarkdownFormatter,
    SlackBlockFormatter,
    JsonFormatter,
    TemplateFormatter,
    get_formatter,
    register_formatter,
)

# =============================================================================
# Hooks
# =============================================================================
from packages.enterprise.notifications.hooks import (
    NotificationHook,
    AsyncNotificationHook,
    LoggingNotificationHook,
    MetricsNotificationHook,
    CompositeNotificationHook,
    CallbackNotificationHook,
)

# =============================================================================
# Registry
# =============================================================================
from packages.enterprise.notifications.registry import (
    NotificationRegistry,
    NotificationDispatcher,
    get_notification_registry,
    reset_notification_registry,
    register_handler,
    unregister_handler,
    notify,
    notify_all,
)

# =============================================================================
# Factory Functions
# =============================================================================
from packages.enterprise.notifications.factory import (
    create_email_handler,
    create_handler_from_config,
    create_sendgrid_email_handler,
    create_ses_email_handler,
    create_slack_handler,
    create_smtp_email_handler,
    create_webhook_handler,
)

__all__ = [
    # Types and Enums
    "NotificationChannel",
    "NotificationLevel",
    "NotificationStatus",
    "NotificationResult",
    "NotificationPayload",
    "NotificationMetadata",
    # Exceptions
    "NotificationError",
    "NotificationSendError",
    "NotificationTimeoutError",
    "NotificationRetryExhaustedError",
    "NotificationConfigError",
    "NotificationHandlerNotFoundError",
    "NotificationFormatterError",
    # Configuration
    "NotificationConfig",
    "SlackConfig",
    "WebhookConfig",
    "EmailConfig",
    "EmailEncryption",
    "EmailProvider",
    "NotificationRetryConfig",
    "DEFAULT_NOTIFICATION_CONFIG",
    "CRITICAL_NOTIFICATION_CONFIG",
    "LENIENT_NOTIFICATION_CONFIG",
    "FAST_NOTIFICATION_CONFIG",
    # Handlers
    "NotificationHandler",
    "AsyncNotificationHandler",
    "SyncNotificationHandler",
    "SlackNotificationHandler",
    "WebhookNotificationHandler",
    "EmailNotificationHandler",
    # Formatters
    "MessageFormatter",
    "TextFormatter",
    "MarkdownFormatter",
    "SlackBlockFormatter",
    "JsonFormatter",
    "TemplateFormatter",
    "get_formatter",
    "register_formatter",
    # Hooks
    "NotificationHook",
    "AsyncNotificationHook",
    "LoggingNotificationHook",
    "MetricsNotificationHook",
    "CompositeNotificationHook",
    "CallbackNotificationHook",
    # Registry
    "NotificationRegistry",
    "NotificationDispatcher",
    "get_notification_registry",
    "reset_notification_registry",
    "register_handler",
    "unregister_handler",
    "notify",
    "notify_all",
    # Factory
    "create_slack_handler",
    "create_webhook_handler",
    "create_email_handler",
    "create_smtp_email_handler",
    "create_sendgrid_email_handler",
    "create_ses_email_handler",
    "create_handler_from_config",
]
