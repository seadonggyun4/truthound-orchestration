"""Handler factory functions.

This module provides factory functions for creating notification handlers
with common configurations. Simplifies handler creation for common use cases.

Example:
    >>> from packages.enterprise.notifications.factory import (
    ...     create_slack_handler,
    ...     create_webhook_handler,
    ...     create_handler_from_config,
    ... )
    >>>
    >>> # Create Slack handler
    >>> slack = create_slack_handler(
    ...     webhook_url="https://hooks.slack.com/services/...",
    ...     channel="#alerts",
    ... )
    >>>
    >>> # Create webhook handler
    >>> webhook = create_webhook_handler(
    ...     url="https://api.example.com/notify",
    ...     auth_token="your-token",
    ... )
"""

from __future__ import annotations

from typing import Any

from packages.enterprise.notifications.config import (
    CRITICAL_NOTIFICATION_CONFIG,
    DEFAULT_NOTIFICATION_CONFIG,
    EmailConfig,
    EmailEncryption,
    EmailProvider,
    NotificationConfig,
    SlackConfig,
    WebhookConfig,
)
from packages.enterprise.notifications.exceptions import NotificationConfigError
from packages.enterprise.notifications.handlers.base import NotificationHandler
from packages.enterprise.notifications.handlers.email import EmailNotificationHandler
from packages.enterprise.notifications.handlers.slack import SlackNotificationHandler
from packages.enterprise.notifications.handlers.webhook import (
    WebhookNotificationHandler,
)
from packages.enterprise.notifications.hooks import NotificationHook
from packages.enterprise.notifications.types import NotificationChannel, NotificationLevel


def create_slack_handler(
    webhook_url: str,
    *,
    channel: str | None = None,
    username: str | None = None,
    icon_emoji: str | None = None,
    name: str | None = None,
    config: NotificationConfig | None = None,
    hooks: list[NotificationHook] | None = None,
) -> SlackNotificationHandler:
    """Create a Slack notification handler.

    Convenience factory for creating Slack handlers with common options.

    Args:
        webhook_url: Slack webhook URL
        channel: Override channel (e.g., "#alerts")
        username: Bot username
        icon_emoji: Bot icon emoji (e.g., ":robot:")
        name: Handler name
        config: Base notification config
        hooks: Notification hooks

    Returns:
        Configured Slack handler

    Example:
        >>> handler = create_slack_handler(
        ...     webhook_url="https://hooks.slack.com/services/...",
        ...     channel="#data-quality-alerts",
        ...     username="DQ Bot",
        ...     icon_emoji=":chart_with_upwards_trend:",
        ... )
    """
    slack_config = SlackConfig(
        webhook_url=webhook_url,
        channel=channel,
        username=username,
        icon_emoji=icon_emoji,
        base_config=config or DEFAULT_NOTIFICATION_CONFIG,
    )

    return SlackNotificationHandler(
        config=slack_config,
        name=name or "slack",
        hooks=hooks,
    )


def create_webhook_handler(
    url: str,
    *,
    method: str = "POST",
    auth_token: str | None = None,
    api_key: str | None = None,
    api_key_header: str = "X-API-Key",
    basic_auth: tuple[str, str] | None = None,
    headers: dict[str, str] | None = None,
    verify_ssl: bool = True,
    name: str | None = None,
    config: NotificationConfig | None = None,
    hooks: list[NotificationHook] | None = None,
) -> WebhookNotificationHandler:
    """Create a webhook notification handler.

    Convenience factory for creating webhook handlers with common auth options.

    Args:
        url: Webhook endpoint URL
        method: HTTP method (POST, PUT, PATCH)
        auth_token: Bearer token for authentication
        api_key: API key for authentication
        api_key_header: Header name for API key
        basic_auth: Tuple of (username, password) for basic auth
        headers: Additional headers
        verify_ssl: Whether to verify SSL certificates
        name: Handler name
        config: Base notification config
        hooks: Notification hooks

    Returns:
        Configured webhook handler

    Example:
        >>> handler = create_webhook_handler(
        ...     url="https://api.example.com/notify",
        ...     auth_token="your-bearer-token",
        ... )
    """
    webhook_config = WebhookConfig(
        url=url,
        method=method,
        headers=tuple((headers or {}).items()),
        verify_ssl=verify_ssl,
        base_config=config or DEFAULT_NOTIFICATION_CONFIG,
    )

    # Apply authentication
    if auth_token:
        webhook_config = webhook_config.with_bearer_token(auth_token)
    elif api_key:
        webhook_config = webhook_config.with_api_key(api_key, api_key_header)
    elif basic_auth:
        username, password = basic_auth
        webhook_config = webhook_config.with_basic_auth(username, password)

    return WebhookNotificationHandler(
        config=webhook_config,
        name=name or "webhook",
        hooks=hooks,
    )


def create_handler_from_config(
    handler_type: str,
    config: dict[str, Any],
    *,
    name: str | None = None,
    hooks: list[NotificationHook] | None = None,
) -> NotificationHandler:
    """Create a handler from a configuration dictionary.

    Factory for creating handlers from configuration files or environment.

    Args:
        handler_type: Type of handler ("slack", "webhook")
        config: Configuration dictionary
        name: Handler name
        hooks: Notification hooks

    Returns:
        Configured handler

    Raises:
        NotificationConfigError: If configuration is invalid

    Example:
        >>> config = {
        ...     "webhook_url": "https://hooks.slack.com/services/...",
        ...     "channel": "#alerts",
        ... }
        >>> handler = create_handler_from_config("slack", config)
    """
    handler_type = handler_type.lower()

    # Parse base config if present
    base_config = None
    if "base_config" in config:
        base_config = NotificationConfig.from_dict(config["base_config"])
    elif any(k in config for k in ["timeout_seconds", "min_level", "retry_config"]):
        # Extract base config fields
        base_fields = {
            "enabled": config.get("enabled", True),
            "timeout_seconds": config.get("timeout_seconds", 30.0),
            "min_level": config.get("min_level", "info"),
            "async_send": config.get("async_send", False),
        }
        if "retry_config" in config:
            base_fields["retry_config"] = config["retry_config"]
        base_config = NotificationConfig.from_dict(base_fields)

    if handler_type == "slack":
        return _create_slack_from_config(config, name, base_config, hooks)
    elif handler_type == "webhook":
        return _create_webhook_from_config(config, name, base_config, hooks)
    elif handler_type == "email":
        return _create_email_from_config(config, name, base_config, hooks)
    else:
        raise NotificationConfigError(
            message=f"Unknown handler type: {handler_type}",
            field_name="handler_type",
            field_value=handler_type,
        )


def _create_slack_from_config(
    config: dict[str, Any],
    name: str | None,
    base_config: NotificationConfig | None,
    hooks: list[NotificationHook] | None,
) -> SlackNotificationHandler:
    """Create Slack handler from config dict."""
    webhook_url = config.get("webhook_url")
    if not webhook_url:
        raise NotificationConfigError(
            message="webhook_url is required for Slack handler",
            field_name="webhook_url",
            channel=NotificationChannel.SLACK,
        )

    return create_slack_handler(
        webhook_url=webhook_url,
        channel=config.get("channel"),
        username=config.get("username"),
        icon_emoji=config.get("icon_emoji"),
        name=name,
        config=base_config,
        hooks=hooks,
    )


def _create_webhook_from_config(
    config: dict[str, Any],
    name: str | None,
    base_config: NotificationConfig | None,
    hooks: list[NotificationHook] | None,
) -> WebhookNotificationHandler:
    """Create webhook handler from config dict."""
    url = config.get("url")
    if not url:
        raise NotificationConfigError(
            message="url is required for webhook handler",
            field_name="url",
            channel=NotificationChannel.WEBHOOK,
        )

    # Parse authentication
    auth_token = config.get("auth_token") or config.get("bearer_token")
    api_key = config.get("api_key")
    api_key_header = config.get("api_key_header", "X-API-Key")

    basic_auth = None
    if "basic_auth" in config:
        ba = config["basic_auth"]
        if isinstance(ba, dict):
            basic_auth = (ba.get("username", ""), ba.get("password", ""))
        elif isinstance(ba, (list, tuple)) and len(ba) == 2:
            basic_auth = tuple(ba)

    return create_webhook_handler(
        url=url,
        method=config.get("method", "POST"),
        auth_token=auth_token,
        api_key=api_key,
        api_key_header=api_key_header,
        basic_auth=basic_auth,
        headers=config.get("headers"),
        verify_ssl=config.get("verify_ssl", True),
        name=name,
        config=base_config,
        hooks=hooks,
    )


def create_critical_slack_handler(
    webhook_url: str,
    *,
    channel: str | None = None,
    name: str = "critical_slack",
) -> SlackNotificationHandler:
    """Create a Slack handler optimized for critical alerts.

    Uses aggressive retry settings and shorter timeouts.

    Args:
        webhook_url: Slack webhook URL
        channel: Alert channel
        name: Handler name

    Returns:
        Handler configured for critical alerts
    """
    return create_slack_handler(
        webhook_url=webhook_url,
        channel=channel,
        username="🚨 Critical Alert",
        icon_emoji=":rotating_light:",
        name=name,
        config=CRITICAL_NOTIFICATION_CONFIG,
    )


def create_critical_webhook_handler(
    url: str,
    *,
    auth_token: str | None = None,
    name: str = "critical_webhook",
) -> WebhookNotificationHandler:
    """Create a webhook handler optimized for critical alerts.

    Uses aggressive retry settings and shorter timeouts.

    Args:
        url: Webhook URL
        auth_token: Bearer token
        name: Handler name

    Returns:
        Handler configured for critical alerts
    """
    return create_webhook_handler(
        url=url,
        auth_token=auth_token,
        name=name,
        config=CRITICAL_NOTIFICATION_CONFIG,
    )


def create_email_handler(
    from_address: str,
    *,
    provider: EmailProvider | str = EmailProvider.SMTP,
    smtp_host: str | None = None,
    smtp_port: int = 587,
    smtp_username: str | None = None,
    smtp_password: str | None = None,
    encryption: EmailEncryption | str = EmailEncryption.STARTTLS,
    api_key: str | None = None,
    from_name: str | None = None,
    default_recipients: tuple[str, ...] | list[str] | None = None,
    cc_recipients: tuple[str, ...] | list[str] | None = None,
    bcc_recipients: tuple[str, ...] | list[str] | None = None,
    subject_prefix: str = "",
    reply_to: str | None = None,
    name: str | None = None,
    config: NotificationConfig | None = None,
    hooks: list[NotificationHook] | None = None,
) -> EmailNotificationHandler:
    """Create an Email notification handler.

    Convenience factory for creating Email handlers with common options.

    Args:
        from_address: Sender email address
        provider: Email provider (smtp, sendgrid, ses, mailgun, postmark, resend)
        smtp_host: SMTP server host (required for SMTP provider)
        smtp_port: SMTP server port
        smtp_username: SMTP authentication username
        smtp_password: SMTP authentication password
        encryption: Encryption mode (none, starttls, ssl_tls)
        api_key: API key (required for API providers)
        from_name: Display name for sender
        default_recipients: Default recipient list
        cc_recipients: CC recipient list
        bcc_recipients: BCC recipient list
        subject_prefix: Prefix for email subjects
        reply_to: Reply-to address
        name: Handler name
        config: Base notification config
        hooks: Notification hooks

    Returns:
        Configured Email handler

    Example:
        >>> # SMTP handler
        >>> handler = create_email_handler(
        ...     from_address="alerts@example.com",
        ...     smtp_host="smtp.example.com",
        ...     smtp_username="user",
        ...     smtp_password="pass",
        ... )
        >>>
        >>> # SendGrid handler
        >>> handler = create_email_handler(
        ...     from_address="alerts@example.com",
        ...     provider=EmailProvider.SENDGRID,
        ...     api_key="SG.xxxx",
        ... )
    """
    # Convert string provider to enum if needed
    if isinstance(provider, str):
        provider = EmailProvider(provider.lower())
    if isinstance(encryption, str):
        encryption = EmailEncryption(encryption.lower())

    email_config = EmailConfig(
        provider=provider,
        from_address=from_address,
        from_name=from_name,
        smtp_host=smtp_host or "",
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        encryption=encryption,
        api_key=api_key,
        default_recipients=tuple(default_recipients or ()),
        cc_recipients=tuple(cc_recipients or ()),
        bcc_recipients=tuple(bcc_recipients or ()),
        subject_prefix=subject_prefix,
        reply_to=reply_to,
        base_config=config or DEFAULT_NOTIFICATION_CONFIG,
    )

    return EmailNotificationHandler(
        config=email_config,
        name=name or "email",
        hooks=hooks,
    )


def create_smtp_email_handler(
    from_address: str,
    smtp_host: str,
    *,
    smtp_port: int = 587,
    smtp_username: str | None = None,
    smtp_password: str | None = None,
    encryption: EmailEncryption | str = EmailEncryption.STARTTLS,
    from_name: str | None = None,
    default_recipients: tuple[str, ...] | list[str] | None = None,
    subject_prefix: str = "",
    name: str | None = None,
    config: NotificationConfig | None = None,
    hooks: list[NotificationHook] | None = None,
) -> EmailNotificationHandler:
    """Create an SMTP-based Email notification handler.

    Convenience factory for creating SMTP email handlers.

    Args:
        from_address: Sender email address
        smtp_host: SMTP server host
        smtp_port: SMTP server port (default: 587)
        smtp_username: SMTP authentication username
        smtp_password: SMTP authentication password
        encryption: Encryption mode (default: STARTTLS)
        from_name: Display name for sender
        default_recipients: Default recipient list
        subject_prefix: Prefix for email subjects
        name: Handler name
        config: Base notification config
        hooks: Notification hooks

    Returns:
        Configured SMTP Email handler

    Example:
        >>> handler = create_smtp_email_handler(
        ...     from_address="alerts@example.com",
        ...     smtp_host="smtp.gmail.com",
        ...     smtp_port=587,
        ...     smtp_username="alerts@example.com",
        ...     smtp_password="app-password",
        ... )
    """
    return create_email_handler(
        from_address=from_address,
        provider=EmailProvider.SMTP,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
        encryption=encryption,
        from_name=from_name,
        default_recipients=default_recipients,
        subject_prefix=subject_prefix,
        name=name,
        config=config,
        hooks=hooks,
    )


def create_sendgrid_email_handler(
    from_address: str,
    api_key: str,
    *,
    from_name: str | None = None,
    default_recipients: tuple[str, ...] | list[str] | None = None,
    subject_prefix: str = "",
    name: str | None = None,
    config: NotificationConfig | None = None,
    hooks: list[NotificationHook] | None = None,
) -> EmailNotificationHandler:
    """Create a SendGrid-based Email notification handler.

    Args:
        from_address: Sender email address
        api_key: SendGrid API key
        from_name: Display name for sender
        default_recipients: Default recipient list
        subject_prefix: Prefix for email subjects
        name: Handler name
        config: Base notification config
        hooks: Notification hooks

    Returns:
        Configured SendGrid Email handler

    Example:
        >>> handler = create_sendgrid_email_handler(
        ...     from_address="alerts@example.com",
        ...     api_key="SG.xxxxxxxxxxxx",
        ... )
    """
    return create_email_handler(
        from_address=from_address,
        provider=EmailProvider.SENDGRID,
        api_key=api_key,
        from_name=from_name,
        default_recipients=default_recipients,
        subject_prefix=subject_prefix,
        name=name,
        config=config,
        hooks=hooks,
    )


def create_ses_email_handler(
    from_address: str,
    *,
    aws_access_key: str | None = None,
    aws_secret_key: str | None = None,
    aws_region: str = "us-east-1",
    from_name: str | None = None,
    default_recipients: tuple[str, ...] | list[str] | None = None,
    subject_prefix: str = "",
    name: str | None = None,
    config: NotificationConfig | None = None,
    hooks: list[NotificationHook] | None = None,
) -> EmailNotificationHandler:
    """Create an AWS SES-based Email notification handler.

    Args:
        from_address: Sender email address
        aws_access_key: AWS access key ID (uses boto3 defaults if not provided)
        aws_secret_key: AWS secret access key
        aws_region: AWS region
        from_name: Display name for sender
        default_recipients: Default recipient list
        subject_prefix: Prefix for email subjects
        name: Handler name
        config: Base notification config
        hooks: Notification hooks

    Returns:
        Configured AWS SES Email handler

    Example:
        >>> handler = create_ses_email_handler(
        ...     from_address="alerts@example.com",
        ...     aws_region="us-west-2",
        ... )
    """
    email_config = EmailConfig(
        provider=EmailProvider.SES,
        from_address=from_address,
        from_name=from_name,
        api_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_region=aws_region,
        default_recipients=tuple(default_recipients or ()),
        subject_prefix=subject_prefix,
        base_config=config or DEFAULT_NOTIFICATION_CONFIG,
    )

    return EmailNotificationHandler(
        config=email_config,
        name=name or "email",
        hooks=hooks,
    )


def create_critical_email_handler(
    from_address: str,
    *,
    smtp_host: str | None = None,
    api_key: str | None = None,
    provider: EmailProvider | str = EmailProvider.SMTP,
    default_recipients: tuple[str, ...] | list[str] | None = None,
    name: str = "critical_email",
) -> EmailNotificationHandler:
    """Create an Email handler optimized for critical alerts.

    Uses aggressive retry settings and shorter timeouts.

    Args:
        from_address: Sender email address
        smtp_host: SMTP host (for SMTP provider)
        api_key: API key (for API providers)
        provider: Email provider
        default_recipients: Default recipient list
        name: Handler name

    Returns:
        Handler configured for critical alerts
    """
    if isinstance(provider, str):
        provider = EmailProvider(provider.lower())

    return create_email_handler(
        from_address=from_address,
        provider=provider,
        smtp_host=smtp_host,
        api_key=api_key,
        default_recipients=default_recipients,
        from_name="🚨 Critical Alert",
        subject_prefix="[CRITICAL] ",
        name=name,
        config=CRITICAL_NOTIFICATION_CONFIG,
    )


def _create_email_from_config(
    config: dict[str, Any],
    name: str | None,
    base_config: NotificationConfig | None,
    hooks: list[NotificationHook] | None,
) -> EmailNotificationHandler:
    """Create Email handler from config dict."""
    from_address = config.get("from_address")
    if not from_address:
        raise NotificationConfigError(
            message="from_address is required for Email handler",
            field_name="from_address",
            channel=NotificationChannel.EMAIL,
        )

    provider = config.get("provider", "smtp")
    if isinstance(provider, str):
        provider = EmailProvider(provider.lower())

    encryption = config.get("encryption", "starttls")
    if isinstance(encryption, str):
        encryption = EmailEncryption(encryption.lower())

    return create_email_handler(
        from_address=from_address,
        provider=provider,
        smtp_host=config.get("smtp_host"),
        smtp_port=config.get("smtp_port", 587),
        smtp_username=config.get("smtp_username"),
        smtp_password=config.get("smtp_password"),
        encryption=encryption,
        api_key=config.get("api_key"),
        from_name=config.get("from_name"),
        default_recipients=config.get("default_recipients"),
        cc_recipients=config.get("cc_recipients"),
        bcc_recipients=config.get("bcc_recipients"),
        subject_prefix=config.get("subject_prefix", ""),
        reply_to=config.get("reply_to"),
        name=name,
        config=base_config,
        hooks=hooks,
    )
