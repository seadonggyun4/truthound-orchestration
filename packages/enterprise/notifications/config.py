"""Notification configuration.

This module provides immutable configuration classes for notification handlers.
All configurations follow the builder pattern for fluent modification while
maintaining immutability.

Features:
    - Frozen dataclasses for thread-safety and immutability
    - Builder methods for configuration modification
    - Preset configurations for common use cases
    - Validation in __post_init__

Example:
    >>> from packages.enterprise.notifications.config import (
    ...     NotificationConfig,
    ...     SlackConfig,
    ...     DEFAULT_NOTIFICATION_CONFIG,
    ... )
    >>>
    >>> # Use preset
    >>> config = DEFAULT_NOTIFICATION_CONFIG
    >>>
    >>> # Or customize with builder pattern
    >>> config = NotificationConfig().with_timeout(10.0).with_retry(count=5)
    >>>
    >>> # Channel-specific config
    >>> slack_config = SlackConfig(
    ...     webhook_url="https://hooks.slack.com/services/...",
    ...     channel="#alerts",
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from packages.enterprise.notifications.exceptions import NotificationConfigError
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
)


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay_seconds: Initial delay between retries
        max_delay_seconds: Maximum delay between retries
        exponential_backoff: Whether to use exponential backoff
        jitter: Whether to add random jitter to delays
    """

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise NotificationConfigError(
                message="max_retries must be non-negative",
                field_name="max_retries",
                field_value=self.max_retries,
            )
        if self.base_delay_seconds <= 0:
            raise NotificationConfigError(
                message="base_delay_seconds must be positive",
                field_name="base_delay_seconds",
                field_value=self.base_delay_seconds,
            )
        if self.max_delay_seconds < self.base_delay_seconds:
            raise NotificationConfigError(
                message="max_delay_seconds must be >= base_delay_seconds",
                field_name="max_delay_seconds",
                field_value=self.max_delay_seconds,
            )

    def with_max_retries(self, max_retries: int) -> RetryConfig:
        """Create a copy with a new max_retries value."""
        return RetryConfig(
            max_retries=max_retries,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            exponential_backoff=self.exponential_backoff,
            jitter=self.jitter,
        )

    def with_delays(
        self,
        base_delay_seconds: float | None = None,
        max_delay_seconds: float | None = None,
    ) -> RetryConfig:
        """Create a copy with new delay values."""
        return RetryConfig(
            max_retries=self.max_retries,
            base_delay_seconds=base_delay_seconds or self.base_delay_seconds,
            max_delay_seconds=max_delay_seconds or self.max_delay_seconds,
            exponential_backoff=self.exponential_backoff,
            jitter=self.jitter,
        )

    def with_exponential_backoff(self, enabled: bool = True) -> RetryConfig:
        """Create a copy with exponential backoff enabled/disabled."""
        return RetryConfig(
            max_retries=self.max_retries,
            base_delay_seconds=self.base_delay_seconds,
            max_delay_seconds=self.max_delay_seconds,
            exponential_backoff=enabled,
            jitter=self.jitter,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_retries": self.max_retries,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "exponential_backoff": self.exponential_backoff,
            "jitter": self.jitter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetryConfig:
        """Create from dictionary."""
        return cls(
            max_retries=data.get("max_retries", 3),
            base_delay_seconds=data.get("base_delay_seconds", 1.0),
            max_delay_seconds=data.get("max_delay_seconds", 30.0),
            exponential_backoff=data.get("exponential_backoff", True),
            jitter=data.get("jitter", True),
        )


@dataclass(frozen=True, slots=True)
class NotificationConfig:
    """Base configuration for notification handlers.

    This is the shared configuration that applies to all notification handlers.
    Channel-specific configurations extend or compose with this.

    Attributes:
        enabled: Whether notifications are enabled
        timeout_seconds: Timeout for send operations
        retry_config: Retry behavior configuration
        min_level: Minimum notification level to send
        tags: Tags to filter notifications
        async_send: Whether to send asynchronously (fire-and-forget)
        include_metadata: Whether to include metadata in notifications
        formatter_name: Name of the formatter to use
        rate_limit_requests: Maximum requests per window (0 = no limit)
        rate_limit_window_seconds: Rate limit time window
    """

    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    min_level: NotificationLevel = NotificationLevel.INFO
    tags: frozenset[str] = field(default_factory=frozenset)
    async_send: bool = False
    include_metadata: bool = True
    formatter_name: str | None = None
    rate_limit_requests: int = 0
    rate_limit_window_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise NotificationConfigError(
                message="timeout_seconds must be positive",
                field_name="timeout_seconds",
                field_value=self.timeout_seconds,
            )
        if self.rate_limit_requests < 0:
            raise NotificationConfigError(
                message="rate_limit_requests must be non-negative",
                field_name="rate_limit_requests",
                field_value=self.rate_limit_requests,
            )

    def with_enabled(self, enabled: bool) -> NotificationConfig:
        """Create a copy with enabled/disabled state."""
        return NotificationConfig(
            enabled=enabled,
            timeout_seconds=self.timeout_seconds,
            retry_config=self.retry_config,
            min_level=self.min_level,
            tags=self.tags,
            async_send=self.async_send,
            include_metadata=self.include_metadata,
            formatter_name=self.formatter_name,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )

    def with_timeout(self, timeout_seconds: float) -> NotificationConfig:
        """Create a copy with a new timeout."""
        return NotificationConfig(
            enabled=self.enabled,
            timeout_seconds=timeout_seconds,
            retry_config=self.retry_config,
            min_level=self.min_level,
            tags=self.tags,
            async_send=self.async_send,
            include_metadata=self.include_metadata,
            formatter_name=self.formatter_name,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )

    def with_retry(
        self,
        count: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
    ) -> NotificationConfig:
        """Create a copy with modified retry settings."""
        new_retry = self.retry_config
        if count is not None:
            new_retry = new_retry.with_max_retries(count)
        if base_delay is not None or max_delay is not None:
            new_retry = new_retry.with_delays(base_delay, max_delay)

        return NotificationConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retry_config=new_retry,
            min_level=self.min_level,
            tags=self.tags,
            async_send=self.async_send,
            include_metadata=self.include_metadata,
            formatter_name=self.formatter_name,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )

    def with_min_level(self, level: NotificationLevel) -> NotificationConfig:
        """Create a copy with a new minimum level."""
        return NotificationConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retry_config=self.retry_config,
            min_level=level,
            tags=self.tags,
            async_send=self.async_send,
            include_metadata=self.include_metadata,
            formatter_name=self.formatter_name,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )

    def with_tags(self, *tags: str) -> NotificationConfig:
        """Create a copy with additional tags."""
        return NotificationConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retry_config=self.retry_config,
            min_level=self.min_level,
            tags=self.tags | frozenset(tags),
            async_send=self.async_send,
            include_metadata=self.include_metadata,
            formatter_name=self.formatter_name,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )

    def with_async_send(self, async_send: bool = True) -> NotificationConfig:
        """Create a copy with async send enabled/disabled."""
        return NotificationConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retry_config=self.retry_config,
            min_level=self.min_level,
            tags=self.tags,
            async_send=async_send,
            include_metadata=self.include_metadata,
            formatter_name=self.formatter_name,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )

    def with_formatter(self, formatter_name: str) -> NotificationConfig:
        """Create a copy with a specific formatter."""
        return NotificationConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retry_config=self.retry_config,
            min_level=self.min_level,
            tags=self.tags,
            async_send=self.async_send,
            include_metadata=self.include_metadata,
            formatter_name=formatter_name,
            rate_limit_requests=self.rate_limit_requests,
            rate_limit_window_seconds=self.rate_limit_window_seconds,
        )

    def with_rate_limit(
        self,
        requests: int,
        window_seconds: float = 60.0,
    ) -> NotificationConfig:
        """Create a copy with rate limiting."""
        return NotificationConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retry_config=self.retry_config,
            min_level=self.min_level,
            tags=self.tags,
            async_send=self.async_send,
            include_metadata=self.include_metadata,
            formatter_name=self.formatter_name,
            rate_limit_requests=requests,
            rate_limit_window_seconds=window_seconds,
        )

    def should_send(self, level: NotificationLevel) -> bool:
        """Check if a notification at the given level should be sent."""
        return self.enabled and level >= self.min_level

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "retry_config": self.retry_config.to_dict(),
            "min_level": self.min_level.value,
            "tags": list(self.tags),
            "async_send": self.async_send,
            "include_metadata": self.include_metadata,
            "formatter_name": self.formatter_name,
            "rate_limit_requests": self.rate_limit_requests,
            "rate_limit_window_seconds": self.rate_limit_window_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotificationConfig:
        """Create from dictionary."""
        retry_data = data.get("retry_config", {})
        return cls(
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            retry_config=RetryConfig.from_dict(retry_data),
            min_level=NotificationLevel(data.get("min_level", "info")),
            tags=frozenset(data.get("tags", [])),
            async_send=data.get("async_send", False),
            include_metadata=data.get("include_metadata", True),
            formatter_name=data.get("formatter_name"),
            rate_limit_requests=data.get("rate_limit_requests", 0),
            rate_limit_window_seconds=data.get("rate_limit_window_seconds", 60.0),
        )


@dataclass(frozen=True, slots=True)
class SlackConfig:
    """Configuration for Slack notifications.

    Attributes:
        webhook_url: Slack incoming webhook URL
        channel: Override channel (optional, uses webhook default)
        username: Bot username to display
        icon_emoji: Emoji icon for the bot (e.g., ":robot:")
        icon_url: URL for bot icon (alternative to emoji)
        unfurl_links: Whether to unfurl links
        unfurl_media: Whether to unfurl media
        mrkdwn: Whether to enable Markdown formatting
        base_config: Base notification configuration
    """

    webhook_url: str
    channel: str | None = None
    username: str | None = None
    icon_emoji: str | None = None
    icon_url: str | None = None
    unfurl_links: bool = False
    unfurl_media: bool = True
    mrkdwn: bool = True
    base_config: NotificationConfig = field(default_factory=NotificationConfig)

    def __post_init__(self) -> None:
        if not self.webhook_url:
            raise NotificationConfigError(
                message="webhook_url is required",
                field_name="webhook_url",
                channel=NotificationChannel.SLACK,
            )
        if not self.webhook_url.startswith("https://"):
            raise NotificationConfigError(
                message="webhook_url must be HTTPS",
                field_name="webhook_url",
                field_value=self.webhook_url[:50],
                channel=NotificationChannel.SLACK,
            )

    def with_channel(self, channel: str) -> SlackConfig:
        """Create a copy with a different channel."""
        return SlackConfig(
            webhook_url=self.webhook_url,
            channel=channel,
            username=self.username,
            icon_emoji=self.icon_emoji,
            icon_url=self.icon_url,
            unfurl_links=self.unfurl_links,
            unfurl_media=self.unfurl_media,
            mrkdwn=self.mrkdwn,
            base_config=self.base_config,
        )

    def with_username(self, username: str) -> SlackConfig:
        """Create a copy with a different username."""
        return SlackConfig(
            webhook_url=self.webhook_url,
            channel=self.channel,
            username=username,
            icon_emoji=self.icon_emoji,
            icon_url=self.icon_url,
            unfurl_links=self.unfurl_links,
            unfurl_media=self.unfurl_media,
            mrkdwn=self.mrkdwn,
            base_config=self.base_config,
        )

    def with_icon(
        self,
        emoji: str | None = None,
        url: str | None = None,
    ) -> SlackConfig:
        """Create a copy with a different icon."""
        return SlackConfig(
            webhook_url=self.webhook_url,
            channel=self.channel,
            username=self.username,
            icon_emoji=emoji or self.icon_emoji,
            icon_url=url or self.icon_url,
            unfurl_links=self.unfurl_links,
            unfurl_media=self.unfurl_media,
            mrkdwn=self.mrkdwn,
            base_config=self.base_config,
        )

    def with_base_config(self, config: NotificationConfig) -> SlackConfig:
        """Create a copy with different base config."""
        return SlackConfig(
            webhook_url=self.webhook_url,
            channel=self.channel,
            username=self.username,
            icon_emoji=self.icon_emoji,
            icon_url=self.icon_url,
            unfurl_links=self.unfurl_links,
            unfurl_media=self.unfurl_media,
            mrkdwn=self.mrkdwn,
            base_config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes sensitive webhook_url)."""
        return {
            "webhook_url": "***MASKED***",
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "icon_url": self.icon_url,
            "unfurl_links": self.unfurl_links,
            "unfurl_media": self.unfurl_media,
            "mrkdwn": self.mrkdwn,
            "base_config": self.base_config.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WebhookConfig:
    """Configuration for generic webhook notifications.

    Supports various webhook formats and authentication methods.

    Attributes:
        url: Webhook endpoint URL
        method: HTTP method (POST, PUT)
        headers: Additional headers to include
        auth_type: Authentication type (none, basic, bearer, api_key)
        auth_credentials: Credentials for authentication (masked in logs)
        content_type: Content-Type header value
        verify_ssl: Whether to verify SSL certificates
        base_config: Base notification configuration
    """

    url: str
    method: str = "POST"
    headers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    auth_type: str = "none"
    auth_credentials: str | None = None
    content_type: str = "application/json"
    verify_ssl: bool = True
    base_config: NotificationConfig = field(default_factory=NotificationConfig)

    def __post_init__(self) -> None:
        if not self.url:
            raise NotificationConfigError(
                message="url is required",
                field_name="url",
                channel=NotificationChannel.WEBHOOK,
            )
        if self.method.upper() not in {"POST", "PUT", "PATCH"}:
            raise NotificationConfigError(
                message="method must be POST, PUT, or PATCH",
                field_name="method",
                field_value=self.method,
                channel=NotificationChannel.WEBHOOK,
            )
        if self.auth_type not in {"none", "basic", "bearer", "api_key"}:
            raise NotificationConfigError(
                message="auth_type must be one of: none, basic, bearer, api_key",
                field_name="auth_type",
                field_value=self.auth_type,
                channel=NotificationChannel.WEBHOOK,
            )

    @property
    def headers_dict(self) -> dict[str, str]:
        """Get headers as a dictionary."""
        return dict(self.headers)

    def with_url(self, url: str) -> WebhookConfig:
        """Create a copy with a different URL."""
        return WebhookConfig(
            url=url,
            method=self.method,
            headers=self.headers,
            auth_type=self.auth_type,
            auth_credentials=self.auth_credentials,
            content_type=self.content_type,
            verify_ssl=self.verify_ssl,
            base_config=self.base_config,
        )

    def with_header(self, key: str, value: str) -> WebhookConfig:
        """Create a copy with an additional header."""
        return WebhookConfig(
            url=self.url,
            method=self.method,
            headers=self.headers + ((key, value),),
            auth_type=self.auth_type,
            auth_credentials=self.auth_credentials,
            content_type=self.content_type,
            verify_ssl=self.verify_ssl,
            base_config=self.base_config,
        )

    def with_basic_auth(self, username: str, password: str) -> WebhookConfig:
        """Create a copy with basic authentication."""
        import base64

        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        return WebhookConfig(
            url=self.url,
            method=self.method,
            headers=self.headers,
            auth_type="basic",
            auth_credentials=credentials,
            content_type=self.content_type,
            verify_ssl=self.verify_ssl,
            base_config=self.base_config,
        )

    def with_bearer_token(self, token: str) -> WebhookConfig:
        """Create a copy with bearer token authentication."""
        return WebhookConfig(
            url=self.url,
            method=self.method,
            headers=self.headers,
            auth_type="bearer",
            auth_credentials=token,
            content_type=self.content_type,
            verify_ssl=self.verify_ssl,
            base_config=self.base_config,
        )

    def with_api_key(self, key: str, header_name: str = "X-API-Key") -> WebhookConfig:
        """Create a copy with API key authentication."""
        return WebhookConfig(
            url=self.url,
            method=self.method,
            headers=self.headers + ((header_name, key),),
            auth_type="api_key",
            auth_credentials=key,
            content_type=self.content_type,
            verify_ssl=self.verify_ssl,
            base_config=self.base_config,
        )

    def with_base_config(self, config: NotificationConfig) -> WebhookConfig:
        """Create a copy with different base config."""
        return WebhookConfig(
            url=self.url,
            method=self.method,
            headers=self.headers,
            auth_type=self.auth_type,
            auth_credentials=self.auth_credentials,
            content_type=self.content_type,
            verify_ssl=self.verify_ssl,
            base_config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes sensitive credentials)."""
        return {
            "url": self.url,
            "method": self.method,
            "headers": dict(self.headers),
            "auth_type": self.auth_type,
            "auth_credentials": "***MASKED***" if self.auth_credentials else None,
            "content_type": self.content_type,
            "verify_ssl": self.verify_ssl,
            "base_config": self.base_config.to_dict(),
        }


# =============================================================================
# Email Configuration
# =============================================================================


class EmailEncryption(str, Enum):
    """Email encryption modes."""

    NONE = "none"
    STARTTLS = "starttls"
    SSL_TLS = "ssl_tls"

    def __str__(self) -> str:
        return self.value


class EmailProvider(str, Enum):
    """Common email providers with preset configurations."""

    SMTP = "smtp"  # Generic SMTP
    SENDGRID = "sendgrid"
    SES = "ses"  # AWS SES
    MAILGUN = "mailgun"
    POSTMARK = "postmark"
    RESEND = "resend"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class EmailConfig:
    """Configuration for email notifications.

    Supports multiple email delivery methods:
        - SMTP: Direct SMTP server connection
        - API: HTTP API for cloud providers (SendGrid, SES, Mailgun, etc.)

    Attributes:
        provider: Email provider type (smtp, sendgrid, ses, etc.)
        from_address: Sender email address
        from_name: Sender display name
        smtp_host: SMTP server hostname
        smtp_port: SMTP server port
        smtp_username: SMTP authentication username
        smtp_password: SMTP authentication password (masked in logs)
        encryption: Encryption mode (none, starttls, ssl_tls)
        api_key: API key for cloud providers
        api_endpoint: Custom API endpoint (optional)
        default_to_addresses: Default recipient addresses
        default_cc_addresses: Default CC addresses
        default_bcc_addresses: Default BCC addresses
        reply_to: Reply-to address
        template_id: Default template ID for template-based emails
        track_opens: Track email opens (API providers only)
        track_clicks: Track link clicks (API providers only)
        sandbox_mode: Enable sandbox mode for testing
        base_config: Base notification configuration

    Example:
        >>> # SMTP configuration
        >>> config = EmailConfig(
        ...     provider=EmailProvider.SMTP,
        ...     from_address="alerts@example.com",
        ...     smtp_host="smtp.example.com",
        ...     smtp_port=587,
        ...     smtp_username="user",
        ...     smtp_password="secret",
        ...     encryption=EmailEncryption.STARTTLS,
        ... )

        >>> # SendGrid configuration
        >>> config = EmailConfig(
        ...     provider=EmailProvider.SENDGRID,
        ...     from_address="alerts@example.com",
        ...     api_key="SG.xxxxx",
        ... )
    """

    # Provider settings
    provider: EmailProvider = EmailProvider.SMTP
    from_address: str = ""
    from_name: str | None = None

    # SMTP settings
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    encryption: EmailEncryption = EmailEncryption.STARTTLS

    # API provider settings
    api_key: str | None = None
    api_endpoint: str | None = None

    # Recipient defaults
    default_to_addresses: tuple[str, ...] = field(default_factory=tuple)
    default_cc_addresses: tuple[str, ...] = field(default_factory=tuple)
    default_bcc_addresses: tuple[str, ...] = field(default_factory=tuple)
    reply_to: str | None = None

    # Template settings
    template_id: str | None = None

    # Tracking settings (API providers)
    track_opens: bool = False
    track_clicks: bool = False

    # Testing
    sandbox_mode: bool = False

    # Base config
    base_config: NotificationConfig = field(default_factory=NotificationConfig)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.from_address:
            raise NotificationConfigError(
                message="from_address is required",
                field_name="from_address",
                channel=NotificationChannel.EMAIL,
            )

        # Validate based on provider
        if self.provider == EmailProvider.SMTP:
            if not self.smtp_host:
                raise NotificationConfigError(
                    message="smtp_host is required for SMTP provider",
                    field_name="smtp_host",
                    channel=NotificationChannel.EMAIL,
                )
            if self.smtp_port < 1 or self.smtp_port > 65535:
                raise NotificationConfigError(
                    message="smtp_port must be between 1 and 65535",
                    field_name="smtp_port",
                    field_value=self.smtp_port,
                    channel=NotificationChannel.EMAIL,
                )
        else:
            # API-based providers require api_key
            if not self.api_key:
                raise NotificationConfigError(
                    message=f"api_key is required for {self.provider.value} provider",
                    field_name="api_key",
                    channel=NotificationChannel.EMAIL,
                )

    def with_from(
        self,
        address: str,
        name: str | None = None,
    ) -> "EmailConfig":
        """Create a copy with a different from address."""
        return EmailConfig(
            provider=self.provider,
            from_address=address,
            from_name=name or self.from_name,
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            smtp_username=self.smtp_username,
            smtp_password=self.smtp_password,
            encryption=self.encryption,
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            default_to_addresses=self.default_to_addresses,
            default_cc_addresses=self.default_cc_addresses,
            default_bcc_addresses=self.default_bcc_addresses,
            reply_to=self.reply_to,
            template_id=self.template_id,
            track_opens=self.track_opens,
            track_clicks=self.track_clicks,
            sandbox_mode=self.sandbox_mode,
            base_config=self.base_config,
        )

    def with_smtp(
        self,
        host: str,
        port: int = 587,
        username: str | None = None,
        password: str | None = None,
        encryption: EmailEncryption = EmailEncryption.STARTTLS,
    ) -> "EmailConfig":
        """Create a copy with SMTP settings."""
        return EmailConfig(
            provider=EmailProvider.SMTP,
            from_address=self.from_address,
            from_name=self.from_name,
            smtp_host=host,
            smtp_port=port,
            smtp_username=username,
            smtp_password=password,
            encryption=encryption,
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            default_to_addresses=self.default_to_addresses,
            default_cc_addresses=self.default_cc_addresses,
            default_bcc_addresses=self.default_bcc_addresses,
            reply_to=self.reply_to,
            template_id=self.template_id,
            track_opens=self.track_opens,
            track_clicks=self.track_clicks,
            sandbox_mode=self.sandbox_mode,
            base_config=self.base_config,
        )

    def with_api(
        self,
        provider: EmailProvider,
        api_key: str,
        api_endpoint: str | None = None,
    ) -> "EmailConfig":
        """Create a copy with API provider settings."""
        return EmailConfig(
            provider=provider,
            from_address=self.from_address,
            from_name=self.from_name,
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            smtp_username=self.smtp_username,
            smtp_password=self.smtp_password,
            encryption=self.encryption,
            api_key=api_key,
            api_endpoint=api_endpoint,
            default_to_addresses=self.default_to_addresses,
            default_cc_addresses=self.default_cc_addresses,
            default_bcc_addresses=self.default_bcc_addresses,
            reply_to=self.reply_to,
            template_id=self.template_id,
            track_opens=self.track_opens,
            track_clicks=self.track_clicks,
            sandbox_mode=self.sandbox_mode,
            base_config=self.base_config,
        )

    def with_default_recipients(
        self,
        to: list[str] | tuple[str, ...] | None = None,
        cc: list[str] | tuple[str, ...] | None = None,
        bcc: list[str] | tuple[str, ...] | None = None,
    ) -> "EmailConfig":
        """Create a copy with default recipients."""
        return EmailConfig(
            provider=self.provider,
            from_address=self.from_address,
            from_name=self.from_name,
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            smtp_username=self.smtp_username,
            smtp_password=self.smtp_password,
            encryption=self.encryption,
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            default_to_addresses=tuple(to) if to else self.default_to_addresses,
            default_cc_addresses=tuple(cc) if cc else self.default_cc_addresses,
            default_bcc_addresses=tuple(bcc) if bcc else self.default_bcc_addresses,
            reply_to=self.reply_to,
            template_id=self.template_id,
            track_opens=self.track_opens,
            track_clicks=self.track_clicks,
            sandbox_mode=self.sandbox_mode,
            base_config=self.base_config,
        )

    def with_tracking(
        self,
        opens: bool = True,
        clicks: bool = True,
    ) -> "EmailConfig":
        """Create a copy with tracking settings."""
        return EmailConfig(
            provider=self.provider,
            from_address=self.from_address,
            from_name=self.from_name,
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            smtp_username=self.smtp_username,
            smtp_password=self.smtp_password,
            encryption=self.encryption,
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            default_to_addresses=self.default_to_addresses,
            default_cc_addresses=self.default_cc_addresses,
            default_bcc_addresses=self.default_bcc_addresses,
            reply_to=self.reply_to,
            template_id=self.template_id,
            track_opens=opens,
            track_clicks=clicks,
            sandbox_mode=self.sandbox_mode,
            base_config=self.base_config,
        )

    def with_sandbox_mode(self, enabled: bool = True) -> "EmailConfig":
        """Create a copy with sandbox mode enabled/disabled."""
        return EmailConfig(
            provider=self.provider,
            from_address=self.from_address,
            from_name=self.from_name,
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            smtp_username=self.smtp_username,
            smtp_password=self.smtp_password,
            encryption=self.encryption,
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            default_to_addresses=self.default_to_addresses,
            default_cc_addresses=self.default_cc_addresses,
            default_bcc_addresses=self.default_bcc_addresses,
            reply_to=self.reply_to,
            template_id=self.template_id,
            track_opens=self.track_opens,
            track_clicks=self.track_clicks,
            sandbox_mode=enabled,
            base_config=self.base_config,
        )

    def with_base_config(self, config: NotificationConfig) -> "EmailConfig":
        """Create a copy with different base config."""
        return EmailConfig(
            provider=self.provider,
            from_address=self.from_address,
            from_name=self.from_name,
            smtp_host=self.smtp_host,
            smtp_port=self.smtp_port,
            smtp_username=self.smtp_username,
            smtp_password=self.smtp_password,
            encryption=self.encryption,
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            default_to_addresses=self.default_to_addresses,
            default_cc_addresses=self.default_cc_addresses,
            default_bcc_addresses=self.default_bcc_addresses,
            reply_to=self.reply_to,
            template_id=self.template_id,
            track_opens=self.track_opens,
            track_clicks=self.track_clicks,
            sandbox_mode=self.sandbox_mode,
            base_config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes sensitive data)."""
        return {
            "provider": self.provider.value,
            "from_address": self.from_address,
            "from_name": self.from_name,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_username": self.smtp_username,
            "smtp_password": "***MASKED***" if self.smtp_password else None,
            "encryption": self.encryption.value,
            "api_key": "***MASKED***" if self.api_key else None,
            "api_endpoint": self.api_endpoint,
            "default_to_addresses": list(self.default_to_addresses),
            "default_cc_addresses": list(self.default_cc_addresses),
            "default_bcc_addresses": list(self.default_bcc_addresses),
            "reply_to": self.reply_to,
            "template_id": self.template_id,
            "track_opens": self.track_opens,
            "track_clicks": self.track_clicks,
            "sandbox_mode": self.sandbox_mode,
            "base_config": self.base_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmailConfig":
        """Create from dictionary."""
        base_config_data = data.get("base_config", {})
        return cls(
            provider=EmailProvider(data.get("provider", "smtp")),
            from_address=data.get("from_address", ""),
            from_name=data.get("from_name"),
            smtp_host=data.get("smtp_host", ""),
            smtp_port=data.get("smtp_port", 587),
            smtp_username=data.get("smtp_username"),
            smtp_password=data.get("smtp_password"),
            encryption=EmailEncryption(data.get("encryption", "starttls")),
            api_key=data.get("api_key"),
            api_endpoint=data.get("api_endpoint"),
            default_to_addresses=tuple(data.get("default_to_addresses", [])),
            default_cc_addresses=tuple(data.get("default_cc_addresses", [])),
            default_bcc_addresses=tuple(data.get("default_bcc_addresses", [])),
            reply_to=data.get("reply_to"),
            template_id=data.get("template_id"),
            track_opens=data.get("track_opens", False),
            track_clicks=data.get("track_clicks", False),
            sandbox_mode=data.get("sandbox_mode", False),
            base_config=NotificationConfig.from_dict(base_config_data),
        )


# =============================================================================
# Preset Configurations
# =============================================================================

DEFAULT_NOTIFICATION_CONFIG = NotificationConfig()
"""Default notification configuration with sensible defaults."""

CRITICAL_NOTIFICATION_CONFIG = NotificationConfig(
    enabled=True,
    timeout_seconds=10.0,
    retry_config=RetryConfig(
        max_retries=5,
        base_delay_seconds=0.5,
        max_delay_seconds=10.0,
    ),
    min_level=NotificationLevel.ERROR,
    async_send=False,
    include_metadata=True,
)
"""Configuration for critical alerts - shorter timeout, more retries."""

LENIENT_NOTIFICATION_CONFIG = NotificationConfig(
    enabled=True,
    timeout_seconds=60.0,
    retry_config=RetryConfig(
        max_retries=1,
        base_delay_seconds=5.0,
        max_delay_seconds=10.0,
    ),
    min_level=NotificationLevel.WARNING,
    async_send=True,
    include_metadata=False,
)
"""Lenient configuration - longer timeout, fewer retries, async."""

FAST_NOTIFICATION_CONFIG = NotificationConfig(
    enabled=True,
    timeout_seconds=5.0,
    retry_config=RetryConfig(
        max_retries=0,
        base_delay_seconds=1.0,
        max_delay_seconds=1.0,
    ),
    min_level=NotificationLevel.INFO,
    async_send=True,
    include_metadata=False,
)
"""Fast fire-and-forget configuration - no retries, short timeout."""

NO_RETRY_CONFIG = RetryConfig(max_retries=0)
"""Retry configuration with no retries."""

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay_seconds=0.5,
    max_delay_seconds=15.0,
    exponential_backoff=True,
    jitter=True,
)
"""Aggressive retry configuration for critical notifications."""


# =============================================================================
# PagerDuty Configuration
# =============================================================================


class PagerDutyRegion(str, Enum):
    """PagerDuty service regions."""

    US = "us"
    EU = "eu"

    def __str__(self) -> str:
        return self.value

    @property
    def events_api_url(self) -> str:
        """Get the Events API v2 URL for this region."""
        if self == PagerDutyRegion.EU:
            return "https://events.eu.pagerduty.com/v2/enqueue"
        return "https://events.pagerduty.com/v2/enqueue"


@dataclass(frozen=True, slots=True)
class PagerDutyConfig:
    """Configuration for PagerDuty notifications.

    Supports PagerDuty Events API v2 for sending alerts/incidents.
    Uses routing keys (integration keys) for authentication.

    Attributes:
        routing_key: PagerDuty integration/routing key (32-char hex string)
        region: PagerDuty region (US or EU)
        default_severity: Default severity when not specified
        default_source: Default source for events
        default_component: Default component for events
        default_group: Default logical grouping
        default_class: Default event class
        client_name: Client name for links
        client_url: Client URL for links
        auto_resolve_timeout_seconds: Auto-resolve timeout (0 = disabled)
        include_context_in_details: Include notification context in custom_details
        dedup_key_prefix: Prefix for generated dedup keys
        base_config: Base notification configuration

    Example:
        >>> config = PagerDutyConfig(
        ...     routing_key="your-32-char-integration-key-here",
        ...     region=PagerDutyRegion.US,
        ...     default_source="data-quality-service",
        ...     default_component="validation",
        ... )
    """

    routing_key: str
    region: PagerDutyRegion = PagerDutyRegion.US
    default_severity: str = "warning"  # critical, error, warning, info
    default_source: str = "truthound-orchestration"
    default_component: str | None = None
    default_group: str | None = None
    default_class: str | None = None
    client_name: str | None = None
    client_url: str | None = None
    auto_resolve_timeout_seconds: int = 0
    include_context_in_details: bool = True
    dedup_key_prefix: str = ""
    base_config: NotificationConfig = field(default_factory=NotificationConfig)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.routing_key:
            raise NotificationConfigError(
                message="routing_key is required",
                field_name="routing_key",
                channel=NotificationChannel.PAGERDUTY,
            )
        if len(self.routing_key) != 32:
            raise NotificationConfigError(
                message="routing_key must be a 32-character hex string",
                field_name="routing_key",
                field_value=f"{self.routing_key[:8]}...",
                channel=NotificationChannel.PAGERDUTY,
            )
        valid_severities = {"critical", "error", "warning", "info"}
        if self.default_severity not in valid_severities:
            raise NotificationConfigError(
                message=f"default_severity must be one of: {valid_severities}",
                field_name="default_severity",
                field_value=self.default_severity,
                channel=NotificationChannel.PAGERDUTY,
            )

    @property
    def api_url(self) -> str:
        """Get the Events API v2 URL."""
        return self.region.events_api_url

    def with_routing_key(self, routing_key: str) -> "PagerDutyConfig":
        """Create a copy with a different routing key."""
        return PagerDutyConfig(
            routing_key=routing_key,
            region=self.region,
            default_severity=self.default_severity,
            default_source=self.default_source,
            default_component=self.default_component,
            default_group=self.default_group,
            default_class=self.default_class,
            client_name=self.client_name,
            client_url=self.client_url,
            auto_resolve_timeout_seconds=self.auto_resolve_timeout_seconds,
            include_context_in_details=self.include_context_in_details,
            dedup_key_prefix=self.dedup_key_prefix,
            base_config=self.base_config,
        )

    def with_region(self, region: PagerDutyRegion) -> "PagerDutyConfig":
        """Create a copy with a different region."""
        return PagerDutyConfig(
            routing_key=self.routing_key,
            region=region,
            default_severity=self.default_severity,
            default_source=self.default_source,
            default_component=self.default_component,
            default_group=self.default_group,
            default_class=self.default_class,
            client_name=self.client_name,
            client_url=self.client_url,
            auto_resolve_timeout_seconds=self.auto_resolve_timeout_seconds,
            include_context_in_details=self.include_context_in_details,
            dedup_key_prefix=self.dedup_key_prefix,
            base_config=self.base_config,
        )

    def with_defaults(
        self,
        *,
        source: str | None = None,
        component: str | None = None,
        group: str | None = None,
        class_type: str | None = None,
        severity: str | None = None,
    ) -> "PagerDutyConfig":
        """Create a copy with default values."""
        return PagerDutyConfig(
            routing_key=self.routing_key,
            region=self.region,
            default_severity=severity or self.default_severity,
            default_source=source or self.default_source,
            default_component=component or self.default_component,
            default_group=group or self.default_group,
            default_class=class_type or self.default_class,
            client_name=self.client_name,
            client_url=self.client_url,
            auto_resolve_timeout_seconds=self.auto_resolve_timeout_seconds,
            include_context_in_details=self.include_context_in_details,
            dedup_key_prefix=self.dedup_key_prefix,
            base_config=self.base_config,
        )

    def with_client(
        self,
        name: str,
        url: str | None = None,
    ) -> "PagerDutyConfig":
        """Create a copy with client information."""
        return PagerDutyConfig(
            routing_key=self.routing_key,
            region=self.region,
            default_severity=self.default_severity,
            default_source=self.default_source,
            default_component=self.default_component,
            default_group=self.default_group,
            default_class=self.default_class,
            client_name=name,
            client_url=url,
            auto_resolve_timeout_seconds=self.auto_resolve_timeout_seconds,
            include_context_in_details=self.include_context_in_details,
            dedup_key_prefix=self.dedup_key_prefix,
            base_config=self.base_config,
        )

    def with_dedup_prefix(self, prefix: str) -> "PagerDutyConfig":
        """Create a copy with a dedup key prefix."""
        return PagerDutyConfig(
            routing_key=self.routing_key,
            region=self.region,
            default_severity=self.default_severity,
            default_source=self.default_source,
            default_component=self.default_component,
            default_group=self.default_group,
            default_class=self.default_class,
            client_name=self.client_name,
            client_url=self.client_url,
            auto_resolve_timeout_seconds=self.auto_resolve_timeout_seconds,
            include_context_in_details=self.include_context_in_details,
            dedup_key_prefix=prefix,
            base_config=self.base_config,
        )

    def with_base_config(self, config: NotificationConfig) -> "PagerDutyConfig":
        """Create a copy with different base config."""
        return PagerDutyConfig(
            routing_key=self.routing_key,
            region=self.region,
            default_severity=self.default_severity,
            default_source=self.default_source,
            default_component=self.default_component,
            default_group=self.default_group,
            default_class=self.default_class,
            client_name=self.client_name,
            client_url=self.client_url,
            auto_resolve_timeout_seconds=self.auto_resolve_timeout_seconds,
            include_context_in_details=self.include_context_in_details,
            dedup_key_prefix=self.dedup_key_prefix,
            base_config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes sensitive routing_key)."""
        return {
            "routing_key": "***MASKED***",
            "region": self.region.value,
            "default_severity": self.default_severity,
            "default_source": self.default_source,
            "default_component": self.default_component,
            "default_group": self.default_group,
            "default_class": self.default_class,
            "client_name": self.client_name,
            "client_url": self.client_url,
            "auto_resolve_timeout_seconds": self.auto_resolve_timeout_seconds,
            "include_context_in_details": self.include_context_in_details,
            "dedup_key_prefix": self.dedup_key_prefix,
            "base_config": self.base_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PagerDutyConfig":
        """Create from dictionary."""
        base_config_data = data.get("base_config", {})
        return cls(
            routing_key=data.get("routing_key", ""),
            region=PagerDutyRegion(data.get("region", "us")),
            default_severity=data.get("default_severity", "warning"),
            default_source=data.get("default_source", "truthound-orchestration"),
            default_component=data.get("default_component"),
            default_group=data.get("default_group"),
            default_class=data.get("default_class"),
            client_name=data.get("client_name"),
            client_url=data.get("client_url"),
            auto_resolve_timeout_seconds=data.get("auto_resolve_timeout_seconds", 0),
            include_context_in_details=data.get("include_context_in_details", True),
            dedup_key_prefix=data.get("dedup_key_prefix", ""),
            base_config=NotificationConfig.from_dict(base_config_data),
        )


# =============================================================================
# Opsgenie Configuration
# =============================================================================


class OpsgenieRegion(str, Enum):
    """Opsgenie service regions."""

    US = "us"
    EU = "eu"

    def __str__(self) -> str:
        return self.value

    @property
    def api_url(self) -> str:
        """Get the Alert API URL for this region."""
        if self == OpsgenieRegion.EU:
            return "https://api.eu.opsgenie.com/v2/alerts"
        return "https://api.opsgenie.com/v2/alerts"


class OpsgeniePriority(str, Enum):
    """Opsgenie alert priority levels."""

    P1 = "P1"  # Critical
    P2 = "P2"  # High
    P3 = "P3"  # Moderate
    P4 = "P4"  # Low
    P5 = "P5"  # Informational

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class OpsgenieResponder:
    """Opsgenie responder definition.

    Attributes:
        id: Responder ID (team or user ID)
        type: Responder type (team, user, escalation, schedule)
        name: Optional responder name (alternative to ID)
    """

    id: str | None = None
    type: str = "team"  # team, user, escalation, schedule
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.id and not self.name:
            raise NotificationConfigError(
                message="Either id or name must be provided for responder",
                field_name="responder",
            )
        if self.type not in {"team", "user", "escalation", "schedule"}:
            raise NotificationConfigError(
                message="Responder type must be one of: team, user, escalation, schedule",
                field_name="type",
                field_value=self.type,
            )

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for API."""
        result: dict[str, str] = {"type": self.type}
        if self.id:
            result["id"] = self.id
        if self.name:
            result["name"] = self.name
        return result


@dataclass(frozen=True, slots=True)
class OpsgenieConfig:
    """Configuration for Opsgenie notifications.

    Supports Opsgenie Alert API v2 for sending alerts.
    Uses API keys for authentication.

    Attributes:
        api_key: Opsgenie API key (GenieKey)
        region: Opsgenie region (US or EU)
        default_priority: Default alert priority (P1-P5)
        default_source: Default source for alerts
        default_tags: Default tags for alerts
        default_responders: Default responders (teams, users, escalations)
        default_visible_to: Default visible to teams/users
        default_actions: Default custom actions
        default_details: Default custom details
        default_entity: Default entity for alerts
        default_alias_prefix: Prefix for alert aliases
        include_context_in_details: Include notification context in details
        base_config: Base notification configuration

    Example:
        >>> config = OpsgenieConfig(
        ...     api_key="your-opsgenie-api-key",
        ...     region=OpsgenieRegion.US,
        ...     default_priority=OpsgeniePriority.P2,
        ...     default_responders=(
        ...         OpsgenieResponder(name="ops-team", type="team"),
        ...     ),
        ... )
    """

    api_key: str
    region: OpsgenieRegion = OpsgenieRegion.US
    default_priority: OpsgeniePriority = OpsgeniePriority.P3
    default_source: str = "truthound-orchestration"
    default_tags: tuple[str, ...] = field(default_factory=tuple)
    default_responders: tuple[OpsgenieResponder, ...] = field(default_factory=tuple)
    default_visible_to: tuple[OpsgenieResponder, ...] = field(default_factory=tuple)
    default_actions: tuple[str, ...] = field(default_factory=tuple)
    default_details: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    default_entity: str | None = None
    default_alias_prefix: str = ""
    include_context_in_details: bool = True
    base_config: NotificationConfig = field(default_factory=NotificationConfig)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.api_key:
            raise NotificationConfigError(
                message="api_key is required",
                field_name="api_key",
                channel=NotificationChannel.OPSGENIE,
            )

    @property
    def api_url(self) -> str:
        """Get the Alert API URL."""
        return self.region.api_url

    @property
    def default_details_dict(self) -> dict[str, str]:
        """Get default details as a dictionary."""
        return dict(self.default_details)

    def with_api_key(self, api_key: str) -> "OpsgenieConfig":
        """Create a copy with a different API key."""
        return OpsgenieConfig(
            api_key=api_key,
            region=self.region,
            default_priority=self.default_priority,
            default_source=self.default_source,
            default_tags=self.default_tags,
            default_responders=self.default_responders,
            default_visible_to=self.default_visible_to,
            default_actions=self.default_actions,
            default_details=self.default_details,
            default_entity=self.default_entity,
            default_alias_prefix=self.default_alias_prefix,
            include_context_in_details=self.include_context_in_details,
            base_config=self.base_config,
        )

    def with_region(self, region: OpsgenieRegion) -> "OpsgenieConfig":
        """Create a copy with a different region."""
        return OpsgenieConfig(
            api_key=self.api_key,
            region=region,
            default_priority=self.default_priority,
            default_source=self.default_source,
            default_tags=self.default_tags,
            default_responders=self.default_responders,
            default_visible_to=self.default_visible_to,
            default_actions=self.default_actions,
            default_details=self.default_details,
            default_entity=self.default_entity,
            default_alias_prefix=self.default_alias_prefix,
            include_context_in_details=self.include_context_in_details,
            base_config=self.base_config,
        )

    def with_priority(self, priority: OpsgeniePriority) -> "OpsgenieConfig":
        """Create a copy with a different default priority."""
        return OpsgenieConfig(
            api_key=self.api_key,
            region=self.region,
            default_priority=priority,
            default_source=self.default_source,
            default_tags=self.default_tags,
            default_responders=self.default_responders,
            default_visible_to=self.default_visible_to,
            default_actions=self.default_actions,
            default_details=self.default_details,
            default_entity=self.default_entity,
            default_alias_prefix=self.default_alias_prefix,
            include_context_in_details=self.include_context_in_details,
            base_config=self.base_config,
        )

    def with_defaults(
        self,
        *,
        source: str | None = None,
        entity: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
    ) -> "OpsgenieConfig":
        """Create a copy with default values."""
        return OpsgenieConfig(
            api_key=self.api_key,
            region=self.region,
            default_priority=self.default_priority,
            default_source=source or self.default_source,
            default_tags=tuple(tags) if tags else self.default_tags,
            default_responders=self.default_responders,
            default_visible_to=self.default_visible_to,
            default_actions=self.default_actions,
            default_details=self.default_details,
            default_entity=entity or self.default_entity,
            default_alias_prefix=self.default_alias_prefix,
            include_context_in_details=self.include_context_in_details,
            base_config=self.base_config,
        )

    def with_responders(self, *responders: OpsgenieResponder) -> "OpsgenieConfig":
        """Create a copy with default responders."""
        return OpsgenieConfig(
            api_key=self.api_key,
            region=self.region,
            default_priority=self.default_priority,
            default_source=self.default_source,
            default_tags=self.default_tags,
            default_responders=self.default_responders + responders,
            default_visible_to=self.default_visible_to,
            default_actions=self.default_actions,
            default_details=self.default_details,
            default_entity=self.default_entity,
            default_alias_prefix=self.default_alias_prefix,
            include_context_in_details=self.include_context_in_details,
            base_config=self.base_config,
        )

    def with_team(self, team_name: str) -> "OpsgenieConfig":
        """Create a copy with a default team responder."""
        return self.with_responders(OpsgenieResponder(name=team_name, type="team"))

    def with_alias_prefix(self, prefix: str) -> "OpsgenieConfig":
        """Create a copy with an alias prefix."""
        return OpsgenieConfig(
            api_key=self.api_key,
            region=self.region,
            default_priority=self.default_priority,
            default_source=self.default_source,
            default_tags=self.default_tags,
            default_responders=self.default_responders,
            default_visible_to=self.default_visible_to,
            default_actions=self.default_actions,
            default_details=self.default_details,
            default_entity=self.default_entity,
            default_alias_prefix=prefix,
            include_context_in_details=self.include_context_in_details,
            base_config=self.base_config,
        )

    def with_base_config(self, config: NotificationConfig) -> "OpsgenieConfig":
        """Create a copy with different base config."""
        return OpsgenieConfig(
            api_key=self.api_key,
            region=self.region,
            default_priority=self.default_priority,
            default_source=self.default_source,
            default_tags=self.default_tags,
            default_responders=self.default_responders,
            default_visible_to=self.default_visible_to,
            default_actions=self.default_actions,
            default_details=self.default_details,
            default_entity=self.default_entity,
            default_alias_prefix=self.default_alias_prefix,
            include_context_in_details=self.include_context_in_details,
            base_config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes sensitive api_key)."""
        return {
            "api_key": "***MASKED***",
            "region": self.region.value,
            "default_priority": self.default_priority.value,
            "default_source": self.default_source,
            "default_tags": list(self.default_tags),
            "default_responders": [r.to_dict() for r in self.default_responders],
            "default_visible_to": [r.to_dict() for r in self.default_visible_to],
            "default_actions": list(self.default_actions),
            "default_details": dict(self.default_details),
            "default_entity": self.default_entity,
            "default_alias_prefix": self.default_alias_prefix,
            "include_context_in_details": self.include_context_in_details,
            "base_config": self.base_config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpsgenieConfig":
        """Create from dictionary."""
        base_config_data = data.get("base_config", {})
        responders_data = data.get("default_responders", [])
        visible_to_data = data.get("default_visible_to", [])

        return cls(
            api_key=data.get("api_key", ""),
            region=OpsgenieRegion(data.get("region", "us")),
            default_priority=OpsgeniePriority(data.get("default_priority", "P3")),
            default_source=data.get("default_source", "truthound-orchestration"),
            default_tags=tuple(data.get("default_tags", [])),
            default_responders=tuple(
                OpsgenieResponder(
                    id=r.get("id"),
                    type=r.get("type", "team"),
                    name=r.get("name"),
                )
                for r in responders_data
            ),
            default_visible_to=tuple(
                OpsgenieResponder(
                    id=r.get("id"),
                    type=r.get("type", "team"),
                    name=r.get("name"),
                )
                for r in visible_to_data
            ),
            default_actions=tuple(data.get("default_actions", [])),
            default_details=tuple(data.get("default_details", {}).items()),
            default_entity=data.get("default_entity"),
            default_alias_prefix=data.get("default_alias_prefix", ""),
            include_context_in_details=data.get("include_context_in_details", True),
            base_config=NotificationConfig.from_dict(base_config_data),
        )
