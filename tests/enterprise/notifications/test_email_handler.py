"""Tests for Email notification handler.

This module provides comprehensive tests for the EmailNotificationHandler,
including configuration, SMTP, and API provider tests.
"""

from __future__ import annotations

import smtplib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.enterprise.notifications.config import (
    EmailConfig,
    EmailEncryption,
    EmailProvider,
    NotificationConfig,
)
from packages.enterprise.notifications.exceptions import (
    NotificationConfigError,
    NotificationSendError,
    NotificationTimeoutError,
)
from packages.enterprise.notifications.factory import (
    create_email_handler,
    create_handler_from_config,
    create_sendgrid_email_handler,
    create_ses_email_handler,
    create_smtp_email_handler,
)
from packages.enterprise.notifications.handlers.email import (
    EmailNotificationHandler,
    LEVEL_COLORS,
    LEVEL_EMOJIS,
    LEVEL_ICONS,
)
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def smtp_config() -> EmailConfig:
    """Create a basic SMTP email configuration."""
    return EmailConfig(
        provider=EmailProvider.SMTP,
        from_address="test@example.com",
        from_name="Test Sender",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user",
        smtp_password="pass",
        encryption=EmailEncryption.STARTTLS,
        default_recipients=("admin@example.com",),
    )


@pytest.fixture
def sendgrid_config() -> EmailConfig:
    """Create a SendGrid email configuration."""
    return EmailConfig(
        provider=EmailProvider.SENDGRID,
        from_address="test@example.com",
        from_name="Test Sender",
        api_key="SG.test-api-key",
        default_recipients=("admin@example.com",),
    )


@pytest.fixture
def ses_config() -> EmailConfig:
    """Create an AWS SES email configuration."""
    return EmailConfig(
        provider=EmailProvider.SES,
        from_address="test@example.com",
        from_name="Test Sender",
        api_key="AKIAIOSFODNN7EXAMPLE",
        aws_secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        aws_region="us-east-1",
        default_recipients=("admin@example.com",),
    )


@pytest.fixture
def basic_payload() -> NotificationPayload:
    """Create a basic notification payload."""
    return NotificationPayload(
        message="Test notification message",
        level=NotificationLevel.INFO,
        title="Test Title",
        context=(("check_id", "chk_123"), ("rows", "1000")),
    )


@pytest.fixture
def error_payload() -> NotificationPayload:
    """Create an error notification payload."""
    return NotificationPayload(
        message="Data quality check failed!",
        level=NotificationLevel.ERROR,
        title="Quality Check Failed",
        context=(
            ("check_id", "chk_456"),
            ("failed_count", "5"),
            ("failure_rate", "2.5%"),
        ),
    )


@pytest.fixture
def smtp_handler(smtp_config: EmailConfig) -> EmailNotificationHandler:
    """Create an SMTP email handler."""
    return EmailNotificationHandler(config=smtp_config)


@pytest.fixture
def sendgrid_handler(sendgrid_config: EmailConfig) -> EmailNotificationHandler:
    """Create a SendGrid email handler."""
    return EmailNotificationHandler(config=sendgrid_config)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestEmailConfig:
    """Tests for EmailConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EmailConfig(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
        )
        assert config.provider == EmailProvider.SMTP
        assert config.smtp_port == 587
        assert config.encryption == EmailEncryption.STARTTLS
        assert config.default_recipients == ()
        assert config.subject_prefix == ""

    def test_smtp_config(self, smtp_config: EmailConfig) -> None:
        """Test SMTP configuration."""
        assert smtp_config.provider == EmailProvider.SMTP
        assert smtp_config.from_address == "test@example.com"
        assert smtp_config.smtp_host == "smtp.example.com"
        assert smtp_config.smtp_port == 587
        assert smtp_config.encryption == EmailEncryption.STARTTLS

    def test_sendgrid_config(self, sendgrid_config: EmailConfig) -> None:
        """Test SendGrid configuration."""
        assert sendgrid_config.provider == EmailProvider.SENDGRID
        assert sendgrid_config.api_key == "SG.test-api-key"

    def test_builder_pattern(self) -> None:
        """Test builder pattern methods."""
        config = EmailConfig(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
        )
        config = config.with_smtp(
            host="new-smtp.example.com",
            port=465,
            username="newuser",
            password="newpass",
        )
        assert config.smtp_host == "new-smtp.example.com"
        assert config.smtp_port == 465
        assert config.smtp_username == "newuser"

    def test_to_dict_masks_sensitive_data(self, smtp_config: EmailConfig) -> None:
        """Test that to_dict masks sensitive data."""
        config_dict = smtp_config.to_dict()
        assert config_dict["smtp_password"] == "***MASKED***"
        assert "pass" not in str(config_dict)

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config_dict = {
            "provider": "smtp",
            "from_address": "test@example.com",
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
        }
        config = EmailConfig.from_dict(config_dict)
        assert config.provider == EmailProvider.SMTP
        assert config.from_address == "test@example.com"

    def test_encryption_modes(self) -> None:
        """Test different encryption modes."""
        for encryption in EmailEncryption:
            config = EmailConfig(
                from_address="test@example.com",
                smtp_host="smtp.example.com",
                encryption=encryption,
            )
            assert config.encryption == encryption


# =============================================================================
# Handler Initialization Tests
# =============================================================================


class TestEmailHandlerInitialization:
    """Tests for EmailNotificationHandler initialization."""

    def test_init_with_config(self, smtp_config: EmailConfig) -> None:
        """Test initialization with full config."""
        handler = EmailNotificationHandler(config=smtp_config)
        assert handler.channel == NotificationChannel.EMAIL
        assert handler.email_config == smtp_config

    def test_init_with_from_address(self) -> None:
        """Test initialization with minimal parameters."""
        handler = EmailNotificationHandler(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
        )
        assert handler.email_config.from_address == "test@example.com"
        assert handler.email_config.provider == EmailProvider.SMTP

    def test_init_requires_config_or_from_address(self) -> None:
        """Test that initialization fails without config or from_address."""
        with pytest.raises(ValueError, match="Either config or from_address"):
            EmailNotificationHandler()

    def test_smtp_requires_host(self) -> None:
        """Test that SMTP provider requires smtp_host."""
        with pytest.raises(NotificationConfigError, match="smtp_host is required"):
            EmailNotificationHandler(
                config=EmailConfig(
                    provider=EmailProvider.SMTP,
                    from_address="test@example.com",
                )
            )

    def test_api_provider_requires_api_key(self) -> None:
        """Test that API providers require api_key."""
        with pytest.raises(NotificationConfigError, match="api_key is required"):
            EmailNotificationHandler(
                config=EmailConfig(
                    provider=EmailProvider.SENDGRID,
                    from_address="test@example.com",
                )
            )

    def test_handler_name(self) -> None:
        """Test handler name configuration."""
        handler = EmailNotificationHandler(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
            name="custom_email",
        )
        assert handler.name == "custom_email"


# =============================================================================
# Message Formatting Tests
# =============================================================================


class TestEmailMessageFormatting:
    """Tests for email message formatting."""

    @pytest.mark.asyncio
    async def test_format_message_structure(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test formatted message structure."""
        formatted = await smtp_handler._format_message(basic_payload)
        assert "subject" in formatted
        assert "html" in formatted
        assert "text" in formatted
        assert "recipients" in formatted

    @pytest.mark.asyncio
    async def test_format_message_html_contains_title(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that HTML body contains the title."""
        formatted = await smtp_handler._format_message(basic_payload)
        assert "Test Title" in formatted["html"]

    @pytest.mark.asyncio
    async def test_format_message_html_contains_message(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that HTML body contains the message."""
        formatted = await smtp_handler._format_message(basic_payload)
        assert "Test notification message" in formatted["html"]

    @pytest.mark.asyncio
    async def test_format_message_contains_context(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that formatted message contains context."""
        formatted = await smtp_handler._format_message(basic_payload)
        assert "check_id" in formatted["html"]
        assert "chk_123" in formatted["html"]

    @pytest.mark.asyncio
    async def test_format_message_level_colors(
        self,
        smtp_handler: EmailNotificationHandler,
    ) -> None:
        """Test that each level has appropriate color in HTML."""
        for level in NotificationLevel:
            payload = NotificationPayload(
                message="Test",
                level=level,
            )
            formatted = await smtp_handler._format_message(payload)
            color = LEVEL_COLORS.get(level)
            if color:
                assert color in formatted["html"]

    @pytest.mark.asyncio
    async def test_format_message_subject_prefix(self) -> None:
        """Test subject prefix in formatted message."""
        config = EmailConfig(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
            subject_prefix="[DQ Alert] ",
        )
        handler = EmailNotificationHandler(config=config)
        payload = NotificationPayload(
            message="Test",
            level=NotificationLevel.WARNING,
            title="Test Alert",
        )
        formatted = await handler._format_message(payload)
        assert formatted["subject"].startswith("[DQ Alert] ")

    def test_simple_message_format(
        self,
        smtp_handler: EmailNotificationHandler,
    ) -> None:
        """Test simple message formatting."""
        simple = smtp_handler.format_simple_message(
            message="Simple test",
            subject="Simple Subject",
            level=NotificationLevel.INFO,
        )
        assert simple["text"] == "Simple test"
        assert "Simple Subject" in simple["subject"]


# =============================================================================
# Recipient Handling Tests
# =============================================================================


class TestEmailRecipients:
    """Tests for email recipient handling."""

    def test_get_recipients_from_config(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test recipients from default config."""
        recipients = smtp_handler._get_recipients(basic_payload)
        assert "admin@example.com" in recipients

    def test_get_recipients_from_context(
        self,
        smtp_handler: EmailNotificationHandler,
    ) -> None:
        """Test recipients from payload context."""
        payload = NotificationPayload(
            message="Test",
            level=NotificationLevel.INFO,
            context=(("recipients", "user1@example.com,user2@example.com"),),
        )
        recipients = smtp_handler._get_recipients(payload)
        assert "user1@example.com" in recipients
        assert "user2@example.com" in recipients

    def test_get_recipients_deduplication(self) -> None:
        """Test that duplicate recipients are removed."""
        config = EmailConfig(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
            default_recipients=("admin@example.com", "Admin@Example.com"),
        )
        handler = EmailNotificationHandler(config=config)
        payload = NotificationPayload(
            message="Test",
            level=NotificationLevel.INFO,
        )
        recipients = handler._get_recipients(payload)
        # Should only have one admin@example.com (case-insensitive dedup)
        assert len([r for r in recipients if r.lower() == "admin@example.com"]) == 1


# =============================================================================
# SMTP Send Tests
# =============================================================================


class TestEmailSMTPSend:
    """Tests for SMTP send functionality."""

    @pytest.mark.asyncio
    async def test_send_success(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test successful SMTP send."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            result = await smtp_handler.send(basic_payload)

            assert result.success
            assert result.channel == NotificationChannel.EMAIL

    @pytest.mark.asyncio
    async def test_send_with_starttls(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test SMTP send with STARTTLS."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            await smtp_handler.send(basic_payload)

            mock_server.starttls.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_with_ssl(
        self,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test SMTP send with SSL/TLS."""
        config = EmailConfig(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
            smtp_port=465,
            encryption=EmailEncryption.SSL_TLS,
            default_recipients=("admin@example.com",),
        )
        handler = EmailNotificationHandler(config=config)

        with patch("smtplib.SMTP_SSL") as mock_smtp_ssl:
            mock_server = MagicMock()
            mock_smtp_ssl.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp_ssl.return_value.__exit__ = MagicMock(return_value=False)

            result = await handler.send(basic_payload)

            assert result.success
            mock_smtp_ssl.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_authentication(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test SMTP authentication."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            await smtp_handler.send(basic_payload)

            mock_server.login.assert_called_once_with("user", "pass")

    @pytest.mark.asyncio
    async def test_send_no_recipients_fails(
        self,
        smtp_handler: EmailNotificationHandler,
    ) -> None:
        """Test that send fails without recipients."""
        # Create a handler with no default recipients
        config = EmailConfig(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
            default_recipients=(),
        )
        handler = EmailNotificationHandler(config=config)
        payload = NotificationPayload(
            message="Test",
            level=NotificationLevel.INFO,
        )

        result = await handler.send(payload)
        assert not result.success
        assert "No recipients" in result.error

    @pytest.mark.asyncio
    async def test_send_smtp_auth_error(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test handling of SMTP authentication error."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_server.login.side_effect = smtplib.SMTPAuthenticationError(
                535, b"Authentication failed"
            )
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            result = await smtp_handler.send(basic_payload)

            assert not result.success
            assert "authentication" in result.error.lower()


# =============================================================================
# SendGrid API Tests
# =============================================================================


class TestEmailSendGridSend:
    """Tests for SendGrid API send functionality."""

    @pytest.mark.asyncio
    async def test_sendgrid_send_success(
        self,
        sendgrid_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test successful SendGrid send."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 202
            mock_response.text = AsyncMock(return_value="")

            mock_post = AsyncMock(return_value=mock_response)
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_client = MagicMock()
            mock_client.post = MagicMock(return_value=mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_client

            result = await sendgrid_handler.send(basic_payload)

            assert result.success
            assert result.channel == NotificationChannel.EMAIL


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestEmailFactoryFunctions:
    """Tests for email factory functions."""

    def test_create_email_handler_smtp(self) -> None:
        """Test create_email_handler for SMTP."""
        handler = create_email_handler(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
        )
        assert handler.email_config.provider == EmailProvider.SMTP
        assert handler.email_config.from_address == "test@example.com"

    def test_create_email_handler_sendgrid(self) -> None:
        """Test create_email_handler for SendGrid."""
        handler = create_email_handler(
            from_address="test@example.com",
            provider=EmailProvider.SENDGRID,
            api_key="SG.test-key",
        )
        assert handler.email_config.provider == EmailProvider.SENDGRID
        assert handler.email_config.api_key == "SG.test-key"

    def test_create_smtp_email_handler(self) -> None:
        """Test create_smtp_email_handler."""
        handler = create_smtp_email_handler(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_username="user",
            smtp_password="pass",
        )
        assert handler.email_config.provider == EmailProvider.SMTP
        assert handler.email_config.smtp_host == "smtp.example.com"

    def test_create_sendgrid_email_handler(self) -> None:
        """Test create_sendgrid_email_handler."""
        handler = create_sendgrid_email_handler(
            from_address="test@example.com",
            api_key="SG.test-key",
        )
        assert handler.email_config.provider == EmailProvider.SENDGRID

    def test_create_handler_from_config_email(self) -> None:
        """Test create_handler_from_config for email."""
        config = {
            "from_address": "test@example.com",
            "provider": "smtp",
            "smtp_host": "smtp.example.com",
        }
        handler = create_handler_from_config("email", config)
        assert isinstance(handler, EmailNotificationHandler)
        assert handler.email_config.provider == EmailProvider.SMTP

    def test_create_handler_from_config_email_sendgrid(self) -> None:
        """Test create_handler_from_config for email with SendGrid."""
        config = {
            "from_address": "test@example.com",
            "provider": "sendgrid",
            "api_key": "SG.test-key",
        }
        handler = create_handler_from_config("email", config)
        assert isinstance(handler, EmailNotificationHandler)
        assert handler.email_config.provider == EmailProvider.SENDGRID

    def test_create_handler_from_config_email_missing_from_address(self) -> None:
        """Test that create_handler_from_config fails without from_address."""
        config = {
            "provider": "smtp",
            "smtp_host": "smtp.example.com",
        }
        with pytest.raises(NotificationConfigError, match="from_address is required"):
            create_handler_from_config("email", config)


# =============================================================================
# Level Mappings Tests
# =============================================================================


class TestLevelMappings:
    """Tests for level color, emoji, and icon mappings."""

    def test_all_levels_have_colors(self) -> None:
        """Test that all notification levels have colors."""
        for level in NotificationLevel:
            assert level in LEVEL_COLORS
            assert LEVEL_COLORS[level].startswith("#")

    def test_all_levels_have_emojis(self) -> None:
        """Test that all notification levels have emojis."""
        for level in NotificationLevel:
            assert level in LEVEL_EMOJIS
            assert len(LEVEL_EMOJIS[level]) > 0

    def test_all_levels_have_icons(self) -> None:
        """Test that all notification levels have icons."""
        for level in NotificationLevel:
            assert level in LEVEL_ICONS
            assert len(LEVEL_ICONS[level]) > 0


# =============================================================================
# Handler State Tests
# =============================================================================


class TestEmailHandlerState:
    """Tests for handler state management."""

    def test_handler_enabled_by_default(
        self,
        smtp_handler: EmailNotificationHandler,
    ) -> None:
        """Test that handler is enabled by default."""
        assert smtp_handler.enabled

    def test_handler_disable_enable(
        self,
        smtp_handler: EmailNotificationHandler,
    ) -> None:
        """Test disabling and enabling handler."""
        smtp_handler.disable()
        assert not smtp_handler.enabled

        smtp_handler.enable()
        assert smtp_handler.enabled

    @pytest.mark.asyncio
    async def test_disabled_handler_skips_send(
        self,
        smtp_handler: EmailNotificationHandler,
        basic_payload: NotificationPayload,
    ) -> None:
        """Test that disabled handler skips sending."""
        smtp_handler.disable()
        result = await smtp_handler.send(basic_payload)
        assert not result.success
        assert "Filtered" in result.error or "skipped" in str(result.status).lower()


# =============================================================================
# From Address Formatting Tests
# =============================================================================


class TestFromAddressFormatting:
    """Tests for from address formatting."""

    def test_format_from_address_with_name(
        self,
        smtp_handler: EmailNotificationHandler,
    ) -> None:
        """Test from address formatting with display name."""
        formatted = smtp_handler._format_from_address()
        assert "Test Sender" in formatted
        assert "test@example.com" in formatted

    def test_format_from_address_without_name(self) -> None:
        """Test from address formatting without display name."""
        config = EmailConfig(
            from_address="test@example.com",
            smtp_host="smtp.example.com",
        )
        handler = EmailNotificationHandler(config=config)
        formatted = handler._format_from_address()
        assert formatted == "test@example.com"
