"""Email notification handler.

This module provides the Email notification handler that sends notifications
via SMTP or various email API providers (SendGrid, AWS SES, Mailgun, etc.).

Features:
    - SMTP support with SSL/TLS and STARTTLS
    - API provider support (SendGrid, SES, Mailgun, Postmark, Resend)
    - HTML and plain text email formatting
    - Template support with variable substitution
    - CC/BCC recipient support
    - Attachment support (planned)
    - Reply-to configuration
    - Retry with exponential backoff
    - Connection pooling for SMTP

Example:
    >>> from packages.enterprise.notifications.handlers import EmailNotificationHandler
    >>> from packages.enterprise.notifications.config import EmailConfig, EmailProvider
    >>>
    >>> # SMTP configuration
    >>> handler = EmailNotificationHandler(
    ...     config=EmailConfig(
    ...         provider=EmailProvider.SMTP,
    ...         from_address="alerts@example.com",
    ...         smtp_host="smtp.example.com",
    ...         smtp_port=587,
    ...         smtp_username="user",
    ...         smtp_password="pass",
    ...     ),
    ... )
    >>>
    >>> # Or with SendGrid
    >>> handler = EmailNotificationHandler(
    ...     config=EmailConfig(
    ...         provider=EmailProvider.SENDGRID,
    ...         from_address="alerts@example.com",
    ...         api_key="SG.xxxx",
    ...     ),
    ... )
    >>>
    >>> result = await handler.send(NotificationPayload(
    ...     message="Data quality check failed!",
    ...     level=NotificationLevel.ERROR,
    ...     recipients=("admin@example.com",),
    ... ))
"""

from __future__ import annotations

import html
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

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
from packages.enterprise.notifications.handlers.base import AsyncNotificationHandler
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
)

if TYPE_CHECKING:
    from packages.enterprise.notifications.hooks import NotificationHook


# Level to color mapping for HTML emails
LEVEL_COLORS: dict[NotificationLevel, str] = {
    NotificationLevel.DEBUG: "#6c757d",    # Gray
    NotificationLevel.INFO: "#28a745",     # Green
    NotificationLevel.WARNING: "#ffc107",  # Yellow/Amber
    NotificationLevel.ERROR: "#dc3545",    # Red
    NotificationLevel.CRITICAL: "#6f42c1", # Purple
}

# Level to emoji mapping for plain text fallback
LEVEL_EMOJIS: dict[NotificationLevel, str] = {
    NotificationLevel.DEBUG: "[DEBUG]",
    NotificationLevel.INFO: "[INFO]",
    NotificationLevel.WARNING: "[WARNING]",
    NotificationLevel.ERROR: "[ERROR]",
    NotificationLevel.CRITICAL: "[CRITICAL]",
}

# Level to icon mapping for HTML emails
LEVEL_ICONS: dict[NotificationLevel, str] = {
    NotificationLevel.DEBUG: "🔍",
    NotificationLevel.INFO: "ℹ️",
    NotificationLevel.WARNING: "⚠️",
    NotificationLevel.ERROR: "❌",
    NotificationLevel.CRITICAL: "🚨",
}


class EmailNotificationHandler(AsyncNotificationHandler):
    """Email notification handler supporting SMTP and API providers.

    Sends notifications via email with support for multiple providers,
    HTML formatting, templates, and rich content.

    Attributes:
        email_config: Email-specific configuration
    """

    def __init__(
        self,
        config: EmailConfig | None = None,
        *,
        from_address: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        api_key: str | None = None,
        provider: EmailProvider | None = None,
        name: str | None = None,
        base_config: NotificationConfig | None = None,
        hooks: list[NotificationHook] | None = None,
    ) -> None:
        """Initialize the Email handler.

        Args:
            config: Complete Email configuration
            from_address: Sender email address (alternative to config)
            smtp_host: SMTP server host (alternative to config)
            smtp_port: SMTP server port (alternative to config)
            api_key: API key for provider (alternative to config)
            provider: Email provider (alternative to config)
            name: Handler name
            base_config: Base notification config (used if config not provided)
            hooks: Notification hooks
        """
        if config is not None:
            self._email_config = config
            base = config.base_config
        elif from_address is not None:
            self._email_config = EmailConfig(
                provider=provider or EmailProvider.SMTP,
                from_address=from_address,
                smtp_host=smtp_host or "",
                smtp_port=smtp_port or 587,
                api_key=api_key,
                base_config=base_config or NotificationConfig(),
            )
            base = self._email_config.base_config
        else:
            raise ValueError("Either config or from_address must be provided")

        super().__init__(
            name=name or "email",
            config=base,
            hooks=hooks,
        )

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the email configuration."""
        if not self._email_config.from_address:
            raise NotificationConfigError(
                message="from_address is required for email notifications",
                config_field="from_address",
            )

        if self._email_config.provider == EmailProvider.SMTP:
            if not self._email_config.smtp_host:
                raise NotificationConfigError(
                    message="smtp_host is required for SMTP provider",
                    config_field="smtp_host",
                )
        else:
            # API providers require api_key
            if not self._email_config.api_key:
                raise NotificationConfigError(
                    message=f"api_key is required for {self._email_config.provider.value} provider",
                    config_field="api_key",
                )

    @property
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        return NotificationChannel.EMAIL

    @property
    def email_config(self) -> EmailConfig:
        """Get the Email-specific configuration."""
        return self._email_config

    async def _format_message(
        self,
        payload: NotificationPayload,
    ) -> dict[str, Any]:
        """Format the message for email.

        Creates both HTML and plain text versions of the email.

        Args:
            payload: The notification payload

        Returns:
            Email message data as dictionary with 'html', 'text', 'subject' keys
        """
        # Determine subject
        level_prefix = LEVEL_EMOJIS.get(payload.level, "[NOTIFICATION]")
        title = payload.title or f"{payload.level.value.upper()} Notification"
        subject = self._email_config.subject_prefix + title

        # Build HTML body
        html_body = self._build_html_body(payload)

        # Build plain text body
        text_body = self._build_text_body(payload)

        return {
            "subject": subject,
            "html": html_body,
            "text": text_body,
            "recipients": self._get_recipients(payload),
        }

    def _get_recipients(self, payload: NotificationPayload) -> list[str]:
        """Get the list of recipients for the email.

        Args:
            payload: The notification payload

        Returns:
            List of recipient email addresses
        """
        recipients = list(self._email_config.default_recipients)

        # Add recipients from payload if available
        if hasattr(payload, "recipients") and payload.recipients:
            recipients.extend(payload.recipients)

        # Check context for recipients
        context = payload.context_dict
        if "recipients" in context:
            r = context["recipients"]
            if isinstance(r, str):
                recipients.extend(r.split(","))
            elif isinstance(r, (list, tuple)):
                recipients.extend(r)

        # Remove duplicates while preserving order
        seen = set()
        unique_recipients = []
        for r in recipients:
            r_lower = r.strip().lower()
            if r_lower not in seen:
                seen.add(r_lower)
                unique_recipients.append(r.strip())

        return unique_recipients

    def _build_html_body(self, payload: NotificationPayload) -> str:
        """Build the HTML body of the email.

        Args:
            payload: The notification payload

        Returns:
            HTML string
        """
        color = LEVEL_COLORS.get(payload.level, "#6c757d")
        icon = LEVEL_ICONS.get(payload.level, "📧")
        title = payload.title or f"{payload.level.value.upper()} Notification"

        # Escape message for HTML
        message_html = html.escape(payload.message).replace("\n", "<br>")

        # Build context section
        context_html = ""
        context_dict = payload.context_dict
        if context_dict:
            context_items = "".join(
                f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold; color: #555;">{html.escape(str(k))}</td>'
                f'<td style="padding: 8px; border-bottom: 1px solid #eee;">{html.escape(str(v))}</td></tr>'
                for k, v in context_dict.items()
            )
            context_html = f'''
            <div style="margin-top: 20px;">
                <h3 style="margin-bottom: 10px; color: #333;">Details</h3>
                <table style="width: 100%; border-collapse: collapse; background: #fafafa; border-radius: 4px;">
                    {context_items}
                </table>
            </div>
            '''

        # Build metadata footer
        metadata_parts = []
        if payload.metadata.source:
            metadata_parts.append(f"Source: {html.escape(payload.metadata.source)}")
        if payload.metadata.correlation_id:
            metadata_parts.append(f"ID: {html.escape(payload.metadata.correlation_id)}")
        metadata_parts.append(f"Level: {payload.level.value}")
        metadata_parts.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        metadata_html = " | ".join(metadata_parts)

        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="border-left: 4px solid {color}; padding-left: 20px; margin-bottom: 20px;">
        <h1 style="margin: 0 0 10px 0; color: {color}; font-size: 24px;">
            {icon} {html.escape(title)}
        </h1>
        <p style="margin: 0; color: #666; font-size: 14px;">
            {payload.level.value.upper()} Notification
        </p>
    </div>

    <div style="background: #fff; border: 1px solid #e9ecef; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 16px; white-space: pre-wrap;">
            {message_html}
        </p>
    </div>

    {context_html}

    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e9ecef; color: #999; font-size: 12px;">
        {metadata_html}
    </div>

    <div style="margin-top: 10px; color: #ccc; font-size: 11px;">
        Sent by Truthound Notification System
    </div>
</body>
</html>'''

    def _build_text_body(self, payload: NotificationPayload) -> str:
        """Build the plain text body of the email.

        Args:
            payload: The notification payload

        Returns:
            Plain text string
        """
        level_prefix = LEVEL_EMOJIS.get(payload.level, "[NOTIFICATION]")
        title = payload.title or f"{payload.level.value.upper()} Notification"

        lines = [
            f"{level_prefix} {title}",
            "=" * 50,
            "",
            payload.message,
            "",
        ]

        # Add context
        context_dict = payload.context_dict
        if context_dict:
            lines.append("-" * 50)
            lines.append("Details:")
            for k, v in context_dict.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # Add metadata
        lines.append("-" * 50)
        if payload.metadata.source:
            lines.append(f"Source: {payload.metadata.source}")
        if payload.metadata.correlation_id:
            lines.append(f"ID: {payload.metadata.correlation_id}")
        lines.append(f"Level: {payload.level.value}")
        lines.append(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")
        lines.append("Sent by Truthound Notification System")

        return "\n".join(lines)

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Send the notification via email.

        Routes to the appropriate provider-specific send method.

        Args:
            payload: The original notification payload
            formatted_message: The formatted email data

        Returns:
            Result of the send operation
        """
        if not isinstance(formatted_message, dict):
            formatted_message = {
                "subject": "Notification",
                "text": str(formatted_message),
                "html": f"<p>{html.escape(str(formatted_message))}</p>",
                "recipients": self._get_recipients(payload),
            }

        recipients = formatted_message.get("recipients", [])
        if not recipients:
            return NotificationResult.failure_result(
                channel=self.channel,
                handler_name=self.name,
                error="No recipients specified for email notification",
                error_type="NoRecipientsError",
            )

        # Route to provider-specific send
        provider = self._email_config.provider
        if provider == EmailProvider.SMTP:
            return await self._send_smtp(formatted_message)
        elif provider == EmailProvider.SENDGRID:
            return await self._send_sendgrid(formatted_message)
        elif provider == EmailProvider.SES:
            return await self._send_ses(formatted_message)
        elif provider == EmailProvider.MAILGUN:
            return await self._send_mailgun(formatted_message)
        elif provider == EmailProvider.POSTMARK:
            return await self._send_postmark(formatted_message)
        elif provider == EmailProvider.RESEND:
            return await self._send_resend(formatted_message)
        else:
            return NotificationResult.failure_result(
                channel=self.channel,
                handler_name=self.name,
                error=f"Unsupported email provider: {provider}",
                error_type="UnsupportedProviderError",
            )

    async def _send_smtp(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Send email via SMTP.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_smtp_sync,
            message_data,
        )

    def _send_smtp_sync(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous SMTP send.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        try:
            # Build MIME message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = message_data["subject"]
            msg["From"] = self._format_from_address()
            msg["To"] = ", ".join(message_data["recipients"])

            if self._email_config.reply_to:
                msg["Reply-To"] = self._email_config.reply_to

            # Add CC and BCC if configured
            cc_recipients = list(self._email_config.cc_recipients)
            bcc_recipients = list(self._email_config.bcc_recipients)

            if cc_recipients:
                msg["Cc"] = ", ".join(cc_recipients)

            # Attach text and HTML parts
            if message_data.get("text"):
                msg.attach(MIMEText(message_data["text"], "plain", "utf-8"))
            if message_data.get("html"):
                msg.attach(MIMEText(message_data["html"], "html", "utf-8"))

            # Combine all recipients for sending
            all_recipients = (
                message_data["recipients"] + cc_recipients + bcc_recipients
            )

            # Connect and send
            if self._email_config.encryption == EmailEncryption.SSL_TLS:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self._email_config.smtp_host,
                    self._email_config.smtp_port,
                    context=context,
                    timeout=self._config.timeout_seconds,
                ) as server:
                    self._smtp_auth_and_send(server, msg, all_recipients)
            else:
                with smtplib.SMTP(
                    self._email_config.smtp_host,
                    self._email_config.smtp_port,
                    timeout=self._config.timeout_seconds,
                ) as server:
                    if self._email_config.encryption == EmailEncryption.STARTTLS:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                    self._smtp_auth_and_send(server, msg, all_recipients)

            return NotificationResult.success_result(
                channel=self.channel,
                handler_name=self.name,
                response_data={
                    "recipients": message_data["recipients"],
                    "subject": message_data["subject"],
                },
            )

        except smtplib.SMTPAuthenticationError as e:
            raise NotificationSendError(
                message=f"SMTP authentication failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e
        except smtplib.SMTPException as e:
            raise NotificationSendError(
                message=f"SMTP error: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e
        except TimeoutError as e:
            raise NotificationTimeoutError(
                message="SMTP connection timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e

    def _smtp_auth_and_send(
        self,
        server: smtplib.SMTP | smtplib.SMTP_SSL,
        msg: MIMEMultipart,
        recipients: list[str],
    ) -> None:
        """Authenticate with SMTP server and send email.

        Args:
            server: SMTP server connection
            msg: MIME message to send
            recipients: List of recipient email addresses
        """
        if self._email_config.smtp_username and self._email_config.smtp_password:
            server.login(
                self._email_config.smtp_username,
                self._email_config.smtp_password,
            )
        server.send_message(msg, to_addrs=recipients)

    def _format_from_address(self) -> str:
        """Format the From address with optional display name.

        Returns:
            Formatted From address
        """
        if self._email_config.from_name:
            return f"{self._email_config.from_name} <{self._email_config.from_address}>"
        return self._email_config.from_address

    async def _send_sendgrid(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Send email via SendGrid API.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        try:
            import aiohttp
        except ImportError:
            return await self._send_sendgrid_sync(message_data)

        import asyncio

        payload = {
            "personalizations": [
                {
                    "to": [{"email": r} for r in message_data["recipients"]],
                }
            ],
            "from": {"email": self._email_config.from_address},
            "subject": message_data["subject"],
            "content": [],
        }

        if self._email_config.from_name:
            payload["from"]["name"] = self._email_config.from_name

        if message_data.get("text"):
            payload["content"].append({
                "type": "text/plain",
                "value": message_data["text"],
            })
        if message_data.get("html"):
            payload["content"].append({
                "type": "text/html",
                "value": message_data["html"],
            })

        if self._email_config.reply_to:
            payload["reply_to"] = {"email": self._email_config.reply_to}

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._email_config.sendgrid_api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._email_config.api_key}",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    if response.status in (200, 201, 202):
                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            response_data={
                                "status": response.status,
                                "provider": "sendgrid",
                            },
                        )
                    else:
                        response_text = await response.text()
                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=f"SendGrid API error ({response.status}): {response_text}",
                            error_type="SendGridAPIError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="SendGrid API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"SendGrid API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_sendgrid_sync(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Fallback synchronous SendGrid send.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_sendgrid_sync_impl,
            message_data,
        )

    def _send_sendgrid_sync_impl(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous SendGrid send implementation.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        try:
            import requests

            payload = {
                "personalizations": [
                    {
                        "to": [{"email": r} for r in message_data["recipients"]],
                    }
                ],
                "from": {"email": self._email_config.from_address},
                "subject": message_data["subject"],
                "content": [],
            }

            if self._email_config.from_name:
                payload["from"]["name"] = self._email_config.from_name

            if message_data.get("text"):
                payload["content"].append({
                    "type": "text/plain",
                    "value": message_data["text"],
                })
            if message_data.get("html"):
                payload["content"].append({
                    "type": "text/html",
                    "value": message_data["html"],
                })

            response = requests.post(
                self._email_config.sendgrid_api_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._email_config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._config.timeout_seconds,
            )

            if response.status_code in (200, 201, 202):
                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    response_data={
                        "status": response.status_code,
                        "provider": "sendgrid",
                    },
                )
            else:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=f"SendGrid API error ({response.status_code}): {response.text}",
                    error_type="SendGridAPIError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="SendGrid API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"SendGrid API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_ses(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Send email via AWS SES.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        import asyncio

        try:
            import boto3
        except ImportError:
            return NotificationResult.failure_result(
                channel=self.channel,
                handler_name=self.name,
                error="boto3 is required for AWS SES provider. Install with: pip install boto3",
                error_type="DependencyError",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_ses_sync,
            message_data,
        )

    def _send_ses_sync(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous AWS SES send.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            # Create SES client
            client_kwargs: dict[str, Any] = {}
            if self._email_config.aws_region:
                client_kwargs["region_name"] = self._email_config.aws_region
            if self._email_config.api_key and self._email_config.aws_secret_key:
                client_kwargs["aws_access_key_id"] = self._email_config.api_key
                client_kwargs["aws_secret_access_key"] = self._email_config.aws_secret_key

            client = boto3.client("ses", **client_kwargs)

            # Build message
            message: dict[str, Any] = {
                "Subject": {
                    "Data": message_data["subject"],
                    "Charset": "UTF-8",
                },
                "Body": {},
            }

            if message_data.get("text"):
                message["Body"]["Text"] = {
                    "Data": message_data["text"],
                    "Charset": "UTF-8",
                }
            if message_data.get("html"):
                message["Body"]["Html"] = {
                    "Data": message_data["html"],
                    "Charset": "UTF-8",
                }

            # Send email
            response = client.send_email(
                Source=self._format_from_address(),
                Destination={
                    "ToAddresses": message_data["recipients"],
                    "CcAddresses": list(self._email_config.cc_recipients),
                    "BccAddresses": list(self._email_config.bcc_recipients),
                },
                Message=message,
                ReplyToAddresses=[self._email_config.reply_to] if self._email_config.reply_to else [],
            )

            return NotificationResult.success_result(
                channel=self.channel,
                handler_name=self.name,
                message_id=response.get("MessageId"),
                response_data={
                    "message_id": response.get("MessageId"),
                    "provider": "ses",
                },
            )

        except (BotoCoreError, ClientError) as e:
            raise NotificationSendError(
                message=f"AWS SES error: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_mailgun(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Send email via Mailgun API.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        try:
            import aiohttp
        except ImportError:
            return await self._send_mailgun_sync(message_data)

        import asyncio
        from aiohttp import BasicAuth

        if not self._email_config.mailgun_domain:
            return NotificationResult.failure_result(
                channel=self.channel,
                handler_name=self.name,
                error="mailgun_domain is required for Mailgun provider",
                error_type="ConfigurationError",
            )

        url = f"{self._email_config.mailgun_api_url}/{self._email_config.mailgun_domain}/messages"

        data = {
            "from": self._format_from_address(),
            "to": message_data["recipients"],
            "subject": message_data["subject"],
        }

        if message_data.get("text"):
            data["text"] = message_data["text"]
        if message_data.get("html"):
            data["html"] = message_data["html"]
        if self._email_config.reply_to:
            data["h:Reply-To"] = self._email_config.reply_to

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        auth = BasicAuth("api", self._email_config.api_key or "")

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=data, auth=auth) as response:
                    response_json = await response.json()

                    if response.status == 200:
                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            message_id=response_json.get("id"),
                            response_data={
                                "id": response_json.get("id"),
                                "message": response_json.get("message"),
                                "provider": "mailgun",
                            },
                        )
                    else:
                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=f"Mailgun API error ({response.status}): {response_json}",
                            error_type="MailgunAPIError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="Mailgun API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Mailgun API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_mailgun_sync(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Fallback synchronous Mailgun send.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_mailgun_sync_impl,
            message_data,
        )

    def _send_mailgun_sync_impl(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous Mailgun send implementation.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        try:
            import requests

            if not self._email_config.mailgun_domain:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error="mailgun_domain is required for Mailgun provider",
                    error_type="ConfigurationError",
                )

            url = f"{self._email_config.mailgun_api_url}/{self._email_config.mailgun_domain}/messages"

            data = {
                "from": self._format_from_address(),
                "to": message_data["recipients"],
                "subject": message_data["subject"],
            }

            if message_data.get("text"):
                data["text"] = message_data["text"]
            if message_data.get("html"):
                data["html"] = message_data["html"]

            response = requests.post(
                url,
                data=data,
                auth=("api", self._email_config.api_key or ""),
                timeout=self._config.timeout_seconds,
            )

            if response.status_code == 200:
                response_json = response.json()
                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    message_id=response_json.get("id"),
                    response_data={
                        "id": response_json.get("id"),
                        "provider": "mailgun",
                    },
                )
            else:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=f"Mailgun API error ({response.status_code}): {response.text}",
                    error_type="MailgunAPIError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="Mailgun API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Mailgun API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_postmark(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Send email via Postmark API.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        try:
            import aiohttp
        except ImportError:
            return await self._send_postmark_sync(message_data)

        import asyncio

        payload = {
            "From": self._format_from_address(),
            "To": ", ".join(message_data["recipients"]),
            "Subject": message_data["subject"],
        }

        if message_data.get("text"):
            payload["TextBody"] = message_data["text"]
        if message_data.get("html"):
            payload["HtmlBody"] = message_data["html"]
        if self._email_config.reply_to:
            payload["ReplyTo"] = self._email_config.reply_to

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._email_config.postmark_api_url,
                    json=payload,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "X-Postmark-Server-Token": self._email_config.api_key or "",
                    },
                ) as response:
                    response_json = await response.json()

                    if response.status == 200:
                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            message_id=response_json.get("MessageID"),
                            response_data={
                                "message_id": response_json.get("MessageID"),
                                "provider": "postmark",
                            },
                        )
                    else:
                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=f"Postmark API error ({response.status}): {response_json}",
                            error_type="PostmarkAPIError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="Postmark API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Postmark API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_postmark_sync(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Fallback synchronous Postmark send."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_postmark_sync_impl,
            message_data,
        )

    def _send_postmark_sync_impl(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous Postmark send implementation."""
        try:
            import requests

            payload = {
                "From": self._format_from_address(),
                "To": ", ".join(message_data["recipients"]),
                "Subject": message_data["subject"],
            }

            if message_data.get("text"):
                payload["TextBody"] = message_data["text"]
            if message_data.get("html"):
                payload["HtmlBody"] = message_data["html"]

            response = requests.post(
                self._email_config.postmark_api_url,
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "X-Postmark-Server-Token": self._email_config.api_key or "",
                },
                timeout=self._config.timeout_seconds,
            )

            if response.status_code == 200:
                response_json = response.json()
                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    message_id=response_json.get("MessageID"),
                    response_data={
                        "message_id": response_json.get("MessageID"),
                        "provider": "postmark",
                    },
                )
            else:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=f"Postmark API error ({response.status_code}): {response.text}",
                    error_type="PostmarkAPIError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="Postmark API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Postmark API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_resend(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Send email via Resend API.

        Args:
            message_data: The formatted email data

        Returns:
            Result of the send operation
        """
        try:
            import aiohttp
        except ImportError:
            return await self._send_resend_sync(message_data)

        import asyncio

        payload = {
            "from": self._format_from_address(),
            "to": message_data["recipients"],
            "subject": message_data["subject"],
        }

        if message_data.get("text"):
            payload["text"] = message_data["text"]
        if message_data.get("html"):
            payload["html"] = message_data["html"]
        if self._email_config.reply_to:
            payload["reply_to"] = self._email_config.reply_to

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._email_config.resend_api_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._email_config.api_key}",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    response_json = await response.json()

                    if response.status == 200:
                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            message_id=response_json.get("id"),
                            response_data={
                                "id": response_json.get("id"),
                                "provider": "resend",
                            },
                        )
                    else:
                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=f"Resend API error ({response.status}): {response_json}",
                            error_type="ResendAPIError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="Resend API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Resend API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_resend_sync(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Fallback synchronous Resend send."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_resend_sync_impl,
            message_data,
        )

    def _send_resend_sync_impl(
        self,
        message_data: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous Resend send implementation."""
        try:
            import requests

            payload = {
                "from": self._format_from_address(),
                "to": message_data["recipients"],
                "subject": message_data["subject"],
            }

            if message_data.get("text"):
                payload["text"] = message_data["text"]
            if message_data.get("html"):
                payload["html"] = message_data["html"]

            response = requests.post(
                self._email_config.resend_api_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._email_config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._config.timeout_seconds,
            )

            if response.status_code == 200:
                response_json = response.json()
                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    message_id=response_json.get("id"),
                    response_data={
                        "id": response_json.get("id"),
                        "provider": "resend",
                    },
                )
            else:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=f"Resend API error ({response.status_code}): {response.text}",
                    error_type="ResendAPIError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="Resend API request timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Resend API request failed: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    def format_simple_message(
        self,
        message: str,
        subject: str | None = None,
        level: NotificationLevel = NotificationLevel.INFO,
    ) -> dict[str, Any]:
        """Create a simple email message without rich formatting.

        Useful for testing or simple notifications.

        Args:
            message: The message text
            subject: Email subject (optional)
            level: Notification level

        Returns:
            Simple email message data
        """
        level_prefix = LEVEL_EMOJIS.get(level, "[NOTIFICATION]")
        final_subject = subject or f"{level_prefix} Notification"

        return {
            "subject": self._email_config.subject_prefix + final_subject,
            "text": message,
            "html": f"<p>{html.escape(message).replace(chr(10), '<br>')}</p>",
            "recipients": list(self._email_config.default_recipients),
        }
