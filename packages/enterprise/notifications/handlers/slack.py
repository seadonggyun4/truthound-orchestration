"""Slack notification handler.

This module provides the Slack notification handler that sends notifications
to Slack channels via incoming webhooks.

Features:
    - Rich message formatting with blocks
    - Emoji and color coding by notification level
    - Attachment support
    - Retry with exponential backoff
    - Rate limiting support

Example:
    >>> from packages.enterprise.notifications.handlers import SlackNotificationHandler
    >>> from packages.enterprise.notifications.config import SlackConfig
    >>>
    >>> handler = SlackNotificationHandler(
    ...     config=SlackConfig(
    ...         webhook_url="https://hooks.slack.com/services/...",
    ...         channel="#alerts",
    ...         username="Data Quality Bot",
    ...     ),
    ... )
    >>>
    >>> result = await handler.send(NotificationPayload(
    ...     message="Data quality check failed!",
    ...     level=NotificationLevel.ERROR,
    ...     context=(("check_id", "chk_123"),),
    ... ))
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from packages.enterprise.notifications.config import (
    NotificationConfig,
    SlackConfig,
)
from packages.enterprise.notifications.exceptions import (
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


# Level to color mapping for Slack attachments
LEVEL_COLORS: dict[NotificationLevel, str] = {
    NotificationLevel.DEBUG: "#808080",    # Gray
    NotificationLevel.INFO: "#36a64f",     # Green
    NotificationLevel.WARNING: "#ffc107",  # Yellow/Amber
    NotificationLevel.ERROR: "#dc3545",    # Red
    NotificationLevel.CRITICAL: "#6f42c1", # Purple
}

# Level to emoji mapping
LEVEL_EMOJIS: dict[NotificationLevel, str] = {
    NotificationLevel.DEBUG: ":mag:",
    NotificationLevel.INFO: ":information_source:",
    NotificationLevel.WARNING: ":warning:",
    NotificationLevel.ERROR: ":x:",
    NotificationLevel.CRITICAL: ":rotating_light:",
}


class SlackNotificationHandler(AsyncNotificationHandler):
    """Slack notification handler using incoming webhooks.

    Sends notifications to Slack channels with rich formatting including
    blocks, attachments, and emoji indicators.

    Attributes:
        slack_config: Slack-specific configuration
    """

    def __init__(
        self,
        config: SlackConfig | None = None,
        *,
        webhook_url: str | None = None,
        channel: str | None = None,
        username: str | None = None,
        name: str | None = None,
        base_config: NotificationConfig | None = None,
        hooks: list[NotificationHook] | None = None,
    ) -> None:
        """Initialize the Slack handler.

        Args:
            config: Complete Slack configuration
            webhook_url: Slack webhook URL (alternative to config)
            channel: Override channel (alternative to config)
            username: Bot username (alternative to config)
            name: Handler name
            base_config: Base notification config (used if config not provided)
            hooks: Notification hooks
        """
        if config is not None:
            self._slack_config = config
            base = config.base_config
        elif webhook_url is not None:
            self._slack_config = SlackConfig(
                webhook_url=webhook_url,
                channel=channel,
                username=username,
                base_config=base_config or NotificationConfig(),
            )
            base = self._slack_config.base_config
        else:
            raise ValueError("Either config or webhook_url must be provided")

        super().__init__(
            name=name or "slack",
            config=base,
            hooks=hooks,
        )

    @property
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        return NotificationChannel.SLACK

    @property
    def slack_config(self) -> SlackConfig:
        """Get the Slack-specific configuration."""
        return self._slack_config

    async def _format_message(
        self,
        payload: NotificationPayload,
    ) -> dict[str, Any]:
        """Format the message for Slack.

        Creates a rich message with:
            - Header with emoji and level
            - Main message text
            - Context fields
            - Metadata footer
            - Color-coded attachment

        Args:
            payload: The notification payload

        Returns:
            Slack message payload as dictionary
        """
        emoji = LEVEL_EMOJIS.get(payload.level, ":bell:")
        color = LEVEL_COLORS.get(payload.level, "#808080")

        # Build blocks
        blocks: list[dict[str, Any]] = []

        # Header block
        title = payload.title or f"{payload.level.value.upper()} Notification"
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} {title}",
                "emoji": True,
            },
        })

        # Main message block
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn" if self._slack_config.mrkdwn else "plain_text",
                "text": payload.message,
            },
        })

        # Context fields (if any)
        context_dict = payload.context_dict
        if context_dict:
            fields = []
            for key, value in context_dict.items():
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}",
                })
            # Slack allows max 10 fields per section
            for i in range(0, len(fields), 10):
                blocks.append({
                    "type": "section",
                    "fields": fields[i:i + 10],
                })

        # Divider before footer
        blocks.append({"type": "divider"})

        # Footer with metadata
        footer_elements = []
        if payload.metadata.source:
            footer_elements.append({
                "type": "mrkdwn",
                "text": f"Source: `{payload.metadata.source}`",
            })
        if payload.metadata.correlation_id:
            footer_elements.append({
                "type": "mrkdwn",
                "text": f"ID: `{payload.metadata.correlation_id}`",
            })
        footer_elements.append({
            "type": "mrkdwn",
            "text": f"Level: `{payload.level.value}`",
        })
        footer_elements.append({
            "type": "mrkdwn",
            "text": f"Time: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        })

        blocks.append({
            "type": "context",
            "elements": footer_elements,
        })

        # Build the message payload
        message: dict[str, Any] = {
            "blocks": blocks,
            "attachments": [
                {
                    "color": color,
                    "fallback": f"{payload.level.value}: {payload.message}",
                }
            ],
        }

        # Add optional fields
        if self._slack_config.channel:
            message["channel"] = self._slack_config.channel
        if self._slack_config.username:
            message["username"] = self._slack_config.username
        if self._slack_config.icon_emoji:
            message["icon_emoji"] = self._slack_config.icon_emoji
        elif self._slack_config.icon_url:
            message["icon_url"] = self._slack_config.icon_url

        message["unfurl_links"] = self._slack_config.unfurl_links
        message["unfurl_media"] = self._slack_config.unfurl_media

        return message

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Send the notification to Slack.

        Uses aiohttp for async HTTP requests with retry support.

        Args:
            payload: The original notification payload
            formatted_message: The formatted Slack message

        Returns:
            Result of the send operation
        """
        import asyncio

        try:
            import aiohttp
        except ImportError:
            # Fallback to synchronous requests if aiohttp not available
            return await self._send_with_requests(formatted_message)

        if not isinstance(formatted_message, dict):
            formatted_message = {"text": str(formatted_message)}

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._slack_config.webhook_url,
                    json=formatted_message,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response_text = await response.text()

                    if response.status == 200 and response_text == "ok":
                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            response_data={"status": response.status},
                        )
                    else:
                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=f"Slack API error: {response_text}",
                            error_type="SlackAPIError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="Slack notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Failed to send Slack notification: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_with_requests(
        self,
        formatted_message: dict[str, Any],
    ) -> NotificationResult:
        """Fallback send using requests library.

        Used when aiohttp is not available.

        Args:
            formatted_message: The formatted Slack message

        Returns:
            Result of the send operation
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_sync,
            formatted_message,
        )

    def _send_sync(
        self,
        formatted_message: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous send using requests library.

        Args:
            formatted_message: The formatted Slack message

        Returns:
            Result of the send operation
        """
        try:
            import requests

            response = requests.post(
                self._slack_config.webhook_url,
                json=formatted_message,
                headers={"Content-Type": "application/json"},
                timeout=self._config.timeout_seconds,
            )

            if response.status_code == 200 and response.text == "ok":
                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    response_data={"status": response.status_code},
                )
            else:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=f"Slack API error: {response.text}",
                    error_type="SlackAPIError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="Slack notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Failed to send Slack notification: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    def format_simple_message(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
    ) -> dict[str, Any]:
        """Create a simple text message without blocks.

        Useful for testing or simple notifications.

        Args:
            message: The message text
            level: Notification level

        Returns:
            Simple Slack message payload
        """
        emoji = LEVEL_EMOJIS.get(level, ":bell:")
        color = LEVEL_COLORS.get(level, "#808080")

        return {
            "text": f"{emoji} {message}",
            "attachments": [{"color": color, "fallback": message}],
        }
