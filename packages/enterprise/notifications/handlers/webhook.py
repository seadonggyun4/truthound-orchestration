"""Generic webhook notification handler.

This module provides a flexible webhook handler that can send notifications
to any HTTP endpoint with customizable formatting and authentication.

Features:
    - Multiple HTTP methods (POST, PUT, PATCH)
    - Various authentication types (Basic, Bearer, API Key)
    - Custom headers support
    - Flexible payload formatting
    - SSL verification control

Example:
    >>> from packages.enterprise.notifications.handlers import WebhookNotificationHandler
    >>> from packages.enterprise.notifications.config import WebhookConfig
    >>>
    >>> handler = WebhookNotificationHandler(
    ...     config=WebhookConfig(
    ...         url="https://api.example.com/notify",
    ...         method="POST",
    ...     ).with_bearer_token("your-token"),
    ... )
    >>>
    >>> result = await handler.send(NotificationPayload(
    ...     message="Data quality check completed",
    ...     level=NotificationLevel.INFO,
    ... ))
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from packages.enterprise.notifications.config import (
    NotificationConfig,
    WebhookConfig,
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


class WebhookNotificationHandler(AsyncNotificationHandler):
    """Generic webhook notification handler.

    Sends notifications to any HTTP endpoint with flexible configuration
    for authentication, headers, and payload formatting.

    Attributes:
        webhook_config: Webhook-specific configuration
    """

    def __init__(
        self,
        config: WebhookConfig | None = None,
        *,
        url: str | None = None,
        method: str = "POST",
        name: str | None = None,
        base_config: NotificationConfig | None = None,
        hooks: list[NotificationHook] | None = None,
    ) -> None:
        """Initialize the webhook handler.

        Args:
            config: Complete webhook configuration
            url: Webhook URL (alternative to config)
            method: HTTP method (alternative to config)
            name: Handler name
            base_config: Base notification config
            hooks: Notification hooks
        """
        if config is not None:
            self._webhook_config = config
            base = config.base_config
        elif url is not None:
            self._webhook_config = WebhookConfig(
                url=url,
                method=method,
                base_config=base_config or NotificationConfig(),
            )
            base = self._webhook_config.base_config
        else:
            raise ValueError("Either config or url must be provided")

        super().__init__(
            name=name or "webhook",
            config=base,
            hooks=hooks,
        )

    @property
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        return NotificationChannel.WEBHOOK

    @property
    def webhook_config(self) -> WebhookConfig:
        """Get the webhook-specific configuration."""
        return self._webhook_config

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the request.

        Combines configured headers with authentication headers.

        Returns:
            Complete headers dictionary
        """
        headers = {"Content-Type": self._webhook_config.content_type}

        # Add custom headers
        headers.update(self._webhook_config.headers_dict)

        # Add authentication header
        if self._webhook_config.auth_type == "basic":
            headers["Authorization"] = f"Basic {self._webhook_config.auth_credentials}"
        elif self._webhook_config.auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self._webhook_config.auth_credentials}"
        # api_key is handled via headers_dict in with_api_key

        return headers

    async def _format_message(
        self,
        payload: NotificationPayload,
    ) -> dict[str, Any]:
        """Format the message for the webhook.

        Creates a structured JSON payload with all notification data.

        Args:
            payload: The notification payload

        Returns:
            Webhook payload as dictionary
        """
        message: dict[str, Any] = {
            "message": payload.message,
            "level": payload.level.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if payload.title:
            message["title"] = payload.title

        # Add context
        if payload.context:
            message["context"] = payload.context_dict

        # Add metadata if configured
        if self._config.include_metadata:
            message["metadata"] = {
                "source": payload.metadata.source,
                "correlation_id": payload.metadata.correlation_id,
                "trace_id": payload.metadata.trace_id,
                "tags": list(payload.metadata.tags),
                "attributes": dict(payload.metadata.attributes),
            }

        return message

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Send the notification to the webhook endpoint.

        Uses aiohttp for async HTTP requests.

        Args:
            payload: The original notification payload
            formatted_message: The formatted webhook payload

        Returns:
            Result of the send operation
        """
        import asyncio

        try:
            import aiohttp
        except ImportError:
            return await self._send_with_requests(formatted_message)

        if isinstance(formatted_message, str):
            formatted_message = {"message": formatted_message}

        headers = self._build_headers()
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)

        ssl_context = None if self._webhook_config.verify_ssl else False

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                method = self._webhook_config.method.upper()

                async with session.request(
                    method,
                    self._webhook_config.url,
                    json=formatted_message,
                    headers=headers,
                    ssl=ssl_context,
                ) as response:
                    response_body = await response.text()

                    # Consider 2xx as success
                    if 200 <= response.status < 300:
                        # Try to parse response as JSON for message_id
                        message_id = None
                        try:
                            response_json = json.loads(response_body)
                            message_id = response_json.get("id") or response_json.get("message_id")
                        except (json.JSONDecodeError, TypeError):
                            pass

                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            message_id=message_id,
                            response_data={
                                "status": response.status,
                                "body": response_body[:500],  # Truncate
                            },
                        )
                    else:
                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=f"HTTP {response.status}: {response_body[:200]}",
                            error_type="HTTPError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="Webhook notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Failed to send webhook notification: {e}",
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
            formatted_message: The formatted webhook payload

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
            formatted_message: The formatted webhook payload

        Returns:
            Result of the send operation
        """
        try:
            import requests

            headers = self._build_headers()
            method = self._webhook_config.method.upper()

            response = requests.request(
                method,
                self._webhook_config.url,
                json=formatted_message,
                headers=headers,
                timeout=self._config.timeout_seconds,
                verify=self._webhook_config.verify_ssl,
            )

            if 200 <= response.status_code < 300:
                message_id = None
                try:
                    response_json = response.json()
                    message_id = response_json.get("id") or response_json.get("message_id")
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    message_id=message_id,
                    response_data={
                        "status": response.status_code,
                        "body": response.text[:500],
                    },
                )
            else:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    error_type="HTTPError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="Webhook notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Failed to send webhook notification: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e


class JsonWebhookHandler(WebhookNotificationHandler):
    """Webhook handler that sends raw JSON payloads.

    A simplified handler that sends the notification data as-is,
    without additional formatting.
    """

    async def _format_message(
        self,
        payload: NotificationPayload,
    ) -> dict[str, Any]:
        """Format as raw JSON.

        Args:
            payload: The notification payload

        Returns:
            Payload data as dictionary
        """
        return payload.to_dict()


class FormWebhookHandler(WebhookNotificationHandler):
    """Webhook handler that sends form-encoded data.

    Useful for legacy systems that expect form data instead of JSON.
    """

    def __init__(
        self,
        config: WebhookConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the form webhook handler."""
        super().__init__(config=config, **kwargs)
        # Override content type for form data
        self._webhook_config = WebhookConfig(
            url=self._webhook_config.url,
            method=self._webhook_config.method,
            headers=self._webhook_config.headers,
            auth_type=self._webhook_config.auth_type,
            auth_credentials=self._webhook_config.auth_credentials,
            content_type="application/x-www-form-urlencoded",
            verify_ssl=self._webhook_config.verify_ssl,
            base_config=self._webhook_config.base_config,
        )

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Send form-encoded data.

        Args:
            payload: The original notification payload
            formatted_message: The formatted data

        Returns:
            Result of the send operation
        """
        import asyncio

        try:
            import aiohttp
        except ImportError:
            return await self._send_form_with_requests(formatted_message)

        if isinstance(formatted_message, str):
            formatted_message = {"message": formatted_message}

        headers = self._build_headers()
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        ssl_context = None if self._webhook_config.verify_ssl else False

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._webhook_config.url,
                    data=formatted_message,
                    headers=headers,
                    ssl=ssl_context,
                ) as response:
                    response_body = await response.text()

                    if 200 <= response.status < 300:
                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            response_data={"status": response.status},
                        )
                    else:
                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=f"HTTP {response.status}: {response_body[:200]}",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="Form webhook notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Failed to send form webhook notification: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def _send_form_with_requests(
        self,
        formatted_message: dict[str, Any],
    ) -> NotificationResult:
        """Fallback form send using requests library."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_form_sync,
            formatted_message,
        )

    def _send_form_sync(
        self,
        formatted_message: dict[str, Any],
    ) -> NotificationResult:
        """Synchronous form send."""
        try:
            import requests

            headers = self._build_headers()

            response = requests.post(
                self._webhook_config.url,
                data=formatted_message,
                headers=headers,
                timeout=self._config.timeout_seconds,
                verify=self._webhook_config.verify_ssl,
            )

            if 200 <= response.status_code < 300:
                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    response_data={"status": response.status_code},
                )
            else:
                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="Form webhook notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Failed to send form webhook notification: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e
