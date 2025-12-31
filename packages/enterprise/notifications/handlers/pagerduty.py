"""PagerDuty notification handler.

This module provides the PagerDuty notification handler that sends alerts
to PagerDuty via the Events API v2.

Features:
    - Events API v2 integration (trigger, acknowledge, resolve)
    - Automatic severity mapping from NotificationLevel
    - Deduplication key support for alert correlation
    - Rich payload with custom details, links, and images
    - US and EU region support
    - Retry with exponential backoff

Example:
    >>> from packages.enterprise.notifications.handlers import PagerDutyNotificationHandler
    >>> from packages.enterprise.notifications.config import PagerDutyConfig
    >>>
    >>> handler = PagerDutyNotificationHandler(
    ...     config=PagerDutyConfig(
    ...         routing_key="your-32-character-routing-key",
    ...         default_source="data-quality-service",
    ...     ),
    ... )
    >>>
    >>> result = await handler.send(NotificationPayload(
    ...     message="Data quality check failed!",
    ...     level=NotificationLevel.CRITICAL,
    ...     title="Database Validation Error",
    ... ))
    >>>
    >>> # Acknowledge an alert
    >>> await handler.acknowledge(dedup_key="alert-123")
    >>>
    >>> # Resolve an alert
    >>> await handler.resolve(dedup_key="alert-123")

API Reference:
    PagerDuty Events API v2:
    https://developer.pagerduty.com/docs/ZG9jOjExMDI5NTgw-events-api-v2-overview
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from packages.enterprise.notifications.config import (
    NotificationConfig,
    PagerDutyConfig,
)
from packages.enterprise.notifications.exceptions import (
    NotificationSendError,
    NotificationTimeoutError,
)
from packages.enterprise.notifications.handlers.incident import (
    IncidentAction,
    IncidentDetails,
    IncidentManagementHandler,
    IncidentSeverity,
)
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationPayload,
    NotificationResult,
)

if TYPE_CHECKING:
    from packages.enterprise.notifications.hooks import NotificationHook


# PagerDuty severity mapping (Events API v2 values)
PAGERDUTY_SEVERITY_MAP: dict[IncidentSeverity, str] = {
    IncidentSeverity.CRITICAL: "critical",
    IncidentSeverity.HIGH: "error",
    IncidentSeverity.MEDIUM: "warning",
    IncidentSeverity.LOW: "info",
    IncidentSeverity.INFO: "info",
}

# PagerDuty action mapping (Events API v2 event_action values)
PAGERDUTY_ACTION_MAP: dict[IncidentAction, str] = {
    IncidentAction.TRIGGER: "trigger",
    IncidentAction.ACKNOWLEDGE: "acknowledge",
    IncidentAction.RESOLVE: "resolve",
}


class PagerDutyNotificationHandler(IncidentManagementHandler):
    """PagerDuty notification handler using Events API v2.

    Sends alerts to PagerDuty with support for triggering, acknowledging,
    and resolving incidents. Uses deduplication keys for alert correlation.

    Attributes:
        pagerduty_config: PagerDuty-specific configuration

    Example:
        >>> handler = PagerDutyNotificationHandler(
        ...     config=PagerDutyConfig(
        ...         routing_key="abcd1234abcd1234abcd1234abcd1234",
        ...         default_source="my-service",
        ...     ),
        ... )
        >>>
        >>> # Trigger an alert
        >>> result = await handler.send(NotificationPayload(
        ...     message="Database connection failed",
        ...     level=NotificationLevel.CRITICAL,
        ... ))
        >>>
        >>> # Or with explicit incident details
        >>> from packages.enterprise.notifications.handlers.incident import (
        ...     IncidentDetails, IncidentSeverity,
        ... )
        >>> details = IncidentDetails(
        ...     dedup_key="db-connection-prod-01",
        ...     severity=IncidentSeverity.CRITICAL,
        ...     source="database-monitor",
        ...     component="postgresql",
        ...     group="production",
        ... )
        >>> result = await handler.trigger(payload, incident_details=details)
    """

    def __init__(
        self,
        config: PagerDutyConfig | None = None,
        *,
        routing_key: str | None = None,
        name: str | None = None,
        base_config: NotificationConfig | None = None,
        hooks: list[NotificationHook] | None = None,
    ) -> None:
        """Initialize the PagerDuty handler.

        Args:
            config: Complete PagerDuty configuration
            routing_key: PagerDuty routing/integration key (alternative to config)
            name: Handler name
            base_config: Base notification config (used if config not provided)
            hooks: Notification hooks

        Raises:
            ValueError: If neither config nor routing_key is provided
        """
        if config is not None:
            self._pagerduty_config = config
            base = config.base_config
        elif routing_key is not None:
            self._pagerduty_config = PagerDutyConfig(
                routing_key=routing_key,
                base_config=base_config or NotificationConfig(),
            )
            base = self._pagerduty_config.base_config
        else:
            raise ValueError("Either config or routing_key must be provided")

        super().__init__(
            name=name or "pagerduty",
            config=base,
            hooks=hooks,
            default_source=self._pagerduty_config.default_source,
            auto_dedup=True,
        )

    @property
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        return NotificationChannel.PAGERDUTY

    @property
    def pagerduty_config(self) -> PagerDutyConfig:
        """Get the PagerDuty-specific configuration."""
        return self._pagerduty_config

    def _map_severity(self, severity: IncidentSeverity) -> str:
        """Map standard severity to PagerDuty severity.

        Args:
            severity: Standard incident severity

        Returns:
            PagerDuty severity string (critical, error, warning, info)
        """
        return PAGERDUTY_SEVERITY_MAP.get(severity, self._pagerduty_config.default_severity)

    def _map_action(self, action: IncidentAction) -> str:
        """Map standard action to PagerDuty event_action.

        Args:
            action: Standard incident action

        Returns:
            PagerDuty event_action string
        """
        return PAGERDUTY_ACTION_MAP.get(action, "trigger")

    def _build_request_body(
        self,
        payload: NotificationPayload,
        incident_details: IncidentDetails,
    ) -> dict[str, Any]:
        """Build the PagerDuty Events API v2 request body.

        Creates a properly formatted request body following the
        PagerDuty Events API v2 specification.

        Args:
            payload: The notification payload
            incident_details: Extracted incident details

        Returns:
            PagerDuty API request body

        Reference:
            https://developer.pagerduty.com/docs/ZG9jOjExMDI5NTgx-send-an-event-to-pagerduty
        """
        event_action = self._map_action(incident_details.action)

        # Build dedup_key with optional prefix
        dedup_key = incident_details.dedup_key or self._generate_dedup_key(payload)
        if self._pagerduty_config.dedup_key_prefix:
            dedup_key = f"{self._pagerduty_config.dedup_key_prefix}{dedup_key}"

        # Base request body
        body: dict[str, Any] = {
            "routing_key": self._pagerduty_config.routing_key,
            "event_action": event_action,
            "dedup_key": dedup_key,
        }

        # For trigger events, include full payload
        if event_action == "trigger":
            body["payload"] = self._build_payload(payload, incident_details)

            # Add optional client info
            if self._pagerduty_config.client_name:
                body["client"] = self._pagerduty_config.client_name
            if self._pagerduty_config.client_url:
                body["client_url"] = self._pagerduty_config.client_url

            # Add links
            if incident_details.links:
                body["links"] = incident_details.links_list

            # Add images
            if incident_details.images:
                body["images"] = [{"src": url} for url in incident_details.images]

        return body

    def _build_payload(
        self,
        payload: NotificationPayload,
        incident_details: IncidentDetails,
    ) -> dict[str, Any]:
        """Build the payload section for trigger events.

        Args:
            payload: The notification payload
            incident_details: Extracted incident details

        Returns:
            PagerDuty payload section
        """
        pd_payload: dict[str, Any] = {
            "summary": payload.title or payload.message[:1024],
            "severity": self._map_severity(incident_details.severity),
            "source": incident_details.source or self._pagerduty_config.default_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add optional fields
        if incident_details.component or self._pagerduty_config.default_component:
            pd_payload["component"] = (
                incident_details.component or self._pagerduty_config.default_component
            )

        if incident_details.group or self._pagerduty_config.default_group:
            pd_payload["group"] = (
                incident_details.group or self._pagerduty_config.default_group
            )

        if incident_details.class_type or self._pagerduty_config.default_class:
            pd_payload["class"] = (
                incident_details.class_type or self._pagerduty_config.default_class
            )

        # Build custom_details
        custom_details: dict[str, Any] = {}

        # Add message as description if different from summary
        if payload.message != pd_payload["summary"]:
            custom_details["description"] = payload.message

        # Add notification context if configured
        if self._pagerduty_config.include_context_in_details:
            context = payload.context_dict
            # Filter out incident-specific keys
            excluded_keys = {
                "dedup_key", "action", "source", "component", "group", "class_type"
            }
            for key, value in context.items():
                if key not in excluded_keys:
                    custom_details[key] = str(value)

        # Add explicit custom details from incident_details
        custom_details.update(incident_details.custom_details_dict)

        # Add metadata
        if payload.metadata.correlation_id:
            custom_details["correlation_id"] = payload.metadata.correlation_id
        if payload.metadata.trace_id:
            custom_details["trace_id"] = payload.metadata.trace_id
        if payload.metadata.tags:
            custom_details["tags"] = list(payload.metadata.tags)

        if custom_details:
            pd_payload["custom_details"] = custom_details

        return pd_payload

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Send the notification to PagerDuty.

        Uses aiohttp for async HTTP requests with retry support.

        Args:
            payload: The original notification payload
            formatted_message: The formatted PagerDuty request body

        Returns:
            Result of the send operation

        Raises:
            NotificationTimeoutError: If the request times out
            NotificationSendError: If the request fails
        """
        import asyncio

        try:
            import aiohttp
        except ImportError:
            return await self._send_with_requests(formatted_message)

        if not isinstance(formatted_message, dict):
            raise NotificationSendError(
                message="Invalid message format for PagerDuty",
                handler_name=self.name,
                channel=self.channel,
            )

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        headers = {
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._pagerduty_config.api_url,
                    json=formatted_message,
                    headers=headers,
                ) as response:
                    response_body = await response.text()

                    # PagerDuty returns 202 Accepted for successful events
                    if response.status in {200, 201, 202}:
                        # Parse response for dedup_key
                        dedup_key = None
                        try:
                            response_json = json.loads(response_body)
                            dedup_key = response_json.get("dedup_key")
                        except (json.JSONDecodeError, TypeError):
                            pass

                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            message_id=dedup_key,
                            response_data={
                                "status": response.status,
                                "dedup_key": dedup_key,
                            },
                        )
                    else:
                        # Parse error response
                        error_msg = f"PagerDuty API error: {response.status}"
                        try:
                            error_json = json.loads(response_body)
                            if "message" in error_json:
                                error_msg = f"PagerDuty API error: {error_json['message']}"
                            elif "errors" in error_json:
                                error_msg = f"PagerDuty API error: {error_json['errors']}"
                        except (json.JSONDecodeError, TypeError):
                            error_msg = f"PagerDuty API error: {response_body[:200]}"

                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=error_msg,
                            error_type="PagerDutyAPIError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="PagerDuty notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Failed to send PagerDuty notification: {e}",
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
            formatted_message: The formatted PagerDuty request body

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
            formatted_message: The formatted PagerDuty request body

        Returns:
            Result of the send operation
        """
        try:
            import requests

            response = requests.post(
                self._pagerduty_config.api_url,
                json=formatted_message,
                headers={"Content-Type": "application/json"},
                timeout=self._config.timeout_seconds,
            )

            if response.status_code in {200, 201, 202}:
                dedup_key = None
                try:
                    response_json = response.json()
                    dedup_key = response_json.get("dedup_key")
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    message_id=dedup_key,
                    response_data={
                        "status": response.status_code,
                        "dedup_key": dedup_key,
                    },
                )
            else:
                error_msg = f"PagerDuty API error: {response.status_code}"
                try:
                    error_json = response.json()
                    if "message" in error_json:
                        error_msg = f"PagerDuty API error: {error_json['message']}"
                    elif "errors" in error_json:
                        error_msg = f"PagerDuty API error: {error_json['errors']}"
                except (json.JSONDecodeError, TypeError, ValueError):
                    error_msg = f"PagerDuty API error: {response.text[:200]}"

                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=error_msg,
                    error_type="PagerDutyAPIError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="PagerDuty notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Failed to send PagerDuty notification: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def trigger_with_details(
        self,
        message: str,
        *,
        title: str | None = None,
        severity: IncidentSeverity = IncidentSeverity.MEDIUM,
        dedup_key: str | None = None,
        source: str | None = None,
        component: str | None = None,
        group: str | None = None,
        class_type: str | None = None,
        custom_details: dict[str, Any] | None = None,
        links: list[tuple[str, str]] | None = None,
    ) -> NotificationResult:
        """Trigger a PagerDuty alert with explicit details.

        Convenience method for triggering alerts with full control over
        all PagerDuty-specific fields.

        Args:
            message: Alert message/description
            title: Alert summary/title (defaults to message)
            severity: Alert severity level
            dedup_key: Deduplication key for alert correlation
            source: Source of the alert
            component: Affected component
            group: Logical grouping
            class_type: Event class
            custom_details: Additional custom details
            links: List of (url, text) tuples for related links

        Returns:
            Result of the trigger operation

        Example:
            >>> result = await handler.trigger_with_details(
            ...     message="Database connection pool exhausted",
            ...     title="DB Pool Critical",
            ...     severity=IncidentSeverity.CRITICAL,
            ...     source="connection-monitor",
            ...     component="postgresql",
            ...     custom_details={"pool_size": 100, "active": 100},
            ...     links=[
            ...         ("https://grafana.example.com/d/db", "DB Dashboard"),
            ...     ],
            ... )
        """
        from packages.enterprise.notifications.types import NotificationLevel

        # Map severity to notification level
        level_map = {
            IncidentSeverity.CRITICAL: NotificationLevel.CRITICAL,
            IncidentSeverity.HIGH: NotificationLevel.ERROR,
            IncidentSeverity.MEDIUM: NotificationLevel.WARNING,
            IncidentSeverity.LOW: NotificationLevel.INFO,
            IncidentSeverity.INFO: NotificationLevel.DEBUG,
        }
        level = level_map.get(severity, NotificationLevel.WARNING)

        payload = NotificationPayload(
            message=message,
            level=level,
            title=title,
        )

        details = IncidentDetails(
            dedup_key=dedup_key,
            severity=severity,
            action=IncidentAction.TRIGGER,
            source=source,
            component=component,
            group=group,
            class_type=class_type,
            custom_details=tuple((custom_details or {}).items()),
            links=tuple(links or []),
        )

        return await self.trigger(payload, incident_details=details)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_pagerduty_handler(
    routing_key: str,
    *,
    source: str | None = None,
    component: str | None = None,
    client_name: str | None = None,
    client_url: str | None = None,
    name: str | None = None,
) -> PagerDutyNotificationHandler:
    """Create a PagerDuty handler with common defaults.

    Convenience function for quickly creating a configured handler.

    Args:
        routing_key: PagerDuty routing/integration key
        source: Default source for alerts
        component: Default component for alerts
        client_name: Client name for links
        client_url: Client URL for links
        name: Handler name

    Returns:
        Configured PagerDutyNotificationHandler

    Example:
        >>> handler = create_pagerduty_handler(
        ...     routing_key="abcd1234abcd1234abcd1234abcd1234",
        ...     source="data-quality",
        ...     client_name="Truthound",
        ... )
    """
    config = PagerDutyConfig(routing_key=routing_key)

    if source or component:
        config = config.with_defaults(source=source, component=component)

    if client_name or client_url:
        config = config.with_client(
            name=client_name or "Truthound",
            url=client_url,
        )

    return PagerDutyNotificationHandler(config=config, name=name)
