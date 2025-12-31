"""Opsgenie notification handler.

This module provides the Opsgenie notification handler that sends alerts
to Opsgenie via the Alert API v2.

Features:
    - Alert API v2 integration (create, acknowledge, close, snooze)
    - Priority mapping from NotificationLevel (P1-P5)
    - Alias-based deduplication for alert correlation
    - Team/user responder support
    - Rich payload with tags, details, and custom actions
    - US and EU region support
    - Retry with exponential backoff

Example:
    >>> from packages.enterprise.notifications.handlers import OpsgenieNotificationHandler
    >>> from packages.enterprise.notifications.config import OpsgenieConfig, OpsgenieResponder
    >>>
    >>> handler = OpsgenieNotificationHandler(
    ...     config=OpsgenieConfig(
    ...         api_key="your-opsgenie-api-key",
    ...         default_responders=(
    ...             OpsgenieResponder(name="ops-team", type="team"),
    ...         ),
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
    >>> await handler.acknowledge(alias="alert-123")
    >>>
    >>> # Close an alert
    >>> await handler.close(alias="alert-123")

API Reference:
    Opsgenie Alert API v2:
    https://docs.opsgenie.com/docs/alert-api
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from packages.enterprise.notifications.config import (
    NotificationConfig,
    OpsgenieConfig,
    OpsgeniePriority,
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


# Opsgenie priority mapping
OPSGENIE_PRIORITY_MAP: dict[IncidentSeverity, OpsgeniePriority] = {
    IncidentSeverity.CRITICAL: OpsgeniePriority.P1,
    IncidentSeverity.HIGH: OpsgeniePriority.P2,
    IncidentSeverity.MEDIUM: OpsgeniePriority.P3,
    IncidentSeverity.LOW: OpsgeniePriority.P4,
    IncidentSeverity.INFO: OpsgeniePriority.P5,
}

# Opsgenie action mapping (API endpoints)
OPSGENIE_ACTION_ENDPOINTS: dict[IncidentAction, str] = {
    IncidentAction.TRIGGER: "",  # POST /v2/alerts
    IncidentAction.ACKNOWLEDGE: "/acknowledge",  # POST /v2/alerts/{id}/acknowledge
    IncidentAction.RESOLVE: "/close",  # POST /v2/alerts/{id}/close
    IncidentAction.SNOOZE: "/snooze",  # POST /v2/alerts/{id}/snooze
    IncidentAction.ADD_NOTE: "/notes",  # POST /v2/alerts/{id}/notes
}


class OpsgenieNotificationHandler(IncidentManagementHandler):
    """Opsgenie notification handler using Alert API v2.

    Sends alerts to Opsgenie with support for creating, acknowledging,
    and closing alerts. Uses aliases for alert correlation/deduplication.

    Attributes:
        opsgenie_config: Opsgenie-specific configuration

    Example:
        >>> handler = OpsgenieNotificationHandler(
        ...     config=OpsgenieConfig(
        ...         api_key="your-api-key",
        ...         default_priority=OpsgeniePriority.P2,
        ...     ),
        ... )
        >>>
        >>> # Create an alert
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
        ...     dedup_key="db-connection-prod-01",  # Used as alias
        ...     severity=IncidentSeverity.CRITICAL,
        ...     source="database-monitor",
        ...     tags=frozenset(["database", "production"]),
        ... )
        >>> result = await handler.trigger(payload, incident_details=details)
    """

    def __init__(
        self,
        config: OpsgenieConfig | None = None,
        *,
        api_key: str | None = None,
        name: str | None = None,
        base_config: NotificationConfig | None = None,
        hooks: list[NotificationHook] | None = None,
    ) -> None:
        """Initialize the Opsgenie handler.

        Args:
            config: Complete Opsgenie configuration
            api_key: Opsgenie API key (alternative to config)
            name: Handler name
            base_config: Base notification config (used if config not provided)
            hooks: Notification hooks

        Raises:
            ValueError: If neither config nor api_key is provided
        """
        if config is not None:
            self._opsgenie_config = config
            base = config.base_config
        elif api_key is not None:
            self._opsgenie_config = OpsgenieConfig(
                api_key=api_key,
                base_config=base_config or NotificationConfig(),
            )
            base = self._opsgenie_config.base_config
        else:
            raise ValueError("Either config or api_key must be provided")

        super().__init__(
            name=name or "opsgenie",
            config=base,
            hooks=hooks,
            default_source=self._opsgenie_config.default_source,
            auto_dedup=True,
        )

    @property
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        return NotificationChannel.OPSGENIE

    @property
    def opsgenie_config(self) -> OpsgenieConfig:
        """Get the Opsgenie-specific configuration."""
        return self._opsgenie_config

    def _map_severity(self, severity: IncidentSeverity) -> str:
        """Map standard severity to Opsgenie priority.

        Args:
            severity: Standard incident severity

        Returns:
            Opsgenie priority string (P1-P5)
        """
        priority = OPSGENIE_PRIORITY_MAP.get(severity, self._opsgenie_config.default_priority)
        return priority.value

    def _map_action_endpoint(self, action: IncidentAction) -> str:
        """Get the API endpoint suffix for an action.

        Args:
            action: Standard incident action

        Returns:
            API endpoint suffix
        """
        return OPSGENIE_ACTION_ENDPOINTS.get(action, "")

    def _build_request_body(
        self,
        payload: NotificationPayload,
        incident_details: IncidentDetails,
    ) -> dict[str, Any]:
        """Build the Opsgenie Alert API v2 request body.

        Creates a properly formatted request body following the
        Opsgenie Alert API v2 specification.

        Args:
            payload: The notification payload
            incident_details: Extracted incident details

        Returns:
            Opsgenie API request body

        Reference:
            https://docs.opsgenie.com/docs/alert-api#create-alert
        """
        action = incident_details.action

        # Build alias with optional prefix (used as dedup key)
        alias = incident_details.dedup_key or self._generate_dedup_key(payload)
        if self._opsgenie_config.default_alias_prefix:
            alias = f"{self._opsgenie_config.default_alias_prefix}{alias}"

        # For non-trigger actions, we need a simpler body
        if action != IncidentAction.TRIGGER:
            return self._build_action_body(action, alias, payload)

        # Build full alert creation body
        body: dict[str, Any] = {
            "message": (payload.title or payload.message)[:130],  # Max 130 chars
            "alias": alias,
            "priority": self._map_severity(incident_details.severity),
            "source": incident_details.source or self._opsgenie_config.default_source,
        }

        # Add description (full message if different from summary)
        if payload.message and payload.message != body["message"]:
            body["description"] = payload.message[:15000]  # Max 15000 chars

        # Add responders
        responders = list(self._opsgenie_config.default_responders)
        if incident_details.responders:
            # Add additional responders from incident details
            for responder in incident_details.responders:
                responders.append({"type": "user", "username": responder})
        if responders:
            body["responders"] = [r.to_dict() if hasattr(r, "to_dict") else r for r in responders]

        # Add visible to
        if self._opsgenie_config.default_visible_to:
            body["visibleTo"] = [
                r.to_dict() for r in self._opsgenie_config.default_visible_to
            ]

        # Add tags
        tags = set(self._opsgenie_config.default_tags)
        tags.update(incident_details.tags)
        tags.update(payload.metadata.tags)
        if tags:
            body["tags"] = list(tags)[:20]  # Max 20 tags

        # Add entity
        if incident_details.component or self._opsgenie_config.default_entity:
            body["entity"] = incident_details.component or self._opsgenie_config.default_entity

        # Add custom actions
        if self._opsgenie_config.default_actions:
            body["actions"] = list(self._opsgenie_config.default_actions)[:10]  # Max 10 actions

        # Build details
        details: dict[str, str] = dict(self._opsgenie_config.default_details)

        # Add notification context if configured
        if self._opsgenie_config.include_context_in_details:
            context = payload.context_dict
            excluded_keys = {
                "dedup_key", "action", "source", "component", "group", "class_type"
            }
            for key, value in context.items():
                if key not in excluded_keys:
                    details[key] = str(value)[:8000]  # Max 8000 chars per value

        # Add explicit custom details from incident_details
        for key, value in incident_details.custom_details_dict.items():
            details[key] = str(value)[:8000]

        # Add metadata
        if payload.metadata.correlation_id:
            details["correlation_id"] = payload.metadata.correlation_id
        if payload.metadata.trace_id:
            details["trace_id"] = payload.metadata.trace_id
        if incident_details.group:
            details["group"] = incident_details.group
        if incident_details.class_type:
            details["class"] = incident_details.class_type

        if details:
            body["details"] = details

        return body

    def _build_action_body(
        self,
        action: IncidentAction,
        alias: str,
        payload: NotificationPayload,
    ) -> dict[str, Any]:
        """Build request body for non-trigger actions.

        Args:
            action: The action to perform
            alias: Alert alias
            payload: The notification payload

        Returns:
            Action request body
        """
        body: dict[str, Any] = {}

        # Add note for acknowledge/resolve actions
        if action in {IncidentAction.ACKNOWLEDGE, IncidentAction.RESOLVE}:
            body["note"] = payload.message[:25000] if payload.message else None

        # Add source
        body["source"] = self._opsgenie_config.default_source

        # Store alias for URL building (will be used by _do_send)
        body["_alias"] = alias
        body["_action"] = action.value

        return body

    def _get_api_url(self, action: IncidentAction, alias: str | None = None) -> str:
        """Get the API URL for an action.

        Args:
            action: The action to perform
            alias: Alert alias (required for non-trigger actions)

        Returns:
            Full API URL
        """
        base_url = self._opsgenie_config.api_url

        if action == IncidentAction.TRIGGER:
            return base_url

        # For other actions, we need to include the alias in the URL
        endpoint = self._map_action_endpoint(action)
        return f"{base_url}/{alias}{endpoint}?identifierType=alias"

    async def _do_send(
        self,
        payload: NotificationPayload,
        formatted_message: str | dict[str, Any],
    ) -> NotificationResult:
        """Send the notification to Opsgenie.

        Uses aiohttp for async HTTP requests with retry support.

        Args:
            payload: The original notification payload
            formatted_message: The formatted Opsgenie request body

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
                message="Invalid message format for Opsgenie",
                handler_name=self.name,
                channel=self.channel,
            )

        # Extract action and alias from body if present
        action_str = formatted_message.pop("_action", "trigger")
        alias = formatted_message.pop("_alias", None)
        action = IncidentAction(action_str)

        # Get appropriate URL
        api_url = self._get_api_url(action, alias)

        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"GenieKey {self._opsgenie_config.api_key}",
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    api_url,
                    json=formatted_message,
                    headers=headers,
                ) as response:
                    response_body = await response.text()

                    # Opsgenie returns 202 Accepted for successful operations
                    if response.status in {200, 201, 202}:
                        request_id = None
                        alert_id = None
                        try:
                            response_json = json.loads(response_body)
                            request_id = response_json.get("requestId")
                            if "data" in response_json:
                                alert_id = response_json["data"].get("alertId")
                        except (json.JSONDecodeError, TypeError):
                            pass

                        return NotificationResult.success_result(
                            channel=self.channel,
                            handler_name=self.name,
                            message_id=alert_id or request_id,
                            response_data={
                                "status": response.status,
                                "request_id": request_id,
                                "alert_id": alert_id,
                            },
                        )
                    else:
                        # Parse error response
                        error_msg = f"Opsgenie API error: {response.status}"
                        try:
                            error_json = json.loads(response_body)
                            if "message" in error_json:
                                error_msg = f"Opsgenie API error: {error_json['message']}"
                            elif "errors" in error_json:
                                error_msg = f"Opsgenie API error: {error_json['errors']}"
                        except (json.JSONDecodeError, TypeError):
                            error_msg = f"Opsgenie API error: {response_body[:200]}"

                        return NotificationResult.failure_result(
                            channel=self.channel,
                            handler_name=self.name,
                            error=error_msg,
                            error_type="OpsgenieAPIError",
                        )

        except asyncio.TimeoutError as e:
            raise NotificationTimeoutError(
                message="Opsgenie notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except aiohttp.ClientError as e:
            raise NotificationSendError(
                message=f"Failed to send Opsgenie notification: {e}",
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
            formatted_message: The formatted Opsgenie request body

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
            formatted_message: The formatted Opsgenie request body

        Returns:
            Result of the send operation
        """
        try:
            import requests

            # Extract action and alias from body if present
            action_str = formatted_message.pop("_action", "trigger")
            alias = formatted_message.pop("_alias", None)
            action = IncidentAction(action_str)

            # Get appropriate URL
            api_url = self._get_api_url(action, alias)

            response = requests.post(
                api_url,
                json=formatted_message,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"GenieKey {self._opsgenie_config.api_key}",
                },
                timeout=self._config.timeout_seconds,
            )

            if response.status_code in {200, 201, 202}:
                request_id = None
                alert_id = None
                try:
                    response_json = response.json()
                    request_id = response_json.get("requestId")
                    if "data" in response_json:
                        alert_id = response_json["data"].get("alertId")
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

                return NotificationResult.success_result(
                    channel=self.channel,
                    handler_name=self.name,
                    message_id=alert_id or request_id,
                    response_data={
                        "status": response.status_code,
                        "request_id": request_id,
                        "alert_id": alert_id,
                    },
                )
            else:
                error_msg = f"Opsgenie API error: {response.status_code}"
                try:
                    error_json = response.json()
                    if "message" in error_json:
                        error_msg = f"Opsgenie API error: {error_json['message']}"
                    elif "errors" in error_json:
                        error_msg = f"Opsgenie API error: {error_json['errors']}"
                except (json.JSONDecodeError, TypeError, ValueError):
                    error_msg = f"Opsgenie API error: {response.text[:200]}"

                return NotificationResult.failure_result(
                    channel=self.channel,
                    handler_name=self.name,
                    error=error_msg,
                    error_type="OpsgenieAPIError",
                )

        except requests.Timeout as e:
            raise NotificationTimeoutError(
                message="Opsgenie notification timed out",
                handler_name=self.name,
                channel=self.channel,
                timeout_seconds=self._config.timeout_seconds,
            ) from e
        except requests.RequestException as e:
            raise NotificationSendError(
                message=f"Failed to send Opsgenie notification: {e}",
                handler_name=self.name,
                channel=self.channel,
                original_error=e,
            ) from e

    async def close(
        self,
        alias: str,
        message: str | None = None,
    ) -> NotificationResult:
        """Close an existing alert.

        Opsgenie uses "close" instead of "resolve" in its API.

        Args:
            alias: The alias of the alert to close
            message: Optional closing note

        Returns:
            Result of the close operation
        """
        # Use the parent's resolve method which will map to close
        return await self.resolve(alias, message)

    async def snooze(
        self,
        alias: str,
        end_time: datetime,
        message: str | None = None,
    ) -> NotificationResult:
        """Snooze an existing alert.

        Args:
            alias: The alias of the alert to snooze
            end_time: When the snooze should end
            message: Optional snooze note

        Returns:
            Result of the snooze operation
        """
        from packages.enterprise.notifications.types import NotificationLevel

        payload = NotificationPayload(
            message=message or f"Alert snoozed until {end_time.isoformat()}",
            level=NotificationLevel.INFO,
        ).with_context(
            dedup_key=alias,
            action=IncidentAction.SNOOZE.value,
            snooze_end_time=end_time.isoformat(),
        )
        return await self.send(payload)

    async def add_note(
        self,
        alias: str,
        note: str,
    ) -> NotificationResult:
        """Add a note to an existing alert.

        Args:
            alias: The alias of the alert
            note: The note to add

        Returns:
            Result of the add_note operation
        """
        from packages.enterprise.notifications.types import NotificationLevel

        payload = NotificationPayload(
            message=note,
            level=NotificationLevel.INFO,
        ).with_context(
            dedup_key=alias,
            action=IncidentAction.ADD_NOTE.value,
        )
        return await self.send(payload)

    async def create_alert(
        self,
        message: str,
        *,
        description: str | None = None,
        priority: OpsgeniePriority | None = None,
        alias: str | None = None,
        source: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        details: dict[str, str] | None = None,
        responders: list[dict[str, str]] | None = None,
    ) -> NotificationResult:
        """Create an Opsgenie alert with explicit parameters.

        Convenience method for creating alerts with full control over
        all Opsgenie-specific fields.

        Args:
            message: Alert message (max 130 chars)
            description: Full alert description (max 15000 chars)
            priority: Alert priority (P1-P5)
            alias: Alert alias for deduplication
            source: Source of the alert
            entity: Entity the alert relates to
            tags: List of tags
            details: Additional custom details
            responders: List of responder dicts with type and id/name

        Returns:
            Result of the create operation

        Example:
            >>> result = await handler.create_alert(
            ...     message="Database connection pool exhausted",
            ...     description="All connections are in use",
            ...     priority=OpsgeniePriority.P1,
            ...     source="connection-monitor",
            ...     tags=["database", "critical"],
            ...     details={"pool_size": "100", "active": "100"},
            ... )
        """
        from packages.enterprise.notifications.types import NotificationLevel

        # Map priority to notification level
        priority_level_map = {
            OpsgeniePriority.P1: NotificationLevel.CRITICAL,
            OpsgeniePriority.P2: NotificationLevel.ERROR,
            OpsgeniePriority.P3: NotificationLevel.WARNING,
            OpsgeniePriority.P4: NotificationLevel.INFO,
            OpsgeniePriority.P5: NotificationLevel.DEBUG,
        }
        level = priority_level_map.get(
            priority or self._opsgenie_config.default_priority,
            NotificationLevel.WARNING,
        )

        # Map priority to severity
        priority_severity_map = {
            OpsgeniePriority.P1: IncidentSeverity.CRITICAL,
            OpsgeniePriority.P2: IncidentSeverity.HIGH,
            OpsgeniePriority.P3: IncidentSeverity.MEDIUM,
            OpsgeniePriority.P4: IncidentSeverity.LOW,
            OpsgeniePriority.P5: IncidentSeverity.INFO,
        }
        severity = priority_severity_map.get(
            priority or self._opsgenie_config.default_priority,
            IncidentSeverity.MEDIUM,
        )

        payload = NotificationPayload(
            message=description or message,
            level=level,
            title=message,
        )

        incident_details = IncidentDetails(
            dedup_key=alias,
            severity=severity,
            action=IncidentAction.TRIGGER,
            source=source,
            component=entity,
            custom_details=tuple((details or {}).items()),
            tags=frozenset(tags or []),
            responders=tuple(
                r.get("name") or r.get("id", "") for r in (responders or [])
            ),
        )

        return await self.trigger(payload, incident_details=incident_details)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_opsgenie_handler(
    api_key: str,
    *,
    team: str | None = None,
    source: str | None = None,
    priority: OpsgeniePriority | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
) -> OpsgenieNotificationHandler:
    """Create an Opsgenie handler with common defaults.

    Convenience function for quickly creating a configured handler.

    Args:
        api_key: Opsgenie API key
        team: Default team to notify
        source: Default source for alerts
        priority: Default priority level
        tags: Default tags for alerts
        name: Handler name

    Returns:
        Configured OpsgenieNotificationHandler

    Example:
        >>> handler = create_opsgenie_handler(
        ...     api_key="your-api-key",
        ...     team="platform-team",
        ...     source="data-quality",
        ...     priority=OpsgeniePriority.P2,
        ... )
    """
    config = OpsgenieConfig(api_key=api_key)

    if priority:
        config = config.with_priority(priority)

    if source or tags:
        config = config.with_defaults(source=source, tags=tags)

    if team:
        config = config.with_team(team)

    return OpsgenieNotificationHandler(config=config, name=name)
