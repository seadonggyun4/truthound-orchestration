"""Abstract incident management handler for PagerDuty, Opsgenie, etc.

This module provides a shared base class for incident management services.
These services share common concepts like:
    - Alert/Incident severity levels
    - Deduplication keys
    - Routing/Escalation
    - Acknowledgment and resolution

The abstract base class handles common functionality while allowing
specific implementations to handle service-specific API details.

Example:
    >>> class MyIncidentHandler(IncidentManagementHandler):
    ...     @property
    ...     def channel(self) -> NotificationChannel:
    ...         return NotificationChannel.CUSTOM
    ...
    ...     async def _do_send(self, payload, formatted_message):
    ...         # Service-specific implementation
    ...         ...

Design Decisions:
    - Uses composition over inheritance for service-specific configs
    - Provides standard severity mapping with service-specific overrides
    - Supports both event-based and alert-based APIs
    - Enables deduplication for alert storm prevention
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from packages.enterprise.notifications.config import NotificationConfig
from packages.enterprise.notifications.handlers.base import AsyncNotificationHandler
from packages.enterprise.notifications.types import (
    NotificationChannel,
    NotificationLevel,
    NotificationPayload,
    NotificationResult,
)

if TYPE_CHECKING:
    from packages.enterprise.notifications.hooks import NotificationHook


class IncidentSeverity(str, Enum):
    """Standardized incident severity levels.

    Maps to service-specific severity levels:
        - PagerDuty: critical, error, warning, info
        - Opsgenie: P1, P2, P3, P4, P5
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_notification_level(cls, level: NotificationLevel) -> "IncidentSeverity":
        """Convert NotificationLevel to IncidentSeverity."""
        mapping = {
            NotificationLevel.CRITICAL: cls.CRITICAL,
            NotificationLevel.ERROR: cls.HIGH,
            NotificationLevel.WARNING: cls.MEDIUM,
            NotificationLevel.INFO: cls.LOW,
            NotificationLevel.DEBUG: cls.INFO,
        }
        return mapping.get(level, cls.MEDIUM)


class IncidentAction(str, Enum):
    """Actions that can be performed on incidents.

    Standard actions supported by most incident management services.
    """

    TRIGGER = "trigger"  # Create new incident/alert
    ACKNOWLEDGE = "acknowledge"  # Acknowledge the incident
    RESOLVE = "resolve"  # Resolve/close the incident
    SNOOZE = "snooze"  # Temporarily suppress (Opsgenie)
    ESCALATE = "escalate"  # Escalate to next level
    ADD_NOTE = "add_note"  # Add note/annotation

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class IncidentDetails:
    """Structured incident details for incident management services.

    Provides a standardized way to specify incident-related information
    that can be converted to service-specific formats.

    Attributes:
        dedup_key: Key for deduplication (prevents duplicate alerts)
        severity: Incident severity level
        action: Action to perform (trigger, acknowledge, resolve)
        source: Source/component that generated the incident
        component: Affected component or service
        group: Logical grouping for the incident
        class_type: Classification of the incident
        custom_details: Additional service-specific details
        links: Related links (URLs)
        images: Related images (URLs)
        tags: Tags for categorization
        responders: Explicit responders (Opsgenie)
        escalation_policy: Escalation policy ID (PagerDuty)
    """

    dedup_key: str | None = None
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    action: IncidentAction = IncidentAction.TRIGGER
    source: str | None = None
    component: str | None = None
    group: str | None = None
    class_type: str | None = None
    custom_details: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    links: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (url, text)
    images: tuple[str, ...] = field(default_factory=tuple)
    tags: frozenset[str] = field(default_factory=frozenset)
    responders: tuple[str, ...] = field(default_factory=tuple)
    escalation_policy: str | None = None

    @property
    def custom_details_dict(self) -> dict[str, Any]:
        """Get custom details as a dictionary."""
        return dict(self.custom_details)

    @property
    def links_list(self) -> list[dict[str, str]]:
        """Get links as a list of dictionaries."""
        return [{"href": url, "text": text} for url, text in self.links]

    def with_dedup_key(self, dedup_key: str) -> "IncidentDetails":
        """Create a copy with a deduplication key."""
        return IncidentDetails(
            dedup_key=dedup_key,
            severity=self.severity,
            action=self.action,
            source=self.source,
            component=self.component,
            group=self.group,
            class_type=self.class_type,
            custom_details=self.custom_details,
            links=self.links,
            images=self.images,
            tags=self.tags,
            responders=self.responders,
            escalation_policy=self.escalation_policy,
        )

    def with_severity(self, severity: IncidentSeverity) -> "IncidentDetails":
        """Create a copy with a different severity."""
        return IncidentDetails(
            dedup_key=self.dedup_key,
            severity=severity,
            action=self.action,
            source=self.source,
            component=self.component,
            group=self.group,
            class_type=self.class_type,
            custom_details=self.custom_details,
            links=self.links,
            images=self.images,
            tags=self.tags,
            responders=self.responders,
            escalation_policy=self.escalation_policy,
        )

    def with_action(self, action: IncidentAction) -> "IncidentDetails":
        """Create a copy with a different action."""
        return IncidentDetails(
            dedup_key=self.dedup_key,
            severity=self.severity,
            action=action,
            source=self.source,
            component=self.component,
            group=self.group,
            class_type=self.class_type,
            custom_details=self.custom_details,
            links=self.links,
            images=self.images,
            tags=self.tags,
            responders=self.responders,
            escalation_policy=self.escalation_policy,
        )

    def with_source(
        self,
        source: str,
        component: str | None = None,
        group: str | None = None,
    ) -> "IncidentDetails":
        """Create a copy with source information."""
        return IncidentDetails(
            dedup_key=self.dedup_key,
            severity=self.severity,
            action=self.action,
            source=source,
            component=component or self.component,
            group=group or self.group,
            class_type=self.class_type,
            custom_details=self.custom_details,
            links=self.links,
            images=self.images,
            tags=self.tags,
            responders=self.responders,
            escalation_policy=self.escalation_policy,
        )

    def with_custom_detail(self, key: str, value: Any) -> "IncidentDetails":
        """Create a copy with an additional custom detail."""
        return IncidentDetails(
            dedup_key=self.dedup_key,
            severity=self.severity,
            action=self.action,
            source=self.source,
            component=self.component,
            group=self.group,
            class_type=self.class_type,
            custom_details=self.custom_details + ((key, value),),
            links=self.links,
            images=self.images,
            tags=self.tags,
            responders=self.responders,
            escalation_policy=self.escalation_policy,
        )

    def with_link(self, url: str, text: str) -> "IncidentDetails":
        """Create a copy with an additional link."""
        return IncidentDetails(
            dedup_key=self.dedup_key,
            severity=self.severity,
            action=self.action,
            source=self.source,
            component=self.component,
            group=self.group,
            class_type=self.class_type,
            custom_details=self.custom_details,
            links=self.links + ((url, text),),
            images=self.images,
            tags=self.tags,
            responders=self.responders,
            escalation_policy=self.escalation_policy,
        )

    def with_tags(self, *tags: str) -> "IncidentDetails":
        """Create a copy with additional tags."""
        return IncidentDetails(
            dedup_key=self.dedup_key,
            severity=self.severity,
            action=self.action,
            source=self.source,
            component=self.component,
            group=self.group,
            class_type=self.class_type,
            custom_details=self.custom_details,
            links=self.links,
            images=self.images,
            tags=self.tags | frozenset(tags),
            responders=self.responders,
            escalation_policy=self.escalation_policy,
        )

    def with_responders(self, *responders: str) -> "IncidentDetails":
        """Create a copy with specified responders."""
        return IncidentDetails(
            dedup_key=self.dedup_key,
            severity=self.severity,
            action=self.action,
            source=self.source,
            component=self.component,
            group=self.group,
            class_type=self.class_type,
            custom_details=self.custom_details,
            links=self.links,
            images=self.images,
            tags=self.tags,
            responders=self.responders + responders,
            escalation_policy=self.escalation_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dedup_key": self.dedup_key,
            "severity": self.severity.value,
            "action": self.action.value,
            "source": self.source,
            "component": self.component,
            "group": self.group,
            "class_type": self.class_type,
            "custom_details": dict(self.custom_details),
            "links": self.links_list,
            "images": list(self.images),
            "tags": list(self.tags),
            "responders": list(self.responders),
            "escalation_policy": self.escalation_policy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IncidentDetails":
        """Create from dictionary."""
        links = tuple(
            (link.get("href", ""), link.get("text", ""))
            for link in data.get("links", [])
        )
        return cls(
            dedup_key=data.get("dedup_key"),
            severity=IncidentSeverity(data.get("severity", "medium")),
            action=IncidentAction(data.get("action", "trigger")),
            source=data.get("source"),
            component=data.get("component"),
            group=data.get("group"),
            class_type=data.get("class_type"),
            custom_details=tuple(data.get("custom_details", {}).items()),
            links=links,
            images=tuple(data.get("images", [])),
            tags=frozenset(data.get("tags", [])),
            responders=tuple(data.get("responders", [])),
            escalation_policy=data.get("escalation_policy"),
        )

    @classmethod
    def from_payload(cls, payload: NotificationPayload) -> "IncidentDetails":
        """Create IncidentDetails from a NotificationPayload.

        Extracts incident-related information from the payload's context
        and metadata.
        """
        context = payload.context_dict
        severity = IncidentSeverity.from_notification_level(payload.level)

        return cls(
            dedup_key=context.get("dedup_key") or payload.metadata.correlation_id,
            severity=severity,
            action=IncidentAction(context.get("action", "trigger")),
            source=context.get("source") or payload.metadata.source,
            component=context.get("component"),
            group=context.get("group"),
            class_type=context.get("class_type"),
            custom_details=tuple(
                (k, v)
                for k, v in context.items()
                if k not in {"dedup_key", "action", "source", "component", "group", "class_type"}
            ),
            tags=payload.metadata.tags,
        )


class IncidentManagementHandler(AsyncNotificationHandler):
    """Abstract base class for incident management handlers.

    Provides common functionality for PagerDuty, Opsgenie, and similar
    incident management services.

    Subclasses must implement:
        - channel property
        - _map_severity method (service-specific severity mapping)
        - _build_request_body method (service-specific payload)
        - _do_send method (actual API call)

    Features:
        - Automatic severity mapping from NotificationLevel
        - Deduplication key generation
        - Incident details extraction from payload
        - Common error handling
    """

    def __init__(
        self,
        name: str | None = None,
        config: NotificationConfig | None = None,
        hooks: list[NotificationHook] | None = None,
        *,
        default_source: str | None = None,
        auto_dedup: bool = True,
    ) -> None:
        """Initialize the incident management handler.

        Args:
            name: Handler name
            config: Base notification config
            hooks: Notification hooks
            default_source: Default source for incidents
            auto_dedup: Whether to auto-generate dedup keys
        """
        super().__init__(name=name, config=config, hooks=hooks)
        self._default_source = default_source or "truthound-orchestration"
        self._auto_dedup = auto_dedup

    @property
    @abstractmethod
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        ...

    @abstractmethod
    def _map_severity(self, severity: IncidentSeverity) -> str:
        """Map standard severity to service-specific severity.

        Args:
            severity: Standard incident severity

        Returns:
            Service-specific severity string
        """
        ...

    @abstractmethod
    def _build_request_body(
        self,
        payload: NotificationPayload,
        incident_details: IncidentDetails,
    ) -> dict[str, Any]:
        """Build the service-specific request body.

        Args:
            payload: The notification payload
            incident_details: Extracted incident details

        Returns:
            Request body dictionary
        """
        ...

    def _generate_dedup_key(self, payload: NotificationPayload) -> str:
        """Generate a deduplication key from the payload.

        Uses a combination of:
            - Source
            - Message hash
            - Correlation ID (if available)

        Args:
            payload: The notification payload

        Returns:
            Deduplication key string
        """
        import hashlib

        components = [
            payload.metadata.source or self._default_source,
            payload.message[:100],  # First 100 chars of message
        ]
        if payload.metadata.correlation_id:
            components.append(payload.metadata.correlation_id)

        combined = "|".join(str(c) for c in components)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    def _extract_incident_details(
        self,
        payload: NotificationPayload,
    ) -> IncidentDetails:
        """Extract incident details from the payload.

        Args:
            payload: The notification payload

        Returns:
            IncidentDetails with extracted information
        """
        details = IncidentDetails.from_payload(payload)

        # Auto-generate dedup key if enabled and not provided
        if self._auto_dedup and not details.dedup_key:
            details = details.with_dedup_key(self._generate_dedup_key(payload))

        # Set default source if not provided
        if not details.source:
            details = details.with_source(self._default_source)

        return details

    async def _format_message(
        self,
        payload: NotificationPayload,
    ) -> dict[str, Any]:
        """Format the message for the incident management service.

        Args:
            payload: The notification payload

        Returns:
            Formatted request body
        """
        incident_details = self._extract_incident_details(payload)
        return self._build_request_body(payload, incident_details)

    async def trigger(
        self,
        payload: NotificationPayload,
        incident_details: IncidentDetails | None = None,
    ) -> NotificationResult:
        """Trigger a new incident/alert.

        Convenience method for creating incidents with explicit details.

        Args:
            payload: The notification payload
            incident_details: Optional explicit incident details

        Returns:
            Result of the trigger operation
        """
        if incident_details:
            # Add incident details to context
            payload = payload.with_context(**incident_details.to_dict())
        return await self.send(payload)

    async def acknowledge(
        self,
        dedup_key: str,
        message: str | None = None,
    ) -> NotificationResult:
        """Acknowledge an existing incident.

        Args:
            dedup_key: The deduplication key of the incident
            message: Optional acknowledgment message

        Returns:
            Result of the acknowledge operation
        """
        payload = NotificationPayload(
            message=message or "Incident acknowledged",
            level=NotificationLevel.INFO,
        ).with_context(
            dedup_key=dedup_key,
            action=IncidentAction.ACKNOWLEDGE.value,
        )
        return await self.send(payload)

    async def resolve(
        self,
        dedup_key: str,
        message: str | None = None,
    ) -> NotificationResult:
        """Resolve an existing incident.

        Args:
            dedup_key: The deduplication key of the incident
            message: Optional resolution message

        Returns:
            Result of the resolve operation
        """
        payload = NotificationPayload(
            message=message or "Incident resolved",
            level=NotificationLevel.INFO,
        ).with_context(
            dedup_key=dedup_key,
            action=IncidentAction.RESOLVE.value,
        )
        return await self.send(payload)


# =============================================================================
# Utility Functions
# =============================================================================


def create_incident_payload(
    message: str,
    *,
    level: NotificationLevel = NotificationLevel.ERROR,
    title: str | None = None,
    dedup_key: str | None = None,
    source: str | None = None,
    component: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    custom_details: dict[str, Any] | None = None,
) -> NotificationPayload:
    """Create a notification payload with incident details.

    Convenience function for creating properly structured incident payloads.

    Args:
        message: The incident message
        level: Notification level (maps to severity)
        title: Optional incident title
        dedup_key: Deduplication key
        source: Source of the incident
        component: Affected component
        group: Logical grouping
        tags: Tags for categorization
        custom_details: Additional custom details

    Returns:
        NotificationPayload configured for incident management

    Example:
        >>> payload = create_incident_payload(
        ...     message="Database connection failed",
        ...     level=NotificationLevel.CRITICAL,
        ...     title="Database Outage",
        ...     source="db-monitor",
        ...     component="postgresql",
        ...     tags=["database", "critical"],
        ... )
        >>> await pagerduty_handler.send(payload)
    """
    context: dict[str, Any] = {}
    if dedup_key:
        context["dedup_key"] = dedup_key
    if source:
        context["source"] = source
    if component:
        context["component"] = component
    if group:
        context["group"] = group
    if custom_details:
        context.update(custom_details)

    payload = NotificationPayload(
        message=message,
        level=level,
        title=title,
        context=tuple(context.items()),
    )

    if tags:
        payload = payload.with_metadata(
            payload.metadata.with_tags(*tags)
        )

    return payload
