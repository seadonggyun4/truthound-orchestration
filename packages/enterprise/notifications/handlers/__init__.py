"""Notification handlers.

This module provides the handler implementations for various notification channels.
All handlers implement the NotificationHandler protocol.

Handlers:
    - SlackNotificationHandler: Send notifications to Slack via webhooks
    - WebhookNotificationHandler: Send notifications to generic webhooks
    - EmailNotificationHandler: Send notifications via email (SMTP/API)
    - PagerDutyNotificationHandler: Send alerts to PagerDuty via Events API v2
    - OpsgenieNotificationHandler: Send alerts to Opsgenie via Alert API v2

Incident Management:
    - IncidentManagementHandler: Abstract base for incident management services
    - IncidentDetails: Structured incident details for alert creation
    - IncidentSeverity: Standardized severity levels
    - IncidentAction: Standard incident actions (trigger, acknowledge, resolve)

Example:
    >>> from packages.enterprise.notifications.handlers import (
    ...     SlackNotificationHandler,
    ...     WebhookNotificationHandler,
    ...     EmailNotificationHandler,
    ...     PagerDutyNotificationHandler,
    ...     OpsgenieNotificationHandler,
    ... )
    >>>
    >>> slack = SlackNotificationHandler(
    ...     webhook_url="https://hooks.slack.com/services/...",
    ... )
    >>> result = await slack.send(payload)
    >>>
    >>> email = EmailNotificationHandler(
    ...     from_address="alerts@example.com",
    ...     smtp_host="smtp.example.com",
    ... )
    >>> result = await email.send(payload)
    >>>
    >>> pagerduty = PagerDutyNotificationHandler(
    ...     routing_key="your-32-char-routing-key-here",
    ... )
    >>> result = await pagerduty.send(payload)
    >>>
    >>> opsgenie = OpsgenieNotificationHandler(
    ...     api_key="your-opsgenie-api-key",
    ... )
    >>> result = await opsgenie.send(payload)
"""

from packages.enterprise.notifications.handlers.base import (
    AsyncNotificationHandler,
    BaseNotificationHandler,
    NotificationHandler,
    SyncNotificationHandler,
)
from packages.enterprise.notifications.handlers.email import (
    EmailNotificationHandler,
)
from packages.enterprise.notifications.handlers.incident import (
    IncidentAction,
    IncidentDetails,
    IncidentManagementHandler,
    IncidentSeverity,
    create_incident_payload,
)
from packages.enterprise.notifications.handlers.opsgenie import (
    OpsgenieNotificationHandler,
    create_opsgenie_handler,
)
from packages.enterprise.notifications.handlers.pagerduty import (
    PagerDutyNotificationHandler,
    create_pagerduty_handler,
)
from packages.enterprise.notifications.handlers.slack import (
    SlackNotificationHandler,
)
from packages.enterprise.notifications.handlers.webhook import (
    WebhookNotificationHandler,
)

__all__ = [
    # Base classes and protocols
    "NotificationHandler",
    "AsyncNotificationHandler",
    "SyncNotificationHandler",
    "BaseNotificationHandler",
    # Channel handlers
    "SlackNotificationHandler",
    "WebhookNotificationHandler",
    "EmailNotificationHandler",
    "PagerDutyNotificationHandler",
    "OpsgenieNotificationHandler",
    # Incident management
    "IncidentManagementHandler",
    "IncidentDetails",
    "IncidentSeverity",
    "IncidentAction",
    # Factory functions
    "create_pagerduty_handler",
    "create_opsgenie_handler",
    "create_incident_payload",
]
