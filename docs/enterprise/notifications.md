---
title: Notifications
---

# Notifications

Multi-channel notification system.

## NotificationRegistry

Notification registry:

```python
from packages.enterprise.notifications import NotificationRegistry

registry = NotificationRegistry()

# Send notification
registry.notify(
    channel="slack",
    message="Data quality check failed",
    severity="warning",
)
```

## Supported Channels

### Slack

```python
from packages.enterprise.notifications.handlers import SlackHandler

handler = SlackHandler(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#data-quality",
)

handler.send(
    message="Data quality check failed",
    severity="warning",
    details={"table": "orders", "failed_count": 10},
)
```

### Email

```python
from packages.enterprise.notifications.handlers import EmailHandler

handler = EmailHandler(
    smtp_host="smtp.example.com",
    smtp_port=587,
    username="user@example.com",
    password="password",
    from_address="alerts@example.com",
    to_addresses=["team@example.com"],
)

handler.send(
    subject="Data Quality Alert",
    message="Data quality check failed for orders table",
)
```

### Webhook

```python
from packages.enterprise.notifications.handlers import WebhookHandler

handler = WebhookHandler(
    url="https://api.example.com/webhooks/alerts",
    headers={"Authorization": "Bearer token"},
)

handler.send(
    message="Data quality check failed",
    payload={"table": "orders", "status": "failed"},
)
```

### PagerDuty

```python
from packages.enterprise.notifications.handlers import PagerDutyHandler

handler = PagerDutyHandler(
    routing_key="your-routing-key",
    severity="critical",
)

handler.send(
    summary="Data quality check failed",
    source="truthound-orchestration",
    details={"table": "orders"},
)
```

### Opsgenie

```python
from packages.enterprise.notifications.handlers import OpsgenieHandler

handler = OpsgenieHandler(
    api_key="your-api-key",
    priority="P2",
)

handler.send(
    message="Data quality check failed",
    alias="data-quality-orders",
    description="10 validation failures in orders table",
)
```

## Formatters

### Text

```python
from packages.enterprise.notifications.formatters import TextFormatter

formatter = TextFormatter()
message = formatter.format(check_result)
```

### Markdown

```python
from packages.enterprise.notifications.formatters import MarkdownFormatter

formatter = MarkdownFormatter()
message = formatter.format(check_result)
```

### SlackBlock

```python
from packages.enterprise.notifications.formatters import SlackBlockFormatter

formatter = SlackBlockFormatter()
blocks = formatter.format(check_result)
```

### JSON

```python
from packages.enterprise.notifications.formatters import JSONFormatter

formatter = JSONFormatter()
json_message = formatter.format(check_result)
```

## Notification Hooks

Notification lifecycle events:

```python
from packages.enterprise.notifications import NotificationHook

class MyNotificationHook(NotificationHook):
    def before_send(self, channel, message, severity):
        # Pre-send processing
        pass

    def after_send(self, channel, message, severity, success):
        # Post-send processing
        pass

    def on_error(self, channel, message, severity, error):
        # Error handling
        pass
```

## Multi-Channel Notifications

Send to multiple channels simultaneously:

```python
from packages.enterprise.notifications import NotificationRegistry

registry = NotificationRegistry()

# Register handlers
registry.register("slack", slack_handler)
registry.register("email", email_handler)
registry.register("pagerduty", pagerduty_handler)

# Send to multiple channels
registry.notify_all(
    channels=["slack", "email"],
    message="Data quality check failed",
    severity="warning",
)
```

## Severity-Based Routing

```python
from packages.enterprise.notifications import SeverityRouter

router = SeverityRouter()
router.add_route("info", ["slack"])
router.add_route("warning", ["slack", "email"])
router.add_route("critical", ["slack", "email", "pagerduty"])

# Route to appropriate channels based on severity
router.route(
    message="Critical data quality failure",
    severity="critical",
)
```

## Retry

Retry on notification send failure:

```python
from packages.enterprise.notifications import RetryingHandler

handler = RetryingHandler(
    base_handler=slack_handler,
    max_retries=3,
    retry_delay_seconds=5,
)

handler.send(message="Alert")
```

## Batch Notifications

Batch multiple notifications together:

```python
from packages.enterprise.notifications import BatchingHandler

handler = BatchingHandler(
    base_handler=email_handler,
    batch_size=10,
    batch_interval_seconds=60,
)

# Notifications are batched and sent together
handler.send(message="Alert 1")
handler.send(message="Alert 2")
# Batch sent after 60 seconds or when 10 messages accumulate
```
