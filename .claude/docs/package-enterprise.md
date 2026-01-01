# Enterprise Package Documentation

## Overview

The Enterprise package (`packages/enterprise/`) provides production-grade features for large-scale deployments:

- **Notifications**: Multi-channel notification system (Slack, Email, Webhook, PagerDuty, Opsgenie)
- **Multi-Tenant**: Complete tenant isolation, context management, and quota enforcement
- **Enterprise Engines**: Adapters for Informatica, Talend, IBM InfoSphere, SAP Data Services
- **Secrets**: Secret management with multiple backends (Vault, AWS, GCP, Azure) and audit logging

```
packages/enterprise/
├── notifications/          # Multi-channel notification system
│   ├── types.py           # Types and enums
│   ├── config.py          # Configuration classes
│   ├── exceptions.py      # Exception hierarchy
│   ├── handlers/          # Channel handlers
│   │   ├── base.py       # Handler protocols
│   │   ├── slack.py      # Slack handler
│   │   ├── email.py      # Email handler (SMTP, SendGrid, SES)
│   │   ├── webhook.py    # Webhook handler
│   │   ├── pagerduty.py  # PagerDuty handler
│   │   ├── opsgenie.py   # Opsgenie handler
│   │   └── incident.py   # Incident management
│   ├── formatters.py      # Message formatters
│   ├── hooks.py           # Notification hooks
│   ├── registry.py        # Handler registry
│   └── factory.py         # Factory functions
├── multi_tenant/          # Multi-tenant support
│   ├── types.py           # Types and protocols
│   ├── config.py          # Configuration classes
│   ├── exceptions.py      # Exception hierarchy
│   ├── context.py         # Context management
│   ├── registry.py        # Tenant registry
│   ├── middleware.py      # HTTP middleware
│   ├── hooks.py           # Lifecycle hooks
│   ├── isolation/         # Isolation strategies
│   │   ├── base.py       # Protocols
│   │   ├── strategies.py # Implementations
│   │   └── validators.py # Validation
│   └── storage/           # Storage backends
│       ├── base.py       # Protocols
│       ├── memory.py     # In-memory storage
│       └── file.py       # File-based storage
├── secrets/               # Secret management system
│   ├── base.py            # SecretProvider protocol, types
│   ├── config.py          # Configuration classes
│   ├── exceptions.py      # Exception hierarchy
│   ├── registry.py        # Provider registry
│   ├── cache.py           # Secret caching (TTL, tiered)
│   ├── encryption.py      # Client-side encryption
│   ├── rotation.py        # Automatic rotation
│   ├── hooks.py           # Audit logging hooks
│   ├── middleware.py      # Provider wrappers
│   ├── tenant.py          # Multi-tenant isolation
│   ├── testing.py         # Test utilities
│   └── backends/          # Storage backends
│       ├── vault.py       # HashiCorp Vault
│       ├── aws.py         # AWS Secrets Manager
│       ├── gcp.py         # GCP Secret Manager
│       ├── azure.py       # Azure Key Vault
│       ├── env.py         # Environment variables
│       ├── file.py        # Encrypted files
│       └── memory.py      # In-memory (testing)
└── engines/               # Enterprise engine adapters
    ├── base.py            # Base classes and protocols
    ├── informatica.py     # Informatica Data Quality
    ├── talend.py          # Talend Data Quality
    ├── ibm_infosphere.py  # IBM InfoSphere
    ├── sap_data_services.py # SAP Data Services
    └── registry.py        # Engine registry
```

---

## Notifications

Multi-channel notification system for data quality alerts and SLA violations.

### Features

- Protocol-based handler abstraction for easy extensibility
- Async-first design with sync compatibility
- Built-in retry and circuit breaker integration
- Flexible message formatting (text, Markdown, structured)
- Hook system for logging, metrics, and custom processing
- Central registry for multi-channel dispatch

### Quick Start

```python
from packages.enterprise.notifications import (
    SlackNotificationHandler,
    WebhookNotificationHandler,
    get_notification_registry,
    NotificationLevel,
)

# Get global registry
registry = get_notification_registry()

# Register handlers
registry.register("slack", SlackNotificationHandler(
    webhook_url="https://hooks.slack.com/services/...",
))
registry.register("webhook", WebhookNotificationHandler(
    url="https://api.example.com/notify",
))

# Send to all channels
results = await registry.notify_all(
    message="Data quality check failed!",
    level=NotificationLevel.ERROR,
    context={"check_id": "chk_123", "failure_count": 5},
)
```

### Types and Enums

#### NotificationChannel

Supported notification channels:

| Channel | Description |
|---------|-------------|
| `SLACK` | Slack webhook |
| `WEBHOOK` | Generic webhook |
| `EMAIL` | Email (SMTP, SendGrid, SES) |
| `PAGERDUTY` | PagerDuty incidents |
| `OPSGENIE` | Opsgenie alerts |
| `TEAMS` | Microsoft Teams |
| `DISCORD` | Discord webhook |
| `SMS` | SMS messages |
| `CUSTOM` | Custom channel |

#### NotificationLevel

Severity levels (ordered from least to most severe):

| Level | Priority | Description |
|-------|----------|-------------|
| `DEBUG` | 0 | Detailed debugging information |
| `INFO` | 1 | General informational messages |
| `WARNING` | 2 | Warning conditions |
| `ERROR` | 3 | Error conditions requiring attention |
| `CRITICAL` | 4 | Critical conditions requiring immediate action |

```python
# Levels are comparable
NotificationLevel.WARNING < NotificationLevel.ERROR  # True
NotificationLevel.CRITICAL.priority  # 4
```

#### NotificationStatus

Status of a notification send attempt:

| Status | Terminal | Success | Description |
|--------|----------|---------|-------------|
| `PENDING` | No | - | Created, not yet sent |
| `SENDING` | No | - | Send in progress |
| `SENT` | Yes | Yes | Successfully sent |
| `DELIVERED` | Yes | Yes | Confirmed delivered |
| `FAILED` | Yes | No | Send failed |
| `RETRYING` | No | - | Retry in progress |
| `SKIPPED` | Yes | - | Intentionally skipped |
| `THROTTLED` | Yes | No | Rate limited |

### Data Types

#### NotificationPayload

Payload for a notification message:

```python
from packages.enterprise.notifications import NotificationPayload, NotificationLevel

payload = NotificationPayload(
    message="Data quality check failed!",
    level=NotificationLevel.ERROR,
    title="Validation Alert",
    context=(("check_id", "chk_123"), ("failure_count", 5)),
)

# Builder pattern
payload = payload.with_title("Alert")
payload = payload.with_context(job_id="job_456")
payload = payload.with_level(NotificationLevel.CRITICAL)
```

#### NotificationResult

Result of a notification send attempt:

```python
from packages.enterprise.notifications import NotificationResult, NotificationChannel

# Create success result
result = NotificationResult.success_result(
    channel=NotificationChannel.SLACK,
    handler_name="primary_slack",
    message_id="msg_123",
    duration_ms=150.5,
)

# Create failure result
result = NotificationResult.failure_result(
    channel=NotificationChannel.EMAIL,
    handler_name="smtp_email",
    error="Connection timeout",
    error_type="TimeoutError",
    retry_count=3,
)
```

#### BatchNotificationResult

Result of sending to multiple channels:

```python
batch_result = BatchNotificationResult.from_results(results_dict)
print(f"Success rate: {batch_result.success_count}/{batch_result.total_count}")
print(f"All succeeded: {batch_result.all_success}")

# Get failures
for name, result in batch_result.get_failures():
    print(f"{name}: {result.error}")
```

### Configuration

#### NotificationConfig

Base notification configuration:

```python
from packages.enterprise.notifications import (
    NotificationConfig,
    DEFAULT_NOTIFICATION_CONFIG,
    CRITICAL_NOTIFICATION_CONFIG,
)

config = NotificationConfig(
    timeout_seconds=30.0,
    max_retries=3,
    retry_delay_seconds=1.0,
    min_level=NotificationLevel.WARNING,
)

# Use presets
config = CRITICAL_NOTIFICATION_CONFIG  # High priority, more retries
```

#### SlackConfig

Slack-specific configuration:

```python
from packages.enterprise.notifications import SlackConfig

config = SlackConfig(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#alerts",
    username="Data Quality Bot",
    icon_emoji=":warning:",
    mention_users=("U12345",),
    mention_groups=("engineering",),
)
```

#### EmailConfig

Email configuration with provider support:

```python
from packages.enterprise.notifications import EmailConfig, EmailProvider, EmailEncryption

# SMTP configuration
config = EmailConfig(
    provider=EmailProvider.SMTP,
    smtp_host="smtp.example.com",
    smtp_port=587,
    username="alerts@example.com",
    password="secret",
    encryption=EmailEncryption.STARTTLS,
    from_address="alerts@example.com",
    to_addresses=("ops@example.com",),
)

# SendGrid configuration
config = EmailConfig(
    provider=EmailProvider.SENDGRID,
    api_key="SG.xxx",
    from_address="alerts@example.com",
    to_addresses=("ops@example.com",),
)

# AWS SES configuration
config = EmailConfig(
    provider=EmailProvider.SES,
    aws_region="us-east-1",
    from_address="alerts@example.com",
    to_addresses=("ops@example.com",),
)
```

### Handlers

#### NotificationHandler Protocol

All handlers implement this protocol:

```python
from packages.enterprise.notifications import (
    NotificationHandler,
    NotificationResult,
    NotificationPayload,
    NotificationChannel,
    NotificationConfig,
)

class CustomHandler(NotificationHandler):
    @property
    def name(self) -> str:
        return "my_custom_handler"

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.CUSTOM

    @property
    def config(self) -> NotificationConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return True

    async def send(
        self,
        payload: NotificationPayload,
    ) -> NotificationResult:
        # Implementation using payload.message, payload.level, payload.context_dict
        return NotificationResult.success_result(
            channel=self.channel,
            handler_name=self.name,
        )

    def should_send(self, payload: NotificationPayload) -> bool:
        return True
```

#### Built-in Handlers

```python
from packages.enterprise.notifications import (
    SlackNotificationHandler,
    WebhookNotificationHandler,
    EmailNotificationHandler,
)

# Slack handler
slack = SlackNotificationHandler(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#data-quality",
)

# Webhook handler
webhook = WebhookNotificationHandler(
    url="https://api.example.com/notify",
    method="POST",
    headers={"Authorization": "Bearer token"},
)

# Email handler
email = EmailNotificationHandler(
    config=EmailConfig(
        provider=EmailProvider.SMTP,
        smtp_host="smtp.example.com",
        from_address="alerts@example.com",
        to_addresses=("ops@example.com",),
    ),
)
```

### Factory Functions

```python
from packages.enterprise.notifications import (
    create_slack_handler,
    create_webhook_handler,
    create_email_handler,
    create_smtp_email_handler,
    create_sendgrid_email_handler,
    create_ses_email_handler,
    create_handler_from_config,
)

# Create Slack handler
handler = create_slack_handler(
    webhook_url="https://hooks.slack.com/...",
    channel="#alerts",
)

# Create from config dict
handler = create_handler_from_config({
    "type": "slack",
    "webhook_url": "https://hooks.slack.com/...",
})
```

### Formatters

Message formatters for different output formats:

```python
from packages.enterprise.notifications import (
    TextFormatter,
    MarkdownFormatter,
    SlackBlockFormatter,
    JsonFormatter,
    TemplateFormatter,
    get_formatter,
    register_formatter,
)

# Use text formatter
formatter = TextFormatter()
message = formatter.format(payload)

# Use Markdown formatter
formatter = MarkdownFormatter()
message = formatter.format(payload)

# Use Slack Block Kit formatter
formatter = SlackBlockFormatter()
blocks = formatter.format(payload)

# Template-based formatter
formatter = TemplateFormatter(
    template="Alert: {title}\n\nMessage: {message}\nLevel: {level}"
)

# Get formatter by name
formatter = get_formatter("markdown")
```

### Hooks

Notification lifecycle hooks for monitoring:

```python
from packages.enterprise.notifications import (
    LoggingNotificationHook,
    MetricsNotificationHook,
    CompositeNotificationHook,
    CallbackNotificationHook,
)

# Logging hook
logging_hook = LoggingNotificationHook()

# Metrics hook
metrics_hook = MetricsNotificationHook()

# Composite hook
composite = CompositeNotificationHook([logging_hook, metrics_hook])

# Custom callback hook
callback_hook = CallbackNotificationHook(
    on_send_start=lambda name, payload: print(f"Sending to {name}"),
    on_send_complete=lambda name, result: print(f"Result: {result.success}"),
)
```

### Registry

Central registry for managing handlers:

```python
from packages.enterprise.notifications import (
    NotificationRegistry,
    get_notification_registry,
    reset_notification_registry,
    register_handler,
    unregister_handler,
    notify,
    notify_all,
)

# Get global registry
registry = get_notification_registry()

# Register handlers
registry.register("primary_slack", slack_handler)
registry.register("backup_webhook", webhook_handler)

# Send to specific handler
result = await registry.notify(
    "primary_slack",
    message="Alert!",
    level=NotificationLevel.ERROR,
)

# Send to all handlers
results = await registry.notify_all(
    message="Critical alert!",
    level=NotificationLevel.CRITICAL,
)

# Convenience functions
register_handler("email", email_handler)
result = await notify("email", "Test message")
results = await notify_all("Broadcast message")
```

### Exceptions

```python
from packages.enterprise.notifications import (
    NotificationError,           # Base exception
    NotificationSendError,       # Send failed
    NotificationTimeoutError,    # Timeout occurred
    NotificationRetryExhaustedError,  # All retries failed
    NotificationConfigError,     # Configuration error
    NotificationHandlerNotFoundError,  # Handler not in registry
    NotificationFormatterError,  # Formatting failed
)
```

---

## Multi-Tenant

Complete multi-tenant support with context management, isolation enforcement, and quota tracking.

### Features

- Thread-safe and async-safe tenant context propagation
- Configurable isolation levels (shared, logical, physical, dedicated)
- Pluggable storage backends for tenant configurations
- HTTP middleware for tenant resolution
- Lifecycle hooks for monitoring and auditing
- Quota enforcement with configurable limits

### Quick Start

```python
from packages.enterprise.multi_tenant import (
    TenantRegistry,
    TenantConfig,
    TenantContextManager,
    get_current_tenant_id,
)

# Create a registry
registry = TenantRegistry()

# Create a tenant
tenant = registry.create_tenant(
    "acme-corp",
    "ACME Corporation",
    activate=True,
)

# Use tenant context
with TenantContextManager("acme-corp", config=tenant):
    print(get_current_tenant_id())  # "acme-corp"
    # All operations run in tenant context
```

### Types and Enums

#### TenantStatus

Tenant lifecycle states:

| Status | Operational | Accessible | Description |
|--------|-------------|------------|-------------|
| `PENDING` | No | No | Created but not activated |
| `ACTIVE` | Yes | Yes | Fully operational |
| `SUSPENDED` | No | Yes | Temporarily disabled |
| `DISABLED` | No | No | Permanently disabled |
| `ARCHIVED` | No | No | Data retained, not accessible |
| `DELETED` | No | No | Marked for deletion |

#### TenantTier

Subscription tiers:

| Tier | Description |
|------|-------------|
| `FREE` | Basic functionality with limited quotas |
| `STARTER` | Entry-level paid tier |
| `PROFESSIONAL` | Standard paid tier |
| `ENTERPRISE` | Full-featured enterprise tier |
| `CUSTOM` | Custom tier with negotiated limits |

#### IsolationLevel

Tenant isolation levels:

| Level | Security | Description |
|-------|----------|-------------|
| `SHARED` | 1 | Resources shared between tenants |
| `LOGICAL` | 2 | Logical separation with shared infrastructure |
| `PHYSICAL` | 3 | Physical separation (e.g., separate databases) |
| `DEDICATED` | 4 | Fully dedicated resources per tenant |

#### QuotaType

Types of quotas:

| Type | Description |
|------|-------------|
| `API_CALLS` | Number of API calls allowed |
| `STORAGE_BYTES` | Storage space in bytes |
| `ENGINES` | Number of data quality engines |
| `RULES` | Number of validation rules |
| `EXECUTIONS` | Number of validation executions |
| `USERS` | Number of users per tenant |
| `CONNECTIONS` | Number of data source connections |
| `CONCURRENT_JOBS` | Number of concurrent jobs |

#### QuotaPeriod

Quota reset periods:

| Period | Description |
|--------|-------------|
| `HOURLY` | Resets every hour |
| `DAILY` | Resets every day |
| `WEEKLY` | Resets every week |
| `MONTHLY` | Resets every month |
| `UNLIMITED` | No time-based limit |

#### Permission

Granular permissions:

| Permission | Description |
|------------|-------------|
| `READ` | Read access to resources |
| `WRITE` | Write/modify access |
| `DELETE` | Delete access |
| `ADMIN` | Administrative access |
| `EXECUTE` | Execute operations |
| `SHARE` | Share resources |
| `EXPORT` | Export data |
| `CONFIGURE` | Modify configuration |

#### ResourceType

Resource categories:

| Type | Description |
|------|-------------|
| `ENGINE` | Data quality engine instances |
| `RULE` | Validation rules |
| `SCHEMA` | Data schemas |
| `CONNECTION` | Data source connections |
| `DATASET` | Data sets |
| `REPORT` | Validation reports |
| `SCHEDULE` | Scheduled jobs |
| `ALERT` | Alert configurations |
| `USER` | User accounts |
| `CONFIG` | Configuration settings |

### Data Types

#### TenantId

Value object for tenant identifiers:

```python
from packages.enterprise.multi_tenant import TenantId

tenant_id = TenantId("acme-corp")
tenant_id = TenantId.from_string("  ACME-CORP  ")  # Normalized to "acme-corp"

# Validation: alphanumeric with - and _
TenantId("invalid@id")  # Raises ValueError
```

#### QuotaLimit

Quota limit definition:

```python
from packages.enterprise.multi_tenant import QuotaLimit, QuotaType, QuotaPeriod

limit = QuotaLimit(
    quota_type=QuotaType.API_CALLS,
    limit=10000,
    period=QuotaPeriod.DAILY,
    warning_threshold=0.8,  # Warn at 80%
)
```

#### QuotaUsage

Current quota usage:

```python
from packages.enterprise.multi_tenant import QuotaUsage

usage = QuotaUsage(
    quota_type=QuotaType.API_CALLS,
    current=7500,
    limit=10000,
    period=QuotaPeriod.DAILY,
    period_start=datetime(2025, 1, 1),
    period_end=datetime(2025, 1, 2),
)

print(usage.remaining)          # 2500
print(usage.usage_percentage)   # 0.75
print(usage.is_exceeded)        # False
```

### Configuration

#### TenantConfig

Tenant configuration:

```python
from packages.enterprise.multi_tenant import (
    TenantConfig,
    TenantTier,
    IsolationLevel,
)

config = TenantConfig(
    tenant_id="acme-corp",
    display_name="ACME Corporation",
    tier=TenantTier.ENTERPRISE,
    isolation_level=IsolationLevel.LOGICAL,
    enabled=True,
    quotas={
        QuotaType.API_CALLS: QuotaLimit(QuotaType.API_CALLS, 100000),
        QuotaType.ENGINES: QuotaLimit(QuotaType.ENGINES, 10),
    },
)
```

#### MultiTenantConfig

System-wide multi-tenant configuration:

```python
from packages.enterprise.multi_tenant import (
    MultiTenantConfig,
    DEFAULT_CONFIG,
    PRODUCTION_CONFIG,
    TESTING_CONFIG,
    STRICT_CONFIG,
)

config = MultiTenantConfig(
    enabled=True,
    default_tier=TenantTier.FREE,
    default_isolation=IsolationLevel.LOGICAL,
    require_tenant_context=True,
    auto_create_tenant=False,
)

# Use presets
config = PRODUCTION_CONFIG
```

### Context Management

Thread-safe and async-safe tenant context:

```python
from packages.enterprise.multi_tenant import (
    TenantContextManager,
    tenant_context,
    get_current_tenant_id,
    get_current_tenant_id_required,
    is_tenant_context_set,
    require_tenant_context,
)

# Context manager
with TenantContextManager("tenant-1", config=tenant_config):
    tenant_id = get_current_tenant_id()  # "tenant-1"

# Decorator
@tenant_context("tenant-1")
def process_data():
    pass

# Async context
async with TenantContextManager("tenant-1"):
    await async_operation()

# Decorator for functions that require tenant context
@require_tenant_context
def tenant_required_function():
    tenant_id = get_current_tenant_id_required()  # Raises if no context
```

#### Context Propagation

Propagate tenant context across threads/tasks:

```python
from packages.enterprise.multi_tenant import (
    TenantContextPropagator,
    context_propagator,
    copy_context_to_thread,
)

# Propagate to thread
with TenantContextManager("tenant-1"):
    copy_context_to_thread(target_thread)

# Use propagator
propagator = TenantContextPropagator()
context = propagator.capture()
# Later in another thread/task
propagator.restore(context)
```

### Registry

Central tenant registry:

```python
from packages.enterprise.multi_tenant import (
    TenantRegistry,
    get_registry,
    configure_registry,
    reset_registry,
    create_tenant,
    get_tenant,
    list_tenants,
    tenant_exists,
)

# Get global registry
registry = get_registry()

# Create tenant
tenant = registry.create_tenant(
    "new-tenant",
    "New Tenant",
    tier=TenantTier.STARTER,
    activate=True,
)

# Get tenant
tenant = registry.get_tenant("new-tenant")

# List tenants
tenants = registry.list_tenants(status=TenantStatus.ACTIVE)

# Convenience functions
tenant = create_tenant("tenant-id", "Display Name")
tenant = get_tenant("tenant-id")
exists = tenant_exists("tenant-id")
```

### Isolation

Tenant isolation enforcement:

```python
from packages.enterprise.multi_tenant import (
    create_isolation_enforcer,
    IsolationLevel,
    ResourceType,
    Permission,
    NoopIsolationEnforcer,
    SharedIsolationEnforcer,
    LogicalIsolationEnforcer,
    PhysicalIsolationEnforcer,
)

# Create enforcer for isolation level
enforcer = create_isolation_enforcer(IsolationLevel.LOGICAL)

# Check cross-tenant access
allowed = enforcer.check_access(
    source_tenant_id="tenant-a",
    target_tenant_id="tenant-b",
    resource_type=ResourceType.ENGINE,
    permission=Permission.READ,
)

# Validate resource ownership
valid = enforcer.validate_isolation(
    tenant_id="tenant-a",
    resource_id="engine-123",
    resource_type=ResourceType.ENGINE,
)
```

### Storage

Pluggable tenant storage backends:

```python
from packages.enterprise.multi_tenant import (
    TenantStorage,
    InMemoryTenantStorage,
    FileTenantStorage,
)

# In-memory storage (default)
storage = InMemoryTenantStorage()

# File-based storage
storage = FileTenantStorage("/path/to/tenants")

# Use with registry
registry = TenantRegistry(storage=storage)
```

### Middleware

HTTP middleware for tenant resolution:

```python
from packages.enterprise.multi_tenant import (
    TenantMiddleware,
    HeaderTenantResolver,
    SubdomainTenantResolver,
    PathTenantResolver,
    JWTClaimTenantResolver,
    CompositeTenantResolver,
    create_default_middleware,
)

# Create middleware
middleware = create_default_middleware(
    registry=registry,
    require_tenant=True,
)

# Custom resolver
resolver = HeaderTenantResolver(header_name="X-Tenant-ID")

# Composite resolver (try multiple strategies)
resolver = CompositeTenantResolver([
    HeaderTenantResolver(),
    SubdomainTenantResolver(),
    JWTClaimTenantResolver(claim_name="tenant_id"),
])

middleware = TenantMiddleware(
    registry=registry,
    resolver=resolver,
)
```

### Hooks

Lifecycle hooks for monitoring and auditing:

```python
from packages.enterprise.multi_tenant import (
    LoggingTenantHook,
    MetricsTenantHook,
    AuditTenantHook,
    CompositeTenantHook,
    CallbackTenantHook,
)

# Logging hook
logging_hook = LoggingTenantHook()

# Metrics hook
metrics_hook = MetricsTenantHook()

# Audit hook
audit_hook = AuditTenantHook()

# Composite hook
composite = CompositeTenantHook([
    logging_hook,
    metrics_hook,
    audit_hook,
])

registry = TenantRegistry(hooks=[composite])
```

### Exceptions

```python
from packages.enterprise.multi_tenant import (
    # Base
    MultiTenantError,
    # Lifecycle
    TenantNotFoundError,
    TenantAlreadyExistsError,
    TenantDisabledError,
    TenantSuspendedError,
    # Configuration
    TenantConfigurationError,
    TenantConfigValidationError,
    # Isolation
    TenantIsolationError,
    CrossTenantAccessError,
    # Authorization
    TenantAuthorizationError,
    TenantPermissionDeniedError,
    # Quota
    TenantQuotaError,
    TenantQuotaExceededError,
    TenantResourceLimitError,
    # Context
    TenantContextError,
    NoTenantContextError,
    TenantContextAlreadySetError,
    # Storage
    TenantStorageError,
    TenantDataNotFoundError,
    # Middleware
    TenantMiddlewareError,
    TenantResolutionError,
)
```

---

## Enterprise Engines

Adapters for enterprise data quality platforms.

### Features

- API-based authentication and connection management
- Complex rule translation to vendor-specific formats
- Result conversion from vendor formats to common types
- Connection pooling and retry mechanisms
- Health checking and lifecycle management
- Lazy vendor SDK loading

### Quick Start

```python
from packages.enterprise.engines import (
    get_enterprise_engine,
    InformaticaAdapter,
    InformaticaConfig,
)

# Get engine from registry
engine = get_enterprise_engine("informatica")

# Or create directly with config
config = InformaticaConfig(
    api_endpoint="https://idq.example.com/api/v2",
    api_key="your-api-key",
)
engine = InformaticaAdapter(config=config)

# Use with context manager
with engine:
    result = engine.check(data, rules)
    profile = engine.profile(data)
```

### Base Types

#### AuthType

Authentication types:

| Type | Description |
|------|-------------|
| `NONE` | No authentication |
| `API_KEY` | API key authentication |
| `BASIC` | Basic auth (username/password) |
| `OAUTH2` | OAuth 2.0 |
| `JWT` | JWT token |
| `CERTIFICATE` | Client certificate |

#### ConnectionMode

Connection modes:

| Mode | Description |
|------|-------------|
| `REST` | REST API |
| `SOAP` | SOAP API |
| `JDBC` | JDBC connection |
| `NATIVE` | Native SDK |

#### DataTransferMode

Data transfer modes:

| Mode | Description |
|------|-------------|
| `INLINE` | Data sent directly in API payload |
| `REFERENCE` | Data reference (path, URL) sent |
| `STREAMING` | Data streamed to engine |

### EnterpriseEngineConfig

Base configuration for enterprise engines:

```python
from packages.enterprise.engines import (
    EnterpriseEngineConfig,
    AuthType,
    ConnectionMode,
    DataTransferMode,
)

config = EnterpriseEngineConfig(
    # API Configuration
    api_endpoint="https://api.example.com",
    api_key="secret-key",
    auth_type=AuthType.API_KEY,
    connection_mode=ConnectionMode.REST,

    # Timeouts
    timeout_seconds=60.0,
    connect_timeout_seconds=10.0,

    # Retry
    max_retries=3,
    retry_delay_seconds=1.0,

    # SSL/Proxy
    verify_ssl=True,
    proxy_url=None,

    # Connection Pool
    pool_size=5,
    pool_timeout_seconds=30.0,

    # Data Transfer
    data_transfer_mode=DataTransferMode.INLINE,
    max_payload_size_mb=100.0,
    batch_size=10000,
)

# Builder pattern
config = config.with_api_endpoint("https://api.example.com")
config = config.with_api_key("secret")
config = config.with_basic_auth("user", "pass")
config = config.with_timeout(60.0, connect_timeout_seconds=10.0)
config = config.with_retry(5, delay_seconds=2.0)
```

### Preset Configurations

```python
from packages.enterprise.engines import (
    DEFAULT_ENTERPRISE_CONFIG,
    PRODUCTION_ENTERPRISE_CONFIG,
    DEVELOPMENT_ENTERPRISE_CONFIG,
    HIGH_THROUGHPUT_CONFIG,
)
```

### Informatica Adapter

```python
from packages.enterprise.engines import (
    InformaticaAdapter,
    InformaticaConfig,
    create_informatica_adapter,
    DEFAULT_INFORMATICA_CONFIG,
    PRODUCTION_INFORMATICA_CONFIG,
)

# Direct creation
config = InformaticaConfig(
    api_endpoint="https://idq.example.com/api/v2",
    api_key="your-api-key",
    domain="Production",
)
engine = InformaticaAdapter(config=config)

# Factory function
engine = create_informatica_adapter(
    api_endpoint="https://idq.example.com/api/v2",
    api_key="secret",
    domain="Production",
)
```

### Talend Adapter

```python
from packages.enterprise.engines import (
    TalendAdapter,
    TalendConfig,
    TalendExecutionMode,
    create_talend_adapter,
    DEFAULT_TALEND_CONFIG,
    EMBEDDED_TALEND_CONFIG,
)

config = TalendConfig(
    api_endpoint="https://tdc.example.com/api",
    api_key="your-api-key",
    execution_mode=TalendExecutionMode.REMOTE,
)
engine = TalendAdapter(config=config)
```

### IBM InfoSphere Adapter

```python
from packages.enterprise.engines import (
    IBMInfoSphereAdapter,
    IBMInfoSphereConfig,
    InfoSphereAnalysisType,
    create_ibm_infosphere_adapter,
    DEFAULT_INFOSPHERE_CONFIG,
    BATCH_INFOSPHERE_CONFIG,
)

config = IBMInfoSphereConfig(
    api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
    username="admin",
    password="secret",
    project="DataQuality",
)
engine = IBMInfoSphereAdapter(config=config)
```

### SAP Data Services Adapter

```python
from packages.enterprise.engines import (
    SAPDataServicesAdapter,
    SAPDataServicesConfig,
    SAPExecutionMode,
    create_sap_data_services_adapter,
    DEFAULT_SAP_DS_CONFIG,
    REALTIME_SAP_DS_CONFIG,
    ADDRESS_CLEANSING_CONFIG,
)

config = SAPDataServicesConfig(
    api_endpoint="https://ds.example.com/api",
    username="admin",
    password="secret",
    repository="Production",
)
engine = SAPDataServicesAdapter(config=config)
```

### Protocols

#### RuleTranslator

Translates common rules to vendor format:

```python
from packages.enterprise.engines import RuleTranslator, BaseRuleTranslator

class CustomRuleTranslator(BaseRuleTranslator):
    def _get_rule_mapping(self) -> dict[str, str]:
        return {
            "not_null": "VENDOR_NOT_NULL",
            "unique": "VENDOR_UNIQUE",
        }

    def _translate_rule_params(
        self,
        rule_type: str,
        rule: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Custom parameter translation
        ...
```

#### ResultConverter

Converts vendor results to common format:

```python
from packages.enterprise.engines import ResultConverter, BaseResultConverter

class CustomResultConverter(BaseResultConverter):
    def _extract_check_items(
        self,
        vendor_result: Any,
    ) -> list[dict[str, Any]]:
        # Parse vendor-specific result format
        ...
```

#### ConnectionManager

Manages connections to enterprise engines:

```python
from packages.enterprise.engines import ConnectionManager, BaseConnectionManager

class CustomConnectionManager(BaseConnectionManager):
    def _do_connect(self) -> Any:
        # Establish connection
        ...

    def _do_disconnect(self) -> None:
        # Close connection
        ...

    def _do_health_check(self) -> bool:
        # Check connection health
        ...
```

### Registry

```python
from packages.enterprise.engines import (
    EnterpriseEngineRegistry,
    get_enterprise_engine_registry,
    get_enterprise_engine,
    register_enterprise_engine,
    list_enterprise_engines,
    is_enterprise_engine_registered,
    register_with_common_registry,
)

# Get global registry
registry = get_enterprise_engine_registry()

# Get engine by name
engine = get_enterprise_engine("informatica")

# Register with common.engines
register_with_common_registry()

# Now accessible via common.engines
from common.engines import get_engine
engine = get_engine("informatica")
```

### Exceptions

```python
from packages.enterprise.engines import (
    EnterpriseEngineError,  # Base exception
    ConnectionError,        # Connection failed
    AuthenticationError,    # Authentication failed
    RateLimitError,         # Rate limit exceeded
    VendorSDKError,         # Vendor SDK error
    RuleTranslationError,   # Rule translation failed
    EngineNotRegisteredError,
    EngineAlreadyRegisteredError,
)
```

---

## Integration Examples

### Data Quality Check with Notifications

```python
from packages.enterprise.notifications import (
    get_notification_registry,
    create_slack_handler,
    NotificationLevel,
)
from packages.enterprise.multi_tenant import (
    TenantContextManager,
    get_current_tenant_id,
)
from common.engines import TruthoundEngine

# Setup
registry = get_notification_registry()
registry.register("slack", create_slack_handler("https://hooks.slack.com/..."))

engine = TruthoundEngine()

async def validate_with_notification(tenant_id: str, data):
    with TenantContextManager(tenant_id):
        result = engine.check(data, auto_schema=True)

        if result.status.name == "FAILED":
            await registry.notify_all(
                message=f"Data quality check failed for {tenant_id}",
                level=NotificationLevel.ERROR,
                context={
                    "tenant_id": get_current_tenant_id(),
                    "failed_count": result.failed_count,
                    "failures": [f.to_dict() for f in result.failures[:5]],
                },
            )

        return result
```

### Multi-Tenant Enterprise Engine

```python
from packages.enterprise.engines import InformaticaAdapter, InformaticaConfig
from packages.enterprise.multi_tenant import (
    TenantRegistry,
    TenantContextManager,
    get_current_tenant_id,
)

# Per-tenant engine configuration
tenant_configs = {
    "tenant-a": InformaticaConfig(
        api_endpoint="https://tenant-a.idq.example.com/api",
        api_key="tenant-a-key",
    ),
    "tenant-b": InformaticaConfig(
        api_endpoint="https://tenant-b.idq.example.com/api",
        api_key="tenant-b-key",
    ),
}

def get_engine_for_tenant() -> InformaticaAdapter:
    tenant_id = get_current_tenant_id()
    config = tenant_configs.get(tenant_id)
    if not config:
        raise ValueError(f"No configuration for tenant: {tenant_id}")
    return InformaticaAdapter(config=config)

# Usage
with TenantContextManager("tenant-a"):
    engine = get_engine_for_tenant()
    with engine:
        result = engine.check(data, rules)
```

---

## Testing

### Notification Testing

```python
from packages.enterprise.notifications import (
    NotificationRegistry,
    NotificationResult,
    NotificationPayload,
    NotificationChannel,
    NotificationConfig,
    NotificationLevel,
)

class MockHandler:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.sent_payloads: list[NotificationPayload] = []
        self._config = NotificationConfig()

    @property
    def name(self) -> str:
        return "mock"

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.CUSTOM

    @property
    def config(self) -> NotificationConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return True

    async def send(self, payload: NotificationPayload) -> NotificationResult:
        self.sent_payloads.append(payload)
        if self.should_fail:
            return NotificationResult.failure_result(
                channel=self.channel,
                handler_name=self.name,
                error="Mock failure",
            )
        return NotificationResult.success_result(
            channel=self.channel,
            handler_name=self.name,
        )

    def should_send(self, payload: NotificationPayload) -> bool:
        return True

# Test
mock = MockHandler()
registry = NotificationRegistry()
registry.register("mock", mock)

await registry.notify("mock", "Test message", level=NotificationLevel.INFO)
assert len(mock.sent_payloads) == 1
assert mock.sent_payloads[0].message == "Test message"
```

### Multi-Tenant Testing

```python
from packages.enterprise.multi_tenant import (
    TenantRegistry,
    TenantContextManager,
    InMemoryTenantStorage,
    get_current_tenant_id,
    TESTING_CONFIG,
)

def test_tenant_isolation():
    storage = InMemoryTenantStorage()
    registry = TenantRegistry(storage=storage, config=TESTING_CONFIG)

    # Create tenants
    registry.create_tenant("tenant-a", "Tenant A", activate=True)
    registry.create_tenant("tenant-b", "Tenant B", activate=True)

    # Test isolation
    with TenantContextManager("tenant-a"):
        assert get_current_tenant_id() == "tenant-a"

    with TenantContextManager("tenant-b"):
        assert get_current_tenant_id() == "tenant-b"
```

---

## Secrets

Secret management system with multiple backend support, caching, encryption, and audit logging.

### Features

- Protocol-based provider abstraction for easy extensibility
- Multiple backend support (Vault, AWS, GCP, Azure, Env, File)
- Client-side encryption (Fernet, AES-GCM, ChaCha20-Poly1305)
- TTL-based caching with tiered cache support
- Automatic secret rotation with configurable schedules
- Audit logging hooks for compliance
- Multi-tenant secret isolation

### Quick Start

```python
from packages.enterprise.secrets import (
    get_secret_registry,
    get_secret,
    set_secret,
    InMemorySecretProvider,
)

# Get global registry
registry = get_secret_registry()

# Register a provider
from packages.enterprise.secrets.backends import InMemorySecretProvider
registry.register("memory", InMemorySecretProvider())

# Use convenience functions
set_secret("database/password", "secret123")
secret = get_secret("database/password")
print(secret.value)
```

### Supported Backends

| Backend | Module | Description |
|---------|--------|-------------|
| HashiCorp Vault | `backends/vault.py` | KV v1/v2, Transit encryption |
| AWS Secrets Manager | `backends/aws.py` | AWS Secrets Manager integration |
| GCP Secret Manager | `backends/gcp.py` | Google Cloud Secret Manager |
| Azure Key Vault | `backends/azure.py` | Azure Key Vault integration |
| Environment Variables | `backends/env.py` | Read secrets from environment |
| Encrypted Files | `backends/file.py` | Local encrypted file storage |
| In-Memory | `backends/memory.py` | For testing only |

### Types and Enums

#### SecretType

Types of secrets:

| Type | Description |
|------|-------------|
| `STRING` | Plain text string (default) |
| `BINARY` | Binary data (base64 encoded) |
| `JSON` | JSON-structured data |
| `CERTIFICATE` | X.509 certificate |
| `KEY_PAIR` | Public/private key pair |
| `API_KEY` | API key or token |
| `PASSWORD` | Password credential |
| `CONNECTION_STRING` | Connection string |

#### BackendType

Storage backend types:

| Type | Description |
|------|-------------|
| `MEMORY` | In-memory storage (testing) |
| `ENV` | Environment variables |
| `FILE` | Encrypted file storage |
| `VAULT` | HashiCorp Vault |
| `AWS_SECRETS_MANAGER` | AWS Secrets Manager |
| `GCP_SECRET_MANAGER` | GCP Secret Manager |
| `AZURE_KEY_VAULT` | Azure Key Vault |

### Data Types

#### SecretValue

Immutable container for a secret:

```python
from packages.enterprise.secrets import SecretValue, SecretType

secret = SecretValue(
    value="my-secret",
    version="v1",
    secret_type=SecretType.PASSWORD,
    expires_at=datetime(2025, 12, 31),
)

print(secret.is_expired)  # False
print(secret.is_binary)   # False
```

### Configuration

#### SecretConfig

Base configuration for secret management:

```python
from packages.enterprise.secrets import (
    SecretConfig,
    BackendType,
    DEFAULT_SECRET_CONFIG,
    PRODUCTION_SECRET_CONFIG,
)

config = SecretConfig(
    backend_type=BackendType.VAULT,
    cache_enabled=True,
    cache_ttl_seconds=300.0,
    encryption_enabled=True,
    audit_enabled=True,
)

# Builder pattern
config = (
    SecretConfig()
    .with_cache(enabled=True, ttl_seconds=300.0)
    .with_retry(enabled=True, max_attempts=3)
    .with_encryption(enabled=True)
)
```

#### Backend-specific Configs

```python
from packages.enterprise.secrets import (
    VaultConfig,
    AWSSecretsManagerConfig,
    GCPSecretManagerConfig,
    AzureKeyVaultConfig,
)

# HashiCorp Vault
vault_config = VaultConfig(
    url="https://vault.example.com:8200",
    token="hvs.xxxxx",
    mount_point="secret",
    namespace="production",
)

# AWS Secrets Manager
aws_config = AWSSecretsManagerConfig(
    region_name="us-east-1",
    prefix="myapp/",
)

# GCP Secret Manager
gcp_config = GCPSecretManagerConfig(
    project_id="my-project",
    prefix="myapp-",
)

# Azure Key Vault
azure_config = AzureKeyVaultConfig(
    vault_url="https://myvault.vault.azure.net/",
)
```

### Providers

#### SecretProvider Protocol

All providers implement this protocol:

```python
from packages.enterprise.secrets import SecretProvider, SecretValue

class CustomProvider(SecretProvider):
    def get(self, path: str) -> SecretValue | None:
        # Retrieve secret
        ...

    def set(self, path: str, value: str | bytes, **kwargs) -> SecretValue:
        # Store secret
        ...

    def delete(self, path: str) -> bool:
        # Delete secret
        ...

    def exists(self, path: str) -> bool:
        # Check if secret exists
        ...

    def list(self, prefix: str = "") -> list[str]:
        # List secret paths
        ...
```

#### Built-in Providers

```python
from packages.enterprise.secrets.backends import (
    InMemorySecretProvider,
    EnvSecretProvider,
    FileSecretProvider,
    VaultSecretProvider,
    AWSSecretsManagerProvider,
    GCPSecretManagerProvider,
    AzureKeyVaultProvider,
)

# In-memory (testing)
provider = InMemorySecretProvider()

# Environment variables
provider = EnvSecretProvider(prefix="APP_SECRET_")

# Encrypted files
provider = FileSecretProvider(
    base_path="/etc/secrets",
    encryption_key=key,
)

# HashiCorp Vault
provider = VaultSecretProvider(config=vault_config)
```

### Caching

TTL-based caching with tiered support:

```python
from packages.enterprise.secrets import SecretCache, TieredSecretCache

# Simple cache
cache = SecretCache(ttl_seconds=300.0, max_size=1000)

# Tiered cache (L1: fast/small, L2: slower/larger)
cache = TieredSecretCache(
    l1_ttl_seconds=60.0,
    l1_max_size=100,
    l2_ttl_seconds=300.0,
    l2_max_size=1000,
)
```

### Encryption

Client-side encryption support:

```python
from packages.enterprise.secrets import (
    FernetEncryptor,
    AESGCMEncryptor,
    ChaCha20Poly1305Encryptor,
    generate_fernet_key,
    generate_aes_key,
)

# Fernet (recommended for simplicity)
key = generate_fernet_key()
encryptor = FernetEncryptor(key)

# AES-256-GCM
key = generate_aes_key()
encryptor = AESGCMEncryptor(key)

# ChaCha20-Poly1305
encryptor = ChaCha20Poly1305Encryptor(key)

# Encrypt/decrypt
ciphertext = encryptor.encrypt(b"secret data")
plaintext = encryptor.decrypt(ciphertext)
```

### Rotation

Automatic secret rotation:

```python
from packages.enterprise.secrets import (
    SecretRotationManager,
    RotationSchedule,
    RotationConfig,
    PasswordGenerator,
)

# Create rotation manager
manager = SecretRotationManager(provider=provider)

# Add rotation schedule
schedule = RotationSchedule(
    path="database/password",
    interval_days=30,
    generator=PasswordGenerator(length=32),
)
manager.add_schedule(schedule)

# Run rotation (usually in a background job)
results = await manager.rotate_all()
```

### Hooks

Audit logging and metrics:

```python
from packages.enterprise.secrets import (
    AuditLoggingHook,
    MetricsSecretHook,
    CompositeSecretHook,
)

# Audit logging (NEVER logs actual secret values)
audit_hook = AuditLoggingHook()

# Metrics collection
metrics_hook = MetricsSecretHook()

# Composite hook
hooks = CompositeSecretHook([audit_hook, metrics_hook])

# Use with middleware
from packages.enterprise.secrets import HookedProviderWrapper

wrapped = HookedProviderWrapper(provider, hooks=[hooks])
```

### Middleware

Provider wrappers for additional functionality:

```python
from packages.enterprise.secrets import (
    CachingProviderWrapper,
    EncryptingProviderWrapper,
    NamespacedProviderWrapper,
    ValidatingProviderWrapper,
    create_wrapped_provider,
)

# Caching wrapper
cached = CachingProviderWrapper(provider, cache=cache)

# Encryption wrapper
encrypted = EncryptingProviderWrapper(provider, encryptor=encryptor)

# Namespace wrapper
namespaced = NamespacedProviderWrapper(provider, namespace="myapp/")

# Factory function (combines all wrappers)
wrapped = create_wrapped_provider(
    provider,
    cache=cache,
    encryptor=encryptor,
    namespace="myapp/",
    hooks=[audit_hook],
)
```

### Multi-Tenant

Tenant-aware secret management:

```python
from packages.enterprise.secrets import (
    TenantAwareSecretProvider,
    create_tenant_provider,
)

# Create tenant-aware provider
tenant_provider = create_tenant_provider(
    base_provider=provider,
    tenant_separator="/",
)

# Secrets are automatically namespaced by tenant
# When tenant context is "acme": "password" -> "acme/password"
```

### Exceptions

```python
from packages.enterprise.secrets import (
    SecretError,              # Base exception
    SecretNotFoundError,      # Secret not found
    SecretAccessDeniedError,  # Access denied
    SecretExpiredError,       # Secret has expired
    SecretBackendError,       # Backend error
    SecretConnectionError,    # Connection failed
    SecretAuthenticationError,  # Authentication failed
    SecretEncryptError,       # Encryption failed
    SecretDecryptError,       # Decryption failed
    SecretRotationError,      # Rotation failed
    SecretValidationError,    # Validation failed
    SecretConfigurationError, # Configuration error
    ProviderNotFoundError,    # Provider not registered
)
```

### Testing

```python
from packages.enterprise.secrets import InMemorySecretProvider
from packages.enterprise.secrets.testing import (
    create_test_provider,
    create_test_secret,
)

# Create test provider with pre-populated secrets
provider = create_test_provider({
    "database/password": "test-password",
    "api/key": "test-api-key",
})

# Create test secret
secret = create_test_secret(value="test", version="v1")
```

---

## Installation

```bash
# Install with enterprise features
pip install truthound-orchestration[enterprise]
```

---

## Configuration Reference

### Environment Variables

```bash
# Notifications
NOTIFICATION_SLACK_WEBHOOK_URL=https://hooks.slack.com/...
NOTIFICATION_EMAIL_SMTP_HOST=smtp.example.com
NOTIFICATION_TIMEOUT_SECONDS=30

# Multi-Tenant
MULTI_TENANT_ENABLED=true
MULTI_TENANT_REQUIRE_CONTEXT=true
MULTI_TENANT_DEFAULT_TIER=free

# Enterprise Engines
INFORMATICA_API_ENDPOINT=https://idq.example.com/api
INFORMATICA_API_KEY=secret
TALEND_API_ENDPOINT=https://tdc.example.com/api

# Secrets
SECRET_BACKEND_TYPE=vault
SECRET_CACHE_ENABLED=true
SECRET_CACHE_TTL_SECONDS=300
SECRET_AUDIT_ENABLED=true

# Vault
VAULT_ADDR=https://vault.example.com:8200
VAULT_TOKEN=hvs.xxxxx
VAULT_NAMESPACE=production

# AWS
AWS_REGION=us-east-1
AWS_SECRET_PREFIX=myapp/

# GCP
GCP_PROJECT_ID=my-project
GCP_SECRET_PREFIX=myapp-

# Azure
AZURE_KEY_VAULT_URL=https://myvault.vault.azure.net/
```

---

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Main project documentation
- [common-modules.md](common-modules.md) - Common module documentation
- [architecture.md](architecture.md) - System architecture
- [roadmap.md](roadmap.md) - Development roadmap
