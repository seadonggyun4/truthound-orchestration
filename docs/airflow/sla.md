---
title: Airflow SLA
---

# Airflow SLA Monitoring

Data quality SLA monitoring and alerting system.

## SLAConfig

Defines SLA configuration:

```python
from packages.airflow.sla import SLAConfig

config = SLAConfig(
    min_pass_rate=0.95,           # Minimum pass rate
    max_duration_seconds=3600,     # Maximum execution time
    max_failed_count=10,           # Maximum failure count
    alert_level="warning",         # Alert level
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_pass_rate` | float | Minimum pass rate (0.0-1.0) |
| `max_duration_seconds` | float | Maximum execution time |
| `max_failed_count` | int | Maximum allowed failure count |
| `alert_level` | str | Alert level (info, warning, error, critical) |

## SLAMetrics

SLA metric collection:

```python
from packages.airflow.sla import SLAMetrics

metrics = SLAMetrics(
    pass_rate=0.98,
    duration_seconds=1200,
    failed_count=5,
    passed_count=245,
)

# Check SLA compliance
is_compliant = metrics.check_compliance(config)
```

## SLAViolation

SLA violation information:

```python
from packages.airflow.sla import SLAViolation, SLAViolationType

violation = SLAViolation(
    violation_type=SLAViolationType.PASS_RATE_LOW,
    expected=0.95,
    actual=0.90,
    message="Pass rate below threshold",
)
```

### Violation Types

| Type | Description |
|------|-------------|
| `PASS_RATE_LOW` | Pass rate below threshold |
| `DURATION_EXCEEDED` | Execution time exceeded |
| `FAILED_COUNT_EXCEEDED` | Failure count exceeded |

## SLAMonitor

SLA monitoring:

```python
from packages.airflow.sla import SLAMonitor, SLAConfig

config = SLAConfig(min_pass_rate=0.95)
monitor = SLAMonitor(config)

# Record results
monitor.record(check_result)

# Check for SLA violations
violations = monitor.check_violations()
for v in violations:
    print(f"Violation: {v.violation_type.name}")
```

## DataQualitySLACallback

Connect SLA callback to DAG:

```python
from airflow import DAG
from packages.airflow.sla import SLAConfig, DataQualitySLACallback
from datetime import datetime

config = SLAConfig(
    min_pass_rate=0.95,
    max_duration_seconds=3600,
)

callback = DataQualitySLACallback(config)

with DAG(
    dag_id="sla_monitored_dag",
    start_date=datetime(2024, 1, 1),
    sla_miss_callback=callback,
) as dag:
    ...
```

## QualityAlertCallback

Quality alert callback:

```python
from packages.airflow.sla import QualityAlertCallback

def send_slack_alert(violation):
    # Send Slack alert
    pass

callback = QualityAlertCallback(
    on_violation=send_slack_alert,
)
```

## CallbackChain

Chain multiple callbacks:

```python
from packages.airflow.sla import CallbackChain

chain = CallbackChain([
    DataQualitySLACallback(config),
    QualityAlertCallback(on_violation=send_alert),
])

with DAG(
    dag_id="chained_callbacks_dag",
    sla_miss_callback=chain,
) as dag:
    ...
```

## SLARegistry

SLA registration and management:

```python
from packages.airflow.sla import SLARegistry, SLAConfig

registry = SLARegistry()

# Register SLA
registry.register(
    name="critical_data",
    config=SLAConfig(min_pass_rate=0.99),
)

# Retrieve SLA
config = registry.get("critical_data")

# List all SLAs
all_slas = registry.list_all()
```

## AlertLevel

Alert levels:

| Level | Description |
|-------|-------------|
| `INFO` | Informational alert |
| `WARNING` | Warning |
| `ERROR` | Error |
| `CRITICAL` | Critical |
