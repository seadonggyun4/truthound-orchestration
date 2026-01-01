---
title: Prefect Flows
---

# Prefect Flows

Flow templates and configurations for data quality pipelines.

## Flow Configuration

### FlowConfig

```python
from packages.prefect.flows import FlowConfig

config = FlowConfig(
    name="quality_flow",
    timeout_seconds=3600,
    retries=3,
    retry_delay_seconds=60,
)
```

### QualityFlowConfig

```python
from packages.prefect.flows import QualityFlowConfig

config = QualityFlowConfig(
    auto_schema=True,
    fail_on_error=True,
    engine_name="truthound",
    parallel=True,
)
```

### PipelineFlowConfig

```python
from packages.prefect.flows import PipelineFlowConfig

config = PipelineFlowConfig(
    stages=["load", "validate", "transform", "save"],
    fail_fast=True,
)
```

## Flow Examples

### Basic Quality Validation Flow

```python
from prefect import flow, task
from packages.prefect.tasks import data_quality_check_task

@task
def load_data():
    return pl.read_parquet("data.parquet")

@task
def save_result(result):
    # Result saving logic
    pass

@flow(name="basic_quality_flow")
def basic_quality_flow():
    data = load_data()
    result = data_quality_check_task(data, auto_schema=True)
    save_result(result)
    return result
```

### Flow with Profiling

```python
from prefect import flow
from packages.prefect.tasks import (
    data_quality_check_task,
    data_quality_profile_task,
)

@flow(name="full_quality_flow")
def full_quality_flow():
    data = load_data()

    # Parallel execution
    check_future = data_quality_check_task.submit(data, auto_schema=True)
    profile_future = data_quality_profile_task.submit(data)

    # Collect results
    check_result = check_future.result()
    profile_result = profile_future.result()

    return {
        "check": check_result,
        "profile": profile_result,
    }
```

### Conditional Validation Flow

```python
from prefect import flow, task
from packages.prefect.tasks import (
    strict_check_task,
    lenient_check_task,
)

@task
def determine_mode(data):
    # Determine validation mode based on data size
    return "strict" if len(data) < 10000 else "lenient"

@flow(name="conditional_quality_flow")
def conditional_quality_flow():
    data = load_data()
    mode = determine_mode(data)

    if mode == "strict":
        result = strict_check_task(data)
    else:
        result = lenient_check_task(data)

    return result
```

### Learn and Validate Flow

```python
from prefect import flow
from packages.prefect.tasks import (
    data_quality_learn_task,
    data_quality_check_task,
)

@flow(name="learn_and_check_flow")
def learn_and_check_flow():
    # Learn schema from baseline data
    baseline = load_baseline_data()
    learn_result = data_quality_learn_task(baseline)

    # Validate new data
    new_data = load_new_data()
    check_result = data_quality_check_task(
        new_data,
        rules=learn_result.rules,
    )

    return check_result
```

## Subflows

```python
from prefect import flow
from packages.prefect.tasks import data_quality_check_task

@flow(name="quality_subflow")
def quality_subflow(data):
    return data_quality_check_task(data, auto_schema=True)

@flow(name="main_flow")
def main_flow():
    data1 = load_data("source1")
    data2 = load_data("source2")

    # Call subflows
    result1 = quality_subflow(data1)
    result2 = quality_subflow(data2)

    return [result1, result2]
```

## Scheduling

```python
from prefect import flow
from prefect.server.schemas.schedules import CronSchedule

@flow(name="scheduled_quality_flow")
def scheduled_quality_flow():
    data = load_data()
    result = data_quality_check_task(data, auto_schema=True)
    return result

# Schedule configuration during deployment
# prefect deployment build ./flow.py:scheduled_quality_flow \
#     --cron "0 0 * * *" \
#     --name "daily-quality-check"
```

## Notifications

```python
from prefect import flow
from prefect.blocks.notifications import SlackWebhook

@flow(name="flow_with_notification")
def flow_with_notification():
    data = load_data()
    result = data_quality_check_task(data, auto_schema=True)

    if result.status.name == "FAILED":
        slack = SlackWebhook.load("my-slack-webhook")
        slack.notify(f"Quality check failed: {result.failed_count} failures")

    return result
```
