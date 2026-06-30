!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Prefect Flows
---

# Prefect Flows

Flows are where Prefect teams usually assemble the real user experience: load, validate, profile, learn, notify, and publish. The Truthound package gives you decorators, factories, and task helpers so those flows stay readable.

## Flow Configuration

The package exposes flow-oriented config types such as:

- `FlowConfig`
- `QualityFlowConfig`
- `PipelineFlowConfig`

## Basic Quality 검증 Flow

```python
from prefect import flow, task
from truthound_prefect.tasks import data_quality_check_task

@task
def load_data():
    return ...

@flow(name="basic_quality_flow")
async def basic_quality_flow():
    data = load_data()
    result = await data_quality_check_task(
        data,
        rules=[{"column": "id", "type": "not_null"}],
    )
    return result
```

## Flow With Profiling

```python
from prefect import flow
from truthound_prefect.tasks import data_quality_check_task, data_quality_profile_task

@flow(name="full_quality_flow")
async def full_quality_flow():
    data = ...
    check_future = data_quality_check_task.submit(data, rules=[{"column": "id", "type": "not_null"}])
    profile_future = data_quality_profile_task.submit(data)
    return {"check": check_future.result(), "profile": profile_future.result()}
```

## Flow Decorators And Factories

Use these when you want reusable patterns instead of rebuilding the same 오케스트레이션 shape repeatedly:

- `quality_checked_flow`
- `profiled_flow`
- `validated_flow`
- `create_quality_flow`
- `create_검증_flow`
- `create_pipeline_flow`

## Recommended Usage Pattern

- start with explicit task calls
- move to a saved block when configuration should be shared
- adopt flow decorators or factories when the same 오케스트레이션 pattern repeats across teams or datasets

## Deployment Guidance

Keep deployment concerns separate from 검증 semantics:

- flows define 오케스트레이션
- blocks define reusable config
- Prefect deployments define schedule, work pool, and environment bindings

See [Deployment Patterns](deployment-patterns.md) for the operational details.

## Related Reading

- [Tasks](tasks.md)
- [SLA and Hooks](sla.md)
- [Deployment Patterns](deployment-patterns.md)
