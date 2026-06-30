!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Prefect Blocks
---

# Prefect Blocks

Blocks are the persisted configuration boundary for the Prefect integration. Use them when configuration should be named, saved, loaded, and reused across flows or deployments.

## DataQualityBlock

`DataQualityBlock` is the main high-level block.

```python
from truthound_prefect.blocks import DataQualityBlock

block = DataQualityBlock(engine_name="truthound", auto_schema=True)
```

It exposes `check`, `profile`, `learn`, and streaming helpers while hiding the lower-level engine block details.

## Main Fields

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine_name` | str | `"truthound"` | engine to use |
| `parallel` | bool | `False` | enable parallel Truthound execution |
| `max_workers` | int \| None | `None` | worker limit for parallel execution |
| `auto_schema` | bool | `False` | Truthound auto-schema behavior |
| `fail_on_error` | bool | `True` | raise on hard failures |
| `warning_threshold` | float \| None | `None` | warn instead of fail when below threshold |
| `timeout_seconds` | float | `300.0` | default operation timeout |

## Common Pattern

```python
from prefect import flow
from truthound_prefect.blocks import DataQualityBlock
from truthound_prefect.tasks import data_quality_check_task

@flow
async def validate_users(data):
    block = DataQualityBlock(engine_name="truthound")
    return await data_quality_check_task(data, block=block)
```

## EngineBlock

`EngineBlock` is the lower-level engine wrapper. Most users should start with `DataQualityBlock`.

## SLABlock

`SLABlock` is the persisted boundary for SLA configuration and evaluation.

## Using Blocks In Flows

Blocks work best when:

- the same configuration is reused across several flows
- the deployment surface needs explicit saved configuration
- you want to separate reusable policy from one-off flow code

## Managing Blocks

- update blocks through the normal Prefect block lifecycle
- delete blocks only after flows and deployments no longer reference them
- name blocks by environment and intent so they are easy to locate in the Prefect UI

## Related Reading

- [Tasks](tasks.md)
- [Flows](flows.md)
- [Deployment Patterns](deployment-patterns.md)
