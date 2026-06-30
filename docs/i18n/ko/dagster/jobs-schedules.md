!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster Jobs and Schedules
---

# Dagster Jobs and Schedules

Truthound integrates cleanly into Dagster jobs when quality checks are modeled as explicit ops or resource-backed asset steps. The goal is to keep scheduling and automation Dagster-native while preserving the shared Truthound runtime contract underneath.

## Who This Is For

- Dagster operators designing scheduled 검증 jobs
- teams combining asset materialization and quality checks in one graph
- platform engineers standardizing reusable quality jobs

## When To Use It

Use this page when:

- a resource-backed check should run on a schedule
- you need a dedicated 검증 job separate from asset materialization
- you want to reuse `data_quality_check_op` across datasets

## Prerequisites

- `truthound-오케스트레이션[dagster]` installed
- Dagster Definitions-based project layout
- familiarity with `DataQualityResource` and the prebuilt ops

## Minimal Quickstart

Create a small quality job with the exported op surface:

```python
from dagster import Definitions, job
from truthound_dagster import DataQualityResource, data_quality_check_op

@job(resource_defs={"data_quality": DataQualityResource()})
def users_quality_job():
    data_quality_check_op.alias("check_users")()

defs = Definitions(jobs=[users_quality_job])
```

For mixed jobs, keep materialization and 검증 as separate nodes:

```python
from dagster import job, op
from truthound_dagster import create_check_op

@op
def load_users():
    return read_users_dataframe()

validate_users = create_check_op(
    name="validate_users",
    rules=[{"column": "id", "check": "not_null"}],
)
```

## Production Pattern

Use one of these shapes:

| Pattern | Best Fit |
|--------|----------|
| asset-only schedule | quality logic naturally belongs to assets |
| dedicated 검증 job | operators want a separate operational surface |
| mixed pipeline job | materialization and 검증 must stay in one Dagster graph |

Recommended schedule policy:

- keep 검증 schedules aligned with materialization freshness windows
- separate “fast gate” checks from expensive profile or learn jobs
- name jobs after the dataset or domain, not the engine

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| schedule is green but asset quality is stale | 검증 job is not aligned with materialization cadence | reschedule to run after the producing asset |
| one 검증 graph becomes too large | too many unrelated checks live in one job | split by domain or ownership boundary |
| jobs are hard to debug | materialization and 검증 share the same op name patterns | use explicit aliases for quality steps |

## Related Pages

- [Dagster Overview](index.md)
- [Ops](ops.md)
- [Assets and Asset Checks](assets.md)
- [Dagster Partitions and Automation](partitions-automation.md)
