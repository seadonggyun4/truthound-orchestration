!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Mage Production Rollout
---

# Mage Production Rollout

Mage is often adopted first in a single pipeline and only later standardized across projects. This page documents the rollout path that keeps Truthound quality blocks maintainable as the surface grows.

## Who This Is For

- teams promoting Mage quality checks from pilot to production
- platform engineers defining a shared rollout pattern
- operators creating incident and rollback procedures for Mage pipelines

## When To Use It

Use this page when:

- more than one Mage pipeline needs the same quality pattern
- SLA-aware rollout and alerting are becoming operational requirements
- a local `io_config.yaml` path needs to become environment-safe

## Prerequisites

- at least one working Truthound Mage pipeline
- project configuration under version control
- an environment strategy for Mage execution

## Minimal Quickstart

Start with a dedicated 검증 transformer block:

```python
from truthound_mage import CheckTransformer, CheckBlockConfig

check_block = CheckTransformer(
    config=CheckBlockConfig(
        rules=[{"column": "id", "check": "not_null"}],
        fail_on_error=True,
    )
)
```

Then add a sensor or condition for the rollout boundary:

```python
from truthound_mage import DataQualityCondition
```

## Production Pattern

Recommended rollout ladder:

1. validate in one transformer block
2. externalize source config through `io_config.yaml` and profiles
3. add sensors or conditions for branching and gating
4. enable SLA hooks and notification routing
5. standardize shared block config across pipelines

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| every pipeline rolls its own quality behavior | no shared rollout template exists | publish one blessed transformer + gating pattern |
| incident response is slow | blocks emit results but no operational hooks are wired | add SLA monitoring before wider rollout |
| local assumptions leak into production | `io_config.yaml` and env policy are not formalized | promote profile and secret conventions early |

## Related Pages

- [Mage Overview](index.md)
- [Block Types and Runtime Context](block-runtime-context.md)
- [Variables, Secrets, and Profiles](variables-secrets.md)
- [Enterprise CI/CD and Production Rollout](../enterprise/ci-cd-production.md)
