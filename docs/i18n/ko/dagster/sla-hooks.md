!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster SLA And Hooks
---

# Dagster SLA And Hooks

Dagster 워크플로우s often need both data-quality correctness and operational guarantees. The SLA layer helps you enforce thresholds such as pass rate or execution time without mixing alerting logic directly into every asset or op.

## Main Types

| Type | Purpose |
|------|---------|
| `SLAConfig` | threshold definition |
| `SLAMonitor` | evaluation logic |
| `SLAResource` | reusable resource-based SLA integration |
| `SLAHook` and hook implementations | alerting and monitoring side effects |

## Recommended Pattern

- use `DataQualityResource` or asset helpers for execution
- use SLA resources or hooks for policy and escalation
- keep quality semantics and escalation semantics separate

## Hook Types

The Dagster package exposes logging-oriented, metrics-oriented, and composite hooks for SLA handling.

## Related Reading

- [Resources](resources.md)
- [Recipes](recipes.md)
