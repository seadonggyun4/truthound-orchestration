!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Prefect SLA And Hooks
---

# Prefect SLA And Hooks

Prefect deployments often need the same operational guardrails as the other hosts: pass-rate thresholds, execution-time expectations, and a clear escalation path when quality drifts outside policy.

## Main Types

| Type | Purpose |
|------|---------|
| `SLAConfig` | threshold configuration |
| `SLABlock` | persisted SLA configuration |
| `SLAMonitor` | evaluation |
| `LoggingSLAHook`, `MetricsSLAHook`, `CompositeSLAHook` | response and telemetry |

## When To Use SLABlock

Use `SLABlock` when SLA policy should be saved and reused independently of any one flow definition.

## Recommended Pattern

- 검증 stays in tasks or flows
- SLA policy lives in config or saved blocks
- hooks handle logging, metrics, or notification side effects

## Related Reading

- [Blocks](blocks.md)
- [Deployment Patterns](deployment-patterns.md)
