!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Multi-Tenant
---

# Multi-Tenant

Truthound's multi-tenant module is for shared deployments where one 오케스트레이션
service, worker fleet, or internal platform must safely serve multiple tenants.

## Main Components

- `TenantRegistry`
- `TenantContextManager`
- isolation strategies
- middleware and context propagation helpers

## When To Use It

Use multi-tenant support when:

- one deployment serves multiple customers or business units
- quotas, isolation, or audit boundaries matter
- the same 오케스트레이션 stack needs tenant-aware secrets or notifications

## Recommended Operating Model

- maintain tenant definitions centrally through the registry
- enter tenant context explicitly around work that should be isolated
- keep isolation strategy aligned with the real risk model of the platform

## Isolation Levels

The module supports shared, logical, and stronger isolation strategies. The right
choice depends on whether you need convenience, logical separation, or tighter
environmental boundaries.

## Practical Guidance

- treat tenant context as required execution metadata
- avoid hidden global state outside the tenant context helpers
- test cross-tenant access rules as part of platform verification, not only at runtime

## Related Pages

- [Secrets](secrets.md)
- [CI/CD and Production Rollout](ci-cd-production.md)
