!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Rollout Topologies
---

# Rollout Topologies

Truthound 오케스트레이션 deployments can stay simple or become highly segmented
depending on whether one team, many teams, or many tenants share the same host.

## Who This Is For

- platform operators planning multi-environment rollout
- teams deciding how to separate development, staging, and production
- organizations introducing tenant isolation or governed secrets

## When To Use It

Use this page before the first organization-wide rollout or whenever a local
single-team setup grows into a shared platform surface.

## Common Topologies

| Topology | Best Fit | Notes |
|----------|----------|-------|
| single team / single host | early rollout | lowest overhead, fastest feedback |
| shared host / multiple teams | central platform teams | requires documented ownership of configs and alerts |
| multi-tenant shared runtime | regulated or heavily shared environments | pair with [Multi-Tenant](multi-tenant.md) and [Secrets](secrets.md) |
| CI-gated package promotion | larger organizations | promotes adapter and docs changes through explicit release gates |

## Production Pattern

- keep environment promotion explicit
- keep host-native secrets and notification paths separate per environment
- treat docs snapshot sync as part of the release flow, not an afterthought

## Related Pages

- [CI/CD and Production Rollout](ci-cd-production.md)
- [Multi-Tenant](multi-tenant.md)
- [Rollback Playbooks](rollback-playbooks.md)
