!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Rollback Playbooks
---

# Rollback Playbooks

Operational maturity is not just about successful rollout. It is also about
knowing how to step back safely when a host package, docs surface, support
matrix, or result contract changes unexpectedly.

## Who This Is For

- release managers
- on-call operators
- maintainers approving first-party adapter changes

## Rollback Triggers

- support-matrix or compatibility regressions
- host-specific API drift in supported minor versions
- docs or public-site regressions that break operator guidance
- result-serialization drift that breaks downstream consumers

## Minimal Rollback Sequence

1. stop promoting the new package or docs snapshot
2. revert to the last green support-matrix and CI gate state
3. confirm host-native execution still works on the prior version
4. restore public docs if operator-facing guidance regressed
5. reopen rollout only after the failing tuple or contract is reproduced and fixed

When the incident involves 워크플로우-triggered release or rollback flows, treat the shared 오케스트레이션 layer as the execution and status propagation surface only. Approval and rollback safety still belong to 워크플로우 policy. See [워크플로우 Pipelines].

## Related Pages

- [CI/CD and Production Rollout](ci-cd-production.md)
- [Rollout Topologies](rollout-topologies.md)
- [워크플로우 Pipelines]
- [Failure Catalog](../common/failure-catalog.md)
