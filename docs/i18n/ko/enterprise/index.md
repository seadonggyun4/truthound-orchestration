!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Enterprise Operations
---

# Enterprise Operations

The enterprise section covers the operator-facing modules in this repository: tenant
isolation, secret management, notification routing, and production rollout practices.

## Who This Is For

- platform teams operating shared Truthound installations
- organizations with multiple tenants, teams, or environments
- operators who need audited secret and notification flows

## What Lives Here

- [Rollout Topologies](rollout-topologies.md)
- [Multi-Tenant](multi-tenant.md)
- [Secrets](secrets.md)
- [Notifications](notifications.md)
- [Governance and Audit](governance-audit.md)
- [CI/CD and Production Rollout](ci-cd-production.md)
- [Rollback Playbooks](rollback-playbooks.md)

## How To Use These Pages

These docs are meant to complement the platform adapter guides. Use them when the
question is operational:

- how do we isolate tenants?
- where do secrets come from?
- how are incidents routed?
- how do we roll out 오케스트레이션 changes safely?
- how do we prove governance or recover from a bad rollout?

## Scope Note

These pages are public operator documentation, not internal design notes. They focus on
supported patterns and practical rollout choices rather than hidden implementation
details.
