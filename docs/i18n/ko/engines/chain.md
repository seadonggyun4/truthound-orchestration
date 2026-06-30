!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Engine Chains
---

# Engine Chains

Engine chains let you layer resolution and fallback behavior instead of assuming one
engine must always serve every dataset the same way.

## When Engine Chains Help

- one engine is the primary path and another is a compatibility fallback
- you need phased migration between engines
- different workloads justify different engines under one 오케스트레이션 host

## Practical Use Cases

- Truthound-first default with a Pandera fallback during migration
- host-specific rollout where legacy datasets stay on Great Expectations temporarily

## Operational Guidance

- keep the primary engine explicit
- treat fallbacks as deliberate policy, not silent magic
- expose the selected engine in logs and metrics so operators can reason about results

## Related Pages

- [Engines Overview](index.md)
- [Lifecycle Management](lifecycle.md)
