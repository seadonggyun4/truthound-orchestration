!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Great Expectations Adapter
---

# Great Expectations Adapter

`GreatExpectationsAdapter` exists for teams that need to preserve prior Great
Expectations investment while moving 오케스트레이션 and runtime handling toward the
Truthound 3.x model.

## When To Use It

Use the Great Expectations adapter when:

- expectations already exist and cannot be replaced immediately
- the organization wants shared 오케스트레이션 semantics before fully migrating rule
  assets

## Strengths

- lets existing expectation work participate in the same host integrations
- creates a migration bridge instead of forcing a hard cutover

## Tradeoffs

- adapter behavior may not expose every Truthound-native capability the same way
- teams should expect some conceptual mismatch between expectation-first and
  Truthound-first 검증 styles

## Production Guidance

- use it as a managed transition path, not only as a permanent abstraction layer
- keep shared runtime behavior explicit so operators understand which parts come from
  the host adapter and which come from the underlying engine

## Related Pages

- [Truthound Engine](truthound.md)
- [Pandera Adapter](pandera.md)
