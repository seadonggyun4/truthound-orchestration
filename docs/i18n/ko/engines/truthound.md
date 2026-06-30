!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Truthound Engine
---

# Truthound — Data Quality 워크플로우 Engine

`TruthoundEngine` is the default engine for the 오케스트레이션 stack. It is the best
fit for teams that want Truthound-first zero-config behavior and the broadest alignment
with the rest of the adapter family in this repository.

## When To Use It

Use `TruthoundEngine` when:

- you want the default and best-supported path
- you are onboarding to Truthound 3.x for the first time
- you want rule vocabulary and runtime semantics to match the shipped adapters closely

## Basic Pattern

```python
from common.engines import TruthoundEngine

with TruthoundEngine() as engine:
    result = engine.check(data, auto_schema=True)
```

## Main Operations

- `check`: run rules or auto-schema-backed 검증
- `profile`: inspect shape and quality characteristics
- `learn`: infer rules or baseline expectations from data

## Why It Is The Recommended Default

- closest alignment with the shared runtime layer
- strong zero-config behavior
- best coverage in the repository's adapter examples and CI
- natural fit for teams using Truthound as the primary quality system rather than as an
  adapter for an older investment

## Production Pattern

- use `auto_schema=True` for fast onboarding or exploratory flows
- promote learned or explicit rules for datasets with production contracts
- keep lifecycle and metrics hooks enabled in long-running orchestrators

## Failure Modes

- preflight failures usually mean runtime compatibility issues, not engine bugs
- inconsistent source shapes often belong in source resolution or serialization, not in
  the engine itself

## Related Pages

- [Batch Processing](batch.md)
- [Lifecycle Management](lifecycle.md)
