!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Streaming 검증
---

# Streaming 검증

Streaming 검증 is the capability-aware path for validating data in batches or
incremental chunks without treating the entire input as one eager dataset.

## When To Use It

Use streaming 검증 when:

- data arrives incrementally
- memory pressure makes eager 검증 unattractive
- operators want early failure or incremental visibility

## Capability Model

Not every engine supports streaming. Check capabilities before designing a streaming
워크플로우 around a specific engine:

```python
from common.engines import get_engine

engine = get_engine("truthound")
supports_streaming = engine.get_capabilities().supports_streaming
```

## Operational Pattern

- normalize the source first
- validate each chunk or batch
- decide whether to fail fast or aggregate later
- emit metrics and summaries at the host level

## Shared Guidance

Streaming works best when paired with:

- shared result serialization
- host-native retries and backoff
- explicit observability around chunk failures and lag

## Related Pages

- [Batch Processing](batch.md)
- [Observability and Resilience](../common/observability-resilience.md)
