!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Custom Engine Authoring
---

# Custom Engine Authoring

`truthound-오케스트레이션` is intentionally Truthound-first, but it is not
closed. The shared runtime exposes engine protocols, a registry, and resolver
hooks so advanced teams can register custom engines while keeping the host
adapters stable.

## Who This Is For

- advanced teams extending the engine layer
- contributors implementing new engine adapters
- operators reviewing whether a custom engine is safe to adopt

## When To Use It

Use this page only when Truthound, Pandera, and Great Expectations are not a
fit for a concrete use case.

## Prerequisites

- familiarity with the shared runtime and engine registry
- clear ownership for the custom engine's lifecycle and support burden

## Minimal Quickstart

```python
from common.engines import register_engine

register_engine(
    "custom_engine",
    factory=my_engine_factory,
    set_default=False,
)
```

The custom engine should still participate in the same shared contracts used by
the host adapters:

- capability reporting
- result serialization expectations
- compatibility and preflight checks

## Production Pattern

- Keep the custom engine behind an explicit engine name.
- Do not replace the default Truthound engine casually.
- Add host-level smoke tests for every adapter that will call into the custom
  engine.
- Document which operations are intentionally unsupported.

## Failure Modes And Troubleshooting

| Symptom | Likely Cause |
|---------|--------------|
| host creates the wrong engine | the registry entry or default setting is misconfigured |
| preflight passes but runtime payloads drift | the custom engine does not align with shared result semantics |
| one host works and another fails | host-specific runtime context assumptions leaked into the custom engine |

## Related Pages

- [Capability Matrix](capability-matrix.md)
- [Error Reporting and Diagnostics](error-reporting.md)
- [Engine Resolution and Selection](../common/engine-resolution-selection.md)
