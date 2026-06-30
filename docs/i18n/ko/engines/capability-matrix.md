!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Engine Capability Matrix
---

# Engine Capability Matrix

The 오케스트레이션 adapters expose a broad quality surface, but not every engine
supports every operation equally. This page helps operators choose the right
engine for `check`, `profile`, `learn`, `batch`, `stream`, `drift`, and
`anomaly` 워크플로우s.

## Who This Is For

- teams deciding whether to stay on Truthound or route to another engine
- operators writing compatibility runbooks
- contributors extending engine support

## When To Use It

Use this page before changing engine selection in production or when a
preflight report says an operation is not supported.

## Prerequisites

- understanding of [Engine Resolution and Selection](../common/engine-resolution-selection.md)
- a known target operation

## Capability View

| Capability | Truthound | Pandera | Great Expectations |
|------------|-----------|---------|--------------------|
| check | primary first-party path | strong dataframe-oriented fit | supported through the GE adapter |
| profile | first-party path | limited compared with Truthound profiling semantics | adapter-dependent |
| learn | first-party path | not the primary Pandera 워크플로우 | not the primary GE 워크플로우 |
| stream check | first-party path in this repo | not the default choice | not the default choice |
| drift | first-party path in this repo | limited | limited |
| anomaly | first-party path in this repo | limited | limited |
| batch helpers | first-party path in this repo | available through shared batch helpers where supported | available where adapter semantics permit |

## What The Matrix Means

- Truthound is the canonical engine for the full first-party feature line.
- Pandera is the best fit when dataframe schema enforcement is the main need.
- Great Expectations is the best fit when a team is intentionally carrying an
  expectations-style 검증 model into the 오케스트레이션 layer.
- The shared runtime can still offer 오케스트레이션 conveniences even when the
  engine surface is narrower.

## Production Pattern

- Prefer Truthound unless a narrower engine fits the problem materially better.
- Use `run_preflight(...)` to confirm capability instead of assuming parity.
- Document the engine choice per host because the same team may use different
  engines in different 오케스트레이션 systems.

## Failure Modes And Troubleshooting

| Symptom | Likely Cause | Read Next |
|---------|--------------|-----------|
| preflight says operation unsupported | selected engine lacks capability | [Selection Guide](selection-guide.md) |
| same rule works in one engine but not another | rule vocabulary or behavior differs | [Truthound Engine](truthound.md), [Pandera](pandera.md), [Great Expectations](great-expectations.md) |

## Related Pages

- [Selection Guide](selection-guide.md)
- [Truthound Engine](truthound.md)
- [Pandera](pandera.md)
- [Great Expectations](great-expectations.md)
