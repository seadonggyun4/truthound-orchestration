!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Engine Selection Guide
---

# Engine Selection Guide

Choosing an engine is less about brand preference and more about matching the
검증 model to the host, data shape, and operational goal.

## Who This Is For

- teams standardizing one engine per platform
- operators reviewing an engine change request
- contributors designing adapter defaults

## When To Use It

Use this guide when you need a default engine recommendation for a host or when
you are deciding whether to keep Truthound as the primary path.

## Minimal Decision Table

| If your main need is... | Start With... |
|-------------------------|---------------|
| the full first-party 오케스트레이션 feature line | Truthound |
| dataframe schema contracts in Python 워크플로우s | Pandera |
| expectations-style 검증 carried over from existing GE usage | Great Expectations |
| stream, drift, anomaly, or learn 워크플로우s | Truthound |

## Host-Specific Defaults

| Host | Recommended Default | Why |
|------|---------------------|-----|
| Airflow | Truthound | best match for operator, sensor, and SLA patterns |
| Dagster | Truthound | best match for assets, asset checks, and metadata-rich outputs |
| Prefect | Truthound | best match for blocks, tasks, and flow factories |
| dbt | Truthound | the package itself is the first-party Truthound SQL surface |
| Mage | Truthound | best match for block-level checks and runtime helpers |
| Kestra | Truthound | best match for scripts, generated flows, outputs, and stream helpers |

## Production Pattern

- Default to one engine per team unless there is a strong operational reason to
  mix them.
- If you introduce a second engine, keep the selection logic in shared runtime
  configuration rather than scattering it across host code.
- Validate the chosen engine in CI using the same host and source shape used in
  production.

## Related Pages

- [Capability Matrix](capability-matrix.md)
- [Custom Engines](custom-engines.md)
- [Engine Resolution and Selection](../common/engine-resolution-selection.md)
