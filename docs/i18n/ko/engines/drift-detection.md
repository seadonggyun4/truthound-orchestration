!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Drift Detection
---

# Drift Detection

Drift detection is an advanced engine capability for comparing current data behavior to
an earlier baseline. It is useful when a dataset can still pass simple checks while
quietly changing in ways operators care about.

## When To Use It

- baseline distributions matter, not just row-level validity
- upstream systems change gradually and silently
- model quality needs trend-aware monitoring

## Operational Guidance

- pair drift detection with profiling and baseline management
- use it as an additional signal, not a replacement for core checks
- keep severity and notification policy explicit so expected seasonal changes do not
  page the team unnecessarily

## Related Pages

- [Anomaly Detection](anomaly-detection.md)
- [Truthound Engine](truthound.md)
