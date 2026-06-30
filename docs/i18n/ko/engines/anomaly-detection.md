!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Anomaly Detection
---

# Anomaly Detection

Anomaly detection is the advanced layer for spotting unusual patterns that might not be
expressed as simple deterministic 검증 rules.

## When To Use It

- the system needs behavior-aware monitoring, not only rule evaluation
- you want an early warning signal for unusual values or trends
- baseline drift and operational outliers both matter

## How To Use It Safely

- treat anomalies as operator signals first, not always as immediate hard failures
- combine them with host-native notifications and review 워크플로우s
- document which anomalies are informational, warning, or blocking

## Related Pages

- [Drift Detection](drift-detection.md)
- [Observability and Resilience](../common/observability-resilience.md)
