!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Airflow Troubleshooting
---

# Airflow Troubleshooting

## Local Paths Work But SQL Fails

Cause:

- local file zero-config is working as intended, but SQL still needs an Airflow connection

Fix:

- configure or reference the Airflow connection explicitly
- verify the hook path before debugging the engine

## DAG Imports Fail On Install

Cause:

- unsupported Airflow or Python tuple
- dependency resolution mismatch

Fix:

- compare the runtime against [Compatibility](../compatibility.md)
- reinstall against the documented Airflow line

## Sensor Waits Forever

Cause:

- threshold is too strict for the current source
- source is stale
- the sensor is waiting on a connection-backed path that never resolves correctly

Fix:

- lower the threshold temporarily for diagnostics
- inspect the same source with `DataQualityCheckOperator`
- prefer `reschedule` or deferrable behavior for long waits

## XCom Consumers Break

Cause:

- downstream code assumes a custom result shape instead of the shared wire contract

Fix:

- consume documented result fields only
- keep any extra metadata additive
