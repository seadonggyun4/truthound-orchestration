!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster Troubleshooting
---

# Dagster Troubleshooting

## Resource Not Initialized

Cause:

- `DataQualityResource` is being used outside Dagster's resource lifecycle

Fix:

- use it through `Definitions` or the proper Dagster execution context

## Asset Check Behavior Differs Across Versions

Cause:

- older Dagster lanes expose slightly different asset-check API surfaces

Fix:

- compare against the documented support matrix
- keep examples aligned with the primary supported line when writing new definitions

## Result Metadata Looks Right But Downstream Logic Fails

Cause:

- Dagster metadata is fine, but downstream code is assuming a non-shared result contract

Fix:

- keep downstream logic on documented result fields
