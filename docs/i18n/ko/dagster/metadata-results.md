!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster Metadata and Result Payloads
---

# Dagster Metadata and Result Payloads

Dagster users care about metadata-rich execution. Truthound supports that by converting shared quality results into Dagster-native metadata rather than inventing a separate result model.

## Who This Is For

- asset owners surfacing quality state in Dagster UI
- platform teams building dashboards from asset metadata
- engineers debugging how a Truthound result appears inside Dagster

## When To Use It

Use this page when:

- a check passes but operators still need detailed metadata
- an asset check should expose failure counts and quality percentages
- you need to translate a shared result into Dagster-native display fields

## Prerequisites

- a Dagster project using `DataQualityResource`, ops, or asset helpers
- familiarity with `to_dagster_metadata`
- understanding of the shared result contract

## Minimal Quickstart

Use the utility directly when custom wrapping is needed:

```python
from truthound_dagster import to_dagster_metadata

metadata = to_dagster_metadata(check_result)
```

For resource-driven checks, keep the result shared and let Dagster-specific metadata wrap it:

```python
def validated_users(data_quality):
    result = data_quality.check(
        load_users(),
        rules=[{"column": "id", "check": "not_null"}],
    )
    return result
```

## Production Pattern

Think in layers:

| Layer | Purpose |
|------|---------|
| shared Truthound result | canonical status, counts, and detailed failures |
| Dagster metadata | UI-friendly display, asset context, and operator navigation |

Recommended practices:

- keep the shared result available for programmatic use
- use Dagster metadata for display and run inspection
- avoid custom ad hoc metadata keys when shared ones already exist

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| metadata is present but hard to automate | only human-readable strings are emitted | preserve structured shared results alongside metadata |
| alerting loses quality detail | only Dagster event success/failure is read | include summary counts from the serialized result |
| two assets show different quality formatting | custom metadata wrappers diverged | standardize on `to_dagster_metadata` |

## Related Pages

- [Dagster Overview](index.md)
- [Resources](resources.md)
- [Assets and Asset Checks](assets.md)
- [Shared Result Serialization](../common/result-serialization.md)
