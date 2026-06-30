!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster Install And Compatibility
---

# Dagster Install And Compatibility

## Supported Dagster Line

The current support anchors are:

- minimum supported: Dagster `1.5.0` on Python `3.11`
- primary supported: Dagster `1.12.18` on Python `3.12`

Always check [Compatibility](../compatibility.md) for the release-blocking matrix before upgrading.

## Install

```bash
pip install truthound-오케스트레이션[dagster] "truthound>=3.0,<4.0"
```

## What To Verify After Install

- `truthound_dagster` imports correctly
- `DataQualityResource()` initializes in a small definitions file
- one asset or op can execute a simple check
- any asset-check helpers you depend on match your Dagster version

## Upgrade Guidance

- upgrade Dagster and Python as a tested tuple
- verify resource initialization before re-testing asset checks
- if a legacy Dagster lane behaves differently, treat that as a compatibility-lane issue first

## Related Reading

- [Dagster Overview](index.md)
- [Resources](resources.md)
- [Assets and Asset Checks](assets.md)
