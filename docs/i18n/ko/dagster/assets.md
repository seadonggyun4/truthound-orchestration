!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster Assets
---

# Dagster Assets And Asset Checks

Dagster assets are where the integration feels most native. The package provides decorators and factories so that 검증 can live next to assets without losing shared runtime behavior.

## `@quality_checked_asset`

Use it when the asset itself should be quality-aware.

## `@profiled_asset`

Use it when the asset should emit profile information as part of its metadata story.

## Asset Factories

`create_quality_asset`, `create_quality_check_asset`, and `create_asset_check` help when you want consistent patterns across many assets.

## Asset Configuration

`QualityAssetConfig`, `ProfileAssetConfig`, and `QualityCheckMode` let you control whether checks run before, after, or around the asset boundary.

## Asset Dependencies

Asset checks should follow the same asset dependency model as the Dagster graph. Keep upstream or downstream intent obvious instead of hiding quality logic in unrelated assets.

## Metadata

Quality metadata should stay Dagster-native, but the underlying 검증 semantics should still come from the shared runtime contract.

## Related Reading

- [Resources](resources.md)
- [Ops](ops.md)
- [SLA and Hooks](sla-hooks.md)
