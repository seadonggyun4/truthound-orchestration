!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster Ops
---

# Dagster Ops

Prebuilt ops are the right choice when you want Dagster jobs to compose explicit 검증 stages instead of putting all behavior inside assets.

## Pre-built Ops

### `data_quality_check_op`

Use it for standard 검증 steps inside jobs and graphs.

### `data_quality_profile_op`

Use it when the job needs descriptive profile output.

### `data_quality_learn_op`

Use it when the job should infer candidate rules from baseline data.

## Op Factories

Use `create_check_op`, `create_profile_op`, and `create_learn_op` when you want preconfigured ops without rewriting boilerplate.

## Op Configuration

`CheckOpConfig`, `ProfileOpConfig`, and `LearnOpConfig` expose operation-level configuration surfaces for explicit job composition.

## Preset Configurations

`STRICT_CHECK_CONFIG` and `LENIENT_CHECK_CONFIG` are useful when teams want named policy modes inside job code.

## Op Chaining

Ops are a strong fit when you want separate stages for:

- load
- validate
- profile
- decide
- notify or persist

## Inputs And Outputs

The op outputs follow the same shared result contract used across the repository, with Dagster-native wrapping where appropriate.


## Related Reading

- [Resources](resources.md)
- [Assets and Asset Checks](assets.md)
- [Recipes](recipes.md)
- [워크플로우 Pipelines]
