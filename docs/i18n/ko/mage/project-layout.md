!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Mage Project Layout
---

# Mage Project Layout

Truthound works best when quality blocks are treated as first-class stages in the Mage
pipeline, not as anonymous helper calls buried inside unrelated transformations.

## Recommended Structure

```text
my_mage_project/
├── io_config.yaml
├── data_loaders/
├── transformers/
│   ├── validate_users.py
│   ├── profile_orders.py
│   └── learn_baseline.py
├── sensors/
│   └── quality_gate.py
└── custom/
    └── truthound_helpers.py
```

## Suggested Responsibility Split

- loader blocks: fetch or read the dataset
- transformer blocks: run Truthound checks, profiles, or schema learning
- sensor/condition blocks: decide whether downstream work should continue
- custom helpers: shared block config factories and formatting helpers

## Why This Layout Helps

- quality logic stays visible in Mage's execution graph
- teams can reason about 데이터 품질 separately from business transforms
- it becomes easier to standardize block-level alerting and SLA behavior

## Block Types To Use

### Transformer Blocks

Use these for the main operation:

- `CheckTransformer`
- `ProfileTransformer`
- `LearnTransformer`

### Gate Blocks

Use these when you need control flow:

- `DataQualitySensor`
- `QualityGateSensor`
- `DataQualityCondition`

## Production Pattern

For production Mage projects:

- keep reusable `CheckBlockConfig` and `SensorBlockConfig` builders in one shared file
- avoid hardcoding credentials or connection strings in block code
- keep block inputs and outputs explicit so run history is easy to inspect

## Related Pages

- [`io_config.yaml`](io-config.md)
- [Recipes](recipes.md)
- [Troubleshooting](troubleshooting.md)
