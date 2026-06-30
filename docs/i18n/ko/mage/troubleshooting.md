!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Mage Troubleshooting
---

# Mage Troubleshooting

## `io_config.yaml` Is Not Being Picked Up

Check:

- the file name is `io_config.yaml` or `io_config.yml`
- the block is running from a project directory or child directory
- the expected source name exists in the file

## The Block Fails During Engine Creation

Mage runs the shared preflight before creating an engine. If preflight fails, inspect:

- engine name
- runtime context assumptions
- missing source metadata
- compatibility rules from the shared runtime layer

## The Quality Result Is Hard To Consume Downstream

Use `result_dict` and the shared serialization helpers instead of passing raw engine
objects through unrelated Mage blocks.

## A Sensor Never Opens The Gate

Check the thresholds:

- `min_pass_rate`
- `max_failure_rate`
- required row-count constraints
- required columns

Sensors are often misconfigured because rollout thresholds are copied from another
pipeline without adapting them to the current dataset.

## Related Pages

- [Shared Runtime Overview](../common/index.md)
- [Observability and Resilience](../common/observability-resilience.md)
