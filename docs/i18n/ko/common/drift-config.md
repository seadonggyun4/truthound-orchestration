!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

# Drift and Anomaly Config

This page describes the configuration objects used by advanced drift, anomaly, and
streaming 워크플로우s.

## Who This Is For

- operators enabling advanced quality monitoring beyond deterministic checks
- maintainers wiring platform-specific drift or anomaly tasks
- teams standardizing reusable config objects across 오케스트레이션 hosts

## `DriftConfig`

`DriftConfig` controls how a baseline is compared with a current dataset.

```python
from common.base import DriftConfig

config = DriftConfig(
    method="ks",
    columns=("revenue", "count"),
    threshold=0.05,
    min_severity="medium",
    timeout_seconds=60,
    extra={"n_permutations": 1000},
)
```

### Builder Pattern

```python
config = DriftConfig()
config = config.with_method("psi")
config = config.with_columns(("revenue", "user_count"))
config = config.with_threshold(0.1)
```

## `AnomalyConfig`

`AnomalyConfig` controls detector choice and sensitivity for anomaly-oriented 워크플로우s.

```python
from common.base import AnomalyConfig

config = AnomalyConfig(
    detector="isolation_forest",
    columns=("amount",),
    contamination=0.05,
    threshold=None,
    timeout_seconds=120,
    extra={"n_estimators": 200},
)
```

### Builder Pattern

```python
config = AnomalyConfig()
config = config.with_detector("lof")
config = config.with_columns(("amount", "frequency"))
config = config.with_contamination(0.03)
```

## `StreamConfig`

`StreamConfig` is used when 검증 happens incrementally or in chunks.

```python
from common.base import StreamConfig

config = StreamConfig(
    batch_size=5000,
    max_batches=100,
    timeout_per_batch_seconds=30.0,
    fail_fast=True,
)
```

### Builder Pattern

```python
config = StreamConfig()
config = config.with_batch_size(10000)
config = config.with_fail_fast(True)
```

## Platform-Specific Config Surfaces

Individual adapters expose platform-native config types on top of the shared concepts:

| Platform | Drift Config | Anomaly Config |
|----------|--------------|----------------|
| Dagster | `DriftOpConfig` | `AnomalyOpConfig` |
| Prefect | `DriftTaskConfig` | `AnomalyTaskConfig` |
| Mage | `DriftBlockConfig` | `AnomalyBlockConfig` |
| Kestra | `DriftScriptConfig` | `AnomalyScriptConfig` |

## Operational Guidance

- keep thresholds explicit and documented
- treat anomaly configs as monitoring policy, not only as model settings
- prefer shared config builders when multiple pipelines should behave consistently

## Related Pages

- [Drift Detection](../engines/drift-detection.md)
- [Anomaly Detection](../engines/anomaly-detection.md)
