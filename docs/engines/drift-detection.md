# Drift Detection (데이터 드리프트 탐지)

데이터 분포 변화를 통계적으로 감지합니다. baseline 데이터와 current 데이터를 비교하여 컬럼별 드리프트를 탐지합니다.

## 개요

- **14개 통계 방법**: KS, PSI, Chi2, KL, JS, Wasserstein, Hellinger, Bhattacharyya, TV, Energy, MMD, CVM, Anderson-Darling, Auto
- **컬럼별 분석**: 각 컬럼에 대해 독립적으로 드리프트 탐지
- **Protocol 기반**: `DriftDetectionEngine` Protocol을 구현하는 엔진에서만 사용 가능
- **Truthound 지원**: TruthoundEngine이 기본적으로 모든 drift 방법을 지원

## 빠른 시작

```python
from common.engines import TruthoundEngine
import polars as pl

engine = TruthoundEngine()

baseline = pl.read_parquet("baseline.parquet")
current = pl.read_parquet("current.parquet")

result = engine.detect_drift(baseline, current)

if result.is_drifted:
    print(f"드리프트 감지! {result.drifted_count}/{result.total_columns} 컬럼")
    for col in result.drifted_columns:
        print(f"  {col.column}: {col.method.name} stat={col.statistic:.4f}")
```

## API

### `detect_drift(baseline, current, *, method, columns, threshold, **kwargs)`

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `baseline` | Any | (필수) | 기준 데이터셋 |
| `current` | Any | (필수) | 비교 대상 데이터셋 |
| `method` | str | `"auto"` | 통계 방법 |
| `columns` | Sequence[str] \| None | None | 검사 대상 컬럼 (None=전체) |
| `threshold` | float \| None | None | 드리프트 판정 임계값 |

### 반환값: `DriftResult`

| 속성 | 타입 | 설명 |
|------|------|------|
| `status` | DriftStatus | NO_DRIFT, DRIFT_DETECTED, WARNING, ERROR |
| `is_drifted` | bool | 드리프트 감지 여부 (property) |
| `drift_rate` | float | 드리프트 비율 (property) |
| `drifted_columns` | tuple[ColumnDrift] | 컬럼별 결과 |
| `total_columns` | int | 전체 컬럼 수 |
| `drifted_count` | int | 드리프트 감지 컬럼 수 |
| `method` | DriftMethod | 사용된 방법 |
| `execution_time_ms` | float | 실행 시간 |

### `ColumnDrift` 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `column` | str | 컬럼명 |
| `method` | DriftMethod | 사용된 통계 방법 |
| `statistic` | float | 검정 통계량 |
| `p_value` | float \| None | p-value |
| `threshold` | float | 판정 임계값 |
| `is_drifted` | bool | 드리프트 여부 |
| `severity` | Severity | 심각도 |
| `baseline_stats` | dict | 기준 데이터 통계 |
| `current_stats` | dict | 현재 데이터 통계 |

## 통계 방법

### 연속형 데이터

| Method | 설명 | 특징 |
|--------|------|------|
| `ks` | Kolmogorov-Smirnov | 가장 일반적, 분포 형태에 민감 |
| `wasserstein` | Earth Mover's Distance | 분포 간 "이동 비용" 측정 |
| `anderson_darling` | Anderson-Darling | 꼬리 분포에 민감 |
| `cvm` | Cramér-von Mises | 누적분포 차이의 제곱 적분 |
| `energy` | Energy Distance | 다변량 분포 비교 가능 |

### 범주형 데이터

| Method | 설명 | 특징 |
|--------|------|------|
| `chi2` | 카이제곱 검정 | 범주형 분포 표준 검정 |
| `psi` | Population Stability Index | 신용평가 산업 표준 |

### 정보 이론 기반

| Method | 설명 | 특징 |
|--------|------|------|
| `kl` | KL Divergence | 비대칭, 방향성 있음 |
| `js` | Jensen-Shannon | KL의 대칭 버전 |
| `hellinger` | Hellinger Distance | [0,1] 범위, 해석 용이 |
| `bhattacharyya` | Bhattacharyya | 분포 겹침 측정 |
| `tv` | Total Variation | 최대 확률 차이 |

### 커널 기반

| Method | 설명 | 특징 |
|--------|------|------|
| `mmd` | Maximum Mean Discrepancy | 고차원 데이터에 적합 |

## DriftConfig

```python
from common.base import DriftConfig

# 기본 생성
config = DriftConfig(method="ks", threshold=0.05)

# 빌더 패턴
config = DriftConfig().with_method("psi").with_columns(("revenue",)).with_threshold(0.1)

# 직렬화
d = config.to_dict()
restored = DriftConfig.from_dict(d)
```

## 엔진 설정

```python
from common.engines import TruthoundEngineConfig, TruthoundEngine

config = TruthoundEngineConfig().with_drift_defaults(
    method="ks",
    threshold=0.05,
)
engine = TruthoundEngine(config=config)
```

## Feature Detection

```python
from common.engines.base import supports_drift

engine = TruthoundEngine()
assert supports_drift(engine) is True

from common.engines import GreatExpectationsAdapter
ge = GreatExpectationsAdapter()
assert supports_drift(ge) is False
```

## 플랫폼 통합

### Airflow

```python
from truthound_airflow.operators.drift import DataQualityDriftOperator

op = DataQualityDriftOperator(
    task_id="drift_check",
    baseline_data_path="s3://bucket/baseline.parquet",
    current_data_path="s3://bucket/current.parquet",
    method="ks",
    threshold=0.05,
    fail_on_drift=True,
)
```

### Dagster

```python
from truthound_dagster.ops.drift import data_quality_drift_op, create_drift_op

# Op으로 직접 사용
result = data_quality_drift_op(context, baseline=baseline_df, current=current_df)

# 팩토리로 사전 구성된 Op 생성
my_drift_op = create_drift_op(method="psi", threshold=0.1)
```

### Prefect

```python
from truthound_prefect.tasks.drift import data_quality_drift_task

result = await data_quality_drift_task(
    block=dq_block,
    baseline_data_path="baseline.parquet",
    current_data_path="current.parquet",
    method="ks",
)
```

### Mage

```python
from truthound_mage.blocks.drift import DriftTransformer, create_drift_transformer

transformer = create_drift_transformer(method="ks", threshold=0.05)
result = transformer.execute(baseline_data, current_data)
```

### Kestra

```python
from truthound_kestra.scripts.drift import drift_detection_script

drift_detection_script(
    baseline_data_path="baseline.parquet",
    current_data_path="current.parquet",
    method="ks",
)
```

### dbt

```yaml
# schema.yml
models:
  - name: orders
    tests:
      - truthound_drift_mean:
          column: revenue
          baseline_model: ref('orders_baseline')
          threshold: 0.1
      - truthound_drift_null_rate:
          column: email
          baseline_model: ref('orders_baseline')
          threshold: 0.05
```

## 직렬화

```python
result_dict = result.to_dict()
result_json = result.to_json()
restored = DriftResult.from_dict(result_dict)
```

## 배치 처리

```python
from common.engines import BatchExecutor

executor = BatchExecutor(engine, config)
batch_result = executor.drift_batch(baseline_list, current_list, method="ks")
batch_result = executor.drift_chunked(large_baseline, large_current)
```

## 멀티 엔진 집계

```python
from common.engines import aggregate_drift_results

aggregated = aggregate_drift_results({
    "engine1": drift_result_1,
    "engine2": drift_result_2,
})
```
