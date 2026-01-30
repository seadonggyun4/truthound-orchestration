# Anomaly Detection (이상탐지)

ML 기반 이상탐지 기능입니다. Isolation Forest, Z-Score, LOF, Ensemble 탐지기를 지원합니다.

## 개요

- **4개 탐지기**: Isolation Forest, Z-Score, LOF, Ensemble
- **컬럼별 분석**: 각 컬럼에 대해 독립적으로 이상치 점수 산출
- **Protocol 기반**: `AnomalyDetectionEngine` Protocol을 구현하는 엔진에서만 사용 가능
- **Truthound 지원**: TruthoundEngine이 기본적으로 모든 탐지기를 지원

## 빠른 시작

```python
from common.engines import TruthoundEngine
import polars as pl

engine = TruthoundEngine()
data = pl.read_parquet("data.parquet")

result = engine.detect_anomalies(data)

if result.has_anomalies:
    print(f"이상치 감지! rate={result.anomaly_rate:.2%}")
    for score in result.anomalies:
        print(f"  {score.column}: score={score.score:.4f}, anomaly={score.is_anomaly}")
```

## API

### `detect_anomalies(data, *, detector, columns, contamination, **kwargs)`

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `data` | Any | (필수) | 분석 대상 데이터셋 |
| `detector` | str | `"isolation_forest"` | 탐지기 이름 |
| `columns` | Sequence[str] \| None | None | 검사 대상 컬럼 (None=전체) |
| `contamination` | float | 0.05 | 예상 이상치 비율 (0 < x < 0.5) |

### 반환값: `AnomalyResult`

| 속성 | 타입 | 설명 |
|------|------|------|
| `status` | AnomalyStatus | NORMAL, ANOMALY_DETECTED, WARNING, ERROR |
| `has_anomalies` | bool | 이상치 감지 여부 (property) |
| `anomaly_rate` | float | 이상치 비율 (property) |
| `anomalies` | tuple[AnomalyScore] | 컬럼별 이상치 점수 |
| `anomalous_row_count` | int | 이상치 행 수 |
| `total_row_count` | int | 전체 행 수 |
| `detector` | str | 사용된 탐지기 |
| `execution_time_ms` | float | 실행 시간 |

### `AnomalyScore` 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `column` | str | 컬럼명 |
| `score` | float | 이상치 점수 |
| `threshold` | float | 판정 임계값 |
| `is_anomaly` | bool | 이상치 여부 |
| `detector` | str | 사용된 탐지기 |

## Detector 비교

| Detector | 알고리즘 | 장점 | 단점 | 용도 |
|----------|---------|------|------|------|
| `isolation_forest` | Isolation Forest | 고차원, 대용량 효율적 | 하이퍼파라미터 민감 | 범용 이상탐지 |
| `z_score` | Z-Score | 단순, 해석 용이 | 정규분포 가정 | 정규분포 데이터 |
| `lof` | Local Outlier Factor | 밀도 기반, 국소 이상치 감지 | 계산 비용 높음 | 군집형 데이터 |
| `ensemble` | 다수결 앙상블 | 높은 정확도, 강건함 | 느림 | 정확도 우선 |

## AnomalyConfig

```python
from common.base import AnomalyConfig

config = AnomalyConfig(detector="ensemble", contamination=0.03)
config = config.with_detector("lof").with_columns(("amount",)).with_contamination(0.01)

d = config.to_dict()
restored = AnomalyConfig.from_dict(d)
```

## 엔진 설정

```python
from common.engines import TruthoundEngineConfig, TruthoundEngine

config = TruthoundEngineConfig().with_anomaly_defaults(
    detector="isolation_forest",
    contamination=0.05,
)
engine = TruthoundEngine(config=config)
```

## Feature Detection

```python
from common.engines.base import supports_anomaly

assert supports_anomaly(TruthoundEngine()) is True
assert supports_anomaly(GreatExpectationsAdapter()) is False
```

## 플랫폼 통합

### Airflow

```python
from truthound_airflow.operators.anomaly import DataQualityAnomalyOperator

op = DataQualityAnomalyOperator(
    task_id="anomaly_check",
    data_path="s3://bucket/data.parquet",
    detector="isolation_forest",
    contamination=0.05,
    fail_on_anomaly=True,
)
```

### Dagster

```python
from truthound_dagster.ops.anomaly import data_quality_anomaly_op, create_anomaly_op

result = data_quality_anomaly_op(context, data=df)
my_op = create_anomaly_op(detector="z_score", contamination=0.03)
```

### Prefect

```python
from truthound_prefect.tasks.anomaly import data_quality_anomaly_task

result = await data_quality_anomaly_task(
    block=dq_block,
    data_path="data.parquet",
    detector="isolation_forest",
)
```

### Mage

```python
from truthound_mage.blocks.anomaly import AnomalyTransformer, create_anomaly_transformer

transformer = create_anomaly_transformer(detector="lof")
result = transformer.execute(data)
```

### Kestra

```python
from truthound_kestra.scripts.anomaly import anomaly_detection_script

anomaly_detection_script(data_path="data.parquet", detector="ensemble")
```

### dbt

```yaml
models:
  - name: transactions
    tests:
      - truthound_anomaly_zscore:
          column: amount
          threshold: 3.0
      - truthound_anomaly_iqr:
          column: amount
          factor: 1.5
```

## 직렬화

```python
result_dict = result.to_dict()
result_json = result.to_json()
restored = AnomalyResult.from_dict(result_dict)
```

## 배치 처리

```python
from common.engines import BatchExecutor

executor = BatchExecutor(engine, config)
batch_result = executor.anomaly_batch(datasets, detector="isolation_forest")
batch_result = executor.anomaly_chunked(large_data, detector="z_score")
```

## 멀티 엔진 집계

```python
from common.engines import aggregate_anomaly_results, AggregationConfig, ResultAggregationStrategy

# CONSENSUS 전략: 다수 엔진이 동의하는 anomaly만 유지
config = AggregationConfig(strategy=ResultAggregationStrategy.CONSENSUS)
aggregated = aggregate_anomaly_results(engine_results, config=config)
```
