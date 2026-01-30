# Drift & Anomaly Config (설정 타입)

Drift Detection과 Anomaly Detection을 위한 설정 타입입니다. 모든 Config는 불변 dataclass이며 빌더 패턴을 지원합니다.

## DriftConfig

```python
from common.base import DriftConfig

config = DriftConfig(
    method="ks",                    # 통계 방법 (기본: "auto")
    columns=("revenue", "count"),   # 대상 컬럼 (기본: None=전체)
    threshold=0.05,                 # 판정 임계값
    min_severity="medium",          # 최소 보고 심각도
    timeout_seconds=60,             # 타임아웃
    extra={"n_permutations": 1000}, # 엔진별 추가 파라미터
)
```

### 빌더 패턴

```python
config = DriftConfig()
config = config.with_method("psi")
config = config.with_columns(("revenue", "user_count"))
config = config.with_threshold(0.1)
```

### 직렬화

```python
d = config.to_dict()
restored = DriftConfig.from_dict(d)
```

## AnomalyConfig

```python
from common.base import AnomalyConfig

config = AnomalyConfig(
    detector="isolation_forest",    # 탐지기 (기본: "isolation_forest")
    columns=("amount",),            # 대상 컬럼 (기본: None=전체)
    contamination=0.05,             # 예상 이상치 비율 (0 < x < 0.5)
    threshold=None,                 # 판정 임계값
    timeout_seconds=120,            # 타임아웃
    extra={"n_estimators": 200},    # 엔진별 추가 파라미터
)
```

### 빌더 패턴

```python
config = AnomalyConfig()
config = config.with_detector("lof")
config = config.with_columns(("amount", "frequency"))
config = config.with_contamination(0.03)
```

### 검증

`contamination`은 `__post_init__`에서 `0 < x < 0.5` 범위로 검증됩니다.

## StreamConfig

```python
from common.base import StreamConfig

config = StreamConfig(
    batch_size=5000,                    # 배치당 레코드 수 (기본: 1000)
    max_batches=100,                    # 최대 배치 수 (기본: None=무제한)
    timeout_per_batch_seconds=30.0,     # 배치별 타임아웃
    fail_fast=True,                     # 첫 실패시 중단 (기본: False)
)
```

### 빌더 패턴

```python
config = StreamConfig()
config = config.with_batch_size(10000)
config = config.with_fail_fast(True)
```

## 플랫폼별 Config

각 플랫폼 패키지는 자체 Config 타입을 제공합니다:

| 플랫폼 | Drift Config | Anomaly Config |
|--------|-------------|----------------|
| Dagster | `DriftOpConfig` | `AnomalyOpConfig` |
| Prefect | `DriftTaskConfig` | `AnomalyTaskConfig` |
| Mage | `DriftBlockConfig` | `AnomalyBlockConfig` |
| Kestra | `DriftScriptConfig` | `AnomalyScriptConfig` |

모든 플랫폼 Config는 동일한 빌더 패턴과 직렬화 인터페이스를 제공합니다.
