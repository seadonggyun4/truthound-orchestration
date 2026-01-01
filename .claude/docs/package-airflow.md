# Package: truthound-airflow

> **Last Updated:** 2025-12-31
> **Document Version:** 2.0.0
> **Package Version:** 0.1.0
> **Status:** Implementation Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [DataQualityCheckOperator](#dataqualitycheckoperator)
5. [DataQualityProfileOperator](#dataqualityprofileoperator)
6. [DataQualityLearnOperator](#dataqualitylearnoperator)
7. [DataQualitySensor](#dataqualitysensor)
8. [DataQualityHook](#dataqualityhook)
9. [XCom Integration](#xcom-integration)
10. [Connection Configuration](#connection-configuration)
11. [Example DAGs](#example-dags)
12. [Testing Strategy](#testing-strategy)
13. [pyproject.toml](#pyprojecttoml)

---

## Overview

### Purpose
`truthound-airflow`는 Apache Airflow용 **범용 데이터 품질** Provider 패키지입니다. `DataQualityEngine` Protocol을 통해 **Truthound, Great Expectations, Pandera 등 다양한 엔진**을 지원하며, Truthound가 기본 엔진으로 제공됩니다.

### Key Features

| Feature | Description |
|---------|-------------|
| **Engine-Agnostic** | DataQualityEngine Protocol로 다양한 엔진 지원 |
| **Native Operators** | Airflow Operator 패턴 완벽 지원 |
| **XCom Integration** | 결과를 XCom으로 전달하여 downstream 활용 |
| **Connection Management** | Airflow Connection으로 설정 중앙화 |
| **Template Support** | Jinja 템플릿으로 동적 파라미터 지원 |
| **Sensor Support** | 품질 조건 충족까지 대기 |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Airflow DAG                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │  Extract      │─▶│  Transform    │─▶│  Load         │   │
│  └───────────────┘  └───────────────┘  └───────┬───────┘   │
│                                                 │           │
│                                                 ▼           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            DataQualityCheckOperator                  │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  • 데이터 로드 (Hook 사용)                    │    │   │
│  │  │  • 품질 검증 실행 (engine.check)             │    │   │
│  │  │  • 결과 XCom 푸시                            │    │   │
│  │  │  • 실패 시 예외 발생                         │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │                        │                             │   │
│  │                        ▼                             │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │         DataQualityEngine (Pluggable)        │    │   │
│  │  │   Truthound | Great Expectations | Custom    │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────┐  ┌───────────────┐                      │
│  │  Notify       │  │  Dashboard    │                      │
│  └───────────────┘  └───────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### PyPI

```bash
pip install truthound-airflow
```

### With Extras

```bash
# PostgreSQL 지원
pip install truthound-airflow[postgres]

# BigQuery 지원
pip install truthound-airflow[bigquery]

# 전체 설치
pip install truthound-airflow[all]
```

### Requirements

| Dependency | Version |
|------------|---------|
| Python | >= 3.11 |
| apache-airflow | >= 2.6.0 |
| truthound | >= 1.0.0 |
| polars | >= 0.20.0 |

---

## Components

### Package Structure

```
packages/airflow/
├── pyproject.toml
├── README.md
├── src/
│   └── truthound_airflow/
│       ├── __init__.py           # Public API exports
│       ├── version.py            # Package version
│       ├── operators/
│       │   ├── __init__.py
│       │   ├── base.py           # BaseDataQualityOperator
│       │   ├── check.py          # DataQualityCheckOperator
│       │   ├── profile.py        # DataQualityProfileOperator
│       │   └── learn.py          # DataQualityLearnOperator
│       ├── sensors/
│       │   ├── __init__.py
│       │   └── quality.py        # DataQualitySensor
│       ├── hooks/
│       │   ├── __init__.py
│       │   └── base.py           # DataQualityHook
│       └── utils/
│           ├── __init__.py
│           ├── serialization.py  # XCom serialization
│           └── connection.py     # Connection helpers
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_operators/
    ├── test_sensors/
    └── test_hooks/
```

### Public API

```python
# truthound_airflow/__init__.py
from truthound_airflow.operators.check import DataQualityCheckOperator
from truthound_airflow.operators.profile import DataQualityProfileOperator
from truthound_airflow.operators.learn import DataQualityLearnOperator
from truthound_airflow.sensors.quality import DataQualitySensor
from truthound_airflow.hooks.base import DataQualityHook

__all__ = [
    "DataQualityCheckOperator",
    "DataQualityProfileOperator",
    "DataQualityLearnOperator",
    "DataQualitySensor",
    "DataQualityHook",
]

__version__ = "0.1.0"
```

---

## DataQualityCheckOperator

### Specification

```python
from typing import Any, Sequence
from airflow.models import BaseOperator
from airflow.utils.context import Context


class DataQualityCheckOperator(BaseOperator):
    """
    데이터 품질 검증을 실행하는 Operator.

    이 Operator는 지정된 데이터 소스에서 데이터를 로드하고,
    정의된 규칙에 따라 품질 검증을 수행합니다. 검증 결과는
    XCom으로 전달되어 downstream task에서 활용할 수 있습니다.

    Parameters
    ----------
    rules : list[dict[str, Any]]
        적용할 검증 규칙 목록. Jinja 템플릿 지원.
        예: [{"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"}]

    data_path : str | None
        검증할 데이터 파일 경로. 템플릿 지원.
        예: "s3://bucket/data/{{ ds }}/events.parquet"

    sql : str | None
        데이터를 가져올 SQL 쿼리. data_path와 상호 배타적.
        예: "SELECT * FROM events WHERE date = '{{ ds }}'"

    connection_id : str
        Airflow Connection ID. 기본값: "truthound_default"

    fail_on_error : bool
        검증 실패 시 Task 실패 여부. 기본값: True

    warning_threshold : float | None
        경고로 처리할 실패율 임계값 (0.0-1.0)
        예: 0.05 (5% 미만 실패 시 경고만)

    sample_size : int | None
        샘플링할 행 수. None이면 전체 데이터.

    timeout_seconds : int
        검증 타임아웃 (초). 기본값: 300

    xcom_push_key : str
        XCom에 저장할 키 이름. 기본값: "data_quality_result"

    Attributes
    ----------
    template_fields : Sequence[str]
        Jinja 템플릿을 지원하는 필드 목록

    template_ext : Sequence[str]
        템플릿 파일 확장자

    ui_color : str
        Airflow UI에서 표시할 색상

    Examples
    --------
    기본 사용:

    >>> check_quality = DataQualityCheckOperator(
    ...     task_id="check_data_quality",
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
    ...         {"column": "age", "type": "in_range", "min": 0, "max": 150},
    ...     ],
    ...     data_path="s3://bucket/data/{{ ds }}/users.parquet",
    ...     connection_id="truthound_s3",
    ... )

    SQL 쿼리 사용:

    >>> check_from_db = DataQualityCheckOperator(
    ...     task_id="check_db_quality",
    ...     rules=[{"column": "amount", "type": "in_range", "min": 0}],
    ...     sql="SELECT * FROM transactions WHERE date = '{{ ds }}'",
    ...     connection_id="truthound_postgres",
    ... )

    경고 임계값 사용:

    >>> check_with_warning = DataQualityCheckOperator(
    ...     task_id="check_with_threshold",
    ...     rules=[{"column": "status", "type": "in_set", "values": ["A", "B", "C"]}],
    ...     data_path="/data/events.parquet",
    ...     warning_threshold=0.01,  # 1% 미만 실패 시 경고만
    ... )

    Raises
    ------
    AirflowException
        fail_on_error=True이고 검증이 실패한 경우
    AirflowSkipException
        데이터가 없는 경우 (선택적)

    Notes
    -----
    - data_path와 sql은 상호 배타적입니다
    - 검증 결과는 항상 XCom으로 푸시됩니다
    - 대용량 데이터의 경우 sample_size 사용을 권장합니다
    """

    template_fields: Sequence[str] = (
        "rules",
        "data_path",
        "sql",
        "connection_id",
    )
    template_ext: Sequence[str] = (".sql", ".json")
    ui_color: str = "#4A90D9"
    ui_fgcolor: str = "#FFFFFF"

    def __init__(
        self,
        *,
        rules: list[dict[str, Any]],
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        fail_on_error: bool = True,
        warning_threshold: float | None = None,
        sample_size: int | None = None,
        timeout_seconds: int = 300,
        xcom_push_key: str = "data_quality_result",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validation
        if data_path and sql:
            raise ValueError("Cannot specify both data_path and sql")
        if not data_path and not sql:
            raise ValueError("Must specify either data_path or sql")
        if warning_threshold is not None and not 0 <= warning_threshold <= 1:
            raise ValueError("warning_threshold must be between 0 and 1")

        self.rules = rules
        self.data_path = data_path
        self.sql = sql
        self.connection_id = connection_id
        self.fail_on_error = fail_on_error
        self.warning_threshold = warning_threshold
        self.sample_size = sample_size
        self.timeout_seconds = timeout_seconds
        self.xcom_push_key = xcom_push_key

    def execute(self, context: Context) -> dict[str, Any]:
        """
        Operator 실행 메인 로직.

        Parameters
        ----------
        context : Context
            Airflow 실행 컨텍스트

        Returns
        -------
        dict[str, Any]
            검증 결과 딕셔너리
        """
        import truthound as th
        from truthound_airflow.hooks.base import DataQualityHook

        self.log.info(f"Starting data quality check with {len(self.rules)} rules")

        # Hook 초기화
        hook = DataQualityHook(connection_id=self.connection_id)

        # 데이터 로드
        if self.data_path:
            self.log.info(f"Loading data from: {self.data_path}")
            data = hook.load_data(self.data_path)
        else:
            self.log.info(f"Executing SQL query")
            data = hook.query(self.sql)

        self.log.info(f"Loaded {len(data)} rows")

        # 샘플링
        if self.sample_size and len(data) > self.sample_size:
            self.log.info(f"Sampling {self.sample_size} rows")
            data = data.sample(n=self.sample_size)

        # 검증 실행
        self.log.info("Executing quality check...")
        result = th.check(
            data,
            rules=self.rules,
            fail_on_error=False,  # 자체 처리
            timeout=self.timeout_seconds,
        )

        # 결과 직렬화
        result_dict = self._serialize_result(result)

        # XCom 푸시
        context["ti"].xcom_push(key=self.xcom_push_key, value=result_dict)
        self.log.info(f"Pushed result to XCom with key: {self.xcom_push_key}")

        # 메트릭 로깅
        self._log_metrics(result_dict)

        # 실패 처리
        if not result_dict["is_success"]:
            failure_rate = result_dict["failure_rate"]

            # 경고 임계값 체크
            if self.warning_threshold and failure_rate <= self.warning_threshold:
                self.log.warning(
                    f"Quality check has warnings: {result_dict['failed_count']} "
                    f"failures ({failure_rate:.2%} <= {self.warning_threshold:.2%})"
                )
            elif self.fail_on_error:
                from airflow.exceptions import AirflowException
                raise AirflowException(
                    f"Quality check failed: {result_dict['failed_count']} rules failed "
                    f"({failure_rate:.2%}). Details: {result_dict['failures']}"
                )
            else:
                self.log.error(
                    f"Quality check failed (not raising): {result_dict['failed_count']} failures"
                )

        self.log.info("Quality check completed successfully")
        return result_dict

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        """검증 결과를 XCom 호환 형식으로 직렬화"""
        return {
            "status": result.status.value,
            "is_success": result.is_success,
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "warning_count": result.warning_count,
            "failure_rate": result.failure_rate,
            "failures": [
                {
                    "rule_name": f.rule_name,
                    "column": f.column,
                    "message": f.message,
                    "severity": f.severity.value,
                    "failed_count": f.failed_count,
                    "total_count": f.total_count,
                }
                for f in result.failures
            ],
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
        }

    def _log_metrics(self, result_dict: dict[str, Any]) -> None:
        """검증 메트릭을 로깅"""
        self.log.info(
            f"Quality Check Results: "
            f"status={result_dict['status']}, "
            f"passed={result_dict['passed_count']}, "
            f"failed={result_dict['failed_count']}, "
            f"duration={result_dict['execution_time_ms']:.2f}ms"
        )
```

### Usage Examples

#### Basic Usage

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from truthound_airflow import DataQualityCheckOperator

with DAG(
    dag_id="data_quality_check",
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:

    check_users = DataQualityCheckOperator(
        task_id="check_users_quality",
        rules=[
            {"column": "user_id", "type": "not_null"},
            {"column": "user_id", "type": "unique"},
            {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
            {"column": "age", "type": "in_range", "min": 0, "max": 150},
            {"column": "created_at", "type": "not_future"},
        ],
        data_path="s3://data-lake/users/{{ ds }}/data.parquet",
        connection_id="truthound_s3",
    )
```

#### With SQL Query

```python
check_transactions = DataQualityCheckOperator(
    task_id="check_transactions",
    rules=[
        {"column": "amount", "type": "in_range", "min": 0},
        {"column": "currency", "type": "in_set", "values": ["USD", "EUR", "GBP"]},
        {"column": "user_id", "type": "foreign_key", "reference": "users.id"},
    ],
    sql="""
        SELECT *
        FROM transactions
        WHERE transaction_date = '{{ ds }}'
          AND status = 'completed'
    """,
    connection_id="truthound_postgres",
)
```

#### With Dynamic Rules

```python
from airflow.models import Variable

def get_rules(**context):
    """동적으로 규칙 생성"""
    env = Variable.get("environment", default_var="dev")
    base_rules = [
        {"column": "id", "type": "not_null"},
    ]
    if env == "prod":
        base_rules.append({"column": "id", "type": "unique"})
    return base_rules

check_dynamic = DataQualityCheckOperator(
    task_id="check_dynamic",
    rules="{{ task_instance.xcom_pull(task_ids='get_rules') }}",
    data_path="/data/events.parquet",
)
```

---

## DataQualityProfileOperator

### Specification

```python
class DataQualityProfileOperator(BaseOperator):
    """
    데이터 프로파일링을 실행하는 Operator.

    Parameters
    ----------
    data_path : str | None
        프로파일링할 데이터 경로

    sql : str | None
        데이터 쿼리

    connection_id : str
        Connection ID

    columns : list[str] | None
        프로파일링할 컬럼 (None=전체)

    include_statistics : bool
        통계 포함 여부. 기본값: True

    include_patterns : bool
        패턴 감지 포함 여부. 기본값: True

    include_distributions : bool
        분포 분석 포함 여부. 기본값: True

    sample_size : int | None
        샘플 크기

    xcom_push_key : str
        XCom 키. 기본값: "data_quality_profile"
    """

    template_fields: Sequence[str] = ("data_path", "sql", "columns")
    ui_color: str = "#9B59B6"

    def __init__(
        self,
        *,
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        columns: list[str] | None = None,
        include_statistics: bool = True,
        include_patterns: bool = True,
        include_distributions: bool = True,
        sample_size: int | None = None,
        xcom_push_key: str = "data_quality_profile",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.sql = sql
        self.connection_id = connection_id
        self.columns = columns
        self.include_statistics = include_statistics
        self.include_patterns = include_patterns
        self.include_distributions = include_distributions
        self.sample_size = sample_size
        self.xcom_push_key = xcom_push_key

    def execute(self, context: Context) -> dict[str, Any]:
        """프로파일링 실행"""
        import truthound as th
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook(connection_id=self.connection_id)

        # 데이터 로드
        data = hook.load_data(self.data_path) if self.data_path else hook.query(self.sql)

        # 프로파일링 실행
        profile = th.profile(
            data,
            columns=self.columns,
            include_statistics=self.include_statistics,
            include_patterns=self.include_patterns,
            include_distributions=self.include_distributions,
            sample_size=self.sample_size,
        )

        # 결과 직렬화 및 XCom 푸시
        result_dict = profile.to_dict()
        context["ti"].xcom_push(key=self.xcom_push_key, value=result_dict)

        return result_dict
```

### Usage Example

```python
profile_data = DataQualityProfileOperator(
    task_id="profile_sales_data",
    data_path="s3://bucket/sales/{{ ds }}/data.parquet",
    columns=["amount", "quantity", "discount"],
    include_distributions=True,
    connection_id="truthound_s3",
)
```

---

## DataQualityLearnOperator

### Specification

```python
class DataQualityLearnOperator(BaseOperator):
    """
    데이터에서 스키마와 검증 규칙을 자동 학습하는 Operator.

    Parameters
    ----------
    data_path : str | None
        학습할 데이터 경로

    sql : str | None
        데이터 쿼리

    connection_id : str
        Connection ID

    output_path : str | None
        학습된 스키마 저장 경로

    strictness : str
        학습 엄격도. "strict", "moderate", "lenient"

    xcom_push_key : str
        XCom 키. 기본값: "data_quality_schema"
    """

    template_fields: Sequence[str] = ("data_path", "sql", "output_path")
    ui_color: str = "#2ECC71"

    def __init__(
        self,
        *,
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        output_path: str | None = None,
        strictness: str = "moderate",
        xcom_push_key: str = "data_quality_schema",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.sql = sql
        self.connection_id = connection_id
        self.output_path = output_path
        self.strictness = strictness
        self.xcom_push_key = xcom_push_key

    def execute(self, context: Context) -> dict[str, Any]:
        """스키마 학습 실행"""
        import truthound as th
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook(connection_id=self.connection_id)
        data = hook.load_data(self.data_path) if self.data_path else hook.query(self.sql)

        # 스키마 학습
        schema = th.learn(data, strictness=self.strictness)

        # 저장 (선택적)
        if self.output_path:
            hook.save_json(schema.to_dict(), self.output_path)
            self.log.info(f"Schema saved to: {self.output_path}")

        # XCom 푸시
        result_dict = schema.to_dict()
        context["ti"].xcom_push(key=self.xcom_push_key, value=result_dict)

        return result_dict
```

---

## DataQualitySensor

### Specification

```python
from airflow.sensors.base import BaseSensorOperator


class DataQualitySensor(BaseSensorOperator):
    """
    데이터 품질 조건이 충족될 때까지 대기하는 Sensor.

    Parameters
    ----------
    rules : list[dict[str, Any]]
        충족해야 할 검증 규칙

    data_path : str | None
        검증할 데이터 경로

    sql : str | None
        데이터 쿼리

    connection_id : str
        Connection ID

    min_pass_rate : float
        최소 통과율 (0.0-1.0). 기본값: 1.0

    poke_interval : int
        재시도 간격 (초). 기본값: 60

    timeout : int
        전체 타임아웃 (초). 기본값: 3600

    mode : str
        센서 모드. "poke" 또는 "reschedule"

    Examples
    --------
    >>> wait_for_quality = DataQualitySensor(
    ...     task_id="wait_for_data_quality",
    ...     rules=[{"column": "status", "type": "not_null"}],
    ...     data_path="s3://bucket/data/{{ ds }}/events.parquet",
    ...     min_pass_rate=0.99,
    ...     poke_interval=300,
    ...     timeout=3600,
    ...     mode="reschedule",
    ... )
    """

    template_fields: Sequence[str] = ("rules", "data_path", "sql")
    ui_color: str = "#E67E22"

    def __init__(
        self,
        *,
        rules: list[dict[str, Any]],
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        min_pass_rate: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rules = rules
        self.data_path = data_path
        self.sql = sql
        self.connection_id = connection_id
        self.min_pass_rate = min_pass_rate

    def poke(self, context: Context) -> bool:
        """
        품질 조건 체크.

        Returns
        -------
        bool
            조건 충족 시 True
        """
        import truthound as th
        from truthound_airflow.hooks.base import DataQualityHook

        hook = DataQualityHook(connection_id=self.connection_id)

        try:
            data = hook.load_data(self.data_path) if self.data_path else hook.query(self.sql)
        except FileNotFoundError:
            self.log.info("Data not found yet, will retry...")
            return False

        result = th.check(data, rules=self.rules, fail_on_error=False)

        pass_rate = result.passed_count / (result.passed_count + result.failed_count)
        self.log.info(f"Current pass rate: {pass_rate:.2%}, required: {self.min_pass_rate:.2%}")

        return pass_rate >= self.min_pass_rate
```

### Usage Example

```python
from airflow import DAG
from truthound_airflow import DataQualitySensor, DataQualityCheckOperator

with DAG(dag_id="quality_gated_pipeline") as dag:

    # 데이터 품질이 충족될 때까지 대기
    wait_for_quality = DataQualitySensor(
        task_id="wait_for_quality",
        rules=[
            {"column": "id", "type": "not_null"},
            {"column": "amount", "type": "in_range", "min": 0},
        ],
        data_path="s3://bucket/incoming/{{ ds }}/data.parquet",
        min_pass_rate=0.95,
        poke_interval=300,  # 5분마다 체크
        timeout=7200,       # 2시간 타임아웃
        mode="reschedule",  # 리소스 효율적
    )

    # 품질 충족 후 상세 검증
    full_check = DataQualityCheckOperator(
        task_id="full_quality_check",
        rules=[...],
        data_path="s3://bucket/incoming/{{ ds }}/data.parquet",
    )

    wait_for_quality >> full_check
```

---

## DataQualityHook

### Specification

```python
from airflow.hooks.base import BaseHook
from typing import Any
import polars as pl


class DataQualityHook(BaseHook):
    """
    Truthound 연결 및 데이터 작업을 위한 Hook.

    Parameters
    ----------
    connection_id : str
        Airflow Connection ID

    Attributes
    ----------
    conn_type : str
        Connection 타입

    hook_name : str
        Hook 이름

    Examples
    --------
    >>> hook = DataQualityHook(connection_id="truthound_s3")
    >>> data = hook.load_data("s3://bucket/data.parquet")
    >>> result = hook.query("SELECT * FROM table")
    """

    conn_type = "truthound"
    conn_name_attr = "connection_id"
    hook_name = "Truthound"

    def __init__(
        self,
        connection_id: str = "truthound_default",
    ) -> None:
        super().__init__()
        self.connection_id = connection_id
        self._conn: Any = None

    def get_conn(self) -> Any:
        """
        Connection 객체 반환.

        Returns
        -------
        Any
            Truthound 연결 객체
        """
        if self._conn is None:
            conn = self.get_connection(self.connection_id)
            self._conn = self._create_connection(conn)
        return self._conn

    def _create_connection(self, conn: Any) -> Any:
        """Connection 파라미터에서 연결 생성"""
        import truthound as th

        conn_type = conn.extra_dejson.get("conn_type", "filesystem")

        if conn_type == "s3":
            return th.connect(
                type="s3",
                bucket=conn.host,
                access_key=conn.login,
                secret_key=conn.password,
                region=conn.extra_dejson.get("region", "us-east-1"),
            )
        elif conn_type == "gcs":
            return th.connect(
                type="gcs",
                bucket=conn.host,
                credentials_path=conn.extra_dejson.get("credentials_path"),
            )
        elif conn_type == "postgres":
            return th.connect(
                type="postgres",
                host=conn.host,
                port=conn.port or 5432,
                database=conn.schema,
                user=conn.login,
                password=conn.password,
            )
        elif conn_type == "bigquery":
            return th.connect(
                type="bigquery",
                project=conn.extra_dejson.get("project"),
                credentials_path=conn.extra_dejson.get("credentials_path"),
            )
        else:
            # 기본: 파일시스템
            return th.connect(type="filesystem")

    def load_data(
        self,
        path: str,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        경로에서 데이터 로드.

        Parameters
        ----------
        path : str
            데이터 경로 (로컬, S3, GCS 등)
        **kwargs
            추가 로드 옵션

        Returns
        -------
        pl.DataFrame
            로드된 데이터
        """
        conn = self.get_conn()
        self.log.info(f"Loading data from: {path}")

        return conn.read(path, **kwargs)

    def query(
        self,
        sql: str,
        **kwargs: Any,
    ) -> pl.DataFrame:
        """
        SQL 쿼리 실행.

        Parameters
        ----------
        sql : str
            실행할 SQL 쿼리
        **kwargs
            추가 쿼리 옵션

        Returns
        -------
        pl.DataFrame
            쿼리 결과
        """
        conn = self.get_conn()
        self.log.info(f"Executing query: {sql[:100]}...")

        return conn.query(sql, **kwargs)

    def save_json(
        self,
        data: dict[str, Any],
        path: str,
    ) -> None:
        """JSON 데이터를 파일로 저장"""
        conn = self.get_conn()
        conn.write_json(data, path)

    def test_connection(self) -> tuple[bool, str]:
        """
        Connection 테스트.

        Returns
        -------
        tuple[bool, str]
            (성공 여부, 메시지)
        """
        try:
            conn = self.get_conn()
            conn.ping()
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)
```

---

## XCom Integration

### XCom Data Flow

```
┌──────────────────┐    XCom Push     ┌──────────────────┐
│  Check Operator  │─────────────────▶│   truthound_     │
│                  │                  │     result       │
└──────────────────┘                  └────────┬─────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
          ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
          │   Notify Task    │      │  Dashboard Task  │      │   Branch Task    │
          │  (Slack, Email)  │      │  (Metrics Push)  │      │  (Conditional)   │
          └──────────────────┘      └──────────────────┘      └──────────────────┘
```

### XCom Schema

```python
# data_quality_result XCom 스키마
{
    "status": "passed" | "failed" | "warning",
    "is_success": bool,
    "passed_count": int,
    "failed_count": int,
    "warning_count": int,
    "failure_rate": float,  # 0.0 - 1.0
    "failures": [
        {
            "rule_name": str,
            "column": str | None,
            "message": str,
            "severity": "critical" | "high" | "medium" | "low" | "info",
            "failed_count": int,
            "total_count": int,
        }
    ],
    "execution_time_ms": float,
    "timestamp": str,  # ISO format
}
```

### XCom Usage Example

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from truthound_airflow import DataQualityCheckOperator

def notify_failure(**context):
    """실패 시 알림 전송"""
    result = context["ti"].xcom_pull(
        task_ids="check_quality",
        key="data_quality_result",
    )

    if not result["is_success"]:
        # Slack 알림 등
        failures = result["failures"]
        message = f"Quality check failed: {len(failures)} issues found"
        send_slack_notification(message)

def choose_branch(**context):
    """결과에 따라 분기"""
    result = context["ti"].xcom_pull(
        task_ids="check_quality",
        key="data_quality_result",
    )

    if result["is_success"]:
        return "process_data"
    elif result["failure_rate"] < 0.05:
        return "process_with_warning"
    else:
        return "reject_data"

with DAG(dag_id="quality_branching") as dag:

    check = DataQualityCheckOperator(
        task_id="check_quality",
        rules=[...],
        data_path="...",
        fail_on_error=False,  # 자체 분기 처리
    )

    branch = BranchPythonOperator(
        task_id="branch_on_quality",
        python_callable=choose_branch,
    )

    notify = PythonOperator(
        task_id="notify_if_failed",
        python_callable=notify_failure,
        trigger_rule="all_done",
    )

    check >> branch >> [process_data, process_with_warning, reject_data]
    check >> notify
```

---

## Connection Configuration

### Connection Types

| Type | Host | Port | Schema | Login | Password | Extra |
|------|------|------|--------|-------|----------|-------|
| **S3** | bucket-name | - | - | access_key | secret_key | `{"conn_type": "s3", "region": "us-east-1"}` |
| **GCS** | bucket-name | - | - | - | - | `{"conn_type": "gcs", "credentials_path": "/path/to/creds.json"}` |
| **PostgreSQL** | hostname | 5432 | database | user | password | `{"conn_type": "postgres"}` |
| **BigQuery** | - | - | - | - | - | `{"conn_type": "bigquery", "project": "my-project", "credentials_path": "..."}` |
| **Filesystem** | - | - | - | - | - | `{"conn_type": "filesystem"}` |

### UI Configuration Example

```
Connection Id: truthound_s3
Connection Type: Truthound
Host: my-data-bucket
Login: AKIAIOSFODNN7EXAMPLE
Password: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Extra: {"conn_type": "s3", "region": "ap-northeast-2"}
```

### Environment Variable Configuration

```bash
# S3 Connection
export AIRFLOW_CONN_TRUTHOUND_S3='{"conn_type": "truthound", "host": "my-bucket", "login": "ACCESS_KEY", "password": "SECRET_KEY", "extra": {"conn_type": "s3", "region": "us-east-1"}}'

# PostgreSQL Connection
export AIRFLOW_CONN_TRUTHOUND_PG='{"conn_type": "truthound", "host": "db.example.com", "port": 5432, "schema": "analytics", "login": "user", "password": "pass", "extra": {"conn_type": "postgres"}}'
```

---

## Example DAGs

### Complete ETL with Quality Gates

```python
"""
예제: 품질 게이트가 포함된 완전한 ETL 파이프라인.

이 DAG는 다음을 수행합니다:
1. 외부 시스템에서 데이터 추출
2. 원본 데이터 품질 검증
3. 데이터 변환
4. 변환된 데이터 품질 검증
5. 데이터 로드
6. 결과 알림
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from truthound_airflow import (
    DataQualityCheckOperator,
    DataQualityProfileOperator,
    DataQualitySensor,
)

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["data-alerts@company.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# 품질 규칙 정의
RAW_DATA_RULES = [
    {"column": "user_id", "type": "not_null"},
    {"column": "user_id", "type": "regex", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"},
    {"column": "event_type", "type": "in_set", "values": ["click", "view", "purchase"]},
    {"column": "timestamp", "type": "not_null"},
    {"column": "timestamp", "type": "not_future"},
    {"column": "amount", "type": "in_range", "min": 0, "max": 1000000},
]

TRANSFORMED_DATA_RULES = [
    {"column": "user_id", "type": "not_null"},
    {"column": "daily_total", "type": "in_range", "min": 0},
    {"column": "event_count", "type": "in_range", "min": 1, "max": 10000},
    {"column": "date", "type": "date_format", "format": "%Y-%m-%d"},
]

with DAG(
    dag_id="quality_gated_etl",
    default_args=default_args,
    description="ETL pipeline with Truthound quality gates",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["etl", "quality", "truthound"],
) as dag:

    # 1. 데이터 도착 대기
    wait_for_data = DataQualitySensor(
        task_id="wait_for_source_data",
        rules=[{"column": "user_id", "type": "not_null"}],
        data_path="s3://raw-data/events/{{ ds }}/data.parquet",
        connection_id="truthound_s3",
        min_pass_rate=0.9,
        poke_interval=300,
        timeout=7200,
        mode="reschedule",
    )

    # 2. 원본 데이터 프로파일링
    profile_raw = DataQualityProfileOperator(
        task_id="profile_raw_data",
        data_path="s3://raw-data/events/{{ ds }}/data.parquet",
        connection_id="truthound_s3",
        include_distributions=True,
    )

    # 3. 원본 데이터 품질 검증
    check_raw = DataQualityCheckOperator(
        task_id="check_raw_data_quality",
        rules=RAW_DATA_RULES,
        data_path="s3://raw-data/events/{{ ds }}/data.parquet",
        connection_id="truthound_s3",
        fail_on_error=True,
        warning_threshold=0.01,
    )

    # 4. 데이터 변환 (예: Spark Job 실행)
    transform_data = PythonOperator(
        task_id="transform_data",
        python_callable=lambda: print("Transforming data..."),
    )

    # 5. 변환 데이터 품질 검증
    check_transformed = DataQualityCheckOperator(
        task_id="check_transformed_quality",
        rules=TRANSFORMED_DATA_RULES,
        data_path="s3://processed-data/daily_summary/{{ ds }}/data.parquet",
        connection_id="truthound_s3",
        fail_on_error=True,
    )

    # 6. 데이터 로드
    load_to_warehouse = SQLExecuteQueryOperator(
        task_id="load_to_warehouse",
        conn_id="warehouse",
        sql="COPY INTO daily_summary FROM 's3://processed-data/daily_summary/{{ ds }}/'",
    )

    # 7. 알림
    def send_success_notification(**context):
        raw_result = context["ti"].xcom_pull(
            task_ids="check_raw_data_quality",
            key="data_quality_result",
        )
        transformed_result = context["ti"].xcom_pull(
            task_ids="check_transformed_quality",
            key="data_quality_result",
        )
        print(f"Raw data: {raw_result['passed_count']} checks passed")
        print(f"Transformed: {transformed_result['passed_count']} checks passed")

    notify = PythonOperator(
        task_id="send_notification",
        python_callable=send_success_notification,
    )

    # DAG 의존성
    wait_for_data >> profile_raw >> check_raw >> transform_data
    transform_data >> check_transformed >> load_to_warehouse >> notify
```

### Schema Evolution Detection

```python
"""
예제: 스키마 변경 감지 파이프라인.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from truthound_airflow import DataQualityLearnOperator, DataQualityCheckOperator

def compare_schemas(**context):
    """이전 스키마와 현재 스키마 비교"""
    current = context["ti"].xcom_pull(task_ids="learn_current_schema", key="data_quality_schema")
    # 이전 스키마 로드 로직...
    previous = load_previous_schema()

    if schemas_match(current, previous):
        return "schema_unchanged"
    else:
        return "schema_changed"

with DAG(dag_id="schema_evolution_detection") as dag:

    learn_schema = DataQualityLearnOperator(
        task_id="learn_current_schema",
        data_path="s3://bucket/data/{{ ds }}/data.parquet",
        connection_id="truthound_s3",
        strictness="moderate",
    )

    compare = BranchPythonOperator(
        task_id="compare_with_previous",
        python_callable=compare_schemas,
    )

    schema_unchanged = DataQualityCheckOperator(
        task_id="schema_unchanged",
        rules="{{ ti.xcom_pull(task_ids='load_baseline_rules') }}",
        data_path="s3://bucket/data/{{ ds }}/data.parquet",
    )

    schema_changed = PythonOperator(
        task_id="schema_changed",
        python_callable=lambda: print("Schema changed! Review required."),
    )

    learn_schema >> compare >> [schema_unchanged, schema_changed]
```

---

## Testing Strategy

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # pytest fixtures
├── test_operators/
│   ├── __init__.py
│   ├── test_check_operator.py
│   ├── test_profile_operator.py
│   └── test_learn_operator.py
├── test_sensors/
│   ├── __init__.py
│   └── test_quality_sensor.py
├── test_hooks/
│   ├── __init__.py
│   └── test_truthound_hook.py
└── test_integration/
    ├── __init__.py
    └── test_dag_integrity.py
```

### Fixtures (conftest.py)

```python
import pytest
from unittest.mock import MagicMock, patch
import polars as pl
from airflow.models import DagBag, Connection
from airflow.utils.state import DagRunState


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """테스트용 샘플 DataFrame"""
    return pl.DataFrame({
        "user_id": ["uuid1", "uuid2", "uuid3", None],
        "email": ["a@b.com", "invalid", "c@d.com", "e@f.com"],
        "age": [25, 30, -5, 150],
        "amount": [100.0, 200.0, 50.0, 1000.0],
    })


@pytest.fixture
def mock_truthound():
    """Truthound 모듈 모킹"""
    with patch("truthound_airflow.operators.check.th") as mock:
        # Mock check result
        mock_result = MagicMock()
        mock_result.status.value = "passed"
        mock_result.is_success = True
        mock_result.passed_count = 5
        mock_result.failed_count = 0
        mock_result.warning_count = 0
        mock_result.failure_rate = 0.0
        mock_result.failures = []
        mock_result.execution_time_ms = 100.0
        mock_result.timestamp.isoformat.return_value = "2024-01-01T00:00:00Z"

        mock.check.return_value = mock_result
        yield mock


@pytest.fixture
def mock_hook():
    """DataQualityHook 모킹"""
    with patch("truthound_airflow.operators.check.DataQualityHook") as mock:
        hook_instance = MagicMock()
        hook_instance.load_data.return_value = pl.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
        })
        mock.return_value = hook_instance
        yield mock


@pytest.fixture
def airflow_context():
    """Airflow 실행 컨텍스트 모킹"""
    ti = MagicMock()
    ti.xcom_push = MagicMock()
    ti.xcom_pull = MagicMock()

    return {
        "ti": ti,
        "ds": "2024-01-01",
        "execution_date": "2024-01-01T00:00:00",
        "dag_run": MagicMock(),
    }


@pytest.fixture
def dag_bag():
    """DAG 무결성 테스트용 DagBag"""
    return DagBag(dag_folder="examples/airflow/dags/", include_examples=False)
```

### Unit Tests

```python
# test_operators/test_check_operator.py
import pytest
from airflow.exceptions import AirflowException
from truthound_airflow import DataQualityCheckOperator


class TestDataQualityCheckOperator:
    """DataQualityCheckOperator 단위 테스트"""

    def test_init_with_data_path(self):
        """data_path로 초기화 테스트"""
        op = DataQualityCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path="/data/test.parquet",
        )

        assert op.data_path == "/data/test.parquet"
        assert op.sql is None
        assert len(op.rules) == 1

    def test_init_with_sql(self):
        """SQL로 초기화 테스트"""
        op = DataQualityCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            sql="SELECT * FROM test",
        )

        assert op.sql == "SELECT * FROM test"
        assert op.data_path is None

    def test_init_fails_with_both_path_and_sql(self):
        """data_path와 sql 동시 지정 시 실패"""
        with pytest.raises(ValueError, match="Cannot specify both"):
            DataQualityCheckOperator(
                task_id="test",
                rules=[],
                data_path="/data/test.parquet",
                sql="SELECT * FROM test",
            )

    def test_init_fails_without_path_or_sql(self):
        """data_path와 sql 모두 없을 시 실패"""
        with pytest.raises(ValueError, match="Must specify either"):
            DataQualityCheckOperator(
                task_id="test",
                rules=[],
            )

    def test_execute_success(
        self,
        mock_truthound,
        mock_hook,
        airflow_context,
    ):
        """성공 케이스 실행 테스트"""
        op = DataQualityCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path="/data/test.parquet",
        )

        result = op.execute(airflow_context)

        assert result["is_success"] is True
        assert result["passed_count"] == 5
        airflow_context["ti"].xcom_push.assert_called_once()

    def test_execute_failure_raises(
        self,
        mock_truthound,
        mock_hook,
        airflow_context,
    ):
        """실패 시 예외 발생 테스트"""
        # 실패 결과로 모킹
        mock_truthound.check.return_value.is_success = False
        mock_truthound.check.return_value.failed_count = 2
        mock_truthound.check.return_value.failure_rate = 0.4

        op = DataQualityCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path="/data/test.parquet",
            fail_on_error=True,
        )

        with pytest.raises(AirflowException, match="Quality check failed"):
            op.execute(airflow_context)

    def test_execute_failure_with_warning_threshold(
        self,
        mock_truthound,
        mock_hook,
        airflow_context,
    ):
        """경고 임계값 이하 실패 시 예외 미발생 테스트"""
        mock_truthound.check.return_value.is_success = False
        mock_truthound.check.return_value.failed_count = 1
        mock_truthound.check.return_value.failure_rate = 0.02  # 2%

        op = DataQualityCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path="/data/test.parquet",
            fail_on_error=True,
            warning_threshold=0.05,  # 5%
        )

        # 예외 없이 실행 완료
        result = op.execute(airflow_context)
        assert result["is_success"] is False

    def test_template_fields(self):
        """템플릿 필드 테스트"""
        op = DataQualityCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path="/data/{{ ds }}/test.parquet",
        )

        assert "data_path" in op.template_fields
        assert "rules" in op.template_fields
        assert "sql" in op.template_fields


class TestDataQualityCheckOperatorIntegration:
    """통합 테스트 (실제 Truthound 호출)"""

    @pytest.mark.integration
    def test_real_check_execution(self, sample_dataframe, tmp_path):
        """실제 데이터로 검증 실행"""
        # 테스트 데이터 저장
        data_path = tmp_path / "test.parquet"
        sample_dataframe.write_parquet(data_path)

        op = DataQualityCheckOperator(
            task_id="test",
            rules=[
                {"column": "user_id", "type": "not_null"},
                {"column": "age", "type": "in_range", "min": 0, "max": 200},
            ],
            data_path=str(data_path),
            connection_id="truthound_default",
            fail_on_error=False,
        )

        # 실행 및 검증
        # Note: 실제 실행 시 환경 설정 필요
```

### DAG Integrity Tests

```python
# test_integration/test_dag_integrity.py
import pytest
from airflow.models import DagBag


class TestDagIntegrity:
    """DAG 무결성 테스트"""

    @pytest.fixture
    def dag_bag(self):
        return DagBag(dag_folder="examples/airflow/dags/", include_examples=False)

    def test_no_import_errors(self, dag_bag):
        """DAG 임포트 에러 없음 확인"""
        assert len(dag_bag.import_errors) == 0, f"Import errors: {dag_bag.import_errors}"

    def test_dag_loaded(self, dag_bag):
        """DAG 로드 확인"""
        assert len(dag_bag.dags) > 0, "No DAGs found"

    @pytest.mark.parametrize("dag_id", [
        "quality_gated_etl",
        "schema_evolution_detection",
    ])
    def test_dag_exists(self, dag_bag, dag_id):
        """특정 DAG 존재 확인"""
        assert dag_id in dag_bag.dags, f"DAG {dag_id} not found"

    def test_truthound_operators_in_dag(self, dag_bag):
        """Truthound Operator 포함 확인"""
        dag = dag_bag.get_dag("quality_gated_etl")
        operator_types = [type(task).__name__ for task in dag.tasks]

        assert "DataQualityCheckOperator" in operator_types
```

### pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    slow: marks tests as slow (deselect with '-m "not slow"')
filterwarnings =
    ignore::DeprecationWarning
```

---

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "truthound-airflow"
version = "0.1.0"
description = "Apache Airflow provider for Truthound data quality framework"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    { name = "Truthound Team", email = "team@truthound.dev" }
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Framework :: Apache Airflow",
    "Framework :: Apache Airflow :: Provider",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Quality Assurance",
]
keywords = ["airflow", "data-quality", "truthound", "validation"]

dependencies = [
    "apache-airflow>=2.6.0",
    "truthound>=1.0.0",
    "polars>=0.20.0",
]

[project.optional-dependencies]
postgres = [
    "psycopg2-binary>=2.9.0",
]
bigquery = [
    "google-cloud-bigquery>=3.0.0",
]
s3 = [
    "boto3>=1.28.0",
]
gcs = [
    "google-cloud-storage>=2.0.0",
]
all = [
    "truthound-airflow[postgres,bigquery,s3,gcs]",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/seadonggyun4/truthound-integrations"
Documentation = "https://truthound.dev/docs/integrations/airflow"
Repository = "https://github.com/seadonggyun4/truthound-integrations"
Issues = "https://github.com/seadonggyun4/truthound-integrations/issues"

[project.entry-points."apache_airflow_provider"]
provider_info = "truthound_airflow:get_provider_info"

[tool.hatch.build.targets.wheel]
packages = ["src/truthound_airflow"]

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "ARG", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[[tool.mypy.overrides]]
module = "airflow.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
markers = [
    "integration: marks tests as integration tests",
    "slow: marks tests as slow",
]

[tool.coverage.run]
source = ["src/truthound_airflow"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

### Provider Info

```python
# src/truthound_airflow/__init__.py

def get_provider_info() -> dict:
    """Airflow Provider 메타데이터"""
    return {
        "package-name": "truthound-airflow",
        "name": "Truthound",
        "description": "Apache Airflow provider for Truthound data quality framework",
        "connection-types": [
            {
                "connection-type": "truthound",
                "hook-class-name": "truthound_airflow.hooks.base.DataQualityHook",
            }
        ],
        "versions": ["0.1.0"],
    }
```

---

## References

- [Apache Airflow Provider Development](https://airflow.apache.org/docs/apache-airflow-providers/howto/create-custom-providers.html)
- [Truthound Documentation](https://truthound.dev/docs)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

---

*이 문서는 truthound-airflow 패키지의 완전한 구현 명세입니다.*
