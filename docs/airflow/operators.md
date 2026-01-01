---
title: Airflow Operators
---

# Airflow Operators

Airflow Operators for data quality validation, profiling, and schema learning.

## DataQualityCheckOperator

Executes data validation.

```python
from packages.airflow.operators import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="quality_check",
    data_source="s3://bucket/data.parquet",
    auto_schema=True,
    fail_on_error=True,
    engine_name="truthound",
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | str | Task ID |
| `data_source` | str | Data source path |
| `auto_schema` | bool | Automatic schema generation |
| `rules` | list | List of validation rules |
| `fail_on_error` | bool | Raise exception on failure |
| `engine_name` | str | Engine name to use |

### XCom Output

Validation results are pushed to XCom:

```python
# Use results in subsequent task
def process_result(**context):
    result = context["ti"].xcom_pull(task_ids="quality_check")
    print(f"Status: {result['status']}")
    print(f"Passed: {result['passed_count']}")
```

## DataQualityProfileOperator

Performs data profiling.

```python
from packages.airflow.operators import DataQualityProfileOperator

profile = DataQualityProfileOperator(
    task_id="profile_data",
    data_source="s3://bucket/data.parquet",
    engine_name="truthound",
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | str | Task ID |
| `data_source` | str | Data source path |
| `engine_name` | str | Engine name to use |

### XCom Output

Profile results are pushed to XCom:

```python
def analyze_profile(**context):
    profile = context["ti"].xcom_pull(task_ids="profile_data")
    for col in profile["columns"]:
        print(f"{col['column_name']}: {col['null_percentage']}% null")
```

## DataQualityLearnOperator

Learns schemas.

```python
from packages.airflow.operators import DataQualityLearnOperator

learn = DataQualityLearnOperator(
    task_id="learn_schema",
    data_source="s3://bucket/baseline.parquet",
    engine_name="truthound",
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | str | Task ID |
| `data_source` | str | Baseline data path |
| `engine_name` | str | Engine name to use |

### XCom Output

Learned rules are pushed to XCom:

```python
def use_learned_schema(**context):
    learn_result = context["ti"].xcom_pull(task_ids="learn_schema")
    for rule in learn_result["rules"]:
        print(f"{rule['column']}: {rule['rule_type']}")
```

## TruthoundCheckOperator

Truthound-specific Operator:

```python
from packages.airflow.operators import TruthoundCheckOperator

check = TruthoundCheckOperator(
    task_id="truthound_check",
    data_source="s3://bucket/data.parquet",
    auto_schema=True,
    parallel=True,
    max_workers=4,
)
```

## BaseDataQualityOperator

Base class for custom Operator implementation:

```python
from packages.airflow.operators import BaseDataQualityOperator

class MyCustomOperator(BaseDataQualityOperator):
    def execute(self, context):
        engine = self.get_engine()
        data = self.load_data()
        result = engine.check(data, auto_schema=True)
        return self.serialize_result(result)
```

## Configuration Classes

### OperatorConfig

```python
from packages.airflow.operators import OperatorConfig

config = OperatorConfig(
    engine_name="truthound",
    fail_on_error=True,
    timeout_seconds=3600,
)
```

### CheckOperatorConfig

```python
from packages.airflow.operators import CheckOperatorConfig

config = CheckOperatorConfig(
    auto_schema=True,
    rules=[
        {"type": "not_null", "column": "id"},
    ],
    fail_on_error=True,
)
```

### ProfileOperatorConfig

```python
from packages.airflow.operators import ProfileOperatorConfig

config = ProfileOperatorConfig(
    include_statistics=True,
)
```

### LearnOperatorConfig

```python
from packages.airflow.operators import LearnOperatorConfig

config = LearnOperatorConfig(
    min_confidence=0.8,
)
```

## Result Handlers

Custom processing of validation results:

```python
from packages.airflow.operators import ResultHandler

class MyResultHandler(ResultHandler):
    def handle(self, result, context):
        if result.status.name == "FAILED":
            send_alert(result)
        return result
```
