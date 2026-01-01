---
title: Airflow Integration
---

# Apache Airflow Integration

Provides Operators, Sensors, and Hooks for data quality validation within Apache Airflow.

## Installation

```bash
pip install truthound-orchestration[airflow]
```

## Components

| Component | Description | Documentation |
|-----------|-------------|---------------|
| Operators | Execute data quality validation in DAGs | [operators.md](operators.md) |
| Sensors | Wait for data quality conditions | [sensors.md](sensors.md) |
| Hooks | Connection and data loading | [hooks.md](hooks.md) |
| SLA | SLA monitoring and alerts | [sla.md](sla.md) |

## Quick Start

```python
from airflow import DAG
from packages.airflow.operators import DataQualityCheckOperator
from datetime import datetime

with DAG(
    dag_id="data_quality_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
) as dag:

    check_task = DataQualityCheckOperator(
        task_id="check_data_quality",
        data_source="s3://bucket/data.parquet",
        auto_schema=True,
    )
```

## Operators

### DataQualityCheckOperator

Executes data validation and pushes results to XCom:

```python
from packages.airflow.operators import DataQualityCheckOperator

check = DataQualityCheckOperator(
    task_id="quality_check",
    data_source="s3://bucket/data.parquet",
    auto_schema=True,
    fail_on_error=True,
)
```

### DataQualityProfileOperator

Performs data profiling:

```python
from packages.airflow.operators import DataQualityProfileOperator

profile = DataQualityProfileOperator(
    task_id="profile_data",
    data_source="s3://bucket/data.parquet",
)
```

### DataQualityLearnOperator

Learns schemas:

```python
from packages.airflow.operators import DataQualityLearnOperator

learn = DataQualityLearnOperator(
    task_id="learn_schema",
    data_source="s3://bucket/baseline.parquet",
)
```

## Sensors

### DataQualitySensor

Waits until data quality conditions are met:

```python
from packages.airflow.sensors import DataQualitySensor

sensor = DataQualitySensor(
    task_id="wait_for_quality",
    data_source="s3://bucket/data.parquet",
    quality_threshold=0.95,
    poke_interval=60,
)
```

## Hooks

### DataQualityHook

Connection management and data loading:

```python
from packages.airflow.hooks import DataQualityHook

hook = DataQualityHook(conn_id="my_data_connection")
data = hook.load_data("s3://bucket/data.parquet")
result = hook.check(data, auto_schema=True)
```

## SLA Monitoring

```python
from packages.airflow.sla import SLAConfig, DataQualitySLACallback

sla_config = SLAConfig(
    min_pass_rate=0.95,
    max_duration_seconds=3600,
)

# Connect SLA callback to DAG
with DAG(
    dag_id="sla_monitored_dag",
    sla_miss_callback=DataQualitySLACallback(sla_config),
) as dag:
    ...
```

## XCom Serialization

Validation results are passed to subsequent tasks via XCom:

```python
from packages.airflow.serializers import AirflowXComSerializer

serializer = AirflowXComSerializer()
serialized = serializer.serialize(check_result)
result = serializer.deserialize(serialized)
```

## Navigation

- [Operators](operators.md) - Detailed Operator usage
- [Sensors](sensors.md) - Sensor configuration
- [Hooks](hooks.md) - Hook utilization
- [SLA](sla.md) - SLA monitoring setup
