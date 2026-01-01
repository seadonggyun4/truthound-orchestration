---
title: Airflow Sensors
---

# Airflow Sensors

Sensors that wait until data quality conditions are met.

## DataQualitySensor

Waits until data quality threshold is satisfied.

```python
from packages.airflow.sensors import DataQualitySensor

sensor = DataQualitySensor(
    task_id="wait_for_quality",
    data_source="s3://bucket/data.parquet",
    quality_threshold=0.95,
    poke_interval=60,
    timeout=3600,
    mode="poke",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_id` | str | - | Task ID |
| `data_source` | str | - | Data source path |
| `quality_threshold` | float | 0.95 | Pass rate threshold (0.0-1.0) |
| `poke_interval` | int | 60 | Retry interval (seconds) |
| `timeout` | int | 3600 | Timeout (seconds) |
| `mode` | str | "poke" | Sensor mode ("poke" or "reschedule") |
| `engine_name` | str | "truthound" | Engine name to use |

### Usage Example

```python
from airflow import DAG
from packages.airflow.sensors import DataQualitySensor
from packages.airflow.operators import DataQualityCheckOperator
from datetime import datetime

with DAG(
    dag_id="quality_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
) as dag:

    # Wait for data quality condition
    wait_for_quality = DataQualitySensor(
        task_id="wait_for_quality",
        data_source="s3://bucket/upstream_data.parquet",
        quality_threshold=0.99,
        poke_interval=300,  # Check every 5 minutes
        timeout=7200,  # 2 hour timeout
    )

    # Process downstream after quality condition met
    process_data = DataQualityCheckOperator(
        task_id="process_data",
        data_source="s3://bucket/upstream_data.parquet",
        auto_schema=True,
    )

    wait_for_quality >> process_data
```

## TruthoundSensor

Truthound-specific Sensor:

```python
from packages.airflow.sensors import TruthoundSensor

sensor = TruthoundSensor(
    task_id="truthound_sensor",
    data_source="s3://bucket/data.parquet",
    quality_threshold=0.95,
    auto_schema=True,
    parallel=True,
)
```

## SensorConfig

Sensor configuration class:

```python
from packages.airflow.sensors import SensorConfig

config = SensorConfig(
    quality_threshold=0.95,
    poke_interval=60,
    timeout=3600,
    mode="reschedule",
    engine_name="truthound",
)

sensor = DataQualitySensor(
    task_id="sensor",
    data_source="s3://bucket/data.parquet",
    config=config,
)
```

## Sensor Behavior

1. `poke()` method is called at `poke_interval` intervals
2. Data quality validation is executed
3. Returns `True` if pass rate meets or exceeds `quality_threshold`
4. Waits until next poke if condition not met
5. Task fails if `timeout` is exceeded

## Modes

| Mode | Description |
|------|-------------|
| `poke` | Occupies worker slot while waiting |
| `reschedule` | Releases slot and reschedules |

`reschedule` mode is recommended for long wait times:

```python
sensor = DataQualitySensor(
    task_id="long_wait_sensor",
    data_source="s3://bucket/data.parquet",
    quality_threshold=0.95,
    poke_interval=600,  # 10 minutes
    timeout=86400,  # 24 hours
    mode="reschedule",  # Efficient slot usage
)
```
