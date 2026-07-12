# Truthound Airflow

Apache Airflow integration for the Truthound data quality orchestration framework.

## Installation

```bash
pip install truthound-airflow
```

## Features

- DataQualityCheckOperator - Run data quality validation checks
- DataQualityProfileOperator - Profile datasets
- DataQualityLearnOperator - Learn validation rules from data
- DataQualitySensor - Wait for quality conditions
- DataQualityHook - Connection management for data sources
- SLA monitoring and callbacks

Depot pipeline operations are documented in [`../../docs/depot-pipelines.md`](../../docs/depot-pipelines.md). The Airflow-specific happy path uses `DepotScheduledValidationOperator` and the other Depot operators exported from `truthound_airflow.operators`.

## Usage

```python
from airflow import DAG
from truthound_airflow.operators import DataQualityCheckOperator

with DAG("data_quality_dag", ...) as dag:
    check = DataQualityCheckOperator(
        task_id="check_data",
        data_path="/path/to/data.parquet",
        fail_on_error=True,
    )
```

With the default `TruthoundEngine`, omitting `rules` selects Truthound 3.x
zero-config auto-schema validation. The host-neutral `rules=[{"type": ...}]`
contract is for alternative engines that explicitly consume that rule format;
the Truthound adapter does not translate those dictionaries into validators.
