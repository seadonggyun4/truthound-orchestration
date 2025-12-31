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

## Usage

```python
from airflow import DAG
from truthound_airflow.operators import DataQualityCheckOperator

with DAG("data_quality_dag", ...) as dag:
    check = DataQualityCheckOperator(
        task_id="check_data",
        data_path="/path/to/data.parquet",
        rules=[
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
        ],
    )
```
