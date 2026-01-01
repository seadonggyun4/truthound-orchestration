---
title: Dagster Assets
---

# Dagster Assets

Decorators and factories for defining Assets with data quality validation.

## @quality_checked_asset

Asset with automatic quality validation:

```python
from packages.dagster.assets import quality_checked_asset

@quality_checked_asset(
    auto_schema=True,
    fail_on_error=True,
)
def my_data_asset():
    return pl.read_parquet("data.parquet")
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `auto_schema` | bool | Automatic schema generation |
| `rules` | list | List of validation rules |
| `fail_on_error` | bool | Raise exception on failure |
| `engine_name` | str | Engine name to use |

## @profiled_asset

Asset with automatic profiling:

```python
from packages.dagster.assets import profiled_asset

@profiled_asset
def profiled_data():
    return pl.read_parquet("data.parquet")
```

Profile results are stored in Asset metadata.

## Asset Factories

### create_quality_asset

Create quality validation Asset:

```python
from packages.dagster.assets import create_quality_asset, QualityAssetConfig

config = QualityAssetConfig(
    auto_schema=True,
    fail_on_error=True,
    engine_name="truthound",
)

my_asset = create_quality_asset(
    name="my_quality_asset",
    compute_fn=lambda: pl.read_parquet("data.parquet"),
    config=config,
)
```

### create_quality_check_asset

Asset that returns only validation results:

```python
from packages.dagster.assets import create_quality_check_asset

check_asset = create_quality_check_asset(
    name="quality_check_result",
    data_source="s3://bucket/data.parquet",
)
```

## Asset Configuration

### QualityAssetConfig

```python
from packages.dagster.assets import QualityAssetConfig

config = QualityAssetConfig(
    auto_schema=True,
    rules=[
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
    ],
    fail_on_error=True,
    engine_name="truthound",
)
```

### ProfileAssetConfig

```python
from packages.dagster.assets import ProfileAssetConfig

config = ProfileAssetConfig(
    include_statistics=True,
    store_in_metadata=True,
)
```

### QualityCheckMode

```python
from packages.dagster.assets import QualityCheckMode

# BEFORE: Validate before returning data
# AFTER: Validate after returning data
# BOTH: Validate both before and after

@quality_checked_asset(mode=QualityCheckMode.BEFORE)
def my_asset():
    return load_data()
```

## Asset Dependencies

```python
from dagster import asset
from packages.dagster.assets import quality_checked_asset

@asset
def raw_data():
    return pl.read_parquet("raw.parquet")

@quality_checked_asset(auto_schema=True)
def clean_data(raw_data):
    # raw_data is quality validated
    return transform(raw_data)
```

## Metadata

Quality validation results are stored in Asset metadata:

```python
from dagster import asset, Output

@quality_checked_asset(auto_schema=True)
def my_asset():
    data = load_data()
    return Output(
        data,
        metadata={
            "row_count": len(data),
            "quality_status": "passed",
        },
    )
```

## Usage in Definitions

```python
from dagster import Definitions
from packages.dagster.assets import quality_checked_asset
from packages.dagster.resources import DataQualityResource

@quality_checked_asset(auto_schema=True)
def my_asset():
    return load_data()

defs = Definitions(
    assets=[my_asset],
    resources={"data_quality": DataQualityResource()},
)
```
