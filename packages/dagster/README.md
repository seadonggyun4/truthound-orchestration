# Truthound Dagster

Official Dagster integration package for Truthound 3.x orchestration workflows.

## Installation

```bash
pip install truthound-dagster "truthound>=3.0,<4.0"
```

## Quick Start

```python
from dagster import Definitions, asset
from truthound_dagster import DataQualityResource


@asset
def users() -> str:
    return "users.parquet"


defs = Definitions(resources={"data_quality": DataQualityResource()})
```

## Features

- Dagster resources as the integration boundary
- Truthound-first zero-config defaults
- Shared resolver and serializer contracts
- Truthound 3.x compatibility line

Depot pipeline operations are documented in [`../../docs/depot-pipelines.md`](../../docs/depot-pipelines.md). The Dagster-specific happy path uses `DepotResource` and the Depot ops exported from `truthound_dagster.ops`.
