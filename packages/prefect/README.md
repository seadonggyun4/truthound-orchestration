# Truthound Prefect

Official Prefect integration package for Truthound 3.x orchestration workflows.

## Installation

```bash
pip install truthound-prefect "truthound>=3.0,<4.0"
```

## Quick Start

```python
from prefect import flow
from truthound_prefect import data_quality_check


@flow
def validate_users() -> dict:
    return data_quality_check(
        data_path="users.parquet",
        rules=[
            {"type": "not_null", "column": "id"},
            {"type": "unique", "column": "email"},
        ],
    )
```

## Features

- Prefect Blocks for persisted configuration
- Zero-config Truthound-first task helpers
- Shared orchestration wire serialization
- Truthound 3.x compatibility line
