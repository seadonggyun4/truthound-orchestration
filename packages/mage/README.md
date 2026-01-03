# Truthound Mage Integration

Mage AI integration for data quality validation with pluggable engine support.

## Installation

```bash
pip install truthound-mage
```

## Quick Start

```python
from truthound_mage.blocks import DataQualityCheckBlock

# Create a data quality check block
block = DataQualityCheckBlock(
    engine_name="truthound",
    rules=[
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
    ],
)

# Execute the block
result = block.execute(dataframe)
```

## Features

- **DataQualityCheckBlock**: Transform block for running data quality checks
- **DataQualitySensorBlock**: Sensor block for waiting on data quality conditions
- **DataQualityConditionBlock**: Condition block for branching based on data quality
- **SLA Monitoring**: Built-in SLA monitoring and alerting

## Supported Engines

- Truthound (default)
- Great Expectations
- Pandera

## License

MIT License
