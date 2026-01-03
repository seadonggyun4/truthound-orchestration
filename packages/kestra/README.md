# Truthound Kestra Integration

Truthound data quality integration for Kestra workflow orchestration.

## Installation

```bash
pip install truthound-kestra
```

## Quick Start

### Using Python Scripts in Kestra

```yaml
id: data_quality_check
namespace: company.data
tasks:
  - id: run_check
    type: io.kestra.plugin.scripts.python.Script
    script: |
      from truthound_kestra.scripts import run_check

      result = run_check(
          data_path="{{ inputs.data_path }}",
          engine_name="truthound",
          rules=[
              {"type": "not_null", "column": "id"},
              {"type": "unique", "column": "email"},
          ],
      )
      print(result)
```

### Using Flow Templates

```python
from truthound_kestra.flows import generate_check_flow

flow_yaml = generate_check_flow(
    flow_id="my_quality_check",
    namespace="company.data",
    data_source="s3://bucket/data.parquet",
    rules=[{"type": "not_null", "column": "id"}],
)
```

## Features

- **Python Scripts**: `run_check`, `run_profile`, `run_learn` for Kestra Python tasks
- **Flow Templates**: Generate Kestra flow YAML configurations
- **Output Handlers**: Parse and handle Kestra task outputs
- **SLA Monitoring**: Built-in SLA monitoring and alerting

## Supported Engines

- Truthound (default)
- Great Expectations
- Pandera

## License

MIT License
