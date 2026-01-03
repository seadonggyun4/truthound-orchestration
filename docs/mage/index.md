# Mage AI Integration

The Mage AI integration package provides data quality validation capabilities for [Mage AI](https://www.mage.ai/) pipelines through custom block implementations.

## Installation

```bash
pip install truthound-orchestration[mage]

# With Truthound engine (default)
pip install truthound
```

## Components

The package provides three categories of Mage blocks:

| Component | Description |
|-----------|-------------|
| Transformer Blocks | Execute data quality operations (check, profile, learn) |
| Sensor Blocks | Monitor data quality conditions |
| Condition Blocks | Enable conditional branching based on quality metrics |

## Transformer Blocks

### CheckTransformer

Validates data against quality rules.

```python
from truthound_mage import CheckTransformer, CheckBlockConfig

config = CheckBlockConfig(
    fail_on_error=True,
    auto_schema=True,
    parallel=True,
)

transformer = CheckTransformer(config=config)

# In Mage block
@transformer(data_source="upstream_block")
def validate_data(data, *args, **kwargs):
    from truthound_mage.blocks import BlockExecutionContext

    context = BlockExecutionContext(
        block_uuid="validate_users",
        pipeline_uuid="data_pipeline",
    )

    result = transformer.execute(data, context)

    if result.result.status.name == "FAILED":
        raise ValueError(f"Validation failed: {result.result.failed_count} failures")

    return result.data
```

### ProfileTransformer

Generates statistical profiles of data.

```python
from truthound_mage import ProfileTransformer, ProfileBlockConfig

config = ProfileBlockConfig(
    include_stats=True,
    include_histograms=False,
    sample_size=10000,
)

transformer = ProfileTransformer(config=config)

@transformer(data_source="raw_data")
def profile_data(data, *args, **kwargs):
    context = BlockExecutionContext(
        block_uuid="profile_users",
        pipeline_uuid="data_pipeline",
    )

    result = transformer.execute(data, context)
    return result.data
```

### LearnTransformer

Infers validation rules from data characteristics.

```python
from truthound_mage import LearnTransformer, LearnBlockConfig

config = LearnBlockConfig(
    min_confidence=0.8,
    include_patterns=True,
    categorical_threshold=50,
)

transformer = LearnTransformer(config=config)

@transformer(data_source="baseline_data")
def learn_schema(data, *args, **kwargs):
    context = BlockExecutionContext(
        block_uuid="learn_schema",
        pipeline_uuid="schema_pipeline",
    )

    result = transformer.execute(data, context)
    return result.data
```

### DataQualityTransformer

Unified facade supporting all operations.

```python
from truthound_mage import DataQualityTransformer

transformer = DataQualityTransformer()

# Use for any operation type
result = transformer.check(data, context, auto_schema=True)
result = transformer.profile(data, context)
result = transformer.learn(data, context)
```

## Sensor Blocks

### DataQualitySensor

Polls for data quality conditions.

```python
from truthound_mage import DataQualitySensor

sensor = DataQualitySensor(
    min_pass_rate=0.95,
    timeout_seconds=300.0,
    poll_interval_seconds=30.0,
)

# Returns True when condition is met
is_ready = sensor.poll(data_path="s3://bucket/data.parquet")
```

### QualityGateSensor

Blocks pipeline progression based on quality thresholds.

```python
from truthound_mage import QualityGateSensor

gate = QualityGateSensor(
    min_pass_rate=0.99,
    max_critical_failures=0,
)

# Raises exception if gate fails
gate.check(result)
```

## Condition Blocks

### DataQualityCondition

Enables conditional branching.

```python
from truthound_mage import DataQualityCondition

condition = DataQualityCondition(
    pass_threshold=0.95,
    on_pass="process_data",
    on_fail="quarantine_data",
)

# Returns branch name based on result
next_block = condition.evaluate(result)
```

## SLA Monitoring

The package includes SLA monitoring for quality metrics.

### Configuration

```python
from truthound_mage.sla import SLAConfig, SLAMonitor, SLAMetrics

config = SLAConfig(
    enabled=True,
    max_failure_rate=0.05,
    min_pass_rate=0.95,
    max_execution_time_seconds=300.0,
    max_consecutive_failures=3,
)

monitor = SLAMonitor(config=config)
```

### Evaluation

```python
metrics = SLAMetrics(
    passed_count=950,
    failed_count=50,
    execution_time_seconds=120.5,
    row_count=1000,
    block_uuid="validate_users",
    pipeline_uuid="data_pipeline",
)

violations = monitor.check(metrics)

for violation in violations:
    print(f"{violation.violation_type.value}: {violation.message}")
```

### Preset Configurations

| Preset | Description |
|--------|-------------|
| `DEFAULT_SLA_CONFIG` | Default thresholds |
| `STRICT_SLA_CONFIG` | Stringent requirements |
| `LENIENT_SLA_CONFIG` | Relaxed thresholds |
| `PRODUCTION_SLA_CONFIG` | Balanced production settings |

```python
from truthound_mage.sla import PRODUCTION_SLA_CONFIG

monitor = SLAMonitor(config=PRODUCTION_SLA_CONFIG)
```

### SLA Registry

Manage multiple SLA monitors.

```python
from truthound_mage.sla import get_sla_registry

registry = get_sla_registry()

# Register monitors
registry.register("users_check", SLAConfig(min_pass_rate=0.95))
registry.register("orders_check", SLAConfig(min_pass_rate=0.99))

# Check SLA
violations = registry.check("users_check", metrics)

# Get statistics
stats = registry.get_all_stats()
```

### SLA Hooks

```python
from truthound_mage.sla import (
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
)

# Combine multiple hooks
hooks = CompositeSLAHook([
    LoggingSLAHook(),
    MetricsSLAHook(),
])

monitor = SLAMonitor(config=config, hooks=[hooks])
```

## IO Configuration

Load data source configurations from `io_config.yaml`.

```python
from truthound_mage.io import load_io_config

config = load_io_config("io_config.yaml")

# Access sources and sinks
source = config.sources["s3_data"]
sink = config.sinks["warehouse"]
```

## Factory Functions

```python
from truthound_mage import (
    create_check_transformer,
    create_profile_transformer,
    create_learn_transformer,
)

# Create with custom engine
from common.engines import GreatExpectationsAdapter

transformer = create_check_transformer(
    config=CheckBlockConfig(fail_on_error=True),
    engine=GreatExpectationsAdapter(),
)
```

## Exception Handling

```python
from truthound_mage.utils.exceptions import (
    DataQualityBlockError,
    BlockConfigurationError,
    BlockExecutionError,
    DataLoadError,
    SLAViolationError,
)

try:
    result = transformer.execute(data, context)
except BlockExecutionError as e:
    print(f"Execution failed: {e}")
except SLAViolationError as e:
    print(f"SLA violated: {e.violations}")
```

## Result Serialization

```python
from truthound_mage.utils.serialization import serialize_result, deserialize_result

# Serialize for storage
serialized = serialize_result(result)

# Deserialize
restored = deserialize_result(serialized, "check")
```
