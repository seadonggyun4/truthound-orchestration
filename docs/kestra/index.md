# Kestra Integration

The Kestra integration package provides data quality validation capabilities for [Kestra](https://kestra.io/) workflow orchestration through Python script executors and YAML flow generators.

## Installation

```bash
pip install truthound-orchestration[kestra]

# With Truthound engine (default)
pip install truthound
```

## Components

| Component | Description |
|-----------|-------------|
| Scripts | Python executors for data quality operations |
| Flows | YAML flow generators for Kestra |
| Outputs | Kestra-native output handlers |
| SLA | Service Level Agreement monitoring |

## Script Executors

### Check Script

Validates data against quality rules.

```python
from truthound_kestra.scripts import check_quality_script, CheckScriptConfig

result = check_quality_script(
    data_uri="s3://bucket/data.parquet",
    config=CheckScriptConfig(
        engine_name="truthound",
        fail_on_error=True,
        auto_schema=True,
        parallel=True,
    ),
)

print(f"Status: {result.status}")
print(f"Passed: {result.passed_count}, Failed: {result.failed_count}")
```

### Profile Script

Generates statistical profiles.

```python
from truthound_kestra.scripts import profile_data_script, ProfileScriptConfig

result = profile_data_script(
    data_uri="s3://bucket/data.parquet",
    config=ProfileScriptConfig(
        include_stats=True,
        sample_size=10000,
    ),
)

for column in result.columns:
    print(f"{column.name}: {column.dtype}, null_rate={column.null_rate}")
```

### Learn Script

Infers validation rules from data.

```python
from truthound_kestra.scripts import learn_schema_script, LearnScriptConfig

result = learn_schema_script(
    data_uri="s3://bucket/baseline.parquet",
    config=LearnScriptConfig(
        min_confidence=0.8,
        include_patterns=True,
    ),
)

for rule in result.rules:
    print(f"{rule.column}: {rule.type} (confidence={rule.confidence})")
```

## Flow Generation

Generate Kestra-compatible YAML flow definitions.

### Check Flow

```python
from truthound_kestra.flows import generate_check_flow

yaml_content = generate_check_flow(
    flow_id="validate_users",
    namespace="production",
    input_uri="s3://bucket/users.parquet",
    rules=[
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "email"},
    ],
    fail_on_error=True,
    schedule="0 0 * * *",  # Daily at midnight
)

with open("flows/validate_users.yaml", "w") as f:
    f.write(yaml_content)
```

### Profile Flow

```python
from truthound_kestra.flows import generate_profile_flow

yaml_content = generate_profile_flow(
    flow_id="profile_users",
    namespace="production",
    input_uri="s3://bucket/users.parquet",
    schedule="0 6 * * 0",  # Weekly on Sunday
)
```

### Learn Flow

```python
from truthound_kestra.flows import generate_learn_flow

yaml_content = generate_learn_flow(
    flow_id="learn_schema",
    namespace="production",
    input_uri="s3://bucket/baseline.parquet",
    min_confidence=0.8,
)
```

### Quality Pipeline

Generate a complete pipeline with profile, check, and learn steps.

```python
from truthound_kestra.flows import generate_quality_pipeline

yaml_content = generate_quality_pipeline(
    flow_id="data_quality_pipeline",
    namespace="production",
    input_uri="s3://bucket/data.parquet",
    rules=[
        {"type": "not_null", "column": "id"},
        {"type": "in_range", "column": "amount", "min": 0},
    ],
    include_profile=True,
    include_learn=False,
    fail_on_error=True,
    schedule="0 6 * * *",  # Daily at 6 AM
)
```

Generated flow structure:

```yaml
id: data_quality_pipeline
namespace: production
description: Data quality pipeline
tasks:
  - id: profile
    type: io.kestra.plugin.scripts.python.Script
    dependsOn: []
  - id: check
    type: io.kestra.plugin.scripts.python.Script
    dependsOn:
      - profile
outputs:
  - name: profile_result
    value: "{{ outputs.profile.result }}"
  - name: check_result
    value: "{{ outputs.check.result }}"
```

### Flow Configuration

```python
from truthound_kestra.flows import (
    FlowConfig,
    TaskConfig,
    TriggerConfig,
    InputConfig,
    OutputConfig,
    RetryConfig,
    TaskType,
    TriggerType,
    RetryPolicy,
    FlowGenerator,
)

# Build custom flow
config = FlowConfig(
    id="custom_flow",
    namespace="production",
    description="Custom data quality flow",
    inputs=(
        InputConfig(name="data_uri", type="STRING", required=True),
    ),
    tasks=(
        TaskConfig(
            id="validate",
            task_type=TaskType.CHECK,
            input_uri="{{ inputs.data_uri }}",
            rules=({"type": "not_null", "column": "id"},),
            fail_on_error=True,
        ),
    ),
    triggers=(
        TriggerConfig(
            id="schedule",
            type=TriggerType.SCHEDULE,
            cron="0 */6 * * *",
        ),
    ),
    retry=RetryConfig(
        max_attempts=3,
        initial_delay_seconds=30.0,
        policy=RetryPolicy.EXPONENTIAL,
    ),
)

generator = FlowGenerator(python_version="3.11")
yaml_content = generator.generate(config)
```

## Output Handlers

Send results to Kestra outputs.

```python
from truthound_kestra.outputs import (
    send_check_result,
    send_profile_result,
    send_outputs,
    KestraOutputHandler,
)

# Send check result
send_check_result(result)

# Send multiple outputs
send_outputs({
    "status": result.status,
    "passed_count": result.passed_count,
    "failed_count": result.failed_count,
})

# Custom handler
handler = KestraOutputHandler()
handler.send("custom_metric", 42)
```

## SLA Monitoring

### Configuration

```python
from truthound_kestra.sla import SLAConfig, SLAMonitor, SLAMetrics

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
    flow_id="validate_users",
    task_id="check_quality",
)

result = monitor.evaluate(metrics)

if not result.is_compliant:
    for violation in result.violations:
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
from truthound_kestra.sla import PRODUCTION_SLA_CONFIG

monitor = SLAMonitor(config=PRODUCTION_SLA_CONFIG)
```

### SLA Registry

```python
from truthound_kestra.sla import (
    register_sla,
    evaluate_sla,
    get_sla_registry,
)

# Register SLAs
register_sla("users_check", SLAConfig(min_pass_rate=0.95))
register_sla("orders_check", SLAConfig(min_pass_rate=0.99))

# Evaluate
result = evaluate_sla("users_check", metrics, raise_on_violation=False)

# Get registry
registry = get_sla_registry()
all_results = registry.get_all_results()
```

### SLA Hooks

```python
from truthound_kestra.sla import (
    LoggingSLAHook,
    MetricsSLAHook,
    CallbackSLAHook,
    KestraNotificationHook,
    CompositeSLAHook,
)

# Logging hook
logging_hook = LoggingSLAHook()

# Metrics collection
metrics_hook = MetricsSLAHook()

# Custom callback
callback_hook = CallbackSLAHook(
    on_pass=lambda r: print(f"SLA passed"),
    on_violation=lambda r: print(f"SLA violated: {len(r.violations)} violations"),
)

# Kestra notification
kestra_hook = KestraNotificationHook()

# Combine hooks
composite = CompositeSLAHook([
    logging_hook,
    metrics_hook,
    kestra_hook,
])

monitor = SLAMonitor(config=config, hooks=[composite])
```

## Configuration Presets

### Script Configurations

```python
from truthound_kestra.scripts import (
    DEFAULT_SCRIPT_CONFIG,
    STRICT_SCRIPT_CONFIG,
    LENIENT_SCRIPT_CONFIG,
    PRODUCTION_SCRIPT_CONFIG,
)

# Strict: fails on any error
check_quality_script(
    data_uri="s3://bucket/data.parquet",
    config=STRICT_SCRIPT_CONFIG,
)

# Production: parallel with balanced settings
check_quality_script(
    data_uri="s3://bucket/data.parquet",
    config=PRODUCTION_SCRIPT_CONFIG,
)
```

## Utility Functions

### Engine Access

```python
from truthound_kestra.scripts import get_engine

# Get engine by name
engine = get_engine("truthound")
engine = get_engine("great_expectations")
engine = get_engine("pandera")
```

### Data Loading

```python
from truthound_kestra.utils import load_data, detect_data_source_type

# Auto-detect and load
data = load_data("s3://bucket/data.parquet")
data = load_data("/path/to/file.csv")
data = load_data("https://example.com/data.json")

# Detect source type
source_type = detect_data_source_type("s3://bucket/data.parquet")
# Returns: DataSourceType.S3
```

### Kestra Integration

```python
from truthound_kestra.utils import (
    get_kestra_variable,
    get_execution_context,
    create_kestra_output,
)

# Get Kestra variable
data_uri = get_kestra_variable("data_uri", default="s3://default/path")

# Get execution context
context = get_execution_context()
print(f"Flow: {context['flow_id']}, Execution: {context['execution_id']}")

# Create output
output = create_kestra_output("result", {"status": "passed"})
```

## Exception Handling

```python
from truthound_kestra.utils.exceptions import (
    DataQualityError,
    ConfigurationError,
    EngineError,
    ScriptError,
    FlowError,
    OutputError,
    SLAViolationError,
)

try:
    result = check_quality_script(data_uri="s3://bucket/data.parquet")
except EngineError as e:
    print(f"Engine error: {e}")
except SLAViolationError as e:
    print(f"SLA violated: {e.violations}")
except DataQualityError as e:
    print(f"Data quality error: {e}")
```

## Serialization

```python
from truthound_kestra.utils.serialization import (
    JsonSerializer,
    YamlSerializer,
    MarkdownSerializer,
)

# JSON serialization
serializer = JsonSerializer()
json_str = serializer.serialize(result)
restored = serializer.deserialize(json_str)

# YAML serialization
serializer = YamlSerializer()
yaml_str = serializer.serialize(result)

# Markdown report
serializer = MarkdownSerializer()
report = serializer.serialize(result)
```
