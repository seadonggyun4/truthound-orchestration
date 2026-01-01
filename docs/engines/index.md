---
title: Engines
---

# Data Quality Engines

The `DataQualityEngine` Protocol enables the use of various data quality engines through a unified interface.

## Supported Engines

| Engine | Status | Characteristics |
|--------|--------|-----------------|
| TruthoundEngine | Default Engine | Schema-based validation, automatic learning, Polars native |
| GreatExpectationsAdapter | Adapter | Expectation-based validation |
| PanderaAdapter | Adapter | Type-safe schema validation |

## DataQualityEngine Protocol

The core Protocol implemented by all engines:

```python
from typing import Protocol, Any, Sequence, Mapping

class DataQualityEngine(Protocol):
    @property
    def engine_name(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CheckResult: ...

    def profile(self, data: Any, **kwargs: Any) -> ProfileResult: ...

    def learn(self, data: Any, **kwargs: Any) -> LearnResult: ...
```

## Engine Registry

```python
from common.engines import (
    get_engine,
    register_engine,
    list_engines,
    get_default_engine,
    set_default_engine,
)

# Get engine by name
engine = get_engine("truthound")
engine = get_engine("great_expectations")
engine = get_engine("pandera")

# List available engines
for name in list_engines():
    print(name)

# Change default engine
set_default_engine("great_expectations")
default = get_default_engine()
```

## Engine Capabilities

```python
from common.engines import get_engine

engine = get_engine("truthound")
caps = engine.get_capabilities()

print(f"Check support: {caps.supports_check}")
print(f"Profile support: {caps.supports_profile}")
print(f"Learn support: {caps.supports_learn}")
print(f"Async support: {caps.supports_async}")
print(f"Streaming support: {caps.supports_streaming}")
print(f"Data types: {caps.supported_data_types}")
print(f"Rule types: {caps.supported_rule_types}")
```

## Advanced Features

Advanced features for engine management:

| Feature | Description | Documentation |
|---------|-------------|---------------|
| Lifecycle | Engine start/stop/health check | [lifecycle.md](lifecycle.md) |
| Batch Processing | Large data chunking | [batch.md](batch.md) |
| Engine Chain | Fallback/load balancing | [chain.md](chain.md) |

## Navigation

- [TruthoundEngine](truthound.md) - Default engine
- [GreatExpectationsAdapter](great-expectations.md) - GE adapter
- [PanderaAdapter](pandera.md) - Pandera adapter
- [Lifecycle Management](lifecycle.md) - Engine lifecycle
