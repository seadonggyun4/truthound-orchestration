---
title: Common Module
---

# Common Module

The `common` module provides core utilities shared across all platform integrations.

## Components

### Core Types

| Module | Description |
|--------|-------------|
| `base.py` | Protocol, Config, and Result type definitions |
| `exceptions.py` | Exception hierarchy definition |
| `config.py` | Configuration loading and validation |
| `serializers.py` | Platform-specific serialization |

### Utilities

| Module | Description |
|--------|-------------|
| `logging.py` | Structured logging with sensitive data masking |
| `retry.py` | Retry decorators with backoff strategies |
| `circuit_breaker.py` | Circuit breaker pattern implementation |
| `health.py` | Health check system |
| `metrics.py` | Metrics collection and distributed tracing |
| `rate_limiter.py` | Rate limiting utilities |
| `cache.py` | Caching (LRU, LFU, TTL) |
| `rule_validation.py` | Rule validation |
| `testing.py` | Test utilities |

### Engines

| Module | Description |
|--------|-------------|
| `engines/` | Data quality engine implementations |

## Core Types

### CheckResult

Represents the result of data validation:

```python
from common.base import CheckResult, CheckStatus

result = CheckResult(
    status=CheckStatus.PASSED,
    passed_count=100,
    failed_count=0,
    metadata={"rows": 1000},
)
```

### CheckStatus

| Status | Description |
|--------|-------------|
| `PASSED` | Validation passed |
| `FAILED` | Validation failed |
| `WARNING` | Warning (threshold exceeded) |
| `SKIPPED` | Validation skipped |
| `ERROR` | Error occurred |

### ProfileResult

Data profiling result:

```python
from common.base import ProfileResult, ColumnProfile

profile = ProfileResult(
    columns=[
        ColumnProfile(
            column_name="id",
            dtype="Int64",
            null_count=0,
            null_percentage=0.0,
            unique_count=1000,
        ),
    ],
    row_count=1000,
)
```

### LearnResult

Schema learning result:

```python
from common.base import LearnResult, LearnedRule

learn = LearnResult(
    rules=[
        LearnedRule(
            column="email",
            rule_type="not_null",
            confidence=0.99,
        ),
    ],
)
```

## Exceptions

### Exception Hierarchy

```
TruthoundIntegrationError (base)
├── ConfigurationError
│   ├── InvalidConfigValueError
│   └── MissingConfigError
├── ValidationExecutionError
│   ├── RuleExecutionError
│   └── DataAccessError
├── SerializationError
│   ├── SerializeError
│   └── DeserializeError
├── PlatformConnectionError
│   └── AuthenticationError
├── IntegrationTimeoutError
└── QualityGateError
    └── ThresholdExceededError
```

### Exception Wrapping

```python
from common.exceptions import wrap_exception, ConfigurationError

try:
    load_config()
except ValueError as e:
    raise wrap_exception(e, ConfigurationError, "Configuration loading failed")
```

## Navigation

- [Logging](logging.md)
- [Retry](retry.md)
- [Circuit Breaker](circuit-breaker.md)
- [Health Check](health.md)
- [Metrics](metrics.md)
- [Rate Limiting](rate-limiter.md)
- [Caching](cache.md)
