---
title: API Reference
---

# API Reference

Core API reference for Truthound Orchestration.

## Module Structure

```
common/
├── base.py           # Core types
├── exceptions.py     # Exceptions
├── logging.py        # Logging
├── retry.py          # Retry
├── circuit_breaker.py # Circuit breaker
├── health.py         # Health check
├── metrics.py        # Metrics
├── rate_limiter.py   # Rate limiting
├── cache.py          # Caching
└── engines/          # Engines
```

## Core Types

### Enums

| Enum | Values | Description |
|------|--------|-------------|
| `CheckStatus` | PASSED, FAILED, WARNING, SKIPPED, ERROR | Validation status |
| `Severity` | CRITICAL, HIGH, MEDIUM, LOW, INFO | Severity level |
| `FailureAction` | RAISE, WARN, LOG, CONTINUE | Action on failure |

### Config Classes

| Class | Description |
|-------|-------------|
| `CheckConfig` | Validation configuration (frozen dataclass) |
| `ProfileConfig` | Profiling configuration |
| `LearnConfig` | Learning configuration |

### Result Classes

| Class | Description |
|-------|-------------|
| `CheckResult` | Validation result |
| `ProfileResult` | Profiling result |
| `LearnResult` | Learning result |
| `ValidationFailure` | Individual failure information |
| `ColumnProfile` | Column profile |
| `LearnedRule` | Learned rule |

## Protocols

### DataQualityEngine

```python
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

### ManagedEngine

```python
class ManagedEngine(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def health_check(self) -> HealthCheckResult: ...
    def get_state(self) -> EngineState: ...
```

## Exception Hierarchy

```
TruthoundIntegrationError
├── ConfigurationError
│   ├── InvalidConfigValueError
│   └── MissingConfigError
├── ValidationExecutionError
│   ├── RuleExecutionError
│   └── DataAccessError
├── SerializationError
├── PlatformConnectionError
├── IntegrationTimeoutError
└── QualityGateError
```

## Utility Functions

### Logging

```python
get_logger(name: str) -> TruthoundLogger
get_performance_logger(name: str) -> PerformanceLogger
```

### Retry

```python
@retry(max_attempts=3, exceptions=(Exception,))
retry_call(func, *args, config=None, **kwargs)
await retry_call_async(func, *args, config=None, **kwargs)
```

### Circuit Breaker

```python
@circuit_breaker(failure_threshold=5)
circuit_breaker_call(func, *args, config=None, **kwargs)
get_circuit_breaker(name: str, config=None) -> CircuitBreaker
```

### Health Check

```python
@health_check(name="component")
check_health(name: str) -> HealthCheckResult
check_all_health(parallel=True) -> HealthCheckResult
register_health_check(name: str, check_fn) -> None
```

### Metrics

```python
counter(name: str, description: str) -> Counter
gauge(name: str, description: str) -> Gauge
histogram(name: str, description: str, buckets=None) -> Histogram
summary(name: str, description: str, quantiles=None) -> Summary
@timed(name: str)
@counted(name: str)
```

### Rate Limiting

```python
@rate_limit(max_requests=100, window_seconds=60.0)
rate_limit_call(func, *args, config=None, **kwargs)
get_rate_limiter(name: str, config=None) -> RateLimiter
```

### Caching

```python
@cached(ttl_seconds=300.0)
cache_get(cache_name: str, key: str, default=None)
cache_set(cache_name: str, key: str, value, ttl_seconds=None)
get_cache(name: str, config=None) -> CacheBackend
```

## Engine Functions

```python
get_engine(name: str) -> DataQualityEngine
register_engine(name: str, engine: DataQualityEngine) -> None
list_engines() -> list[str]
get_default_engine() -> DataQualityEngine
set_default_engine(name: str) -> None
```

## Navigation

- [Common Module](common.md) - Common utility details
- [Engines](engines.md) - Engine details
