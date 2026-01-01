---
title: Engine Lifecycle
---

# Engine Lifecycle

A lifecycle management system for engine initialization, health checks, and termination.

## Basic Usage

```python
from common.engines import TruthoundEngine

# Context manager usage (recommended)
with TruthoundEngine() as engine:
    result = engine.check(data, auto_schema=True)
    # Automatic stop() on context exit

# Explicit lifecycle management
engine = TruthoundEngine()
engine.start()
try:
    result = engine.check(data)
    health = engine.health_check()
    print(f"Health: {health.status.name}")
finally:
    engine.stop()
```

## Engine States

| State | Description |
|-------|-------------|
| `CREATED` | Engine created, not yet started |
| `STARTING` | Initialization in progress |
| `RUNNING` | Operating normally |
| `STOPPING` | Shutdown in progress |
| `STOPPED` | Terminated, resources released |
| `FAILED` | Error occurred |

## State Tracking

```python
from common.engines import TruthoundEngine, EngineState, EngineStateTracker

engine = TruthoundEngine()
print(f"State: {engine.get_state().name}")  # CREATED

engine.start()
print(f"State: {engine.get_state().name}")  # RUNNING

# Query state snapshot
snapshot = engine.get_state_snapshot()
print(f"Uptime: {snapshot.uptime_seconds}s")
print(f"Error count: {snapshot.error_count}")
```

## Health Checks

```python
from common.engines import TruthoundEngine
from common.health import HealthStatus

engine = TruthoundEngine()
engine.start()

result = engine.health_check()
print(f"Status: {result.status.name}")     # HEALTHY, DEGRADED, UNHEALTHY
print(f"Message: {result.message}")
print(f"Duration: {result.duration_ms}ms")
```

## Lifecycle Manager

Wrapping engines without lifecycle support:

```python
from common.engines import EngineLifecycleManager, EngineConfig

config = EngineConfig(
    auto_start=True,
    health_check_interval_seconds=60.0,
)
manager = EngineLifecycleManager(engine, config=config)

with manager:
    result = manager.check(data, rules)
    health = manager.health_check()
```

## Lifecycle Hooks

```python
from common.engines import (
    LoggingLifecycleHook,
    MetricsLifecycleHook,
    CompositeLifecycleHook,
)

# Logging hook
logging_hook = LoggingLifecycleHook()

# Metrics hook
metrics_hook = MetricsLifecycleHook()

# Composite hook
composite = CompositeLifecycleHook([logging_hook, metrics_hook])

# Hook events:
# - on_start(engine_name)
# - on_stop(engine_name)
# - on_health_check(engine_name, result)
# - on_state_change(engine_name, old_state, new_state)
# - on_error(engine_name, error)
```

## Async Engines

```python
from common.engines import SyncEngineAsyncAdapter, TruthoundEngine

# Wrap synchronous engine for async usage
sync_engine = TruthoundEngine()
async_engine = SyncEngineAsyncAdapter(sync_engine)

async with async_engine:
    result = await async_engine.check(data, auto_schema=True)
    health = await async_engine.health_check()
```

## Exception Handling

```python
from common.engines import (
    TruthoundEngine,
    EngineNotStartedError,
    EngineAlreadyStartedError,
)

engine = TruthoundEngine()

try:
    result = engine.check(data)  # Attempting check without starting
except EngineNotStartedError:
    print("Engine has not been started")
    engine.start()
    result = engine.check(data)

try:
    engine.start()  # Already started
except EngineAlreadyStartedError:
    print("Engine is already running")
```

## ManagedEngineMixin

Adding lifecycle support to custom engines:

```python
from common.engines.lifecycle import ManagedEngineMixin
from common.engines.base import EngineInfoMixin

class MyCustomEngine(ManagedEngineMixin, EngineInfoMixin):
    @property
    def engine_name(self) -> str:
        return "my_custom_engine"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    def check(self, data, rules, **kwargs):
        ...

    def profile(self, data, **kwargs):
        ...

    def learn(self, data, **kwargs):
        ...

# Usable as context manager
with MyCustomEngine() as engine:
    result = engine.check(data, rules)
```
