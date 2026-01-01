---
title: Engine Chain
---

# Engine Chain

Engine chains combine multiple engines to implement advanced patterns such as fallback, load balancing, and conditional routing.

## Basic Usage

```python
from common.engines import EngineChain, TruthoundEngine, GreatExpectationsAdapter

primary = TruthoundEngine()
backup = GreatExpectationsAdapter()
chain = EngineChain([primary, backup])

# Use like a regular engine (automatically tries backup on primary failure)
result = chain.check(data, rules)
```

## Fallback Strategies

| Strategy | Description |
|----------|-------------|
| `SEQUENTIAL` | Try sequentially (default) |
| `FIRST_AVAILABLE` | Use first healthy engine |
| `ROUND_ROBIN` | Distribute via round robin |
| `RANDOM` | Random selection |
| `PRIORITY` | Priority-based selection |
| `WEIGHTED` | Weighted random selection |

## Execution Modes

| Mode | Description |
|------|-------------|
| `FAIL_FAST` | Stop immediately on first failure |
| `FALLBACK` | Try next engine on failure (default) |
| `ALL` | Execute all engines, aggregate results |
| `FIRST_SUCCESS` | Execute until first success |
| `QUORUM` | Execute until quorum success |

## Preset Configurations

```python
from common.engines import (
    DEFAULT_FALLBACK_CONFIG,       # Default: sequential fallback
    RETRY_FALLBACK_CONFIG,         # Retry: 3 attempts
    HEALTH_AWARE_FALLBACK_CONFIG,  # Health check: healthy engines only
    LOAD_BALANCED_CONFIG,          # Load balancing: round robin
    WEIGHTED_CONFIG,               # Weighted: weight-based selection
)
```

## Builder Pattern Configuration

```python
from common.engines import FallbackConfig, FallbackStrategy

config = FallbackConfig()
config = config.with_strategy(FallbackStrategy.ROUND_ROBIN)
config = config.with_retry(count=3, delay_seconds=1.0)
config = config.with_health_check(enabled=True, skip_unhealthy=True)
config = config.with_timeout(30.0)
config = config.with_weights(truthound=2.0, ge=1.0)

chain = EngineChain([engine1, engine2], config=config)
```

## Factory Functions

```python
from common.engines import (
    create_fallback_chain,
    create_load_balanced_chain,
)

# Simple fallback chain
chain = create_fallback_chain(
    primary,
    backup1,
    backup2,
    retry_count=2,
    check_health=True,
    name="production_chain",
)

# Load-balanced chain
chain = create_load_balanced_chain(
    engine1,
    engine2,
    engine3,
    strategy=FallbackStrategy.WEIGHTED,
    weights={"engine1": 3.0, "engine2": 2.0, "engine3": 1.0},
)
```

## Conditional Routing

```python
from common.engines import ConditionalEngineChain

chain = ConditionalEngineChain(name="smart_router")

# Add conditions
chain.add_route(
    lambda data, rules: len(data) > 1_000_000,  # Large data
    heavy_engine,
    priority=10,
    name="large_data",
)

chain.add_route(
    lambda data, rules: any(r.get("type") == "regex" for r in rules),
    regex_engine,
    priority=5,
    name="regex_rules",
)

# Default engine
chain.set_default_engine(general_engine)

result = chain.check(data, rules)
```

## Hooks

```python
from common.engines import (
    LoggingChainHook,
    MetricsChainHook,
    CompositeChainHook,
)

logging_hook = LoggingChainHook()
metrics_hook = MetricsChainHook()
composite = CompositeChainHook([logging_hook, metrics_hook])

chain = EngineChain([primary, backup], hooks=[composite])
result = chain.check(data, rules)

# Query metrics
print(metrics_hook.get_chain_success_rate("chain"))
print(metrics_hook.get_fallback_rate("chain"))
print(metrics_hook.get_average_duration_ms("chain"))
```

## Execution Result Query

```python
chain = EngineChain([primary, backup])
result = chain.check(data, rules)

exec_result = chain.last_execution_result

print(f"Success: {exec_result.success}")
print(f"Final engine: {exec_result.final_engine}")
print(f"Attempt count: {exec_result.attempt_count}")
print(f"Total duration: {exec_result.total_duration_ms}ms")
```

## Exception Handling

```python
from common.engines import (
    AllEnginesFailedError,
    NoEngineSelectedError,
    EngineChainConfigError,
)

try:
    result = chain.check(data, rules)
except AllEnginesFailedError as e:
    print(f"All engines failed: {e.attempted_engines}")
except NoEngineSelectedError as e:
    print(f"No engine selected: {e.chain_name}")
```

## Async Chain

```python
from common.engines import AsyncEngineChain, create_async_fallback_chain

chain = AsyncEngineChain([async_engine1, async_engine2])
result = await chain.check(data, rules)
```
