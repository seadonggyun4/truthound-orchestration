---
title: Batch Operations
---

# Batch Operations

A batch operation system for efficient processing of large-scale data.

## Basic Usage

```python
from common.engines import BatchExecutor, BatchConfig, TruthoundEngine

engine = TruthoundEngine()
config = BatchConfig(batch_size=10000, max_workers=4)
executor = BatchExecutor(engine, config)

result = executor.check_batch(large_dataframe, auto_schema=True)

print(f"Status: {result.status.name}")
print(f"Total chunks: {result.total_chunks}")
print(f"Passed: {result.passed_chunks}")
print(f"Duration: {result.duration_seconds:.2f}s")
```

## Preset Configurations

```python
from common.engines import (
    DEFAULT_BATCH_CONFIG,      # Default: 1000 rows, 4 workers
    PARALLEL_BATCH_CONFIG,     # Parallel: 8 workers
    SEQUENTIAL_BATCH_CONFIG,   # Sequential: single thread
    FAIL_FAST_BATCH_CONFIG,    # Stop on first failure
    LARGE_DATA_BATCH_CONFIG,   # Large data: 50000 rows
)

executor = BatchExecutor(engine, config=LARGE_DATA_BATCH_CONFIG)
```

## Execution Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SEQUENTIAL` | Process chunks sequentially | Memory-constrained environments |
| `PARALLEL` | Process chunks in parallel | Multi-core systems |
| `ADAPTIVE` | Automatic selection based on data size | General usage |

## Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| `MERGE` | Merge all results into one |
| `WORST` | Return worst status |
| `BEST` | Return best status |
| `MAJORITY` | Return majority status |
| `FIRST_FAILURE` | Stop and return on first failure |
| `ALL` | Return all individual results |

## Builder Pattern Configuration

```python
from common.engines import BatchConfig, ExecutionStrategy, AggregationStrategy

config = BatchConfig()
config = config.with_batch_size(5000)
config = config.with_max_workers(8)
config = config.with_execution_strategy(ExecutionStrategy.PARALLEL)
config = config.with_aggregation_strategy(AggregationStrategy.MERGE)
config = config.with_fail_fast(True)
config = config.with_timeout(300.0)
```

## Batch Operation Types

```python
# Validation batch
check_result = executor.check_batch(data, auto_schema=True)

# Profiling batch
profile_result = executor.profile_batch(data)

# Learning batch
learn_result = executor.learn_batch(data)
```

## Async Batch Operations

```python
from common.engines import AsyncBatchExecutor, SyncEngineAsyncAdapter

async_engine = SyncEngineAsyncAdapter(TruthoundEngine())
executor = AsyncBatchExecutor(async_engine, config)

async def validate_large_data(data):
    result = await executor.check_batch(data, auto_schema=True)
    return result
```

## Data Chunkers

```python
from common.engines import RowCountChunker, PolarsChunker, DatasetListChunker

# Row count-based chunking
chunker = RowCountChunker(chunk_size=5000)

# Polars-optimized chunking
chunker = PolarsChunker(chunk_size=10000)

# Dataset list chunking
chunker = DatasetListChunker()
```

## Hooks

```python
from common.engines import (
    LoggingBatchHook,
    MetricsBatchHook,
    CompositeBatchHook,
)

logging_hook = LoggingBatchHook()
metrics_hook = MetricsBatchHook()
composite = CompositeBatchHook([logging_hook, metrics_hook])

executor = BatchExecutor(engine, config, hooks=[composite])
result = executor.check_batch(data)

# Query metrics
print(metrics_hook.chunks_processed)
print(metrics_hook.chunks_failed)
print(metrics_hook.average_chunk_duration_ms)
```

## Exception Handling

```python
from common.engines import (
    BatchExecutionError,
    ChunkingError,
    AggregationError,
)

try:
    result = executor.check_batch(data)
except ChunkingError as e:
    print(f"Chunking failed: {e}")
except BatchExecutionError as e:
    print(f"Batch execution failed: {e}")
except AggregationError as e:
    print(f"Result aggregation failed: {e}")
```
