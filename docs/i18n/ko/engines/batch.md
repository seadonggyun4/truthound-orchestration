!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Batch Processing
---

# Batch Processing

Large 검증s often need chunking, aggregation, and worker control instead of a
single eager call. The engine layer provides batch executors for that use case.

## Main Components

- `BatchExecutor`
- `AsyncBatchExecutor`
- `BatchConfig`
- chunkers such as `RowCountChunker`, `PolarsChunker`, and `DatasetListChunker`
- hooks such as `LoggingBatchHook` and `MetricsBatchHook`

## Basic Usage

```python
from common.engines import BatchConfig, BatchExecutor, TruthoundEngine

engine = TruthoundEngine()
executor = BatchExecutor(engine, BatchConfig(batch_size=10000, max_workers=4))
result = executor.check_batch(large_dataframe, auto_schema=True)
```

## When To Use Batch Execution

- datasets are too large for one simple in-memory pass
- you need explicit worker and chunk-size control
- operators want aggregated results across many chunks

## Operational Choices

Choose:

- sequential execution for simpler failure analysis or tight resource limits
- parallel execution for large workloads on capable runners
- fail-fast behavior when the first serious violation should stop the job

## Related Pages

- [Streaming 검증](streaming.md)
- [Lifecycle Management](lifecycle.md)
