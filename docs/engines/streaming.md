# Streaming Validation (스트리밍 검증)

스트리밍 데이터를 배치 단위로 검증하는 기능입니다. Iterator/Generator 패턴으로 메모리 효율적 처리를 제공합니다.

## 개요

- **Generator 패턴**: 각 배치를 독립 `CheckResult`로 반환하여 메모리 효율적
- **Protocol 기반**: `StreamingEngine` / `AsyncStreamingEngine` Protocol 구현
- **Truthound 지원**: TruthoundEngine이 기본적으로 스트리밍 검증 지원
- **유연한 입력**: Iterator, Generator, Kafka/Kinesis 어댑터 등 모든 Iterable 지원

## 빠른 시작

```python
from common.engines import TruthoundEngine

engine = TruthoundEngine()

def data_stream():
    for chunk in read_large_file("data.csv", chunk_size=10000):
        yield chunk

for batch_result in engine.check_stream(data_stream(), batch_size=5000):
    print(f"Batch: {batch_result.status.name}")
    if batch_result.status.name == "FAILED":
        break
```

## API

### `check_stream(stream, *, batch_size, schema, auto_schema, **kwargs)`

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `stream` | Any (Iterable) | (필수) | 데이터 스트림 |
| `batch_size` | int | 1000 | 배치당 레코드 수 |
| `schema` | Any \| None | None | 검증 스키마 (`learn()`에서 획득) |
| `auto_schema` | bool | False | 첫 배치에서 자동 스키마 생성 |

### 반환값: `Iterator[CheckResult]`

각 배치에 대한 `CheckResult`를 Generator로 반환합니다.

## 사용 패턴

### 스키마 지정

```python
schema = engine.get_schema(baseline_df)
for result in engine.check_stream(stream, batch_size=1000, schema=schema):
    process(result)
```

### 자동 스키마

```python
for result in engine.check_stream(stream, auto_schema=True):
    process(result)
```

### Fail-Fast

```python
for result in engine.check_stream(stream):
    if result.status.name == "FAILED":
        handle_failure(result)
        break
```

### 결과 수집

```python
results = list(engine.check_stream(stream, batch_size=5000))
total_failed = sum(r.failed_count for r in results)
```

## StreamConfig

```python
from common.base import StreamConfig

config = StreamConfig(
    batch_size=5000,
    max_batches=100,
    fail_fast=True,
    timeout_per_batch_seconds=30.0,
)
config = config.with_batch_size(10000).with_fail_fast(True)
```

## Feature Detection

```python
from common.engines.base import supports_streaming

assert supports_streaming(TruthoundEngine()) is True
assert supports_streaming(GreatExpectationsAdapter()) is False
```

## 메트릭 수집

```python
from common.engines import InstrumentedEngine, StatsCollectorHook

hook = StatsCollectorHook()
instrumented = InstrumentedEngine(engine, hooks=[hook])

for result in instrumented.check_stream(stream, batch_size=1000):
    process(result)

print(f"Stream checks: {hook.get_stats().stream_check_count}")
```
