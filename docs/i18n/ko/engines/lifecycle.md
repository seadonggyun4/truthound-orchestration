!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Lifecycle Management
---

# Lifecycle Management

Long-running 오케스트레이션 hosts need more than a bare `check()` call. The engine layer
includes lifecycle helpers for startup, shutdown, health checks, and hook-based
instrumentation.

## Main Components

- `ManagedEngineMixin`
- `EngineLifecycleManager`
- `EngineHealthChecker`
- lifecycle hooks such as `LoggingLifecycleHook` and `MetricsLifecycleHook`

## When This Matters

Lifecycle management is especially important in:

- Airflow workers and sensors
- Dagster or Prefect processes with reusable resources or blocks
- services that hold engine instances across multiple requests

## Basic Pattern

```python
from common.engines import EngineLifecycleManager, TruthoundEngine

engine = TruthoundEngine()
manager = EngineLifecycleManager(engine)
manager.start()
health = manager.health_check()
manager.stop()
```

## Production Guidance

- use context managers where possible
- keep health checks separate from normal 검증 output
- wire lifecycle hooks into the same logging and metrics systems used by the host

## Related Pages

- [Batch Processing](batch.md)
- [Shared Runtime Overview](../common/index.md)
