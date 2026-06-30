!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: API Reference
---

# API Reference

## Shared Runtime APIs

### Engine Creation

```python
from common.engines import EngineCreationRequest, create_engine
```

Use `create_engine(...)` with either a string engine name or an `EngineCreationRequest`.

### Runtime Context

```python
from common.engines import PlatformRuntimeContext, AutoConfigPolicy, normalize_runtime_context
```

Use runtime context objects to describe the platform, connection hints, source descriptors, and zero-config policy.

### Source Resolution

```python
from common.engines import resolve_data_source
```

Supported normalized source kinds:

- dataframe
- local path
- remote URI
- SQL
- callable
- generic object

### 호환성 And Preflight

```python
from common.engines import build_compatibility_report, run_preflight
```

These APIs validate engine resolution, serializer readiness, and whether a source needs credentials that have not been supplied.

## Result Serialization

```python
from common.serializers import serialize_result_wire
```

The shared wire format is the contract that platform packages should build on.
