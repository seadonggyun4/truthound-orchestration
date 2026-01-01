# Common Modules

> **Last Updated:** 2025-12-31
> **Document Version:** 2.0.0
> **Status:** Implementation Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Module Structure](#module-structure)
3. [engines/](#engines)
4. [Engine Configuration System](#engine-configuration-system)
5. [Engine Context Manager](#engine-context-manager)
6. [Async Engine Support](#async-engine-support)
7. [Result Aggregation](#result-aggregation)
8. [Engine Versioning](#engine-versioning)
9. [Engine Chain](#engine-chain)
10. [Plugin Discovery](#plugin-discovery)
11. [base.py](#basepy)
12. [config.py](#configpy)
13. [logging.py](#loggingpy)
14. [retry.py](#retrypy)
15. [serializers.py](#serializerspy)
16. [exceptions.py](#exceptionspy)
17. [testing.py](#testingpy)
18. [rule_validation.py](#rule_validationpy)
19. [Usage Examples](#usage-examples)

---

## Overview

### Purpose
`common/` 디렉토리는 모든 워크플로우 통합 패키지에서 공유하는 핵심 모듈을 포함합니다. **핵심은 `DataQualityEngine` Protocol**로, 이를 통해 Truthound, Great Expectations, Pandera 등 다양한 데이터 품질 엔진을 추상화합니다.

### Design Principles

| Principle | Description |
|-----------|-------------|
| **Engine-Agnostic** | DataQualityEngine Protocol로 엔진 추상화 |
| **DRY** | 중복 코드 제거 |
| **Protocol-Based** | 추상 클래스 대신 Protocol 사용 |
| **Immutable** | 가능한 불변 데이터 구조 사용 |
| **Type-Safe** | 완전한 타입 힌트 |
| **Testable** | 쉬운 Mock 생성 |

### Architecture

```
common/
├── __init__.py           # Public API exports
├── base.py               # Protocols (DataQualityEngine 포함), Config, Result types
├── config.py             # Environment & platform config
├── logging.py            # Structured logging & masking
├── retry.py              # Retry decorator & backoff strategies
├── serializers.py        # Serialization utilities
├── exceptions.py         # Exception hierarchy
├── testing.py            # Test utilities & mocks (sync + async)
├── rule_validation.py    # Rule schema validation & normalization
├── engines/              # Engine implementations
│   ├── __init__.py       # Engine registry & exports
│   ├── base.py           # DataQualityEngine, AsyncDataQualityEngine Protocol
│   ├── aggregation.py    # Result aggregation (multi-engine, strategies, comparison)
│   ├── batch.py          # Batch operations (BatchExecutor, chunking, aggregation)
│   ├── chain.py          # Engine chain/fallback (EngineChain, ConditionalChain, strategies)
│   ├── config.py         # Engine configuration system (Builder, Loader, Validator)
│   ├── context.py        # Context management (EngineContext, EngineSession, savepoints)
│   ├── lifecycle.py      # ManagedEngine, AsyncManagedEngine, lifecycle management
│   ├── metrics.py        # Engine metrics (InstrumentedEngine, hooks)
│   ├── plugin.py         # Plugin discovery (entry points, sources, validation)
│   ├── registry.py       # Engine registry (get_engine, register_engine)
│   ├── version.py        # Semantic versioning, compatibility checking
│   ├── truthound.py      # Truthound engine (default)
│   ├── great_expectations.py  # GE adapter
│   └── pandera.py        # Pandera adapter
└── py.typed              # PEP 561 marker
```

---

## Module Structure

### __init__.py

```python
# common/__init__.py
"""
Data Quality Orchestration - Common Module.

이 모듈은 모든 워크플로우 통합 패키지에서 공유하는
핵심 타입, 프로토콜, 유틸리티를 제공합니다.
DataQualityEngine Protocol을 통해 다양한 엔진을 지원합니다.
"""

from common.base import (
    # Core Engine Protocol
    DataQualityEngine,
    AsyncDataQualityEngine,
    # Workflow Protocols
    WorkflowIntegration,
    AsyncWorkflowIntegration,
    # Enums
    CheckStatus,
    Severity,
    FailureAction,
    SerializationFormat,
    # Config Types
    CheckConfig,
    ProfileConfig,
    LearnConfig,
    # Result Types
    ValidationFailure,
    CheckResult,
    ProfileResult,
    LearnResult,
)
from common.engines import (
    # Engine Registry
    EngineRegistry,
    get_engine,
    register_engine,
    # Engine Implementations
    TruthoundEngine,
    GreatExpectationsAdapter,
    PanderaAdapter,
)
from common.config import (
    DataQualityConfig,
    load_config,
    get_platform_config,
)
from common.serializers import (
    Serializer,
    JSONSerializer,
    DictSerializer,
    SerializerFactory,
)
from common.exceptions import (
    DataQualityIntegrationError,
    ConfigurationError,
    EngineNotFoundError,
    ValidationExecutionError,
    SerializationError,
    PlatformConnectionError,
    TimeoutError,
    QualityGateError,
)
from common.testing import (
    MockTruthound,
    create_mock_check_result,
    create_mock_profile_result,
    assert_check_result,
)

__all__ = [
    # Protocols
    "WorkflowIntegration",
    "AsyncWorkflowIntegration",
    # Enums
    "CheckStatus",
    "Severity",
    "FailureAction",
    "SerializationFormat",
    # Configs
    "CheckConfig",
    "ProfileConfig",
    "LearnConfig",
    "TruthoundConfig",
    "load_config",
    "get_platform_config",
    # Results
    "ValidationFailure",
    "CheckResult",
    "ProfileResult",
    "LearnResult",
    # Serializers
    "Serializer",
    "JSONSerializer",
    "DictSerializer",
    "SerializerFactory",
    # Exceptions
    "TruthoundIntegrationError",
    "ConfigurationError",
    "ValidationExecutionError",
    "SerializationError",
    "PlatformConnectionError",
    "TimeoutError",
    "QualityGateError",
    # Testing
    "MockTruthound",
    "create_mock_check_result",
    "create_mock_profile_result",
    "assert_check_result",
]

__version__ = "0.1.0"
```

---

## engines/

`engines/` 디렉토리는 데이터 품질 엔진 구현체와 라이프사이클 관리를 담당합니다.

### 주요 컴포넌트

| 파일 | 설명 |
|-----|------|
| `base.py` | DataQualityEngine, AsyncDataQualityEngine Protocol |
| `batch.py` | 배치 작업 (BatchExecutor, 청킹, 결과 집계) |
| `config.py` | 엔진 설정 시스템 (Builder, Loader, Validator, Registry) |
| `lifecycle.py` | ManagedEngine, AsyncManagedEngine, 라이프사이클 관리 |
| `version.py` | 시맨틱 버저닝, 버전 제약조건, 호환성 검사, 버전 레지스트리 |
| `registry.py` | 엔진 레지스트리 (get_engine, register_engine) |
| `truthound.py` | Truthound 엔진 구현체 (기본값) |
| `great_expectations.py` | Great Expectations 어댑터 |
| `pandera.py` | Pandera 어댑터 |

### 엔진 Protocols

```python
from common.engines import DataQualityEngine, AsyncDataQualityEngine

# 동기 엔진 Protocol
@runtime_checkable
class DataQualityEngine(Protocol):
    @property
    def engine_name(self) -> str: ...
    @property
    def engine_version(self) -> str: ...
    def check(self, data, rules, **kwargs) -> CheckResult: ...
    def profile(self, data, **kwargs) -> ProfileResult: ...
    def learn(self, data, **kwargs) -> LearnResult: ...

# 비동기 엔진 Protocol
@runtime_checkable
class AsyncDataQualityEngine(Protocol):
    @property
    def engine_name(self) -> str: ...
    @property
    def engine_version(self) -> str: ...
    async def check(self, data, rules, **kwargs) -> CheckResult: ...
    async def profile(self, data, **kwargs) -> ProfileResult: ...
    async def learn(self, data, **kwargs) -> LearnResult: ...
```

### 라이프사이클 관리

```python
from common.engines import (
    ManagedEngine,
    AsyncManagedEngine,
    EngineState,
    EngineLifecycleManager,
    AsyncEngineLifecycleManager,
)

# 동기 라이프사이클
with EngineLifecycleManager(engine) as managed:
    result = managed.check(data, rules)
    health = managed.health_check()

# 비동기 라이프사이클
async with AsyncEngineLifecycleManager(async_engine) as managed:
    result = await managed.check(data, rules)
    health = await managed.health_check()
```

---

## Engine Configuration System

`engines/config.py` 모듈은 엔진별 설정을 유연하게 관리하기 위한 시스템을 제공합니다. 환경변수, 파일, 딕셔너리 등 다양한 소스에서 설정을 로드하고 검증할 수 있습니다.

### 핵심 클래스

| 클래스 | 설명 |
|-------|------|
| `BaseEngineConfig` | 모든 엔진 설정의 기본 클래스 (builder 메서드 포함) |
| `ConfigBuilder` | 플루언트 빌더 패턴으로 설정 생성 |
| `ConfigLoader` | 환경변수, JSON, YAML, TOML 파일에서 설정 로드 |
| `ConfigValidator` | 커스텀 제약조건으로 설정 검증 |
| `ConfigRegistry` | 명명된 설정 저장 및 관리 |
| `EnvironmentConfig` | 환경별 (dev, test, staging, prod) 설정 관리 |

### 설정 소스 및 환경

```python
from common.engines import ConfigSource, ConfigEnvironment

# 설정 소스
ConfigSource.ENV       # 환경변수
ConfigSource.FILE      # 파일 (JSON, YAML, TOML)
ConfigSource.DICT      # 딕셔너리
ConfigSource.DEFAULT   # 기본값

# 환경
ConfigEnvironment.DEVELOPMENT
ConfigEnvironment.TESTING
ConfigEnvironment.STAGING
ConfigEnvironment.PRODUCTION
```

### BaseEngineConfig

모든 엔진 설정의 기본 클래스입니다. 불변(frozen) dataclass이며 builder 패턴을 지원합니다.

```python
from dataclasses import dataclass
from common.engines.config import BaseEngineConfig

@dataclass(frozen=True, slots=True)
class BaseEngineConfig:
    """엔진 설정 기본 클래스."""
    auto_start: bool = False
    auto_stop: bool = True
    health_check_enabled: bool = True
    health_check_interval_seconds: float = 30.0
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Builder 메서드
    def with_auto_start(self, enabled: bool) -> Self: ...
    def with_auto_stop(self, enabled: bool) -> Self: ...
    def with_health_check_enabled(self, enabled: bool) -> Self: ...
    def with_health_check_interval(self, seconds: float) -> Self: ...
    def with_timeouts(self, startup: float, shutdown: float) -> Self: ...
    def with_retries(self, max_retries: int, delay: float) -> Self: ...

    # 직렬화
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...
```

### 엔진별 설정

```python
from common.engines import (
    TruthoundEngineConfig,
    GreatExpectationsConfig,
    PanderaConfig,
)

# Truthound 설정
config = TruthoundEngineConfig(
    auto_start=True,
    parallel=True,
    max_workers=4,
    min_severity="medium",
    cache_schemas=True,
    infer_constraints=True,
    categorical_threshold=20,
)

# Builder 패턴
config = (
    TruthoundEngineConfig()
    .with_auto_start(True)
    .with_parallel(True, max_workers=4)
    .with_min_severity("medium")
    .with_cache_schemas(True)
)

# Great Expectations 설정
ge_config = GreatExpectationsConfig(
    result_format="COMPLETE",
    context_root_dir="/path/to/context",
    include_profiling=True,
    catch_exceptions=True,
    enable_data_docs=True,
)

# Pandera 설정
pandera_config = PanderaConfig(
    lazy=True,
    strict=True,
    coerce=False,
    unique_column_names=True,
    report_duplicates="all",
)
```

### ConfigLoader

다양한 소스에서 설정을 로드합니다.

```python
from common.engines import ConfigLoader, TruthoundEngineConfig

loader = ConfigLoader(TruthoundEngineConfig)

# 환경변수에서 로드
# TRUTHOUND_AUTO_START=true
# TRUTHOUND_PARALLEL=true
config = loader.from_env(prefix="TRUTHOUND")

# JSON 파일에서 로드
config = loader.from_file("config.json")

# YAML 파일에서 로드
config = loader.from_file("config.yaml")

# TOML 파일에서 로드
config = loader.from_file("config.toml")

# 딕셔너리에서 로드
config = loader.from_dict({
    "auto_start": True,
    "parallel": True,
    "max_workers": 4,
})

# 기존 설정과 병합
base = TruthoundEngineConfig(cache_schemas=True)
config = loader.from_env(prefix="TRUTHOUND", base_config=base)
```

### ConfigBuilder

플루언트 빌더 패턴으로 설정을 생성합니다.

```python
from common.engines import ConfigBuilder, TruthoundEngineConfig

builder = ConfigBuilder(TruthoundEngineConfig)

config = (
    builder
    .with_auto_start(True)
    .with_health_check_enabled(True)
    .set("parallel", True)
    .set("max_workers", 4)
    .set("min_severity", "medium")
    .merge({"cache_schemas": True})
    .build(validate=True)  # 검증 후 빌드
)
```

### ConfigValidator

설정을 검증합니다.

```python
from common.engines import (
    ConfigValidator,
    FieldConstraint,
    ValidationResult,
)

# 검증기 생성
validator = ConfigValidator()

# 필드별 제약조건 추가
validator.add_constraint(
    "max_workers",
    FieldConstraint(min_value=1, max_value=32),
)
validator.add_constraint(
    "min_severity",
    FieldConstraint(choices=["critical", "high", "medium", "low"]),
)
validator.add_constraint(
    "context_root_dir",
    FieldConstraint(pattern=r"^/[a-zA-Z0-9/_-]+$"),  # 정규식 패턴
)

# 검증 실행
result: ValidationResult = validator.validate(config)

if not result.is_valid:
    for error in result.errors:
        print(f"Field '{error.field}': {error.message}")
else:
    print("Configuration is valid")
```

### ConfigRegistry

명명된 설정을 저장하고 관리합니다.

```python
from common.engines import ConfigRegistry, TruthoundEngineConfig

registry = ConfigRegistry()

# 설정 등록
registry.register("fast", TruthoundEngineConfig(parallel=True, max_workers=8))
registry.register("safe", TruthoundEngineConfig(parallel=False))
registry.register("production", TruthoundEngineConfig(
    auto_start=True,
    parallel=True,
    max_workers=4,
    health_check_enabled=True,
))

# 설정 가져오기
config = registry.get("fast")

# 등록된 설정 목록
names = registry.list_names()  # ["fast", "safe", "production"]

# 설정 제거
registry.unregister("safe")
```

### EnvironmentConfig

환경별 설정을 관리합니다.

```python
from common.engines import (
    EnvironmentConfig,
    ConfigEnvironment,
    TruthoundEngineConfig,
)

# 환경별 설정 관리
env_config = EnvironmentConfig(TruthoundEngineConfig)

# 환경별 설정 등록
env_config.register(
    ConfigEnvironment.DEVELOPMENT,
    TruthoundEngineConfig(
        auto_start=False,
        parallel=False,
        health_check_enabled=False,
    ),
)

env_config.register(
    ConfigEnvironment.PRODUCTION,
    TruthoundEngineConfig(
        auto_start=True,
        parallel=True,
        max_workers=8,
        health_check_enabled=True,
        health_check_interval_seconds=60.0,
    ),
)

# 환경별 설정 가져오기
config = env_config.get(ConfigEnvironment.PRODUCTION)
```

### 프리셋 설정

미리 정의된 설정을 사용할 수 있습니다.

```python
from common.engines import (
    # Truthound 프리셋
    DEFAULT_TRUTHOUND_CONFIG,
    PARALLEL_TRUTHOUND_CONFIG,
    PRODUCTION_TRUTHOUND_CONFIG,

    # Great Expectations 프리셋
    DEFAULT_GE_CONFIG,
    PRODUCTION_GE_CONFIG,
    DEVELOPMENT_GE_CONFIG,

    # Pandera 프리셋
    DEFAULT_PANDERA_CONFIG,
    PRODUCTION_PANDERA_CONFIG,
    DEVELOPMENT_PANDERA_CONFIG,
)

# 프리셋 기반으로 커스터마이징
config = PRODUCTION_TRUTHOUND_CONFIG.with_max_workers(8)
```

### 병합 전략

설정을 병합할 때 사용할 전략을 지정할 수 있습니다.

```python
from common.engines import MergeStrategy

# OVERRIDE: 새 값으로 덮어쓰기 (기본값)
# DEEP_MERGE: 중첩 딕셔너리 깊이 병합
# KEEP_FIRST: 기존 값 유지
# KEEP_LAST: 새 값 사용 (OVERRIDE와 동일)
```

### 편의 함수

```python
from common.engines import load_config, create_config_for_environment

# 파일에서 설정 로드 (확장자 자동 감지)
config = load_config(TruthoundEngineConfig, "config.yaml")

# 환경변수에서 로드
config = load_config(TruthoundEngineConfig, env_prefix="TRUTHOUND")

# 환경별 설정 생성
config = create_config_for_environment(
    TruthoundEngineConfig,
    "production",  # 문자열로 환경 지정
)
```

### 예외 타입

```python
from common.engines import (
    ConfigurationError,      # 설정 관련 기본 예외
    ConfigValidationError,   # 검증 실패
    ConfigLoadError,         # 로드 실패
)

try:
    config = loader.from_file("invalid.json")
except ConfigLoadError as e:
    print(f"Failed to load config: {e}")

try:
    validator.validate(config, raise_on_error=True)
except ConfigValidationError as e:
    print(f"Validation failed: {e.errors}")
```

---

## Engine Context Manager

`engines/context.py` 모듈은 `with engine:` 패턴과 트랜잭션/세션 관리를 위한 컨텍스트 매니저 시스템을 제공합니다.

### 핵심 컴포넌트

| 컴포넌트 | 설명 |
|---------|------|
| `EngineContext` | 기본 컨텍스트 매니저 (엔진 라이프사이클, 리소스 추적) |
| `EngineSession` | 트랜잭션 스타일 세션 (commit/rollback 시맨틱) |
| `MultiEngineContext` | 복수 엔진 조율 (병렬/순차 시작/종료) |
| `ContextStack` | 중첩 컨텍스트 스택 관리 |
| `AsyncEngineContext` | 비동기 컨텍스트 매니저 |
| `ResourceTracker` | 리소스 추적 및 LIFO 순서 정리 |
| `SavepointManager` | 세이브포인트 생성/롤백 관리 |

### Enums

```python
from common.engines import ContextState, SessionState, CleanupStrategy

# 컨텍스트 상태
class ContextState(Enum):
    CREATED = "created"       # 생성됨
    ENTERING = "entering"     # 진입 중
    ACTIVE = "active"         # 활성
    EXITING = "exiting"       # 종료 중
    EXITED = "exited"         # 종료됨
    FAILED = "failed"         # 실패

# 세션 상태
class SessionState(Enum):
    PENDING = "pending"           # 대기 중
    ACTIVE = "active"             # 활성
    COMMITTING = "committing"     # 커밋 중
    COMMITTED = "committed"       # 커밋됨
    ROLLING_BACK = "rolling_back" # 롤백 중
    ROLLED_BACK = "rolled_back"   # 롤백됨
    FAILED = "failed"             # 실패

# 정리 전략
class CleanupStrategy(Enum):
    ALWAYS = "always"         # 항상 정리
    ON_SUCCESS = "on_success" # 성공시에만 정리
    ON_FAILURE = "on_failure" # 실패시에만 정리
    NEVER = "never"           # 정리하지 않음
```

### ContextConfig

컨텍스트 설정을 위한 불변 dataclass입니다.

```python
from common.engines import ContextConfig, CleanupStrategy

@dataclass(frozen=True, slots=True)
class ContextConfig:
    auto_start_engine: bool = True       # 진입시 엔진 자동 시작
    auto_stop_engine: bool = True        # 종료시 엔진 자동 중지
    cleanup_strategy: CleanupStrategy = CleanupStrategy.ALWAYS
    track_resources: bool = True         # 리소스 추적 활성화
    enable_savepoints: bool = True       # 세이브포인트 활성화
    propagate_exceptions: bool = True    # 정리 후 예외 재발생
    cleanup_timeout_seconds: float = 30.0  # 정리 타임아웃

# 프리셋 설정
from common.engines import (
    DEFAULT_CONTEXT_CONFIG,      # 기본 설정
    LIGHTWEIGHT_CONTEXT_CONFIG,  # 경량 (리소스 추적 비활성화)
    STRICT_CONTEXT_CONFIG,       # 엄격 (정리 실패시 예외)
    TESTING_CONTEXT_CONFIG,      # 테스트용
)
```

### EngineContext

기본 컨텍스트 매니저입니다. 엔진 라이프사이클 관리와 리소스 추적을 제공합니다.

```python
from common.engines import EngineContext, TruthoundEngine

# 기본 사용법
with EngineContext(TruthoundEngine()) as ctx:
    result = ctx.engine.check(data, auto_schema=True)
    # 엔진 자동 시작/종료

# 리소스 추적
with EngineContext(TruthoundEngine()) as ctx:
    ctx.track_resource(connection, cleanup_fn=lambda c: c.close())
    ctx.track_resource(temp_file, cleanup_fn=lambda f: f.unlink())
    # LIFO 순서로 자동 정리

# 설정 사용
config = ContextConfig(
    auto_start_engine=True,
    cleanup_strategy=CleanupStrategy.ALWAYS,
    track_resources=True,
)
with EngineContext(engine, config=config) as ctx:
    result = ctx.engine.check(data)
```

### EngineSession

트랜잭션 스타일 세션 관리를 제공합니다.

```python
from common.engines import EngineSession, TruthoundEngine

# commit/rollback 시맨틱
with EngineSession(TruthoundEngine()) as session:
    result1 = session.execute(lambda e: e.check(data1, auto_schema=True))
    result2 = session.execute(lambda e: e.check(data2, auto_schema=True))

    if all_valid(result1, result2):
        session.commit()   # 성공 표시
    else:
        session.rollback() # 롤백 (정리 트리거)

# 상태 전이: PENDING -> ACTIVE -> COMMITTING/ROLLING_BACK -> COMMITTED/ROLLED_BACK
```

### SavepointManager

세이브포인트 패턴을 통한 부분 롤백을 지원합니다.

```python
from common.engines import EngineSession

with EngineSession(engine) as session:
    # 위험 작업 전 세이브포인트 생성
    savepoint = session.create_savepoint("before_transform")

    try:
        result = session.execute(risky_transform)
    except TransformError:
        # 세이브포인트로 롤백 후 대안 실행
        session.rollback_to_savepoint(savepoint)
        result = session.execute(safe_transform)

    session.commit()
```

### MultiEngineContext

복수 엔진을 조율합니다.

```python
from common.engines import MultiEngineContext, TruthoundEngine, GreatExpectationsAdapter

engines = {
    "truthound": TruthoundEngine(),
    "ge": GreatExpectationsAdapter(),
}

with MultiEngineContext(engines) as ctx:
    result1 = ctx.get_engine("truthound").check(data)
    result2 = ctx.get_engine("ge").check(data, rules)
    # 모든 엔진 함께 시작/종료

# 병렬 시작
with MultiEngineContext(engines, parallel=True, max_workers=4) as ctx:
    ...
```

### ContextStack

중첩 컨텍스트를 관리합니다.

```python
from common.engines import ContextStack, EngineContext

stack = ContextStack()

# 스택에 컨텍스트 푸시
ctx1 = stack.push(EngineContext(engine1))
ctx2 = stack.push(EngineContext(engine2))

# 현재 컨텍스트
current = stack.current()

# LIFO 순서로 팝 (정리)
stack.pop()  # ctx2 종료
stack.pop()  # ctx1 종료

# 컨텍스트 매니저로 사용
with ContextStack() as stack:
    stack.push(EngineContext(engine1))
    stack.push(EngineContext(engine2))
    # 종료시 모든 컨텍스트 정리
```

### AsyncEngineContext

비동기 컨텍스트 매니저입니다.

```python
from common.engines import AsyncEngineContext, SyncEngineAsyncAdapter

# 동기 엔진을 비동기로 래핑
async_engine = SyncEngineAsyncAdapter(TruthoundEngine())

async with AsyncEngineContext(async_engine) as ctx:
    result = await ctx.engine.check(data, auto_schema=True)

# 리소스 추적 포함
async with AsyncEngineContext(async_engine) as ctx:
    ctx.track_resource(connection, cleanup_fn=lambda c: c.close())
    result = await ctx.engine.check(data)
```

### Context Hooks

컨텍스트 이벤트를 관찰합니다.

```python
from common.engines import (
    ContextHook,
    LoggingContextHook,
    MetricsContextHook,
    CompositeContextHook,
)

# ContextHook Protocol
class ContextHook(Protocol):
    def on_enter(self, context: EngineContext) -> None: ...
    def on_exit(self, context: EngineContext, exception: Exception | None) -> None: ...
    def on_resource_tracked(self, context: EngineContext, resource: TrackedResource) -> None: ...
    def on_resource_cleanup(self, context: EngineContext, resource: TrackedResource, success: bool) -> None: ...
    def on_savepoint_created(self, context: EngineContext, savepoint: Savepoint) -> None: ...
    def on_savepoint_rollback(self, context: EngineContext, savepoint: Savepoint) -> None: ...

# 로깅 훅
logging_hook = LoggingContextHook()

# 메트릭 훅
metrics_hook = MetricsContextHook()

# 복합 훅
composite = CompositeContextHook([logging_hook, metrics_hook])

with EngineContext(engine, hooks=[composite]) as ctx:
    result = ctx.engine.check(data)
```

### 팩토리 함수

```python
from common.engines import (
    create_engine_context,
    create_engine_session,
    create_multi_engine_context,
    engine_context,
    async_engine_context,
)

# 팩토리 함수
ctx = create_engine_context(engine, config=config)
session = create_engine_session(engine)
multi_ctx = create_multi_engine_context(engines)

# 데코레이터 스타일
@engine_context(TruthoundEngine())
def process_data(ctx, data):
    return ctx.engine.check(data)

# 비동기 데코레이터
@async_engine_context(async_engine)
async def async_process(ctx, data):
    return await ctx.engine.check(data)
```

### 예외 계층

```python
from common.engines import (
    ContextError,              # 컨텍스트 기본 예외
    ContextNotActiveError,     # 비활성 컨텍스트 작업 시도
    ContextAlreadyActiveError, # 이미 활성화된 컨텍스트 진입 시도
    SavepointError,            # 세이브포인트 작업 실패
    ResourceCleanupError,      # 리소스 정리 실패
)

try:
    with EngineContext(engine) as ctx:
        result = ctx.engine.check(data)
except ContextNotActiveError:
    print("컨텍스트가 활성화되지 않음")
except ResourceCleanupError as e:
    print(f"리소스 정리 실패: {e}")
```

### 스레드 안전성

모든 컨텍스트 관리 작업은 스레드 안전합니다. `EngineContext`와 `EngineSession`은 `threading.RLock`을 사용하여 동시 접근 시 일관된 상태 전이를 보장합니다.

```python
from concurrent.futures import ThreadPoolExecutor
from common.engines import EngineContext, TruthoundEngine

# 안전한 동시 컨텍스트 생성
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(
            lambda d: EngineContext(TruthoundEngine()).__enter__().engine.check(d),
            data
        )
        for data in datasets
    ]
    results = [f.result() for f in futures]
```

---

## Async Engine Support

Prefect, FastAPI 등 비동기 프레임워크와의 통합을 위한 비동기 엔진 지원입니다.

### SyncEngineAsyncAdapter

기존 동기 엔진을 비동기 컨텍스트에서 사용할 수 있는 어댑터입니다.

```python
from common.engines import TruthoundEngine, SyncEngineAsyncAdapter

# 동기 엔진을 비동기로 래핑
sync_engine = TruthoundEngine()
async_engine = SyncEngineAsyncAdapter(sync_engine)

# 비동기 컨텍스트에서 사용
async def validate_data(data):
    async with async_engine:
        result = await async_engine.check(data, auto_schema=True)
        health = await async_engine.health_check()
        return result
```

### AsyncManagedEngineMixin

커스텀 비동기 엔진에 라이프사이클 관리를 추가하는 Mixin입니다.

```python
from common.engines import AsyncManagedEngineMixin
from common.engines.base import EngineInfoMixin

class MyAsyncEngine(AsyncManagedEngineMixin, EngineInfoMixin):
    @property
    def engine_name(self) -> str:
        return "my_async_engine"

    @property
    def engine_version(self) -> str:
        return "1.0.0"

    async def _do_start(self) -> None:
        self._connection = await create_async_connection()

    async def _do_stop(self) -> None:
        await self._connection.close()

    async def _do_health_check(self) -> HealthCheckResult:
        if await self._connection.ping():
            return HealthCheckResult.healthy(self.engine_name)
        return HealthCheckResult.unhealthy(self.engine_name)

    async def check(self, data, rules, **kwargs):
        # 비동기 검증 로직
        ...

# 사용
async with MyAsyncEngine() as engine:
    result = await engine.check(data, rules)
```

### 비동기 라이프사이클 훅

```python
from common.engines import (
    AsyncLoggingLifecycleHook,
    AsyncMetricsLifecycleHook,
    AsyncCompositeLifecycleHook,
    SyncToAsyncLifecycleHookAdapter,
)

# 네이티브 비동기 훅
logging_hook = AsyncLoggingLifecycleHook()
metrics_hook = AsyncMetricsLifecycleHook()

# 복합 훅
composite = AsyncCompositeLifecycleHook([logging_hook, metrics_hook])

# 기존 동기 훅을 비동기로 어댑팅
from common.engines import LoggingLifecycleHook
sync_hook = LoggingLifecycleHook()
async_hook = SyncToAsyncLifecycleHookAdapter(sync_hook)
```

### FastAPI 통합 예시

```python
from fastapi import FastAPI
from common.engines import SyncEngineAsyncAdapter, TruthoundEngine

app = FastAPI()
engine: SyncEngineAsyncAdapter | None = None

@app.on_event("startup")
async def startup():
    global engine
    engine = SyncEngineAsyncAdapter(TruthoundEngine())
    await engine.start()

@app.on_event("shutdown")
async def shutdown():
    if engine:
        await engine.stop()

@app.post("/validate")
async def validate_data(data: dict):
    result = await engine.check(data, auto_schema=True)
    return {"status": result.status.name}

@app.get("/health")
async def health_check():
    result = await engine.health_check()
    return {"status": result.status.name}
```

### Prefect 통합 예시

```python
from prefect import flow, task
from common.engines import SyncEngineAsyncAdapter, TruthoundEngine

@task
async def validate_data(data):
    async with SyncEngineAsyncAdapter(TruthoundEngine()) as engine:
        result = await engine.check(data, auto_schema=True)
        if result.status.name == "FAILED":
            raise ValueError(f"Validation failed: {result.failed_count} failures")
        return result

@flow
async def data_quality_flow():
    data = await load_data()
    result = await validate_data(data)
    return result
```

---

## Batch Operations (engines/batch.py)

`engines/batch.py` 모듈은 대용량 데이터 처리를 위한 배치 작업 시스템을 제공합니다. 데이터 청킹, 병렬/순차 실행, 결과 집계를 지원합니다.

### 핵심 클래스

| 클래스 | 설명 |
|-------|------|
| `BatchConfig` | 배치 설정 (불변 dataclass, builder 패턴 지원) |
| `BatchExecutor` | 동기 배치 실행기 |
| `AsyncBatchExecutor` | 비동기 배치 실행기 |
| `BatchResult` | 배치 실행 결과 |
| `BatchItemResult` | 개별 청크 결과 |

### 전략 열거형

```python
from enum import Enum, auto

class ExecutionStrategy(Enum):
    """실행 전략"""
    SEQUENTIAL = auto()  # 순차 실행
    PARALLEL = auto()    # 병렬 실행
    ADAPTIVE = auto()    # 데이터 크기에 따라 자동 선택

class AggregationStrategy(Enum):
    """결과 집계 전략"""
    MERGE = auto()         # 모든 결과 병합
    WORST = auto()         # 가장 나쁜 상태 반환
    BEST = auto()          # 가장 좋은 상태 반환
    MAJORITY = auto()      # 다수 상태 반환
    FIRST_FAILURE = auto() # 첫 실패시 중단
    ALL = auto()           # 모든 개별 결과 반환

class ChunkingStrategy(Enum):
    """청킹 전략"""
    ROW_COUNT = auto()      # 행 수 기반
    BYTE_SIZE = auto()      # 바이트 크기 기반
    COLUMN_GROUPS = auto()  # 컬럼 그룹 기반
    CUSTOM = auto()         # 커스텀 청커
```

### BatchConfig

```python
from dataclasses import dataclass
from common.engines.batch import BatchConfig, ExecutionStrategy, AggregationStrategy

@dataclass(frozen=True, slots=True)
class BatchConfig:
    """배치 설정"""
    batch_size: int = 1000
    max_workers: int = 4
    execution_strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MERGE
    fail_fast: bool = False
    timeout_seconds: float | None = None
    retry_failed_chunks: bool = False
    max_chunk_retries: int = 3

    # Builder 메서드
    def with_batch_size(self, size: int) -> "BatchConfig": ...
    def with_max_workers(self, workers: int) -> "BatchConfig": ...
    def with_execution_strategy(self, strategy: ExecutionStrategy) -> "BatchConfig": ...
    def with_aggregation_strategy(self, strategy: AggregationStrategy) -> "BatchConfig": ...
    def with_fail_fast(self, enabled: bool) -> "BatchConfig": ...
    def with_timeout(self, seconds: float) -> "BatchConfig": ...
```

### 프리셋 설정

```python
from common.engines import (
    DEFAULT_BATCH_CONFIG,      # 기본: 1000행, 4 워커, PARALLEL
    PARALLEL_BATCH_CONFIG,     # 병렬: 8 워커
    SEQUENTIAL_BATCH_CONFIG,   # 순차: 1 워커
    FAIL_FAST_BATCH_CONFIG,    # 첫 실패시 중단
    LARGE_DATA_BATCH_CONFIG,   # 대용량: 50000행
)

# 프리셋 커스터마이징
config = LARGE_DATA_BATCH_CONFIG.with_max_workers(16)
```

### BatchExecutor (동기)

```python
from common.engines import BatchExecutor, BatchConfig, TruthoundEngine

engine = TruthoundEngine()
config = BatchConfig(batch_size=10000, max_workers=4)
executor = BatchExecutor(engine, config)

# 배치 검증
result = executor.check_batch(data, auto_schema=True)
print(f"Total chunks: {result.total_chunks}")
print(f"Passed: {result.passed_chunks}")
print(f"Failed: {result.failed_chunks}")
print(f"Duration: {result.duration_seconds:.2f}s")

# 배치 프로파일링
profile_result = executor.profile_batch(data)

# 배치 학습
learn_result = executor.learn_batch(data)
```

### AsyncBatchExecutor (비동기)

```python
from common.engines import AsyncBatchExecutor, SyncEngineAsyncAdapter

async_engine = SyncEngineAsyncAdapter(TruthoundEngine())
executor = AsyncBatchExecutor(async_engine, config)

async def validate_large_data(data):
    result = await executor.check_batch(data, auto_schema=True)
    return result

# 여러 데이터셋 동시 처리
async def process_multiple(datasets):
    tasks = [executor.check_batch(ds) for ds in datasets]
    results = await asyncio.gather(*tasks)
    return results
```

### DataChunker Protocol

```python
from typing import Protocol, Any, Iterator

class DataChunker(Protocol):
    """데이터 청킹 프로토콜"""
    def chunk(self, data: Any, chunk_size: int) -> Iterator[Any]:
        """데이터를 청크로 분할"""
        ...

    def get_total_size(self, data: Any) -> int:
        """전체 데이터 크기 반환"""
        ...
```

### 청커 구현체

```python
from common.engines import RowCountChunker, PolarsChunker, DatasetListChunker

# 행 수 기반 청커 (일반적인 사용)
chunker = RowCountChunker(chunk_size=5000)
for chunk in chunker.chunk(data, chunk_size=5000):
    process(chunk)

# Polars 최적화 청커
chunker = PolarsChunker(chunk_size=10000)

# 데이터셋 리스트 청커
chunker = DatasetListChunker()
datasets = [df1, df2, df3]
for ds in chunker.chunk(datasets, chunk_size=1):
    process(ds)

# 커스텀 청커
class MyChunker:
    def chunk(self, data, chunk_size: int):
        yield from custom_logic(data, chunk_size)

    def get_total_size(self, data) -> int:
        return calculate_size(data)
```

### ResultAggregator Protocol

```python
from typing import Protocol, Sequence

class ResultAggregator(Protocol[T]):
    """결과 집계 프로토콜"""
    def aggregate(self, results: Sequence[T]) -> T:
        """여러 결과를 하나로 집계"""
        ...
```

### 집계기 구현체

```python
from common.engines import (
    CheckResultAggregator,
    ProfileResultAggregator,
    LearnResultAggregator,
    AggregationStrategy,
)

# CheckResult 집계
aggregator = CheckResultAggregator(strategy=AggregationStrategy.MERGE)
combined = aggregator.aggregate(chunk_results)

# WORST 전략: 가장 나쁜 상태 반환
aggregator = CheckResultAggregator(strategy=AggregationStrategy.WORST)

# ProfileResult 집계
profile_aggregator = ProfileResultAggregator()
combined_profile = profile_aggregator.aggregate(profile_results)

# LearnResult 집계
learn_aggregator = LearnResultAggregator()
combined_learn = learn_aggregator.aggregate(learn_results)
```

### BatchHook Protocol

```python
from typing import Protocol

class BatchHook(Protocol):
    """배치 이벤트 훅 프로토콜"""

    def on_batch_start(self, total_chunks: int) -> None:
        """배치 시작 시 호출"""
        ...

    def on_chunk_start(self, chunk_index: int, chunk_size: int) -> None:
        """청크 처리 시작 시 호출"""
        ...

    def on_chunk_complete(
        self,
        chunk_index: int,
        result: Any,
        duration_ms: float,
    ) -> None:
        """청크 처리 완료 시 호출"""
        ...

    def on_chunk_error(
        self,
        chunk_index: int,
        error: Exception,
        duration_ms: float,
    ) -> None:
        """청크 처리 오류 시 호출"""
        ...

    def on_batch_complete(
        self,
        result: "BatchResult",
        duration_seconds: float,
    ) -> None:
        """배치 완료 시 호출"""
        ...
```

### 훅 구현체

```python
from common.engines import (
    LoggingBatchHook,
    MetricsBatchHook,
    CompositeBatchHook,
)

# 로깅 훅
logging_hook = LoggingBatchHook()

# 메트릭 훅
metrics_hook = MetricsBatchHook()

# 복합 훅
composite = CompositeBatchHook([logging_hook, metrics_hook])

# 훅 사용
executor = BatchExecutor(engine, config, hooks=[composite])
result = executor.check_batch(data)

# 메트릭 조회
print(f"Processed: {metrics_hook.chunks_processed}")
print(f"Failed: {metrics_hook.chunks_failed}")
print(f"Avg duration: {metrics_hook.average_chunk_duration_ms}ms")
print(f"Total duration: {metrics_hook.total_duration_seconds}s")
```

### BatchResult

```python
from dataclasses import dataclass
from common.base import CheckStatus

@dataclass(frozen=True, slots=True)
class BatchResult:
    """배치 실행 결과"""
    status: CheckStatus
    total_chunks: int
    passed_chunks: int
    failed_chunks: int
    skipped_chunks: int
    duration_seconds: float
    aggregated_result: CheckResult | ProfileResult | LearnResult | None
    chunk_results: tuple[BatchItemResult, ...] | None
    metadata: dict[str, Any]

    @property
    def success_rate(self) -> float:
        """성공률 계산"""
        if self.total_chunks == 0:
            return 0.0
        return self.passed_chunks / self.total_chunks

    @property
    def is_success(self) -> bool:
        """전체 성공 여부"""
        return self.status == CheckStatus.PASSED
```

### BatchItemResult

```python
@dataclass(frozen=True, slots=True)
class BatchItemResult:
    """개별 청크 결과"""
    chunk_index: int
    status: CheckStatus
    result: CheckResult | ProfileResult | LearnResult | None
    error: Exception | None
    duration_ms: float
    chunk_size: int
```

### 예외 계층

```python
from common.engines.batch import (
    BatchOperationError,    # 기본 예외
    BatchExecutionError,    # 실행 오류
    ChunkingError,          # 청킹 오류
    AggregationError,       # 집계 오류
)

try:
    result = executor.check_batch(data)
except ChunkingError as e:
    print(f"청킹 실패: {e}")
except BatchExecutionError as e:
    print(f"실행 실패: {e}")
    print(f"실패한 청크: {e.failed_chunks}")
except AggregationError as e:
    print(f"집계 실패: {e}")
```

### 사용 예시

```python
from common.engines import (
    TruthoundEngine,
    BatchExecutor,
    BatchConfig,
    ExecutionStrategy,
    AggregationStrategy,
    LoggingBatchHook,
    MetricsBatchHook,
    CompositeBatchHook,
)
import polars as pl

# 대용량 데이터 로드
data = pl.read_parquet("large_dataset.parquet")

# 엔진 및 설정 생성
engine = TruthoundEngine()
config = (
    BatchConfig()
    .with_batch_size(50000)
    .with_max_workers(8)
    .with_execution_strategy(ExecutionStrategy.PARALLEL)
    .with_aggregation_strategy(AggregationStrategy.MERGE)
    .with_fail_fast(False)
)

# 훅 설정
logging_hook = LoggingBatchHook()
metrics_hook = MetricsBatchHook()
hooks = [CompositeBatchHook([logging_hook, metrics_hook])]

# 실행
executor = BatchExecutor(engine, config, hooks=hooks)
result = executor.check_batch(data, auto_schema=True)

# 결과 확인
print(f"Status: {result.status.name}")
print(f"Chunks: {result.passed_chunks}/{result.total_chunks}")
print(f"Success rate: {result.success_rate:.1%}")
print(f"Duration: {result.duration_seconds:.2f}s")

# 메트릭 확인
print(f"Avg chunk time: {metrics_hook.average_chunk_duration_ms:.0f}ms")
```

---

## Engine Metrics Integration (engines/metrics.py)

엔진 작업(check, profile, learn)에 대한 메트릭 수집, 로깅, 분산 추적을 위한 통합 시스템입니다.

### 핵심 개념

| 컴포넌트 | 설명 |
|----------|------|
| `InstrumentedEngine` | 메트릭 수집을 추가하는 투명 엔진 래퍼 |
| `AsyncInstrumentedEngine` | 비동기 엔진용 메트릭 래퍼 |
| `EngineMetricsHook` | 엔진 작업 이벤트 수신 프로토콜 |
| `MetricsEngineHook` | Prometheus 스타일 메트릭 수집 |
| `LoggingEngineHook` | 엔진 작업 자동 로깅 |
| `TracingEngineHook` | 분산 추적 스팬 생성 |
| `StatsCollectorHook` | 인메모리 통계 수집 |

### EngineMetricsHook Protocol

```python
from typing import Protocol, Any
from common.base import CheckResult, ProfileResult, LearnResult

@runtime_checkable
class EngineMetricsHook(Protocol):
    """엔진 메트릭 수집을 위한 훅 프로토콜"""

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None: ...

    def on_check_end(
        self,
        engine_name: str,
        result: CheckResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None: ...

    def on_profile_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None: ...

    def on_profile_end(
        self,
        engine_name: str,
        result: ProfileResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None: ...

    def on_learn_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None: ...

    def on_learn_end(
        self,
        engine_name: str,
        result: LearnResult,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None: ...

    def on_error(
        self,
        engine_name: str,
        operation: EngineOperation,
        exception: Exception,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None: ...
```

### EngineMetricsConfig

```python
@dataclass(frozen=True, slots=True)
class EngineMetricsConfig:
    """엔진 메트릭 설정"""
    enabled: bool = True
    logging_enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    record_data_size: bool = True
    record_result_details: bool = True
    log_level: str = "INFO"

    def with_enabled(self, enabled: bool) -> EngineMetricsConfig:
        return dataclasses.replace(self, enabled=enabled)

    def with_logging_enabled(self, enabled: bool) -> EngineMetricsConfig:
        return dataclasses.replace(self, logging_enabled=enabled)

    def with_metrics_enabled(self, enabled: bool) -> EngineMetricsConfig:
        return dataclasses.replace(self, metrics_enabled=enabled)

    def with_tracing_enabled(self, enabled: bool) -> EngineMetricsConfig:
        return dataclasses.replace(self, tracing_enabled=enabled)
```

### 프리셋 설정

```python
DEFAULT_ENGINE_METRICS_CONFIG = EngineMetricsConfig()

DISABLED_ENGINE_METRICS_CONFIG = EngineMetricsConfig(
    enabled=False,
    logging_enabled=False,
    metrics_enabled=False,
    tracing_enabled=False,
)

MINIMAL_ENGINE_METRICS_CONFIG = EngineMetricsConfig(
    enabled=True,
    logging_enabled=True,
    metrics_enabled=False,
    tracing_enabled=False,
)

FULL_ENGINE_METRICS_CONFIG = EngineMetricsConfig(
    enabled=True,
    logging_enabled=True,
    metrics_enabled=True,
    tracing_enabled=True,
    record_data_size=True,
    record_result_details=True,
)
```

### EngineOperation 및 OperationStatus

```python
from enum import Enum, auto

class EngineOperation(Enum):
    """엔진 작업 유형"""
    CHECK = auto()
    PROFILE = auto()
    LEARN = auto()

class OperationStatus(Enum):
    """작업 상태"""
    STARTED = auto()
    COMPLETED = auto()
    FAILED = auto()
```

### InstrumentedEngine

```python
class InstrumentedEngine:
    """메트릭 수집이 포함된 투명 엔진 래퍼"""

    def __init__(
        self,
        engine: DataQualityEngine,
        hooks: Sequence[EngineMetricsHook] | None = None,
        config: EngineMetricsConfig | None = None,
    ) -> None:
        self._engine = engine
        self._hooks = list(hooks) if hooks else []
        self._config = config or DEFAULT_ENGINE_METRICS_CONFIG

    @property
    def engine_name(self) -> str:
        return self._engine.engine_name

    @property
    def engine_version(self) -> str:
        return self._engine.engine_version

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] = (),
        **kwargs: Any,
    ) -> CheckResult:
        """메트릭 수집과 함께 check 실행"""
        context = self._build_context(data, kwargs)
        data_size = self._get_data_size(data)

        # 훅 호출 (격리됨)
        self._call_hooks_check_start(data_size, context)

        start_time = time.perf_counter()
        try:
            result = self._engine.check(data, rules, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._call_hooks_check_end(result, duration_ms, context)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._call_hooks_error(EngineOperation.CHECK, e, duration_ms, context)
            raise
```

### StatsCollectorHook

```python
class StatsCollectorHook(BaseEngineMetricsHook):
    """인메모리 통계 수집 훅"""

    def __init__(self) -> None:
        self._check_count = 0
        self._profile_count = 0
        self._learn_count = 0
        self._check_success_count = 0
        self._profile_success_count = 0
        self._learn_success_count = 0
        self._error_count = 0
        self._total_duration_ms = 0.0
        self._lock = threading.Lock()

    def get_stats(self) -> EngineOperationStats:
        """통계 반환"""
        with self._lock:
            return EngineOperationStats(
                total_operations=self._check_count + self._profile_count + self._learn_count,
                successful_operations=self._check_success_count + self._profile_success_count + self._learn_success_count,
                failed_operations=self._error_count,
                check_count=self._check_count,
                profile_count=self._profile_count,
                learn_count=self._learn_count,
                check_success_count=self._check_success_count,
                profile_success_count=self._profile_success_count,
                learn_success_count=self._learn_success_count,
                total_duration_ms=self._total_duration_ms,
            )

    def reset(self) -> None:
        """통계 초기화"""
        with self._lock:
            self._check_count = 0
            # ... 모든 카운터 초기화
```

### EngineOperationStats

```python
@dataclass(frozen=True, slots=True)
class EngineOperationStats:
    """엔진 작업 통계"""
    total_operations: int
    successful_operations: int
    failed_operations: int
    check_count: int
    profile_count: int
    learn_count: int
    check_success_count: int
    profile_success_count: int
    learn_success_count: int
    total_duration_ms: float

    @property
    def average_duration_ms(self) -> float:
        """평균 작업 시간"""
        if self.total_operations == 0:
            return 0.0
        return self.total_duration_ms / self.total_operations

    @property
    def check_success_rate(self) -> float:
        """Check 성공률"""
        if self.check_count == 0:
            return 0.0
        return self.check_success_count / self.check_count

    @property
    def overall_success_rate(self) -> float:
        """전체 성공률"""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
```

### CompositeEngineHook

```python
class CompositeEngineHook(BaseEngineMetricsHook):
    """여러 훅을 조합하는 복합 훅"""

    def __init__(self, hooks: Sequence[EngineMetricsHook]) -> None:
        self._hooks = list(hooks)

    def on_check_start(
        self,
        engine_name: str,
        data_size: int | None,
        context: dict[str, Any],
    ) -> None:
        for hook in self._hooks:
            with contextlib.suppress(Exception):
                hook.on_check_start(engine_name, data_size, context)

    # 다른 메서드들도 동일한 패턴
```

### 팩토리 함수

```python
def create_instrumented_engine(
    engine: DataQualityEngine,
    hooks: Sequence[EngineMetricsHook] | None = None,
    config: EngineMetricsConfig | None = None,
    enable_logging: bool = False,
    enable_metrics: bool = False,
    enable_tracing: bool = False,
) -> InstrumentedEngine:
    """계측된 엔진 생성"""
    all_hooks = list(hooks) if hooks else []

    if enable_logging:
        all_hooks.append(LoggingEngineHook())
    if enable_metrics:
        all_hooks.append(MetricsEngineHook())
    if enable_tracing:
        all_hooks.append(TracingEngineHook())

    return InstrumentedEngine(engine, hooks=all_hooks, config=config)


async def create_async_instrumented_engine(
    engine: AsyncDataQualityEngine,
    hooks: Sequence[AsyncEngineMetricsHook] | None = None,
    config: EngineMetricsConfig | None = None,
    enable_logging: bool = False,
    enable_metrics: bool = False,
    enable_tracing: bool = False,
) -> AsyncInstrumentedEngine:
    """비동기 계측 엔진 생성"""
    # 동일한 패턴
    ...
```

### 사용 예시

```python
from common.engines import (
    TruthoundEngine,
    InstrumentedEngine,
    StatsCollectorHook,
    MetricsEngineHook,
    LoggingEngineHook,
    CompositeEngineHook,
    create_instrumented_engine,
)

# 1. 기본 사용법
engine = TruthoundEngine()
hook = StatsCollectorHook()
instrumented = InstrumentedEngine(engine, hooks=[hook])

result = instrumented.check(data, auto_schema=True)

# 통계 조회
stats = hook.get_stats()
print(f"Total: {stats.total_operations}")
print(f"Success rate: {stats.check_success_rate:.2%}")
print(f"Avg duration: {stats.average_duration_ms:.2f}ms")

# 2. 팩토리 함수 사용
instrumented = create_instrumented_engine(
    TruthoundEngine(),
    enable_logging=True,
    enable_metrics=True,
    enable_tracing=True,
)

# 3. 복합 훅 사용
composite = CompositeEngineHook([
    MetricsEngineHook(prefix="truthound"),
    LoggingEngineHook(log_level="DEBUG"),
    StatsCollectorHook(),
])
instrumented = InstrumentedEngine(engine, hooks=[composite])

# 4. 비동기 사용
from common.engines import (
    SyncEngineAsyncAdapter,
    AsyncInstrumentedEngine,
    SyncToAsyncEngineHookAdapter,
)

async_engine = SyncEngineAsyncAdapter(TruthoundEngine())
sync_hook = StatsCollectorHook()
async_hook = SyncToAsyncEngineHookAdapter(sync_hook)

async_instrumented = AsyncInstrumentedEngine(async_engine, hooks=[async_hook])
result = await async_instrumented.check(data, auto_schema=True)
```

### 훅 격리

훅 내부 오류는 엔진 작업에 영향을 주지 않습니다:

```python
class BuggyHook(BaseEngineMetricsHook):
    def on_check_start(self, *args, **kwargs):
        raise RuntimeError("Hook error!")

# 훅 오류가 발생해도 엔진 작업은 정상 진행
instrumented = InstrumentedEngine(engine, hooks=[BuggyHook()])
result = instrumented.check(data)  # 정상 동작
```

---

## Result Aggregation

`engines/aggregation.py` 모듈은 다중 엔진의 데이터 품질 검증 결과를 통합, 집계, 비교하기 위한 시스템을 제공합니다. 엔터프라이즈 환경에서 여러 엔진의 결과를 종합적으로 분석할 때 사용합니다.

### 핵심 개념

| 컴포넌트 | 설명 |
|----------|------|
| `ResultAggregationStrategy` | 집계 전략 열거형 (MERGE, WORST, BEST, MAJORITY 등) |
| `AggregationConfig` | 집계 설정 (불변 dataclass, 빌더 패턴 지원) |
| `MultiEngineAggregator` | 다중 엔진 결과 집계 메인 클래스 |
| `AggregatedResult` | 집계된 결과 컨테이너 |
| `ComparisonResult` | 엔진 간 비교 결과 (불일치 감지) |
| `AggregatorRegistry` | 커스텀 집계기 등록 레지스트리 |
| `AggregationHook` | 집계 이벤트 관찰 프로토콜 |

### ResultAggregationStrategy

```python
from enum import Enum, auto

class ResultAggregationStrategy(Enum):
    """결과 집계 전략"""
    MERGE = auto()        # 모든 결과 병합 (실패 통합)
    WORST = auto()        # 가장 나쁜 상태 반환
    BEST = auto()         # 가장 좋은 상태 반환
    MAJORITY = auto()     # 다수결 투표
    WEIGHTED = auto()     # 가중치 기반 집계
    CONSENSUS = auto()    # 임계값 이상 합의
    STRICT_ALL = auto()   # 모두 PASSED여야 PASSED
    LENIENT_ANY = auto()  # 하나라도 PASSED면 PASSED
```

### AggregationConfig

```python
from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True, slots=True)
class AggregationConfig:
    """집계 설정 (불변)"""
    strategy: ResultAggregationStrategy = ResultAggregationStrategy.MERGE
    weights: Mapping[str, float] = field(default_factory=dict)
    consensus_threshold: float = 0.5
    conflict_resolution: ConflictResolution = ConflictResolution.USE_WORST
    include_metadata: bool = True
    normalize_weights: bool = True
    strict_mode: bool = False

    # 빌더 메서드
    def with_strategy(self, strategy: ResultAggregationStrategy) -> "AggregationConfig":
        return AggregationConfig(strategy=strategy, **self._other_fields())

    def with_weights(self, weights: Mapping[str, float]) -> "AggregationConfig":
        return AggregationConfig(weights=weights, **self._other_fields())

    def with_consensus_threshold(self, threshold: float) -> "AggregationConfig":
        return AggregationConfig(consensus_threshold=threshold, **self._other_fields())
```

### MultiEngineAggregator

```python
from common.engines import MultiEngineAggregator, AggregationConfig

# 집계기 생성
aggregator = MultiEngineAggregator()

# 여러 엔진의 결과
engine_results = {
    "truthound": truthound_result,
    "great_expectations": ge_result,
    "pandera": pandera_result,
}

# 결과 집계
aggregated = aggregator.aggregate_check_results(
    engine_results,
    config=AggregationConfig(strategy=ResultAggregationStrategy.MERGE),
)

print(f"Status: {aggregated.result.status.name}")
print(f"Engines: {aggregated.contributing_engines}")
print(f"Duration: {aggregated.aggregation_time_ms:.2f}ms")
```

### AggregatedResult

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class AggregatedResult(Generic[T]):
    """집계된 결과 컨테이너"""
    result: T                                        # 집계된 결과 객체
    contributing_engines: tuple[str, ...]            # 기여 엔진 목록
    strategy_used: ResultAggregationStrategy         # 사용된 전략
    engine_results: Mapping[str, EngineResultEntry[T]]  # 개별 엔진 결과
    aggregation_time_ms: float                       # 집계 소요 시간
    metadata: Mapping[str, Any]                      # 메타데이터

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AggregatedResult[T]":
        """딕셔너리에서 복원"""
        ...
```

### ComparisonResult (불일치 감지)

```python
from common.engines import MultiEngineAggregator

aggregator = MultiEngineAggregator()

# 엔진 간 결과 비교
comparison = aggregator.compare_check_results(engine_results)

print(f"All agree: {comparison.all_agree}")
print(f"Agreement rate: {comparison.agreement_rate:.1%}")

if comparison.discrepancies:
    for disc in comparison.discrepancies:
        print(f"Engines {disc.engines} disagree on: {disc.description}")
```

### 집계 전략별 동작

| 전략 | 동작 | 사용 사례 |
|------|------|----------|
| `MERGE` | 모든 실패를 하나의 결과로 병합 | 종합적인 품질 리포트 |
| `WORST` | 가장 나쁜 상태 반환 (FAILED > WARNING > PASSED) | 보수적인 품질 관리 |
| `BEST` | 가장 좋은 상태 반환 | 관대한 품질 관리 |
| `MAJORITY` | 다수결로 상태 결정 | 민주적 의사결정 |
| `WEIGHTED` | 엔진별 가중치 적용 후 점수 계산 | 신뢰도 기반 집계 |
| `CONSENSUS` | 임계값 이상 동의 시 PASSED | 합의 기반 품질 관리 |
| `STRICT_ALL` | 모두 PASSED여야 PASSED | 엄격한 품질 관리 |
| `LENIENT_ANY` | 하나라도 PASSED면 PASSED | 유연한 품질 관리 |

### 가중치 기반 집계

```python
from common.engines import (
    MultiEngineAggregator,
    AggregationConfig,
    ResultAggregationStrategy,
)

# 가중치 설정 (신뢰도 반영)
config = AggregationConfig(
    strategy=ResultAggregationStrategy.WEIGHTED,
    weights={
        "truthound": 0.5,         # 50% 가중치
        "great_expectations": 0.3,  # 30% 가중치
        "pandera": 0.2,           # 20% 가중치
    },
    normalize_weights=True,  # 합계 1.0으로 정규화
)

aggregator = MultiEngineAggregator()
aggregated = aggregator.aggregate_check_results(engine_results, config=config)

# 가중 점수 확인
weighted_score = aggregated.metadata.get("weighted_score")
print(f"Weighted score: {weighted_score:.2f}")
```

### 프리셋 설정

```python
from common.engines import (
    DEFAULT_AGGREGATION_CONFIG,     # 기본: MERGE 전략
    STRICT_AGGREGATION_CONFIG,      # 엄격: STRICT_ALL, strict_mode=True
    LENIENT_AGGREGATION_CONFIG,     # 관대: LENIENT_ANY
    CONSENSUS_AGGREGATION_CONFIG,   # 합의: CONSENSUS, 0.5 임계값
    WEIGHTED_AGGREGATION_CONFIG,    # 가중치: WEIGHTED
)

aggregator = MultiEngineAggregator(config=STRICT_AGGREGATION_CONFIG)
```

### AggregationHook (관찰 훅)

```python
from typing import Protocol, Any
from common.engines import AggregatedResult

@runtime_checkable
class AggregationHook(Protocol):
    """집계 이벤트 관찰 프로토콜"""

    def on_aggregation_start(
        self,
        engine_names: tuple[str, ...],
        strategy: ResultAggregationStrategy,
        context: dict[str, Any],
    ) -> None: ...

    def on_aggregation_end(
        self,
        result: AggregatedResult[Any],
        duration_ms: float,
        context: dict[str, Any],
    ) -> None: ...

    def on_comparison_complete(
        self,
        comparison: ComparisonResult,
        context: dict[str, Any],
    ) -> None: ...

    def on_error(
        self,
        exception: Exception,
        context: dict[str, Any],
    ) -> None: ...
```

### 내장 훅

```python
from common.engines import (
    LoggingAggregationHook,       # 집계 이벤트 로깅
    MetricsAggregationHook,       # 메트릭 수집 (집계 수, 소요 시간)
    CompositeAggregationHook,     # 여러 훅 조합
)

# 로깅 + 메트릭 훅 조합
logging_hook = LoggingAggregationHook()
metrics_hook = MetricsAggregationHook()
composite = CompositeAggregationHook([logging_hook, metrics_hook])

aggregator = MultiEngineAggregator(hooks=[composite])
aggregated = aggregator.aggregate_check_results(engine_results)

# 메트릭 조회
print(f"Total aggregations: {metrics_hook.total_aggregations}")
print(f"Average duration: {metrics_hook.average_duration_ms:.2f}ms")
print(f"Comparison count: {metrics_hook.comparison_count}")
```

### AggregatorRegistry

```python
from common.engines import (
    get_aggregator_registry,
    register_aggregator,
    get_aggregator,
)

# 커스텀 집계기 등록
class MyCustomAggregator(BaseResultAggregator[CheckResult]):
    def aggregate(self, results: Mapping[str, CheckResult]) -> CheckResult:
        # 커스텀 집계 로직
        ...

register_aggregator("custom", MyCustomAggregator())

# 등록된 집계기 사용
aggregator = get_aggregator("custom")
result = aggregator.aggregate(engine_results)

# 레지스트리 관리
registry = get_aggregator_registry()
print(f"Registered: {registry.list_names()}")
```

### 편의 함수

```python
from common.engines import (
    aggregate_check_results,
    compare_check_results,
    aggregate_profile_results,
    aggregate_learn_results,
)

# 간편 집계
aggregated = aggregate_check_results(
    engine_results,
    strategy=ResultAggregationStrategy.MAJORITY,
)

# 간편 비교
comparison = compare_check_results(engine_results)

# 프로파일 결과 집계
profile_aggregated = aggregate_profile_results(profile_results)

# 학습 결과 집계
learn_aggregated = aggregate_learn_results(learn_results)
```

### 타입별 집계기

```python
from common.engines import (
    CheckResultMergeAggregator,    # CheckResult 병합 집계
    CheckResultWeightedAggregator, # CheckResult 가중치 집계
    ProfileResultAggregator,       # ProfileResult 집계
    LearnResultAggregator,         # LearnResult 집계
)

# CheckResult 병합 집계
merge_aggregator = CheckResultMergeAggregator()
merged = merge_aggregator.aggregate(check_results)

# CheckResult 가중치 집계
weighted_aggregator = CheckResultWeightedAggregator()
weighted = weighted_aggregator.aggregate_with_weights(
    check_results,
    weights={"truthound": 0.6, "ge": 0.4},
)
```

### 예외 타입

```python
from common.engines import (
    AggregationError,           # 기본 집계 예외
    EmptyResultsError,          # 빈 결과 집계 시도
    IncompatibleResultsError,   # 호환 불가능한 결과 타입
    WeightConfigurationError,   # 가중치 설정 오류
    StrategyNotSupportedError,  # 지원하지 않는 전략
)

try:
    aggregated = aggregator.aggregate_check_results({})
except EmptyResultsError:
    print("No results to aggregate")

try:
    aggregated = aggregator.aggregate_check_results(
        results,
        config=AggregationConfig(
            strategy=ResultAggregationStrategy.WEIGHTED,
            weights={},  # 빈 가중치
        ),
    )
except WeightConfigurationError as e:
    print(f"Weight error: {e}")
```

### 실제 사용 예시

```python
from common.engines import (
    TruthoundEngine,
    GreatExpectationsAdapter,
    PanderaAdapter,
    MultiEngineAggregator,
    AggregationConfig,
    ResultAggregationStrategy,
)
import polars as pl

# 여러 엔진 준비
engines = {
    "truthound": TruthoundEngine(),
    "great_expectations": GreatExpectationsAdapter(),
    "pandera": PanderaAdapter(),
}

# 데이터 로드
df = pl.read_parquet("data.parquet")

# 규칙 정의
rules = [
    {"type": "not_null", "column": "id"},
    {"type": "unique", "column": "email"},
]

# 각 엔진으로 검증
results = {}
for name, engine in engines.items():
    with engine:
        if name == "truthound":
            results[name] = engine.check(df, auto_schema=True)
        else:
            results[name] = engine.check(df, rules=rules)

# 결과 집계 (가중치 기반)
aggregator = MultiEngineAggregator()
config = AggregationConfig(
    strategy=ResultAggregationStrategy.WEIGHTED,
    weights={"truthound": 0.5, "great_expectations": 0.3, "pandera": 0.2},
)

aggregated = aggregator.aggregate_check_results(results, config=config)

# 결과 출력
print(f"Final Status: {aggregated.result.status.name}")
print(f"Passed: {aggregated.result.passed_count}")
print(f"Failed: {aggregated.result.failed_count}")
print(f"Engines: {', '.join(aggregated.contributing_engines)}")

# 불일치 검사
comparison = aggregator.compare_check_results(results)
if not comparison.all_agree:
    print(f"Warning: Engines disagree (agreement: {comparison.agreement_rate:.0%})")
    for disc in comparison.discrepancies:
        print(f"  - {disc.description}")
```

---

## Engine Versioning

`common/engines/version.py`는 엔진 버전 관리를 위한 시맨틱 버저닝 시스템을 제공합니다. SemVer 2.0.0 명세를 완벽히 지원합니다.

### 핵심 타입

| 타입 | 설명 |
|------|------|
| `SemanticVersion` | SemVer 2.0.0 호환 버전 (불변 dataclass, total ordering 지원) |
| `VersionConstraint` | 단일 버전 제약조건 (>=1.0.0, ^2.0, ~1.2.3) |
| `VersionRange` | 복수 제약조건 조합 (>=1.0.0,<2.0.0) |
| `VersionCompatibilityResult` | 호환성 검사 결과 |
| `VersionCompatibilityConfig` | 호환성 검사 설정 (불변 dataclass, builder 패턴) |
| `EngineVersionRequirement` | 엔진 버전 요구사항 정의 |
| `VersionRegistry` | 엔진 버전 등록 및 요구사항 검증 레지스트리 |

### SemanticVersion

```python
from common.engines import SemanticVersion, parse_version

# 생성자로 직접 생성
version = SemanticVersion(major=1, minor=2, patch=3)
version = SemanticVersion(1, 2, 3, prerelease="alpha.1", build="build.123")

# 문자열에서 파싱
version = parse_version("1.2.3")
version = parse_version("2.0.0-beta.1+build.456")

# 비교 (total ordering 지원)
v1 = parse_version("1.0.0-alpha")
v2 = parse_version("1.0.0-beta")
v3 = parse_version("1.0.0")
print(v1 < v2 < v3)  # True (alpha < beta < release)

# 문자열 변환
print(str(version))          # "2.0.0-beta.1+build.456"
print(version.base_version)  # "2.0.0" (prerelease/build 제외)
```

### VersionConstraint

npm/Cargo 스타일의 버전 제약조건을 지원합니다.

```python
from common.engines import parse_constraint, VersionOperator

# 단일 제약조건
constraint = parse_constraint(">=1.0.0")
constraint = parse_constraint("^2.0.0")  # Caret: >=2.0.0,<3.0.0
constraint = parse_constraint("~1.2.3")  # Tilde: >=1.2.3,<1.3.0
constraint = parse_constraint("*")       # Wildcard: 모든 버전

# 제약조건 검사
print(constraint.satisfies(parse_version("2.5.0")))  # True

# 지원 연산자
# VersionOperator.EQ (==), NE (!=), GT (>), GE (>=), LT (<), LE (<=)
# VersionOperator.CARET (^): 메이저 버전 내 호환
# VersionOperator.TILDE (~): 마이너 버전 내 호환
# VersionOperator.WILDCARD (*): 모든 버전
```

### VersionRange

```python
from common.engines import parse_range

# 범위 생성 (AND 조합)
range_ = parse_range(">=1.0.0,<2.0.0")
range_ = parse_range(">=1.0.0 <2.0.0")  # 공백도 지원
range_ = parse_range("^1.2")            # >=1.2.0,<2.0.0

# 범위 검사
print(range_.satisfies(parse_version("1.5.0")))  # True
print(range_.satisfies(parse_version("2.0.0")))  # False
```

### 호환성 전략

| 전략 | 설명 |
|------|------|
| `STRICT` | 정확히 일치해야 함 |
| `SEMVER` | 동일 메이저, engine >= required (기본값) |
| `MAJOR` | 메이저 버전만 일치 |
| `MINOR` | 메이저 + 마이너 버전 일치 |
| `ANY` | 항상 호환 |

```python
from common.engines import (
    VersionCompatibilityChecker,
    VersionCompatibilityConfig,
    CompatibilityStrategy,
    CompatibilityLevel,
)

# 검사기 생성
config = VersionCompatibilityConfig(strategy=CompatibilityStrategy.SEMVER)
checker = VersionCompatibilityChecker(config)

# 호환성 검사
result = checker.check("1.5.0", ">=1.0.0")
print(f"Compatible: {result.compatible}")
print(f"Level: {result.level.name}")  # COMPATIBLE, INCOMPATIBLE, UNKNOWN, UNTESTED
print(f"Message: {result.message}")

# 문자열 또는 SemanticVersion 모두 지원
result = checker.check(
    engine_version=parse_version("1.5.0"),
    required=">=1.0.0,<2.0.0",
    engine_name="truthound",
)
```

### 프리셋 설정

```python
from common.engines import (
    DEFAULT_VERSION_COMPATIBILITY_CONFIG,   # SEMVER 전략 (기본)
    STRICT_VERSION_COMPATIBILITY_CONFIG,    # STRICT 전략
    LENIENT_VERSION_COMPATIBILITY_CONFIG,   # ANY 전략
)

checker = VersionCompatibilityChecker(STRICT_VERSION_COMPATIBILITY_CONFIG)
```

### 설정 빌더 패턴

```python
from common.engines import VersionCompatibilityConfig, CompatibilityStrategy, CompatibilityLevel

config = VersionCompatibilityConfig()
config = config.with_strategy(CompatibilityStrategy.MAJOR)
config = config.with_allow_prerelease(True)
config = config.with_default_level(CompatibilityLevel.UNTESTED)
```

### VersionRegistry

엔진 버전 등록 및 요구사항 관리를 위한 레지스트리입니다.

```python
from common.engines import (
    VersionRegistry,
    EngineVersionRequirement,
    get_version_registry,
    reset_version_registry,
)

# 전역 레지스트리 사용
registry = get_version_registry()

# 엔진 버전 등록
registry.register_engine("truthound", "1.5.0")
registry.register_engine("great_expectations", "0.18.0")

# 엔진 정보와 함께 등록
from common.engines import EngineInfo
registry.register_engine(
    "pandera",
    "0.17.0",
    info=EngineInfo(name="pandera", version="0.17.0", description="Pandera adapter"),
)

# 버전 요구사항 추가
registry.add_requirement(EngineVersionRequirement(
    engine_name="truthound",
    version_range=">=1.0.0,<2.0.0",
    description="Truthound 1.x 버전 필요",
    is_required=True,
))

# 개별 엔진 검증
result = registry.validate_engine("truthound")
print(f"Truthound compatible: {result.compatible}")

# 전체 검증
results = registry.validate_all()
for engine, result in results.items():
    print(f"{engine}: {result.compatible}")

# 등록된 엔진 조회
engines = registry.list_engines()
version = registry.get_engine_version("truthound")

# 레지스트리 초기화
reset_version_registry()
```

### 데코레이터

버전 요구사항을 데코레이터로 선언적으로 정의합니다.

```python
from common.engines import require_version, version_required

# 클래스 데코레이터 - 전역 레지스트리에 요구사항 등록
@require_version("truthound", ">=1.0.0")
@require_version("great_expectations", ">=0.15.0")
class MyWorkflow:
    pass

# 함수 래퍼 데코레이터 - 호출 시 버전 검증
@version_required({
    "truthound": ">=1.0.0",
    "great_expectations": ">=0.15.0,<1.0.0",
})
def process_data(data):
    return validate(data)

# 런타임에 버전 불일치 시 VersionIncompatibleError 발생
try:
    result = process_data(data)
except VersionIncompatibleError as e:
    print(f"Incompatible: {e.engine_name} {e.actual_version} vs {e.required_version}")
```

### 편의 함수

```python
from common.engines import (
    parse_version,
    parse_constraint,
    parse_range,
    check_version_compatibility,
    is_version_compatible,
    compare_versions,
    versions_compatible,
)

# 버전 파싱
version = parse_version("1.2.3")
constraint = parse_constraint(">=1.0.0")
range_ = parse_range(">=1.0.0,<2.0.0")

# 호환성 검사 (결과 객체 반환)
result = check_version_compatibility("1.5.0", ">=1.0.0,<2.0.0")
print(f"Compatible: {result.compatible}")

# 호환성 검사 (불리언 반환)
if is_version_compatible("1.5.0", ">=1.0.0"):
    print("Compatible!")

# 버전 비교 (-1, 0, 1 반환)
result = compare_versions("1.2.0", "1.3.0")
print(result)  # -1 (v1 < v2)

# 두 버전 호환성 검사 (SemVer 전략)
if versions_compatible("1.5.0", "1.2.0"):  # 1.5.0 >= 1.2.0, 같은 major
    print("Compatible per SemVer!")
```

### Prerelease 및 Build 메타데이터

SemVer 2.0.0 명세에 따른 완전한 지원:

```python
from common.engines import parse_version

# Prerelease 버전 순서
alpha = parse_version("1.0.0-alpha")
alpha_1 = parse_version("1.0.0-alpha.1")
beta = parse_version("1.0.0-beta")
rc_1 = parse_version("1.0.0-rc.1")
release = parse_version("1.0.0")

# 순서: alpha < alpha.1 < beta < rc.1 < release
print(alpha < alpha_1 < beta < rc_1 < release)  # True

# Prerelease는 release보다 항상 낮음
print(parse_version("1.0.0-alpha") < parse_version("1.0.0"))  # True

# Build 메타데이터 (비교에서 무시됨)
v1 = parse_version("1.0.0+build.1")
v2 = parse_version("1.0.0+build.2")
print(v1 == v2)  # True (build 무시)

# Prerelease 식별자 비교 규칙
# 1. 숫자 식별자: 숫자로 비교
# 2. 영숫자 식별자: ASCII 사전순 비교
# 3. 숫자 < 영숫자
```

### 예외 타입

```python
from common.engines import (
    VersionError,              # 기본 버전 예외
    VersionParseError,         # 버전 파싱 실패
    VersionConstraintError,    # 제약조건 파싱/평가 실패
    VersionIncompatibleError,  # 버전 호환 불가
)

try:
    version = parse_version("invalid")
except VersionParseError as e:
    print(f"Parse error: {e}")

try:
    constraint = parse_constraint(">>>1.0.0")  # 잘못된 연산자
except VersionConstraintError as e:
    print(f"Constraint error: {e}")

try:
    @require_version("truthound", ">=999.0.0")
    class Workflow:
        pass
except VersionIncompatibleError as e:
    print(f"Incompatible: {e.engine_name}")
    print(f"  Actual: {e.actual_version}")
    print(f"  Required: {e.required_version}")
```

### 엔진 통합 예시

```python
from common.engines import (
    TruthoundEngine,
    VersionRegistry,
    EngineVersionRequirement,
    get_version_registry,
)

# 엔진 생성 및 버전 등록
engine = TruthoundEngine()
registry = get_version_registry()
registry.register_engine(engine.engine_name, engine.engine_version)

# 버전 요구사항 정의
registry.add_requirement(EngineVersionRequirement(
    engine_name="truthound",
    version_range=">=1.0.0",
    description="Truthound 1.x or higher required",
))

# 전체 요구사항 검증
results = registry.validate_all()
all_compatible = all(r.compatible for r in results.values())

if not all_compatible:
    for engine, result in results.items():
        if not result.compatible:
            print(f"Engine '{engine}' version mismatch: {result.message}")
    raise RuntimeError("Engine version requirements not met")

# 정상 진행
result = engine.check(data, auto_schema=True)
```

### 워크플로우 통합 예시

```python
from common.engines import require_version, version_required, get_version_registry

# 워크플로우 클래스에 버전 요구사항 선언
@require_version("truthound", ">=1.0.0,<2.0.0")
@require_version("great_expectations", ">=0.15.0")
class DataQualityWorkflow:
    def __init__(self, engine):
        self.engine = engine

    @version_required({"truthound": ">=1.0.0"})
    def validate(self, data):
        return self.engine.check(data, auto_schema=True)

# 시작 시 요구사항 검증
def initialize_workflow():
    registry = get_version_registry()

    # 실제 엔진 버전 등록
    registry.register_engine("truthound", "1.5.0")
    registry.register_engine("great_expectations", "0.18.0")

    # 요구사항 검증
    results = registry.validate_all()
    for engine, result in results.items():
        if not result.compatible:
            raise RuntimeError(
                f"Engine '{engine}' incompatible: "
                f"required {result.required}, "
                f"got {result.engine_version}"
            )

    return DataQualityWorkflow(TruthoundEngine())
```

---

## Engine Chain

`common/engines/chain.py`는 여러 엔진을 조합하여 폴백, 로드밸런싱, 조건부 라우팅 등의 고급 패턴을 구현합니다.

### 핵심 타입

| 타입 | 설명 |
|------|------|
| `FallbackStrategy` | SEQUENTIAL, FIRST_AVAILABLE, ROUND_ROBIN, RANDOM, PRIORITY, WEIGHTED |
| `ChainExecutionMode` | FAIL_FAST, FALLBACK, ALL, FIRST_SUCCESS, QUORUM |
| `FailureReason` | EXCEPTION, TIMEOUT, UNHEALTHY, RESULT_CHECK, SKIPPED |
| `FallbackConfig` | 폴백 설정 (불변 dataclass, builder 패턴) |
| `ChainExecutionAttempt` | 단일 엔진 실행 시도 결과 |
| `ChainExecutionResult` | 전체 체인 실행 결과 |
| `EngineChain` | 메인 체인 클래스 (폴백, 재시도, 헬스체크 통합) |
| `ConditionalEngineChain` | 조건 기반 라우팅 체인 |
| `SelectorEngineChain` | 커스텀 선택기 기반 체인 |
| `AsyncEngineChain` | 비동기 엔진 체인 |

### 기본 사용법

```python
from common.engines import EngineChain, TruthoundEngine, GreatExpectationsAdapter

# 간단한 폴백 체인 (primary 실패 시 backup 사용)
primary = TruthoundEngine()
backup = GreatExpectationsAdapter()
chain = EngineChain([primary, backup])

# 일반 엔진처럼 사용
result = chain.check(data, rules)  # primary 실패 시 자동으로 backup 시도

# 체인은 DataQualityEngine 프로토콜 구현
# -> 어디서든 엔진 대신 사용 가능
```

### 팩토리 함수

```python
from common.engines import (
    create_fallback_chain,
    create_load_balanced_chain,
    create_async_fallback_chain,
)

# 간단한 폴백 체인
chain = create_fallback_chain(
    primary,
    backup1,
    backup2,
    retry_count=2,
    check_health=True,
    name="production_chain",
)

# 로드밸런싱 체인
chain = create_load_balanced_chain(
    engine1,
    engine2,
    engine3,
    strategy=FallbackStrategy.ROUND_ROBIN,
)

# 비동기 폴백 체인
async_chain = create_async_fallback_chain(
    async_primary,
    async_backup,
    retry_count=3,
)
```

### 폴백 전략

| 전략 | 설명 |
|------|------|
| `SEQUENTIAL` | 순차적으로 시도 (기본값) |
| `FIRST_AVAILABLE` | 첫 번째 건강한 엔진 사용 |
| `ROUND_ROBIN` | 라운드 로빈으로 분산 |
| `RANDOM` | 무작위 선택 |
| `PRIORITY` | 우선순위 기반 선택 |
| `WEIGHTED` | 가중치 기반 랜덤 선택 |

### 실행 모드

| 모드 | 설명 |
|------|------|
| `FAIL_FAST` | 첫 실패 시 즉시 중단 |
| `FALLBACK` | 실패 시 다음 엔진 시도 (기본값) |
| `ALL` | 모든 엔진 실행, 결과 집계 |
| `FIRST_SUCCESS` | 첫 성공까지 실행 |
| `QUORUM` | 쿼럼 성공까지 실행 |

### 프리셋 설정

```python
from common.engines import (
    DEFAULT_FALLBACK_CONFIG,       # 기본: 순차 폴백, 1회 시도
    RETRY_FALLBACK_CONFIG,         # 재시도: 3회 시도, 1초 딜레이
    HEALTH_AWARE_FALLBACK_CONFIG,  # 헬스체크: 건강한 엔진만 사용
    LOAD_BALANCED_CONFIG,          # 로드밸런싱: 라운드 로빈
    WEIGHTED_CONFIG,               # 가중치: 가중치 기반 선택
)
```

### 설정 빌더 패턴

```python
from common.engines import FallbackConfig, FallbackStrategy, ChainExecutionMode

config = FallbackConfig()
config = config.with_strategy(FallbackStrategy.ROUND_ROBIN)
config = config.with_retry(count=3, delay_seconds=1.0)
config = config.with_health_check(enabled=True, skip_unhealthy=True)
config = config.with_timeout(30.0)
config = config.with_weights(truthound=2.0, ge=1.0)  # 가중치 설정

chain = EngineChain([engine1, engine2], config=config)
```

### 조건부 라우팅

데이터/규칙 특성에 따라 엔진을 선택합니다.

```python
from common.engines import ConditionalEngineChain

chain = ConditionalEngineChain(name="smart_router")

# 조건 추가 (높은 priority가 먼저 평가됨)
chain.add_route(
    lambda data, rules: len(data) > 1_000_000,  # 대용량 데이터
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

# 기본 엔진 (조건이 없을 때)
chain.set_default_engine(general_engine)

result = chain.check(data, rules)  # 적절한 엔진으로 라우팅
```

### 커스텀 선택기

더 복잡한 선택 로직이 필요할 때 EngineSelector 구현:

```python
from common.engines import SelectorEngineChain, EngineSelector

class DataSizeSelector:
    """데이터 크기 기반 선택기"""

    def select_engine(self, data, rules, available_engines, context):
        data_size = len(data) if hasattr(data, "__len__") else 0

        for engine in available_engines:
            if data_size > 1_000_000 and "heavy" in engine.engine_name:
                return engine
            if data_size <= 1_000_000 and "light" in engine.engine_name:
                return engine

        return available_engines[0] if available_engines else None

chain = SelectorEngineChain(
    engines=[light_engine, heavy_engine],
    selector=DataSizeSelector(),
)
```

### 훅 (Hooks)

체인 실행 이벤트를 모니터링합니다.

```python
from common.engines import (
    EngineChain,
    LoggingChainHook,
    MetricsChainHook,
    CompositeChainHook,
)

# 로깅 훅: 체인 이벤트 자동 로깅
logging_hook = LoggingChainHook()

# 메트릭 훅: 통계 수집
metrics_hook = MetricsChainHook()

# 복합 훅
composite = CompositeChainHook([logging_hook, metrics_hook])

chain = EngineChain(
    [primary, backup],
    hooks=[composite],
    name="monitored_chain",
)

# 작업 수행
result = chain.check(data, rules)

# 메트릭 조회
print(metrics_hook.get_chain_success_rate("monitored_chain"))  # 1.0
print(metrics_hook.get_fallback_rate("monitored_chain"))       # 0.0 또는 1.0
print(metrics_hook.get_average_duration_ms("monitored_chain")) # 평균 소요시간

# 엔진별 통계
stats = metrics_hook.get_engine_stats("monitored_chain")
# {"attempts": {"primary": 1}, "successes": {"primary": 1}, "failures": {}}
```

### 실행 결과 조회

```python
from common.engines import EngineChain

chain = EngineChain([primary, backup])
result = chain.check(data, rules)

# 마지막 실행 결과 조회
exec_result = chain.last_execution_result

print(f"성공: {exec_result.success}")
print(f"최종 엔진: {exec_result.final_engine}")
print(f"시도 횟수: {exec_result.attempt_count}")
print(f"총 소요시간: {exec_result.total_duration_ms}ms")
print(f"실패한 엔진들: {exec_result.failed_engines}")

# 개별 시도 정보
for attempt in exec_result.attempts:
    print(f"  - {attempt.engine_name}: success={attempt.success}")
    if attempt.failure_reason:
        print(f"    reason: {attempt.failure_reason.name}")
```

### 비동기 체인

```python
from common.engines import AsyncEngineChain, create_async_fallback_chain

chain = AsyncEngineChain([async_engine1, async_engine2])

# 비동기 사용
result = await chain.check(data, rules)

# 컨텍스트 매니저
async with chain:
    result = await chain.check(data, rules)
```

### 체인 조합 (Composability)

체인은 DataQualityEngine을 구현하므로 중첩 가능합니다.

```python
# 내부 체인
inner_chain = EngineChain([engine1, engine2], name="inner")

# 외부 체인 (내부 체인을 엔진으로 사용)
outer_chain = EngineChain([inner_chain, backup_engine], name="outer")

# 조건부 체인 내에서 폴백 체인 사용
conditional = ConditionalEngineChain()
conditional.add_route(
    lambda data, rules: len(rules) > 10,
    EngineChain([heavy_primary, heavy_backup]),  # 폴백 체인
)
conditional.set_default_engine(light_engine)
```

### 예외 처리

```python
from common.engines import (
    EngineChain,
    AllEnginesFailedError,
    NoEngineSelectedError,
    EngineChainConfigError,
)

try:
    result = chain.check(data, rules)
except AllEnginesFailedError as e:
    print(f"모든 엔진 실패: {e.attempted_engines}")
    for engine_name, exception in e.exceptions.items():
        print(f"  {engine_name}: {exception}")
except NoEngineSelectedError as e:
    print(f"선택된 엔진 없음: {e.chain_name}")

# 설정 오류 (엔진 목록이 비어있을 때)
try:
    chain = EngineChain([])  # 빈 목록
except EngineChainConfigError as e:
    print(f"설정 오류: {e}")
```

### 엔진 반복자/선택기

```python
from common.engines import (
    SequentialEngineIterator,
    RoundRobinEngineIterator,
    WeightedRandomSelector,
    PrioritySelector,
)

engines = [engine1, engine2, engine3]

# 순차 반복자
iterator = SequentialEngineIterator(engines)
while (engine := iterator.next()) is not None:
    print(engine.engine_name)
iterator.reset()

# 라운드 로빈 반복자 (무한 순환)
robin = RoundRobinEngineIterator(engines)
for _ in range(10):
    engine = robin.next()  # 순환

# 가중치 기반 랜덤 선택
weights = {"engine1": 3.0, "engine2": 1.0, "engine3": 1.0}
selector = WeightedRandomSelector(engines, weights)
engine = selector.select()  # 가중치에 따라 선택

# 우선순위 기반 선택
priorities = {"engine1": 10, "engine2": 5, "engine3": 1}
priority_iter = PrioritySelector(engines, priorities)
# 우선순위 높은 순서대로 반환
```

### 컨텍스트 매니저

```python
from common.engines import EngineChain

# 컨텍스트 매니저로 사용 시 ManagedEngine 자동 시작/종료
with EngineChain([managed_engine1, managed_engine2]) as chain:
    result = chain.check(data, rules)
# 자동으로 모든 ManagedEngine stop() 호출
```

### 워크플로우 통합 예시

```python
from common.engines import (
    EngineChain,
    ConditionalEngineChain,
    TruthoundEngine,
    GreatExpectationsAdapter,
    PanderaAdapter,
    FallbackConfig,
    FallbackStrategy,
    MetricsChainHook,
)

# 프로덕션 엔진 체인 구성
def create_production_chain():
    # 1. 기본 폴백 체인: Truthound -> GE -> Pandera
    primary = TruthoundEngine()
    backup1 = GreatExpectationsAdapter()
    backup2 = PanderaAdapter()

    fallback_chain = EngineChain(
        [primary, backup1, backup2],
        config=FallbackConfig(
            retry_count=2,
            retry_delay_seconds=1.0,
            check_health=True,
            skip_unhealthy=True,
        ),
        name="fallback",
    )

    # 2. 조건부 라우팅: 데이터 크기에 따라 분기
    conditional = ConditionalEngineChain(name="production_router")

    # 대용량 데이터는 Truthound (스트리밍 지원)
    conditional.add_route(
        lambda data, rules: len(data) > 1_000_000,
        TruthoundEngine(),
        priority=10,
        name="large_data",
    )

    # 기본은 폴백 체인 사용
    conditional.set_default_engine(fallback_chain)

    # 3. 메트릭 수집
    metrics_hook = MetricsChainHook()

    return conditional, metrics_hook


# 사용
chain, metrics = create_production_chain()

with chain:
    result = chain.check(data, rules)

print(f"Success rate: {metrics.get_chain_success_rate('production_router'):.0%}")
```

---

## Plugin Discovery

`common/engines/plugin.py`는 Python 엔트리 포인트를 통한 자동 엔진 탐지 시스템을 제공합니다. 서드파티 패키지가 코어 프레임워크 수정 없이 커스텀 데이터 품질 엔진을 등록할 수 있습니다.

### 설계 원칙

플러그인 발견 시스템은 다음 원칙을 따릅니다:

| 원칙 | 설명 |
|------|------|
| **Open-Closed Principle** | 기존 코드 수정 없이 새 엔진 추가 가능 |
| **Lazy Loading** | 필요 시점까지 엔진 인스턴스화 지연 |
| **Priority Resolution** | 동일 이름 충돌 시 우선순위로 해결 |
| **Protocol Validation** | 로딩 시 `DataQualityEngine` 준수 검증 |
| **Multi-Source Discovery** | 엔트리 포인트, 내장, 프로그래밍 방식 등록 지원 |

### 핵심 타입

| 타입 | 설명 |
|------|------|
| `PluginSpec` | 플러그인 사양 (불변 dataclass, 이름, 경로, 우선순위) |
| `PluginMetadata` | 플러그인 메타데이터 (버전, 저작자, 라이선스) |
| `PluginInstance` | 플러그인 인스턴스 컨테이너 (사양, 인스턴스, 상태, 오류) |
| `ValidationResult` | 검증 결과 (유효 여부, 오류 목록, 경고 목록) |
| `DiscoveryConfig` | 발견 설정 (엔트리 포인트 그룹, 로드 전략) |
| `PluginRegistry` | 중앙 플러그인 레지스트리 |

### 아키텍처

```
                    ┌─────────────────────────────────┐
                    │        PluginRegistry           │
                    │     (중앙 발견 관리자)           │
                    └─────────────┬───────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ EntryPointSource│    │  BuiltinSource  │    │   DictSource    │
│   (서드파티)     │    │   (코어 엔진)    │    │  (프로그래밍)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PluginLoader                             │
│                  (클래스 로딩 및 인스턴스화)                       │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PluginValidator                             │
│                 (프로토콜 준수 검증)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 기본 사용법

```python
from common.engines import (
    PluginRegistry,
    discover_plugins,
    get_plugin_engine,
)

# 사용 가능한 모든 플러그인 발견
specs = discover_plugins(include_builtins=True, include_entry_points=True)
for spec in specs:
    print(f"발견된 플러그인: {spec.name} ({spec.full_path})")

# 엔진 인스턴스 획득 및 사용
engine = get_plugin_engine("truthound")
result = engine.check(data, auto_schema=True)
```

### 서드파티 엔진 등록

외부 패키지는 `pyproject.toml`의 Python 엔트리 포인트를 통해 엔진을 등록합니다:

```toml
[project.entry-points."truthound.engines"]
my_engine = "my_package.engines:MyCustomEngine"
```

설치 후 엔진은 자동으로 발견 가능합니다:

```python
from common.engines import get_plugin_engine

# 서드파티 엔진 사용 가능
engine = get_plugin_engine("my_engine")
result = engine.check(data, rules)
```

### 플러그인 사양

```python
from common.engines import PluginSpec, PluginType

spec = PluginSpec(
    name="custom_engine",
    module_path="my_package.engines",
    class_name="CustomEngine",
    plugin_type=PluginType.ENGINE,
    priority=100,  # 높은 우선순위가 충돌 시 승리
    enabled=True,
    aliases=("custom", "ce"),  # 대체 이름
    config={"timeout": 30.0},  # 엔진 설정
)

# 불변 dataclass이므로 with_* 메서드로 수정
new_spec = spec.with_priority(200)
new_spec = spec.with_enabled(False)
new_spec = spec.with_aliases(("alias1", "alias2"))
```

### 발견 설정

```python
from common.engines import DiscoveryConfig, LoadStrategy

config = DiscoveryConfig(
    entry_point_group="truthound.engines",
    load_strategy=LoadStrategy.LAZY,  # LAZY 또는 EAGER
    disabled_plugins=frozenset(["deprecated_engine"]),
    priority_overrides={"critical_engine": 1000},
    validate_on_load=True,
)

registry = PluginRegistry(config=config)
specs = registry.discover()
```

### 로드 전략

| 전략 | 동작 | 사용 사례 |
|------|------|----------|
| `LAZY` | 최초 접근 시 엔진 로딩 | 일반적인 사용 (기본값) |
| `EAGER` | 발견 시점에 모든 엔진 로딩 | 시작 시 검증 필요 시 |

### 플러그인 소스

시스템은 `PluginSource` 프로토콜을 통해 다양한 플러그인 소스를 지원합니다:

```python
from common.engines import (
    EntryPointPluginSource,
    BuiltinPluginSource,
    DictPluginSource,
)

# 엔트리 포인트 소스 (서드파티 패키지용)
ep_source = EntryPointPluginSource(group="truthound.engines")

# 내장 소스 (코어 엔진용)
builtin_source = BuiltinPluginSource()  # truthound, great_expectations, pandera

# 딕셔너리 소스 (프로그래밍 방식 등록용)
dict_source = DictPluginSource({
    "custom": "my_package.engines:CustomEngine",
})

# 레지스트리에 커스텀 소스 추가
registry = PluginRegistry()
registry.add_source(dict_source)
```

### 검증 시스템

플러그인은 `DataQualityEngine` 프로토콜 준수 여부를 검증합니다:

```python
from common.engines import (
    DataQualityEngineValidator,
    CompositeValidator,
    validate_plugin,
)

# 플러그인 사양 검증
result = validate_plugin(spec)
if result.is_valid:
    print("플러그인이 유효합니다")
else:
    print(f"검증 오류: {result.errors}")
    print(f"경고: {result.warnings}")

# 커스텀 검증기 조합
validator = CompositeValidator([
    DataQualityEngineValidator(),
    CustomValidator(),
])
result = validator.validate(engine_class)
```

### 훅 시스템

훅을 통해 플러그인 라이프사이클 이벤트를 모니터링합니다:

```python
from common.engines import (
    PluginRegistry,
    LoggingPluginHook,
    MetricsPluginHook,
    CompositePluginHook,
)

# 디버깅용 로깅 훅
logging_hook = LoggingPluginHook()

# 통계용 메트릭 훅
metrics_hook = MetricsPluginHook()

# 다중 훅 조합
composite = CompositePluginHook([logging_hook, metrics_hook])

registry = PluginRegistry(hooks=[composite])
registry.discover()

# 메트릭 조회
print(f"발견: {metrics_hook.discovery_count}")
print(f"로딩: {metrics_hook.loaded_count}")
print(f"오류: {metrics_hook.error_count}")
print(f"성공률: {metrics_hook.success_rate:.2%}")

stats = metrics_hook.get_stats()
# {"discovery_count": 3, "loaded_count": 1, "error_count": 0, ...}
```

### EngineRegistry 통합

플러그인 시스템은 기존 `EngineRegistry`와 원활하게 통합됩니다:

```python
from common.engines import EngineRegistry

registry = EngineRegistry()

# 자동 플러그인 발견 활성화
registry.enable_plugin_discovery()

# 플러그인 발견 및 등록
plugins = registry.discover_plugins(include_builtins=True)

# 레지스트리를 통해 발견된 엔진 접근
engine = registry.get("truthound")

# 비활성화
registry.disable_plugin_discovery()
```

### 우선순위 해결

동일 이름의 플러그인을 여러 소스가 제공할 경우, 최고 우선순위 사양이 유지됩니다:

```python
from common.engines import DictPluginSource, PluginRegistry

# 낮은 우선순위 소스
source1 = DictPluginSource(
    {"engine": "package1:Engine"},
    default_priority=50,
)

# 높은 우선순위 소스
source2 = DictPluginSource(
    {"engine": "package2:Engine"},
    default_priority=100,
)

registry = PluginRegistry()
registry.add_source(source1)
registry.add_source(source2)
registry.discover()

# source2의 엔진 사용 (높은 우선순위)
spec = registry.get_spec("engine")
assert spec.module_path == "package2"
assert spec.priority == 100
```

### 전역 함수

```python
from common.engines import (
    # 레지스트리 관리
    get_plugin_registry,
    reset_plugin_registry,

    # 발견
    discover_plugins,
    load_plugins,

    # 접근
    get_plugin_engine,

    # 등록
    register_plugin,

    # 검증
    validate_plugin,
)

# 전역 싱글톤 레지스트리 획득
registry = get_plugin_registry()

# 레지스트리 초기화 (테스트에 유용)
reset_plugin_registry()

# 단일 단계 발견 및 로딩
engines = load_plugins(include_builtins=True)
```

### 예외 처리

```python
from common.engines import (
    PluginError,
    PluginNotFoundError,
    PluginLoadError,
    PluginValidationError,
    PluginConflictError,
    PluginDiscoveryError,
)

try:
    engine = get_plugin_engine("nonexistent")
except PluginNotFoundError as e:
    print(f"플러그인 미발견: {e.plugin_name}")

try:
    registry.get_engine("broken_plugin")
except PluginLoadError as e:
    print(f"로딩 실패: {e.spec.name}")
    print(f"원인: {e.original_error}")

try:
    validate_plugin(invalid_spec, strict=True)
except PluginValidationError as e:
    print(f"검증 실패: {e.errors}")
```

### 커스텀 플러그인 소스 구현

`PluginSource` 프로토콜을 구현하여 커스텀 소스를 생성할 수 있습니다:

```python
from typing import Sequence
from common.engines import PluginSource, PluginSpec

class DatabasePluginSource:
    """데이터베이스에서 플러그인 정보를 로드하는 소스"""

    def __init__(self, connection_string: str):
        self._conn = connection_string

    @property
    def source_name(self) -> str:
        return "database"

    def discover(self) -> Sequence[PluginSpec]:
        # 데이터베이스에서 플러그인 정보 조회
        plugins = fetch_plugins_from_db(self._conn)
        return [
            PluginSpec(
                name=p.name,
                module_path=p.module,
                class_name=p.class_name,
            )
            for p in plugins
        ]

# 레지스트리에 추가
registry = PluginRegistry()
registry.add_source(DatabasePluginSource("postgresql://..."))
registry.discover()
```

### 워크플로우 통합 예시

```python
from common.engines import (
    PluginRegistry,
    DiscoveryConfig,
    LoadStrategy,
    LoggingPluginHook,
    MetricsPluginHook,
    CompositePluginHook,
)

def setup_plugin_system():
    """프로덕션 환경을 위한 플러그인 시스템 설정"""

    # 발견 설정
    config = DiscoveryConfig(
        entry_point_group="truthound.engines",
        load_strategy=LoadStrategy.LAZY,
        disabled_plugins=frozenset(["deprecated_v1_engine"]),
        priority_overrides={
            "production_engine": 1000,  # 프로덕션 엔진 최우선
        },
        validate_on_load=True,
    )

    # 훅 설정
    hooks = CompositePluginHook([
        LoggingPluginHook(),
        MetricsPluginHook(),
    ])

    # 레지스트리 생성
    registry = PluginRegistry(config=config, hooks=[hooks])

    # 발견 실행
    specs = registry.discover(
        include_builtins=True,
        include_entry_points=True,
    )

    print(f"총 {len(specs)}개 플러그인 발견")
    for spec in specs:
        print(f"  - {spec.name} (priority={spec.priority})")

    return registry

# 사용
registry = setup_plugin_system()
engine = registry.get_engine("truthound")
result = engine.check(data, auto_schema=True)
```

---

## base.py

### Protocols

```python
# common/base.py
"""
핵심 Protocol 및 데이터 타입 정의.

이 모듈은 모든 플랫폼 어댑터가 구현해야 하는 Protocol과
공통 데이터 구조를 정의합니다.
"""

from __future__ import annotations

from typing import (
    Protocol,
    TypeVar,
    Any,
    Iterator,
    runtime_checkable,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json

# Polars import (optional)
try:
    import polars as pl
    DataFrame = pl.DataFrame | pl.LazyFrame
except ImportError:
    DataFrame = Any  # type: ignore


# =============================================================================
# Enums
# =============================================================================

class CheckStatus(Enum):
    """검증 결과 상태"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class Severity(Enum):
    """검증 실패 심각도"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def __lt__(self, other: "Severity") -> bool:
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
        return order.index(self) < order.index(other)


class FailureAction(Enum):
    """검증 실패 시 동작"""
    RAISE = "raise"
    WARN = "warn"
    LOG = "log"
    CONTINUE = "continue"


class SerializationFormat(Enum):
    """직렬화 형식"""
    JSON = "json"
    DICT = "dict"
    PYDANTIC = "pydantic"
    AIRFLOW_XCOM = "airflow_xcom"
    DAGSTER_OUTPUT = "dagster_output"
    PREFECT_ARTIFACT = "prefect_artifact"


# =============================================================================
# Protocols
# =============================================================================

@runtime_checkable
class WorkflowIntegration(Protocol):
    """
    워크플로우 통합을 위한 핵심 Protocol.

    모든 플랫폼 어댑터는 이 Protocol을 구현해야 합니다.
    Protocol은 덕 타이핑을 지원하므로 명시적 상속 없이도
    호환 가능합니다.

    Examples
    --------
    Protocol 구현:

    >>> class MyAdapter:
    ...     @property
    ...     def platform_name(self) -> str:
    ...         return "my_platform"
    ...
    ...     def check(self, data, config):
    ...         # 구현
    ...         pass

    타입 체크:

    >>> def run_check(adapter: WorkflowIntegration) -> CheckResult:
    ...     return adapter.check(data, config)
    """

    @property
    def platform_name(self) -> str:
        """플랫폼 식별자 (예: 'airflow', 'dagster', 'prefect')"""
        ...

    @property
    def platform_version(self) -> str:
        """지원하는 플랫폼 버전 범위"""
        ...

    def check(
        self,
        data: DataFrame,
        config: "CheckConfig",
    ) -> "CheckResult":
        """
        데이터 품질 검증 실행.

        Parameters
        ----------
        data : DataFrame
            검증할 데이터 (Polars DataFrame 또는 LazyFrame)

        config : CheckConfig
            검증 설정

        Returns
        -------
        CheckResult
            검증 결과

        Raises
        ------
        ValidationExecutionError
            검증 실행 중 오류 발생
        TimeoutError
            검증 타임아웃
        """
        ...

    def profile(
        self,
        data: DataFrame,
        config: "ProfileConfig",
    ) -> "ProfileResult":
        """
        데이터 프로파일링 실행.

        Parameters
        ----------
        data : DataFrame
            프로파일링할 데이터

        config : ProfileConfig
            프로파일링 설정

        Returns
        -------
        ProfileResult
            프로파일링 결과
        """
        ...

    def learn(
        self,
        data: DataFrame,
        config: "LearnConfig | None" = None,
    ) -> "LearnResult":
        """
        데이터에서 스키마 자동 학습.

        Parameters
        ----------
        data : DataFrame
            학습할 데이터

        config : LearnConfig | None
            학습 설정 (선택적)

        Returns
        -------
        LearnResult
            학습된 스키마 및 규칙
        """
        ...


@runtime_checkable
class AsyncWorkflowIntegration(Protocol):
    """비동기 워크플로우 통합 Protocol"""

    @property
    def platform_name(self) -> str:
        ...

    async def check_async(
        self,
        data: DataFrame,
        config: "CheckConfig",
    ) -> "CheckResult":
        """비동기 데이터 품질 검증"""
        ...

    async def profile_async(
        self,
        data: DataFrame,
        config: "ProfileConfig",
    ) -> "ProfileResult":
        """비동기 데이터 프로파일링"""
        ...

    async def learn_async(
        self,
        data: DataFrame,
        config: "LearnConfig | None" = None,
    ) -> "LearnResult":
        """비동기 스키마 학습"""
        ...


# =============================================================================
# Configuration Types
# =============================================================================

@dataclass(frozen=True)
class CheckConfig:
    """
    데이터 품질 검증 설정.

    불변 데이터클래스로 스레드 안전합니다.

    Parameters
    ----------
    rules : tuple[dict[str, Any], ...]
        적용할 검증 규칙 목록

    fail_on_error : bool
        검증 실패 시 예외 발생 여부. 기본값: True

    failure_action : FailureAction
        실패 시 동작. 기본값: RAISE

    sample_size : int | None
        샘플링할 행 수. None=전체

    parallel : bool
        병렬 실행 여부. 기본값: True

    timeout_seconds : int
        타임아웃 (초). 기본값: 300

    tags : frozenset[str]
        메타데이터 태그

    extra : dict[str, Any]
        플랫폼별 추가 설정

    Examples
    --------
    >>> config = CheckConfig(
    ...     rules=(
    ...         {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
    ...         {"column": "age", "type": "in_range", "min": 0, "max": 150},
    ...     ),
    ...     fail_on_error=True,
    ...     timeout_seconds=600,
    ... )
    """
    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    fail_on_error: bool = True
    failure_action: FailureAction = FailureAction.RAISE
    sample_size: int | None = None
    parallel: bool = True
    timeout_seconds: int = 300
    tags: frozenset[str] = field(default_factory=frozenset)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """유효성 검증"""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be positive or None")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckConfig":
        """딕셔너리에서 생성"""
        return cls(
            rules=tuple(data.get("rules", [])),
            fail_on_error=data.get("fail_on_error", True),
            failure_action=FailureAction(data.get("failure_action", "raise")),
            sample_size=data.get("sample_size"),
            parallel=data.get("parallel", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
        )

    def with_rules(self, rules: list[dict[str, Any]]) -> "CheckConfig":
        """규칙을 변경한 새 설정 반환"""
        return CheckConfig(
            rules=tuple(rules),
            fail_on_error=self.fail_on_error,
            failure_action=self.failure_action,
            sample_size=self.sample_size,
            parallel=self.parallel,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            extra=self.extra,
        )

    def with_timeout(self, seconds: int) -> "CheckConfig":
        """타임아웃을 변경한 새 설정 반환"""
        return CheckConfig(
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            failure_action=self.failure_action,
            sample_size=self.sample_size,
            parallel=self.parallel,
            timeout_seconds=seconds,
            tags=self.tags,
            extra=self.extra,
        )

    def to_truthound_kwargs(self) -> dict[str, Any]:
        """Truthound API 호출용 kwargs로 변환"""
        return {
            "rules": list(self.rules),
            "fail_on_error": self.fail_on_error,
            "sample_size": self.sample_size,
            "parallel": self.parallel,
            "timeout": self.timeout_seconds,
            **self.extra,
        }


@dataclass(frozen=True)
class ProfileConfig:
    """데이터 프로파일링 설정"""
    columns: frozenset[str] | None = None
    include_statistics: bool = True
    include_patterns: bool = True
    include_distributions: bool = True
    sample_size: int | None = None
    timeout_seconds: int = 300
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileConfig":
        """딕셔너리에서 생성"""
        columns = data.get("columns")
        return cls(
            columns=frozenset(columns) if columns else None,
            include_statistics=data.get("include_statistics", True),
            include_patterns=data.get("include_patterns", True),
            include_distributions=data.get("include_distributions", True),
            sample_size=data.get("sample_size"),
            timeout_seconds=data.get("timeout_seconds", 300),
            extra=data.get("extra", {}),
        )

    def to_truthound_kwargs(self) -> dict[str, Any]:
        """Truthound API 호출용 kwargs로 변환"""
        kwargs: dict[str, Any] = {
            "include_statistics": self.include_statistics,
            "include_patterns": self.include_patterns,
            "include_distributions": self.include_distributions,
            "timeout": self.timeout_seconds,
        }
        if self.columns:
            kwargs["columns"] = list(self.columns)
        if self.sample_size:
            kwargs["sample_size"] = self.sample_size
        return {**kwargs, **self.extra}


@dataclass(frozen=True)
class LearnConfig:
    """스키마 학습 설정"""
    strictness: str = "moderate"  # strict, moderate, lenient
    include_constraints: bool = True
    include_patterns: bool = True
    sample_size: int | None = None
    timeout_seconds: int = 300
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strictness not in ("strict", "moderate", "lenient"):
            raise ValueError(
                f"strictness must be 'strict', 'moderate', or 'lenient', "
                f"got '{self.strictness}'"
            )


# =============================================================================
# Result Types
# =============================================================================

@dataclass(frozen=True)
class ValidationFailure:
    """
    개별 검증 실패 정보.

    Parameters
    ----------
    rule_name : str
        실패한 규칙 이름

    column : str | None
        관련 컬럼 (모델 레벨 규칙은 None)

    message : str
        실패 메시지

    severity : Severity
        심각도

    failed_count : int
        실패한 행 수

    total_count : int
        검사한 총 행 수

    sample_values : tuple[Any, ...]
        실패한 값 샘플
    """
    rule_name: str
    column: str | None
    message: str
    severity: Severity
    failed_count: int
    total_count: int
    sample_values: tuple[Any, ...] = field(default_factory=tuple)

    @property
    def failure_rate(self) -> float:
        """실패율 계산"""
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "rule_name": self.rule_name,
            "column": self.column,
            "message": self.message,
            "severity": self.severity.value,
            "failed_count": self.failed_count,
            "total_count": self.total_count,
            "failure_rate": self.failure_rate,
            "sample_values": list(self.sample_values[:5]),
        }


@dataclass(frozen=True)
class CheckResult:
    """
    데이터 품질 검증 결과.

    Parameters
    ----------
    status : CheckStatus
        전체 검증 상태

    passed_count : int
        통과한 규칙 수

    failed_count : int
        실패한 규칙 수

    warning_count : int
        경고 규칙 수

    skipped_count : int
        스킵된 규칙 수

    failures : tuple[ValidationFailure, ...]
        실패 상세 정보

    execution_time_ms : float
        실행 시간 (밀리초)

    timestamp : datetime
        실행 시각

    metadata : dict[str, Any]
        추가 메타데이터
    """
    status: CheckStatus
    passed_count: int
    failed_count: int
    warning_count: int = 0
    skipped_count: int = 0
    failures: tuple[ValidationFailure, ...] = field(default_factory=tuple)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """검증 성공 여부"""
        return self.status == CheckStatus.PASSED

    @property
    def total_count(self) -> int:
        """전체 규칙 수"""
        return (
            self.passed_count
            + self.failed_count
            + self.warning_count
            + self.skipped_count
        )

    @property
    def pass_rate(self) -> float:
        """통과율"""
        if self.total_count == 0:
            return 1.0
        return self.passed_count / self.total_count

    @property
    def failure_rate(self) -> float:
        """실패율"""
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count

    def iter_failures(
        self,
        min_severity: Severity = Severity.INFO,
    ) -> Iterator[ValidationFailure]:
        """
        심각도 필터링된 실패 순회.

        Parameters
        ----------
        min_severity : Severity
            최소 심각도 (이상만 반환)

        Yields
        ------
        ValidationFailure
            필터링된 실패
        """
        for failure in self.failures:
            if failure.severity <= min_severity:
                yield failure

    def get_critical_failures(self) -> list[ValidationFailure]:
        """크리티컬 실패만 반환"""
        return list(self.iter_failures(Severity.CRITICAL))

    @classmethod
    def from_truthound(cls, th_result: Any) -> "CheckResult":
        """Truthound 결과에서 변환"""
        failures = tuple(
            ValidationFailure(
                rule_name=f.rule_name,
                column=f.column,
                message=f.message,
                severity=Severity(f.severity) if isinstance(f.severity, str) else f.severity,
                failed_count=f.failed_count,
                total_count=f.total_count,
                sample_values=tuple(f.sample_values[:5]) if hasattr(f, "sample_values") else (),
            )
            for f in th_result.failures
        )

        return cls(
            status=CheckStatus(th_result.status) if isinstance(th_result.status, str) else th_result.status,
            passed_count=th_result.passed_count,
            failed_count=th_result.failed_count,
            warning_count=getattr(th_result, "warning_count", 0),
            skipped_count=getattr(th_result, "skipped_count", 0),
            failures=failures,
            execution_time_ms=th_result.execution_time_ms,
            metadata=getattr(th_result, "metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환 (XCom/직렬화 호환)"""
        return {
            "status": self.status.value,
            "is_success": self.is_success,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "skipped_count": self.skipped_count,
            "total_count": self.total_count,
            "pass_rate": self.pass_rate,
            "failure_rate": self.failure_rate,
            "failures": [f.to_dict() for f in self.failures],
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class ProfileResult:
    """프로파일링 결과"""
    columns: dict[str, dict[str, Any]]
    row_count: int
    execution_time_ms: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_truthound(cls, th_result: Any) -> "ProfileResult":
        """Truthound 결과에서 변환"""
        return cls(
            columns=th_result.columns,
            row_count=th_result.row_count,
            execution_time_ms=th_result.execution_time_ms,
            metadata=getattr(th_result, "metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "columns": self.columns,
            "row_count": self.row_count,
            "column_count": len(self.columns),
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LearnResult:
    """스키마 학습 결과"""
    rules: tuple[dict[str, Any], ...]
    columns: dict[str, dict[str, Any]]
    strictness: str
    execution_time_ms: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "rules": list(self.rules),
            "columns": self.columns,
            "strictness": self.strictness,
            "rules_count": len(self.rules),
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_check_config(self) -> CheckConfig:
        """CheckConfig로 변환"""
        return CheckConfig(rules=self.rules)
```

---

## config.py

```python
# common/config.py
"""
환경 설정 로딩 및 플랫폼별 설정 변환.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class TruthoundConfig:
    """
    Truthound 통합 설정.

    환경 변수 또는 설정 파일에서 로드할 수 있습니다.

    Parameters
    ----------
    connection_string : str | None
        데이터 소스 연결 문자열

    default_timeout : int
        기본 타임아웃 (초)

    fail_on_error : bool
        기본 실패 동작

    parallel : bool
        병렬 실행 기본값

    log_level : str
        로그 레벨

    platform_specific : dict[str, Any]
        플랫폼별 설정
    """
    connection_string: str | None = None
    default_timeout: int = 300
    fail_on_error: bool = True
    parallel: bool = True
    log_level: str = "INFO"
    platform_specific: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = "TRUTHOUND") -> "TruthoundConfig":
        """
        환경 변수에서 설정 로드.

        Parameters
        ----------
        prefix : str
            환경 변수 접두사

        Returns
        -------
        TruthoundConfig
            로드된 설정

        Examples
        --------
        >>> # TRUTHOUND_CONNECTION_STRING="postgresql://..."
        >>> # TRUTHOUND_DEFAULT_TIMEOUT="600"
        >>> config = TruthoundConfig.from_env()
        """
        def get_env(key: str, default: Any = None) -> Any:
            return os.environ.get(f"{prefix}_{key}", default)

        def get_bool(key: str, default: bool) -> bool:
            val = get_env(key)
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")

        def get_int(key: str, default: int) -> int:
            val = get_env(key)
            if val is None:
                return default
            return int(val)

        return cls(
            connection_string=get_env("CONNECTION_STRING"),
            default_timeout=get_int("DEFAULT_TIMEOUT", 300),
            fail_on_error=get_bool("FAIL_ON_ERROR", True),
            parallel=get_bool("PARALLEL", True),
            log_level=get_env("LOG_LEVEL", "INFO"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "TruthoundConfig":
        """
        설정 파일에서 로드.

        Parameters
        ----------
        path : str | Path
            설정 파일 경로 (JSON 또는 YAML)

        Returns
        -------
        TruthoundConfig
            로드된 설정
        """
        path = Path(path)
        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
        else:
            data = json.loads(content)

        return cls(
            connection_string=data.get("connection_string"),
            default_timeout=data.get("default_timeout", 300),
            fail_on_error=data.get("fail_on_error", True),
            parallel=data.get("parallel", True),
            log_level=data.get("log_level", "INFO"),
            platform_specific=data.get("platform_specific", {}),
        )

    def for_platform(self, platform: str) -> dict[str, Any]:
        """
        플랫폼별 설정 반환.

        Parameters
        ----------
        platform : str
            플랫폼 이름 (airflow, dagster, prefect)

        Returns
        -------
        dict[str, Any]
            플랫폼별 설정
        """
        base = {
            "connection_string": self.connection_string,
            "default_timeout": self.default_timeout,
            "fail_on_error": self.fail_on_error,
            "parallel": self.parallel,
        }

        platform_config = self.platform_specific.get(platform, {})
        return {**base, **platform_config}


def load_config(
    env_prefix: str = "TRUTHOUND",
    config_path: str | Path | None = None,
) -> TruthoundConfig:
    """
    설정 로드 (환경 변수 우선).

    Parameters
    ----------
    env_prefix : str
        환경 변수 접두사

    config_path : str | Path | None
        설정 파일 경로 (선택적)

    Returns
    -------
    TruthoundConfig
        로드된 설정
    """
    # 환경 변수에서 설정 파일 경로 확인
    if config_path is None:
        config_path = os.environ.get(f"{env_prefix}_CONFIG_PATH")

    # 설정 파일이 있으면 로드
    if config_path and Path(config_path).exists():
        config = TruthoundConfig.from_file(config_path)
    else:
        config = TruthoundConfig()

    # 환경 변수로 오버라이드
    env_config = TruthoundConfig.from_env(env_prefix)

    # 병합 (환경 변수 우선)
    if env_config.connection_string:
        config.connection_string = env_config.connection_string
    if os.environ.get(f"{env_prefix}_DEFAULT_TIMEOUT"):
        config.default_timeout = env_config.default_timeout

    return config


def get_platform_config(
    platform: str,
    config: TruthoundConfig | None = None,
) -> dict[str, Any]:
    """
    플랫폼별 설정 반환.

    Parameters
    ----------
    platform : str
        플랫폼 이름

    config : TruthoundConfig | None
        설정 (None이면 자동 로드)

    Returns
    -------
    dict[str, Any]
        플랫폼별 설정
    """
    if config is None:
        config = load_config()

    return config.for_platform(platform)
```

---

## logging.py

Enterprise-grade logging utilities with structured logging, context propagation, and sensitive data masking.

### Core Components

```python
# common/logging.py
"""
구조화된 로깅 유틸리티.

주요 기능:
- 구조화된 로깅 (Structured Logging)
- 컨텍스트 전파 (Context Propagation)
- 민감정보 자동 마스킹 (Sensitive Data Masking)
- 플랫폼별 어댑터 (Airflow, Dagster, Prefect)
- 성능 타이밍 (Performance Timing)
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from enum import Enum
from contextvars import ContextVar


# =============================================================================
# Log Levels
# =============================================================================

class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    def to_stdlib(self) -> int:
        """Convert to stdlib logging level."""
        return self.value


# =============================================================================
# Context Management
# =============================================================================

@dataclass(frozen=True, slots=True)
class LogContextData:
    """Immutable container for log context data."""
    operation: str | None = None
    platform: str | None = None
    correlation_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LogContext:
    """
    Context manager for log context propagation.

    Examples
    --------
    >>> with LogContext(operation="validate", platform="airflow"):
    ...     logger.info("Starting")  # Includes operation and platform
    ...     with LogContext(task_id="task_1"):
    ...         logger.info("Processing")  # Includes all context
    """

    def __init__(
        self,
        *,
        operation: str | None = None,
        platform: str | None = None,
        correlation_id: str | None = None,
        **extra: Any,
    ) -> None:
        self._new_context = LogContextData(
            operation=operation,
            platform=platform,
            correlation_id=correlation_id,
            extra=extra,
        )
        self._token = None

    def __enter__(self):
        # Merge with current context
        current = get_current_context()
        merged = current.merge(self._new_context)
        self._token = _log_context.set(merged)
        return self

    def __exit__(self, *args):
        if self._token is not None:
            _log_context.reset(self._token)


# =============================================================================
# Sensitive Data Masking
# =============================================================================

class SensitiveDataMasker:
    """
    Masks sensitive data in log messages and structured data.

    Auto-masks:
    - password, secret, token, api_key in strings and dicts
    - URL credentials (postgres://user:pass@host)
    - AWS credentials (AKIA...)

    Examples
    --------
    >>> masker = SensitiveDataMasker()
    >>> masker.mask_string("password=secret123")
    'password=***MASKED***'
    >>> masker.mask_dict({"password": "secret", "name": "test"})
    {'password': '***MASKED***', 'name': 'test'}
    """

    MASK_VALUE = "***MASKED***"

    DEFAULT_SENSITIVE_KEYS = frozenset({
        "password", "secret", "token", "api_key",
        "access_token", "refresh_token", "credentials",
        "connection_string", "private_key",
    })

    def mask_string(self, value: str) -> str:
        """Mask sensitive patterns in string."""
        ...

    def mask_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask sensitive keys in dictionary (recursive)."""
        ...


# =============================================================================
# Logger Implementation
# =============================================================================

class TruthoundLogger:
    """
    Main logger class for Truthound integrations.

    Features:
    - Structured logging with key-value pairs
    - Automatic context inclusion
    - Sensitive data masking
    - Multiple handlers support

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing data", rows=1000)

    >>> with LogContext(operation="validate"):
    ...     logger.warning("Validation issue", column="email")
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.DEBUG,
        handlers: list[LogHandler] | None = None,
    ) -> None:
        self.name = name
        self.level = level
        self._handlers = handlers or []
        self._masker = SensitiveDataMasker()

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: Exception | None = None, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: Exception | None = None, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with stack trace."""
        exc_info = sys.exc_info()[1]
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)


# =============================================================================
# Handlers
# =============================================================================

class StreamHandler:
    """Handler that writes to a stream (stdout/stderr)."""
    ...

class BufferingHandler:
    """Handler that buffers records before flushing."""
    ...

class NullHandler:
    """Handler that discards all records."""
    ...


# =============================================================================
# Formatters
# =============================================================================

class TextFormatter:
    """Plain text log formatter (human-readable)."""
    ...

class JSONFormatter:
    """JSON log formatter (structured logging systems)."""
    ...


# =============================================================================
# Filters
# =============================================================================

class LevelFilter:
    """Filter by log level."""
    ...

class ContextFilter:
    """Filter by context field values."""
    ...

class RegexFilter:
    """Filter by message pattern."""
    ...


# =============================================================================
# Platform Adapters
# =============================================================================

class AirflowLoggerAdapter:
    """Adapter for Apache Airflow logging."""
    ...

class DagsterLoggerAdapter:
    """Adapter for Dagster op context logging."""
    ...

class PrefectLoggerAdapter:
    """Adapter for Prefect flow/task logging."""
    ...

def create_platform_handler(platform: str, context: Any = None) -> LogHandler:
    """
    Create a platform-specific log handler.

    Parameters
    ----------
    platform : str
        Platform name ('airflow', 'dagster', 'prefect', 'stdlib')
    context : Any
        Platform-specific context object

    Returns
    -------
    LogHandler
        Configured handler for the platform

    Examples
    --------
    >>> handler = create_platform_handler("airflow", task_instance)
    >>> logger.add_handler(handler)
    """
    ...


# =============================================================================
# Performance Logging
# =============================================================================

class PerformanceLogger:
    """
    Logger for performance timing.

    Examples
    --------
    >>> perf = get_performance_logger(__name__)
    >>> with perf.timed("database_query"):
    ...     result = execute_query()
    # Logs: "database_query completed in 123.45ms"

    >>> @perf.timed_decorator()
    ... def process_data(data):
    ...     return transform(data)
    """

    def __init__(
        self,
        logger: TruthoundLogger,
        slow_threshold_ms: float = 1000.0,
    ) -> None:
        self._logger = logger
        self._slow_threshold_ms = slow_threshold_ms

    def timed(self, operation: str, **metadata: Any):
        """Context manager for timing operations."""
        ...

    def timed_decorator(self, operation: str | None = None):
        """Decorator for timing functions."""
        ...


# =============================================================================
# Convenience Functions
# =============================================================================

def get_logger(name: str) -> TruthoundLogger:
    """Get a logger by name."""
    ...

def get_performance_logger(name: str, slow_threshold_ms: float = 1000.0) -> PerformanceLogger:
    """Get a performance logger."""
    ...

def configure_logging(level: LogLevel = LogLevel.INFO, format: str = "text") -> None:
    """Configure global logging settings."""
    ...


# =============================================================================
# Decorators
# =============================================================================

def log_call(logger: TruthoundLogger | None = None, include_result: bool = False):
    """Decorator to log function calls."""
    ...

def log_errors(logger: TruthoundLogger | None = None, reraise: bool = True):
    """Decorator to log exceptions."""
    ...
```

### Usage Examples

```python
from common import get_logger, LogContext, get_performance_logger

# Basic structured logging
logger = get_logger(__name__)
logger.info("Processing data", rows=1000, platform="airflow")

# Context propagation
with LogContext(operation="validate", platform="airflow"):
    logger.info("Starting validation")  # Includes operation, platform
    with LogContext(task_id="task_1"):
        logger.warning("Issue found", column="email")  # All context included

# Performance timing
perf = get_performance_logger(__name__)

with perf.timed("database_query", table="users"):
    result = execute_query()
# Logs: "database_query completed in 123.45ms"

@perf.timed_decorator()
def process_batch(data):
    return transform(data)

# Sensitive data masking (automatic)
logger.info("Connecting", password="secret")  # password=***MASKED***
logger.info("URL: postgres://user:pass@host/db")  # pass masked

# Platform adapters
from common import create_platform_handler

handler = create_platform_handler("airflow", task_instance)
logger.add_handler(handler)
```

---

## retry.py

Enterprise-grade retry utilities with configurable backoff strategies, exception filtering, and observability hooks.

### Core Components

```python
# common/retry.py
"""
재시도 데코레이터 및 유틸리티.

주요 기능:
- 설정 가능한 재시도 전략 (Fixed, Exponential, Linear, Fibonacci)
- Jitter를 통한 Thundering Herd 방지
- 유연한 예외 필터링
- 관찰성 Hook (로깅, 메트릭)
- 동기/비동기 함수 지원
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Exceptions
# =============================================================================

class RetryError(TruthoundIntegrationError):
    """재시도 관련 기본 예외."""
    pass


class RetryExhaustedError(RetryError):
    """최대 재시도 횟수 초과 시 발생."""
    pass


class NonRetryableError(RetryError):
    """재시도 불가능한 오류. 즉시 실패 처리."""
    pass


# =============================================================================
# Retry Strategy
# =============================================================================

class RetryStrategy(Enum):
    """재시도 지연 전략."""
    FIXED = "fixed"           # 고정 지연
    EXPONENTIAL = "exponential"  # 지수 백오프
    LINEAR = "linear"         # 선형 증가
    FIBONACCI = "fibonacci"   # 피보나치 지연


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RetryConfig:
    """
    재시도 설정.

    불변 데이터클래스로 스레드 안전하며, 빌더 패턴을 지원합니다.

    Parameters
    ----------
    max_attempts : int
        최대 시도 횟수 (1 이상). 기본값: 3

    strategy : RetryStrategy
        지연 계산 전략. 기본값: EXPONENTIAL

    base_delay : float
        기본 지연 시간 (초). 기본값: 1.0

    max_delay : float
        최대 지연 시간 (초). 기본값: 60.0

    jitter_factor : float
        지터 비율 (0.0 ~ 1.0). 기본값: 0.1

    exceptions : tuple[type[Exception], ...]
        재시도할 예외 타입. 기본값: (Exception,)

    Examples
    --------
    >>> config = RetryConfig(
    ...     max_attempts=5,
    ...     strategy=RetryStrategy.EXPONENTIAL,
    ...     base_delay=2.0,
    ... )
    >>> # 빌더 패턴
    >>> new_config = config.with_max_attempts(10).with_jitter(0.2)
    """
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1
    exceptions: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        """유효성 검증."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if not (0.0 <= self.jitter_factor <= 1.0):
            raise ValueError("jitter_factor must be between 0.0 and 1.0")

    # Builder methods
    def with_max_attempts(self, value: int) -> "RetryConfig":
        """max_attempts를 변경한 새 설정 반환."""
        return RetryConfig(
            max_attempts=value,
            strategy=self.strategy,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            jitter_factor=self.jitter_factor,
            exceptions=self.exceptions,
        )

    def with_strategy(self, value: RetryStrategy) -> "RetryConfig":
        """strategy를 변경한 새 설정 반환."""
        ...

    def with_base_delay(self, value: float) -> "RetryConfig":
        """base_delay를 변경한 새 설정 반환."""
        ...

    def with_max_delay(self, value: float) -> "RetryConfig":
        """max_delay를 변경한 새 설정 반환."""
        ...

    def with_jitter(self, value: float) -> "RetryConfig":
        """jitter_factor를 변경한 새 설정 반환."""
        ...

    def with_exceptions(self, *exceptions: type[Exception]) -> "RetryConfig":
        """exceptions를 변경한 새 설정 반환."""
        ...


# Preset configurations
DEFAULT_RETRY_CONFIG = RetryConfig()

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=10,
    base_delay=0.5,
    max_delay=30.0,
    jitter_factor=0.2,
)

CONSERVATIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=5.0,
    max_delay=120.0,
    jitter_factor=0.1,
)

NO_DELAY_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.0,
    max_delay=0.0,
    jitter_factor=0.0,
)


# =============================================================================
# Delay Calculators
# =============================================================================

class DelayCalculator(Protocol):
    """지연 시간 계산 프로토콜."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        """attempt 번째 시도에 대한 지연 시간(초) 계산."""
        ...


class FixedDelayCalculator:
    """고정 지연 계산기."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        return config.base_delay


class ExponentialDelayCalculator:
    """지수 백오프 계산기. delay = base_delay * (2 ^ attempt)"""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        delay = config.base_delay * (2 ** attempt)
        return min(delay, config.max_delay)


class LinearDelayCalculator:
    """선형 증가 계산기. delay = base_delay * (attempt + 1)"""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        delay = config.base_delay * (attempt + 1)
        return min(delay, config.max_delay)


class FibonacciDelayCalculator:
    """피보나치 지연 계산기."""

    def calculate(self, attempt: int, config: RetryConfig) -> float:
        a, b = 0, 1
        for _ in range(attempt):
            a, b = b, a + b
        delay = config.base_delay * b
        return min(delay, config.max_delay)


# =============================================================================
# Exception Filters
# =============================================================================

class ExceptionFilter(Protocol):
    """예외 필터 프로토콜."""

    def should_retry(self, exception: Exception) -> bool:
        """이 예외에 대해 재시도해야 하는지 결정."""
        ...


class TypeBasedExceptionFilter:
    """타입 기반 예외 필터."""

    def __init__(self, exceptions: tuple[type[Exception], ...]) -> None:
        self._exceptions = exceptions

    def should_retry(self, exception: Exception) -> bool:
        return isinstance(exception, self._exceptions)


class CallableExceptionFilter:
    """콜러블 기반 예외 필터."""

    def __init__(self, predicate: Callable[[Exception], bool]) -> None:
        self._predicate = predicate

    def should_retry(self, exception: Exception) -> bool:
        return self._predicate(exception)


class CompositeExceptionFilter:
    """복합 예외 필터 (AND/OR 조합)."""

    def __init__(
        self,
        filters: list[ExceptionFilter],
        mode: str = "any",  # "any" or "all"
    ) -> None:
        self._filters = filters
        self._mode = mode

    def should_retry(self, exception: Exception) -> bool:
        if self._mode == "any":
            return any(f.should_retry(exception) for f in self._filters)
        return all(f.should_retry(exception) for f in self._filters)


# =============================================================================
# Hooks
# =============================================================================

class RetryHook(Protocol):
    """재시도 이벤트 Hook 프로토콜."""

    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
    ) -> None:
        """재시도 전 호출."""
        ...

    def on_success(self, attempt: int, result: Any) -> None:
        """성공 시 호출."""
        ...

    def on_failure(self, attempt: int, exception: Exception) -> None:
        """최종 실패 시 호출."""
        ...


class LoggingRetryHook:
    """로깅 Hook."""

    def __init__(self, logger: TruthoundLogger | None = None) -> None:
        self._logger = logger or get_logger(__name__)

    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
    ) -> None:
        self._logger.warning(
            "Retry attempt",
            attempt=attempt,
            exception=str(exception),
            delay_seconds=delay,
        )

    def on_success(self, attempt: int, result: Any) -> None:
        if attempt > 1:
            self._logger.info(
                "Operation succeeded after retry",
                attempts=attempt,
            )

    def on_failure(self, attempt: int, exception: Exception) -> None:
        self._logger.error(
            "Operation failed after all retries",
            attempts=attempt,
            exception=str(exception),
        )


class MetricsRetryHook:
    """메트릭 수집 Hook."""

    def __init__(self) -> None:
        self.retry_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_delay = 0.0

    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
    ) -> None:
        self.retry_count += 1
        self.total_delay += delay

    def on_success(self, attempt: int, result: Any) -> None:
        self.success_count += 1

    def on_failure(self, attempt: int, exception: Exception) -> None:
        self.failure_count += 1

    def reset(self) -> None:
        """메트릭 초기화."""
        self.retry_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_delay = 0.0


class CompositeRetryHook:
    """복합 Hook (여러 Hook 조합)."""

    def __init__(self, hooks: list[RetryHook]) -> None:
        self._hooks = hooks

    def on_retry(
        self,
        attempt: int,
        exception: Exception,
        delay: float,
    ) -> None:
        for hook in self._hooks:
            hook.on_retry(attempt, exception, delay)

    def on_success(self, attempt: int, result: Any) -> None:
        for hook in self._hooks:
            hook.on_success(attempt, result)

    def on_failure(self, attempt: int, exception: Exception) -> None:
        for hook in self._hooks:
            hook.on_failure(attempt, exception)


# =============================================================================
# Retry Executor
# =============================================================================

class RetryExecutor:
    """
    재시도 실행기.

    설정에 따라 함수를 재시도하며, 동기/비동기 모두 지원합니다.

    Examples
    --------
    >>> executor = RetryExecutor(config)
    >>> result = executor.execute(unstable_function, arg1, arg2)
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        hooks: list[RetryHook] | None = None,
    ) -> None:
        self._config = config or DEFAULT_RETRY_CONFIG
        self._hooks = CompositeRetryHook(hooks or [])
        self._delay_calculator = self._get_calculator()
        self._exception_filter = TypeBasedExceptionFilter(self._config.exceptions)

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """동기 함수 실행."""
        ...

    async def execute_async(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """비동기 함수 실행."""
        ...


# =============================================================================
# Decorator
# =============================================================================

def retry(
    *,
    max_attempts: int = 3,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.1,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    hooks: list[RetryHook] | None = None,
    config: RetryConfig | None = None,
) -> Callable:
    """
    재시도 데코레이터.

    동기 및 비동기 함수 모두 지원합니다.

    Parameters
    ----------
    max_attempts : int
        최대 시도 횟수

    strategy : RetryStrategy
        지연 계산 전략

    base_delay : float
        기본 지연 시간 (초)

    max_delay : float
        최대 지연 시간 (초)

    jitter_factor : float
        지터 비율 (0.0 ~ 1.0)

    exceptions : tuple[type[Exception], ...]
        재시도할 예외 타입

    hooks : list[RetryHook] | None
        관찰성 Hook 목록

    config : RetryConfig | None
        설정 객체 (지정 시 다른 파라미터 무시)

    Examples
    --------
    >>> @retry(max_attempts=3, exceptions=(ConnectionError,))
    ... def fetch_data():
    ...     return api.get("/data")

    >>> @retry(config=AGGRESSIVE_RETRY_CONFIG)
    ... async def fetch_data_async():
    ...     return await api.get_async("/data")
    """
    ...


# =============================================================================
# Helper Functions
# =============================================================================

def retry_call(
    func: Callable[..., T],
    *args: Any,
    config: RetryConfig | None = None,
    hooks: list[RetryHook] | None = None,
    **kwargs: Any,
) -> T:
    """
    함수를 재시도와 함께 호출 (데코레이터 없이).

    Examples
    --------
    >>> result = retry_call(
    ...     unstable_api.get,
    ...     "/users",
    ...     config=AGGRESSIVE_RETRY_CONFIG,
    ... )
    """
    executor = RetryExecutor(config=config, hooks=hooks)
    return executor.execute(func, *args, **kwargs)


async def retry_call_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    hooks: list[RetryHook] | None = None,
    **kwargs: Any,
) -> T:
    """비동기 함수를 재시도와 함께 호출."""
    executor = RetryExecutor(config=config, hooks=hooks)
    return await executor.execute_async(func, *args, **kwargs)
```

### Usage Examples

```python
from common import (
    retry, RetryConfig, RetryStrategy,
    AGGRESSIVE_RETRY_CONFIG, LoggingRetryHook,
    retry_call,
)

# 기본 데코레이터 사용
@retry(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
def fetch_data():
    return api.get("/data")

# 비동기 함수
@retry(max_attempts=5, strategy=RetryStrategy.EXPONENTIAL)
async def fetch_data_async():
    return await api.get_async("/data")

# Preset 설정 사용
@retry(config=AGGRESSIVE_RETRY_CONFIG)
def aggressive_fetch():
    return api.get("/critical")

# 빌더 패턴으로 설정 커스터마이징
custom_config = (
    RetryConfig()
    .with_max_attempts(5)
    .with_strategy(RetryStrategy.FIBONACCI)
    .with_base_delay(2.0)
    .with_jitter(0.2)
)

@retry(config=custom_config)
def custom_fetch():
    return api.get("/custom")

# Hook을 통한 관찰성
logging_hook = LoggingRetryHook()
metrics_hook = MetricsRetryHook()

@retry(
    max_attempts=3,
    hooks=[logging_hook, metrics_hook],
)
def monitored_fetch():
    return api.get("/monitored")

# 데코레이터 없이 사용
result = retry_call(
    api.get,
    "/data",
    config=CONSERVATIVE_RETRY_CONFIG,
)
```

### Strategy Comparison

| Strategy | Pattern | Use Case |
|----------|---------|----------|
| FIXED | 1s, 1s, 1s | 일정한 간격 필요 시 |
| EXPONENTIAL | 1s, 2s, 4s, 8s | 일반적인 백오프 (기본값) |
| LINEAR | 1s, 2s, 3s, 4s | 점진적 증가 |
| FIBONACCI | 1s, 1s, 2s, 3s, 5s | 부드러운 증가 |

---

## serializers.py

```python
# common/serializers.py
"""
결과 직렬화 유틸리티.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, Any, Generic
from abc import abstractmethod
import json
from datetime import datetime

from common.base import (
    CheckResult,
    ProfileResult,
    CheckStatus,
    Severity,
    ValidationFailure,
    SerializationFormat,
)


T = TypeVar("T")


class Serializer(Protocol[T]):
    """직렬화 프로토콜"""

    def serialize(self, obj: T) -> Any:
        """객체를 직렬화된 형태로 변환"""
        ...

    def deserialize(self, data: Any) -> T:
        """직렬화된 데이터를 객체로 복원"""
        ...


class JSONSerializer:
    """JSON 직렬화기"""

    def serialize(self, obj: CheckResult) -> str:
        """CheckResult를 JSON 문자열로 변환"""
        return obj.to_json()

    def deserialize(self, data: str) -> CheckResult:
        """JSON 문자열에서 CheckResult 복원"""
        parsed = json.loads(data)
        return self._dict_to_result(parsed)

    def _dict_to_result(self, data: dict[str, Any]) -> CheckResult:
        """딕셔너리를 CheckResult로 변환"""
        failures = tuple(
            ValidationFailure(
                rule_name=f["rule_name"],
                column=f.get("column"),
                message=f["message"],
                severity=Severity(f["severity"]),
                failed_count=f["failed_count"],
                total_count=f["total_count"],
            )
            for f in data.get("failures", [])
        )

        return CheckResult(
            status=CheckStatus(data["status"]),
            passed_count=data["passed_count"],
            failed_count=data["failed_count"],
            warning_count=data.get("warning_count", 0),
            skipped_count=data.get("skipped_count", 0),
            failures=failures,
            execution_time_ms=data.get("execution_time_ms", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


class DictSerializer:
    """딕셔너리 직렬화기 (XCom 호환)"""

    def serialize(self, obj: CheckResult) -> dict[str, Any]:
        """CheckResult를 딕셔너리로 변환"""
        return obj.to_dict()

    def deserialize(self, data: dict[str, Any]) -> CheckResult:
        """딕셔너리에서 CheckResult 복원"""
        return JSONSerializer()._dict_to_result(data)


class AirflowXComSerializer:
    """
    Airflow XCom 전용 직렬화기.

    XCom의 크기 제한을 고려하여 결과를 최적화합니다.
    """

    MAX_FAILURES = 50  # XCom에 저장할 최대 실패 수

    def serialize(self, obj: CheckResult) -> dict[str, Any]:
        """CheckResult를 XCom 호환 딕셔너리로 변환"""
        result = obj.to_dict()

        # 실패 수 제한
        if len(result["failures"]) > self.MAX_FAILURES:
            result["failures"] = result["failures"][:self.MAX_FAILURES]
            result["_truncated"] = True
            result["_original_failure_count"] = len(obj.failures)

        return result

    def deserialize(self, data: dict[str, Any]) -> CheckResult:
        """딕셔너리에서 CheckResult 복원"""
        return DictSerializer().deserialize(data)


class DagsterOutputSerializer:
    """
    Dagster Output 전용 직렬화기.

    Dagster 메타데이터 형식으로 변환합니다.
    """

    def serialize(self, obj: CheckResult) -> dict[str, Any]:
        """CheckResult를 Dagster 메타데이터로 변환"""
        from dagster import MetadataValue

        return {
            "quality_status": MetadataValue.text(obj.status.value),
            "passed_checks": MetadataValue.int(obj.passed_count),
            "failed_checks": MetadataValue.int(obj.failed_count),
            "pass_rate": MetadataValue.float(obj.pass_rate),
            "execution_time_ms": MetadataValue.float(obj.execution_time_ms),
            "is_success": MetadataValue.bool(obj.is_success),
            "failures_json": MetadataValue.json([
                f.to_dict() for f in obj.failures[:20]
            ]),
        }


class PrefectArtifactSerializer:
    """
    Prefect Artifact 전용 직렬화기.
    """

    def serialize(self, obj: CheckResult) -> dict[str, Any]:
        """CheckResult를 Prefect Artifact 형식으로 변환"""
        return {
            "key": "quality-check-result",
            "type": "table",
            "data": [
                {"Metric": "Status", "Value": obj.status.value},
                {"Metric": "Passed", "Value": str(obj.passed_count)},
                {"Metric": "Failed", "Value": str(obj.failed_count)},
                {"Metric": "Pass Rate", "Value": f"{obj.pass_rate:.2%}"},
            ],
            "description": f"Quality Check: {obj.status.value}",
        }


class SerializerFactory:
    """
    직렬화기 팩토리.

    지정된 형식에 맞는 직렬화기를 생성합니다.
    """

    _serializers: dict[SerializationFormat, type] = {
        SerializationFormat.JSON: JSONSerializer,
        SerializationFormat.DICT: DictSerializer,
        SerializationFormat.AIRFLOW_XCOM: AirflowXComSerializer,
        SerializationFormat.DAGSTER_OUTPUT: DagsterOutputSerializer,
        SerializationFormat.PREFECT_ARTIFACT: PrefectArtifactSerializer,
    }

    @classmethod
    def get_serializer(cls, format: SerializationFormat) -> Any:
        """
        형식에 맞는 직렬화기 반환.

        Parameters
        ----------
        format : SerializationFormat
            직렬화 형식

        Returns
        -------
        Serializer
            해당 직렬화기

        Raises
        ------
        ValueError
            지원하지 않는 형식
        """
        serializer_class = cls._serializers.get(format)
        if not serializer_class:
            raise ValueError(f"Unsupported serialization format: {format}")
        return serializer_class()

    @classmethod
    def register(cls, format: SerializationFormat, serializer_class: type) -> None:
        """
        새 직렬화기 등록.

        Parameters
        ----------
        format : SerializationFormat
            형식

        serializer_class : type
            직렬화기 클래스
        """
        cls._serializers[format] = serializer_class
```

---

## exceptions.py

```python
# common/exceptions.py
"""
예외 계층 정의.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from common.base import CheckResult


class TruthoundIntegrationError(Exception):
    """
    통합 계층 기본 예외.

    모든 Truthound 통합 예외의 기본 클래스입니다.

    Parameters
    ----------
    message : str
        에러 메시지

    context : dict[str, Any] | None
        추가 컨텍스트 정보

    cause : Exception | None
        원인 예외
    """

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환 (로깅/직렬화용)"""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self!s}, context={self.context})"


class ConfigurationError(TruthoundIntegrationError):
    """
    설정 관련 오류.

    잘못된 설정이나 누락된 필수 설정이 있을 때 발생합니다.

    Examples
    --------
    >>> raise ConfigurationError(
    ...     "Missing required 'rules' parameter",
    ...     context={"parameter": "rules"},
    ... )
    """
    pass


class ValidationExecutionError(TruthoundIntegrationError):
    """
    검증 실행 중 오류.

    Truthound 검증 실행 중 발생하는 오류입니다.

    Examples
    --------
    >>> raise ValidationExecutionError(
    ...     "Failed to execute check",
    ...     context={"rules_count": 5},
    ...     cause=original_error,
    ... )
    """
    pass


class TruthoundCheckError(TruthoundIntegrationError):
    """
    검증 실패 오류.

    검증이 실패했을 때 발생하며, 결과를 포함합니다.

    Parameters
    ----------
    message : str
        에러 메시지

    result : CheckResult
        검증 결과

    Examples
    --------
    >>> if not result.is_success:
    ...     raise TruthoundCheckError(
    ...         f"Quality check failed: {result.failed_count} failures",
    ...         result=result,
    ...     )
    """

    def __init__(
        self,
        message: str,
        *,
        result: "CheckResult",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        base = super().to_dict()
        base["result"] = self.result.to_dict()
        return base


class SerializationError(TruthoundIntegrationError):
    """
    직렬화/역직렬화 오류.

    결과 변환 중 발생하는 오류입니다.
    """
    pass


class PlatformConnectionError(TruthoundIntegrationError):
    """
    플랫폼 연결 오류.

    데이터 소스 연결 실패 시 발생합니다.

    Examples
    --------
    >>> raise PlatformConnectionError(
    ...     "Failed to connect to PostgreSQL",
    ...     context={"host": "db.example.com", "port": 5432},
    ... )
    """
    pass


class TimeoutError(TruthoundIntegrationError):
    """
    타임아웃 오류.

    검증이 설정된 시간 내에 완료되지 않았을 때 발생합니다.
    """
    pass


class QualityGateError(TruthoundIntegrationError):
    """
    품질 게이트 통과 실패.

    파이프라인의 품질 게이트를 통과하지 못했을 때 발생합니다.

    Examples
    --------
    >>> if pass_rate < min_pass_rate:
    ...     raise QualityGateError(
    ...         f"Pass rate {pass_rate:.2%} < threshold {min_pass_rate:.2%}",
    ...         context={"pass_rate": pass_rate, "threshold": min_pass_rate},
    ...     )
    """
    pass
```

---

## testing.py

```python
# common/testing.py
"""
테스트 유틸리티 및 Mock 팩토리.
"""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from unittest.mock import MagicMock
import polars as pl

from common.base import (
    CheckResult,
    ProfileResult,
    CheckStatus,
    Severity,
    ValidationFailure,
)


class MockTruthound:
    """
    Truthound 모듈 Mock.

    테스트에서 실제 Truthound 대신 사용할 수 있는 Mock입니다.

    Parameters
    ----------
    check_result : CheckResult | None
        check() 호출 시 반환할 결과

    profile_result : ProfileResult | None
        profile() 호출 시 반환할 결과

    should_fail : bool
        검증 실패 시뮬레이션 여부

    Examples
    --------
    >>> mock = MockTruthound(should_fail=False)
    >>> result = mock.check(data, rules=[])
    >>> assert result.is_success
    """

    def __init__(
        self,
        check_result: CheckResult | None = None,
        profile_result: ProfileResult | None = None,
        should_fail: bool = False,
    ) -> None:
        self._check_result = check_result or create_mock_check_result(
            is_success=not should_fail
        )
        self._profile_result = profile_result or create_mock_profile_result()
        self._call_history: list[dict[str, Any]] = []

    def check(
        self,
        data: pl.DataFrame,
        rules: list[dict[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Mock 검증 실행"""
        self._call_history.append({
            "method": "check",
            "data_shape": data.shape,
            "rules": rules,
            "kwargs": kwargs,
        })
        return self._check_result

    def profile(
        self,
        data: pl.DataFrame,
        **kwargs: Any,
    ) -> ProfileResult:
        """Mock 프로파일링 실행"""
        self._call_history.append({
            "method": "profile",
            "data_shape": data.shape,
            "kwargs": kwargs,
        })
        return self._profile_result

    def learn(
        self,
        data: pl.DataFrame,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock 스키마 학습"""
        self._call_history.append({
            "method": "learn",
            "data_shape": data.shape,
            "kwargs": kwargs,
        })
        return {
            "rules": [{"column": "id", "type": "not_null"}],
            "columns": {},
        }

    def connect(self, connection_string: str) -> "MockTruthound":
        """Mock 연결"""
        self._call_history.append({
            "method": "connect",
            "connection_string": connection_string,
        })
        return self

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """호출 이력 반환"""
        return self._call_history.copy()

    def reset(self) -> None:
        """호출 이력 초기화"""
        self._call_history.clear()


def create_mock_check_result(
    is_success: bool = True,
    passed_count: int = 5,
    failed_count: int = 0,
    execution_time_ms: float = 50.0,
    failures: list[dict[str, Any]] | None = None,
) -> CheckResult:
    """
    Mock CheckResult 생성.

    Parameters
    ----------
    is_success : bool
        성공 여부

    passed_count : int
        통과 수

    failed_count : int
        실패 수

    execution_time_ms : float
        실행 시간

    failures : list[dict[str, Any]] | None
        실패 상세

    Returns
    -------
    CheckResult
        Mock 결과

    Examples
    --------
    >>> result = create_mock_check_result(is_success=False, failed_count=3)
    >>> assert not result.is_success
    >>> assert result.failed_count == 3
    """
    if failures is None and not is_success:
        failures = [
            {
                "rule_name": "not_null",
                "column": "id",
                "message": "Column contains NULL values",
                "severity": "high",
                "failed_count": 10,
                "total_count": 100,
            }
        ]

    failure_tuples = tuple(
        ValidationFailure(
            rule_name=f["rule_name"],
            column=f.get("column"),
            message=f["message"],
            severity=Severity(f.get("severity", "medium")),
            failed_count=f["failed_count"],
            total_count=f["total_count"],
        )
        for f in (failures or [])
    )

    return CheckResult(
        status=CheckStatus.PASSED if is_success else CheckStatus.FAILED,
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=0,
        skipped_count=0,
        failures=failure_tuples,
        execution_time_ms=execution_time_ms,
        timestamp=datetime.now(timezone.utc),
    )


def create_mock_profile_result(
    row_count: int = 1000,
    columns: dict[str, dict[str, Any]] | None = None,
    execution_time_ms: float = 100.0,
) -> ProfileResult:
    """
    Mock ProfileResult 생성.

    Parameters
    ----------
    row_count : int
        행 수

    columns : dict[str, dict[str, Any]] | None
        컬럼 프로파일

    execution_time_ms : float
        실행 시간

    Returns
    -------
    ProfileResult
        Mock 결과
    """
    if columns is None:
        columns = {
            "id": {"dtype": "int64", "null_count": 0, "unique_count": 1000},
            "name": {"dtype": "string", "null_count": 5, "unique_count": 950},
            "amount": {"dtype": "float64", "null_count": 0, "min": 0.0, "max": 1000.0},
        }

    return ProfileResult(
        columns=columns,
        row_count=row_count,
        execution_time_ms=execution_time_ms,
        timestamp=datetime.now(timezone.utc),
    )


def create_sample_dataframe(
    rows: int = 100,
    with_nulls: bool = False,
    with_duplicates: bool = False,
) -> pl.DataFrame:
    """
    테스트용 샘플 DataFrame 생성.

    Parameters
    ----------
    rows : int
        행 수

    with_nulls : bool
        NULL 값 포함 여부

    with_duplicates : bool
        중복 값 포함 여부

    Returns
    -------
    pl.DataFrame
        샘플 데이터
    """
    ids = list(range(1, rows + 1))
    if with_duplicates:
        ids[-1] = ids[0]  # 마지막 ID를 첫 번째와 동일하게

    emails = [f"user{i}@example.com" for i in range(1, rows + 1)]
    if with_nulls:
        emails[-1] = None  # 마지막 이메일을 NULL로

    ages = [25 + (i % 50) for i in range(rows)]
    amounts = [100.0 * (i + 1) for i in range(rows)]

    return pl.DataFrame({
        "id": ids,
        "email": emails,
        "age": ages,
        "amount": amounts,
    })


def assert_check_result(
    result: CheckResult,
    *,
    is_success: bool | None = None,
    min_passed: int | None = None,
    max_failed: int | None = None,
    status: CheckStatus | None = None,
) -> None:
    """
    CheckResult 검증 헬퍼.

    Parameters
    ----------
    result : CheckResult
        검증할 결과

    is_success : bool | None
        기대 성공 여부

    min_passed : int | None
        최소 통과 수

    max_failed : int | None
        최대 실패 수

    status : CheckStatus | None
        기대 상태

    Raises
    ------
    AssertionError
        조건 미충족 시

    Examples
    --------
    >>> assert_check_result(
    ...     result,
    ...     is_success=True,
    ...     min_passed=3,
    ... )
    """
    if is_success is not None:
        assert result.is_success == is_success, (
            f"Expected is_success={is_success}, got {result.is_success}"
        )

    if min_passed is not None:
        assert result.passed_count >= min_passed, (
            f"Expected at least {min_passed} passed, got {result.passed_count}"
        )

    if max_failed is not None:
        assert result.failed_count <= max_failed, (
            f"Expected at most {max_failed} failed, got {result.failed_count}"
        )

    if status is not None:
        assert result.status == status, (
            f"Expected status={status}, got {result.status}"
        )


def create_mock_context(platform: str = "test") -> MagicMock:
    """
    플랫폼 컨텍스트 Mock 생성.

    Parameters
    ----------
    platform : str
        플랫폼 이름

    Returns
    -------
    MagicMock
        Mock 컨텍스트
    """
    context = MagicMock()
    context.log = MagicMock()
    context.log.info = MagicMock()
    context.log.warning = MagicMock()
    context.log.error = MagicMock()

    if platform == "airflow":
        context.ti = MagicMock()
        context.ti.xcom_push = MagicMock()
        context.ti.xcom_pull = MagicMock()
        context.ds = "2024-01-01"

    elif platform == "dagster":
        context.add_output_metadata = MagicMock()

    return context


# =============================================================================
# Async Mock Engines
# =============================================================================


class AsyncMockDataQualityEngine:
    """
    비동기 Mock 엔진.

    AsyncDataQualityEngine Protocol을 구현하며, 비동기 테스트에서 사용됩니다.

    Examples
    --------
    >>> engine = AsyncMockDataQualityEngine()
    >>> engine.configure_check(success=True, delay_seconds=0.1)
    >>> result = await engine.check(data, rules=[])
    >>> assert result.is_success
    """

    def __init__(self, name: str = "async_mock", version: str = "1.0.0") -> None:
        self._name = name
        self._version = version
        # 설정 및 호출 기록 관리
        ...

    def configure_check(
        self,
        *,
        success: bool = True,
        delay_seconds: float = 0.0,
        raise_error: Exception | None = None,
    ) -> None:
        """check 동작 설정."""
        ...

    async def check(self, data, rules, **kwargs) -> CheckResult:
        """비동기 검증 실행."""
        ...

    async def profile(self, data, **kwargs) -> ProfileResult:
        """비동기 프로파일링."""
        ...

    async def learn(self, data, **kwargs) -> LearnResult:
        """비동기 스키마 학습."""
        ...


class AsyncMockManagedEngine:
    """
    라이프사이클을 지원하는 비동기 Mock 엔진.

    Examples
    --------
    >>> engine = AsyncMockManagedEngine()
    >>> engine.configure_lifecycle(health_status="healthy")
    >>> async with engine:
    ...     result = await engine.check(data, rules)
    ...     health = await engine.health_check()
    """

    async def start(self) -> None:
        """비동기 시작."""
        ...

    async def stop(self) -> None:
        """비동기 종료."""
        ...

    async def health_check(self) -> HealthCheckResult:
        """비동기 헬스 체크."""
        ...


def create_async_mock_engine(
    *,
    name: str = "async_mock",
    check_success: bool = True,
    check_delay_seconds: float = 0.0,
) -> AsyncMockDataQualityEngine:
    """
    설정된 비동기 Mock 엔진 생성.

    Examples
    --------
    >>> engine = create_async_mock_engine(check_success=True)
    >>> result = await engine.check(data, rules=[])
    """
    engine = AsyncMockDataQualityEngine(name=name)
    engine.configure_check(success=check_success, delay_seconds=check_delay_seconds)
    return engine


def create_async_mock_managed_engine(
    *,
    name: str = "async_mock_managed",
    health_status: str = "healthy",
) -> AsyncMockManagedEngine:
    """
    라이프사이클 지원 비동기 Mock 엔진 생성.

    Examples
    --------
    >>> engine = create_async_mock_managed_engine(health_status="healthy")
    >>> async with engine:
    ...     health = await engine.health_check()
    """
    engine = AsyncMockManagedEngine(name=name)
    engine.configure_lifecycle(health_status=health_status)
    return engine
```

---

## rule_validation.py

규칙 딕셔너리의 스키마를 검증하고 정규화하는 모듈입니다. 런타임 에러를 방지하기 위해 규칙 타입, 필수 필드, 필드 값을 사전에 검증합니다.

### Purpose

- **런타임 에러 방지**: 잘못된 규칙 타입이나 파라미터로 인한 런타임 에러 사전 감지
- **엔진별 검증**: 각 엔진(Truthound, Great Expectations, Pandera)의 지원 규칙 검증
- **규칙 정규화**: 별칭 해석 및 기본값 적용
- **확장성**: 커스텀 규칙 스키마 등록 지원

### Core Types

#### Enums

```python
from enum import Enum, auto

class FieldType(Enum):
    """규칙 필드의 데이터 타입."""
    STRING = auto()           # 문자열
    INTEGER = auto()          # 정수
    FLOAT = auto()            # 실수
    BOOLEAN = auto()          # 불리언
    LIST = auto()             # 리스트
    DICT = auto()             # 딕셔너리
    ANY = auto()              # 모든 타입
    COLUMN_NAME = auto()      # 컬럼명 (문자열, 유효성 검사 포함)
    REGEX_PATTERN = auto()    # 정규식 패턴 (컴파일 가능성 검사)
    POSITIVE_INTEGER = auto() # 양의 정수
    POSITIVE_FLOAT = auto()   # 양의 실수
    NON_EMPTY_STRING = auto() # 비어있지 않은 문자열
    NON_EMPTY_LIST = auto()   # 비어있지 않은 리스트


class RuleCategory(Enum):
    """규칙 카테고리."""
    COMPLETENESS = auto()     # 완전성 (not_null, required)
    UNIQUENESS = auto()       # 고유성 (unique, distinct)
    VALIDITY = auto()         # 유효성 (in_set, regex, dtype)
    CONSISTENCY = auto()      # 일관성 (cross-column checks)
    ACCURACY = auto()         # 정확성 (range checks, bounds)
    CUSTOM = auto()           # 사용자 정의
```

#### Schema Types

```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class FieldSchema:
    """규칙 필드 스키마 정의."""
    name: str                                      # 필드 이름
    field_type: FieldType                          # 필드 타입
    required: bool = True                          # 필수 여부
    default: Any = None                            # 기본값
    choices: tuple[Any, ...] | None = None         # 허용 값 목록
    min_value: float | None = None                 # 최소값 (숫자형)
    max_value: float | None = None                 # 최대값 (숫자형)
    description: str | None = None                 # 설명


@dataclass(frozen=True)
class RuleSchema:
    """규칙 스키마 정의."""
    rule_type: str                                 # 규칙 타입 (예: "not_null")
    fields: tuple[FieldSchema, ...]                # 필드 스키마 목록
    aliases: tuple[str, ...] = ()                  # 별칭 목록 (예: ("notnull", "notna"))
    category: RuleCategory = RuleCategory.CUSTOM   # 규칙 카테고리
    engines: tuple[str, ...] = ()                  # 지원 엔진 (비어있으면 모든 엔진)
    description: str | None = None                 # 규칙 설명


@dataclass(frozen=True)
class RuleValidationResult:
    """단일 규칙 검증 결과."""
    is_valid: bool                                 # 유효 여부
    errors: tuple[str, ...] = ()                   # 에러 메시지 목록
    normalized_rule: dict[str, Any] | None = None  # 정규화된 규칙


@dataclass(frozen=True)
class BatchValidationResult:
    """복수 규칙 검증 결과."""
    is_valid: bool                                 # 전체 유효 여부
    results: tuple[RuleValidationResult, ...]      # 개별 결과
    total_count: int                               # 전체 규칙 수
    valid_count: int                               # 유효 규칙 수
    invalid_count: int                             # 무효 규칙 수
```

### Exception Hierarchy

```python
class RuleValidationError(DataQualityIntegrationError):
    """규칙 검증 기본 예외."""
    pass


class UnknownRuleTypeError(RuleValidationError):
    """알 수 없는 규칙 타입."""
    def __init__(self, rule_type: str, available_types: tuple[str, ...] = ()):
        self.rule_type = rule_type
        self.available_types = available_types
        super().__init__(f"Unknown rule type: '{rule_type}'")


class MissingFieldError(RuleValidationError):
    """필수 필드 누락."""
    def __init__(self, field_name: str, rule_type: str):
        self.field_name = field_name
        self.rule_type = rule_type
        super().__init__(f"Missing required field '{field_name}' for rule type '{rule_type}'")


class InvalidFieldTypeError(RuleValidationError):
    """잘못된 필드 타입."""
    def __init__(self, field_name: str, expected_type: FieldType, actual_type: type):
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        super().__init__(
            f"Invalid type for field '{field_name}': "
            f"expected {expected_type.name}, got {actual_type.__name__}"
        )


class InvalidFieldValueError(RuleValidationError):
    """잘못된 필드 값."""
    def __init__(self, field_name: str, value: Any, message: str):
        self.field_name = field_name
        self.value = value
        self.message = message
        super().__init__(f"Invalid value for field '{field_name}': {message}")


class MultipleRuleValidationErrors(RuleValidationError):
    """복수 검증 오류."""
    def __init__(self, errors: list[RuleValidationError]):
        self.errors = tuple(errors)
        super().__init__(f"Multiple validation errors: {len(errors)} errors")
```

### Validators

```python
from typing import Protocol

class RuleValidator(Protocol):
    """규칙 검증기 프로토콜."""

    def validate(self, rule: dict[str, Any]) -> RuleValidationResult:
        """단일 규칙 검증."""
        ...

    def validate_batch(self, rules: Sequence[dict[str, Any]]) -> BatchValidationResult:
        """복수 규칙 검증."""
        ...

    def is_supported(self, rule_type: str) -> bool:
        """규칙 타입 지원 여부 확인."""
        ...


class StandardRuleValidator:
    """표준 규칙 검증기."""

    def __init__(self, registry: RuleRegistry | None = None):
        self._registry = registry or get_rule_registry()

    def validate(self, rule: dict[str, Any]) -> RuleValidationResult:
        """
        단일 규칙 검증.

        Parameters
        ----------
        rule : dict[str, Any]
            검증할 규칙 딕셔너리

        Returns
        -------
        RuleValidationResult
            검증 결과
        """
        ...


class TruthoundRuleValidator(StandardRuleValidator):
    """Truthound 전용 검증기."""
    pass


class GreatExpectationsRuleValidator(StandardRuleValidator):
    """Great Expectations 전용 검증기."""
    pass


class PanderaRuleValidator(StandardRuleValidator):
    """Pandera 전용 검증기."""
    pass
```

### Rule Registry

```python
class RuleRegistry:
    """규칙 스키마 레지스트리 (싱글톤)."""

    _instance: RuleRegistry | None = None

    @classmethod
    def get_instance(cls) -> RuleRegistry:
        """싱글톤 인스턴스 반환."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, schema: RuleSchema) -> None:
        """규칙 스키마 등록."""
        ...

    def get_schema(self, rule_type: str) -> RuleSchema | None:
        """규칙 스키마 조회."""
        ...

    def get_all_types(self, engine: str | None = None) -> tuple[str, ...]:
        """모든 규칙 타입 목록."""
        ...

    def resolve_alias(self, alias: str) -> str | None:
        """별칭을 규칙 타입으로 해석."""
        ...


# 전역 레지스트리 접근 함수
def get_rule_registry() -> RuleRegistry:
    """전역 규칙 레지스트리 반환."""
    return RuleRegistry.get_instance()

def reset_rule_registry() -> None:
    """전역 레지스트리 초기화 (테스트용)."""
    RuleRegistry._instance = None
```

### Rule Normalizer

```python
class RuleNormalizer:
    """규칙 정규화기."""

    def __init__(self, registry: RuleRegistry | None = None):
        self._registry = registry or get_rule_registry()

    def normalize(self, rule: dict[str, Any]) -> dict[str, Any]:
        """
        단일 규칙 정규화.

        - 별칭을 표준 규칙 타입으로 해석
        - 누락된 선택 필드에 기본값 적용
        - 불필요한 필드 제거 (선택적)

        Parameters
        ----------
        rule : dict[str, Any]
            정규화할 규칙

        Returns
        -------
        dict[str, Any]
            정규화된 규칙
        """
        ...

    def normalize_rules(self, rules: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """복수 규칙 정규화."""
        ...
```

### Engine Integration

```python
class ValidatingEngineWrapper:
    """검증 기능이 포함된 엔진 래퍼."""

    def __init__(
        self,
        engine: DataQualityEngine,
        validator: RuleValidator | None = None,
        strict: bool = True,
        normalize: bool = True,
    ):
        self._engine = engine
        self._validator = validator or get_validator_for_engine(engine.engine_name)
        self._strict = strict
        self._normalize = normalize

    @property
    def engine_name(self) -> str:
        return self._engine.engine_name

    @property
    def engine_version(self) -> str:
        return self._engine.engine_version

    def check(self, data: Any, rules: Sequence[dict[str, Any]], **kwargs) -> CheckResult:
        """검증 후 check 실행."""
        if rules:
            result = self._validator.validate_batch(rules)
            if not result.is_valid and self._strict:
                raise MultipleRuleValidationErrors([...])

            if self._normalize:
                rules = [r.normalized_rule for r in result.results if r.is_valid]

        return self._engine.check(data, rules, **kwargs)


def wrap_engine_with_validation(
    engine: DataQualityEngine,
    strict: bool = True,
    normalize: bool = True,
) -> ValidatingEngineWrapper:
    """엔진을 검증 래퍼로 감싸기."""
    return ValidatingEngineWrapper(engine, strict=strict, normalize=normalize)


def validate_rules_decorator(
    engine: str | None = None,
    strict: bool = True,
):
    """규칙 검증 데코레이터."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rules = kwargs.get("rules", [])
            if rules:
                result = validate_rules(rules, engine=engine, strict=strict)
                if not result.is_valid and strict:
                    raise MultipleRuleValidationErrors([...])
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Built-in Rule Schemas

```python
# 공통 규칙 스키마
COMMON_RULE_SCHEMAS = (
    RuleSchema(
        rule_type="not_null",
        fields=(
            FieldSchema(name="column", field_type=FieldType.COLUMN_NAME),
        ),
        aliases=("notnull", "notna", "not_na", "required"),
        category=RuleCategory.COMPLETENESS,
        description="Check that column values are not null",
    ),
    RuleSchema(
        rule_type="unique",
        fields=(
            FieldSchema(name="column", field_type=FieldType.COLUMN_NAME),
        ),
        aliases=("distinct",),
        category=RuleCategory.UNIQUENESS,
        description="Check that column values are unique",
    ),
    RuleSchema(
        rule_type="in_set",
        fields=(
            FieldSchema(name="column", field_type=FieldType.COLUMN_NAME),
            FieldSchema(name="values", field_type=FieldType.NON_EMPTY_LIST),
        ),
        aliases=("isin", "in_list", "allowed_values"),
        category=RuleCategory.VALIDITY,
        description="Check that column values are in a set of allowed values",
    ),
    RuleSchema(
        rule_type="in_range",
        fields=(
            FieldSchema(name="column", field_type=FieldType.COLUMN_NAME),
            FieldSchema(name="min", field_type=FieldType.FLOAT, required=False),
            FieldSchema(name="max", field_type=FieldType.FLOAT, required=False),
        ),
        aliases=("between", "range"),
        category=RuleCategory.ACCURACY,
        description="Check that column values are within a range",
    ),
    RuleSchema(
        rule_type="regex",
        fields=(
            FieldSchema(name="column", field_type=FieldType.COLUMN_NAME),
            FieldSchema(name="pattern", field_type=FieldType.REGEX_PATTERN),
        ),
        aliases=("matches", "pattern"),
        category=RuleCategory.VALIDITY,
        description="Check that column values match a regex pattern",
    ),
    # ... 추가 규칙 스키마
)
```

### Convenience Functions

```python
def validate_rule(
    rule: dict[str, Any],
    engine: str | None = None,
    strict: bool = False,
) -> RuleValidationResult:
    """단일 규칙 검증."""
    validator = get_validator_for_engine(engine)
    result = validator.validate(rule)
    if not result.is_valid and strict:
        # 첫 번째 에러로 예외 발생
        raise RuleValidationError(result.errors[0])
    return result


def validate_rules(
    rules: Sequence[dict[str, Any]],
    engine: str | None = None,
    strict: bool = False,
) -> BatchValidationResult:
    """복수 규칙 검증."""
    validator = get_validator_for_engine(engine)
    result = validator.validate_batch(rules)
    if not result.is_valid and strict:
        errors = [RuleValidationError(e) for r in result.results for e in r.errors]
        raise MultipleRuleValidationErrors(errors)
    return result


def normalize_rules(
    rules: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """규칙 정규화."""
    normalizer = RuleNormalizer()
    return normalizer.normalize_rules(rules)


def get_validator_for_engine(engine: str | None = None) -> RuleValidator:
    """엔진별 검증기 팩토리."""
    if engine is None or engine == "truthound":
        return TruthoundRuleValidator()
    elif engine == "great_expectations":
        return GreatExpectationsRuleValidator()
    elif engine == "pandera":
        return PanderaRuleValidator()
    else:
        return StandardRuleValidator()


def register_rule_schema(schema: RuleSchema) -> None:
    """전역 레지스트리에 규칙 스키마 등록."""
    get_rule_registry().register(schema)


def get_supported_rule_types(engine: str | None = None) -> tuple[str, ...]:
    """지원 규칙 타입 목록."""
    return get_rule_registry().get_all_types(engine)
```

### Usage Examples

```python
from common import (
    validate_rules,
    validate_rule,
    normalize_rules,
    wrap_engine_with_validation,
    register_rule_schema,
    RuleSchema,
    FieldSchema,
    FieldType,
    RuleCategory,
    RuleValidationError,
    UnknownRuleTypeError,
    MissingFieldError,
)
from common.engines import TruthoundEngine

# 1. 기본 검증
rule = {"type": "not_null", "column": "id"}
result = validate_rule(rule)
print(f"Valid: {result.is_valid}")

# 2. 엔진별 검증
rules = [
    {"type": "not_null", "column": "id"},
    {"type": "unique", "column": "email"},
    {"type": "in_range", "column": "age", "min": 0, "max": 150},
]
result = validate_rules(rules, engine="great_expectations")
print(f"Valid: {result.valid_count}/{result.total_count}")

# 3. 정규화
normalized = normalize_rules([
    {"type": "notnull", "column": "id"},  # 별칭 사용
    {"type": "between", "column": "age"},  # min/max 생략
])
# [{"type": "not_null", "column": "id"}, {"type": "in_range", "column": "age", "min": None, "max": None}]

# 4. 엔진 래퍼
engine = TruthoundEngine()
validating_engine = wrap_engine_with_validation(engine)
result = validating_engine.check(data, rules=rules)

# 5. 커스텀 규칙 등록
custom_schema = RuleSchema(
    rule_type="statistical_outlier",
    fields=(
        FieldSchema(name="column", field_type=FieldType.COLUMN_NAME),
        FieldSchema(name="method", field_type=FieldType.STRING,
                   choices=("zscore", "iqr", "mad")),
        FieldSchema(name="threshold", field_type=FieldType.POSITIVE_FLOAT, default=3.0),
    ),
    category=RuleCategory.ACCURACY,
)
register_rule_schema(custom_schema)

# 6. 예외 처리
try:
    validate_rules([{"type": "unknown_rule"}], strict=True)
except UnknownRuleTypeError as e:
    print(f"Unknown: {e.rule_type}")
except MissingFieldError as e:
    print(f"Missing: {e.field_name}")
```

---

## Usage Examples

### Using Common Modules in Airflow

```python
from common.base import CheckConfig, CheckResult, WorkflowIntegration
from common.exceptions import TruthoundCheckError
from common.serializers import SerializerFactory, SerializationFormat


class AirflowAdapter:
    @property
    def platform_name(self) -> str:
        return "airflow"

    def check(self, data, config: CheckConfig) -> CheckResult:
        import truthound as th

        result = th.check(data, **config.to_truthound_kwargs())
        return CheckResult.from_truthound(result)


# Operator에서 사용
def execute(self, context):
    config = CheckConfig.from_dict({
        "rules": self.rules,
        "fail_on_error": self.fail_on_error,
    })

    adapter = AirflowAdapter()
    result = adapter.check(data, config)

    # XCom 직렬화
    serializer = SerializerFactory.get_serializer(SerializationFormat.AIRFLOW_XCOM)
    context["ti"].xcom_push(key="result", value=serializer.serialize(result))

    if not result.is_success:
        raise TruthoundCheckError("Check failed", result=result)
```

### Using Common Modules in Tests

```python
import pytest
from common.testing import (
    MockTruthound,
    create_mock_check_result,
    create_sample_dataframe,
    assert_check_result,
)


def test_quality_check():
    # Mock 준비
    mock = MockTruthound(should_fail=False)
    data = create_sample_dataframe(rows=100)

    # 실행
    result = mock.check(data, rules=[{"column": "id", "type": "not_null"}])

    # 검증
    assert_check_result(result, is_success=True, min_passed=1)
    assert len(mock.call_history) == 1


def test_quality_check_failure():
    # 실패 결과 Mock
    mock_result = create_mock_check_result(
        is_success=False,
        failed_count=5,
        failures=[
            {
                "rule_name": "email_format",
                "column": "email",
                "message": "Invalid email",
                "severity": "high",
                "failed_count": 50,
                "total_count": 100,
            }
        ],
    )

    mock = MockTruthound(check_result=mock_result)
    data = create_sample_dataframe(with_nulls=True)

    result = mock.check(data, rules=[])

    assert_check_result(result, is_success=False, max_failed=10)
```

---

## References

- [Python Protocols (PEP 544)](https://peps.python.org/pep-0544/)
- [Python Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Truthound Documentation](https://truthound.dev/docs)

---

*이 문서는 common/ 디렉토리의 완전한 구현 명세입니다.*
