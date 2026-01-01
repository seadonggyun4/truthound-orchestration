# Architecture - Data Quality Orchestration Framework

> **Last Updated:** 2025-12-31
> **Document Version:** 2.0.0
> **Status:** Approved

---

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Core Abstractions](#core-abstractions)
3. [DataQualityEngine Protocol](#dataqualityengine-protocol)
4. [WorkflowIntegration Protocol](#workflowintegration-protocol)
5. [Common Module Design](#common-module-design)
6. [Platform Adapter Pattern](#platform-adapter-pattern)
7. [Error Handling Strategy](#error-handling-strategy)
8. [Serialization Patterns](#serialization-patterns)
9. [Data Flow](#data-flow)

---

## System Architecture Overview

### High-Level Architecture

본 프레임워크는 **엔진 독립적(Engine-Agnostic)** 설계를 채택합니다. `DataQualityEngine` Protocol을 통해 어떤 데이터 품질 엔진이든 플러그인할 수 있으며, Truthound가 기본 엔진으로 제공됩니다.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Workflow Orchestrators                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Airflow   │  │   Dagster   │  │   Prefect   │  │     dbt     │          │
│  │   DAGs      │  │   Graphs    │  │   Flows     │  │   Models    │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Platform Integration Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ truthound-  │  │ truthound-  │  │ truthound-  │  │ truthound-  │          │
│  │  airflow    │  │  dagster    │  │  prefect    │  │    dbt      │          │
│  │             │  │             │  │             │  │             │          │
│  │ • Operators │  │ • Resources │  │ • Blocks    │  │ • Macros    │          │
│  │ • Sensors   │  │ • Assets    │  │ • Tasks     │  │ • Tests     │          │
│  │ • Hooks     │  │ • Ops       │  │ • Flows     │  │             │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
└─────────┼────────────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │                │
          └────────────────┴───────┬────────┴────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Common Layer                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                         common/                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │    │
│  │  │   base.py   │  │  config.py  │  │serializers  │  │ exceptions  │  │    │
│  │  │             │  │             │  │    .py      │  │    .py      │  │    │
│  │  │ • Protocol  │  │ • Env Load  │  │ • JSON      │  │ • Hierarchy │  │    │
│  │  │ • Config    │  │ • Transform │  │ • Dict      │  │ • Context   │  │    │
│  │  │ • Result    │  │ • Validate  │  │ • Pydantic  │  │ • Recovery  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │    │
│  │                                                                       │    │
│  │  ┌────────────────────────────────────────────────────────────────┐  │    │
│  │  │                   DataQualityEngine Protocol                    │  │    │
│  │  │   check(data, rules) -> CheckResult                             │  │    │
│  │  │   profile(data) -> ProfileResult                                │  │    │
│  │  │   learn(data) -> LearnResult                                    │  │    │
│  │  └────────────────────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   TruthoundEngine │  │   GreatExpect.   │  │   Custom Engine  │
│    (Default)      │  │     Adapter      │  │    (Optional)    │
│                   │  │                  │  │                  │
│  th.check()       │  │  ge.validate()   │  │  custom.run()    │
│  th.profile()     │  │  ge.get_batch()  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Data Layer                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Polars    │  │   Pandas    │  │    RDBMS    │  │  Cloud DW   │          │
│  │  DataFrame  │  │  DataFrame  │  │  (PG, My)   │  │ (BQ, SF)    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
┌───────────────────────────────────────────────────────────────────┐
│                    Request Flow (Engine-Agnostic)                  │
└───────────────────────────────────────────────────────────────────┘

 Orchestrator                Integration              DataQualityEngine
     │                           │                         │
     │   1. Execute Task         │                         │
     ├──────────────────────────▶│                         │
     │                           │                         │
     │                           │  2. Build Config        │
     │                           ├────────────┐            │
     │                           │            │            │
     │                           │◀───────────┘            │
     │                           │                         │
     │                           │  3. engine.check(data)  │
     │                           ├────────────────────────▶│ (Truthound/GE/Custom)
     │                           │                         │
     │                           │                         │  4. Execute
     │                           │                         ├───────┐
     │                           │                         │       │
     │                           │                         │◀──────┘
     │                           │                         │
     │                           │  5. CheckResult         │
     │                           │◀────────────────────────┤
     │                           │                         │
     │                           │  6. Serialize Result    │
     │                           ├────────────┐            │
     │                           │            │            │
     │                           │◀───────────┘            │
     │                           │                         │
     │   7. Return Result        │                         │
     │◀──────────────────────────┤                         │
     │                           │                         │
     ▼                           ▼                         ▼
```

---

## Core Abstractions

### Design Principles

| Principle | Description | Rationale |
|-----------|-------------|-----------|
| **Protocol-First** | 모든 컴포넌트는 Protocol을 통해 정의 | 느슨한 결합, 테스트 용이성 |
| **Immutable Config** | 설정 객체는 불변 | 스레드 안전성, 예측 가능성 |
| **Result Monad** | 결과는 성공/실패를 명시적으로 표현 | 에러 처리 일관성 |
| **Lazy Evaluation** | 가능한 경우 연산 지연 | 성능 최적화 |
| **Fail-Fast** | 오류 발생 시 즉시 실패 | 디버깅 용이성 |

### Core Types

```python
from typing import Protocol, TypeVar, Generic, Any
from dataclasses import dataclass
from enum import Enum
import polars as pl

# Type Variables
T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound="BaseConfig")
ResultT = TypeVar("ResultT", bound="BaseResult")


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
```

---

## DataQualityEngine Protocol

### Core Engine Abstraction

`DataQualityEngine`은 본 프레임워크의 핵심 추상화입니다. 모든 데이터 품질 엔진(Truthound, Great Expectations, Pandera 등)은 이 프로토콜을 구현하여 플랫폼 통합에 사용됩니다.

```python
from typing import Protocol, runtime_checkable, Any, Sequence


@runtime_checkable
class DataQualityEngine(Protocol):
    """
    데이터 품질 엔진의 핵심 프로토콜.

    모든 데이터 품질 엔진은 이 프로토콜을 구현해야 합니다.
    Truthound, Great Expectations, Pandera 등 어떤 엔진이든
    이 프로토콜을 구현하면 플랫폼 통합에서 사용 가능합니다.
    """

    @property
    def engine_name(self) -> str:
        """엔진 식별자 (예: 'truthound', 'great_expectations')"""
        ...

    @property
    def engine_version(self) -> str:
        """엔진 버전"""
        ...

    def check(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]],
        **kwargs,
    ) -> "CheckResult":
        """
        데이터 품질 검증을 실행합니다.

        Args:
            data: 검증할 데이터 (Polars/Pandas DataFrame 등)
            rules: 검증 규칙 목록
            **kwargs: 엔진별 추가 옵션

        Returns:
            CheckResult: 검증 결과

        Raises:
            ValidationExecutionError: 검증 실행 중 오류 발생
        """
        ...

    def profile(
        self,
        data: Any,
        **kwargs,
    ) -> "ProfileResult":
        """
        데이터 프로파일링을 실행합니다.

        Args:
            data: 프로파일링할 데이터
            **kwargs: 엔진별 추가 옵션

        Returns:
            ProfileResult: 프로파일링 결과
        """
        ...

    def learn(
        self,
        data: Any,
        **kwargs,
    ) -> "LearnResult":
        """
        데이터에서 검증 규칙을 자동 학습합니다.

        Args:
            data: 학습할 데이터
            **kwargs: 엔진별 추가 옵션

        Returns:
            LearnResult: 학습된 규칙
        """
        ...


@runtime_checkable
class AsyncDataQualityEngine(Protocol):
    """비동기 데이터 품질 엔진 프로토콜"""

    async def check_async(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]],
        **kwargs,
    ) -> "CheckResult":
        """비동기 데이터 품질 검증"""
        ...

    async def profile_async(
        self,
        data: Any,
        **kwargs,
    ) -> "ProfileResult":
        """비동기 데이터 프로파일링"""
        ...
```

### Engine Implementations

#### TruthoundEngine (Default)

```python
class TruthoundEngine:
    """
    Truthound 기반 기본 엔진 구현.

    Truthound가 설치된 경우 기본 엔진으로 사용됩니다.
    """

    @property
    def engine_name(self) -> str:
        return "truthound"

    @property
    def engine_version(self) -> str:
        import truthound
        return truthound.__version__

    def check(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        rules: Sequence[dict[str, Any]],
        **kwargs,
    ) -> CheckResult:
        import truthound as th

        result = th.check(data, list(rules), **kwargs)
        return CheckResult.from_truthound(result)

    def profile(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        **kwargs,
    ) -> ProfileResult:
        import truthound as th

        result = th.profile(data, **kwargs)
        return ProfileResult.from_truthound(result)

    def learn(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        **kwargs,
    ) -> LearnResult:
        import truthound as th

        result = th.learn(data, **kwargs)
        return LearnResult.from_truthound(result)
```

#### GreatExpectationsAdapter

```python
class GreatExpectationsAdapter:
    """
    Great Expectations를 DataQualityEngine으로 어댑팅.

    GE의 Expectation Suite를 rules 형식으로 변환하여 사용합니다.
    """

    @property
    def engine_name(self) -> str:
        return "great_expectations"

    def check(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]],
        **kwargs,
    ) -> CheckResult:
        import great_expectations as ge

        # rules를 GE Expectation으로 변환
        suite = self._rules_to_expectations(rules)

        # GE 검증 실행
        result = ge.validate(data, suite)

        # 결과를 CheckResult로 변환
        return self._convert_result(result)
```

#### Custom Engine Example

```python
class CustomEngine:
    """
    사용자 정의 엔진 예시.

    DataQualityEngine Protocol만 구현하면
    어떤 엔진이든 플러그인 가능합니다.
    """

    @property
    def engine_name(self) -> str:
        return "custom"

    def check(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]],
        **kwargs,
    ) -> CheckResult:
        # 사용자 정의 검증 로직
        results = []
        for rule in rules:
            passed = self._evaluate_rule(data, rule)
            results.append(passed)

        return CheckResult(
            status=CheckStatus.PASSED if all(results) else CheckStatus.FAILED,
            passed_count=sum(results),
            failed_count=len(results) - sum(results),
        )
```

### Engine Registry

```python
from typing import Type


class EngineRegistry:
    """데이터 품질 엔진 레지스트리"""

    _engines: dict[str, Type[DataQualityEngine]] = {}
    _default_engine: str = "truthound"

    @classmethod
    def register(cls, name: str, engine_class: Type[DataQualityEngine]) -> None:
        """엔진 등록"""
        cls._engines[name] = engine_class

    @classmethod
    def get(cls, name: str | None = None) -> DataQualityEngine:
        """엔진 인스턴스 반환 (None이면 기본 엔진)"""
        engine_name = name or cls._default_engine
        if engine_name not in cls._engines:
            raise EngineNotFoundError(f"Engine '{engine_name}' not found")
        return cls._engines[engine_name]()

    @classmethod
    def set_default(cls, name: str) -> None:
        """기본 엔진 설정"""
        if name not in cls._engines:
            raise EngineNotFoundError(f"Engine '{name}' not found")
        cls._default_engine = name


# 기본 엔진 등록
EngineRegistry.register("truthound", TruthoundEngine)
```

---

## WorkflowIntegration Protocol

### Protocol Definition

```python
from typing import Protocol, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class WorkflowIntegration(Protocol):
    """
    워크플로우 통합을 위한 핵심 프로토콜.

    모든 플랫폼 어댑터는 이 프로토콜을 구현해야 합니다.
    engine 속성을 통해 사용할 DataQualityEngine을 지정합니다.
    """

    @property
    def platform_name(self) -> str:
        """플랫폼 식별자 (예: 'airflow', 'dagster')"""
        ...

    @property
    def engine(self) -> DataQualityEngine:
        """사용할 데이터 품질 엔진"""
        ...

    @property
    def platform_version(self) -> str:
        """지원하는 플랫폼 버전"""
        ...

    def check(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        config: "CheckConfig",
    ) -> "CheckResult":
        """
        데이터 품질 검증을 실행합니다.

        Args:
            data: 검증할 데이터
            config: 검증 설정

        Returns:
            CheckResult: 검증 결과

        Raises:
            TruthoundCheckError: 검증 실행 중 오류 발생
        """
        ...

    def profile(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        config: "ProfileConfig",
    ) -> "ProfileResult":
        """
        데이터 프로파일링을 실행합니다.

        Args:
            data: 프로파일링할 데이터
            config: 프로파일링 설정

        Returns:
            ProfileResult: 프로파일링 결과
        """
        ...

    def learn(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        config: "LearnConfig | None" = None,
    ) -> "Schema":
        """
        데이터에서 스키마를 자동 학습합니다.

        Args:
            data: 학습할 데이터
            config: 학습 설정 (선택적)

        Returns:
            Schema: 학습된 스키마
        """
        ...

    def serialize_result(
        self,
        result: "BaseResult",
        format: "SerializationFormat",
    ) -> str | dict:
        """
        결과를 지정된 형식으로 직렬화합니다.

        Args:
            result: 직렬화할 결과
            format: 출력 형식

        Returns:
            직렬화된 결과
        """
        ...


@runtime_checkable
class AsyncWorkflowIntegration(Protocol):
    """비동기 워크플로우 통합 프로토콜"""

    async def check_async(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        config: "CheckConfig",
    ) -> "CheckResult":
        """비동기 데이터 품질 검증"""
        ...

    async def profile_async(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        config: "ProfileConfig",
    ) -> "ProfileResult":
        """비동기 데이터 프로파일링"""
        ...
```

### Protocol Usage Example

```python
from common.base import WorkflowIntegration, DataQualityEngine, CheckConfig, CheckResult
from common.engines import TruthoundEngine, EngineRegistry


class AirflowDataQualityAdapter:
    """Airflow용 데이터 품질 어댑터 (엔진 독립적)"""

    def __init__(self, engine: DataQualityEngine | None = None):
        # 기본값: TruthoundEngine
        self._engine = engine or EngineRegistry.get()

    @property
    def platform_name(self) -> str:
        return "airflow"

    @property
    def platform_version(self) -> str:
        return ">=2.6.0"

    @property
    def engine(self) -> DataQualityEngine:
        return self._engine

    def check(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        config: CheckConfig,
    ) -> CheckResult:
        # 엔진 추상화를 통한 호출 (Truthound, GE, 커스텀 등)
        return self._engine.check(data, config.rules, **config.options)


# 다양한 엔진으로 어댑터 사용
adapter_with_truthound = AirflowDataQualityAdapter()  # 기본값: Truthound
adapter_with_ge = AirflowDataQualityAdapter(engine=GreatExpectationsAdapter())
adapter_with_custom = AirflowDataQualityAdapter(engine=CustomEngine())

# 타입 체크 - 어떤 플랫폼 어댑터든, 어떤 엔진이든 사용 가능
def execute_check(adapter: WorkflowIntegration) -> CheckResult:
    """어떤 플랫폼 어댑터든 사용 가능"""
    return adapter.check(data, config)
```

---

## Common Module Design

### CheckConfig

```python
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class FailureAction(Enum):
    """검증 실패 시 동작"""
    RAISE = "raise"       # 예외 발생
    WARN = "warn"         # 경고만 출력
    LOG = "log"           # 로그 기록
    CONTINUE = "continue" # 무시하고 계속


@dataclass(frozen=True)
class CheckConfig:
    """
    데이터 품질 검증 설정.

    불변 객체로 설계되어 스레드 안전합니다.

    Attributes:
        rules: 적용할 검증 규칙 목록
        fail_on_error: 오류 발생 시 실패 여부
        failure_action: 검증 실패 시 동작
        sample_size: 샘플링할 행 수 (None=전체)
        parallel: 병렬 실행 여부
        timeout_seconds: 타임아웃 (초)
        tags: 메타데이터 태그
        extra: 플랫폼별 추가 설정

    Example:
        >>> config = CheckConfig(
        ...     rules=[
        ...         {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
        ...         {"column": "age", "type": "in_range", "min": 0, "max": 150},
        ...     ],
        ...     fail_on_error=True,
        ...     failure_action=FailureAction.RAISE,
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
            raise ValueError("sample_size must be positive")

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

    def to_truthound_kwargs(self) -> dict[str, Any]:
        """Truthound API 호출용 kwargs 변환"""
        return {
            "rules": list(self.rules),
            "fail_on_error": self.fail_on_error,
            "sample_size": self.sample_size,
            "parallel": self.parallel,
            "timeout": self.timeout_seconds,
            **self.extra,
        }
```

### CheckResult

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
import json


@dataclass(frozen=True)
class ValidationFailure:
    """개별 검증 실패 정보"""
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


@dataclass(frozen=True)
class CheckResult:
    """
    데이터 품질 검증 결과.

    Attributes:
        status: 전체 검증 상태
        passed_count: 통과한 규칙 수
        failed_count: 실패한 규칙 수
        warning_count: 경고 규칙 수
        failures: 실패 상세 정보
        execution_time_ms: 실행 시간 (밀리초)
        timestamp: 실행 시각
        metadata: 추가 메타데이터

    Example:
        >>> result = CheckResult(
        ...     status=CheckStatus.FAILED,
        ...     passed_count=8,
        ...     failed_count=2,
        ...     failures=(
        ...         ValidationFailure(
        ...             rule_name="email_format",
        ...             column="email",
        ...             message="Invalid email format",
        ...             severity=Severity.HIGH,
        ...             failed_count=150,
        ...             total_count=10000,
        ...         ),
        ...     ),
        ... )
        >>> result.is_success
        False
        >>> result.failure_rate
        0.2
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
    def failure_rate(self) -> float:
        """실패율"""
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count

    def iter_failures(
        self,
        min_severity: Severity = Severity.INFO,
    ) -> Iterator[ValidationFailure]:
        """심각도 필터링된 실패 순회"""
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        min_order = severity_order[min_severity]

        for failure in self.failures:
            if severity_order[failure.severity] <= min_order:
                yield failure

    @classmethod
    def from_truthound(cls, th_result: Any) -> "CheckResult":
        """Truthound 결과에서 변환"""
        failures = tuple(
            ValidationFailure(
                rule_name=f.rule_name,
                column=f.column,
                message=f.message,
                severity=Severity(f.severity),
                failed_count=f.failed_count,
                total_count=f.total_count,
                sample_values=tuple(f.sample_values[:5]),
            )
            for f in th_result.failures
        )

        return cls(
            status=CheckStatus(th_result.status),
            passed_count=th_result.passed_count,
            failed_count=th_result.failed_count,
            warning_count=th_result.warning_count,
            skipped_count=th_result.skipped_count,
            failures=failures,
            execution_time_ms=th_result.execution_time_ms,
            metadata=th_result.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "status": self.status.value,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "skipped_count": self.skipped_count,
            "failures": [
                {
                    "rule_name": f.rule_name,
                    "column": f.column,
                    "message": f.message,
                    "severity": f.severity.value,
                    "failed_count": f.failed_count,
                    "total_count": f.total_count,
                    "failure_rate": f.failure_rate,
                }
                for f in self.failures
            ],
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "is_success": self.is_success,
            "failure_rate": self.failure_rate,
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=2)
```

### ProfileConfig

```python
@dataclass(frozen=True)
class ProfileConfig:
    """
    데이터 프로파일링 설정.

    Attributes:
        columns: 프로파일링할 컬럼 (None=전체)
        include_statistics: 통계 포함 여부
        include_patterns: 패턴 감지 여부
        include_distributions: 분포 분석 여부
        sample_size: 샘플 크기
        timeout_seconds: 타임아웃
    """
    columns: frozenset[str] | None = None
    include_statistics: bool = True
    include_patterns: bool = True
    include_distributions: bool = True
    sample_size: int | None = None
    timeout_seconds: int = 300
    extra: dict[str, Any] = field(default_factory=dict)

    def to_truthound_kwargs(self) -> dict[str, Any]:
        """Truthound API 호출용 kwargs 변환"""
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
```

---

## Platform Adapter Pattern

### Adapter Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     Platform Adapter                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Platform Native API                    │   │
│  │  (Operator, Resource, Task, Macro)                       │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Adapter Core                           │   │
│  │  - Config 변환                                            │   │
│  │  - 플랫폼 컨텍스트 주입                                    │   │
│  │  - 결과 후처리                                            │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │               WorkflowIntegration Protocol                │   │
│  │  check(), profile(), learn(), serialize_result()         │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Truthound   │
                    │     Core      │
                    └───────────────┘
```

### Platform-Specific Implementations

#### Airflow Adapter

```python
from airflow.models import BaseOperator
from airflow.utils.context import Context
from common.base import WorkflowIntegration, CheckConfig, CheckResult, DataQualityEngine
from common.engines import EngineRegistry


class DataQualityCheckOperator(BaseOperator):
    """
    Airflow Operator로 래핑된 데이터 품질 검증.

    엔진 독립적 설계:
    - engine 파라미터로 사용할 엔진 지정 (기본값: Truthound)
    - 어떤 DataQualityEngine 구현체든 사용 가능

    플랫폼 특화:
    - XCom으로 결과 전달
    - Airflow Connection 활용
    - Task Context 주입
    """

    template_fields = ("rules", "data_path", "connection_id")

    def __init__(
        self,
        *,
        rules: list[dict],
        data_path: str | None = None,
        connection_id: str = "data_quality_default",
        fail_on_error: bool = True,
        engine: DataQualityEngine | None = None,  # 엔진 플러그인
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rules = rules
        self.data_path = data_path
        self.connection_id = connection_id
        self.fail_on_error = fail_on_error
        self._engine = engine or EngineRegistry.get()  # 기본값: Truthound

    def execute(self, context: Context) -> dict:
        # 데이터 로드
        data = self._load_data(context)

        # 엔진을 통한 검증 실행 (Truthound, GE, Custom 등)
        result = self._engine.check(
            data,
            self.rules,
            context=self._extract_context(context),
        )

        # XCom 푸시
        context["ti"].xcom_push(key="quality_result", value=result.to_dict())

        # 실패 처리
        if not result.is_success and self.fail_on_error:
            raise AirflowException(f"Quality check failed: {result.failed_count} failures")

        return result.to_dict()


# 사용 예시
validate_with_truthound = DataQualityCheckOperator(
    task_id="validate_data",
    rules=[{"column": "id", "type": "not_null"}],
)

validate_with_ge = DataQualityCheckOperator(
    task_id="validate_data",
    rules=[{"column": "id", "type": "not_null"}],
    engine=GreatExpectationsAdapter(),  # GE 사용
)
```

#### Dagster Adapter

```python
from dagster import ConfigurableResource, asset, AssetExecutionContext
from common.base import WorkflowIntegration, CheckConfig, CheckResult, DataQualityEngine
from common.engines import EngineRegistry


class DataQualityResource(ConfigurableResource):
    """
    Dagster Resource로 제공되는 데이터 품질 검증.

    엔진 독립적 설계:
    - engine_name으로 사용할 엔진 지정 (기본값: 'truthound')
    - 런타임에 엔진 인스턴스 주입 가능

    플랫폼 특화:
    - Dagster 설정 통합
    - Asset Materialization 메타데이터
    - IO Manager 통합
    """

    connection_string: str | None = None
    default_timeout: int = 300
    engine_name: str = "truthound"  # 기본 엔진

    def _get_engine(self) -> DataQualityEngine:
        return EngineRegistry.get(self.engine_name)

    def check(
        self,
        data: pl.DataFrame,
        rules: list[dict],
        **kwargs,
    ) -> CheckResult:
        engine = self._get_engine()
        return engine.check(data, rules, timeout=self.default_timeout, **kwargs)


@asset
def quality_checked_data(
    context: AssetExecutionContext,
    data_quality: DataQualityResource,  # 엔진 독립적 리소스
    raw_data: pl.DataFrame,
) -> pl.DataFrame:
    """품질 검증된 데이터 Asset"""
    result = data_quality.check(
        raw_data,
        rules=[{"column": "id", "type": "not_null"}],
    )

    # Asset 메타데이터 추가
    context.add_output_metadata({
        "quality_status": result.status.value,
        "passed_checks": result.passed_count,
        "failed_checks": result.failed_count,
        "engine": data_quality.engine_name,
    })

    if not result.is_success:
        raise Exception(f"Quality check failed: {result.failed_count}")

    return raw_data


# 다양한 엔진으로 리소스 구성
defs_with_truthound = Definitions(
    assets=[quality_checked_data],
    resources={"data_quality": DataQualityResource()},  # 기본값: Truthound
)

defs_with_ge = Definitions(
    assets=[quality_checked_data],
    resources={"data_quality": DataQualityResource(engine_name="great_expectations")},
)
```

#### Prefect Adapter

```python
from prefect import task, flow
from prefect.blocks.core import Block
from common.base import WorkflowIntegration, CheckConfig, CheckResult, DataQualityEngine
from common.engines import EngineRegistry


class DataQualityBlock(Block):
    """
    Prefect Block으로 저장되는 데이터 품질 설정.

    엔진 독립적 설계:
    - engine_name으로 사용할 엔진 지정 (기본값: 'truthound')
    - Block 저장소에 설정 영구 저장

    플랫폼 특화:
    - Prefect Block 저장소 통합
    - Artifact 생성
    - 로깅 통합
    """

    _block_type_name = "data-quality"
    _block_type_slug = "data-quality"

    connection_string: str | None = None
    default_rules: list[dict] = []
    engine_name: str = "truthound"  # 기본 엔진

    def get_engine(self) -> DataQualityEngine:
        return EngineRegistry.get(self.engine_name)

    def check(self, data: pl.DataFrame, rules: list[dict] | None = None) -> CheckResult:
        engine = self.get_engine()
        return engine.check(data, rules or self.default_rules)


@task(name="data_quality_check")
def data_quality_check(
    data: pl.DataFrame,
    rules: list[dict],
    engine: DataQualityEngine | None = None,  # 엔진 직접 주입 가능
) -> CheckResult:
    """Prefect Task로 래핑된 데이터 품질 검증"""
    from prefect.artifacts import create_table_artifact

    # 엔진 결정: 직접 주입 > 기본값(Truthound)
    _engine = engine or EngineRegistry.get()
    result = _engine.check(data, rules)

    # Artifact 생성
    create_table_artifact(
        key="quality-check-result",
        table=[f.to_dict() for f in result.failures],
        description=f"Quality Check: {result.status.value} (Engine: {_engine.engine_name})",
    )

    return result


# 사용 예시
@flow
def validation_pipeline(data: pl.DataFrame):
    # 기본 엔진 (Truthound) 사용
    result1 = data_quality_check(data, rules=[{"column": "id", "type": "not_null"}])

    # GE 엔진 사용
    result2 = data_quality_check(
        data,
        rules=[{"column": "id", "type": "not_null"}],
        engine=GreatExpectationsAdapter(),
    )
```

---

## Error Handling Strategy

### Exception Hierarchy

```python
class DataQualityIntegrationError(Exception):
    """통합 계층 기본 예외"""

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.context = context or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }


class ConfigurationError(DataQualityIntegrationError):
    """설정 관련 오류"""
    pass


class ValidationExecutionError(DataQualityIntegrationError):
    """검증 실행 중 오류"""
    pass


class SerializationError(DataQualityIntegrationError):
    """직렬화/역직렬화 오류"""
    pass


class PlatformConnectionError(DataQualityIntegrationError):
    """플랫폼 연결 오류"""
    pass


class EngineNotFoundError(DataQualityIntegrationError):
    """요청된 엔진을 찾을 수 없음"""
    pass


class IntegrationTimeoutError(DataQualityIntegrationError):
    """타임아웃 오류"""
    pass
```

### Error Handling Pattern

```python
from contextlib import contextmanager
from typing import Generator
import logging

logger = logging.getLogger(__name__)


@contextmanager
def handle_truthound_errors(
    *,
    operation: str,
    context: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """
    Truthound 오류를 일관되게 처리하는 컨텍스트 매니저.

    Example:
        >>> with handle_truthound_errors(operation="check", context={"task": "validate"}):
        ...     result = th.check(data, rules)
    """
    ctx = context or {}

    try:
        yield
    except TimeoutError as e:
        logger.error(f"Timeout during {operation}: {e}", extra=ctx)
        raise TimeoutError(
            f"Operation '{operation}' timed out",
            context=ctx,
            cause=e,
        ) from e
    except ValueError as e:
        logger.error(f"Configuration error in {operation}: {e}", extra=ctx)
        raise ConfigurationError(
            f"Invalid configuration for '{operation}': {e}",
            context=ctx,
            cause=e,
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error in {operation}: {e}", extra=ctx)
        raise ValidationExecutionError(
            f"Failed to execute '{operation}': {e}",
            context=ctx,
            cause=e,
        ) from e
```

### Platform-Specific Error Mapping

```python
# Airflow
from airflow.exceptions import AirflowException, AirflowSkipException

def map_to_airflow_exception(error: TruthoundIntegrationError) -> AirflowException:
    """Truthound 예외를 Airflow 예외로 변환"""
    if isinstance(error, ConfigurationError):
        return AirflowException(f"Configuration error: {error}")
    if isinstance(error, TimeoutError):
        return AirflowException(f"Task timeout: {error}")
    return AirflowException(str(error))


# Dagster
from dagster import Failure, RetryRequested

def map_to_dagster_exception(error: TruthoundIntegrationError) -> Failure:
    """Truthound 예외를 Dagster 예외로 변환"""
    return Failure(
        description=str(error),
        metadata={
            "error_type": error.__class__.__name__,
            **error.context,
        },
    )


# Prefect
from prefect.exceptions import PrefectException

def map_to_prefect_exception(error: TruthoundIntegrationError) -> PrefectException:
    """Truthound 예외를 Prefect 예외로 변환"""
    return PrefectException(str(error))
```

---

## Serialization Patterns

### Serialization Strategy

```python
from enum import Enum
from typing import Protocol, TypeVar, Any
import json
from dataclasses import asdict


class SerializationFormat(Enum):
    """지원하는 직렬화 형식"""
    JSON = "json"
    DICT = "dict"
    PYDANTIC = "pydantic"
    AIRFLOW_XCOM = "airflow_xcom"
    DAGSTER_OUTPUT = "dagster_output"
    PREFECT_ARTIFACT = "prefect_artifact"


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
    """JSON 직렬화"""

    def serialize(self, obj: CheckResult) -> str:
        return obj.to_json()

    def deserialize(self, data: str) -> CheckResult:
        parsed = json.loads(data)
        return CheckResult(
            status=CheckStatus(parsed["status"]),
            passed_count=parsed["passed_count"],
            failed_count=parsed["failed_count"],
            # ... 나머지 필드
        )


class DictSerializer:
    """딕셔너리 직렬화 (XCom 호환)"""

    def serialize(self, obj: CheckResult) -> dict[str, Any]:
        return obj.to_dict()

    def deserialize(self, data: dict[str, Any]) -> CheckResult:
        return CheckResult(
            status=CheckStatus(data["status"]),
            passed_count=data["passed_count"],
            failed_count=data["failed_count"],
            # ... 나머지 필드
        )


class SerializerFactory:
    """직렬화기 팩토리"""

    _serializers: dict[SerializationFormat, type[Serializer]] = {
        SerializationFormat.JSON: JSONSerializer,
        SerializationFormat.DICT: DictSerializer,
    }

    @classmethod
    def get_serializer(cls, format: SerializationFormat) -> Serializer:
        serializer_class = cls._serializers.get(format)
        if not serializer_class:
            raise ValueError(f"Unsupported format: {format}")
        return serializer_class()
```

---

## Data Flow

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Data Flow                                       │
└─────────────────────────────────────────────────────────────────────────────┘

1. Configuration Phase
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │   User       │───▶│  Platform    │───▶│  CheckConfig │
   │   Input      │    │  Config      │    │  (Common)    │
   │  (YAML/API)  │    │  Transform   │    │              │
   └──────────────┘    └──────────────┘    └──────────────┘

2. Execution Phase
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Data        │───▶│  Platform    │───▶│  Truthound   │
   │  Source      │    │  Adapter     │    │  Core        │
   │              │    │  (check)     │    │  (th.check)  │
   └──────────────┘    └──────────────┘    └──────────────┘

3. Result Phase
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Truthound   │───▶│  CheckResult │───▶│  Serializer  │
   │  Result      │    │  (Common)    │    │              │
   └──────────────┘    └──────────────┘    └──────────────┘

4. Platform Output Phase
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Serialized  │───▶│  Platform    │───▶│  Downstream  │
   │  Result      │    │  Output      │    │  Tasks       │
   │              │    │  (XCom/Meta) │    │              │
   └──────────────┘    └──────────────┘    └──────────────┘
```

### State Machine

```
                         ┌─────────────────────────────────────┐
                         │                                     │
                         ▼                                     │
┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────┴───────┐
│  INIT   │───▶│  VALIDATING  │───▶│   SUCCESS    │    │    RETRY     │
└─────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                      │                                        ▲
                      │                                        │
                      ▼                                        │
               ┌──────────────┐                               │
               │   FAILED     │───────────────────────────────┘
               └──────────────┘
                      │
                      ▼
               ┌──────────────┐
               │   ERROR      │
               └──────────────┘
```

---

## Design Decisions

### Why Protocol over ABC?

| Aspect | Protocol | ABC |
|--------|----------|-----|
| **Structural Typing** | ✅ Duck typing 지원 | ❌ 명시적 상속 필요 |
| **Runtime Checking** | ✅ `runtime_checkable` | ✅ 항상 가능 |
| **Third-party Extension** | ✅ 코드 수정 없이 호환 | ❌ 상속 필요 |
| **Testing** | ✅ Mock 작성 용이 | ⚠️ ABC Mock 필요 |

### Why Frozen Dataclasses?

1. **Thread Safety**: 불변 객체는 락 없이 스레드 간 공유 가능
2. **Hashability**: `frozen=True`는 자동으로 `__hash__` 제공
3. **Predictability**: 상태 변경 불가로 디버깅 용이
4. **Caching**: 결과 캐싱에 적합

### Why Tuple over List in Config?

```python
# ✅ Good: 불변
rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)

# ❌ Bad: 가변 (외부에서 수정 가능)
rules: list[dict[str, Any]] = field(default_factory=list)
```

---

## References

### Supported Engines
- [Truthound](https://github.com/seadonggyun4/Truthound) - Default data quality engine
- [Great Expectations](https://greatexpectations.io/) - Via adapter
- [Pandera](https://pandera.readthedocs.io/) - Via adapter

### Python & Design Patterns
- [Python Protocol PEP 544](https://peps.python.org/pep-0544/)

### Orchestration Platforms
- [Airflow Provider Packages](https://airflow.apache.org/docs/apache-airflow-providers/)
- [Dagster Integration Guide](https://docs.dagster.io/integrations)
- [Prefect Task Design](https://docs.prefect.io/concepts/tasks/)
- [dbt Testing](https://docs.getdbt.com/docs/build/data-tests)

---

## Pending Enterprise Features

다음 기능들은 `packages/` 레벨에서 구현이 필요합니다.

### 엔터프라이즈 엔진 어댑터

`common/` 모듈의 `DataQualityEngine` Protocol을 구현하는 벤더별 어댑터입니다.

```
packages/enterprise/engines/
├── informatica.py    # Informatica Data Quality
├── talend.py         # Talend Data Quality
├── infosphere.py     # IBM InfoSphere
└── sap.py            # SAP Data Services
```

**상태**: 미구현 (v0.5.0+ 예정)

### SLA 관리 및 알림

플랫폼별 SLA 통합과 알림 채널 시스템입니다.

```
packages/
├── airflow/sla/          # Airflow SLA 콜백 통합
├── dagster/sla/          # Dagster Freshness Policy 통합
├── prefect/sla/          # Prefect Automations 통합
└── enterprise/notifications/
    ├── base.py           # Notifier Protocol
    ├── slack.py          # Slack 알림
    ├── email.py          # Email 알림
    ├── webhook.py        # Webhook 알림
    ├── pagerduty.py      # PagerDuty 연동
    └── opsgenie.py       # Opsgenie 연동
```

**상태**: 미구현 (v0.3.0+ 예정)

**기반 인프라** (common/에 구현됨):
- Hook 시스템 - 이벤트 기반 알림 트리거
- Health Check - 상태 모니터링
- Metrics - 지연시간/오류율 측정
- Circuit Breaker - 장애 감지

### 멀티테넌트 지원

Tenant별 격리와 리소스 관리 시스템입니다.

```
packages/enterprise/multi_tenant/
├── isolation.py      # Tenant 격리 모듈
├── config.py         # Tenant 설정 관리
├── resources.py      # Tenant별 리소스 할당
└── auth.py           # Tenant 인증/인가
```

**상태**: 미구현 (v0.4.0+ 예정)

**기반 인프라** (common/에 구현됨):
- Rate Limiter - key 기반 분리
- Cache - key 기반 파티셔닝

---

*이 문서는 범용 데이터 품질 오케스트레이션 프레임워크의 핵심 아키텍처를 정의합니다.*
*DataQualityEngine Protocol을 통해 Truthound, Great Expectations, Pandera 등 다양한 엔진을 지원합니다.*
