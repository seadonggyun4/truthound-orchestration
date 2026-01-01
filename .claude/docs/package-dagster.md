# Package: truthound-dagster

> **Last Updated:** 2025-12-31
> **Document Version:** 2.0.0
> **Package Version:** 0.1.0
> **Status:** Implementation Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [DataQualityResource](#dataqualityresource)
5. [Quality Check Assets](#quality-check-assets)
6. [Dagster Ops](#dagster-ops)
7. [SLA Monitoring](#sla-monitoring)
8. [Configuration](#configuration)
9. [Usage Examples](#usage-examples)
10. [Testing Strategy](#testing-strategy)
11. [pyproject.toml](#pyprojecttoml)

---

## Overview

### Purpose
`truthound-dagster`는 Dagster용 **범용 데이터 품질** 통합 패키지입니다. `DataQualityEngine` Protocol을 통해 **Truthound, Great Expectations, Pandera 등 다양한 엔진**을 지원하며, Dagster의 Software-Defined Assets 패러다임과 완벽하게 통합됩니다.

### Key Features

| Feature | Description |
|---------|-------------|
| **Engine-Agnostic** | DataQualityEngine Protocol로 다양한 엔진 지원 |
| **ConfigurableResource** | Dagster 리소스 패턴 완벽 지원 |
| **Asset Factory** | 품질 검증 Asset 자동 생성 |
| **Metadata Integration** | Asset Materialization 메타데이터 자동 추가 |
| **Type Checking** | Dagster 타입 시스템과 통합 |
| **Partitioned Assets** | 파티션 기반 검증 지원 |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Dagster Definitions                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Asset Graph                                 │   │
│  │                                                                  │   │
│  │   ┌──────────┐      ┌──────────────────┐      ┌──────────┐     │   │
│  │   │  raw_    │─────▶│ quality_checked_ │─────▶│  marts_  │     │   │
│  │   │  data    │      │      data        │      │   data   │     │   │
│  │   └──────────┘      └────────┬─────────┘      └──────────┘     │   │
│  │                              │                                  │   │
│  │                    Uses DataQualityResource                     │   │
│  │                              │                                  │   │
│  └──────────────────────────────┼──────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                   DataQualityResource                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │   check()   │  │  profile()  │  │   learn()   │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  │                          │                                       │   │
│  │                          ▼                                       │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │          DataQualityEngine (Pluggable)                   │    │   │
│  │  │     Truthound | Great Expectations | Custom              │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### PyPI

```bash
pip install truthound-dagster
```

### With Extras

```bash
# Snowflake 지원
pip install truthound-dagster[snowflake]

# BigQuery 지원
pip install truthound-dagster[bigquery]

# 전체 설치
pip install truthound-dagster[all]
```

### Requirements

| Dependency | Version |
|------------|---------|
| Python | >= 3.11 |
| dagster | >= 1.5.0 |
| dagster-webserver | >= 1.5.0 |
| truthound | >= 1.0.0 |
| polars | >= 0.20.0 |

---

## Components

### Package Structure

```
packages/dagster/
├── pyproject.toml
├── README.md
├── src/
│   └── truthound_dagster/
│       ├── __init__.py           # Public API exports
│       ├── version.py            # Package version
│       ├── resources/
│       │   ├── __init__.py
│       │   ├── base.py           # DataQualityResource
│       │   └── engine.py         # EngineResource
│       ├── assets/
│       │   ├── __init__.py
│       │   ├── decorators.py     # quality_checked_asset, profiled_asset
│       │   └── factory.py        # Asset factories
│       ├── ops/
│       │   ├── __init__.py
│       │   ├── check.py          # data_quality_check_op
│       │   ├── profile.py        # data_quality_profile_op
│       │   └── learn.py          # data_quality_learn_op
│       ├── sla/
│       │   ├── __init__.py
│       │   ├── config.py         # SLAConfig, SLAMetrics
│       │   ├── monitor.py        # SLAMonitor, SLARegistry
│       │   ├── resource.py       # SLAResource
│       │   └── hooks.py          # SLAHook implementations
│       └── utils/
│           ├── __init__.py
│           ├── types.py          # Output types
│           ├── serialization.py  # Result serialization
│           ├── exceptions.py     # Custom exceptions
│           └── helpers.py        # Utility functions
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_resources.py
    ├── test_assets.py
    ├── test_ops.py
    └── test_sla.py
```

### Public API

```python
# truthound_dagster/__init__.py
from truthound_dagster.resources import (
    DataQualityResource,
    EngineResource,
    EngineResourceConfig,
)
from truthound_dagster.assets import (
    quality_checked_asset,
    profiled_asset,
    create_quality_asset,
    create_quality_check_asset,
    QualityAssetConfig,
)
from truthound_dagster.ops import (
    data_quality_check_op,
    data_quality_profile_op,
    data_quality_learn_op,
    create_check_op,
    CheckOpConfig,
)
from truthound_dagster.sla import (
    SLAConfig,
    SLAMonitor,
    SLAResource,
    SLAHook,
)
from truthound_dagster.utils import (
    QualityCheckOutput,
    ProfileOutput,
    DataQualityError,
    serialize_result,
)

__all__ = [
    # Resources
    "DataQualityResource",
    "EngineResource",
    "EngineResourceConfig",
    # Asset Decorators & Factories
    "quality_checked_asset",
    "profiled_asset",
    "create_quality_asset",
    "create_quality_check_asset",
    "QualityAssetConfig",
    # Ops
    "data_quality_check_op",
    "data_quality_profile_op",
    "data_quality_learn_op",
    "create_check_op",
    "CheckOpConfig",
    # SLA
    "SLAConfig",
    "SLAMonitor",
    "SLAResource",
    "SLAHook",
    # Utils
    "QualityCheckOutput",
    "ProfileOutput",
    "DataQualityError",
    "serialize_result",
]

__version__ = "0.1.0"
```

---

## DataQualityResource

### Specification

```python
from dagster import ConfigurableResource, InitResourceContext
from pydantic import Field
from typing import Any
import polars as pl


class DataQualityResource(ConfigurableResource):
    """
    Dagster용 Truthound ConfigurableResource.

    이 리소스는 Truthound의 핵심 기능을 Dagster 파이프라인에서
    사용할 수 있도록 래핑합니다. ConfigurableResource를 상속하여
    Dagster의 설정 시스템과 완벽하게 통합됩니다.

    Parameters
    ----------
    connection_string : str | None
        데이터 소스 연결 문자열.
        예: "postgresql://user:pass@host/db"

    default_timeout : int
        기본 검증 타임아웃 (초). 기본값: 300

    fail_on_error : bool
        검증 실패 시 예외 발생 여부. 기본값: True

    parallel : bool
        병렬 검증 활성화. 기본값: True

    cache_schemas : bool
        학습된 스키마 캐싱 여부. 기본값: True

    Attributes
    ----------
    _client : TruthoundClient
        내부 Truthound 클라이언트 (지연 초기화)

    Examples
    --------
    리소스 정의:

    >>> data_quality = DataQualityResource(
    ...     connection_string="postgresql://localhost/analytics",
    ...     default_timeout=600,
    ... )

    Definitions에 등록:

    >>> defs = Definitions(
    ...     assets=[my_asset],
    ...     resources={"data_quality": data_quality},
    ... )

    Asset에서 사용:

    >>> @asset
    ... def quality_checked_data(
    ...     context: AssetExecutionContext,
    ...     data_quality: DataQualityResource,
    ...     raw_data: pl.DataFrame,
    ... ) -> pl.DataFrame:
    ...     result = data_quality.check(raw_data, rules=[...])
    ...     if not result.is_success:
    ...         raise Exception("Quality check failed")
    ...     return raw_data

    Notes
    -----
    - 리소스는 스레드 안전합니다
    - 연결은 필요할 때까지 지연 생성됩니다
    - 스키마 캐싱은 메모리에 저장됩니다
    """

    connection_string: str | None = Field(
        default=None,
        description="데이터 소스 연결 문자열",
    )
    default_timeout: int = Field(
        default=300,
        description="기본 검증 타임아웃 (초)",
        ge=1,
        le=3600,
    )
    fail_on_error: bool = Field(
        default=True,
        description="검증 실패 시 예외 발생 여부",
    )
    parallel: bool = Field(
        default=True,
        description="병렬 검증 활성화",
    )
    cache_schemas: bool = Field(
        default=True,
        description="학습된 스키마 캐싱",
    )

    _client: Any = None
    _schema_cache: dict[str, Any] = {}

    def setup_for_execution(self, context: InitResourceContext) -> None:
        """
        실행 시 리소스 초기화.

        Parameters
        ----------
        context : InitResourceContext
            Dagster 초기화 컨텍스트
        """
        import truthound as th

        self._log = context.log
        self._log.info("Initializing DataQualityResource")

        if self.connection_string:
            self._client = th.connect(self.connection_string)
        else:
            self._client = th

        self._schema_cache = {}

    def check(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        rules: list[dict[str, Any]],
        *,
        fail_on_error: bool | None = None,
        timeout: int | None = None,
        sample_size: int | None = None,
    ) -> "QualityCheckOutput":
        """
        데이터 품질 검증 실행.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame
            검증할 데이터

        rules : list[dict[str, Any]]
            적용할 검증 규칙 목록.
            예: [{"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"}]

        fail_on_error : bool | None
            검증 실패 시 예외 발생. None이면 리소스 기본값 사용.

        timeout : int | None
            타임아웃 (초). None이면 리소스 기본값 사용.

        sample_size : int | None
            샘플링할 행 수. None이면 전체 검증.

        Returns
        -------
        QualityCheckOutput
            검증 결과

        Raises
        ------
        DataQualityError
            fail_on_error=True이고 검증 실패 시

        Examples
        --------
        >>> result = data_quality.check(
        ...     data=my_dataframe,
        ...     rules=[
        ...         {"column": "user_id", "type": "not_null"},
        ...         {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
        ...     ],
        ... )
        >>> if result.is_success:
        ...     print("All checks passed!")
        """
        import truthound as th

        effective_timeout = timeout or self.default_timeout
        effective_fail = fail_on_error if fail_on_error is not None else self.fail_on_error

        self._log.info(f"Running quality check with {len(rules)} rules")

        # 검증 실행
        result = self._client.check(
            data,
            rules=rules,
            fail_on_error=False,  # 자체 처리
            timeout=effective_timeout,
            parallel=self.parallel,
            sample_size=sample_size,
        )

        # 결과 래핑
        check_result = QualityCheckOutput.from_truthound(result)

        self._log.info(
            f"Quality check complete: "
            f"passed={check_result.passed_count}, "
            f"failed={check_result.failed_count}"
        )

        # 실패 처리
        if not check_result.is_success and effective_fail:
            raise DataQualityError(
                f"Quality check failed: {check_result.failed_count} rules failed",
                result=check_result,
            )

        return check_result

    def profile(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        columns: list[str] | None = None,
        include_statistics: bool = True,
        include_patterns: bool = True,
        include_distributions: bool = True,
        sample_size: int | None = None,
    ) -> "ProfileOutput":
        """
        데이터 프로파일링 실행.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame
            프로파일링할 데이터

        columns : list[str] | None
            프로파일링할 컬럼. None이면 전체.

        include_statistics : bool
            통계 포함 여부. 기본값: True

        include_patterns : bool
            패턴 감지 포함. 기본값: True

        include_distributions : bool
            분포 분석 포함. 기본값: True

        sample_size : int | None
            샘플 크기. None이면 전체.

        Returns
        -------
        ProfileOutput
            프로파일링 결과
        """
        self._log.info("Running data profiling")

        result = self._client.profile(
            data,
            columns=columns,
            include_statistics=include_statistics,
            include_patterns=include_patterns,
            include_distributions=include_distributions,
            sample_size=sample_size,
        )

        return ProfileOutput.from_truthound(result)

    def learn(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        strictness: str = "moderate",
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        """
        데이터에서 스키마 자동 학습.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame
            학습할 데이터

        strictness : str
            학습 엄격도. "strict", "moderate", "lenient"

        cache_key : str | None
            캐시 키. 설정 시 결과를 캐싱.

        Returns
        -------
        dict[str, Any]
            학습된 스키마 및 추천 규칙
        """
        # 캐시 확인
        if cache_key and cache_key in self._schema_cache:
            self._log.info(f"Using cached schema for: {cache_key}")
            return self._schema_cache[cache_key]

        self._log.info(f"Learning schema with strictness: {strictness}")

        result = self._client.learn(data, strictness=strictness)
        schema_dict = result.to_dict()

        # 캐싱
        if cache_key and self.cache_schemas:
            self._schema_cache[cache_key] = schema_dict

        return schema_dict

    def validate_with_schema(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        schema: dict[str, Any],
    ) -> "QualityCheckOutput":
        """
        저장된 스키마로 검증 실행.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame
            검증할 데이터

        schema : dict[str, Any]
            검증에 사용할 스키마

        Returns
        -------
        QualityCheckOutput
            검증 결과
        """
        rules = schema.get("rules", [])
        return self.check(data, rules)
```

### Result Types

```python
# types/results.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum


class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationFailure:
    """검증 실패 상세 정보"""
    rule_name: str
    column: str | None
    message: str
    severity: str
    failed_count: int
    total_count: int

    @property
    def failure_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.failed_count / self.total_count


@dataclass(frozen=True)
class QualityCheckOutput:
    """
    품질 검증 결과.

    Dagster 메타데이터와 호환되는 형식으로 구조화됩니다.
    """
    status: CheckStatus
    passed_count: int
    failed_count: int
    warning_count: int
    failures: tuple[ValidationFailure, ...]
    execution_time_ms: float
    timestamp: datetime

    @property
    def is_success(self) -> bool:
        return self.status == CheckStatus.PASSED

    @property
    def total_count(self) -> int:
        return self.passed_count + self.failed_count + self.warning_count

    @property
    def pass_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return self.passed_count / self.total_count

    @classmethod
    def from_truthound(cls, result: Any) -> "QualityCheckOutput":
        """Truthound 결과에서 변환"""
        failures = tuple(
            ValidationFailure(
                rule_name=f.rule_name,
                column=f.column,
                message=f.message,
                severity=f.severity.value,
                failed_count=f.failed_count,
                total_count=f.total_count,
            )
            for f in result.failures
        )

        return cls(
            status=CheckStatus(result.status.value),
            passed_count=result.passed_count,
            failed_count=result.failed_count,
            warning_count=result.warning_count,
            failures=failures,
            execution_time_ms=result.execution_time_ms,
            timestamp=result.timestamp,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Dagster 메타데이터로 변환"""
        from dagster import MetadataValue

        return {
            "quality_status": MetadataValue.text(self.status.value),
            "passed_checks": MetadataValue.int(self.passed_count),
            "failed_checks": MetadataValue.int(self.failed_count),
            "pass_rate": MetadataValue.float(self.pass_rate),
            "execution_time_ms": MetadataValue.float(self.execution_time_ms),
            "failures": MetadataValue.json([
                {
                    "rule": f.rule_name,
                    "column": f.column,
                    "message": f.message,
                }
                for f in self.failures[:10]  # 상위 10개만
            ]),
        }


@dataclass(frozen=True)
class ProfileOutput:
    """프로파일링 결과"""
    columns: dict[str, dict[str, Any]]
    row_count: int
    execution_time_ms: float
    timestamp: datetime

    @classmethod
    def from_truthound(cls, result: Any) -> "ProfileOutput":
        return cls(
            columns=result.columns,
            row_count=result.row_count,
            execution_time_ms=result.execution_time_ms,
            timestamp=result.timestamp,
        )

    def to_metadata(self) -> dict[str, Any]:
        from dagster import MetadataValue

        return {
            "row_count": MetadataValue.int(self.row_count),
            "column_count": MetadataValue.int(len(self.columns)),
            "execution_time_ms": MetadataValue.float(self.execution_time_ms),
            "profile_summary": MetadataValue.json(self._summary()),
        }

    def _summary(self) -> dict[str, Any]:
        return {
            col: {
                "dtype": info.get("dtype"),
                "null_count": info.get("null_count"),
                "unique_count": info.get("unique_count"),
            }
            for col, info in list(self.columns.items())[:10]
        }
```

---

## Quality Check Assets

### Asset Factory

```python
# assets/factory.py
from dagster import (
    asset,
    AssetExecutionContext,
    AssetIn,
    Output,
    MetadataValue,
    DagsterType,
    In,
    Out,
)
from typing import Any, Callable
import polars as pl


def create_quality_check_asset(
    name: str,
    upstream_asset: str,
    rules: list[dict[str, Any]],
    *,
    description: str | None = None,
    group_name: str | None = None,
    fail_on_error: bool = True,
    output_type: str = "validated",
    tags: dict[str, str] | None = None,
) -> Callable:
    """
    품질 검증 Asset 팩토리.

    주어진 규칙으로 upstream asset을 검증하는 새 asset을 생성합니다.

    Parameters
    ----------
    name : str
        생성할 asset 이름

    upstream_asset : str
        검증할 upstream asset 이름

    rules : list[dict[str, Any]]
        적용할 검증 규칙

    description : str | None
        Asset 설명

    group_name : str | None
        Asset 그룹

    fail_on_error : bool
        실패 시 materialization 실패 여부. 기본값: True

    output_type : str
        출력 타입. "validated" (데이터 반환), "result" (결과만 반환)

    tags : dict[str, str] | None
        Asset 태그

    Returns
    -------
    Callable
        Dagster asset 함수

    Examples
    --------
    >>> validated_users = create_quality_check_asset(
    ...     name="validated_users",
    ...     upstream_asset="raw_users",
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
    ...     ],
    ...     description="Quality validated user data",
    ...     group_name="staging",
    ... )
    """
    asset_tags = {"data_quality": "quality_check", **(tags or {})}

    @asset(
        name=name,
        ins={"upstream": AssetIn(key=upstream_asset)},
        description=description or f"Quality checked {upstream_asset}",
        group_name=group_name,
        tags=asset_tags,
    )
    def quality_check_asset(
        context: AssetExecutionContext,
        data_quality: "DataQualityResource",
        upstream: pl.DataFrame,
    ) -> Output[pl.DataFrame]:
        """Quality check asset implementation"""
        context.log.info(f"Running quality check on {upstream_asset}")
        context.log.info(f"Data shape: {upstream.shape}")

        # 검증 실행
        result = data_quality.check(
            upstream,
            rules=rules,
            fail_on_error=fail_on_error,
        )

        # 메타데이터 추가
        metadata = result.to_metadata()
        metadata["input_asset"] = MetadataValue.text(upstream_asset)
        metadata["rules_count"] = MetadataValue.int(len(rules))

        context.log.info(
            f"Quality check complete: {result.passed_count} passed, "
            f"{result.failed_count} failed"
        )

        if output_type == "validated":
            return Output(upstream, metadata=metadata)
        else:
            return Output(result.to_dict(), metadata=metadata)

    return quality_check_asset


def create_quality_profile_asset(
    name: str,
    upstream_asset: str,
    *,
    description: str | None = None,
    group_name: str | None = None,
    include_distributions: bool = True,
    tags: dict[str, str] | None = None,
) -> Callable:
    """
    데이터 프로파일링 Asset 팩토리.

    Parameters
    ----------
    name : str
        생성할 asset 이름

    upstream_asset : str
        프로파일링할 upstream asset 이름

    description : str | None
        Asset 설명

    group_name : str | None
        Asset 그룹

    include_distributions : bool
        분포 분석 포함 여부

    tags : dict[str, str] | None
        Asset 태그

    Returns
    -------
    Callable
        Dagster asset 함수
    """
    asset_tags = {"data_quality": "profile", **(tags or {})}

    @asset(
        name=name,
        ins={"upstream": AssetIn(key=upstream_asset)},
        description=description or f"Profile of {upstream_asset}",
        group_name=group_name,
        tags=asset_tags,
    )
    def profile_asset(
        context: AssetExecutionContext,
        data_quality: "DataQualityResource",
        upstream: pl.DataFrame,
    ) -> Output[dict]:
        """Profile asset implementation"""
        context.log.info(f"Profiling {upstream_asset}")

        result = data_quality.profile(
            upstream,
            include_distributions=include_distributions,
        )

        return Output(
            result.columns,
            metadata=result.to_metadata(),
        )

    return profile_asset


def build_quality_assets(
    assets_config: list[dict[str, Any]],
    *,
    default_group: str | None = None,
) -> list[Callable]:
    """
    설정에서 여러 품질 검증 asset 일괄 생성.

    Parameters
    ----------
    assets_config : list[dict[str, Any]]
        Asset 설정 목록

    default_group : str | None
        기본 그룹 이름

    Returns
    -------
    list[Callable]
        생성된 asset 목록

    Examples
    --------
    >>> assets = build_quality_assets([
    ...     {
    ...         "name": "validated_users",
    ...         "upstream": "raw_users",
    ...         "rules": [{"column": "id", "type": "not_null"}],
    ...     },
    ...     {
    ...         "name": "validated_orders",
    ...         "upstream": "raw_orders",
    ...         "rules": [{"column": "order_id", "type": "unique"}],
    ...     },
    ... ])
    """
    assets = []

    for config in assets_config:
        asset_fn = create_quality_check_asset(
            name=config["name"],
            upstream_asset=config["upstream"],
            rules=config["rules"],
            description=config.get("description"),
            group_name=config.get("group", default_group),
            fail_on_error=config.get("fail_on_error", True),
            tags=config.get("tags"),
        )
        assets.append(asset_fn)

    return assets
```

### Usage Examples

```python
# definitions.py
from dagster import Definitions, asset
from truthound_dagster import (
    DataQualityResource,
    create_quality_check_asset,
    build_quality_assets,
)
import polars as pl


# Raw data asset
@asset(group_name="raw")
def raw_users() -> pl.DataFrame:
    """Load raw user data"""
    return pl.read_parquet("s3://bucket/users.parquet")


@asset(group_name="raw")
def raw_orders() -> pl.DataFrame:
    """Load raw order data"""
    return pl.read_parquet("s3://bucket/orders.parquet")


# 개별 품질 검증 asset
validated_users = create_quality_check_asset(
    name="validated_users",
    upstream_asset="raw_users",
    rules=[
        {"column": "user_id", "type": "not_null"},
        {"column": "user_id", "type": "unique"},
        {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
        {"column": "age", "type": "in_range", "min": 0, "max": 150},
    ],
    group_name="staging",
    fail_on_error=True,
)


# 일괄 생성
quality_assets = build_quality_assets(
    [
        {
            "name": "validated_orders",
            "upstream": "raw_orders",
            "rules": [
                {"column": "order_id", "type": "not_null"},
                {"column": "order_id", "type": "regex", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"},
                {"column": "amount", "type": "in_range", "min": 0},
            ],
        },
    ],
    default_group="staging",
)


# Definitions 등록
defs = Definitions(
    assets=[
        raw_users,
        raw_orders,
        validated_users,
        *quality_assets,
    ],
    resources={
        "data_quality": DataQualityResource(
            connection_string=None,
            default_timeout=300,
        ),
    },
)
```

---

## Dagster Ops

### Check Op

```python
# ops/check.py
from dagster import op, In, Out, Output, OpExecutionContext
from typing import Any
import polars as pl


@op(
    ins={
        "data": In(dagster_type=pl.DataFrame),
        "rules": In(dagster_type=list),
    },
    out=Out(dagster_type=dict),
    required_resource_keys={"data_quality"},
    tags={"kind": "quality_check"},
)
def data_quality_check_op(
    context: OpExecutionContext,
    data: pl.DataFrame,
    rules: list[dict[str, Any]],
) -> Output[dict]:
    """
    데이터 품질 검증 Op.

    Parameters
    ----------
    context : OpExecutionContext
        Dagster 실행 컨텍스트

    data : pl.DataFrame
        검증할 데이터

    rules : list[dict[str, Any]]
        검증 규칙

    Returns
    -------
    Output[dict]
        검증 결과 딕셔너리
    """
    data_quality = context.resources.data_quality

    result = data_quality.check(data, rules, fail_on_error=False)

    return Output(
        result.to_dict(),
        metadata=result.to_metadata(),
    )


@op(
    ins={
        "data": In(dagster_type=pl.DataFrame),
        "rules": In(dagster_type=list),
    },
    out={
        "validated_data": Out(dagster_type=pl.DataFrame, is_required=True),
        "check_result": Out(dagster_type=dict, is_required=False),
    },
    required_resource_keys={"data_quality"},
)
def data_quality_validate_op(
    context: OpExecutionContext,
    data: pl.DataFrame,
    rules: list[dict[str, Any]],
) -> tuple[pl.DataFrame, dict]:
    """
    데이터 검증 후 데이터와 결과 모두 반환.

    실패 시 예외 발생.
    """
    data_quality = context.resources.data_quality

    result = data_quality.check(data, rules, fail_on_error=True)

    return (
        Output(data, output_name="validated_data", metadata=result.to_metadata()),
        Output(result.to_dict(), output_name="check_result"),
    )
```

### Profile Op

```python
# ops/profile.py
from dagster import op, In, Out, Output, OpExecutionContext
import polars as pl


@op(
    ins={"data": In(dagster_type=pl.DataFrame)},
    out=Out(dagster_type=dict),
    required_resource_keys={"data_quality"},
    tags={"kind": "profiling"},
)
def data_quality_profile_op(
    context: OpExecutionContext,
    data: pl.DataFrame,
) -> Output[dict]:
    """
    데이터 프로파일링 Op.

    Parameters
    ----------
    context : OpExecutionContext
        Dagster 실행 컨텍스트

    data : pl.DataFrame
        프로파일링할 데이터

    Returns
    -------
    Output[dict]
        프로파일링 결과
    """
    data_quality = context.resources.data_quality

    result = data_quality.profile(data)

    return Output(
        result.to_dict(),
        metadata=result.to_metadata(),
    )
```

### Learn Op

```python
# ops/learn.py
from dagster import op, In, Out, Output, OpExecutionContext, Config
from pydantic import Field
import polars as pl


class LearnConfig(Config):
    strictness: str = Field(
        default="moderate",
        description="학습 엄격도: strict, moderate, lenient",
    )
    cache_key: str | None = Field(
        default=None,
        description="스키마 캐시 키",
    )


@op(
    ins={"data": In(dagster_type=pl.DataFrame)},
    out=Out(dagster_type=dict),
    required_resource_keys={"data_quality"},
    tags={"kind": "schema_learning"},
)
def data_quality_learn_op(
    context: OpExecutionContext,
    config: LearnConfig,
    data: pl.DataFrame,
) -> Output[dict]:
    """
    데이터에서 스키마 자동 학습 Op.
    """
    data_quality = context.resources.data_quality

    schema = data_quality.learn(
        data,
        strictness=config.strictness,
        cache_key=config.cache_key,
    )

    return Output(
        schema,
        metadata={
            "rules_count": len(schema.get("rules", [])),
            "columns_count": len(schema.get("columns", {})),
        },
    )
```

---

## SLA Monitoring

SLA (Service Level Agreement) 모니터링을 위한 모듈입니다. 데이터 품질 작업의 실패율, 통과율, 실행 시간 등을 추적하고 위반을 감지합니다.

### SLAConfig

SLA 임계값을 정의하는 불변 설정 클래스입니다.

```python
# sla/config.py
from truthound_dagster.sla import SLAConfig, AlertLevel

# 기본 SLA 설정
config = SLAConfig(
    max_failure_rate=0.05,           # 최대 실패율 5%
    min_pass_rate=0.95,              # 최소 통과율 95%
    max_execution_time_seconds=300.0, # 최대 실행 시간 5분
    min_row_count=1000,              # 최소 행 수
    max_row_count=10_000_000,        # 최대 행 수
    max_consecutive_failures=3,      # 최대 연속 실패 횟수
    alert_on_warning=True,           # 경고 시 알림
    alert_level=AlertLevel.ERROR,    # 기본 알림 레벨
    enabled=True,                    # SLA 모니터링 활성화
)

# 빌더 패턴으로 설정 수정
strict_config = config.with_failure_rate(0.01).with_pass_rate(0.99)
```

#### 프리셋 설정

```python
from truthound_dagster.sla import (
    DEFAULT_SLA_CONFIG,     # 기본 설정
    STRICT_SLA_CONFIG,      # 엄격: 1% 실패율, 99% 통과율
    LENIENT_SLA_CONFIG,     # 관대: 10% 실패율, 90% 통과율
)
```

### SLAMetrics

SLA 평가를 위한 메트릭 데이터 컨테이너입니다.

```python
from truthound_dagster.sla import SLAMetrics

# 메트릭 생성
metrics = SLAMetrics(
    passed_count=95,
    failed_count=5,
    warning_count=2,
    execution_time_ms=1500.0,
    row_count=10000,
    asset_key="users",
    run_id="abc-123",
)

# 자동 계산 속성
print(f"Pass rate: {metrics.pass_rate:.2%}")      # 95.00%
print(f"Failure rate: {metrics.failure_rate:.2%}")  # 5.00%
print(f"Execution time: {metrics.execution_time_seconds}s")  # 1.5s

# CheckResult에서 생성
metrics = SLAMetrics.from_check_result(
    result=check_result.to_dict(),
    asset_key="users",
    run_id=context.run_id,
)
```

### SLAMonitor

SLA 위반을 감지하는 모니터 클래스입니다.

```python
from truthound_dagster.sla import SLAMonitor, SLAConfig, SLAMetrics

# 모니터 생성
monitor = SLAMonitor(
    config=SLAConfig(max_failure_rate=0.05),
    name="users_quality",
)

# 메트릭 검사
violations = monitor.check(metrics)

for v in violations:
    print(f"[{v.alert_level.value}] {v.message}")
    print(f"  Type: {v.violation_type.value}")
    print(f"  Threshold: {v.threshold}, Actual: {v.actual}")

# 연속 실패 추적
monitor.record_failure()
monitor.record_failure()
monitor.record_failure()  # 임계값 도달시 위반 생성

# 상태 요약
summary = monitor.get_summary()
print(f"Consecutive failures: {summary['consecutive_failures']}")
print(f"Average pass rate: {summary['average_pass_rate']:.2%}")
```

### SLARegistry

여러 SLA 모니터를 중앙 관리하는 레지스트리입니다.

```python
from truthound_dagster.sla import SLARegistry, SLAConfig, get_sla_registry

# 레지스트리 생성 또는 전역 레지스트리 사용
registry = SLARegistry()
# 또는
registry = get_sla_registry()

# 모니터 등록
registry.register(
    "users_check",
    SLAConfig(max_failure_rate=0.05, min_pass_rate=0.95),
)

registry.register(
    "orders_check",
    SLAConfig(max_failure_rate=0.01),  # 더 엄격한 설정
)

# 모니터 조회
monitor = registry.get("users_check")
violations = monitor.check(metrics)

# 전체 검사
all_violations = registry.check_all({
    "users_check": users_metrics,
    "orders_check": orders_metrics,
})

# 전체 요약
summaries = registry.get_summary_all()
```

### SLAResource

Dagster 리소스로 SLA 모니터링을 제공합니다.

```python
from dagster import Definitions, asset, AssetExecutionContext
from truthound_dagster.sla import SLAResource, SLAMetrics

@asset
def users(context: AssetExecutionContext, sla: SLAResource):
    """SLA 모니터링이 포함된 asset"""
    data = load_users()

    # 품질 검사 실행
    result = run_quality_check(data)

    # SLA 검사
    violations = sla.check_result(
        result=result.to_dict(),
        name="users_quality",
        asset_key="users",
        run_id=context.run_id,
    )

    if violations:
        for v in violations:
            context.log.warning(f"SLA Violation: {v.message}")

    return data

# Definitions에 리소스 등록
defs = Definitions(
    assets=[users],
    resources={
        "sla": SLAResource(
            max_failure_rate=0.05,
            min_pass_rate=0.95,
            max_execution_time_seconds=300.0,
            alert_level="error",
        ),
    },
)
```

### SLA Hooks

SLA 이벤트를 처리하는 훅 시스템입니다.

```python
from truthound_dagster.sla import (
    SLAHook,
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
)

# 로깅 훅: 이벤트 자동 로깅
logging_hook = LoggingSLAHook()

# 메트릭 훅: 통계 수집
metrics_hook = MetricsSLAHook()

# 복합 훅: 여러 훅 조합
composite = CompositeSLAHook([logging_hook, metrics_hook])

# 메트릭 조회
stats = metrics_hook.get_stats()
print(f"Total checks: {stats['check_count']}")
print(f"Violations: {stats['violation_count']}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

#### 커스텀 훅 구현

```python
from truthound_dagster.sla import SLAHook, SLAMetrics, SLAViolation

class SlackAlertHook(SLAHook):
    """SLA 위반 시 Slack 알림 전송"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def on_check(
        self,
        metrics: SLAMetrics,
        violations: list[SLAViolation],
        context: dict | None = None,
    ) -> None:
        if violations:
            self._send_slack_alert(violations)

    def on_violation(
        self,
        violation: SLAViolation,
        context: dict | None = None,
    ) -> None:
        self._send_slack_alert([violation])

    def _send_slack_alert(self, violations: list[SLAViolation]) -> None:
        # Slack webhook 호출
        ...
```

### SLA Violation Types

지원하는 SLA 위반 유형입니다.

| Type | Description |
|------|-------------|
| `FAILURE_RATE_EXCEEDED` | 실패율이 임계값 초과 |
| `PASS_RATE_BELOW_MINIMUM` | 통과율이 최소값 미만 |
| `EXECUTION_TIME_EXCEEDED` | 실행 시간이 제한 초과 |
| `ROW_COUNT_BELOW_MINIMUM` | 행 수가 최소값 미만 |
| `ROW_COUNT_ABOVE_MAXIMUM` | 행 수가 최대값 초과 |
| `CONSECUTIVE_FAILURES` | 연속 실패 횟수가 임계값 도달 |
| `CUSTOM` | 커스텀 위반 유형 |

### 전체 사용 예시

```python
from dagster import Definitions, asset, AssetExecutionContext
from truthound_dagster import DataQualityResource
from truthound_dagster.sla import (
    SLAResource,
    SLAConfig,
    SLAMetrics,
    LoggingSLAHook,
    MetricsSLAHook,
    CompositeSLAHook,
)


@asset
def validated_users(
    context: AssetExecutionContext,
    data_quality: DataQualityResource,
    sla: SLAResource,
):
    """품질 검증 및 SLA 모니터링이 포함된 asset"""
    import time

    start_time = time.time()
    data = load_users()

    # 품질 검사 실행
    result = data_quality.check(
        data,
        rules=[
            {"column": "id", "type": "not_null"},
            {"column": "email", "type": "unique"},
        ],
    )

    execution_time_ms = (time.time() - start_time) * 1000

    # SLA 메트릭 생성
    metrics = SLAMetrics(
        passed_count=result.passed_count,
        failed_count=result.failed_count,
        execution_time_ms=execution_time_ms,
        row_count=len(data),
        asset_key="validated_users",
        run_id=context.run_id,
    )

    # SLA 검사
    violations = sla.check(
        metrics,
        name="users_sla",
        config=SLAConfig(
            max_failure_rate=0.05,
            max_execution_time_seconds=60.0,
        ),
    )

    # 메타데이터에 SLA 정보 추가
    context.add_output_metadata({
        "sla_pass_rate": metrics.pass_rate,
        "sla_execution_time_ms": execution_time_ms,
        "sla_violations": len(violations),
    })

    if violations:
        for v in violations:
            context.log.warning(f"SLA Violation: {v.message}")

    return data


defs = Definitions(
    assets=[validated_users],
    resources={
        "data_quality": DataQualityResource(),
        "sla": SLAResource(
            max_failure_rate=0.05,
            min_pass_rate=0.95,
        ),
    },
)
```

---

## Configuration

### dagster.yaml

```yaml
# dagster.yaml
telemetry:
  enabled: false

run_queue:
  max_concurrent_runs: 10

storage:
  postgres:
    postgres_db:
      hostname: localhost
      username: dagster
      password: ${DAGSTER_PG_PASSWORD}
      db_name: dagster

run_storage:
  module: dagster_postgres.run_storage
  class: PostgresRunStorage
  config:
    postgres_db:
      hostname: localhost
      username: dagster
      password: ${DAGSTER_PG_PASSWORD}
      db_name: dagster

event_log_storage:
  module: dagster_postgres.event_log
  class: PostgresEventLogStorage
  config:
    postgres_db:
      hostname: localhost
      username: dagster
      password: ${DAGSTER_PG_PASSWORD}
      db_name: dagster

schedule_storage:
  module: dagster_postgres.schedule_storage
  class: PostgresScheduleStorage
  config:
    postgres_db:
      hostname: localhost
      username: dagster
      password: ${DAGSTER_PG_PASSWORD}
      db_name: dagster
```

### Environment-Based Configuration

```python
# resources.py
import os
from dagster import EnvVar
from truthound_dagster import DataQualityResource


def get_data_quality_resource() -> DataQualityResource:
    """환경 기반 리소스 설정"""
    env = os.getenv("DAGSTER_ENV", "dev")

    if env == "prod":
        return DataQualityResource(
            connection_string=EnvVar("DATA_QUALITY_CONNECTION_STRING"),
            default_timeout=600,
            fail_on_error=True,
        )
    else:
        return DataQualityResource(
            connection_string=None,
            default_timeout=120,
            fail_on_error=False,
        )
```

---

## Usage Examples

### Complete Pipeline

```python
# definitions.py
from dagster import (
    Definitions,
    asset,
    AssetExecutionContext,
    ScheduleDefinition,
    define_asset_job,
)
from truthound_dagster import (
    DataQualityResource,
    create_quality_check_asset,
    quality_sensor,
)
import polars as pl


# 1. Raw data assets
@asset(group_name="raw")
def raw_customers() -> pl.DataFrame:
    """Extract customer data"""
    return pl.read_parquet("s3://data-lake/raw/customers/")


@asset(group_name="raw")
def raw_orders() -> pl.DataFrame:
    """Extract order data"""
    return pl.read_parquet("s3://data-lake/raw/orders/")


# 2. Quality validated assets
validated_customers = create_quality_check_asset(
    name="validated_customers",
    upstream_asset="raw_customers",
    rules=[
        {"column": "customer_id", "type": "not_null"},
        {"column": "customer_id", "type": "unique"},
        {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
        {"column": "status", "type": "in_set", "values": ["active", "inactive"]},
    ],
    group_name="staging",
)


validated_orders = create_quality_check_asset(
    name="validated_orders",
    upstream_asset="raw_orders",
    rules=[
        {"column": "order_id", "type": "not_null"},
        {"column": "order_id", "type": "regex", "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"},
        {"column": "customer_id", "type": "not_null"},
        {"column": "amount", "type": "in_range", "min": 0},
        {"column": "created_at", "type": "not_future"},
    ],
    group_name="staging",
)


# 3. Marts (downstream of validated data)
@asset(
    group_name="marts",
    deps=["validated_customers", "validated_orders"],
)
def customer_orders_summary(
    context: AssetExecutionContext,
    validated_customers: pl.DataFrame,
    validated_orders: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregated customer orders"""
    return (
        validated_orders
        .group_by("customer_id")
        .agg([
            pl.count("order_id").alias("order_count"),
            pl.sum("amount").alias("total_amount"),
        ])
        .join(validated_customers, on="customer_id", how="left")
    )


# 4. Jobs
quality_check_job = define_asset_job(
    name="quality_check_job",
    selection=["validated_*"],
    description="Run all quality checks",
)

full_pipeline_job = define_asset_job(
    name="full_pipeline_job",
    selection="*",
    description="Run full data pipeline",
)


# 5. Schedules
daily_pipeline_schedule = ScheduleDefinition(
    name="daily_pipeline",
    job=full_pipeline_job,
    cron_schedule="0 6 * * *",  # 매일 06:00
)


# 6. Definitions
defs = Definitions(
    assets=[
        raw_customers,
        raw_orders,
        validated_customers,
        validated_orders,
        customer_orders_summary,
    ],
    jobs=[quality_check_job, full_pipeline_job],
    schedules=[daily_pipeline_schedule],
    sensors=[quality_sensor],
    resources={
        "data_quality": DataQualityResource(
            default_timeout=300,
            fail_on_error=True,
        ),
    },
)
```

---

## Testing Strategy

### Test Structure

```
tests/
├── __init__.py
├── conftest.py
├── test_resources/
│   ├── __init__.py
│   └── test_data_quality_resource.py
├── test_assets/
│   ├── __init__.py
│   └── test_factory.py
├── test_ops/
│   ├── __init__.py
│   ├── test_check_op.py
│   └── test_profile_op.py
└── test_sensors/
    ├── __init__.py
    └── test_quality_sensor.py
```

### Fixtures

```python
# conftest.py
import pytest
from unittest.mock import MagicMock, patch
import polars as pl
from dagster import build_op_context, build_asset_context


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    return pl.DataFrame({
        "user_id": ["uuid1", "uuid2", "uuid3"],
        "email": ["a@b.com", "c@d.com", "e@f.com"],
        "age": [25, 30, 45],
    })


@pytest.fixture
def mock_data_quality_resource():
    with patch("truthound_dagster.resources.data_quality.th") as mock:
        mock_result = MagicMock()
        mock_result.status.value = "passed"
        mock_result.is_success = True
        mock_result.passed_count = 3
        mock_result.failed_count = 0
        mock_result.failures = []

        resource = MagicMock()
        resource.check.return_value = mock_result

        yield resource


@pytest.fixture
def op_context(mock_data_quality_resource):
    return build_op_context(
        resources={"data_quality": mock_data_quality_resource},
    )
```

### Unit Tests

```python
# test_resources/test_data_quality_resource.py
import pytest
from truthound_dagster import DataQualityResource


class TestDataQualityResource:

    def test_init_defaults(self):
        resource = DataQualityResource()

        assert resource.connection_string is None
        assert resource.default_timeout == 300
        assert resource.fail_on_error is True

    def test_init_with_config(self):
        resource = DataQualityResource(
            connection_string="postgresql://localhost/db",
            default_timeout=600,
            fail_on_error=False,
        )

        assert resource.connection_string == "postgresql://localhost/db"
        assert resource.default_timeout == 600
        assert resource.fail_on_error is False

    def test_check_success(self, sample_dataframe, mock_data_quality_resource):
        rules = [{"column": "user_id", "type": "not_null"}]

        result = mock_data_quality_resource.check(sample_dataframe, rules)

        assert result.is_success
        mock_data_quality_resource.check.assert_called_once()

    def test_check_failure_raises(self, sample_dataframe):
        # 실패 시나리오 테스트
        pass
```

### Asset Tests

```python
# test_assets/test_factory.py
import pytest
from dagster import materialize
from truthound_dagster import create_quality_check_asset, DataQualityResource


class TestQualityCheckAsset:

    def test_create_asset(self):
        asset_fn = create_quality_check_asset(
            name="test_validated",
            upstream_asset="test_raw",
            rules=[{"column": "id", "type": "not_null"}],
        )

        assert asset_fn is not None
        assert asset_fn.__name__ == "quality_check_asset"

    def test_asset_execution(self, sample_dataframe, mock_data_quality_resource):
        # Asset 실행 테스트
        pass
```

---

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "truthound-dagster"
version = "0.1.0"
description = "Dagster integration for Truthound data quality framework"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    { name = "Truthound Team", email = "team@truthound.dev" }
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Framework :: Dagster",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
keywords = ["dagster", "data-quality", "truthound", "validation"]

dependencies = [
    "dagster>=1.5.0",
    "truthound>=1.0.0",
    "polars>=0.20.0",
]

[project.optional-dependencies]
snowflake = ["dagster-snowflake>=0.21.0"]
bigquery = ["dagster-gcp>=0.21.0"]
postgres = ["dagster-postgres>=0.21.0"]
all = ["truthound-dagster[snowflake,bigquery,postgres]"]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "dagster-webserver>=1.5.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/seadonggyun4/truthound-integrations"
Documentation = "https://truthound.dev/docs/integrations/dagster"
Repository = "https://github.com/seadonggyun4/truthound-integrations"

[tool.hatch.build.targets.wheel]
packages = ["src/truthound_dagster"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true

[[tool.mypy.overrides]]
module = "dagster.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## References

- [Dagster Software-Defined Assets](https://docs.dagster.io/concepts/assets/software-defined-assets)
- [Dagster Resources](https://docs.dagster.io/concepts/resources)
- [Dagster Testing](https://docs.dagster.io/concepts/testing)
- [Truthound Documentation](https://truthound.dev/docs)

---

*이 문서는 truthound-dagster 패키지의 완전한 구현 명세입니다.*
