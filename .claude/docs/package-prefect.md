# Package: truthound-prefect

> **Last Updated:** 2025-12-31
> **Document Version:** 2.0.0
> **Package Version:** 0.1.0
> **Status:** Implementation Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [DataQualityBlock](#dataqualityblock)
5. [Tasks](#tasks)
6. [Flows](#flows)
7. [Artifacts](#artifacts)
8. [Deployment Configuration](#deployment-configuration)
9. [Usage Examples](#usage-examples)
10. [Testing Strategy](#testing-strategy)
11. [pyproject.toml](#pyprojecttoml)

---

## Overview

### Purpose
`truthound-prefect`는 Prefect용 **범용 데이터 품질** 통합 패키지입니다. `DataQualityEngine` Protocol을 통해 **Truthound, Great Expectations, Pandera 등 다양한 엔진**을 지원하며, Prefect의 함수형 워크플로우 패러다임과 완벽하게 통합됩니다.

### Key Features

| Feature | Description |
|---------|-------------|
| **Engine-Agnostic** | DataQualityEngine Protocol로 다양한 엔진 지원 |
| **Prefect Blocks** | 재사용 가능한 설정 저장 |
| **Native Tasks** | `@task` 데코레이터 기반 검증 |
| **Flow Templates** | 품질 파이프라인 템플릿 제공 |
| **Artifacts** | 검증 결과를 Prefect Artifacts로 저장 |
| **Retries & Caching** | Prefect의 재시도/캐싱 완벽 지원 |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Prefect Flow                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    @flow: data_pipeline                          │   │
│  │                                                                  │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │   │
│  │   │ @task        │───▶│ @task        │───▶│ @task        │     │   │
│  │   │ extract_data │    │ data_quality │    │ load_data    │     │   │
│  │   │              │    │ _check       │    │              │     │   │
│  │   └──────────────┘    └──────┬───────┘    └──────────────┘     │   │
│  │                              │                                  │   │
│  │                              ▼                                  │   │
│  │                    ┌──────────────────┐                        │   │
│  │                    │ DataQualityBlock │                        │   │
│  │                    │ (Prefect Block)  │                        │   │
│  │                    └────────┬─────────┘                        │   │
│  │                             │                                   │   │
│  │                             ▼                                   │   │
│  │            ┌────────────────────────────────────┐              │   │
│  │            │   DataQualityEngine (Pluggable)    │              │   │
│  │            │  Truthound | Great Expectations    │              │   │
│  │            └────────────────────────────────────┘              │   │
│  │                              │                                  │   │
│  └──────────────────────────────┼──────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│                    ┌──────────────────────┐                            │
│                    │   Prefect Artifacts  │                            │
│                    │  (Quality Reports)   │                            │
│                    └──────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### PyPI

```bash
pip install truthound-prefect
```

### With Extras

```bash
# S3 지원
pip install truthound-prefect[aws]

# GCS 지원
pip install truthound-prefect[gcp]

# 전체 설치
pip install truthound-prefect[all]
```

### Requirements

| Dependency | Version |
|------------|---------|
| Python | >= 3.11 |
| prefect | >= 2.14.0 |
| truthound | >= 1.0.0 |
| polars | >= 0.20.0 |

---

## Components

### Package Structure

```
packages/prefect/
├── pyproject.toml
├── README.md
├── src/
│   └── truthound_prefect/
│       ├── __init__.py           # Public API exports
│       ├── version.py            # Package version
│       ├── blocks/
│       │   ├── __init__.py
│       │   └── data_quality.py   # DataQualityBlock
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── check.py          # data_quality_check_task
│       │   ├── profile.py        # data_quality_profile_task
│       │   └── learn.py          # data_quality_learn_task
│       ├── flows/
│       │   ├── __init__.py
│       │   └── templates.py      # Flow templates
│       ├── artifacts/
│       │   ├── __init__.py
│       │   └── quality.py        # Quality artifacts
│       └── utils/
│           ├── __init__.py
│           └── serialization.py  # Result serialization
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_blocks/
    ├── test_tasks/
    └── test_flows/
```

### Public API

```python
# truthound_prefect/__init__.py
from truthound_prefect.blocks import DataQualityBlock
from truthound_prefect.tasks import (
    data_quality_check_task,
    data_quality_profile_task,
    data_quality_learn_task,
)
from truthound_prefect.flows import (
    quality_checked_flow,
    profiled_flow,
    validated_flow,
    create_quality_flow,
    create_validation_flow,
    create_pipeline_flow,
)
from truthound_prefect.sla import (
    SLABlock,
    SLAConfig,
    SLAMonitor,
    SLAViolation,
)

__all__ = [
    # Blocks
    "DataQualityBlock",
    "SLABlock",
    # Tasks
    "data_quality_check_task",
    "data_quality_profile_task",
    "data_quality_learn_task",
    # Flow Decorators
    "quality_checked_flow",
    "profiled_flow",
    "validated_flow",
    # Flow Factories
    "create_quality_flow",
    "create_validation_flow",
    "create_pipeline_flow",
    # SLA
    "SLAConfig",
    "SLAMonitor",
    "SLAViolation",
]

__version__ = "0.1.0"
```

---

## DataQualityBlock

### Specification

```python
from prefect.blocks.core import Block
from pydantic import Field
from typing import Any
import polars as pl


class DataQualityBlock(Block):
    """
    데이터 품질 설정을 저장하는 Prefect Block.

    이 Block은 DataQualityEngine 연결 정보와 기본 설정을 저장하여
    여러 Flow에서 재사용할 수 있습니다. Prefect UI에서 관리하거나
    코드로 생성할 수 있습니다.

    Parameters
    ----------
    engine_name : str
        사용할 엔진 이름. 기본값: "truthound"

    auto_schema : bool
        자동 스키마 생성 여부. 기본값: True

    fail_on_error : bool
        검증 실패 시 예외 발생 여부. 기본값: True

    timeout_seconds : float
        기본 검증 타임아웃 (초). 기본값: 300.0

    default_rules : list[dict[str, Any]]
        기본 검증 규칙 목록

    Attributes
    ----------
    _block_type_name : str
        Block 타입 이름

    _block_type_slug : str
        Block 타입 slug

    Examples
    --------
    Block 생성 및 저장:

    >>> block = DataQualityBlock(
    ...     engine_name="truthound",
    ...     auto_schema=True,
    ...     default_rules=[
    ...         {"column": "id", "type": "not_null"},
    ...     ],
    ... )
    >>> await block.save("production-quality")

    Block 로드 및 사용:

    >>> block = await DataQualityBlock.load("production-quality")
    >>> result = await block.check(data, rules=[...])

    CLI로 Block 등록:

    ```bash
    prefect block register -m truthound_prefect.blocks
    ```

    Notes
    -----
    - Block은 Prefect Server/Cloud에 저장됩니다
    - 동일한 Block을 여러 Flow에서 공유할 수 있습니다
    - engine_name으로 Truthound, Great Expectations, Pandera 선택 가능
    """

    _block_type_name = "DataQuality"
    _block_type_slug = "data-quality"
    _documentation_url = "https://truthound.dev/docs/integrations/prefect"

    engine_name: str = Field(
        default="truthound",
        description="사용할 엔진 이름 (truthound, great_expectations, pandera)",
    )
    auto_schema: bool = Field(
        default=True,
        description="자동 스키마 생성 여부",
    )
    fail_on_error: bool = Field(
        default=True,
        description="검증 실패 시 예외 발생 여부",
    )
    timeout_seconds: float = Field(
        default=300.0,
        description="기본 검증 타임아웃 (초)",
        ge=1.0,
        le=3600.0,
    )
    default_rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="기본 검증 규칙 목록",
    )

    def get_engine(self) -> "DataQualityEngine":
        """
        DataQualityEngine 인스턴스 반환.

        Returns
        -------
        DataQualityEngine
            데이터 품질 엔진 인스턴스
        """
        from common.engines import get_engine

        return get_engine(self.engine_name)

    async def check(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        rules: list[dict[str, Any]] | None = None,
        *,
        fail_on_error: bool | None = None,
        auto_schema: bool | None = None,
    ) -> "QualityCheckOutput":
        """
        데이터 품질 검증 실행.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame
            검증할 데이터

        rules : list[dict[str, Any]] | None
            검증 규칙. None이면 default_rules 사용.

        fail_on_error : bool | None
            실패 시 예외 발생. None이면 Block 기본값.

        auto_schema : bool | None
            자동 스키마. None이면 Block 기본값.

        Returns
        -------
        QualityCheckOutput
            검증 결과

        Raises
        ------
        DataQualityError
            검증 실패 및 fail_on_error=True인 경우
        """
        from prefect import get_run_logger

        logger = get_run_logger()

        effective_rules = rules or self.default_rules
        effective_fail = fail_on_error if fail_on_error is not None else self.fail_on_error
        effective_auto_schema = auto_schema if auto_schema is not None else self.auto_schema

        logger.info(f"Running quality check with engine: {self.engine_name}")

        engine = self.get_engine()
        result = engine.check(
            data,
            rules=effective_rules,
            auto_schema=effective_auto_schema,
        )

        logger.info(
            f"Quality check complete: "
            f"passed={result.passed_count}, failed={result.failed_count}"
        )

        if result.status.name == "FAILED" and effective_fail:
            from truthound_prefect.utils import DataQualityError
            raise DataQualityError(
                f"Quality check failed: {result.failed_count} rules failed",
                result=result,
            )

        return result

    async def profile(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        columns: list[str] | None = None,
    ) -> "ProfileOutput":
        """
        데이터 프로파일링 실행.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame
            프로파일링할 데이터

        columns : list[str] | None
            프로파일링할 컬럼

        Returns
        -------
        ProfileOutput
            프로파일링 결과
        """
        from prefect import get_run_logger

        logger = get_run_logger()
        logger.info("Running data profiling")

        engine = self.get_engine()
        result = engine.profile(data, columns=columns)

        return result

    async def learn(
        self,
        data: pl.DataFrame | pl.LazyFrame,
    ) -> "LearnOutput":
        """
        데이터에서 스키마 자동 학습.

        Parameters
        ----------
        data : pl.DataFrame | pl.LazyFrame
            학습할 데이터

        Returns
        -------
        LearnOutput
            학습된 스키마 및 규칙
        """
        from prefect import get_run_logger

        logger = get_run_logger()
        logger.info("Learning schema from data")

        engine = self.get_engine()
        result = engine.learn(data)

        return result
```

### Block Registration

```bash
# Block 타입 등록
prefect block register -m truthound_prefect.blocks

# Block 인스턴스 생성 (CLI)
prefect block create data-quality/production --engine-name "truthound"
```

---

## Tasks

### data_quality_check_task

```python
# tasks/check.py
from prefect import task, get_run_logger
from typing import Any
import polars as pl


@task(
    name="data_quality_check",
    description="데이터 품질 검증",
    tags=["data-quality", "check"],
    retries=2,
    retry_delay_seconds=10,
)
async def data_quality_check_task(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    block: "DataQualityBlock | None" = None,
    rules: list[dict[str, Any]] | None = None,
    auto_schema: bool = True,
    fail_on_error: bool = True,
    create_artifact: bool = True,
) -> "QualityCheckOutput":
    """
    데이터 품질 검증 Task.

    이 Task는 DataQualityEngine을 사용하여 데이터 품질을 검증합니다.
    Block을 사용하거나 직접 설정을 전달할 수 있습니다.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        검증할 데이터

    block : DataQualityBlock | None
        사용할 DataQualityBlock. None이면 기본 설정 사용.

    rules : list[dict[str, Any]] | None
        적용할 검증 규칙 목록

    auto_schema : bool
        자동 스키마 생성 여부. 기본값: True

    fail_on_error : bool
        검증 실패 시 Task 실패 여부. 기본값: True

    create_artifact : bool
        Prefect Artifact 생성 여부. 기본값: True

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
    Flow에서 사용:

    >>> @flow
    ... async def my_pipeline():
    ...     data = load_data()
    ...     result = await data_quality_check_task(
    ...         data=data,
    ...         rules=[
    ...             {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
    ...             {"column": "age", "type": "in_range", "min": 0, "max": 150},
    ...         ],
    ...     )
    ...     if result.status.name == "PASSED":
    ...         process_data(data)

    Block 사용:

    >>> block = DataQualityBlock(engine_name="truthound", auto_schema=True)
    >>> result = await data_quality_check_task(
    ...     data=data,
    ...     block=block,
    ...     rules=[...],
    ... )
    """
    from common.engines import get_engine
    from truthound_prefect.utils import DataQualityError, to_prefect_artifact

    logger = get_run_logger()

    # Block 또는 기본 설정 사용
    if block:
        engine = block.get_engine()
        effective_rules = rules or block.default_rules
        effective_auto_schema = auto_schema if auto_schema else block.auto_schema
    else:
        engine = get_engine("truthound")
        effective_rules = rules or []
        effective_auto_schema = auto_schema

    logger.info(f"Starting quality check with engine: {engine.engine_name}")

    result = engine.check(
        data,
        rules=effective_rules,
        auto_schema=effective_auto_schema,
    )

    # Artifact 생성
    if create_artifact:
        await to_prefect_artifact(result)

    logger.info(
        f"Quality check complete: "
        f"passed={result.passed_count}, failed={result.failed_count}"
    )

    # 실패 처리
    if result.status.name == "FAILED" and fail_on_error:
        raise DataQualityError(
            f"Quality check failed: {result.failed_count} rules failed",
            result=result,
        )

    return result
```

### data_quality_profile_task

```python
# tasks/profile.py
from prefect import task, get_run_logger
from typing import Any
import polars as pl
from datetime import timedelta


@task(
    name="data_quality_profile",
    description="데이터 프로파일링",
    tags=["data-quality", "profiling"],
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
async def data_quality_profile_task(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    block: "DataQualityBlock | None" = None,
    columns: list[str] | None = None,
    create_artifact: bool = True,
) -> "ProfileOutput":
    """
    데이터 프로파일링 Task.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        프로파일링할 데이터

    block : DataQualityBlock | None
        사용할 DataQualityBlock

    columns : list[str] | None
        프로파일링할 컬럼. None이면 전체.

    create_artifact : bool
        Artifact 생성 여부

    Returns
    -------
    ProfileOutput
        프로파일링 결과
    """
    from common.engines import get_engine
    from truthound_prefect.utils import to_prefect_artifact

    logger = get_run_logger()
    logger.info("Starting data profiling")

    if block:
        engine = block.get_engine()
    else:
        engine = get_engine("truthound")

    result = engine.profile(data, columns=columns)

    if create_artifact:
        await to_prefect_artifact(result)

    logger.info(f"Profiling complete: {len(result.columns)} columns analyzed")

    return result
```

### data_quality_learn_task

```python
# tasks/learn.py
from prefect import task, get_run_logger
from typing import Any
import polars as pl


@task(
    name="data_quality_learn",
    description="데이터에서 스키마 자동 학습",
    tags=["data-quality", "schema", "learning"],
)
async def data_quality_learn_task(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    block: "DataQualityBlock | None" = None,
) -> "LearnOutput":
    """
    데이터에서 스키마와 검증 규칙을 자동 학습.

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame
        학습할 데이터

    block : DataQualityBlock | None
        사용할 DataQualityBlock

    Returns
    -------
    LearnOutput
        학습된 스키마 및 추천 규칙

    Examples
    --------
    >>> result = await data_quality_learn_task(data)
    >>> print(f"Learned {len(result.rules)} rules")
    """
    from common.engines import get_engine

    logger = get_run_logger()
    logger.info("Learning schema from data")

    if block:
        engine = block.get_engine()
    else:
        engine = get_engine("truthound")

    result = engine.learn(data)

    logger.info(f"Learned {len(result.rules)} rules")

    return result
```

---

## Flows

### Flow Decorators

```python
# flows/decorators.py
from prefect import flow
from functools import wraps
from typing import Any, Callable
import polars as pl


def quality_checked_flow(
    rules: list[dict[str, Any]] | None = None,
    *,
    auto_schema: bool = True,
    fail_on_error: bool = True,
    **flow_kwargs,
) -> Callable:
    """
    품질 검증이 포함된 Flow 데코레이터.

    Flow 함수의 반환값에 대해 자동으로 품질 검증을 수행합니다.

    Parameters
    ----------
    rules : list[dict[str, Any]] | None
        검증 규칙. None이면 auto_schema 사용.

    auto_schema : bool
        자동 스키마 생성 여부

    fail_on_error : bool
        검증 실패 시 예외 발생 여부

    Examples
    --------
    >>> @quality_checked_flow(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     fail_on_error=True,
    ... )
    ... async def process_users():
    ...     return load_and_transform_users()
    """
    def decorator(fn: Callable) -> Callable:
        @flow(**flow_kwargs)
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            from truthound_prefect.tasks import data_quality_check_task

            result = await fn(*args, **kwargs)

            if isinstance(result, (pl.DataFrame, pl.LazyFrame)):
                await data_quality_check_task(
                    data=result,
                    rules=rules,
                    auto_schema=auto_schema,
                    fail_on_error=fail_on_error,
                )

            return result
        return wrapper
    return decorator


def profiled_flow(**flow_kwargs) -> Callable:
    """
    프로파일링이 포함된 Flow 데코레이터.

    Flow 함수의 반환값에 대해 자동으로 프로파일링을 수행합니다.
    """
    def decorator(fn: Callable) -> Callable:
        @flow(**flow_kwargs)
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            from truthound_prefect.tasks import data_quality_profile_task

            result = await fn(*args, **kwargs)

            if isinstance(result, (pl.DataFrame, pl.LazyFrame)):
                await data_quality_profile_task(data=result)

            return result
        return wrapper
    return decorator


def validated_flow(
    rules: list[dict[str, Any]],
    **flow_kwargs,
) -> Callable:
    """
    검증이 포함된 Flow 데코레이터 (실패시 예외 발생).
    """
    return quality_checked_flow(
        rules=rules,
        fail_on_error=True,
        **flow_kwargs,
    )
```

### Flow Factories

```python
# flows/factories.py
from prefect import flow, get_run_logger
from typing import Any, Callable
import polars as pl


def create_quality_flow(
    name: str,
    rules: list[dict[str, Any]],
    *,
    auto_schema: bool = True,
    fail_on_error: bool = True,
    retries: int = 1,
) -> Callable:
    """
    품질 검증 Flow 팩토리.

    Parameters
    ----------
    name : str
        Flow 이름

    rules : list[dict[str, Any]]
        검증 규칙

    auto_schema : bool
        자동 스키마 생성 여부

    fail_on_error : bool
        실패 시 예외 발생 여부

    retries : int
        재시도 횟수

    Returns
    -------
    Callable
        생성된 Flow 함수

    Examples
    --------
    >>> user_quality_flow = create_quality_flow(
    ...     name="user_quality_check",
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
    ...     ],
    ... )
    >>> result = await user_quality_flow(data=user_df)
    """
    from truthound_prefect.tasks import data_quality_check_task

    @flow(name=name, retries=retries)
    async def quality_flow(data: pl.DataFrame) -> "QualityCheckOutput":
        logger = get_run_logger()
        logger.info(f"Running quality flow: {name}")

        result = await data_quality_check_task(
            data=data,
            rules=rules,
            auto_schema=auto_schema,
            fail_on_error=fail_on_error,
        )

        logger.info(
            f"Quality check complete: "
            f"passed={result.passed_count}, failed={result.failed_count}"
        )

        return result

    return quality_flow


def create_validation_flow(
    name: str,
    rules: list[dict[str, Any]],
) -> Callable:
    """
    검증 Flow 팩토리 (검증된 데이터 반환).

    검증 실패시 예외를 발생시키고, 성공시 원본 데이터를 반환합니다.
    """
    from truthound_prefect.tasks import data_quality_check_task

    @flow(name=name)
    async def validation_flow(data: pl.DataFrame) -> pl.DataFrame:
        logger = get_run_logger()
        logger.info(f"Running validation flow: {name}")

        await data_quality_check_task(
            data=data,
            rules=rules,
            fail_on_error=True,
        )

        return data

    return validation_flow


def create_pipeline_flow(
    name: str,
    *,
    pre_check_rules: list[dict[str, Any]] | None = None,
    post_check_rules: list[dict[str, Any]] | None = None,
    auto_schema: bool = True,
) -> Callable:
    """
    파이프라인 Flow 팩토리 (전처리/후처리 검증 포함).

    Parameters
    ----------
    name : str
        Flow 이름

    pre_check_rules : list[dict[str, Any]] | None
        전처리 검증 규칙

    post_check_rules : list[dict[str, Any]] | None
        후처리 검증 규칙

    auto_schema : bool
        자동 스키마 생성 여부
    """
    from truthound_prefect.tasks import data_quality_check_task

    def decorator(transform_fn: Callable) -> Callable:
        @flow(name=name)
        async def pipeline_flow(data: pl.DataFrame) -> pl.DataFrame:
            logger = get_run_logger()

            # 전처리 검증
            if pre_check_rules:
                logger.info("Running pre-check validation")
                await data_quality_check_task(
                    data=data,
                    rules=pre_check_rules,
                    auto_schema=auto_schema,
                    fail_on_error=True,
                )

            # 변환 실행
            result = await transform_fn(data)

            # 후처리 검증
            if post_check_rules:
                logger.info("Running post-check validation")
                await data_quality_check_task(
                    data=result,
                    rules=post_check_rules,
                    auto_schema=auto_schema,
                    fail_on_error=True,
                )

            return result

        return pipeline_flow
    return decorator
```

---

## Artifacts

### Quality Artifacts

```python
# utils/serialization.py
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from typing import Any, Union


async def to_prefect_artifact(
    result: Union["QualityCheckOutput", "ProfileOutput", "LearnOutput"],
    *,
    key: str | None = None,
    include_failures: bool = True,
    max_failures: int = 20,
) -> str:
    """
    데이터 품질 결과를 Prefect Artifact로 생성.

    Parameters
    ----------
    result : QualityCheckOutput | ProfileOutput | LearnOutput
        검증/프로파일링/학습 결과

    key : str | None
        Artifact 키. None이면 결과 타입에 따라 자동 생성.

    include_failures : bool
        실패 상세 포함 여부 (QualityCheckOutput만 해당)

    max_failures : int
        표시할 최대 실패 수

    Returns
    -------
    str
        생성된 Artifact ID
    """
    from truthound_prefect.utils import QualityCheckOutput, ProfileOutput, LearnOutput

    if isinstance(result, QualityCheckOutput):
        return await _create_check_artifact(
            result,
            key=key or "quality-check",
            include_failures=include_failures,
            max_failures=max_failures,
        )
    elif isinstance(result, ProfileOutput):
        return await _create_profile_artifact(result, key=key or "data-profile")
    elif isinstance(result, LearnOutput):
        return await _create_learn_artifact(result, key=key or "schema-learn")
    else:
        raise ValueError(f"Unsupported result type: {type(result)}")


async def _create_check_artifact(
    result: "QualityCheckOutput",
    *,
    key: str = "quality-check",
    include_failures: bool = True,
    max_failures: int = 20,
) -> str:
    """품질 검증 결과를 Prefect Artifact로 생성."""
    # 요약 테이블
    summary_table = [
        {"Metric": "Status", "Value": result.status.name},
        {"Metric": "Passed", "Value": str(result.passed_count)},
        {"Metric": "Failed", "Value": str(result.failed_count)},
        {"Metric": "Pass Rate", "Value": f"{result.pass_rate:.2%}"},
        {"Metric": "Duration", "Value": f"{result.execution_time_ms:.2f}ms"},
    ]

    await create_table_artifact(
        key=f"{key}-summary",
        table=summary_table,
        description="Quality Check Summary",
    )

    # 실패 상세
    if include_failures and result.failures:
        failure_table = [
            {
                "Rule": f.rule_name,
                "Column": f.column or "-",
                "Message": f.message,
                "Severity": f.severity.name,
                "Failed": f.failed_count,
                "Total": f.total_count,
            }
            for f in result.failures[:max_failures]
        ]

        await create_table_artifact(
            key=f"{key}-failures",
            table=failure_table,
            description=f"Quality Check Failures (top {max_failures})",
        )

    # Markdown 리포트
    status_emoji = "PASSED" if result.status.name == "PASSED" else "FAILED"
    markdown = f"""
# Quality Check Report: {status_emoji}

## Summary
- **Status:** {result.status.name}
- **Passed:** {result.passed_count}
- **Failed:** {result.failed_count}
- **Pass Rate:** {result.pass_rate:.2%}
- **Duration:** {result.execution_time_ms:.2f}ms
    """

    if result.failures:
        markdown += "\n## Top Failures\n"
        for f in result.failures[:5]:
            markdown += f"- **{f.rule_name}** ({f.column}): {f.message}\n"

    artifact_id = await create_markdown_artifact(
        key=key,
        markdown=markdown,
        description="Quality Check Report",
    )

    return artifact_id


async def _create_profile_artifact(
    result: "ProfileOutput",
    *,
    key: str = "data-profile",
    max_columns: int = 30,
) -> str:
    """프로파일링 결과를 Prefect Artifact로 생성."""
    # 컬럼 테이블
    column_table = [
        {
            "Column": col.column_name,
            "Type": col.dtype,
            "Nulls": col.null_count,
            "Null %": f"{col.null_percentage:.1f}%",
            "Uniques": col.unique_count,
        }
        for col in result.columns[:max_columns]
    ]

    await create_table_artifact(
        key=f"{key}-columns",
        table=column_table,
        description=f"Data Profile ({len(result.columns)} columns)",
    )

    markdown = f"""
# Data Profile Report

## Overview
- **Row Count:** {result.row_count:,}
- **Column Count:** {len(result.columns)}

## Columns
| Column | Type | Nulls | Null % | Uniques |
|--------|------|-------|--------|---------|
"""
    for col in result.columns[:10]:
        markdown += f"| {col.column_name} | {col.dtype} | {col.null_count} | {col.null_percentage:.1f}% | {col.unique_count} |\n"

    if len(result.columns) > 10:
        markdown += f"\n*... and {len(result.columns) - 10} more columns*\n"

    artifact_id = await create_markdown_artifact(
        key=key,
        markdown=markdown,
        description="Data Profile Report",
    )

    return artifact_id


async def _create_learn_artifact(
    result: "LearnOutput",
    *,
    key: str = "schema-learn",
) -> str:
    """스키마 학습 결과를 Prefect Artifact로 생성."""
    # 규칙 테이블
    rules_table = [
        {
            "Column": rule.column,
            "Rule Type": rule.rule_type,
            "Confidence": f"{rule.confidence:.2f}",
        }
        for rule in result.rules[:20]
    ]

    await create_table_artifact(
        key=f"{key}-rules",
        table=rules_table,
        description=f"Learned Rules ({len(result.rules)} rules)",
    )

    markdown = f"""
# Schema Learn Report

## Summary
- **Rules Learned:** {len(result.rules)}

## Top Rules (by confidence)
| Column | Rule Type | Confidence |
|--------|-----------|------------|
"""
    for rule in sorted(result.rules, key=lambda r: r.confidence, reverse=True)[:10]:
        markdown += f"| {rule.column} | {rule.rule_type} | {rule.confidence:.2f} |\n"

    artifact_id = await create_markdown_artifact(
        key=key,
        markdown=markdown,
        description="Schema Learn Report",
    )

    return artifact_id
```

---

## Deployment Configuration

### prefect.yaml

```yaml
# prefect.yaml
name: truthound-pipelines
prefect-version: 2.14.0

# 빌드 설정
build:
  - prefect_docker.deployments.steps.build_docker_image:
      id: build_image
      requires: prefect-docker>=0.4.0
      image_name: "{{ $DOCKER_REGISTRY }}/truthound-pipelines"
      tag: "{{ $GIT_SHA }}"
      dockerfile: Dockerfile

# 푸시 설정
push:
  - prefect_docker.deployments.steps.push_docker_image:
      requires: prefect-docker>=0.4.0
      image_name: "{{ build_image.image_name }}"
      tag: "{{ build_image.tag }}"

# 배포 정의
deployments:
  - name: daily-quality-check
    version: "1.0.0"
    tags: ["production", "quality"]
    description: "Daily data quality validation"
    entrypoint: flows/quality.py:daily_quality_flow
    parameters:
      block_name: production-truthound
      fail_on_error: true
    schedule:
      cron: "0 6 * * *"
      timezone: "Asia/Seoul"
    work_pool:
      name: kubernetes-pool
      work_queue_name: quality

  - name: hourly-monitoring
    version: "1.0.0"
    tags: ["production", "monitoring"]
    description: "Hourly quality monitoring"
    entrypoint: flows/monitoring.py:hourly_monitoring_flow
    parameters:
      block_name: production-truthound
      alert_on_failure: true
    schedule:
      interval: 3600
    work_pool:
      name: kubernetes-pool
      work_queue_name: monitoring

  - name: ad-hoc-validation
    version: "1.0.0"
    tags: ["development", "ad-hoc"]
    description: "On-demand quality validation"
    entrypoint: flows/quality.py:validation_flow
    work_pool:
      name: local-pool
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . .

# Prefect 설정
ENV PREFECT_API_URL=${PREFECT_API_URL}
ENV PREFECT_API_KEY=${PREFECT_API_KEY}

# 진입점
CMD ["prefect", "worker", "start", "-p", "kubernetes-pool"]
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: truthound-prefect-worker
  labels:
    app: truthound-prefect
spec:
  replicas: 2
  selector:
    matchLabels:
      app: truthound-prefect
  template:
    metadata:
      labels:
        app: truthound-prefect
    spec:
      containers:
        - name: worker
          image: registry/truthound-pipelines:latest
          env:
            - name: PREFECT_API_URL
              valueFrom:
                secretKeyRef:
                  name: prefect-secrets
                  key: api-url
            - name: PREFECT_API_KEY
              valueFrom:
                secretKeyRef:
                  name: prefect-secrets
                  key: api-key
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
```

---

## Usage Examples

### Complete Pipeline Example

```python
# flows/quality_pipeline.py
from prefect import flow, task, get_run_logger
from truthound_prefect import (
    DataQualityBlock,
    data_quality_check_task,
    data_quality_profile_task,
)
import polars as pl


@task(name="extract_data")
async def extract_data(source: str) -> pl.DataFrame:
    """데이터 추출"""
    return pl.read_parquet(source)


@task(name="transform_data")
async def transform_data(data: pl.DataFrame) -> pl.DataFrame:
    """데이터 변환"""
    return (
        data
        .filter(pl.col("status") == "active")
        .with_columns([
            pl.col("amount").round(2),
        ])
    )


@task(name="load_data")
async def load_data(data: pl.DataFrame, destination: str) -> None:
    """데이터 로드"""
    data.write_parquet(destination)


@flow(name="etl_with_quality_gates")
async def etl_pipeline(
    source: str,
    destination: str,
) -> dict:
    """
    품질 게이트가 포함된 ETL 파이프라인.

    Parameters
    ----------
    source : str
        소스 데이터 경로

    destination : str
        대상 경로

    Returns
    -------
    dict
        파이프라인 실행 결과
    """
    logger = get_run_logger()
    results = {}

    # Block 설정
    block = DataQualityBlock(engine_name="truthound", auto_schema=True)

    # 1. 데이터 추출
    logger.info("Extracting data...")
    raw_data = await extract_data(source)
    results["extracted_rows"] = len(raw_data)

    # 2. 원본 데이터 품질 검증
    logger.info("Checking raw data quality...")
    raw_quality = await data_quality_check_task(
        data=raw_data,
        block=block,
        rules=[
            {"column": "id", "type": "not_null"},
            {"column": "id", "type": "unique"},
            {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
            {"column": "amount", "type": "in_range", "min": 0},
        ],
        fail_on_error=True,
    )
    results["raw_quality"] = raw_quality

    # 3. 데이터 변환
    logger.info("Transforming data...")
    transformed_data = await transform_data(raw_data)
    results["transformed_rows"] = len(transformed_data)

    # 4. 변환 데이터 품질 검증
    logger.info("Checking transformed data quality...")
    await data_quality_check_task(
        data=transformed_data,
        block=block,
        rules=[
            {"column": "id", "type": "not_null"},
            {"column": "amount", "type": "in_range", "min": 0},
        ],
        fail_on_error=True,
    )

    # 5. 데이터 로드
    logger.info("Loading data...")
    await load_data(transformed_data, destination)
    results["loaded_rows"] = len(transformed_data)

    logger.info("ETL pipeline completed successfully!")
    return results


# 실행
if __name__ == "__main__":
    import asyncio

    asyncio.run(etl_pipeline(
        source="s3://bucket/raw/data.parquet",
        destination="s3://bucket/processed/data.parquet",
    ))
```

### With Concurrency

```python
@flow(name="parallel_quality_checks")
async def parallel_validation_flow(
    datasets: list[dict[str, Any]],
) -> list[dict]:
    """
    여러 데이터셋을 병렬로 검증.

    Parameters
    ----------
    datasets : list[dict[str, Any]]
        검증할 데이터셋 목록
        [{"name": "users", "path": "...", "rules": [...]}]

    Returns
    -------
    list[dict]
        각 데이터셋의 검증 결과
    """
    from prefect import task
    from prefect.tasks import task_map
    from truthound_prefect import data_quality_check_task, DataQualityBlock

    block = DataQualityBlock(engine_name="truthound", auto_schema=True)

    @task
    async def check_dataset(dataset: dict) -> dict:
        data = pl.read_parquet(dataset["path"])
        result = await data_quality_check_task(
            data=data,
            block=block,
            rules=dataset["rules"],
            fail_on_error=False,
        )
        return {"name": dataset["name"], "result": result}

    # 병렬 실행
    results = await task_map(check_dataset, datasets)

    return results
```

---

## Testing Strategy

### Test Structure

```
tests/
├── __init__.py
├── conftest.py
├── test_blocks/
│   ├── __init__.py
│   └── test_data_quality_block.py
├── test_tasks/
│   ├── __init__.py
│   ├── test_check.py
│   ├── test_profile.py
│   └── test_learn.py
└── test_flows/
    ├── __init__.py
    └── test_decorators.py
```

### Fixtures

```python
# conftest.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import polars as pl
from datetime import datetime


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3],
        "email": ["a@b.com", "c@d.com", "e@f.com"],
        "amount": [100.0, 200.0, 300.0],
    })


@pytest.fixture
def mock_engine():
    with patch("common.engines.get_engine") as mock:
        mock_result = MagicMock()
        mock_result.status.name = "PASSED"
        mock_result.passed_count = 3
        mock_result.failed_count = 0
        mock_result.failures = []
        mock_result.execution_time_ms = 50.0

        engine = MagicMock()
        engine.check.return_value = mock_result
        engine.engine_name = "truthound"

        mock.return_value = engine
        yield mock


@pytest.fixture
def mock_block():
    from truthound_prefect.blocks import DataQualityBlock
    from truthound_prefect.utils import QualityCheckOutput, CheckStatus

    block = AsyncMock(spec=DataQualityBlock)
    block.check.return_value = QualityCheckOutput(
        status=CheckStatus.PASSED,
        passed_count=3,
        failed_count=0,
        failures=(),
        execution_time_ms=50.0,
    )
    return block
```

### Unit Tests

```python
# test_tasks/test_check.py
import pytest
from truthound_prefect.tasks import data_quality_check_task
from truthound_prefect.utils import DataQualityError


class TestDataQualityCheckTask:

    @pytest.mark.asyncio
    async def test_check_success(self, sample_dataframe, mock_engine):
        rules = [{"column": "id", "type": "not_null"}]

        result = await data_quality_check_task(
            data=sample_dataframe,
            rules=rules,
            fail_on_error=False,
        )

        assert result.status.name == "PASSED"
        assert result.passed_count == 3

    @pytest.mark.asyncio
    async def test_check_failure_raises(self, sample_dataframe, mock_engine):
        mock_engine.return_value.check.return_value.status.name = "FAILED"
        mock_engine.return_value.check.return_value.failed_count = 2

        rules = [{"column": "id", "type": "not_null"}]

        with pytest.raises(DataQualityError):
            await data_quality_check_task(
                data=sample_dataframe,
                rules=rules,
                fail_on_error=True,
            )

    @pytest.mark.asyncio
    async def test_check_with_block(self, sample_dataframe, mock_block):
        result = await data_quality_check_task(
            data=sample_dataframe,
            block=mock_block,
            rules=[{"column": "id", "type": "not_null"}],
        )

        mock_block.get_engine.assert_called_once()
        assert result.status.name == "PASSED"
```

---

## pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "truthound-prefect"
version = "0.1.0"
description = "Prefect integration for Truthound data quality framework"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    { name = "Truthound Team", email = "team@truthound.dev" }
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Framework :: Prefect",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
keywords = ["prefect", "data-quality", "truthound", "validation", "workflow"]

dependencies = [
    "prefect>=2.14.0",
    "truthound>=1.0.0",
    "polars>=0.20.0",
]

[project.optional-dependencies]
aws = ["prefect-aws>=0.4.0"]
gcp = ["prefect-gcp>=0.5.0"]
all = ["truthound-prefect[aws,gcp]"]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
]

[project.urls]
Homepage = "https://github.com/seadonggyun4/truthound-integrations"
Documentation = "https://truthound.dev/docs/integrations/prefect"
Repository = "https://github.com/seadonggyun4/truthound-integrations"

[project.entry-points."prefect.blocks"]
data-quality = "truthound_prefect.blocks:DataQualityBlock"

[tool.hatch.build.targets.wheel]
packages = ["src/truthound_prefect"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true

[[tool.mypy.overrides]]
module = "prefect.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
```

---

## References

- [Prefect Tasks](https://docs.prefect.io/concepts/tasks/)
- [Prefect Flows](https://docs.prefect.io/concepts/flows/)
- [Prefect Blocks](https://docs.prefect.io/concepts/blocks/)
- [Prefect Artifacts](https://docs.prefect.io/concepts/artifacts/)
- [Truthound Documentation](https://truthound.dev/docs)

---

*이 문서는 truthound-prefect 패키지의 완전한 구현 명세입니다.*
