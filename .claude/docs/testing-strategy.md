# Testing Strategy

> **Last Updated:** 2024-12-30
> **Document Version:** 1.0.0
> **Status:** Implementation Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Test Pyramid](#test-pyramid)
3. [Package Test Structures](#package-test-structures)
4. [Mock Strategy](#mock-strategy)
5. [Platform Test Containers](#platform-test-containers)
6. [Integration Test Environment](#integration-test-environment)
7. [Coverage Requirements](#coverage-requirements)
8. [pytest Configuration](#pytest-configuration)

---

## Overview

### Testing Philosophy

| Principle | Description |
|-----------|-------------|
| **Fast Feedback** | 단위 테스트는 밀리초 내 완료 |
| **Isolation** | 각 테스트는 독립적으로 실행 |
| **Determinism** | 동일 입력에 항상 동일 결과 |
| **Clarity** | 실패 시 명확한 원인 파악 가능 |
| **Coverage** | 핵심 로직 80%+ 커버리지 |

### Test Categories

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Test Pyramid                                    │
│                                                                         │
│                              ▲                                          │
│                             /│\                                         │
│                            / │ \        E2E Tests                       │
│                           /  │  \       (Few, Slow)                     │
│                          /───┼───\                                      │
│                         /    │    \                                     │
│                        /     │     \    Integration Tests               │
│                       /      │      \   (Some, Medium)                  │
│                      /───────┼───────\                                  │
│                     /        │        \                                 │
│                    /         │         \   Unit Tests                   │
│                   /          │          \  (Many, Fast)                 │
│                  /───────────┼───────────\                              │
│                 ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔                             │
│                                                                         │
│  Distribution:  70% Unit  |  20% Integration  |  10% E2E               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Test Pyramid

### Unit Tests (70%)

**목적**: 개별 함수/클래스의 정확성 검증

```python
# packages/airflow/tests/test_operators/test_check_operator.py
import pytest
from unittest.mock import MagicMock, patch
from truthound_airflow import TruthoundCheckOperator


class TestTruthoundCheckOperator:
    """TruthoundCheckOperator 단위 테스트"""

    def test_init_with_data_path(self):
        """data_path 파라미터로 초기화"""
        op = TruthoundCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path="/data/test.parquet",
        )

        assert op.data_path == "/data/test.parquet"
        assert op.sql is None
        assert len(op.rules) == 1

    def test_init_fails_without_source(self):
        """data_path와 sql 모두 없으면 실패"""
        with pytest.raises(ValueError, match="Must specify either"):
            TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
            )

    def test_template_fields(self):
        """템플릿 필드 정의 확인"""
        op = TruthoundCheckOperator(
            task_id="test",
            rules=[],
            data_path="/data/test.parquet",
        )

        assert "data_path" in op.template_fields
        assert "rules" in op.template_fields

    @patch("truthound_airflow.operators.check.TruthoundHook")
    @patch("truthound_airflow.operators.check.th")
    def test_execute_success(self, mock_th, mock_hook_class, airflow_context):
        """성공적인 실행"""
        # Mock 설정
        mock_hook = MagicMock()
        mock_hook.load_data.return_value = MagicMock()
        mock_hook_class.return_value = mock_hook

        mock_result = MagicMock()
        mock_result.is_success = True
        mock_result.passed_count = 3
        mock_result.failed_count = 0
        mock_result.failures = []
        mock_th.check.return_value = mock_result

        # 실행
        op = TruthoundCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path="/data/test.parquet",
        )
        result = op.execute(airflow_context)

        # 검증
        assert result["is_success"] is True
        mock_hook.load_data.assert_called_once()
        mock_th.check.assert_called_once()
```

### Integration Tests (20%)

**목적**: 컴포넌트 간 상호작용 검증

```python
# packages/airflow/tests/test_integration/test_operator_hook.py
import pytest
import polars as pl
from truthound_airflow import TruthoundCheckOperator, TruthoundHook


@pytest.mark.integration
class TestOperatorHookIntegration:
    """Operator와 Hook 통합 테스트"""

    @pytest.fixture
    def sample_data(self, tmp_path) -> str:
        """테스트 데이터 파일 생성"""
        data = pl.DataFrame({
            "id": [1, 2, 3],
            "email": ["a@b.com", "c@d.com", "e@f.com"],
        })
        path = tmp_path / "test.parquet"
        data.write_parquet(path)
        return str(path)

    def test_operator_loads_data_via_hook(self, sample_data, airflow_context):
        """Operator가 Hook을 통해 데이터 로드"""
        op = TruthoundCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path=sample_data,
        )

        # 실제 Hook과 실제 파일로 테스트
        result = op.execute(airflow_context)

        assert result["passed_count"] >= 1

    def test_xcom_push_on_success(self, sample_data, airflow_context):
        """성공 시 XCom 푸시 확인"""
        op = TruthoundCheckOperator(
            task_id="test",
            rules=[{"column": "id", "type": "not_null"}],
            data_path=sample_data,
        )

        op.execute(airflow_context)

        airflow_context["ti"].xcom_push.assert_called()
        call_args = airflow_context["ti"].xcom_push.call_args
        assert call_args.kwargs["key"] == "truthound_result"
```

### E2E Tests (10%)

**목적**: 전체 워크플로우 검증

```python
# tests/e2e/test_airflow_dag.py
import pytest
from airflow.models import DagBag


@pytest.mark.e2e
class TestAirflowDAGExecution:
    """Airflow DAG E2E 테스트"""

    @pytest.fixture
    def dag_bag(self):
        """예제 DAG 로드"""
        return DagBag(
            dag_folder="examples/airflow/dags/",
            include_examples=False,
        )

    def test_dag_loads_without_errors(self, dag_bag):
        """DAG 로드 오류 없음"""
        assert len(dag_bag.import_errors) == 0

    def test_dag_task_dependencies(self, dag_bag):
        """태스크 의존성 검증"""
        dag = dag_bag.get_dag("quality_gated_etl")

        # 품질 체크가 로드 전에 실행되어야 함
        check_task = dag.get_task("check_raw_data_quality")
        load_task = dag.get_task("load_to_warehouse")

        assert check_task in dag.get_task("transform_data").upstream_list

    @pytest.mark.slow
    def test_dag_run_completes(self, dag_bag, airflow_db):
        """DAG 실행 완료"""
        from airflow.utils.state import State
        from airflow.executors.debug_executor import DebugExecutor

        dag = dag_bag.get_dag("quality_gated_etl")

        dag.run(
            start_date=dag.start_date,
            end_date=dag.start_date,
            executor=DebugExecutor(),
        )

        # 모든 태스크 성공 확인
        for task in dag.tasks:
            assert task.state == State.SUCCESS
```

---

## Package Test Structures

### Airflow Package

```
packages/airflow/tests/
├── __init__.py
├── conftest.py                    # Fixtures
├── test_operators/
│   ├── __init__.py
│   ├── test_check_operator.py     # TruthoundCheckOperator 테스트
│   ├── test_profile_operator.py   # TruthoundProfileOperator 테스트
│   └── test_learn_operator.py     # TruthoundLearnOperator 테스트
├── test_sensors/
│   ├── __init__.py
│   └── test_quality_sensor.py     # TruthoundSensor 테스트
├── test_hooks/
│   ├── __init__.py
│   └── test_truthound_hook.py     # TruthoundHook 테스트
├── test_integration/
│   ├── __init__.py
│   ├── test_operator_hook.py      # Operator-Hook 통합
│   └── test_xcom_flow.py          # XCom 흐름 테스트
└── test_e2e/
    ├── __init__.py
    └── test_dag_execution.py      # DAG 실행 테스트
```

### Dagster Package

```
packages/dagster/tests/
├── __init__.py
├── conftest.py
├── test_resources/
│   ├── __init__.py
│   └── test_truthound_resource.py
├── test_assets/
│   ├── __init__.py
│   ├── test_factory.py
│   └── test_quality_assets.py
├── test_ops/
│   ├── __init__.py
│   ├── test_check_op.py
│   └── test_profile_op.py
├── test_sensors/
│   ├── __init__.py
│   └── test_quality_sensor.py
└── test_integration/
    ├── __init__.py
    └── test_asset_graph.py
```

### Prefect Package

```
packages/prefect/tests/
├── __init__.py
├── conftest.py
├── test_blocks/
│   ├── __init__.py
│   └── test_truthound_block.py
├── test_tasks/
│   ├── __init__.py
│   ├── test_check_task.py
│   ├── test_profile_task.py
│   └── test_learn_task.py
├── test_flows/
│   ├── __init__.py
│   └── test_templates.py
└── test_integration/
    ├── __init__.py
    └── test_flow_execution.py
```

### dbt Package

```
packages/dbt/integration_tests/
├── dbt_project.yml
├── profiles.yml
├── seeds/
│   ├── test_valid_data.csv
│   └── test_invalid_data.csv
├── models/
│   ├── test_model_valid.sql
│   └── test_model_invalid.sql
└── tests/
    └── schema.yml
```

---

## Mock Strategy

### Truthound API Mocking

```python
# common/testing.py (발췌)
from unittest.mock import MagicMock
from typing import Any
import polars as pl


class MockTruthound:
    """
    Truthound 모듈 Mock.

    테스트에서 실제 Truthound 호출을 대체합니다.
    """

    def __init__(
        self,
        should_pass: bool = True,
        passed_count: int = 5,
        failed_count: int = 0,
    ):
        self.should_pass = should_pass
        self.passed_count = passed_count
        self.failed_count = failed_count
        self._calls = []

    def check(self, data: pl.DataFrame, **kwargs) -> MagicMock:
        """Mock 검증"""
        self._calls.append(("check", data, kwargs))

        result = MagicMock()
        result.is_success = self.should_pass
        result.passed_count = self.passed_count
        result.failed_count = self.failed_count
        result.status.value = "passed" if self.should_pass else "failed"
        result.failures = []
        result.execution_time_ms = 50.0

        return result

    def profile(self, data: pl.DataFrame, **kwargs) -> MagicMock:
        """Mock 프로파일링"""
        self._calls.append(("profile", data, kwargs))

        result = MagicMock()
        result.columns = {"id": {}, "name": {}}
        result.row_count = len(data)
        result.execution_time_ms = 30.0

        return result

    @property
    def calls(self) -> list:
        return self._calls


# 사용 예시
@pytest.fixture
def mock_truthound():
    """Truthound Mock Fixture"""
    with patch("truthound_airflow.operators.check.th") as mock:
        mock_instance = MockTruthound(should_pass=True)
        mock.check = mock_instance.check
        mock.profile = mock_instance.profile
        yield mock_instance
```

### Platform Context Mocking

```python
# packages/airflow/tests/conftest.py
import pytest
from unittest.mock import MagicMock
from datetime import datetime


@pytest.fixture
def airflow_context():
    """Airflow 실행 컨텍스트 Mock"""
    context = {
        "ti": MagicMock(),
        "ds": "2024-01-01",
        "execution_date": datetime(2024, 1, 1),
        "dag_run": MagicMock(),
        "task": MagicMock(),
        "task_instance": MagicMock(),
    }

    # XCom Mock
    context["ti"].xcom_push = MagicMock()
    context["ti"].xcom_pull = MagicMock(return_value=None)

    return context


# packages/dagster/tests/conftest.py
@pytest.fixture
def dagster_context():
    """Dagster 실행 컨텍스트 Mock"""
    from dagster import build_op_context

    context = build_op_context(
        resources={"truthound": MagicMock()},
    )
    return context


# packages/prefect/tests/conftest.py
@pytest.fixture
def prefect_context():
    """Prefect 실행 컨텍스트 Mock"""
    from unittest.mock import AsyncMock

    context = MagicMock()
    context.log = MagicMock()
    return context
```

### Data Fixtures

```python
# packages/airflow/tests/conftest.py
import pytest
import polars as pl


@pytest.fixture
def valid_dataframe() -> pl.DataFrame:
    """유효한 테스트 데이터"""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com"],
        "age": [25, 30, 35, 40, 45],
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


@pytest.fixture
def invalid_dataframe() -> pl.DataFrame:
    """품질 문제가 있는 테스트 데이터"""
    return pl.DataFrame({
        "id": [1, 2, 2, None, 5],  # 중복, NULL
        "email": ["a@b.com", "invalid", None, "d@e.com", ""],  # 잘못된 형식, NULL, 빈 값
        "age": [25, -5, 200, 30, 35],  # 음수, 범위 초과
        "amount": [100.0, 0.0, -50.0, None, 500.0],  # 음수, NULL
    })


@pytest.fixture
def large_dataframe() -> pl.DataFrame:
    """대용량 테스트 데이터 (성능 테스트용)"""
    n = 100_000
    return pl.DataFrame({
        "id": range(n),
        "email": [f"user{i}@example.com" for i in range(n)],
        "age": [20 + (i % 60) for i in range(n)],
        "amount": [float(i * 10) for i in range(n)],
    })


@pytest.fixture
def sample_rules() -> list:
    """기본 검증 규칙"""
    return [
        {"column": "id", "type": "not_null"},
        {"column": "id", "type": "unique"},
        {"column": "email", "type": "regex", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
        {"column": "age", "type": "in_range", "min": 0, "max": 150},
        {"column": "amount", "type": "in_range", "min": 0},
    ]
```

---

## Platform Test Containers

### Airflow Test Container

```python
# packages/airflow/tests/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQL 컨테이너 (Airflow 메타데이터용)"""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def airflow_db(postgres_container):
    """Airflow 데이터베이스 초기화"""
    import os
    from airflow import settings
    from airflow.utils.db import initdb

    os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = postgres_container.get_connection_url()

    # Airflow DB 초기화
    initdb()

    yield settings.Session
```

### Dagster Test Container

```python
# packages/dagster/tests/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def dagster_postgres():
    """Dagster 저장소용 PostgreSQL"""
    with PostgresContainer("postgres:15") as postgres:
        yield postgres


@pytest.fixture
def dagster_instance(dagster_postgres, tmp_path):
    """Dagster 인스턴스"""
    from dagster import DagsterInstance

    instance = DagsterInstance.ephemeral()
    yield instance
    instance.dispose()
```

### dbt Test Container

```yaml
# packages/dbt/integration_tests/profiles.yml
integration_tests:
  target: ci
  outputs:
    ci:
      type: postgres
      host: localhost
      port: 5432
      user: postgres
      password: postgres
      dbname: postgres
      schema: truthound_test
      threads: 4
```

---

## Integration Test Environment

### Docker Compose for Testing

```yaml
# docker-compose.test.yml
version: "3.8"

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: test_db
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  airflow-init:
    image: apache/airflow:2.7.0-python3.11
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://test:test@postgres/test_db
    command: >
      bash -c "
        airflow db init &&
        airflow users create --username admin --password admin --firstname Test --lastname User --role Admin --email admin@example.com
      "

  airflow-webserver:
    image: apache/airflow:2.7.0-python3.11
    depends_on:
      - airflow-init
    ports:
      - "8080:8080"
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://test:test@postgres/test_db
    command: webserver
```

### Running Integration Tests

```bash
# 테스트 환경 시작
docker-compose -f docker-compose.test.yml up -d

# 환경 준비 대기
sleep 30

# 통합 테스트 실행
pytest tests/integration/ -v -m integration

# 환경 정리
docker-compose -f docker-compose.test.yml down -v
```

---

## Coverage Requirements

### Coverage Targets

| Category | Target | Minimum |
|----------|--------|---------|
| Unit Tests | 90%+ | 80% |
| Integration Tests | 70%+ | 60% |
| Overall | 80%+ | 70% |

### Coverage Configuration

```toml
# pyproject.toml
[tool.coverage.run]
source = [
    "packages/airflow/src",
    "packages/dagster/src",
    "packages/prefect/src",
    "common",
]
branch = true
parallel = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "@abstractmethod",
    "\\.\\.\\.",
]
fail_under = 80
show_missing = true

[tool.coverage.html]
directory = "htmlcov"
```

### Running Coverage

```bash
# 전체 커버리지
pytest --cov --cov-report=html --cov-report=xml

# 패키지별 커버리지
pytest packages/airflow/tests/ --cov=truthound_airflow --cov-report=term-missing

# 커버리지 최소 요구사항 확인
pytest --cov --cov-fail-under=80
```

---

## pytest Configuration

### pytest.ini

```ini
# pytest.ini
[pytest]
testpaths =
    packages/airflow/tests
    packages/dagster/tests
    packages/prefect/tests
    tests

python_files = test_*.py
python_classes = Test*
python_functions = test_*

addopts =
    -v
    --tb=short
    --strict-markers
    -ra

markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, requires dependencies)
    e2e: End-to-end tests (slowest, full pipeline)
    slow: Slow tests (> 5 seconds)
    airflow: Airflow-specific tests
    dagster: Dagster-specific tests
    prefect: Prefect-specific tests
    dbt: dbt-specific tests

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning

asyncio_mode = auto
```

### conftest.py (Root)

```python
# tests/conftest.py
import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "common"))


def pytest_configure(config):
    """pytest 설정"""
    config.addinivalue_line(
        "markers",
        "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """테스트 수집 후 처리"""
    # 마커가 없는 테스트는 unit으로 처리
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)

    # -m 옵션이 없으면 e2e 제외
    if config.option.markexpr == "":
        skip_e2e = pytest.mark.skip(reason="E2E tests skipped by default")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
```

### Running Tests

```bash
# 모든 테스트
pytest

# 단위 테스트만
pytest -m unit

# 통합 테스트만
pytest -m integration

# E2E 테스트 포함
pytest -m "e2e or integration or unit"

# 특정 패키지
pytest packages/airflow/tests/ -v

# 특정 테스트 클래스
pytest packages/airflow/tests/test_operators/test_check_operator.py::TestTruthoundCheckOperator -v

# 병렬 실행
pytest -n auto

# 실패한 테스트만 재실행
pytest --lf

# 자세한 출력
pytest -vvs
```

---

## Test Examples

### Complete Test Class Example

```python
# packages/airflow/tests/test_operators/test_check_operator.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from airflow.exceptions import AirflowException
import polars as pl

from truthound_airflow import TruthoundCheckOperator


class TestTruthoundCheckOperator:
    """TruthoundCheckOperator 테스트 스위트"""

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    class TestInit:
        """초기화 테스트"""

        def test_init_with_data_path(self):
            """data_path로 초기화"""
            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                data_path="/data/test.parquet",
            )
            assert op.data_path == "/data/test.parquet"
            assert op.sql is None

        def test_init_with_sql(self):
            """SQL로 초기화"""
            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                sql="SELECT * FROM test",
            )
            assert op.sql == "SELECT * FROM test"
            assert op.data_path is None

        def test_init_fails_with_both(self):
            """data_path와 sql 동시 지정 시 실패"""
            with pytest.raises(ValueError, match="Cannot specify both"):
                TruthoundCheckOperator(
                    task_id="test",
                    rules=[],
                    data_path="/data/test.parquet",
                    sql="SELECT * FROM test",
                )

        def test_init_fails_with_neither(self):
            """둘 다 없으면 실패"""
            with pytest.raises(ValueError, match="Must specify either"):
                TruthoundCheckOperator(
                    task_id="test",
                    rules=[],
                )

        def test_init_with_all_options(self):
            """모든 옵션으로 초기화"""
            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                data_path="/data/test.parquet",
                connection_id="custom_conn",
                fail_on_error=False,
                warning_threshold=0.05,
                sample_size=1000,
                timeout_seconds=600,
            )
            assert op.connection_id == "custom_conn"
            assert op.fail_on_error is False
            assert op.warning_threshold == 0.05
            assert op.sample_size == 1000
            assert op.timeout_seconds == 600

    # =========================================================================
    # Execution Tests
    # =========================================================================

    class TestExecute:
        """실행 테스트"""

        @pytest.fixture
        def mock_dependencies(self):
            """의존성 Mock"""
            with patch("truthound_airflow.operators.check.TruthoundHook") as mock_hook_cls, \
                 patch("truthound_airflow.operators.check.th") as mock_th:

                mock_hook = MagicMock()
                mock_hook.load_data.return_value = pl.DataFrame({
                    "id": [1, 2, 3],
                    "value": ["a", "b", "c"],
                })
                mock_hook_cls.return_value = mock_hook

                mock_result = MagicMock()
                mock_result.is_success = True
                mock_result.passed_count = 3
                mock_result.failed_count = 0
                mock_result.warning_count = 0
                mock_result.failures = []
                mock_result.execution_time_ms = 50.0
                mock_result.timestamp.isoformat.return_value = "2024-01-01T00:00:00"
                mock_result.status.value = "passed"
                mock_th.check.return_value = mock_result

                yield {
                    "hook_cls": mock_hook_cls,
                    "hook": mock_hook,
                    "th": mock_th,
                    "result": mock_result,
                }

        def test_execute_success(self, mock_dependencies, airflow_context):
            """성공적인 실행"""
            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                data_path="/data/test.parquet",
            )

            result = op.execute(airflow_context)

            assert result["is_success"] is True
            assert result["passed_count"] == 3
            mock_dependencies["hook"].load_data.assert_called_once()
            mock_dependencies["th"].check.assert_called_once()

        def test_execute_pushes_xcom(self, mock_dependencies, airflow_context):
            """XCom 푸시 확인"""
            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                data_path="/data/test.parquet",
            )

            op.execute(airflow_context)

            airflow_context["ti"].xcom_push.assert_called_once()
            call_kwargs = airflow_context["ti"].xcom_push.call_args.kwargs
            assert call_kwargs["key"] == "truthound_result"

        def test_execute_failure_raises(self, mock_dependencies, airflow_context):
            """실패 시 예외 발생"""
            mock_dependencies["result"].is_success = False
            mock_dependencies["result"].failed_count = 2
            mock_dependencies["result"].failure_rate = 0.4

            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                data_path="/data/test.parquet",
                fail_on_error=True,
            )

            with pytest.raises(AirflowException, match="Quality check failed"):
                op.execute(airflow_context)

        def test_execute_failure_no_raise_when_disabled(
            self,
            mock_dependencies,
            airflow_context,
        ):
            """fail_on_error=False면 예외 미발생"""
            mock_dependencies["result"].is_success = False
            mock_dependencies["result"].failed_count = 2

            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                data_path="/data/test.parquet",
                fail_on_error=False,
            )

            result = op.execute(airflow_context)
            assert result["is_success"] is False

        def test_execute_warning_threshold(self, mock_dependencies, airflow_context):
            """경고 임계값 내 실패는 예외 미발생"""
            mock_dependencies["result"].is_success = False
            mock_dependencies["result"].failed_count = 1
            mock_dependencies["result"].failure_rate = 0.02  # 2%

            op = TruthoundCheckOperator(
                task_id="test",
                rules=[{"column": "id", "type": "not_null"}],
                data_path="/data/test.parquet",
                fail_on_error=True,
                warning_threshold=0.05,  # 5%
            )

            # 예외 없이 실행
            result = op.execute(airflow_context)
            assert result["is_success"] is False

    # =========================================================================
    # Template Tests
    # =========================================================================

    class TestTemplateFields:
        """템플릿 필드 테스트"""

        def test_template_fields_defined(self):
            """템플릿 필드 정의"""
            op = TruthoundCheckOperator(
                task_id="test",
                rules=[],
                data_path="/data/test.parquet",
            )

            assert "data_path" in op.template_fields
            assert "rules" in op.template_fields
            assert "sql" in op.template_fields
            assert "connection_id" in op.template_fields

        def test_template_ext_defined(self):
            """템플릿 확장자 정의"""
            op = TruthoundCheckOperator(
                task_id="test",
                rules=[],
                data_path="/data/test.parquet",
            )

            assert ".sql" in op.template_ext
            assert ".json" in op.template_ext
```

---

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [testcontainers-python](https://testcontainers-python.readthedocs.io/)
- [Airflow Testing](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html#testing-a-dag)
- [Dagster Testing](https://docs.dagster.io/concepts/testing)

---

*이 문서는 Truthound Integrations의 테스트 전략을 정의합니다.*
