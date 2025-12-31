"""Pytest configuration and fixtures for truthound-airflow tests."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Mock Airflow modules before any imports
# =============================================================================

# Create mock airflow module hierarchy
mock_airflow = MagicMock()
mock_airflow.models = MagicMock()

# Create a proper BaseOperator mock class with log attribute
class MockBaseOperator:
    """Mock BaseOperator with log support."""

    template_fields: tuple[str, ...] = ()
    template_ext: tuple[str, ...] = ()
    ui_color: str = "#FFFFFF"
    ui_fgcolor: str = "#000000"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.task_id = kwargs.get("task_id", "mock_task")
        self.log = MagicMock()

mock_airflow.models.BaseOperator = MockBaseOperator

mock_airflow.sensors = MagicMock()
mock_airflow.sensors.base = MagicMock()

# Create a proper BaseSensorOperator mock class with log attribute
class MockBaseSensorOperator(MockBaseOperator):
    """Mock BaseSensorOperator with log support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

mock_airflow.sensors.base.BaseSensorOperator = MockBaseSensorOperator
mock_airflow.hooks = MagicMock()
mock_airflow.hooks.base = MagicMock()

# Create a proper BaseHook mock class that supports patching
class MockBaseHook:
    """Mock BaseHook with get_connection support."""

    _mock_connection: MagicMock | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.log = MagicMock()

    def get_connection(self, conn_id: str) -> MagicMock:
        """Mock get_connection method (instance method)."""
        if MockBaseHook._mock_connection is not None:
            return MockBaseHook._mock_connection
        return MagicMock()

    @classmethod
    def get_connection(cls, conn_id: str) -> MagicMock:  # type: ignore[no-redef]
        """Mock get_connection method (class method for static calls)."""
        if cls._mock_connection is not None:
            return cls._mock_connection
        return MagicMock()

mock_airflow.hooks.base.BaseHook = MockBaseHook
mock_airflow.utils = MagicMock()
mock_airflow.utils.context = MagicMock()
mock_airflow.exceptions = MagicMock()
mock_airflow.exceptions.AirflowException = Exception
mock_airflow.exceptions.AirflowSkipException = Exception

# Register mock modules if airflow is not installed
if "airflow" not in sys.modules:
    sys.modules["airflow"] = mock_airflow
    sys.modules["airflow.models"] = mock_airflow.models
    sys.modules["airflow.sensors"] = mock_airflow.sensors
    sys.modules["airflow.sensors.base"] = mock_airflow.sensors.base
    sys.modules["airflow.hooks"] = mock_airflow.hooks
    sys.modules["airflow.hooks.base"] = mock_airflow.hooks.base
    sys.modules["airflow.utils"] = mock_airflow.utils
    sys.modules["airflow.utils.context"] = mock_airflow.utils.context
    sys.modules["airflow.exceptions"] = mock_airflow.exceptions

# Try to import polars, skip if not available
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None  # type: ignore


# =============================================================================
# Markers Configuration
# =============================================================================


def pytest_configure(config: Any) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "requires_polars: marks tests requiring polars")


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_dataframe() -> Any:
    """Create a sample DataFrame for testing."""
    if not HAS_POLARS:
        pytest.skip("polars not installed")

    return pl.DataFrame(
        {
            "user_id": ["uuid1", "uuid2", "uuid3", None],
            "email": ["a@b.com", "invalid", "c@d.com", "e@f.com"],
            "age": [25, 30, -5, 150],
            "amount": [100.0, 200.0, 50.0, 1000.0],
            "status": ["active", "inactive", "active", "pending"],
        }
    )


@pytest.fixture
def valid_dataframe() -> Any:
    """Create a valid DataFrame with no quality issues."""
    if not HAS_POLARS:
        pytest.skip("polars not installed")

    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "email": ["a@b.com", "b@c.com", "c@d.com", "d@e.com", "e@f.com"],
            "age": [25, 30, 35, 40, 45],
            "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def sample_rules() -> list[dict[str, Any]]:
    """Sample validation rules."""
    return [
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "id"},
        {"type": "in_range", "column": "age", "min": 0, "max": 150},
    ]


# =============================================================================
# Mock Result Fixtures
# =============================================================================


@dataclass
class MockStatus:
    """Mock status with value attribute (like an Enum)."""

    value: str = "passed"


@dataclass
class MockSeverity:
    """Mock severity with value attribute (like an Enum)."""

    value: str = "error"


@dataclass
class MockFailure:
    """Mock validation failure."""

    rule_name: str = "not_null"
    column: str = "id"
    message: str = "Found null values"
    severity: MockSeverity | None = None
    failed_count: int = 1
    total_count: int = 100
    failure_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.severity is None:
            self.severity = MockSeverity("error")


@dataclass
class MockCheckResult:
    """Mock check result for testing."""

    status: MockStatus | None = None
    is_success: bool = True
    passed_count: int = 5
    failed_count: int = 0
    warning_count: int = 0
    failure_rate: float = 0.0
    failures: list[MockFailure] | None = None
    execution_time_ms: float = 100.0
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.status is None:
            self.status = MockStatus("passed")
        if self.failures is None:
            self.failures = []
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@pytest.fixture
def mock_success_result() -> MockCheckResult:
    """Create a successful check result."""
    return MockCheckResult(
        status=MockStatus("passed"),
        is_success=True,
        passed_count=5,
        failed_count=0,
        warning_count=0,
        failure_rate=0.0,
        failures=[],
    )


@pytest.fixture
def mock_failure_result() -> MockCheckResult:
    """Create a failed check result."""
    return MockCheckResult(
        status=MockStatus("failed"),
        is_success=False,
        passed_count=3,
        failed_count=2,
        warning_count=0,
        failure_rate=0.4,
        failures=[
            MockFailure(rule_name="not_null", column="user_id", failed_count=1),
            MockFailure(rule_name="in_range", column="age", failed_count=1),
        ],
    )


@pytest.fixture
def mock_warning_result() -> MockCheckResult:
    """Create a warning check result."""
    return MockCheckResult(
        status=MockStatus("warning"),
        is_success=False,
        passed_count=48,
        failed_count=2,
        warning_count=1,
        failure_rate=0.04,
        failures=[MockFailure(rule_name="not_null", column="optional_field", failed_count=2)],
    )


# =============================================================================
# Airflow Context Fixtures
# =============================================================================


@pytest.fixture
def mock_task_instance() -> MagicMock:
    """Create a mock Airflow TaskInstance."""
    ti = MagicMock()
    ti.xcom_push = MagicMock()
    ti.xcom_pull = MagicMock(return_value=None)
    ti.task_id = "test_task"
    ti.dag_id = "test_dag"
    ti.run_id = "test_run_123"
    return ti


@pytest.fixture
def mock_dag_run() -> MagicMock:
    """Create a mock Airflow DagRun."""
    dag_run = MagicMock()
    dag_run.run_id = "test_run_123"
    dag_run.execution_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dag_run.start_date = datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    return dag_run


@pytest.fixture
def airflow_context(mock_task_instance: MagicMock, mock_dag_run: MagicMock) -> dict[str, Any]:
    """Create a mock Airflow execution context."""
    return {
        "ti": mock_task_instance,
        "task_instance": mock_task_instance,
        "ds": "2024-01-01",
        "ds_nodash": "20240101",
        "execution_date": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        "dag_run": mock_dag_run,
        "run_id": "test_run_123",
        "dag": MagicMock(dag_id="test_dag"),
        "task": MagicMock(task_id="test_task"),
        "params": {},
        "var": {"json": MagicMock(), "value": MagicMock()},
        "conn": MagicMock(),
    }


# =============================================================================
# Connection Fixtures
# =============================================================================


@pytest.fixture
def mock_connection() -> MagicMock:
    """Create a mock Airflow Connection."""
    conn = MagicMock()
    conn.conn_id = "test_connection"
    conn.conn_type = "truthound"
    conn.host = "test-bucket"
    conn.port = None
    conn.schema = None
    conn.login = "access_key"
    conn.password = "secret_key"
    conn.extra = '{"conn_type": "s3", "region": "us-east-1"}'
    conn.extra_dejson = {"conn_type": "s3", "region": "us-east-1"}
    return conn


@pytest.fixture
def mock_postgres_connection() -> MagicMock:
    """Create a mock PostgreSQL Connection."""
    conn = MagicMock()
    conn.conn_id = "postgres_connection"
    conn.conn_type = "postgres"
    conn.host = "localhost"
    conn.port = 5432
    conn.schema = "test_db"
    conn.login = "user"
    conn.password = "password"
    conn.extra = '{"conn_type": "postgres"}'
    conn.extra_dejson = {"conn_type": "postgres"}
    return conn


# =============================================================================
# Engine Fixtures
# =============================================================================


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock DataQualityEngine."""
    engine = MagicMock()
    engine.engine_name = "mock_engine"
    engine.engine_version = "1.0.0"

    # Default successful check result
    check_result = MagicMock()
    check_result.status.value = "passed"
    check_result.is_success = True
    check_result.passed_count = 5
    check_result.failed_count = 0
    check_result.warning_count = 0
    check_result.failures = []
    check_result.execution_time_ms = 100.0
    check_result.timestamp = datetime.now(timezone.utc)
    engine.check.return_value = check_result

    # Default profile result
    profile_result = MagicMock()
    profile_result.columns = []
    profile_result.row_count = 100
    engine.profile.return_value = profile_result

    # Default learn result
    learn_result = MagicMock()
    learn_result.rules = []
    engine.learn.return_value = learn_result

    return engine


# =============================================================================
# Hook Fixtures
# =============================================================================


@pytest.fixture
def mock_hook(sample_dataframe: Any, mock_connection: MagicMock) -> MagicMock:
    """Create a mock DataQualityHook."""
    hook = MagicMock()
    hook.connection_id = "test_connection"
    hook.load_data.return_value = sample_dataframe
    hook.query.return_value = sample_dataframe
    hook.get_connection.return_value = mock_connection
    return hook


# =============================================================================
# Patch Fixtures
# =============================================================================


@pytest.fixture
def patch_base_hook() -> Any:
    """Patch BaseHook.get_connection."""
    with patch("airflow.hooks.base.BaseHook.get_connection") as mock:
        yield mock


@pytest.fixture
def patch_engine() -> Any:
    """Patch the default engine registry."""
    with patch("common.engines.get_engine") as mock:
        yield mock


# =============================================================================
# SLA Fixtures
# =============================================================================


@pytest.fixture
def sample_sla_config() -> dict[str, Any]:
    """Sample SLA configuration."""
    return {
        "max_failure_rate": 0.05,
        "min_pass_rate": 0.95,
        "max_execution_time_seconds": 300.0,
        "max_consecutive_failures": 3,
    }


@pytest.fixture
def sample_sla_metrics() -> dict[str, Any]:
    """Sample SLA metrics."""
    return {
        "passed_count": 95,
        "failed_count": 5,
        "warning_count": 2,
        "execution_time_ms": 1500.0,
        "row_count": 10000,
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def temp_parquet_file(tmp_path: Any, sample_dataframe: Any) -> Any:
    """Create a temporary parquet file with sample data."""
    if not HAS_POLARS:
        pytest.skip("polars not installed")

    file_path = tmp_path / "test_data.parquet"
    sample_dataframe.write_parquet(file_path)
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path: Any, sample_dataframe: Any) -> Any:
    """Create a temporary CSV file with sample data."""
    if not HAS_POLARS:
        pytest.skip("polars not installed")

    file_path = tmp_path / "test_data.csv"
    sample_dataframe.write_csv(file_path)
    return file_path
