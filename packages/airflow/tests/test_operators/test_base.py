"""Tests for BaseDataQualityOperator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestOperatorConfig:
    """Tests for OperatorConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from truthound_airflow.operators.base import OperatorConfig

        config = OperatorConfig()

        assert config.connection_id == "truthound_default"
        assert config.fail_on_error is True
        assert config.timeout_seconds == 300
        assert config.xcom_push_key == "data_quality_result"
        assert config.tags == frozenset()
        assert config.extra == {}

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        from truthound_airflow.operators.base import OperatorConfig

        config = OperatorConfig(
            connection_id="custom_conn",
            fail_on_error=False,
            timeout_seconds=600,
            xcom_push_key="custom_key",
            tags=frozenset({"production", "critical"}),
            extra={"key": "value"},
        )

        assert config.connection_id == "custom_conn"
        assert config.fail_on_error is False
        assert config.timeout_seconds == 600
        assert config.xcom_push_key == "custom_key"
        assert "production" in config.tags
        assert config.extra == {"key": "value"}

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        from truthound_airflow.operators.base import OperatorConfig

        config = OperatorConfig()

        with pytest.raises(AttributeError):
            config.connection_id = "new_value"  # type: ignore

    def test_with_connection_id(self) -> None:
        """Test builder method for connection_id."""
        from truthound_airflow.operators.base import OperatorConfig

        config = OperatorConfig()
        new_config = config.with_connection_id("new_conn")

        assert config.connection_id == "truthound_default"
        assert new_config.connection_id == "new_conn"
        assert new_config.fail_on_error == config.fail_on_error

    def test_with_fail_on_error(self) -> None:
        """Test builder method for fail_on_error."""
        from truthound_airflow.operators.base import OperatorConfig

        config = OperatorConfig()
        new_config = config.with_fail_on_error(False)

        assert config.fail_on_error is True
        assert new_config.fail_on_error is False

    def test_with_timeout(self) -> None:
        """Test builder method for timeout."""
        from truthound_airflow.operators.base import OperatorConfig

        config = OperatorConfig()
        new_config = config.with_timeout(600)

        assert config.timeout_seconds == 300
        assert new_config.timeout_seconds == 600


class TestBaseDataQualityOperator:
    """Tests for BaseDataQualityOperator."""

    def test_initialization(self) -> None:
        """Test operator initialization."""
        from truthound_airflow.operators.base import BaseDataQualityOperator

        # Create a concrete subclass for testing
        class TestOperator(BaseDataQualityOperator):
            def _execute_operation(
                self,
                data: Any,
                context: dict[str, Any],
            ) -> Any:
                return MagicMock()

            def _serialize_result(self, result: Any) -> dict[str, Any]:
                return {"status": "success"}

        op = TestOperator(
            task_id="test_task",
            data_path="/data/test.parquet",
        )

        assert op.task_id == "test_task"
        assert op.data_path == "/data/test.parquet"
        assert op.sql is None
        assert op.connection_id == "truthound_default"

    def test_initialization_with_sql(self) -> None:
        """Test operator initialization with SQL."""
        from truthound_airflow.operators.base import BaseDataQualityOperator

        class TestOperator(BaseDataQualityOperator):
            def _execute_operation(
                self,
                data: Any,
                context: dict[str, Any],
            ) -> Any:
                return MagicMock()

            def _serialize_result(self, result: Any) -> dict[str, Any]:
                return {"status": "success"}

        op = TestOperator(
            task_id="test_task",
            sql="SELECT * FROM users",
        )

        assert op.sql == "SELECT * FROM users"
        assert op.data_path is None

    def test_initialization_with_parameters(self) -> None:
        """Test operator initialization with custom parameters."""
        from truthound_airflow.operators.base import BaseDataQualityOperator

        class TestOperator(BaseDataQualityOperator):
            def _execute_operation(
                self,
                data: Any,
                context: dict[str, Any],
            ) -> Any:
                return MagicMock()

            def _serialize_result(self, result: Any) -> dict[str, Any]:
                return {"status": "success"}

        op = TestOperator(
            task_id="test_task",
            data_path="/data/test.parquet",
            connection_id="custom_conn",
            fail_on_error=False,
            timeout_seconds=600,
        )

        assert op.connection_id == "custom_conn"
        assert op.fail_on_error is False
        assert op.timeout_seconds == 600

    def test_validation_both_path_and_sql(self) -> None:
        """Test that specifying both data_path and sql raises error."""
        from truthound_airflow.operators.base import BaseDataQualityOperator

        class TestOperator(BaseDataQualityOperator):
            def _execute_operation(
                self,
                data: Any,
                context: dict[str, Any],
            ) -> Any:
                return MagicMock()

            def _serialize_result(self, result: Any) -> dict[str, Any]:
                return {"status": "success"}

        with pytest.raises(ValueError, match="Cannot specify both data_path and sql"):
            TestOperator(
                task_id="test_task",
                data_path="/data/test.parquet",
                sql="SELECT * FROM users",
            )

    def test_validation_neither_path_nor_sql(self) -> None:
        """Test that specifying neither data_path nor sql raises error."""
        from truthound_airflow.operators.base import BaseDataQualityOperator

        class TestOperator(BaseDataQualityOperator):
            def _execute_operation(
                self,
                data: Any,
                context: dict[str, Any],
            ) -> Any:
                return MagicMock()

            def _serialize_result(self, result: Any) -> dict[str, Any]:
                return {"status": "success"}

        with pytest.raises(ValueError, match="Must specify either data_path or sql"):
            TestOperator(task_id="test_task")

    def test_template_fields(self) -> None:
        """Test template fields are correctly defined."""
        from truthound_airflow.operators.base import BaseDataQualityOperator

        class TestOperator(BaseDataQualityOperator):
            def _execute_operation(
                self,
                data: Any,
                context: dict[str, Any],
            ) -> Any:
                return MagicMock()

            def _serialize_result(self, result: Any) -> dict[str, Any]:
                return {"status": "success"}

        op = TestOperator(
            task_id="test_task",
            data_path="/data/{{ ds }}/test.parquet",
        )

        assert "data_path" in op.template_fields
        assert "sql" in op.template_fields
        assert "connection_id" in op.template_fields

    def test_ui_color(self) -> None:
        """Test UI color is correctly defined."""
        from truthound_airflow.operators.base import BaseDataQualityOperator

        assert hasattr(BaseDataQualityOperator, "ui_color")
        assert BaseDataQualityOperator.ui_color.startswith("#")
