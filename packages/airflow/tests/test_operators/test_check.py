"""Tests for DataQualityCheckOperator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestDataQualityCheckOperator:
    """Tests for DataQualityCheckOperator."""

    def test_initialization_with_rules_and_path(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test initialization with rules and data path."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        op = DataQualityCheckOperator(
            task_id="test_check",
            rules=sample_rules,
            data_path="/data/test.parquet",
        )

        assert op.task_id == "test_check"
        assert len(op.rules) == 3
        assert op.data_path == "/data/test.parquet"
        assert op.sql is None

    def test_initialization_with_sql(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test initialization with SQL query."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        op = DataQualityCheckOperator(
            task_id="test_check",
            rules=sample_rules,
            sql="SELECT * FROM users",
        )

        assert op.sql == "SELECT * FROM users"
        assert op.data_path is None

    def test_initialization_with_warning_threshold(
        self,
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test initialization with warning threshold."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        op = DataQualityCheckOperator(
            task_id="test_check",
            rules=sample_rules,
            data_path="/data/test.parquet",
            warning_threshold=0.05,
        )

        assert op.warning_threshold == 0.05

    def test_invalid_warning_threshold_raises(
        self,
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test that invalid warning threshold raises error."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        with pytest.raises(ValueError, match="warning_threshold must be between 0 and 1"):
            DataQualityCheckOperator(
                task_id="test_check",
                rules=sample_rules,
                data_path="/data/test.parquet",
                warning_threshold=1.5,
            )

    def test_template_fields_include_rules(self) -> None:
        """Test that rules is in template_fields."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        op = DataQualityCheckOperator(
            task_id="test_check",
            rules=[{"type": "not_null", "column": "{{ params.column }}"}],
            data_path="/data/test.parquet",
        )

        assert "rules" in op.template_fields

    def test_ui_color(self) -> None:
        """Test UI color is correctly defined."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        assert hasattr(DataQualityCheckOperator, "ui_color")
        assert DataQualityCheckOperator.ui_color.startswith("#")

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_success(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_success_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test successful execution."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        # Setup mock
        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check.return_value = mock_success_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityCheckOperator(
                task_id="test_check",
                rules=sample_rules,
                data_path="/data/test.parquet",
            )

            result = op.execute(airflow_context)

            assert result is not None
            assert "status" in result
            mock_hook.load_data.assert_called_once()
            airflow_context["ti"].xcom_push.assert_called()

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_failure_raises(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_failure_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test that execution failure raises AirflowException."""
        from airflow.exceptions import AirflowException

        from truthound_airflow.operators.check import DataQualityCheckOperator

        # Setup mock
        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check.return_value = mock_failure_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityCheckOperator(
                task_id="test_check",
                rules=sample_rules,
                data_path="/data/test.parquet",
                fail_on_error=True,
            )

            with pytest.raises(AirflowException, match="Quality check failed"):
                op.execute(airflow_context)

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_failure_no_raise_when_disabled(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_failure_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test that failure doesn't raise when fail_on_error is False."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check.return_value = mock_failure_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityCheckOperator(
                task_id="test_check",
                rules=sample_rules,
                data_path="/data/test.parquet",
                fail_on_error=False,
            )

            # Should not raise
            result = op.execute(airflow_context)
            assert result is not None

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_with_warning_threshold(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_warning_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test execution with warning threshold allowing minor failures."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check.return_value = mock_warning_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityCheckOperator(
                task_id="test_check",
                rules=sample_rules,
                data_path="/data/test.parquet",
                warning_threshold=0.05,  # 5% threshold
                fail_on_error=True,
            )

            # Should not raise because failure_rate (4%) < threshold (5%)
            result = op.execute(airflow_context)
            assert result is not None

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_with_sampling(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_success_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test execution with sampling."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.check.return_value = mock_success_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityCheckOperator(
                task_id="test_check",
                rules=sample_rules,
                data_path="/data/test.parquet",
                sample_size=2,
            )

            result = op.execute(airflow_context)
            assert result is not None

    def test_xcom_push_key_customization(
        self,
        sample_rules: list[dict[str, Any]],
    ) -> None:
        """Test custom XCom push key."""
        from truthound_airflow.operators.check import DataQualityCheckOperator

        op = DataQualityCheckOperator(
            task_id="test_check",
            rules=sample_rules,
            data_path="/data/test.parquet",
            xcom_push_key="custom_result_key",
        )

        assert op.xcom_push_key == "custom_result_key"


class TestTruthoundCheckOperatorAlias:
    """Tests for TruthoundCheckOperator legacy alias."""

    def test_alias_exists(self) -> None:
        """Test that legacy alias exists."""
        from truthound_airflow.operators.check import (
            DataQualityCheckOperator,
            TruthoundCheckOperator,
        )

        assert TruthoundCheckOperator is DataQualityCheckOperator

    def test_alias_works(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test that legacy alias works correctly."""
        from truthound_airflow.operators.check import TruthoundCheckOperator

        op = TruthoundCheckOperator(
            task_id="test_check",
            rules=sample_rules,
            data_path="/data/test.parquet",
        )

        assert op.task_id == "test_check"
