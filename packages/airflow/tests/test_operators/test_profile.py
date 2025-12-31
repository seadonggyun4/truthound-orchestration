"""Tests for DataQualityProfileOperator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestDataQualityProfileOperator:
    """Tests for DataQualityProfileOperator."""

    def test_initialization_with_path(self) -> None:
        """Test initialization with data path."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        op = DataQualityProfileOperator(
            task_id="test_profile",
            data_path="/data/test.parquet",
        )

        assert op.task_id == "test_profile"
        assert op.data_path == "/data/test.parquet"
        assert op.sql is None

    def test_initialization_with_columns(self) -> None:
        """Test initialization with specific columns."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        op = DataQualityProfileOperator(
            task_id="test_profile",
            data_path="/data/test.parquet",
            columns=["col1", "col2", "col3"],
        )

        assert op.columns == ["col1", "col2", "col3"]

    def test_initialization_with_options(self) -> None:
        """Test initialization with profiling options."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        op = DataQualityProfileOperator(
            task_id="test_profile",
            data_path="/data/test.parquet",
            include_statistics=True,
            include_patterns=False,
            include_distributions=True,
        )

        assert op.include_statistics is True
        assert op.include_patterns is False
        assert op.include_distributions is True

    def test_template_fields_include_columns(self) -> None:
        """Test that columns is in template_fields."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        op = DataQualityProfileOperator(
            task_id="test_profile",
            data_path="/data/test.parquet",
        )

        assert "columns" in op.template_fields

    def test_ui_color(self) -> None:
        """Test UI color is correctly defined."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        assert hasattr(DataQualityProfileOperator, "ui_color")
        assert DataQualityProfileOperator.ui_color.startswith("#")

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_success(
        self,
        mock_hook_class: MagicMock,
        sample_dataframe: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test successful execution."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_profile_result = MagicMock()
            mock_profile_result.columns = []
            mock_profile_result.row_count = 100
            mock_profile_result.to_dict.return_value = {
                "row_count": 100,
                "column_count": 5,
                "columns": [],
            }
            mock_engine.profile.return_value = mock_profile_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityProfileOperator(
                task_id="test_profile",
                data_path="/data/test.parquet",
            )

            result = op.execute(airflow_context)

            assert result is not None
            assert "row_count" in result
            mock_hook.load_data.assert_called_once()
            airflow_context["ti"].xcom_push.assert_called()

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_with_sql(
        self,
        mock_hook_class: MagicMock,
        sample_dataframe: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test execution with SQL query."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        mock_hook = MagicMock()
        mock_hook.query.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_profile_result = MagicMock()
            mock_profile_result.to_dict.return_value = {"row_count": 100, "columns": []}
            mock_engine.profile.return_value = mock_profile_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityProfileOperator(
                task_id="test_profile",
                sql="SELECT * FROM users",
            )

            result = op.execute(airflow_context)

            assert result is not None
            mock_hook.query.assert_called_once()

    def test_xcom_push_key_default(self) -> None:
        """Test default XCom push key for profile."""
        from truthound_airflow.operators.profile import DataQualityProfileOperator

        op = DataQualityProfileOperator(
            task_id="test_profile",
            data_path="/data/test.parquet",
        )

        assert op.xcom_push_key == "data_quality_profile"


class TestTruthoundProfileOperatorAlias:
    """Tests for TruthoundProfileOperator legacy alias."""

    def test_alias_exists(self) -> None:
        """Test that legacy alias exists."""
        from truthound_airflow.operators.profile import (
            DataQualityProfileOperator,
            TruthoundProfileOperator,
        )

        assert TruthoundProfileOperator is DataQualityProfileOperator
