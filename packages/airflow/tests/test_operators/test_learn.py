"""Tests for DataQualityLearnOperator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestDataQualityLearnOperator:
    """Tests for DataQualityLearnOperator."""

    def test_initialization_with_path(self) -> None:
        """Test initialization with data path."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        op = DataQualityLearnOperator(
            task_id="test_learn",
            data_path="/data/test.parquet",
        )

        assert op.task_id == "test_learn"
        assert op.data_path == "/data/test.parquet"
        assert op.sql is None

    def test_initialization_with_output_path(self) -> None:
        """Test initialization with output path."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        op = DataQualityLearnOperator(
            task_id="test_learn",
            data_path="/data/test.parquet",
            output_path="/schemas/learned_schema.json",
        )

        assert op.output_path == "/schemas/learned_schema.json"

    def test_initialization_with_strictness(self) -> None:
        """Test initialization with strictness level."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        op = DataQualityLearnOperator(
            task_id="test_learn",
            data_path="/data/test.parquet",
            strictness="strict",
        )

        assert op.strictness == "strict"

    def test_invalid_strictness_raises(self) -> None:
        """Test that invalid strictness raises error."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        with pytest.raises(ValueError, match="strictness must be one of"):
            DataQualityLearnOperator(
                task_id="test_learn",
                data_path="/data/test.parquet",
                strictness="invalid",
            )

    def test_template_fields_include_output_path(self) -> None:
        """Test that output_path is in template_fields."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        op = DataQualityLearnOperator(
            task_id="test_learn",
            data_path="/data/test.parquet",
        )

        assert "output_path" in op.template_fields

    def test_ui_color(self) -> None:
        """Test UI color is correctly defined."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        assert hasattr(DataQualityLearnOperator, "ui_color")
        assert DataQualityLearnOperator.ui_color.startswith("#")

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_success(
        self,
        mock_hook_class: MagicMock,
        sample_dataframe: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test successful execution."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_learn_result = MagicMock()
            mock_learn_result.rules = [
                MagicMock(
                    rule_type="not_null",
                    column="id",
                    confidence=0.95,
                    to_dict=lambda: {"type": "not_null", "column": "id"},
                ),
            ]
            mock_learn_result.to_dict.return_value = {
                "rules": [{"type": "not_null", "column": "id"}],
            }
            mock_engine.learn.return_value = mock_learn_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityLearnOperator(
                task_id="test_learn",
                data_path="/data/test.parquet",
            )

            result = op.execute(airflow_context)

            assert result is not None
            mock_hook.load_data.assert_called_once()
            airflow_context["ti"].xcom_push.assert_called()

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_execute_with_output_path(
        self,
        mock_hook_class: MagicMock,
        sample_dataframe: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test execution with output path saves schema."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        with patch("common.engines.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_learn_result = MagicMock()
            mock_learn_result.rules = []
            mock_learn_result.to_dict.return_value = {"rules": []}
            mock_engine.learn.return_value = mock_learn_result
            mock_get_engine.return_value = mock_engine

            op = DataQualityLearnOperator(
                task_id="test_learn",
                data_path="/data/test.parquet",
                output_path="/schemas/learned.json",
            )

            result = op.execute(airflow_context)

            assert result is not None
            mock_hook.save_json.assert_called_once()

    def test_xcom_push_key_default(self) -> None:
        """Test default XCom push key for learn."""
        from truthound_airflow.operators.learn import DataQualityLearnOperator

        op = DataQualityLearnOperator(
            task_id="test_learn",
            data_path="/data/test.parquet",
        )

        assert op.xcom_push_key == "data_quality_schema"


class TestTruthoundLearnOperatorAlias:
    """Tests for TruthoundLearnOperator legacy alias."""

    def test_alias_exists(self) -> None:
        """Test that legacy alias exists."""
        from truthound_airflow.operators.learn import (
            DataQualityLearnOperator,
            TruthoundLearnOperator,
        )

        assert TruthoundLearnOperator is DataQualityLearnOperator
