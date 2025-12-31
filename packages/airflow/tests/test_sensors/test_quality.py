"""Tests for DataQualitySensor."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestSensorConfig:
    """Tests for SensorConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        from truthound_airflow.sensors.quality import SensorConfig

        config = SensorConfig()

        assert config.min_pass_rate == 1.0
        assert config.min_row_count is None
        assert config.max_failure_count is None
        assert config.check_data_exists is True
        assert config.continue_on_error is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        from truthound_airflow.sensors.quality import SensorConfig

        config = SensorConfig(
            min_pass_rate=0.95,
            min_row_count=100,
            max_failure_count=10,
            check_data_exists=False,
            continue_on_error=True,
        )

        assert config.min_pass_rate == 0.95
        assert config.min_row_count == 100
        assert config.max_failure_count == 10
        assert config.check_data_exists is False
        assert config.continue_on_error is True

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        from truthound_airflow.sensors.quality import SensorConfig

        config = SensorConfig()

        with pytest.raises(AttributeError):
            config.min_pass_rate = 0.5  # type: ignore

    def test_validation_min_pass_rate_range(self) -> None:
        """Test validation of min_pass_rate range."""
        from truthound_airflow.sensors.quality import SensorConfig

        with pytest.raises(ValueError, match="min_pass_rate"):
            SensorConfig(min_pass_rate=1.5)

        with pytest.raises(ValueError, match="min_pass_rate"):
            SensorConfig(min_pass_rate=-0.1)

    def test_validation_min_row_count_negative(self) -> None:
        """Test validation of min_row_count."""
        from truthound_airflow.sensors.quality import SensorConfig

        with pytest.raises(ValueError, match="min_row_count"):
            SensorConfig(min_row_count=-1)


class TestDataQualitySensor:
    """Tests for DataQualitySensor."""

    def test_initialization_with_rules(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test initialization with rules."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
        )

        assert sensor.task_id == "test_sensor"
        assert len(sensor.rules) == 3
        assert sensor.data_path == "/data/test.parquet"

    def test_initialization_with_min_pass_rate(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test initialization with min pass rate."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
            min_pass_rate=0.95,
        )

        assert sensor.min_pass_rate == 0.95

    def test_initialization_with_sql(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test initialization with SQL query."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            sql="SELECT * FROM my_table",
        )

        assert sensor.sql == "SELECT * FROM my_table"
        assert sensor.data_path is None

    def test_initialization_both_data_sources_raises(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test that specifying both data_path and sql raises error."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        with pytest.raises(ValueError, match="Cannot specify both"):
            DataQualitySensor(
                task_id="test_sensor",
                rules=sample_rules,
                data_path="/data/test.parquet",
                sql="SELECT * FROM my_table",
            )

    def test_initialization_no_data_source_raises(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test that not specifying data source raises error."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        with pytest.raises(ValueError, match="Must specify either"):
            DataQualitySensor(
                task_id="test_sensor",
                rules=sample_rules,
            )

    def test_template_fields(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test template fields are correctly defined."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/{{ ds }}/test.parquet",
        )

        assert "rules" in sensor.template_fields
        assert "data_path" in sensor.template_fields
        assert "sql" in sensor.template_fields

    def test_ui_color(self) -> None:
        """Test UI color is correctly defined."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        assert hasattr(DataQualitySensor, "ui_color")
        assert DataQualitySensor.ui_color.startswith("#")

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_poke_returns_true_on_success(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_success_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test poke returns True when quality conditions are met."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
            min_pass_rate=1.0,
        )

        # Set up the engine mock - directly setting _engine bypasses get_engine()
        mock_engine = MagicMock()
        mock_engine.check.return_value = mock_success_result
        sensor._engine = mock_engine

        result = sensor.poke(airflow_context)

        assert result is True

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_poke_returns_false_below_threshold(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_failure_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test poke returns False when below pass rate threshold."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
            min_pass_rate=0.95,
        )

        # Set up the engine mock
        mock_engine = MagicMock()
        mock_engine.check.return_value = mock_failure_result
        sensor._engine = mock_engine

        result = sensor.poke(airflow_context)

        assert result is False

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_poke_returns_false_on_file_not_found(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        airflow_context: dict[str, Any],
    ) -> None:
        """Test poke returns False when data file not found."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        mock_hook = MagicMock()
        mock_hook.load_data.side_effect = FileNotFoundError("Data not found")
        mock_hook_class.return_value = mock_hook

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
        )

        result = sensor.poke(airflow_context)

        assert result is False

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_poke_with_row_count_validation(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        mock_success_result: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test poke with row count validation."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe
        mock_hook_class.return_value = mock_hook

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
            min_row_count=1,  # Sample has 4-5 rows
        )

        # Set up the engine mock
        mock_engine = MagicMock()
        mock_engine.check.return_value = mock_success_result
        sensor._engine = mock_engine

        result = sensor.poke(airflow_context)

        assert result is True

    @patch("truthound_airflow.hooks.base.DataQualityHook")
    def test_poke_returns_false_below_min_row_count(
        self,
        mock_hook_class: MagicMock,
        sample_rules: list[dict[str, Any]],
        sample_dataframe: Any,
        airflow_context: dict[str, Any],
    ) -> None:
        """Test poke returns False when row count is below minimum."""
        from truthound_airflow.sensors.quality import DataQualitySensor

        mock_hook = MagicMock()
        mock_hook.load_data.return_value = sample_dataframe  # 4-5 rows
        mock_hook_class.return_value = mock_hook

        sensor = DataQualitySensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
            min_row_count=1000,  # Much higher than sample
        )

        result = sensor.poke(airflow_context)

        assert result is False


class TestTruthoundSensorAlias:
    """Tests for TruthoundSensor legacy alias."""

    def test_alias_exists(self) -> None:
        """Test that legacy alias exists."""
        from truthound_airflow.sensors.quality import (
            DataQualitySensor,
            TruthoundSensor,
        )

        assert TruthoundSensor is DataQualitySensor

    def test_alias_works(self, sample_rules: list[dict[str, Any]]) -> None:
        """Test that legacy alias works correctly."""
        from truthound_airflow.sensors.quality import TruthoundSensor

        sensor = TruthoundSensor(
            task_id="test_sensor",
            rules=sample_rules,
            data_path="/data/test.parquet",
        )

        assert sensor.task_id == "test_sensor"
