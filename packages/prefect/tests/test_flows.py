"""Tests for truthound_prefect.flows module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound_prefect.flows.config import (
    AUTO_SCHEMA_FLOW_CONFIG,
    DEFAULT_FLOW_CONFIG,
    DEFAULT_PIPELINE_CONFIG,
    DEFAULT_QUALITY_FLOW_CONFIG,
    FULL_PIPELINE_CONFIG,
    LENIENT_QUALITY_FLOW_CONFIG,
    STRICT_QUALITY_FLOW_CONFIG,
    FlowConfig,
    PipelineFlowConfig,
    QualityFlowConfig,
)
from truthound_prefect.flows.decorators import (
    profiled_flow,
    quality_checked_flow,
    validated_flow,
)
from truthound_prefect.flows.factories import (
    create_multi_table_quality_flows,
    create_pipeline_flow,
    create_quality_flow,
    create_validation_flow,
)


class TestFlowConfig:
    """Tests for FlowConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FlowConfig()
        assert config.enabled is True
        assert config.name == ""
        assert config.description == ""
        assert config.timeout_seconds is None
        assert config.retries == 0
        assert config.retry_delay_seconds == 60.0
        assert config.tags == frozenset()
        assert config.log_prints is True
        assert config.validate_parameters is True

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = FlowConfig()
        with pytest.raises(AttributeError):
            config.name = "test"  # type: ignore

    def test_with_name(self) -> None:
        """Test with_name builder method."""
        config = FlowConfig()
        new_config = config.with_name("test_flow")
        assert new_config.name == "test_flow"

    def test_with_timeout(self) -> None:
        """Test with_timeout builder method."""
        config = FlowConfig()
        new_config = config.with_timeout(7200.0)
        assert new_config.timeout_seconds == 7200.0

    def test_with_retries(self) -> None:
        """Test with_retries builder method."""
        config = FlowConfig()
        new_config = config.with_retries(3, retry_delay_seconds=120.0)
        assert new_config.retries == 3
        assert new_config.retry_delay_seconds == 120.0

    def test_with_tags(self) -> None:
        """Test with_tags builder method."""
        config = FlowConfig()
        new_config = config.with_tags("production", "critical")
        assert "production" in new_config.tags
        assert "critical" in new_config.tags


class TestQualityFlowConfig:
    """Tests for QualityFlowConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = QualityFlowConfig()
        assert config.fail_on_error is True
        assert config.auto_schema is False
        assert config.store_results is True
        assert config.rules == ()
        assert config.engine_name == "truthound"

    def test_with_fail_on_error(self) -> None:
        """Test with_fail_on_error builder method."""
        config = QualityFlowConfig()
        new_config = config.with_fail_on_error(False)
        assert new_config.fail_on_error is False

    def test_with_auto_schema(self) -> None:
        """Test with_auto_schema builder method."""
        config = QualityFlowConfig()
        new_config = config.with_auto_schema(True)
        assert new_config.auto_schema is True

    def test_with_engine(self) -> None:
        """Test with_engine builder method."""
        config = QualityFlowConfig()
        new_config = config.with_engine("great_expectations")
        assert new_config.engine_name == "great_expectations"

    def test_with_rules(self) -> None:
        """Test with_rules builder method."""
        rules = [{"type": "not_null", "column": "id"}]
        config = QualityFlowConfig()
        new_config = config.with_rules(rules)
        assert len(new_config.rules) == 1


class TestPipelineFlowConfig:
    """Tests for PipelineFlowConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PipelineFlowConfig()
        assert config.profile_data is False
        assert config.learn_rules is False
        assert config.parallel_checks is False
        assert config.max_workers == 4

    def test_with_profiling(self) -> None:
        """Test with_profiling builder method."""
        config = PipelineFlowConfig()
        new_config = config.with_profiling(True)
        assert new_config.profile_data is True

    def test_with_learning(self) -> None:
        """Test with_learning builder method."""
        config = PipelineFlowConfig()
        new_config = config.with_learning(True)
        assert new_config.learn_rules is True

    def test_with_parallel(self) -> None:
        """Test with_parallel builder method."""
        config = PipelineFlowConfig()
        new_config = config.with_parallel(True, max_workers=8)
        assert new_config.parallel_checks is True
        assert new_config.max_workers == 8


class TestPresetFlowConfigs:
    """Tests for preset flow configurations."""

    def test_default_flow_config(self) -> None:
        """Test DEFAULT_FLOW_CONFIG preset."""
        assert DEFAULT_FLOW_CONFIG.enabled is True
        assert DEFAULT_FLOW_CONFIG.timeout_seconds is None

    def test_default_quality_flow_config(self) -> None:
        """Test DEFAULT_QUALITY_FLOW_CONFIG preset."""
        assert DEFAULT_QUALITY_FLOW_CONFIG.fail_on_error is True

    def test_strict_quality_flow_config(self) -> None:
        """Test STRICT_QUALITY_FLOW_CONFIG preset."""
        assert STRICT_QUALITY_FLOW_CONFIG.fail_on_error is True
        assert "strict" in STRICT_QUALITY_FLOW_CONFIG.tags

    def test_lenient_quality_flow_config(self) -> None:
        """Test LENIENT_QUALITY_FLOW_CONFIG preset."""
        assert LENIENT_QUALITY_FLOW_CONFIG.fail_on_error is False
        assert LENIENT_QUALITY_FLOW_CONFIG.warning_threshold == 0.10

    def test_auto_schema_flow_config(self) -> None:
        """Test AUTO_SCHEMA_FLOW_CONFIG preset."""
        assert AUTO_SCHEMA_FLOW_CONFIG.auto_schema is True
        assert AUTO_SCHEMA_FLOW_CONFIG.engine_name == "truthound"

    def test_default_pipeline_config(self) -> None:
        """Test DEFAULT_PIPELINE_CONFIG preset."""
        assert DEFAULT_PIPELINE_CONFIG.profile_data is False

    def test_full_pipeline_config(self) -> None:
        """Test FULL_PIPELINE_CONFIG preset."""
        assert FULL_PIPELINE_CONFIG.profile_data is True
        assert FULL_PIPELINE_CONFIG.parallel_checks is True


class TestFlowDecorators:
    """Tests for flow decorators."""

    def test_quality_checked_flow_decorator(self) -> None:
        """Test quality_checked_flow decorator."""
        @quality_checked_flow(
            rules=[{"type": "not_null", "column": "id"}],
            fail_on_error=True,
        )
        async def my_flow():
            return {"data": "test"}

        # Check decorator was applied
        assert callable(my_flow)

    def test_profiled_flow_decorator(self) -> None:
        """Test profiled_flow decorator."""
        @profiled_flow(include_histograms=True)
        async def my_flow():
            return {"data": "test"}

        assert callable(my_flow)

    def test_validated_flow_decorator(self) -> None:
        """Test validated_flow decorator."""
        @validated_flow(
            check_before=True,
            check_after=True,
            rules=[{"type": "not_null", "column": "id"}],
        )
        async def my_flow(data):
            return data

        assert callable(my_flow)


class TestFlowFactories:
    """Tests for flow factory functions."""

    def test_create_quality_flow(self) -> None:
        """Test create_quality_flow factory."""
        def loader():
            return {"data": "test"}

        flow = create_quality_flow(
            name="test_quality_flow",
            loader=loader,
            config=STRICT_QUALITY_FLOW_CONFIG,
        )
        assert flow is not None
        assert callable(flow)

    def test_create_validation_flow(self) -> None:
        """Test create_validation_flow factory."""
        def source():
            return {"data": "test"}

        flow = create_validation_flow(
            name="test_validation_flow",
            source=source,
            rules=[{"type": "not_null", "column": "id"}],
        )
        assert flow is not None
        assert callable(flow)

    def test_create_pipeline_flow(self) -> None:
        """Test create_pipeline_flow factory."""
        stages = [
            {"name": "extract", "loader": lambda: {"data": "raw"}},
            {"name": "transform", "loader": lambda d: {"data": "transformed"}},
            {"name": "load", "loader": lambda d: {"data": "loaded"}, "check": False},
        ]
        flow = create_pipeline_flow(
            name="test_pipeline",
            stages=stages,
            config=DEFAULT_PIPELINE_CONFIG,
        )
        assert flow is not None
        assert callable(flow)

    def test_create_multi_table_quality_flows(self) -> None:
        """Test create_multi_table_quality_flows factory."""
        tables = {
            "users": {
                "loader": lambda: {"data": "users"},
                "rules": [{"type": "not_null", "column": "id"}],
            },
            "orders": {
                "loader": lambda: {"data": "orders"},
                "rules": [{"type": "not_null", "column": "order_id"}],
            },
        }
        flows = create_multi_table_quality_flows(
            tables=tables,
            group_name="validate",
        )
        assert len(flows) == 2
        assert "users" in flows
        assert "orders" in flows
        assert all(callable(f) for f in flows.values())
