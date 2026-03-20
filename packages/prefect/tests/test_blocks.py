"""Tests for truthound_prefect.blocks module."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.testing import MockDataQualityEngine
from truthound_prefect.blocks.base import (
    BaseBlock,
    BlockConfig,
    DEFAULT_BLOCK_CONFIG,
    PRODUCTION_BLOCK_CONFIG,
)
from truthound_prefect.blocks.engine import (
    AUTO_SCHEMA_ENGINE_CONFIG,
    DEFAULT_ENGINE_CONFIG,
    PARALLEL_ENGINE_CONFIG,
    DataQualityBlock,
    EngineBlock,
    EngineBlockConfig,
    create_ephemeral_truthound_block,
)


class TestBlockConfig:
    """Tests for BlockConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BlockConfig()
        assert config.enabled is True
        assert config.timeout_seconds == 300.0
        assert config.tags == frozenset()
        assert config.description == ""

    def test_immutability(self) -> None:
        """Test that config is immutable."""
        config = BlockConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore

    def test_with_enabled(self) -> None:
        """Test with_enabled builder method."""
        config = BlockConfig()
        new_config = config.with_enabled(False)
        assert config.enabled is True  # Original unchanged
        assert new_config.enabled is False

    def test_with_timeout(self) -> None:
        """Test with_timeout builder method."""
        config = BlockConfig()
        new_config = config.with_timeout(600.0)
        assert new_config.timeout_seconds == 600.0

    def test_with_tags(self) -> None:
        """Test with_tags builder method."""
        config = BlockConfig()
        new_config = config.with_tags("production", "critical")
        assert "production" in new_config.tags
        assert "critical" in new_config.tags

    def test_with_description(self) -> None:
        """Test with_description builder method."""
        config = BlockConfig()
        new_config = config.with_description("Test config")
        assert new_config.description == "Test config"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        config = BlockConfig(
            enabled=True,
            timeout_seconds=600.0,
            tags=frozenset(["test"]),
            description="Test",
        )
        d = config.to_dict()
        assert d["enabled"] is True
        assert d["timeout_seconds"] == 600.0
        assert "test" in d["tags"]

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {
            "enabled": False,
            "timeout_seconds": 600.0,
            "tags": ["test"],
            "description": "Test",
        }
        config = BlockConfig.from_dict(d)
        assert config.enabled is False
        assert config.timeout_seconds == 600.0


class TestEngineBlockConfig:
    """Tests for EngineBlockConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EngineBlockConfig()
        assert config.engine_name == "truthound"
        assert config.parallel is False
        assert config.max_workers is None
        assert config.auto_start is True
        assert config.auto_stop is True
        assert config.auto_schema is False
        assert config.fail_on_error is True

    def test_with_engine(self) -> None:
        """Test with_engine builder method."""
        config = EngineBlockConfig()
        new_config = config.with_engine("great_expectations")
        assert new_config.engine_name == "great_expectations"

    def test_with_parallel(self) -> None:
        """Test with_parallel builder method."""
        config = EngineBlockConfig()
        new_config = config.with_parallel(True, max_workers=8)
        assert new_config.parallel is True
        assert new_config.max_workers == 8

    def test_with_auto_schema(self) -> None:
        """Test with_auto_schema builder method."""
        config = EngineBlockConfig()
        new_config = config.with_auto_schema(True)
        assert new_config.auto_schema is True

    def test_with_fail_on_error(self) -> None:
        """Test with_fail_on_error builder method."""
        config = EngineBlockConfig()
        new_config = config.with_fail_on_error(False)
        assert new_config.fail_on_error is False

    def test_with_lifecycle(self) -> None:
        """Test with_lifecycle builder method."""
        config = EngineBlockConfig()
        new_config = config.with_lifecycle(auto_start=False, auto_stop=False)
        assert new_config.auto_start is False
        assert new_config.auto_stop is False


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_default_block_config(self) -> None:
        """Test DEFAULT_BLOCK_CONFIG preset."""
        assert DEFAULT_BLOCK_CONFIG.enabled is True

    def test_production_block_config(self) -> None:
        """Test PRODUCTION_BLOCK_CONFIG preset."""
        assert PRODUCTION_BLOCK_CONFIG.timeout_seconds == 600.0
        assert "production" in PRODUCTION_BLOCK_CONFIG.tags

    def test_default_engine_config(self) -> None:
        """Test DEFAULT_ENGINE_CONFIG preset."""
        assert DEFAULT_ENGINE_CONFIG.engine_name == "truthound"

    def test_parallel_engine_config(self) -> None:
        """Test PARALLEL_ENGINE_CONFIG preset."""
        assert PARALLEL_ENGINE_CONFIG.parallel is True
        assert PARALLEL_ENGINE_CONFIG.max_workers == 4

    def test_auto_schema_engine_config(self) -> None:
        """Test AUTO_SCHEMA_ENGINE_CONFIG preset."""
        assert AUTO_SCHEMA_ENGINE_CONFIG.auto_schema is True
        assert AUTO_SCHEMA_ENGINE_CONFIG.engine_name == "truthound"


class TestEngineBlock:
    """Tests for EngineBlock."""

    def test_default_config(self) -> None:
        """Test that EngineBlock uses default config."""
        block = EngineBlock()
        assert block.config.engine_name == "truthound"

    def test_custom_config(self) -> None:
        """Test EngineBlock with custom config."""
        config = EngineBlockConfig(engine_name="pandera")
        block = EngineBlock(config)
        assert block.config.engine_name == "pandera"

    def test_is_initialized_false_initially(self) -> None:
        """Test that block is not initialized before setup."""
        block = EngineBlock()
        assert block.is_initialized is False

    def test_engine_not_available_before_setup(self) -> None:
        """Test that engine raises error before setup."""
        block = EngineBlock()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = block.engine

    def test_provided_engine(self, mock_engine: MagicMock) -> None:
        """Test using a pre-configured engine."""
        block = EngineBlock(engine=mock_engine)
        block.setup()
        assert block.engine == mock_engine
        assert block.is_initialized is True
        block.teardown()

    def test_context_manager(self, mock_engine: MagicMock) -> None:
        """Test using block as context manager."""
        block = EngineBlock(engine=mock_engine)
        with block:
            assert block.is_initialized is True
            assert block.engine == mock_engine
        assert block.is_initialized is False

    def test_create_truthound_engine(self) -> None:
        """Test creating a Truthound engine using mock."""
        mock_engine = MagicMock()
        mock_engine.engine_name = "truthound"
        mock_engine.engine_version = "1.0.0"

        config = EngineBlockConfig(engine_name="truthound")
        block = EngineBlock(config)

        # Mock the _create_engine method to return our mock
        with patch.object(block, "_create_engine", return_value=mock_engine):
            block.setup()
            assert block.is_initialized is True
            assert block.engine == mock_engine
            block.teardown()

    def test_create_engine_uses_common_resolver(self) -> None:
        """Test engine creation delegates to common resolver."""
        block = EngineBlock(
            EngineBlockConfig(
                engine_name="truthound",
                parallel=True,
                max_workers=2,
            )
        )
        mock_engine = MagicMock()
        mock_preflight = MagicMock()
        mock_preflight.compatible = True

        with (
            patch("common.engines.run_preflight", return_value=mock_preflight) as run_preflight,
            patch("common.engines.create_engine", return_value=mock_engine) as create_engine,
        ):
            engine = block._create_engine()

        assert engine == mock_engine
        run_preflight.assert_called_once()
        request = create_engine.call_args.args[0]
        assert request.engine_name == "truthound"
        assert request.runtime_context.platform == "prefect"
        create_engine.assert_called_once_with(
            request,
            auto_start=False,
            auto_stop=False,
            parallel=True,
            max_workers=2,
        )

    def test_engine_name_property(self, mock_engine: MagicMock) -> None:
        """Test engine_name property."""
        block = EngineBlock(engine=mock_engine)
        block.setup()
        assert block.engine_name == "mock_engine"
        block.teardown()

    def test_engine_version_property(self, mock_engine: MagicMock) -> None:
        """Test engine_version property."""
        block = EngineBlock(engine=mock_engine)
        block.setup()
        assert block.engine_version == "1.0.0"
        block.teardown()

    def test_create_ephemeral_truthound_block(self) -> None:
        """Test zero-config helper returns an in-memory Truthound block."""
        block = create_ephemeral_truthound_block(auto_schema=True, warning_threshold=0.1)

        assert isinstance(block, DataQualityBlock)
        assert block.engine_name == "truthound"
        assert block.auto_schema is True
        assert block.warning_threshold == 0.1

    def test_engine_block_stream(self) -> None:
        """EngineBlock should expose shared streaming batches and summary."""
        engine = MockDataQualityEngine()
        engine.configure_check(success=True, passed_count=1)
        block = EngineBlock(engine=engine)
        block.setup()

        result = block.stream(iter([{"id": 1}, {"id": 2}, {"id": 3}]), rules=[], batch_size=2)

        assert result["summary"]["total_batches"] == 2
        assert len(result["batches"]) == 2
        block.teardown()

    @pytest.mark.asyncio
    async def test_data_quality_block_astream(self) -> None:
        """DataQualityBlock should expose async streaming helper."""
        engine = MockDataQualityEngine()
        engine.configure_check(success=True, passed_count=1)
        block = DataQualityBlock()
        block._engine_block = EngineBlock(engine=engine)
        block._engine_block.setup()

        async def async_stream():
            for row in [{"id": 1}, {"id": 2}]:
                yield row

        result = await block.astream(async_stream(), rules=[], batch_size=1)

        assert result["summary"]["total_batches"] == 2

    def test_engine_block_emits_openlineage_host_execution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        openlineage_collector: Any,
    ) -> None:
        """Prefect block should emit shared host execution metadata."""
        import prefect.runtime as prefect_runtime

        monkeypatch.setattr(
            prefect_runtime,
            "flow_run",
            type("FlowRun", (), {"id": "flow-123", "deployment_id": "deployment-123"})(),
            raising=False,
        )
        monkeypatch.setattr(
            prefect_runtime,
            "task_run",
            type("TaskRun", (), {"id": "task-123"})(),
            raising=False,
        )

        engine = MockDataQualityEngine(name="truthound", version="3.1.0")
        engine.configure_check(success=True, passed_count=1)
        block = EngineBlock(
            EngineBlockConfig(
                observability={
                    "backend": "openlineage",
                    "endpoint": openlineage_collector.endpoint,
                    "namespace": "truthound",
                    "job_name": "prefect-check",
                }
            ),
            engine=engine,
        )
        block.setup()

        try:
            block.check([{"id": 1}], rules=None)
        finally:
            block.teardown()

        assert len(openlineage_collector.received) == 2
        facet = openlineage_collector.received[-1]["run"]["facets"]["truthound"]
        assert facet["host_execution"]["flow_run_id"] == "flow-123"
        assert facet["host_execution"]["task_run_id"] == "task-123"
        assert facet["host_execution"]["deployment_id"] == "deployment-123"
