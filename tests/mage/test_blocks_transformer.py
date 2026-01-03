"""Tests for Mage transformer blocks."""

from __future__ import annotations

from typing import Any

import pytest

from truthound_mage.blocks.transformer import (
    BaseDataQualityTransformer,
    DataQualityTransformer,
    CheckTransformer,
    ProfileTransformer,
    LearnTransformer,
    create_check_transformer,
    create_profile_transformer,
    create_learn_transformer,
)
from truthound_mage.blocks.base import (
    CheckBlockConfig,
    ProfileBlockConfig,
    LearnBlockConfig,
    BlockExecutionContext,
)
from common.base import CheckStatus
from common.testing import MockDataQualityEngine


@pytest.fixture
def mock_engine() -> MockDataQualityEngine:
    """Create a mock engine that returns successful results."""
    engine = MockDataQualityEngine()
    engine.configure_check(success=True)
    engine.configure_profile(success=True)
    engine.configure_learn(success=True)
    return engine


@pytest.fixture
def failing_mock_engine() -> MockDataQualityEngine:
    """Create a mock engine that returns failed results."""
    engine = MockDataQualityEngine()
    engine.configure_check(success=False, failed_count=5)
    engine.configure_profile(success=False)
    engine.configure_learn(success=False)
    return engine


@pytest.fixture
def sample_data() -> dict[str, Any]:
    """Create sample data for testing."""
    return {"id": [1, 2, 3], "name": ["a", "b", "c"]}


class TestCheckTransformer:
    """Tests for CheckTransformer."""

    def test_creation_with_config(self) -> None:
        """Test transformer creation with config."""
        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
            fail_on_error=True,
        )
        transformer = CheckTransformer(config=config)
        assert transformer.config.fail_on_error is True
        assert len(transformer.config.rules) == 1

    def test_creation_with_engine(self, mock_engine: MockDataQualityEngine) -> None:
        """Test transformer creation with engine."""
        transformer = CheckTransformer(engine=mock_engine)
        assert transformer._engine is mock_engine

    def test_execute_success(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test successful execution."""
        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
        )
        transformer = CheckTransformer(config=config, engine=mock_engine)

        result = transformer.execute(sample_data)

        assert result.success is True
        # BlockResult doesn't have operation field - check result_dict instead
        assert result.result is not None

    def test_execute_with_context(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test execution with context."""
        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
        )
        context = BlockExecutionContext(
            block_uuid="check_1",
            pipeline_uuid="pipeline_1",
        )
        transformer = CheckTransformer(config=config, engine=mock_engine)

        result = transformer.execute(sample_data, context=context)

        # BlockResult has metadata which contains block_uuid
        assert result.metadata.get("block_uuid") == "check_1"

    def test_execute_fail_on_error(
        self, failing_mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test execution with fail_on_error raises exception."""
        from truthound_mage.utils.exceptions import BlockExecutionError

        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
            fail_on_error=True,
        )
        transformer = CheckTransformer(config=config, engine=failing_mock_engine)

        # Should raise when fail_on_error=True and check fails
        with pytest.raises(BlockExecutionError):
            transformer.execute(sample_data)

    def test_config_with_warning_threshold(self) -> None:
        """Test config with warning threshold."""
        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
            warning_threshold=0.05,
        )
        transformer = CheckTransformer(config=config)
        assert transformer.config.warning_threshold == 0.05

    def test_config_with_parallel(self) -> None:
        """Test config with parallel setting."""
        config = CheckBlockConfig(
            rules=({"type": "not_null", "column": "id"},),
            parallel=False,
        )
        transformer = CheckTransformer(config=config)
        assert transformer.config.parallel is False

    def test_config_with_auto_schema(self) -> None:
        """Test config with auto_schema setting."""
        config = CheckBlockConfig(
            rules=(),
            auto_schema=True,
        )
        transformer = CheckTransformer(config=config)
        assert transformer.config.auto_schema is True


class TestProfileTransformer:
    """Tests for ProfileTransformer."""

    def test_creation_with_config(self) -> None:
        """Test transformer creation with config."""
        config = ProfileBlockConfig(
            include_statistics=True,
            include_distributions=True,
        )
        transformer = ProfileTransformer(config=config)
        assert transformer.config.include_statistics is True
        assert transformer.config.include_distributions is True

    def test_creation_with_columns(self) -> None:
        """Test transformer creation with specific columns."""
        config = ProfileBlockConfig(
            columns=frozenset(["id", "name"]),
        )
        transformer = ProfileTransformer(config=config)
        assert transformer.config.columns == frozenset(["id", "name"])

    def test_creation_with_sample_size(self) -> None:
        """Test transformer creation with sample size."""
        config = ProfileBlockConfig(
            sample_size=1000,
        )
        transformer = ProfileTransformer(config=config)
        assert transformer.config.sample_size == 1000

    def test_execute_success(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test successful profiling."""
        transformer = ProfileTransformer(engine=mock_engine)

        result = transformer.execute(sample_data)

        assert result.success is True
        assert result.result is not None


class TestLearnTransformer:
    """Tests for LearnTransformer."""

    def test_creation_with_config(self) -> None:
        """Test transformer creation with config."""
        config = LearnBlockConfig(
            strictness="strict",
            infer_constraints=True,
        )
        transformer = LearnTransformer(config=config)
        assert transformer.config.strictness == "strict"
        assert transformer.config.infer_constraints is True

    def test_creation_with_output_path(self) -> None:
        """Test transformer creation with output path."""
        config = LearnBlockConfig(
            output_path="/tmp/schema.json",
        )
        transformer = LearnTransformer(config=config)
        assert transformer.config.output_path == "/tmp/schema.json"

    def test_creation_with_categorical_threshold(self) -> None:
        """Test transformer creation with categorical threshold."""
        config = LearnBlockConfig(
            categorical_threshold=50,
        )
        transformer = LearnTransformer(config=config)
        assert transformer.config.categorical_threshold == 50

    def test_execute_success(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test successful learning."""
        transformer = LearnTransformer(engine=mock_engine)

        result = transformer.execute(sample_data)

        assert result.success is True
        assert result.result is not None


class TestDataQualityTransformer:
    """Tests for DataQualityTransformer facade."""

    def test_check_operation(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test check operation via facade."""
        # DataQualityTransformer.check() requires rules to be passed as argument
        # The config passed to constructor is for base settings only
        transformer = DataQualityTransformer(engine=mock_engine)

        rules = [{"type": "not_null", "column": "id"}]
        result = transformer.check(sample_data, rules=rules)

        assert result.success is True
        assert result.result is not None

    def test_profile_operation(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test profile operation via facade."""
        config = ProfileBlockConfig()
        transformer = DataQualityTransformer(config=config, engine=mock_engine)

        result = transformer.profile(sample_data)

        assert result.success is True
        assert result.result is not None

    def test_learn_operation(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test learn operation via facade."""
        config = LearnBlockConfig()
        transformer = DataQualityTransformer(config=config, engine=mock_engine)

        result = transformer.learn(sample_data)

        assert result.success is True
        assert result.result is not None

    def test_check_with_rules(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test check with rules passed to method."""
        transformer = DataQualityTransformer(engine=mock_engine)

        rules = [{"type": "not_null", "column": "id"}]
        result = transformer.check(sample_data, rules=rules)

        assert result.success is True

    def test_profile_with_columns(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test profile with columns passed to method."""
        transformer = DataQualityTransformer(engine=mock_engine)

        result = transformer.profile(sample_data, columns=["id", "name"])

        assert result.success is True


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_check_transformer(self) -> None:
        """Test check transformer factory."""
        rules = [{"type": "not_null", "column": "id"}]
        transformer = create_check_transformer(
            rules=rules,
            fail_on_error=False,
            warning_threshold=0.05,
        )

        assert len(transformer.config.rules) == 1
        assert transformer.config.fail_on_error is False
        assert transformer.config.warning_threshold == 0.05

    def test_create_check_transformer_with_timeout(self) -> None:
        """Test check transformer factory with timeout."""
        rules = [{"type": "not_null", "column": "id"}]
        transformer = create_check_transformer(
            rules=rules,
            timeout_seconds=60,
        )

        assert transformer.config.timeout_seconds == 60

    def test_create_profile_transformer(self) -> None:
        """Test profile transformer factory."""
        transformer = create_profile_transformer(
            include_statistics=True,
            include_distributions=True,
        )

        assert transformer.config.include_statistics is True
        assert transformer.config.include_distributions is True

    def test_create_profile_transformer_with_columns(self) -> None:
        """Test profile transformer factory with columns."""
        transformer = create_profile_transformer(
            columns=["amount", "quantity"],
        )

        assert transformer.config.columns == frozenset(["amount", "quantity"])

    def test_create_learn_transformer(self) -> None:
        """Test learn transformer factory."""
        transformer = create_learn_transformer(
            strictness="strict",
            infer_constraints=False,
        )

        assert transformer.config.strictness == "strict"
        assert transformer.config.infer_constraints is False

    def test_create_learn_transformer_with_output_path(self) -> None:
        """Test learn transformer factory with output path."""
        transformer = create_learn_transformer(
            output_path="/tmp/schema.json",
        )

        assert transformer.config.output_path == "/tmp/schema.json"


class TestTransformerDefaultConfig:
    """Tests for transformer default configurations."""

    def test_check_transformer_default_config(self) -> None:
        """Test CheckTransformer uses CheckBlockConfig by default."""
        transformer = CheckTransformer()
        assert isinstance(transformer.config, CheckBlockConfig)
        assert transformer.config.rules == ()
        assert transformer.config.parallel is True

    def test_profile_transformer_default_config(self) -> None:
        """Test ProfileTransformer uses ProfileBlockConfig by default."""
        transformer = ProfileTransformer()
        assert isinstance(transformer.config, ProfileBlockConfig)
        assert transformer.config.include_statistics is True
        assert transformer.config.columns is None

    def test_learn_transformer_default_config(self) -> None:
        """Test LearnTransformer uses LearnBlockConfig by default."""
        transformer = LearnTransformer()
        assert isinstance(transformer.config, LearnBlockConfig)
        assert transformer.config.strictness == "moderate"
        assert transformer.config.infer_constraints is True


class TestTransformerConfigBuilder:
    """Tests for transformer config builder methods."""

    def test_check_config_with_rules(self) -> None:
        """Test CheckBlockConfig.with_rules builder."""
        config = CheckBlockConfig().with_rules([{"type": "not_null", "column": "id"}])
        transformer = CheckTransformer(config=config)
        assert len(transformer.config.rules) == 1

    def test_profile_config_with_columns(self) -> None:
        """Test ProfileBlockConfig.with_columns builder."""
        config = ProfileBlockConfig().with_columns(["id", "name"])
        transformer = ProfileTransformer(config=config)
        assert transformer.config.columns == frozenset(["id", "name"])

    def test_learn_config_with_strictness(self) -> None:
        """Test LearnBlockConfig.with_strictness builder."""
        config = LearnBlockConfig().with_strictness("strict")
        transformer = LearnTransformer(config=config)
        assert transformer.config.strictness == "strict"


class TestTransformerExecution:
    """Tests for transformer execution behavior."""

    def test_execution_time_tracked(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test execution time is tracked."""
        config = CheckBlockConfig(rules=({"type": "not_null", "column": "id"},))
        transformer = CheckTransformer(config=config, engine=mock_engine)
        result = transformer.execute(sample_data)

        assert result.execution_time_ms >= 0

    def test_metadata_includes_engine_info(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test metadata includes engine information."""
        config = CheckBlockConfig(rules=({"type": "not_null", "column": "id"},))
        transformer = CheckTransformer(config=config, engine=mock_engine)
        result = transformer.execute(sample_data)

        assert "engine" in result.metadata
        assert "engine_version" in result.metadata
        assert "timestamp" in result.metadata

    def test_result_dict_available(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test result_dict is available for serialization."""
        config = CheckBlockConfig(rules=({"type": "not_null", "column": "id"},))
        transformer = CheckTransformer(config=config, engine=mock_engine)
        result = transformer.execute(sample_data)

        assert result.result_dict is not None
        assert isinstance(result.result_dict, dict)

    def test_data_passed_through(
        self, mock_engine: MockDataQualityEngine, sample_data: dict[str, Any]
    ) -> None:
        """Test original data is passed through in result."""
        config = CheckBlockConfig(rules=({"type": "not_null", "column": "id"},))
        transformer = CheckTransformer(config=config, engine=mock_engine)
        result = transformer.execute(sample_data)

        assert result.data is sample_data
