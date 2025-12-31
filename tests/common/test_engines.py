"""Tests for Data Quality Engine module.

This module tests the engine abstraction layer including:
- DataQualityEngine protocol compliance
- TruthoundEngine implementation
- GreatExpectationsAdapter implementation
- PanderaAdapter implementation
- EngineRegistry functionality
- MockDataQualityEngine for testing
"""

from __future__ import annotations

import pytest

from common import (
    CheckStatus,
    LearnStatus,
    ProfileStatus,
)
from common.engines import (
    DataQualityEngine,
    EngineCapabilities,
    EngineInfo,
    EngineRegistry,
    GreatExpectationsAdapter,
    PanderaAdapter,
    TruthoundEngine,
    get_default_engine,
    get_engine,
    get_engine_registry,
    list_engines,
    register_engine,
)
from common.engines.base import EngineInfoMixin
from common.engines.registry import (
    EngineAlreadyRegisteredError,
    EngineNotFoundError,
)
from common.testing import MockDataQualityEngine


# =============================================================================
# Engine Capabilities Tests
# =============================================================================


class TestEngineCapabilities:
    """Tests for EngineCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = EngineCapabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is True
        assert caps.supports_async is False
        assert caps.supports_streaming is False
        assert caps.supported_data_types == ("polars",)
        assert caps.supported_rule_types == ()

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = EngineCapabilities(
            supports_check=True,
            supports_profile=False,
            supports_learn=False,
            supports_async=True,
            supported_data_types=("pandas", "polars", "spark"),
            supported_rule_types=("not_null", "unique"),
        )

        assert caps.supports_check is True
        assert caps.supports_profile is False
        assert caps.supports_learn is False
        assert caps.supports_async is True
        assert "pandas" in caps.supported_data_types

    def test_supports_data_type(self):
        """Test data type support checking."""
        caps = EngineCapabilities(
            supported_data_types=("polars", "pandas"),
        )

        assert caps.supports_data_type("polars") is True
        assert caps.supports_data_type("Polars") is True  # Case insensitive
        assert caps.supports_data_type("PANDAS") is True
        assert caps.supports_data_type("spark") is False

    def test_supports_rule_type(self):
        """Test rule type support checking."""
        # Empty supported_rule_types means all are supported
        caps = EngineCapabilities(supported_rule_types=())
        assert caps.supports_rule_type("not_null") is True
        assert caps.supports_rule_type("any_rule") is True

        # Specific rule types
        caps = EngineCapabilities(
            supported_rule_types=("not_null", "unique"),
        )
        assert caps.supports_rule_type("not_null") is True
        assert caps.supports_rule_type("unique") is True
        assert caps.supports_rule_type("in_range") is False


class TestEngineInfo:
    """Tests for EngineInfo dataclass."""

    def test_engine_info_creation(self):
        """Test creating engine info."""
        info = EngineInfo(
            name="test_engine",
            version="1.0.0",
            description="A test engine",
            homepage="https://example.com",
        )

        assert info.name == "test_engine"
        assert info.version == "1.0.0"
        assert info.description == "A test engine"
        assert info.homepage == "https://example.com"

    def test_engine_info_to_dict(self):
        """Test converting engine info to dictionary."""
        caps = EngineCapabilities(supports_async=True)
        info = EngineInfo(
            name="test",
            version="1.0.0",
            capabilities=caps,
        )

        result = info.to_dict()

        assert result["name"] == "test"
        assert result["version"] == "1.0.0"
        assert result["capabilities"]["supports_async"] is True


# =============================================================================
# MockDataQualityEngine Tests
# =============================================================================


class TestMockDataQualityEngine:
    """Tests for MockDataQualityEngine."""

    def test_protocol_compliance(self):
        """Test that MockDataQualityEngine implements DataQualityEngine protocol."""
        engine = MockDataQualityEngine()
        assert isinstance(engine, DataQualityEngine)

    def test_engine_properties(self):
        """Test engine name and version properties."""
        engine = MockDataQualityEngine(name="test", version="2.0.0")

        assert engine.engine_name == "test"
        assert engine.engine_version == "2.0.0"

    def test_check_success(self):
        """Test successful check execution."""
        engine = MockDataQualityEngine()
        engine.configure_check(success=True, passed_count=5)

        result = engine.check(None, rules=[{"type": "not_null"}])

        assert result.is_success
        assert result.status == CheckStatus.PASSED
        assert result.passed_count == 5
        assert engine.check_call_count == 1

    def test_check_failure(self):
        """Test failed check execution."""
        engine = MockDataQualityEngine()
        engine.configure_check(success=False, failed_count=3)

        result = engine.check(None, rules=[])

        assert not result.is_success
        assert result.status == CheckStatus.FAILED
        assert result.failed_count == 3

    def test_check_raises_error(self):
        """Test check raising configured error."""
        engine = MockDataQualityEngine()
        engine.configure_check(raise_error=ValueError("test error"))

        with pytest.raises(ValueError, match="test error"):
            engine.check(None, rules=[])

    def test_profile_success(self):
        """Test successful profile execution."""
        engine = MockDataQualityEngine()
        engine.configure_profile(success=True, row_count=100)

        result = engine.profile(None)

        assert result.is_success
        assert result.status == ProfileStatus.COMPLETED
        assert result.row_count == 100
        assert engine.profile_call_count == 1

    def test_learn_success(self):
        """Test successful learn execution."""
        engine = MockDataQualityEngine()
        engine.configure_learn(success=True, columns_analyzed=10)

        result = engine.learn(None)

        assert result.is_success
        assert result.status == LearnStatus.COMPLETED
        assert result.columns_analyzed == 10
        assert engine.learn_call_count == 1

    def test_call_tracking(self):
        """Test that calls are properly tracked."""
        engine = MockDataQualityEngine()

        engine.check("data1", rules=[{"type": "a"}])
        engine.check("data2", rules=[{"type": "b"}, {"type": "c"}])
        engine.profile("data3", columns=["id"])
        engine.learn("data4", confidence=0.9)

        assert engine.check_call_count == 2
        assert engine.profile_call_count == 1
        assert engine.learn_call_count == 1

        check_calls = engine.get_check_calls()
        assert check_calls[0][0] == "data1"
        assert check_calls[1][1] == ({"type": "b"}, {"type": "c"})

    def test_reset(self):
        """Test resetting the mock engine."""
        engine = MockDataQualityEngine()
        engine.configure_check(success=False)
        engine.check(None, rules=[])
        engine.profile(None)

        engine.reset()

        assert engine.check_call_count == 0
        assert engine.profile_call_count == 0
        # Default config should be restored
        result = engine.check(None, rules=[])
        assert result.is_success


# =============================================================================
# TruthoundEngine Tests
# =============================================================================


class TestTruthoundEngine:
    """Tests for TruthoundEngine."""

    def test_protocol_compliance(self):
        """Test that TruthoundEngine implements DataQualityEngine protocol."""
        engine = TruthoundEngine()
        assert isinstance(engine, DataQualityEngine)

    def test_engine_name(self):
        """Test engine name property."""
        engine = TruthoundEngine()
        assert engine.engine_name == "truthound"

    def test_capabilities(self):
        """Test engine capabilities."""
        engine = TruthoundEngine()
        caps = engine.get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is True
        assert caps.supports_data_type("polars")
        assert caps.supports_data_type("pandas")

    def test_engine_info(self):
        """Test engine info."""
        engine = TruthoundEngine()
        info = engine.get_info()

        assert info.name == "truthound"
        assert "truthound" in info.description.lower()

    def test_check_with_truthound(self):
        """Test check works with truthound installed."""
        import polars as pl

        engine = TruthoundEngine()

        # Create test data
        df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })

        result = engine.check(df)
        assert result.status == CheckStatus.PASSED
        assert result.is_success is True
        assert "truthound" in result.metadata.get("engine", "")

    def test_check_with_issues(self):
        """Test check detects data quality issues."""
        import polars as pl

        engine = TruthoundEngine()

        # Create test data with issues
        df = pl.DataFrame({
            "id": [1, 2, None],
            "email": ["a@test.com", "invalid", "c@test.com"],
        })

        result = engine.check(df)
        assert result.status in (CheckStatus.FAILED, CheckStatus.WARNING)
        assert len(result.failures) > 0

    def test_profile_with_truthound(self):
        """Test profile works with truthound installed."""
        import polars as pl

        engine = TruthoundEngine()

        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        })

        result = engine.profile(df)
        assert result.status == ProfileStatus.COMPLETED
        assert result.row_count == 5
        assert result.column_count == 2

    def test_learn_with_truthound(self):
        """Test learn works with truthound installed."""
        import polars as pl

        engine = TruthoundEngine()

        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "status": ["active", "active", "inactive", "active", "inactive"],
        })

        result = engine.learn(df)
        assert result.status == LearnStatus.COMPLETED
        assert result.columns_analyzed == 2
        assert len(result.rules) > 0

    def test_get_schema(self):
        """Test get_schema returns usable schema for validation."""
        import polars as pl

        engine = TruthoundEngine()

        # Learn schema from baseline
        baseline_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })
        schema = engine.get_schema(baseline_df)

        # Use schema to validate same data (should pass)
        result = engine.check(baseline_df, schema=schema)
        assert result.status == CheckStatus.PASSED

        # New data with different values will trigger warnings
        # because Truthound learns allowed_values
        new_df = pl.DataFrame({
            "id": [4, 5, 6],
            "name": ["David", "Eve", "Frank"],
        })
        result_new = engine.check(new_df, schema=schema)
        # This may have warnings due to values not in learned allowed set
        assert result_new.status in (CheckStatus.PASSED, CheckStatus.WARNING)


# =============================================================================
# GreatExpectationsAdapter Tests
# =============================================================================


class TestGreatExpectationsAdapter:
    """Tests for GreatExpectationsAdapter."""

    def test_protocol_compliance(self):
        """Test that adapter implements DataQualityEngine protocol."""
        adapter = GreatExpectationsAdapter()
        assert isinstance(adapter, DataQualityEngine)

    def test_engine_name(self):
        """Test engine name property."""
        adapter = GreatExpectationsAdapter()
        assert adapter.engine_name == "great_expectations"

    def test_capabilities(self):
        """Test engine capabilities."""
        adapter = GreatExpectationsAdapter()
        caps = adapter.get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is True
        assert caps.supports_data_type("pandas")

    def test_check_without_ge(self):
        """Test check fails gracefully without GE installed."""
        adapter = GreatExpectationsAdapter()

        from common.exceptions import ValidationExecutionError

        with pytest.raises(ValidationExecutionError, match="not installed"):
            adapter.check(None, rules=[{"type": "not_null"}])


# =============================================================================
# PanderaAdapter Tests
# =============================================================================


class TestPanderaAdapter:
    """Tests for PanderaAdapter."""

    def test_protocol_compliance(self):
        """Test that adapter implements DataQualityEngine protocol."""
        adapter = PanderaAdapter()
        assert isinstance(adapter, DataQualityEngine)

    def test_engine_name(self):
        """Test engine name property."""
        adapter = PanderaAdapter()
        assert adapter.engine_name == "pandera"

    def test_capabilities(self):
        """Test engine capabilities."""
        adapter = PanderaAdapter()
        caps = adapter.get_capabilities()

        assert caps.supports_check is True
        assert caps.supports_profile is True
        assert caps.supports_learn is True
        assert caps.supports_data_type("pandas")
        assert caps.supports_data_type("polars")

    def test_check_without_pandera(self):
        """Test check fails gracefully without pandera installed."""
        adapter = PanderaAdapter()

        from common.exceptions import ValidationExecutionError

        with pytest.raises(ValidationExecutionError, match="not installed"):
            adapter.check(None, rules=[{"type": "not_null"}])


# =============================================================================
# EngineRegistry Tests
# =============================================================================


class TestEngineRegistry:
    """Tests for EngineRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving an engine."""
        registry = EngineRegistry()
        engine = MockDataQualityEngine(name="test")

        registry.register("test", engine)

        retrieved = registry.get("test")
        assert retrieved is engine
        assert retrieved.engine_name == "test"

    def test_register_sets_default(self):
        """Test that first registration sets default."""
        registry = EngineRegistry()
        engine = MockDataQualityEngine()

        registry.register("first", engine)

        assert registry.default_engine_name == "first"

    def test_register_with_set_as_default(self):
        """Test explicitly setting default during registration."""
        registry = EngineRegistry()
        engine1 = MockDataQualityEngine(name="first")
        engine2 = MockDataQualityEngine(name="second")

        registry.register("first", engine1)
        registry.register("second", engine2, set_as_default=True)

        assert registry.default_engine_name == "second"

    def test_register_duplicate_fails(self):
        """Test that duplicate registration raises error."""
        registry = EngineRegistry()
        engine = MockDataQualityEngine()

        registry.register("test", engine)

        with pytest.raises(EngineAlreadyRegisteredError) as exc_info:
            registry.register("test", engine)

        assert exc_info.value.engine_name == "test"

    def test_register_override(self):
        """Test overriding existing registration."""
        registry = EngineRegistry()
        engine1 = MockDataQualityEngine(name="v1")
        engine2 = MockDataQualityEngine(name="v2")

        registry.register("test", engine1)
        registry.register("test", engine2, allow_override=True)

        assert registry.get("test").engine_name == "v2"

    def test_get_not_found(self):
        """Test getting non-existent engine raises error."""
        registry = EngineRegistry()
        registry.register("existing", MockDataQualityEngine())

        with pytest.raises(EngineNotFoundError) as exc_info:
            registry.get("nonexistent")

        assert exc_info.value.engine_name == "nonexistent"
        assert "existing" in exc_info.value.available_engines

    def test_get_or_none(self):
        """Test get_or_none returns None for missing engines."""
        registry = EngineRegistry()

        result = registry.get_or_none("missing")
        assert result is None

        engine = MockDataQualityEngine()
        registry.register("present", engine)
        assert registry.get_or_none("present") is engine

    def test_unregister(self):
        """Test unregistering an engine."""
        registry = EngineRegistry()
        engine = MockDataQualityEngine()

        registry.register("test", engine)
        removed = registry.unregister("test")

        assert removed is engine
        assert registry.get_or_none("test") is None

    def test_unregister_default_sets_new_default(self):
        """Test that unregistering default sets new default."""
        registry = EngineRegistry()
        engine1 = MockDataQualityEngine(name="first")
        engine2 = MockDataQualityEngine(name="second")

        registry.register("first", engine1)
        registry.register("second", engine2)

        registry.unregister("first")

        assert registry.default_engine_name == "second"

    def test_get_default(self):
        """Test getting the default engine."""
        registry = EngineRegistry()
        engine = MockDataQualityEngine()

        registry.register("test", engine)

        default = registry.get_default()
        assert default is engine

    def test_get_default_empty_raises(self):
        """Test getting default from empty registry raises error."""
        registry = EngineRegistry()

        with pytest.raises(EngineNotFoundError):
            registry.get_default()

    def test_set_default(self):
        """Test setting the default engine."""
        registry = EngineRegistry()
        engine1 = MockDataQualityEngine()
        engine2 = MockDataQualityEngine()

        registry.register("first", engine1)
        registry.register("second", engine2)

        registry.set_default("second")

        assert registry.default_engine_name == "second"

    def test_set_default_not_found(self):
        """Test setting default to non-existent engine raises error."""
        registry = EngineRegistry()

        with pytest.raises(EngineNotFoundError):
            registry.set_default("nonexistent")

    def test_list(self):
        """Test listing registered engines."""
        registry = EngineRegistry()

        registry.register("a", MockDataQualityEngine())
        registry.register("b", MockDataQualityEngine())
        registry.register("c", MockDataQualityEngine())

        names = registry.list()
        assert set(names) == {"a", "b", "c"}

    def test_has(self):
        """Test checking if engine exists."""
        registry = EngineRegistry()
        registry.register("test", MockDataQualityEngine())

        assert registry.has("test") is True
        assert registry.has("missing") is False

    def test_contains(self):
        """Test __contains__ protocol."""
        registry = EngineRegistry()
        registry.register("test", MockDataQualityEngine())

        assert "test" in registry
        assert "missing" not in registry

    def test_len(self):
        """Test __len__ protocol."""
        registry = EngineRegistry()

        assert len(registry) == 0

        registry.register("a", MockDataQualityEngine())
        registry.register("b", MockDataQualityEngine())

        assert len(registry) == 2

    def test_iter(self):
        """Test iteration over registry."""
        registry = EngineRegistry()
        registry.register("a", MockDataQualityEngine())
        registry.register("b", MockDataQualityEngine())

        names = list(registry)
        assert set(names) == {"a", "b"}

    def test_clear(self):
        """Test clearing all engines."""
        registry = EngineRegistry()
        registry.register("a", MockDataQualityEngine())
        registry.register("b", MockDataQualityEngine())

        registry.clear()

        assert len(registry) == 0
        assert registry.default_engine_name is None


# =============================================================================
# Global Registry Function Tests
# =============================================================================


class TestGlobalRegistryFunctions:
    """Tests for global registry convenience functions."""

    def test_get_engine_registry(self):
        """Test getting the global registry."""
        registry = get_engine_registry()

        assert isinstance(registry, EngineRegistry)
        # Should have default engines registered
        assert "truthound" in registry

    def test_list_engines(self):
        """Test listing engines from global registry."""
        engines = list_engines()

        assert "truthound" in engines
        assert "great_expectations" in engines
        assert "pandera" in engines
        assert "ge" in engines  # Alias

    def test_get_default_engine(self):
        """Test getting default engine."""
        engine = get_default_engine()

        assert engine.engine_name == "truthound"

    def test_get_engine(self):
        """Test getting engine by name."""
        engine = get_engine("truthound")
        assert engine.engine_name == "truthound"

        engine = get_engine("pandera")
        assert engine.engine_name == "pandera"

    def test_get_engine_not_found(self):
        """Test getting non-existent engine."""
        with pytest.raises(EngineNotFoundError):
            get_engine("nonexistent_engine")

    def test_register_custom_engine(self):
        """Test registering a custom engine."""
        custom = MockDataQualityEngine(name="custom_test")

        register_engine("custom_test_engine", custom, allow_override=True)

        retrieved = get_engine("custom_test_engine")
        assert retrieved.engine_name == "custom_test"


# =============================================================================
# EngineInfoMixin Tests
# =============================================================================


class TestEngineInfoMixin:
    """Tests for EngineInfoMixin."""

    def test_get_info(self):
        """Test get_info method from mixin."""

        class TestEngine(EngineInfoMixin):
            @property
            def engine_name(self) -> str:
                return "test"

            @property
            def engine_version(self) -> str:
                return "1.0.0"

        engine = TestEngine()
        info = engine.get_info()

        assert info.name == "test"
        assert info.version == "1.0.0"
        assert "test" in info.description

    def test_custom_capabilities(self):
        """Test overriding capabilities in mixin."""

        class CustomEngine(EngineInfoMixin):
            @property
            def engine_name(self) -> str:
                return "custom"

            @property
            def engine_version(self) -> str:
                return "2.0.0"

            def _get_capabilities(self) -> EngineCapabilities:
                return EngineCapabilities(
                    supports_async=True,
                    supports_streaming=True,
                )

        engine = CustomEngine()
        caps = engine.get_capabilities()

        assert caps.supports_async is True
        assert caps.supports_streaming is True

    def test_custom_description(self):
        """Test overriding description in mixin."""

        class DescribedEngine(EngineInfoMixin):
            @property
            def engine_name(self) -> str:
                return "described"

            @property
            def engine_version(self) -> str:
                return "1.0.0"

            def _get_description(self) -> str:
                return "A custom description"

        engine = DescribedEngine()
        info = engine.get_info()

        assert info.description == "A custom description"
