"""Tests for Plugin Discovery System.

This module tests the plugin discovery system including:
- Plugin specifications and metadata
- Plugin sources (entry points, builtins, dict)
- Plugin loading and validation
- Plugin registry management
- Hook system
- Integration with EngineRegistry
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

from common.base import CheckResult, CheckStatus, LearnResult, ProfileResult
from common.engines.plugin import (
    # Constants
    DEFAULT_PLUGIN_PRIORITY,
    ENTRY_POINT_GROUP,
    # Enums
    LoadStrategy,
    PluginState,
    PluginType,
    # Exceptions
    PluginConflictError,
    PluginDiscoveryError,
    PluginError,
    PluginLoadError,
    PluginNotFoundError,
    PluginValidationError,
    # Data Types
    DiscoveryConfig,
    PluginInstance,
    PluginMetadata,
    PluginSpec,
    ValidationResult,
    # Protocols / Implementations
    BasePluginHook,
    BuiltinPluginSource,
    CompositePluginHook,
    CompositeValidator,
    DataQualityEngineValidator,
    DefaultPluginFactory,
    DictPluginSource,
    EntryPointPluginSource,
    LoggingPluginHook,
    MetricsPluginHook,
    PluginLoader,
    PluginRegistry,
    # Global Functions
    auto_discover_engines,
    discover_plugins,
    get_plugin_engine,
    get_plugin_registry,
    load_plugins,
    register_plugin,
    reset_plugin_registry,
    validate_plugin,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reset_global_registry():
    """Reset the global plugin registry before and after each test."""
    reset_plugin_registry()
    yield
    reset_plugin_registry()


@pytest.fixture
def mock_engine_class():
    """Create a mock engine class for testing."""

    class MockEngine:
        @property
        def engine_name(self) -> str:
            return "mock_engine"

        @property
        def engine_version(self) -> str:
            return "1.0.0"

        def check(self, data: Any, rules: Sequence, **kwargs) -> CheckResult:
            return CheckResult(status=CheckStatus.PASSED, passed_count=1)

        def profile(self, data: Any, **kwargs) -> ProfileResult:
            return ProfileResult(row_count=0, column_count=0, columns=())

        def learn(self, data: Any, **kwargs) -> LearnResult:
            return LearnResult(rules=())

    return MockEngine


@pytest.fixture
def invalid_engine_class():
    """Create an invalid engine class for testing validation."""

    class InvalidEngine:
        pass

    return InvalidEngine


# =============================================================================
# Tests: PluginMetadata
# =============================================================================


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    def test_create_basic_metadata(self):
        """Test creating basic metadata."""
        meta = PluginMetadata(name="test_plugin")

        assert meta.name == "test_plugin"
        assert meta.version == ""
        assert meta.description == ""
        assert meta.author == ""
        assert meta.homepage == ""
        assert meta.license == ""
        assert meta.tags == ()
        assert meta.dependencies == ()
        assert meta.python_requires == ""

    def test_create_full_metadata(self):
        """Test creating metadata with all fields."""
        meta = PluginMetadata(
            name="test_plugin",
            version="1.2.3",
            description="A test plugin",
            author="Test Author",
            homepage="https://example.com",
            license="MIT",
            tags=("test", "example"),
            dependencies=("dep1", "dep2"),
            python_requires=">=3.11",
        )

        assert meta.name == "test_plugin"
        assert meta.version == "1.2.3"
        assert meta.description == "A test plugin"
        assert meta.author == "Test Author"
        assert meta.homepage == "https://example.com"
        assert meta.license == "MIT"
        assert meta.tags == ("test", "example")
        assert meta.dependencies == ("dep1", "dep2")
        assert meta.python_requires == ">=3.11"

    def test_from_distribution_without_dist(self):
        """Test creating metadata without distribution."""
        meta = PluginMetadata.from_distribution("test_plugin", None)

        assert meta.name == "test_plugin"
        assert meta.version == ""

    def test_metadata_is_immutable(self):
        """Test that metadata is immutable."""
        meta = PluginMetadata(name="test")

        with pytest.raises(AttributeError):
            meta.name = "changed"  # type: ignore


# =============================================================================
# Tests: PluginSpec
# =============================================================================


class TestPluginSpec:
    """Tests for PluginSpec dataclass."""

    def test_create_basic_spec(self):
        """Test creating basic plugin spec."""
        spec = PluginSpec(
            name="test_plugin",
            module_path="test.module",
            class_name="TestEngine",
        )

        assert spec.name == "test_plugin"
        assert spec.module_path == "test.module"
        assert spec.class_name == "TestEngine"
        assert spec.plugin_type == PluginType.ENGINE
        assert spec.priority == DEFAULT_PLUGIN_PRIORITY
        assert spec.enabled is True
        assert spec.source == "entry_point"
        assert spec.aliases == ()
        assert spec.config == {}

    def test_full_path_property(self):
        """Test full_path property."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        assert spec.full_path == "test.module:TestEngine"

    def test_with_priority(self):
        """Test with_priority creates a new spec."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
            priority=100,
        )

        new_spec = spec.with_priority(50)

        assert spec.priority == 100  # Original unchanged
        assert new_spec.priority == 50
        assert new_spec.name == spec.name
        assert new_spec.module_path == spec.module_path

    def test_with_enabled(self):
        """Test with_enabled creates a new spec."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
            enabled=True,
        )

        new_spec = spec.with_enabled(False)

        assert spec.enabled is True
        assert new_spec.enabled is False

    def test_with_aliases(self):
        """Test with_aliases creates a new spec."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
            aliases=("a",),
        )

        new_spec = spec.with_aliases("b", "c")

        assert spec.aliases == ("a",)
        assert new_spec.aliases == ("a", "b", "c")

    def test_with_config(self):
        """Test with_config creates a new spec."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
            config={"a": 1},
        )

        new_spec = spec.with_config({"b": 2})

        assert spec.config == {"a": 1}
        assert new_spec.config == {"a": 1, "b": 2}


# =============================================================================
# Tests: PluginInstance
# =============================================================================


class TestPluginInstance:
    """Tests for PluginInstance dataclass."""

    def test_create_instance(self):
        """Test creating plugin instance."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )
        instance = PluginInstance(spec=spec)

        assert instance.spec == spec
        assert instance.instance is None
        assert instance.state == PluginState.DISCOVERED
        assert instance.is_loaded is False
        assert instance.is_failed is False

    def test_is_loaded_property(self):
        """Test is_loaded property."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )
        mock_engine = MagicMock()

        instance = PluginInstance(
            spec=spec,
            instance=mock_engine,
            state=PluginState.LOADED,
        )

        assert instance.is_loaded is True
        assert instance.is_failed is False

    def test_is_failed_property(self):
        """Test is_failed property."""
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        instance = PluginInstance(
            spec=spec,
            state=PluginState.FAILED,
            error=RuntimeError("Test error"),
        )

        assert instance.is_loaded is False
        assert instance.is_failed is True


# =============================================================================
# Tests: ValidationResult
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        result = ValidationResult.success()

        assert result.is_valid is True
        assert result.errors == ()
        assert result.warnings == ()

    def test_success_with_warnings(self):
        """Test creating success result with warnings."""
        result = ValidationResult.success(warnings=["Warning 1", "Warning 2"])

        assert result.is_valid is True
        assert result.warnings == ("Warning 1", "Warning 2")

    def test_failure_result(self):
        """Test creating failure result."""
        result = ValidationResult.failure(["Error 1", "Error 2"])

        assert result.is_valid is False
        assert result.errors == ("Error 1", "Error 2")

    def test_failure_with_warnings(self):
        """Test creating failure result with warnings."""
        result = ValidationResult.failure(
            ["Error 1"],
            warnings=["Warning 1"],
        )

        assert result.is_valid is False
        assert result.errors == ("Error 1",)
        assert result.warnings == ("Warning 1",)


# =============================================================================
# Tests: DiscoveryConfig
# =============================================================================


class TestDiscoveryConfig:
    """Tests for DiscoveryConfig dataclass."""

    def test_default_config(self):
        """Test default discovery config."""
        config = DiscoveryConfig()

        assert config.entry_point_group == ENTRY_POINT_GROUP
        assert config.load_strategy == LoadStrategy.LAZY
        assert config.validate_plugins is True
        assert config.ignore_errors is True
        assert config.disabled_plugins == ()
        assert config.priority_overrides == {}
        assert config.include_patterns == ("*",)
        assert config.exclude_patterns == ()

    def test_with_entry_point_group(self):
        """Test with_entry_point_group creates new config."""
        config = DiscoveryConfig()
        new_config = config.with_entry_point_group("custom.group")

        assert config.entry_point_group == ENTRY_POINT_GROUP
        assert new_config.entry_point_group == "custom.group"

    def test_with_load_strategy(self):
        """Test with_load_strategy creates new config."""
        config = DiscoveryConfig()
        new_config = config.with_load_strategy(LoadStrategy.EAGER)

        assert config.load_strategy == LoadStrategy.LAZY
        assert new_config.load_strategy == LoadStrategy.EAGER

    def test_with_disabled_plugins(self):
        """Test with_disabled_plugins creates new config."""
        config = DiscoveryConfig(disabled_plugins=("a",))
        new_config = config.with_disabled_plugins("b", "c")

        assert config.disabled_plugins == ("a",)
        assert new_config.disabled_plugins == ("a", "b", "c")

    def test_with_priority_override(self):
        """Test with_priority_override creates new config."""
        config = DiscoveryConfig()
        new_config = config.with_priority_override("test", 50)

        assert config.priority_overrides == {}
        assert new_config.priority_overrides == {"test": 50}


# =============================================================================
# Tests: Plugin Sources
# =============================================================================


class TestBuiltinPluginSource:
    """Tests for BuiltinPluginSource."""

    def test_discover_builtins(self):
        """Test discovering builtin plugins."""
        source = BuiltinPluginSource()
        plugins = list(source.discover())

        assert len(plugins) == 3
        names = {p.name for p in plugins}
        assert names == {"truthound", "great_expectations", "pandera"}

    def test_builtin_source_name(self):
        """Test source name property."""
        source = BuiltinPluginSource()
        assert source.name == "builtin"

    def test_builtin_priorities(self):
        """Test builtin priorities are set correctly."""
        source = BuiltinPluginSource()
        plugins = list(source.discover())

        truthound = next(p for p in plugins if p.name == "truthound")
        assert truthound.priority == 0  # Highest priority


class TestDictPluginSource:
    """Tests for DictPluginSource."""

    def test_discover_from_dict_with_strings(self):
        """Test discovering plugins from dict with string paths."""
        plugins_dict = {
            "engine1": "my.module:Engine1",
            "engine2": "my.other.module.Engine2",
        }
        source = DictPluginSource(plugins_dict)
        plugins = list(source.discover())

        assert len(plugins) == 2
        assert plugins[0].name == "engine1"
        assert plugins[0].module_path == "my.module"
        assert plugins[0].class_name == "Engine1"

    def test_discover_from_dict_with_classes(self, mock_engine_class):
        """Test discovering plugins from dict with classes."""
        plugins_dict = {"my_engine": mock_engine_class}
        source = DictPluginSource(plugins_dict)
        plugins = list(source.discover())

        assert len(plugins) == 1
        assert plugins[0].name == "my_engine"
        assert plugins[0].class_name == "MockEngine"

    def test_source_name(self):
        """Test custom source name."""
        source = DictPluginSource({}, source_name="custom")
        assert source.name == "custom"


class TestEntryPointPluginSource:
    """Tests for EntryPointPluginSource."""

    def test_source_name(self):
        """Test source name property."""
        source = EntryPointPluginSource()
        assert source.name == f"entry_point:{ENTRY_POINT_GROUP}"

    def test_discover_with_no_entry_points(self):
        """Test discovering when no entry points exist."""
        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = []
            mock_eps.return_value = mock_result

            source = EntryPointPluginSource("nonexistent.group")
            plugins = list(source.discover())

            assert plugins == []


# =============================================================================
# Tests: Validators
# =============================================================================


class TestDataQualityEngineValidator:
    """Tests for DataQualityEngineValidator."""

    def test_validate_valid_engine(self, mock_engine_class):
        """Test validating a valid engine."""
        validator = DataQualityEngineValidator()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="MockEngine",
        )

        result = validator.validate(mock_engine_class, spec)

        assert result.is_valid is True

    def test_validate_invalid_engine(self, invalid_engine_class):
        """Test validating an invalid engine."""
        validator = DataQualityEngineValidator()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="InvalidEngine",
        )

        result = validator.validate(invalid_engine_class, spec)

        assert result.is_valid is False
        assert "Missing required method: check" in result.errors
        assert "Missing required method: profile" in result.errors
        assert "Missing required method: learn" in result.errors

    def test_validate_missing_properties(self):
        """Test validating engine with missing properties."""

        class EngineNoProps:
            def check(self, data, rules, **kwargs):
                pass

            def profile(self, data, **kwargs):
                pass

            def learn(self, data, **kwargs):
                pass

        validator = DataQualityEngineValidator()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="EngineNoProps",
        )

        result = validator.validate(EngineNoProps, spec)

        assert result.is_valid is False
        assert any("engine_name" in e for e in result.errors)
        assert any("engine_version" in e for e in result.errors)


class TestCompositeValidator:
    """Tests for CompositeValidator."""

    def test_composite_runs_all_validators(self, mock_engine_class):
        """Test that composite validator runs all validators."""
        validator1 = MagicMock()
        validator1.validate.return_value = ValidationResult.success(["Warning 1"])

        validator2 = MagicMock()
        validator2.validate.return_value = ValidationResult.success(["Warning 2"])

        composite = CompositeValidator([validator1, validator2])
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="MockEngine",
        )

        result = composite.validate(mock_engine_class, spec)

        assert result.is_valid is True
        assert "Warning 1" in result.warnings
        assert "Warning 2" in result.warnings

    def test_composite_combines_errors(self, invalid_engine_class):
        """Test that composite combines errors from all validators."""
        validator1 = MagicMock()
        validator1.validate.return_value = ValidationResult.failure(["Error 1"])

        validator2 = MagicMock()
        validator2.validate.return_value = ValidationResult.failure(["Error 2"])

        composite = CompositeValidator([validator1, validator2])
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="InvalidEngine",
        )

        result = composite.validate(invalid_engine_class, spec)

        assert result.is_valid is False
        assert "Error 1" in result.errors
        assert "Error 2" in result.errors


# =============================================================================
# Tests: Plugin Factory
# =============================================================================


class TestDefaultPluginFactory:
    """Tests for DefaultPluginFactory."""

    def test_create_instance(self, mock_engine_class):
        """Test creating a plugin instance."""
        factory = DefaultPluginFactory()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="MockEngine",
        )

        instance = factory.create(spec, mock_engine_class)

        assert instance is not None
        assert instance.engine_name == "mock_engine"

    def test_create_with_config(self):
        """Test creating instance with config."""

        class ConfigurableEngine:
            def __init__(self, option: str = "default"):
                self.option = option

            @property
            def engine_name(self):
                return "configurable"

            @property
            def engine_version(self):
                return "1.0.0"

            def check(self, data, rules, **kwargs):
                pass

            def profile(self, data, **kwargs):
                pass

            def learn(self, data, **kwargs):
                pass

        factory = DefaultPluginFactory()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="ConfigurableEngine",
            config={"option": "custom"},
        )

        instance = factory.create(spec, ConfigurableEngine)

        assert instance.option == "custom"


# =============================================================================
# Tests: Plugin Loader
# =============================================================================


class TestPluginLoader:
    """Tests for PluginLoader."""

    def test_load_class(self):
        """Test loading a class by spec."""
        loader = PluginLoader()
        spec = PluginSpec(
            name="truthound",
            module_path="common.engines.truthound",
            class_name="TruthoundEngine",
        )

        engine_class = loader.load_class(spec)

        assert engine_class is not None
        assert engine_class.__name__ == "TruthoundEngine"

    def test_load_class_invalid_module(self):
        """Test loading from invalid module raises error."""
        loader = PluginLoader()
        spec = PluginSpec(
            name="test",
            module_path="nonexistent.module",
            class_name="Engine",
        )

        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_class(spec)

        assert "nonexistent.module" in str(exc_info.value)

    def test_load_class_invalid_class(self):
        """Test loading invalid class raises error."""
        loader = PluginLoader()
        spec = PluginSpec(
            name="test",
            module_path="common.engines.truthound",
            class_name="NonexistentClass",
        )

        with pytest.raises(PluginLoadError) as exc_info:
            loader.load_class(spec)

        assert "NonexistentClass" in str(exc_info.value)

    def test_load_complete(self, mock_engine_class):
        """Test complete load of a plugin."""
        loader = PluginLoader()

        # Create spec that points to a real loadable class
        spec = PluginSpec(
            name="truthound",
            module_path="common.engines.truthound",
            class_name="TruthoundEngine",
        )

        instance = loader.load(spec, validate=True)

        assert instance.is_loaded
        assert instance.state == PluginState.LOADED
        assert instance.instance is not None
        assert instance.engine_class is not None
        assert instance.load_time is not None


# =============================================================================
# Tests: Plugin Hooks
# =============================================================================


class TestBasePluginHook:
    """Tests for BasePluginHook."""

    def test_base_hook_methods_are_noop(self):
        """Test that base hook methods do nothing."""
        hook = BasePluginHook()
        config = DiscoveryConfig()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        # These should not raise
        hook.on_discovery_start(config)
        hook.on_discovery_end([], [])
        hook.on_plugin_discovered(spec)
        hook.on_plugin_loading(spec)
        hook.on_plugin_loaded(spec, None)  # type: ignore[arg-type]
        hook.on_plugin_error(spec, RuntimeError("test"))


class TestLoggingPluginHook:
    """Tests for LoggingPluginHook."""

    def test_logging_hook_logs_events(self, caplog):
        """Test that logging hook logs events."""
        import logging

        hook = LoggingPluginHook(level=logging.INFO)
        config = DiscoveryConfig()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        with caplog.at_level(logging.INFO):
            hook.on_discovery_start(config)
            hook.on_plugin_discovered(spec)

        assert "Starting plugin discovery" in caplog.text
        assert "Discovered plugin 'test'" in caplog.text


class TestMetricsPluginHook:
    """Tests for MetricsPluginHook."""

    def test_metrics_hook_tracks_counts(self):
        """Test that metrics hook tracks counts."""
        hook = MetricsPluginHook()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        hook.on_plugin_discovered(spec)
        hook.on_plugin_loaded(spec, None)  # type: ignore[arg-type]
        hook.on_plugin_error(spec, RuntimeError("test"))

        assert hook.discovery_count == 1
        assert hook.loaded_count == 1
        assert hook.error_count == 1

    def test_metrics_hook_success_rate(self):
        """Test success rate calculation."""
        hook = MetricsPluginHook()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        hook.on_plugin_loaded(spec, None)  # type: ignore[arg-type]
        hook.on_plugin_loaded(spec, None)  # type: ignore[arg-type]
        hook.on_plugin_error(spec, RuntimeError("test"))

        assert hook.success_rate == pytest.approx(2 / 3)

    def test_metrics_hook_reset(self):
        """Test resetting metrics."""
        hook = MetricsPluginHook()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        hook.on_plugin_discovered(spec)
        hook.on_plugin_loaded(spec, None)  # type: ignore[arg-type]
        hook.reset()

        assert hook.discovery_count == 0
        assert hook.loaded_count == 0

    def test_get_stats(self):
        """Test getting all statistics."""
        hook = MetricsPluginHook()
        spec = PluginSpec(
            name="test",
            module_path="test.module",
            class_name="TestEngine",
        )

        hook.on_plugin_discovered(spec)
        hook.on_plugin_loaded(spec, None)  # type: ignore[arg-type]

        stats = hook.get_stats()

        assert stats["discovery_count"] == 1
        assert stats["loaded_count"] == 1
        assert stats["error_count"] == 0


class TestCompositePluginHook:
    """Tests for CompositePluginHook."""

    def test_composite_delegates_to_all_hooks(self):
        """Test that composite delegates to all hooks."""
        hook1 = MagicMock()
        hook2 = MagicMock()
        composite = CompositePluginHook([hook1, hook2])
        config = DiscoveryConfig()

        composite.on_discovery_start(config)

        hook1.on_discovery_start.assert_called_once_with(config)
        hook2.on_discovery_start.assert_called_once_with(config)

    def test_composite_add_remove_hooks(self):
        """Test adding and removing hooks."""
        hook1 = MagicMock()
        hook2 = MagicMock()
        composite = CompositePluginHook()

        composite.add_hook(hook1)
        composite.add_hook(hook2)
        composite.on_discovery_start(DiscoveryConfig())

        hook1.on_discovery_start.assert_called_once()
        hook2.on_discovery_start.assert_called_once()

        composite.remove_hook(hook1)
        hook1.reset_mock()
        hook2.reset_mock()

        composite.on_discovery_start(DiscoveryConfig())
        hook1.on_discovery_start.assert_not_called()
        hook2.on_discovery_start.assert_called_once()

    def test_composite_handles_hook_errors(self):
        """Test that composite handles hook errors gracefully."""
        hook1 = MagicMock()
        hook1.on_discovery_start.side_effect = RuntimeError("Hook error")
        hook2 = MagicMock()
        composite = CompositePluginHook([hook1, hook2])

        # Should not raise
        composite.on_discovery_start(DiscoveryConfig())

        # Hook2 should still be called
        hook2.on_discovery_start.assert_called_once()


# =============================================================================
# Tests: Plugin Registry
# =============================================================================


class TestPluginRegistry:
    """Tests for PluginRegistry."""

    def test_create_registry(self):
        """Test creating a registry."""
        registry = PluginRegistry()

        # Note: list_plugins() triggers auto-discovery, so we check _specs directly
        assert len(registry._specs) == 0
        assert registry._discovered is False

    def test_discover_builtins(self):
        """Test discovering builtin plugins."""
        registry = PluginRegistry()
        specs = registry.discover(
            include_builtins=True,
            include_entry_points=False,
        )

        names = [s.name for s in specs]
        assert "truthound" in names
        assert "great_expectations" in names
        assert "pandera" in names

    def test_get_engine(self):
        """Test getting an engine by name."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        engine = registry.get_engine("truthound")

        assert engine is not None
        assert engine.engine_name == "truthound"

    def test_get_engine_not_found(self):
        """Test getting non-existent engine raises error."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        with pytest.raises(PluginNotFoundError) as exc_info:
            registry.get_engine("nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert exc_info.value.available_plugins

    def test_get_spec(self):
        """Test getting a spec by name."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        spec = registry.get_spec("truthound")

        assert spec.name == "truthound"
        assert spec.class_name == "TruthoundEngine"

    def test_list_plugins(self):
        """Test listing all plugins."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        plugins = registry.list_plugins()

        assert "truthound" in plugins
        assert "great_expectations" in plugins
        assert "pandera" in plugins

    def test_list_aliases(self):
        """Test listing aliases."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        aliases = registry.list_aliases()

        assert "th" in aliases
        assert aliases["th"] == "truthound"
        assert "ge" in aliases
        assert aliases["ge"] == "great_expectations"

    def test_has_plugin(self):
        """Test checking if plugin exists."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        assert registry.has_plugin("truthound")
        assert registry.has_plugin("th")  # Alias
        assert not registry.has_plugin("nonexistent")

    def test_is_loaded(self):
        """Test checking if plugin is loaded."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        assert not registry.is_loaded("truthound")

        registry.get_engine("truthound")

        assert registry.is_loaded("truthound")

    def test_disable_enable_plugin(self):
        """Test disabling and enabling plugins."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        # Load first
        registry.get_engine("truthound")
        assert registry.is_loaded("truthound")

        # Disable
        registry.disable_plugin("truthound")
        spec = registry.get_spec("truthound")
        assert not spec.enabled
        assert not registry.is_loaded("truthound")

        # Enable
        registry.enable_plugin("truthound")
        spec = registry.get_spec("truthound")
        assert spec.enabled

    def test_unload_plugin(self):
        """Test unloading a plugin."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)

        registry.get_engine("truthound")
        assert registry.is_loaded("truthound")

        registry.unload_plugin("truthound")
        assert not registry.is_loaded("truthound")

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)
        registry.get_engine("truthound")

        registry.clear()

        # Note: list_plugins() triggers auto-discovery, so check _specs directly
        assert len(registry._specs) == 0
        assert registry._discovered is False

    def test_rediscover(self):
        """Test rediscovering plugins."""
        registry = PluginRegistry()
        specs1 = registry.discover(include_builtins=True, include_entry_points=False)

        specs2 = registry.rediscover(include_builtins=True, include_entry_points=False)

        assert len(specs1) == len(specs2)

    def test_get_stats(self):
        """Test getting registry stats."""
        registry = PluginRegistry()
        registry.discover(include_builtins=True, include_entry_points=False)
        registry.get_engine("truthound")

        stats = registry.get_stats()

        assert stats["discovered_count"] == 3
        assert stats["loaded_count"] == 1
        assert stats["is_discovered"] is True

    def test_add_source(self):
        """Test adding a custom source."""
        registry = PluginRegistry()
        custom_source = DictPluginSource(
            {"custom": "common.engines.truthound:TruthoundEngine"}
        )
        registry.add_source(custom_source)

        specs = registry.discover(include_builtins=False, include_entry_points=False)

        assert any(s.name == "custom" for s in specs)

    def test_priority_conflict_resolution(self):
        """Test that higher priority wins in conflicts."""
        registry = PluginRegistry()

        # Create two sources with same plugin name
        source1 = DictPluginSource(
            {"test": "common.engines.truthound:TruthoundEngine"}
        )
        source2 = DictPluginSource(
            {"test": "common.engines.pandera:PanderaAdapter"}
        )

        registry.add_source(source1)
        registry.add_source(source2)

        registry.discover(include_builtins=False, include_entry_points=False)

        # First one should win (same priority, first registered)
        # The registry stores only one spec per name
        assert "test" in registry._specs
        spec = registry._specs["test"]
        assert spec.class_name == "TruthoundEngine"  # First one wins

    def test_config_disabled_plugins(self):
        """Test that disabled plugins are respected."""
        config = DiscoveryConfig().with_disabled_plugins("truthound")
        registry = PluginRegistry(config=config)

        specs = registry.discover(include_builtins=True, include_entry_points=False)

        truthound_spec = next((s for s in specs if s.name == "truthound"), None)
        assert truthound_spec is not None
        assert not truthound_spec.enabled

    def test_config_priority_overrides(self):
        """Test that priority overrides are applied."""
        config = DiscoveryConfig().with_priority_override("truthound", 999)
        registry = PluginRegistry(config=config)

        specs = registry.discover(include_builtins=True, include_entry_points=False)

        truthound_spec = next(s for s in specs if s.name == "truthound")
        assert truthound_spec.priority == 999


# =============================================================================
# Tests: Global Functions
# =============================================================================


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_plugin_registry_returns_singleton(self, reset_global_registry):
        """Test that get_plugin_registry returns the same instance."""
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()

        assert registry1 is registry2

    def test_reset_plugin_registry(self, reset_global_registry):
        """Test resetting the global registry."""
        registry1 = get_plugin_registry()
        reset_plugin_registry()
        registry2 = get_plugin_registry()

        assert registry1 is not registry2

    def test_discover_plugins_function(self, reset_global_registry):
        """Test discover_plugins convenience function."""
        specs = discover_plugins(
            include_builtins=True,
            include_entry_points=False,
        )

        assert len(specs) >= 3
        names = [s.name for s in specs]
        assert "truthound" in names

    def test_load_plugins_function(self, reset_global_registry):
        """Test load_plugins convenience function."""
        discover_plugins(include_builtins=True, include_entry_points=False)
        engines = load_plugins(["truthound", "pandera"])

        assert "truthound" in engines
        assert "pandera" in engines

    def test_get_plugin_engine_function(self, reset_global_registry):
        """Test get_plugin_engine convenience function."""
        discover_plugins(include_builtins=True, include_entry_points=False)
        engine = get_plugin_engine("truthound")

        assert engine.engine_name == "truthound"

    def test_validate_plugin_function(self, mock_engine_class):
        """Test validate_plugin convenience function."""
        result = validate_plugin(mock_engine_class)

        assert result.is_valid is True

    def test_register_plugin_function(self, reset_global_registry):
        """Test register_plugin convenience function."""
        spec = register_plugin(
            "custom",
            "common.engines.truthound:TruthoundEngine",
            aliases=["c"],
        )

        assert spec.name == "custom"
        assert "c" in spec.aliases

        # Should be accessible
        registry = get_plugin_registry()
        assert registry.has_plugin("custom")


# =============================================================================
# Tests: Integration with EngineRegistry
# =============================================================================


class TestEngineRegistryIntegration:
    """Tests for integration with EngineRegistry."""

    def test_auto_discover_engines(self, reset_global_registry):
        """Test auto_discover_engines function."""
        from common.engines.registry import get_engine_registry

        # Reset engine registry
        engine_registry = get_engine_registry()
        initial_count = len(engine_registry.list())

        # Auto-discover (should register plugins)
        specs = auto_discover_engines()

        # Check that plugins were registered
        assert len(specs) > 0

    def test_enable_plugin_discovery(self, reset_global_registry):
        """Test enabling plugin discovery on EngineRegistry."""
        from common.engines.registry import (
            enable_plugin_discovery,
            get_engine_registry,
            is_plugin_discovery_enabled,
        )

        assert not is_plugin_discovery_enabled()

        enable_plugin_discovery()

        assert is_plugin_discovery_enabled()

    def test_discover_and_register_plugins(self, reset_global_registry):
        """Test discover_and_register_plugins function."""
        from common.engines.registry import (
            discover_and_register_plugins,
            get_engine_registry,
        )

        specs = discover_and_register_plugins(
            include_builtins=True,
            include_entry_points=False,
        )

        # Builtins are already registered, so this mainly tests the flow
        assert len(specs) >= 3


# =============================================================================
# Tests: Exceptions
# =============================================================================


class TestPluginExceptions:
    """Tests for plugin exceptions."""

    def test_plugin_error_basic(self):
        """Test basic PluginError."""
        error = PluginError("Test error")

        assert str(error) == "Test error"
        assert error.plugin_name is None
        assert error.details == {}
        assert error.cause is None

    def test_plugin_error_with_details(self):
        """Test PluginError with all details."""
        cause = RuntimeError("Original")
        error = PluginError(
            "Test error",
            plugin_name="test",
            details={"key": "value"},
            cause=cause,
        )

        assert error.plugin_name == "test"
        assert error.details == {"key": "value"}
        assert error.cause is cause

    def test_plugin_not_found_error(self):
        """Test PluginNotFoundError."""
        error = PluginNotFoundError(
            "missing",
            available_plugins=["a", "b", "c"],
        )

        assert "missing" in str(error)
        assert error.plugin_name == "missing"
        assert error.available_plugins == ["a", "b", "c"]

    def test_plugin_validation_error(self):
        """Test PluginValidationError."""
        error = PluginValidationError(
            "Validation failed",
            plugin_name="test",
            validation_errors=["Error 1", "Error 2"],
        )

        assert error.plugin_name == "test"
        assert error.validation_errors == ["Error 1", "Error 2"]

    def test_plugin_conflict_error(self):
        """Test PluginConflictError."""
        error = PluginConflictError(
            "Conflict detected",
            plugin_name="test",
            conflicting_plugins=["a", "b"],
        )

        assert error.plugin_name == "test"
        assert error.conflicting_plugins == ["a", "b"]
