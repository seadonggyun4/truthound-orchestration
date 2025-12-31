"""Tests for Enterprise Engine Registry.

This module tests the registry functionality including:
- EnterpriseEngineRegistry
- EngineRegistration
- Global registry functions
- Plugin integration
"""

from __future__ import annotations

import pytest

from packages.enterprise.engines.registry import (
    # Registry
    EnterpriseEngineRegistry,
    EngineRegistration,
    # Global functions
    get_enterprise_engine_registry,
    reset_enterprise_engine_registry,
    get_enterprise_engine,
    register_enterprise_engine,
    list_enterprise_engines,
    is_enterprise_engine_registered,
    # Plugin integration
    create_plugin_spec,
    register_with_common_registry,
    # Exceptions
    EngineNotRegisteredError,
    EngineAlreadyRegisteredError,
)
from packages.enterprise.engines.base import EnterpriseEngineAdapter, EnterpriseEngineConfig
from packages.enterprise.engines.informatica import InformaticaAdapter, InformaticaConfig
from packages.enterprise.engines.talend import TalendAdapter, TalendConfig


# =============================================================================
# EngineRegistration Tests
# =============================================================================


class TestEngineRegistration:
    """Tests for EngineRegistration dataclass."""

    def test_basic_registration(self):
        """Test basic registration creation."""
        registration = EngineRegistration(
            name="test",
            engine_class=InformaticaAdapter,
        )

        assert registration.name == "test"
        assert registration.engine_class is InformaticaAdapter
        assert registration.factory is None
        assert registration.default_config is None
        assert registration.aliases == ()
        assert registration.priority == 100

    def test_registration_with_config(self):
        """Test registration with default config."""
        config = InformaticaConfig(api_endpoint="https://test.example.com")
        registration = EngineRegistration(
            name="informatica",
            engine_class=InformaticaAdapter,
            default_config=config,
        )

        assert registration.default_config is config

    def test_registration_with_aliases(self):
        """Test registration with aliases."""
        registration = EngineRegistration(
            name="informatica",
            engine_class=InformaticaAdapter,
            aliases=("idq", "informatica_dq"),
        )

        assert "idq" in registration.aliases
        assert "informatica_dq" in registration.aliases

    def test_create_engine(self):
        """Test creating engine from registration."""
        registration = EngineRegistration(
            name="informatica",
            engine_class=InformaticaAdapter,
        )

        engine = registration.create_engine()

        assert isinstance(engine, InformaticaAdapter)
        assert engine.engine_name == "informatica"

    def test_create_engine_with_config(self):
        """Test creating engine with config override."""
        default_config = InformaticaConfig(api_endpoint="https://default.example.com")
        override_config = InformaticaConfig(api_endpoint="https://override.example.com")

        registration = EngineRegistration(
            name="informatica",
            engine_class=InformaticaAdapter,
            default_config=default_config,
        )

        # Create with default
        engine1 = registration.create_engine()
        assert engine1._config.api_endpoint == "https://default.example.com"

        # Create with override
        engine2 = registration.create_engine(config=override_config)
        assert engine2._config.api_endpoint == "https://override.example.com"

    def test_create_engine_with_factory(self):
        """Test creating engine with custom factory."""
        created_with_factory = []

        def custom_factory(config=None, **kwargs):
            created_with_factory.append(True)
            return InformaticaAdapter(config=config)

        registration = EngineRegistration(
            name="informatica",
            engine_class=InformaticaAdapter,
            factory=custom_factory,
        )

        engine = registration.create_engine()

        assert len(created_with_factory) == 1
        assert isinstance(engine, InformaticaAdapter)


# =============================================================================
# EnterpriseEngineRegistry Tests
# =============================================================================


class TestEnterpriseEngineRegistry:
    """Tests for EnterpriseEngineRegistry."""

    def test_registry_initialization(self):
        """Test registry is properly initialized."""
        registry = EnterpriseEngineRegistry()

        # Should have built-in engines after first access
        engines = registry.list_engines()

        assert "informatica" in engines
        assert "talend" in engines

    def test_register_custom_engine(self):
        """Test registering a custom engine."""
        registry = EnterpriseEngineRegistry()

        # Create a mock engine class
        class CustomAdapter(EnterpriseEngineAdapter):
            @property
            def engine_name(self):
                return "custom"

            @property
            def engine_version(self):
                return "1.0.0"

        registry.register(
            "custom",
            CustomAdapter,
            description="Custom engine",
        )

        assert registry.is_registered("custom")

    def test_register_with_aliases(self):
        """Test registering engine with aliases."""
        registry = EnterpriseEngineRegistry()

        class CustomAdapter(EnterpriseEngineAdapter):
            @property
            def engine_name(self):
                return "custom"

            @property
            def engine_version(self):
                return "1.0.0"

        registry.register(
            "custom",
            CustomAdapter,
            aliases=("c", "cust"),
        )

        assert registry.is_registered("custom")
        assert registry.is_registered("c")
        assert registry.is_registered("cust")

    def test_register_duplicate_fails(self):
        """Test that duplicate registration raises error."""
        registry = EnterpriseEngineRegistry()

        # informatica is already registered during initialization
        with pytest.raises(EngineAlreadyRegisteredError):
            registry.register("informatica", InformaticaAdapter)

    def test_register_with_force(self):
        """Test overriding registration with force=True."""
        registry = EnterpriseEngineRegistry()

        # Should succeed with force=True
        registry.register(
            "informatica",
            InformaticaAdapter,
            force=True,
            description="Overridden",
        )

        reg = registry.get_registration("informatica")
        assert reg.description == "Overridden"

    def test_get_engine(self):
        """Test getting an engine instance."""
        registry = EnterpriseEngineRegistry()

        engine = registry.get("informatica")

        assert isinstance(engine, InformaticaAdapter)
        assert engine.engine_name == "informatica"

    def test_get_engine_by_alias(self):
        """Test getting engine by alias."""
        registry = EnterpriseEngineRegistry()

        engine = registry.get("idq")  # Alias for informatica

        assert isinstance(engine, InformaticaAdapter)

    def test_get_engine_not_found(self):
        """Test getting non-existent engine raises error."""
        registry = EnterpriseEngineRegistry()

        with pytest.raises(EngineNotRegisteredError) as exc_info:
            registry.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_registration(self):
        """Test getting registration entry."""
        registry = EnterpriseEngineRegistry()

        reg = registry.get_registration("informatica")

        assert reg is not None
        assert reg.name == "informatica"
        assert reg.engine_class is InformaticaAdapter

    def test_unregister(self):
        """Test unregistering an engine."""
        registry = EnterpriseEngineRegistry()

        class TempAdapter(EnterpriseEngineAdapter):
            @property
            def engine_name(self):
                return "temp"

            @property
            def engine_version(self):
                return "1.0.0"

        registry.register("temp", TempAdapter, aliases=("t",))
        assert registry.is_registered("temp")
        assert registry.is_registered("t")

        result = registry.unregister("temp")

        assert result is True
        assert not registry.is_registered("temp")
        assert not registry.is_registered("t")

    def test_unregister_not_found(self):
        """Test unregistering non-existent engine."""
        registry = EnterpriseEngineRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_list_engines(self):
        """Test listing registered engines."""
        registry = EnterpriseEngineRegistry()

        engines = registry.list_engines()

        assert isinstance(engines, list)
        assert "informatica" in engines
        assert "talend" in engines

    def test_list_all_names(self):
        """Test listing all names including aliases."""
        registry = EnterpriseEngineRegistry()

        names = registry.list_all_names()

        assert "informatica" in names
        assert "idq" in names  # Alias
        assert "talend" in names
        assert "tdq" in names  # Alias

    def test_get_info(self):
        """Test getting info about all engines."""
        registry = EnterpriseEngineRegistry()

        info = registry.get_info()

        assert "informatica" in info
        assert "talend" in info
        assert info["informatica"]["class"] == "InformaticaAdapter"
        assert "idq" in info["informatica"]["aliases"]

    def test_reset(self):
        """Test resetting the registry."""
        registry = EnterpriseEngineRegistry()

        class TempAdapter(EnterpriseEngineAdapter):
            @property
            def engine_name(self):
                return "temp"

            @property
            def engine_version(self):
                return "1.0.0"

        registry.register("temp", TempAdapter)
        assert registry.is_registered("temp")

        registry.reset()

        # temp should be gone, but builtins should be back
        assert not registry.is_registered("temp")
        assert registry.is_registered("informatica")
        assert registry.is_registered("talend")


# =============================================================================
# Global Registry Function Tests
# =============================================================================


class TestGlobalRegistryFunctions:
    """Tests for global registry convenience functions."""

    def test_get_enterprise_engine_registry(self):
        """Test getting the global registry."""
        registry = get_enterprise_engine_registry()

        assert isinstance(registry, EnterpriseEngineRegistry)
        assert registry.is_registered("informatica")

    def test_get_enterprise_engine(self):
        """Test getting engine from global registry."""
        engine = get_enterprise_engine("informatica")

        assert isinstance(engine, InformaticaAdapter)

    def test_get_enterprise_engine_with_config(self):
        """Test getting engine with config override."""
        config = InformaticaConfig(api_endpoint="https://custom.example.com")

        engine = get_enterprise_engine("informatica", config=config)

        assert engine._config.api_endpoint == "https://custom.example.com"

    def test_list_enterprise_engines(self):
        """Test listing engines from global registry."""
        engines = list_enterprise_engines()

        assert "informatica" in engines
        assert "talend" in engines

    def test_is_enterprise_engine_registered(self):
        """Test checking if engine is registered."""
        assert is_enterprise_engine_registered("informatica") is True
        assert is_enterprise_engine_registered("idq") is True  # Alias
        assert is_enterprise_engine_registered("nonexistent") is False

    def test_register_enterprise_engine(self):
        """Test registering engine via global function."""

        class TestAdapter(EnterpriseEngineAdapter):
            @property
            def engine_name(self):
                return "test_global"

            @property
            def engine_version(self):
                return "1.0.0"

        register_enterprise_engine(
            "test_global_reg",
            TestAdapter,
            aliases=("tgr",),
            force=True,
        )

        assert is_enterprise_engine_registered("test_global_reg")
        assert is_enterprise_engine_registered("tgr")

        # Cleanup
        get_enterprise_engine_registry().unregister("test_global_reg")


# =============================================================================
# Plugin Integration Tests
# =============================================================================


class TestPluginIntegration:
    """Tests for plugin discovery integration."""

    def test_create_plugin_spec(self):
        """Test creating a plugin specification."""
        spec = create_plugin_spec(
            "informatica",
            InformaticaAdapter,
            priority=100,
            aliases=("idq",),
        )

        assert spec["name"] == "informatica"
        assert spec["class_name"] == "InformaticaAdapter"
        assert spec["plugin_type"] == "ENGINE"
        assert spec["priority"] == 100
        assert "idq" in spec["aliases"]
        assert spec["metadata"]["enterprise"] is True

    def test_register_with_common_registry_no_error(self):
        """Test that register_with_common_registry doesn't raise errors."""
        # Should not raise even if common.engines is not available
        # or if engines are already registered
        register_with_common_registry()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestRegistryThreadSafety:
    """Tests for thread safety of registry operations."""

    def test_concurrent_access(self):
        """Test concurrent access to registry."""
        import threading
        import concurrent.futures

        registry = EnterpriseEngineRegistry()
        results = []
        errors = []

        def get_engine(name: str):
            try:
                engine = registry.get(name)
                results.append(engine.engine_name)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for _ in range(50):
                futures.append(executor.submit(get_engine, "informatica"))
                futures.append(executor.submit(get_engine, "talend"))

            concurrent.futures.wait(futures)

        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_registration(self):
        """Test concurrent registration doesn't corrupt state."""
        import threading
        import concurrent.futures

        registry = EnterpriseEngineRegistry()
        errors = []

        class ConcurrentAdapter(EnterpriseEngineAdapter):
            def __init__(self, idx):
                super().__init__()
                self._idx = idx

            @property
            def engine_name(self):
                return f"concurrent_{self._idx}"

            @property
            def engine_version(self):
                return "1.0.0"

        def register_engine(idx: int):
            try:
                # Create unique class for each registration
                class UniqueAdapter(EnterpriseEngineAdapter):
                    @property
                    def engine_name(self):
                        return f"concurrent_{idx}"

                    @property
                    def engine_version(self):
                        return "1.0.0"

                registry.register(
                    f"concurrent_{idx}",
                    UniqueAdapter,
                    force=True,
                )
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(register_engine, i)
                for i in range(20)
            ]
            concurrent.futures.wait(futures)

        assert len(errors) == 0

        # Verify all registrations succeeded
        for i in range(20):
            assert registry.is_registered(f"concurrent_{i}")

        # Cleanup
        for i in range(20):
            registry.unregister(f"concurrent_{i}")
