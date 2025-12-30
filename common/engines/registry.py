"""Engine Registry for managing data quality engines.

This module provides a registry pattern for managing and accessing
data quality engines throughout the application. It supports both
manual registration and automatic plugin discovery via entry points.

Example:
    >>> from common.engines import register_engine, get_engine
    >>> register_engine("custom", CustomEngine())
    >>> engine = get_engine("custom")
    >>> result = engine.check(data, rules)

Using the global registry:
    >>> from common.engines import get_engine_registry
    >>> registry = get_engine_registry()
    >>> registry.register("my_engine", MyEngine())
    >>> engine = registry.get("my_engine")

Using plugin discovery:
    >>> from common.engines import enable_plugin_discovery, discover_and_register_plugins
    >>> enable_plugin_discovery()  # Enable auto-discovery on first get_engine call
    >>> # Or explicitly discover plugins:
    >>> discover_and_register_plugins()  # Discover and register third-party engines
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from common.exceptions import TruthoundIntegrationError


if TYPE_CHECKING:
    from collections.abc import Sequence

    from common.engines.base import DataQualityEngine
    from common.engines.plugin import PluginHook, PluginSpec


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class EngineNotFoundError(TruthoundIntegrationError):
    """Exception raised when a requested engine is not found.

    Attributes:
        engine_name: Name of the engine that was not found.
        available_engines: List of available engine names.
    """

    def __init__(
        self,
        engine_name: str,
        *,
        available_engines: list[str] | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize engine not found error.

        Args:
            engine_name: Name of the engine that was not found.
            available_engines: List of available engine names.
            details: Additional error details.
            cause: Original exception that caused this error.
        """
        message = f"Engine '{engine_name}' not found"
        if available_engines:
            message += f". Available engines: {', '.join(available_engines)}"

        details = details or {}
        details["engine_name"] = engine_name
        if available_engines:
            details["available_engines"] = available_engines

        super().__init__(message, details=details, cause=cause)
        self.engine_name = engine_name
        self.available_engines = available_engines or []


class EngineAlreadyRegisteredError(TruthoundIntegrationError):
    """Exception raised when attempting to register a duplicate engine.

    Attributes:
        engine_name: Name of the engine that already exists.
    """

    def __init__(
        self,
        engine_name: str,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize engine already registered error.

        Args:
            engine_name: Name of the engine that already exists.
            details: Additional error details.
            cause: Original exception that caused this error.
        """
        message = f"Engine '{engine_name}' is already registered"

        details = details or {}
        details["engine_name"] = engine_name

        super().__init__(message, details=details, cause=cause)
        self.engine_name = engine_name


# =============================================================================
# Engine Registry
# =============================================================================


class EngineRegistry:
    """Registry for managing data quality engines.

    Thread-safe registry that allows registering, retrieving, and
    managing data quality engines. Supports a default engine that
    is used when no specific engine is requested.

    Attributes:
        _engines: Dictionary of registered engines.
        _default_engine_name: Name of the default engine.
        _lock: Thread lock for safe concurrent access.

    Example:
        >>> registry = EngineRegistry()
        >>> registry.register("truthound", TruthoundEngine())
        >>> registry.set_default("truthound")
        >>> engine = registry.get_default()
    """

    def __init__(self) -> None:
        """Initialize the engine registry."""
        self._engines: dict[str, DataQualityEngine] = {}
        self._default_engine_name: str | None = None
        self._lock = threading.RLock()
        self._plugin_discovery_enabled: bool = False
        self._plugins_discovered: bool = False

    def register(
        self,
        name: str,
        engine: DataQualityEngine,
        *,
        set_as_default: bool = False,
        allow_override: bool = False,
    ) -> None:
        """Register a data quality engine.

        Args:
            name: Unique name for the engine.
            engine: Engine instance to register.
            set_as_default: Whether to set this as the default engine.
            allow_override: Whether to allow overriding existing registration.

        Raises:
            EngineAlreadyRegisteredError: If engine exists and override not allowed.

        Example:
            >>> registry.register("custom", CustomEngine(), set_as_default=True)
        """
        with self._lock:
            if name in self._engines and not allow_override:
                raise EngineAlreadyRegisteredError(name)

            self._engines[name] = engine

            if set_as_default or self._default_engine_name is None:
                self._default_engine_name = name

    def unregister(self, name: str) -> DataQualityEngine | None:
        """Unregister an engine.

        Args:
            name: Name of the engine to unregister.

        Returns:
            The unregistered engine, or None if not found.

        Example:
            >>> engine = registry.unregister("old_engine")
        """
        with self._lock:
            engine = self._engines.pop(name, None)

            if self._default_engine_name == name:
                # Set a new default if we removed the default
                self._default_engine_name = (
                    next(iter(self._engines.keys())) if self._engines else None
                )

            return engine

    def get(self, name: str) -> DataQualityEngine:
        """Get an engine by name.

        If plugin discovery is enabled and the engine is not found,
        it will attempt to discover and load plugins first.

        Args:
            name: Name of the engine to retrieve.

        Returns:
            The requested engine.

        Raises:
            EngineNotFoundError: If the engine is not found.

        Example:
            >>> engine = registry.get("truthound")
        """
        with self._lock:
            engine = self._engines.get(name)
            if engine is None:
                # Try plugin discovery if enabled
                if self._plugin_discovery_enabled and not self._plugins_discovered:
                    self._discover_plugins_internal()
                    engine = self._engines.get(name)

            if engine is None:
                raise EngineNotFoundError(
                    name,
                    available_engines=list(self._engines.keys()),
                )
            return engine

    def enable_plugin_discovery(self) -> None:
        """Enable automatic plugin discovery.

        When enabled, the registry will automatically discover and load
        plugins from entry points when an unknown engine is requested.
        """
        with self._lock:
            self._plugin_discovery_enabled = True

    def disable_plugin_discovery(self) -> None:
        """Disable automatic plugin discovery."""
        with self._lock:
            self._plugin_discovery_enabled = False

    def is_plugin_discovery_enabled(self) -> bool:
        """Check if plugin discovery is enabled.

        Returns:
            True if plugin discovery is enabled.
        """
        with self._lock:
            return self._plugin_discovery_enabled

    def discover_plugins(
        self,
        include_builtins: bool = False,
        include_entry_points: bool = True,
        hooks: Sequence[PluginHook] | None = None,
    ) -> list[PluginSpec]:
        """Discover and register plugins.

        Args:
            include_builtins: Whether to include built-in engines.
            include_entry_points: Whether to include entry point plugins.
            hooks: Plugin lifecycle hooks.

        Returns:
            List of discovered plugin specifications.

        Example:
            >>> specs = registry.discover_plugins()
            >>> for spec in specs:
            ...     print(f"Discovered: {spec.name}")
        """
        with self._lock:
            return self._discover_plugins_internal(
                include_builtins=include_builtins,
                include_entry_points=include_entry_points,
                hooks=hooks,
            )

    def _discover_plugins_internal(
        self,
        include_builtins: bool = False,
        include_entry_points: bool = True,
        hooks: Sequence[PluginHook] | None = None,
    ) -> list[PluginSpec]:
        """Internal method to discover and register plugins.

        Args:
            include_builtins: Whether to include built-in engines.
            include_entry_points: Whether to include entry point plugins.
            hooks: Plugin lifecycle hooks.

        Returns:
            List of discovered plugin specifications.
        """
        from common.engines.plugin import (
            PluginRegistry,
            get_plugin_registry,
        )

        plugin_registry = get_plugin_registry()

        # Add hooks if provided
        if hooks:
            for hook in hooks:
                plugin_registry.add_hook(hook)

        # Discover plugins
        specs = plugin_registry.discover(
            include_builtins=include_builtins,
            include_entry_points=include_entry_points,
        )

        # Register discovered engines
        for spec in specs:
            if spec.enabled:
                try:
                    engine = plugin_registry.get_engine(spec.name)
                    # Don't override existing engines unless they came from plugins
                    if spec.name not in self._engines:
                        self._engines[spec.name] = engine
                        logger.debug("Registered plugin engine: %s", spec.name)

                        # Register aliases
                        for alias in spec.aliases:
                            if alias not in self._engines:
                                self._engines[alias] = engine
                                logger.debug("Registered plugin alias: %s -> %s", alias, spec.name)

                except Exception as e:
                    logger.warning("Failed to register plugin '%s': %s", spec.name, e)

        self._plugins_discovered = True
        return specs

    def get_or_none(self, name: str) -> DataQualityEngine | None:
        """Get an engine by name, returning None if not found.

        Args:
            name: Name of the engine to retrieve.

        Returns:
            The requested engine, or None if not found.
        """
        with self._lock:
            return self._engines.get(name)

    def get_default(self) -> DataQualityEngine:
        """Get the default engine.

        Returns:
            The default engine.

        Raises:
            EngineNotFoundError: If no default engine is set.
        """
        with self._lock:
            if self._default_engine_name is None:
                raise EngineNotFoundError(
                    "default",
                    available_engines=list(self._engines.keys()),
                )
            return self.get(self._default_engine_name)

    def set_default(self, name: str) -> None:
        """Set the default engine.

        Args:
            name: Name of the engine to set as default.

        Raises:
            EngineNotFoundError: If the engine is not found.
        """
        with self._lock:
            if name not in self._engines:
                raise EngineNotFoundError(
                    name,
                    available_engines=list(self._engines.keys()),
                )
            self._default_engine_name = name

    def list(self) -> list[str]:
        """List all registered engine names.

        Returns:
            List of engine names.
        """
        with self._lock:
            return list(self._engines.keys())

    def has(self, name: str) -> bool:
        """Check if an engine is registered.

        Args:
            name: Name of the engine to check.

        Returns:
            True if engine is registered, False otherwise.
        """
        with self._lock:
            return name in self._engines

    @property
    def default_engine_name(self) -> str | None:
        """Get the name of the default engine."""
        with self._lock:
            return self._default_engine_name

    def clear(self) -> None:
        """Clear all registered engines."""
        with self._lock:
            self._engines.clear()
            self._default_engine_name = None
            self._plugins_discovered = False

    def __len__(self) -> int:
        """Return the number of registered engines."""
        with self._lock:
            return len(self._engines)

    def __contains__(self, name: str) -> bool:
        """Check if an engine is registered."""
        return self.has(name)

    def __iter__(self):
        """Iterate over registered engine names."""
        with self._lock:
            return iter(list(self._engines.keys()))


# =============================================================================
# Global Registry
# =============================================================================

_global_registry: EngineRegistry | None = None
_registry_lock = threading.Lock()


def get_engine_registry() -> EngineRegistry:
    """Get the global engine registry.

    Lazily creates the global registry and registers default engines.

    Returns:
        The global EngineRegistry instance.

    Example:
        >>> registry = get_engine_registry()
        >>> registry.register("custom", CustomEngine())
    """
    global _global_registry

    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = EngineRegistry()
                _register_default_engines(_global_registry)

    return _global_registry


def _register_default_engines(registry: EngineRegistry) -> None:
    """Register default engines in the registry.

    Args:
        registry: Registry to populate with default engines.
    """
    # Import here to avoid circular imports
    from common.engines.great_expectations import GreatExpectationsAdapter
    from common.engines.pandera import PanderaAdapter
    from common.engines.truthound import TruthoundEngine

    # Register Truthound as the default engine
    registry.register("truthound", TruthoundEngine(), set_as_default=True)

    # Register other adapters
    registry.register("great_expectations", GreatExpectationsAdapter())
    registry.register("ge", GreatExpectationsAdapter())  # Alias
    registry.register("pandera", PanderaAdapter())


# =============================================================================
# Convenience Functions
# =============================================================================


def get_engine(name: str) -> DataQualityEngine:
    """Get an engine by name from the global registry.

    Args:
        name: Name of the engine to retrieve.

    Returns:
        The requested engine.

    Raises:
        EngineNotFoundError: If the engine is not found.

    Example:
        >>> engine = get_engine("truthound")
        >>> result = engine.check(data, rules)
    """
    return get_engine_registry().get(name)


def get_default_engine() -> DataQualityEngine:
    """Get the default engine from the global registry.

    Returns:
        The default engine (Truthound by default).

    Example:
        >>> engine = get_default_engine()
        >>> result = engine.check(data, rules)
    """
    return get_engine_registry().get_default()


def set_default_engine(name: str) -> None:
    """Set the default engine in the global registry.

    Args:
        name: Name of the engine to set as default.

    Example:
        >>> set_default_engine("great_expectations")
    """
    get_engine_registry().set_default(name)


def register_engine(
    name: str,
    engine: DataQualityEngine,
    *,
    set_as_default: bool = False,
    allow_override: bool = False,
) -> None:
    """Register an engine in the global registry.

    Args:
        name: Unique name for the engine.
        engine: Engine instance to register.
        set_as_default: Whether to set this as the default engine.
        allow_override: Whether to allow overriding existing registration.

    Example:
        >>> register_engine("custom", CustomEngine(), set_as_default=True)
    """
    get_engine_registry().register(
        name,
        engine,
        set_as_default=set_as_default,
        allow_override=allow_override,
    )


def list_engines() -> list[str]:
    """List all registered engine names from the global registry.

    Returns:
        List of engine names.

    Example:
        >>> engines = list_engines()
        >>> print(engines)  # ['truthound', 'great_expectations', 'pandera']
    """
    return get_engine_registry().list()


# =============================================================================
# Plugin Discovery Functions
# =============================================================================


def enable_plugin_discovery() -> None:
    """Enable automatic plugin discovery for the global registry.

    When enabled, the registry will automatically discover and load
    plugins from entry points when an unknown engine is requested.

    Example:
        >>> enable_plugin_discovery()
        >>> engine = get_engine("my_third_party_engine")  # Auto-discovered
    """
    get_engine_registry().enable_plugin_discovery()


def disable_plugin_discovery() -> None:
    """Disable automatic plugin discovery for the global registry."""
    get_engine_registry().disable_plugin_discovery()


def is_plugin_discovery_enabled() -> bool:
    """Check if plugin discovery is enabled for the global registry.

    Returns:
        True if plugin discovery is enabled.
    """
    return get_engine_registry().is_plugin_discovery_enabled()


def discover_and_register_plugins(
    include_builtins: bool = False,
    include_entry_points: bool = True,
    hooks: Sequence[PluginHook] | None = None,
) -> list[PluginSpec]:
    """Discover plugins from entry points and register with global registry.

    This function explicitly triggers plugin discovery and registration.
    Useful when you want to discover plugins before they are needed.

    Args:
        include_builtins: Whether to include built-in engines (normally
            already registered).
        include_entry_points: Whether to include entry point plugins.
        hooks: Plugin lifecycle hooks for monitoring discovery.

    Returns:
        List of discovered plugin specifications.

    Example:
        >>> from common.engines import discover_and_register_plugins
        >>> specs = discover_and_register_plugins()
        >>> for spec in specs:
        ...     print(f"Discovered: {spec.name} from {spec.source}")
        >>> # Now you can use the discovered engines
        >>> engine = get_engine("my_third_party_engine")
    """
    return get_engine_registry().discover_plugins(
        include_builtins=include_builtins,
        include_entry_points=include_entry_points,
        hooks=hooks,
    )
