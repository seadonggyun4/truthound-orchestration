"""Enterprise Engine Registry and Plugin Registration.

This module provides a registry for enterprise data quality engines,
enabling dynamic discovery and registration of enterprise adapters.

The registry integrates with the common.engines plugin discovery system,
allowing enterprise engines to be discovered via entry points.

Example:
    >>> from packages.enterprise.engines import (
    ...     EnterpriseEngineRegistry,
    ...     get_enterprise_engine,
    ...     register_enterprise_engine,
    ... )
    >>>
    >>> # Get a registered engine
    >>> engine = get_enterprise_engine("informatica")
    >>>
    >>> # Register a custom enterprise engine
    >>> register_enterprise_engine("custom", CustomAdapter)

Entry Point Registration:
    Add to pyproject.toml:
    ```toml
    [project.entry-points."truthound.engines"]
    informatica = "packages.enterprise.engines:InformaticaAdapter"
    talend = "packages.enterprise.engines:TalendAdapter"
    ```
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from packages.enterprise.engines.base import (
    EnterpriseEngineAdapter,
    EnterpriseEngineConfig,
    EnterpriseEngineError,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


# =============================================================================
# Types
# =============================================================================

EngineT = TypeVar("EngineT", bound=EnterpriseEngineAdapter)
EngineFactory = Callable[..., EnterpriseEngineAdapter]


# =============================================================================
# Exceptions
# =============================================================================


class EngineNotRegisteredError(EnterpriseEngineError):
    """Raised when an engine is not found in the registry."""

    def __init__(self, engine_name: str) -> None:
        """Initialize error.

        Args:
            engine_name: Name of the unregistered engine.
        """
        super().__init__(
            f"Enterprise engine '{engine_name}' is not registered. "
            f"Use register_enterprise_engine() to register it.",
            engine_name=engine_name,
        )


class EngineAlreadyRegisteredError(EnterpriseEngineError):
    """Raised when trying to register an already registered engine."""

    def __init__(self, engine_name: str) -> None:
        """Initialize error.

        Args:
            engine_name: Name of the already registered engine.
        """
        super().__init__(
            f"Enterprise engine '{engine_name}' is already registered. "
            f"Use force=True to override.",
            engine_name=engine_name,
        )


# =============================================================================
# Engine Registration Entry
# =============================================================================


@dataclass(frozen=True)
class EngineRegistration:
    """Registration entry for an enterprise engine.

    Attributes:
        name: Engine name.
        engine_class: Engine adapter class.
        factory: Optional factory function.
        default_config: Default configuration.
        aliases: Alternative names for the engine.
        description: Human-readable description.
        priority: Priority for conflict resolution (higher wins).
        metadata: Additional registration metadata.
    """

    name: str
    engine_class: type[EnterpriseEngineAdapter]
    factory: EngineFactory | None = None
    default_config: EnterpriseEngineConfig | None = None
    aliases: tuple[str, ...] = ()
    description: str = ""
    priority: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_engine(
        self,
        config: EnterpriseEngineConfig | None = None,
        **kwargs: Any,
    ) -> EnterpriseEngineAdapter:
        """Create an engine instance.

        Args:
            config: Optional configuration override.
            **kwargs: Additional arguments.

        Returns:
            Engine instance.
        """
        effective_config = config or self.default_config

        if self.factory:
            return self.factory(config=effective_config, **kwargs)
        else:
            return self.engine_class(config=effective_config)


# =============================================================================
# Enterprise Engine Registry
# =============================================================================


class EnterpriseEngineRegistry:
    """Registry for enterprise data quality engines.

    Provides centralized management of enterprise engine adapters
    with support for:
    - Engine registration and lookup
    - Alias support
    - Default configuration management
    - Factory functions for custom instantiation
    - Integration with plugin discovery system

    Thread-safe implementation using RLock.

    Example:
        >>> registry = EnterpriseEngineRegistry()
        >>>
        >>> # Register an engine
        >>> registry.register(
        ...     "informatica",
        ...     InformaticaAdapter,
        ...     default_config=InformaticaConfig(api_endpoint="..."),
        ... )
        >>>
        >>> # Get an engine
        >>> engine = registry.get("informatica")
        >>>
        >>> # List registered engines
        >>> for name in registry.list_engines():
        ...     print(name)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._registrations: dict[str, EngineRegistration] = {}
        self._aliases: dict[str, str] = {}
        self._lock = threading.RLock()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure registry is initialized with default engines."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Register built-in enterprise engines
            self._register_builtin_engines()
            self._initialized = True

    def _register_builtin_engines(self) -> None:
        """Register built-in enterprise engines."""
        # Import here to avoid circular imports
        from packages.enterprise.engines.informatica import (
            InformaticaAdapter,
            DEFAULT_INFORMATICA_CONFIG,
        )
        from packages.enterprise.engines.talend import (
            TalendAdapter,
            DEFAULT_TALEND_CONFIG,
        )
        from packages.enterprise.engines.ibm_infosphere import (
            IBMInfoSphereAdapter,
            DEFAULT_INFOSPHERE_CONFIG,
        )
        from packages.enterprise.engines.sap_data_services import (
            SAPDataServicesAdapter,
            DEFAULT_SAP_DS_CONFIG,
        )

        # Register Informatica
        self._register_internal(
            name="informatica",
            engine_class=InformaticaAdapter,
            default_config=DEFAULT_INFORMATICA_CONFIG,
            aliases=("idq", "informatica_dq"),
            description="Informatica Data Quality adapter",
            priority=100,
        )

        # Register Talend
        self._register_internal(
            name="talend",
            engine_class=TalendAdapter,
            default_config=DEFAULT_TALEND_CONFIG,
            aliases=("talend_dq", "tdq"),
            description="Talend Data Quality adapter",
            priority=100,
        )

        # Register IBM InfoSphere
        self._register_internal(
            name="ibm_infosphere",
            engine_class=IBMInfoSphereAdapter,
            default_config=DEFAULT_INFOSPHERE_CONFIG,
            aliases=("infosphere", "iis", "ibm_iis"),
            description="IBM InfoSphere Information Server adapter",
            priority=100,
        )

        # Register SAP Data Services
        self._register_internal(
            name="sap_data_services",
            engine_class=SAPDataServicesAdapter,
            default_config=DEFAULT_SAP_DS_CONFIG,
            aliases=("sap_ds", "sap", "bods"),
            description="SAP Data Services (BODS) adapter",
            priority=100,
        )

    def _register_internal(
        self,
        name: str,
        engine_class: type[EnterpriseEngineAdapter],
        factory: EngineFactory | None = None,
        default_config: EnterpriseEngineConfig | None = None,
        aliases: tuple[str, ...] = (),
        description: str = "",
        priority: int = 100,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Internal registration without initialization check.

        Args:
            name: Engine name.
            engine_class: Engine adapter class.
            factory: Optional factory function.
            default_config: Default configuration.
            aliases: Alternative names.
            description: Description.
            priority: Priority.
            metadata: Additional metadata.
        """
        registration = EngineRegistration(
            name=name,
            engine_class=engine_class,
            factory=factory,
            default_config=default_config,
            aliases=aliases,
            description=description,
            priority=priority,
            metadata=metadata or {},
        )

        self._registrations[name] = registration

        # Register aliases
        for alias in aliases:
            self._aliases[alias] = name

    def register(
        self,
        name: str,
        engine_class: type[EnterpriseEngineAdapter],
        *,
        factory: EngineFactory | None = None,
        default_config: EnterpriseEngineConfig | None = None,
        aliases: tuple[str, ...] = (),
        description: str = "",
        priority: int = 100,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        """Register an enterprise engine.

        Args:
            name: Engine name (should be lowercase, no spaces).
            engine_class: Engine adapter class.
            factory: Optional factory function for custom instantiation.
            default_config: Default configuration for this engine.
            aliases: Alternative names for the engine.
            description: Human-readable description.
            priority: Priority for conflict resolution (higher wins).
            metadata: Additional registration metadata.
            force: Whether to override existing registration.

        Raises:
            EngineAlreadyRegisteredError: If engine already registered and force=False.

        Example:
            >>> registry.register(
            ...     "my_engine",
            ...     MyEngineAdapter,
            ...     aliases=("me", "my"),
            ...     default_config=MyConfig(api_endpoint="..."),
            ... )
        """
        self._ensure_initialized()

        with self._lock:
            # Check if already registered
            if name in self._registrations and not force:
                existing = self._registrations[name]
                if existing.priority > priority:
                    return  # Higher priority already registered
                if existing.priority == priority:
                    raise EngineAlreadyRegisteredError(name)

            # Check alias conflicts
            for alias in aliases:
                if alias in self._aliases and not force:
                    existing_name = self._aliases[alias]
                    if existing_name != name:
                        existing = self._registrations.get(existing_name)
                        if existing and existing.priority >= priority:
                            raise EngineAlreadyRegisteredError(
                                f"{name} (alias '{alias}' conflicts with '{existing_name}')"
                            )

            self._register_internal(
                name=name,
                engine_class=engine_class,
                factory=factory,
                default_config=default_config,
                aliases=aliases,
                description=description,
                priority=priority,
                metadata=metadata,
            )

    def unregister(self, name: str) -> bool:
        """Unregister an enterprise engine.

        Args:
            name: Engine name or alias.

        Returns:
            True if engine was unregistered, False if not found.
        """
        self._ensure_initialized()

        with self._lock:
            # Resolve alias
            actual_name = self._aliases.get(name, name)

            if actual_name not in self._registrations:
                return False

            registration = self._registrations.pop(actual_name)

            # Remove aliases
            for alias in registration.aliases:
                self._aliases.pop(alias, None)

            return True

    def get(
        self,
        name: str,
        config: EnterpriseEngineConfig | None = None,
        **kwargs: Any,
    ) -> EnterpriseEngineAdapter:
        """Get an engine instance.

        Args:
            name: Engine name or alias.
            config: Optional configuration override.
            **kwargs: Additional arguments for engine creation.

        Returns:
            Engine instance.

        Raises:
            EngineNotRegisteredError: If engine not found.

        Example:
            >>> engine = registry.get("informatica")
            >>> engine = registry.get("informatica", config=custom_config)
        """
        self._ensure_initialized()

        with self._lock:
            # Resolve alias
            actual_name = self._aliases.get(name, name)

            if actual_name not in self._registrations:
                raise EngineNotRegisteredError(name)

            registration = self._registrations[actual_name]
            return registration.create_engine(config=config, **kwargs)

    def get_registration(self, name: str) -> EngineRegistration | None:
        """Get registration entry for an engine.

        Args:
            name: Engine name or alias.

        Returns:
            Registration entry or None if not found.
        """
        self._ensure_initialized()

        with self._lock:
            actual_name = self._aliases.get(name, name)
            return self._registrations.get(actual_name)

    def is_registered(self, name: str) -> bool:
        """Check if an engine is registered.

        Args:
            name: Engine name or alias.

        Returns:
            True if registered.
        """
        self._ensure_initialized()

        with self._lock:
            actual_name = self._aliases.get(name, name)
            return actual_name in self._registrations

    def list_engines(self) -> list[str]:
        """List all registered engine names.

        Returns:
            List of engine names (not including aliases).
        """
        self._ensure_initialized()

        with self._lock:
            return list(self._registrations.keys())

    def list_all_names(self) -> list[str]:
        """List all registered names including aliases.

        Returns:
            List of all names and aliases.
        """
        self._ensure_initialized()

        with self._lock:
            names = set(self._registrations.keys())
            names.update(self._aliases.keys())
            return sorted(names)

    def get_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all registered engines.

        Returns:
            Dictionary mapping engine names to their info.
        """
        self._ensure_initialized()

        with self._lock:
            info: dict[str, dict[str, Any]] = {}
            for name, reg in self._registrations.items():
                info[name] = {
                    "class": reg.engine_class.__name__,
                    "module": reg.engine_class.__module__,
                    "aliases": list(reg.aliases),
                    "description": reg.description,
                    "priority": reg.priority,
                    "has_factory": reg.factory is not None,
                    "has_default_config": reg.default_config is not None,
                    "metadata": reg.metadata,
                }
            return info

    def reset(self) -> None:
        """Reset the registry to initial state.

        Clears all registrations and re-registers built-in engines.
        """
        with self._lock:
            self._registrations.clear()
            self._aliases.clear()
            self._initialized = False
            self._ensure_initialized()


# =============================================================================
# Global Registry Instance
# =============================================================================

_registry: EnterpriseEngineRegistry | None = None
_registry_lock = threading.Lock()


def get_enterprise_engine_registry() -> EnterpriseEngineRegistry:
    """Get the global enterprise engine registry.

    Returns:
        Global EnterpriseEngineRegistry instance.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = EnterpriseEngineRegistry()
    return _registry


def reset_enterprise_engine_registry() -> None:
    """Reset the global enterprise engine registry."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.reset()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_enterprise_engine(
    name: str,
    config: EnterpriseEngineConfig | None = None,
    **kwargs: Any,
) -> EnterpriseEngineAdapter:
    """Get an enterprise engine by name.

    Convenience function that uses the global registry.

    Args:
        name: Engine name or alias (e.g., "informatica", "talend").
        config: Optional configuration override.
        **kwargs: Additional arguments for engine creation.

    Returns:
        Engine instance.

    Raises:
        EngineNotRegisteredError: If engine not found.

    Example:
        >>> engine = get_enterprise_engine("informatica")
        >>> with engine:
        ...     result = engine.check(data, rules)
    """
    return get_enterprise_engine_registry().get(name, config=config, **kwargs)


def register_enterprise_engine(
    name: str,
    engine_class: type[EnterpriseEngineAdapter],
    *,
    factory: EngineFactory | None = None,
    default_config: EnterpriseEngineConfig | None = None,
    aliases: tuple[str, ...] = (),
    description: str = "",
    force: bool = False,
) -> None:
    """Register an enterprise engine.

    Convenience function that uses the global registry.

    Args:
        name: Engine name.
        engine_class: Engine adapter class.
        factory: Optional factory function.
        default_config: Default configuration.
        aliases: Alternative names.
        description: Description.
        force: Override existing registration.

    Example:
        >>> register_enterprise_engine(
        ...     "my_custom",
        ...     MyCustomAdapter,
        ...     aliases=("custom",),
        ... )
    """
    get_enterprise_engine_registry().register(
        name=name,
        engine_class=engine_class,
        factory=factory,
        default_config=default_config,
        aliases=aliases,
        description=description,
        force=force,
    )


def list_enterprise_engines() -> list[str]:
    """List all registered enterprise engines.

    Convenience function that uses the global registry.

    Returns:
        List of engine names.
    """
    return get_enterprise_engine_registry().list_engines()


def is_enterprise_engine_registered(name: str) -> bool:
    """Check if an enterprise engine is registered.

    Convenience function that uses the global registry.

    Args:
        name: Engine name or alias.

    Returns:
        True if registered.
    """
    return get_enterprise_engine_registry().is_registered(name)


# =============================================================================
# Plugin Discovery Integration
# =============================================================================


def create_plugin_spec(
    name: str,
    engine_class: type[EnterpriseEngineAdapter],
    *,
    priority: int = 100,
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create a plugin specification for an enterprise engine.

    Use this to register enterprise engines with the common plugin system.

    Args:
        name: Engine name.
        engine_class: Engine adapter class.
        priority: Plugin priority.
        aliases: Alternative names.

    Returns:
        Plugin specification dictionary.

    Example:
        In your package's __init__.py:
        >>> from common.engines import register_plugin
        >>> from packages.enterprise.engines import create_plugin_spec, InformaticaAdapter
        >>>
        >>> spec = create_plugin_spec(
        ...     "informatica",
        ...     InformaticaAdapter,
        ...     aliases=("idq",),
        ... )
        >>> register_plugin(spec)
    """
    return {
        "name": name,
        "module_path": engine_class.__module__,
        "class_name": engine_class.__name__,
        "plugin_type": "ENGINE",
        "priority": priority,
        "aliases": aliases,
        "metadata": {
            "enterprise": True,
            "adapter_base": "EnterpriseEngineAdapter",
        },
    }


def register_with_common_registry() -> None:
    """Register enterprise engines with the common engine registry.

    This makes enterprise engines available through the standard
    get_engine() function in common.engines.

    Example:
        >>> from packages.enterprise.engines import register_with_common_registry
        >>> register_with_common_registry()
        >>>
        >>> from common.engines import get_engine
        >>> engine = get_engine("informatica")  # Now works!
    """
    try:
        from common.engines import register_engine
    except ImportError:
        return  # common.engines not available

    registry = get_enterprise_engine_registry()

    for name in registry.list_engines():
        registration = registry.get_registration(name)
        if registration:
            # Register with common registry
            try:
                register_engine(
                    name,
                    registration.engine_class,
                    force=False,  # Don't override if already registered
                )
                # Register aliases
                for alias in registration.aliases:
                    try:
                        register_engine(alias, registration.engine_class, force=False)
                    except Exception:
                        pass
            except Exception:
                pass  # Ignore registration errors
