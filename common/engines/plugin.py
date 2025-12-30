"""Plugin Discovery System for Data Quality Engines.

This module provides an extensible plugin discovery mechanism based on
Python entry points. Third-party packages can register their engines
by declaring entry points in their pyproject.toml or setup.py.

Entry Point Group: truthound.engines

Example pyproject.toml configuration:
    [project.entry-points."truthound.engines"]
    my_engine = "my_package.engines:MyEngine"
    another_engine = "my_package:AnotherEngine"

Example setup.py configuration:
    setup(
        entry_points={
            "truthound.engines": [
                "my_engine = my_package.engines:MyEngine",
            ],
        },
    )

Quick Start:
    >>> from common.engines.plugin import discover_plugins, load_plugins
    >>> # Discover all available plugins
    >>> plugins = discover_plugins()
    >>> for plugin in plugins:
    ...     print(f"{plugin.name}: {plugin.module_path}")
    >>> # Load and register all plugins
    >>> engines = load_plugins()
    >>> # Use with registry
    >>> from common.engines.plugin import auto_discover_engines
    >>> auto_discover_engines()  # Automatically registers discovered engines

Plugin Lifecycle Hooks:
    >>> from common.engines.plugin import PluginHook, LoggingPluginHook
    >>> hook = LoggingPluginHook()
    >>> plugins = discover_plugins(hooks=[hook])

Custom Plugin Sources:
    >>> from common.engines.plugin import PluginSource, ConfigFilePluginSource
    >>> source = ConfigFilePluginSource("plugins.yaml")
    >>> plugins = discover_plugins(sources=[source])

Plugin Validation:
    >>> from common.engines.plugin import PluginValidator, validate_plugin
    >>> result = validate_plugin(MyEngine)
    >>> if not result.is_valid:
    ...     print(f"Errors: {result.errors}")
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
import threading
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from common.engines.base import DataQualityEngine


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

ENTRY_POINT_GROUP = "truthound.engines"
"""The entry point group name for Truthound engine plugins."""

DEFAULT_PLUGIN_PRIORITY = 100
"""Default priority for plugins (lower = higher priority)."""


# =============================================================================
# Enums
# =============================================================================


class PluginState(Enum):
    """Plugin lifecycle state."""

    DISCOVERED = auto()
    """Plugin has been discovered but not loaded."""

    LOADING = auto()
    """Plugin is currently being loaded."""

    LOADED = auto()
    """Plugin has been successfully loaded."""

    FAILED = auto()
    """Plugin failed to load."""

    DISABLED = auto()
    """Plugin is explicitly disabled."""


class PluginType(Enum):
    """Type of plugin."""

    ENGINE = auto()
    """Data quality engine plugin."""

    ADAPTER = auto()
    """Adapter for existing engines."""

    HOOK = auto()
    """Hook plugin for extensibility."""

    VALIDATOR = auto()
    """Rule validator plugin."""


class LoadStrategy(Enum):
    """Strategy for loading plugins."""

    EAGER = auto()
    """Load plugin immediately on discovery."""

    LAZY = auto()
    """Load plugin only when first accessed."""

    ON_DEMAND = auto()
    """Load plugin only when explicitly requested."""


# =============================================================================
# Exceptions
# =============================================================================


class PluginError(Exception):
    """Base exception for plugin-related errors.

    Attributes:
        plugin_name: Name of the plugin that caused the error.
        details: Additional error details.
        cause: Original exception that caused this error.
    """

    def __init__(
        self,
        message: str,
        *,
        plugin_name: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize plugin error.

        Args:
            message: Error message.
            plugin_name: Name of the plugin that caused the error.
            details: Additional error details.
            cause: Original exception that caused this error.
        """
        super().__init__(message)
        self.plugin_name = plugin_name
        self.details = details or {}
        self.cause = cause


class PluginDiscoveryError(PluginError):
    """Error during plugin discovery.

    Raised when plugins cannot be discovered from entry points or other sources.
    """

    pass


class PluginLoadError(PluginError):
    """Error loading a plugin.

    Raised when a discovered plugin cannot be loaded or instantiated.
    """

    pass


class PluginValidationError(PluginError):
    """Error validating a plugin.

    Raised when a plugin does not meet the required interface or constraints.

    Attributes:
        validation_errors: List of validation error messages.
    """

    def __init__(
        self,
        message: str,
        *,
        plugin_name: str | None = None,
        validation_errors: list[str] | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Error message.
            plugin_name: Name of the plugin that caused the error.
            validation_errors: List of validation error messages.
            details: Additional error details.
            cause: Original exception that caused this error.
        """
        super().__init__(
            message,
            plugin_name=plugin_name,
            details=details,
            cause=cause,
        )
        self.validation_errors = validation_errors or []


class PluginConflictError(PluginError):
    """Error when plugins conflict.

    Raised when multiple plugins provide the same functionality
    and conflict resolution fails.

    Attributes:
        conflicting_plugins: List of conflicting plugin names.
    """

    def __init__(
        self,
        message: str,
        *,
        plugin_name: str | None = None,
        conflicting_plugins: list[str] | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize conflict error.

        Args:
            message: Error message.
            plugin_name: Name of the plugin that caused the error.
            conflicting_plugins: List of conflicting plugin names.
            details: Additional error details.
            cause: Original exception that caused this error.
        """
        super().__init__(
            message,
            plugin_name=plugin_name,
            details=details,
            cause=cause,
        )
        self.conflicting_plugins = conflicting_plugins or []


class PluginNotFoundError(PluginError):
    """Error when a requested plugin is not found.

    Attributes:
        available_plugins: List of available plugin names.
    """

    def __init__(
        self,
        plugin_name: str,
        *,
        available_plugins: list[str] | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize not found error.

        Args:
            plugin_name: Name of the plugin that was not found.
            available_plugins: List of available plugin names.
            details: Additional error details.
            cause: Original exception that caused this error.
        """
        message = f"Plugin '{plugin_name}' not found"
        if available_plugins:
            message += f". Available plugins: {', '.join(available_plugins)}"

        super().__init__(
            message,
            plugin_name=plugin_name,
            details=details,
            cause=cause,
        )
        self.available_plugins = available_plugins or []


# =============================================================================
# Data Types
# =============================================================================


@dataclass(frozen=True)
class PluginMetadata:
    """Immutable metadata about a plugin.

    Attributes:
        name: Plugin name (from entry point name).
        version: Plugin version (from package version).
        description: Plugin description.
        author: Plugin author.
        homepage: Plugin homepage URL.
        license: Plugin license.
        tags: Plugin tags for categorization.
        dependencies: Plugin dependencies.
        python_requires: Python version requirement.
    """

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""
    tags: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    python_requires: str = ""

    @classmethod
    def from_distribution(
        cls,
        name: str,
        dist: importlib.metadata.Distribution | None = None,
    ) -> PluginMetadata:
        """Create metadata from a distribution.

        Args:
            name: Plugin name.
            dist: Python distribution object.

        Returns:
            Plugin metadata extracted from the distribution.
        """
        if dist is None:
            return cls(name=name)

        metadata = dist.metadata

        return cls(
            name=name,
            version=metadata.get("Version", ""),
            description=metadata.get("Summary", ""),
            author=metadata.get("Author", "") or metadata.get("Author-email", ""),
            homepage=metadata.get("Home-page", "") or metadata.get("Project-URL", ""),
            license=metadata.get("License", ""),
            python_requires=metadata.get("Requires-Python", ""),
        )


@dataclass(frozen=True)
class PluginSpec:
    """Specification for a discovered plugin.

    Contains all information needed to load and instantiate a plugin.

    Attributes:
        name: Unique plugin name.
        module_path: Full module path (e.g., "my_package.engines").
        class_name: Class name within the module (e.g., "MyEngine").
        plugin_type: Type of plugin.
        priority: Loading priority (lower = higher priority).
        enabled: Whether the plugin is enabled.
        metadata: Additional metadata about the plugin.
        source: Where the plugin was discovered from.
        aliases: Alternative names for the plugin.
        config: Default configuration for the plugin.
    """

    name: str
    module_path: str
    class_name: str
    plugin_type: PluginType = PluginType.ENGINE
    priority: int = DEFAULT_PLUGIN_PRIORITY
    enabled: bool = True
    metadata: PluginMetadata | None = None
    source: str = "entry_point"
    aliases: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def full_path(self) -> str:
        """Get the full import path (module:class)."""
        return f"{self.module_path}:{self.class_name}"

    @classmethod
    def from_entry_point(
        cls,
        ep: importlib.metadata.EntryPoint,
        *,
        plugin_type: PluginType = PluginType.ENGINE,
        priority: int = DEFAULT_PLUGIN_PRIORITY,
    ) -> PluginSpec:
        """Create a plugin spec from an entry point.

        Args:
            ep: Entry point object.
            plugin_type: Type of plugin.
            priority: Loading priority.

        Returns:
            Plugin specification.
        """
        # Parse the entry point value (module:class or module.class)
        if ":" in ep.value:
            module_path, class_name = ep.value.rsplit(":", 1)
        else:
            module_path, class_name = ep.value.rsplit(".", 1)

        # Try to get metadata from the distribution
        metadata: PluginMetadata | None = None
        try:
            if hasattr(ep, "dist") and ep.dist is not None:
                metadata = PluginMetadata.from_distribution(ep.name, ep.dist)
        except Exception:
            metadata = PluginMetadata(name=ep.name)

        return cls(
            name=ep.name,
            module_path=module_path,
            class_name=class_name,
            plugin_type=plugin_type,
            priority=priority,
            metadata=metadata,
            source="entry_point",
        )

    def with_priority(self, priority: int) -> PluginSpec:
        """Create a copy with a new priority.

        Args:
            priority: New priority value.

        Returns:
            New PluginSpec with updated priority.
        """
        return PluginSpec(
            name=self.name,
            module_path=self.module_path,
            class_name=self.class_name,
            plugin_type=self.plugin_type,
            priority=priority,
            enabled=self.enabled,
            metadata=self.metadata,
            source=self.source,
            aliases=self.aliases,
            config=self.config,
        )

    def with_enabled(self, enabled: bool) -> PluginSpec:
        """Create a copy with enabled flag changed.

        Args:
            enabled: Whether the plugin should be enabled.

        Returns:
            New PluginSpec with updated enabled flag.
        """
        return PluginSpec(
            name=self.name,
            module_path=self.module_path,
            class_name=self.class_name,
            plugin_type=self.plugin_type,
            priority=self.priority,
            enabled=enabled,
            metadata=self.metadata,
            source=self.source,
            aliases=self.aliases,
            config=self.config,
        )

    def with_aliases(self, *aliases: str) -> PluginSpec:
        """Create a copy with additional aliases.

        Args:
            *aliases: Alias names for the plugin.

        Returns:
            New PluginSpec with updated aliases.
        """
        return PluginSpec(
            name=self.name,
            module_path=self.module_path,
            class_name=self.class_name,
            plugin_type=self.plugin_type,
            priority=self.priority,
            enabled=self.enabled,
            metadata=self.metadata,
            source=self.source,
            aliases=self.aliases + aliases,
            config=self.config,
        )

    def with_config(self, config: dict[str, Any]) -> PluginSpec:
        """Create a copy with configuration.

        Args:
            config: Default configuration for the plugin.

        Returns:
            New PluginSpec with updated config.
        """
        return PluginSpec(
            name=self.name,
            module_path=self.module_path,
            class_name=self.class_name,
            plugin_type=self.plugin_type,
            priority=self.priority,
            enabled=self.enabled,
            metadata=self.metadata,
            source=self.source,
            aliases=self.aliases,
            config={**self.config, **config},
        )


@dataclass
class PluginInstance:
    """Container for a loaded plugin instance.

    Attributes:
        spec: Plugin specification.
        instance: The loaded plugin instance.
        state: Current plugin state.
        load_time: When the plugin was loaded.
        error: Error if loading failed.
        engine_class: The engine class (before instantiation).
    """

    spec: PluginSpec
    instance: DataQualityEngine | None = None
    state: PluginState = PluginState.DISCOVERED
    load_time: datetime | None = None
    error: Exception | None = None
    engine_class: type | None = None

    @property
    def is_loaded(self) -> bool:
        """Check if the plugin is successfully loaded."""
        return self.state == PluginState.LOADED and self.instance is not None

    @property
    def is_failed(self) -> bool:
        """Check if the plugin failed to load."""
        return self.state == PluginState.FAILED


@dataclass(frozen=True)
class ValidationResult:
    """Result of plugin validation.

    Attributes:
        is_valid: Whether the plugin passed validation.
        errors: List of error messages.
        warnings: List of warning messages.
        checked_at: When the validation was performed.
    """

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checked_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def success(cls, warnings: Sequence[str] | None = None) -> ValidationResult:
        """Create a successful validation result.

        Args:
            warnings: Optional warning messages.

        Returns:
            Successful validation result.
        """
        return cls(
            is_valid=True,
            warnings=tuple(warnings) if warnings else (),
        )

    @classmethod
    def failure(
        cls,
        errors: Sequence[str],
        warnings: Sequence[str] | None = None,
    ) -> ValidationResult:
        """Create a failed validation result.

        Args:
            errors: Error messages.
            warnings: Optional warning messages.

        Returns:
            Failed validation result.
        """
        return cls(
            is_valid=False,
            errors=tuple(errors),
            warnings=tuple(warnings) if warnings else (),
        )


@dataclass(frozen=True)
class DiscoveryConfig:
    """Configuration for plugin discovery.

    Attributes:
        entry_point_group: Entry point group name.
        load_strategy: How to load discovered plugins.
        validate_plugins: Whether to validate plugins on discovery.
        ignore_errors: Whether to ignore discovery errors.
        disabled_plugins: List of plugin names to disable.
        priority_overrides: Plugin priority overrides (name -> priority).
        include_patterns: Glob patterns for plugin names to include.
        exclude_patterns: Glob patterns for plugin names to exclude.
    """

    entry_point_group: str = ENTRY_POINT_GROUP
    load_strategy: LoadStrategy = LoadStrategy.LAZY
    validate_plugins: bool = True
    ignore_errors: bool = True
    disabled_plugins: tuple[str, ...] = ()
    priority_overrides: dict[str, int] = field(default_factory=dict)
    include_patterns: tuple[str, ...] = ("*",)
    exclude_patterns: tuple[str, ...] = ()

    def with_entry_point_group(self, group: str) -> DiscoveryConfig:
        """Create a copy with a different entry point group.

        Args:
            group: Entry point group name.

        Returns:
            New config with updated group.
        """
        return DiscoveryConfig(
            entry_point_group=group,
            load_strategy=self.load_strategy,
            validate_plugins=self.validate_plugins,
            ignore_errors=self.ignore_errors,
            disabled_plugins=self.disabled_plugins,
            priority_overrides=self.priority_overrides,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
        )

    def with_load_strategy(self, strategy: LoadStrategy) -> DiscoveryConfig:
        """Create a copy with a different load strategy.

        Args:
            strategy: Load strategy.

        Returns:
            New config with updated strategy.
        """
        return DiscoveryConfig(
            entry_point_group=self.entry_point_group,
            load_strategy=strategy,
            validate_plugins=self.validate_plugins,
            ignore_errors=self.ignore_errors,
            disabled_plugins=self.disabled_plugins,
            priority_overrides=self.priority_overrides,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
        )

    def with_disabled_plugins(self, *plugins: str) -> DiscoveryConfig:
        """Create a copy with disabled plugins.

        Args:
            *plugins: Plugin names to disable.

        Returns:
            New config with updated disabled list.
        """
        return DiscoveryConfig(
            entry_point_group=self.entry_point_group,
            load_strategy=self.load_strategy,
            validate_plugins=self.validate_plugins,
            ignore_errors=self.ignore_errors,
            disabled_plugins=self.disabled_plugins + plugins,
            priority_overrides=self.priority_overrides,
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
        )

    def with_priority_override(self, name: str, priority: int) -> DiscoveryConfig:
        """Create a copy with a priority override.

        Args:
            name: Plugin name.
            priority: Priority value.

        Returns:
            New config with updated priorities.
        """
        return DiscoveryConfig(
            entry_point_group=self.entry_point_group,
            load_strategy=self.load_strategy,
            validate_plugins=self.validate_plugins,
            ignore_errors=self.ignore_errors,
            disabled_plugins=self.disabled_plugins,
            priority_overrides={**self.priority_overrides, name: priority},
            include_patterns=self.include_patterns,
            exclude_patterns=self.exclude_patterns,
        )


# Default configuration
DEFAULT_DISCOVERY_CONFIG = DiscoveryConfig()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class PluginSource(Protocol):
    """Protocol for plugin sources.

    Plugin sources provide plugin specifications from various locations
    such as entry points, configuration files, or directories.
    """

    @property
    def name(self) -> str:
        """Get the source name."""
        ...

    def discover(self) -> Iterator[PluginSpec]:
        """Discover plugins from this source.

        Yields:
            Plugin specifications.
        """
        ...


@runtime_checkable
class PluginHook(Protocol):
    """Protocol for plugin lifecycle hooks.

    Hooks are called at various points during plugin discovery and loading.
    """

    def on_discovery_start(self, config: DiscoveryConfig) -> None:
        """Called when discovery starts.

        Args:
            config: Discovery configuration.
        """
        ...

    def on_discovery_end(
        self,
        plugins: Sequence[PluginSpec],
        errors: Sequence[Exception],
    ) -> None:
        """Called when discovery ends.

        Args:
            plugins: Discovered plugins.
            errors: Errors during discovery.
        """
        ...

    def on_plugin_discovered(self, spec: PluginSpec) -> None:
        """Called when a plugin is discovered.

        Args:
            spec: Plugin specification.
        """
        ...

    def on_plugin_loading(self, spec: PluginSpec) -> None:
        """Called when a plugin starts loading.

        Args:
            spec: Plugin specification.
        """
        ...

    def on_plugin_loaded(
        self,
        spec: PluginSpec,
        instance: DataQualityEngine,
    ) -> None:
        """Called when a plugin is successfully loaded.

        Args:
            spec: Plugin specification.
            instance: Loaded plugin instance.
        """
        ...

    def on_plugin_error(
        self,
        spec: PluginSpec,
        error: Exception,
    ) -> None:
        """Called when a plugin fails to load.

        Args:
            spec: Plugin specification.
            error: The error that occurred.
        """
        ...


@runtime_checkable
class PluginValidator(Protocol):
    """Protocol for plugin validators.

    Validators check if a plugin meets certain requirements.
    """

    def validate(
        self,
        engine_class: type,
        spec: PluginSpec,
    ) -> ValidationResult:
        """Validate a plugin class.

        Args:
            engine_class: The engine class to validate.
            spec: Plugin specification.

        Returns:
            Validation result.
        """
        ...


@runtime_checkable
class PluginFactory(Protocol):
    """Protocol for plugin factories.

    Factories create plugin instances from specifications.
    """

    def create(
        self,
        spec: PluginSpec,
        engine_class: type,
        **kwargs: Any,
    ) -> DataQualityEngine:
        """Create a plugin instance.

        Args:
            spec: Plugin specification.
            engine_class: The engine class.
            **kwargs: Additional arguments for instantiation.

        Returns:
            Created plugin instance.
        """
        ...


# =============================================================================
# Hook Implementations
# =============================================================================


class BasePluginHook:
    """Base implementation of PluginHook with no-op methods.

    Subclass this to implement only the hooks you need.
    """

    def on_discovery_start(self, config: DiscoveryConfig) -> None:
        """Called when discovery starts."""
        pass

    def on_discovery_end(
        self,
        plugins: Sequence[PluginSpec],
        errors: Sequence[Exception],
    ) -> None:
        """Called when discovery ends."""
        pass

    def on_plugin_discovered(self, spec: PluginSpec) -> None:
        """Called when a plugin is discovered."""
        pass

    def on_plugin_loading(self, spec: PluginSpec) -> None:
        """Called when a plugin starts loading."""
        pass

    def on_plugin_loaded(
        self,
        spec: PluginSpec,
        instance: DataQualityEngine,
    ) -> None:
        """Called when a plugin is successfully loaded."""
        pass

    def on_plugin_error(
        self,
        spec: PluginSpec,
        error: Exception,
    ) -> None:
        """Called when a plugin fails to load."""
        pass


class LoggingPluginHook(BasePluginHook):
    """Plugin hook that logs all events.

    Useful for debugging plugin discovery and loading.
    """

    def __init__(
        self,
        logger_name: str = "common.engines.plugin",
        level: int = logging.DEBUG,
    ) -> None:
        """Initialize the logging hook.

        Args:
            logger_name: Name for the logger.
            level: Logging level.
        """
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def on_discovery_start(self, config: DiscoveryConfig) -> None:
        """Log discovery start."""
        self._logger.log(
            self._level,
            "Starting plugin discovery with group '%s'",
            config.entry_point_group,
        )

    def on_discovery_end(
        self,
        plugins: Sequence[PluginSpec],
        errors: Sequence[Exception],
    ) -> None:
        """Log discovery end."""
        self._logger.log(
            self._level,
            "Plugin discovery completed: %d plugins, %d errors",
            len(plugins),
            len(errors),
        )

    def on_plugin_discovered(self, spec: PluginSpec) -> None:
        """Log plugin discovery."""
        self._logger.log(
            self._level,
            "Discovered plugin '%s' from %s",
            spec.name,
            spec.full_path,
        )

    def on_plugin_loading(self, spec: PluginSpec) -> None:
        """Log plugin loading start."""
        self._logger.log(
            self._level,
            "Loading plugin '%s'",
            spec.name,
        )

    def on_plugin_loaded(
        self,
        spec: PluginSpec,
        instance: DataQualityEngine,
    ) -> None:
        """Log successful plugin load."""
        self._logger.log(
            self._level,
            "Successfully loaded plugin '%s'",
            spec.name,
        )

    def on_plugin_error(
        self,
        spec: PluginSpec,
        error: Exception,
    ) -> None:
        """Log plugin error."""
        self._logger.warning(
            "Failed to load plugin '%s': %s",
            spec.name,
            str(error),
        )


class MetricsPluginHook(BasePluginHook):
    """Plugin hook that collects metrics.

    Tracks discovery and loading statistics.
    """

    def __init__(self) -> None:
        """Initialize the metrics hook."""
        self._discovery_count = 0
        self._loaded_count = 0
        self._error_count = 0
        self._discovery_times: list[float] = []
        self._load_times: dict[str, float] = {}
        self._lock = threading.RLock()  # RLock for reentrant access in get_stats

    @property
    def discovery_count(self) -> int:
        """Get total discovered plugins count."""
        with self._lock:
            return self._discovery_count

    @property
    def loaded_count(self) -> int:
        """Get successfully loaded plugins count."""
        with self._lock:
            return self._loaded_count

    @property
    def error_count(self) -> int:
        """Get failed plugins count."""
        with self._lock:
            return self._error_count

    @property
    def success_rate(self) -> float:
        """Get loading success rate (0.0 to 1.0)."""
        with self._lock:
            total = self._loaded_count + self._error_count
            if total == 0:
                return 1.0
            return self._loaded_count / total

    def get_stats(self) -> dict[str, Any]:
        """Get all statistics.

        Returns:
            Dictionary of statistics.
        """
        with self._lock:
            return {
                "discovery_count": self._discovery_count,
                "loaded_count": self._loaded_count,
                "error_count": self._error_count,
                "success_rate": self.success_rate,
                "load_times": dict(self._load_times),
            }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._discovery_count = 0
            self._loaded_count = 0
            self._error_count = 0
            self._discovery_times.clear()
            self._load_times.clear()

    def on_plugin_discovered(self, spec: PluginSpec) -> None:
        """Track discovered plugin."""
        with self._lock:
            self._discovery_count += 1

    def on_plugin_loaded(
        self,
        spec: PluginSpec,
        instance: DataQualityEngine,
    ) -> None:
        """Track loaded plugin."""
        with self._lock:
            self._loaded_count += 1

    def on_plugin_error(
        self,
        spec: PluginSpec,
        error: Exception,
    ) -> None:
        """Track failed plugin."""
        with self._lock:
            self._error_count += 1


class CompositePluginHook(BasePluginHook):
    """Composite hook that delegates to multiple hooks.

    Useful for combining multiple hooks together.
    """

    def __init__(self, hooks: Sequence[PluginHook] | None = None) -> None:
        """Initialize composite hook.

        Args:
            hooks: List of hooks to delegate to.
        """
        self._hooks: list[PluginHook] = list(hooks) if hooks else []

    def add_hook(self, hook: PluginHook) -> None:
        """Add a hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.append(hook)

    def remove_hook(self, hook: PluginHook) -> None:
        """Remove a hook.

        Args:
            hook: Hook to remove.
        """
        if hook in self._hooks:
            self._hooks.remove(hook)

    def on_discovery_start(self, config: DiscoveryConfig) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_discovery_start(config)
            except Exception as e:
                logger.warning("Hook error in on_discovery_start: %s", e)

    def on_discovery_end(
        self,
        plugins: Sequence[PluginSpec],
        errors: Sequence[Exception],
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_discovery_end(plugins, errors)
            except Exception as e:
                logger.warning("Hook error in on_discovery_end: %s", e)

    def on_plugin_discovered(self, spec: PluginSpec) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_plugin_discovered(spec)
            except Exception as e:
                logger.warning("Hook error in on_plugin_discovered: %s", e)

    def on_plugin_loading(self, spec: PluginSpec) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_plugin_loading(spec)
            except Exception as e:
                logger.warning("Hook error in on_plugin_loading: %s", e)

    def on_plugin_loaded(
        self,
        spec: PluginSpec,
        instance: DataQualityEngine,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_plugin_loaded(spec, instance)
            except Exception as e:
                logger.warning("Hook error in on_plugin_loaded: %s", e)

    def on_plugin_error(
        self,
        spec: PluginSpec,
        error: Exception,
    ) -> None:
        """Delegate to all hooks."""
        for hook in self._hooks:
            try:
                hook.on_plugin_error(spec, error)
            except Exception as e:
                logger.warning("Hook error in on_plugin_error: %s", e)


# =============================================================================
# Plugin Sources
# =============================================================================


class EntryPointPluginSource:
    """Plugin source that discovers plugins from entry points.

    This is the primary source for plugin discovery, using Python's
    standard entry point mechanism.
    """

    def __init__(
        self,
        group: str = ENTRY_POINT_GROUP,
        plugin_type: PluginType = PluginType.ENGINE,
    ) -> None:
        """Initialize entry point source.

        Args:
            group: Entry point group name.
            plugin_type: Type of plugins to discover.
        """
        self._group = group
        self._plugin_type = plugin_type

    @property
    def name(self) -> str:
        """Get the source name."""
        return f"entry_point:{self._group}"

    def discover(self) -> Iterator[PluginSpec]:
        """Discover plugins from entry points.

        Yields:
            Plugin specifications.
        """
        try:
            eps = importlib.metadata.entry_points()

            # Handle different Python versions
            if hasattr(eps, "select"):
                # Python 3.10+
                group_eps = eps.select(group=self._group)
            elif hasattr(eps, "get"):
                # Python 3.9
                group_eps = eps.get(self._group, [])
            else:
                # Fallback
                group_eps = getattr(eps, self._group, [])

            for ep in group_eps:
                try:
                    yield PluginSpec.from_entry_point(ep, plugin_type=self._plugin_type)
                except Exception as e:
                    logger.warning(
                        "Failed to parse entry point '%s': %s",
                        ep.name,
                        e,
                    )
        except Exception as e:
            logger.warning("Failed to discover entry points: %s", e)


class BuiltinPluginSource:
    """Plugin source for built-in engines.

    Provides the default Truthound engines as plugins.
    """

    def __init__(self) -> None:
        """Initialize builtin source."""
        self._builtin_specs: list[PluginSpec] = [
            PluginSpec(
                name="truthound",
                module_path="common.engines.truthound",
                class_name="TruthoundEngine",
                plugin_type=PluginType.ENGINE,
                priority=0,  # Highest priority for builtins
                source="builtin",
                aliases=("th", "default"),
            ),
            PluginSpec(
                name="great_expectations",
                module_path="common.engines.great_expectations",
                class_name="GreatExpectationsAdapter",
                plugin_type=PluginType.ADAPTER,
                priority=10,
                source="builtin",
                aliases=("ge", "gx"),
            ),
            PluginSpec(
                name="pandera",
                module_path="common.engines.pandera",
                class_name="PanderaAdapter",
                plugin_type=PluginType.ADAPTER,
                priority=10,
                source="builtin",
                aliases=("pa",),
            ),
        ]

    @property
    def name(self) -> str:
        """Get the source name."""
        return "builtin"

    def discover(self) -> Iterator[PluginSpec]:
        """Discover built-in plugins.

        Yields:
            Plugin specifications for built-in engines.
        """
        yield from self._builtin_specs


class DictPluginSource:
    """Plugin source from a dictionary.

    Useful for programmatic plugin registration.
    """

    def __init__(
        self,
        plugins: dict[str, str | type],
        source_name: str = "dict",
    ) -> None:
        """Initialize dict source.

        Args:
            plugins: Dictionary mapping name to module:class or class.
            source_name: Name for this source.
        """
        self._plugins = plugins
        self._source_name = source_name

    @property
    def name(self) -> str:
        """Get the source name."""
        return self._source_name

    def discover(self) -> Iterator[PluginSpec]:
        """Discover plugins from dictionary.

        Yields:
            Plugin specifications.
        """
        for name, value in self._plugins.items():
            if isinstance(value, str):
                # Parse module:class string
                if ":" in value:
                    module_path, class_name = value.rsplit(":", 1)
                else:
                    module_path, class_name = value.rsplit(".", 1)
            else:
                # It's a class
                module_path = value.__module__
                class_name = value.__name__

            yield PluginSpec(
                name=name,
                module_path=module_path,
                class_name=class_name,
                source=self._source_name,
            )


# =============================================================================
# Validators
# =============================================================================


class DataQualityEngineValidator:
    """Validator for DataQualityEngine implementations.

    Checks that a class properly implements the DataQualityEngine protocol.
    """

    REQUIRED_METHODS = ("check", "profile", "learn")
    REQUIRED_PROPERTIES = ("engine_name", "engine_version")

    def validate(
        self,
        engine_class: type,
        spec: PluginSpec,
    ) -> ValidationResult:
        """Validate an engine class.

        Args:
            engine_class: The engine class to validate.
            spec: Plugin specification.

        Returns:
            Validation result.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check required methods
        for method in self.REQUIRED_METHODS:
            if not hasattr(engine_class, method):
                errors.append(f"Missing required method: {method}")
            elif not callable(getattr(engine_class, method)):
                errors.append(f"'{method}' is not callable")

        # Check required properties
        for prop in self.REQUIRED_PROPERTIES:
            if not hasattr(engine_class, prop):
                errors.append(f"Missing required property: {prop}")

        # Check for optional but recommended methods
        if not hasattr(engine_class, "get_capabilities"):
            warnings.append("Missing recommended method: get_capabilities")

        if not hasattr(engine_class, "get_info"):
            warnings.append("Missing recommended method: get_info")

        if errors:
            return ValidationResult.failure(errors, warnings)
        return ValidationResult.success(warnings)


class CompositeValidator:
    """Composite validator that runs multiple validators.

    Combines results from all validators.
    """

    def __init__(self, validators: Sequence[PluginValidator] | None = None) -> None:
        """Initialize composite validator.

        Args:
            validators: List of validators to run.
        """
        self._validators: list[PluginValidator] = list(validators) if validators else []

    def add_validator(self, validator: PluginValidator) -> None:
        """Add a validator.

        Args:
            validator: Validator to add.
        """
        self._validators.append(validator)

    def validate(
        self,
        engine_class: type,
        spec: PluginSpec,
    ) -> ValidationResult:
        """Run all validators.

        Args:
            engine_class: The engine class to validate.
            spec: Plugin specification.

        Returns:
            Combined validation result.
        """
        all_errors: list[str] = []
        all_warnings: list[str] = []

        for validator in self._validators:
            try:
                result = validator.validate(engine_class, spec)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
            except Exception as e:
                all_errors.append(f"Validator error: {e}")

        if all_errors:
            return ValidationResult.failure(all_errors, all_warnings)
        return ValidationResult.success(all_warnings)


# Default validator
DEFAULT_VALIDATOR = DataQualityEngineValidator()


# =============================================================================
# Plugin Factory
# =============================================================================


class DefaultPluginFactory:
    """Default factory for creating plugin instances.

    Handles standard instantiation with optional config injection.
    """

    def create(
        self,
        spec: PluginSpec,
        engine_class: type,
        **kwargs: Any,
    ) -> DataQualityEngine:
        """Create a plugin instance.

        Args:
            spec: Plugin specification.
            engine_class: The engine class.
            **kwargs: Additional arguments for instantiation.

        Returns:
            Created plugin instance.
        """
        # Merge spec config with kwargs
        config = {**spec.config, **kwargs}

        # Try to instantiate with config if the class accepts it
        try:
            return engine_class(**config)
        except TypeError:
            # Fall back to no-arg constructor
            try:
                return engine_class()
            except TypeError as e:
                raise PluginLoadError(
                    f"Failed to instantiate {spec.name}: {e}",
                    plugin_name=spec.name,
                    cause=e,
                ) from e


# Default factory instance
DEFAULT_FACTORY = DefaultPluginFactory()


# =============================================================================
# Plugin Loader
# =============================================================================


class PluginLoader:
    """Loader for plugins.

    Handles importing modules and instantiating plugin classes.
    """

    def __init__(
        self,
        validator: PluginValidator | None = None,
        factory: PluginFactory | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            validator: Validator for loaded plugins.
            factory: Factory for creating instances.
        """
        self._validator = validator or DEFAULT_VALIDATOR
        self._factory = factory or DEFAULT_FACTORY

    def load_class(self, spec: PluginSpec) -> type:
        """Load a plugin class without instantiating.

        Args:
            spec: Plugin specification.

        Returns:
            The loaded class.

        Raises:
            PluginLoadError: If the class cannot be loaded.
        """
        try:
            module = importlib.import_module(spec.module_path)
            engine_class = getattr(module, spec.class_name)
            return engine_class
        except ImportError as e:
            raise PluginLoadError(
                f"Failed to import module '{spec.module_path}': {e}",
                plugin_name=spec.name,
                cause=e,
            ) from e
        except AttributeError as e:
            raise PluginLoadError(
                f"Class '{spec.class_name}' not found in module '{spec.module_path}': {e}",
                plugin_name=spec.name,
                cause=e,
            ) from e

    def validate_class(
        self,
        engine_class: type,
        spec: PluginSpec,
    ) -> ValidationResult:
        """Validate a plugin class.

        Args:
            engine_class: The engine class.
            spec: Plugin specification.

        Returns:
            Validation result.
        """
        return self._validator.validate(engine_class, spec)

    def create_instance(
        self,
        spec: PluginSpec,
        engine_class: type,
        **kwargs: Any,
    ) -> DataQualityEngine:
        """Create a plugin instance.

        Args:
            spec: Plugin specification.
            engine_class: The engine class.
            **kwargs: Additional arguments.

        Returns:
            Created instance.
        """
        return self._factory.create(spec, engine_class, **kwargs)

    def load(
        self,
        spec: PluginSpec,
        validate: bool = True,
        **kwargs: Any,
    ) -> PluginInstance:
        """Load a plugin completely.

        Args:
            spec: Plugin specification.
            validate: Whether to validate the class.
            **kwargs: Additional arguments for instantiation.

        Returns:
            Plugin instance container.
        """
        result = PluginInstance(spec=spec, state=PluginState.LOADING)

        try:
            # Load the class
            engine_class = self.load_class(spec)
            result.engine_class = engine_class

            # Validate if requested
            if validate:
                validation = self.validate_class(engine_class, spec)
                if not validation.is_valid:
                    raise PluginValidationError(
                        f"Plugin '{spec.name}' failed validation",
                        plugin_name=spec.name,
                        validation_errors=list(validation.errors),
                    )

            # Create instance
            instance = self.create_instance(spec, engine_class, **kwargs)
            result.instance = instance
            result.state = PluginState.LOADED
            result.load_time = datetime.now()

        except Exception as e:
            result.state = PluginState.FAILED
            result.error = e
            raise

        return result


# =============================================================================
# Plugin Registry
# =============================================================================


class PluginRegistry:
    """Registry for discovered and loaded plugins.

    Thread-safe registry that manages plugin discovery, loading,
    and access throughout the application lifecycle.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.discover()
        >>> engine = registry.get_engine("my_plugin")
        >>> # Or use lazy loading
        >>> engine = registry.get_engine("my_plugin", lazy=True)
    """

    def __init__(
        self,
        config: DiscoveryConfig | None = None,
        loader: PluginLoader | None = None,
        sources: Sequence[PluginSource] | None = None,
        hooks: Sequence[PluginHook] | None = None,
    ) -> None:
        """Initialize the registry.

        Args:
            config: Discovery configuration.
            loader: Plugin loader.
            sources: Plugin sources.
            hooks: Plugin lifecycle hooks.
        """
        self._config = config or DEFAULT_DISCOVERY_CONFIG
        self._loader = loader or PluginLoader()
        self._sources: list[PluginSource] = list(sources) if sources else []
        self._hooks = CompositePluginHook(hooks)

        # Registry state
        self._specs: dict[str, PluginSpec] = {}
        self._instances: dict[str, PluginInstance] = {}
        self._aliases: dict[str, str] = {}  # alias -> name
        self._lock = threading.RLock()
        self._discovered = False

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def add_source(self, source: PluginSource) -> None:
        """Add a plugin source.

        Args:
            source: Plugin source to add.
        """
        with self._lock:
            self._sources.append(source)

    def add_hook(self, hook: PluginHook) -> None:
        """Add a plugin hook.

        Args:
            hook: Hook to add.
        """
        self._hooks.add_hook(hook)

    def remove_hook(self, hook: PluginHook) -> None:
        """Remove a plugin hook.

        Args:
            hook: Hook to remove.
        """
        self._hooks.remove_hook(hook)

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def discover(
        self,
        include_builtins: bool = True,
        include_entry_points: bool = True,
    ) -> list[PluginSpec]:
        """Discover all available plugins.

        Args:
            include_builtins: Whether to include built-in engines.
            include_entry_points: Whether to include entry point plugins.

        Returns:
            List of discovered plugin specifications.

        Raises:
            PluginDiscoveryError: If discovery fails and ignore_errors is False.
        """
        with self._lock:
            self._hooks.on_discovery_start(self._config)

            discovered: list[PluginSpec] = []
            errors: list[Exception] = []

            # Build source list
            sources: list[PluginSource] = []
            if include_builtins:
                sources.append(BuiltinPluginSource())
            if include_entry_points:
                sources.append(EntryPointPluginSource(self._config.entry_point_group))
            sources.extend(self._sources)

            # Discover from all sources
            for source in sources:
                try:
                    for spec in source.discover():
                        try:
                            # Apply config overrides
                            spec = self._apply_config(spec)

                            # Check if plugin should be included
                            if not self._should_include(spec):
                                continue

                            # Register the spec
                            self._register_spec(spec)
                            discovered.append(spec)
                            self._hooks.on_plugin_discovered(spec)

                        except Exception as e:
                            logger.warning(
                                "Error processing plugin '%s' from source '%s': %s",
                                getattr(spec, "name", "unknown"),
                                source.name,
                                e,
                            )
                            errors.append(e)
                            if not self._config.ignore_errors:
                                raise PluginDiscoveryError(
                                    f"Failed to process plugin: {e}",
                                    cause=e,
                                ) from e

                except Exception as e:
                    logger.warning("Error discovering from source '%s': %s", source.name, e)
                    errors.append(e)
                    if not self._config.ignore_errors:
                        raise PluginDiscoveryError(
                            f"Failed to discover from {source.name}: {e}",
                            cause=e,
                        ) from e

            self._discovered = True
            self._hooks.on_discovery_end(discovered, errors)

            # Load eagerly if configured
            if self._config.load_strategy == LoadStrategy.EAGER:
                self._load_all()

            return discovered

    def _apply_config(self, spec: PluginSpec) -> PluginSpec:
        """Apply configuration overrides to a spec.

        Args:
            spec: Plugin specification.

        Returns:
            Modified specification.
        """
        # Apply priority override
        if spec.name in self._config.priority_overrides:
            spec = spec.with_priority(self._config.priority_overrides[spec.name])

        # Apply disabled status
        if spec.name in self._config.disabled_plugins:
            spec = spec.with_enabled(False)

        return spec

    def _should_include(self, spec: PluginSpec) -> bool:
        """Check if a plugin should be included.

        Args:
            spec: Plugin specification.

        Returns:
            True if plugin should be included.
        """
        import fnmatch

        name = spec.name

        # Check exclude patterns
        for pattern in self._config.exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return False

        # Check include patterns
        for pattern in self._config.include_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True

        return False

    def _register_spec(self, spec: PluginSpec) -> None:
        """Register a plugin spec.

        Args:
            spec: Plugin specification.
        """
        # Handle conflicts
        if spec.name in self._specs:
            existing = self._specs[spec.name]
            if spec.priority < existing.priority:
                # New plugin has higher priority, replace
                logger.debug(
                    "Replacing plugin '%s' (priority %d) with higher priority plugin (priority %d)",
                    spec.name,
                    existing.priority,
                    spec.priority,
                )
            else:
                # Keep existing
                logger.debug(
                    "Keeping existing plugin '%s' (priority %d) over new plugin (priority %d)",
                    spec.name,
                    existing.priority,
                    spec.priority,
                )
                return

        self._specs[spec.name] = spec

        # Register aliases
        for alias in spec.aliases:
            if alias not in self._aliases:
                self._aliases[alias] = spec.name

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load_plugin(
        self,
        name: str,
        **kwargs: Any,
    ) -> PluginInstance:
        """Load a specific plugin by name.

        Args:
            name: Plugin name or alias.
            **kwargs: Additional arguments for instantiation.

        Returns:
            Loaded plugin instance.

        Raises:
            PluginNotFoundError: If the plugin is not found.
            PluginLoadError: If loading fails.
        """
        with self._lock:
            # Ensure discovery has run
            if not self._discovered:
                self.discover()

            # Resolve alias
            resolved_name = self._aliases.get(name, name)

            # Check if already loaded
            if resolved_name in self._instances:
                instance = self._instances[resolved_name]
                if instance.is_loaded:
                    return instance

            # Get spec
            if resolved_name not in self._specs:
                raise PluginNotFoundError(
                    resolved_name,
                    available_plugins=list(self._specs.keys()),
                )

            spec = self._specs[resolved_name]

            # Check if disabled
            if not spec.enabled:
                raise PluginLoadError(
                    f"Plugin '{resolved_name}' is disabled",
                    plugin_name=resolved_name,
                )

            # Load the plugin
            self._hooks.on_plugin_loading(spec)
            try:
                instance = self._loader.load(
                    spec,
                    validate=self._config.validate_plugins,
                    **kwargs,
                )
                self._instances[resolved_name] = instance
                self._hooks.on_plugin_loaded(spec, instance.instance)
                return instance

            except Exception as e:
                self._hooks.on_plugin_error(spec, e)
                raise

    def _load_all(self) -> None:
        """Load all discovered plugins."""
        for name, spec in self._specs.items():
            if spec.enabled and name not in self._instances:
                try:
                    self.load_plugin(name)
                except Exception as e:
                    logger.warning("Failed to load plugin '%s': %s", name, e)

    # -------------------------------------------------------------------------
    # Access
    # -------------------------------------------------------------------------

    def get_engine(
        self,
        name: str,
        lazy: bool = True,
        **kwargs: Any,
    ) -> DataQualityEngine:
        """Get an engine by name.

        Args:
            name: Plugin name or alias.
            lazy: Whether to load lazily (on first access).
            **kwargs: Additional arguments for instantiation.

        Returns:
            The engine instance.

        Raises:
            PluginNotFoundError: If the plugin is not found.
            PluginLoadError: If loading fails.
        """
        with self._lock:
            # Resolve alias
            resolved_name = self._aliases.get(name, name)

            # Check if already loaded
            if resolved_name in self._instances:
                instance = self._instances[resolved_name]
                if instance.is_loaded:
                    return instance.instance  # type: ignore

            # Load the plugin
            instance = self.load_plugin(name, **kwargs)
            return instance.instance  # type: ignore

    def get_engine_class(self, name: str) -> type:
        """Get an engine class without instantiating.

        Args:
            name: Plugin name or alias.

        Returns:
            The engine class.

        Raises:
            PluginNotFoundError: If the plugin is not found.
            PluginLoadError: If loading fails.
        """
        with self._lock:
            # Resolve alias
            resolved_name = self._aliases.get(name, name)

            # Check if already loaded
            if resolved_name in self._instances:
                instance = self._instances[resolved_name]
                if instance.engine_class is not None:
                    return instance.engine_class

            # Get spec
            if resolved_name not in self._specs:
                # Ensure discovery has run
                if not self._discovered:
                    self.discover()

                if resolved_name not in self._specs:
                    raise PluginNotFoundError(
                        resolved_name,
                        available_plugins=list(self._specs.keys()),
                    )

            spec = self._specs[resolved_name]
            return self._loader.load_class(spec)

    def get_spec(self, name: str) -> PluginSpec:
        """Get a plugin specification.

        Args:
            name: Plugin name or alias.

        Returns:
            The plugin specification.

        Raises:
            PluginNotFoundError: If the plugin is not found.
        """
        with self._lock:
            # Ensure discovery has run
            if not self._discovered:
                self.discover()

            # Resolve alias
            resolved_name = self._aliases.get(name, name)

            if resolved_name not in self._specs:
                raise PluginNotFoundError(
                    resolved_name,
                    available_plugins=list(self._specs.keys()),
                )

            return self._specs[resolved_name]

    def list_plugins(self) -> list[str]:
        """List all discovered plugin names.

        Returns:
            List of plugin names.
        """
        with self._lock:
            if not self._discovered:
                self.discover()
            return list(self._specs.keys())

    def list_aliases(self) -> dict[str, str]:
        """List all plugin aliases.

        Returns:
            Dictionary mapping alias to plugin name.
        """
        with self._lock:
            if not self._discovered:
                self.discover()
            return dict(self._aliases)

    def has_plugin(self, name: str) -> bool:
        """Check if a plugin is available.

        Args:
            name: Plugin name or alias.

        Returns:
            True if plugin is available.
        """
        with self._lock:
            if not self._discovered:
                self.discover()

            resolved_name = self._aliases.get(name, name)
            return resolved_name in self._specs

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded.

        Args:
            name: Plugin name or alias.

        Returns:
            True if plugin is loaded.
        """
        with self._lock:
            resolved_name = self._aliases.get(name, name)
            if resolved_name in self._instances:
                return self._instances[resolved_name].is_loaded
            return False

    # -------------------------------------------------------------------------
    # Management
    # -------------------------------------------------------------------------

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin.

        Args:
            name: Plugin name or alias.

        Returns:
            True if plugin was unloaded.
        """
        with self._lock:
            resolved_name = self._aliases.get(name, name)

            if resolved_name in self._instances:
                instance = self._instances.pop(resolved_name)

                # Call stop if the engine supports lifecycle
                if instance.instance is not None:
                    if hasattr(instance.instance, "stop"):
                        try:
                            instance.instance.stop()  # type: ignore
                        except Exception as e:
                            logger.warning("Error stopping plugin '%s': %s", name, e)

                return True
            return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin.

        Args:
            name: Plugin name or alias.

        Returns:
            True if plugin was disabled.
        """
        with self._lock:
            resolved_name = self._aliases.get(name, name)

            if resolved_name in self._specs:
                spec = self._specs[resolved_name]
                self._specs[resolved_name] = spec.with_enabled(False)

                # Unload if loaded
                self.unload_plugin(name)
                return True
            return False

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin.

        Args:
            name: Plugin name or alias.

        Returns:
            True if plugin was enabled.
        """
        with self._lock:
            resolved_name = self._aliases.get(name, name)

            if resolved_name in self._specs:
                spec = self._specs[resolved_name]
                self._specs[resolved_name] = spec.with_enabled(True)
                return True
            return False

    def clear(self) -> None:
        """Clear all plugins and reset state."""
        with self._lock:
            # Unload all
            for name in list(self._instances.keys()):
                self.unload_plugin(name)

            self._specs.clear()
            self._instances.clear()
            self._aliases.clear()
            self._discovered = False

    def rediscover(
        self,
        include_builtins: bool = True,
        include_entry_points: bool = True,
    ) -> list[PluginSpec]:
        """Clear and rediscover all plugins.

        Args:
            include_builtins: Whether to include built-in engines.
            include_entry_points: Whether to include entry point plugins.

        Returns:
            List of discovered plugin specifications.
        """
        self.clear()
        return self.discover(
            include_builtins=include_builtins,
            include_entry_points=include_entry_points,
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary of statistics.
        """
        with self._lock:
            loaded = sum(1 for i in self._instances.values() if i.is_loaded)
            failed = sum(1 for i in self._instances.values() if i.is_failed)

            return {
                "discovered_count": len(self._specs),
                "loaded_count": loaded,
                "failed_count": failed,
                "alias_count": len(self._aliases),
                "source_count": len(self._sources),
                "is_discovered": self._discovered,
            }


# =============================================================================
# Global Registry
# =============================================================================

_global_plugin_registry: PluginRegistry | None = None
_plugin_registry_lock = threading.Lock()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry.

    Returns:
        The global PluginRegistry instance.
    """
    global _global_plugin_registry

    if _global_plugin_registry is None:
        with _plugin_registry_lock:
            if _global_plugin_registry is None:
                _global_plugin_registry = PluginRegistry()

    return _global_plugin_registry


def reset_plugin_registry() -> None:
    """Reset the global plugin registry."""
    global _global_plugin_registry

    with _plugin_registry_lock:
        if _global_plugin_registry is not None:
            _global_plugin_registry.clear()
            _global_plugin_registry = None


# =============================================================================
# Convenience Functions
# =============================================================================


def discover_plugins(
    config: DiscoveryConfig | None = None,
    sources: Sequence[PluginSource] | None = None,
    hooks: Sequence[PluginHook] | None = None,
    include_builtins: bool = True,
    include_entry_points: bool = True,
) -> list[PluginSpec]:
    """Discover available plugins.

    Args:
        config: Discovery configuration.
        sources: Additional plugin sources.
        hooks: Plugin hooks.
        include_builtins: Whether to include built-in engines.
        include_entry_points: Whether to include entry point plugins.

    Returns:
        List of discovered plugin specifications.

    Example:
        >>> plugins = discover_plugins()
        >>> for plugin in plugins:
        ...     print(f"{plugin.name}: {plugin.full_path}")
    """
    registry = get_plugin_registry()

    if config is not None:
        registry._config = config

    if sources:
        for source in sources:
            registry.add_source(source)

    if hooks:
        for hook in hooks:
            registry.add_hook(hook)

    return registry.discover(
        include_builtins=include_builtins,
        include_entry_points=include_entry_points,
    )


def load_plugins(
    names: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str, DataQualityEngine]:
    """Load plugins by name.

    Args:
        names: Plugin names to load (all if None).
        **kwargs: Additional arguments for instantiation.

    Returns:
        Dictionary mapping name to engine instance.

    Example:
        >>> engines = load_plugins(["truthound", "pandera"])
        >>> result = engines["truthound"].check(data, rules)
    """
    registry = get_plugin_registry()

    if not registry._discovered:
        registry.discover()

    if names is None:
        names = registry.list_plugins()

    result: dict[str, DataQualityEngine] = {}
    for name in names:
        try:
            result[name] = registry.get_engine(name, **kwargs)
        except Exception as e:
            logger.warning("Failed to load plugin '%s': %s", name, e)

    return result


def get_plugin_engine(
    name: str,
    **kwargs: Any,
) -> DataQualityEngine:
    """Get a plugin engine by name.

    Args:
        name: Plugin name or alias.
        **kwargs: Additional arguments for instantiation.

    Returns:
        The engine instance.

    Example:
        >>> engine = get_plugin_engine("truthound")
        >>> result = engine.check(data, rules)
    """
    return get_plugin_registry().get_engine(name, **kwargs)


def validate_plugin(
    engine_class: type,
    spec: PluginSpec | None = None,
) -> ValidationResult:
    """Validate a plugin class.

    Args:
        engine_class: The engine class to validate.
        spec: Optional plugin specification.

    Returns:
        Validation result.

    Example:
        >>> result = validate_plugin(MyEngine)
        >>> if not result.is_valid:
        ...     print(f"Errors: {result.errors}")
    """
    if spec is None:
        spec = PluginSpec(
            name=engine_class.__name__,
            module_path=engine_class.__module__,
            class_name=engine_class.__name__,
        )

    return DEFAULT_VALIDATOR.validate(engine_class, spec)


def auto_discover_engines(
    register_with_engine_registry: bool = True,
) -> list[PluginSpec]:
    """Automatically discover and optionally register engines.

    This is the main integration point with the existing EngineRegistry.

    Args:
        register_with_engine_registry: Whether to register with EngineRegistry.

    Returns:
        List of discovered plugin specifications.

    Example:
        >>> from common.engines.plugin import auto_discover_engines
        >>> auto_discover_engines()  # Discovers and registers all plugins
        >>> from common.engines import get_engine
        >>> engine = get_engine("my_third_party_engine")
    """
    plugins = discover_plugins()

    if register_with_engine_registry:
        from common.engines.registry import get_engine_registry

        engine_registry = get_engine_registry()
        plugin_registry = get_plugin_registry()

        for spec in plugins:
            if spec.enabled:
                try:
                    # Get the engine instance
                    engine = plugin_registry.get_engine(spec.name)

                    # Register with engine registry
                    engine_registry.register(
                        spec.name,
                        engine,
                        allow_override=True,
                    )

                    # Register aliases
                    for alias in spec.aliases:
                        engine_registry.register(
                            alias,
                            engine,
                            allow_override=True,
                        )

                except Exception as e:
                    logger.warning(
                        "Failed to register plugin '%s' with engine registry: %s",
                        spec.name,
                        e,
                    )

    return plugins


def register_plugin(
    name: str,
    engine_class: type | str,
    *,
    aliases: Sequence[str] | None = None,
    priority: int = DEFAULT_PLUGIN_PRIORITY,
    config: dict[str, Any] | None = None,
) -> PluginSpec:
    """Programmatically register a plugin.

    Args:
        name: Plugin name.
        engine_class: Engine class or import path.
        aliases: Alternative names.
        priority: Loading priority.
        config: Default configuration.

    Returns:
        Plugin specification.

    Example:
        >>> register_plugin("my_engine", MyEngine, aliases=["me"])
        >>> engine = get_plugin_engine("me")
    """
    if isinstance(engine_class, str):
        if ":" in engine_class:
            module_path, class_name = engine_class.rsplit(":", 1)
        else:
            module_path, class_name = engine_class.rsplit(".", 1)
    else:
        module_path = engine_class.__module__
        class_name = engine_class.__name__

    spec = PluginSpec(
        name=name,
        module_path=module_path,
        class_name=class_name,
        priority=priority,
        aliases=tuple(aliases) if aliases else (),
        config=config or {},
        source="programmatic",
    )

    registry = get_plugin_registry()
    registry._register_spec(spec)

    return spec
