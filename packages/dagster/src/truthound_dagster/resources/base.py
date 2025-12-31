"""Base Resource Classes for Dagster Integration.

This module provides base classes and configuration types for
Dagster resources in the data quality integration.

Example:
    >>> from truthound_dagster.resources import BaseResource, ResourceConfig
    >>>
    >>> config = ResourceConfig(
    ...     enabled=True,
    ...     timeout_seconds=300.0,
    ... )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    """Base configuration for Dagster resources.

    Attributes:
        enabled: Whether the resource is enabled.
        timeout_seconds: Default timeout for operations.
        tags: Metadata tags for the resource.

    Example:
        >>> config = ResourceConfig(
        ...     enabled=True,
        ...     timeout_seconds=300.0,
        ...     tags=frozenset({"production"}),
        ... )
    """

    enabled: bool = True
    timeout_seconds: float = 300.0
    tags: "frozenset[str]" = field(default_factory=frozenset)

    def with_enabled(self, enabled: bool) -> "ResourceConfig":
        """Return new config with updated enabled status."""
        return ResourceConfig(
            enabled=enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
        )

    def with_timeout(self, timeout_seconds: float) -> "ResourceConfig":
        """Return new config with updated timeout."""
        return ResourceConfig(
            enabled=self.enabled,
            timeout_seconds=timeout_seconds,
            tags=self.tags,
        )

    def with_tags(self, *tags: str) -> "ResourceConfig":
        """Return new config with additional tags."""
        return ResourceConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags | frozenset(tags),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
        }


ConfigT = TypeVar("ConfigT", bound=ResourceConfig)


class BaseResource(ABC, Generic[ConfigT]):
    """Abstract base class for Dagster resources.

    This class provides a common interface for all data quality
    resources in Dagster integration.

    Parameters
    ----------
    config : ConfigT
        Resource configuration.

    Attributes
    ----------
    config : ConfigT
        The resource configuration.

    Example:
        >>> class MyResource(BaseResource[ResourceConfig]):
        ...     def setup(self, context):
        ...         pass
        ...
        ...     def teardown(self, context):
        ...         pass
    """

    def __init__(self, config: Optional[ConfigT] = None) -> None:
        """Initialize base resource.

        Args:
            config: Resource configuration. Uses default if None.
        """
        self._config = config or self._default_config()
        self._initialized = False

    @property
    def config(self) -> ConfigT:
        """Get resource configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if resource is initialized."""
        return self._initialized

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Return default configuration.

        Subclasses must implement this to provide their default config.

        Returns:
            ConfigT: Default configuration instance.
        """
        ...

    def setup(self, context: Any) -> None:
        """Set up the resource.

        Called by Dagster when the resource is initialized.
        Override in subclasses to perform initialization.

        Args:
            context: Dagster initialization context.
        """
        self._initialized = True

    def teardown(self, context: Any) -> None:
        """Tear down the resource.

        Called by Dagster when the resource is being disposed.
        Override in subclasses to perform cleanup.

        Args:
            context: Dagster initialization context.
        """
        self._initialized = False

    def __enter__(self) -> "BaseResource[ConfigT]":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass
