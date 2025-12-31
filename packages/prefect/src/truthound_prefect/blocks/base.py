"""Base block classes and configurations for truthound-prefect.

This module provides the base abstractions for Prefect Blocks,
following the immutable frozen dataclass pattern with builder methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from prefect.blocks.core import Block

ConfigT = TypeVar("ConfigT", bound="BlockConfig")


@dataclass(frozen=True, slots=True)
class BlockConfig:
    """Base configuration for all blocks.

    Immutable configuration with builder methods for customization.

    Attributes:
        enabled: Whether the block is enabled.
        timeout_seconds: Default timeout for operations.
        tags: Tags for categorization and filtering.
        description: Human-readable description.
    """

    enabled: bool = True
    timeout_seconds: float = 300.0
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    def with_enabled(self, enabled: bool) -> BlockConfig:
        """Return a new config with enabled flag changed."""
        return BlockConfig(
            enabled=enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
        )

    def with_timeout(self, timeout_seconds: float) -> BlockConfig:
        """Return a new config with timeout changed."""
        return BlockConfig(
            enabled=self.enabled,
            timeout_seconds=timeout_seconds,
            tags=self.tags,
            description=self.description,
        )

    def with_tags(self, *tags: str) -> BlockConfig:
        """Return a new config with additional tags."""
        new_tags = self.tags | frozenset(tags)
        return BlockConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=new_tags,
            description=self.description,
        )

    def with_description(self, description: str) -> BlockConfig:
        """Return a new config with description changed."""
        return BlockConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=description,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            tags=frozenset(data.get("tags", [])),
            description=data.get("description", ""),
        )


class BaseBlock(ABC, Generic[ConfigT]):
    """Abstract base class for all truthound-prefect blocks.

    Provides common lifecycle management and configuration handling.

    Type Parameters:
        ConfigT: The configuration type for this block.
    """

    def __init__(self, config: ConfigT | None = None) -> None:
        """Initialize the block.

        Args:
            config: Optional configuration. Uses default if not provided.
        """
        self._config = config or self._default_config()
        self._initialized = False

    @property
    def config(self) -> ConfigT:
        """Get the current configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the block has been initialized."""
        return self._initialized

    @classmethod
    @abstractmethod
    def _default_config(cls) -> ConfigT:
        """Return the default configuration for this block type."""
        ...

    def setup(self) -> None:
        """Initialize the block.

        Called before the block is used. Override in subclasses for
        custom initialization logic.
        """
        self._initialized = True

    def teardown(self) -> None:
        """Clean up the block.

        Called when the block is no longer needed. Override in subclasses
        for custom cleanup logic.
        """
        self._initialized = False

    def __enter__(self) -> BaseBlock[ConfigT]:
        """Enter context manager."""
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.teardown()

    async def __aenter__(self) -> BaseBlock[ConfigT]:
        """Enter async context manager."""
        self.setup()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        self.teardown()


# Preset configurations
DEFAULT_BLOCK_CONFIG = BlockConfig()

PRODUCTION_BLOCK_CONFIG = BlockConfig(
    enabled=True,
    timeout_seconds=600.0,
    tags=frozenset({"production"}),
    description="Production block configuration",
)

DEVELOPMENT_BLOCK_CONFIG = BlockConfig(
    enabled=True,
    timeout_seconds=60.0,
    tags=frozenset({"development"}),
    description="Development block configuration",
)


__all__ = [
    "BlockConfig",
    "BaseBlock",
    "DEFAULT_BLOCK_CONFIG",
    "PRODUCTION_BLOCK_CONFIG",
    "DEVELOPMENT_BLOCK_CONFIG",
]
