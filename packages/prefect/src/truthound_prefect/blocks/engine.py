"""Engine block for data quality operations.

This module provides Prefect Block implementations that wrap data quality
engines from the common/ module, following the Protocol-based design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from prefect.blocks.core import Block
from pydantic import Field

from truthound_prefect.blocks.base import BaseBlock, BlockConfig
from truthound_prefect.utils.exceptions import EngineError
from truthound_prefect.utils.serialization import serialize_result

if TYPE_CHECKING:
    from common.base import CheckResult, LearnResult, ProfileResult
    from common.engines.base import DataQualityEngine


@dataclass(frozen=True, slots=True)
class EngineBlockConfig(BlockConfig):
    """Configuration for engine blocks.

    Extends BlockConfig with engine-specific settings.

    Attributes:
        engine_name: Name of the engine to use (truthound, great_expectations, pandera).
        parallel: Enable parallel processing.
        max_workers: Maximum number of parallel workers.
        auto_start: Automatically start the engine on setup.
        auto_stop: Automatically stop the engine on teardown.
        auto_schema: Use auto-schema mode for Truthound engine.
        fail_on_error: Raise exception on check failures.
    """

    engine_name: str = "truthound"
    parallel: bool = False
    max_workers: int | None = None
    auto_start: bool = True
    auto_stop: bool = True
    auto_schema: bool = False
    fail_on_error: bool = True

    def with_engine(self, engine_name: str) -> EngineBlockConfig:
        """Return a new config with engine name changed."""
        return EngineBlockConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            engine_name=engine_name,
            parallel=self.parallel,
            max_workers=self.max_workers,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            auto_schema=self.auto_schema,
            fail_on_error=self.fail_on_error,
        )

    def with_parallel(
        self,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> EngineBlockConfig:
        """Return a new config with parallel settings changed."""
        return EngineBlockConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            engine_name=self.engine_name,
            parallel=parallel,
            max_workers=max_workers if max_workers else self.max_workers,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            auto_schema=self.auto_schema,
            fail_on_error=self.fail_on_error,
        )

    def with_auto_schema(self, auto_schema: bool = True) -> EngineBlockConfig:
        """Return a new config with auto_schema changed."""
        return EngineBlockConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            engine_name=self.engine_name,
            parallel=self.parallel,
            max_workers=self.max_workers,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            auto_schema=auto_schema,
            fail_on_error=self.fail_on_error,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> EngineBlockConfig:
        """Return a new config with fail_on_error changed."""
        return EngineBlockConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            engine_name=self.engine_name,
            parallel=self.parallel,
            max_workers=self.max_workers,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            auto_schema=self.auto_schema,
            fail_on_error=fail_on_error,
        )

    def with_lifecycle(
        self,
        auto_start: bool = True,
        auto_stop: bool = True,
    ) -> EngineBlockConfig:
        """Return a new config with lifecycle settings changed."""
        return EngineBlockConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            engine_name=self.engine_name,
            parallel=self.parallel,
            max_workers=self.max_workers,
            auto_start=auto_start,
            auto_stop=auto_stop,
            auto_schema=self.auto_schema,
            fail_on_error=self.fail_on_error,
        )


class EngineBlock(BaseBlock[EngineBlockConfig]):
    """Low-level block wrapping a DataQualityEngine.

    This block provides direct access to the data quality engine
    from the common/ module. Use this when you need fine-grained
    control over engine operations.

    Example:
        >>> config = EngineBlockConfig(engine_name="truthound", parallel=True)
        >>> with EngineBlock(config) as block:
        ...     result = block.check(data, auto_schema=True)
    """

    def __init__(
        self,
        config: EngineBlockConfig | None = None,
        engine: DataQualityEngine | None = None,
    ) -> None:
        """Initialize the engine block.

        Args:
            config: Block configuration.
            engine: Optional pre-configured engine instance.
        """
        super().__init__(config)
        self._provided_engine = engine
        self._engine: DataQualityEngine | None = None

    @classmethod
    def _default_config(cls) -> EngineBlockConfig:
        """Return the default engine block configuration."""
        return EngineBlockConfig()

    @property
    def engine(self) -> DataQualityEngine:
        """Get the underlying engine.

        Raises:
            RuntimeError: If the block has not been initialized.
        """
        if self._engine is None:
            raise RuntimeError("Engine block not initialized. Call setup() first.")
        return self._engine

    @property
    def engine_name(self) -> str:
        """Get the engine name."""
        if self._engine is not None:
            return self._engine.engine_name
        return self._config.engine_name

    @property
    def engine_version(self) -> str:
        """Get the engine version."""
        if self._engine is not None:
            return self._engine.engine_version
        return "unknown"

    def setup(self) -> None:
        """Initialize the block and create the engine."""
        if self._provided_engine is not None:
            self._engine = self._provided_engine
        else:
            self._engine = self._create_engine()

        # Start managed engines if configured
        if self._config.auto_start and hasattr(self._engine, "start"):
            self._engine.start()

        super().setup()

    def teardown(self) -> None:
        """Clean up the block and stop the engine."""
        if self._engine is not None and self._config.auto_stop:
            if hasattr(self._engine, "stop"):
                self._engine.stop()
        self._engine = None
        super().teardown()

    def _create_engine(self) -> DataQualityEngine:
        """Create the engine based on configuration.

        Returns:
            The created engine instance.

        Raises:
            EngineError: If engine creation fails.
        """
        engine_name = self._config.engine_name.lower()

        try:
            if engine_name == "truthound":
                return self._create_truthound_engine()
            elif engine_name in ("great_expectations", "ge"):
                return self._create_ge_engine()
            elif engine_name == "pandera":
                return self._create_pandera_engine()
            else:
                # Fallback to registry
                from common.engines import get_engine
                return get_engine(engine_name)
        except Exception as e:
            raise EngineError(
                message=f"Failed to create engine: {engine_name}",
                engine_name=engine_name,
                operation="create",
                original_error=e,
            ) from e

    def _create_truthound_engine(self) -> DataQualityEngine:
        """Create a Truthound engine."""
        from common.engines import TruthoundEngine, TruthoundEngineConfig

        config = TruthoundEngineConfig(
            parallel=self._config.parallel,
            max_workers=self._config.max_workers,
        )
        return TruthoundEngine(config=config)

    def _create_ge_engine(self) -> DataQualityEngine:
        """Create a Great Expectations engine."""
        from common.engines import GreatExpectationsAdapter
        return GreatExpectationsAdapter()

    def _create_pandera_engine(self) -> DataQualityEngine:
        """Create a Pandera engine."""
        from common.engines import PanderaAdapter
        return PanderaAdapter()

    def check(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> CheckResult:
        """Execute a data quality check.

        Args:
            data: The data to check.
            rules: Optional list of rules to check.
            **kwargs: Additional engine-specific arguments.

        Returns:
            The check result.

        Raises:
            EngineError: If the check fails.
        """
        try:
            # Apply auto_schema for Truthound
            if self._config.auto_schema and "auto_schema" not in kwargs:
                kwargs["auto_schema"] = True

            return self.engine.check(data, rules or [], **kwargs)
        except Exception as e:
            raise EngineError(
                message=f"Check operation failed: {e}",
                engine_name=self.engine_name,
                operation="check",
                original_error=e,
            ) from e

    def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        """Profile the data.

        Args:
            data: The data to profile.
            **kwargs: Additional engine-specific arguments.

        Returns:
            The profile result.

        Raises:
            EngineError: If profiling fails.
        """
        try:
            return self.engine.profile(data, **kwargs)
        except Exception as e:
            raise EngineError(
                message=f"Profile operation failed: {e}",
                engine_name=self.engine_name,
                operation="profile",
                original_error=e,
            ) from e

    def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        """Learn rules from the data.

        Args:
            data: The data to learn from.
            **kwargs: Additional engine-specific arguments.

        Returns:
            The learn result.

        Raises:
            EngineError: If learning fails.
        """
        try:
            return self.engine.learn(data, **kwargs)
        except Exception as e:
            raise EngineError(
                message=f"Learn operation failed: {e}",
                engine_name=self.engine_name,
                operation="learn",
                original_error=e,
            ) from e


class DataQualityBlock(Block):
    """Prefect Block for data quality operations.

    This is the high-level, Prefect-native block that can be saved,
    loaded, and used in Prefect flows and tasks. It wraps an EngineBlock
    and provides a convenient interface.

    Example:
        >>> # Create and save the block
        >>> block = DataQualityBlock(engine_name="truthound")
        >>> await block.save("my-dq-block")
        >>>
        >>> # Load and use in a flow
        >>> block = await DataQualityBlock.load("my-dq-block")
        >>> result = block.check(data)
    """

    _block_type_name = "Data Quality"
    _block_type_slug = "data-quality"
    _logo_url = "https://example.com/logo.png"
    _documentation_url = "https://github.com/truthound/truthound-orchestration"

    # Block configuration fields
    engine_name: str = Field(
        default="truthound",
        description="Name of the data quality engine (truthound, great_expectations, pandera)",
    )
    parallel: bool = Field(
        default=False,
        description="Enable parallel processing",
    )
    max_workers: int | None = Field(
        default=None,
        description="Maximum number of parallel workers",
    )
    auto_schema: bool = Field(
        default=False,
        description="Use auto-schema mode (Truthound only)",
    )
    fail_on_error: bool = Field(
        default=True,
        description="Raise exception on check failures",
    )
    warning_threshold: float | None = Field(
        default=None,
        description="Failure rate threshold for warnings (0.0 to 1.0)",
    )
    timeout_seconds: float = Field(
        default=300.0,
        description="Timeout for operations in seconds",
    )

    _engine_block: EngineBlock | None = None

    def _get_engine_block(self) -> EngineBlock:
        """Get or create the engine block."""
        if self._engine_block is None:
            config = EngineBlockConfig(
                engine_name=self.engine_name,
                parallel=self.parallel,
                max_workers=self.max_workers,
                auto_schema=self.auto_schema,
                fail_on_error=self.fail_on_error,
                timeout_seconds=self.timeout_seconds,
            )
            self._engine_block = EngineBlock(config)
            self._engine_block.setup()
        return self._engine_block

    def check(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a data quality check.

        Args:
            data: The data to check.
            rules: Optional list of rules to check.
            **kwargs: Additional engine-specific arguments.

        Returns:
            Serialized check result as a dictionary.
        """
        engine_block = self._get_engine_block()
        result = engine_block.check(data, rules, **kwargs)
        serialized = serialize_result(result)

        # Handle failure/warning logic
        if not serialized.get("is_success", True):
            failure_rate = serialized.get("failure_rate", 0.0)
            is_warning = (
                self.warning_threshold is not None
                and failure_rate <= self.warning_threshold
            )

            if not is_warning and self.fail_on_error:
                from truthound_prefect.utils.exceptions import DataQualityError
                raise DataQualityError(
                    message=f"Data quality check failed with {serialized['failed_count']} failures",
                    result=serialized,
                )

        return serialized

    def profile(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        """Profile the data.

        Args:
            data: The data to profile.
            **kwargs: Additional engine-specific arguments.

        Returns:
            Serialized profile result as a dictionary.
        """
        engine_block = self._get_engine_block()
        result = engine_block.profile(data, **kwargs)
        return serialize_result(result)

    def learn(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        """Learn rules from the data.

        Args:
            data: The data to learn from.
            **kwargs: Additional engine-specific arguments.

        Returns:
            Serialized learn result as a dictionary.
        """
        engine_block = self._get_engine_block()
        result = engine_block.learn(data, **kwargs)
        return serialize_result(result)

    def __del__(self) -> None:
        """Clean up resources."""
        if self._engine_block is not None:
            self._engine_block.teardown()


# Preset configurations
DEFAULT_ENGINE_CONFIG = EngineBlockConfig()

PARALLEL_ENGINE_CONFIG = EngineBlockConfig(
    parallel=True,
    max_workers=4,
    description="Parallel processing configuration",
)

PRODUCTION_ENGINE_CONFIG = EngineBlockConfig(
    enabled=True,
    timeout_seconds=600.0,
    parallel=True,
    max_workers=4,
    auto_start=True,
    auto_stop=True,
    fail_on_error=True,
    tags=frozenset({"production"}),
    description="Production engine configuration",
)

DEVELOPMENT_ENGINE_CONFIG = EngineBlockConfig(
    enabled=True,
    timeout_seconds=60.0,
    parallel=False,
    auto_start=True,
    auto_stop=True,
    fail_on_error=False,
    tags=frozenset({"development"}),
    description="Development engine configuration",
)

AUTO_SCHEMA_ENGINE_CONFIG = EngineBlockConfig(
    engine_name="truthound",
    auto_schema=True,
    description="Truthound with auto-schema mode",
)


__all__ = [
    # Configs
    "EngineBlockConfig",
    # Blocks
    "EngineBlock",
    "DataQualityBlock",
    # Presets
    "DEFAULT_ENGINE_CONFIG",
    "PARALLEL_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    "DEVELOPMENT_ENGINE_CONFIG",
    "AUTO_SCHEMA_ENGINE_CONFIG",
]
