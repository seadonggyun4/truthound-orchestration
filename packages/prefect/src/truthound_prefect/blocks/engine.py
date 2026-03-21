"""Engine block for data quality operations.

This module provides Prefect Block implementations that wrap data quality
engines from the common/ module, following the Protocol-based design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Sequence

from prefect.blocks.core import Block

try:
    from pydantic.v1 import Field, PrivateAttr
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import Field, PrivateAttr

from common.engines import normalize_runtime_context
from common.orchestration import (
    StreamCheckpointState,
    StreamRequest,
    execute_operation,
    run_stream_check,
    run_stream_check_async,
    summarize_stream,
)
from truthound_prefect.blocks.base import BaseBlock, BlockConfig
from truthound_prefect.utils.exceptions import EngineError
from truthound_prefect.utils.serialization import serialize_result

if TYPE_CHECKING:
    from common.base import CheckResult, LearnResult, ProfileResult
    from common.engines.base import DataQualityEngine


def _build_prefect_runtime_context(
    *,
    block_type: str,
    operation: str | None = None,
) -> Any:
    """Build Prefect runtime context with host-native execution identifiers."""

    host_execution: dict[str, Any] = {}
    try:  # pragma: no cover - depends on Prefect runtime context
        from prefect.runtime import flow_run, task_run

        host_execution = {
            "flow_run_id": getattr(flow_run, "id", None),
            "task_run_id": getattr(task_run, "id", None),
            "deployment_id": getattr(flow_run, "deployment_id", None),
        }
    except Exception:
        host_execution = {}

    host_metadata = {"block_type": block_type}
    if operation is not None:
        host_metadata["operation"] = operation

    return normalize_runtime_context(
        platform="prefect",
        host_metadata=host_metadata,
        host_execution=host_execution,
    )


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
    observability: dict[str, Any] = field(default_factory=dict)

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
            observability=self.observability,
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
            observability=self.observability,
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
            observability=self.observability,
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
            observability=self.observability,
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
            observability=self.observability,
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
        self._runtime_context = _build_prefect_runtime_context(block_type=type(self).__name__)

    def _refresh_runtime_context(self, operation: str | None = None) -> Any:
        self._runtime_context = _build_prefect_runtime_context(
            block_type=type(self).__name__,
            operation=operation,
        )
        return self._runtime_context

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
            from common.engines import (
                EngineCreationRequest,
                create_engine,
                normalize_runtime_context,
                run_preflight,
            )

            runtime_context = self._refresh_runtime_context("engine_create")
            request = EngineCreationRequest(
                engine_name=engine_name,
                runtime_context=runtime_context,
                observability=self._config.observability,
            )
            preflight = run_preflight(request, observability=self._config.observability)
            if not preflight.compatible:
                failures = "; ".join(
                    check.message for check in preflight.compatibility.failures
                )
                raise EngineError(
                    message=f"Preflight failed: {failures}",
                    engine_name=engine_name,
                    operation="preflight",
                )

            return create_engine(
                request,
                auto_start=False,
                auto_stop=False,
                parallel=self._config.parallel,
                max_workers=self._config.max_workers,
            )
        except Exception as e:
            raise EngineError(
                message=f"Failed to create engine: {engine_name}",
                engine_name=engine_name,
                operation="create",
                original_error=e,
            ) from e

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

            return execute_operation(
                "check",
                self.engine,
                data=data,
                rules=rules,
                runtime_context=self._refresh_runtime_context("check"),
                observability=self._config.observability,
                **kwargs,
            )
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
            return execute_operation(
                "profile",
                self.engine,
                data=data,
                runtime_context=self._refresh_runtime_context("profile"),
                observability=self._config.observability,
                **kwargs,
            )
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
            return execute_operation(
                "learn",
                self.engine,
                data=data,
                runtime_context=self._refresh_runtime_context("learn"),
                observability=self._config.observability,
                **kwargs,
            )
        except Exception as e:
            raise EngineError(
                message=f"Learn operation failed: {e}",
                engine_name=self.engine_name,
                operation="learn",
                original_error=e,
            ) from e

    def stream(
        self,
        stream: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        *,
        batch_size: int = 1000,
        checkpoint: dict[str, Any] | None = None,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        checkpoint_state = (
            StreamCheckpointState(**checkpoint) if checkpoint is not None else None
        )
        envelopes = list(
            run_stream_check(
                self.engine,
                StreamRequest(
                    stream=stream,
                    rules=rules,
                    batch_size=batch_size,
                    checkpoint=checkpoint_state,
                    max_batches=max_batches,
                    kwargs=kwargs,
                ),
                runtime_context=self._refresh_runtime_context("stream"),
                observability=self._config.observability,
            )
        )
        return {
            "batches": [envelope.to_dict() for envelope in envelopes],
            "summary": summarize_stream(envelopes).to_dict(),
        }

    async def astream(
        self,
        stream: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        *,
        batch_size: int = 1000,
        checkpoint: dict[str, Any] | None = None,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not inspect.isasyncgen(stream) and not hasattr(stream, "__aiter__"):
            return self.stream(
                stream,
                rules=rules,
                batch_size=batch_size,
                checkpoint=checkpoint,
                max_batches=max_batches,
                **kwargs,
            )

        checkpoint_state = (
            StreamCheckpointState(**checkpoint) if checkpoint is not None else None
        )
        envelopes = [
            envelope
            async for envelope in run_stream_check_async(
                self.engine,
                StreamRequest(
                    stream=stream,
                    rules=rules,
                    batch_size=batch_size,
                    checkpoint=checkpoint_state,
                    max_batches=max_batches,
                    kwargs=kwargs,
                ),
                runtime_context=self._refresh_runtime_context("stream"),
                observability=self._config.observability,
            )
        ]
        return {
            "batches": [envelope.to_dict() for envelope in envelopes],
            "summary": summarize_stream(envelopes).to_dict(),
        }


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
    observability: dict[str, Any] = Field(
        default_factory=dict,
        description="Shared observability configuration for runtime events",
    )

    _engine_block: EngineBlock | None = PrivateAttr(default=None)

    def _current_engine_block(self) -> EngineBlock | None:
        engine_block = getattr(self, "_engine_block", None)
        return engine_block if isinstance(engine_block, EngineBlock) else None

    def _get_engine_block(self) -> EngineBlock:
        """Get or create the engine block."""
        engine_block = self._current_engine_block()
        if engine_block is None:
            config = EngineBlockConfig(
                engine_name=self.engine_name,
                parallel=self.parallel,
                max_workers=self.max_workers,
                auto_schema=self.auto_schema,
                fail_on_error=self.fail_on_error,
                timeout_seconds=self.timeout_seconds,
                observability=self.observability,
            )
            engine_block = EngineBlock(config)
            engine_block.setup()
            self._engine_block = engine_block
        return engine_block

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

    def stream(
        self,
        stream: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        *,
        batch_size: int = 1000,
        checkpoint: dict[str, Any] | None = None,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        engine_block = self._get_engine_block()
        return engine_block.stream(
            stream,
            rules=rules,
            batch_size=batch_size,
            checkpoint=checkpoint,
            max_batches=max_batches,
            **kwargs,
        )

    async def astream(
        self,
        stream: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        *,
        batch_size: int = 1000,
        checkpoint: dict[str, Any] | None = None,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        engine_block = self._get_engine_block()
        return await engine_block.astream(
            stream,
            rules=rules,
            batch_size=batch_size,
            checkpoint=checkpoint,
            max_batches=max_batches,
            **kwargs,
        )

    def __del__(self) -> None:
        """Clean up resources."""
        engine_block = self._current_engine_block()
        if engine_block is not None:
            engine_block.teardown()


def create_ephemeral_truthound_block(
    *,
    auto_schema: bool = False,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    timeout_seconds: float = 300.0,
) -> DataQualityBlock:
    """Create an in-memory Truthound-backed block for zero-config execution."""

    return DataQualityBlock(
        engine_name="truthound",
        auto_schema=auto_schema,
        fail_on_error=fail_on_error,
        warning_threshold=warning_threshold,
        timeout_seconds=timeout_seconds,
    )


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
    "create_ephemeral_truthound_block",
    # Presets
    "DEFAULT_ENGINE_CONFIG",
    "PARALLEL_ENGINE_CONFIG",
    "PRODUCTION_ENGINE_CONFIG",
    "DEVELOPMENT_ENGINE_CONFIG",
    "AUTO_SCHEMA_ENGINE_CONFIG",
]
