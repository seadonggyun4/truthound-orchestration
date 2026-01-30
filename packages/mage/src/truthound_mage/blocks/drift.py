"""Data Quality Drift Detection Block for Mage AI.

This module provides the DriftTransformer block for executing drift
detection between baseline and current datasets in Mage AI pipelines.

Example:
    >>> from truthound_mage.blocks.drift import DriftTransformer
    >>>
    >>> transformer = DriftTransformer(method="ks", threshold=0.05)
    >>> result = transformer.execute(baseline_data, current_data)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from common.base import DriftResult
    from common.engines.base import DataQualityEngine

from truthound_mage.blocks.base import (
    BlockConfig,
    BlockExecutionContext,
    BlockResult,
    DriftBlockConfig,
)


# =============================================================================
# Base Drift Transformer
# =============================================================================


class BaseDriftTransformer(ABC):
    """Abstract base class for drift detection transformers.

    This transformer requires TWO data inputs (baseline and current)
    and cannot extend BaseDataQualityTransformer which uses single input.

    Subclasses must implement:
        - _execute_drift: The core drift detection logic

    Attributes:
        config: DriftBlockConfig with drift detection settings.

    Example:
        >>> class CustomDriftTransformer(BaseDriftTransformer):
        ...     def _execute_drift(self, baseline, current, **kwargs):
        ...         return self.engine.detect_drift(baseline, current, **kwargs)
    """

    def __init__(
        self,
        config: DriftBlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        hooks: list[Any] | None = None,
    ) -> None:
        """Initialize drift transformer block.

        Args:
            config: Drift block configuration. Uses default if None.
            engine: DataQualityEngine instance. Uses registry if None.
            engine_name: Name of the engine to use from registry.
            hooks: Lifecycle hooks to invoke during execution.
        """
        from truthound_mage.blocks.base import DEFAULT_DRIFT_BLOCK_CONFIG

        self._config = config or DEFAULT_DRIFT_BLOCK_CONFIG
        self._engine_name = engine_name or self._config.engine_name
        self._engine = engine
        self._hooks: list[Any] = list(hooks) if hooks else []

    @property
    def config(self) -> DriftBlockConfig:
        """Get the drift block configuration."""
        return self._config

    @property
    def engine(self) -> DataQualityEngine:
        """Get the data quality engine instance.

        Returns:
            DataQualityEngine: The configured engine instance.

        Note:
            Lazily initializes the engine on first access.
        """
        if self._engine is None:
            from common.engines import get_engine

            self._engine = get_engine(self._engine_name)
        return self._engine

    def add_hook(self, hook: Any) -> None:
        """Add a lifecycle hook.

        Args:
            hook: Hook instance with on_block_start/on_block_success/on_block_error.
        """
        self._hooks.append(hook)

    def execute(
        self,
        baseline_data: Any,
        current_data: Any,
        context: BlockExecutionContext | None = None,
        **kwargs: Any,
    ) -> BlockResult:
        """Execute drift detection between baseline and current data.

        Args:
            baseline_data: Baseline dataset.
            current_data: Current dataset.
            context: Block execution context. Creates default if None.
            **kwargs: Additional keyword arguments.

        Returns:
            BlockResult containing the drift detection result.

        Raises:
            BlockExecutionError: If drift detected and fail_on_drift is True.
        """
        context = context or BlockExecutionContext()
        start_time = time.perf_counter()

        # Invoke pre-execution hooks
        self._invoke_start_hooks(context)

        try:
            result = self._execute_drift(baseline_data, current_data, **kwargs)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            serialized = self._serialize_result(result)
            serialized["_metadata"] = self._create_metadata(context)

            block_result = BlockResult(
                success=True,
                result=None,
                data=serialized,
                result_dict=serialized,
                metadata={
                    "engine": self.engine.engine_name,
                    "engine_version": self.engine.engine_version,
                    "duration_ms": execution_time_ms,
                    "status": result.status.name,
                    "drifted_count": result.drifted_count,
                    "total_columns": result.total_columns,
                    "drift_rate": result.drift_rate,
                },
                execution_time_ms=execution_time_ms,
            )

            # Log results if configured
            if self._config.log_results:
                self._log_metrics(result, execution_time_ms)

            # Invoke success hooks
            self._invoke_success_hooks(block_result, context)

            # Handle result (may raise on drift)
            self._handle_result(result, context)

            return block_result

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            block_result = BlockResult(
                success=False,
                data=None,
                result_dict={"error": str(e)},
                metadata=self._create_metadata(context),
                execution_time_ms=execution_time_ms,
                error=e,
            )

            # Invoke error hooks
            self._invoke_error_hooks(e, context)

            if self._config.fail_on_error:
                from truthound_mage.utils.exceptions import BlockExecutionError

                raise BlockExecutionError(
                    f"Drift detection failed: {e}",
                    block_uuid=context.block_uuid,
                    original_error=e,
                ) from e

            return block_result

    @abstractmethod
    def _execute_drift(
        self,
        baseline_data: Any,
        current_data: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute the drift detection operation.

        Args:
            baseline_data: Baseline dataset.
            current_data: Current dataset.
            **kwargs: Additional arguments.

        Returns:
            DriftResult with drift detection results.
        """
        ...

    def _serialize_result(self, result: DriftResult) -> dict[str, Any]:
        """Serialize drift result to dictionary.

        Args:
            result: DriftResult to serialize.

        Returns:
            Serialized result dictionary.
        """
        return result.to_dict()

    def _handle_result(
        self,
        result: DriftResult,
        context: BlockExecutionContext,
    ) -> None:
        """Handle drift result, raising on drift if configured.

        Args:
            result: The drift result.
            context: Block execution context.

        Raises:
            BlockExecutionError: If drift detected and fail_on_drift is True.
        """
        if result.is_drifted and self._config.fail_on_drift:
            from truthound_mage.utils.exceptions import BlockExecutionError

            drifted_names = [c.column for c in result.drifted_columns if c.is_drifted]
            msg = (
                f"Drift detected: {result.drifted_count}/{result.total_columns} columns "
                f"({result.drift_rate:.2f}%). "
                f"Drifted: {', '.join(drifted_names[:3])}"
            )
            if len(drifted_names) > 3:
                msg += f" ... ({len(drifted_names) - 3} more)"

            raise BlockExecutionError(
                msg,
                block_uuid=context.block_uuid,
            )

    def _log_metrics(self, result: DriftResult, duration_ms: float) -> None:
        """Log drift detection metrics.

        Args:
            result: The drift result.
            duration_ms: Execution duration in milliseconds.
        """
        from common import get_logger

        logger = get_logger(__name__)
        logger.info(
            "Drift detection completed",
            engine=self.engine.engine_name,
            status=result.status.name,
            drifted_count=result.drifted_count,
            total_columns=result.total_columns,
            drift_rate=result.drift_rate,
            duration_ms=duration_ms,
        )

    def _create_metadata(self, context: BlockExecutionContext) -> dict[str, Any]:
        """Create execution metadata.

        Args:
            context: Block execution context.

        Returns:
            Metadata dictionary.
        """
        return {
            "engine": self.engine.engine_name,
            "engine_version": self.engine.engine_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "block_uuid": context.block_uuid,
            "pipeline_uuid": context.pipeline_uuid,
            "partition": context.partition,
            "run_id": context.run_id,
            "tags": list(self._config.tags),
        }

    def _invoke_start_hooks(self, context: BlockExecutionContext) -> None:
        """Invoke pre-execution hooks."""
        for hook in self._hooks:
            if hasattr(hook, "on_block_start"):
                try:
                    hook.on_block_start(
                        block_name=self.__class__.__name__,
                        config=self._config,
                        context=context,
                    )
                except Exception:
                    pass  # Hooks should not break execution

    def _invoke_success_hooks(
        self,
        result: BlockResult,
        context: BlockExecutionContext,
    ) -> None:
        """Invoke post-execution success hooks."""
        for hook in self._hooks:
            if hasattr(hook, "on_block_success"):
                try:
                    hook.on_block_success(
                        block_name=self.__class__.__name__,
                        result=result,
                        context=context,
                    )
                except Exception:
                    pass

    def _invoke_error_hooks(
        self,
        error: Exception,
        context: BlockExecutionContext,
    ) -> None:
        """Invoke error hooks."""
        for hook in self._hooks:
            if hasattr(hook, "on_block_error"):
                try:
                    hook.on_block_error(
                        block_name=self.__class__.__name__,
                        error=error,
                        context=context,
                    )
                except Exception:
                    pass


# =============================================================================
# Drift Transformer
# =============================================================================


class DriftTransformer(BaseDriftTransformer):
    """Concrete drift detection transformer.

    Executes drift detection using the configured engine's
    detect_drift method.

    Attributes:
        config: DriftBlockConfig with drift detection settings.

    Example:
        >>> transformer = DriftTransformer(
        ...     config=DriftBlockConfig(method="ks", threshold=0.05),
        ... )
        >>> result = transformer.execute(baseline_df, current_df)
        >>> if result.success:
        ...     print(f"Status: {result.metadata['status']}")
    """

    def __init__(
        self,
        config: DriftBlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        method: str | None = None,
        columns: list[str] | None = None,
        threshold: float | None = None,
        fail_on_drift: bool | None = None,
        hooks: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize drift transformer.

        Args:
            config: Drift block configuration. Built from kwargs if None.
            engine: DataQualityEngine instance. Uses registry if None.
            engine_name: Name of the engine to use from registry.
            method: Drift detection method.
            columns: Columns to check for drift.
            threshold: Detection threshold.
            fail_on_drift: Whether to fail on drift detection.
            hooks: Lifecycle hooks to invoke during execution.
            **kwargs: Additional configuration options.
        """
        if config is None:
            config_kwargs: dict[str, Any] = {}
            if engine_name is not None:
                config_kwargs["engine_name"] = engine_name
            if method is not None:
                config_kwargs["method"] = method
            if columns is not None:
                config_kwargs["columns"] = tuple(columns)
            if threshold is not None:
                config_kwargs["threshold"] = threshold
            if fail_on_drift is not None:
                config_kwargs["fail_on_drift"] = fail_on_drift
            config = DriftBlockConfig(**config_kwargs) if config_kwargs else None

        super().__init__(
            config=config,
            engine=engine,
            engine_name=engine_name,
            hooks=hooks,
        )

    def _execute_drift(
        self,
        baseline_data: Any,
        current_data: Any,
        **kwargs: Any,
    ) -> DriftResult:
        """Execute drift detection using the engine.

        Args:
            baseline_data: Baseline dataset.
            current_data: Current dataset.
            **kwargs: Additional arguments passed to detect_drift.

        Returns:
            DriftResult with drift detection results.

        Raises:
            ValueError: If engine does not support drift detection.
        """
        from common.engines.base import supports_drift

        if not supports_drift(self.engine):
            raise ValueError(
                f"Engine '{self.engine.engine_name}' does not support drift detection."
            )

        detect_kwargs: dict[str, Any] = {"method": self._config.method}
        if self._config.columns is not None:
            detect_kwargs["columns"] = list(self._config.columns)
        if self._config.threshold is not None:
            detect_kwargs["threshold"] = self._config.threshold
        detect_kwargs.update(kwargs)

        return self.engine.detect_drift(baseline_data, current_data, **detect_kwargs)


# =============================================================================
# Factory Functions
# =============================================================================


def create_drift_transformer(
    engine_name: str | None = None,
    method: str = "auto",
    threshold: float | None = None,
    fail_on_drift: bool = True,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> DriftTransformer:
    """Create a drift transformer with the given configuration.

    Args:
        engine_name: Name of engine to use.
        method: Drift detection method.
        threshold: Detection threshold.
        fail_on_drift: Whether to fail on drift.
        columns: Columns to check for drift.
        **kwargs: Additional configuration options.

    Returns:
        Configured DriftTransformer instance.

    Example:
        >>> transformer = create_drift_transformer(
        ...     method="ks",
        ...     threshold=0.05,
        ... )
    """
    config = DriftBlockConfig(
        engine_name=engine_name,
        method=method,
        threshold=threshold,
        fail_on_drift=fail_on_drift,
        columns=tuple(columns) if columns else None,
        **kwargs,
    )
    return DriftTransformer(config=config)
