"""Transformer blocks for data quality operations in Mage.

This module provides transformer blocks that execute data quality operations
on input data and return the processed results.

Components:
    - BaseDataQualityTransformer: Abstract base for all transformers
    - CheckTransformer: Execute validation checks
    - ProfileTransformer: Generate data profiles
    - LearnTransformer: Learn schemas from data

Example:
    >>> from truthound_mage import CheckTransformer, CheckBlockConfig
    >>>
    >>> config = CheckBlockConfig(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ... )
    >>> transformer = CheckTransformer(config=config)
    >>> result = transformer.execute(df)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from truthound_mage.blocks.base import (
    BlockConfig,
    BlockExecutionContext,
    BlockResult,
    CheckBlockConfig,
    LearnBlockConfig,
    ProfileBlockConfig,
)

if TYPE_CHECKING:
    import polars as pl

    from common.base import CheckResult, LearnResult, ProfileResult
    from common.engines.base import DataQualityEngine


# =============================================================================
# Base Transformer
# =============================================================================


class BaseDataQualityTransformer(ABC):
    """Abstract base class for data quality transformer blocks.

    This class provides the common infrastructure for data quality operations
    in Mage AI, including:

    - Engine management (supports any DataQualityEngine)
    - Data handling with Polars DataFrames
    - Result serialization for downstream blocks
    - Error handling with configurable failure behavior

    Subclasses must implement:
        - _execute_operation: The core operation logic
        - _serialize_result: Result serialization

    Attributes:
        config: Block configuration.

    Example:
        >>> class CustomTransformer(BaseDataQualityTransformer):
        ...     def _execute_operation(self, data, context):
        ...         return self.engine.check(data, self.config.rules)
    """

    def __init__(
        self,
        config: BlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize transformer block.

        Args:
            config: Block configuration. Uses default if None.
            engine: DataQualityEngine instance. Uses registry if None.
            hooks: Lifecycle hooks to invoke during execution.
        """
        self.config = config or BlockConfig()
        self._engine = engine
        self._hooks = list(hooks) if hooks else []

    @property
    def engine(self) -> DataQualityEngine:
        """Get the data quality engine instance.

        Returns:
            DataQualityEngine: The configured engine instance.

        Note:
            Lazily initializes the engine on first access.
            Uses the provided engine, or gets one from the registry.
        """
        if self._engine is None:
            from common.engines import get_engine

            self._engine = get_engine(self.config.engine_name)
        return self._engine

    def execute(
        self,
        data: pl.DataFrame | Any,
        context: BlockExecutionContext | None = None,
        **kwargs: Any,
    ) -> BlockResult:
        """Execute the data quality operation.

        This is the main entry point for transformer execution. It:
        1. Prepares the execution context
        2. Invokes pre-execution hooks
        3. Executes the subclass-specific operation
        4. Serializes results
        5. Invokes post-execution hooks
        6. Handles failures based on configuration

        Args:
            data: Input data (typically Polars DataFrame).
            context: Execution context. Creates default if None.
            **kwargs: Additional arguments passed to operation.

        Returns:
            BlockResult containing the operation result.

        Raises:
            DataQualityBlockError: If operation fails and fail_on_error is True.
        """
        from truthound_mage.utils.exceptions import BlockExecutionError

        context = context or BlockExecutionContext()
        start_time = time.perf_counter()

        # Invoke pre-execution hooks
        self._invoke_start_hooks(context)

        try:
            # Execute the operation
            result = self._execute_operation(data, context, **kwargs)

            # Serialize result
            result_dict = self._serialize_result(result)

            # Add metadata
            result_dict["_metadata"] = self._create_metadata(context)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            block_result = BlockResult(
                success=True,
                result=result,
                data=data,
                result_dict=result_dict,
                metadata=result_dict["_metadata"],
                execution_time_ms=execution_time_ms,
            )

            # Log results if configured
            if self.config.log_results:
                self._log_result(block_result)

            # Invoke success hooks
            self._invoke_success_hooks(block_result, context)

            # Handle result (may raise on failure)
            self._handle_result(result, block_result, context)

            return block_result

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            block_result = BlockResult(
                success=False,
                data=data,
                result_dict={"error": str(e)},
                metadata=self._create_metadata(context),
                execution_time_ms=execution_time_ms,
                error=e,
            )

            # Invoke error hooks
            self._invoke_error_hooks(e, context)

            if self.config.fail_on_error:
                raise BlockExecutionError(
                    f"Data quality operation failed: {e}",
                    block_uuid=context.block_uuid,
                    original_error=e,
                ) from e

            return block_result

    @abstractmethod
    def _execute_operation(
        self,
        data: Any,
        context: BlockExecutionContext,
        **kwargs: Any,
    ) -> CheckResult | ProfileResult | LearnResult:
        """Execute the specific data quality operation.

        Subclasses must implement this method to define their
        specific operation logic.

        Args:
            data: The input data (typically Polars DataFrame).
            context: Block execution context.
            **kwargs: Additional operation arguments.

        Returns:
            The operation result (CheckResult, ProfileResult, or LearnResult).
        """
        ...

    @abstractmethod
    def _serialize_result(
        self,
        result: CheckResult | ProfileResult | LearnResult,
    ) -> dict[str, Any]:
        """Serialize operation result.

        Subclasses must implement this to convert their result
        type to a dictionary for downstream blocks.

        Args:
            result: The operation result.

        Returns:
            dict[str, Any]: Serialized result dictionary.
        """
        ...

    def _create_metadata(self, context: BlockExecutionContext) -> dict[str, Any]:
        """Create execution metadata.

        Args:
            context: Block execution context.

        Returns:
            dict[str, Any]: Metadata dictionary.
        """
        return {
            "engine": self.engine.engine_name,
            "engine_version": self.engine.engine_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "block_uuid": context.block_uuid,
            "pipeline_uuid": context.pipeline_uuid,
            "partition": context.partition,
            "run_id": context.run_id,
            "tags": list(self.config.tags),
        }

    def _handle_result(
        self,
        result: Any,
        block_result: BlockResult,
        context: BlockExecutionContext,
    ) -> None:
        """Handle operation result, potentially raising on failure.

        Override this method to customize failure handling behavior.

        Args:
            result: The raw operation result.
            block_result: The wrapped block result.
            context: Block execution context.

        Raises:
            BlockExecutionError: If result indicates failure and fail_on_error is True.
        """
        # Default implementation does nothing
        # Subclasses override for specific handling

    def _log_result(self, result: BlockResult) -> None:
        """Log operation result.

        Args:
            result: The block result to log.
        """
        from common import get_logger

        logger = get_logger(__name__)
        logger.info(
            "Data quality operation completed",
            engine=self.engine.engine_name,
            execution_time_ms=result.execution_time_ms,
            success=result.success,
        )

    def _invoke_start_hooks(self, context: BlockExecutionContext) -> None:
        """Invoke pre-execution hooks."""
        for hook in self._hooks:
            if hasattr(hook, "on_block_start"):
                try:
                    hook.on_block_start(
                        block_name=self.__class__.__name__,
                        config=self.config,
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
# Check Transformer
# =============================================================================


class CheckTransformer(BaseDataQualityTransformer):
    """Transformer block for data quality checks.

    Executes validation checks on input data and returns the check result
    along with the original data for downstream processing.

    Attributes:
        config: CheckBlockConfig with validation rules and settings.

    Example:
        >>> config = CheckBlockConfig(
        ...     rules=[
        ...         {"column": "id", "type": "not_null"},
        ...         {"column": "email", "type": "regex", "pattern": r".*@.*"},
        ...     ],
        ...     warning_threshold=0.05,
        ... )
        >>> transformer = CheckTransformer(config=config)
        >>> result = transformer.execute(df)
        >>> if result.success:
        ...     print(f"Passed: {result.result_dict['passed_count']}")
    """

    def __init__(
        self,
        config: CheckBlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize check transformer.

        Args:
            config: Check block configuration. Uses default if None.
            engine: DataQualityEngine instance. Uses registry if None.
            hooks: Lifecycle hooks to invoke during execution.
        """
        super().__init__(config=config or CheckBlockConfig(), engine=engine, hooks=hooks)

    @property
    def config(self) -> CheckBlockConfig:
        """Get typed config."""
        return self._config  # type: ignore

    @config.setter
    def config(self, value: BlockConfig) -> None:
        """Set config with type validation."""
        if not isinstance(value, CheckBlockConfig):
            value = CheckBlockConfig.from_dict(value.to_dict())
        self._config = value

    def _execute_operation(
        self,
        data: Any,
        context: BlockExecutionContext,
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation checks on data.

        Args:
            data: Input data to validate.
            context: Block execution context.
            **kwargs: Additional check arguments.

        Returns:
            CheckResult with validation results.
        """
        # Build check kwargs
        check_kwargs: dict[str, Any] = {}

        if self.config.rules:
            check_kwargs["rules"] = list(self.config.rules)

        if self.config.parallel:
            check_kwargs["parallel"] = True

        if self.config.auto_schema:
            check_kwargs["auto_schema"] = True

        if self.config.min_severity:
            check_kwargs["min_severity"] = self.config.min_severity

        # Merge with any additional kwargs
        check_kwargs.update(kwargs)

        return self.engine.check(data, **check_kwargs)

    def _serialize_result(self, result: CheckResult) -> dict[str, Any]:
        """Serialize check result.

        Args:
            result: CheckResult to serialize.

        Returns:
            Serialized result dictionary.
        """
        from truthound_mage.utils.serialization import serialize_check_result

        return serialize_check_result(result)

    def _handle_result(
        self,
        result: CheckResult,
        block_result: BlockResult,
        context: BlockExecutionContext,
    ) -> None:
        """Handle check result, raising on failure if configured.

        Args:
            result: The check result.
            block_result: The wrapped block result.
            context: Block execution context.

        Raises:
            BlockExecutionError: If check failed and fail_on_error is True.
        """
        from common.base import CheckStatus

        from truthound_mage.utils.exceptions import BlockExecutionError

        if result.status in (CheckStatus.FAILED, CheckStatus.ERROR):
            if self.config.fail_on_error:
                raise BlockExecutionError(
                    f"Data quality check failed: {result.failed_count} failures",
                    block_uuid=context.block_uuid,
                )

        # Check warning threshold
        if (
            self.config.warning_threshold is not None
            and result.failure_rate > self.config.warning_threshold
        ):
            from common import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "Data quality warning: failure rate exceeds threshold",
                failure_rate=result.failure_rate,
                threshold=self.config.warning_threshold,
            )


# =============================================================================
# Profile Transformer
# =============================================================================


class ProfileTransformer(BaseDataQualityTransformer):
    """Transformer block for data profiling.

    Generates statistical profiles of input data including column statistics,
    distributions, and pattern detection.

    Attributes:
        config: ProfileBlockConfig with profiling settings.

    Example:
        >>> config = ProfileBlockConfig(
        ...     columns=frozenset(["amount", "quantity"]),
        ...     include_distributions=True,
        ... )
        >>> transformer = ProfileTransformer(config=config)
        >>> result = transformer.execute(df)
        >>> for col in result.result.columns:
        ...     print(f"{col.column_name}: {col.dtype}")
    """

    def __init__(
        self,
        config: ProfileBlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize profile transformer.

        Args:
            config: Profile block configuration. Uses default if None.
            engine: DataQualityEngine instance. Uses registry if None.
            hooks: Lifecycle hooks to invoke during execution.
        """
        super().__init__(config=config or ProfileBlockConfig(), engine=engine, hooks=hooks)

    @property
    def config(self) -> ProfileBlockConfig:
        """Get typed config."""
        return self._config  # type: ignore

    @config.setter
    def config(self, value: BlockConfig) -> None:
        """Set config with type validation."""
        if not isinstance(value, ProfileBlockConfig):
            value = ProfileBlockConfig.from_dict(value.to_dict())
        self._config = value

    def _execute_operation(
        self,
        data: Any,
        context: BlockExecutionContext,
        **kwargs: Any,
    ) -> ProfileResult:
        """Execute profiling on data.

        Args:
            data: Input data to profile.
            context: Block execution context.
            **kwargs: Additional profile arguments.

        Returns:
            ProfileResult with profiling results.
        """
        profile_kwargs: dict[str, Any] = {}

        if self.config.columns:
            profile_kwargs["columns"] = list(self.config.columns)

        if self.config.sample_size:
            profile_kwargs["sample_size"] = self.config.sample_size

        profile_kwargs.update(kwargs)

        return self.engine.profile(data, **profile_kwargs)

    def _serialize_result(self, result: ProfileResult) -> dict[str, Any]:
        """Serialize profile result.

        Args:
            result: ProfileResult to serialize.

        Returns:
            Serialized result dictionary.
        """
        from truthound_mage.utils.serialization import serialize_profile_result

        return serialize_profile_result(result)


# =============================================================================
# Learn Transformer
# =============================================================================


class LearnTransformer(BaseDataQualityTransformer):
    """Transformer block for schema learning.

    Learns data schemas and validation rules from input data,
    optionally saving them for later use.

    Attributes:
        config: LearnBlockConfig with learning settings.

    Example:
        >>> config = LearnBlockConfig(
        ...     output_path="schemas/users.json",
        ...     strictness="moderate",
        ... )
        >>> transformer = LearnTransformer(config=config)
        >>> result = transformer.execute(df)
        >>> for rule in result.result.rules:
        ...     print(f"{rule.column}: {rule.rule_type}")
    """

    def __init__(
        self,
        config: LearnBlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize learn transformer.

        Args:
            config: Learn block configuration. Uses default if None.
            engine: DataQualityEngine instance. Uses registry if None.
            hooks: Lifecycle hooks to invoke during execution.
        """
        super().__init__(config=config or LearnBlockConfig(), engine=engine, hooks=hooks)

    @property
    def config(self) -> LearnBlockConfig:
        """Get typed config."""
        return self._config  # type: ignore

    @config.setter
    def config(self, value: BlockConfig) -> None:
        """Set config with type validation."""
        if not isinstance(value, LearnBlockConfig):
            value = LearnBlockConfig.from_dict(value.to_dict())
        self._config = value

    def _execute_operation(
        self,
        data: Any,
        context: BlockExecutionContext,
        **kwargs: Any,
    ) -> LearnResult:
        """Execute schema learning on data.

        Args:
            data: Input data to learn from.
            context: Block execution context.
            **kwargs: Additional learn arguments.

        Returns:
            LearnResult with learned schema.
        """
        learn_kwargs: dict[str, Any] = {}

        if self.config.infer_constraints:
            learn_kwargs["infer_constraints"] = True

        if self.config.categorical_threshold:
            learn_kwargs["categorical_threshold"] = self.config.categorical_threshold

        learn_kwargs.update(kwargs)

        return self.engine.learn(data, **learn_kwargs)

    def _serialize_result(self, result: LearnResult) -> dict[str, Any]:
        """Serialize learn result.

        Args:
            result: LearnResult to serialize.

        Returns:
            Serialized result dictionary.
        """
        from truthound_mage.utils.serialization import serialize_learn_result

        return serialize_learn_result(result)


# =============================================================================
# Generic Transformer (Facade)
# =============================================================================


class DataQualityTransformer(BaseDataQualityTransformer):
    """Generic data quality transformer that delegates to specialized transformers.

    This class provides a unified interface for all data quality operations,
    automatically selecting the appropriate transformer based on the operation type.

    Example:
        >>> transformer = DataQualityTransformer()
        >>> result = transformer.check(df, rules=[{"column": "id", "type": "not_null"}])
        >>> profile = transformer.profile(df)
        >>> schema = transformer.learn(df)
    """

    def __init__(
        self,
        config: BlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize generic transformer.

        Args:
            config: Base block configuration. Uses default if None.
            engine: DataQualityEngine instance. Uses registry if None.
            hooks: Lifecycle hooks to invoke during execution.
        """
        super().__init__(config=config, engine=engine, hooks=hooks)
        self._check_transformer: CheckTransformer | None = None
        self._profile_transformer: ProfileTransformer | None = None
        self._learn_transformer: LearnTransformer | None = None

    def check(
        self,
        data: Any,
        rules: Sequence[dict[str, Any]] | None = None,
        context: BlockExecutionContext | None = None,
        **kwargs: Any,
    ) -> BlockResult:
        """Execute data quality checks.

        Args:
            data: Input data to validate.
            rules: Validation rules to apply.
            context: Block execution context.
            **kwargs: Additional check arguments.

        Returns:
            BlockResult with check results.
        """
        if self._check_transformer is None:
            config = CheckBlockConfig(
                engine_name=self.config.engine_name,
                fail_on_error=self.config.fail_on_error,
                timeout_seconds=self.config.timeout_seconds,
                tags=self.config.tags,
            )
            self._check_transformer = CheckTransformer(
                config=config,
                engine=self._engine,
                hooks=self._hooks,
            )

        if rules:
            self._check_transformer.config = self._check_transformer.config.with_rules(rules)

        return self._check_transformer.execute(data, context, **kwargs)

    def profile(
        self,
        data: Any,
        columns: Sequence[str] | None = None,
        context: BlockExecutionContext | None = None,
        **kwargs: Any,
    ) -> BlockResult:
        """Execute data profiling.

        Args:
            data: Input data to profile.
            columns: Columns to profile.
            context: Block execution context.
            **kwargs: Additional profile arguments.

        Returns:
            BlockResult with profile results.
        """
        if self._profile_transformer is None:
            config = ProfileBlockConfig(
                engine_name=self.config.engine_name,
                fail_on_error=self.config.fail_on_error,
                timeout_seconds=self.config.timeout_seconds,
                tags=self.config.tags,
            )
            self._profile_transformer = ProfileTransformer(
                config=config,
                engine=self._engine,
                hooks=self._hooks,
            )

        if columns:
            self._profile_transformer.config = self._profile_transformer.config.with_columns(
                columns
            )

        return self._profile_transformer.execute(data, context, **kwargs)

    def learn(
        self,
        data: Any,
        context: BlockExecutionContext | None = None,
        **kwargs: Any,
    ) -> BlockResult:
        """Execute schema learning.

        Args:
            data: Input data to learn from.
            context: Block execution context.
            **kwargs: Additional learn arguments.

        Returns:
            BlockResult with learn results.
        """
        if self._learn_transformer is None:
            config = LearnBlockConfig(
                engine_name=self.config.engine_name,
                fail_on_error=self.config.fail_on_error,
                timeout_seconds=self.config.timeout_seconds,
                tags=self.config.tags,
            )
            self._learn_transformer = LearnTransformer(
                config=config,
                engine=self._engine,
                hooks=self._hooks,
            )

        return self._learn_transformer.execute(data, context, **kwargs)

    def _execute_operation(
        self,
        data: Any,
        context: BlockExecutionContext,
        **kwargs: Any,
    ) -> CheckResult:
        """Default operation: execute check."""
        return self.engine.check(data, **kwargs)

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        """Serialize any result type."""
        from truthound_mage.utils.serialization import serialize_result

        return serialize_result(result)


# =============================================================================
# Factory Functions
# =============================================================================


def create_check_transformer(
    rules: Sequence[dict[str, Any]],
    engine_name: str | None = None,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    **kwargs: Any,
) -> CheckTransformer:
    """Create a check transformer with the given configuration.

    Args:
        rules: Validation rules to apply.
        engine_name: Name of engine to use.
        fail_on_error: Whether to fail on validation errors.
        warning_threshold: Failure rate threshold for warnings.
        **kwargs: Additional configuration options.

    Returns:
        Configured CheckTransformer instance.

    Example:
        >>> transformer = create_check_transformer(
        ...     rules=[{"column": "id", "type": "not_null"}],
        ...     warning_threshold=0.05,
        ... )
    """
    config = CheckBlockConfig(
        engine_name=engine_name,
        fail_on_error=fail_on_error,
        rules=tuple(rules),
        warning_threshold=warning_threshold,
        **kwargs,
    )
    return CheckTransformer(config=config)


def create_profile_transformer(
    columns: Sequence[str] | None = None,
    engine_name: str | None = None,
    include_statistics: bool = True,
    include_distributions: bool = True,
    **kwargs: Any,
) -> ProfileTransformer:
    """Create a profile transformer with the given configuration.

    Args:
        columns: Columns to profile (None = all).
        engine_name: Name of engine to use.
        include_statistics: Whether to include statistics.
        include_distributions: Whether to include distributions.
        **kwargs: Additional configuration options.

    Returns:
        Configured ProfileTransformer instance.

    Example:
        >>> transformer = create_profile_transformer(
        ...     columns=["amount", "quantity"],
        ...     include_distributions=True,
        ... )
    """
    config = ProfileBlockConfig(
        engine_name=engine_name,
        columns=frozenset(columns) if columns else None,
        include_statistics=include_statistics,
        include_distributions=include_distributions,
        **kwargs,
    )
    return ProfileTransformer(config=config)


def create_learn_transformer(
    output_path: str | None = None,
    engine_name: str | None = None,
    strictness: str = "moderate",
    infer_constraints: bool = True,
    **kwargs: Any,
) -> LearnTransformer:
    """Create a learn transformer with the given configuration.

    Args:
        output_path: Path to save learned schema.
        engine_name: Name of engine to use.
        strictness: Learning strictness level.
        infer_constraints: Whether to infer constraints.
        **kwargs: Additional configuration options.

    Returns:
        Configured LearnTransformer instance.

    Example:
        >>> transformer = create_learn_transformer(
        ...     output_path="schemas/users.json",
        ...     strictness="strict",
        ... )
    """
    config = LearnBlockConfig(
        engine_name=engine_name,
        output_path=output_path,
        strictness=strictness,
        infer_constraints=infer_constraints,
        **kwargs,
    )
    return LearnTransformer(config=config)
