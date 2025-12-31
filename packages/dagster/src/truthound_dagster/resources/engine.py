"""Engine Resources for Dagster Integration.

This module provides Dagster resources for data quality engine integration.
The resources wrap DataQualityEngine implementations and provide a
Dagster-native interface for data validation operations.

Example:
    >>> from dagster import Definitions, asset
    >>> from truthound_dagster.resources import DataQualityResource
    >>>
    >>> @asset
    ... def validated_data(data_quality: DataQualityResource):
    ...     result = data_quality.check(
    ...         data=load_data(),
    ...         rules=[{"column": "id", "type": "not_null"}],
    ...     )
    ...     if not result.is_success:
    ...         raise ValueError("Data quality check failed")
    ...     return result
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from dagster import ConfigurableResource, InitResourceContext

from truthound_dagster.resources.base import BaseResource, ResourceConfig

if TYPE_CHECKING:
    from common.base import CheckResult, LearnResult, ProfileResult
    from common.engines.base import DataQualityEngine


@dataclass(frozen=True, slots=True)
class EngineResourceConfig(ResourceConfig):
    """Configuration for engine resource.

    Attributes:
        enabled: Whether the resource is enabled.
        timeout_seconds: Default timeout for operations.
        engine_name: Name of engine to use from registry.
        auto_start: Whether to auto-start managed engines.
        auto_stop: Whether to auto-stop managed engines.
        parallel: Enable parallel validation (Truthound).
        max_workers: Maximum parallel workers.

    Example:
        >>> config = EngineResourceConfig(
        ...     engine_name="truthound",
        ...     parallel=True,
        ...     max_workers=4,
        ... )
    """

    engine_name: str = "truthound"
    auto_start: bool = True
    auto_stop: bool = True
    parallel: bool = False
    max_workers: Optional[int] = None
    tags: "frozenset[str]" = field(default_factory=frozenset)

    def with_engine(self, engine_name: str) -> "EngineResourceConfig":
        """Return new config with updated engine name."""
        return EngineResourceConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            engine_name=engine_name,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            parallel=self.parallel,
            max_workers=self.max_workers,
            tags=self.tags,
        )

    def with_parallel(
        self,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> "EngineResourceConfig":
        """Return new config with updated parallel settings."""
        return EngineResourceConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            engine_name=self.engine_name,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            parallel=parallel,
            max_workers=max_workers if max_workers is not None else self.max_workers,
            tags=self.tags,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = ResourceConfig.to_dict(self)
        return {
            **base_dict,
            "engine_name": self.engine_name,
            "auto_start": self.auto_start,
            "auto_stop": self.auto_stop,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
        }


class EngineResource(BaseResource["EngineResourceConfig"]):
    """Low-level engine resource for direct engine access.

    This resource provides direct access to a DataQualityEngine instance.
    For most use cases, prefer DataQualityResource which provides a
    higher-level interface.

    Parameters
    ----------
    config : EngineResourceConfig | None
        Engine configuration. Uses default if None.

    engine : DataQualityEngine | None
        Pre-configured engine instance. If provided, engine_name is ignored.

    Attributes
    ----------
    engine : DataQualityEngine
        The underlying data quality engine.

    Example:
        >>> resource = EngineResource(
        ...     config=EngineResourceConfig(engine_name="truthound"),
        ... )
        >>> with resource:
        ...     result = resource.engine.check(data, rules)
    """

    def __init__(
        self,
        config: Optional["EngineResourceConfig"] = None,
        engine: Optional["DataQualityEngine"] = None,
    ) -> None:
        """Initialize engine resource.

        Args:
            config: Engine configuration.
            engine: Pre-configured engine instance.
        """
        super().__init__(config)
        self._provided_engine = engine
        self._engine: Optional["DataQualityEngine"] = None

    @classmethod
    def _default_config(cls) -> "EngineResourceConfig":
        """Return default engine configuration."""
        return EngineResourceConfig()

    @property
    def engine(self) -> "DataQualityEngine":
        """Get the underlying engine.

        Returns:
            DataQualityEngine: The engine instance.

        Raises:
            RuntimeError: If engine is not initialized.
        """
        if self._engine is None:
            msg = "Engine not initialized. Call setup() first or use as context manager."
            raise RuntimeError(msg)
        return self._engine

    def setup(self, context: Optional[InitResourceContext] = None) -> None:
        """Set up the engine resource.

        Args:
            context: Dagster initialization context.
        """
        if self._provided_engine is not None:
            self._engine = self._provided_engine
        else:
            self._engine = self._create_engine()

        # Start managed engines if configured
        if self.config.auto_start and hasattr(self._engine, "start"):
            self._engine.start()

        self._initialized = True

    def teardown(self, context: Optional[InitResourceContext] = None) -> None:
        """Tear down the engine resource.

        Args:
            context: Dagster initialization context.
        """
        if self._engine is not None and self.config.auto_stop:
            if hasattr(self._engine, "stop"):
                self._engine.stop()

        self._engine = None
        self._initialized = False

    def _create_engine(self) -> "DataQualityEngine":
        """Create engine from configuration.

        Returns:
            DataQualityEngine: Created engine instance.

        Raises:
            ImportError: If engine dependencies are not available.
            ValueError: If engine name is unknown.
        """
        engine_name = self.config.engine_name.lower()

        if engine_name == "truthound":
            return self._create_truthound_engine()
        elif engine_name in ("great_expectations", "ge"):
            return self._create_ge_engine()
        elif engine_name == "pandera":
            return self._create_pandera_engine()
        else:
            # Try to get from registry
            try:
                from common.engines import get_engine

                return get_engine(engine_name)
            except (ImportError, KeyError) as e:
                msg = f"Unknown engine: {engine_name}"
                raise ValueError(msg) from e

    def _create_truthound_engine(self) -> "DataQualityEngine":
        """Create Truthound engine with configuration."""
        try:
            from common.engines import TruthoundEngine, TruthoundEngineConfig

            config = TruthoundEngineConfig(
                auto_start=False,  # We handle lifecycle
                auto_stop=False,
                parallel=self.config.parallel,
                max_workers=self.config.max_workers,
            )
            return TruthoundEngine(config=config)
        except ImportError as e:
            msg = "Truthound is not installed. Install with: pip install truthound"
            raise ImportError(msg) from e

    def _create_ge_engine(self) -> "DataQualityEngine":
        """Create Great Expectations engine."""
        try:
            from common.engines import GreatExpectationsAdapter

            return GreatExpectationsAdapter()
        except ImportError as e:
            msg = "Great Expectations is not installed. Install with: pip install great-expectations"
            raise ImportError(msg) from e

    def _create_pandera_engine(self) -> "DataQualityEngine":
        """Create Pandera engine."""
        try:
            from common.engines import PanderaAdapter

            return PanderaAdapter()
        except ImportError as e:
            msg = "Pandera is not installed. Install with: pip install pandera"
            raise ImportError(msg) from e

    def __enter__(self) -> "EngineResource":
        """Context manager entry."""
        self.setup(None)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.teardown(None)


@dataclass(frozen=True, slots=True)
class DataQualityResourceConfig(EngineResourceConfig):
    """Configuration for data quality resource.

    Extends engine configuration with additional options for
    data quality operations.

    Attributes:
        fail_on_error: Whether to raise on validation failure.
        warning_threshold: Failure rate threshold for warning.
        sample_size: Number of rows to sample for validation.
        store_results: Whether to store results in metadata.
        result_format: Format for result storage.

    Example:
        >>> config = DataQualityResourceConfig(
        ...     engine_name="truthound",
        ...     fail_on_error=True,
        ...     warning_threshold=0.05,
        ... )
    """

    fail_on_error: bool = True
    warning_threshold: Optional[float] = None
    sample_size: Optional[int] = None
    store_results: bool = True
    result_format: str = "dict"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.warning_threshold is not None:
            if not 0 <= self.warning_threshold <= 1:
                msg = "warning_threshold must be between 0 and 1"
                raise ValueError(msg)

        if self.sample_size is not None and self.sample_size < 1:
            msg = "sample_size must be positive"
            raise ValueError(msg)

    def with_fail_on_error(self, fail_on_error: bool) -> "DataQualityResourceConfig":
        """Return new config with updated fail_on_error."""
        return DataQualityResourceConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            engine_name=self.engine_name,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            parallel=self.parallel,
            max_workers=self.max_workers,
            tags=self.tags,
            fail_on_error=fail_on_error,
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            store_results=self.store_results,
            result_format=self.result_format,
        )

    def with_warning_threshold(
        self,
        threshold: Optional[float],
    ) -> "DataQualityResourceConfig":
        """Return new config with updated warning threshold."""
        return DataQualityResourceConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            engine_name=self.engine_name,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            parallel=self.parallel,
            max_workers=self.max_workers,
            tags=self.tags,
            fail_on_error=self.fail_on_error,
            warning_threshold=threshold,
            sample_size=self.sample_size,
            store_results=self.store_results,
            result_format=self.result_format,
        )

    def with_sample_size(self, sample_size: Optional[int]) -> "DataQualityResourceConfig":
        """Return new config with updated sample size."""
        return DataQualityResourceConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            engine_name=self.engine_name,
            auto_start=self.auto_start,
            auto_stop=self.auto_stop,
            parallel=self.parallel,
            max_workers=self.max_workers,
            tags=self.tags,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            sample_size=sample_size,
            store_results=self.store_results,
            result_format=self.result_format,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = EngineResourceConfig.to_dict(self)
        return {
            **base_dict,
            "fail_on_error": self.fail_on_error,
            "warning_threshold": self.warning_threshold,
            "sample_size": self.sample_size,
            "store_results": self.store_results,
            "result_format": self.result_format,
        }


class DataQualityResource(ConfigurableResource):
    """Dagster resource for data quality operations.

    This is the primary resource for integrating data quality checks
    into Dagster pipelines. It provides high-level methods for
    check, profile, and learn operations.

    The resource is designed to work with Dagster's native resource
    system and can be configured using Dagster's configuration system.

    Parameters
    ----------
    engine_name : str
        Name of the data quality engine to use.
        Options: "truthound", "great_expectations", "pandera"

    timeout_seconds : float
        Default timeout for operations in seconds.

    fail_on_error : bool
        Whether to raise exception on validation failure.

    warning_threshold : float | None
        Failure rate threshold for warning instead of failure.

    parallel : bool
        Enable parallel validation (Truthound only).

    max_workers : int | None
        Maximum parallel workers (Truthound only).

    Example:
        >>> from dagster import Definitions, asset
        >>> from truthound_dagster import DataQualityResource
        >>>
        >>> @asset
        ... def users_validated(data_quality: DataQualityResource):
        ...     data = load_users()
        ...     result = data_quality.check(
        ...         data=data,
        ...         rules=[
        ...             {"column": "user_id", "type": "not_null"},
        ...             {"column": "email", "type": "unique"},
        ...         ],
        ...     )
        ...     return {"data": data, "quality": result}
        >>>
        >>> defs = Definitions(
        ...     assets=[users_validated],
        ...     resources={"data_quality": DataQualityResource()},
        ... )
    """

    # Dagster configuration fields
    engine_name: str = "truthound"
    timeout_seconds: float = 300.0
    fail_on_error: bool = True
    warning_threshold: Optional[float] = None
    parallel: bool = False
    max_workers: Optional[int] = None

    # Internal state (not configurable)
    _engine: Optional["DataQualityEngine"] = None

    def setup_for_execution(self, context: InitResourceContext) -> None:
        """Set up the resource for execution.

        Called by Dagster before the resource is used.

        Args:
            context: Dagster initialization context.
        """
        config = DataQualityResourceConfig(
            engine_name=self.engine_name,
            timeout_seconds=self.timeout_seconds,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )

        engine_resource = EngineResource(config=config)
        engine_resource.setup(context)
        self._engine = engine_resource.engine

    def teardown_after_execution(self, context: InitResourceContext) -> None:
        """Tear down the resource after execution.

        Args:
            context: Dagster initialization context.
        """
        if self._engine is not None and hasattr(self._engine, "stop"):
            self._engine.stop()
        self._engine = None

    @property
    def engine(self) -> "DataQualityEngine":
        """Get the underlying engine.

        Returns:
            DataQualityEngine: The engine instance.

        Raises:
            RuntimeError: If engine is not initialized.
        """
        if self._engine is None:
            msg = "Engine not initialized. Resource must be used within Dagster context."
            raise RuntimeError(msg)
        return self._engine

    def check(
        self,
        data: Any,
        rules: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        auto_schema: bool = False,
        fail_on_error: Optional[bool] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> "CheckResult":
        """Execute data quality validation check.

        Args:
            data: Data to validate (DataFrame, path, etc.).
            rules: Validation rules to apply.
            auto_schema: Auto-generate schema from data (Truthound).
            fail_on_error: Override fail_on_error setting.
            timeout: Override timeout setting.
            **kwargs: Additional engine-specific arguments.

        Returns:
            CheckResult: Validation result.

        Raises:
            DataQualityError: If validation fails and fail_on_error is True.
        """
        from truthound_dagster.utils.exceptions import DataQualityError

        actual_fail = fail_on_error if fail_on_error is not None else self.fail_on_error
        actual_timeout = timeout if timeout is not None else self.timeout_seconds

        # Execute check
        result = self.engine.check(
            data,
            rules or [],
            auto_schema=auto_schema,
            timeout=actual_timeout,
            **kwargs,
        )

        # Handle failure
        if not result.is_success and actual_fail:
            # Check warning threshold
            if self.warning_threshold is not None:
                if result.failure_rate <= self.warning_threshold:
                    return result

            raise DataQualityError(
                message=f"Data quality check failed: {result.failed_count} rules failed",
                result=result,
            )

        return result

    def profile(
        self,
        data: Any,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> "ProfileResult":
        """Execute data profiling.

        Args:
            data: Data to profile.
            timeout: Override timeout setting.
            **kwargs: Additional engine-specific arguments.

        Returns:
            ProfileResult: Profiling result.
        """
        actual_timeout = timeout if timeout is not None else self.timeout_seconds

        return self.engine.profile(
            data,
            timeout=actual_timeout,
            **kwargs,
        )

    def learn(
        self,
        data: Any,
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> "LearnResult":
        """Learn validation rules from data.

        Args:
            data: Data to learn from.
            timeout: Override timeout setting.
            **kwargs: Additional engine-specific arguments.

        Returns:
            LearnResult: Learning result with suggested rules.
        """
        actual_timeout = timeout if timeout is not None else self.timeout_seconds

        return self.engine.learn(
            data,
            timeout=actual_timeout,
            **kwargs,
        )


# Preset configurations
DEFAULT_ENGINE_CONFIG = EngineResourceConfig()

PARALLEL_ENGINE_CONFIG = EngineResourceConfig(
    parallel=True,
    max_workers=4,
)

PRODUCTION_ENGINE_CONFIG = EngineResourceConfig(
    parallel=True,
    max_workers=8,
    timeout_seconds=600.0,
)

DEFAULT_DQ_CONFIG = DataQualityResourceConfig()

STRICT_DQ_CONFIG = DataQualityResourceConfig(
    fail_on_error=True,
    warning_threshold=None,
)

LENIENT_DQ_CONFIG = DataQualityResourceConfig(
    fail_on_error=False,
    warning_threshold=0.10,
)
