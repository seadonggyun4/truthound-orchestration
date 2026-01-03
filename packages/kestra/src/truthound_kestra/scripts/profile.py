"""Profile data script for Kestra tasks.

This module provides the main entry point for data profiling
operations in Kestra workflows.

Example Kestra Flow:
    ```yaml
    id: data_profiling
    namespace: production
    tasks:
      - id: profile_users
        type: io.kestra.plugin.scripts.python.Script
        script: |
          from truthound_kestra.scripts.profile import profile_data_script
          profile_data_script(
              input_uri="{{ inputs.data_uri }}",
              include_stats=True,
              include_histograms=False
          )
    ```

Python Usage:
    >>> from truthound_kestra.scripts.profile import (
    ...     profile_data_script,
    ...     ProfileScriptExecutor,
    ... )
    >>>
    >>> result = profile_data_script(data=df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from truthound_kestra.scripts.base import (
    DataQualityEngineProtocol,
    ProfileScriptConfig,
    get_engine,
)
from truthound_kestra.utils.exceptions import EngineError, ScriptError
from truthound_kestra.utils.helpers import (
    Timer,
    create_kestra_output,
    get_execution_context,
    get_logger,
    kestra_outputs,
    load_data,
)
from truthound_kestra.utils.types import (
    CheckStatus,
    ColumnProfile,
    ExecutionContext,
    OperationType,
    ScriptOutput,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "profile_data_script",
    "ProfileScriptExecutor",
    "ProfileScriptResult",
]

logger = get_logger(__name__)


@dataclass
class ProfileScriptResult:
    """Result of a profile data script execution.

    Attributes:
        output: The ScriptOutput containing profile results.
        context: Kestra execution context.
        config: Configuration used for profiling.
        raw_result: Raw result from the engine.
    """

    output: ScriptOutput
    context: ExecutionContext | None = None
    config: ProfileScriptConfig | None = None
    raw_result: Any = None

    @property
    def is_success(self) -> bool:
        """Check if profiling succeeded."""
        return self.output.is_success

    @property
    def columns(self) -> tuple[ColumnProfile, ...]:
        """Get column profiles."""
        return self.output.columns

    @property
    def column_count(self) -> int:
        """Get number of profiled columns."""
        return len(self.output.columns)

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.output.execution_time_ms

    def get_column(self, name: str) -> ColumnProfile | None:
        """Get profile for a specific column."""
        for col in self.columns:
            if col.column_name == name:
                return col
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output": self.output.to_dict(),
            "context": self.context.to_dict() if self.context else None,
            "config": self.config.to_dict() if self.config else None,
        }

    def to_kestra_output(self) -> dict[str, Any]:
        """Convert to Kestra-compatible output."""
        return create_kestra_output(self.output)


@dataclass
class ProfileScriptExecutor:
    """Executor for profile data scripts.

    Attributes:
        config: Profile script configuration.
        engine: Data quality engine instance.
    """

    config: ProfileScriptConfig = field(default_factory=ProfileScriptConfig)
    engine: DataQualityEngineProtocol | None = None

    def __post_init__(self) -> None:
        """Initialize engine if not provided."""
        if self.engine is None:
            self.engine = get_engine(self.config.engine_name)

    def execute(
        self,
        data: Any,
        context: ExecutionContext | None = None,
        **kwargs: Any,
    ) -> ProfileScriptResult:
        """Execute the profile operation.

        Args:
            data: Data to profile (DataFrame or path/URI).
            context: Optional execution context.
            **kwargs: Additional arguments for the engine.

        Returns:
            ProfileScriptResult containing the profile results.

        Raises:
            ScriptError: If script execution fails.
        """
        if not self.config.enabled:
            return self._create_skipped_result(context)

        # Get execution context
        if context is None:
            try:
                context = get_execution_context()
            except Exception:
                context = ExecutionContext(
                    execution_id="local",
                    flow_id="script",
                    namespace="default",
                )

        # Load data if string
        if isinstance(data, str):
            data = load_data(data)

        # Sample data if configured
        if self.config.sample_size > 0 and hasattr(data, "head"):
            data = data.head(self.config.sample_size)

        # Execute profile with timing
        with Timer("profile") as timer:
            try:
                raw_result = self._execute_profile(data, **kwargs)
            except Exception as e:
                return self._handle_engine_error(e, context, timer.elapsed_ms)

        # Convert to ScriptOutput
        output = self._convert_result(
            raw_result,
            timer.elapsed_ms,
            len(data) if hasattr(data, "__len__") else 0,
        )

        result = ProfileScriptResult(
            output=output,
            context=context,
            config=self.config,
            raw_result=raw_result,
        )

        logger.info(
            f"Profile completed: {result.column_count} columns "
            f"({result.execution_time_ms:.2f}ms)"
        )

        return result

    def _execute_profile(self, data: Any, **kwargs: Any) -> Any:
        """Execute the engine profile operation."""
        assert self.engine is not None
        return self.engine.profile(data, **kwargs)

    def _convert_result(
        self,
        raw_result: Any,
        elapsed_ms: float,
        total_rows: int,
    ) -> ScriptOutput:
        """Convert engine result to ScriptOutput."""
        # Extract column profiles
        columns = self._extract_columns(raw_result)

        return ScriptOutput(
            status=CheckStatus.PASSED,
            operation=OperationType.PROFILE,
            total_rows=total_rows,
            execution_time_ms=elapsed_ms,
            columns=tuple(columns),
            metadata={
                "engine": self.config.engine_name,
                "include_stats": self.config.include_stats,
                "include_histograms": self.config.include_histograms,
            },
            timestamp=datetime.now(timezone.utc),
        )

    def _extract_columns(self, raw_result: Any) -> list[ColumnProfile]:
        """Extract ColumnProfile objects from engine result."""
        columns = []
        raw_columns = getattr(raw_result, "columns", [])

        for c in raw_columns:
            if isinstance(c, ColumnProfile):
                columns.append(c)
            elif isinstance(c, dict):
                columns.append(ColumnProfile.from_dict(c))
            elif hasattr(c, "to_dict"):
                columns.append(ColumnProfile.from_dict(c.to_dict()))
            else:
                # Best effort extraction
                columns.append(
                    ColumnProfile(
                        column_name=getattr(c, "column_name", str(c)),
                        dtype=getattr(c, "dtype", "unknown"),
                        null_count=getattr(c, "null_count", 0),
                        null_percentage=getattr(c, "null_percentage", 0.0),
                        unique_count=getattr(c, "unique_count", 0),
                        unique_percentage=getattr(c, "unique_percentage", 0.0),
                    )
                )

        return columns

    def _create_skipped_result(
        self,
        context: ExecutionContext | None,
    ) -> ProfileScriptResult:
        """Create a skipped result when script is disabled."""
        output = ScriptOutput(
            status=CheckStatus.SKIPPED,
            operation=OperationType.PROFILE,
            metadata={"reason": "Script disabled"},
        )
        return ProfileScriptResult(
            output=output,
            context=context,
            config=self.config,
        )

    def _handle_engine_error(
        self,
        error: Exception,
        context: ExecutionContext | None,
        elapsed_ms: float,
    ) -> ProfileScriptResult:
        """Handle engine errors and return error result."""
        logger.error(f"Engine error: {error}")

        output = ScriptOutput(
            status=CheckStatus.ERROR,
            operation=OperationType.PROFILE,
            execution_time_ms=elapsed_ms,
            metadata={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        return ProfileScriptResult(
            output=output,
            context=context,
            config=self.config,
        )


def profile_data_script(
    data: Any | None = None,
    input_uri: str | None = None,
    engine_name: str = "truthound",
    include_stats: bool = True,
    include_histograms: bool = False,
    sample_size: int = 0,
    timeout_seconds: float = 300.0,
    output_to_kestra: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main entry point for Kestra profile data tasks.

    Args:
        data: Data to profile (DataFrame).
        input_uri: URI to load data from.
        engine_name: Name of the engine to use.
        include_stats: Whether to include statistics.
        include_histograms: Whether to include histograms.
        sample_size: Maximum rows to sample (0 = all).
        timeout_seconds: Maximum execution time.
        output_to_kestra: Whether to send outputs to Kestra.
        **kwargs: Additional arguments for the engine.

    Returns:
        Dictionary containing profile results.

    Raises:
        ScriptError: If data loading fails.
    """
    if data is None and input_uri is None:
        raise ScriptError(
            message="Either 'data' or 'input_uri' must be provided",
            script_name="profile_data_script",
        )

    if data is None and input_uri is not None:
        data = load_data(input_uri)

    config = ProfileScriptConfig(
        engine_name=engine_name,
        include_stats=include_stats,
        include_histograms=include_histograms,
        sample_size=sample_size,
        timeout_seconds=timeout_seconds,
    )

    executor = ProfileScriptExecutor(config=config)
    result = executor.execute(data, **kwargs)

    output = result.to_kestra_output()

    if output_to_kestra:
        kestra_outputs(output)

    return output
