"""Data Quality Drift Detection Script for Kestra.

This module provides the drift detection script executor for
comparing baseline and current datasets in Kestra workflows.

Example Kestra Flow:
    ```yaml
    id: data_quality_drift
    namespace: production
    tasks:
      - id: detect_drift
        type: io.kestra.plugin.scripts.python.Script
        script: |
          from truthound_kestra.scripts.drift import drift_detection_script
          drift_detection_script(
              baseline_data_path="/data/baseline.parquet",
              current_data_path="/data/current.parquet",
              method="ks",
              threshold=0.05,
          )
    ```

Python Usage:
    >>> from truthound_kestra.scripts.drift import (
    ...     drift_detection_script,
    ...     DriftScriptExecutor,
    ... )
    >>>
    >>> result = drift_detection_script(
    ...     baseline_data_path="/data/baseline.parquet",
    ...     current_data_path="/data/current.parquet",
    ...     method="ks",
    ...     threshold=0.05,
    ... )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from truthound_kestra.scripts.base import (
    DataQualityEngineProtocol,
    DriftScriptConfig,
    get_engine,
)
from truthound_kestra.utils.exceptions import (
    DataQualityError,
    EngineError,
    ScriptError,
)
from truthound_kestra.utils.helpers import (
    Timer,
    create_kestra_output,
    get_logger,
    kestra_outputs,
    load_data,
    log_operation,
)
from truthound_kestra.utils.serialization import serialize_result

if TYPE_CHECKING:
    from common.base import DriftResult

__all__ = [
    "drift_detection_script",
    "DriftScriptExecutor",
    "DriftScriptResult",
]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DriftScriptResult:
    """Result container for drift detection scripts.

    Attributes:
        status: Drift detection status name.
        drifted_count: Number of drifted columns.
        total_columns: Total columns analyzed.
        drift_rate: Percentage of drifted columns.
        execution_time_ms: Execution duration in milliseconds.
        result_dict: Full serialized result dictionary.
        engine_name: Engine used for detection.
        method: Statistical method used.
    """

    status: str
    drifted_count: int
    total_columns: int
    drift_rate: float
    execution_time_ms: float
    result_dict: dict[str, Any] = field(default_factory=dict)
    engine_name: str = ""
    method: str = "auto"

    @property
    def is_drifted(self) -> bool:
        """Check if drift was detected."""
        return self.drifted_count > 0

    @property
    def summary(self) -> str:
        """Get a human-readable summary."""
        return (
            f"Drift Detection: status={self.status}, "
            f"drifted={self.drifted_count}/{self.total_columns}, "
            f"drift_rate={self.drift_rate:.2f}%, "
            f"duration={self.execution_time_ms:.2f}ms"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status,
            "drifted_count": self.drifted_count,
            "total_columns": self.total_columns,
            "drift_rate": self.drift_rate,
            "execution_time_ms": self.execution_time_ms,
            "result_dict": self.result_dict,
            "engine_name": self.engine_name,
            "method": self.method,
        }


@dataclass
class DriftScriptExecutor:
    """Executor for drift detection scripts.

    This class provides a reusable executor for drift detection
    that can be configured and invoked multiple times.

    Attributes:
        config: Drift script configuration.
        engine: Data quality engine instance.

    Example:
        >>> config = DriftScriptConfig(method="ks", threshold=0.05)
        >>> executor = DriftScriptExecutor(config=config)
        >>> result = executor.execute(
        ...     baseline_data_path="/data/baseline.parquet",
        ...     current_data_path="/data/current.parquet",
        ... )
    """

    config: DriftScriptConfig = field(default_factory=DriftScriptConfig)
    engine: DataQualityEngineProtocol | None = None

    def __post_init__(self) -> None:
        """Initialize engine if not provided."""
        if self.engine is None:
            self.engine = get_engine(self.config.engine_name)

    def execute(
        self,
        baseline_data_path: str | None = None,
        current_data_path: str | None = None,
        baseline_sql: str | None = None,
        current_sql: str | None = None,
        **kwargs: Any,
    ) -> DriftScriptResult:
        """Execute drift detection.

        Args:
            baseline_data_path: Path to baseline data file.
            current_data_path: Path to current data file.
            baseline_sql: SQL query for baseline data.
            current_sql: SQL query for current data.
            **kwargs: Additional engine kwargs.

        Returns:
            DriftScriptResult with detection results.

        Raises:
            ScriptError: If data sources not specified.
            EngineError: If engine does not support drift detection.
            DataQualityError: If drift detected and fail_on_drift is True.
        """
        if not self.config.enabled:
            return self._create_skipped_result()

        # Resolve data paths from config or parameters
        baseline_path = baseline_data_path or self.config.baseline_data_path
        current_path = current_data_path or self.config.current_data_path
        b_sql = baseline_sql or self.config.baseline_sql
        c_sql = current_sql or self.config.current_sql

        # Validate inputs
        if not baseline_path and not b_sql:
            raise ScriptError(
                message="Must specify either baseline_data_path or baseline_sql",
                script_name="drift_detection_script",
            )
        if not current_path and not c_sql:
            raise ScriptError(
                message="Must specify either current_data_path or current_sql",
                script_name="drift_detection_script",
            )

        # Verify engine supports drift detection
        from common.engines.base import supports_drift

        assert self.engine is not None
        if not supports_drift(self.engine):
            raise EngineError(
                message=(
                    f"Engine '{self.engine.engine_name}' does not support "
                    f"drift detection. Use an engine that implements "
                    f"DriftDetectionEngine protocol."
                ),
                engine_name=self.engine.engine_name,
                operation="detect_drift",
            )

        # Load data
        with log_operation("load_baseline_data", logger, source=baseline_path or b_sql):
            baseline = load_data(baseline_path or b_sql)

        with log_operation("load_current_data", logger, source=current_path or c_sql):
            current = load_data(current_path or c_sql)

        # Build detection kwargs
        detect_kwargs: dict[str, Any] = {"method": self.config.method}
        if self.config.columns is not None:
            detect_kwargs["columns"] = list(self.config.columns)
        if self.config.threshold is not None:
            detect_kwargs["threshold"] = self.config.threshold
        detect_kwargs.update(kwargs)

        # Execute detection with timing
        with Timer("detect_drift") as timer:
            try:
                raw_result: DriftResult = self.engine.detect_drift(
                    baseline, current, **detect_kwargs
                )
            except Exception as e:
                return self._handle_engine_error(e, timer.elapsed_ms)

        # Build script result
        result_dict = raw_result.to_dict()
        result_dict["_metadata"] = {
            "engine": self.engine.engine_name,
            "engine_version": self.engine.engine_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": self.config.method,
        }

        script_result = DriftScriptResult(
            status=raw_result.status.name,
            drifted_count=raw_result.drifted_count,
            total_columns=raw_result.total_columns,
            drift_rate=raw_result.drift_rate,
            execution_time_ms=timer.elapsed_ms,
            result_dict=result_dict,
            engine_name=self.engine.engine_name,
            method=self.config.method,
        )

        # Log result
        self._log_result(script_result)

        # Handle drift
        if script_result.is_drifted and self.config.fail_on_drift:
            drifted_names = [
                c.column
                for c in raw_result.drifted_columns
                if c.is_drifted
            ]
            msg = (
                f"Drift detected: {raw_result.drifted_count}/"
                f"{raw_result.total_columns} columns "
                f"({raw_result.drift_rate:.2f}%). "
                f"Drifted: {', '.join(drifted_names[:3])}"
            )
            if len(drifted_names) > 3:
                msg += f" ... ({len(drifted_names) - 3} more)"
            raise DataQualityError(
                message=msg,
                result=script_result.to_dict(),
                metadata={"drifted_columns": drifted_names},
            )

        return script_result

    def _create_skipped_result(self) -> DriftScriptResult:
        """Create a skipped result when script is disabled."""
        return DriftScriptResult(
            status="SKIPPED",
            drifted_count=0,
            total_columns=0,
            drift_rate=0.0,
            execution_time_ms=0.0,
            result_dict={"reason": "Script disabled"},
            engine_name=self.config.engine_name,
            method=self.config.method,
        )

    def _handle_engine_error(
        self,
        error: Exception,
        elapsed_ms: float,
    ) -> DriftScriptResult:
        """Handle engine errors and return error result."""
        logger.error(f"Engine error during drift detection: {error}")

        if self.config.fail_on_drift:
            assert self.engine is not None
            raise EngineError(
                message=f"Engine drift detection failed: {error}",
                engine_name=self.engine.engine_name,
                operation="detect_drift",
                original_error=error,
            ) from error

        return DriftScriptResult(
            status="ERROR",
            drifted_count=0,
            total_columns=0,
            drift_rate=0.0,
            execution_time_ms=elapsed_ms,
            result_dict={
                "error": str(error),
                "error_type": type(error).__name__,
            },
            engine_name=self.config.engine_name,
            method=self.config.method,
        )

    def _log_result(self, result: DriftScriptResult) -> None:
        """Log the drift detection result."""
        if result.is_drifted:
            logger.warning(result.summary)
        else:
            logger.info(result.summary)


def drift_detection_script(
    baseline_data_path: str | None = None,
    current_data_path: str | None = None,
    baseline_sql: str | None = None,
    current_sql: str | None = None,
    engine_name: str = "truthound",
    method: str = "auto",
    columns: list[str] | None = None,
    threshold: float | None = None,
    fail_on_drift: bool = True,
    timeout_seconds: float = 300.0,
    output_to_kestra: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main entry point for Kestra drift detection tasks.

    This function provides a simple interface for executing drift
    detection from Kestra Python script tasks.

    Args:
        baseline_data_path: Path to baseline data file.
        current_data_path: Path to current data file.
        baseline_sql: SQL query for baseline data.
        current_sql: SQL query for current data.
        engine_name: Name of the engine to use.
        method: Statistical method for drift detection.
        columns: Columns to check for drift.
        threshold: Detection threshold.
        fail_on_drift: Whether to raise on drift detection.
        timeout_seconds: Maximum execution time.
        output_to_kestra: Whether to send outputs to Kestra.
        **kwargs: Additional engine kwargs.

    Returns:
        Dictionary containing drift detection results.

    Raises:
        ScriptError: If data sources not specified.
        DataQualityError: If drift detected and fail_on_drift is True.

    Example:
        >>> result = drift_detection_script(
        ...     baseline_data_path="/data/baseline.parquet",
        ...     current_data_path="/data/current.parquet",
        ...     method="ks",
        ...     threshold=0.05,
        ... )
    """
    # Validate inputs
    if not baseline_data_path and not baseline_sql:
        raise ScriptError(
            message="Either 'baseline_data_path' or 'baseline_sql' must be provided",
            script_name="drift_detection_script",
        )
    if not current_data_path and not current_sql:
        raise ScriptError(
            message="Either 'current_data_path' or 'current_sql' must be provided",
            script_name="drift_detection_script",
        )

    # Create configuration
    config = DriftScriptConfig(
        engine_name=engine_name,
        method=method,
        columns=tuple(columns) if columns else None,
        threshold=threshold,
        fail_on_drift=fail_on_drift,
        timeout_seconds=timeout_seconds,
        baseline_data_path=baseline_data_path,
        current_data_path=current_data_path,
        baseline_sql=baseline_sql,
        current_sql=current_sql,
    )

    # Create executor and execute
    executor = DriftScriptExecutor(config=config)
    result = executor.execute(
        baseline_data_path=baseline_data_path,
        current_data_path=current_data_path,
        baseline_sql=baseline_sql,
        current_sql=current_sql,
        **kwargs,
    )

    # Create output
    output = result.to_dict()

    # Send to Kestra if configured
    if output_to_kestra:
        kestra_outputs(output)

    return output
