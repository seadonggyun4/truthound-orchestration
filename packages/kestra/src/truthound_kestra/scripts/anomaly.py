"""Data Quality Anomaly Detection Script for Kestra.

This module provides the anomaly detection script executor for
identifying anomalous patterns in datasets within Kestra workflows.

Example Kestra Flow:
    ```yaml
    id: data_quality_anomaly
    namespace: production
    tasks:
      - id: detect_anomalies
        type: io.kestra.plugin.scripts.python.Script
        script: |
          from truthound_kestra.scripts.anomaly import anomaly_detection_script
          anomaly_detection_script(
              data_path="/data/dataset.parquet",
              detector="isolation_forest",
              contamination=0.05,
          )
    ```

Python Usage:
    >>> from truthound_kestra.scripts.anomaly import (
    ...     anomaly_detection_script,
    ...     AnomalyScriptExecutor,
    ... )
    >>>
    >>> result = anomaly_detection_script(
    ...     data_path="/data/dataset.parquet",
    ...     detector="isolation_forest",
    ...     contamination=0.05,
    ... )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from truthound_kestra.scripts.base import (
    AnomalyScriptConfig,
    DataQualityEngineProtocol,
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
    from common.base import AnomalyResult

__all__ = [
    "anomaly_detection_script",
    "AnomalyScriptExecutor",
    "AnomalyScriptResult",
]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AnomalyScriptResult:
    """Result container for anomaly detection scripts.

    Attributes:
        status: Anomaly detection status name.
        anomaly_count: Number of columns with anomalies.
        total_columns: Total columns analyzed.
        anomaly_rate: Percentage of anomalous columns.
        execution_time_ms: Execution duration in milliseconds.
        result_dict: Full serialized result dictionary.
        engine_name: Engine used for detection.
        detector: Anomaly detection algorithm used.
    """

    status: str
    anomaly_count: int
    total_columns: int
    anomaly_rate: float
    execution_time_ms: float
    result_dict: dict[str, Any] = field(default_factory=dict)
    engine_name: str = ""
    detector: str = "isolation_forest"

    @property
    def has_anomalies(self) -> bool:
        """Check if anomalies were detected."""
        return self.anomaly_count > 0

    @property
    def summary(self) -> str:
        """Get a human-readable summary."""
        return (
            f"Anomaly Detection: status={self.status}, "
            f"anomalies={self.anomaly_count}/{self.total_columns}, "
            f"anomaly_rate={self.anomaly_rate:.2f}%, "
            f"duration={self.execution_time_ms:.2f}ms"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status,
            "anomaly_count": self.anomaly_count,
            "total_columns": self.total_columns,
            "anomaly_rate": self.anomaly_rate,
            "execution_time_ms": self.execution_time_ms,
            "result_dict": self.result_dict,
            "engine_name": self.engine_name,
            "detector": self.detector,
        }


@dataclass
class AnomalyScriptExecutor:
    """Executor for anomaly detection scripts.

    This class provides a reusable executor for anomaly detection
    that can be configured and invoked multiple times.

    Attributes:
        config: Anomaly script configuration.
        engine: Data quality engine instance.

    Example:
        >>> config = AnomalyScriptConfig(
        ...     detector="isolation_forest",
        ...     contamination=0.05,
        ... )
        >>> executor = AnomalyScriptExecutor(config=config)
        >>> result = executor.execute(data_path="/data/dataset.parquet")
    """

    config: AnomalyScriptConfig = field(default_factory=AnomalyScriptConfig)
    engine: DataQualityEngineProtocol | None = None

    def __post_init__(self) -> None:
        """Initialize engine if not provided."""
        if self.engine is None:
            self.engine = get_engine(self.config.engine_name)

    def execute(
        self,
        data_path: str | None = None,
        data_sql: str | None = None,
        **kwargs: Any,
    ) -> AnomalyScriptResult:
        """Execute anomaly detection.

        Args:
            data_path: Path to data file.
            data_sql: SQL query for data.
            **kwargs: Additional engine kwargs.

        Returns:
            AnomalyScriptResult with detection results.

        Raises:
            ScriptError: If data source not specified.
            EngineError: If engine does not support anomaly detection.
            DataQualityError: If anomalies detected and fail_on_anomaly is True.
        """
        if not self.config.enabled:
            return self._create_skipped_result()

        # Resolve data source from config or parameters
        d_path = data_path or self.config.data_path
        d_sql = data_sql or self.config.data_sql

        # Validate inputs
        if not d_path and not d_sql:
            raise ScriptError(
                message="Must specify either data_path or data_sql",
                script_name="anomaly_detection_script",
            )

        # Verify engine supports anomaly detection
        from common.engines.base import supports_anomaly

        assert self.engine is not None
        if not supports_anomaly(self.engine):
            raise EngineError(
                message=(
                    f"Engine '{self.engine.engine_name}' does not support "
                    f"anomaly detection. Use an engine that implements "
                    f"AnomalyDetectionEngine protocol."
                ),
                engine_name=self.engine.engine_name,
                operation="detect_anomalies",
            )

        # Load data
        with log_operation("load_data", logger, source=d_path or d_sql):
            data = load_data(d_path or d_sql)

        # Build detection kwargs
        detect_kwargs: dict[str, Any] = {
            "detector": self.config.detector,
            "contamination": self.config.contamination,
        }
        if self.config.columns is not None:
            detect_kwargs["columns"] = list(self.config.columns)
        detect_kwargs.update(kwargs)

        # Execute detection with timing
        with Timer("detect_anomalies") as timer:
            try:
                raw_result: AnomalyResult = self.engine.detect_anomalies(
                    data, **detect_kwargs
                )
            except Exception as e:
                return self._handle_engine_error(e, timer.elapsed_ms)

        # Build script result
        result_dict = raw_result.to_dict()
        result_dict["_metadata"] = {
            "engine": self.engine.engine_name,
            "engine_version": self.engine.engine_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detector": self.config.detector,
        }

        script_result = AnomalyScriptResult(
            status=raw_result.status.name,
            anomaly_count=raw_result.anomaly_count,
            total_columns=raw_result.total_columns,
            anomaly_rate=raw_result.anomaly_rate,
            execution_time_ms=timer.elapsed_ms,
            result_dict=result_dict,
            engine_name=self.engine.engine_name,
            detector=self.config.detector,
        )

        # Log result
        self._log_result(script_result)

        # Handle anomalies
        if script_result.has_anomalies and self.config.fail_on_anomaly:
            anomaly_names = [
                a.column
                for a in raw_result.anomalies
                if a.is_anomaly
            ]
            msg = (
                f"Anomalies detected: {raw_result.anomaly_count}/"
                f"{raw_result.total_columns} columns "
                f"({raw_result.anomaly_rate:.2f}%). "
                f"Anomalous: {', '.join(anomaly_names[:3])}"
            )
            if len(anomaly_names) > 3:
                msg += f" ... ({len(anomaly_names) - 3} more)"
            raise DataQualityError(
                message=msg,
                result=script_result.to_dict(),
                metadata={"anomalous_columns": anomaly_names},
            )

        return script_result

    def _create_skipped_result(self) -> AnomalyScriptResult:
        """Create a skipped result when script is disabled."""
        return AnomalyScriptResult(
            status="SKIPPED",
            anomaly_count=0,
            total_columns=0,
            anomaly_rate=0.0,
            execution_time_ms=0.0,
            result_dict={"reason": "Script disabled"},
            engine_name=self.config.engine_name,
            detector=self.config.detector,
        )

    def _handle_engine_error(
        self,
        error: Exception,
        elapsed_ms: float,
    ) -> AnomalyScriptResult:
        """Handle engine errors and return error result."""
        logger.error(f"Engine error during anomaly detection: {error}")

        if self.config.fail_on_anomaly:
            assert self.engine is not None
            raise EngineError(
                message=f"Engine anomaly detection failed: {error}",
                engine_name=self.engine.engine_name,
                operation="detect_anomalies",
                original_error=error,
            ) from error

        return AnomalyScriptResult(
            status="ERROR",
            anomaly_count=0,
            total_columns=0,
            anomaly_rate=0.0,
            execution_time_ms=elapsed_ms,
            result_dict={
                "error": str(error),
                "error_type": type(error).__name__,
            },
            engine_name=self.config.engine_name,
            detector=self.config.detector,
        )

    def _log_result(self, result: AnomalyScriptResult) -> None:
        """Log the anomaly detection result."""
        if result.has_anomalies:
            logger.warning(result.summary)
        else:
            logger.info(result.summary)


def anomaly_detection_script(
    data_path: str | None = None,
    data_sql: str | None = None,
    engine_name: str = "truthound",
    detector: str = "isolation_forest",
    columns: list[str] | None = None,
    contamination: float = 0.05,
    fail_on_anomaly: bool = True,
    timeout_seconds: float = 300.0,
    output_to_kestra: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main entry point for Kestra anomaly detection tasks.

    This function provides a simple interface for executing anomaly
    detection from Kestra Python script tasks.

    Args:
        data_path: Path to data file.
        data_sql: SQL query for data.
        engine_name: Name of the engine to use.
        detector: Anomaly detection algorithm.
        columns: Columns to check for anomalies.
        contamination: Expected proportion of anomalies.
        fail_on_anomaly: Whether to raise on anomaly detection.
        timeout_seconds: Maximum execution time.
        output_to_kestra: Whether to send outputs to Kestra.
        **kwargs: Additional engine kwargs.

    Returns:
        Dictionary containing anomaly detection results.

    Raises:
        ScriptError: If data source not specified.
        DataQualityError: If anomalies detected and fail_on_anomaly is True.

    Example:
        >>> result = anomaly_detection_script(
        ...     data_path="/data/dataset.parquet",
        ...     detector="isolation_forest",
        ...     contamination=0.05,
        ... )
    """
    # Validate inputs
    if not data_path and not data_sql:
        raise ScriptError(
            message="Either 'data_path' or 'data_sql' must be provided",
            script_name="anomaly_detection_script",
        )

    # Create configuration
    config = AnomalyScriptConfig(
        engine_name=engine_name,
        detector=detector,
        columns=tuple(columns) if columns else None,
        contamination=contamination,
        fail_on_anomaly=fail_on_anomaly,
        timeout_seconds=timeout_seconds,
        data_path=data_path,
        data_sql=data_sql,
    )

    # Create executor and execute
    executor = AnomalyScriptExecutor(config=config)
    result = executor.execute(
        data_path=data_path,
        data_sql=data_sql,
        **kwargs,
    )

    # Create output
    output = result.to_dict()

    # Send to Kestra if configured
    if output_to_kestra:
        kestra_outputs(output)

    return output
