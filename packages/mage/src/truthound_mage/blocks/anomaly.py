"""Data Quality Anomaly Detection Block for Mage AI.

This module provides the AnomalyTransformer block for detecting
anomalous patterns in datasets within Mage AI pipelines.

Example:
    >>> from truthound_mage.blocks.anomaly import AnomalyTransformer
    >>>
    >>> transformer = AnomalyTransformer(detector="isolation_forest")
    >>> result = transformer.execute(data)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from common.base import AnomalyResult
    from common.engines.base import DataQualityEngine

from truthound_mage.blocks.base import (
    AnomalyBlockConfig,
    BlockConfig,
    BlockExecutionContext,
    BlockResult,
)
from truthound_mage.blocks.transformer import BaseDataQualityTransformer


# =============================================================================
# Anomaly Transformer
# =============================================================================


class AnomalyTransformer(BaseDataQualityTransformer):
    """Anomaly detection transformer block.

    Extends BaseDataQualityTransformer to detect anomalous patterns
    in a single dataset using the configured engine's anomaly detection.

    Attributes:
        config: AnomalyBlockConfig with anomaly detection settings.

    Example:
        >>> config = AnomalyBlockConfig(
        ...     detector="isolation_forest",
        ...     contamination=0.05,
        ... )
        >>> transformer = AnomalyTransformer(config=config)
        >>> result = transformer.execute(df)
        >>> if result.success:
        ...     print(f"Status: {result.result_dict['status']}")
    """

    def __init__(
        self,
        config: AnomalyBlockConfig | None = None,
        engine: DataQualityEngine | None = None,
        hooks: Sequence[Any] | None = None,
        engine_name: str | None = None,
        detector: str | None = None,
        columns: list[str] | None = None,
        contamination: float | None = None,
        fail_on_anomaly: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize anomaly transformer.

        Args:
            config: Anomaly block configuration. Built from kwargs if None.
            engine: DataQualityEngine instance. Uses registry if None.
            hooks: Lifecycle hooks to invoke during execution.
            engine_name: Name of the engine to use from registry.
            detector: Anomaly detection algorithm.
            columns: Columns to check for anomalies.
            contamination: Expected proportion of anomalies.
            fail_on_anomaly: Whether to fail on anomaly detection.
            **kwargs: Additional configuration options.
        """
        if config is None:
            config_kwargs: dict[str, Any] = {}
            if engine_name is not None:
                config_kwargs["engine_name"] = engine_name
            if detector is not None:
                config_kwargs["detector"] = detector
            if columns is not None:
                config_kwargs["columns"] = tuple(columns)
            if contamination is not None:
                config_kwargs["contamination"] = contamination
            if fail_on_anomaly is not None:
                config_kwargs["fail_on_anomaly"] = fail_on_anomaly
            config = AnomalyBlockConfig(**config_kwargs) if config_kwargs else None

        super().__init__(
            config=config or AnomalyBlockConfig(),
            engine=engine,
            hooks=hooks,
        )

    @property
    def config(self) -> AnomalyBlockConfig:
        """Get typed config."""
        return self._config  # type: ignore[return-value]

    @config.setter
    def config(self, value: BlockConfig) -> None:
        """Set config with type validation."""
        if not isinstance(value, AnomalyBlockConfig):
            value = AnomalyBlockConfig.from_dict(value.to_dict())
        self._config = value

    @property
    def anomaly_config(self) -> AnomalyBlockConfig:
        """Get typed anomaly config (alias for config)."""
        return self.config

    def _execute_operation(
        self,
        data: Any,
        context: BlockExecutionContext,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Execute anomaly detection on data.

        Args:
            data: Input data to analyze.
            context: Block execution context.
            **kwargs: Additional anomaly detection arguments.

        Returns:
            AnomalyResult with anomaly detection results.

        Raises:
            ValueError: If engine does not support anomaly detection.
        """
        from common.engines.base import supports_anomaly

        if not supports_anomaly(self.engine):
            raise ValueError(
                f"Engine '{self.engine.engine_name}' does not support anomaly detection."
            )

        detect_kwargs: dict[str, Any] = {
            "detector": self.anomaly_config.detector,
            "contamination": self.anomaly_config.contamination,
        }
        if self.anomaly_config.columns is not None:
            detect_kwargs["columns"] = list(self.anomaly_config.columns)
        detect_kwargs.update(kwargs)

        return self.engine.detect_anomalies(data, **detect_kwargs)

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        """Serialize anomaly result.

        Args:
            result: AnomalyResult to serialize.

        Returns:
            Serialized result dictionary.
        """
        result_dict = result.to_dict()
        result_dict["_metadata"] = {
            "engine": self.engine.engine_name,
            "engine_version": self.engine.engine_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return result_dict

    def _handle_result(
        self,
        result: Any,
        block_result: BlockResult,
        context: BlockExecutionContext,
    ) -> None:
        """Handle anomaly result, raising on anomaly if configured.

        Args:
            result: The anomaly result.
            block_result: The wrapped block result.
            context: Block execution context.

        Raises:
            BlockExecutionError: If anomalies detected and fail_on_anomaly is True.
        """
        if result.has_anomalies and self.anomaly_config.fail_on_anomaly:
            from truthound_mage.utils.exceptions import BlockExecutionError

            anomaly_names = [a.column for a in result.anomalies if a.is_anomaly]
            msg = (
                f"Anomalies detected: {result.anomaly_count}/{result.total_columns} columns "
                f"({result.anomaly_rate:.2f}%). "
                f"Anomalous: {', '.join(anomaly_names[:3])}"
            )
            if len(anomaly_names) > 3:
                msg += f" ... ({len(anomaly_names) - 3} more)"

            raise BlockExecutionError(
                msg,
                block_uuid=context.block_uuid,
            )

    def _log_result(self, result: BlockResult) -> None:
        """Log anomaly detection result.

        Args:
            result: The block result to log.
        """
        from common import get_logger

        logger = get_logger(__name__)
        logger.info(
            "Anomaly detection completed",
            engine=self.engine.engine_name,
            execution_time_ms=result.execution_time_ms,
            success=result.success,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_anomaly_transformer(
    engine_name: str | None = None,
    detector: str = "isolation_forest",
    contamination: float = 0.05,
    fail_on_anomaly: bool = True,
    columns: list[str] | None = None,
    **kwargs: Any,
) -> AnomalyTransformer:
    """Create an anomaly transformer with the given configuration.

    Args:
        engine_name: Name of engine to use.
        detector: Anomaly detection algorithm.
        contamination: Expected proportion of anomalies.
        fail_on_anomaly: Whether to fail on anomaly detection.
        columns: Columns to check for anomalies.
        **kwargs: Additional configuration options.

    Returns:
        Configured AnomalyTransformer instance.

    Example:
        >>> transformer = create_anomaly_transformer(
        ...     detector="isolation_forest",
        ...     contamination=0.05,
        ... )
    """
    config = AnomalyBlockConfig(
        engine_name=engine_name,
        detector=detector,
        contamination=contamination,
        fail_on_anomaly=fail_on_anomaly,
        columns=tuple(columns) if columns else None,
        **kwargs,
    )
    return AnomalyTransformer(config=config)
