"""Base block abstractions for data quality operations in Mage.

This module provides abstract base classes and configuration types for data quality
blocks in Mage AI. The design follows these principles:

1. Engine-Agnostic: Works with any DataQualityEngine implementation
2. Protocol-First: Uses structural typing for maximum flexibility
3. Immutable Configs: Thread-safe frozen dataclasses for configuration
4. Extensible: Easy to extend with custom blocks and behaviors

Architecture:
    BlockConfig (frozen dataclass)
        ├── CheckBlockConfig
        ├── ProfileBlockConfig
        ├── LearnBlockConfig
        ├── DriftBlockConfig
        └── AnomalyBlockConfig

Example:
    >>> config = CheckBlockConfig(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     fail_on_error=True,
    ... )
    >>> transformer = DataQualityTransformer(config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from common.base import CheckResult, LearnResult, ProfileResult
    from common.engines.base import DataQualityEngine


# =============================================================================
# Enums
# =============================================================================


class BlockType(str, Enum):
    """Type of data quality block."""

    TRANSFORMER = "transformer"
    SENSOR = "sensor"
    CONDITION = "condition"
    DATA_LOADER = "data_loader"
    DATA_EXPORTER = "data_exporter"


class ExecutionMode(str, Enum):
    """Block execution mode."""

    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class BlockHook(Protocol):
    """Protocol for block lifecycle hooks."""

    def on_block_start(
        self,
        block_name: str,
        config: BlockConfig,
        context: BlockExecutionContext,
    ) -> None:
        """Called when block execution starts."""
        ...

    def on_block_success(
        self,
        block_name: str,
        result: BlockResult,
        context: BlockExecutionContext,
    ) -> None:
        """Called when block execution succeeds."""
        ...

    def on_block_error(
        self,
        block_name: str,
        error: Exception,
        context: BlockExecutionContext,
    ) -> None:
        """Called when block execution fails."""
        ...


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class BlockConfig:
    """Base configuration for data quality blocks.

    This immutable configuration holds common settings shared across
    all data quality blocks.

    Attributes:
        engine_name: Name of the data quality engine to use.
        fail_on_error: Whether to raise exception on validation failure.
        timeout_seconds: Operation timeout in seconds.
        tags: Metadata tags for the operation.
        output_key: Key for storing results in block outputs.
        log_results: Whether to log operation results.
        extra: Additional block-specific options.

    Example:
        >>> config = BlockConfig(
        ...     engine_name="truthound",
        ...     fail_on_error=True,
        ...     timeout_seconds=300,
        ... )
    """

    engine_name: str | None = None
    fail_on_error: bool = True
    timeout_seconds: int = 300
    output_key: str = "data_quality_result"
    log_results: bool = True
    tags: frozenset[str] = field(default_factory=frozenset)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.timeout_seconds <= 0:
            msg = "timeout_seconds must be positive"
            raise ValueError(msg)

    def with_engine_name(self, engine_name: str) -> BlockConfig:
        """Return new config with updated engine_name."""
        return BlockConfig(
            engine_name=engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> BlockConfig:
        """Return new config with updated fail_on_error."""
        return BlockConfig(
            engine_name=self.engine_name,
            fail_on_error=fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
        )

    def with_timeout(self, timeout_seconds: int) -> BlockConfig:
        """Return new config with updated timeout."""
        return BlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
        )

    def with_tags(self, tags: frozenset[str] | set[str]) -> BlockConfig:
        """Return new config with updated tags."""
        return BlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=frozenset(tags),
            extra=self.extra,
        )

    def with_extra(self, **kwargs: Any) -> BlockConfig:
        """Return new config with updated extra options."""
        new_extra = dict(self.extra)
        new_extra.update(kwargs)
        return BlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=new_extra,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "engine_name": self.engine_name,
            "fail_on_error": self.fail_on_error,
            "timeout_seconds": self.timeout_seconds,
            "output_key": self.output_key,
            "log_results": self.log_results,
            "tags": list(self.tags),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockConfig:
        """Create configuration from dictionary."""
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
        )


@dataclass(frozen=True, slots=True)
class CheckBlockConfig(BlockConfig):
    """Configuration specific to check operations.

    Attributes:
        rules: Validation rules to apply.
        warning_threshold: Failure rate threshold for warning (0.0-1.0).
        sample_size: Number of rows to sample (None=all).
        parallel: Whether to run checks in parallel.
        auto_schema: Whether to auto-generate schema from data.
        min_severity: Minimum severity to report (for Truthound).

    Example:
        >>> config = CheckBlockConfig(
        ...     rules=(
        ...         {"column": "id", "type": "not_null"},
        ...         {"column": "email", "type": "regex", "pattern": r".*@.*"},
        ...     ),
        ...     warning_threshold=0.05,
        ... )
    """

    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    warning_threshold: float | None = None
    sample_size: int | None = None
    parallel: bool = True
    auto_schema: bool = False
    min_severity: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        BlockConfig.__post_init__(self)
        if self.warning_threshold is not None:
            if not 0 <= self.warning_threshold <= 1:
                msg = "warning_threshold must be between 0 and 1"
                raise ValueError(msg)
        if self.sample_size is not None and self.sample_size <= 0:
            msg = "sample_size must be positive"
            raise ValueError(msg)
        if self.min_severity is not None:
            valid_severities = {"critical", "high", "medium", "low"}
            if self.min_severity.lower() not in valid_severities:
                msg = f"min_severity must be one of {valid_severities}"
                raise ValueError(msg)

    def with_rules(self, rules: Sequence[dict[str, Any]]) -> CheckBlockConfig:
        """Return new config with updated rules."""
        return CheckBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            rules=tuple(rules),
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            parallel=self.parallel,
            auto_schema=self.auto_schema,
            min_severity=self.min_severity,
        )

    def with_warning_threshold(self, threshold: float) -> CheckBlockConfig:
        """Return new config with updated warning_threshold."""
        return CheckBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            rules=self.rules,
            warning_threshold=threshold,
            sample_size=self.sample_size,
            parallel=self.parallel,
            auto_schema=self.auto_schema,
            min_severity=self.min_severity,
        )

    def with_sample_size(self, sample_size: int | None) -> CheckBlockConfig:
        """Return new config with updated sample_size."""
        return CheckBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            rules=self.rules,
            warning_threshold=self.warning_threshold,
            sample_size=sample_size,
            parallel=self.parallel,
            auto_schema=self.auto_schema,
            min_severity=self.min_severity,
        )

    def with_parallel(self, parallel: bool) -> CheckBlockConfig:
        """Return new config with updated parallel setting."""
        return CheckBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            rules=self.rules,
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            parallel=parallel,
            auto_schema=self.auto_schema,
            min_severity=self.min_severity,
        )

    def with_auto_schema(self, auto_schema: bool) -> CheckBlockConfig:
        """Return new config with updated auto_schema setting."""
        return CheckBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            rules=self.rules,
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            parallel=self.parallel,
            auto_schema=auto_schema,
            min_severity=self.min_severity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        base = BlockConfig.to_dict(self)
        base.update({
            "rules": list(self.rules),
            "warning_threshold": self.warning_threshold,
            "sample_size": self.sample_size,
            "parallel": self.parallel,
            "auto_schema": self.auto_schema,
            "min_severity": self.min_severity,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckBlockConfig:
        """Create configuration from dictionary."""
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
            rules=tuple(data.get("rules", [])),
            warning_threshold=data.get("warning_threshold"),
            sample_size=data.get("sample_size"),
            parallel=data.get("parallel", True),
            auto_schema=data.get("auto_schema", False),
            min_severity=data.get("min_severity"),
        )


@dataclass(frozen=True, slots=True)
class ProfileBlockConfig(BlockConfig):
    """Configuration specific to profile operations.

    Attributes:
        columns: Columns to profile (None=all).
        include_statistics: Whether to include statistical analysis.
        include_patterns: Whether to detect data patterns.
        include_distributions: Whether to analyze distributions.
        sample_size: Number of rows to sample.

    Example:
        >>> config = ProfileBlockConfig(
        ...     columns=frozenset(["amount", "quantity"]),
        ...     include_statistics=True,
        ...     include_distributions=True,
        ... )
    """

    columns: frozenset[str] | None = None
    include_statistics: bool = True
    include_patterns: bool = True
    include_distributions: bool = True
    sample_size: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        BlockConfig.__post_init__(self)
        if self.sample_size is not None and self.sample_size <= 0:
            msg = "sample_size must be positive"
            raise ValueError(msg)

    def with_columns(
        self,
        columns: frozenset[str] | set[str] | Sequence[str] | None,
    ) -> ProfileBlockConfig:
        """Return new config with updated columns."""
        cols = frozenset(columns) if columns else None
        return ProfileBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            columns=cols,
            include_statistics=self.include_statistics,
            include_patterns=self.include_patterns,
            include_distributions=self.include_distributions,
            sample_size=self.sample_size,
        )

    def with_statistics(self, include: bool) -> ProfileBlockConfig:
        """Return new config with updated include_statistics."""
        return ProfileBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            columns=self.columns,
            include_statistics=include,
            include_patterns=self.include_patterns,
            include_distributions=self.include_distributions,
            sample_size=self.sample_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        base = BlockConfig.to_dict(self)
        base.update({
            "columns": list(self.columns) if self.columns else None,
            "include_statistics": self.include_statistics,
            "include_patterns": self.include_patterns,
            "include_distributions": self.include_distributions,
            "sample_size": self.sample_size,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileBlockConfig:
        """Create configuration from dictionary."""
        columns = data.get("columns")
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
            columns=frozenset(columns) if columns else None,
            include_statistics=data.get("include_statistics", True),
            include_patterns=data.get("include_patterns", True),
            include_distributions=data.get("include_distributions", True),
            sample_size=data.get("sample_size"),
        )


@dataclass(frozen=True, slots=True)
class LearnBlockConfig(BlockConfig):
    """Configuration specific to learn operations.

    Attributes:
        output_path: Path to save learned schema.
        strictness: Learning strictness level.
        infer_constraints: Whether to infer value constraints.
        categorical_threshold: Max unique values for categorical detection.

    Example:
        >>> config = LearnBlockConfig(
        ...     output_path="s3://bucket/schemas/users.json",
        ...     strictness="moderate",
        ... )
    """

    output_path: str | None = None
    strictness: str = "moderate"
    infer_constraints: bool = True
    categorical_threshold: int = 20

    def __post_init__(self) -> None:
        """Validate configuration values."""
        BlockConfig.__post_init__(self)
        valid_strictness = {"strict", "moderate", "lenient"}
        if self.strictness not in valid_strictness:
            msg = f"strictness must be one of {valid_strictness}"
            raise ValueError(msg)
        if self.categorical_threshold <= 0:
            msg = "categorical_threshold must be positive"
            raise ValueError(msg)

    def with_output_path(self, output_path: str | None) -> LearnBlockConfig:
        """Return new config with updated output_path."""
        return LearnBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            output_path=output_path,
            strictness=self.strictness,
            infer_constraints=self.infer_constraints,
            categorical_threshold=self.categorical_threshold,
        )

    def with_strictness(self, strictness: str) -> LearnBlockConfig:
        """Return new config with updated strictness."""
        return LearnBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            output_path=self.output_path,
            strictness=strictness,
            infer_constraints=self.infer_constraints,
            categorical_threshold=self.categorical_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        base = BlockConfig.to_dict(self)
        base.update({
            "output_path": self.output_path,
            "strictness": self.strictness,
            "infer_constraints": self.infer_constraints,
            "categorical_threshold": self.categorical_threshold,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnBlockConfig:
        """Create configuration from dictionary."""
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
            output_path=data.get("output_path"),
            strictness=data.get("strictness", "moderate"),
            infer_constraints=data.get("infer_constraints", True),
            categorical_threshold=data.get("categorical_threshold", 20),
        )


# =============================================================================
# Execution Context
# =============================================================================


@dataclass(frozen=True, slots=True)
class BlockExecutionContext:
    """Context information for block execution.

    Attributes:
        block_uuid: Unique identifier for the block.
        pipeline_uuid: Unique identifier for the pipeline.
        partition: Data partition identifier.
        execution_date: Execution timestamp.
        upstream_outputs: Outputs from upstream blocks.
        variables: Pipeline variables.
        run_id: Current run identifier.

    Example:
        >>> context = BlockExecutionContext(
        ...     block_uuid="check_quality_1",
        ...     pipeline_uuid="data_pipeline",
        ...     execution_date=datetime.now(timezone.utc),
        ... )
    """

    block_uuid: str = ""
    pipeline_uuid: str = ""
    partition: str | None = None
    execution_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    upstream_outputs: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the context."""
        return self.variables.get(key, default)

    def get_upstream_output(self, block_uuid: str, default: Any = None) -> Any:
        """Get output from an upstream block."""
        return self.upstream_outputs.get(block_uuid, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "block_uuid": self.block_uuid,
            "pipeline_uuid": self.pipeline_uuid,
            "partition": self.partition,
            "execution_date": self.execution_date.isoformat(),
            "run_id": self.run_id,
        }

    @classmethod
    def from_mage_context(
        cls,
        block_uuid: str = "",
        pipeline_uuid: str = "",
        partition: str | None = None,
        execution_date: datetime | None = None,
        **kwargs: Any,
    ) -> BlockExecutionContext:
        """Create context from Mage execution parameters.

        This factory method adapts Mage's execution context to our internal format.

        Args:
            block_uuid: Block UUID from Mage.
            pipeline_uuid: Pipeline UUID from Mage.
            partition: Data partition from Mage.
            execution_date: Execution date from Mage.
            **kwargs: Additional context values.

        Returns:
            BlockExecutionContext instance.
        """
        return cls(
            block_uuid=block_uuid,
            pipeline_uuid=pipeline_uuid,
            partition=partition,
            execution_date=execution_date or datetime.now(timezone.utc),
            upstream_outputs=kwargs.get("upstream_outputs", {}),
            variables=kwargs.get("variables", {}),
            run_id=kwargs.get("run_id"),
        )


# =============================================================================
# Block Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class BlockResult:
    """Result of a data quality block execution.

    Attributes:
        success: Whether the operation succeeded.
        result: The raw result object (CheckResult, ProfileResult, LearnResult).
        data: The processed data (may be same as input for check operations).
        result_dict: Serialized result dictionary.
        metadata: Additional metadata about the execution.
        execution_time_ms: Execution time in milliseconds.
        error: Exception if operation failed.

    Example:
        >>> result = BlockResult(
        ...     success=True,
        ...     result=check_result,
        ...     result_dict={"status": "PASSED", "passed_count": 100},
        ...     execution_time_ms=150.5,
        ... )
    """

    success: bool
    result: CheckResult | ProfileResult | LearnResult | None = None
    data: Any = None
    result_dict: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: Exception | None = None

    @property
    def is_success(self) -> bool:
        """Alias for success property."""
        return self.success

    @property
    def has_error(self) -> bool:
        """Check if result has an error."""
        return self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "result": self.result_dict,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "error": str(self.error) if self.error else None,
        }


# =============================================================================
# Preset Configurations
# =============================================================================


DEFAULT_BLOCK_CONFIG = BlockConfig()
"""Default block configuration with standard settings."""

STRICT_BLOCK_CONFIG = BlockConfig(
    fail_on_error=True,
    timeout_seconds=120,
)
"""Strict block configuration that fails fast on errors."""

LENIENT_BLOCK_CONFIG = BlockConfig(
    fail_on_error=False,
    timeout_seconds=600,
)
"""Lenient block configuration that tolerates failures."""


# =============================================================================
# Drift Block Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class DriftBlockConfig(BlockConfig):
    """Configuration specific to drift detection operations.

    Attributes:
        method: Drift detection method (e.g., "ks", "psi", "auto").
        columns: Columns to check for drift (None=all).
        threshold: Detection threshold (None=engine default).
        fail_on_drift: Whether to raise exception when drift is detected.
        baseline_output_key: Key for storing baseline data in block outputs.
        current_output_key: Key for storing current data in block outputs.

    Example:
        >>> config = DriftBlockConfig(
        ...     method="ks",
        ...     threshold=0.05,
        ...     fail_on_drift=True,
        ... )
    """

    method: str = "auto"
    columns: tuple[str, ...] | None = None
    threshold: float | None = None
    fail_on_drift: bool = True
    baseline_output_key: str = "drift_baseline"
    current_output_key: str = "drift_current"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        BlockConfig.__post_init__(self)
        if self.threshold is not None and self.threshold <= 0:
            msg = "threshold must be positive"
            raise ValueError(msg)

    def with_method(self, method: str) -> DriftBlockConfig:
        """Return new config with updated method."""
        return DriftBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            method=method,
            columns=self.columns,
            threshold=self.threshold,
            fail_on_drift=self.fail_on_drift,
            baseline_output_key=self.baseline_output_key,
            current_output_key=self.current_output_key,
        )

    def with_columns(
        self,
        columns: Sequence[str] | tuple[str, ...] | None,
    ) -> DriftBlockConfig:
        """Return new config with updated columns."""
        cols = tuple(columns) if columns else None
        return DriftBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            method=self.method,
            columns=cols,
            threshold=self.threshold,
            fail_on_drift=self.fail_on_drift,
            baseline_output_key=self.baseline_output_key,
            current_output_key=self.current_output_key,
        )

    def with_threshold(self, threshold: float | None) -> DriftBlockConfig:
        """Return new config with updated threshold."""
        return DriftBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            method=self.method,
            columns=self.columns,
            threshold=threshold,
            fail_on_drift=self.fail_on_drift,
            baseline_output_key=self.baseline_output_key,
            current_output_key=self.current_output_key,
        )

    def with_fail_on_drift(self, fail_on_drift: bool) -> DriftBlockConfig:
        """Return new config with updated fail_on_drift."""
        return DriftBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            method=self.method,
            columns=self.columns,
            threshold=self.threshold,
            fail_on_drift=fail_on_drift,
            baseline_output_key=self.baseline_output_key,
            current_output_key=self.current_output_key,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        base = BlockConfig.to_dict(self)
        base.update({
            "method": self.method,
            "columns": list(self.columns) if self.columns else None,
            "threshold": self.threshold,
            "fail_on_drift": self.fail_on_drift,
            "baseline_output_key": self.baseline_output_key,
            "current_output_key": self.current_output_key,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftBlockConfig:
        """Create configuration from dictionary."""
        columns = data.get("columns")
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
            method=data.get("method", "auto"),
            columns=tuple(columns) if columns else None,
            threshold=data.get("threshold"),
            fail_on_drift=data.get("fail_on_drift", True),
            baseline_output_key=data.get("baseline_output_key", "drift_baseline"),
            current_output_key=data.get("current_output_key", "drift_current"),
        )


DEFAULT_DRIFT_BLOCK_CONFIG = DriftBlockConfig()
"""Default drift detection block configuration."""

STRICT_DRIFT_BLOCK_CONFIG = DriftBlockConfig(
    fail_on_drift=True,
    threshold=0.01,
    timeout_seconds=120,
)
"""Strict drift detection configuration with low threshold."""

LENIENT_DRIFT_BLOCK_CONFIG = DriftBlockConfig(
    fail_on_drift=False,
    threshold=0.1,
    timeout_seconds=600,
)
"""Lenient drift detection configuration that tolerates moderate drift."""


# =============================================================================
# Anomaly Block Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class AnomalyBlockConfig(BlockConfig):
    """Configuration specific to anomaly detection operations.

    Attributes:
        detector: Anomaly detection algorithm (e.g., "isolation_forest").
        columns: Columns to check for anomalies (None=all).
        contamination: Expected proportion of anomalies (0.0-1.0).
        fail_on_anomaly: Whether to raise exception when anomalies are detected.

    Example:
        >>> config = AnomalyBlockConfig(
        ...     detector="isolation_forest",
        ...     contamination=0.05,
        ...     fail_on_anomaly=True,
        ... )
    """

    detector: str = "isolation_forest"
    columns: tuple[str, ...] | None = None
    contamination: float = 0.05
    fail_on_anomaly: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        BlockConfig.__post_init__(self)
        if not 0 < self.contamination < 1:
            msg = "contamination must be between 0 and 1 (exclusive)"
            raise ValueError(msg)

    def with_detector(self, detector: str) -> AnomalyBlockConfig:
        """Return new config with updated detector."""
        return AnomalyBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            detector=detector,
            columns=self.columns,
            contamination=self.contamination,
            fail_on_anomaly=self.fail_on_anomaly,
        )

    def with_columns(
        self,
        columns: Sequence[str] | tuple[str, ...] | None,
    ) -> AnomalyBlockConfig:
        """Return new config with updated columns."""
        cols = tuple(columns) if columns else None
        return AnomalyBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            detector=self.detector,
            columns=cols,
            contamination=self.contamination,
            fail_on_anomaly=self.fail_on_anomaly,
        )

    def with_contamination(self, contamination: float) -> AnomalyBlockConfig:
        """Return new config with updated contamination."""
        return AnomalyBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            detector=self.detector,
            columns=self.columns,
            contamination=contamination,
            fail_on_anomaly=self.fail_on_anomaly,
        )

    def with_fail_on_anomaly(self, fail_on_anomaly: bool) -> AnomalyBlockConfig:
        """Return new config with updated fail_on_anomaly."""
        return AnomalyBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            detector=self.detector,
            columns=self.columns,
            contamination=self.contamination,
            fail_on_anomaly=fail_on_anomaly,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        base = BlockConfig.to_dict(self)
        base.update({
            "detector": self.detector,
            "columns": list(self.columns) if self.columns else None,
            "contamination": self.contamination,
            "fail_on_anomaly": self.fail_on_anomaly,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyBlockConfig:
        """Create configuration from dictionary."""
        columns = data.get("columns")
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
            detector=data.get("detector", "isolation_forest"),
            columns=tuple(columns) if columns else None,
            contamination=data.get("contamination", 0.05),
            fail_on_anomaly=data.get("fail_on_anomaly", True),
        )


DEFAULT_ANOMALY_BLOCK_CONFIG = AnomalyBlockConfig()
"""Default anomaly detection block configuration."""

STRICT_ANOMALY_BLOCK_CONFIG = AnomalyBlockConfig(
    fail_on_anomaly=True,
    contamination=0.01,
    timeout_seconds=120,
)
"""Strict anomaly detection configuration with low contamination threshold."""

LENIENT_ANOMALY_BLOCK_CONFIG = AnomalyBlockConfig(
    fail_on_anomaly=False,
    contamination=0.1,
    timeout_seconds=600,
)
"""Lenient anomaly detection configuration that tolerates more anomalies."""
