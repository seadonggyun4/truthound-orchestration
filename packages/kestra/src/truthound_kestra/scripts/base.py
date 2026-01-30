"""Base configuration classes for Kestra data quality scripts.

This module provides the foundational configuration classes used by
all data quality scripts in the Kestra integration.

Example:
    >>> from truthound_kestra.scripts.base import (
    ...     ScriptConfig,
    ...     CheckScriptConfig,
    ...     ProfileScriptConfig,
    ... )
    >>>
    >>> config = CheckScriptConfig(
    ...     engine_name="truthound",
    ...     fail_on_error=True,
    ...     timeout_seconds=300.0
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

from truthound_kestra.utils.exceptions import ConfigurationError
from truthound_kestra.utils.types import (
    OperationType,
    RuleDict,
    Severity,
)

if TYPE_CHECKING:
    pass

__all__ = [
    # Protocols
    "DataQualityEngineProtocol",
    "ScriptExecutorProtocol",
    # Config classes
    "ScriptConfig",
    "CheckScriptConfig",
    "ProfileScriptConfig",
    "LearnScriptConfig",
    "DriftScriptConfig",
    "AnomalyScriptConfig",
    # Presets
    "DEFAULT_SCRIPT_CONFIG",
    "STRICT_SCRIPT_CONFIG",
    "LENIENT_SCRIPT_CONFIG",
    "PRODUCTION_SCRIPT_CONFIG",
    "DEFAULT_DRIFT_SCRIPT_CONFIG",
    "STRICT_DRIFT_SCRIPT_CONFIG",
    "LENIENT_DRIFT_SCRIPT_CONFIG",
    "DEFAULT_ANOMALY_SCRIPT_CONFIG",
    "STRICT_ANOMALY_SCRIPT_CONFIG",
    "LENIENT_ANOMALY_SCRIPT_CONFIG",
    # Utility
    "get_engine",
    "create_script_config",
]

ConfigT = TypeVar("ConfigT", bound="ScriptConfig")
ResultT = TypeVar("ResultT")


@runtime_checkable
class DataQualityEngineProtocol(Protocol):
    """Protocol for data quality engines.

    This protocol defines the interface that all data quality engines
    must implement to be used with Kestra scripts.
    """

    @property
    def engine_name(self) -> str:
        """Get the engine name."""
        ...

    @property
    def engine_version(self) -> str:
        """Get the engine version."""
        ...

    def check(self, data: Any, rules: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        """Check data quality against rules."""
        ...

    def profile(self, data: Any, **kwargs: Any) -> Any:
        """Profile data to generate statistics."""
        ...

    def learn(self, data: Any, **kwargs: Any) -> Any:
        """Learn schema/rules from data."""
        ...


@runtime_checkable
class ScriptExecutorProtocol(Protocol[ConfigT, ResultT]):
    """Protocol for script executors."""

    @property
    def config(self) -> ConfigT:
        """Get the script configuration."""
        ...

    def execute(self, data: Any, **kwargs: Any) -> ResultT:
        """Execute the script operation."""
        ...


@dataclass(frozen=True, slots=True)
class ScriptConfig:
    """Base configuration for all data quality scripts.

    This is the base class for script configurations. It provides
    common settings shared by all script types.

    Attributes:
        engine_name: Name of the data quality engine to use.
        enabled: Whether the script is enabled.
        timeout_seconds: Maximum execution time in seconds.
        tags: Tags for categorizing the script.
        description: Optional description of the script.
        metadata: Additional metadata for the script.

    Example:
        >>> config = ScriptConfig(
        ...     engine_name="truthound",
        ...     timeout_seconds=300.0,
        ...     tags=frozenset(["production", "users"])
        ... )
    """

    engine_name: str = "truthound"
    enabled: bool = True
    timeout_seconds: float = 300.0
    tags: frozenset[str] = field(default_factory=frozenset)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.timeout_seconds <= 0:
            msg = "timeout_seconds must be positive"
            raise ConfigurationError(
                message=msg,
                field="timeout_seconds",
                value=self.timeout_seconds,
                reason=msg,
            )

    def with_engine(self, engine_name: str) -> ScriptConfig:
        """Return new config with updated engine name."""
        return ScriptConfig(
            engine_name=engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
        )

    def with_timeout(self, timeout_seconds: float) -> ScriptConfig:
        """Return new config with updated timeout."""
        return ScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
        )

    def with_tags(self, *tags: str) -> ScriptConfig:
        """Return new config with additional tags."""
        return ScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags | frozenset(tags),
            description=self.description,
            metadata=self.metadata,
        )

    def with_metadata(self, **metadata: Any) -> ScriptConfig:
        """Return new config with additional metadata."""
        return ScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata={**self.metadata, **metadata},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "engine_name": self.engine_name,
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScriptConfig:
        """Create from dictionary."""
        return cls(
            engine_name=data.get("engine_name", "truthound"),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            tags=frozenset(data.get("tags", [])),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class CheckScriptConfig(ScriptConfig):
    """Configuration for check quality scripts.

    This configuration class extends ScriptConfig with settings
    specific to data quality check operations.

    Attributes:
        rules: Tuple of rule dictionaries to validate against.
        fail_on_error: Whether to raise exception on check failure.
        auto_schema: Whether to use auto-generated schema (Truthound).
        min_severity: Minimum severity level to report.
        sample_failures: Maximum number of failure samples to include.
        parallel: Whether to use parallel validation.
        max_workers: Maximum parallel workers (if parallel=True).

    Example:
        >>> config = CheckScriptConfig(
        ...     engine_name="truthound",
        ...     rules=(
        ...         {"type": "not_null", "column": "id"},
        ...         {"type": "unique", "column": "email"},
        ...     ),
        ...     fail_on_error=True,
        ...     auto_schema=True
        ... )
    """

    rules: tuple[RuleDict, ...] = ()
    fail_on_error: bool = True
    auto_schema: bool = False
    min_severity: Severity = Severity.LOW
    sample_failures: int = 100
    parallel: bool = False
    max_workers: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Use explicit class call for frozen dataclass with slots
        ScriptConfig.__post_init__(self)

        if self.sample_failures < 0:
            msg = "sample_failures must be non-negative"
            raise ConfigurationError(
                message=msg,
                field="sample_failures",
                value=self.sample_failures,
                reason=msg,
            )

        if self.max_workers is not None and self.max_workers < 1:
            msg = "max_workers must be at least 1"
            raise ConfigurationError(
                message=msg,
                field="max_workers",
                value=self.max_workers,
                reason=msg,
            )

    def with_rules(self, rules: list[RuleDict]) -> CheckScriptConfig:
        """Return new config with updated rules."""
        return CheckScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            rules=tuple(rules),
            fail_on_error=self.fail_on_error,
            auto_schema=self.auto_schema,
            min_severity=self.min_severity,
            sample_failures=self.sample_failures,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> CheckScriptConfig:
        """Return new config with updated fail_on_error."""
        return CheckScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            rules=self.rules,
            fail_on_error=fail_on_error,
            auto_schema=self.auto_schema,
            min_severity=self.min_severity,
            sample_failures=self.sample_failures,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )

    def with_auto_schema(self, auto_schema: bool) -> CheckScriptConfig:
        """Return new config with updated auto_schema."""
        return CheckScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            auto_schema=auto_schema,
            min_severity=self.min_severity,
            sample_failures=self.sample_failures,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )

    def with_min_severity(self, min_severity: Severity | str) -> CheckScriptConfig:
        """Return new config with updated min_severity."""
        if isinstance(min_severity, str):
            min_severity = Severity(min_severity.lower())
        return CheckScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            auto_schema=self.auto_schema,
            min_severity=min_severity,
            sample_failures=self.sample_failures,
            parallel=self.parallel,
            max_workers=self.max_workers,
        )

    def with_parallel(
        self,
        parallel: bool,
        max_workers: int | None = None,
    ) -> CheckScriptConfig:
        """Return new config with updated parallel settings."""
        return CheckScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            auto_schema=self.auto_schema,
            min_severity=self.min_severity,
            sample_failures=self.sample_failures,
            parallel=parallel,
            max_workers=max_workers or self.max_workers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "rules": list(self.rules),
            "fail_on_error": self.fail_on_error,
            "auto_schema": self.auto_schema,
            "min_severity": self.min_severity.value,
            "sample_failures": self.sample_failures,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckScriptConfig:
        """Create from dictionary."""
        return cls(
            engine_name=data.get("engine_name", "truthound"),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            tags=frozenset(data.get("tags", [])),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            rules=tuple(data.get("rules", [])),
            fail_on_error=data.get("fail_on_error", True),
            auto_schema=data.get("auto_schema", False),
            min_severity=Severity(data.get("min_severity", "low")),
            sample_failures=data.get("sample_failures", 100),
            parallel=data.get("parallel", False),
            max_workers=data.get("max_workers"),
        )


@dataclass(frozen=True, slots=True)
class ProfileScriptConfig(ScriptConfig):
    """Configuration for data profiling scripts.

    This configuration class extends ScriptConfig with settings
    specific to data profiling operations.

    Attributes:
        include_stats: Whether to include statistical summaries.
        include_histograms: Whether to include value histograms.
        sample_size: Maximum rows to sample (0 = all).
        top_n: Number of top values to include per column.

    Example:
        >>> config = ProfileScriptConfig(
        ...     engine_name="truthound",
        ...     include_stats=True,
        ...     sample_size=10000
        ... )
    """

    include_stats: bool = True
    include_histograms: bool = False
    sample_size: int = 0
    top_n: int = 10

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Use explicit class call for frozen dataclass with slots
        ScriptConfig.__post_init__(self)

        if self.sample_size < 0:
            msg = "sample_size must be non-negative"
            raise ConfigurationError(
                message=msg,
                field="sample_size",
                value=self.sample_size,
                reason=msg,
            )

        if self.top_n < 0:
            msg = "top_n must be non-negative"
            raise ConfigurationError(
                message=msg,
                field="top_n",
                value=self.top_n,
                reason=msg,
            )

    def with_stats(self, include_stats: bool) -> ProfileScriptConfig:
        """Return new config with updated include_stats."""
        return ProfileScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            include_stats=include_stats,
            include_histograms=self.include_histograms,
            sample_size=self.sample_size,
            top_n=self.top_n,
        )

    def with_sample_size(self, sample_size: int) -> ProfileScriptConfig:
        """Return new config with updated sample_size."""
        return ProfileScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            include_stats=self.include_stats,
            include_histograms=self.include_histograms,
            sample_size=sample_size,
            top_n=self.top_n,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "include_stats": self.include_stats,
            "include_histograms": self.include_histograms,
            "sample_size": self.sample_size,
            "top_n": self.top_n,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileScriptConfig:
        """Create from dictionary."""
        return cls(
            engine_name=data.get("engine_name", "truthound"),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            tags=frozenset(data.get("tags", [])),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            include_stats=data.get("include_stats", True),
            include_histograms=data.get("include_histograms", False),
            sample_size=data.get("sample_size", 0),
            top_n=data.get("top_n", 10),
        )


@dataclass(frozen=True, slots=True)
class LearnScriptConfig(ScriptConfig):
    """Configuration for schema learning scripts.

    This configuration class extends ScriptConfig with settings
    specific to schema/rule learning operations.

    Attributes:
        min_confidence: Minimum confidence threshold for learned rules.
        include_patterns: Whether to learn regex patterns.
        include_ranges: Whether to learn value ranges.
        include_categories: Whether to learn categorical values.
        categorical_threshold: Max unique values to treat as categorical.
        sample_size: Maximum rows to sample (0 = all).

    Example:
        >>> config = LearnScriptConfig(
        ...     engine_name="truthound",
        ...     min_confidence=0.8,
        ...     include_patterns=True
        ... )
    """

    min_confidence: float = 0.8
    include_patterns: bool = True
    include_ranges: bool = True
    include_categories: bool = True
    categorical_threshold: int = 50
    sample_size: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Use explicit class call for frozen dataclass with slots
        ScriptConfig.__post_init__(self)

        if not 0.0 <= self.min_confidence <= 1.0:
            msg = "min_confidence must be between 0.0 and 1.0"
            raise ConfigurationError(
                message=msg,
                field="min_confidence",
                value=self.min_confidence,
                reason=msg,
            )

        if self.categorical_threshold < 1:
            msg = "categorical_threshold must be at least 1"
            raise ConfigurationError(
                message=msg,
                field="categorical_threshold",
                value=self.categorical_threshold,
                reason=msg,
            )

    def with_min_confidence(self, min_confidence: float) -> LearnScriptConfig:
        """Return new config with updated min_confidence."""
        return LearnScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            min_confidence=min_confidence,
            include_patterns=self.include_patterns,
            include_ranges=self.include_ranges,
            include_categories=self.include_categories,
            categorical_threshold=self.categorical_threshold,
            sample_size=self.sample_size,
        )

    def with_sample_size(self, sample_size: int) -> LearnScriptConfig:
        """Return new config with updated sample_size."""
        return LearnScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            min_confidence=self.min_confidence,
            include_patterns=self.include_patterns,
            include_ranges=self.include_ranges,
            include_categories=self.include_categories,
            categorical_threshold=self.categorical_threshold,
            sample_size=sample_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "min_confidence": self.min_confidence,
            "include_patterns": self.include_patterns,
            "include_ranges": self.include_ranges,
            "include_categories": self.include_categories,
            "categorical_threshold": self.categorical_threshold,
            "sample_size": self.sample_size,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnScriptConfig:
        """Create from dictionary."""
        return cls(
            engine_name=data.get("engine_name", "truthound"),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            tags=frozenset(data.get("tags", [])),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            min_confidence=data.get("min_confidence", 0.8),
            include_patterns=data.get("include_patterns", True),
            include_ranges=data.get("include_ranges", True),
            include_categories=data.get("include_categories", True),
            categorical_threshold=data.get("categorical_threshold", 50),
            sample_size=data.get("sample_size", 0),
        )


@dataclass(frozen=True, slots=True)
class DriftScriptConfig(ScriptConfig):
    """Configuration for drift detection scripts.

    This configuration class extends ScriptConfig with settings
    specific to data drift detection operations.

    Attributes:
        method: Statistical method for drift detection.
        columns: Columns to check for drift. None means all.
        threshold: Detection threshold. None uses engine default.
        fail_on_drift: Whether to raise exception on drift detection.
        baseline_data_path: Path to baseline data file.
        current_data_path: Path to current data file.
        baseline_sql: SQL query for baseline data.
        current_sql: SQL query for current data.

    Example:
        >>> config = DriftScriptConfig(
        ...     engine_name="truthound",
        ...     method="ks",
        ...     threshold=0.05,
        ...     fail_on_drift=True,
        ... )
    """

    method: str = "auto"
    columns: tuple[str, ...] | None = None
    threshold: float | None = None
    fail_on_drift: bool = True
    baseline_data_path: str | None = None
    current_data_path: str | None = None
    baseline_sql: str | None = None
    current_sql: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        ScriptConfig.__post_init__(self)

        if self.threshold is not None and self.threshold < 0:
            msg = "threshold must be non-negative"
            raise ConfigurationError(
                message=msg,
                field="threshold",
                value=self.threshold,
                reason=msg,
            )

    def with_method(self, method: str) -> DriftScriptConfig:
        """Return new config with updated method."""
        return DriftScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            method=method,
            columns=self.columns,
            threshold=self.threshold,
            fail_on_drift=self.fail_on_drift,
            baseline_data_path=self.baseline_data_path,
            current_data_path=self.current_data_path,
            baseline_sql=self.baseline_sql,
            current_sql=self.current_sql,
        )

    def with_columns(self, columns: Sequence[str]) -> DriftScriptConfig:
        """Return new config with updated columns."""
        return DriftScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            method=self.method,
            columns=tuple(columns),
            threshold=self.threshold,
            fail_on_drift=self.fail_on_drift,
            baseline_data_path=self.baseline_data_path,
            current_data_path=self.current_data_path,
            baseline_sql=self.baseline_sql,
            current_sql=self.current_sql,
        )

    def with_threshold(self, threshold: float) -> DriftScriptConfig:
        """Return new config with updated threshold."""
        return DriftScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            method=self.method,
            columns=self.columns,
            threshold=threshold,
            fail_on_drift=self.fail_on_drift,
            baseline_data_path=self.baseline_data_path,
            current_data_path=self.current_data_path,
            baseline_sql=self.baseline_sql,
            current_sql=self.current_sql,
        )

    def with_fail_on_drift(self, fail_on_drift: bool) -> DriftScriptConfig:
        """Return new config with updated fail_on_drift."""
        return DriftScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            method=self.method,
            columns=self.columns,
            threshold=self.threshold,
            fail_on_drift=fail_on_drift,
            baseline_data_path=self.baseline_data_path,
            current_data_path=self.current_data_path,
            baseline_sql=self.baseline_sql,
            current_sql=self.current_sql,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "method": self.method,
            "columns": list(self.columns) if self.columns else None,
            "threshold": self.threshold,
            "fail_on_drift": self.fail_on_drift,
            "baseline_data_path": self.baseline_data_path,
            "current_data_path": self.current_data_path,
            "baseline_sql": self.baseline_sql,
            "current_sql": self.current_sql,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftScriptConfig:
        """Create from dictionary."""
        columns = data.get("columns")
        return cls(
            engine_name=data.get("engine_name", "truthound"),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            tags=frozenset(data.get("tags", [])),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            method=data.get("method", "auto"),
            columns=tuple(columns) if columns else None,
            threshold=data.get("threshold"),
            fail_on_drift=data.get("fail_on_drift", True),
            baseline_data_path=data.get("baseline_data_path"),
            current_data_path=data.get("current_data_path"),
            baseline_sql=data.get("baseline_sql"),
            current_sql=data.get("current_sql"),
        )


@dataclass(frozen=True, slots=True)
class AnomalyScriptConfig(ScriptConfig):
    """Configuration for anomaly detection scripts.

    This configuration class extends ScriptConfig with settings
    specific to anomaly detection operations.

    Attributes:
        detector: Anomaly detection algorithm name.
        columns: Columns to check for anomalies. None means all.
        contamination: Expected proportion of anomalies (0.0 to 1.0).
        fail_on_anomaly: Whether to raise exception on anomaly detection.
        data_path: Path to data file.
        data_sql: SQL query for data.

    Example:
        >>> config = AnomalyScriptConfig(
        ...     engine_name="truthound",
        ...     detector="isolation_forest",
        ...     contamination=0.05,
        ...     fail_on_anomaly=True,
        ... )
    """

    detector: str = "isolation_forest"
    columns: tuple[str, ...] | None = None
    contamination: float = 0.05
    fail_on_anomaly: bool = True
    data_path: str | None = None
    data_sql: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        ScriptConfig.__post_init__(self)

        if not 0.0 < self.contamination < 1.0:
            msg = "contamination must be between 0.0 and 1.0 (exclusive)"
            raise ConfigurationError(
                message=msg,
                field="contamination",
                value=self.contamination,
                reason=msg,
            )

    def with_detector(self, detector: str) -> AnomalyScriptConfig:
        """Return new config with updated detector."""
        return AnomalyScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            detector=detector,
            columns=self.columns,
            contamination=self.contamination,
            fail_on_anomaly=self.fail_on_anomaly,
            data_path=self.data_path,
            data_sql=self.data_sql,
        )

    def with_columns(self, columns: Sequence[str]) -> AnomalyScriptConfig:
        """Return new config with updated columns."""
        return AnomalyScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            detector=self.detector,
            columns=tuple(columns),
            contamination=self.contamination,
            fail_on_anomaly=self.fail_on_anomaly,
            data_path=self.data_path,
            data_sql=self.data_sql,
        )

    def with_contamination(self, contamination: float) -> AnomalyScriptConfig:
        """Return new config with updated contamination."""
        return AnomalyScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            detector=self.detector,
            columns=self.columns,
            contamination=contamination,
            fail_on_anomaly=self.fail_on_anomaly,
            data_path=self.data_path,
            data_sql=self.data_sql,
        )

    def with_fail_on_anomaly(self, fail_on_anomaly: bool) -> AnomalyScriptConfig:
        """Return new config with updated fail_on_anomaly."""
        return AnomalyScriptConfig(
            engine_name=self.engine_name,
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            description=self.description,
            metadata=self.metadata,
            detector=self.detector,
            columns=self.columns,
            contamination=self.contamination,
            fail_on_anomaly=fail_on_anomaly,
            data_path=self.data_path,
            data_sql=self.data_sql,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        base = super().to_dict()
        base.update({
            "detector": self.detector,
            "columns": list(self.columns) if self.columns else None,
            "contamination": self.contamination,
            "fail_on_anomaly": self.fail_on_anomaly,
            "data_path": self.data_path,
            "data_sql": self.data_sql,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyScriptConfig:
        """Create from dictionary."""
        columns = data.get("columns")
        return cls(
            engine_name=data.get("engine_name", "truthound"),
            enabled=data.get("enabled", True),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            tags=frozenset(data.get("tags", [])),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            detector=data.get("detector", "isolation_forest"),
            columns=tuple(columns) if columns else None,
            contamination=data.get("contamination", 0.05),
            fail_on_anomaly=data.get("fail_on_anomaly", True),
            data_path=data.get("data_path"),
            data_sql=data.get("data_sql"),
        )


# =============================================================================
# Preset Configurations
# =============================================================================


DEFAULT_SCRIPT_CONFIG = ScriptConfig()
"""Default script configuration."""

STRICT_SCRIPT_CONFIG = CheckScriptConfig(
    fail_on_error=True,
    min_severity=Severity.LOW,
    sample_failures=1000,
)
"""Strict configuration that fails on any error."""

LENIENT_SCRIPT_CONFIG = CheckScriptConfig(
    fail_on_error=False,
    min_severity=Severity.HIGH,
    sample_failures=10,
)
"""Lenient configuration that only reports high severity issues."""

PRODUCTION_SCRIPT_CONFIG = CheckScriptConfig(
    fail_on_error=True,
    min_severity=Severity.MEDIUM,
    sample_failures=100,
    parallel=True,
    timeout_seconds=600.0,
)
"""Production configuration with balanced settings."""

DEFAULT_DRIFT_SCRIPT_CONFIG = DriftScriptConfig()
"""Default drift detection configuration."""

STRICT_DRIFT_SCRIPT_CONFIG = DriftScriptConfig(
    method="ks",
    threshold=0.01,
    fail_on_drift=True,
    timeout_seconds=600.0,
)
"""Strict drift detection configuration with low threshold."""

LENIENT_DRIFT_SCRIPT_CONFIG = DriftScriptConfig(
    method="auto",
    threshold=0.1,
    fail_on_drift=False,
)
"""Lenient drift detection configuration with high threshold."""

DEFAULT_ANOMALY_SCRIPT_CONFIG = AnomalyScriptConfig()
"""Default anomaly detection configuration."""

STRICT_ANOMALY_SCRIPT_CONFIG = AnomalyScriptConfig(
    detector="isolation_forest",
    contamination=0.01,
    fail_on_anomaly=True,
    timeout_seconds=600.0,
)
"""Strict anomaly detection configuration with low contamination."""

LENIENT_ANOMALY_SCRIPT_CONFIG = AnomalyScriptConfig(
    detector="isolation_forest",
    contamination=0.1,
    fail_on_anomaly=False,
)
"""Lenient anomaly detection configuration with high contamination."""


# =============================================================================
# Utility Functions
# =============================================================================


def get_engine(name: str = "truthound") -> DataQualityEngineProtocol:
    """Get a data quality engine by name.

    This function provides a unified way to obtain engine instances
    from the truthound-orchestration common package.

    Args:
        name: Name of the engine to get.

    Returns:
        Engine instance implementing DataQualityEngineProtocol.

    Raises:
        ConfigurationError: If the engine is not found.

    Example:
        >>> engine = get_engine("truthound")
        >>> result = engine.check(data, auto_schema=True)
    """
    try:
        from common.engines import get_engine as _get_engine

        return _get_engine(name)
    except ImportError:
        # Fallback for when common package is not fully available
        try:
            if name == "truthound":
                from common.engines.truthound import TruthoundEngine

                return TruthoundEngine()
            elif name == "great_expectations":
                from common.engines.great_expectations import GreatExpectationsAdapter

                return GreatExpectationsAdapter()
            elif name == "pandera":
                from common.engines.pandera import PanderaAdapter

                return PanderaAdapter()
        except ImportError:
            pass

        raise ConfigurationError(
            message=f"Engine '{name}' not found",
            field="engine_name",
            value=name,
            reason="Engine not available. Install the required dependencies.",
        )
    except Exception as e:
        raise ConfigurationError(
            message=f"Failed to get engine '{name}': {e}",
            field="engine_name",
            value=name,
            reason=str(e),
        ) from e


def create_script_config(
    operation: OperationType | str,
    **kwargs: Any,
) -> ScriptConfig:
    """Create a script configuration for the specified operation.

    Factory function to create the appropriate configuration class
    based on the operation type.

    Args:
        operation: Type of operation (check, profile, learn).
        **kwargs: Configuration arguments.

    Returns:
        Appropriate ScriptConfig subclass instance.

    Raises:
        ConfigurationError: If operation type is invalid.

    Example:
        >>> config = create_script_config("check", fail_on_error=True)
        >>> config = create_script_config(OperationType.PROFILE, sample_size=1000)
    """
    if isinstance(operation, str):
        try:
            operation = OperationType(operation.lower())
        except ValueError:
            raise ConfigurationError(
                message=f"Invalid operation type: {operation}",
                field="operation",
                value=operation,
                reason="Must be one of: check, profile, learn",
            )

    config_map = {
        OperationType.CHECK: CheckScriptConfig,
        OperationType.PROFILE: ProfileScriptConfig,
        OperationType.LEARN: LearnScriptConfig,
    }

    config_class = config_map.get(operation)
    if config_class is None:
        raise ConfigurationError(
            message=f"No configuration class for operation: {operation}",
            field="operation",
            value=operation.value,
        )

    return config_class(**kwargs)
