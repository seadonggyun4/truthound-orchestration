"""Base Op Classes and Configuration for Dagster Integration.

This module provides base configuration types and utilities for
Dagster ops in the data quality integration.

Example:
    >>> from truthound_dagster.ops import CheckOpConfig
    >>>
    >>> config = CheckOpConfig(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     fail_on_error=True,
    ... )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class BaseOpConfig:
    """Base configuration for data quality ops.

    Attributes:
        timeout_seconds: Operation timeout in seconds.
        tags: Metadata tags.

    Example:
        >>> config = BaseOpConfig(timeout_seconds=300.0)
    """

    timeout_seconds: float = 300.0
    tags: "frozenset[str]" = field(default_factory=frozenset)

    def with_timeout(self, timeout_seconds: float) -> "BaseOpConfig":
        """Return new config with updated timeout."""
        return BaseOpConfig(
            timeout_seconds=timeout_seconds,
            tags=self.tags,
        )

    def with_tags(self, *tags: str) -> "BaseOpConfig":
        """Return new config with additional tags."""
        return BaseOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags | frozenset(tags),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class CheckOpConfig(BaseOpConfig):
    """Configuration for check ops.

    Attributes:
        rules: Validation rules to apply.
        fail_on_error: Whether to raise on validation failure.
        warning_threshold: Failure rate threshold for warning.
        sample_size: Number of rows to sample.
        auto_schema: Auto-generate schema from data.

    Example:
        >>> config = CheckOpConfig(
        ...     rules=[{"column": "id", "type": "not_null"}],
        ...     fail_on_error=True,
        ...     warning_threshold=0.05,
        ... )
    """

    rules: "Tuple[Dict[str, Any], ...]" = field(default_factory=tuple)
    fail_on_error: bool = True
    warning_threshold: Optional[float] = None
    sample_size: Optional[int] = None
    auto_schema: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.warning_threshold is not None:
            if not 0 <= self.warning_threshold <= 1:
                msg = "warning_threshold must be between 0 and 1"
                raise ValueError(msg)

        if self.sample_size is not None and self.sample_size < 1:
            msg = "sample_size must be positive"
            raise ValueError(msg)

    def with_rules(self, rules: List[Dict[str, Any]]) -> "CheckOpConfig":
        """Return new config with updated rules."""
        return CheckOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            rules=tuple(rules),
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            auto_schema=self.auto_schema,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> "CheckOpConfig":
        """Return new config with updated fail_on_error."""
        return CheckOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            rules=self.rules,
            fail_on_error=fail_on_error,
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            auto_schema=self.auto_schema,
        )

    def with_warning_threshold(self, threshold: Optional[float]) -> "CheckOpConfig":
        """Return new config with updated warning threshold."""
        return CheckOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=threshold,
            sample_size=self.sample_size,
            auto_schema=self.auto_schema,
        )

    def with_sample_size(self, sample_size: Optional[int]) -> "CheckOpConfig":
        """Return new config with updated sample size."""
        return CheckOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            sample_size=sample_size,
            auto_schema=self.auto_schema,
        )

    def with_auto_schema(self, auto_schema: bool) -> "CheckOpConfig":
        """Return new config with updated auto_schema."""
        return CheckOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            auto_schema=auto_schema,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = BaseOpConfig.to_dict(self)
        return {
            **base_dict,
            "rules": list(self.rules),
            "fail_on_error": self.fail_on_error,
            "warning_threshold": self.warning_threshold,
            "sample_size": self.sample_size,
            "auto_schema": self.auto_schema,
        }


@dataclass(frozen=True, slots=True)
class ProfileOpConfig(BaseOpConfig):
    """Configuration for profile ops.

    Attributes:
        include_histograms: Include histogram data in profile.
        include_samples: Include sample values.
        sample_size: Number of sample values to include.

    Example:
        >>> config = ProfileOpConfig(
        ...     include_histograms=True,
        ...     sample_size=10,
        ... )
    """

    include_histograms: bool = True
    include_samples: bool = True
    sample_size: int = 10

    def with_histograms(self, include: bool) -> "ProfileOpConfig":
        """Return new config with updated histogram setting."""
        return ProfileOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            include_histograms=include,
            include_samples=self.include_samples,
            sample_size=self.sample_size,
        )

    def with_samples(
        self,
        include: bool,
        sample_size: Optional[int] = None,
    ) -> "ProfileOpConfig":
        """Return new config with updated sample settings."""
        return ProfileOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            include_histograms=self.include_histograms,
            include_samples=include,
            sample_size=sample_size if sample_size is not None else self.sample_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = BaseOpConfig.to_dict(self)
        return {
            **base_dict,
            "include_histograms": self.include_histograms,
            "include_samples": self.include_samples,
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True, slots=True)
class LearnOpConfig(BaseOpConfig):
    """Configuration for learn ops.

    Attributes:
        infer_constraints: Infer constraints from data.
        min_confidence: Minimum confidence for learned rules.
        categorical_threshold: Max unique values for categorical.

    Example:
        >>> config = LearnOpConfig(
        ...     infer_constraints=True,
        ...     min_confidence=0.9,
        ... )
    """

    infer_constraints: bool = True
    min_confidence: float = 0.8
    categorical_threshold: int = 20

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.min_confidence <= 1:
            msg = "min_confidence must be between 0 and 1"
            raise ValueError(msg)

        if self.categorical_threshold < 1:
            msg = "categorical_threshold must be positive"
            raise ValueError(msg)

    def with_min_confidence(self, confidence: float) -> "LearnOpConfig":
        """Return new config with updated min confidence."""
        return LearnOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            infer_constraints=self.infer_constraints,
            min_confidence=confidence,
            categorical_threshold=self.categorical_threshold,
        )

    def with_categorical_threshold(self, threshold: int) -> "LearnOpConfig":
        """Return new config with updated threshold."""
        return LearnOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            infer_constraints=self.infer_constraints,
            min_confidence=self.min_confidence,
            categorical_threshold=threshold,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = BaseOpConfig.to_dict(self)
        return {
            **base_dict,
            "infer_constraints": self.infer_constraints,
            "min_confidence": self.min_confidence,
            "categorical_threshold": self.categorical_threshold,
        }


# Preset configurations
DEFAULT_CHECK_CONFIG = CheckOpConfig()

STRICT_CHECK_CONFIG = CheckOpConfig(
    fail_on_error=True,
    warning_threshold=None,
)

LENIENT_CHECK_CONFIG = CheckOpConfig(
    fail_on_error=False,
    warning_threshold=0.10,
)

AUTO_SCHEMA_CHECK_CONFIG = CheckOpConfig(
    auto_schema=True,
    fail_on_error=True,
)

DEFAULT_PROFILE_CONFIG = ProfileOpConfig()

MINIMAL_PROFILE_CONFIG = ProfileOpConfig(
    include_histograms=False,
    include_samples=False,
)

DEFAULT_LEARN_CONFIG = LearnOpConfig()

HIGH_CONFIDENCE_LEARN_CONFIG = LearnOpConfig(
    min_confidence=0.95,
)


@dataclass(frozen=True, slots=True)
class DriftOpConfig(BaseOpConfig):
    """Configuration for drift detection ops.

    Attributes:
        method: Drift detection method (e.g., "auto", "ks", "psi").
        columns: Columns to check for drift. None means all columns.
        threshold: Drift threshold. None uses engine default.
        fail_on_drift: Whether to raise on drift detection.

    Example:
        >>> config = DriftOpConfig(
        ...     method="ks",
        ...     threshold=0.05,
        ...     fail_on_drift=True,
        ... )
    """

    method: str = "auto"
    columns: "tuple[str, ...] | None" = None
    threshold: Optional[float] = None
    fail_on_drift: bool = True

    def with_method(self, method: str) -> "DriftOpConfig":
        """Return new config with updated method."""
        return DriftOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            method=method,
            columns=self.columns,
            threshold=self.threshold,
            fail_on_drift=self.fail_on_drift,
        )

    def with_columns(self, columns: Sequence[str]) -> "DriftOpConfig":
        """Return new config with updated columns."""
        return DriftOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            method=self.method,
            columns=tuple(columns),
            threshold=self.threshold,
            fail_on_drift=self.fail_on_drift,
        )

    def with_threshold(self, threshold: float) -> "DriftOpConfig":
        """Return new config with updated threshold."""
        return DriftOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            method=self.method,
            columns=self.columns,
            threshold=threshold,
            fail_on_drift=self.fail_on_drift,
        )

    def with_fail_on_drift(self, fail_on_drift: bool) -> "DriftOpConfig":
        """Return new config with updated fail_on_drift."""
        return DriftOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            method=self.method,
            columns=self.columns,
            threshold=self.threshold,
            fail_on_drift=fail_on_drift,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = BaseOpConfig.to_dict(self)
        return {
            **base_dict,
            "method": self.method,
            "columns": list(self.columns) if self.columns else None,
            "threshold": self.threshold,
            "fail_on_drift": self.fail_on_drift,
        }


DEFAULT_DRIFT_CONFIG = DriftOpConfig()

STRICT_DRIFT_CONFIG = DriftOpConfig(
    threshold=0.01,
    fail_on_drift=True,
)

LENIENT_DRIFT_CONFIG = DriftOpConfig(
    threshold=0.1,
    fail_on_drift=False,
)


@dataclass(frozen=True, slots=True)
class AnomalyOpConfig(BaseOpConfig):
    """Configuration for anomaly detection ops.

    Attributes:
        detector: Anomaly detection algorithm (e.g., "isolation_forest").
        columns: Columns to check for anomalies. None means all columns.
        contamination: Expected proportion of anomalies.
        fail_on_anomaly: Whether to raise on anomaly detection.

    Example:
        >>> config = AnomalyOpConfig(
        ...     detector="isolation_forest",
        ...     contamination=0.05,
        ...     fail_on_anomaly=True,
        ... )
    """

    detector: str = "isolation_forest"
    columns: "tuple[str, ...] | None" = None
    contamination: float = 0.05
    fail_on_anomaly: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 < self.contamination < 1:
            msg = "contamination must be between 0 and 1 (exclusive)"
            raise ValueError(msg)

    def with_detector(self, detector: str) -> "AnomalyOpConfig":
        """Return new config with updated detector."""
        return AnomalyOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            detector=detector,
            columns=self.columns,
            contamination=self.contamination,
            fail_on_anomaly=self.fail_on_anomaly,
        )

    def with_columns(self, columns: Sequence[str]) -> "AnomalyOpConfig":
        """Return new config with updated columns."""
        return AnomalyOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            detector=self.detector,
            columns=tuple(columns),
            contamination=self.contamination,
            fail_on_anomaly=self.fail_on_anomaly,
        )

    def with_contamination(self, contamination: float) -> "AnomalyOpConfig":
        """Return new config with updated contamination."""
        return AnomalyOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            detector=self.detector,
            columns=self.columns,
            contamination=contamination,
            fail_on_anomaly=self.fail_on_anomaly,
        )

    def with_fail_on_anomaly(self, fail_on_anomaly: bool) -> "AnomalyOpConfig":
        """Return new config with updated fail_on_anomaly."""
        return AnomalyOpConfig(
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
            detector=self.detector,
            columns=self.columns,
            contamination=self.contamination,
            fail_on_anomaly=fail_on_anomaly,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = BaseOpConfig.to_dict(self)
        return {
            **base_dict,
            "detector": self.detector,
            "columns": list(self.columns) if self.columns else None,
            "contamination": self.contamination,
            "fail_on_anomaly": self.fail_on_anomaly,
        }


DEFAULT_ANOMALY_CONFIG = AnomalyOpConfig()

STRICT_ANOMALY_CONFIG = AnomalyOpConfig(
    contamination=0.01,
    fail_on_anomaly=True,
)

LENIENT_ANOMALY_CONFIG = AnomalyOpConfig(
    contamination=0.1,
    fail_on_anomaly=False,
)
