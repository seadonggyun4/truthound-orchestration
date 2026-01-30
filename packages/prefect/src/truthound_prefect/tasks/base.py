"""Base task configurations for data quality operations.

This module provides immutable configuration classes for tasks,
following the frozen dataclass pattern with builder methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class BaseTaskConfig:
    """Base configuration for all data quality tasks.

    Attributes:
        enabled: Whether the task is enabled.
        timeout_seconds: Timeout for the task in seconds.
        retries: Number of retries on failure.
        retry_delay_seconds: Delay between retries.
        tags: Tags for categorization and filtering.
        cache_key: Optional cache key for result caching.
        cache_expiration_seconds: Cache expiration time.
    """

    enabled: bool = True
    timeout_seconds: float = 300.0
    retries: int = 0
    retry_delay_seconds: float = 10.0
    tags: frozenset[str] = field(default_factory=frozenset)
    cache_key: str | None = None
    cache_expiration_seconds: float | None = None

    def with_timeout(self, timeout_seconds: float) -> BaseTaskConfig:
        """Return a new config with timeout changed."""
        return BaseTaskConfig(
            enabled=self.enabled,
            timeout_seconds=timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
        )

    def with_retries(
        self,
        retries: int,
        retry_delay_seconds: float | None = None,
    ) -> BaseTaskConfig:
        """Return a new config with retry settings changed."""
        return BaseTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds or self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
        )

    def with_cache(
        self,
        cache_key: str,
        expiration_seconds: float | None = None,
    ) -> BaseTaskConfig:
        """Return a new config with caching enabled."""
        return BaseTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=cache_key,
            cache_expiration_seconds=expiration_seconds,
        )

    def with_tags(self, *tags: str) -> BaseTaskConfig:
        """Return a new config with additional tags."""
        new_tags = self.tags | frozenset(tags)
        return BaseTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=new_tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
        )


@dataclass(frozen=True, slots=True)
class CheckTaskConfig(BaseTaskConfig):
    """Configuration for check tasks.

    Attributes:
        rules: Tuple of rules to check.
        fail_on_error: Raise exception on check failures.
        warning_threshold: Failure rate threshold for warnings (0.0 to 1.0).
        auto_schema: Use auto-schema mode (Truthound only).
        store_result: Store result as Prefect artifact.
        result_key: Key for storing the result.
    """

    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    fail_on_error: bool = True
    warning_threshold: float | None = None
    auto_schema: bool = False
    store_result: bool = True
    result_key: str = "check_result"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.warning_threshold is not None:
            if not 0.0 <= self.warning_threshold <= 1.0:
                raise ValueError("warning_threshold must be between 0.0 and 1.0")

    def with_rules(self, rules: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> CheckTaskConfig:
        """Return a new config with rules changed."""
        return CheckTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            rules=tuple(rules),
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_result=self.store_result,
            result_key=self.result_key,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> CheckTaskConfig:
        """Return a new config with fail_on_error changed."""
        return CheckTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            rules=self.rules,
            fail_on_error=fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_result=self.store_result,
            result_key=self.result_key,
        )

    def with_warning_threshold(self, threshold: float | None) -> CheckTaskConfig:
        """Return a new config with warning_threshold changed."""
        return CheckTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=threshold,
            auto_schema=self.auto_schema,
            store_result=self.store_result,
            result_key=self.result_key,
        )

    def with_auto_schema(self, auto_schema: bool = True) -> CheckTaskConfig:
        """Return a new config with auto_schema changed."""
        return CheckTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=auto_schema,
            store_result=self.store_result,
            result_key=self.result_key,
        )


@dataclass(frozen=True, slots=True)
class ProfileTaskConfig(BaseTaskConfig):
    """Configuration for profile tasks.

    Attributes:
        include_histograms: Include histogram data in profile.
        sample_size: Maximum rows to sample for profiling.
        store_result: Store result as Prefect artifact.
        result_key: Key for storing the result.
    """

    include_histograms: bool = False
    sample_size: int | None = None
    store_result: bool = True
    result_key: str = "profile_result"

    def with_histograms(self, include: bool = True) -> ProfileTaskConfig:
        """Return a new config with histogram setting changed."""
        return ProfileTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            include_histograms=include,
            sample_size=self.sample_size,
            store_result=self.store_result,
            result_key=self.result_key,
        )

    def with_sample_size(self, sample_size: int | None) -> ProfileTaskConfig:
        """Return a new config with sample size changed."""
        return ProfileTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            include_histograms=self.include_histograms,
            sample_size=sample_size,
            store_result=self.store_result,
            result_key=self.result_key,
        )


@dataclass(frozen=True, slots=True)
class LearnTaskConfig(BaseTaskConfig):
    """Configuration for learn tasks.

    Attributes:
        infer_constraints: Infer constraints from data.
        min_confidence: Minimum confidence for learned rules.
        categorical_threshold: Threshold for categorical column detection.
        store_result: Store result as Prefect artifact.
        result_key: Key for storing the result.
    """

    infer_constraints: bool = True
    min_confidence: float = 0.9
    categorical_threshold: int = 20
    store_result: bool = True
    result_key: str = "learn_result"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.categorical_threshold < 1:
            raise ValueError("categorical_threshold must be at least 1")

    def with_min_confidence(self, confidence: float) -> LearnTaskConfig:
        """Return a new config with min_confidence changed."""
        return LearnTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            infer_constraints=self.infer_constraints,
            min_confidence=confidence,
            categorical_threshold=self.categorical_threshold,
            store_result=self.store_result,
            result_key=self.result_key,
        )

    def with_categorical_threshold(self, threshold: int) -> LearnTaskConfig:
        """Return a new config with categorical_threshold changed."""
        return LearnTaskConfig(
            enabled=self.enabled,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            cache_key=self.cache_key,
            cache_expiration_seconds=self.cache_expiration_seconds,
            infer_constraints=self.infer_constraints,
            min_confidence=self.min_confidence,
            categorical_threshold=threshold,
            store_result=self.store_result,
            result_key=self.result_key,
        )


# Preset configurations
DEFAULT_CHECK_CONFIG = CheckTaskConfig()

STRICT_CHECK_CONFIG = CheckTaskConfig(
    fail_on_error=True,
    warning_threshold=None,
    tags=frozenset({"strict"}),
)

LENIENT_CHECK_CONFIG = CheckTaskConfig(
    fail_on_error=False,
    warning_threshold=0.10,
    tags=frozenset({"lenient"}),
)

AUTO_SCHEMA_CHECK_CONFIG = CheckTaskConfig(
    auto_schema=True,
    tags=frozenset({"auto-schema"}),
)

DEFAULT_PROFILE_CONFIG = ProfileTaskConfig()

MINIMAL_PROFILE_CONFIG = ProfileTaskConfig(
    include_histograms=False,
    sample_size=10000,
    tags=frozenset({"minimal"}),
)

FULL_PROFILE_CONFIG = ProfileTaskConfig(
    include_histograms=True,
    sample_size=None,
    tags=frozenset({"full"}),
)

DEFAULT_LEARN_CONFIG = LearnTaskConfig()

STRICT_LEARN_CONFIG = LearnTaskConfig(
    min_confidence=0.95,
    tags=frozenset({"strict"}),
)


@dataclass(frozen=True, slots=True)
class DriftTaskConfig(BaseTaskConfig):
    """Configuration for drift detection tasks.

    Attributes:
        method: Statistical method for drift detection.
        columns: Columns to check. None means all.
        threshold: Drift detection threshold.
        fail_on_drift: Raise exception on drift detection.
    """

    method: str = "auto"
    columns: tuple[str, ...] | None = None
    threshold: float | None = None
    fail_on_drift: bool = True

    def with_method(self, method: str) -> DriftTaskConfig:
        """Return a new config with method changed."""
        return DriftTaskConfig(**{**self.__dict__, "method": method})

    def with_columns(self, columns: Sequence[str]) -> DriftTaskConfig:
        """Return a new config with columns changed."""
        return DriftTaskConfig(**{**self.__dict__, "columns": tuple(columns)})

    def with_threshold(self, threshold: float) -> DriftTaskConfig:
        """Return a new config with threshold changed."""
        return DriftTaskConfig(**{**self.__dict__, "threshold": threshold})

    def with_fail_on_drift(self, fail_on_drift: bool) -> DriftTaskConfig:
        """Return a new config with fail_on_drift changed."""
        return DriftTaskConfig(**{**self.__dict__, "fail_on_drift": fail_on_drift})


DEFAULT_DRIFT_TASK_CONFIG = DriftTaskConfig()

STRICT_DRIFT_TASK_CONFIG = DriftTaskConfig(
    threshold=0.01,
    fail_on_drift=True,
    tags=frozenset({"strict"}),
)

LENIENT_DRIFT_TASK_CONFIG = DriftTaskConfig(
    threshold=0.1,
    fail_on_drift=False,
    tags=frozenset({"lenient"}),
)


@dataclass(frozen=True, slots=True)
class AnomalyTaskConfig(BaseTaskConfig):
    """Configuration for anomaly detection tasks.

    Attributes:
        detector: Anomaly detection algorithm.
        columns: Columns to check. None means all.
        contamination: Expected proportion of anomalies.
        fail_on_anomaly: Raise exception on anomaly detection.
    """

    detector: str = "isolation_forest"
    columns: tuple[str, ...] | None = None
    contamination: float = 0.05
    fail_on_anomaly: bool = True

    def with_detector(self, detector: str) -> AnomalyTaskConfig:
        """Return a new config with detector changed."""
        return AnomalyTaskConfig(**{**self.__dict__, "detector": detector})

    def with_columns(self, columns: Sequence[str]) -> AnomalyTaskConfig:
        """Return a new config with columns changed."""
        return AnomalyTaskConfig(**{**self.__dict__, "columns": tuple(columns)})

    def with_contamination(self, contamination: float) -> AnomalyTaskConfig:
        """Return a new config with contamination changed."""
        return AnomalyTaskConfig(**{**self.__dict__, "contamination": contamination})

    def with_fail_on_anomaly(self, fail_on_anomaly: bool) -> AnomalyTaskConfig:
        """Return a new config with fail_on_anomaly changed."""
        return AnomalyTaskConfig(**{**self.__dict__, "fail_on_anomaly": fail_on_anomaly})


DEFAULT_ANOMALY_TASK_CONFIG = AnomalyTaskConfig()

STRICT_ANOMALY_TASK_CONFIG = AnomalyTaskConfig(
    contamination=0.01,
    fail_on_anomaly=True,
    tags=frozenset({"strict"}),
)

LENIENT_ANOMALY_TASK_CONFIG = AnomalyTaskConfig(
    contamination=0.1,
    fail_on_anomaly=False,
    tags=frozenset({"lenient"}),
)


__all__ = [
    # Base
    "BaseTaskConfig",
    # Check
    "CheckTaskConfig",
    "DEFAULT_CHECK_CONFIG",
    "STRICT_CHECK_CONFIG",
    "LENIENT_CHECK_CONFIG",
    "AUTO_SCHEMA_CHECK_CONFIG",
    # Profile
    "ProfileTaskConfig",
    "DEFAULT_PROFILE_CONFIG",
    "MINIMAL_PROFILE_CONFIG",
    "FULL_PROFILE_CONFIG",
    # Learn
    "LearnTaskConfig",
    "DEFAULT_LEARN_CONFIG",
    "STRICT_LEARN_CONFIG",
    # Drift
    "DriftTaskConfig",
    "DEFAULT_DRIFT_TASK_CONFIG",
    "STRICT_DRIFT_TASK_CONFIG",
    "LENIENT_DRIFT_TASK_CONFIG",
    # Anomaly
    "AnomalyTaskConfig",
    "DEFAULT_ANOMALY_TASK_CONFIG",
    "STRICT_ANOMALY_TASK_CONFIG",
    "LENIENT_ANOMALY_TASK_CONFIG",
]
