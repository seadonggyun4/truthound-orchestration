"""Asset Configuration Types for Dagster Integration.

This module provides configuration types for quality-aware assets.

Example:
    >>> from truthound_dagster.assets import QualityAssetConfig
    >>>
    >>> config = QualityAssetConfig(
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     fail_on_error=True,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QualityCheckMode(Enum):
    """When to run quality checks.

    Attributes:
        BEFORE: Check data before returning from asset.
        AFTER: Check data after asset completes.
        BOTH: Check before and after.
        NONE: Skip quality checks.
    """

    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class QualityAssetConfig:
    """Configuration for quality-checked assets.

    Attributes:
        rules: Validation rules to apply.
        check_mode: When to run quality checks.
        fail_on_error: Whether to fail asset on quality failure.
        warning_threshold: Failure rate threshold for warning.
        auto_schema: Auto-generate schema from data.
        store_result: Store quality result in metadata.
        result_metadata_key: Key for storing result in metadata.

    Example:
        >>> config = QualityAssetConfig(
        ...     rules=[
        ...         {"column": "id", "type": "not_null"},
        ...         {"column": "email", "type": "unique"},
        ...     ],
        ...     fail_on_error=True,
        ... )
    """

    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    check_mode: QualityCheckMode = QualityCheckMode.AFTER
    fail_on_error: bool = True
    warning_threshold: float | None = None
    auto_schema: bool = False
    store_result: bool = True
    result_metadata_key: str = "quality_result"
    timeout_seconds: float = 300.0
    tags: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.warning_threshold is not None:
            if not 0 <= self.warning_threshold <= 1:
                msg = "warning_threshold must be between 0 and 1"
                raise ValueError(msg)

    def with_rules(self, rules: list[dict[str, Any]]) -> QualityAssetConfig:
        """Return new config with updated rules."""
        return QualityAssetConfig(
            rules=tuple(rules),
            check_mode=self.check_mode,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_result=self.store_result,
            result_metadata_key=self.result_metadata_key,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
        )

    def with_check_mode(self, mode: QualityCheckMode) -> QualityAssetConfig:
        """Return new config with updated check mode."""
        return QualityAssetConfig(
            rules=self.rules,
            check_mode=mode,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_result=self.store_result,
            result_metadata_key=self.result_metadata_key,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> QualityAssetConfig:
        """Return new config with updated fail_on_error."""
        return QualityAssetConfig(
            rules=self.rules,
            check_mode=self.check_mode,
            fail_on_error=fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_result=self.store_result,
            result_metadata_key=self.result_metadata_key,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
        )

    def with_auto_schema(self, auto_schema: bool) -> QualityAssetConfig:
        """Return new config with updated auto_schema."""
        return QualityAssetConfig(
            rules=self.rules,
            check_mode=self.check_mode,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=auto_schema,
            store_result=self.store_result,
            result_metadata_key=self.result_metadata_key,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rules": list(self.rules),
            "check_mode": self.check_mode.value,
            "fail_on_error": self.fail_on_error,
            "warning_threshold": self.warning_threshold,
            "auto_schema": self.auto_schema,
            "store_result": self.store_result,
            "result_metadata_key": self.result_metadata_key,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class ProfileAssetConfig:
    """Configuration for profiled assets.

    Attributes:
        include_histograms: Include histogram data.
        include_samples: Include sample values.
        sample_size: Number of sample values.
        store_result: Store profile in metadata.
        result_metadata_key: Key for storing profile.

    Example:
        >>> config = ProfileAssetConfig(
        ...     include_histograms=True,
        ...     sample_size=10,
        ... )
    """

    include_histograms: bool = True
    include_samples: bool = True
    sample_size: int = 10
    store_result: bool = True
    result_metadata_key: str = "profile_result"
    timeout_seconds: float = 300.0
    tags: frozenset[str] = field(default_factory=frozenset)

    def with_histograms(self, include: bool) -> ProfileAssetConfig:
        """Return new config with updated histogram setting."""
        return ProfileAssetConfig(
            include_histograms=include,
            include_samples=self.include_samples,
            sample_size=self.sample_size,
            store_result=self.store_result,
            result_metadata_key=self.result_metadata_key,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
        )

    def with_samples(
        self,
        include: bool,
        sample_size: int | None = None,
    ) -> ProfileAssetConfig:
        """Return new config with updated sample settings."""
        return ProfileAssetConfig(
            include_histograms=self.include_histograms,
            include_samples=include,
            sample_size=sample_size if sample_size is not None else self.sample_size,
            store_result=self.store_result,
            result_metadata_key=self.result_metadata_key,
            timeout_seconds=self.timeout_seconds,
            tags=self.tags,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "include_histograms": self.include_histograms,
            "include_samples": self.include_samples,
            "sample_size": self.sample_size,
            "store_result": self.store_result,
            "result_metadata_key": self.result_metadata_key,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
        }


# Preset configurations
DEFAULT_QUALITY_CONFIG = QualityAssetConfig()

STRICT_QUALITY_CONFIG = QualityAssetConfig(
    fail_on_error=True,
    warning_threshold=None,
    check_mode=QualityCheckMode.AFTER,
)

LENIENT_QUALITY_CONFIG = QualityAssetConfig(
    fail_on_error=False,
    warning_threshold=0.10,
    check_mode=QualityCheckMode.AFTER,
)

AUTO_SCHEMA_QUALITY_CONFIG = QualityAssetConfig(
    auto_schema=True,
    fail_on_error=True,
    check_mode=QualityCheckMode.AFTER,
)

DEFAULT_PROFILE_CONFIG = ProfileAssetConfig()

MINIMAL_PROFILE_CONFIG = ProfileAssetConfig(
    include_histograms=False,
    include_samples=False,
)
