"""Flow configurations for data quality operations.

This module provides immutable configuration classes for flows,
following the frozen dataclass pattern with builder methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from truthound_prefect.utils.types import QualityCheckMode


@dataclass(frozen=True, slots=True)
class FlowConfig:
    """Base configuration for all data quality flows.

    Attributes:
        enabled: Whether the flow is enabled.
        name: Name of the flow.
        description: Human-readable description.
        timeout_seconds: Timeout for the flow in seconds.
        retries: Number of retries on failure.
        retry_delay_seconds: Delay between retries.
        tags: Tags for categorization and filtering.
        log_prints: Whether to log print statements.
        validate_parameters: Whether to validate parameters.
    """

    enabled: bool = True
    name: str = ""
    description: str = ""
    timeout_seconds: float | None = None
    retries: int = 0
    retry_delay_seconds: float = 60.0
    tags: frozenset[str] = field(default_factory=frozenset)
    log_prints: bool = True
    validate_parameters: bool = True

    def with_name(self, name: str) -> FlowConfig:
        """Return a new config with name changed."""
        return FlowConfig(
            enabled=self.enabled,
            name=name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
        )

    def with_timeout(self, timeout_seconds: float) -> FlowConfig:
        """Return a new config with timeout changed."""
        return FlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
        )

    def with_retries(
        self,
        retries: int,
        retry_delay_seconds: float | None = None,
    ) -> FlowConfig:
        """Return a new config with retry settings changed."""
        return FlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds or self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
        )

    def with_tags(self, *tags: str) -> FlowConfig:
        """Return a new config with additional tags."""
        new_tags = self.tags | frozenset(tags)
        return FlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=new_tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
        )


@dataclass(frozen=True, slots=True)
class QualityFlowConfig(FlowConfig):
    """Configuration for quality-checked flows.

    Extends FlowConfig with quality-specific settings.

    Attributes:
        check_mode: When to perform quality checks (before, after, both, none).
        rules: Tuple of rules to check.
        fail_on_error: Raise exception on check failures.
        warning_threshold: Failure rate threshold for warnings (0.0 to 1.0).
        auto_schema: Use auto-schema mode (Truthound only).
        store_results: Store results as Prefect artifacts.
        engine_name: Name of the engine to use.
    """

    check_mode: QualityCheckMode = QualityCheckMode.AFTER
    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    fail_on_error: bool = True
    warning_threshold: float | None = None
    auto_schema: bool = False
    store_results: bool = True
    engine_name: str = "truthound"

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.warning_threshold is not None:
            if not 0.0 <= self.warning_threshold <= 1.0:
                raise ValueError("warning_threshold must be between 0.0 and 1.0")

    def with_check_mode(self, mode: QualityCheckMode | str) -> QualityFlowConfig:
        """Return a new config with check mode changed."""
        if isinstance(mode, str):
            mode = QualityCheckMode(mode)
        return QualityFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=mode,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
        )

    def with_rules(self, rules: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> QualityFlowConfig:
        """Return a new config with rules changed."""
        return QualityFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=tuple(rules),
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> QualityFlowConfig:
        """Return a new config with fail_on_error changed."""
        return QualityFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=self.rules,
            fail_on_error=fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
        )

    def with_warning_threshold(self, threshold: float | None) -> QualityFlowConfig:
        """Return a new config with warning_threshold changed."""
        return QualityFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
        )

    def with_auto_schema(self, auto_schema: bool = True) -> QualityFlowConfig:
        """Return a new config with auto_schema changed."""
        return QualityFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
        )

    def with_engine(self, engine_name: str) -> QualityFlowConfig:
        """Return a new config with engine changed."""
        return QualityFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=engine_name,
        )


@dataclass(frozen=True, slots=True)
class PipelineFlowConfig(QualityFlowConfig):
    """Configuration for data pipeline flows with quality checks.

    Extends QualityFlowConfig with pipeline-specific settings.

    Attributes:
        profile_data: Whether to profile data.
        learn_rules: Whether to learn rules from baseline data.
        parallel_checks: Run checks in parallel.
        max_workers: Maximum parallel workers.
        checkpoint_results: Save intermediate results.
    """

    profile_data: bool = False
    learn_rules: bool = False
    parallel_checks: bool = False
    max_workers: int = 4
    checkpoint_results: bool = True

    def with_profiling(self, profile_data: bool = True) -> PipelineFlowConfig:
        """Return a new config with profiling enabled/disabled."""
        return PipelineFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
            profile_data=profile_data,
            learn_rules=self.learn_rules,
            parallel_checks=self.parallel_checks,
            max_workers=self.max_workers,
            checkpoint_results=self.checkpoint_results,
        )

    def with_learning(self, learn_rules: bool = True) -> PipelineFlowConfig:
        """Return a new config with rule learning enabled/disabled."""
        return PipelineFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
            profile_data=self.profile_data,
            learn_rules=learn_rules,
            parallel_checks=self.parallel_checks,
            max_workers=self.max_workers,
            checkpoint_results=self.checkpoint_results,
        )

    def with_parallel(
        self,
        parallel: bool = True,
        max_workers: int = 4,
    ) -> PipelineFlowConfig:
        """Return a new config with parallel settings changed."""
        return PipelineFlowConfig(
            enabled=self.enabled,
            name=self.name,
            description=self.description,
            timeout_seconds=self.timeout_seconds,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            tags=self.tags,
            log_prints=self.log_prints,
            validate_parameters=self.validate_parameters,
            check_mode=self.check_mode,
            rules=self.rules,
            fail_on_error=self.fail_on_error,
            warning_threshold=self.warning_threshold,
            auto_schema=self.auto_schema,
            store_results=self.store_results,
            engine_name=self.engine_name,
            profile_data=self.profile_data,
            learn_rules=self.learn_rules,
            parallel_checks=parallel,
            max_workers=max_workers,
            checkpoint_results=self.checkpoint_results,
        )


# Preset configurations
DEFAULT_FLOW_CONFIG = FlowConfig()

DEFAULT_QUALITY_FLOW_CONFIG = QualityFlowConfig()

STRICT_QUALITY_FLOW_CONFIG = QualityFlowConfig(
    check_mode=QualityCheckMode.AFTER,
    fail_on_error=True,
    warning_threshold=None,
    tags=frozenset({"strict", "data-quality"}),
)

LENIENT_QUALITY_FLOW_CONFIG = QualityFlowConfig(
    check_mode=QualityCheckMode.AFTER,
    fail_on_error=False,
    warning_threshold=0.10,
    tags=frozenset({"lenient", "data-quality"}),
)

AUTO_SCHEMA_FLOW_CONFIG = QualityFlowConfig(
    check_mode=QualityCheckMode.AFTER,
    auto_schema=True,
    engine_name="truthound",
    tags=frozenset({"auto-schema", "data-quality"}),
)

DEFAULT_PIPELINE_CONFIG = PipelineFlowConfig()

FULL_PIPELINE_CONFIG = PipelineFlowConfig(
    profile_data=True,
    learn_rules=False,
    parallel_checks=True,
    max_workers=4,
    store_results=True,
    tags=frozenset({"full", "pipeline"}),
)


__all__ = [
    # Configs
    "FlowConfig",
    "QualityFlowConfig",
    "PipelineFlowConfig",
    # Presets
    "DEFAULT_FLOW_CONFIG",
    "DEFAULT_QUALITY_FLOW_CONFIG",
    "STRICT_QUALITY_FLOW_CONFIG",
    "LENIENT_QUALITY_FLOW_CONFIG",
    "AUTO_SCHEMA_FLOW_CONFIG",
    "DEFAULT_PIPELINE_CONFIG",
    "FULL_PIPELINE_CONFIG",
]
