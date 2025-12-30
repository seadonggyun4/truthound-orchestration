"""Truthound Engine implementation.

This module provides the TruthoundEngine class, the default data quality
engine that wraps the Truthound library for validation, profiling, and
rule learning operations.

The TruthoundEngine supports full lifecycle management including:
- Explicit start/stop lifecycle
- Health checking
- Resource management via context managers
- State introspection

Example:
    >>> from common.engines import TruthoundEngine
    >>> engine = TruthoundEngine()
    >>> result = engine.check(data=df)
    >>> print(result.status)

    >>> # With lifecycle management
    >>> with TruthoundEngine() as engine:
    ...     result = engine.check(data=df)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from common.engines.lifecycle import EngineStateSnapshot

from common.base import (
    CheckResult,
    CheckStatus,
    ColumnProfile,
    LearnedRule,
    LearnResult,
    LearnStatus,
    ProfileResult,
    ProfileStatus,
    Severity,
    ValidationFailure,
)
from common.engines.base import EngineCapabilities, EngineInfoMixin
from common.engines.lifecycle import (
    EngineAlreadyStartedError,
    EngineConfig,
    EngineInitializationError,
    EngineShutdownError,
    EngineState,
    EngineStateTracker,
    EngineStoppedError,
)
from common.exceptions import ValidationExecutionError
from common.health import HealthCheckResult, HealthStatus


# =============================================================================
# Severity Mapping
# =============================================================================

SEVERITY_MAPPING: dict[str, Severity] = {
    "critical": Severity.CRITICAL,
    "high": Severity.ERROR,
    "medium": Severity.WARNING,
    "low": Severity.INFO,
    "info": Severity.INFO,
}


# =============================================================================
# Engine Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class TruthoundEngineConfig(EngineConfig):
    """Configuration for TruthoundEngine.

    Extends base EngineConfig with Truthound-specific settings.

    Attributes:
        parallel: Whether to run validators in parallel by default.
        max_workers: Maximum number of parallel workers.
        min_severity: Minimum severity level to report (critical, high, medium, low).
        cache_schemas: Whether to cache learned schemas.
        infer_constraints: Whether to infer value constraints during learn.
        categorical_threshold: Max unique values to consider as categorical.

    Example:
        >>> config = TruthoundEngineConfig(
        ...     parallel=True,
        ...     max_workers=4,
        ...     auto_start=True,
        ... )
        >>> engine = TruthoundEngine(config=config)

        >>> # Using builder pattern
        >>> config = (
        ...     TruthoundEngineConfig()
        ...     .with_parallel(True, max_workers=4)
        ...     .with_min_severity("medium")
        ...     .with_auto_start(True)
        ... )
    """

    parallel: bool = False
    max_workers: int | None = None
    min_severity: str | None = None
    cache_schemas: bool = True
    infer_constraints: bool = True
    categorical_threshold: int = 20

    def __post_init__(self) -> None:
        """Validate Truthound-specific configuration."""
        self._validate_base_config()
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.min_severity is not None:
            valid_severities = {"critical", "high", "medium", "low"}
            if self.min_severity.lower() not in valid_severities:
                raise ValueError(
                    f"min_severity must be one of {valid_severities}, "
                    f"got '{self.min_severity}'"
                )
        if self.categorical_threshold < 1:
            raise ValueError("categorical_threshold must be at least 1")

    def with_parallel(
        self,
        enabled: bool = True,
        max_workers: int | None = None,
    ) -> Self:
        """Create config with parallel settings.

        Args:
            enabled: Whether to enable parallel processing.
            max_workers: Maximum number of worker threads.

        Returns:
            New configuration with parallel settings.
        """
        return self._copy_with(
            parallel=enabled,
            max_workers=max_workers if max_workers is not None else self.max_workers,
        )

    def with_min_severity(self, severity: str) -> Self:
        """Create config with minimum severity.

        Args:
            severity: Minimum severity level (critical, high, medium, low).

        Returns:
            New configuration with severity setting.
        """
        return self._copy_with(min_severity=severity)

    def with_cache_schemas(self, enabled: bool) -> Self:
        """Create config with schema caching setting.

        Args:
            enabled: Whether to cache learned schemas.

        Returns:
            New configuration with caching setting.
        """
        return self._copy_with(cache_schemas=enabled)

    def with_infer_constraints(self, enabled: bool) -> Self:
        """Create config with constraint inference setting.

        Args:
            enabled: Whether to infer value constraints during learn.

        Returns:
            New configuration with inference setting.
        """
        return self._copy_with(infer_constraints=enabled)

    def with_categorical_threshold(self, threshold: int) -> Self:
        """Create config with categorical threshold.

        Args:
            threshold: Maximum unique values to consider as categorical.

        Returns:
            New configuration with threshold setting.
        """
        return self._copy_with(categorical_threshold=threshold)


# Default Truthound configurations
DEFAULT_TRUTHOUND_CONFIG = TruthoundEngineConfig()

PARALLEL_TRUTHOUND_CONFIG = TruthoundEngineConfig(
    parallel=True,
    max_workers=4,
)

PRODUCTION_TRUTHOUND_CONFIG = TruthoundEngineConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    parallel=True,
    max_workers=4,
    min_severity="medium",
)


# =============================================================================
# Truthound Engine
# =============================================================================


class TruthoundEngine(EngineInfoMixin):
    """Default data quality engine using the Truthound library.

    This engine provides a standardized interface to the Truthound
    data quality library, supporting validation checks, data profiling,
    and automatic rule learning.

    Truthound is a zero-configuration data quality validation framework
    designed for modern data engineering pipelines, leveraging Polars
    for high-performance data validation.

    The engine implements the ManagedEngine protocol for lifecycle management:
    - start(): Initialize the Truthound library
    - stop(): Release resources
    - health_check(): Verify engine health
    - Context manager support for automatic resource cleanup

    Attributes:
        _truthound: Lazily loaded Truthound library instance.
        _config: Engine configuration.
        _state_tracker: State tracking for lifecycle management.

    Example:
        >>> engine = TruthoundEngine()
        >>> result = engine.check(data=df)
        >>> if result.is_success:
        ...     print("All validations passed!")

        >>> # With lifecycle management
        >>> with TruthoundEngine(config=PRODUCTION_TRUTHOUND_CONFIG) as engine:
        ...     result = engine.check(data=df)

    Note:
        The Truthound library is lazily imported to avoid requiring it
        as a hard dependency. If Truthound is not installed, an error
        is raised when attempting operations.
    """

    def __init__(
        self,
        config: TruthoundEngineConfig | None = None,
    ) -> None:
        """Initialize the Truthound engine.

        Args:
            config: Engine configuration. Defaults to DEFAULT_TRUTHOUND_CONFIG.
        """
        self._config = config or DEFAULT_TRUTHOUND_CONFIG
        self._truthound: Any | None = None
        self._version: str | None = None
        self._state_tracker = EngineStateTracker("truthound")
        self._lock = threading.RLock()
        self._schema_cache: dict[int, Any] = {}

        # Auto-start if configured
        if self._config.auto_start:
            self.start()

    @property
    def engine_name(self) -> str:
        """Return the engine name."""
        return "truthound"

    @property
    def engine_version(self) -> str:
        """Return the engine version.

        Returns the version of the Truthound library if available,
        otherwise returns "0.0.0".
        """
        if self._version is None:
            try:
                import truthound

                self._version = getattr(truthound, "__version__", "0.0.0")
            except ImportError:
                self._version = "0.0.0"
        return self._version

    def _get_capabilities(self) -> EngineCapabilities:
        """Return Truthound capabilities."""
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
            supports_async=False,
            supports_streaming=True,
            supported_data_types=("polars", "pandas", "csv", "parquet", "sql"),
            supported_rule_types=(
                "null",
                "invalid_format",
                "duplicate",
                "outlier",
                "range",
                "pattern",
                "schema",
            ),
        )

    def _get_description(self) -> str:
        """Return engine description."""
        return (
            "Truthound: Zero-configuration data quality validation framework "
            "with 275+ validators and high-performance Polars backend"
        )

    def _get_homepage(self) -> str:
        """Return engine homepage."""
        return "https://github.com/seadonggyun4/Truthound"

    # =========================================================================
    # Lifecycle Management (ManagedEngine Protocol)
    # =========================================================================

    @property
    def config(self) -> TruthoundEngineConfig:
        """Return engine configuration."""
        return self._config

    def start(self) -> None:
        """Start the engine and initialize resources.

        Initializes the Truthound library and prepares the engine for use.

        Raises:
            EngineAlreadyStartedError: If engine is already running.
            EngineStoppedError: If engine was stopped.
            EngineInitializationError: If initialization fails.

        Example:
            >>> engine = TruthoundEngine()
            >>> engine.start()
            >>> result = engine.check(data)
            >>> engine.stop()
        """
        with self._lock:
            state = self._state_tracker.state

            if state.is_terminal:
                raise EngineStoppedError("truthound")
            if state.is_active:
                raise EngineAlreadyStartedError("truthound")

            self._state_tracker.transition_to(EngineState.STARTING)

            try:
                # Pre-load Truthound library
                self._ensure_truthound()
                self._state_tracker.transition_to(EngineState.RUNNING)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineInitializationError(
                    f"Failed to start Truthound engine: {e}",
                    engine_name="truthound",
                    cause=e,
                ) from e

    def stop(self) -> None:
        """Stop the engine and cleanup resources.

        Clears cached schemas and releases the Truthound library reference.

        Raises:
            EngineShutdownError: If shutdown fails.

        Example:
            >>> engine.stop()
        """
        with self._lock:
            state = self._state_tracker.state

            if state == EngineState.STOPPED:
                return
            if not state.can_stop:
                return

            self._state_tracker.transition_to(EngineState.STOPPING)

            try:
                # Clear caches
                self._schema_cache.clear()
                # Release library reference (allows garbage collection)
                self._truthound = None
                self._version = None
                self._state_tracker.transition_to(EngineState.STOPPED)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineShutdownError(
                    f"Failed to stop Truthound engine: {e}",
                    engine_name="truthound",
                    cause=e,
                ) from e

    def health_check(self) -> HealthCheckResult:
        """Perform health check on the engine.

        Verifies:
        - Engine is in RUNNING state
        - Truthound library is accessible
        - Basic validation can be performed

        Returns:
            HealthCheckResult with engine health status.

        Example:
            >>> result = engine.health_check()
            >>> if result.is_healthy:
            ...     print("Engine is healthy")
        """
        start_time = time.perf_counter()
        state = self._state_tracker.state

        # Check state
        if state != EngineState.RUNNING:
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
                duration_ms=(time.perf_counter() - start_time) * 1000,
                details={"state": state.name},
            )
            self._state_tracker.record_health_check(result.status)
            return result

        try:
            # Verify Truthound is accessible
            truthound = self._ensure_truthound()

            # Check version is accessible
            version = getattr(truthound, "__version__", "unknown")

            duration_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.HEALTHY,
                message=f"Truthound v{version} is operational",
                duration_ms=duration_ms,
                details={
                    "version": version,
                    "state": state.name,
                    "schema_cache_size": len(self._schema_cache),
                },
            )
            self._state_tracker.record_health_check(result.status)
            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                duration_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__},
            )
            self._state_tracker.record_health_check(result.status)
            return result

    def get_state(self) -> EngineState:
        """Return current engine state.

        Returns:
            Current EngineState.
        """
        return self._state_tracker.state

    def get_state_snapshot(self) -> EngineStateSnapshot:
        """Return snapshot of engine state.

        Returns:
            EngineStateSnapshot with full state details.
        """
        return self._state_tracker.get_snapshot()

    def __enter__(self) -> Self:
        """Enter context manager, starting engine if needed.

        Returns:
            Self for use in with statement.
        """
        if self._state_tracker.state == EngineState.CREATED:
            self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager, stopping engine if configured."""
        if self._config.auto_stop:
            self.stop()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _ensure_truthound(self) -> Any:
        """Lazily import and return the Truthound library.

        Returns:
            Truthound module.

        Raises:
            ValidationExecutionError: If Truthound is not installed.
        """
        if self._truthound is None:
            try:
                import truthound

                self._truthound = truthound
            except ImportError as e:
                raise ValidationExecutionError(
                    "Truthound library is not installed. "
                    "Install it with: pip install truthound",
                    cause=e,
                ) from e
        return self._truthound

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]] | None = None,
        *,
        fail_on_error: bool = True,
        schema: Any | None = None,
        auto_schema: bool = False,
        parallel: bool = False,
        max_workers: int | None = None,
        min_severity: str | None = None,
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation checks on the data.

        Args:
            data: Data to validate (Polars DataFrame, Pandas DataFrame, or file path).
            rules: Optional sequence of validation rule dictionaries (not used by Truthound).
            fail_on_error: Whether to fail on execution errors.
            schema: Schema to validate against (learned from th.learn()).
            auto_schema: Whether to auto-generate schema from data.
            parallel: Whether to run validators in parallel.
            max_workers: Maximum number of parallel workers.
            min_severity: Minimum severity level to report.
            **kwargs: Additional Truthound-specific parameters.

        Returns:
            CheckResult with validation outcomes.

        Raises:
            ValidationExecutionError: If validation execution fails.

        Example:
            >>> result = engine.check(data=df)
            >>> result = engine.check(data=df, schema=learned_schema)
        """
        start_time = time.perf_counter()

        try:
            truthound = self._ensure_truthound()

            # Build Truthound kwargs
            th_kwargs: dict[str, Any] = {
                "parallel": parallel,
            }
            if schema is not None:
                th_kwargs["schema"] = schema
            if auto_schema:
                th_kwargs["auto_schema"] = auto_schema
            if max_workers is not None:
                th_kwargs["max_workers"] = max_workers
            if min_severity is not None:
                th_kwargs["min_severity"] = min_severity
            th_kwargs.update(kwargs)

            # Execute validation
            report = truthound.check(data, **th_kwargs)

            # Convert to CheckResult
            return self._convert_check_result(report, start_time)

        except ValidationExecutionError:
            raise
        except Exception as e:
            if fail_on_error:
                raise ValidationExecutionError(
                    f"Truthound check failed: {e}",
                    cause=e,
                ) from e
            # Return error result
            return CheckResult(
                status=CheckStatus.ERROR,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    def _convert_check_result(
        self,
        report: Any,
        start_time: float,
    ) -> CheckResult:
        """Convert Truthound Report to CheckResult.

        Args:
            report: Report from Truthound check.
            start_time: Start time for execution timing.

        Returns:
            CheckResult instance.
        """
        failures: list[ValidationFailure] = []
        failed = 0
        warnings = 0

        # Get issues from report
        report_dict = report.to_dict()
        issues = report_dict.get("issues", [])

        for issue in issues:
            severity_str = issue.get("severity", "high")
            severity = SEVERITY_MAPPING.get(severity_str, Severity.ERROR)

            if severity in (Severity.CRITICAL, Severity.ERROR):
                failed += 1
            else:
                warnings += 1

            failures.append(
                ValidationFailure(
                    rule_name=issue.get("issue_type", "unknown"),
                    column=issue.get("column"),
                    message=issue.get("details", ""),
                    severity=severity,
                    failed_count=issue.get("count", 0),
                    total_count=report_dict.get("row_count", 0),
                )
            )

        # Determine status
        if report.has_critical or failed > 0:
            status = CheckStatus.FAILED
        elif warnings > 0:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.PASSED

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Calculate passed count (columns without issues)
        total_columns = report_dict.get("column_count", 0)
        columns_with_issues = len({f.column for f in failures if f.column})
        passed = total_columns - columns_with_issues

        return CheckResult(
            status=status,
            passed_count=max(passed, 0),
            failed_count=failed,
            warning_count=warnings,
            failures=tuple(failures),
            execution_time_ms=execution_time_ms,
            metadata={
                "engine": self.engine_name,
                "source": report_dict.get("source", "unknown"),
                "row_count": report_dict.get("row_count", 0),
                "column_count": report_dict.get("column_count", 0),
            },
        )

    def profile(
        self,
        data: Any,
        *,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ProfileResult:
        """Profile the data to understand its characteristics.

        Args:
            data: Data to profile (Polars DataFrame, Pandas DataFrame, or file path).
            columns: Specific columns to profile (None = all).
            **kwargs: Additional Truthound-specific parameters.

        Returns:
            ProfileResult with profiling outcomes.

        Raises:
            ValidationExecutionError: If profiling execution fails.

        Example:
            >>> result = engine.profile(data=df)
            >>> for col in result.columns:
            ...     print(f"{col.column_name}: {col.dtype}")
        """
        start_time = time.perf_counter()

        try:
            truthound = self._ensure_truthound()

            # Execute profiling
            profile_report = truthound.profile(data)

            # Convert to ProfileResult
            return self._convert_profile_result(profile_report, start_time, columns)

        except ValidationExecutionError:
            raise
        except Exception as e:
            raise ValidationExecutionError(
                f"Truthound profile failed: {e}",
                cause=e,
            ) from e

    def _convert_profile_result(
        self,
        profile_report: Any,
        start_time: float,
        columns_filter: Sequence[str] | None = None,
    ) -> ProfileResult:
        """Convert Truthound ProfileReport to ProfileResult.

        Args:
            profile_report: ProfileReport from Truthound.
            start_time: Start time for execution timing.
            columns_filter: Optional filter for specific columns.

        Returns:
            ProfileResult instance.
        """
        column_profiles: list[ColumnProfile] = []
        report_dict = profile_report.to_dict()

        for col_data in report_dict.get("columns", []):
            col_name = col_data.get("name", "unknown")

            # Skip if not in filter
            if columns_filter and col_name not in columns_filter:
                continue

            # Parse percentages
            null_pct_str = col_data.get("null_pct", "0.0%")
            null_pct = float(null_pct_str.replace("%", "")) if null_pct_str else 0.0

            unique_pct_str = col_data.get("unique_pct", "0%")
            unique_pct = float(unique_pct_str.replace("%", "")) if unique_pct_str else 0.0

            # Parse min/max values
            min_val = col_data.get("min")
            max_val = col_data.get("max")
            if min_val == "-":
                min_val = None
            if max_val == "-":
                max_val = None

            # Calculate counts from percentages
            row_count = report_dict.get("row_count", 0)
            null_count = int(row_count * null_pct / 100) if row_count > 0 else 0
            unique_count = int(row_count * unique_pct / 100) if row_count > 0 else 0

            column_profiles.append(
                ColumnProfile(
                    column_name=col_name,
                    dtype=col_data.get("dtype", "unknown"),
                    null_count=null_count,
                    null_percentage=null_pct,
                    unique_count=unique_count,
                    unique_percentage=unique_pct,
                    min_value=min_val,
                    max_value=max_val,
                )
            )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return ProfileResult(
            status=ProfileStatus.COMPLETED,
            row_count=report_dict.get("row_count", 0),
            column_count=len(column_profiles),
            columns=tuple(column_profiles),
            execution_time_ms=execution_time_ms,
            metadata={
                "engine": self.engine_name,
                "source": report_dict.get("source", "unknown"),
                "size_bytes": report_dict.get("size_bytes"),
            },
        )

    def learn(
        self,
        data: Any,
        *,
        columns: Sequence[str] | None = None,
        infer_constraints: bool = True,
        categorical_threshold: int = 20,
        confidence_threshold: float = 0.95,
        **kwargs: Any,
    ) -> LearnResult:
        """Learn validation rules (schema) from the data.

        Args:
            data: Data to learn from (Polars DataFrame, Pandas DataFrame, or file path).
            columns: Specific columns to learn from (None = all).
            infer_constraints: Whether to infer value constraints.
            categorical_threshold: Max unique values to consider as categorical.
            confidence_threshold: Minimum confidence for learned rules.
            **kwargs: Additional Truthound-specific parameters.

        Returns:
            LearnResult with learned rules.

        Raises:
            ValidationExecutionError: If learning execution fails.

        Example:
            >>> result = engine.learn(data=df)
            >>> for rule in result.rules:
            ...     print(f"{rule.column}: {rule.rule_type}")
        """
        start_time = time.perf_counter()

        try:
            truthound = self._ensure_truthound()

            # Execute learning
            schema = truthound.learn(
                data,
                infer_constraints=infer_constraints,
                categorical_threshold=categorical_threshold,
            )

            # Convert to LearnResult
            return self._convert_learn_result(
                schema, start_time, columns, confidence_threshold
            )

        except ValidationExecutionError:
            raise
        except Exception as e:
            raise ValidationExecutionError(
                f"Truthound learn failed: {e}",
                cause=e,
            ) from e

    def _convert_learn_result(
        self,
        schema: Any,
        start_time: float,
        columns_filter: Sequence[str] | None = None,
        confidence_threshold: float = 0.95,
    ) -> LearnResult:
        """Convert Truthound Schema to LearnResult.

        Args:
            schema: Schema from Truthound learn.
            start_time: Start time for execution timing.
            columns_filter: Optional filter for specific columns.
            confidence_threshold: Minimum confidence for learned rules.

        Returns:
            LearnResult instance.
        """
        learned_rules: list[LearnedRule] = []
        schema_dict = schema.to_dict()
        columns_data = schema_dict.get("columns", {})
        row_count = schema_dict.get("row_count", 0)

        for col_name, col_schema in columns_data.items():
            # Skip if not in filter
            if columns_filter and col_name not in columns_filter:
                continue

            # Learn dtype rule
            if col_schema.get("dtype"):
                learned_rules.append(
                    LearnedRule(
                        rule_type="dtype",
                        column=col_name,
                        parameters={"dtype": col_schema["dtype"]},
                        confidence=1.0,
                        sample_size=row_count,
                    )
                )

            # Learn nullable rule
            null_ratio = col_schema.get("null_ratio", 0.0)
            if null_ratio < (1 - confidence_threshold):
                learned_rules.append(
                    LearnedRule(
                        rule_type="not_null",
                        column=col_name,
                        confidence=1 - null_ratio,
                        sample_size=row_count,
                    )
                )

            # Learn unique rule
            unique_ratio = col_schema.get("unique_ratio", 0.0)
            if unique_ratio >= confidence_threshold:
                learned_rules.append(
                    LearnedRule(
                        rule_type="unique",
                        column=col_name,
                        confidence=unique_ratio,
                        sample_size=row_count,
                    )
                )

            # Learn range rule for numeric columns
            min_val = col_schema.get("min_value")
            max_val = col_schema.get("max_value")
            if min_val is not None and max_val is not None:
                learned_rules.append(
                    LearnedRule(
                        rule_type="in_range",
                        column=col_name,
                        parameters={"min": min_val, "max": max_val},
                        confidence=1.0,
                        sample_size=row_count,
                    )
                )

            # Learn allowed values for categorical columns
            allowed_values = col_schema.get("allowed_values")
            if allowed_values and len(allowed_values) <= 20:
                learned_rules.append(
                    LearnedRule(
                        rule_type="in_set",
                        column=col_name,
                        parameters={"values": allowed_values},
                        confidence=1.0,
                        sample_size=row_count,
                    )
                )

            # Learn string length constraints
            min_length = col_schema.get("min_length")
            max_length = col_schema.get("max_length")
            if min_length is not None:
                learned_rules.append(
                    LearnedRule(
                        rule_type="min_length",
                        column=col_name,
                        parameters={"value": min_length},
                        confidence=1.0,
                        sample_size=row_count,
                    )
                )
            if max_length is not None:
                learned_rules.append(
                    LearnedRule(
                        rule_type="max_length",
                        column=col_name,
                        parameters={"value": max_length},
                        confidence=1.0,
                        sample_size=row_count,
                    )
                )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return LearnResult(
            status=LearnStatus.COMPLETED,
            rules=tuple(learned_rules),
            columns_analyzed=len(columns_data),
            execution_time_ms=execution_time_ms,
            metadata={
                "engine": self.engine_name,
                "schema_version": schema_dict.get("version", "1.0"),
                "row_count": row_count,
            },
        )

    def get_schema(self, data: Any, **kwargs: Any) -> Any:
        """Get the learned schema object for use with check().

        This returns the raw Truthound Schema object which can be
        passed to check() for validation.

        Args:
            data: Data to learn schema from.
            **kwargs: Additional parameters for learn().

        Returns:
            Truthound Schema object.

        Example:
            >>> schema = engine.get_schema(baseline_df)
            >>> result = engine.check(new_df, schema=schema)
        """
        truthound = self._ensure_truthound()
        return truthound.learn(data, **kwargs)
