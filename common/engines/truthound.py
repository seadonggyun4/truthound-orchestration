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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from common.engines.lifecycle import EngineStateSnapshot

from common.base import (
    AnomalyResult,
    AnomalyScore,
    AnomalyStatus,
    CheckResult,
    CheckStatus,
    ColumnDrift,
    ColumnProfile,
    DriftMethod,
    DriftResult,
    DriftStatus,
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
    default_drift_method: str = "auto"
    default_drift_threshold: float | None = None
    default_anomaly_detector: str = "isolation_forest"
    default_contamination: float = 0.05

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
        valid_drift_methods = {m.value for m in DriftMethod}
        if self.default_drift_method not in valid_drift_methods:
            raise ValueError(
                f"default_drift_method must be one of {valid_drift_methods}, "
                f"got '{self.default_drift_method}'"
            )
        valid_detectors = {"isolation_forest", "z_score", "lof", "ensemble"}
        if self.default_anomaly_detector not in valid_detectors:
            raise ValueError(
                f"default_anomaly_detector must be one of {valid_detectors}, "
                f"got '{self.default_anomaly_detector}'"
            )
        if not (0.0 < self.default_contamination < 0.5):
            raise ValueError(
                f"default_contamination must be between 0 and 0.5, "
                f"got {self.default_contamination}"
            )

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

    def with_drift_defaults(
        self,
        method: str = "auto",
        threshold: float | None = None,
    ) -> Self:
        """Create config with drift detection defaults.

        Args:
            method: Default drift method (e.g., "ks", "psi", "auto").
            threshold: Default drift threshold.

        Returns:
            New configuration with drift defaults.
        """
        return self._copy_with(
            default_drift_method=method,
            default_drift_threshold=threshold,
        )

    def with_anomaly_defaults(
        self,
        detector: str = "isolation_forest",
        contamination: float = 0.05,
    ) -> Self:
        """Create config with anomaly detection defaults.

        Args:
            detector: Default anomaly detector.
            contamination: Default contamination rate.

        Returns:
            New configuration with anomaly defaults.
        """
        return self._copy_with(
            default_anomaly_detector=detector,
            default_contamination=contamination,
        )


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
            supports_drift=True,
            supports_anomaly=True,
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
            supported_drift_methods=tuple(m.value for m in DriftMethod),
            supported_anomaly_detectors=(
                "isolation_forest",
                "z_score",
                "lof",
                "ensemble",
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

    # =========================================================================
    # Drift Detection (DriftDetectionEngine Protocol)
    # =========================================================================

    def detect_drift(
        self,
        baseline: Any,
        current: Any,
        *,
        method: str | None = None,
        columns: Sequence[str] | None = None,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> DriftResult:
        """Detect data drift between baseline and current datasets.

        Uses Truthound's ``compare()`` API to compute statistical drift
        across columns using one of 14 supported methods.

        Args:
            baseline: Baseline dataset (DataFrame or file path).
            current: Current dataset to compare against baseline.
            method: Statistical method (e.g., "ks", "psi", "auto").
                Falls back to config default if None.
            columns: Specific columns to check (None = all).
            threshold: Drift threshold (None = method default).
                Falls back to config default if None.
            **kwargs: Additional Truthound-specific parameters.

        Returns:
            DriftResult with per-column drift outcomes.

        Raises:
            ValidationExecutionError: If drift detection fails.
        """
        start_time = time.perf_counter()
        resolved_method = method or self._config.default_drift_method
        resolved_threshold = threshold or self._config.default_drift_threshold

        try:
            truthound = self._ensure_truthound()

            compare_kwargs: dict[str, Any] = {"method": resolved_method}
            if columns is not None:
                compare_kwargs["columns"] = list(columns)
            if resolved_threshold is not None:
                compare_kwargs["threshold"] = resolved_threshold
            compare_kwargs.update(kwargs)

            drift_report = truthound.compare(baseline, current, **compare_kwargs)
            return self._convert_drift_result(drift_report, start_time, resolved_method)

        except ValidationExecutionError:
            raise
        except Exception as e:
            raise ValidationExecutionError(
                f"Truthound drift detection failed: {e}",
                cause=e,
            ) from e

    def _convert_drift_result(
        self,
        drift_report: Any,
        start_time: float,
        method: str,
    ) -> DriftResult:
        """Convert Truthound drift report to DriftResult.

        Args:
            drift_report: Report from Truthound compare().
            start_time: Start time for execution timing.
            method: Method string used.

        Returns:
            DriftResult instance.
        """
        report_dict = drift_report.to_dict()
        column_results = report_dict.get("columns", [])

        try:
            drift_method = DriftMethod(method)
        except ValueError:
            drift_method = DriftMethod.AUTO

        drifted_columns: list[ColumnDrift] = []
        drifted_count = 0

        for col_data in column_results:
            is_drifted = col_data.get("is_drifted", False)
            if is_drifted:
                drifted_count += 1

            col_method_str = col_data.get("method", method)
            try:
                col_method = DriftMethod(col_method_str)
            except ValueError:
                col_method = drift_method

            severity_str = col_data.get("severity", "info")
            severity = SEVERITY_MAPPING.get(severity_str, Severity.INFO)

            drifted_columns.append(
                ColumnDrift(
                    column=col_data.get("column", "unknown"),
                    method=col_method,
                    statistic=col_data.get("statistic", 0.0),
                    p_value=col_data.get("p_value"),
                    threshold=col_data.get("threshold", 0.0),
                    is_drifted=is_drifted,
                    severity=severity,
                    baseline_stats=col_data.get("baseline_stats", {}),
                    current_stats=col_data.get("current_stats", {}),
                    metadata=col_data.get("metadata", {}),
                )
            )

        total_columns = len(column_results)
        if drifted_count > 0:
            status = DriftStatus.DRIFT_DETECTED
        else:
            status = DriftStatus.NO_DRIFT

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return DriftResult(
            status=status,
            drifted_columns=tuple(drifted_columns),
            total_columns=total_columns,
            drifted_count=drifted_count,
            method=drift_method,
            execution_time_ms=execution_time_ms,
            metadata={
                "engine": self.engine_name,
                "method": method,
            },
        )

    # =========================================================================
    # Anomaly Detection (AnomalyDetectionEngine Protocol)
    # =========================================================================

    def detect_anomalies(
        self,
        data: Any,
        *,
        detector: str | None = None,
        columns: Sequence[str] | None = None,
        contamination: float | None = None,
        **kwargs: Any,
    ) -> AnomalyResult:
        """Detect anomalies in the data using ML-based detectors.

        Uses Truthound's ``ml`` module to run anomaly detection with
        Isolation Forest, Z-Score, LOF, or Ensemble detectors.

        Args:
            data: Data to analyze (DataFrame or file path).
            detector: Anomaly detector name. Falls back to config default.
            columns: Specific columns to check (None = all).
            contamination: Expected anomaly proportion. Falls back to config default.
            **kwargs: Additional Truthound-specific parameters.

        Returns:
            AnomalyResult with per-column anomaly scores.

        Raises:
            ValidationExecutionError: If anomaly detection fails.
        """
        start_time = time.perf_counter()
        resolved_detector = detector or self._config.default_anomaly_detector
        resolved_contamination = contamination or self._config.default_contamination

        try:
            truthound = self._ensure_truthound()

            detector_cls = self._get_detector_class(truthound, resolved_detector)
            det_instance = detector_cls(contamination=resolved_contamination, **kwargs)

            if columns is not None:
                det_instance.fit(data, columns=list(columns))
            else:
                det_instance.fit(data)

            anomaly_report = det_instance.predict(data)
            return self._convert_anomaly_result(
                anomaly_report, start_time, resolved_detector, data
            )

        except ValidationExecutionError:
            raise
        except Exception as e:
            raise ValidationExecutionError(
                f"Truthound anomaly detection failed: {e}",
                cause=e,
            ) from e

    @staticmethod
    def _get_detector_class(truthound: Any, detector: str) -> Any:
        """Resolve detector name to Truthound ML detector class.

        Args:
            truthound: Truthound module.
            detector: Detector name string.

        Returns:
            Truthound detector class.

        Raises:
            ValidationExecutionError: If detector is not supported.
        """
        detector_map = {
            "isolation_forest": "IsolationForestDetector",
            "z_score": "ZScoreDetector",
            "lof": "LocalOutlierFactor",
            "ensemble": "EnsembleDetector",
        }
        class_name = detector_map.get(detector)
        if class_name is None:
            raise ValidationExecutionError(
                f"Unsupported anomaly detector: '{detector}'. "
                f"Supported: {list(detector_map.keys())}"
            )
        ml_module = getattr(truthound, "ml", None)
        if ml_module is None:
            raise ValidationExecutionError(
                "Truthound ml module not available. "
                "Ensure Truthound is installed with ML support."
            )
        detector_cls = getattr(ml_module, class_name, None)
        if detector_cls is None:
            raise ValidationExecutionError(
                f"Detector class '{class_name}' not found in truthound.ml"
            )
        return detector_cls

    def _convert_anomaly_result(
        self,
        anomaly_report: Any,
        start_time: float,
        detector: str,
        data: Any,
    ) -> AnomalyResult:
        """Convert Truthound anomaly report to AnomalyResult.

        Args:
            anomaly_report: Report from Truthound detector.predict().
            start_time: Start time for execution timing.
            detector: Detector name used.
            data: Original data for row count.

        Returns:
            AnomalyResult instance.
        """
        report_dict = anomaly_report.to_dict()
        column_scores = report_dict.get("columns", [])

        anomalies: list[AnomalyScore] = []
        for col_data in column_scores:
            anomalies.append(
                AnomalyScore(
                    column=col_data.get("column", "unknown"),
                    score=col_data.get("score", 0.0),
                    threshold=col_data.get("threshold", 0.0),
                    is_anomaly=col_data.get("is_anomaly", False),
                    detector=detector,
                    metadata=col_data.get("metadata", {}),
                )
            )

        anomalous_row_count = report_dict.get("anomalous_row_count", 0)
        total_row_count = report_dict.get("total_row_count", 0)

        if not total_row_count:
            total_row_count = len(data) if hasattr(data, "__len__") else 0

        if anomalous_row_count > 0:
            status = AnomalyStatus.ANOMALY_DETECTED
        else:
            status = AnomalyStatus.NORMAL

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return AnomalyResult(
            status=status,
            anomalies=tuple(anomalies),
            anomalous_row_count=anomalous_row_count,
            total_row_count=total_row_count,
            detector=detector,
            execution_time_ms=execution_time_ms,
            metadata={
                "engine": self.engine_name,
                "detector": detector,
            },
        )

    # =========================================================================
    # Streaming Support (StreamingEngine Protocol)
    # =========================================================================

    def check_stream(
        self,
        stream: Any,
        *,
        batch_size: int = 1000,
        schema: Any | None = None,
        auto_schema: bool = False,
        **kwargs: Any,
    ) -> Iterator[CheckResult]:
        """Validate streaming data in batches.

        Processes an iterable stream in ``batch_size`` chunks, yielding
        an independent ``CheckResult`` for each batch. The generator
        pattern ensures memory-efficient processing of arbitrarily large
        streams.

        Args:
            stream: Iterable data stream (Iterator, Generator, Kafka adapter, etc.).
            batch_size: Number of records per batch.
            schema: Schema to validate against.
            auto_schema: Whether to auto-generate schema from first batch.
            **kwargs: Additional Truthound-specific parameters.

        Yields:
            CheckResult for each processed batch.

        Raises:
            ValidationExecutionError: If stream processing fails.
        """
        try:
            import polars as pl
        except ImportError:
            pass

        truthound = self._ensure_truthound()
        batch: list[Any] = []
        batch_index = 0
        resolved_schema = schema

        for record in stream:
            batch.append(record)

            if len(batch) >= batch_size:
                batch_df = self._to_dataframe(batch)

                if auto_schema and resolved_schema is None:
                    resolved_schema = truthound.learn(batch_df)

                check_kwargs: dict[str, Any] = {}
                if resolved_schema is not None:
                    check_kwargs["schema"] = resolved_schema
                check_kwargs.update(kwargs)

                start_time = time.perf_counter()
                try:
                    report = truthound.check(batch_df, **check_kwargs)
                    result = self._convert_check_result(report, start_time)
                except Exception as e:
                    result = CheckResult(
                        status=CheckStatus.ERROR,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                        metadata={
                            "engine": self.engine_name,
                            "batch_index": batch_index,
                            "error": str(e),
                        },
                    )

                yield result
                batch = []
                batch_index += 1

        # Process remaining records
        if batch:
            batch_df = self._to_dataframe(batch)

            if auto_schema and resolved_schema is None:
                resolved_schema = truthound.learn(batch_df)

            check_kwargs = {}
            if resolved_schema is not None:
                check_kwargs["schema"] = resolved_schema
            check_kwargs.update(kwargs)

            start_time = time.perf_counter()
            try:
                report = truthound.check(batch_df, **check_kwargs)
                result = self._convert_check_result(report, start_time)
            except Exception as e:
                result = CheckResult(
                    status=CheckStatus.ERROR,
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    metadata={
                        "engine": self.engine_name,
                        "batch_index": batch_index,
                        "error": str(e),
                    },
                )

            yield result

    @staticmethod
    def _to_dataframe(records: list[Any]) -> Any:
        """Convert a list of records to a Polars DataFrame.

        Handles dicts and existing DataFrames.

        Args:
            records: List of records.

        Returns:
            Polars DataFrame.
        """
        try:
            import polars as pl

            if records and isinstance(records[0], dict):
                return pl.DataFrame(records)
            return pl.DataFrame(records)
        except ImportError:
            return records
