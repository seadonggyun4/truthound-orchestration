"""Great Expectations Adapter for Data Quality Engine.

This module provides an adapter that wraps Great Expectations library
to conform to the DataQualityEngine protocol, with full lifecycle management.

Example:
    >>> from common.engines import GreatExpectationsAdapter
    >>> engine = GreatExpectationsAdapter()
    >>> result = engine.check(
    ...     data=df,
    ...     rules=[{"type": "expect_column_to_exist", "column": "id"}],
    ... )

    >>> # With lifecycle management
    >>> with GreatExpectationsAdapter() as engine:
    ...     result = engine.check(data=df, rules=rules)

Note:
    This adapter translates common rule types to Great Expectations
    expectations. Custom GE expectations can be passed directly.
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
# Rule Type Mapping
# =============================================================================

# Map common rule types to Great Expectations expectations
RULE_TYPE_MAPPING: dict[str, str] = {
    "not_null": "expect_column_values_to_not_be_null",
    "unique": "expect_column_values_to_be_unique",
    "in_set": "expect_column_values_to_be_in_set",
    "in_range": "expect_column_values_to_be_between",
    "regex": "expect_column_values_to_match_regex",
    "dtype": "expect_column_values_to_be_of_type",
    "min_length": "expect_column_value_lengths_to_be_between",
    "max_length": "expect_column_value_lengths_to_be_between",
    "column_exists": "expect_column_to_exist",
}


# =============================================================================
# Engine Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class GreatExpectationsConfig(EngineConfig):
    """Configuration for GreatExpectationsAdapter.

    Extends base EngineConfig with GE-specific settings.

    Attributes:
        result_format: Default result format (BOOLEAN_ONLY, BASIC, COMPLETE).
        context_root_dir: Path to GE context root directory.
        include_profiling: Whether to include data docs profiling.
        catch_exceptions: Whether to catch exceptions in expectations.
        enable_data_docs: Whether to enable data docs generation.

    Example:
        >>> config = GreatExpectationsConfig(
        ...     result_format="COMPLETE",
        ...     auto_start=True,
        ... )
        >>> engine = GreatExpectationsAdapter(config=config)

        >>> # Using builder pattern
        >>> config = (
        ...     GreatExpectationsConfig()
        ...     .with_result_format("BASIC")
        ...     .with_context_root_dir("/path/to/ge")
        ...     .with_auto_start(True)
        ... )
    """

    result_format: str = "COMPLETE"
    context_root_dir: str | None = None
    include_profiling: bool = True
    catch_exceptions: bool = True
    enable_data_docs: bool = False

    def __post_init__(self) -> None:
        """Validate GE-specific configuration."""
        self._validate_base_config()
        valid_formats = {"BOOLEAN_ONLY", "BASIC", "SUMMARY", "COMPLETE"}
        if self.result_format.upper() not in valid_formats:
            raise ValueError(
                f"result_format must be one of {valid_formats}, "
                f"got '{self.result_format}'"
            )

    def with_result_format(self, format_: str) -> Self:
        """Create config with result format.

        Args:
            format_: Result format (BOOLEAN_ONLY, BASIC, SUMMARY, COMPLETE).

        Returns:
            New configuration with format setting.
        """
        return self._copy_with(result_format=format_.upper())

    def with_context_root_dir(self, path: str) -> Self:
        """Create config with GE context root directory.

        Args:
            path: Path to GE context root directory.

        Returns:
            New configuration with context path.
        """
        return self._copy_with(context_root_dir=path)

    def with_profiling(self, enabled: bool) -> Self:
        """Create config with profiling setting.

        Args:
            enabled: Whether to include profiling.

        Returns:
            New configuration with profiling setting.
        """
        return self._copy_with(include_profiling=enabled)

    def with_catch_exceptions(self, enabled: bool) -> Self:
        """Create config with exception catching setting.

        Args:
            enabled: Whether to catch exceptions in expectations.

        Returns:
            New configuration with exception handling setting.
        """
        return self._copy_with(catch_exceptions=enabled)

    def with_data_docs(self, enabled: bool) -> Self:
        """Create config with data docs setting.

        Args:
            enabled: Whether to enable data docs generation.

        Returns:
            New configuration with data docs setting.
        """
        return self._copy_with(enable_data_docs=enabled)


DEFAULT_GE_CONFIG = GreatExpectationsConfig()

# Production preset
PRODUCTION_GE_CONFIG = GreatExpectationsConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    result_format="COMPLETE",
    include_profiling=True,
    catch_exceptions=True,
    enable_data_docs=True,
)

# Development preset
DEVELOPMENT_GE_CONFIG = GreatExpectationsConfig(
    auto_start=False,
    auto_stop=True,
    health_check_enabled=False,
    result_format="BASIC",
    include_profiling=False,
    catch_exceptions=False,
    enable_data_docs=False,
)


# =============================================================================
# Great Expectations Adapter
# =============================================================================


class GreatExpectationsAdapter(EngineInfoMixin):
    """Adapter for Great Expectations data quality library.

    This adapter translates the DataQualityEngine interface to
    Great Expectations API calls, allowing GE to be used as a
    pluggable engine within the orchestration framework.

    Implements the ManagedEngine protocol for lifecycle management.

    Attributes:
        _ge: Lazily loaded Great Expectations module.
        _config: Engine configuration.

    Example:
        >>> adapter = GreatExpectationsAdapter()
        >>> result = adapter.check(
        ...     data=df,
        ...     rules=[
        ...         {"type": "not_null", "column": "id"},
        ...         {"type": "unique", "column": "email"},
        ...     ],
        ... )

        >>> # With lifecycle management
        >>> with GreatExpectationsAdapter() as engine:
        ...     result = engine.check(data=df, rules=rules)

    Note:
        Great Expectations is lazily imported to avoid requiring it
        as a hard dependency.
    """

    def __init__(
        self,
        config: GreatExpectationsConfig | None = None,
    ) -> None:
        """Initialize the Great Expectations adapter.

        Args:
            config: Engine configuration.
        """
        self._config = config or DEFAULT_GE_CONFIG
        self._ge: Any | None = None
        self._version: str | None = None
        self._state_tracker = EngineStateTracker("great_expectations")
        self._lock = threading.RLock()

        if self._config.auto_start:
            self.start()

    @property
    def engine_name(self) -> str:
        """Return the engine name."""
        return "great_expectations"

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        if self._version is None:
            try:
                import great_expectations as ge

                self._version = getattr(ge, "__version__", "0.0.0")
            except ImportError:
                self._version = "0.0.0"
        return self._version

    def _get_capabilities(self) -> EngineCapabilities:
        """Return Great Expectations capabilities."""
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
            supports_async=False,
            supports_streaming=False,
            supports_drift=False,
            supports_anomaly=False,
            supported_data_types=("pandas", "polars", "spark"),
            supported_rule_types=tuple(RULE_TYPE_MAPPING.keys()),
        )

    def _get_description(self) -> str:
        """Return engine description."""
        return (
            "Great Expectations adapter for comprehensive data validation "
            "and documentation"
        )

    def _get_homepage(self) -> str:
        """Return engine homepage."""
        return "https://greatexpectations.io/"

    # =========================================================================
    # Lifecycle Management (ManagedEngine Protocol)
    # =========================================================================

    @property
    def config(self) -> GreatExpectationsConfig:
        """Return engine configuration."""
        return self._config

    def start(self) -> None:
        """Start the engine and initialize resources."""
        with self._lock:
            state = self._state_tracker.state
            if state.is_terminal:
                raise EngineStoppedError("great_expectations")
            if state.is_active:
                raise EngineAlreadyStartedError("great_expectations")

            self._state_tracker.transition_to(EngineState.STARTING)
            try:
                self._ensure_ge()
                self._state_tracker.transition_to(EngineState.RUNNING)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineInitializationError(
                    f"Failed to start Great Expectations adapter: {e}",
                    engine_name="great_expectations",
                    cause=e,
                ) from e

    def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        with self._lock:
            state = self._state_tracker.state
            if state == EngineState.STOPPED:
                return
            if not state.can_stop:
                return

            self._state_tracker.transition_to(EngineState.STOPPING)
            try:
                self._ge = None
                self._version = None
                self._state_tracker.transition_to(EngineState.STOPPED)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineShutdownError(
                    f"Failed to stop Great Expectations adapter: {e}",
                    engine_name="great_expectations",
                    cause=e,
                ) from e

    def health_check(self) -> HealthCheckResult:
        """Perform health check on the engine."""
        start_time = time.perf_counter()
        state = self._state_tracker.state

        if state != EngineState.RUNNING:
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
            self._state_tracker.record_health_check(result.status)
            return result

        try:
            ge = self._ensure_ge()
            version = getattr(ge, "__version__", "unknown")
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.HEALTHY,
                message=f"Great Expectations v{version} is operational",
                duration_ms=duration_ms,
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
            )
            self._state_tracker.record_health_check(result.status)
            return result

    def get_state(self) -> EngineState:
        """Return current engine state."""
        return self._state_tracker.state

    def get_state_snapshot(self) -> EngineStateSnapshot:
        """Return snapshot of engine state."""
        return self._state_tracker.get_snapshot()

    def __enter__(self) -> Self:
        """Enter context manager."""
        if self._state_tracker.state == EngineState.CREATED:
            self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager."""
        if self._config.auto_stop:
            self.stop()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _ensure_ge(self) -> Any:
        """Lazily import and return the Great Expectations library.

        Returns:
            Great Expectations module.

        Raises:
            ValidationExecutionError: If GE is not installed.
        """
        if self._ge is None:
            try:
                import great_expectations as ge

                self._ge = ge
            except ImportError as e:
                raise ValidationExecutionError(
                    "Great Expectations is not installed. "
                    "Install it with: pip install great-expectations",
                    cause=e,
                ) from e
        return self._ge

    def _convert_to_pandas(self, data: Any) -> Any:
        """Convert data to pandas DataFrame for GE compatibility.

        Args:
            data: Input data (Polars or Pandas DataFrame).

        Returns:
            Pandas DataFrame.
        """
        # Check if it's a Polars DataFrame
        if hasattr(data, "to_pandas"):
            return data.to_pandas()
        return data

    def _translate_rule(self, rule: Mapping[str, Any]) -> dict[str, Any]:
        """Translate common rule format to GE expectation.

        Args:
            rule: Rule dictionary with common format.

        Returns:
            GE expectation dictionary.
        """
        rule_type = rule.get("type", "")
        column = rule.get("column")

        # Check if it's already a GE expectation
        if rule_type.startswith("expect_"):
            return dict(rule)

        # Translate common rule types
        ge_type = RULE_TYPE_MAPPING.get(rule_type, rule_type)

        translated: dict[str, Any] = {
            "expectation_type": ge_type,
        }

        if column:
            translated["column"] = column

        # Handle specific rule parameters
        if rule_type == "in_set":
            translated["value_set"] = rule.get("values", [])
        elif rule_type == "in_range":
            translated["min_value"] = rule.get("min")
            translated["max_value"] = rule.get("max")
        elif rule_type == "regex":
            translated["regex"] = rule.get("pattern", ".*")
        elif rule_type == "dtype":
            translated["type_"] = rule.get("dtype", "object")
        elif rule_type in ("min_length", "max_length"):
            if rule_type == "min_length":
                translated["min_value"] = rule.get("value", 0)
            else:
                translated["max_value"] = rule.get("value")

        # Copy any additional parameters
        excluded_keys = {"type", "column", "values", "min", "max", "pattern", "dtype", "value"}
        translated.update({k: v for k, v in rule.items() if k not in excluded_keys})

        return translated

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        *,
        fail_on_error: bool = True,
        result_format: str = "COMPLETE",
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation checks using Great Expectations.

        Args:
            data: Data to validate (Polars or Pandas DataFrame).
            rules: Sequence of validation rule dictionaries.
            fail_on_error: Whether to fail on execution errors.
            result_format: GE result format (BOOLEAN_ONLY, BASIC, COMPLETE).
            **kwargs: Additional GE-specific parameters.

        Returns:
            CheckResult with validation outcomes.

        Raises:
            ValidationExecutionError: If validation execution fails.
        """
        start_time = time.perf_counter()

        try:
            ge = self._ensure_ge()
            pandas_data = self._convert_to_pandas(data)

            # Create GE DataFrame
            ge_df = ge.from_pandas(pandas_data)

            failures: list[ValidationFailure] = []
            passed = 0
            failed = 0
            warnings = 0

            # Execute each rule as an expectation
            for rule in rules:
                translated = self._translate_rule(rule)
                expectation_type = translated.pop("expectation_type", None)

                if not expectation_type:
                    continue

                try:
                    # Get the expectation method
                    expectation_method = getattr(ge_df, expectation_type, None)
                    if expectation_method is None:
                        if fail_on_error:
                            raise ValidationExecutionError(
                                f"Unknown expectation type: {expectation_type}"
                            )
                        failed += 1
                        failures.append(
                            ValidationFailure(
                                rule_name=expectation_type,
                                column=translated.get("column"),
                                message=f"Unknown expectation: {expectation_type}",
                                severity=Severity.ERROR,
                            )
                        )
                        continue

                    # Execute the expectation
                    result = expectation_method(
                        result_format=result_format,
                        **translated,
                    )

                    # Process result
                    if result.success:
                        passed += 1
                    else:
                        severity_str = rule.get("severity", "error")
                        if severity_str.lower() == "warning":
                            warnings += 1
                        else:
                            failed += 1
                            failures.append(
                                self._create_failure_from_result(
                                    expectation_type,
                                    translated.get("column"),
                                    result,
                                )
                            )

                except Exception as e:
                    if fail_on_error:
                        raise ValidationExecutionError(
                            f"Expectation {expectation_type} failed: {e}",
                            cause=e,
                        ) from e
                    failed += 1
                    failures.append(
                        ValidationFailure(
                            rule_name=expectation_type,
                            column=translated.get("column"),
                            message=str(e),
                            severity=Severity.ERROR,
                        )
                    )

            # Determine status
            status = CheckStatus.PASSED
            if failed > 0:
                status = CheckStatus.FAILED
            elif warnings > 0:
                status = CheckStatus.WARNING

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return CheckResult(
                status=status,
                passed_count=passed,
                failed_count=failed,
                warning_count=warnings,
                failures=tuple(failures),
                execution_time_ms=execution_time_ms,
                metadata={"engine": self.engine_name},
            )

        except ValidationExecutionError:
            raise
        except Exception as e:
            if fail_on_error:
                raise ValidationExecutionError(
                    f"Great Expectations check failed: {e}",
                    cause=e,
                ) from e
            return CheckResult(
                status=CheckStatus.ERROR,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": str(e), "engine": self.engine_name},
            )

    def _create_failure_from_result(
        self,
        expectation_type: str,
        column: str | None,
        result: Any,
    ) -> ValidationFailure:
        """Create ValidationFailure from GE result.

        Args:
            expectation_type: Type of expectation that failed.
            column: Column name (if applicable).
            result: GE expectation result.

        Returns:
            ValidationFailure instance.
        """
        # Extract failure details from GE result
        result_dict = result.result if hasattr(result, "result") else {}

        failed_count = result_dict.get("unexpected_count", 0)
        total_count = result_dict.get("element_count", 0)
        sample_values = result_dict.get("partial_unexpected_list", [])[:5]
        message = result_dict.get("exception_info", {}).get("exception_message", "")

        if not message:
            message = f"Expectation {expectation_type} failed"
            if failed_count > 0:
                message += f": {failed_count} unexpected values"

        return ValidationFailure(
            rule_name=expectation_type,
            column=column,
            message=message,
            severity=Severity.ERROR,
            failed_count=failed_count,
            total_count=total_count,
            sample_values=tuple(sample_values),
        )

    def profile(
        self,
        data: Any,
        *,
        columns: Sequence[str] | None = None,
        include_histograms: bool = True,
        **kwargs: Any,
    ) -> ProfileResult:
        """Profile data using Great Expectations profiler.

        Args:
            data: Data to profile.
            columns: Specific columns to profile (None = all).
            include_histograms: Whether to compute histograms.
            **kwargs: Additional GE-specific parameters.

        Returns:
            ProfileResult with profiling outcomes.

        Raises:
            ValidationExecutionError: If profiling fails.
        """
        start_time = time.perf_counter()

        try:
            ge = self._ensure_ge()
            pandas_data = self._convert_to_pandas(data)

            # Use GE's basic profiling (we use pandas directly for stats)
            _ = ge  # GE module loaded for potential future use

            column_profiles: list[ColumnProfile] = []
            profile_columns = list(columns) if columns else list(pandas_data.columns)

            for col in profile_columns:
                try:
                    col_data = pandas_data[col]

                    # Basic statistics
                    null_count = int(col_data.isnull().sum())
                    total_count = len(col_data)
                    unique_count = int(col_data.nunique())

                    profile = ColumnProfile(
                        column_name=col,
                        dtype=str(col_data.dtype),
                        null_count=null_count,
                        null_percentage=(
                            (null_count / total_count * 100) if total_count > 0 else 0.0
                        ),
                        unique_count=unique_count,
                        unique_percentage=(
                            (unique_count / total_count * 100) if total_count > 0 else 0.0
                        ),
                    )

                    # Add numeric stats if applicable
                    if col_data.dtype.kind in ("i", "f"):
                        profile = ColumnProfile(
                            column_name=profile.column_name,
                            dtype=profile.dtype,
                            null_count=profile.null_count,
                            null_percentage=profile.null_percentage,
                            unique_count=profile.unique_count,
                            unique_percentage=profile.unique_percentage,
                            min_value=col_data.min() if not col_data.empty else None,
                            max_value=col_data.max() if not col_data.empty else None,
                            mean=float(col_data.mean()) if not col_data.empty else None,
                            std=float(col_data.std()) if not col_data.empty else None,
                        )

                    column_profiles.append(profile)
                except Exception:
                    # Skip columns that can't be profiled
                    continue

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return ProfileResult(
                status=ProfileStatus.COMPLETED,
                row_count=len(pandas_data),
                column_count=len(column_profiles),
                columns=tuple(column_profiles),
                execution_time_ms=execution_time_ms,
                metadata={"engine": self.engine_name},
            )

        except Exception as e:
            raise ValidationExecutionError(
                f"Great Expectations profile failed: {e}",
                cause=e,
            ) from e

    def learn(
        self,
        data: Any,
        *,
        columns: Sequence[str] | None = None,
        confidence_threshold: float = 0.95,
        **kwargs: Any,
    ) -> LearnResult:
        """Learn validation rules using GE's auto-profiling.

        Args:
            data: Data to learn from.
            columns: Specific columns to learn from (None = all).
            confidence_threshold: Minimum confidence for learned rules.
            **kwargs: Additional GE-specific parameters.

        Returns:
            LearnResult with learned rules.

        Raises:
            ValidationExecutionError: If learning fails.
        """
        start_time = time.perf_counter()

        try:
            self._ensure_ge()  # Verify GE is available
            pandas_data = self._convert_to_pandas(data)

            learned_rules: list[LearnedRule] = []
            learn_columns = list(columns) if columns else list(pandas_data.columns)

            for col in learn_columns:
                try:
                    col_data = pandas_data[col]
                    total_count = len(col_data)

                    # Learn not_null if mostly non-null
                    null_rate = col_data.isnull().sum() / total_count if total_count > 0 else 0
                    if null_rate < (1 - confidence_threshold):
                        learned_rules.append(
                            LearnedRule(
                                rule_type="expect_column_values_to_not_be_null",
                                column=col,
                                confidence=1 - null_rate,
                                sample_size=total_count,
                            )
                        )

                    # Learn uniqueness if highly unique
                    unique_rate = col_data.nunique() / total_count if total_count > 0 else 0
                    if unique_rate >= confidence_threshold:
                        learned_rules.append(
                            LearnedRule(
                                rule_type="expect_column_values_to_be_unique",
                                column=col,
                                confidence=unique_rate,
                                sample_size=total_count,
                            )
                        )

                    # Learn range for numeric columns
                    if col_data.dtype.kind in ("i", "f"):
                        min_val = col_data.min()
                        max_val = col_data.max()
                        if min_val is not None and max_val is not None:
                            learned_rules.append(
                                LearnedRule(
                                    rule_type="expect_column_values_to_be_between",
                                    column=col,
                                    parameters={"min_value": min_val, "max_value": max_val},
                                    confidence=1.0,
                                    sample_size=total_count,
                                )
                            )

                except Exception:
                    # Skip columns that can't be analyzed
                    continue

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return LearnResult(
                status=LearnStatus.COMPLETED,
                rules=tuple(learned_rules),
                columns_analyzed=len(learn_columns),
                execution_time_ms=execution_time_ms,
                metadata={"engine": self.engine_name},
            )

        except Exception as e:
            raise ValidationExecutionError(
                f"Great Expectations learn failed: {e}",
                cause=e,
            ) from e
