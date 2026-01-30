"""Pandera Adapter for Data Quality Engine.

This module provides an adapter that wraps the Pandera library
to conform to the DataQualityEngine protocol, with full lifecycle management.

Example:
    >>> from common.engines import PanderaAdapter
    >>> engine = PanderaAdapter()
    >>> result = engine.check(
    ...     data=df,
    ...     rules=[{"type": "not_null", "column": "id"}],
    ... )

    >>> # With lifecycle management
    >>> with PanderaAdapter() as engine:
    ...     result = engine.check(data=df, rules=rules)

Note:
    Pandera specializes in schema validation with strong typing.
    This adapter translates common rule types to Pandera checks.
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
# Engine Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class PanderaConfig(EngineConfig):
    """Configuration for PanderaAdapter.

    Extends base EngineConfig with Pandera-specific settings.

    Attributes:
        lazy: Whether to collect all errors (True) or fail fast (False).
        strict: Whether to require exact column match.
        coerce: Whether to coerce types before validation.
        unique_column_names: Whether to require unique column names.
        report_duplicates: How to report duplicate rows (first, all, none).

    Example:
        >>> config = PanderaConfig(lazy=True, strict=False)
        >>> engine = PanderaAdapter(config=config)

        >>> # Using builder pattern
        >>> config = (
        ...     PanderaConfig()
        ...     .with_lazy(True)
        ...     .with_strict(True)
        ...     .with_coerce(True)
        ...     .with_auto_start(True)
        ... )
    """

    lazy: bool = True
    strict: bool = False
    coerce: bool = False
    unique_column_names: bool = False
    report_duplicates: str = "all"

    def __post_init__(self) -> None:
        """Validate Pandera-specific configuration."""
        self._validate_base_config()
        valid_report_duplicates = {"first", "all", "none"}
        if self.report_duplicates.lower() not in valid_report_duplicates:
            raise ValueError(
                f"report_duplicates must be one of {valid_report_duplicates}, "
                f"got '{self.report_duplicates}'"
            )

    def with_lazy(self, enabled: bool) -> Self:
        """Create config with lazy validation setting.

        Args:
            enabled: Whether to collect all errors (True) or fail fast (False).

        Returns:
            New configuration with lazy setting.
        """
        return self._copy_with(lazy=enabled)

    def with_strict(self, enabled: bool) -> Self:
        """Create config with strict column matching.

        Args:
            enabled: Whether to require exact column match.

        Returns:
            New configuration with strict setting.
        """
        return self._copy_with(strict=enabled)

    def with_coerce(self, enabled: bool) -> Self:
        """Create config with type coercion setting.

        Args:
            enabled: Whether to coerce types before validation.

        Returns:
            New configuration with coercion setting.
        """
        return self._copy_with(coerce=enabled)

    def with_unique_column_names(self, enabled: bool) -> Self:
        """Create config with unique column names requirement.

        Args:
            enabled: Whether to require unique column names.

        Returns:
            New configuration with column names setting.
        """
        return self._copy_with(unique_column_names=enabled)

    def with_report_duplicates(self, mode: str) -> Self:
        """Create config with duplicate reporting mode.

        Args:
            mode: How to report duplicates (first, all, none).

        Returns:
            New configuration with duplicate reporting setting.
        """
        return self._copy_with(report_duplicates=mode.lower())


DEFAULT_PANDERA_CONFIG = PanderaConfig()

# Production preset
PRODUCTION_PANDERA_CONFIG = PanderaConfig(
    auto_start=True,
    auto_stop=True,
    health_check_enabled=True,
    lazy=True,
    strict=True,
    coerce=False,
    unique_column_names=True,
    report_duplicates="all",
)

# Development preset
DEVELOPMENT_PANDERA_CONFIG = PanderaConfig(
    auto_start=False,
    auto_stop=True,
    health_check_enabled=False,
    lazy=True,
    strict=False,
    coerce=True,
    unique_column_names=False,
    report_duplicates="first",
)


# =============================================================================
# Pandera Adapter
# =============================================================================


class PanderaAdapter(EngineInfoMixin):
    """Adapter for Pandera data validation library.

    This adapter translates the DataQualityEngine interface to
    Pandera API calls, allowing Pandera schemas and checks to be
    used within the orchestration framework.

    Implements the ManagedEngine protocol for lifecycle management.

    Pandera excels at:
    - Schema-based validation
    - Type checking with pandas/polars integration
    - Statistical hypothesis testing
    - Custom check functions

    Attributes:
        _pa: Lazily loaded Pandera module.
        _config: Engine configuration.

    Example:
        >>> adapter = PanderaAdapter()
        >>> result = adapter.check(
        ...     data=df,
        ...     rules=[
        ...         {"type": "not_null", "column": "id"},
        ...         {"type": "dtype", "column": "value", "dtype": "float64"},
        ...     ],
        ... )

        >>> # With lifecycle management
        >>> with PanderaAdapter() as engine:
        ...     result = engine.check(data=df, rules=rules)

    Note:
        Pandera is lazily imported to avoid requiring it as a hard
        dependency.
    """

    def __init__(
        self,
        config: PanderaConfig | None = None,
    ) -> None:
        """Initialize the Pandera adapter.

        Args:
            config: Engine configuration.
        """
        self._config = config or DEFAULT_PANDERA_CONFIG
        self._pa: Any | None = None
        self._version: str | None = None
        self._state_tracker = EngineStateTracker("pandera")
        self._lock = threading.RLock()

        if self._config.auto_start:
            self.start()

    @property
    def engine_name(self) -> str:
        """Return the engine name."""
        return "pandera"

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        if self._version is None:
            try:
                import pandera as pa

                self._version = getattr(pa, "__version__", "0.0.0")
            except ImportError:
                self._version = "0.0.0"
        return self._version

    def _get_capabilities(self) -> EngineCapabilities:
        """Return Pandera capabilities."""
        return EngineCapabilities(
            supports_check=True,
            supports_profile=True,
            supports_learn=True,
            supports_async=False,
            supports_streaming=False,
            supports_drift=False,
            supports_anomaly=False,
            supported_data_types=("pandas", "polars"),
            supported_rule_types=(
                "not_null",
                "unique",
                "in_set",
                "in_range",
                "regex",
                "dtype",
                "min_length",
                "max_length",
                "greater_than",
                "less_than",
            ),
        )

    def _get_description(self) -> str:
        """Return engine description."""
        return "Pandera adapter for schema-based data validation with strong typing"

    def _get_homepage(self) -> str:
        """Return engine homepage."""
        return "https://pandera.readthedocs.io/"

    # =========================================================================
    # Lifecycle Management (ManagedEngine Protocol)
    # =========================================================================

    @property
    def config(self) -> PanderaConfig:
        """Return engine configuration."""
        return self._config

    def start(self) -> None:
        """Start the engine and initialize resources."""
        with self._lock:
            state = self._state_tracker.state
            if state.is_terminal:
                raise EngineStoppedError("pandera")
            if state.is_active:
                raise EngineAlreadyStartedError("pandera")

            self._state_tracker.transition_to(EngineState.STARTING)
            try:
                self._ensure_pandera()
                self._state_tracker.transition_to(EngineState.RUNNING)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineInitializationError(
                    f"Failed to start Pandera adapter: {e}",
                    engine_name="pandera",
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
                self._pa = None
                self._version = None
                self._state_tracker.transition_to(EngineState.STOPPED)
            except Exception as e:
                self._state_tracker.transition_to(EngineState.FAILED)
                raise EngineShutdownError(
                    f"Failed to stop Pandera adapter: {e}",
                    engine_name="pandera",
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
            pa = self._ensure_pandera()
            version = getattr(pa, "__version__", "unknown")
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.HEALTHY,
                message=f"Pandera v{version} is operational",
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

    def _ensure_pandera(self) -> Any:
        """Lazily import and return the Pandera library.

        Returns:
            Pandera module.

        Raises:
            ValidationExecutionError: If Pandera is not installed.
        """
        if self._pa is None:
            try:
                import pandera as pa

                self._pa = pa
            except ImportError as e:
                raise ValidationExecutionError(
                    "Pandera is not installed. "
                    "Install it with: pip install pandera",
                    cause=e,
                ) from e
        return self._pa

    def _convert_to_pandas(self, data: Any) -> Any:
        """Convert data to pandas DataFrame for Pandera compatibility.

        Args:
            data: Input data (Polars or Pandas DataFrame).

        Returns:
            Pandas DataFrame.
        """
        if hasattr(data, "to_pandas"):
            return data.to_pandas()
        return data

    def _build_check(self, rule: Mapping[str, Any]) -> Any:
        """Build a Pandera Check from a rule dictionary.

        Args:
            rule: Rule dictionary.

        Returns:
            Pandera Check instance.
        """
        pa = self._ensure_pandera()

        rule_type = rule.get("type", "")
        checks: list[Any] = []

        if rule_type == "not_null":
            checks.append(pa.Check.not_null())
        elif rule_type == "unique":
            checks.append(pa.Check.unique())
        elif rule_type == "in_set":
            values = rule.get("values", [])
            checks.append(pa.Check.isin(values))
        elif rule_type == "in_range":
            min_val = rule.get("min")
            max_val = rule.get("max")
            checks.append(pa.Check.in_range(min_val, max_val))
        elif rule_type == "regex":
            pattern = rule.get("pattern", ".*")
            checks.append(pa.Check.str_matches(pattern))
        elif rule_type == "greater_than":
            value = rule.get("value", 0)
            checks.append(pa.Check.greater_than(value))
        elif rule_type == "less_than":
            value = rule.get("value", 0)
            checks.append(pa.Check.less_than(value))
        elif rule_type == "min_length":
            value = rule.get("value", 0)
            checks.append(pa.Check.str_length(min_value=value))
        elif rule_type == "max_length":
            value = rule.get("value")
            checks.append(pa.Check.str_length(max_value=value))

        return checks

    def _map_dtype(self, dtype_str: str) -> Any:
        """Map dtype string to Pandera dtype.

        Args:
            dtype_str: Data type string.

        Returns:
            Pandera dtype or None.
        """
        pa = self._ensure_pandera()

        dtype_mapping = {
            "int": pa.Int,
            "int32": pa.Int32,
            "int64": pa.Int64,
            "float": pa.Float,
            "float32": pa.Float32,
            "float64": pa.Float64,
            "str": pa.String,
            "string": pa.String,
            "bool": pa.Bool,
            "boolean": pa.Bool,
            "datetime": pa.DateTime,
            "date": pa.Date,
        }

        return dtype_mapping.get(dtype_str.lower())

    def check(
        self,
        data: Any,
        rules: Sequence[Mapping[str, Any]],
        *,
        fail_on_error: bool = True,
        lazy: bool = True,
        **kwargs: Any,
    ) -> CheckResult:
        """Execute validation checks using Pandera.

        Args:
            data: Data to validate (Polars or Pandas DataFrame).
            rules: Sequence of validation rule dictionaries.
            fail_on_error: Whether to fail on execution errors.
            lazy: Whether to collect all errors (True) or fail fast (False).
            **kwargs: Additional Pandera-specific parameters.

        Returns:
            CheckResult with validation outcomes.

        Raises:
            ValidationExecutionError: If validation execution fails.
        """
        start_time = time.perf_counter()

        try:
            pa = self._ensure_pandera()
            pandas_data = self._convert_to_pandas(data)

            # Group rules by column
            column_rules: dict[str, list[Mapping[str, Any]]] = {}
            for rule in rules:
                col = rule.get("column", "__dataframe__")
                if col not in column_rules:
                    column_rules[col] = []
                column_rules[col].append(rule)

            # Build schema from rules
            columns: dict[str, Any] = {}
            for col, col_rules in column_rules.items():
                if col == "__dataframe__":
                    continue

                checks: list[Any] = []
                dtype = None
                nullable = True

                for rule in col_rules:
                    rule_type = rule.get("type", "")

                    if rule_type == "not_null":
                        nullable = False
                    elif rule_type == "dtype":
                        dtype = self._map_dtype(rule.get("dtype", ""))
                    else:
                        checks.extend(self._build_check(rule))

                columns[col] = pa.Column(
                    dtype=dtype,
                    nullable=nullable,
                    checks=checks if checks else None,
                )

            # Create schema
            schema = pa.DataFrameSchema(columns=columns, strict=False)

            # Validate
            failures: list[ValidationFailure] = []
            passed = 0
            failed = 0

            try:
                schema.validate(pandas_data, lazy=lazy)
                passed = len(rules)
            except pa.errors.SchemaErrors as e:
                # Process all schema errors
                for err in e.schema_errors:
                    failure_data = err.get("failure_cases", [])
                    failed_count = len(failure_data) if failure_data else 1

                    failures.append(
                        ValidationFailure(
                            rule_name=err.get("check", "unknown"),
                            column=err.get("column"),
                            message=str(err.get("error", "")),
                            severity=Severity.ERROR,
                            failed_count=failed_count,
                            total_count=len(pandas_data),
                        )
                    )
                    failed += 1

                passed = len(rules) - failed

            except pa.errors.SchemaError as e:
                # Single schema error (when lazy=False)
                failures.append(
                    ValidationFailure(
                        rule_name=str(e.check),
                        column=e.schema.name if hasattr(e.schema, "name") else None,
                        message=str(e),
                        severity=Severity.ERROR,
                        failed_count=1,
                        total_count=len(pandas_data),
                    )
                )
                failed = 1
                passed = len(rules) - 1

            # Determine status
            status = CheckStatus.PASSED if failed == 0 else CheckStatus.FAILED
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return CheckResult(
                status=status,
                passed_count=passed,
                failed_count=failed,
                failures=tuple(failures),
                execution_time_ms=execution_time_ms,
                metadata={"engine": self.engine_name},
            )

        except ValidationExecutionError:
            raise
        except Exception as e:
            if fail_on_error:
                raise ValidationExecutionError(
                    f"Pandera check failed: {e}",
                    cause=e,
                ) from e
            return CheckResult(
                status=CheckStatus.ERROR,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                metadata={"error": str(e), "engine": self.engine_name},
            )

    def profile(
        self,
        data: Any,
        *,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ProfileResult:
        """Profile data characteristics.

        Args:
            data: Data to profile.
            columns: Specific columns to profile (None = all).
            **kwargs: Additional parameters.

        Returns:
            ProfileResult with profiling outcomes.

        Raises:
            ValidationExecutionError: If profiling fails.
        """
        start_time = time.perf_counter()

        try:
            pandas_data = self._convert_to_pandas(data)

            column_profiles: list[ColumnProfile] = []
            profile_columns = list(columns) if columns else list(pandas_data.columns)

            for col in profile_columns:
                try:
                    col_data = pandas_data[col]
                    total_count = len(col_data)

                    null_count = int(col_data.isnull().sum())
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
                f"Pandera profile failed: {e}",
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
        """Learn validation rules (infer schema) from data.

        Args:
            data: Data to learn from.
            columns: Specific columns to learn from (None = all).
            confidence_threshold: Minimum confidence for learned rules.
            **kwargs: Additional parameters.

        Returns:
            LearnResult with learned rules.

        Raises:
            ValidationExecutionError: If learning fails.
        """
        start_time = time.perf_counter()

        try:
            pa = self._ensure_pandera()
            pandas_data = self._convert_to_pandas(data)

            # Use Pandera's schema inference
            inferred_schema = pa.infer_schema(pandas_data)

            learned_rules: list[LearnedRule] = []
            learn_columns = list(columns) if columns else list(pandas_data.columns)

            for col in learn_columns:
                if col not in inferred_schema.columns:
                    continue

                col_schema = inferred_schema.columns[col]
                total_count = len(pandas_data)

                # Learn dtype
                if col_schema.dtype is not None:
                    learned_rules.append(
                        LearnedRule(
                            rule_type="dtype",
                            column=col,
                            parameters={"dtype": str(col_schema.dtype)},
                            confidence=1.0,
                            sample_size=total_count,
                        )
                    )

                # Learn nullable
                col_data = pandas_data[col]
                null_rate = col_data.isnull().sum() / total_count if total_count > 0 else 0
                if null_rate < (1 - confidence_threshold):
                    learned_rules.append(
                        LearnedRule(
                            rule_type="not_null",
                            column=col,
                            confidence=1 - null_rate,
                            sample_size=total_count,
                        )
                    )

                # Learn unique
                unique_rate = col_data.nunique() / total_count if total_count > 0 else 0
                if unique_rate >= confidence_threshold:
                    learned_rules.append(
                        LearnedRule(
                            rule_type="unique",
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
                                rule_type="in_range",
                                column=col,
                                parameters={"min": float(min_val), "max": float(max_val)},
                                confidence=1.0,
                                sample_size=total_count,
                            )
                        )

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
                f"Pandera learn failed: {e}",
                cause=e,
            ) from e
