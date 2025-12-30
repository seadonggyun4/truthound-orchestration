"""Testing utilities for Truthound Integrations.

This module provides test utilities, mocks, and helpers for testing
platform integrations. It includes:
- MockTruthound: Simulates Truthound behavior for unit tests
- Result factories: Create mock results with customizable parameters
- Test data generators: Create sample DataFrames for testing
- Assertion helpers: Verify results match expectations

Example:
    >>> from common.testing import MockTruthound, create_mock_check_result
    >>> mock = MockTruthound()
    >>> mock.configure_check(success=True, passed_count=10)
    >>> result = mock.check(data, config)
    >>> assert result.is_success
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from common.base import (
    CheckConfig,
    CheckResult,
    CheckStatus,
    ColumnProfile,
    FailureAction,
    LearnedRule,
    LearnResult,
    LearnStatus,
    ProfileResult,
    ProfileStatus,
    Severity,
    ValidationFailure,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl

    from common.base import LearnConfig, ProfileConfig


# =============================================================================
# Mock Data Quality Engine
# =============================================================================


class MockDataQualityEngine:
    """Mock implementation of DataQualityEngine for testing.

    Provides configurable mock responses for check, profile, and learn
    operations. Implements the DataQualityEngine protocol for use in tests.

    Example:
        >>> engine = MockDataQualityEngine()
        >>> engine.configure_check(success=True, passed_count=10)
        >>> result = engine.check(data, rules=[{"type": "not_null", "column": "id"}])
        >>> assert result.is_success
        >>> assert engine.check_call_count == 1
    """

    def __init__(
        self,
        name: str = "mock",
        version: str = "1.0.0",
    ) -> None:
        """Initialize the mock engine.

        Args:
            name: Engine name.
            version: Engine version.
        """
        self._name = name
        self._version = version
        self._check_config: MockCheckConfig = MockCheckConfig()
        self._profile_config: MockProfileConfig = MockProfileConfig()
        self._learn_config: MockLearnConfig = MockLearnConfig()

        self._check_calls: list[tuple[Any, tuple[dict[str, Any], ...]]] = []
        self._profile_calls: list[tuple[Any, dict[str, Any]]] = []
        self._learn_calls: list[tuple[Any, dict[str, Any]]] = []

    @property
    def engine_name(self) -> str:
        """Return the engine name."""
        return self._name

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        return self._version

    @property
    def check_call_count(self) -> int:
        """Return the number of check calls."""
        return len(self._check_calls)

    @property
    def profile_call_count(self) -> int:
        """Return the number of profile calls."""
        return len(self._profile_calls)

    @property
    def learn_call_count(self) -> int:
        """Return the number of learn calls."""
        return len(self._learn_calls)

    def get_check_calls(self) -> list[tuple[Any, tuple[dict[str, Any], ...]]]:
        """Get all check call arguments."""
        return list(self._check_calls)

    def get_profile_calls(self) -> list[tuple[Any, dict[str, Any]]]:
        """Get all profile call arguments."""
        return list(self._profile_calls)

    def get_learn_calls(self) -> list[tuple[Any, dict[str, Any]]]:
        """Get all learn call arguments."""
        return list(self._learn_calls)

    def configure_check(
        self,
        *,
        success: bool = True,
        passed_count: int = 10,
        failed_count: int = 0,
        warning_count: int = 0,
        failures: tuple[ValidationFailure, ...] | None = None,
        execution_time_ms: float = 100.0,
        raise_error: Exception | None = None,
    ) -> None:
        """Configure mock check behavior.

        Args:
            success: Whether check should succeed.
            passed_count: Number of passed validations.
            failed_count: Number of failed validations.
            warning_count: Number of warnings.
            failures: Custom failures to return.
            execution_time_ms: Simulated execution time.
            raise_error: Exception to raise.
        """
        self._check_config = MockCheckConfig(
            success=success,
            passed_count=passed_count,
            failed_count=failed_count,
            warning_count=warning_count,
            failures=failures or (),
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
        )

    def configure_profile(
        self,
        *,
        success: bool = True,
        row_count: int = 1000,
        columns: tuple[ColumnProfile, ...] | None = None,
        execution_time_ms: float = 200.0,
        raise_error: Exception | None = None,
    ) -> None:
        """Configure mock profile behavior.

        Args:
            success: Whether profiling should succeed.
            row_count: Simulated row count.
            columns: Mock column profiles.
            execution_time_ms: Simulated execution time.
            raise_error: Exception to raise.
        """
        self._profile_config = MockProfileConfig(
            success=success,
            row_count=row_count,
            columns=columns or (),
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
        )

    def configure_learn(
        self,
        *,
        success: bool = True,
        rules: tuple[LearnedRule, ...] | None = None,
        columns_analyzed: int = 5,
        execution_time_ms: float = 500.0,
        raise_error: Exception | None = None,
    ) -> None:
        """Configure mock learn behavior.

        Args:
            success: Whether learning should succeed.
            rules: Mock learned rules.
            columns_analyzed: Number of columns analyzed.
            execution_time_ms: Simulated execution time.
            raise_error: Exception to raise.
        """
        self._learn_config = MockLearnConfig(
            success=success,
            rules=rules or (),
            columns_analyzed=columns_analyzed,
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
        )

    def check(
        self,
        data: Any,
        rules: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute mock validation check.

        Args:
            data: Data to validate.
            rules: Validation rules.
            **kwargs: Additional parameters.

        Returns:
            Mock CheckResult.

        Raises:
            Exception: If configured to raise an error.
        """
        self._check_calls.append((data, tuple(rules)))

        if self._check_config.raise_error:
            raise self._check_config.raise_error

        cfg = self._check_config
        status = CheckStatus.PASSED if cfg.success else CheckStatus.FAILED
        if cfg.warning_count > 0 and cfg.failed_count == 0:
            status = CheckStatus.WARNING

        return CheckResult(
            status=status,
            passed_count=cfg.passed_count,
            failed_count=cfg.failed_count,
            warning_count=cfg.warning_count,
            failures=cfg.failures,
            execution_time_ms=cfg.execution_time_ms,
            metadata={"engine": self._name},
        )

    def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        """Execute mock profiling.

        Args:
            data: Data to profile.
            **kwargs: Additional parameters.

        Returns:
            Mock ProfileResult.

        Raises:
            Exception: If configured to raise an error.
        """
        self._profile_calls.append((data, kwargs))

        if self._profile_config.raise_error:
            raise self._profile_config.raise_error

        cfg = self._profile_config
        return ProfileResult(
            status=ProfileStatus.COMPLETED if cfg.success else ProfileStatus.FAILED,
            row_count=cfg.row_count,
            column_count=len(cfg.columns),
            columns=cfg.columns,
            execution_time_ms=cfg.execution_time_ms,
            metadata={"engine": self._name},
        )

    def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        """Execute mock schema learning.

        Args:
            data: Data to learn from.
            **kwargs: Additional parameters.

        Returns:
            Mock LearnResult.

        Raises:
            Exception: If configured to raise an error.
        """
        self._learn_calls.append((data, kwargs))

        if self._learn_config.raise_error:
            raise self._learn_config.raise_error

        cfg = self._learn_config
        return LearnResult(
            status=LearnStatus.COMPLETED if cfg.success else LearnStatus.FAILED,
            rules=cfg.rules,
            columns_analyzed=cfg.columns_analyzed,
            execution_time_ms=cfg.execution_time_ms,
            metadata={"engine": self._name},
        )

    def reset(self) -> None:
        """Reset all call history and configurations."""
        self._check_calls.clear()
        self._profile_calls.clear()
        self._learn_calls.clear()
        self._check_config = MockCheckConfig()
        self._profile_config = MockProfileConfig()
        self._learn_config = MockLearnConfig()


# =============================================================================
# Mock Truthound (Legacy - wraps MockDataQualityEngine)
# =============================================================================


@dataclass
class MockCheckConfig:
    """Configuration for mock check behavior.

    Attributes:
        success: Whether check should succeed.
        passed_count: Number of passed validations.
        failed_count: Number of failed validations.
        warning_count: Number of warnings.
        failures: Custom failures to return.
        execution_time_ms: Simulated execution time.
        raise_error: Exception to raise (if any).
    """

    success: bool = True
    passed_count: int = 10
    failed_count: int = 0
    warning_count: int = 0
    failures: tuple[ValidationFailure, ...] = ()
    execution_time_ms: float = 100.0
    raise_error: Exception | None = None


@dataclass
class MockProfileConfig:
    """Configuration for mock profile behavior.

    Attributes:
        success: Whether profiling should succeed.
        row_count: Simulated row count.
        columns: Mock column profiles.
        execution_time_ms: Simulated execution time.
        raise_error: Exception to raise (if any).
    """

    success: bool = True
    row_count: int = 1000
    columns: tuple[ColumnProfile, ...] = ()
    execution_time_ms: float = 200.0
    raise_error: Exception | None = None


@dataclass
class MockLearnConfig:
    """Configuration for mock learn behavior.

    Attributes:
        success: Whether learning should succeed.
        rules: Mock learned rules.
        columns_analyzed: Number of columns analyzed.
        execution_time_ms: Simulated execution time.
        raise_error: Exception to raise (if any).
    """

    success: bool = True
    rules: tuple[LearnedRule, ...] = ()
    columns_analyzed: int = 5
    execution_time_ms: float = 500.0
    raise_error: Exception | None = None


class MockTruthound:
    """Mock Truthound implementation for testing.

    Provides configurable mock responses for check, profile, and learn
    operations. Tracks call history for verification in tests.

    Example:
        >>> mock = MockTruthound()
        >>> mock.configure_check(success=True, passed_count=10)
        >>>
        >>> result = mock.check(data, config)
        >>> assert result.is_success
        >>> assert mock.check_call_count == 1
    """

    def __init__(self) -> None:
        """Initialize the mock Truthound."""
        self._check_config = MockCheckConfig()
        self._profile_config = MockProfileConfig()
        self._learn_config = MockLearnConfig()

        self._check_calls: list[tuple[Any, CheckConfig]] = []
        self._profile_calls: list[tuple[Any, ProfileConfig]] = []
        self._learn_calls: list[tuple[Any, LearnConfig]] = []

        self._check_hook: Callable[[Any, CheckConfig], CheckResult | None] | None = None
        self._profile_hook: Callable[[Any, ProfileConfig], ProfileResult | None] | None = None
        self._learn_hook: Callable[[Any, LearnConfig], LearnResult | None] | None = None

    @property
    def platform_name(self) -> str:
        """Return the platform name."""
        return "mock"

    @property
    def platform_version(self) -> str:
        """Return the platform version."""
        return "1.0.0"

    @property
    def check_call_count(self) -> int:
        """Return the number of check calls."""
        return len(self._check_calls)

    @property
    def profile_call_count(self) -> int:
        """Return the number of profile calls."""
        return len(self._profile_calls)

    @property
    def learn_call_count(self) -> int:
        """Return the number of learn calls."""
        return len(self._learn_calls)

    def get_check_calls(self) -> list[tuple[Any, CheckConfig]]:
        """Get all check call arguments."""
        return list(self._check_calls)

    def get_profile_calls(self) -> list[tuple[Any, ProfileConfig]]:
        """Get all profile call arguments."""
        return list(self._profile_calls)

    def get_learn_calls(self) -> list[tuple[Any, LearnConfig]]:
        """Get all learn call arguments."""
        return list(self._learn_calls)

    def configure_check(
        self,
        *,
        success: bool = True,
        passed_count: int = 10,
        failed_count: int = 0,
        warning_count: int = 0,
        failures: tuple[ValidationFailure, ...] | None = None,
        execution_time_ms: float = 100.0,
        raise_error: Exception | None = None,
    ) -> None:
        """Configure mock check behavior.

        Args:
            success: Whether check should succeed.
            passed_count: Number of passed validations.
            failed_count: Number of failed validations.
            warning_count: Number of warnings.
            failures: Custom failures to return.
            execution_time_ms: Simulated execution time.
            raise_error: Exception to raise.
        """
        self._check_config = MockCheckConfig(
            success=success,
            passed_count=passed_count,
            failed_count=failed_count,
            warning_count=warning_count,
            failures=failures or (),
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
        )

    def configure_profile(
        self,
        *,
        success: bool = True,
        row_count: int = 1000,
        columns: tuple[ColumnProfile, ...] | None = None,
        execution_time_ms: float = 200.0,
        raise_error: Exception | None = None,
    ) -> None:
        """Configure mock profile behavior.

        Args:
            success: Whether profiling should succeed.
            row_count: Simulated row count.
            columns: Mock column profiles.
            execution_time_ms: Simulated execution time.
            raise_error: Exception to raise.
        """
        self._profile_config = MockProfileConfig(
            success=success,
            row_count=row_count,
            columns=columns or (),
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
        )

    def configure_learn(
        self,
        *,
        success: bool = True,
        rules: tuple[LearnedRule, ...] | None = None,
        columns_analyzed: int = 5,
        execution_time_ms: float = 500.0,
        raise_error: Exception | None = None,
    ) -> None:
        """Configure mock learn behavior.

        Args:
            success: Whether learning should succeed.
            rules: Mock learned rules.
            columns_analyzed: Number of columns analyzed.
            execution_time_ms: Simulated execution time.
            raise_error: Exception to raise.
        """
        self._learn_config = MockLearnConfig(
            success=success,
            rules=rules or (),
            columns_analyzed=columns_analyzed,
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
        )

    def set_check_hook(
        self,
        hook: Callable[[Any, CheckConfig], CheckResult | None] | None,
    ) -> None:
        """Set a custom hook for check calls.

        The hook is called with (data, config) and should return a
        CheckResult or None. If None, the default mock behavior is used.

        Args:
            hook: Custom hook function.
        """
        self._check_hook = hook

    def set_profile_hook(
        self,
        hook: Callable[[Any, ProfileConfig], ProfileResult | None] | None,
    ) -> None:
        """Set a custom hook for profile calls.

        Args:
            hook: Custom hook function.
        """
        self._profile_hook = hook

    def set_learn_hook(
        self,
        hook: Callable[[Any, LearnConfig], LearnResult | None] | None,
    ) -> None:
        """Set a custom hook for learn calls.

        Args:
            hook: Custom hook function.
        """
        self._learn_hook = hook

    def check(self, data: Any, config: CheckConfig) -> CheckResult:
        """Execute mock validation check.

        Args:
            data: Data to validate.
            config: Check configuration.

        Returns:
            Mock CheckResult.

        Raises:
            Exception: If configured to raise an error.
        """
        self._check_calls.append((data, config))

        if self._check_config.raise_error:
            raise self._check_config.raise_error

        if self._check_hook:
            result = self._check_hook(data, config)
            if result is not None:
                return result

        cfg = self._check_config
        status = CheckStatus.PASSED if cfg.success else CheckStatus.FAILED
        if cfg.warning_count > 0 and cfg.failed_count == 0:
            status = CheckStatus.WARNING

        return CheckResult(
            status=status,
            passed_count=cfg.passed_count,
            failed_count=cfg.failed_count,
            warning_count=cfg.warning_count,
            failures=cfg.failures,
            execution_time_ms=cfg.execution_time_ms,
        )

    def profile(self, data: Any, config: ProfileConfig) -> ProfileResult:
        """Execute mock profiling.

        Args:
            data: Data to profile.
            config: Profile configuration.

        Returns:
            Mock ProfileResult.

        Raises:
            Exception: If configured to raise an error.
        """
        self._profile_calls.append((data, config))

        if self._profile_config.raise_error:
            raise self._profile_config.raise_error

        if self._profile_hook:
            result = self._profile_hook(data, config)
            if result is not None:
                return result

        cfg = self._profile_config
        return ProfileResult(
            status=ProfileStatus.COMPLETED if cfg.success else ProfileStatus.FAILED,
            row_count=cfg.row_count,
            column_count=len(cfg.columns),
            columns=cfg.columns,
            execution_time_ms=cfg.execution_time_ms,
        )

    def learn(self, data: Any, config: LearnConfig) -> LearnResult:
        """Execute mock schema learning.

        Args:
            data: Data to learn from.
            config: Learn configuration.

        Returns:
            Mock LearnResult.

        Raises:
            Exception: If configured to raise an error.
        """
        self._learn_calls.append((data, config))

        if self._learn_config.raise_error:
            raise self._learn_config.raise_error

        if self._learn_hook:
            result = self._learn_hook(data, config)
            if result is not None:
                return result

        cfg = self._learn_config
        return LearnResult(
            status=LearnStatus.COMPLETED if cfg.success else LearnStatus.FAILED,
            rules=cfg.rules,
            columns_analyzed=cfg.columns_analyzed,
            execution_time_ms=cfg.execution_time_ms,
        )

    def reset(self) -> None:
        """Reset all call history and configurations."""
        self._check_calls.clear()
        self._profile_calls.clear()
        self._learn_calls.clear()
        self._check_config = MockCheckConfig()
        self._profile_config = MockProfileConfig()
        self._learn_config = MockLearnConfig()
        self._check_hook = None
        self._profile_hook = None
        self._learn_hook = None


# =============================================================================
# Result Factories
# =============================================================================


def create_mock_check_result(
    *,
    success: bool = True,
    passed_count: int = 10,
    failed_count: int = 0,
    warning_count: int = 0,
    skipped_count: int = 0,
    failures: list[ValidationFailure] | None = None,
    execution_time_ms: float = 100.0,
    metadata: dict[str, Any] | None = None,
) -> CheckResult:
    """Create a mock CheckResult with customizable parameters.

    Args:
        success: Whether the result represents success.
        passed_count: Number of passed validations.
        failed_count: Number of failed validations.
        warning_count: Number of warnings.
        skipped_count: Number of skipped validations.
        failures: List of validation failures.
        execution_time_ms: Execution time in milliseconds.
        metadata: Additional metadata.

    Returns:
        CheckResult instance.

    Example:
        >>> result = create_mock_check_result(success=True, passed_count=5)
        >>> assert result.is_success
        >>> assert result.passed_count == 5
    """
    if success:
        status = CheckStatus.PASSED
    elif warning_count > 0 and failed_count == 0:
        status = CheckStatus.WARNING
    else:
        status = CheckStatus.FAILED

    return CheckResult(
        status=status,
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=warning_count,
        skipped_count=skipped_count,
        failures=tuple(failures or []),
        execution_time_ms=execution_time_ms,
        metadata=metadata or {},
    )


def create_mock_failure(
    *,
    rule_name: str = "not_null",
    column: str | None = "test_column",
    message: str = "Validation failed",
    severity: Severity = Severity.ERROR,
    failed_count: int = 5,
    total_count: int = 100,
    sample_values: list[Any] | None = None,
) -> ValidationFailure:
    """Create a mock ValidationFailure.

    Args:
        rule_name: Name of the validation rule.
        column: Column name.
        message: Failure message.
        severity: Failure severity.
        failed_count: Number of failed records.
        total_count: Total records checked.
        sample_values: Sample of failing values.

    Returns:
        ValidationFailure instance.

    Example:
        >>> failure = create_mock_failure(rule_name="not_null", failed_count=5)
        >>> assert failure.rule_name == "not_null"
    """
    return ValidationFailure(
        rule_name=rule_name,
        column=column,
        message=message,
        severity=severity,
        failed_count=failed_count,
        total_count=total_count,
        sample_values=tuple(sample_values or []),
    )


def create_mock_profile_result(
    *,
    success: bool = True,
    row_count: int = 1000,
    columns: list[ColumnProfile] | None = None,
    execution_time_ms: float = 200.0,
) -> ProfileResult:
    """Create a mock ProfileResult.

    Args:
        success: Whether profiling succeeded.
        row_count: Number of rows profiled.
        columns: Column profiles.
        execution_time_ms: Execution time.

    Returns:
        ProfileResult instance.
    """
    columns = columns or []
    return ProfileResult(
        status=ProfileStatus.COMPLETED if success else ProfileStatus.FAILED,
        row_count=row_count,
        column_count=len(columns),
        columns=tuple(columns),
        execution_time_ms=execution_time_ms,
    )


def create_mock_column_profile(
    *,
    column_name: str = "test_column",
    dtype: str = "Int64",
    null_count: int = 0,
    null_percentage: float = 0.0,
    unique_count: int = 100,
    unique_percentage: float = 10.0,
    min_value: Any = 0,
    max_value: Any = 100,
    mean: float | None = 50.0,
    std: float | None = 25.0,
) -> ColumnProfile:
    """Create a mock ColumnProfile.

    Args:
        column_name: Name of the column.
        dtype: Data type.
        null_count: Null value count.
        null_percentage: Null percentage.
        unique_count: Unique value count.
        unique_percentage: Unique percentage.
        min_value: Minimum value.
        max_value: Maximum value.
        mean: Mean value.
        std: Standard deviation.

    Returns:
        ColumnProfile instance.
    """
    return ColumnProfile(
        column_name=column_name,
        dtype=dtype,
        null_count=null_count,
        null_percentage=null_percentage,
        unique_count=unique_count,
        unique_percentage=unique_percentage,
        min_value=min_value,
        max_value=max_value,
        mean=mean,
        std=std,
    )


def create_mock_learn_result(
    *,
    success: bool = True,
    rules: list[LearnedRule] | None = None,
    columns_analyzed: int = 5,
    execution_time_ms: float = 500.0,
) -> LearnResult:
    """Create a mock LearnResult.

    Args:
        success: Whether learning succeeded.
        rules: Learned rules.
        columns_analyzed: Number of columns analyzed.
        execution_time_ms: Execution time.

    Returns:
        LearnResult instance.
    """
    return LearnResult(
        status=LearnStatus.COMPLETED if success else LearnStatus.FAILED,
        rules=tuple(rules or []),
        columns_analyzed=columns_analyzed,
        execution_time_ms=execution_time_ms,
    )


def create_mock_learned_rule(
    *,
    rule_type: str = "not_null",
    column: str = "test_column",
    parameters: dict[str, Any] | None = None,
    confidence: float = 0.99,
) -> LearnedRule:
    """Create a mock LearnedRule.

    Args:
        rule_type: Type of the rule.
        column: Column the rule applies to.
        parameters: Rule parameters.
        confidence: Confidence score.

    Returns:
        LearnedRule instance.
    """
    return LearnedRule(
        rule_type=rule_type,
        column=column,
        parameters=parameters or {},
        confidence=confidence,
    )


# =============================================================================
# Test Data Generators
# =============================================================================


def create_sample_dataframe(
    rows: int = 100,
    columns: list[str] | None = None,
    include_nulls: bool = False,
    null_rate: float = 0.1,
) -> pl.DataFrame:
    """Create a sample Polars DataFrame for testing.

    Args:
        rows: Number of rows.
        columns: Column names (default: ["id", "name", "value", "category"]).
        include_nulls: Whether to include null values.
        null_rate: Rate of null values (0.0-1.0).

    Returns:
        Polars DataFrame.

    Example:
        >>> df = create_sample_dataframe(rows=100, include_nulls=True)
        >>> assert len(df) == 100
    """
    import polars as pl

    columns = columns or ["id", "name", "value", "category"]
    data: dict[str, list[Any]] = {}

    for col in columns:
        if col == "id":
            values: list[Any] = list(range(rows))
        elif col == "name":
            values = [f"item_{i}" for i in range(rows)]
        elif col == "value":
            values = [random.uniform(0, 100) for _ in range(rows)]
        elif col == "category":
            categories = ["A", "B", "C", "D"]
            values = [random.choice(categories) for _ in range(rows)]
        else:
            values = [f"{col}_{i}" for i in range(rows)]

        if include_nulls:
            for i in range(rows):
                if random.random() < null_rate:
                    values[i] = None

        data[col] = values

    return pl.DataFrame(data)


def create_sample_config(
    *,
    rules: list[dict[str, Any]] | None = None,
    failure_action: FailureAction = FailureAction.RAISE,
    timeout_seconds: int = 60,
) -> CheckConfig:
    """Create a sample CheckConfig for testing.

    Args:
        rules: Validation rules.
        failure_action: Action on failure.
        timeout_seconds: Timeout in seconds.

    Returns:
        CheckConfig instance.

    Example:
        >>> config = create_sample_config(rules=[{"type": "not_null", "column": "id"}])
        >>> assert len(config.rules) == 1
    """
    default_rules = [
        {"type": "not_null", "column": "id"},
        {"type": "unique", "column": "id"},
    ]
    return CheckConfig(
        rules=tuple(rules or default_rules),
        failure_action=failure_action,
        timeout_seconds=timeout_seconds,
    )


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_check_result(
    result: CheckResult,
    *,
    expected_success: bool | None = None,
    expected_status: CheckStatus | None = None,
    min_pass_rate: float | None = None,
    max_failures: int | None = None,
) -> None:
    """Assert CheckResult matches expectations.

    Args:
        result: Result to check.
        expected_success: Expected success status.
        expected_status: Expected status enum.
        min_pass_rate: Minimum pass rate percentage.
        max_failures: Maximum number of failures.

    Raises:
        AssertionError: If expectations not met.

    Example:
        >>> result = create_mock_check_result(success=True)
        >>> assert_check_result(result, expected_success=True, min_pass_rate=90.0)
    """
    if expected_success is not None:
        assert result.is_success == expected_success, (
            f"Expected success={expected_success}, got {result.is_success}"
        )

    if expected_status is not None:
        assert result.status == expected_status, (
            f"Expected status={expected_status}, got {result.status}"
        )

    if min_pass_rate is not None:
        assert result.pass_rate >= min_pass_rate, (
            f"Expected pass_rate >= {min_pass_rate}, got {result.pass_rate}"
        )

    if max_failures is not None:
        assert len(result.failures) <= max_failures, (
            f"Expected max {max_failures} failures, got {len(result.failures)}"
        )


def assert_profile_result(
    result: ProfileResult,
    *,
    expected_success: bool | None = None,
    expected_row_count: int | None = None,
    expected_columns: list[str] | None = None,
) -> None:
    """Assert ProfileResult matches expectations.

    Args:
        result: Result to check.
        expected_success: Expected success status.
        expected_row_count: Expected row count.
        expected_columns: Expected column names.

    Raises:
        AssertionError: If expectations not met.
    """
    if expected_success is not None:
        assert result.is_success == expected_success, (
            f"Expected success={expected_success}, got {result.is_success}"
        )

    if expected_row_count is not None:
        assert result.row_count == expected_row_count, (
            f"Expected row_count={expected_row_count}, got {result.row_count}"
        )

    if expected_columns is not None:
        actual_columns = [c.column_name for c in result.columns]
        assert set(actual_columns) == set(expected_columns), (
            f"Expected columns={expected_columns}, got {actual_columns}"
        )


# =============================================================================
# Platform Context Mocks
# =============================================================================


@dataclass
class MockAirflowContext:
    """Mock Airflow task instance context.

    Attributes:
        task_id: Task identifier.
        dag_id: DAG identifier.
        run_id: Run identifier.
        execution_date: Execution date.
        xcom_values: XCom push/pull storage.
    """

    task_id: str = "test_task"
    dag_id: str = "test_dag"
    run_id: str = "test_run"
    execution_date: datetime = field(default_factory=lambda: datetime.now(UTC))
    xcom_values: dict[str, Any] = field(default_factory=dict)

    def xcom_push(self, key: str, value: Any) -> None:
        """Push a value to XCom."""
        self.xcom_values[key] = value

    def xcom_pull(self, task_ids: str, key: str = "return_value") -> Any:
        """Pull a value from XCom."""
        return self.xcom_values.get(key)


def create_mock_airflow_context(**kwargs: Any) -> MockAirflowContext:
    """Create a mock Airflow context.

    Args:
        **kwargs: Context attributes to override.

    Returns:
        MockAirflowContext instance.
    """
    return MockAirflowContext(**kwargs)


@dataclass
class MockDagsterContext:
    """Mock Dagster op context.

    Attributes:
        op_name: Operation name.
        run_id: Run identifier.
        resources: Mock resources.
        log: Mock logger.
    """

    op_name: str = "test_op"
    run_id: str = "test_run"
    resources: dict[str, Any] = field(default_factory=dict)
    log: Any = field(default_factory=lambda: MagicMock())

    def get_resource(self, name: str) -> Any:
        """Get a resource by name."""
        return self.resources.get(name)


def create_mock_dagster_context(**kwargs: Any) -> MockDagsterContext:
    """Create a mock Dagster context.

    Args:
        **kwargs: Context attributes to override.

    Returns:
        MockDagsterContext instance.
    """
    return MockDagsterContext(**kwargs)


@dataclass
class MockPrefectContext:
    """Mock Prefect task run context.

    Attributes:
        task_run_id: Task run identifier.
        flow_run_id: Flow run identifier.
        task_name: Task name.
    """

    task_run_id: str = "test_task_run"
    flow_run_id: str = "test_flow_run"
    task_name: str = "test_task"


def create_mock_prefect_context(**kwargs: Any) -> MockPrefectContext:
    """Create a mock Prefect context.

    Args:
        **kwargs: Context attributes to override.

    Returns:
        MockPrefectContext instance.
    """
    return MockPrefectContext(**kwargs)


# =============================================================================
# Fixtures and Context Managers
# =============================================================================


class TruthoundTestContext:
    """Context manager for Truthound tests.

    Provides a clean testing environment with mock Truthound and
    sample data.

    Example:
        >>> with TruthoundTestContext() as ctx:
        ...     ctx.mock.configure_check(success=True)
        ...     result = ctx.mock.check(ctx.sample_data, ctx.sample_config)
        ...     assert result.is_success
    """

    def __init__(
        self,
        rows: int = 100,
        include_nulls: bool = False,
    ) -> None:
        """Initialize the test context.

        Args:
            rows: Number of rows in sample data.
            include_nulls: Whether sample data includes nulls.
        """
        self._rows = rows
        self._include_nulls = include_nulls
        self._mock: MockTruthound | None = None
        self._sample_data: pl.DataFrame | None = None
        self._sample_config: CheckConfig | None = None

    @property
    def mock(self) -> MockTruthound:
        """Get the mock Truthound instance."""
        if self._mock is None:
            raise RuntimeError("Context not entered")
        return self._mock

    @property
    def sample_data(self) -> pl.DataFrame:
        """Get the sample DataFrame."""
        if self._sample_data is None:
            raise RuntimeError("Context not entered")
        return self._sample_data

    @property
    def sample_config(self) -> CheckConfig:
        """Get the sample config."""
        if self._sample_config is None:
            raise RuntimeError("Context not entered")
        return self._sample_config

    def __enter__(self) -> TruthoundTestContext:
        """Enter the test context."""
        self._mock = MockTruthound()
        self._sample_data = create_sample_dataframe(
            rows=self._rows,
            include_nulls=self._include_nulls,
        )
        self._sample_config = create_sample_config()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the test context."""
        if self._mock is not None:
            self._mock.reset()
        self._mock = None
        self._sample_data = None
        self._sample_config = None


# =============================================================================
# Cache Test Utilities
# =============================================================================


class MockCacheBackend:
    """Mock cache backend for testing.

    Provides configurable cache behavior with call tracking.

    Example:
        >>> cache = MockCacheBackend()
        >>> cache.set("key", "value")
        >>> assert cache.get("key") == "value"
        >>> assert cache.set_call_count == 1
    """

    def __init__(
        self,
        *,
        simulate_miss: bool = False,
        raise_on_get: Exception | None = None,
        raise_on_set: Exception | None = None,
    ) -> None:
        """Initialize mock cache backend.

        Args:
            simulate_miss: Always return None on get (simulate miss).
            raise_on_get: Exception to raise on get.
            raise_on_set: Exception to raise on set.
        """
        self._data: dict[str, Any] = {}
        self._simulate_miss = simulate_miss
        self._raise_on_get = raise_on_get
        self._raise_on_set = raise_on_set

        self._get_calls: list[str] = []
        self._set_calls: list[tuple[str, Any, float | None]] = []
        self._delete_calls: list[str] = []

    @property
    def get_call_count(self) -> int:
        """Get number of get calls."""
        return len(self._get_calls)

    @property
    def set_call_count(self) -> int:
        """Get number of set calls."""
        return len(self._set_calls)

    @property
    def delete_call_count(self) -> int:
        """Get number of delete calls."""
        return len(self._delete_calls)

    def get_get_calls(self) -> list[str]:
        """Get all get call keys."""
        return list(self._get_calls)

    def get_set_calls(self) -> list[tuple[str, Any, float | None]]:
        """Get all set call arguments."""
        return list(self._set_calls)

    def get(self, key: str) -> Any | None:
        """Get a value from the cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None.
        """
        self._get_calls.append(key)

        if self._raise_on_get:
            raise self._raise_on_get

        if self._simulate_miss:
            return None

        return self._data.get(key)

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Optional TTL.
        """
        self._set_calls.append((key, value, ttl_seconds))

        if self._raise_on_set:
            raise self._raise_on_set

        self._data[key] = value

    def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key.

        Returns:
            True if deleted.
        """
        self._delete_calls.append(key)
        if key in self._data:
            del self._data[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all values."""
        self._data.clear()

    def contains(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: The cache key.

        Returns:
            True if exists.
        """
        if self._simulate_miss:
            return False
        return key in self._data

    def size(self) -> int:
        """Get number of items.

        Returns:
            Number of cached items.
        """
        return len(self._data)

    def reset(self) -> None:
        """Reset all state and call history."""
        self._data.clear()
        self._get_calls.clear()
        self._set_calls.clear()
        self._delete_calls.clear()


class MockCacheHook:
    """Mock cache hook for testing.

    Records all cache events for verification in tests.

    Example:
        >>> hook = MockCacheHook()
        >>> hook.on_hit("key", "value", {})
        >>> assert hook.hit_count == 1
    """

    def __init__(self) -> None:
        """Initialize mock hook."""
        self._hits: list[tuple[str, Any, dict[str, Any]]] = []
        self._misses: list[tuple[str, dict[str, Any]]] = []
        self._sets: list[tuple[str, Any, float | None, dict[str, Any]]] = []
        self._evicts: list[tuple[str, str, dict[str, Any]]] = []

    @property
    def hit_count(self) -> int:
        """Get hit event count."""
        return len(self._hits)

    @property
    def miss_count(self) -> int:
        """Get miss event count."""
        return len(self._misses)

    @property
    def set_count(self) -> int:
        """Get set event count."""
        return len(self._sets)

    @property
    def evict_count(self) -> int:
        """Get evict event count."""
        return len(self._evicts)

    def get_hits(self) -> list[tuple[str, Any, dict[str, Any]]]:
        """Get all hit events."""
        return list(self._hits)

    def get_misses(self) -> list[tuple[str, dict[str, Any]]]:
        """Get all miss events."""
        return list(self._misses)

    def get_sets(self) -> list[tuple[str, Any, float | None, dict[str, Any]]]:
        """Get all set events."""
        return list(self._sets)

    def get_evicts(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Get all evict events."""
        return list(self._evicts)

    def on_hit(
        self,
        key: str,
        value: Any,
        context: dict[str, Any],
    ) -> None:
        """Record cache hit."""
        self._hits.append((key, value, context))

    def on_miss(
        self,
        key: str,
        context: dict[str, Any],
    ) -> None:
        """Record cache miss."""
        self._misses.append((key, context))

    def on_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None,
        context: dict[str, Any],
    ) -> None:
        """Record cache set."""
        self._sets.append((key, value, ttl_seconds, context))

    def on_evict(
        self,
        key: str,
        reason: str,
        context: dict[str, Any],
    ) -> None:
        """Record cache eviction."""
        self._evicts.append((key, reason, context))

    def reset(self) -> None:
        """Reset all recorded events."""
        self._hits.clear()
        self._misses.clear()
        self._sets.clear()
        self._evicts.clear()


def create_mock_cache_backend(
    *,
    initial_data: dict[str, Any] | None = None,
    simulate_miss: bool = False,
    raise_on_get: Exception | None = None,
    raise_on_set: Exception | None = None,
) -> MockCacheBackend:
    """Create a mock cache backend.

    Args:
        initial_data: Initial cache data.
        simulate_miss: Always simulate cache miss.
        raise_on_get: Exception to raise on get.
        raise_on_set: Exception to raise on set.

    Returns:
        MockCacheBackend instance.

    Example:
        >>> cache = create_mock_cache_backend(initial_data={"key": "value"})
        >>> assert cache.get("key") == "value"
    """
    backend = MockCacheBackend(
        simulate_miss=simulate_miss,
        raise_on_get=raise_on_get,
        raise_on_set=raise_on_set,
    )
    if initial_data:
        for key, value in initial_data.items():
            backend._data[key] = value
    return backend


def create_mock_cache_hook() -> MockCacheHook:
    """Create a mock cache hook.

    Returns:
        MockCacheHook instance.

    Example:
        >>> hook = create_mock_cache_hook()
        >>> hook.on_hit("key", "value", {})
        >>> assert hook.hit_count == 1
    """
    return MockCacheHook()


def assert_cache_stats(
    stats: Any,
    *,
    expected_size: int | None = None,
    min_hit_rate: float | None = None,
    max_miss_rate: float | None = None,
    max_evictions: int | None = None,
) -> None:
    """Assert cache statistics match expectations.

    Args:
        stats: CacheStats to check.
        expected_size: Expected cache size.
        min_hit_rate: Minimum hit rate (0.0 to 1.0).
        max_miss_rate: Maximum miss rate (0.0 to 1.0).
        max_evictions: Maximum eviction count.

    Raises:
        AssertionError: If expectations not met.

    Example:
        >>> from common.cache import LRUCache
        >>> cache = LRUCache(max_size=10)
        >>> cache.set("key", "value")
        >>> cache.get("key")
        >>> assert_cache_stats(cache.get_stats(), expected_size=1, min_hit_rate=0.5)
    """
    if expected_size is not None:
        assert stats.size == expected_size, (
            f"Expected size={expected_size}, got {stats.size}"
        )

    if min_hit_rate is not None:
        assert stats.hit_rate >= min_hit_rate, (
            f"Expected hit_rate >= {min_hit_rate}, got {stats.hit_rate}"
        )

    if max_miss_rate is not None:
        assert stats.miss_rate <= max_miss_rate, (
            f"Expected miss_rate <= {max_miss_rate}, got {stats.miss_rate}"
        )

    if max_evictions is not None:
        assert stats.evictions <= max_evictions, (
            f"Expected evictions <= {max_evictions}, got {stats.evictions}"
        )


# =============================================================================
# Async Mock Data Quality Engine
# =============================================================================


class AsyncMockDataQualityEngine:
    """Async mock implementation of AsyncDataQualityEngine for testing.

    Provides configurable async mock responses for check, profile, and learn
    operations. Implements the AsyncDataQualityEngine protocol for use in tests.

    All methods are async and can simulate delays for testing async behavior.

    Example:
        >>> engine = AsyncMockDataQualityEngine()
        >>> engine.configure_check(success=True, passed_count=10)
        >>> result = await engine.check(data, rules=[{"type": "not_null", "column": "id"}])
        >>> assert result.is_success
        >>> assert engine.check_call_count == 1

    With simulated delay:
        >>> engine.configure_check(success=True, delay_seconds=0.5)
        >>> # This call will take ~0.5 seconds
        >>> result = await engine.check(data, rules=[])
    """

    def __init__(
        self,
        name: str = "async_mock",
        version: str = "1.0.0",
    ) -> None:
        """Initialize the async mock engine.

        Args:
            name: Engine name.
            version: Engine version.
        """
        self._name = name
        self._version = version
        self._check_config: AsyncMockCheckConfig = AsyncMockCheckConfig()
        self._profile_config: AsyncMockProfileConfig = AsyncMockProfileConfig()
        self._learn_config: AsyncMockLearnConfig = AsyncMockLearnConfig()

        self._check_calls: list[tuple[Any, tuple[dict[str, Any], ...]]] = []
        self._profile_calls: list[tuple[Any, dict[str, Any]]] = []
        self._learn_calls: list[tuple[Any, dict[str, Any]]] = []

    @property
    def engine_name(self) -> str:
        """Return the engine name."""
        return self._name

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        return self._version

    @property
    def check_call_count(self) -> int:
        """Return the number of check calls."""
        return len(self._check_calls)

    @property
    def profile_call_count(self) -> int:
        """Return the number of profile calls."""
        return len(self._profile_calls)

    @property
    def learn_call_count(self) -> int:
        """Return the number of learn calls."""
        return len(self._learn_calls)

    def get_check_calls(self) -> list[tuple[Any, tuple[dict[str, Any], ...]]]:
        """Get all check call arguments."""
        return list(self._check_calls)

    def get_profile_calls(self) -> list[tuple[Any, dict[str, Any]]]:
        """Get all profile call arguments."""
        return list(self._profile_calls)

    def get_learn_calls(self) -> list[tuple[Any, dict[str, Any]]]:
        """Get all learn call arguments."""
        return list(self._learn_calls)

    def configure_check(
        self,
        *,
        success: bool = True,
        passed_count: int = 10,
        failed_count: int = 0,
        warning_count: int = 0,
        failures: tuple[ValidationFailure, ...] | None = None,
        execution_time_ms: float = 100.0,
        raise_error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """Configure mock check behavior.

        Args:
            success: Whether check should succeed.
            passed_count: Number of passed validations.
            failed_count: Number of failed validations.
            warning_count: Number of warnings.
            failures: Custom failures to return.
            execution_time_ms: Simulated execution time in result.
            raise_error: Exception to raise.
            delay_seconds: Async delay before returning (for testing async behavior).
        """
        self._check_config = AsyncMockCheckConfig(
            success=success,
            passed_count=passed_count,
            failed_count=failed_count,
            warning_count=warning_count,
            failures=failures or (),
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
            delay_seconds=delay_seconds,
        )

    def configure_profile(
        self,
        *,
        success: bool = True,
        row_count: int = 1000,
        columns: tuple[ColumnProfile, ...] | None = None,
        execution_time_ms: float = 200.0,
        raise_error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """Configure mock profile behavior.

        Args:
            success: Whether profiling should succeed.
            row_count: Simulated row count.
            columns: Mock column profiles.
            execution_time_ms: Simulated execution time in result.
            raise_error: Exception to raise.
            delay_seconds: Async delay before returning.
        """
        self._profile_config = AsyncMockProfileConfig(
            success=success,
            row_count=row_count,
            columns=columns or (),
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
            delay_seconds=delay_seconds,
        )

    def configure_learn(
        self,
        *,
        success: bool = True,
        rules: tuple[LearnedRule, ...] | None = None,
        columns_analyzed: int = 5,
        execution_time_ms: float = 500.0,
        raise_error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        """Configure mock learn behavior.

        Args:
            success: Whether learning should succeed.
            rules: Mock learned rules.
            columns_analyzed: Number of columns analyzed.
            execution_time_ms: Simulated execution time in result.
            raise_error: Exception to raise.
            delay_seconds: Async delay before returning.
        """
        self._learn_config = AsyncMockLearnConfig(
            success=success,
            rules=rules or (),
            columns_analyzed=columns_analyzed,
            execution_time_ms=execution_time_ms,
            raise_error=raise_error,
            delay_seconds=delay_seconds,
        )

    async def check(
        self,
        data: Any,
        rules: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute async mock validation check.

        Args:
            data: Data to validate.
            rules: Validation rules.
            **kwargs: Additional parameters.

        Returns:
            Mock CheckResult.

        Raises:
            Exception: If configured to raise an error.
        """
        import asyncio

        self._check_calls.append((data, tuple(rules)))

        if self._check_config.delay_seconds > 0:
            await asyncio.sleep(self._check_config.delay_seconds)

        if self._check_config.raise_error:
            raise self._check_config.raise_error

        cfg = self._check_config
        status = CheckStatus.PASSED if cfg.success else CheckStatus.FAILED
        if cfg.warning_count > 0 and cfg.failed_count == 0:
            status = CheckStatus.WARNING

        return CheckResult(
            status=status,
            passed_count=cfg.passed_count,
            failed_count=cfg.failed_count,
            warning_count=cfg.warning_count,
            failures=cfg.failures,
            execution_time_ms=cfg.execution_time_ms,
            metadata={"engine": self._name, "async": True},
        )

    async def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        """Execute async mock profiling.

        Args:
            data: Data to profile.
            **kwargs: Additional parameters.

        Returns:
            Mock ProfileResult.

        Raises:
            Exception: If configured to raise an error.
        """
        import asyncio

        self._profile_calls.append((data, kwargs))

        if self._profile_config.delay_seconds > 0:
            await asyncio.sleep(self._profile_config.delay_seconds)

        if self._profile_config.raise_error:
            raise self._profile_config.raise_error

        cfg = self._profile_config
        return ProfileResult(
            status=ProfileStatus.COMPLETED if cfg.success else ProfileStatus.FAILED,
            row_count=cfg.row_count,
            column_count=len(cfg.columns),
            columns=cfg.columns,
            execution_time_ms=cfg.execution_time_ms,
            metadata={"engine": self._name, "async": True},
        )

    async def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        """Execute async mock schema learning.

        Args:
            data: Data to learn from.
            **kwargs: Additional parameters.

        Returns:
            Mock LearnResult.

        Raises:
            Exception: If configured to raise an error.
        """
        import asyncio

        self._learn_calls.append((data, kwargs))

        if self._learn_config.delay_seconds > 0:
            await asyncio.sleep(self._learn_config.delay_seconds)

        if self._learn_config.raise_error:
            raise self._learn_config.raise_error

        cfg = self._learn_config
        return LearnResult(
            status=LearnStatus.COMPLETED if cfg.success else LearnStatus.FAILED,
            rules=cfg.rules,
            columns_analyzed=cfg.columns_analyzed,
            execution_time_ms=cfg.execution_time_ms,
            metadata={"engine": self._name, "async": True},
        )

    def reset(self) -> None:
        """Reset all call history and configurations."""
        self._check_calls.clear()
        self._profile_calls.clear()
        self._learn_calls.clear()
        self._check_config = AsyncMockCheckConfig()
        self._profile_config = AsyncMockProfileConfig()
        self._learn_config = AsyncMockLearnConfig()


@dataclass
class AsyncMockCheckConfig:
    """Configuration for async mock check behavior.

    Attributes:
        success: Whether check should succeed.
        passed_count: Number of passed validations.
        failed_count: Number of failed validations.
        warning_count: Number of warnings.
        failures: Custom failures to return.
        execution_time_ms: Simulated execution time.
        raise_error: Exception to raise (if any).
        delay_seconds: Async delay before returning.
    """

    success: bool = True
    passed_count: int = 10
    failed_count: int = 0
    warning_count: int = 0
    failures: tuple[ValidationFailure, ...] = ()
    execution_time_ms: float = 100.0
    raise_error: Exception | None = None
    delay_seconds: float = 0.0


@dataclass
class AsyncMockProfileConfig:
    """Configuration for async mock profile behavior.

    Attributes:
        success: Whether profiling should succeed.
        row_count: Simulated row count.
        columns: Mock column profiles.
        execution_time_ms: Simulated execution time.
        raise_error: Exception to raise (if any).
        delay_seconds: Async delay before returning.
    """

    success: bool = True
    row_count: int = 1000
    columns: tuple[ColumnProfile, ...] = ()
    execution_time_ms: float = 200.0
    raise_error: Exception | None = None
    delay_seconds: float = 0.0


@dataclass
class AsyncMockLearnConfig:
    """Configuration for async mock learn behavior.

    Attributes:
        success: Whether learning should succeed.
        rules: Mock learned rules.
        columns_analyzed: Number of columns analyzed.
        execution_time_ms: Simulated execution time.
        raise_error: Exception to raise (if any).
        delay_seconds: Async delay before returning.
    """

    success: bool = True
    rules: tuple[LearnedRule, ...] = ()
    columns_analyzed: int = 5
    execution_time_ms: float = 500.0
    raise_error: Exception | None = None
    delay_seconds: float = 0.0


# =============================================================================
# Async Mock Managed Engine
# =============================================================================


class AsyncMockManagedEngine:
    """Async mock implementation of AsyncManagedEngine for testing.

    Provides configurable async mock responses for check, profile, learn,
    and lifecycle operations. Implements both AsyncDataQualityEngine and
    AsyncManagedEngine protocols for complete testing coverage.

    Example:
        >>> engine = AsyncMockManagedEngine()
        >>> async with engine:
        ...     result = await engine.check(data, rules=[])
        ...     health = await engine.health_check()

        >>> # Configure lifecycle behavior
        >>> engine.configure_lifecycle(start_delay=0.1, health_status="unhealthy")
        >>> await engine.start()  # Takes ~0.1 seconds
        >>> health = await engine.health_check()  # Returns UNHEALTHY
    """

    def __init__(
        self,
        name: str = "async_mock_managed",
        version: str = "1.0.0",
    ) -> None:
        """Initialize the async mock managed engine.

        Args:
            name: Engine name.
            version: Engine version.
        """
        from common.engines.lifecycle import (
            DEFAULT_ENGINE_CONFIG,
            EngineStateTracker,
        )
        from common.health import HealthStatus

        self._name = name
        self._version = version
        self._engine = AsyncMockDataQualityEngine(name=name, version=version)

        # Lifecycle state
        self._lifecycle_config = DEFAULT_ENGINE_CONFIG
        self._state_tracker = EngineStateTracker(name)

        # Lifecycle mock config
        self._start_delay_seconds: float = 0.0
        self._stop_delay_seconds: float = 0.0
        self._health_check_delay_seconds: float = 0.0
        self._start_raise_error: Exception | None = None
        self._stop_raise_error: Exception | None = None
        self._health_status: HealthStatus = HealthStatus.HEALTHY
        self._health_message: str = "Mock engine is healthy"

    @property
    def engine_name(self) -> str:
        """Return the engine name."""
        return self._name

    @property
    def engine_version(self) -> str:
        """Return the engine version."""
        return self._version

    def configure_check(self, **kwargs: Any) -> None:
        """Configure mock check behavior. Delegates to inner engine."""
        self._engine.configure_check(**kwargs)

    def configure_profile(self, **kwargs: Any) -> None:
        """Configure mock profile behavior. Delegates to inner engine."""
        self._engine.configure_profile(**kwargs)

    def configure_learn(self, **kwargs: Any) -> None:
        """Configure mock learn behavior. Delegates to inner engine."""
        self._engine.configure_learn(**kwargs)

    def configure_lifecycle(
        self,
        *,
        start_delay_seconds: float = 0.0,
        stop_delay_seconds: float = 0.0,
        health_check_delay_seconds: float = 0.0,
        start_raise_error: Exception | None = None,
        stop_raise_error: Exception | None = None,
        health_status: str = "healthy",
        health_message: str = "Mock engine is healthy",
    ) -> None:
        """Configure mock lifecycle behavior.

        Args:
            start_delay_seconds: Delay during start().
            stop_delay_seconds: Delay during stop().
            health_check_delay_seconds: Delay during health_check().
            start_raise_error: Exception to raise during start().
            stop_raise_error: Exception to raise during stop().
            health_status: Health status ("healthy", "degraded", "unhealthy", "unknown").
            health_message: Message to include in health check result.
        """
        from common.health import HealthStatus

        self._start_delay_seconds = start_delay_seconds
        self._stop_delay_seconds = stop_delay_seconds
        self._health_check_delay_seconds = health_check_delay_seconds
        self._start_raise_error = start_raise_error
        self._stop_raise_error = stop_raise_error
        self._health_message = health_message

        status_map = {
            "healthy": HealthStatus.HEALTHY,
            "degraded": HealthStatus.DEGRADED,
            "unhealthy": HealthStatus.UNHEALTHY,
            "unknown": HealthStatus.UNKNOWN,
        }
        self._health_status = status_map.get(health_status.lower(), HealthStatus.HEALTHY)

    async def check(
        self,
        data: Any,
        rules: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> CheckResult:
        """Execute async mock validation check."""
        return await self._engine.check(data, rules, **kwargs)

    async def profile(self, data: Any, **kwargs: Any) -> ProfileResult:
        """Execute async mock profiling."""
        return await self._engine.profile(data, **kwargs)

    async def learn(self, data: Any, **kwargs: Any) -> LearnResult:
        """Execute async mock schema learning."""
        return await self._engine.learn(data, **kwargs)

    async def start(self) -> None:
        """Start the async mock engine."""
        import asyncio

        from common.engines.lifecycle import (
            EngineAlreadyStartedError,
            EngineInitializationError,
            EngineState,
            EngineStoppedError,
        )

        state = self._state_tracker.state
        if state.is_terminal:
            raise EngineStoppedError(self.engine_name)
        if state.is_active:
            raise EngineAlreadyStartedError(self.engine_name)

        self._state_tracker.transition_to(EngineState.STARTING)

        if self._start_delay_seconds > 0:
            await asyncio.sleep(self._start_delay_seconds)

        if self._start_raise_error:
            self._state_tracker.transition_to(EngineState.FAILED)
            raise EngineInitializationError(
                f"Mock start failure: {self._start_raise_error}",
                engine_name=self.engine_name,
                cause=self._start_raise_error,
            )

        self._state_tracker.transition_to(EngineState.RUNNING)

    async def stop(self) -> None:
        """Stop the async mock engine."""
        import asyncio

        from common.engines.lifecycle import EngineShutdownError, EngineState

        state = self._state_tracker.state
        if state == EngineState.STOPPED:
            return
        if not state.can_stop:
            return

        self._state_tracker.transition_to(EngineState.STOPPING)

        if self._stop_delay_seconds > 0:
            await asyncio.sleep(self._stop_delay_seconds)

        if self._stop_raise_error:
            self._state_tracker.transition_to(EngineState.FAILED)
            raise EngineShutdownError(
                f"Mock stop failure: {self._stop_raise_error}",
                engine_name=self.engine_name,
                cause=self._stop_raise_error,
            )

        self._state_tracker.transition_to(EngineState.STOPPED)

    async def health_check(self) -> Any:
        """Perform async mock health check."""
        import asyncio

        from common.engines.lifecycle import EngineState
        from common.health import HealthCheckResult, HealthStatus

        if self._health_check_delay_seconds > 0:
            await asyncio.sleep(self._health_check_delay_seconds)

        state = self._state_tracker.state
        if state != EngineState.RUNNING:
            return HealthCheckResult(
                name=self.engine_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Engine not running (state: {state.name})",
            )

        if self._health_status == HealthStatus.HEALTHY:
            return HealthCheckResult.healthy(self.engine_name, message=self._health_message)
        elif self._health_status == HealthStatus.DEGRADED:
            return HealthCheckResult.degraded(self.engine_name, message=self._health_message)
        elif self._health_status == HealthStatus.UNHEALTHY:
            return HealthCheckResult.unhealthy(self.engine_name, message=self._health_message)
        else:
            return HealthCheckResult.unknown(self.engine_name, message=self._health_message)

    def get_state(self) -> Any:
        """Get current engine state."""
        return self._state_tracker.state

    def get_state_snapshot(self) -> Any:
        """Get state snapshot."""
        return self._state_tracker.get_snapshot()

    @property
    def check_call_count(self) -> int:
        """Return the number of check calls."""
        return self._engine.check_call_count

    @property
    def profile_call_count(self) -> int:
        """Return the number of profile calls."""
        return self._engine.profile_call_count

    @property
    def learn_call_count(self) -> int:
        """Return the number of learn calls."""
        return self._engine.learn_call_count

    def reset(self) -> None:
        """Reset all call history, configurations, and state."""
        from common.engines.lifecycle import EngineStateTracker

        self._engine.reset()
        self._state_tracker = EngineStateTracker(self._name)
        self._start_delay_seconds = 0.0
        self._stop_delay_seconds = 0.0
        self._health_check_delay_seconds = 0.0
        self._start_raise_error = None
        self._stop_raise_error = None

    async def __aenter__(self) -> AsyncMockManagedEngine:
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager."""
        await self.stop()


# =============================================================================
# Async Test Utilities
# =============================================================================


def create_async_mock_engine(
    *,
    name: str = "async_mock",
    version: str = "1.0.0",
    check_success: bool = True,
    check_delay_seconds: float = 0.0,
) -> AsyncMockDataQualityEngine:
    """Create a configured async mock engine.

    Args:
        name: Engine name.
        version: Engine version.
        check_success: Whether check should succeed by default.
        check_delay_seconds: Delay for check calls.

    Returns:
        Configured AsyncMockDataQualityEngine.

    Example:
        >>> engine = create_async_mock_engine(check_success=True)
        >>> result = await engine.check(data, rules=[])
        >>> assert result.is_success
    """
    engine = AsyncMockDataQualityEngine(name=name, version=version)
    engine.configure_check(success=check_success, delay_seconds=check_delay_seconds)
    return engine


def create_async_mock_managed_engine(
    *,
    name: str = "async_mock_managed",
    version: str = "1.0.0",
    check_success: bool = True,
    health_status: str = "healthy",
) -> AsyncMockManagedEngine:
    """Create a configured async mock managed engine.

    Args:
        name: Engine name.
        version: Engine version.
        check_success: Whether check should succeed by default.
        health_status: Health status ("healthy", "degraded", "unhealthy", "unknown").

    Returns:
        Configured AsyncMockManagedEngine.

    Example:
        >>> engine = create_async_mock_managed_engine(health_status="healthy")
        >>> async with engine:
        ...     health = await engine.health_check()
        ...     assert health.status == HealthStatus.HEALTHY
    """
    engine = AsyncMockManagedEngine(name=name, version=version)
    engine.configure_check(success=check_success)
    engine.configure_lifecycle(health_status=health_status)
    return engine


# =============================================================================
# Async Lifecycle Hook Mock
# =============================================================================


class AsyncMockLifecycleHook:
    """Async mock lifecycle hook for testing.

    Records all lifecycle events for verification in tests.

    Example:
        >>> hook = AsyncMockLifecycleHook()
        >>> manager = AsyncEngineLifecycleManager(engine, hooks=[hook])
        >>> await manager.start()
        >>> assert hook.start_count == 1
        >>> assert "my_engine" in hook.started_engines
    """

    def __init__(self) -> None:
        """Initialize async mock hook."""
        self._starting_events: list[tuple[str, dict[str, Any]]] = []
        self._started_events: list[tuple[str, float, dict[str, Any]]] = []
        self._stopping_events: list[tuple[str, dict[str, Any]]] = []
        self._stopped_events: list[tuple[str, float, dict[str, Any]]] = []
        self._error_events: list[tuple[str, Exception, dict[str, Any]]] = []
        self._health_check_events: list[tuple[str, Any, dict[str, Any]]] = []

    @property
    def start_count(self) -> int:
        """Get number of started events."""
        return len(self._started_events)

    @property
    def stop_count(self) -> int:
        """Get number of stopped events."""
        return len(self._stopped_events)

    @property
    def error_count(self) -> int:
        """Get number of error events."""
        return len(self._error_events)

    @property
    def health_check_count(self) -> int:
        """Get number of health check events."""
        return len(self._health_check_events)

    @property
    def started_engines(self) -> list[str]:
        """Get list of engines that were started."""
        return [name for name, _, _ in self._started_events]

    @property
    def stopped_engines(self) -> list[str]:
        """Get list of engines that were stopped."""
        return [name for name, _, _ in self._stopped_events]

    def get_errors(self) -> list[tuple[str, Exception]]:
        """Get list of (engine_name, error) tuples."""
        return [(name, error) for name, error, _ in self._error_events]

    async def on_starting(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record starting event."""
        self._starting_events.append((engine_name, context))

    async def on_started(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record started event."""
        self._started_events.append((engine_name, duration_ms, context))

    async def on_stopping(
        self,
        engine_name: str,
        context: dict[str, Any],
    ) -> None:
        """Record stopping event."""
        self._stopping_events.append((engine_name, context))

    async def on_stopped(
        self,
        engine_name: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record stopped event."""
        self._stopped_events.append((engine_name, duration_ms, context))

    async def on_error(
        self,
        engine_name: str,
        error: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record error event."""
        self._error_events.append((engine_name, error, context))

    async def on_health_check(
        self,
        engine_name: str,
        result: Any,
        context: dict[str, Any],
    ) -> None:
        """Record health check event."""
        self._health_check_events.append((engine_name, result, context))

    def reset(self) -> None:
        """Reset all recorded events."""
        self._starting_events.clear()
        self._started_events.clear()
        self._stopping_events.clear()
        self._stopped_events.clear()
        self._error_events.clear()
        self._health_check_events.clear()


def create_async_mock_lifecycle_hook() -> AsyncMockLifecycleHook:
    """Create an async mock lifecycle hook.

    Returns:
        AsyncMockLifecycleHook instance.

    Example:
        >>> hook = create_async_mock_lifecycle_hook()
        >>> manager = AsyncEngineLifecycleManager(engine, hooks=[hook])
        >>> await manager.start()
        >>> assert hook.start_count == 1
    """
    return AsyncMockLifecycleHook()
