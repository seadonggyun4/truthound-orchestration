"""Base operator abstractions for data quality operations.

This module provides abstract base classes and protocols for data quality
operators in Apache Airflow. The design follows these principles:

1. Engine-Agnostic: Works with any DataQualityEngine implementation
2. Protocol-First: Uses structural typing for maximum flexibility
3. Immutable Configs: Thread-safe frozen dataclasses for configuration
4. Extensible: Easy to extend with custom operators and behaviors

Architecture:
    BaseDataQualityOperator (ABC)
        ├── DataQualityCheckOperator
        ├── DataQualityProfileOperator
        └── DataQualityLearnOperator

Example:
    >>> class CustomCheckOperator(BaseDataQualityOperator):
    ...     def _execute_operation(self, data, context):
    ...         return self.engine.check(data, self.rules)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from airflow.models import BaseOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.base import CheckResult, LearnResult, ProfileResult
    from common.engines.base import DataQualityEngine


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    """Base configuration for data quality operators.

    This immutable configuration holds common settings shared across
    all data quality operators.

    Attributes:
        connection_id: Airflow Connection ID for data source.
        fail_on_error: Whether to raise exception on validation failure.
        timeout_seconds: Operation timeout in seconds.
        xcom_push_key: Key for pushing results to XCom.
        tags: Metadata tags for the operation.
        extra: Additional operator-specific options.

    Example:
        >>> config = OperatorConfig(
        ...     connection_id="my_db",
        ...     fail_on_error=True,
        ...     timeout_seconds=300,
        ... )
    """

    connection_id: str = "truthound_default"
    fail_on_error: bool = True
    timeout_seconds: int = 300
    xcom_push_key: str = "data_quality_result"
    tags: frozenset[str] = field(default_factory=frozenset)
    extra: dict[str, Any] = field(default_factory=dict)

    def with_connection_id(self, connection_id: str) -> OperatorConfig:
        """Return new config with updated connection_id."""
        return OperatorConfig(
            connection_id=connection_id,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            xcom_push_key=self.xcom_push_key,
            tags=self.tags,
            extra=self.extra,
        )

    def with_fail_on_error(self, fail_on_error: bool) -> OperatorConfig:
        """Return new config with updated fail_on_error."""
        return OperatorConfig(
            connection_id=self.connection_id,
            fail_on_error=fail_on_error,
            timeout_seconds=self.timeout_seconds,
            xcom_push_key=self.xcom_push_key,
            tags=self.tags,
            extra=self.extra,
        )

    def with_timeout(self, timeout_seconds: int) -> OperatorConfig:
        """Return new config with updated timeout."""
        return OperatorConfig(
            connection_id=self.connection_id,
            fail_on_error=self.fail_on_error,
            timeout_seconds=timeout_seconds,
            xcom_push_key=self.xcom_push_key,
            tags=self.tags,
            extra=self.extra,
        )


@dataclass(frozen=True, slots=True)
class CheckOperatorConfig(OperatorConfig):
    """Configuration specific to check operations.

    Attributes:
        rules: Validation rules to apply.
        warning_threshold: Failure rate threshold for warning (0.0-1.0).
        sample_size: Number of rows to sample (None=all).
        parallel: Whether to run checks in parallel.

    Example:
        >>> config = CheckOperatorConfig(
        ...     rules=(
        ...         {"column": "id", "type": "not_null"},
        ...         {"column": "email", "type": "regex", "pattern": r".*@.*"},
        ...     ),
        ...     warning_threshold=0.05,
        ... )
    """

    rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    warning_threshold: float | None = None
    sample_size: int | None = None
    parallel: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.warning_threshold is not None:
            if not 0 <= self.warning_threshold <= 1:
                msg = "warning_threshold must be between 0 and 1"
                raise ValueError(msg)
        if self.sample_size is not None and self.sample_size <= 0:
            msg = "sample_size must be positive"
            raise ValueError(msg)

    def with_rules(self, rules: Sequence[dict[str, Any]]) -> CheckOperatorConfig:
        """Return new config with updated rules."""
        return CheckOperatorConfig(
            connection_id=self.connection_id,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            xcom_push_key=self.xcom_push_key,
            tags=self.tags,
            extra=self.extra,
            rules=tuple(rules),
            warning_threshold=self.warning_threshold,
            sample_size=self.sample_size,
            parallel=self.parallel,
        )


@dataclass(frozen=True, slots=True)
class ProfileOperatorConfig(OperatorConfig):
    """Configuration specific to profile operations.

    Attributes:
        columns: Columns to profile (None=all).
        include_statistics: Whether to include statistical analysis.
        include_patterns: Whether to detect data patterns.
        include_distributions: Whether to analyze distributions.
        sample_size: Number of rows to sample.

    Example:
        >>> config = ProfileOperatorConfig(
        ...     columns=frozenset(["amount", "quantity"]),
        ...     include_statistics=True,
        ...     include_distributions=True,
        ... )
    """

    columns: frozenset[str] | None = None
    include_statistics: bool = True
    include_patterns: bool = True
    include_distributions: bool = True
    sample_size: int | None = None


@dataclass(frozen=True, slots=True)
class LearnOperatorConfig(OperatorConfig):
    """Configuration specific to learn operations.

    Attributes:
        output_path: Path to save learned schema.
        strictness: Learning strictness level.
        infer_constraints: Whether to infer value constraints.
        categorical_threshold: Max unique values for categorical detection.

    Example:
        >>> config = LearnOperatorConfig(
        ...     output_path="s3://bucket/schemas/users.json",
        ...     strictness="moderate",
        ... )
    """

    output_path: str | None = None
    strictness: str = "moderate"
    infer_constraints: bool = True
    categorical_threshold: int = 20


# =============================================================================
# Base Operator
# =============================================================================


class BaseDataQualityOperator(BaseOperator, ABC):
    """Abstract base class for all data quality operators.

    This class provides the common infrastructure for data quality operations
    in Airflow, including:

    - Engine management (supports any DataQualityEngine)
    - Data source handling (file path or SQL query)
    - Hook integration for data loading
    - Result serialization and XCom pushing
    - Failure handling with configurable thresholds

    Subclasses must implement:
        - _execute_operation: The core operation logic
        - _serialize_result: Result serialization for XCom

    Attributes:
        template_fields: Fields that support Jinja templating.
        template_ext: File extensions for template files.
        ui_color: Color shown in Airflow UI.
        ui_fgcolor: Foreground color in Airflow UI.

    Example:
        >>> class CustomOperator(BaseDataQualityOperator):
        ...     def _execute_operation(self, data, context):
        ...         return self.engine.check(data, self.config.rules)
        ...
        ...     def _serialize_result(self, result):
        ...         return result.to_dict()
    """

    # Template fields for Jinja rendering
    template_fields: Sequence[str] = (
        "data_path",
        "sql",
        "connection_id",
    )
    template_ext: Sequence[str] = (".sql", ".json", ".yaml")

    # UI customization
    ui_color: str = "#4A90D9"
    ui_fgcolor: str = "#FFFFFF"

    def __init__(
        self,
        *,
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        fail_on_error: bool = True,
        timeout_seconds: int = 300,
        xcom_push_key: str = "data_quality_result",
        **kwargs: Any,
    ) -> None:
        """Initialize base data quality operator.

        Args:
            data_path: Path to data file (S3, GCS, local, etc.).
            sql: SQL query to fetch data. Mutually exclusive with data_path.
            connection_id: Airflow Connection ID for data source.
            engine: DataQualityEngine instance to use. If None, uses registry.
            engine_name: Name of engine to get from registry (if engine is None).
            fail_on_error: Whether to raise exception on failure.
            timeout_seconds: Operation timeout in seconds.
            xcom_push_key: Key for XCom result push.
            **kwargs: Additional BaseOperator arguments.

        Raises:
            ValueError: If both data_path and sql are specified.
            ValueError: If neither data_path nor sql is specified.
        """
        super().__init__(**kwargs)

        # Validate data source
        if data_path and sql:
            msg = "Cannot specify both data_path and sql"
            raise ValueError(msg)
        if not data_path and not sql:
            msg = "Must specify either data_path or sql"
            raise ValueError(msg)

        self.data_path = data_path
        self.sql = sql
        self.connection_id = connection_id
        self.fail_on_error = fail_on_error
        self.timeout_seconds = timeout_seconds
        self.xcom_push_key = xcom_push_key

        # Engine configuration
        self._engine = engine
        self._engine_name = engine_name

    @property
    def engine(self) -> DataQualityEngine:
        """Get the data quality engine instance.

        Returns:
            DataQualityEngine: The configured engine instance.

        Note:
            Lazily initializes the engine on first access.
            Uses the provided engine, or gets one from the registry.
        """
        if self._engine is None:
            from common.engines import get_engine

            self._engine = get_engine(self._engine_name)
        return self._engine

    def execute(self, context: Context) -> dict[str, Any]:
        """Execute the data quality operation.

        This is the main entry point called by Airflow. It:
        1. Loads data from the configured source
        2. Executes the subclass-specific operation
        3. Serializes and pushes results to XCom
        4. Handles failures based on configuration

        Args:
            context: Airflow execution context.

        Returns:
            dict[str, Any]: Serialized operation result.

        Raises:
            AirflowException: If operation fails and fail_on_error is True.
        """
        from truthound_airflow.hooks.base import DataQualityHook

        self.log.info(f"Starting data quality operation with engine: {self.engine.engine_name}")

        # Initialize hook and load data
        hook = DataQualityHook(connection_id=self.connection_id)

        if self.data_path:
            self.log.info(f"Loading data from path: {self.data_path}")
            data = hook.load_data(self.data_path)
        else:
            self.log.info("Executing SQL query")
            data = hook.query(self.sql)

        self.log.info(f"Loaded {len(data) if hasattr(data, '__len__') else 'unknown'} rows")

        # Execute subclass operation
        result = self._execute_operation(data, context)

        # Serialize and push to XCom
        result_dict = self._serialize_result(result)
        result_dict["_metadata"] = {
            "engine": self.engine.engine_name,
            "engine_version": self.engine.engine_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": self.task_id,
            "dag_id": context.get("dag", {}).dag_id if context.get("dag") else None,
        }

        context["ti"].xcom_push(key=self.xcom_push_key, value=result_dict)
        self.log.info(f"Pushed result to XCom with key: {self.xcom_push_key}")

        # Log metrics
        self._log_metrics(result_dict)

        # Handle failures
        self._handle_result(result, result_dict, context)

        return result_dict

    @abstractmethod
    def _execute_operation(
        self,
        data: Any,
        context: Context,
    ) -> CheckResult | ProfileResult | LearnResult:
        """Execute the specific data quality operation.

        Subclasses must implement this method to define their
        specific operation logic.

        Args:
            data: The loaded data (typically Polars DataFrame).
            context: Airflow execution context.

        Returns:
            The operation result (CheckResult, ProfileResult, or LearnResult).
        """
        ...

    @abstractmethod
    def _serialize_result(
        self,
        result: CheckResult | ProfileResult | LearnResult,
    ) -> dict[str, Any]:
        """Serialize operation result for XCom.

        Subclasses must implement this to convert their result
        type to a dictionary suitable for XCom serialization.

        Args:
            result: The operation result.

        Returns:
            dict[str, Any]: Serialized result dictionary.
        """
        ...

    def _handle_result(
        self,
        result: Any,
        result_dict: dict[str, Any],
        context: Context,
    ) -> None:
        """Handle operation result, potentially raising on failure.

        Override this method to customize failure handling behavior.

        Args:
            result: The raw operation result.
            result_dict: The serialized result dictionary.
            context: Airflow execution context.

        Raises:
            AirflowException: If result indicates failure and fail_on_error is True.
        """
        # Default implementation does nothing
        # Subclasses override for specific handling
        pass

    def _log_metrics(self, result_dict: dict[str, Any]) -> None:
        """Log operation metrics.

        Override this method to customize metric logging.

        Args:
            result_dict: The serialized result dictionary.
        """
        self.log.info(
            f"Operation completed - "
            f"engine={result_dict.get('_metadata', {}).get('engine', 'unknown')}"
        )


# =============================================================================
# Result Handler Protocol
# =============================================================================


class ResultHandler:
    """Handler for processing and routing operation results.

    This class provides utilities for handling operation results,
    including serialization, XCom integration, and callback invocation.

    Attributes:
        on_success: Callback for successful operations.
        on_failure: Callback for failed operations.
        on_warning: Callback for operations with warnings.

    Example:
        >>> handler = ResultHandler(
        ...     on_success=lambda r: notify_slack("Success!"),
        ...     on_failure=lambda r: notify_pagerduty("Failure!"),
        ... )
        >>> handler.handle(result, context)
    """

    def __init__(
        self,
        *,
        on_success: Any | None = None,
        on_failure: Any | None = None,
        on_warning: Any | None = None,
    ) -> None:
        """Initialize result handler with callbacks.

        Args:
            on_success: Callable invoked on success.
            on_failure: Callable invoked on failure.
            on_warning: Callable invoked on warning.
        """
        self.on_success = on_success
        self.on_failure = on_failure
        self.on_warning = on_warning

    def handle(
        self,
        result: CheckResult,
        context: Context,
    ) -> None:
        """Handle a check result, invoking appropriate callbacks.

        Args:
            result: The check result to handle.
            context: Airflow execution context.
        """
        from common.base import CheckStatus

        if result.status == CheckStatus.PASSED:
            if self.on_success:
                self.on_success(result, context)
        elif result.status == CheckStatus.WARNING:
            if self.on_warning:
                self.on_warning(result, context)
        else:  # FAILED or ERROR
            if self.on_failure:
                self.on_failure(result, context)
