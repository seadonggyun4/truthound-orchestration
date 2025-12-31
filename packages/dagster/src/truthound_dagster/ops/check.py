"""Data Quality Check Ops for Dagster.

This module provides Dagster ops for executing data quality validation
checks. The ops integrate with the DataQualityResource for engine access.

Example:
    >>> from dagster import job
    >>> from truthound_dagster.ops import data_quality_check_op, create_check_op
    >>> from truthound_dagster.resources import DataQualityResource
    >>>
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def quality_job():
    ...     data_quality_check_op()
    >>>
    >>> # Or create customized op
    >>> users_check = create_check_op(
    ...     name="users_check",
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "unique"},
    ...     ],
    ... )
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Dict, List

from dagster import Config, In, OpExecutionContext, Out, op

from truthound_dagster.ops.base import CheckOpConfig

if TYPE_CHECKING:
    from common.base import CheckResult

    from truthound_dagster.resources import DataQualityResource


class CheckOpDagsterConfig(Config):
    """Dagster configuration schema for check ops.

    This class defines the configuration fields that can be set
    in Dagster's configuration system (YAML, Python, etc.).

    Note: Rules are passed as a JSON string due to Dagster's config
    limitations with nested dict structures. Use json.dumps() to
    convert rules list to string format.
    """

    rules_json: str = ""  # JSON string of rules list
    fail_on_error: bool = True
    warning_threshold: float = -1.0  # -1 means not set
    sample_size: int = -1  # -1 means not set
    auto_schema: bool = False
    timeout_seconds: float = 300.0

    def get_rules(self) -> List[Dict[str, Any]]:
        """Parse rules from JSON string."""
        if not self.rules_json:
            return []
        import json

        return json.loads(self.rules_json)

    def get_warning_threshold(self) -> Optional[float]:
        """Get warning threshold, None if not set."""
        return None if self.warning_threshold < 0 else self.warning_threshold

    def get_sample_size(self) -> Optional[int]:
        """Get sample size, None if not set."""
        return None if self.sample_size < 0 else self.sample_size


def _sample_data(data: Any, sample_size: int) -> Any:
    """Sample data if it exceeds sample size.

    Args:
        data: Input data.
        sample_size: Maximum rows to include.

    Returns:
        Sampled data or original if smaller.
    """
    if not hasattr(data, "__len__"):
        return data

    if len(data) <= sample_size:
        return data

    if hasattr(data, "sample"):
        # Polars/Pandas DataFrame
        return data.sample(n=sample_size)
    elif hasattr(data, "head"):
        # Fallback to head
        return data.head(sample_size)

    return data


def _serialize_check_result(result: "CheckResult") -> Dict[str, Any]:
    """Serialize CheckResult for Dagster metadata.

    Args:
        result: Check result to serialize.

    Returns:
        Dict[str, Any]: Serialized result.
    """
    return {
        "status": result.status.value,
        "is_success": result.is_success,
        "passed_count": result.passed_count,
        "failed_count": result.failed_count,
        "warning_count": result.warning_count,
        "failure_rate": result.failure_rate,
        "failures": [
            {
                "rule_name": f.rule_name,
                "column": f.column,
                "message": f.message,
                "severity": f.severity.value,
                "failed_count": f.failed_count,
                "total_count": f.total_count,
            }
            for f in result.failures
        ],
        "execution_time_ms": result.execution_time_ms,
        "timestamp": result.timestamp.isoformat(),
    }


@op(
    name="data_quality_check",
    description="Execute data quality validation check using configured engine.",
    ins={"data": In(description="Data to validate (DataFrame, path, etc.)")},
    out=Out(description="Check result with validation status and failures"),
    tags={"kind": "data_quality", "operation": "check"},
    required_resource_keys={"data_quality"},
)
def data_quality_check_op(
    context: OpExecutionContext,
    data: Any,
    config: CheckOpDagsterConfig,
) -> Dict[str, Any]:
    """Execute data quality validation check.

    This op validates data against a set of rules using the configured
    DataQualityResource. It logs the results and returns a serialized
    result dictionary.

    Parameters
    ----------
    context : OpExecutionContext
        Dagster execution context.

    data : Any
        Data to validate. Can be a DataFrame, file path, or other
        data source supported by the engine.

    config : CheckOpDagsterConfig
        Op configuration with rules and settings.

    Returns
    -------
    Dict[str, Any]
        Serialized check result containing:
        - status: Check status (passed, failed, warning, etc.)
        - is_success: Whether check passed
        - passed_count: Number of passed rules
        - failed_count: Number of failed rules
        - failures: List of failure details
        - execution_time_ms: Execution time

    Raises
    ------
    DataQualityError
        If validation fails and fail_on_error is True.

    Examples
    --------
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def validate_job():
    ...     data = load_data_op()
    ...     result = data_quality_check_op(data)
    ...     return result
    """
    dq_resource: DataQualityResource = context.resources.data_quality

    # Get rules from config
    rules = config.get_rules()

    context.log.info(f"Starting data quality check with {len(rules)} rules")

    # Sample data if configured
    actual_data = data
    sample_size = config.get_sample_size()
    if sample_size is not None:
        actual_data = _sample_data(data, sample_size)
        context.log.info(f"Sampled data to {sample_size} rows")

    # Execute check
    result = dq_resource.check(
        data=actual_data,
        rules=rules,
        auto_schema=config.auto_schema,
        fail_on_error=config.fail_on_error,
        timeout=config.timeout_seconds,
    )

    # Log results
    if result.is_success:
        context.log.info(
            f"Quality check PASSED: {result.passed_count} rules passed "
            f"in {result.execution_time_ms:.2f}ms"
        )
    else:
        context.log.warning(
            f"Quality check FAILED: {result.failed_count} rules failed "
            f"({result.failure_rate:.2%} failure rate)"
        )
        for failure in result.failures[:5]:
            context.log.warning(
                f"  - {failure.rule_name}[{failure.column}]: {failure.message}"
            )
        if len(result.failures) > 5:
            context.log.warning(f"  ... and {len(result.failures) - 5} more failures")

    # Serialize and return
    result_dict = _serialize_check_result(result)

    # Add metadata to context
    context.add_output_metadata(
        {
            "status": result.status.value,
            "passed_count": result.passed_count,
            "failed_count": result.failed_count,
            "execution_time_ms": result.execution_time_ms,
        }
    )

    return result_dict


def create_check_op(
    name: str,
    rules: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    fail_on_error: bool = True,
    warning_threshold: float | None = None,
    sample_size: int | None = None,
    auto_schema: bool = False,
    timeout_seconds: float = 300.0,
    description: str | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Create a customized data quality check op.

    This factory function creates a new op with preset configuration.
    Use this when you need the same check configuration in multiple places.

    Parameters
    ----------
    name : str
        Name for the op.

    rules : Optional[Sequence[Dict[str, Any]]]
        Validation rules to apply.

    fail_on_error : bool
        Whether to raise on validation failure.

    warning_threshold : float | None
        Failure rate threshold for warning.

    sample_size : int | None
        Number of rows to sample.

    auto_schema : bool
        Auto-generate schema from data.

    timeout_seconds : float
        Operation timeout.

    description : str | None
        Op description.

    tags : Optional[Dict[str, str]]
        Additional op tags.

    Returns
    -------
    Callable
        Configured Dagster op.

    Examples
    --------
    >>> users_check = create_check_op(
    ...     name="users_check",
    ...     rules=[
    ...         {"column": "user_id", "type": "not_null"},
    ...         {"column": "email", "type": "unique"},
    ...     ],
    ...     fail_on_error=True,
    ... )
    >>>
    >>> @job
    ... def pipeline():
    ...     data = load_users()
    ...     users_check(data)
    """
    # Build configuration
    config = CheckOpConfig(
        rules=tuple(rules or []),
        fail_on_error=fail_on_error,
        warning_threshold=warning_threshold,
        sample_size=sample_size,
        auto_schema=auto_schema,
        timeout_seconds=timeout_seconds,
    )

    # Build tags
    op_tags = {"kind": "data_quality", "operation": "check"}
    if tags:
        op_tags.update(tags)

    # Build description
    op_description = description or f"Data quality check: {name}"

    @op(
        name=name,
        description=op_description,
        ins={"data": In(description="Data to validate")},
        out=Out(description="Check result"),
        tags=op_tags,
        required_resource_keys={"data_quality"},
    )
    def check_op_impl(context: OpExecutionContext, data: Any) -> Dict[str, Any]:
        """Execute configured data quality check."""
        dq_resource: DataQualityResource = context.resources.data_quality

        context.log.info(f"Starting {name} with {len(config.rules)} rules")

        # Sample data if configured
        actual_data = data
        if config.sample_size is not None:
            actual_data = _sample_data(data, config.sample_size)

        # Execute check
        result = dq_resource.check(
            data=actual_data,
            rules=list(config.rules),
            auto_schema=config.auto_schema,
            fail_on_error=config.fail_on_error,
            timeout=config.timeout_seconds,
        )

        # Log results
        if result.is_success:
            context.log.info(
                f"{name} PASSED: {result.passed_count} rules in "
                f"{result.execution_time_ms:.2f}ms"
            )
        else:
            context.log.warning(
                f"{name} FAILED: {result.failed_count} failures "
                f"({result.failure_rate:.2%})"
            )

        result_dict = _serialize_check_result(result)

        context.add_output_metadata(
            {
                "status": result.status.value,
                "passed_count": result.passed_count,
                "failed_count": result.failed_count,
            }
        )

        return result_dict

    return check_op_impl
