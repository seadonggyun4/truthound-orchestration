"""Check quality script for Kestra tasks.

This module provides the main entry point for data quality check
operations in Kestra workflows.

Example Kestra Flow:
    ```yaml
    id: data_quality_check
    namespace: production
    tasks:
      - id: check_users
        type: io.kestra.plugin.scripts.python.Script
        script: |
          from truthound_kestra.scripts.check import check_quality_script
          check_quality_script(
              input_uri="{{ inputs.data_uri }}",
              rules=[
                  {"type": "not_null", "column": "id"},
                  {"type": "unique", "column": "email"},
              ],
              fail_on_error=True
          )
    ```

Python Usage:
    >>> from truthound_kestra.scripts.check import (
    ...     check_quality_script,
    ...     CheckScriptExecutor,
    ... )
    >>>
    >>> # Simple function call
    >>> result = check_quality_script(
    ...     data=df,
    ...     rules=[{"type": "not_null", "column": "id"}],
    ... )
    >>>
    >>> # Or use executor class for more control
    >>> executor = CheckScriptExecutor(config=CheckScriptConfig(...))
    >>> result = executor.execute(df)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from truthound_kestra.scripts.base import (
    CheckScriptConfig,
    DataQualityEngineProtocol,
    get_engine,
)
from truthound_kestra.utils.exceptions import (
    DataQualityError,
    EngineError,
    ScriptError,
)
from truthound_kestra.utils.helpers import (
    Timer,
    create_kestra_output,
    get_execution_context,
    get_logger,
    kestra_outputs,
    load_data,
    log_operation,
    validate_rules,
)
from truthound_kestra.utils.serialization import serialize_result
from truthound_kestra.utils.types import (
    CheckStatus,
    ExecutionContext,
    OperationType,
    RuleDict,
    ScriptOutput,
    Severity,
    ValidationFailure,
)

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "check_quality_script",
    "CheckScriptExecutor",
    "CheckScriptResult",
]

logger = get_logger(__name__)


@dataclass
class CheckScriptResult:
    """Result of a check quality script execution.

    This class provides a structured container for check results
    with convenient accessors and serialization.

    Attributes:
        output: The ScriptOutput containing check results.
        context: Kestra execution context.
        config: Configuration used for the check.
        raw_result: Raw result from the engine (if available).

    Example:
        >>> result = CheckScriptResult(output=script_output)
        >>> if result.is_success:
        ...     print(f"Check passed: {result.pass_rate:.0%}")
        >>> else:
        ...     print(f"Check failed: {result.failed_count} failures")
    """

    output: ScriptOutput
    context: ExecutionContext | None = None
    config: CheckScriptConfig | None = None
    raw_result: Any = None

    @property
    def is_success(self) -> bool:
        """Check if the validation passed."""
        return self.output.is_success

    @property
    def status(self) -> CheckStatus:
        """Get the check status."""
        return self.output.status

    @property
    def passed_count(self) -> int:
        """Get number of passed validations."""
        return self.output.passed_count

    @property
    def failed_count(self) -> int:
        """Get number of failed validations."""
        return self.output.failed_count

    @property
    def pass_rate(self) -> float:
        """Get the pass rate (0.0 to 1.0)."""
        return self.output.pass_rate

    @property
    def failures(self) -> tuple[ValidationFailure, ...]:
        """Get validation failures."""
        return self.output.failures

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.output.execution_time_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output": self.output.to_dict(),
            "context": self.context.to_dict() if self.context else None,
            "config": self.config.to_dict() if self.config else None,
        }

    def to_kestra_output(self) -> dict[str, Any]:
        """Convert to Kestra-compatible output."""
        return create_kestra_output(self.output)


@dataclass
class CheckScriptExecutor:
    """Executor for check quality scripts.

    This class provides a reusable executor for data quality checks
    that can be configured and invoked multiple times.

    Attributes:
        config: Check script configuration.
        engine: Data quality engine instance.

    Example:
        >>> config = CheckScriptConfig(
        ...     rules=({"type": "not_null", "column": "id"},),
        ...     fail_on_error=True
        ... )
        >>> executor = CheckScriptExecutor(config=config)
        >>> result = executor.execute(df)
    """

    config: CheckScriptConfig = field(default_factory=CheckScriptConfig)
    engine: DataQualityEngineProtocol | None = None

    def __post_init__(self) -> None:
        """Initialize engine if not provided."""
        if self.engine is None:
            self.engine = get_engine(self.config.engine_name)

    def execute(
        self,
        data: Any,
        rules: list[RuleDict] | None = None,
        context: ExecutionContext | None = None,
        **kwargs: Any,
    ) -> CheckScriptResult:
        """Execute the check operation.

        Args:
            data: Data to validate (DataFrame or path/URI).
            rules: Optional rules to override config rules.
            context: Optional execution context.
            **kwargs: Additional arguments for the engine.

        Returns:
            CheckScriptResult containing the check results.

        Raises:
            ScriptError: If script execution fails.
            DataQualityError: If check fails and fail_on_error is True.
        """
        if not self.config.enabled:
            return self._create_skipped_result(context)

        # Get execution context if not provided
        if context is None:
            try:
                context = get_execution_context()
            except Exception:
                context = ExecutionContext(
                    execution_id="local",
                    flow_id="script",
                    namespace="default",
                )

        # Merge rules from config and parameters
        effective_rules = list(rules) if rules else list(self.config.rules)

        # Validate rules
        rule_errors = validate_rules(effective_rules)
        if rule_errors:
            logger.warning(f"Rule validation warnings: {rule_errors}")

        # Load data if string (path or URI)
        if isinstance(data, str):
            with log_operation("load_data", logger, source=data):
                data = load_data(data)

        # Execute check with timing
        with Timer("check") as timer:
            try:
                raw_result = self._execute_check(
                    data, effective_rules, **kwargs
                )
            except Exception as e:
                return self._handle_engine_error(e, context, timer.elapsed_ms)

        # Convert to ScriptOutput
        output = self._convert_result(
            raw_result,
            timer.elapsed_ms,
            len(data) if hasattr(data, "__len__") else 0,
        )

        result = CheckScriptResult(
            output=output,
            context=context,
            config=self.config,
            raw_result=raw_result,
        )

        # Log result
        self._log_result(result)

        # Raise if configured and check failed
        if self.config.fail_on_error and not result.is_success:
            raise DataQualityError(
                message=f"Data quality check failed: {result.failed_count} failures",
                result=result.to_dict(),
                metadata={"context": context.to_dict() if context else None},
            )

        return result

    def _execute_check(
        self,
        data: Any,
        rules: list[RuleDict],
        **kwargs: Any,
    ) -> Any:
        """Execute the engine check operation."""
        assert self.engine is not None

        # Prepare engine kwargs
        engine_kwargs: dict[str, Any] = {}

        # Truthound-specific: auto_schema
        if self.config.auto_schema:
            engine_kwargs["auto_schema"] = True

        # Truthound-specific: parallel processing
        if self.config.parallel:
            engine_kwargs["parallel"] = True
            if self.config.max_workers:
                engine_kwargs["max_workers"] = self.config.max_workers

        # Min severity (Truthound)
        if self.config.min_severity != Severity.LOW:
            engine_kwargs["min_severity"] = self.config.min_severity.value

        # Override with any additional kwargs
        engine_kwargs.update(kwargs)

        # Execute check
        if rules:
            return self.engine.check(data, rules=rules, **engine_kwargs)
        else:
            return self.engine.check(data, **engine_kwargs)

    def _convert_result(
        self,
        raw_result: Any,
        elapsed_ms: float,
        total_rows: int,
    ) -> ScriptOutput:
        """Convert engine result to ScriptOutput."""
        # Extract status
        if hasattr(raw_result, "status"):
            status_val = raw_result.status
            if hasattr(status_val, "name"):
                # Handle Enum with any value type (int or str)
                status = CheckStatus(status_val.name.lower())
            elif isinstance(status_val, str):
                status = CheckStatus(status_val.lower())
            else:
                status = CheckStatus.PASSED if getattr(raw_result, "passed", True) else CheckStatus.FAILED
        else:
            status = CheckStatus.PASSED

        # Extract counts
        passed_count = getattr(raw_result, "passed_count", 0)
        failed_count = getattr(raw_result, "failed_count", 0)
        warning_count = getattr(raw_result, "warning_count", 0)

        # Extract failures
        failures = self._extract_failures(raw_result)

        # Limit failures if configured
        if self.config.sample_failures > 0:
            failures = failures[: self.config.sample_failures]

        return ScriptOutput(
            status=status,
            operation=OperationType.CHECK,
            passed_count=passed_count,
            failed_count=failed_count,
            warning_count=warning_count,
            total_rows=total_rows,
            execution_time_ms=elapsed_ms,
            failures=tuple(failures),
            metadata={
                "engine": self.config.engine_name,
                "rules_count": len(self.config.rules),
            },
            timestamp=datetime.now(timezone.utc),
        )

    def _extract_failures(self, raw_result: Any) -> list[ValidationFailure]:
        """Extract ValidationFailure objects from engine result."""
        failures = []
        raw_failures = getattr(raw_result, "failures", [])

        for f in raw_failures:
            if isinstance(f, ValidationFailure):
                failures.append(f)
            elif isinstance(f, dict):
                failures.append(ValidationFailure.from_dict(f))
            elif hasattr(f, "to_dict"):
                failures.append(ValidationFailure.from_dict(f.to_dict()))
            else:
                # Best effort extraction
                failures.append(
                    ValidationFailure(
                        rule_type=getattr(f, "rule_type", "unknown"),
                        column=getattr(f, "column", "unknown"),
                        message=getattr(f, "message", str(f)),
                        severity=Severity(getattr(f, "severity", "medium")),
                        failed_count=getattr(f, "failed_count", 0),
                    )
                )

        return failures

    def _create_skipped_result(
        self,
        context: ExecutionContext | None,
    ) -> CheckScriptResult:
        """Create a skipped result when script is disabled."""
        output = ScriptOutput(
            status=CheckStatus.SKIPPED,
            operation=OperationType.CHECK,
            metadata={"reason": "Script disabled"},
        )
        return CheckScriptResult(
            output=output,
            context=context,
            config=self.config,
        )

    def _handle_engine_error(
        self,
        error: Exception,
        context: ExecutionContext | None,
        elapsed_ms: float,
    ) -> CheckScriptResult:
        """Handle engine errors and return error result."""
        logger.error(f"Engine error: {error}")

        output = ScriptOutput(
            status=CheckStatus.ERROR,
            operation=OperationType.CHECK,
            execution_time_ms=elapsed_ms,
            metadata={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        if self.config.fail_on_error:
            raise EngineError(
                message=f"Engine check failed: {error}",
                engine_name=self.config.engine_name,
                operation="check",
                original_error=error,
            ) from error

        return CheckScriptResult(
            output=output,
            context=context,
            config=self.config,
        )

    def _log_result(self, result: CheckScriptResult) -> None:
        """Log the check result."""
        if result.is_success:
            logger.info(
                f"Check passed: {result.passed_count} passed, "
                f"{result.failed_count} failed "
                f"({result.execution_time_ms:.2f}ms)"
            )
        else:
            logger.warning(
                f"Check failed: {result.passed_count} passed, "
                f"{result.failed_count} failed "
                f"({result.execution_time_ms:.2f}ms)"
            )


def check_quality_script(
    data: Any | None = None,
    input_uri: str | None = None,
    rules: list[RuleDict] | None = None,
    engine_name: str = "truthound",
    fail_on_error: bool = True,
    auto_schema: bool = False,
    timeout_seconds: float = 300.0,
    output_to_kestra: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main entry point for Kestra check quality tasks.

    This function provides a simple interface for executing data quality
    checks from Kestra Python script tasks.

    Args:
        data: Data to validate (DataFrame).
        input_uri: URI to load data from (alternative to data).
        rules: List of rule dictionaries.
        engine_name: Name of the engine to use.
        fail_on_error: Whether to raise exception on failure.
        auto_schema: Whether to use auto-generated schema.
        timeout_seconds: Maximum execution time.
        output_to_kestra: Whether to send outputs to Kestra.
        **kwargs: Additional arguments for the engine.

    Returns:
        Dictionary containing check results.

    Raises:
        ScriptError: If data loading fails.
        DataQualityError: If check fails and fail_on_error is True.

    Example:
        >>> # In Kestra Python Script task
        >>> result = check_quality_script(
        ...     input_uri="{{ outputs.extract.uri }}",
        ...     rules=[
        ...         {"type": "not_null", "column": "id"},
        ...         {"type": "unique", "column": "email"},
        ...     ],
        ...     fail_on_error=True
        ... )
    """
    # Validate inputs
    if data is None and input_uri is None:
        raise ScriptError(
            message="Either 'data' or 'input_uri' must be provided",
            script_name="check_quality_script",
        )

    # Load data if URI provided
    if data is None and input_uri is not None:
        data = load_data(input_uri)

    # Create configuration
    config = CheckScriptConfig(
        engine_name=engine_name,
        rules=tuple(rules or []),
        fail_on_error=fail_on_error,
        auto_schema=auto_schema,
        timeout_seconds=timeout_seconds,
    )

    # Create executor and execute
    executor = CheckScriptExecutor(config=config)
    result = executor.execute(data, rules=rules, **kwargs)

    # Create output
    output = result.to_kestra_output()

    # Send to Kestra if configured
    if output_to_kestra:
        kestra_outputs(output)

    return output
