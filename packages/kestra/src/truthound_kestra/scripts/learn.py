"""Learn schema script for Kestra tasks.

This module provides the main entry point for schema/rule learning
operations in Kestra workflows.

Example Kestra Flow:
    ```yaml
    id: schema_learning
    namespace: production
    tasks:
      - id: learn_schema
        type: io.kestra.plugin.scripts.python.Script
        script: |
          from truthound_kestra.scripts.learn import learn_schema_script
          learn_schema_script(
              input_uri="{{ inputs.baseline_data_uri }}",
              min_confidence=0.9
          )
    ```

Python Usage:
    >>> from truthound_kestra.scripts.learn import (
    ...     learn_schema_script,
    ...     LearnScriptExecutor,
    ... )
    >>>
    >>> result = learn_schema_script(data=baseline_df)
    >>> rules = [r.to_rule_dict() for r in result["rules"]]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from truthound_kestra.scripts.base import (
    DataQualityEngineProtocol,
    LearnScriptConfig,
    get_engine,
)
from truthound_kestra.utils.exceptions import ScriptError
from truthound_kestra.utils.helpers import (
    Timer,
    create_kestra_output,
    get_execution_context,
    get_logger,
    kestra_outputs,
    load_data,
)
from truthound_kestra.utils.types import (
    CheckStatus,
    ExecutionContext,
    LearnedRule,
    OperationType,
    ScriptOutput,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "learn_schema_script",
    "LearnScriptExecutor",
    "LearnScriptResult",
]

logger = get_logger(__name__)


@dataclass
class LearnScriptResult:
    """Result of a learn schema script execution.

    Attributes:
        output: The ScriptOutput containing learn results.
        context: Kestra execution context.
        config: Configuration used for learning.
        raw_result: Raw result from the engine.
    """

    output: ScriptOutput
    context: ExecutionContext | None = None
    config: LearnScriptConfig | None = None
    raw_result: Any = None

    @property
    def is_success(self) -> bool:
        """Check if learning succeeded."""
        return self.output.is_success

    @property
    def rules(self) -> tuple[LearnedRule, ...]:
        """Get learned rules."""
        return self.output.rules

    @property
    def rule_count(self) -> int:
        """Get number of learned rules."""
        return len(self.output.rules)

    @property
    def execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        return self.output.execution_time_ms

    def get_rules_for_column(self, column: str) -> list[LearnedRule]:
        """Get rules for a specific column."""
        return [r for r in self.rules if r.column == column]

    def get_rules_by_type(self, rule_type: str) -> list[LearnedRule]:
        """Get rules of a specific type."""
        return [r for r in self.rules if r.rule_type == rule_type]

    def to_rule_dicts(self) -> list[dict[str, Any]]:
        """Convert learned rules to rule dictionaries for engine consumption."""
        return [r.to_rule_dict() for r in self.rules]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output": self.output.to_dict(),
            "context": self.context.to_dict() if self.context else None,
            "config": self.config.to_dict() if self.config else None,
        }

    def to_kestra_output(self) -> dict[str, Any]:
        """Convert to Kestra-compatible output."""
        output = create_kestra_output(self.output)
        # Add rules as separate output for easy access
        output["learned_rules"] = self.to_rule_dicts()
        return output


@dataclass
class LearnScriptExecutor:
    """Executor for learn schema scripts.

    Attributes:
        config: Learn script configuration.
        engine: Data quality engine instance.
    """

    config: LearnScriptConfig = field(default_factory=LearnScriptConfig)
    engine: DataQualityEngineProtocol | None = None

    def __post_init__(self) -> None:
        """Initialize engine if not provided."""
        if self.engine is None:
            self.engine = get_engine(self.config.engine_name)

    def execute(
        self,
        data: Any,
        context: ExecutionContext | None = None,
        **kwargs: Any,
    ) -> LearnScriptResult:
        """Execute the learn operation.

        Args:
            data: Data to learn from (DataFrame or path/URI).
            context: Optional execution context.
            **kwargs: Additional arguments for the engine.

        Returns:
            LearnScriptResult containing the learned rules.

        Raises:
            ScriptError: If script execution fails.
        """
        if not self.config.enabled:
            return self._create_skipped_result(context)

        # Get execution context
        if context is None:
            try:
                context = get_execution_context()
            except Exception:
                context = ExecutionContext(
                    execution_id="local",
                    flow_id="script",
                    namespace="default",
                )

        # Load data if string
        if isinstance(data, str):
            data = load_data(data)

        # Sample data if configured
        if self.config.sample_size > 0 and hasattr(data, "head"):
            data = data.head(self.config.sample_size)

        # Execute learn with timing
        with Timer("learn") as timer:
            try:
                raw_result = self._execute_learn(data, **kwargs)
            except Exception as e:
                return self._handle_engine_error(e, context, timer.elapsed_ms)

        # Convert to ScriptOutput
        output = self._convert_result(
            raw_result,
            timer.elapsed_ms,
            len(data) if hasattr(data, "__len__") else 0,
        )

        result = LearnScriptResult(
            output=output,
            context=context,
            config=self.config,
            raw_result=raw_result,
        )

        logger.info(
            f"Learn completed: {result.rule_count} rules "
            f"({result.execution_time_ms:.2f}ms)"
        )

        return result

    def _execute_learn(self, data: Any, **kwargs: Any) -> Any:
        """Execute the engine learn operation."""
        assert self.engine is not None

        # Prepare engine kwargs
        engine_kwargs: dict[str, Any] = {}

        # Truthound-specific options
        if hasattr(self.engine, "engine_name") and self.engine.engine_name == "truthound":
            if self.config.include_patterns:
                engine_kwargs["infer_patterns"] = True
            if self.config.categorical_threshold:
                engine_kwargs["categorical_threshold"] = self.config.categorical_threshold

        engine_kwargs.update(kwargs)

        return self.engine.learn(data, **engine_kwargs)

    def _convert_result(
        self,
        raw_result: Any,
        elapsed_ms: float,
        total_rows: int,
    ) -> ScriptOutput:
        """Convert engine result to ScriptOutput."""
        # Extract rules
        rules = self._extract_rules(raw_result)

        # Filter by confidence
        rules = [r for r in rules if r.confidence >= self.config.min_confidence]

        return ScriptOutput(
            status=CheckStatus.PASSED,
            operation=OperationType.LEARN,
            total_rows=total_rows,
            execution_time_ms=elapsed_ms,
            rules=tuple(rules),
            metadata={
                "engine": self.config.engine_name,
                "min_confidence": self.config.min_confidence,
                "rules_before_filter": len(self._extract_rules(raw_result)),
                "rules_after_filter": len(rules),
            },
            timestamp=datetime.now(timezone.utc),
        )

    def _extract_rules(self, raw_result: Any) -> list[LearnedRule]:
        """Extract LearnedRule objects from engine result."""
        rules = []
        raw_rules = getattr(raw_result, "rules", [])

        for r in raw_rules:
            if isinstance(r, LearnedRule):
                rules.append(r)
            elif isinstance(r, dict):
                rules.append(LearnedRule.from_dict(r))
            elif hasattr(r, "to_dict"):
                rules.append(LearnedRule.from_dict(r.to_dict()))
            else:
                # Best effort extraction
                rules.append(
                    LearnedRule(
                        rule_type=getattr(r, "rule_type", "unknown"),
                        column=getattr(r, "column", "unknown"),
                        parameters=getattr(r, "parameters", {}),
                        confidence=getattr(r, "confidence", 1.0),
                    )
                )

        return rules

    def _create_skipped_result(
        self,
        context: ExecutionContext | None,
    ) -> LearnScriptResult:
        """Create a skipped result when script is disabled."""
        output = ScriptOutput(
            status=CheckStatus.SKIPPED,
            operation=OperationType.LEARN,
            metadata={"reason": "Script disabled"},
        )
        return LearnScriptResult(
            output=output,
            context=context,
            config=self.config,
        )

    def _handle_engine_error(
        self,
        error: Exception,
        context: ExecutionContext | None,
        elapsed_ms: float,
    ) -> LearnScriptResult:
        """Handle engine errors and return error result."""
        logger.error(f"Engine error: {error}")

        output = ScriptOutput(
            status=CheckStatus.ERROR,
            operation=OperationType.LEARN,
            execution_time_ms=elapsed_ms,
            metadata={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        return LearnScriptResult(
            output=output,
            context=context,
            config=self.config,
        )


def learn_schema_script(
    data: Any | None = None,
    input_uri: str | None = None,
    engine_name: str = "truthound",
    min_confidence: float = 0.8,
    include_patterns: bool = True,
    include_ranges: bool = True,
    include_categories: bool = True,
    categorical_threshold: int = 50,
    sample_size: int = 0,
    timeout_seconds: float = 300.0,
    output_to_kestra: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Main entry point for Kestra learn schema tasks.

    Args:
        data: Data to learn from (DataFrame).
        input_uri: URI to load data from.
        engine_name: Name of the engine to use.
        min_confidence: Minimum confidence for learned rules.
        include_patterns: Whether to learn regex patterns.
        include_ranges: Whether to learn value ranges.
        include_categories: Whether to learn categorical values.
        categorical_threshold: Max unique values for categorical.
        sample_size: Maximum rows to sample (0 = all).
        timeout_seconds: Maximum execution time.
        output_to_kestra: Whether to send outputs to Kestra.
        **kwargs: Additional arguments for the engine.

    Returns:
        Dictionary containing learn results.

    Raises:
        ScriptError: If data loading fails.
    """
    if data is None and input_uri is None:
        raise ScriptError(
            message="Either 'data' or 'input_uri' must be provided",
            script_name="learn_schema_script",
        )

    if data is None and input_uri is not None:
        data = load_data(input_uri)

    config = LearnScriptConfig(
        engine_name=engine_name,
        min_confidence=min_confidence,
        include_patterns=include_patterns,
        include_ranges=include_ranges,
        include_categories=include_categories,
        categorical_threshold=categorical_threshold,
        sample_size=sample_size,
        timeout_seconds=timeout_seconds,
    )

    executor = LearnScriptExecutor(config=config)
    result = executor.execute(data, **kwargs)

    output = result.to_kestra_output()

    if output_to_kestra:
        kestra_outputs(output)

    return output
