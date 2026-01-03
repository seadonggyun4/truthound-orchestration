"""Condition blocks for data quality routing in Mage.

This module provides condition blocks that enable conditional routing
of pipeline execution based on data quality results.

Components:
    - BaseConditionBlock: Abstract base for all conditions
    - DataQualityCondition: Route based on quality check results

Example:
    >>> from truthound_mage import DataQualityCondition, ConditionBlockConfig
    >>>
    >>> config = ConditionBlockConfig(
    ...     pass_threshold=0.95,
    ...     route_on_pass="production_load",
    ...     route_on_fail="quarantine",
    ... )
    >>> condition = DataQualityCondition(config=config)
    >>> route = condition.evaluate(check_result)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Sequence

from truthound_mage.blocks.base import (
    BlockConfig,
    BlockExecutionContext,
)

if TYPE_CHECKING:
    from common.base import CheckResult


# =============================================================================
# Enums
# =============================================================================


class RouteDecision(str, Enum):
    """Routing decision types."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


# =============================================================================
# Configuration Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConditionBlockConfig(BlockConfig):
    """Configuration for condition blocks.

    Attributes:
        pass_threshold: Pass rate threshold for PASS routing (0.0-1.0).
        warning_threshold: Pass rate threshold for WARNING routing (0.0-1.0).
        route_on_pass: Block UUID to route to on PASS.
        route_on_fail: Block UUID to route to on FAIL.
        route_on_warning: Block UUID to route to on WARNING.
        route_on_error: Block UUID to route to on ERROR.
        default_route: Default route when no condition matches.
        evaluate_all: Whether to evaluate all conditions or short-circuit.

    Example:
        >>> config = ConditionBlockConfig(
        ...     pass_threshold=0.95,
        ...     warning_threshold=0.90,
        ...     route_on_pass="load_to_warehouse",
        ...     route_on_fail="quarantine_data",
        ... )
    """

    pass_threshold: float = 0.95
    warning_threshold: float | None = None
    route_on_pass: str | None = None
    route_on_fail: str | None = None
    route_on_warning: str | None = None
    route_on_error: str | None = None
    default_route: str | None = None
    evaluate_all: bool = False
    custom_routes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        BlockConfig.__post_init__(self)

        if not 0 <= self.pass_threshold <= 1:
            msg = "pass_threshold must be between 0 and 1"
            raise ValueError(msg)

        if self.warning_threshold is not None:
            if not 0 <= self.warning_threshold <= 1:
                msg = "warning_threshold must be between 0 and 1"
                raise ValueError(msg)
            if self.warning_threshold >= self.pass_threshold:
                msg = "warning_threshold must be less than pass_threshold"
                raise ValueError(msg)

    def with_thresholds(
        self,
        pass_threshold: float,
        warning_threshold: float | None = None,
    ) -> ConditionBlockConfig:
        """Return new config with updated thresholds."""
        return ConditionBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            pass_threshold=pass_threshold,
            warning_threshold=warning_threshold,
            route_on_pass=self.route_on_pass,
            route_on_fail=self.route_on_fail,
            route_on_warning=self.route_on_warning,
            route_on_error=self.route_on_error,
            default_route=self.default_route,
            evaluate_all=self.evaluate_all,
            custom_routes=self.custom_routes,
        )

    def with_routes(
        self,
        on_pass: str | None = None,
        on_fail: str | None = None,
        on_warning: str | None = None,
        on_error: str | None = None,
        default: str | None = None,
    ) -> ConditionBlockConfig:
        """Return new config with updated routes."""
        return ConditionBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            pass_threshold=self.pass_threshold,
            warning_threshold=self.warning_threshold,
            route_on_pass=on_pass if on_pass is not None else self.route_on_pass,
            route_on_fail=on_fail if on_fail is not None else self.route_on_fail,
            route_on_warning=on_warning if on_warning is not None else self.route_on_warning,
            route_on_error=on_error if on_error is not None else self.route_on_error,
            default_route=default if default is not None else self.default_route,
            evaluate_all=self.evaluate_all,
            custom_routes=self.custom_routes,
        )

    def with_custom_route(self, condition_name: str, route: str) -> ConditionBlockConfig:
        """Return new config with added custom route."""
        new_routes = dict(self.custom_routes)
        new_routes[condition_name] = route
        return ConditionBlockConfig(
            engine_name=self.engine_name,
            fail_on_error=self.fail_on_error,
            timeout_seconds=self.timeout_seconds,
            output_key=self.output_key,
            log_results=self.log_results,
            tags=self.tags,
            extra=self.extra,
            pass_threshold=self.pass_threshold,
            warning_threshold=self.warning_threshold,
            route_on_pass=self.route_on_pass,
            route_on_fail=self.route_on_fail,
            route_on_warning=self.route_on_warning,
            route_on_error=self.route_on_error,
            default_route=self.default_route,
            evaluate_all=self.evaluate_all,
            custom_routes=new_routes,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        base = BlockConfig.to_dict(self)
        base.update({
            "pass_threshold": self.pass_threshold,
            "warning_threshold": self.warning_threshold,
            "route_on_pass": self.route_on_pass,
            "route_on_fail": self.route_on_fail,
            "route_on_warning": self.route_on_warning,
            "route_on_error": self.route_on_error,
            "default_route": self.default_route,
            "evaluate_all": self.evaluate_all,
            "custom_routes": self.custom_routes,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConditionBlockConfig:
        """Create configuration from dictionary."""
        return cls(
            engine_name=data.get("engine_name"),
            fail_on_error=data.get("fail_on_error", True),
            timeout_seconds=data.get("timeout_seconds", 300),
            output_key=data.get("output_key", "data_quality_result"),
            log_results=data.get("log_results", True),
            tags=frozenset(data.get("tags", [])),
            extra=data.get("extra", {}),
            pass_threshold=data.get("pass_threshold", 0.95),
            warning_threshold=data.get("warning_threshold"),
            route_on_pass=data.get("route_on_pass"),
            route_on_fail=data.get("route_on_fail"),
            route_on_warning=data.get("route_on_warning"),
            route_on_error=data.get("route_on_error"),
            default_route=data.get("default_route"),
            evaluate_all=data.get("evaluate_all", False),
            custom_routes=data.get("custom_routes", {}),
        )


# =============================================================================
# Condition Result
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConditionResult:
    """Result of a condition evaluation.

    Attributes:
        decision: The routing decision (PASS, FAIL, WARNING, etc.).
        route: The target block UUID to route to.
        metrics: Metrics that were evaluated.
        reasons: List of reasons for the decision.
        all_evaluations: Results of all condition evaluations.

    Example:
        >>> result = ConditionResult(
        ...     decision=RouteDecision.PASS,
        ...     route="load_warehouse",
        ...     metrics={"pass_rate": 0.98},
        ... )
    """

    decision: RouteDecision
    route: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = field(default_factory=tuple)
    all_evaluations: dict[str, bool] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        """Check if decision is PASS."""
        return self.decision == RouteDecision.PASS

    @property
    def is_fail(self) -> bool:
        """Check if decision is FAIL."""
        return self.decision == RouteDecision.FAIL

    @property
    def is_warning(self) -> bool:
        """Check if decision is WARNING."""
        return self.decision == RouteDecision.WARNING

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "decision": self.decision.value,
            "route": self.route,
            "metrics": self.metrics,
            "reasons": list(self.reasons),
            "all_evaluations": self.all_evaluations,
        }


# =============================================================================
# Base Condition Block
# =============================================================================


class BaseConditionBlock(ABC):
    """Abstract base class for condition blocks.

    Condition blocks evaluate data quality results and determine
    routing decisions for pipeline execution.

    Subclasses must implement:
        - evaluate: Evaluate conditions and return routing decision

    Attributes:
        config: Condition block configuration.

    Example:
        >>> class CustomCondition(BaseConditionBlock):
        ...     def evaluate(self, check_result, context):
        ...         if check_result.pass_rate >= 0.99:
        ...             return ConditionResult(RouteDecision.PASS, "production")
        ...         return ConditionResult(RouteDecision.FAIL, "quarantine")
    """

    def __init__(
        self,
        config: ConditionBlockConfig | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize condition block.

        Args:
            config: Condition configuration. Uses default if None.
            hooks: Lifecycle hooks to invoke during execution.
        """
        self.config = config or ConditionBlockConfig()
        self._hooks = list(hooks) if hooks else []

    @abstractmethod
    def evaluate(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> ConditionResult:
        """Evaluate conditions and determine routing.

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            ConditionResult with routing decision.
        """
        ...

    def get_route(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> str | None:
        """Get the target route for given check result.

        Convenience method that returns just the route string.

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            Target block UUID or None.
        """
        result = self.evaluate(check_result, context)
        return result.route

    def _extract_metrics(
        self,
        check_result: CheckResult | dict[str, Any],
    ) -> dict[str, Any]:
        """Extract metrics from check result.

        Args:
            check_result: Check result to extract metrics from.

        Returns:
            Dictionary of metrics.
        """
        if isinstance(check_result, dict):
            return {
                "pass_rate": check_result.get("pass_rate", 0.0),
                "failure_rate": check_result.get("failure_rate", 0.0),
                "passed_count": check_result.get("passed_count", 0),
                "failed_count": check_result.get("failed_count", 0),
                "status": check_result.get("status", "UNKNOWN"),
            }
        else:
            return {
                "pass_rate": check_result.pass_rate,
                "failure_rate": check_result.failure_rate,
                "passed_count": check_result.passed_count,
                "failed_count": check_result.failed_count,
                "status": check_result.status.name if check_result.status else "UNKNOWN",
            }


# =============================================================================
# Data Quality Condition
# =============================================================================


class DataQualityCondition(BaseConditionBlock):
    """Condition block for data quality routing.

    Evaluates check results against configured thresholds and
    routes to appropriate downstream blocks.

    The routing logic follows this priority:
    1. ERROR if check status is ERROR
    2. PASS if pass_rate >= pass_threshold
    3. WARNING if warning_threshold configured and pass_rate >= warning_threshold
    4. FAIL otherwise

    Attributes:
        config: Condition configuration.
        custom_conditions: Additional custom condition functions.

    Example:
        >>> condition = DataQualityCondition(
        ...     config=ConditionBlockConfig(
        ...         pass_threshold=0.95,
        ...         route_on_pass="warehouse",
        ...         route_on_fail="quarantine",
        ...     ),
        ... )
        >>> result = condition.evaluate(check_result)
        >>> print(f"Route to: {result.route}")
    """

    def __init__(
        self,
        config: ConditionBlockConfig | None = None,
        custom_conditions: dict[str, Callable[[dict], bool]] | None = None,
        hooks: Sequence[Any] | None = None,
    ) -> None:
        """Initialize data quality condition.

        Args:
            config: Condition configuration.
            custom_conditions: Dict of condition_name -> condition_fn.
            hooks: Lifecycle hooks.
        """
        super().__init__(config=config, hooks=hooks)
        self._custom_conditions = custom_conditions or {}

    def add_condition(
        self,
        name: str,
        condition_fn: Callable[[dict], bool],
        route: str,
    ) -> None:
        """Add a custom condition with associated route.

        Args:
            name: Condition name.
            condition_fn: Function that takes metrics dict and returns bool.
            route: Block UUID to route to when condition is true.
        """
        self._custom_conditions[name] = condition_fn
        self.config = self.config.with_custom_route(name, route)

    def evaluate(
        self,
        check_result: CheckResult | dict[str, Any],
        context: BlockExecutionContext | None = None,
    ) -> ConditionResult:
        """Evaluate conditions and determine routing.

        Args:
            check_result: Result from data quality check.
            context: Block execution context.

        Returns:
            ConditionResult with routing decision.
        """
        metrics = self._extract_metrics(check_result)
        reasons: list[str] = []
        evaluations: dict[str, bool] = {}

        # Check for ERROR status first
        status = metrics.get("status", "UNKNOWN")
        if status == "ERROR":
            return ConditionResult(
                decision=RouteDecision.ERROR,
                route=self.config.route_on_error or self.config.default_route,
                metrics=metrics,
                reasons=("Check status is ERROR",),
            )

        pass_rate = metrics.get("pass_rate", 0.0)

        # Evaluate custom conditions if configured
        for condition_name, condition_fn in self._custom_conditions.items():
            try:
                result = condition_fn(metrics)
                evaluations[condition_name] = result
                if result and condition_name in self.config.custom_routes:
                    return ConditionResult(
                        decision=RouteDecision.PASS,
                        route=self.config.custom_routes[condition_name],
                        metrics=metrics,
                        reasons=(f"Custom condition '{condition_name}' matched",),
                        all_evaluations=evaluations,
                    )
            except Exception as e:
                evaluations[condition_name] = False
                reasons.append(f"Custom condition '{condition_name}' error: {e}")

        # Check pass threshold
        evaluations["pass_threshold"] = pass_rate >= self.config.pass_threshold
        if evaluations["pass_threshold"]:
            reasons.append(
                f"Pass rate {pass_rate:.2%} >= threshold {self.config.pass_threshold:.2%}"
            )
            return ConditionResult(
                decision=RouteDecision.PASS,
                route=self.config.route_on_pass or self.config.default_route,
                metrics=metrics,
                reasons=tuple(reasons),
                all_evaluations=evaluations,
            )

        # Check warning threshold if configured
        if self.config.warning_threshold is not None:
            evaluations["warning_threshold"] = pass_rate >= self.config.warning_threshold
            if evaluations["warning_threshold"]:
                reasons.append(
                    f"Pass rate {pass_rate:.2%} >= warning threshold "
                    f"{self.config.warning_threshold:.2%}"
                )
                return ConditionResult(
                    decision=RouteDecision.WARNING,
                    route=self.config.route_on_warning or self.config.default_route,
                    metrics=metrics,
                    reasons=tuple(reasons),
                    all_evaluations=evaluations,
                )

        # Default to FAIL
        reasons.append(
            f"Pass rate {pass_rate:.2%} below threshold {self.config.pass_threshold:.2%}"
        )
        return ConditionResult(
            decision=RouteDecision.FAIL,
            route=self.config.route_on_fail or self.config.default_route,
            metrics=metrics,
            reasons=tuple(reasons),
            all_evaluations=evaluations,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_quality_condition(
    pass_threshold: float = 0.95,
    warning_threshold: float | None = None,
    route_on_pass: str | None = None,
    route_on_fail: str | None = None,
    route_on_warning: str | None = None,
    default_route: str | None = None,
    custom_conditions: dict[str, Callable[[dict], bool]] | None = None,
    **kwargs: Any,
) -> DataQualityCondition:
    """Create a data quality condition with the given configuration.

    Args:
        pass_threshold: Pass rate threshold for PASS routing.
        warning_threshold: Pass rate threshold for WARNING routing.
        route_on_pass: Block to route to on PASS.
        route_on_fail: Block to route to on FAIL.
        route_on_warning: Block to route to on WARNING.
        default_route: Default route when no condition matches.
        custom_conditions: Dict of custom condition functions.
        **kwargs: Additional configuration options.

    Returns:
        Configured DataQualityCondition instance.

    Example:
        >>> condition = create_quality_condition(
        ...     pass_threshold=0.99,
        ...     route_on_pass="production",
        ...     route_on_fail="staging",
        ... )
    """
    config = ConditionBlockConfig(
        pass_threshold=pass_threshold,
        warning_threshold=warning_threshold,
        route_on_pass=route_on_pass,
        route_on_fail=route_on_fail,
        route_on_warning=route_on_warning,
        default_route=default_route,
        **kwargs,
    )
    return DataQualityCondition(config=config, custom_conditions=custom_conditions)
