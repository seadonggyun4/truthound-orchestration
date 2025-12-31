"""Data Quality Learn Ops for Dagster.

This module provides Dagster ops for learning validation rules from data.
Learning analyzes data to suggest appropriate validation rules.

Example:
    >>> from dagster import job
    >>> from truthound_dagster.ops import data_quality_learn_op
    >>> from truthound_dagster.resources import DataQualityResource
    >>>
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def learn_job():
    ...     data_quality_learn_op()
"""



from typing import TYPE_CHECKING, Any, Callable, Optional, Dict, List

from dagster import Config, In, OpExecutionContext, Out, op

from truthound_dagster.ops.base import LearnOpConfig

if TYPE_CHECKING:
    from common.base import LearnResult

    from truthound_dagster.resources import DataQualityResource


class LearnOpDagsterConfig(Config):
    """Dagster configuration schema for learn ops."""

    infer_constraints: bool = True
    min_confidence: float = 0.8
    categorical_threshold: int = 20
    timeout_seconds: float = 300.0


def _serialize_learn_result(result: "LearnResult") -> Dict[str, Any]:
    """Serialize LearnResult for Dagster metadata.

    Args:
        result: Learn result to serialize.

    Returns:
        Dict[str, Any]: Serialized result.
    """
    rules = []
    for rule in result.rules:
        rule_data = {
            "column": rule.column,
            "rule_type": rule.rule_type,
            "confidence": rule.confidence,
            "parameters": rule.parameters,
        }
        rules.append(rule_data)

    return {
        "rule_count": len(result.rules),
        "rules": rules,
        "execution_time_ms": result.execution_time_ms,
        "timestamp": result.timestamp.isoformat(),
    }


@op(
    name="data_quality_learn",
    description="Learn validation rules from data patterns.",
    ins={"data": In(description="Data to learn from (DataFrame, path, etc.)")},
    out=Out(description="Learn result with suggested rules"),
    tags={"kind": "data_quality", "operation": "learn"},
    required_resource_keys={"data_quality"},
)
def data_quality_learn_op(
    context: OpExecutionContext,
    data: Any,
    config: LearnOpDagsterConfig,
) -> Dict[str, Any]:
    """Learn validation rules from data.

    This op analyzes data to suggest appropriate validation rules.
    It infers constraints, types, and patterns from the data and
    generates rules with confidence scores.

    Parameters
    ----------
    context : OpExecutionContext
        Dagster execution context.

    data : Any
        Data to learn from.

    config : LearnOpDagsterConfig
        Op configuration.

    Returns
    -------
    Dict[str, Any]
        Serialized learn result containing:
        - rule_count: Number of learned rules
        - rules: List of suggested rules with confidence
        - execution_time_ms: Execution time

    Examples
    --------
    >>> @job(resource_defs={"data_quality": DataQualityResource()})
    ... def learn_job():
    ...     data = load_data_op()
    ...     rules = data_quality_learn_op(data)
    ...     return rules
    """
    dq_resource: DataQualityResource = context.resources.data_quality

    context.log.info("Starting rule learning")

    # Execute learning
    result = dq_resource.learn(
        data=data,
        timeout=config.timeout_seconds,
        infer_constraints=config.infer_constraints,
        min_confidence=config.min_confidence,
        categorical_threshold=config.categorical_threshold,
    )

    context.log.info(
        f"Learning complete: {len(result.rules)} rules learned "
        f"in {result.execution_time_ms:.2f}ms"
    )

    # Log learned rules
    high_confidence_rules = [r for r in result.rules if r.confidence >= 0.9]
    context.log.info(f"  High confidence rules (>=90%): {len(high_confidence_rules)}")

    for rule in result.rules[:5]:
        context.log.info(
            f"  {rule.column}: {rule.rule_type} "
            f"(confidence={rule.confidence:.1%})"
        )
    if len(result.rules) > 5:
        context.log.info(f"  ... and {len(result.rules) - 5} more rules")

    # Serialize and return
    result_dict = _serialize_learn_result(result)

    # Add metadata
    context.add_output_metadata(
        {
            "rule_count": len(result.rules),
            "high_confidence_count": len(high_confidence_rules),
            "execution_time_ms": result.execution_time_ms,
        }
    )

    return result_dict


def create_learn_op(
    name: str,
    *,
    infer_constraints: bool = True,
    min_confidence: float = 0.8,
    categorical_threshold: int = 20,
    timeout_seconds: float = 300.0,
    description: str | None = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Create a customized rule learning op.

    Parameters
    ----------
    name : str
        Name for the op.

    infer_constraints : bool
        Infer constraints from data.

    min_confidence : float
        Minimum confidence for rules.

    categorical_threshold : int
        Max unique values for categorical.

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
    >>> users_learn = create_learn_op(
    ...     name="users_learn",
    ...     min_confidence=0.95,
    ... )
    """
    config = LearnOpConfig(
        infer_constraints=infer_constraints,
        min_confidence=min_confidence,
        categorical_threshold=categorical_threshold,
        timeout_seconds=timeout_seconds,
    )

    op_tags = {"kind": "data_quality", "operation": "learn"}
    if tags:
        op_tags.update(tags)

    op_description = description or f"Rule learning: {name}"

    @op(
        name=name,
        description=op_description,
        ins={"data": In(description="Data to learn from")},
        out=Out(description="Learn result"),
        tags=op_tags,
        required_resource_keys={"data_quality"},
    )
    def learn_op_impl(context: OpExecutionContext, data: Any) -> Dict[str, Any]:
        """Execute configured rule learning."""
        dq_resource: DataQualityResource = context.resources.data_quality

        context.log.info(f"Starting {name}")

        result = dq_resource.learn(
            data=data,
            timeout=config.timeout_seconds,
            infer_constraints=config.infer_constraints,
            min_confidence=config.min_confidence,
            categorical_threshold=config.categorical_threshold,
        )

        context.log.info(f"{name} complete: {len(result.rules)} rules learned")

        result_dict = _serialize_learn_result(result)

        context.add_output_metadata(
            {
                "rule_count": len(result.rules),
            }
        )

        return result_dict

    return learn_op_impl
