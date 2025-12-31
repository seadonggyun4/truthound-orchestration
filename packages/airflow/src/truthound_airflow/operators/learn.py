"""Data Quality Learn Operator for Apache Airflow.

This module provides the DataQualityLearnOperator for automatically learning
validation rules and schemas from data. Useful for establishing baselines
and detecting schema drift.

Example:
    >>> from truthound_airflow import DataQualityLearnOperator
    >>>
    >>> learn = DataQualityLearnOperator(
    ...     task_id="learn_schema",
    ...     data_path="s3://bucket/baseline/data.parquet",
    ...     output_path="s3://bucket/schemas/baseline.json",
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from truthound_airflow.operators.base import BaseDataQualityOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.base import LearnResult
    from common.engines.base import DataQualityEngine


class DataQualityLearnOperator(BaseDataQualityOperator):
    """Learn validation rules and schema from data.

    This operator analyzes data to automatically discover validation rules
    and schema constraints. The learned rules can be saved to a file and
    used for subsequent validation checks.

    Use cases:
    - Establishing baseline validation rules
    - Detecting schema drift between data versions
    - Auto-generating quality rules for new data sources

    Parameters
    ----------
    data_path : str | None
        Path to data file. Mutually exclusive with `sql`.

    sql : str | None
        SQL query to fetch data. Mutually exclusive with `data_path`.

    connection_id : str
        Airflow Connection ID. Default: "truthound_default"

    output_path : str | None
        Path to save learned schema/rules.
        If None, rules are only pushed to XCom.

    strictness : str
        Learning strictness level.
        Options: "strict", "moderate", "lenient"
        Default: "moderate"

    infer_constraints : bool
        Whether to infer value constraints (ranges, patterns, etc.).
        Default: True

    categorical_threshold : int
        Maximum unique values to consider column as categorical.
        Default: 20

    sample_size : int | None
        Number of rows to sample for learning.

    engine : DataQualityEngine | None
        Custom engine instance. Default: Truthound.

    xcom_push_key : str
        Key for XCom result. Default: "data_quality_schema"

    Examples
    --------
    Basic schema learning:

    >>> learn = DataQualityLearnOperator(
    ...     task_id="learn_baseline",
    ...     data_path="s3://bucket/baseline.parquet",
    ...     output_path="s3://bucket/schemas/baseline.json",
    ... )

    Strict learning with constraints:

    >>> learn = DataQualityLearnOperator(
    ...     task_id="learn_strict",
    ...     data_path="s3://bucket/data.parquet",
    ...     strictness="strict",
    ...     infer_constraints=True,
    ...     categorical_threshold=50,
    ... )

    Learn and compare with previous:

    >>> learn_new = DataQualityLearnOperator(
    ...     task_id="learn_new_schema",
    ...     data_path="s3://bucket/new_data.parquet",
    ... )
    >>>
    >>> def compare_schemas(**context):
    ...     new = context['ti'].xcom_pull(task_ids='learn_new_schema')
    ...     old = load_previous_schema()
    ...     return compare(new, old)
    """

    template_fields: Sequence[str] = (
        "data_path",
        "sql",
        "output_path",
        "connection_id",
    )
    ui_color: str = "#2ECC71"

    def __init__(
        self,
        *,
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        output_path: str | None = None,
        strictness: str = "moderate",
        infer_constraints: bool = True,
        categorical_threshold: int = 20,
        sample_size: int | None = None,
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        timeout_seconds: int = 300,
        xcom_push_key: str = "data_quality_schema",
        **kwargs: Any,
    ) -> None:
        """Initialize data quality learn operator."""
        # Validate strictness
        valid_strictness = {"strict", "moderate", "lenient"}
        if strictness not in valid_strictness:
            msg = f"strictness must be one of {valid_strictness}"
            raise ValueError(msg)

        super().__init__(
            data_path=data_path,
            sql=sql,
            connection_id=connection_id,
            engine=engine,
            engine_name=engine_name,
            fail_on_error=False,  # Learning doesn't fail
            timeout_seconds=timeout_seconds,
            xcom_push_key=xcom_push_key,
            **kwargs,
        )

        self.output_path = output_path
        self.strictness = strictness
        self.infer_constraints = infer_constraints
        self.categorical_threshold = categorical_threshold
        self.sample_size = sample_size

    def _execute_operation(
        self,
        data: Any,
        context: Context,
    ) -> LearnResult:
        """Execute schema/rule learning.

        Args:
            data: Loaded data.
            context: Airflow execution context.

        Returns:
            LearnResult: Learning result with discovered rules.
        """
        self.log.info(f"Learning schema with strictness={self.strictness}")

        # Sample data if configured
        if self.sample_size and hasattr(data, "__len__") and len(data) > self.sample_size:
            self.log.info(f"Sampling {self.sample_size} rows for learning")
            if hasattr(data, "sample"):
                data = data.sample(n=self.sample_size)

        # Execute learning
        result = self.engine.learn(
            data,
            strictness=self.strictness,
            infer_constraints=self.infer_constraints,
            categorical_threshold=self.categorical_threshold,
            timeout=self.timeout_seconds,
        )

        # Save to output path if configured
        if self.output_path:
            self._save_schema(result, context)

        return result

    def _save_schema(self, result: LearnResult, context: Context) -> None:
        """Save learned schema to output path.

        Args:
            result: Learning result.
            context: Airflow execution context.
        """
        from truthound_airflow.hooks.base import DataQualityHook

        self.log.info(f"Saving learned schema to: {self.output_path}")

        hook = DataQualityHook(connection_id=self.connection_id)

        # Serialize and save
        schema_dict = self._serialize_result(result)
        hook.save_json(schema_dict, self.output_path)

        self.log.info(f"Schema saved successfully to: {self.output_path}")

    def _serialize_result(self, result: LearnResult) -> dict[str, Any]:
        """Serialize LearnResult for XCom.

        Args:
            result: The learn result.

        Returns:
            dict[str, Any]: XCom-compatible dictionary.
        """
        # LearnResult has to_dict method
        if hasattr(result, "to_dict"):
            return result.to_dict()

        # Fallback serialization
        return {
            "rules": [
                {
                    "column": rule.column,
                    "rule_type": rule.rule_type,
                    "parameters": rule.parameters,
                    "confidence": rule.confidence,
                }
                for rule in getattr(result, "rules", [])
            ],
            "schema": {
                "columns": [
                    {
                        "name": col.name,
                        "dtype": str(col.dtype),
                        "nullable": col.nullable,
                    }
                    for col in getattr(result, "columns", [])
                ],
            },
            "strictness": self.strictness,
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
        }

    def _log_metrics(self, result_dict: dict[str, Any]) -> None:
        """Log learning metrics.

        Args:
            result_dict: Serialized result dictionary.
        """
        rules_count = len(result_dict.get("rules", []))
        columns_count = len(result_dict.get("schema", {}).get("columns", []))

        self.log.info(
            f"Learning Results: "
            f"rules_learned={rules_count}, "
            f"columns_analyzed={columns_count}, "
            f"strictness={self.strictness}, "
            f"duration={result_dict.get('execution_time_ms', 0):.2f}ms"
        )


# Alias for backwards compatibility
TruthoundLearnOperator = DataQualityLearnOperator
