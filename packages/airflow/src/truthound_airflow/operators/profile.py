"""Data Quality Profile Operator for Apache Airflow.

This module provides the DataQualityProfileOperator for executing data
profiling operations in Airflow DAGs. Profiling analyzes data characteristics
including statistics, patterns, and distributions.

Example:
    >>> from truthound_airflow import DataQualityProfileOperator
    >>>
    >>> profile = DataQualityProfileOperator(
    ...     task_id="profile_sales_data",
    ...     data_path="s3://bucket/sales/{{ ds }}/data.parquet",
    ...     columns=["amount", "quantity", "discount"],
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from truthound_airflow.operators.base import BaseDataQualityOperator

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.base import ProfileResult
    from common.engines.base import DataQualityEngine


class DataQualityProfileOperator(BaseDataQualityOperator):
    """Execute data profiling analysis.

    This operator loads data from a file or SQL query and runs profiling
    analysis using the configured DataQualityEngine. Results include
    statistics, pattern detection, and distribution analysis.

    Parameters
    ----------
    data_path : str | None
        Path to data file. Mutually exclusive with `sql`.

    sql : str | None
        SQL query to fetch data. Mutually exclusive with `data_path`.

    connection_id : str
        Airflow Connection ID for data source.
        Default: "truthound_default"

    columns : list[str] | None
        Specific columns to profile. None means all columns.

    include_statistics : bool
        Whether to include statistical analysis (mean, std, etc.).
        Default: True

    include_patterns : bool
        Whether to detect data patterns (email, phone, etc.).
        Default: True

    include_distributions : bool
        Whether to analyze value distributions.
        Default: True

    sample_size : int | None
        Number of rows to sample. None means all rows.

    engine : DataQualityEngine | None
        Custom engine instance. Default: Truthound.

    timeout_seconds : int
        Profiling timeout in seconds. Default: 300

    xcom_push_key : str
        Key for XCom result. Default: "data_quality_profile"

    Examples
    --------
    Basic profiling:

    >>> profile = DataQualityProfileOperator(
    ...     task_id="profile_data",
    ...     data_path="s3://bucket/data.parquet",
    ... )

    Profile specific columns:

    >>> profile = DataQualityProfileOperator(
    ...     task_id="profile_numeric",
    ...     data_path="s3://bucket/transactions.parquet",
    ...     columns=["amount", "quantity", "discount"],
    ...     include_distributions=True,
    ... )

    Profile from SQL:

    >>> profile = DataQualityProfileOperator(
    ...     task_id="profile_users",
    ...     sql="SELECT * FROM users WHERE created_at > '{{ ds }}'",
    ...     connection_id="postgres_default",
    ...     sample_size=10000,
    ... )
    """

    template_fields: Sequence[str] = (
        "data_path",
        "sql",
        "columns",
        "connection_id",
    )
    ui_color: str = "#9B59B6"

    def __init__(
        self,
        *,
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str = "truthound_default",
        columns: list[str] | None = None,
        include_statistics: bool = True,
        include_patterns: bool = True,
        include_distributions: bool = True,
        sample_size: int | None = None,
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        timeout_seconds: int = 300,
        xcom_push_key: str = "data_quality_profile",
        **kwargs: Any,
    ) -> None:
        """Initialize data quality profile operator."""
        super().__init__(
            data_path=data_path,
            sql=sql,
            connection_id=connection_id,
            engine=engine,
            engine_name=engine_name,
            fail_on_error=False,  # Profiling doesn't fail
            timeout_seconds=timeout_seconds,
            xcom_push_key=xcom_push_key,
            **kwargs,
        )

        self.columns = columns
        self.include_statistics = include_statistics
        self.include_patterns = include_patterns
        self.include_distributions = include_distributions
        self.sample_size = sample_size

    def _execute_operation(
        self,
        data: Any,
        context: Context,
    ) -> ProfileResult:
        """Execute data profiling.

        Args:
            data: Loaded data.
            context: Airflow execution context.

        Returns:
            ProfileResult: Profiling result.
        """
        columns_desc = f" for columns {self.columns}" if self.columns else ""
        self.log.info(f"Executing data profiling{columns_desc}")

        # Sample data if configured
        if self.sample_size and hasattr(data, "__len__") and len(data) > self.sample_size:
            self.log.info(f"Sampling {self.sample_size} rows")
            if hasattr(data, "sample"):
                data = data.sample(n=self.sample_size)

        # Execute profiling
        result = self.engine.profile(
            data,
            columns=self.columns,
            include_statistics=self.include_statistics,
            include_patterns=self.include_patterns,
            include_distributions=self.include_distributions,
            timeout=self.timeout_seconds,
        )

        return result

    def _serialize_result(self, result: ProfileResult) -> dict[str, Any]:
        """Serialize ProfileResult for XCom.

        Args:
            result: The profile result.

        Returns:
            dict[str, Any]: XCom-compatible dictionary.
        """
        # ProfileResult has to_dict method
        if hasattr(result, "to_dict"):
            return result.to_dict()

        # Fallback serialization
        return {
            "row_count": getattr(result, "row_count", 0),
            "column_count": getattr(result, "column_count", 0),
            "columns": [
                {
                    "column_name": col.column_name,
                    "dtype": str(col.dtype),
                    "null_count": col.null_count,
                    "null_percentage": col.null_percentage,
                    "unique_count": col.unique_count,
                    "statistics": col.statistics if self.include_statistics else {},
                    "patterns": col.patterns if self.include_patterns else [],
                    "distribution": col.distribution if self.include_distributions else {},
                }
                for col in getattr(result, "columns", [])
            ],
            "execution_time_ms": getattr(result, "execution_time_ms", 0.0),
        }

    def _log_metrics(self, result_dict: dict[str, Any]) -> None:
        """Log profiling metrics.

        Args:
            result_dict: Serialized result dictionary.
        """
        self.log.info(
            f"Profiling Results: "
            f"rows={result_dict.get('row_count', 'N/A')}, "
            f"columns={result_dict.get('column_count', 'N/A')}, "
            f"duration={result_dict.get('execution_time_ms', 0):.2f}ms"
        )


# Alias for backwards compatibility
TruthoundProfileOperator = DataQualityProfileOperator
