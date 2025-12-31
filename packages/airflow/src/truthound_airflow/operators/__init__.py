"""Data Quality Operators for Apache Airflow.

This module provides operators for executing data quality operations
in Airflow DAGs. All operators are engine-agnostic and support
Truthound, Great Expectations, Pandera, and custom engines.

Available Operators:
    - DataQualityCheckOperator: Execute validation checks
    - DataQualityProfileOperator: Run data profiling
    - DataQualityLearnOperator: Learn validation rules from data

Legacy Aliases (for backwards compatibility):
    - TruthoundCheckOperator -> DataQualityCheckOperator
    - TruthoundProfileOperator -> DataQualityProfileOperator
    - TruthoundLearnOperator -> DataQualityLearnOperator

Example:
    >>> from truthound_airflow.operators import (
    ...     DataQualityCheckOperator,
    ...     DataQualityProfileOperator,
    ...     DataQualityLearnOperator,
    ... )
    >>>
    >>> check = DataQualityCheckOperator(
    ...     task_id="check_quality",
    ...     rules=[{"column": "id", "type": "not_null"}],
    ...     data_path="s3://bucket/data.parquet",
    ... )
"""

from truthound_airflow.operators.base import (
    BaseDataQualityOperator,
    CheckOperatorConfig,
    LearnOperatorConfig,
    OperatorConfig,
    ProfileOperatorConfig,
    ResultHandler,
)
from truthound_airflow.operators.check import (
    DataQualityCheckOperator,
    TruthoundCheckOperator,
)
from truthound_airflow.operators.learn import (
    DataQualityLearnOperator,
    TruthoundLearnOperator,
)
from truthound_airflow.operators.profile import (
    DataQualityProfileOperator,
    TruthoundProfileOperator,
)

__all__ = [
    # Base classes
    "BaseDataQualityOperator",
    "ResultHandler",
    # Configuration
    "OperatorConfig",
    "CheckOperatorConfig",
    "ProfileOperatorConfig",
    "LearnOperatorConfig",
    # Engine-agnostic operators (recommended)
    "DataQualityCheckOperator",
    "DataQualityProfileOperator",
    "DataQualityLearnOperator",
    # Legacy aliases (backwards compatibility)
    "TruthoundCheckOperator",
    "TruthoundProfileOperator",
    "TruthoundLearnOperator",
]
