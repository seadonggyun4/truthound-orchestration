"""Prefect Tasks for data quality operations.

This package provides Prefect Task implementations for data quality
operations including check, profile, and learn.

Tasks:
    - data_quality_check_task: Execute data quality checks
    - data_quality_profile_task: Profile data characteristics
    - data_quality_learn_task: Learn rules from data

Factory Functions:
    - create_check_task: Create configured check tasks
    - create_profile_task: Create configured profile tasks
    - create_learn_task: Create configured learn tasks

Example:
    >>> from prefect import flow
    >>> from truthound_prefect.tasks import data_quality_check_task
    >>> from truthound_prefect.blocks import DataQualityBlock
    >>>
    >>> @flow
    ... async def validate_data():
    ...     block = DataQualityBlock(engine_name="truthound")
    ...     data = load_data()
    ...     result = await data_quality_check_task(
    ...         data=data,
    ...         block=block,
    ...         auto_schema=True,
    ...     )
    ...     return result
"""

from truthound_prefect.tasks.base import (
    AUTO_SCHEMA_CHECK_CONFIG,
    DEFAULT_CHECK_CONFIG,
    DEFAULT_LEARN_CONFIG,
    DEFAULT_PROFILE_CONFIG,
    FULL_PROFILE_CONFIG,
    LENIENT_CHECK_CONFIG,
    MINIMAL_PROFILE_CONFIG,
    STRICT_CHECK_CONFIG,
    STRICT_LEARN_CONFIG,
    BaseTaskConfig,
    CheckTaskConfig,
    LearnTaskConfig,
    ProfileTaskConfig,
)
from truthound_prefect.tasks.check import (
    auto_schema_check_task,
    create_check_task,
    data_quality_check_task,
    lenient_check_task,
    strict_check_task,
)
from truthound_prefect.tasks.learn import (
    create_learn_task,
    data_quality_learn_task,
    standard_learn_task,
    strict_learn_task,
)
from truthound_prefect.tasks.profile import (
    create_profile_task,
    data_quality_profile_task,
    full_profile_task,
    minimal_profile_task,
)

__all__ = [
    # Base configs
    "BaseTaskConfig",
    "CheckTaskConfig",
    "ProfileTaskConfig",
    "LearnTaskConfig",
    # Check presets
    "DEFAULT_CHECK_CONFIG",
    "STRICT_CHECK_CONFIG",
    "LENIENT_CHECK_CONFIG",
    "AUTO_SCHEMA_CHECK_CONFIG",
    # Profile presets
    "DEFAULT_PROFILE_CONFIG",
    "MINIMAL_PROFILE_CONFIG",
    "FULL_PROFILE_CONFIG",
    # Learn presets
    "DEFAULT_LEARN_CONFIG",
    "STRICT_LEARN_CONFIG",
    # Check tasks
    "data_quality_check_task",
    "create_check_task",
    "strict_check_task",
    "lenient_check_task",
    "auto_schema_check_task",
    # Profile tasks
    "data_quality_profile_task",
    "create_profile_task",
    "minimal_profile_task",
    "full_profile_task",
    # Learn tasks
    "data_quality_learn_task",
    "create_learn_task",
    "standard_learn_task",
    "strict_learn_task",
]
