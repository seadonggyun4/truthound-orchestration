"""Script modules for Kestra data quality integration.

This package provides script entry points for data quality operations
in Kestra Python script tasks.

Main Entry Points:
    check_quality_script: Execute data quality checks.
    profile_data_script: Generate data profiles.
    learn_schema_script: Learn schema/rules from data.

Executor Classes:
    CheckScriptExecutor: Reusable check executor.
    ProfileScriptExecutor: Reusable profile executor.
    LearnScriptExecutor: Reusable learn executor.

Configuration Classes:
    ScriptConfig: Base configuration.
    CheckScriptConfig: Check-specific configuration.
    ProfileScriptConfig: Profile-specific configuration.
    LearnScriptConfig: Learn-specific configuration.

Example:
    >>> # In Kestra Python Script task
    >>> from truthound_kestra.scripts import check_quality_script
    >>>
    >>> check_quality_script(
    ...     input_uri="{{ outputs.extract.uri }}",
    ...     rules=[{"type": "not_null", "column": "id"}],
    ...     fail_on_error=True
    ... )
"""

from truthound_kestra.scripts.base import (
    DEFAULT_SCRIPT_CONFIG,
    LENIENT_SCRIPT_CONFIG,
    PRODUCTION_SCRIPT_CONFIG,
    STRICT_SCRIPT_CONFIG,
    CheckScriptConfig,
    DataQualityEngineProtocol,
    LearnScriptConfig,
    ProfileScriptConfig,
    ScriptConfig,
    ScriptExecutorProtocol,
    create_script_config,
    get_engine,
)
from truthound_kestra.scripts.check import (
    CheckScriptExecutor,
    CheckScriptResult,
    check_quality_script,
)
from truthound_kestra.scripts.learn import (
    LearnScriptExecutor,
    LearnScriptResult,
    learn_schema_script,
)
from truthound_kestra.scripts.profile import (
    ProfileScriptExecutor,
    ProfileScriptResult,
    profile_data_script,
)

__all__ = [
    # Main entry points
    "check_quality_script",
    "profile_data_script",
    "learn_schema_script",
    # Executors
    "CheckScriptExecutor",
    "ProfileScriptExecutor",
    "LearnScriptExecutor",
    # Results
    "CheckScriptResult",
    "ProfileScriptResult",
    "LearnScriptResult",
    # Configuration
    "ScriptConfig",
    "CheckScriptConfig",
    "ProfileScriptConfig",
    "LearnScriptConfig",
    # Presets
    "DEFAULT_SCRIPT_CONFIG",
    "STRICT_SCRIPT_CONFIG",
    "LENIENT_SCRIPT_CONFIG",
    "PRODUCTION_SCRIPT_CONFIG",
    # Protocols
    "DataQualityEngineProtocol",
    "ScriptExecutorProtocol",
    # Utilities
    "get_engine",
    "create_script_config",
]
