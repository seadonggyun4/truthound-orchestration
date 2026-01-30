"""Script modules for Kestra data quality integration.

This package provides script entry points for data quality operations
in Kestra Python script tasks.

Main Entry Points:
    check_quality_script: Execute data quality checks.
    profile_data_script: Generate data profiles.
    learn_schema_script: Learn schema/rules from data.
    drift_detection_script: Detect data drift between datasets.
    anomaly_detection_script: Detect anomalies in datasets.

Executor Classes:
    CheckScriptExecutor: Reusable check executor.
    ProfileScriptExecutor: Reusable profile executor.
    LearnScriptExecutor: Reusable learn executor.
    DriftScriptExecutor: Reusable drift detection executor.
    AnomalyScriptExecutor: Reusable anomaly detection executor.

Configuration Classes:
    ScriptConfig: Base configuration.
    CheckScriptConfig: Check-specific configuration.
    ProfileScriptConfig: Profile-specific configuration.
    LearnScriptConfig: Learn-specific configuration.
    DriftScriptConfig: Drift detection configuration.
    AnomalyScriptConfig: Anomaly detection configuration.

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

from truthound_kestra.scripts.anomaly import (
    AnomalyScriptExecutor,
    AnomalyScriptResult,
    anomaly_detection_script,
)
from truthound_kestra.scripts.base import (
    DEFAULT_ANOMALY_SCRIPT_CONFIG,
    DEFAULT_DRIFT_SCRIPT_CONFIG,
    DEFAULT_SCRIPT_CONFIG,
    LENIENT_ANOMALY_SCRIPT_CONFIG,
    LENIENT_DRIFT_SCRIPT_CONFIG,
    LENIENT_SCRIPT_CONFIG,
    PRODUCTION_SCRIPT_CONFIG,
    STRICT_ANOMALY_SCRIPT_CONFIG,
    STRICT_DRIFT_SCRIPT_CONFIG,
    STRICT_SCRIPT_CONFIG,
    AnomalyScriptConfig,
    CheckScriptConfig,
    DataQualityEngineProtocol,
    DriftScriptConfig,
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
from truthound_kestra.scripts.drift import (
    DriftScriptExecutor,
    DriftScriptResult,
    drift_detection_script,
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
    "drift_detection_script",
    "anomaly_detection_script",
    # Executors
    "CheckScriptExecutor",
    "ProfileScriptExecutor",
    "LearnScriptExecutor",
    "DriftScriptExecutor",
    "AnomalyScriptExecutor",
    # Results
    "CheckScriptResult",
    "ProfileScriptResult",
    "LearnScriptResult",
    "DriftScriptResult",
    "AnomalyScriptResult",
    # Configuration
    "ScriptConfig",
    "CheckScriptConfig",
    "ProfileScriptConfig",
    "LearnScriptConfig",
    "DriftScriptConfig",
    "AnomalyScriptConfig",
    # Presets
    "DEFAULT_SCRIPT_CONFIG",
    "STRICT_SCRIPT_CONFIG",
    "LENIENT_SCRIPT_CONFIG",
    "PRODUCTION_SCRIPT_CONFIG",
    "DEFAULT_DRIFT_SCRIPT_CONFIG",
    "STRICT_DRIFT_SCRIPT_CONFIG",
    "LENIENT_DRIFT_SCRIPT_CONFIG",
    "DEFAULT_ANOMALY_SCRIPT_CONFIG",
    "STRICT_ANOMALY_SCRIPT_CONFIG",
    "LENIENT_ANOMALY_SCRIPT_CONFIG",
    # Protocols
    "DataQualityEngineProtocol",
    "ScriptExecutorProtocol",
    # Utilities
    "get_engine",
    "create_script_config",
]
