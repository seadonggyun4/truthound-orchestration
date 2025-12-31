"""Enterprise Data Quality Engine Adapters.

This package provides adapters for enterprise data quality engines,
enabling integration with platforms like Informatica, Talend, and IBM InfoSphere.

Enterprise engines extend the common DataQualityEngine protocol with
additional features for:
- API-based authentication and connection management
- Complex rule translation to vendor-specific formats
- Result conversion from vendor formats to common types
- Connection pooling and retry mechanisms
- Health checking and lifecycle management

Quick Start:
    >>> from packages.enterprise.engines import (
    ...     get_enterprise_engine,
    ...     InformaticaAdapter,
    ...     TalendAdapter,
    ...     IBMInfoSphereAdapter,
    ... )
    >>>
    >>> # Get engine from registry
    >>> engine = get_enterprise_engine("informatica")
    >>> engine = get_enterprise_engine("ibm_infosphere")
    >>>
    >>> # Or create directly with config
    >>> from packages.enterprise.engines import InformaticaConfig
    >>> config = InformaticaConfig(
    ...     api_endpoint="https://idq.example.com/api/v2",
    ...     api_key="your-api-key",
    ... )
    >>> engine = InformaticaAdapter(config=config)
    >>>
    >>> # Use with context manager
    >>> with engine:
    ...     result = engine.check(data, rules)
    ...     profile = engine.profile(data)

Factory Functions:
    >>> from packages.enterprise.engines import (
    ...     create_informatica_adapter,
    ...     create_talend_adapter,
    ...     create_ibm_infosphere_adapter,
    ... )
    >>>
    >>> adapter = create_informatica_adapter(
    ...     api_endpoint="https://idq.example.com/api/v2",
    ...     api_key="secret",
    ...     domain="Production",
    ... )
    >>>
    >>> adapter = create_ibm_infosphere_adapter(
    ...     api_endpoint="https://iis.example.com/ibm/iis/ia/api/v1",
    ...     username="admin",
    ...     password="secret",
    ...     project="DataQuality",
    ... )

Entry Point Registration:
    Add to pyproject.toml to enable plugin discovery:
    ```toml
    [project.entry-points."truthound.engines"]
    informatica = "packages.enterprise.engines:InformaticaAdapter"
    talend = "packages.enterprise.engines:TalendAdapter"
    ibm_infosphere = "packages.enterprise.engines:IBMInfoSphereAdapter"
    ```

Integration with Common Registry:
    >>> from packages.enterprise.engines import register_with_common_registry
    >>> register_with_common_registry()
    >>>
    >>> # Now use through common.engines
    >>> from common.engines import get_engine
    >>> engine = get_engine("informatica")
    >>> engine = get_engine("ibm_infosphere")
"""

from __future__ import annotations

# =============================================================================
# Base Classes and Protocols
# =============================================================================

from packages.enterprise.engines.base import (
    # Main base class
    EnterpriseEngineAdapter,
    EnterpriseEngineConfig,
    # Protocols
    RuleTranslator,
    ResultConverter,
    ConnectionManager,
    # Base implementations
    BaseRuleTranslator,
    BaseResultConverter,
    BaseConnectionManager,
    # Enums
    AuthType,
    ConnectionMode,
    DataTransferMode,
    # Exceptions
    EnterpriseEngineError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    VendorSDKError,
    RuleTranslationError,
    # Preset configurations
    DEFAULT_ENTERPRISE_CONFIG,
    PRODUCTION_ENTERPRISE_CONFIG,
    DEVELOPMENT_ENTERPRISE_CONFIG,
    HIGH_THROUGHPUT_CONFIG,
)

# =============================================================================
# Informatica Adapter
# =============================================================================

from packages.enterprise.engines.informatica import (
    # Main adapter
    InformaticaAdapter,
    InformaticaConfig,
    # Internal components (for extension)
    InformaticaRuleTranslator,
    InformaticaResultConverter,
    InformaticaConnectionManager,
    # Factory function
    create_informatica_adapter,
    # Preset configurations
    DEFAULT_INFORMATICA_CONFIG,
    PRODUCTION_INFORMATICA_CONFIG,
    DEVELOPMENT_INFORMATICA_CONFIG,
)

# =============================================================================
# Talend Adapter
# =============================================================================

from packages.enterprise.engines.talend import (
    # Main adapter
    TalendAdapter,
    TalendConfig,
    # Enums
    TalendExecutionMode,
    TalendIndicatorType,
    # Internal components (for extension)
    TalendRuleTranslator,
    TalendResultConverter,
    TalendConnectionManager,
    # Factory function
    create_talend_adapter,
    # Preset configurations
    DEFAULT_TALEND_CONFIG,
    PRODUCTION_TALEND_CONFIG,
    DEVELOPMENT_TALEND_CONFIG,
    EMBEDDED_TALEND_CONFIG,
)

# =============================================================================
# IBM InfoSphere Adapter
# =============================================================================

from packages.enterprise.engines.ibm_infosphere import (
    # Main adapter
    IBMInfoSphereAdapter,
    IBMInfoSphereConfig,
    # Enums
    InfoSphereAnalysisType,
    InfoSphereRuleType,
    InfoSphereExecutionMode,
    # Internal components (for extension)
    IBMInfoSphereRuleTranslator,
    IBMInfoSphereResultConverter,
    IBMInfoSphereConnectionManager,
    # Factory function
    create_ibm_infosphere_adapter,
    # Preset configurations
    DEFAULT_INFOSPHERE_CONFIG,
    PRODUCTION_INFOSPHERE_CONFIG,
    DEVELOPMENT_INFOSPHERE_CONFIG,
    BATCH_INFOSPHERE_CONFIG,
)

# =============================================================================
# SAP Data Services Adapter
# =============================================================================

from packages.enterprise.engines.sap_data_services import (
    # Main adapter
    SAPDataServicesAdapter,
    SAPDataServicesConfig,
    # Enums
    SAPExecutionMode,
    SAPRuleType,
    SAPDataType,
    SAPJobStatus,
    # Internal components (for extension)
    SAPDataServicesRuleTranslator,
    SAPDataServicesResultConverter,
    SAPDataServicesConnectionManager,
    # Factory function
    create_sap_data_services_adapter,
    # Preset configurations
    DEFAULT_SAP_DS_CONFIG,
    PRODUCTION_SAP_DS_CONFIG,
    DEVELOPMENT_SAP_DS_CONFIG,
    REALTIME_SAP_DS_CONFIG,
    ADDRESS_CLEANSING_CONFIG,
)

# =============================================================================
# Registry
# =============================================================================

from packages.enterprise.engines.registry import (
    # Registry class
    EnterpriseEngineRegistry,
    EngineRegistration,
    # Global functions
    get_enterprise_engine_registry,
    reset_enterprise_engine_registry,
    get_enterprise_engine,
    register_enterprise_engine,
    list_enterprise_engines,
    is_enterprise_engine_registered,
    # Plugin integration
    create_plugin_spec,
    register_with_common_registry,
    # Exceptions
    EngineNotRegisteredError,
    EngineAlreadyRegisteredError,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Base Classes and Protocols
    # -------------------------------------------------------------------------
    "EnterpriseEngineAdapter",
    "EnterpriseEngineConfig",
    "RuleTranslator",
    "ResultConverter",
    "ConnectionManager",
    "BaseRuleTranslator",
    "BaseResultConverter",
    "BaseConnectionManager",
    # Enums
    "AuthType",
    "ConnectionMode",
    "DataTransferMode",
    # Exceptions
    "EnterpriseEngineError",
    "ConnectionError",
    "AuthenticationError",
    "RateLimitError",
    "VendorSDKError",
    "RuleTranslationError",
    # Preset configurations
    "DEFAULT_ENTERPRISE_CONFIG",
    "PRODUCTION_ENTERPRISE_CONFIG",
    "DEVELOPMENT_ENTERPRISE_CONFIG",
    "HIGH_THROUGHPUT_CONFIG",
    # -------------------------------------------------------------------------
    # Informatica Adapter
    # -------------------------------------------------------------------------
    "InformaticaAdapter",
    "InformaticaConfig",
    "InformaticaRuleTranslator",
    "InformaticaResultConverter",
    "InformaticaConnectionManager",
    "create_informatica_adapter",
    "DEFAULT_INFORMATICA_CONFIG",
    "PRODUCTION_INFORMATICA_CONFIG",
    "DEVELOPMENT_INFORMATICA_CONFIG",
    # -------------------------------------------------------------------------
    # Talend Adapter
    # -------------------------------------------------------------------------
    "TalendAdapter",
    "TalendConfig",
    "TalendExecutionMode",
    "TalendIndicatorType",
    "TalendRuleTranslator",
    "TalendResultConverter",
    "TalendConnectionManager",
    "create_talend_adapter",
    "DEFAULT_TALEND_CONFIG",
    "PRODUCTION_TALEND_CONFIG",
    "DEVELOPMENT_TALEND_CONFIG",
    "EMBEDDED_TALEND_CONFIG",
    # -------------------------------------------------------------------------
    # IBM InfoSphere Adapter
    # -------------------------------------------------------------------------
    "IBMInfoSphereAdapter",
    "IBMInfoSphereConfig",
    "InfoSphereAnalysisType",
    "InfoSphereRuleType",
    "InfoSphereExecutionMode",
    "IBMInfoSphereRuleTranslator",
    "IBMInfoSphereResultConverter",
    "IBMInfoSphereConnectionManager",
    "create_ibm_infosphere_adapter",
    "DEFAULT_INFOSPHERE_CONFIG",
    "PRODUCTION_INFOSPHERE_CONFIG",
    "DEVELOPMENT_INFOSPHERE_CONFIG",
    "BATCH_INFOSPHERE_CONFIG",
    # -------------------------------------------------------------------------
    # SAP Data Services Adapter
    # -------------------------------------------------------------------------
    "SAPDataServicesAdapter",
    "SAPDataServicesConfig",
    "SAPExecutionMode",
    "SAPRuleType",
    "SAPDataType",
    "SAPJobStatus",
    "SAPDataServicesRuleTranslator",
    "SAPDataServicesResultConverter",
    "SAPDataServicesConnectionManager",
    "create_sap_data_services_adapter",
    "DEFAULT_SAP_DS_CONFIG",
    "PRODUCTION_SAP_DS_CONFIG",
    "DEVELOPMENT_SAP_DS_CONFIG",
    "REALTIME_SAP_DS_CONFIG",
    "ADDRESS_CLEANSING_CONFIG",
    # -------------------------------------------------------------------------
    # Registry
    # -------------------------------------------------------------------------
    "EnterpriseEngineRegistry",
    "EngineRegistration",
    "get_enterprise_engine_registry",
    "reset_enterprise_engine_registry",
    "get_enterprise_engine",
    "register_enterprise_engine",
    "list_enterprise_engines",
    "is_enterprise_engine_registered",
    "create_plugin_spec",
    "register_with_common_registry",
    "EngineNotRegisteredError",
    "EngineAlreadyRegisteredError",
]

# Version
__version__ = "0.1.0"
