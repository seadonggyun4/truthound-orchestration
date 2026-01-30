<img width="300" height="300" alt="Truthound_icon" src="https://github.com/user-attachments/assets/90d9e806-8895-45ec-97dc-f8300da4d997" />

> **Alpha Stage** — This project is currently in **alpha**. APIs may change without notice. Not recommended for production use yet.

# Truthound Orchestration

[![PyPI version](https://img.shields.io/pypi/v/truthound-orchestration.svg)](https://pypi.org/project/truthound-orchestration/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Downloads](https://img.shields.io/pepy/dt/truthound-orchestration?color=brightgreen)](https://pepy.tech/project/truthound-orchestration)

**Truthound Orchestration** is a **generic data quality integration framework** that provides adapters for major workflow orchestration platforms. While [Truthound](https://github.com/seadonggyun4/Truthound) serves as the default data quality engine, the framework supports **any data quality engine** through its Protocol-based architecture.

**Documentation**: [https://truthound.netlify.app](https://truthound.netlify.app/orchestration/)

---

## Quick Start

```bash
# Install from PyPI
pip install truthound-orchestration

# With platform integration
pip install truthound-orchestration[airflow]
pip install truthound-orchestration[dagster]
pip install truthound-orchestration[prefect]

# All platforms
pip install truthound-orchestration[all]
```

```python
from common.engines import TruthoundEngine
import polars as pl

engine = TruthoundEngine()
df = pl.read_csv("data.csv")

with engine:
    # Data validation
    result = engine.check(df, auto_schema=True)
    print(f"Status: {result.status.name}")

    # Drift detection (14 statistical methods)
    drift = engine.detect_drift(baseline_df, current_df, method="ks")
    print(f"Drifted: {drift.is_drifted}, Rate: {drift.drift_rate:.2%}")

    # Anomaly detection (ML-based)
    anomalies = engine.detect_anomalies(df, detector="isolation_forest")
    print(f"Anomalies: {anomalies.has_anomalies}, Rate: {anomalies.anomaly_rate:.2%}")
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Architecture](#architecture)
- [Implementation Status](#implementation-status)
- [Supported Platforms](#supported-platforms)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
  - [Drift Detection](#drift-detection)
  - [Anomaly Detection](#anomaly-detection)
  - [Streaming Validation](#streaming-validation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Common Module Documentation](#common-module-documentation)
- [Enterprise Features](#enterprise-features)
- [Related Projects](#related-projects)
- [Support](#support)

---

## Overview

### Motivation

Modern data ecosystems require robust quality assurance mechanisms that integrate natively with workflow orchestration tools. Truthound Orchestration addresses this requirement by providing **engine-agnostic** adapters for each supported platform, ensuring that data quality validation becomes a first-class operation within existing pipeline architectures—regardless of which data quality engine you choose.

### Design Principles

| Principle | Description |
|-----------|-------------|
| Engine-Agnostic Design | Supports any data quality engine via the `DataQualityEngine` Protocol |
| Platform-Native Patterns | Adheres to the idiomatic conventions of each target platform |
| Protocol-Based Architecture | Employs Python Protocols for loose coupling and extensibility |
| Independent Versioning | Maintains separate release cycles aligned with platform evolution |
| Zero-Configuration Defaults | Provides sensible defaults (Truthound) while supporting advanced customization |

### Core Capabilities

- **Engine Abstraction**: Plug in any data quality engine (Truthound, Great Expectations, Pandera, custom engines)
- **Data Validation**: Execute comprehensive validation rules across multiple data quality dimensions
- **Data Profiling**: Perform automated statistical analysis and pattern detection
- **Schema Learning**: Automatically infer validation rules from data characteristics
- **Data Drift Detection**: Detect distribution changes between baseline and current data using 14 statistical methods (KS, PSI, Chi2, KL, JS, Wasserstein, etc.)
- **Anomaly Detection**: ML-based anomaly detection with Isolation Forest, Z-Score, LOF, and Ensemble detectors
- **Streaming Validation**: Memory-efficient batch-by-batch validation of streaming data via Iterator/Generator patterns
- **Cross-Platform Consistency**: Maintain uniform validation semantics across all supported platforms

---

## Architecture

The system architecture employs a layered design pattern with **engine abstraction** at its core. The `DataQualityEngine` Protocol enables any data quality engine to be plugged in, with Truthound as the default implementation. Extended Protocols (`DriftDetectionEngine`, `AnomalyDetectionEngine`, `StreamingEngine`) provide opt-in advanced capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Workflow Orchestration Layer                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────┐ ┌──────┐ ┌────────┐          │
│  │ Airflow │ │ Dagster │ │ Prefect │ │ dbt │ │ Mage │ │ Kestra │          │
│  └────┬────┘ └────┬────┘ └────┬────┘ └──┬──┘ └──┬───┘ └───┬────┘          │
└───────┼───────────┼───────────┼─────────┼───────┼─────────┼────────────────┘
        └───────────┴───────────┴────┬────┴───────┴─────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Common Module                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Protocols  │  │    Config    │  │  Serializers │  │  Exceptions  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │               DataQualityEngine Protocol (Core)                      │   │
│  │   check(data, rules) -> CheckResult                                  │   │
│  │   profile(data) -> ProfileResult                                     │   │
│  │   learn(data) -> LearnResult                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Extended Protocols (Opt-in)                              │   │
│  │   DriftDetectionEngine  -> detect_drift()      (14 methods)          │   │
│  │   AnomalyDetectionEngine -> detect_anomalies() (4 detectors)         │   │
│  │   StreamingEngine        -> check_stream()     (Iterator pattern)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐  ┌─────────────────────────┐  ┌─────────────────┐
│    Truthound    │  │   Great Expectations    │  │  Custom Engine  │
│    (Default)    │  │      (Optional)         │  │   (Optional)    │
│ Drift/Anomaly/  │  └─────────────────────────┘  └─────────────────┘
│   Streaming     │
└─────────────────┘
```

---

## Implementation Status

### Common Module

| Status | Complete |
|--------|----------|

The Common Module provides the foundational types and utilities shared across all platform integrations, including the core `DataQualityEngine` Protocol.

| Component | Description | Status |
|-----------|-------------|--------|
| `base.py` | Protocols (including `DataQualityEngine`), enumerations, configuration, and result types | Complete |
| `config.py` | Environment and file-based configuration management | Complete |
| `exceptions.py` | Hierarchical exception system | Complete |
| `logging.py` | Structured logging with context propagation and sensitive data masking | Complete |
| `retry.py` | Retry decorator with configurable backoff strategies | Complete |
| `circuit_breaker.py` | Circuit breaker pattern for fault tolerance | Complete |
| `health.py` | Health check system with composite checks and aggregation strategies | Complete |
| `metrics.py` | Metrics collection and distributed tracing | Complete |
| `rate_limiter.py` | Rate limiting with multiple algorithms (Token Bucket, Sliding Window, etc.) | Complete |
| `cache.py` | Caching infrastructure with configurable eviction policies and backend abstraction | Complete |
| `serializers.py` | Platform-specific serialization utilities | Complete |
| `testing.py` | Mock objects, fixtures, and assertion helpers | Complete |
| `rule_validation.py` | Rule validation with schema definitions and engine-specific validators | Complete |
| `engines/` | Engine implementations (Truthound default, adapter for other engines) | Complete |
| `engines/batch.py` | Batch operations with chunking, parallel execution, and result aggregation | Complete |
| `engines/config.py` | Engine configuration system (Builder, Loader, Validator, Registry) | Complete |
| `engines/metrics.py` | Engine metrics integration with hooks for logging, metrics, and tracing | Complete |
| `engines/aggregation.py` | Multi-engine result aggregation with strategies, comparison, and weighted scoring | Complete |
| `engines/version.py` | Semantic versioning, version constraints, compatibility checking, and version registry | Complete |
| `exporters/prometheus.py` | Prometheus metrics export with Push Gateway, HTTP server, and multi-tenant support | Complete |

**Key Components:**

- **Engine Protocol**: `DataQualityEngine`, `AsyncDataQualityEngine` - core abstraction for any data quality engine
- **Extended Protocols**: `DriftDetectionEngine`, `AnomalyDetectionEngine`, `StreamingEngine`, `AsyncStreamingEngine` - opt-in protocols for advanced capabilities
- **Engine Lifecycle**: `ManagedEngine`, `AsyncManagedEngine`, `EngineLifecycleManager`, `AsyncEngineLifecycleManager` - lifecycle management with start/stop/health_check
- **Engine Implementations**: `TruthoundEngine` (default, supports drift/anomaly/streaming), `GreatExpectationsAdapter`, `PanderaAdapter`
- **Engine Configuration**: `BaseEngineConfig`, `ConfigBuilder`, `ConfigLoader`, `ConfigValidator`, `ConfigRegistry`, `EnvironmentConfig` - flexible configuration with builder pattern, multi-source loading, and validation
- **Protocols**: `WorkflowIntegration`, `AsyncWorkflowIntegration`, `ExtendedWorkflowIntegration`
- **Configuration Types**: `CheckConfig`, `ProfileConfig`, `LearnConfig`, `DriftConfig`, `AnomalyConfig`, `StreamConfig`, `RetryConfig`, `CircuitBreakerConfig`, `HealthCheckConfig`, `MetricsConfig`, `TracingConfig`, `RateLimitConfig`
- **Result Types**: `CheckResult`, `ProfileResult`, `LearnResult`, `DriftResult`, `AnomalyResult`, `HealthCheckResult`
- **Drift/Anomaly Types**: `DriftStatus`, `AnomalyStatus`, `DriftMethod`, `ColumnDrift`, `AnomalyScore`
- **Feature Detection**: `supports_drift()`, `supports_anomaly()`, `supports_streaming()` - runtime engine capability checking
- **Serializers**: `AirflowXComSerializer`, `DagsterOutputSerializer`, `PrefectArtifactSerializer`
- **Logging**: `TruthoundLogger`, `LogContext`, `PerformanceLogger`, `SensitiveDataMasker`
- **Retry**: `retry`, `RetryConfig`, `RetryStrategy`, `RetryExecutor`, `LoggingRetryHook`
- **Circuit Breaker**: `circuit_breaker`, `CircuitBreaker`, `CircuitBreakerConfig`, `CircuitState`
- **Health Check**: `health_check`, `HealthCheckConfig`, `HealthStatus`, `CompositeHealthChecker`, `AggregationStrategy`
- **Metrics**: `Counter`, `Gauge`, `Histogram`, `Summary`, `MetricsRegistry`, `timed`, `counted`
- **Tracing**: `Span`, `trace`, `TracingRegistry`, `TraceContext`, `SpanKind`, `SpanStatus`
- **Rate Limiting**: `rate_limit`, `RateLimitConfig`, `RateLimitAlgorithm`, `RateLimiterRegistry`, `TokenBucketRateLimiter`, `SlidingWindowRateLimiter`
- **Caching**: `cached`, `CacheConfig`, `EvictionPolicy`, `CacheBackend`, `LRUCache`, `LFUCache`, `TTLCache`, `CacheRegistry`, `CacheExecutor`
- **Engine Lifecycle**: `ManagedEngine`, `AsyncManagedEngine`, `EngineState`, `EngineConfig`, `EngineLifecycleManager`, `AsyncEngineLifecycleManager`, `EngineHealthChecker`, `AsyncEngineHealthChecker`, `ManagedEngineMixin`, `AsyncManagedEngineMixin`
- **Async Adapters**: `SyncEngineAsyncAdapter`, `SyncToAsyncLifecycleHookAdapter` - wrap sync engines/hooks for async contexts
- **Async Lifecycle Hooks**: `AsyncLifecycleHook`, `AsyncLoggingLifecycleHook`, `AsyncMetricsLifecycleHook`, `AsyncCompositeLifecycleHook`
- **Batch Operations**: `BatchExecutor`, `AsyncBatchExecutor`, `BatchConfig`, `ExecutionStrategy`, `AggregationStrategy`, `ChunkingStrategy` - large dataset processing with chunking and parallel execution
- **Engine Metrics**: `InstrumentedEngine`, `AsyncInstrumentedEngine`, `EngineMetricsHook`, `MetricsEngineHook`, `LoggingEngineHook`, `TracingEngineHook`, `StatsCollectorHook` - automatic metrics collection for engine operations
- **Result Aggregation**: `MultiEngineAggregator`, `AggregationConfig`, `ResultAggregationStrategy`, `CheckResultMergeAggregator`, `CheckResultWeightedAggregator`, `AggregatorRegistry` - multi-engine result combination with strategies
- **Engine Versioning**: `SemanticVersion`, `VersionConstraint`, `VersionRange`, `VersionCompatibilityChecker`, `VersionRegistry`, `parse_version`, `parse_constraint`, `require_version` - SemVer 2.0.0 support with compatibility checking
- **Testing Utilities**: `MockDataQualityEngine`, `MockDriftDetectionEngine`, `MockAnomalyDetectionEngine`, `MockStreamingEngine`, `MockFullEngine`, `AsyncMockDataQualityEngine`, `AsyncMockManagedEngine`, `MockCacheBackend`, `DataQualityTestContext`
- **Prometheus Export**: `PrometheusExporter`, `PrometheusPushGatewayClient`, `PrometheusHttpServer`, `TenantAwarePrometheusExporter`, `AsyncPrometheusExporter`

### Platform Integrations

| Status | Complete |
|--------|----------|

The platform integration layer provides native adapters for major workflow orchestration systems, enabling seamless incorporation of data quality validation into existing pipeline architectures.

| Platform | Package | Description | Status |
|----------|---------|-------------|--------|
| Apache Airflow | `packages/airflow/` | Operators, Sensors, Hooks with SLA integration | Complete |
| Dagster | `packages/dagster/` | Resources, Assets, Ops with native type support | Complete |
| Prefect | `packages/prefect/` | Blocks, Tasks, Flows with async support | Complete |
| dbt | `packages/dbt/` | Generic Tests, Jinja macros, cross-adapter support | Complete |
| Mage AI | `packages/mage/` | Transformer, Sensor, Condition blocks with SLA monitoring | Complete |
| Kestra | `packages/kestra/` | Python scripts, YAML flow generators, output handlers | Complete |

### Enterprise Extensions

| Status | Complete |
|--------|----------|

The enterprise module extends the core framework with production-grade capabilities for large-scale deployments.

| Component | Location | Description | Status |
|-----------|----------|-------------|--------|
| Enterprise Engines | `packages/enterprise/engines/` | Informatica, Talend, IBM InfoSphere, SAP Data Services adapters | Complete |
| Notifications | `packages/enterprise/notifications/` | Multi-channel notification system (Slack, Email, Webhook, PagerDuty, Opsgenie) | Complete |
| Multi-Tenant | `packages/enterprise/multi_tenant/` | Tenant isolation, quota management, context propagation | Complete |
| Secrets | `packages/enterprise/secrets/` | Secret management with multiple backends and audit logging | Complete |

---

## Supported Platforms

### truthound-airflow

| Status | Complete |
|--------|----------|

Apache Airflow Provider package implementing native Operators, Sensors, and Hooks with **pluggable engine support**.

| Component | Description |
|-----------|-------------|
| `DataQualityCheckOperator` | Execute data quality validation with any engine (default: Truthound) |
| `DataQualityProfileOperator` | Perform statistical profiling of datasets |
| `DataQualityLearnOperator` | Automatically infer validation schemas |
| `DataQualityDriftOperator` | Detect data drift between baseline and current datasets |
| `DataQualityAnomalyOperator` | Detect anomalies using ML-based detectors |
| `DataQualitySensor` | Monitor data quality conditions |
| `DataQualityHook` | Manage connections and data source interactions |

**Key Features:**
- Native Airflow Provider architecture with XCom serialization
- SLA monitoring integration with configurable alerting thresholds
- Connection management via Airflow Hooks
- Support for multiple data sources (S3, GCS, BigQuery, Snowflake)

---

### truthound-dagster

| Status | Complete |
|--------|----------|

Dagster integration utilizing ConfigurableResource and Software-Defined Assets with **engine abstraction**.

| Component | Description |
|-----------|-------------|
| `DataQualityResource` | Configurable resource with pluggable engine support |
| `create_quality_check_asset` | Factory function for quality validation assets |
| `data_quality_check_op` | Op implementation for graph-based workflows |
| `data_quality_drift_op` | Drift detection op with dual-input (baseline + current) |
| `data_quality_anomaly_op` | Anomaly detection op with ML detectors |
| `DataQualitySensor` | Event-driven quality monitoring |

**Key Features:**
- Software-Defined Assets with automatic lineage tracking
- Type-safe configuration via Pydantic integration
- Native IOManager support for result persistence
- Freshness policies for data quality SLAs

---

### truthound-prefect

| Status | Complete |
|--------|----------|

Prefect integration providing Blocks, Tasks, and Flow templates with **engine-agnostic design**.

| Component | Description |
|-----------|-------------|
| `DataQualityBlock` | Persistent configuration storage with engine selection |
| `data_quality_check` | Task decorator for validation operations |
| `data_quality_profile` | Task decorator for profiling operations |
| `data_quality_drift_task` | Async drift detection task with table artifact visualization |
| `data_quality_anomaly_task` | Async anomaly detection task with table artifact visualization |
| `validation_flow` | Reusable flow template for quality pipelines |

**Key Features:**
- Native async/await support for concurrent execution
- Block-based configuration with versioned storage
- Artifact generation for validation result visualization
- Integration with Prefect Cloud for centralized monitoring

---

### truthound-dbt

| Status | Complete |
|--------|----------|

dbt package providing Generic Tests, Jinja macros, and Python utilities for SQL-based data quality validation with cross-adapter support.

| Component | Description |
|-----------|-------------|
| `test_truthound_check` | Generic test for declarative rule specification |
| `truthound_check.sql` | Main validation macro |
| `truthound_rules.sql` | Rule-specific SQL generators |
| `truthound_utils.sql` | Cross-adapter utility macros |
| `adapters/` | Database-specific optimizations (Snowflake, BigQuery, Redshift, Databricks, PostgreSQL) |
| Python Package | Adapters, converters, generators, parsers, and hooks |
| Drift SQL Handlers | SQL-based drift detection (mean, stddev, null rate, distinct count, row count) |
| Anomaly SQL Handlers | SQL-based anomaly detection (Z-Score, IQR, range) |

**Supported Databases:**
- PostgreSQL (default)
- Snowflake
- BigQuery
- Redshift
- Databricks

---

### truthound-mage

| Status | Complete |
|--------|----------|

Mage AI integration providing custom block implementations for data quality operations.

| Component | Description |
|-----------|-------------|
| `CheckTransformer` | Transformer block for data quality validation |
| `ProfileTransformer` | Transformer block for statistical profiling |
| `LearnTransformer` | Transformer block for schema inference |
| `DriftTransformer` | Transformer block for drift detection (dual-input) |
| `AnomalyTransformer` | Transformer block for ML anomaly detection |
| `DataQualitySensor` | Sensor block for quality condition monitoring |
| `DataQualityCondition` | Condition block for pipeline branching |
| `SLAMonitor` | SLA monitoring with violation tracking |

**Key Features:**
- Native Mage block architecture with execution context
- SLA monitoring with configurable thresholds and hooks
- Thread-safe consecutive failure tracking
- Builder pattern for immutable configuration

---

### truthound-kestra

| Status | Complete |
|--------|----------|

Kestra integration providing Python script executors and YAML flow generators.

| Component | Description |
|-----------|-------------|
| `check_quality_script` | Script executor for data validation |
| `profile_data_script` | Script executor for statistical profiling |
| `learn_schema_script` | Script executor for schema inference |
| `drift_detection_script` | Script executor for drift detection |
| `anomaly_detection_script` | Script executor for anomaly detection |
| `FlowGenerator` | YAML flow generation from configuration |
| `KestraOutputHandler` | Native Kestra output integration |
| `SLAMonitor` | SLA monitoring with evaluation results |

**Key Features:**
- YAML flow generation for check, profile, learn, and pipeline flows
- Script executors with Kestra-native output handling
- Support for schedule, flow, and webhook triggers
- Retry configuration with exponential backoff

---

## Installation

### Requirements

| Requirement | Version |
|-------------|---------|
| Python | >= 3.11 |
| Truthound (optional, default engine) | >= 1.0.0 |

### Package Installation

This project employs a **single package with optional dependencies** architecture to optimize developer experience:

```bash
# Core package only (includes common module + engine adapters)
pip install truthound-orchestration

# With specific platform integration
pip install truthound-orchestration[airflow]
pip install truthound-orchestration[dagster]
pip install truthound-orchestration[prefect]
pip install truthound-orchestration[mage]
pip install truthound-orchestration[kestra]

# Multiple platforms
pip install truthound-orchestration[airflow,dagster]

# All platforms + development tools
pip install truthound-orchestration[all]
```

#### Engine Installation

Engines are installed separately based on your choice:

```bash
# Truthound (default, recommended)
pip install truthound

# Great Expectations
pip install great-expectations

# Pandera
pip install pandera
```

#### Complete Example

```bash
# Airflow user with Truthound engine
pip install truthound-orchestration[airflow] truthound

# Dagster user with Great Expectations engine
pip install truthound-orchestration[dagster] great-expectations

# Multi-platform with Pandera
pip install truthound-orchestration[airflow,prefect] pandera
```

### Why Single Package?

| Aspect | Single Package (`truthound-orchestration[airflow]`) | Multi Package (`truthound-airflow`) |
|--------|-----------------------------------------------------|-------------------------------------|
| **Developer Experience** | Simple, consistent | Multiple package names to remember |
| **Versioning** | Unified version | Separate version per package |
| **Size** | ~50KB core + platform deps | Same total size |
| **Maintenance** | Single PyPI package | 5 separate packages |

The `common/` module is lightweight (approximately 50KB of pure Python). Platform-specific dependencies (Airflow approximately 200MB, Dagster approximately 150MB) are installed only when the corresponding extra is specified.

### dbt Integration

For dbt integration, add the following to your `packages.yml`:

```yaml
packages:
  - package: truthound/truthound
    version: ">=0.1.0"
```

### Version Compatibility

| Extra | Default Engine | Platform Requirement |
|-------|----------------|----------------------|
| `[airflow]` | Truthound (optional) | Apache Airflow >= 2.6.0 |
| `[dagster]` | Truthound (optional) | Dagster >= 1.5.0 |
| `[prefect]` | Truthound (optional) | Prefect >= 2.14.0 |
| `[opentelemetry]` | - | OpenTelemetry SDK >= 1.20.0 |

**Supported Engines:**
- [Truthound](https://github.com/seadonggyun4/Truthound) (default, recommended)
- [Great Expectations](https://greatexpectations.io/) (via adapter)
- [Pandera](https://pandera.readthedocs.io/) (via adapter)
- Custom engines (implement `DataQualityEngine` Protocol)

---

## Usage Examples

The following examples demonstrate the API for each platform integration with **pluggable engine support**.

### Apache Airflow

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from truthound_airflow import DataQualityCheckOperator

with DAG(
    dag_id="data_quality_pipeline",
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:

    # Using default Truthound engine
    validate_data = DataQualityCheckOperator(
        task_id="validate_user_data",
        rules=[
            {"column": "user_id", "type": "not_null"},
            {"column": "user_id", "type": "unique"},
            {"column": "email", "type": "regex", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
        ],
        data_path="s3://data-lake/users/{{ ds }}/data.parquet",
        fail_on_error=True,
    )

    # Using custom engine
    from my_project import CustomEngine
    validate_with_custom = DataQualityCheckOperator(
        task_id="validate_with_custom",
        engine=CustomEngine(),  # Plug in any DataQualityEngine
        rules=[...],
        data_path="...",
    )
```

### Dagster

```python
from dagster import asset, Definitions
from truthound_dagster import DataQualityResource, create_quality_check_asset
import polars as pl

@asset(group_name="raw")
def raw_users() -> pl.DataFrame:
    return pl.read_parquet("s3://bucket/users.parquet")

validated_users = create_quality_check_asset(
    name="validated_users",
    upstream_asset="raw_users",
    rules=[
        {"column": "user_id", "type": "not_null"},
        {"column": "email", "type": "regex", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
    ],
)

# Default: uses Truthound engine
defs = Definitions(
    assets=[raw_users, validated_users],
    resources={"data_quality": DataQualityResource()},
)

# Custom engine example
from common.engines import GreatExpectationsAdapter
defs = Definitions(
    assets=[raw_users, validated_users],
    resources={"data_quality": DataQualityResource(engine=GreatExpectationsAdapter())},
)
```

### Prefect

```python
from prefect import flow, task
from truthound_prefect import data_quality_check, DataQualityBlock
import polars as pl

@task
def load_data() -> pl.DataFrame:
    return pl.read_parquet("s3://bucket/data.parquet")

@flow(name="quality_validation_pipeline")
async def validation_pipeline():
    data = load_data()

    # Using default Truthound engine
    result = await data_quality_check(
        data=data,
        rules=[
            {"column": "id", "type": "not_null"},
            {"column": "amount", "type": "in_range", "min": 0},
        ],
    )

    # Or using a configured block with custom engine
    block = await DataQualityBlock.load("my-ge-config")
    result = await block.check(data, rules=[...])

    return result
```

### dbt

```yaml
# models/staging/schema.yml
version: 2

models:
  - name: stg_customers
    tests:
      - data_quality_check:
          rules:
            - column: customer_id
              type: not_null
            - column: customer_id
              type: unique
            - column: email
              type: regex
              pattern: "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"
```

### Drift Detection

```python
from common.engines import TruthoundEngine
import polars as pl

engine = TruthoundEngine()

baseline = pl.read_parquet("baseline.parquet")
current = pl.read_parquet("current.parquet")

# Detect drift with auto method selection
result = engine.detect_drift(baseline, current)

if result.is_drifted:
    print(f"Drift detected! {result.drifted_count}/{result.total_columns} columns")
    for col in result.drifted_columns:
        print(f"  {col.column}: {col.method.name} stat={col.statistic:.4f}")

# Specify method and columns
result = engine.detect_drift(
    baseline, current,
    method="ks",
    columns=["revenue", "user_count"],
    threshold=0.05,
)
```

### Anomaly Detection

```python
from common.engines import TruthoundEngine
import polars as pl

engine = TruthoundEngine()
data = pl.read_parquet("data.parquet")

# Detect anomalies with Isolation Forest (default)
result = engine.detect_anomalies(data)

if result.has_anomalies:
    print(f"Anomalies found! rate={result.anomaly_rate:.2%}")
    for score in result.anomalies:
        print(f"  {score.column}: score={score.score:.4f}, anomaly={score.is_anomaly}")

# Use Z-Score detector on specific columns
result = engine.detect_anomalies(
    data,
    detector="z_score",
    columns=["transaction_amount", "login_count"],
    contamination=0.03,
)
```

### Streaming Validation

```python
from common.engines import TruthoundEngine

engine = TruthoundEngine()

def data_stream():
    for chunk in read_large_file("data.csv", chunk_size=10000):
        yield chunk

# Memory-efficient batch-by-batch validation
for batch_result in engine.check_stream(data_stream(), batch_size=5000):
    print(f"Batch: {batch_result.status.name}")
    if batch_result.status.name == "FAILED":
        break  # fail-fast
```

---

## Development

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/seadonggyun4/truthound-orchestration.git
cd truthound-orchestration

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync --all-extras

# Install pre-commit hooks
pre-commit install
```

### Code Quality

| Tool | Purpose | Configuration |
|------|---------|---------------|
| Ruff | Linting and formatting | `ruff.toml` |
| MyPy | Static type checking | `mypy.ini` |
| pytest | Testing framework | `pyproject.toml` |
| pre-commit | Git hooks | `.pre-commit-config.yaml` |

### Commands

```bash
# Run linter
ruff check .

# Run type checker
mypy common/

# Run tests
pytest

# Run all pre-commit checks
pre-commit run --all-files
```

### Repository Structure

```
truthound-orchestration/
├── common/                     # Shared module (Complete)
│   ├── base.py                 # Protocols (DataQualityEngine), Config, Result types
│   ├── config.py               # Environment/file configuration
│   ├── exceptions.py           # Exception hierarchy
│   ├── logging.py              # Structured logging, masking
│   ├── retry.py                # Retry decorator, backoff strategies
│   ├── circuit_breaker.py      # Circuit breaker pattern
│   ├── health.py               # Health check system
│   ├── metrics.py              # Metrics and distributed tracing
│   ├── rate_limiter.py         # Rate limiting algorithms
│   ├── cache.py                # Caching infrastructure
│   ├── serializers.py          # Platform serialization
│   ├── testing.py              # Mock objects, fixtures
│   ├── exporters/              # Metric exporters
│   │   └── prometheus.py       # Prometheus export (Push Gateway, HTTP Server)
│   └── engines/                # Engine implementations
│       ├── base.py             # DataQualityEngine Protocol
│       ├── batch.py            # Batch operations (BatchExecutor, chunking)
│       ├── chain.py            # Engine chain/fallback (EngineChain, strategies)
│       ├── config.py           # Engine configuration system (Builder, Loader)
│       ├── context.py          # Context managers (EngineContext, EngineSession)
│       ├── lifecycle.py        # Lifecycle management (ManagedEngine, EngineState)
│       ├── metrics.py          # Engine metrics (InstrumentedEngine, hooks)
│       ├── aggregation.py      # Result aggregation (MultiEngineAggregator)
│       ├── version.py          # Semantic versioning, compatibility checking
│       ├── plugin.py           # Plugin discovery (Entry Point based)
│       ├── registry.py         # Engine registry
│       ├── truthound.py        # Truthound engine (default)
│       ├── great_expectations.py  # Great Expectations adapter
│       └── pandera.py          # Pandera adapter
├── packages/                   # Platform integrations (Complete)
│   ├── airflow/                # Airflow Operators, Sensors, Hooks, SLA (Complete)
│   │   ├── operators/          # DataQualityCheckOperator, ProfileOperator
│   │   ├── sensors/            # DataQualitySensor
│   │   ├── hooks/              # DataQualityHook
│   │   └── sla/                # SLA monitoring integration
│   ├── dagster/                # Dagster Resources, Assets, Ops (Complete)
│   │   ├── resources/          # DataQualityResource
│   │   ├── assets/             # Quality check asset factories
│   │   ├── ops/                # data_quality_check_op
│   │   └── sensors/            # DataQualitySensor
│   ├── prefect/                # Prefect Blocks, Tasks, Flows (Complete)
│   │   ├── blocks/             # DataQualityBlock
│   │   ├── tasks/              # data_quality_check, data_quality_profile
│   │   └── flows/              # validation_flow templates
│   ├── dbt/                    # dbt Tests (Complete)
│   │   ├── src/truthound_dbt/  # Python package
│   │   │   ├── adapters/       # Database adapters (Postgres, Snowflake, BigQuery, Redshift, Databricks)
│   │   │   ├── converters/     # Rule converters
│   │   │   ├── generators/     # SQL, schema, and test generators
│   │   │   ├── parsers/        # Manifest and results parsers
│   │   │   └── hooks/          # dbt hook system
│   │   ├── macros/             # SQL macros (truthound_check, truthound_rules, truthound_utils)
│   │   ├── tests/generic/      # Generic test implementations
│   │   └── integration_tests/  # Integration test suite
│   ├── mage/                   # Mage AI Blocks (Complete)
│   │   ├── blocks/             # Transformer, Sensor, Condition blocks
│   │   ├── io/                 # IO configuration
│   │   ├── sla/                # SLA monitoring
│   │   └── utils/              # Utilities and exceptions
│   ├── kestra/                 # Kestra Scripts and Flows (Complete)
│   │   ├── scripts/            # Python script executors
│   │   ├── flows/              # YAML flow generators
│   │   ├── outputs/            # Output handlers
│   │   ├── sla/                # SLA monitoring
│   │   └── utils/              # Utilities and exceptions
│   └── enterprise/             # Enterprise extensions (Complete)
│       ├── engines/            # Enterprise engine adapters
│       │   ├── base.py         # EnterpriseEngineAdapter, protocols
│       │   ├── informatica.py  # Informatica Data Quality adapter
│       │   ├── talend.py       # Talend Data Quality adapter
│       │   ├── ibm_infosphere.py  # IBM InfoSphere adapter
│       │   ├── sap_data_services.py  # SAP Data Services adapter
│       │   └── registry.py     # Enterprise engine registry
│       ├── notifications/      # Multi-channel notification system
│       │   ├── types.py        # NotificationChannel, NotificationLevel
│       │   ├── handlers/       # Slack, Email, Webhook, PagerDuty, Opsgenie
│       │   ├── formatters/     # Message formatting (Markdown, HTML, Plain)
│       │   ├── routing.py      # NotificationRouter, routing rules
│       │   └── registry.py     # NotificationRegistry
│       ├── multi_tenant/       # Multi-tenant support
│       │   ├── types.py        # TenantStatus, IsolationLevel, QuotaType
│       │   ├── context.py      # TenantContext, context propagation
│       │   ├── registry.py     # TenantRegistry
│       │   ├── isolation.py    # IsolationEnforcer, TenantIsolator
│       │   ├── storage/        # TenantStorage backends
│       │   └── middleware.py   # TenantMiddleware for web frameworks
│       └── secrets/            # Secret management
│           ├── base.py         # SecretProvider protocol, types
│           ├── config.py       # Configuration classes
│           ├── registry.py     # Provider registry
│           ├── cache.py        # Secret caching (TTL, tiered)
│           ├── encryption.py   # Client-side encryption
│           ├── rotation.py     # Automatic secret rotation
│           ├── hooks.py        # Audit logging hooks
│           └── backends/       # Storage backends (Vault, AWS, GCP, Azure)
├── docs/                       # User documentation
│   ├── index.md                # Documentation index
│   ├── getting-started.md      # Installation and quick start
│   ├── common/                 # Common module guides
│   ├── engines/                # Engine documentation
│   ├── airflow/                # Airflow integration guide
│   ├── dagster/                # Dagster integration guide
│   ├── prefect/                # Prefect integration guide
│   ├── dbt/                    # dbt integration guide
│   ├── mage/                   # Mage AI integration guide
│   ├── kestra/                 # Kestra integration guide
│   ├── enterprise/             # Enterprise features guide
│   └── api-reference/          # API reference documentation
└── tests/                      # Test suites
    ├── common/                 # Common module tests
    ├── dbt/                    # dbt module tests
    ├── mage/                   # Mage module tests
    ├── kestra/                 # Kestra module tests
    └── enterprise/             # Enterprise module tests
```

---

## Contributing

Contributions are welcome.

This project follows [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:

```
<type>(<scope>): <description>

feat(airflow): add TruthoundSensor for quality monitoring
fix(dagster): resolve resource initialization error
docs(common): update configuration examples
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Common Module Documentation

The `common/` module provides foundational components shared across all workflow orchestration integrations.

### Basic Usage

```python
from common.engines import TruthoundEngine

# Context manager usage (recommended)
with TruthoundEngine() as engine:
    result = engine.check(data, auto_schema=True)
    print(f"Status: {result.status.name}")
    print(f"Passed: {result.passed_count}, Failed: {result.failed_count}")
```

### Using Different Engines

```python
from common.engines import get_engine

# Get engine by name
engine = get_engine("truthound")  # Default
engine = get_engine("great_expectations")
engine = get_engine("pandera")
```

### Feature Detection

```python
from common.engines.base import supports_drift, supports_anomaly, supports_streaming

engine = get_engine("truthound")
print(supports_drift(engine))      # True
print(supports_anomaly(engine))    # True
print(supports_streaming(engine))  # True

ge = get_engine("great_expectations")
print(supports_drift(ge))          # False
```

---

## Enterprise Features

The enterprise module (`packages/enterprise/`) provides production-grade capabilities designed for large-scale, multi-tenant deployments requiring integration with commercial data quality platforms and sophisticated operational workflows.

### Enterprise Engine Adapters

The framework provides adapters for leading commercial data quality platforms, enabling organizations to leverage existing investments while benefiting from the unified orchestration layer.

| Engine | Module | Description |
|--------|--------|-------------|
| Informatica Data Quality | `informatica.py` | IDQ integration with scorecard support |
| Talend Data Quality | `talend.py` | TMC integration with profiling capabilities |
| IBM InfoSphere Information Analyzer | `ibm_infosphere.py` | Analysis and rule management |
| SAP Data Services | `sap_data_services.py` | Address cleansing and validation |

```python
from packages.enterprise.engines import (
    get_enterprise_engine,
    create_informatica_adapter,
    create_ibm_infosphere_adapter,
)

# Retrieve engine from registry
engine = get_enterprise_engine("informatica")

# Create with explicit configuration
adapter = create_informatica_adapter(
    api_endpoint="https://idq.example.com/api/v2",
    api_key="your-api-key",
    domain="Production",
)

# Usage follows standard DataQualityEngine protocol
with adapter:
    result = adapter.check(data, rules)
```

### Multi-Channel Notification System

The notification subsystem enables automated alerting across multiple communication channels with configurable routing, formatting, and retry logic.

| Channel | Handler | Description |
|---------|---------|-------------|
| Slack | `SlackNotificationHandler` | Webhook-based Slack integration |
| Email | `EmailNotificationHandler` | SMTP-based email delivery |
| Webhook | `WebhookNotificationHandler` | Generic HTTP endpoint integration |
| PagerDuty | `PagerDutyNotificationHandler` | Incident management integration |
| Opsgenie | `OpsgenieNotificationHandler` | Alert management integration |

```python
from packages.enterprise.notifications import (
    NotificationManager,
    NotificationPayload,
    NotificationLevel,
    create_slack_handler,
)

# Configure notification handler
slack = create_slack_handler(
    webhook_url="https://hooks.slack.com/services/...",
    default_channel="#data-quality-alerts",
)

# Create notification manager
manager = NotificationManager(handlers=[slack])

# Send notification
payload = NotificationPayload(
    message="Data quality check failed: 15 records with null values",
    level=NotificationLevel.ERROR,
    title="Validation Failure Alert",
)
result = await manager.notify(payload)
```

### Multi-Tenant Architecture

The multi-tenant module provides comprehensive isolation, quota management, and context propagation for organizations serving multiple tenants from a shared infrastructure.

| Component | Description |
|-----------|-------------|
| `TenantContext` | Thread-local tenant context propagation |
| `TenantRegistry` | Tenant lifecycle and metadata management |
| `IsolationEnforcer` | Resource access control and isolation |
| `TenantStorage` | Backend abstraction (Memory, Redis, Database) |
| `TenantMiddleware` | Web framework integration (ASGI/WSGI) |

```python
from packages.enterprise.multi_tenant import (
    TenantContext,
    TenantRegistry,
    IsolationLevel,
    create_memory_storage,
)

# Initialize tenant registry
storage = create_memory_storage()
registry = TenantRegistry(storage=storage)

# Register tenant
await registry.register_tenant(
    tenant_id="acme-corp",
    name="ACME Corporation",
    tier=TenantTier.ENTERPRISE,
    isolation_level=IsolationLevel.DEDICATED,
)

# Set tenant context for current execution
with TenantContext(tenant_id="acme-corp"):
    # All operations within this context are tenant-scoped
    result = engine.check(data, rules)
```

**Isolation Levels:**

| Level | Description | Use Case |
|-------|-------------|----------|
| `SHARED` | Resources shared between tenants | Cost-optimized multi-tenant |
| `LOGICAL` | Logical separation with shared infrastructure | Standard multi-tenant |
| `PHYSICAL` | Physical separation (e.g., separate databases) | Compliance requirements |
| `DEDICATED` | Fully dedicated resources per tenant | Enterprise isolation |

### Secret Management

The secrets module provides a unified interface for secret storage and retrieval across multiple backend systems, with support for caching, encryption, rotation, and audit logging.

| Backend | Description |
|---------|-------------|
| HashiCorp Vault | KV v1/v2 secret engine integration |
| AWS Secrets Manager | AWS-native secret storage |
| GCP Secret Manager | Google Cloud secret storage |
| Azure Key Vault | Azure-native secret storage |
| Environment Variables | Environment-based secret injection |
| Encrypted Files | Local encrypted file storage |

```python
from packages.enterprise.secrets import (
    get_secret_registry,
    get_secret,
    set_secret,
)
from packages.enterprise.secrets.backends import InMemorySecretProvider

# Initialize registry and register provider
registry = get_secret_registry()
registry.register("memory", InMemorySecretProvider())

# Store and retrieve secrets
set_secret("database/password", "secret-value")
secret = get_secret("database/password")
```

**Security Features:**

| Feature | Description |
|---------|-------------|
| Client-side Encryption | Fernet, AES-GCM, ChaCha20-Poly1305 algorithms |
| Secret Caching | TTL-based caching with tiered cache support |
| Automatic Rotation | Configurable rotation schedules with generators |
| Audit Logging | Comprehensive audit hooks (values never logged) |
| Multi-Tenant Isolation | Tenant-scoped secret namespacing |

### Observability

The framework provides Prometheus-compatible metrics export for monitoring and alerting integration.

```python
from common.exporters import (
    create_prometheus_exporter,
    create_pushgateway_exporter,
    create_prometheus_http_server,
)

# Create Prometheus exporter
exporter = create_prometheus_exporter(
    namespace="truthound",
    job_name="data_quality",
)

# Export metrics to Push Gateway
pushgateway = create_pushgateway_exporter(
    gateway_url="http://pushgateway:9091",
    job_name="batch_job",
)

# Expose HTTP endpoint for scraping
server = create_prometheus_http_server(port=9090)
server.start()
```

**Export Capabilities:**

| Feature | Description |
|---------|-------------|
| Push Gateway | Batch job metrics via HTTP POST |
| HTTP Server | Scrape endpoint exposure |
| Multi-Tenant | Tenant-aware metric isolation |
| Async Support | Non-blocking export operations |

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Truthound](https://github.com/seadonggyun4/Truthound) | Core data quality validation framework |
| [Apache Airflow](https://airflow.apache.org/) | Workflow orchestration platform |
| [Dagster](https://dagster.io/) | Data orchestration platform |
| [Prefect](https://www.prefect.io/) | Workflow automation platform |
| [dbt](https://www.getdbt.com/) | Data transformation tool |

---

## Support

- **Issues**: [GitHub Issues](https://github.com/seadonggyun4/truthound-orchestration/issues)
- **Documentation**: [Truthound Docs](https://truthound.dev/docs)
