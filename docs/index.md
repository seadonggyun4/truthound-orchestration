---
title: Truthound Orchestration
---

# Truthound Orchestration

A universal data quality integration framework that provides adapter interfaces between workflow orchestration platforms and various data quality engines.

## Overview

Truthound Orchestration is designed to facilitate seamless integration of data quality validation within data pipelines. The framework abstracts the complexity of different data quality engines and orchestration platforms through a unified protocol-based architecture.

### Core Design Principles

- **Protocol-Based Abstraction**: The `DataQualityEngine` Protocol provides a unified interface for interacting with diverse data quality engines
- **Platform Independence**: Native support for major orchestration platforms including Airflow, Dagster, Prefect, and dbt
- **Engine Agnosticism**: Supports multiple data quality engines with Truthound as the default implementation

### Supported Platforms

| Platform | Status | Primary Components |
|----------|--------|-------------------|
| Apache Airflow | Implemented | Operators, Sensors, Hooks, SLA Monitoring |
| Dagster | Implemented | Resources, Ops, Assets, SLA Monitoring |
| Prefect | Implemented | Blocks, Tasks, Flows, SLA Monitoring |
| dbt | Implemented | SQL Macros, Generic Tests, Python Package, Cross-Adapter Support |

### Supported Data Quality Engines

| Engine | Status | Characteristics |
|--------|--------|-----------------|
| Truthound | Default Engine | Schema-based validation, automatic learning |
| Great Expectations | Adapter | Expectation-based validation |
| Pandera | Adapter | Type-safe schema validation |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Workflow Orchestration                      │
│     (Airflow / Dagster / Prefect / dbt)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 truthound-orchestration                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   common module                         │ │
│  │  (logging, retry, circuit_breaker, metrics, cache...)  │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              DataQualityEngine Protocol                 │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  Data Quality Engines                        │
│     (Truthound / Great Expectations / Pandera)              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Common Module

- **Logging**: Structured logging with sensitive data masking and platform-specific adapters
- **Retry**: Multiple backoff strategies including exponential, linear, and Fibonacci
- **Circuit Breaker**: Failure threshold-based request blocking for fault tolerance
- **Health Check**: Component health monitoring with aggregation strategies
- **Metrics**: Counter, Gauge, Histogram, and Summary metric types
- **Distributed Tracing**: W3C Trace Context specification support
- **Rate Limiting**: Token Bucket, Sliding Window, Fixed Window, and Leaky Bucket algorithms
- **Caching**: LRU, LFU, and TTL-based cache implementations

### Engine Management

- **Lifecycle Management**: Engine initialization, health checking, and graceful shutdown
- **Batch Processing**: Data chunking and parallel execution for large datasets
- **Engine Chain**: Fallback patterns, load balancing, and conditional routing
- **Context Manager**: Resource tracking with automatic cleanup
- **Result Aggregation**: Multi-engine result merging and comparison
- **Version Management**: SemVer 2.0.0 compliant version compatibility checking
- **Plugin System**: Entry point-based engine discovery mechanism

### Enterprise Features

- **Multi-Tenancy**: Tenant context management, isolation strategies, and storage backends
- **Secret Management**: Integration with Vault, AWS Secrets Manager, GCP Secret Manager, and Azure Key Vault
- **Notifications**: Multi-channel alerting via Slack, Email, Webhook, PagerDuty, and Opsgenie

## Navigation

- [Getting Started](getting-started.md) - Installation and quick start guide
- [Common Module](common/index.md) - Shared utilities and infrastructure
- [Engines](engines/index.md) - Data quality engine implementations
- [Airflow Integration](airflow/index.md) - Apache Airflow usage guide
- [Dagster Integration](dagster/index.md) - Dagster usage guide
- [Prefect Integration](prefect/index.md) - Prefect usage guide
- [dbt Integration](dbt/index.md) - dbt usage guide
- [Enterprise Features](enterprise/index.md) - Multi-tenancy, secrets, and notifications
- [API Reference](api-reference/index.md) - Complete API documentation
