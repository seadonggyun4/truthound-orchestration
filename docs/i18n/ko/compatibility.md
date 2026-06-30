!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Compatibility
---

# 호환성

This page defines the official compatibility contract for `truthound-오케스트레이션 3.x`: what Truthound line it supports, which host versions are exercised in CI, and how to interpret "minimum supported" versus "primary supported" host lanes.

## Version Line

`truthound-오케스트레이션 3.x` supports `Truthound 3.x` only.

| Package Line | Supported Truthound Range | Status |
|--------------|---------------------------|--------|
| `truthound-오케스트레이션 3.x` | `>=3.0,<4.0` | Active first-party line |
| `truthound-오케스트레이션 1.x` | Pre-Truthound-3 lines | Legacy |

## How To Read This Contract

- "minimum supported" means a release-blocking host-plus-Python tuple still exercised in CI
- "primary supported" means the main documentation and default examples track that tuple
- compatibility is defined at the surface level, not only by package version names
- convenience aggregates such as `truthound-오케스트레이션[all]` are useful for canaries but are not the primary release contract

## Platform Expectations

| Platform | Runtime Expectation |
|----------|---------------------|
| Airflow | `truthound-오케스트레이션[airflow]` with `Truthound 3.x` |
| Dagster | `truthound-오케스트레이션[dagster]` with `Truthound 3.x` |
| Prefect | `truthound-오케스트레이션[prefect]` with `Truthound 3.x` |
| Mage | `truthound-오케스트레이션[mage]` inside a Mage runtime that already provides `mage-ai` |
| Kestra | `truthound-오케스트레이션[kestra]` with `Truthound 3.x` |
| dbt | `truthound-오케스트레이션[dbt]` with the bundled first-party `truthound_dbt` package |

Release-blocking compatibility and security guarantees apply to these per-surface installs. `truthound-오케스트레이션[all]` remains available as a convenience aggregate and nightly canary surface.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
## Generated CI Support Matrix

Truthound release line: `>=3.0,<4.0`

| Lane | Airflow | Prefect | Dagster | Mage | Kestra | dbt Compile | dbt Execute |
|------|---------|---------|---------|------|--------|-------------|-------------|
| PR | `3.2.0` on `Python 3.12` | `3.6.29` on `Python 3.12` | `1.12.18` on `Python 3.12` | Primary host smoke | Primary host smoke | `postgres` | No |
| Main | `3.2.0` on `Python 3.12` | `3.6.29` on `Python 3.12` | `1.12.18` on `Python 3.12` | Primary host smoke | Primary host smoke | `postgres` | No |
| Release | `2.6.0` on `Python 3.11`, `3.2.0` on `Python 3.12` | `2.14.0` on `Python 3.11`, `3.6.29` on `Python 3.12` | `1.5.0` on `Python 3.11`, `1.12.18` on `Python 3.12` | Primary host smoke | Primary host smoke | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |
| Nightly | `3.2.0` on `Python 3.12` | `3.6.29` on `Python 3.12` | `1.12.18` on `Python 3.12` | Primary host smoke + `mage-ai` runtime canary | Primary host smoke | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |

## Supported Host Version Anchors

| Platform | Minimum Supported | Primary Supported |
|----------|-------------------|-------------------|
| Airflow | `2.6.0` on `Python 3.11` | `3.2.0` on `Python 3.12` |
| Prefect | `2.14.0` on `Python 3.11` | `3.6.29` on `Python 3.12` |
| Dagster | `1.5.0` on `Python 3.11` | `1.12.18` on `Python 3.12` |
| Kestra | `1.3.0` on `Python 3.12` | `1.3.0` on `Python 3.12` |
| dbt | `dbt-core>=1.10.0,<1.11.0` + `dbt-postgres>=1.10.0,<1.11.0` | `dbt-core 1.10.20` + `dbt-postgres 1.10.0` |

## Security Audit Surfaces

| Surface | Install Surface | Release Blocking |
|---------|-----------------|------------------|
| `base` | `truthound-오케스트레이션` | Yes |
| `prefect` | `truthound-오케스트레이션[prefect]` | Yes |
| `dagster` | `truthound-오케스트레이션[dagster]` | Yes |
| `dbt` | `truthound-오케스트레이션[dbt]` | Yes |
| `kestra` | `truthound-오케스트레이션[kestra]` | Yes |
| `opentelemetry` | `truthound-오케스트레이션[opentelemetry]` | Yes |
| `airflow` | `truthound-오케스트레이션[airflow]` | Push/release advisory only |
| `all` | `truthound-오케스트레이션[all]` | Nightly canary only |

First-party release guarantees apply to per-surface installs. `truthound-오케스트레이션[all]` remains available as a convenience aggregate and nightly canary surface.
Minimum compatibility guarantees are validated as tested host-plus-Python tuples, not as host versions on every Python runtime.
Airflow security audits install with the official Airflow constraints file for the pinned version and Python version.
<!-- END GENERATED SUPPORT MATRIX -->

## What This Means In Practice

Before rollout:

- pin the host package to one of the documented tuples
- test the same tuple in your deployment environment
- use the host-specific install guidance in each platform section rather than assuming the latest upstream release is supported

When upgrading:

- upgrade the host runtime and Python version as a tested pair
- keep dbt adapter versions aligned with the supported core line
- treat minimum lanes as compatibility anchors, not as a suggestion that every newer transitive dependency is safe

## Supported Scope

This release line is intentionally scoped to:

- Truthound 3.x runtime/result contracts
- first-party platform packages in this repository
- advanced-tier Great Expectations and Pandera adapters

It does not promise:

- backward runtime compatibility with Truthound 1.x or 2.x
- equal docs prominence for advanced engines
- hidden auto-migration of old workspaces or result formats

## 호환성 APIs

Use the shared runtime helpers when you need explicit 검증 inside platform code or deployment checks:

```python
from common.engines import (
    EngineCreationRequest,
    normalize_runtime_context,
    resolve_data_source,
    run_preflight,
)

runtime_context = normalize_runtime_context(platform="prefect")
resolved_source = resolve_data_source("users.parquet")

report = run_preflight(
    EngineCreationRequest(
        engine_name="truthound",
        runtime_context=runtime_context.with_source(resolved_source),
    )
)
```

`run_preflight(...)` checks engine resolution, serializer readiness, source normalization, and whether the normalized source needs a connection or profile.

## Related Reading

- [Getting Started](getting-started.md)
- [Choose a Platform](choose-a-platform.md)
- [Shared Runtime: Preflight and Compatibility](common/preflight-compatibility.md)
- platform-specific install and compatibility pages
