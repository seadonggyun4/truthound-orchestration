---
title: Compatibility
---

# Compatibility

## Version Line

`truthound-orchestration 3.x` supports `Truthound 3.x` only.

| Package Line | Supported Truthound Range | Status |
|--------------|---------------------------|--------|
| `truthound-orchestration 3.x` | `>=3.0,<4.0` | Active first-party line |
| `truthound-orchestration 1.x` | Pre-Truthound-3 lines | Legacy |

## Platform Expectations

| Platform | Runtime Expectation |
|----------|---------------------|
| Airflow | `truthound-orchestration[airflow]` with `Truthound 3.x` |
| Dagster | `truthound-orchestration[dagster]` with `Truthound 3.x` |
| Prefect | `truthound-orchestration[prefect]` with `Truthound 3.x` |
| Mage | `truthound-orchestration[mage]` inside a Mage runtime that already provides `mage-ai` |
| Kestra | `truthound-orchestration[kestra]` with `Truthound 3.x` |
| dbt | `truthound-orchestration[dbt]` with the bundled first-party `truthound_dbt` package |

Release-blocking compatibility and security guarantees apply to these per-surface installs. `truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
## Generated CI Support Matrix

Truthound release line: `>=3.0,<4.0`

| Lane | Python | Airflow | Prefect | Dagster | Mage / Kestra | dbt Compile | dbt Execute |
|------|--------|---------|---------|---------|----------------|-------------|-------------|
| PR | `3.12` | `3.1.8` | `3.6.22` | `1.12.18` | Primary host smoke | `postgres` | No |
| Main | `3.12` | `2.6.0`, `3.1.8` | `2.14.0`, `3.6.22` | `1.5.0`, `1.12.18` | Primary host smoke | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |
| Release | `3.12` | `2.6.0`, `3.1.8` | `2.14.0`, `3.6.22` | `1.5.0`, `1.12.18` | Primary host smoke | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |
| Nightly | `3.11`, `3.12`, `3.13` | `3.1.8` | `3.6.22` | `1.12.18` | Primary host smoke + advanced-tier canary | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |

## Supported Host Version Anchors

| Platform | Minimum Supported | Primary Supported |
|----------|-------------------|-------------------|
| Airflow | `2.6.0` | `3.1.8` |
| Prefect | `2.14.0` | `3.6.22` |
| Dagster | `1.5.0` | `1.12.18` |
| dbt | `dbt-core>=1.8.0,<2.0.0` + `dbt-postgres>=1.8.0,<2.0.0` | `dbt-core 1.9.1` + `dbt-postgres 1.9.1` |

## Security Audit Surfaces

| Surface | Install Surface | Release Blocking |
|---------|-----------------|------------------|
| `base` | `truthound-orchestration` | Yes |
| `airflow` | `truthound-orchestration[airflow]` | Yes |
| `prefect` | `truthound-orchestration[prefect]` | Yes |
| `dagster` | `truthound-orchestration[dagster]` | Yes |
| `dbt` | `truthound-orchestration[dbt]` | Yes |
| `kestra` | `truthound-orchestration[kestra]` | Yes |
| `opentelemetry` | `truthound-orchestration[opentelemetry]` | Yes |
| `all` | `truthound-orchestration[all]` | Nightly canary only |

First-party release guarantees apply to per-surface installs. `truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.
<!-- END GENERATED SUPPORT MATRIX -->

## Supported Scope

This release line is intentionally scoped to:

- Truthound 3.x runtime/result contracts
- first-party platform packages in this repository
- advanced-tier Great Expectations and Pandera adapters

It does not promise:

- backward runtime compatibility with Truthound 1.x or 2.x
- equal docs prominence for advanced engines
- hidden auto-migration of old workspaces or result formats

## Compatibility APIs

Use the shared runtime helpers when you need explicit validation:

```python
from common.engines import EngineCreationRequest, normalize_runtime_context, run_preflight

report = run_preflight(
    EngineCreationRequest(
        engine_name="truthound",
        runtime_context=normalize_runtime_context(platform="prefect"),
    )
)
```

`run_preflight(...)` checks engine resolution, serializer readiness, and whether the normalized source needs a connection or profile.
