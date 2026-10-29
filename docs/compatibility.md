---
title: Compatibility
---

# Compatibility

This page defines the official compatibility contract for `truthound-orchestration 3.x`: what Truthound line it supports, which host versions are exercised in CI, and how to interpret "minimum supported" versus "primary supported" host lanes.

## Version Line

`truthound-orchestration 3.x` supports `Truthound 3.x` only.

| Package Line | Supported Truthound Range | Status |
|--------------|---------------------------|--------|
| `truthound-orchestration 3.x` | `>=3.0,<4.0` | Active first-party line |
| `truthound-orchestration 1.x` | Pre-Truthound-3 lines | Legacy |

## How To Read This Contract

- "minimum supported" means a release-blocking host-plus-Python tuple still exercised in CI
- "primary supported" means the main documentation and default examples track that tuple
- compatibility is defined at the surface level, not only by package version names
- convenience aggregates such as `truthound-orchestration[all]` are useful for canaries but are not the primary release contract

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

## Core 3.3.0 verification

The source revision containing these fixes was verified with the published
`truthound==3.3.0` wheel on Python 3.11.14 / macOS ARM64. The project remains
Alpha. These results apply to wheels built from this revision; they do not
retroactively certify the previously published orchestration 3.0.3 package.
No new package publication is implied by this source update.

The adapter preserves failed checks whose detailed issues were omitted, including
Core and orchestration serialization round trips. It uses the Core drift namespace
and nested drift verdicts, and maps native ML row counts and row indices. Prefect
fields follow the Pydantic generation of the installed host. Kestra explicitly
converts common failure names and severity levels into its output contract.

The same change fixes synchronous health checks inside async aggregation and
slotted Kestra configuration serialization. Async tests now own their event loops. Runtime metadata serialization also avoids
recursing through dynamically fabricated conversion methods.

| Surface | Version / execution verified |
| --- | --- |
| Core | Published 3.3.0 wheel; check, profile, learn, failure round trip |
| Airflow | 3.2.0; real operator with local CSV input and captured XCom output |
| Dagster | 1.12.18; resource inside an in-process job |
| Prefect | 3.6.29; native Block with real Core execution and a fresh-process regression |
| Mage | Packaged transformer execution with real Core; Mage server not started |
| Kestra | Python SDK 1.3.0 and script executor; Kestra server not started |
| dbt | Core 1.10.20 / postgres adapter 1.10.0; compile, seed, run, test and macro execution on isolated PostgreSQL 14.20 |
| Advanced Core | Real KS drift and Isolation Forest, Z-score and Ensemble anomaly detection with NumPy/SciPy/scikit-learn installed |

The dbt data suite produced 12 passes, 7 expected warnings for invalid seed data,
zero errors and zero skips. Manifest coverage was 100% against the existing 80%
gate. This validates PostgreSQL execution, not live Snowflake, BigQuery, Redshift
or Databricks connections. Host minimum-version lanes, other Python versions,
and complete scheduler/server deployments are outside this local verification.
Core 3.3 does not export the adapter's legacy `lof` detector name; requesting it
raises an explicit unsupported-class error. The three ML detectors above are the
verified choices. Optional ML dependencies are required for that advanced lane.

Reproduce installed-package checks in separate environments with the selected
host version and the wheel built from this revision:

```bash
python -m build
python -m pip install "truthound==3.3.0" dist/truthound_orchestration-3.0.3-py3-none-any.whl
python -I scripts/ci/verify_core_runtime.py base
# After installing NumPy, SciPy and scikit-learn:
python -I scripts/ci/verify_core_runtime.py base --advanced
# Use an environment containing the respective host dependency:
python -I scripts/ci/verify_core_runtime.py prefect
```

Other runner labels are `airflow`, `dagster`, `mage`, and `kestra`. The runner
asserts that both packages load from site-packages rather than the source tree.
For dbt, use `packages/dbt/scripts/run_first_party_suite.py --mode all --target
postgres` against an isolated test database, then run the manifest coverage gate.
Existing optional-test skips are not evidence that those optional engines passed.

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
| `base` | `truthound-orchestration` | Yes |
| `prefect` | `truthound-orchestration[prefect]` | Yes |
| `dagster` | `truthound-orchestration[dagster]` | Yes |
| `dbt` | `truthound-orchestration[dbt]` | Yes |
| `kestra` | `truthound-orchestration[kestra]` | Yes |
| `opentelemetry` | `truthound-orchestration[opentelemetry]` | Yes |
| `airflow` | `truthound-orchestration[airflow]` | Push/release advisory only |
| `all` | `truthound-orchestration[all]` | Nightly canary only |

First-party release guarantees apply to per-surface installs. `truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.
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

## Compatibility APIs

Use the shared runtime helpers when you need explicit validation inside platform code or deployment checks:

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
