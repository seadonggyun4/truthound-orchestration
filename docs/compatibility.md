---
title: Compatibility
---

# Compatibility

This page defines the official compatibility contract for `truthound-orchestration 3.x`: what Truthound line it supports, which host versions are exercised in CI, and how to interpret "minimum supported" versus "primary supported" host lanes.

## Scheduler security baseline (2026-10-29)

The companion now declares NumPy as a runtime dependency: Core 3.3 imports it
for default parallel validation. A fresh Mage companion install reproduced the
missing-dependency failure; a real Core regression covers good and bad data
without relying on NumPy installed transitively by another host.

| Platform | Minimum / primary | Verification and limitations |
|---|---|---|
| Prefect | 3.9.0 / 3.9.0 | Python 3.11 / 3.12; API server, Runner and persisted COMPLETED / FAILED states with Core 3.3.0 |
| Dagster | 1.13.1 / 1.13.27 | Python 3.11 / 3.12; webserver, daemon, GraphQL submission and SUCCESS / FAILURE runs |
| Kestra Python SDK | 1.3.0 / 2.0.0 | Python 3.11 / 3.12; both execute on the separately versioned Java server 2.0.8 |
| Kestra Java server | 2.0.8 | Authenticated native execution, quality failures and literal-IP / DNS deny-list regression |
| Mage host | No security-supported released version | 0.9.80 is functionally verified only, with a separate Core 3.3 interpreter |

Older Prefect/Dagster tuples recorded below are historical functional evidence,
not current supported installations. Root extras, the aggregate extra and the
standalone Prefect/Dagster package metadata enforce the new host floors.
Prefect 3.6.29 passes a fresh dependency audit but fails server initialization
with current FastAPI (Body(min_items=...) is no longer accepted); 3.9.0 passes
the existing fresh-process regression and real server execution without legacy
FastAPI/Starlette constraints. Both current Python lanes are audited after normal
dependency resolution with pip check and pip-audit.

Dagster 1.13.1 fixes dynamic-partition SQL injection:
[GHSA-mjw2-v2hm-wj34](https://github.com/dagster-io/dagster/security/advisories/GHSA-mjw2-v2hm-wj34).
Kestra 2.0.7 reproduced a DNS-name bypass of the HTTP deny list against a local
synthetic HTTP canary. The same regression passes on 2.0.8 with both SDK versions:
direct IP and DNS requests fail, and the canary receives no request. See
[GHSA-f5h6-rg96-px74](https://github.com/kestra-io/kestra/security/advisories/GHSA-f5h6-rg96-px74).
CI verifies the official 2.0.8 server artifact SHA-256 before execution:
fcc48c9812e6b5a0534f80b5bff9fddcc021280115434900da43870152f8f8f8.
The Python SDK audit is not an audit of Java dependencies; no complete Java SBOM
vulnerability scan is claimed.

Mage 0.9.80 still pins vulnerable transitive dependencies, including
PyJWT==2.6.0, aiohttp==3.10.0, cryptography==42.0.4 and Werkzeug==3.0.3.
A normal resolver rejects mage-ai==0.9.80 together with PyJWT>=2.15.0.
There is no newer released host to raise the floor to.
The real scheduler completes good data and fails bad data, but that functional
result does not make the host security-supported. Interpreter separation resolves
the Core/Mage Typer conflict only. Do not deploy this host as a security-compliant
baseline or override dependency metadata with --no-deps. Upstream must publish
a compatible patched dependency set before host security support can resume.
Existing blocking security gates remain enabled; Mage host security is unresolved.

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
| Mage | `truthound-orchestration[mage]` with a separate Core interpreter for Mage 0.9.80 (see below) |
| Kestra | `truthound-orchestration[kestra]` with `Truthound 3.x` |
| dbt | `truthound-orchestration[dbt]` with the bundled first-party `truthound_dbt` package |

Release-blocking guarantees apply to the declared audited surfaces; Mage host security remains unsupported. `truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.

## Security dependency baseline

The current supported dbt Core range is **`>=1.11.14,<1.12.0`**, with 1.11.16 as
primary and dbt-postgres 1.10.0 retained. Both the Python extra and the SQL package
reject older Core versions. Install `sqlparse>=0.6.0,<0.7.0`; a compatible Core
version alone still permits vulnerable parser versions. Previously verified
1.10.9/1.10.20 tuples are historical functional results, not current supported
security baselines.

The [upstream dbt advisory](https://github.com/dbt-labs/dbt/security/advisories/GHSA-pxwq-479r-hhwr)
explains that the old parser ceiling prevents fixes for denial of service on SQL
compilation paths. Core 1.11.14 relaxes that ceiling. The 1.10 line still cannot
resolve the fixed parser normally; do not override its metadata with `--no-deps`.

Redshift compile verification requires `dbt-redshift>=1.11.1,<1.12.0`.
Versions through 1.11.0 independently pin `sqlparse<0.6.0`, so upgrading only
dbt Core is insufficient for that adapter. PostgreSQL, Snowflake, BigQuery and
Databricks retain their 1.10 adapter lanes.

OpenTelemetry API, SDK and both OTLP exporters now require **1.45.1 or newer**.
This fixes unbounded parsing of malformed inbound `tracestate` headers described
in the [upstream OpenTelemetry advisory](https://github.com/open-telemetry/opentelemetry-python/security/advisories/GHSA-xcq7-p3pf-p88h).
The dbt extra also enforces the API floor because its host dependencies use that
API. HTTP gzip export now passes the SDK compression enum instead of a string.

Security CI resolves the complete selected extra and runs `pip check` before
`pip-audit`; it no longer installs the project with `--no-deps`. No vulnerability
exceptions were added. Independent Core 3.3.0 environments for dbt 1.11.14,
dbt 1.11.16 and OpenTelemetry 1.45.1 each passed dependency checks and reported
zero known vulnerabilities. The OpenTelemetry audit retains its Protobuf 5
constraint; dbt uses Protobuf 6.

Regression coverage exercises native dbt CTE injection, the package security
floors, malformed header parsing, and real gzip-compressed trace/metric delivery
to local HTTP and gRPC collectors through both exporter factories. Both dbt
versions run the PostgreSQL suite and its existing manifest coverage gate.
These are tests of the revised source and built artifacts, not a new package
publication or certification of every other host's security audit.

## Core 3.3.0 verification

The source revision containing these fixes was verified with the published
`truthound==3.3.0` wheel. Initial adapter checks used Python 3.11.14 / macOS
ARM64; subsequent server and minimum-version results are recorded separately below. The project remains
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
| Mage | Packaged transformer execution with real Core; subsequent server results below |
| Kestra | Python SDK 1.3.0 and script executor; subsequent server results below |
| dbt | Core 1.10.20 / postgres adapter 1.10.0; compile, seed, run, test and macro execution on isolated PostgreSQL 14.20 |
| Advanced Core | Real KS drift and Isolation Forest, Z-score and Ensemble anomaly detection with NumPy/SciPy/scikit-learn installed |

The dbt data suite produced 12 passes, 7 expected warnings for invalid seed data,
zero errors and zero skips. Manifest coverage was 100% against the existing 80%
gate. This validates PostgreSQL execution, not live Snowflake, BigQuery, Redshift
or Databricks connections. Subsequent minimum-host and scheduler/server verification is recorded below.
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

## Server and minimum-host verification

This section records earlier verification. Airflow 2.6.0 and 3.2.0 are no longer
supported; use the Airflow security baseline and generated CI matrix below for
the current support range.

Actual local scheduler/server execution now covers the following combinations on
macOS ARM64 with published Core 3.3.0. Each server accepted synthetic work and
persisted a successful result for valid input and a failed result for null input.
These are isolated verification deployments, not production deployments.

| Host | Versions | Python | Verified execution |
| --- | --- | --- | --- |
| Airflow | 2.6.0 / 3.2.0 | 3.11.14 / 3.12.11 | Scheduler, web/API server, task process and XCom success/failure payloads |
| Dagster | 1.5.0 / 1.12.18 | 3.11.14 / 3.12.11 | Webserver, daemon, GraphQL submission and child run processes |
| Prefect | 2.14.0 / 3.6.29 | 3.11.14 / 3.12.11 | API server, served flow and Runner; COMPLETED / FAILED states |
| Mage | 0.9.80 | 3.11.14 / 3.12.11; separate Core interpreter | API schedule and scheduler block execution; completed / failed states |
| Kestra | Server 2.0.7; Python SDK 1.3.0 | 3.11.14; server Java 25 | Standalone server and process worker; SUCCESS / FAILED states |
| dbt | Core 1.10.9 / 1.10.20; postgres adapter 1.10.0 | 3.12.11 | PostgreSQL 14.20 compile, seed, run, test and macros |

Both dbt versions returned 12 passes, 7 expected warnings, zero errors/skips and
100% manifest coverage. Airflow's minimum and primary package suites each passed
273 tests; Prefect's each passed 169. Both Dagster suites passed 200 tests.
The Kestra version in the generated matrix is the **Python SDK**, not a minimum
Java server version. This run verifies server 2.0.7 only. Mage server checks passed on both Python 3.11 and 3.12, with normally resolved
host and Core environments. The Python 3.12 run used the final rebuilt wheel.

Reproduce Airflow, Dagster and Prefect server checks after installing the selected
host and orchestration wheel in separate environments:

```bash
python scripts/ci/verify_host_server.py airflow --output-dir /tmp/airflow-verification
python scripts/ci/verify_host_server.py dagster --output-dir /tmp/dagster-verification
python scripts/ci/verify_host_server.py prefect --output-dir /tmp/prefect-verification
python scripts/ci/verify_mage_server.py --host-python /path/to/mage/bin/python \
  --core-python /path/to/core/bin/python --output-dir /tmp/mage-verification
```

Output directories must be new. Runners retain local evidence and shut down their
own processes. Keep databases, generated configuration and raw Mage server logs
private; CI uploads only selected summaries and test logs.

Installation findings and boundaries:

- Airflow 2.6.0 has no published `constraints-3.11.txt`; the former CI URL returns
  404. The minimum lane uses published 3.10 dependency pins on Python 3.11. This
  is an orchestration-tested tuple, not an upstream Python 3.11 support claim.
  On macOS only, the verification runner uses Airflow's custom task-runner hook
  to spawn an interpreter: the default fork path hangs in native `setproctitle`.
  Linux retains the default runner. Airflow 3.2.0 uses official Python 3.12
  constraints to avoid an incompatible unconstrained Pydantic installation.
- Prefect 2.14.0 server startup needs `cachetools`, now included in the Prefect
  extra, and `fastapi<0.100` for its legacy Starlette generation. Prefect 3.6.29
  uses `fastapi<0.130` because newer FastAPI removes its `Body(min_items=...)`
  argument. These host constraints are recorded in `ci/support-matrix.toml`.
- dbt-core 1.10.0 and dbt-postgres 1.10.0 conflict on Protobuf major versions.
  The first resolvable 1.10 Core was 1.10.9, but the security baseline above
  now supersedes that functional minimum with 1.11.14.
- Mage 0.9.80 pins `typer[all]==0.9.0`, while Core 3.3.0 requires `typer>=0.12.0`.
  A same-interpreter install is **unsupported**. The verified Mage block calls
  the packaged first-party transformer in a separately resolved Core interpreter
  and propagates quality failures to Mage. Do not bypass dependency resolution
  with `--no-deps`. This process boundary is required for this combination.
- These results do not certify other host versions, live cloud warehouses or
  production infrastructure. The project remains Alpha.

## Airflow security baseline

The primary host and blocking security audit use Airflow **3.3.2** with Truthound
**3.3.0**. The former audit pinned Airflow 3.3.0 and reported 15 upstream
advisories. Airflow 3.3.2 contains the fixes; no advisory is suppressed.
In particular, [CVE-2026-82355](https://lists.apache.org/thread/3p7zpdvv40tn01m0xk5mt2rxg88mw8w1)
allowed a cached session-cookie identity to override an explicit Bearer token.
The native authentication regression checks Bearer precedence, invalid-token
rejection, valid cookie caching, and rejection of untrusted cached identities.
It fails on 3.3.0 and passes on 3.3.2.

The minimum and primary supported Airflow version is **3.3.2**. The minimum
Python tuple is **3.11**, and the primary tuple is **3.12**, each using the same
audited security constraints. Both root-package extras (`airflow` and `all`) and
the `truthound-airflow` provider require `apache-airflow>=3.3.2`. Functional
support for older hosts, including 2.6.0, has ended. Historical results below
do not define the current support range.

The blocking audit resolves the complete Airflow extra with
`cryptography>=46.0.5` and the dependency floors listed below, runs `pip check`, native authentication
regressions, and `pip-audit` without ignored findings. Host compatibility CI
uses the same security constraints and executes actual
servers and persisted success/failure XCom results.

Before the minimum-version increase, verification on 2026-10-29 passed all 273 Airflow adapter tests in each host
environment, four native authentication regressions on 3.3.2, and actual
scheduler/web/API/task execution on 2.6.0/Python 3.11 and 3.3.2/Python 3.12.
Both servers persisted `good=success`, `bad=failed` and matching XCom payloads
with Core 3.3.0. The complete 3.3.2 audit found zero known vulnerabilities.
Initial concurrent local attempts encountered worker termination on 2.6.0
and execution-API timeouts on 3.3.2; those failed records were retained. Fresh
isolated reruns passed without changing host code or weakening assertions.

The 2.6-only `pytest<9.2` and `packaging<24` constraints have been removed.
`pip check` still runs after test-tool installation. Artifact names include
the lane and Python version to avoid upload collisions when both jobs test
the same Airflow version. Regression coverage enforces the package floor,
Python-specific runtime lanes and distinct artifact names.

### Transitive dependency security floors

Auditing the upstream 3.3.2 constraints snapshot found nine advisory reports
across five packages (including duplicate certifi reports). The package extras,
provider metadata, runtime CI and security audit now share these floors:

| Dependency | Minimum |
| --- | --- |
| certifi | 2026.10.19 |
| dnspython | 2.9.0 |
| opentelemetry-api | 1.45.1 |
| PyJWT | 2.15.0 |
| urllib3 | 2.8.0 |
| cryptography | 46.0.5 (existing security floor) |

The fixes cover certificate trust, DNS/tracestate parsing, JWT validation and
HTTP proxy/streaming handling. See the upstream [certifi release](https://github.com/certifi/python-certifi/releases/tag/2026.10.19),
[dnspython release](https://github.com/rthalley/dnspython/releases/tag/v2.9.0),
[PyJWT changelog](https://pyjwt.readthedocs.io/en/stable/changelog.html#v2-15-0),
[urllib3 changelog](https://urllib3.readthedocs.io/en/stable/changelog.html#id1)
and the OpenTelemetry advisory linked above. The old upstream constraints
snapshot cannot be combined with these floors; CI resolves normally using the
repository's security constraints, without ignored vulnerabilities or `--no-deps`.

Current-floor verification passed on Airflow 3.3.2 with Core 3.3.0 on both
Python 3.11 and 3.12: 274 adapter/header regression tests, four native auth
regressions, `pip check`, and zero known vulnerabilities in each complete
installed environment. Actual scheduler/API/task execution persisted
`good=success`, `bad=failed` and matching XCom results. The Python 3.12 server
loaded the built wheel from site-packages. Both built distributions' metadata
and dependency resolution reject Airflow 3.3.1. Wheel/sdist validation, package
privacy checks, strict documentation build and bilingual link checks passed.
This verifies the revised source and artifacts; it does not publish a release.

The Linux Python 3.11 runner also bundled setuptools 79.0.1, which the audit
reported separately from runtime dependencies. CI now upgrades bootstrap tools
to `setuptools>=84.1.0` before host installation; it does not suppress those
findings. See the [upstream setuptools advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w).
Existing environments should update their bootstrap tools as well.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
## Generated CI Support Matrix

Truthound release line: `>=3.0,<4.0`

| Lane | Airflow | Prefect | Dagster | Mage | Kestra | dbt Compile | dbt Execute |
|------|---------|---------|---------|------|--------|-------------|-------------|
| PR | `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.12` | `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `2.0.0` on `Python 3.12` | `postgres` | No |
| Main | `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.12` | `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `2.0.0` on `Python 3.12` | `postgres` | No |
| Release | `3.3.2` on `Python 3.11`, `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.11`, `3.9.0` on `Python 3.12` | `1.13.1` on `Python 3.11`, `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `1.3.0` on `Python 3.11`, `2.0.0` on `Python 3.12` | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |
| Nightly | `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.12` | `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `2.0.0` on `Python 3.12` | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |

## Supported Host Version Anchors

| Platform | Minimum Supported | Primary Supported |
|----------|-------------------|-------------------|
| Airflow | `3.3.2` on `Python 3.11` | `3.3.2` on `Python 3.12` |
| Prefect | `3.9.0` on `Python 3.11` | `3.9.0` on `Python 3.12` |
| Dagster | `1.13.1` on `Python 3.11` | `1.13.27` on `Python 3.12` |
| Kestra SDK | `1.3.0` on `Python 3.11` | `2.0.0` on `Python 3.12` |
| Kestra Java server | `2.0.8` | `2.0.8` |
| dbt | `dbt-core>=1.11.14,<1.12.0` + `dbt-postgres>=1.10.0,<1.11.0` | `dbt-core 1.11.16` + `dbt-postgres 1.10.0` |

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

First-party release guarantees apply to declared audited surfaces; Mage host security is unsupported. `truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.
Minimum compatibility guarantees are validated as tested host-plus-Python tuples, not as host versions on every Python runtime.
Airflow runtime lanes and security audits resolve the complete extra with the same security constraints and pinned supported host.
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
