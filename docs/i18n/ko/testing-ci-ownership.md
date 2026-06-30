!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Testing and CI Ownership
---

# Testing and CI Ownership

Use this guide when you are adding or moving tests in `truthound-오케스트레이션`.

## Who This Is For

- contributors adding adapter coverage
- maintainers deciding where a new regression test should live
- CI owners updating 워크플로우 routing or artifact boundaries

## When To Use It

Use this page whenever you need to decide:

- whether a test belongs under `packages/<adapter>/tests` or `tests/<adapter>`
- which fixtures should stay shared versus adapter-local
- which 워크플로우 should prove the behavior in CI
- which file owns a version change versus a host compatibility change

## Prerequisites

- a local checkout of `truthound-오케스트레이션`
- the development environment from the repository README
- a clear understanding of whether the behavior is adapter-native or repo-contract level

## Canonical Ownership Model

| Location | Owns | Examples |
|----------|------|----------|
| `packages/<adapter>/tests/` | adapter-native unit and integration tests | host primitives, adapter config, result shaping, host-specific utilities |
| `tests/<adapter>/` | repo-level contract and harness tests | first-party suite runners, support-matrix contracts, CI harness behavior, multi-package wiring |
| `tests/common/` | monorepo-wide CI and shared runtime contracts | support-matrix exports, 워크플로우 routing, docs contract checks |

Default rule:

- if a test imports only adapter package code and shared fixtures, it belongs package-local
- if a test verifies repo wiring, CI contracts, install surfaces, or multi-package behavior, it belongs at the repo root
- if a test verifies shared 워크플로우 operation, flow, failure, or observability contracts across hosts, it belongs under `tests/common/`

## Minimal Quickstart

For a new adapter-native test:

```bash
PYTHONPATH=.:packages/mage/src pytest --import-mode=importlib packages/mage/tests
```

For a repo-level dbt contract test:

```bash
PYTHONPATH=. pytest --import-mode=importlib tests/dbt/test_first_party_suite.py
```

## Production Pattern

Treat every adapter family as a self-contained product surface:

1. keep happy-path and failure-path tests in `packages/<adapter>/tests`
2. keep repo-level contracts thin and intentional
3. route CI by adapter 워크플로우 so failures are visible per host
4. preserve shared fixtures only when they reduce duplication across multiple adapters

### Decision Table

| Question | If yes | If no |
|----------|--------|-------|
| does the test exercise only one adapter package? | place it under `packages/<adapter>/tests` | continue evaluating |
| does the test validate a repo script, CI harness, or support-matrix contract? | keep it under `tests/<adapter>` or `tests/common` | continue evaluating |
| does the test need shared runtime fixtures but no repo harness behavior? | keep shared fixtures in `common.testing`, but place the test package-local | continue evaluating |
| does the test verify multiple packages working together? | keep it at the repo root | package-local is usually correct |

## Fixture Placement

Use this fixture policy:

- shared fixtures in `common.testing` or top-level `conftest.py` for host-agnostic engines and payloads
- adapter-local `conftest.py` for host contexts, adapter config shapes, or host-native expectations
- root contract fixtures only when the contract itself is repo-owned

## CI 워크플로우 Ownership

Each adapter family should report health independently.

| 워크플로우 | Owns |
|----------|------|
| `airflow.yml` | Airflow package-native suites |
| `dagster.yml` | Dagster package-native suites |
| `prefect.yml` | Prefect package-native suites |
| `dbt.yml` | dbt package-native suites plus repo-level first-party suite contracts |
| `mage.yml` | Mage package-native suites |
| `kestra.yml` | Kestra package-native suites |
| `shared-runtime.yml` | shared runtime and cross-adapter common behavior |
| `ci-foundation.yml` | lint, build, doc contracts, support-matrix sync |

### Result Surface

- package-local adapter failures should appear in the adapter 워크플로우, not inside a bundled multi-host lane
- root contract failures should remain visible in their contract lane, such as `dbt` or `foundation`
- summary jobs should aggregate adapter results, not hide ownership boundaries
- 워크플로우 shared smoke belongs in `shared-runtime.yml`, while adapter-native 워크플로우 projection tests remain package-local

## Version Ownership

Release-version surfaces and host compatibility surfaces are intentionally split:

| Surface Type | Source Of Truth | Examples |
|--------------|-----------------|----------|
| release-version surfaces | `ci/version-surfaces.toml` | root/package `pyproject.toml`, adapter `version.py`, `packages/dbt/dbt_project.yml`, release tag checks |
| host compatibility surfaces | `ci/support-matrix.toml` | Airflow/Dagster/Prefect minimums, dbt execution versions, security audit surfaces |

Default rule:

- if you are changing `3.0.0`, `v3.0.0`, `3.x`, `>=3.0,<4.0`, or `>=3.0.0,<4.0.0`, edit `ci/version-surfaces.toml` and run `python scripts/ci/sync_version_surfaces.py write`
- if you are changing supported host versions or CI lane matrices, edit `ci/support-matrix.toml`

## Failure Modes And Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| a Mage-only change triggers Kestra CI | PR routing still groups adapters together | update `scripts/ci/pr_change_filter.py` and 워크플로우 wiring |
| a dbt unit test still lives at root | adapter-native test was never migrated | move it to `packages/dbt/tests` unless it validates repo harness behavior |
| package-local tests cannot find fixtures | shared fixtures are still root-only | move generic fixtures to `common.testing` or adapter-local `conftest.py` |
| CI artifacts bundle multiple adapters together | 워크플로우 boundaries are still shared | split the 워크플로우 and artifact names by adapter |
| support-matrix docs drift from 워크플로우 reality | matrix export logic or docs sync is stale | run `python scripts/ci/export_support_matrix.py sync-docs --path docs/compatibility.md --write` |
| package or runtime versions drift from one another | release-version files were hand-edited | run `python scripts/ci/sync_version_surfaces.py write` and commit the generated changes |

## Adding A New Adapter-Native Test

Use this checklist:

1. put the test under `packages/<adapter>/tests`
2. keep fixtures shared only if another adapter genuinely reuses them
3. ensure the adapter 워크플로우 runs the new file by path or directory
4. add or adjust repo-level contract tests only if package behavior alone is not enough
5. verify PR routing still points the change at the correct 워크플로우

## Related Pages

- [Compatibility](compatibility.md)
- [워크플로우 Pipelines]
- [Production Readiness](production-readiness.md)
- [Troubleshooting](troubleshooting.md)
- [Releasing](releasing.md)
