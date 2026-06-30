!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Preflight and Runtime Failure Catalog
---

# Preflight and Runtime Failure Catalog

This page translates common failure classes in `truthound-오케스트레이션` into
operator-facing diagnostics. It is intentionally grounded in real failure
patterns that have already appeared in this repository.

## Who This Is For

- operators triaging host failures quickly
- teams writing runbooks and release gates
- contributors who need a shared language for failure categories

## When To Use It

Use this page when a quality 워크플로우 fails before execution, during result
serialization, or inside CI support-matrix 검증.

## Prerequisites

- access to host logs or CI artifacts
- the host, engine, and source shape involved in the failing run

## Minimal Quickstart

A useful first pass is:

1. identify the host and operation (`check`, `profile`, `learn`, `stream`)
2. check whether failure happened in preflight or execution
3. compare the failure against the catalog below

## Failure Catalog

### 호환성 Tuple Mismatch

Symptoms:

- installation or test matrix fails before runtime
- host version and Python version fall outside the support matrix

Examples already seen in this repository:

- old Dagster API differences around `asset_check`
- old Prefect and Pydantic compatibility issues
- dbt adapter version and package-resolution mismatches

First places to read:

- [Compatibility](../compatibility.md)
- host-specific install and compatibility pages

### Preflight Incompatibility

Symptoms:

- run aborts before engine creation or execution
- compatibility report contains one or more failures

Likely causes:

- operation not supported by the selected engine
- source shape needs a connection or serializer the runtime cannot provide
- host-level auto-config does not match the selected execution path

### Source Resolution Errors

Symptoms:

- local path is treated differently in one host than another
- SQL path reaches execution without a connection
- stream execution cannot infer resumable batches

Likely causes:

- ambiguous source shape
- missing host-native connection layer
- unsupported stream resume assumptions

### Result Serialization Drift

Symptoms:

- host task succeeds but downstream consumers cannot parse the payload
- custom code reads logs instead of the documented result contract

Likely causes:

- bypassing shared serializer helpers
- host-native payload assumptions not aligned with the shared runtime contract

### Docs And CI Contract Regressions

Symptoms:

- docs-only change fails `foundation` or public-doc checks
- top-level CI summaries fail even though runtime code did not change

Examples already seen:

- brittle docs contract wording on the homepage
- public snapshot count drift after 오케스트레이션-doc expansion

### Host-Specific Legacy API Drift

Symptoms:

- a host package imports or decorator signatures differ across supported minors

Examples already seen:

- Dagster asset-check surface differences
- Prefect block serialization and private-attribute issues

## Production Pattern

- Make preflight visible in CI and deployment smoke paths.
- Keep host-specific troubleshooting tables close to the adapter section.
- Treat docs, support-matrix, and snapshot failures as first-class operator
  issues, not as “just docs”.

## Related Pages

- [Preflight and Compatibility](preflight-compatibility.md)
- [Compatibility](../compatibility.md)
- [Troubleshooting](../troubleshooting.md)
