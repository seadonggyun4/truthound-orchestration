!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: dbt Adapter Behavior
---

# dbt Adapter Behavior

Truthound keeps one public macro surface and routes warehouse-specific SQL through
`adapter.dispatch`. This page explains what that means in practice.

## Why Adapter Dispatch Matters

Even simple 검증 rules are not fully portable across warehouses. Different
adapters disagree on:

- regex operators
- length functions
- safe cast syntax
- current timestamp helpers
- percentage rounding behavior
- sampling and limit syntax

Truthound hides those differences behind shared macro names in `truthound_utils.sql`
and related files.

## What Stays Stable

The following stay stable across adapters:

- test names
- rule dictionaries
- top-level run-operation macro names
- the general result model exposed back to dbt

## What Changes By Adapter

The compiled SQL may change for:

- `regex_match`
- `regex_not_match`
- `safe_cast`
- `round_percentage`
- `current_timestamp`
- window and sample helpers

That is expected. A compile diff between PostgreSQL and Snowflake does not mean the
package surface changed.

## Supported Warehouses

Truthound is designed for adapter-dispatched behavior across common dbt warehouses.
The first-party suite in this repository currently proves the package through compile
and PostgreSQL execution, while keeping the macro design ready for the broader dbt
adapter family.

## Operational Implications

- compile success proves macro discovery and basic SQL generation
- execute success proves that the adapter path works on a real warehouse
- an adapter-specific runtime failure usually belongs in the utility macro layer, not
  in every individual test definition

## Recent Compatibility Lessons

The package intentionally keeps adapter-specific logic centralized. For example:

- package-qualified test discovery avoids ambiguous macro resolution
- URL and numeric handling regressions are fixed in macro helpers rather than in each
  test case
- percentage rounding must respect warehouse function signatures

## 문제 해결 Checklist

If one warehouse fails but another compiles:

1. confirm the failing SQL path is adapter-specific
2. inspect the compiled macro output under `target/compiled`
3. check utility macros before touching dozens of test definitions

## Related Pages

- [Generic Tests](generic-tests.md)
- [Macros and Operations](macros.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
