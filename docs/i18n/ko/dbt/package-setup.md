!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: dbt Package Setup
---

# dbt Package Setup

This page covers the supported setup path for installing Truthound into a dbt project.

## Who This Is For

- teams adopting Truthound in an existing dbt repository
- platform engineers standardizing package and adapter installation
- CI maintainers who need reproducible compile and execution lanes

## Prerequisites

- a dbt project with a working `profiles.yml`
- a supported dbt core and adapter combination
- warehouse connectivity if you plan to run execution tests locally

See [Compatibility](../compatibility.md) for the current 오케스트레이션 support matrix.

## Install From dbt Hub

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=3.0.3,<4.0.0"
```

Then install:

```bash
dbt deps
```

## Local Development Install

For work inside this repository or for testing a local checkout:

```yaml
packages:
  - local: "../truthound-오케스트레이션/packages/dbt"
```

This is the same style used by the first-party integration project under
`packages/dbt/integration_tests`.

## Project Conventions That Matter

Truthound follows normal dbt conventions:

- tests are declared in `schema.yml`
- models are referenced with `ref()`
- sources are referenced with `source()`
- package macros are available after `dbt deps`

The package does not need a custom adapter plugin of its own. It runs on top of the
warehouse adapter already used by the dbt project.

## Recommended First 검증

Start with one model-level grouped test:

```yaml
version: 2

models:
  - name: dim_users
    tests:
      - truthound.truthound_check:
          arguments:
            rules:
              - column: id
                check: not_null
              - column: id
                check: unique
              - column: email
                check: email_format
```

Then run:

```bash
dbt test --select dim_users
```

## Profiles And Execution Environments

Truthound does not replace dbt profile handling. It expects:

- the project profile name to exist
- the selected target to be valid
- the warehouse adapter to be installed

The repository's first-party execute suite runs with a project-local `profiles.yml` so
that CI can validate compile and warehouse execution deterministically.

## Package-Qualified Test Names

The safest pattern is to qualify generic and convenience tests with the package name:

- `truthound.truthound_check`
- `truthound.truthound_not_null`
- `truthound.truthound_unique`

This makes test resolution explicit and avoids confusion in multi-package projects.

## Production Pattern

For mature deployments:

- pin the package version range intentionally
- keep `dbt deps` in CI, not only local 워크플로우s
- test both compile-only and execute-against-warehouse paths
- prefer package-qualified test names consistently across the project

## Related Pages

- [Generic Tests](generic-tests.md)
- [Macros and Operations](macros.md)
- [CI and First-Party Suite](ci-first-party-suite.md)
