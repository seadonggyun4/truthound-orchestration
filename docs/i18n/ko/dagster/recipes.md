!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Dagster Recipes
---

# Dagster Recipes

## Add A Resource To Existing Definitions

Start by adding `DataQualityResource()` to your existing `Definitions` object and validating one asset with a simple rule.

## Profile Before Turning On Hard Fails

Use `profiled_asset` or a profile op in staging before enabling strict asset checks in production.

## Use Asset Checks For Catalog Visibility

Prefer asset checks when you want failures to show up as first-class Dagster quality signals rather than buried inside asset logic.

## Separate Quality Execution From Alerting

Run 검증 through resources or asset helpers, then attach SLA hooks or separate downstream ops for alerting and incident routing.
