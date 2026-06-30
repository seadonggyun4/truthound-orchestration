!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Prefect Deployment Patterns
---

# Prefect Deployment Patterns

## Ephemeral Local Flow

Best for:

- local development
- smoke checks
- onboarding

Pattern:

- use task helpers directly
- omit saved blocks
- keep source access simple

## Shared Deployment With Saved Block

Best for:

- repeated runs in a work pool
- shared environment-specific configuration
- teams that want named reusable config objects

Pattern:

- save a `DataQualityBlock`
- load it inside the flow or deployment
- keep schedule and environment config in Prefect deployment settings

## Flow Factory Pattern

Best for:

- several datasets following the same quality 워크플로우 shape
- teams standardizing deployment construction

Pattern:

- use flow decorators or factory helpers
- keep only dataset-specific parameters at the call site

## Operational Advice

- choose one pattern per environment when possible
- avoid mixing ephemeral and persisted configuration invisibly
- document block ownership and naming
