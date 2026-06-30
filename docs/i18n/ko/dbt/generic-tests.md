!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: dbt Generic Tests
---

# dbt Generic Tests

Truthound's main dbt authoring surface is a set of generic tests. They let you keep
검증 next to the model contract instead of moving the logic into separate SQL
files or Python 오케스트레이션 code.

## Quick Model-Level Example

```yaml
version: 2

models:
  - name: test_model_valid
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
              - column: user_uuid
                check: uuid_format
              - column: url
                check: url_format
```

## Column-Level Convenience Tests

Truthound also ships shorthand tests for common cases:

```yaml
columns:
  - name: id
    tests:
      - truthound.truthound_not_null
      - truthound.truthound_unique

  - name: email
    tests:
      - truthound.truthound_email_format

  - name: age
    tests:
      - truthound.truthound_range:
          arguments:
            min: 0
            max: 150
```

## Supported Rule Families

The tested first-party project exercises the following families:

- completeness: `not_null`, `not_empty`
- uniqueness: `unique`
- numeric checks: `range`, `positive`, `non_negative`
- format checks: `email_format`, `uuid_format`, `url_format`
- set membership: `in_set`
- relational checks: `referential_integrity`

## Warning vs Error Behavior

Truthound uses normal dbt test config semantics.

Use `severity: warn` when:

- you are validating intentionally bad fixtures
- you want visibility before hard enforcement
- you are migrating an existing model toward a stricter contract

```yaml
tests:
  - truthound.truthound_check:
      arguments:
        rules:
          - column: status
            check: in_set
            values: ["active", "inactive", "pending"]
      config:
        severity: warn
```

## Modern YAML Shape

dbt 1.10 warns when test arguments are supplied as top-level keys instead of under
`arguments`. The package still compiles in that shape today, but new docs should use
the modern form shown above.

## Source Testing

The same package works on dbt sources:

```yaml
version: 2

sources:
  - name: raw
    tables:
      - name: users
        tests:
          - truthound.truthound_check:
              arguments:
                rules:
                  - column: user_id
                    check: not_null
                  - column: email
                    check: email_format
```

## Production Guidance

- keep grouped model-level tests for business-critical datasets
- use column-level convenience tests for obvious single-column guarantees
- prefer explicit package qualification everywhere
- use warnings only intentionally and document why they are warnings

## Failure Modes

- undefined generic tests usually mean the package was not installed or names were not
  resolved the way you expected
- passing compilation with unexpected runtime failures often indicates an adapter edge
  case rather than a YAML problem

## Related Pages

- [Package Setup](package-setup.md)
- [Macros and Operations](macros.md)
- [Troubleshooting](troubleshooting.md)
