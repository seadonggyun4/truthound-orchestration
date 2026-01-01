# Package: truthound-dbt

> **Last Updated:** 2026-01-02
> **Document Version:** 2.1.0
> **Package Version:** 0.1.0
> **Status:** ✅ Complete

---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Components](#components)
4. [Generic Tests](#generic-tests)
5. [Jinja Macros](#jinja-macros)
6. [Manifest Parsing](#manifest-parsing)
7. [dbt Project Configuration](#dbt-project-configuration)
8. [Usage Examples](#usage-examples)
9. [dbt Cloud Integration](#dbt-cloud-integration)
10. [Testing Strategy](#testing-strategy)

---

## Overview

### Purpose
`truthound-dbt`는 dbt (Data Build Tool)용 **범용 데이터 품질** 패키지입니다. dbt의 네이티브 테스트 프레임워크와 seamless하게 통합되어, SQL 모델에 대해 데이터 품질 검증을 수행할 수 있습니다. Truthound가 기본 엔진으로 제공됩니다.

### Key Features

| Feature | Description |
|---------|-------------|
| **Engine-Agnostic Rules** | 다양한 검증 규칙을 선언적으로 정의 |
| **Generic Tests** | dbt 네이티브 테스트로 규칙 실행 |
| **Jinja Macros** | 재사용 가능한 검증 매크로 |
| **YAML Configuration** | 선언적 규칙 정의 |
| **dbt Cloud Compatible** | dbt Cloud 환경 완벽 지원 |
| **Cross-Adapter** | 모든 dbt 어댑터 지원 (Snowflake, BigQuery, Redshift 등) |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            dbt Project                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      models/                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │  staging/   │─▶│   marts/    │─▶│  reporting/ │              │   │
│  │  │             │  │             │  │             │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    schema.yml (tests)                            │   │
│  │                                                                  │   │
│  │  - data_quality_check:                                          │   │
│  │      rules:                                                      │   │
│  │        - column: email                                           │   │
│  │          type: regex
              pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"                                     │   │
│  │        - column: age                                             │   │
│  │          type: in_range                                            │   │
│  │          min: 0                                                  │   │
│  │          max: 150                                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  data_quality package                            │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │ test_data_      │  │ data_quality_   │  │ dq_utils        │  │   │
│  │  │ quality_check   │  │ check.sql       │  │     .sql        │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│                      ┌─────────────────┐                               │
│                      │   Data Warehouse │                               │
│                      │  (Test Queries)  │                               │
│                      └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### How It Works

```
1. YAML에 Truthound 규칙 선언
         │
         ▼
2. dbt가 Generic Test 매크로 호출
         │
         ▼
3. 매크로가 SQL 검증 쿼리 생성
         │
         ▼
4. dbt가 데이터 웨어하우스에서 쿼리 실행
         │
         ▼
5. 실패 행이 있으면 테스트 실패
```

---

## Installation

### dbt Hub (권장)

```yaml
# packages.yml
packages:
  - package: truthound/truthound
    version: ">=0.1.0"
```

```bash
dbt deps
```

### Git Installation

```yaml
# packages.yml
packages:
  - git: "https://github.com/seadonggyun4/truthound-integrations.git"
    subdirectory: "packages/dbt"
    revision: "v0.1.0"
```

### Local Installation (개발용)

```yaml
# packages.yml
packages:
  - local: "../truthound-integrations/packages/dbt"
```

### Requirements

| Dependency | Version |
|------------|---------|
| dbt-core | >= 1.6.0 |
| Supported Adapters | snowflake, bigquery, redshift, postgres, databricks |

---

## Components

### Package Structure

```
packages/dbt/
├── dbt_project.yml           # 패키지 메타데이터
├── README.md
├── scripts/
│   └── manifest_parser.py    # 매니페스트 파서
├── src/truthound_dbt/        # Python 패키지
│   ├── __init__.py
│   ├── adapters/             # 어댑터 구현체
│   │   ├── __init__.py
│   │   ├── base.py           # 기본 어댑터
│   │   ├── postgres.py       # PostgreSQL 어댑터
│   │   ├── snowflake.py      # Snowflake 어댑터
│   │   ├── bigquery.py       # BigQuery 어댑터
│   │   ├── redshift.py       # Redshift 어댑터
│   │   └── databricks.py     # Databricks 어댑터
│   ├── converters/           # 변환기
│   │   ├── __init__.py
│   │   ├── base.py           # 기본 변환기
│   │   └── rules.py          # 규칙 변환기
│   ├── generators/           # 생성기
│   │   ├── __init__.py
│   │   ├── sql.py            # SQL 생성기
│   │   ├── schema.py         # 스키마 생성기
│   │   └── test.py           # 테스트 생성기
│   ├── parsers/              # 파서
│   │   ├── __init__.py
│   │   ├── manifest.py       # 매니페스트 파서
│   │   └── results.py        # 결과 파서
│   ├── hooks/                # 훅 시스템
│   │   ├── __init__.py
│   │   └── base.py           # 기본 훅
│   └── testing/              # 테스트 유틸리티
│       ├── __init__.py
│       └── mocks.py          # Mock 객체
├── macros/
│   ├── truthound_check.sql   # 메인 검증 매크로
│   ├── truthound_utils.sql   # 유틸리티 매크로
│   ├── truthound_rules.sql   # 규칙별 SQL 생성기
│   └── adapters/
│       ├── default.sql       # 기본 어댑터
│       ├── snowflake.sql     # Snowflake 최적화
│       ├── bigquery.sql      # BigQuery 최적화
│       ├── redshift.sql      # Redshift 최적화
│       └── databricks.sql    # Databricks 최적화
├── tests/
│   └── generic/
│       └── test_truthound_check.sql  # Generic Test
├── integration_tests/
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── test_model_valid.sql
│   │   ├── test_model_invalid.sql
│   │   ├── test_orders_model.sql
│   │   └── test_reference_model.sql
│   └── tests/
└── docs/
    └── README.md

**Python 파일: 23개 | SQL 파일: 13개 | 테스트: 6개**
```

---

## Generic Tests

### test_truthound_check

dbt의 Generic Test로 구현된 메인 검증 테스트입니다.

#### Specification

```sql
-- tests/generic/test_truthound_check.sql
{% test truthound_check(model, rules, fail_on_first=false, sample_size=none) %}
{#
    Truthound 품질 검증 Generic Test.

    이 테스트는 모델에 대해 여러 Truthound 규칙을 실행하고,
    규칙을 위반하는 행을 반환합니다. 반환된 행이 있으면 테스트가 실패합니다.

    Parameters
    ----------
    model : Relation
        테스트할 dbt 모델 또는 소스

    rules : list[dict]
        적용할 검증 규칙 목록
        예: [{"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"}]

    fail_on_first : bool (default: false)
        첫 번째 실패에서 중단할지 여부

    sample_size : int | none (default: none)
        샘플링할 행 수. none이면 전체 테스트.

    Supported Rules
    ---------------
    - not_null: NULL 값 체크
    - unique: 고유성 체크
    - in_set: 허용 값 목록 체크
    - not_in_set: 금지 값 목록 체크
    - range: 숫자 범위 체크
    - length: 문자열 길이 체크
    - regex: 정규식 패턴 체크
    - email_format: 이메일 형식 체크
    - url_format: URL 형식 체크
    - date_format: 날짜 형식 체크
    - positive: 양수 체크
    - negative: 음수 체크
    - not_future: 미래 날짜 아님 체크
    - not_past: 과거 날짜 아님 체크

    Returns
    -------
    SQL Query
        규칙을 위반하는 행을 선택하는 쿼리.
        반환된 행이 있으면 테스트 실패.

    Examples
    --------
    schema.yml에서 사용:

    ```yaml
    models:
      - name: users
        tests:
          - truthound_check:
              rules:
                - column: email
                  type: regex
              pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
                - column: age
                  type: in_range
                  min: 0
                  max: 150
    ```
#}

{{ config(
    severity = 'error',
    tags = ['truthound', 'quality']
) }}

{% set rule_queries = [] %}

{% for rule in rules %}
    {% set rule_sql = truthound.generate_rule_sql(model, rule) %}
    {% do rule_queries.append(rule_sql) %}
{% endfor %}

{% if sample_size %}
    {% set base_query %}
        select * from {{ model }}
        {{ limit_sample(sample_size) }}
    {% endset %}
{% else %}
    {% set base_query %}
        select * from {{ model }}
    {% endset %}
{% endif %}

with validation_source as (
    {{ base_query }}
),

{% for idx, rule_query in rule_queries | enumerate %}
rule_{{ idx }}_failures as (
    {{ rule_query }}
),
{% endfor %}

all_failures as (
    {% for idx in range(rule_queries | length) %}
    select
        '{{ rules[idx].column }}' as _truthound_column,
        '{{ rules[idx].check }}' as _truthound_check,
        *
    from rule_{{ idx }}_failures
    {% if not loop.last %}
    union all
    {% endif %}
    {% endfor %}
)

select * from all_failures
{% if fail_on_first %}
limit 1
{% endif %}

{% endtest %}
```

#### Usage in schema.yml

```yaml
# models/staging/schema.yml
version: 2

models:
  - name: stg_users
    description: "Staged user data with quality checks"
    tests:
      # 모델 레벨 테스트
      - truthound_check:
          rules:
            - column: user_id
              type: not_null
            - column: user_id
              type: unique
            - column: email
              type: regex
              pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            - column: age
              type: in_range
              min: 0
              max: 150
            - column: status
              check: in_set
              values: ['active', 'inactive', 'pending']
          config:
            severity: error
            tags: ['critical', 'pii']

    columns:
      - name: user_id
        description: "Unique user identifier"
        tests:
          # 컬럼 레벨 단일 규칙 테스트
          - truthound_check:
              rules:
                - check: uuid_format

      - name: created_at
        tests:
          - truthound_check:
              rules:
                - type: not_null
                - check: not_future
```

---

## Jinja Macros

### truthound_check.sql

메인 검증 매크로입니다.

```sql
-- macros/truthound_check.sql

{% macro truthound_check(model, rules, options={}) %}
{#
    모델에 대해 Truthound 규칙을 실행하는 메인 매크로.

    Parameters
    ----------
    model : Relation
        테스트할 모델

    rules : list[dict]
        검증 규칙 목록

    options : dict
        추가 옵션 (sample_size, fail_fast 등)

    Returns
    -------
    SQL Query
        실패 행을 반환하는 쿼리

    Example
    -------
    {% set failures = truthound_check(ref('users'), [
        {"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
        {"column": "age", "type": "in_range", "min": 0, "max": 150}
    ]) %}
#}

{% set sample_size = options.get('sample_size', none) %}
{% set fail_fast = options.get('fail_fast', false) %}

{% set rule_queries = [] %}
{% for rule in rules %}
    {% set rule_sql = generate_rule_sql(model, rule) %}
    {% do rule_queries.append(rule_sql) %}
{% endfor %}

with source_data as (
    select *
    from {{ model }}
    {% if sample_size %}
    {{ limit_sample(sample_size) }}
    {% endif %}
),

{% for idx, query in rule_queries | enumerate %}
check_{{ idx }} as (
    {{ query }}
),
{% endfor %}

combined_failures as (
    {% for idx in range(rule_queries | length) %}
    select
        '{{ rules[idx].get("column", "_model_") }}' as _truthound_column,
        '{{ rules[idx]["check"] }}' as _truthound_check,
        '{{ rules[idx].get("message", "") }}' as _truthound_message,
        t.*
    from check_{{ idx }} t
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
)

select * from combined_failures
{% if fail_fast %}limit 1{% endif %}

{% endmacro %}
```

### truthound_rules.sql

각 규칙에 대한 SQL을 생성하는 매크로입니다.

```sql
-- macros/truthound_rules.sql

{% macro generate_rule_sql(model, rule) %}
{#
    규칙에 해당하는 SQL 생성.

    Parameters
    ----------
    model : Relation
        대상 모델
    rule : dict
        규칙 정의

    Returns
    -------
    str
        실패 행을 선택하는 SQL
#}

{% set check_type = rule['check'] %}
{% set column = rule.get('column') %}

{% if check_type == 'not_null' %}
    {{ rule_not_null(model, column) }}

{% elif check_type == 'unique' %}
    {{ rule_unique(model, column) }}

{% elif check_type == 'in_set' %}
    {{ rule_in_set(model, column, rule['values']) }}

{% elif check_type == 'not_in_set' %}
    {{ rule_not_in_set(model, column, rule['values']) }}

{% elif check_type == 'range' %}
    {{ rule_range(model, column, rule.get('min'), rule.get('max')) }}

{% elif check_type == 'length' %}
    {{ rule_length(model, column, rule.get('min'), rule.get('max')) }}

{% elif check_type == 'regex' %}
    {{ rule_regex(model, column, rule['pattern']) }}

{% elif check_type == 'email_format' %}
    {{ rule_email_format(model, column) }}

{% elif check_type == 'url_format' %}
    {{ rule_url_format(model, column) }}

{% elif check_type == 'positive' %}
    {{ rule_positive(model, column) }}

{% elif check_type == 'negative' %}
    {{ rule_negative(model, column) }}

{% elif check_type == 'not_future' %}
    {{ rule_not_future(model, column) }}

{% elif check_type == 'not_past' %}
    {{ rule_not_past(model, column) }}

{% elif check_type == 'uuid_format' %}
    {{ rule_uuid_format(model, column) }}

{% elif check_type == 'date_format' %}
    {{ rule_date_format(model, column, rule.get('format', 'YYYY-MM-DD')) }}

{% else %}
    {{ exceptions.raise_compiler_error("Unknown Truthound check type: " ~ check_type) }}

{% endif %}

{% endmacro %}


{# ===== Individual Rule Macros ===== #}

{% macro rule_not_null(model, column) %}
{#
    NULL 값 체크 규칙.

    실패 조건: column IS NULL
#}
select *
from {{ model }}
where {{ column }} is null
{% endmacro %}


{% macro rule_unique(model, column) %}
{#
    고유성 체크 규칙.

    실패 조건: 중복 값 존재
#}
select t.*
from {{ model }} t
inner join (
    select {{ column }}
    from {{ model }}
    group by {{ column }}
    having count(*) > 1
) duplicates
on t.{{ column }} = duplicates.{{ column }}
{% endmacro %}


{% macro rule_in_set(model, column, values) %}
{#
    허용 값 목록 체크 규칙.

    실패 조건: column 값이 values에 없음
#}
{% set quoted_values = [] %}
{% for v in values %}
    {% do quoted_values.append("'" ~ v ~ "'") %}
{% endfor %}

select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} not in ({{ quoted_values | join(', ') }})
{% endmacro %}


{% macro rule_not_in_set(model, column, values) %}
{#
    금지 값 목록 체크 규칙.

    실패 조건: column 값이 values에 있음
#}
{% set quoted_values = [] %}
{% for v in values %}
    {% do quoted_values.append("'" ~ v ~ "'") %}
{% endfor %}

select *
from {{ model }}
where {{ column }} in ({{ quoted_values | join(', ') }})
{% endmacro %}


{% macro rule_range(model, column, min_val=none, max_val=none) %}
{#
    숫자 범위 체크 규칙.

    실패 조건: min_val <= column <= max_val 범위 벗어남
#}
select *
from {{ model }}
where {{ column }} is not null
  and (
    {% if min_val is not none %}
    {{ column }} < {{ min_val }}
    {% endif %}
    {% if min_val is not none and max_val is not none %}
    or
    {% endif %}
    {% if max_val is not none %}
    {{ column }} > {{ max_val }}
    {% endif %}
  )
{% endmacro %}


{% macro rule_length(model, column, min_len=none, max_len=none) %}
{#
    문자열 길이 체크 규칙.

    실패 조건: 길이가 min_len ~ max_len 범위 벗어남
#}
select *
from {{ model }}
where {{ column }} is not null
  and (
    {% if min_len is not none %}
    {{ length(column) }} < {{ min_len }}
    {% endif %}
    {% if min_len is not none and max_len is not none %}
    or
    {% endif %}
    {% if max_len is not none %}
    {{ length(column) }} > {{ max_len }}
    {% endif %}
  )
{% endmacro %}


{% macro rule_regex(model, column, pattern) %}
{#
    정규식 패턴 체크 규칙.

    실패 조건: 패턴과 매칭되지 않음
#}
select *
from {{ model }}
where {{ column }} is not null
  and not {{ regex_match(column, pattern) }}
{% endmacro %}


{% macro rule_email_format(model, column) %}
{#
    이메일 형식 체크 규칙.

    실패 조건: 유효하지 않은 이메일 형식
#}
{% set email_pattern = '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$' %}

select *
from {{ model }}
where {{ column }} is not null
  and not {{ regex_match(column, email_pattern) }}
{% endmacro %}


{% macro rule_url_format(model, column) %}
{#
    URL 형식 체크 규칙.

    실패 조건: 유효하지 않은 URL 형식
#}
{% set url_pattern = '^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}' %}

select *
from {{ model }}
where {{ column }} is not null
  and not {{ regex_match(column, url_pattern) }}
{% endmacro %}


{% macro rule_uuid_format(model, column) %}
{#
    UUID 형식 체크 규칙.

    실패 조건: 유효하지 않은 UUID 형식
#}
{% set uuid_pattern = '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$' %}

select *
from {{ model }}
where {{ column }} is not null
  and not {{ regex_match(column, uuid_pattern) }}
{% endmacro %}


{% macro rule_positive(model, column) %}
{#
    양수 체크 규칙.

    실패 조건: column <= 0
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} <= 0
{% endmacro %}


{% macro rule_negative(model, column) %}
{#
    음수 체크 규칙.

    실패 조건: column >= 0
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} >= 0
{% endmacro %}


{% macro rule_not_future(model, column) %}
{#
    미래 날짜 아님 체크 규칙.

    실패 조건: column > current_date
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} > {{ current_timestamp() }}
{% endmacro %}


{% macro rule_not_past(model, column) %}
{#
    과거 날짜 아님 체크 규칙.

    실패 조건: column < current_date
#}
select *
from {{ model }}
where {{ column }} is not null
  and {{ column }} < {{ current_timestamp() }}
{% endmacro %}


{% macro rule_date_format(model, column, format) %}
{#
    날짜 형식 체크 규칙.

    실패 조건: 지정된 형식과 불일치
#}
select *
from {{ model }}
where {{ column }} is not null
  and not {{ is_valid_date_format(column, format) }}
{% endmacro %}
```

### truthound_utils.sql

유틸리티 매크로입니다.

```sql
-- macros/truthound_utils.sql

{% macro length(column) %}
{#
    문자열 길이를 반환하는 크로스-어댑터 매크로.
#}
{{ adapter.dispatch('length', 'truthound')(column) }}
{% endmacro %}


{% macro default__length(column) %}
length({{ column }})
{% endmacro %}


{% macro bigquery__length(column) %}
length({{ column }})
{% endmacro %}


{% macro snowflake__length(column) %}
length({{ column }})
{% endmacro %}


{% macro regex_match(column, pattern) %}
{#
    정규식 매칭을 수행하는 크로스-어댑터 매크로.
#}
{{ adapter.dispatch('regex_match', 'truthound')(column, pattern) }}
{% endmacro %}


{% macro default__regex_match(column, pattern) %}
{{ column }} ~ '{{ pattern }}'
{% endmacro %}


{% macro bigquery__regex_match(column, pattern) %}
regexp_contains({{ column }}, r'{{ pattern }}')
{% endmacro %}


{% macro snowflake__regex_match(column, pattern) %}
regexp_like({{ column }}, '{{ pattern }}')
{% endmacro %}


{% macro redshift__regex_match(column, pattern) %}
{{ column }} ~ '{{ pattern }}'
{% endmacro %}


{% macro databricks__regex_match(column, pattern) %}
{{ column }} rlike '{{ pattern }}'
{% endmacro %}


{% macro current_timestamp() %}
{#
    현재 타임스탬프를 반환하는 크로스-어댑터 매크로.
#}
{{ adapter.dispatch('current_timestamp', 'truthound')() }}
{% endmacro %}


{% macro default__current_timestamp() %}
current_timestamp
{% endmacro %}


{% macro bigquery__current_timestamp() %}
current_timestamp()
{% endmacro %}


{% macro snowflake__current_timestamp() %}
current_timestamp()
{% endmacro %}


{% macro limit_sample(n) %}
{#
    샘플링을 위한 LIMIT 절 생성.
#}
{{ adapter.dispatch('limit_sample', 'truthound')(n) }}
{% endmacro %}


{% macro default__limit_sample(n) %}
order by random()
limit {{ n }}
{% endmacro %}


{% macro bigquery__limit_sample(n) %}
order by rand()
limit {{ n }}
{% endmacro %}


{% macro snowflake__limit_sample(n) %}
sample ({{ n }} rows)
{% endmacro %}


{% macro is_valid_date_format(column, format) %}
{#
    날짜 형식 유효성 검증.
#}
{{ adapter.dispatch('is_valid_date_format', 'truthound')(column, format) }}
{% endmacro %}


{% macro default__is_valid_date_format(column, format) %}
{# PostgreSQL/Default: TRY_CAST 시뮬레이션 #}
(
    case
        when {{ column }} is null then true
        else {{ column }}::date is not null
    end
)
{% endmacro %}


{% macro snowflake__is_valid_date_format(column, format) %}
try_to_date({{ column }}, '{{ format }}') is not null
{% endmacro %}


{% macro bigquery__is_valid_date_format(column, format) %}
safe.parse_date('{{ format | replace("YYYY", "%Y") | replace("MM", "%m") | replace("DD", "%d") }}', {{ column }}) is not null
{% endmacro %}
```

### Adapter-Specific Optimizations

```sql
-- macros/adapters/snowflake.sql

{% macro snowflake__rule_unique(model, column) %}
{#
    Snowflake 최적화된 고유성 체크.
    QUALIFY 절 사용으로 더 효율적.
#}
select *
from {{ model }}
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{% macro snowflake__rule_regex(model, column, pattern) %}
{#
    Snowflake REGEXP 함수 사용.
#}
select *
from {{ model }}
where {{ column }} is not null
  and not regexp_like({{ column }}, '{{ pattern }}')
{% endmacro %}
```

```sql
-- macros/adapters/bigquery.sql

{% macro bigquery__rule_unique(model, column) %}
{#
    BigQuery 최적화된 고유성 체크.
    QUALIFY 절 사용.
#}
select *
from {{ model }}
qualify count(*) over (partition by {{ column }}) > 1
{% endmacro %}


{% macro bigquery__rule_email_format(model, column) %}
{#
    BigQuery REGEXP_CONTAINS 사용.
#}
select *
from {{ model }}
where {{ column }} is not null
  and not regexp_contains({{ column }}, r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
{% endmacro %}
```

---

## Manifest Parsing

### Manifest Integration Strategy

dbt의 `manifest.json`을 파싱하여 Truthound 메타데이터를 추출합니다.

```python
# Python 스크립트: manifest_parser.py (CI/분석용)

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class TruthoundTestInfo:
    """Truthound 테스트 정보"""
    model_name: str
    test_name: str
    rules: list[dict[str, Any]]
    config: dict[str, Any]


def parse_manifest(manifest_path: Path) -> list[TruthoundTestInfo]:
    """
    manifest.json에서 Truthound 테스트 정보 추출.

    Parameters
    ----------
    manifest_path : Path
        manifest.json 경로

    Returns
    -------
    list[TruthoundTestInfo]
        Truthound 테스트 목록
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    truthound_tests = []

    for node_id, node in manifest.get("nodes", {}).items():
        if node["resource_type"] != "test":
            continue

        # Truthound 테스트 필터링
        test_metadata = node.get("test_metadata", {})
        if test_metadata.get("name") != "truthound_check":
            continue

        kwargs = test_metadata.get("kwargs", {})

        truthound_tests.append(TruthoundTestInfo(
            model_name=node.get("refs", [{}])[0].get("name", "unknown"),
            test_name=node["name"],
            rules=kwargs.get("rules", []),
            config=node.get("config", {}),
        ))

    return truthound_tests


def generate_test_report(tests: list[TruthoundTestInfo]) -> dict[str, Any]:
    """
    Truthound 테스트 리포트 생성.

    Parameters
    ----------
    tests : list[TruthoundTestInfo]
        테스트 목록

    Returns
    -------
    dict
        리포트 데이터
    """
    models_with_tests = set(t.model_name for t in tests)
    total_rules = sum(len(t.rules) for t in tests)

    rules_by_type = {}
    for test in tests:
        for rule in test.rules:
            check_type = rule.get("check", "unknown")
            rules_by_type[check_type] = rules_by_type.get(check_type, 0) + 1

    return {
        "summary": {
            "total_tests": len(tests),
            "models_covered": len(models_with_tests),
            "total_rules": total_rules,
        },
        "rules_by_type": rules_by_type,
        "tests": [
            {
                "model": t.model_name,
                "rules_count": len(t.rules),
                "severity": t.config.get("severity", "error"),
            }
            for t in tests
        ],
    }


if __name__ == "__main__":
    manifest = Path("target/manifest.json")
    tests = parse_manifest(manifest)
    report = generate_test_report(tests)
    print(json.dumps(report, indent=2))
```

### Manifest 활용 예시

```bash
# CI에서 Truthound 테스트 커버리지 체크
python scripts/manifest_parser.py > truthound_report.json

# 커버리지 체크
jq '.summary.models_covered' truthound_report.json
```

---

## dbt Project Configuration

### dbt_project.yml

```yaml
# packages/dbt/dbt_project.yml
name: 'truthound'
version: '0.1.0'
config-version: 2

require-dbt-version: [">=1.6.0", "<2.0.0"]

# 패키지 설정
vars:
  truthound:
    # 기본 설정
    default_severity: 'error'
    fail_on_first: false
    sample_size: null

    # 규칙별 설정
    email_pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    uuid_pattern: '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'

# 테스트 설정
tests:
  truthound:
    +severity: error
    +tags: ['truthound', 'quality']

# 매크로 경로
macro-paths: ["macros"]
test-paths: ["tests"]

# 패키지 문서
docs-paths: ["docs"]
asset-paths: ["assets"]
```

### 사용자 프로젝트 설정

```yaml
# 사용자 dbt_project.yml
name: 'my_project'
version: '1.0.0'

vars:
  # Truthound 설정 오버라이드
  truthound:
    default_severity: 'warn'  # 기본 경고로 변경
    sample_size: 10000        # 1만 행 샘플링

# 모델별 테스트 설정
models:
  my_project:
    staging:
      +tests:
        - truthound_check:
            rules:
              - column: _loaded_at
                type: not_null
```

---

## Usage Examples

### Basic Usage

```yaml
# models/staging/schema.yml
version: 2

models:
  - name: stg_customers
    description: "Staged customer data"
    tests:
      - truthound_check:
          rules:
            - column: customer_id
              type: not_null
            - column: customer_id
              type: unique
            - column: email
              type: regex
              pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            - column: phone
              check: regex
              pattern: '^\+?[1-9]\d{1,14}$'

  - name: stg_orders
    tests:
      - truthound_check:
          rules:
            - column: order_id
              type: not_null
            - column: order_id
              check: uuid_format
            - column: amount
              type: in_range
              min: 0
            - column: quantity
              type: in_range
              min: 1
              max: 1000
            - column: status
              check: in_set
              values: ['pending', 'processing', 'shipped', 'delivered', 'cancelled']
            - column: created_at
              check: not_future
```

### Advanced Usage with Configuration

```yaml
# models/marts/schema.yml
version: 2

models:
  - name: fct_daily_sales
    description: "Daily sales fact table with strict quality requirements"
    tests:
      # 크리티컬 테스트 (실패 시 빌드 중단)
      - truthound_check:
          rules:
            - column: date_key
              type: not_null
            - column: date_key
              check: date_format
              format: 'YYYY-MM-DD'
            - column: total_revenue
              type: in_range
              min: 0
          config:
            severity: error
            tags: ['critical', 'finance']
            where: "date_key >= '2024-01-01'"

      # 경고 테스트 (실패해도 빌드 계속)
      - truthound_check:
          rules:
            - column: discount_rate
              type: in_range
              min: 0
              max: 1
            - column: customer_count
              type: in_range
              min: 0
              max: 1000000
          config:
            severity: warn
            tags: ['monitoring']
```

### Source Testing

```yaml
# models/staging/sources.yml
version: 2

sources:
  - name: raw_data
    database: raw_db
    schema: public
    tables:
      - name: raw_events
        description: "Raw event stream from Kafka"
        tests:
          - truthound_check:
              rules:
                - column: event_id
                  type: not_null
                - column: event_type
                  check: in_set
                  values: ['click', 'view', 'purchase', 'signup']
                - column: timestamp
                  type: not_null
                - column: user_id
                  check: uuid_format
```

### Using with dbt run-operation

```sql
-- ad-hoc 검증 실행
{% macro validate_table(table_name) %}

{% set rules = [
    {"column": "id", "type": "not_null"},
    {"column": "id", "type": "unique"},
] %}

{% set failures = truthound_check(ref(table_name), rules) %}

{% do log("Validation for " ~ table_name ~ ":", info=True) %}

{% set results = run_query(failures) %}
{% if results | length > 0 %}
    {% do log("FAILED: " ~ results | length ~ " rows failed validation", info=True) %}
    {% for row in results %}
        {% do log("  - " ~ row, info=True) %}
    {% endfor %}
{% else %}
    {% do log("PASSED: All validations passed", info=True) %}
{% endif %}

{% endmacro %}
```

```bash
# 실행
dbt run-operation validate_table --args '{"table_name": "stg_users"}'
```

---

## dbt Cloud Integration

### Webhook Configuration

```yaml
# .github/workflows/dbt-cloud-webhook.yml
name: dbt Cloud Quality Gate

on:
  workflow_dispatch:
    inputs:
      run_id:
        description: 'dbt Cloud Run ID'
        required: true

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - name: Get Run Results
        id: get_results
        run: |
          RESULTS=$(curl -s -H "Authorization: Token ${{ secrets.DBT_CLOUD_TOKEN }}" \
            "https://cloud.getdbt.com/api/v2/accounts/${{ secrets.DBT_CLOUD_ACCOUNT_ID }}/runs/${{ github.event.inputs.run_id }}/")
          echo "results=$RESULTS" >> $GITHUB_OUTPUT

      - name: Check Truthound Tests
        run: |
          echo '${{ steps.get_results.outputs.results }}' | jq '.data.run_results[] | select(.node.test_metadata.name == "truthound_check")'

      - name: Notify on Failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Truthound quality checks failed in dbt Cloud run ${{ github.event.inputs.run_id }}"
            }
```

### dbt Cloud Job Configuration

```yaml
# profiles.yml (dbt Cloud)
my_project:
  target: prod
  outputs:
    prod:
      type: snowflake
      account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
      user: "{{ env_var('SNOWFLAKE_USER') }}"
      password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"
      role: TRANSFORMER
      database: ANALYTICS
      warehouse: TRANSFORMING
      schema: PUBLIC
      threads: 4

# dbt Cloud Job 설정
# 1. dbt deps (패키지 설치)
# 2. dbt build --select tag:truthound (테스트 포함 빌드)
# 3. dbt run-operation generate_truthound_report (리포트 생성)
```

---

## Testing Strategy

### Test Structure

```
packages/dbt/
├── integration_tests/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── seeds/
│   │   ├── test_valid_data.csv
│   │   └── test_invalid_data.csv
│   ├── models/
│   │   ├── test_model_valid.sql
│   │   └── test_model_invalid.sql
│   └── tests/
│       ├── schema.yml
│       └── custom/
```

### Integration Test dbt_project.yml

```yaml
# integration_tests/dbt_project.yml
name: 'truthound_integration_tests'
version: '1.0.0'
config-version: 2

profile: 'integration_tests'

model-paths: ["models"]
seed-paths: ["seeds"]
test-paths: ["tests"]

vars:
  truthound:
    default_severity: 'error'
```

### Test Seeds

```csv
# seeds/test_valid_data.csv
id,email,age,status,amount
1,valid@email.com,25,active,100.50
2,another@test.org,30,inactive,200.00
3,user@domain.io,45,pending,50.25
```

```csv
# seeds/test_invalid_data.csv
id,email,age,status,amount
1,invalid-email,25,active,100.50
2,valid@email.com,-5,unknown,-50.00
3,,150,active,0
```

### Test Models

```sql
-- models/test_model_valid.sql
{{ config(materialized='view') }}

select * from {{ ref('test_valid_data') }}
```

```sql
-- models/test_model_invalid.sql
{{ config(materialized='view') }}

select * from {{ ref('test_invalid_data') }}
```

### Test Schema

```yaml
# tests/schema.yml
version: 2

models:
  # 이 테스트는 통과해야 함
  - name: test_model_valid
    tests:
      - truthound_check:
          rules:
            - column: id
              type: not_null
            - column: email
              type: regex
              pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            - column: age
              type: in_range
              min: 0
              max: 150
            - column: status
              check: in_set
              values: ['active', 'inactive', 'pending']
            - column: amount
              type: in_range
              min: 0

  # 이 테스트는 실패해야 함 (역 테스트)
  - name: test_model_invalid
    tests:
      - truthound_check:
          rules:
            - column: email
              type: regex
              pattern: "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
          config:
            severity: warn  # 실패해도 빌드 계속
```

### Running Tests

```bash
# 통합 테스트 실행
cd integration_tests

# 시드 데이터 로드
dbt seed

# 모델 빌드
dbt run

# 테스트 실행
dbt test

# 특정 테스트만 실행
dbt test --select truthound_check

# 상세 로그
dbt test --debug
```

### CI Configuration

```yaml
# .github/workflows/test-dbt.yml
name: Test dbt Package

on:
  push:
    paths:
      - 'packages/dbt/**'
  pull_request:
    paths:
      - 'packages/dbt/**'

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dbt
        run: pip install dbt-postgres

      - name: Run Integration Tests
        working-directory: packages/dbt/integration_tests
        env:
          DBT_PROFILES_DIR: .
        run: |
          dbt deps
          dbt seed
          dbt run
          dbt test

      - name: Generate Docs
        working-directory: packages/dbt/integration_tests
        run: dbt docs generate

      - name: Upload Test Results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: dbt-artifacts
          path: packages/dbt/integration_tests/target/
```

### profiles.yml for CI

```yaml
# integration_tests/profiles.yml
integration_tests:
  target: ci
  outputs:
    ci:
      type: postgres
      host: localhost
      port: 5432
      user: postgres
      password: postgres
      dbname: postgres
      schema: truthound_test
      threads: 4
```

---

## Package Release

### dbt Hub 배포 프로세스

1. **버전 업데이트**
   ```yaml
   # dbt_project.yml
   version: '0.2.0'  # 버전 증가
   ```

2. **태그 생성**
   ```bash
   git tag dbt-v0.2.0
   git push origin dbt-v0.2.0
   ```

3. **dbt Hub 등록**
   - [hub.getdbt.com](https://hub.getdbt.com) 에서 패키지 등록
   - GitHub 저장소 연결
   - 버전 자동 동기화

### 버전 호환성 매트릭스

| truthound-dbt | dbt-core | Snowflake | BigQuery | Redshift | Postgres |
|---------------|----------|-----------|----------|----------|----------|
| 0.1.x | >= 1.6.0 | >= 1.6.0 | >= 1.6.0 | >= 1.6.0 | >= 1.6.0 |
| 0.2.x | >= 1.7.0 | >= 1.7.0 | >= 1.7.0 | >= 1.7.0 | >= 1.7.0 |

---

## References

- [dbt Package Development](https://docs.getdbt.com/docs/build/packages)
- [dbt Generic Tests](https://docs.getdbt.com/docs/build/tests)
- [dbt Jinja Macros](https://docs.getdbt.com/docs/build/jinja-macros)
- [Truthound Documentation](https://truthound.dev/docs)

---

*이 문서는 truthound-dbt 패키지의 완전한 구현 명세입니다.*
