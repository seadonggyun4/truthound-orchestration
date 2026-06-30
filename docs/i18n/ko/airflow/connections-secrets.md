!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Airflow Connections and Secrets
---

# Airflow Connections and Secrets

Truthound's Airflow integration is designed to follow Airflow's connection model instead of inventing a second secret registry. The `DataQualityHook` resolves connection-backed execution when a source requires credentials, while local-file onboarding stays zero-config.

## Who This Is For

- Airflow operators standardizing connection IDs across DAGs
- teams moving from local-file 검증 to warehouse-backed 검증
- platform engineers deciding where Truthound should read credentials

## When To Use It

Use this page when:

- a DAG moves from `data_path=` to `sql=`
- 검증 reads from Postgres, Snowflake, or another connection-backed source
- you want Airflow Variables and Secrets Backends to remain the source of truth

## Prerequisites

- `truthound-오케스트레이션[airflow]` installed on the Airflow workers
- an Airflow connection available for the target system
- a DAG that passes either `data_path=` or `sql=` into a Truthound operator or sensor

## Minimal Quickstart

Use the default connection contract when SQL execution needs credentials:

```python
from airflow import DAG
from truthound_airflow import DataQualityCheckOperator

with DAG("warehouse_quality", schedule="@daily", catchup=False) as dag:
    validate_users = DataQualityCheckOperator(
        task_id="validate_users",
        sql="select * from analytics.dim_users",
        connection_id="warehouse_primary",
        rules=[
            {"column": "id", "check": "not_null"},
            {"column": "email", "check": "email_format"},
        ],
    )
```

For hook-first loading, use the same connection boundary explicitly:

```python
from truthound_airflow import DataQualityHook

hook = DataQualityHook(connection_id="warehouse_primary")
data = hook.load_data(sql="select * from analytics.dim_users")
```

## Production Pattern

Treat Airflow connections as the control plane for credentials and keep Truthound focused on 검증 semantics.

| Concern | Recommended Airflow Boundary | Why |
|--------|-------------------------------|-----|
| Warehouse credentials | Airflow Connection | reuses existing governance and rotation |
| Runtime flags | DAG code or env vars | keeps changes versioned with the DAG |
| Token or password storage | Secrets Backend or masked extras | avoids inline secrets in DAG definitions |
| 검증 rules | DAG code or imported rule modules | keeps quality intent visible in code review |

Recommended rollout sequence:

1. validate with local files first
2. introduce `connection_id` only when the source requires it
3. move secrets into the Airflow backend before enabling SQL checks in production
4. keep the same connection IDs across operators and sensors to avoid split routing

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| operator works with `data_path` but fails with `sql` | no Airflow connection provided | add `connection_id` and verify the connection exists |
| 검증 unexpectedly attempts network access | source resolution detected a connection-backed URI or SQL path | confirm the source shape in the task arguments |
| secrets differ between workers | environment-only credentials are not synchronized | move them to Airflow Connections or a Secrets Backend |
| result differs between tasks using the same warehouse | two connection IDs point at different environments | standardize connection naming by environment |

## Related Pages

- [Airflow Overview](index.md)
- [Hooks](hooks.md)
- [Install and Compatibility](install-compatibility.md)
- [Shared Runtime Source Resolution](../common/source-resolution.md)
- [Enterprise Secrets](../enterprise/secrets.md)
