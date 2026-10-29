!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Compatibility
---

# 호환성

This page defines the official compatibility contract for `truthound-오케스트레이션 3.x`: what Truthound line it supports, which host versions are exercised in CI, and how to interpret "minimum supported" versus "primary supported" host lanes.

## Version Line

`truthound-오케스트레이션 3.x` supports `Truthound 3.x` only.

| Package Line | Supported Truthound Range | Status |
|--------------|---------------------------|--------|
| `truthound-오케스트레이션 3.x` | `>=3.0,<4.0` | Active first-party line |
| `truthound-오케스트레이션 1.x` | Pre-Truthound-3 lines | Legacy |

## How To Read This Contract

- "minimum supported" means a release-blocking host-plus-Python tuple still exercised in CI
- "primary supported" means the main documentation and default examples track that tuple
- compatibility is defined at the surface level, not only by package version names
- convenience aggregates such as `truthound-오케스트레이션[all]` are useful for canaries but are not the primary release contract

## Platform Expectations

| Platform | Runtime Expectation |
|----------|---------------------|
| Airflow | `truthound-오케스트레이션[airflow]` with `Truthound 3.x` |
| Dagster | `truthound-오케스트레이션[dagster]` with `Truthound 3.x` |
| Prefect | `truthound-오케스트레이션[prefect]` with `Truthound 3.x` |
| Mage | `truthound-오케스트레이션[mage]` inside a Mage runtime that already provides `mage-ai` |
| Kestra | `truthound-오케스트레이션[kestra]` with `Truthound 3.x` |
| dbt | `truthound-오케스트레이션[dbt]` with the bundled first-party `truthound_dbt` package |

Release-blocking compatibility and security guarantees apply to these per-surface installs. `truthound-오케스트레이션[all]` remains available as a convenience aggregate and nightly canary surface.

## Core 3.3.0 검증

이번 수정이 포함된 소스에서 빌드한 패키지를 공개 배포된 `truthound==3.3.0`
wheel과 함께 Python 3.11.14 / macOS ARM64에서 검증했습니다. 프로젝트는
계속 Alpha 단계입니다. 이 결과는 이번 수정본으로 만든 wheel에 적용되며,
기존에 배포된 orchestration 3.0.3 패키지를 소급하여 인증하지 않습니다.
이번 소스 업데이트는 새 패키지 발행을 의미하지 않습니다.

상세 이슈가 생략된 실패 결과를 Core 및 오케스트레이션 직렬화 전후에도
실패로 유지합니다. Core의 drift 모듈과 중첩된 판정 구조, ML의 행 단위
개수·인덱스를 변환합니다. Prefect는 호스트가 사용하는 Pydantic 세대에
맞는 필드를 사용하며, Kestra는 공통 실패 이름·심각도를 출력 계약으로
명시적으로 변환합니다.

같은 변경에서 비동기 집계 내부의 동기 헬스체크 실행과 Kestra 슬롯 설정의
직렬화도 수정했습니다. 비동기 테스트는 자체 이벤트 루프를 생성·정리합니다. 동적 메타데이터가
변환 메서드를 계속 만들어 내는 경우에도 직렬화가 재귀에 빠지지 않도록 했습니다.

| 표면 | 검증한 버전·실행 |
| --- | --- |
| Core | 공개 3.3.0 wheel; 검증·프로파일링·학습·실패 결과 직렬화 왕복 |
| Airflow | 3.2.0; 실제 Operator에서 로컬 CSV 검증 및 XCom 출력 캡처 |
| Dagster | 1.12.18; in-process job 내부 Resource 실행 |
| Prefect | 3.6.29; 실제 Block과 Core 실행, 별도 프로세스 회귀 테스트 |
| Mage | 배포 패키지 Transformer와 실제 Core 실행; Mage 서버는 기동하지 않음 |
| Kestra | Python SDK 1.3.0 및 script executor; Kestra 서버는 기동하지 않음 |
| dbt | Core 1.10.20 / postgres adapter 1.10.0; 독립 PostgreSQL 14.20에서 compile·seed·run·test·매크로 실행 |
| 고급 Core 기능 | NumPy/SciPy/scikit-learn 환경에서 실제 KS 드리프트 및 Isolation Forest·Z-score·Ensemble 이상치 탐지 |

dbt 데이터 검증은 통과 12건, 불량 seed에 대한 예상 경고 7건, 오류 0건,
스킵 0건이었습니다. Manifest coverage는 기존 80% 기준에서 100%였습니다.
이는 PostgreSQL 실행 검증이며 Snowflake·BigQuery·Redshift·Databricks의
실제 연결 검증은 포함하지 않습니다. 호스트 최소 버전, 다른 Python 버전,
스케줄러·서버 전체 배포도 이번 로컬 검증 범위에 포함하지 않습니다.
Core 3.3은 어댑터의 기존 `lof` 이름에 해당하는 클래스를 내보내지 않으므로
요청하면 명시적인 클래스 미지원 오류가 발생합니다. 위의 세 ML 탐지기가
실행 확인된 선택지이며, 고급 기능에는 선택 ML 의존성이 필요합니다.

각 호스트 버전과 이번 수정본 wheel을 별도 환경에 설치한 뒤 재현할 수 있습니다.

```bash
python -m build
python -m pip install "truthound==3.3.0" dist/truthound_orchestration-3.0.3-py3-none-any.whl
python -I scripts/ci/verify_core_runtime.py base
# NumPy, SciPy, scikit-learn 설치 후:
python -I scripts/ci/verify_core_runtime.py base --advanced
# 해당 호스트 의존성을 설치한 환경에서:
python -I scripts/ci/verify_core_runtime.py prefect
```

나머지 실행 이름은 `airflow`, `dagster`, `mage`, `kestra`입니다. 실행기는
두 패키지를 소스 트리가 아닌 site-packages에서 불러왔는지 검사합니다.
dbt는 독립 테스트 DB를 대상으로
`packages/dbt/scripts/run_first_party_suite.py --mode all --target postgres`와
manifest coverage 검사를 실행하세요. 기존 선택 테스트의 스킵은 해당 엔진의
호환성 통과 근거가 아닙니다.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
## Generated CI Support Matrix

Truthound release line: `>=3.0,<4.0`

| Lane | Airflow | Prefect | Dagster | Mage | Kestra | dbt Compile | dbt Execute |
|------|---------|---------|---------|------|--------|-------------|-------------|
| PR | `3.2.0` on `Python 3.12` | `3.6.29` on `Python 3.12` | `1.12.18` on `Python 3.12` | Primary host smoke | Primary host smoke | `postgres` | No |
| Main | `3.2.0` on `Python 3.12` | `3.6.29` on `Python 3.12` | `1.12.18` on `Python 3.12` | Primary host smoke | Primary host smoke | `postgres` | No |
| Release | `2.6.0` on `Python 3.11`, `3.2.0` on `Python 3.12` | `2.14.0` on `Python 3.11`, `3.6.29` on `Python 3.12` | `1.5.0` on `Python 3.11`, `1.12.18` on `Python 3.12` | Primary host smoke | Primary host smoke | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |
| Nightly | `3.2.0` on `Python 3.12` | `3.6.29` on `Python 3.12` | `1.12.18` on `Python 3.12` | Primary host smoke + `mage-ai` runtime canary | Primary host smoke | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |

## Supported Host Version Anchors

| Platform | Minimum Supported | Primary Supported |
|----------|-------------------|-------------------|
| Airflow | `2.6.0` on `Python 3.11` | `3.2.0` on `Python 3.12` |
| Prefect | `2.14.0` on `Python 3.11` | `3.6.29` on `Python 3.12` |
| Dagster | `1.5.0` on `Python 3.11` | `1.12.18` on `Python 3.12` |
| Kestra | `1.3.0` on `Python 3.12` | `1.3.0` on `Python 3.12` |
| dbt | `dbt-core>=1.10.0,<1.11.0` + `dbt-postgres>=1.10.0,<1.11.0` | `dbt-core 1.10.20` + `dbt-postgres 1.10.0` |

## Security Audit Surfaces

| Surface | Install Surface | Release Blocking |
|---------|-----------------|------------------|
| `base` | `truthound-오케스트레이션` | Yes |
| `prefect` | `truthound-오케스트레이션[prefect]` | Yes |
| `dagster` | `truthound-오케스트레이션[dagster]` | Yes |
| `dbt` | `truthound-오케스트레이션[dbt]` | Yes |
| `kestra` | `truthound-오케스트레이션[kestra]` | Yes |
| `opentelemetry` | `truthound-오케스트레이션[opentelemetry]` | Yes |
| `airflow` | `truthound-오케스트레이션[airflow]` | Push/release advisory only |
| `all` | `truthound-오케스트레이션[all]` | Nightly canary only |

First-party release guarantees apply to per-surface installs. `truthound-오케스트레이션[all]` remains available as a convenience aggregate and nightly canary surface.
Minimum compatibility guarantees are validated as tested host-plus-Python tuples, not as host versions on every Python runtime.
Airflow security audits install with the official Airflow constraints file for the pinned version and Python version.
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

## 호환성 APIs

Use the shared runtime helpers when you need explicit 검증 inside platform code or deployment checks:

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
