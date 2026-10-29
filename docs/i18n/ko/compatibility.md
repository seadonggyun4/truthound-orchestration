!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Compatibility
---

# 호환성

This page defines the official compatibility contract for `truthound-orchestration 3.x`: what Truthound line it supports, which host versions are exercised in CI, and how to interpret "minimum supported" versus "primary supported" host lanes.

## 스케줄러 보안 기준 (2026-10-29)

Core 3.3의 기본 병렬 검증이 사용하는 NumPy를 companion 실행 의존성에
명시했습니다. 최소 Mage companion 설치에서 누락을 재현했고, 다른 호스트의
전이 의존성에 기대지 않는 실제 Core 정상·불량 데이터 회귀 테스트를 추가했습니다.

| 플랫폼 | 최소 / 기본 버전 | 검증 및 제한 |
|---|---|---|
| Prefect | 3.9.0 / 3.9.0 | Python 3.11 / 3.12, API 서버·Runner에서 Core 3.3.0 실행과 COMPLETED / FAILED 저장 |
| Dagster | 1.13.1 / 1.13.27 | Python 3.11 / 3.12, 웹서버·daemon·GraphQL 제출과 SUCCESS / FAILURE 실행 |
| Kestra Python SDK | 1.3.0 / 2.0.0 | Python 3.11 / 3.12, 별도 버전인 Java 서버 2.0.8에서 모두 실행 |
| Kestra Java 서버 | 2.0.8 | 인증된 실제 실행, 품질 실패, IP·DNS 차단 목록 회귀 검증 |
| Mage 호스트 | 보안 지원 가능한 배포 버전 없음 | 0.9.80은 별도 Core 3.3 인터프리터로 기능만 검증 |

아래의 이전 Prefect·Dagster 버전 기록은 과거 기능 검증이며 현재 지원 기준이 아닙니다.
루트 extra·all extra·독립 Prefect/Dagster 패키지의 의존성 하한을 일치시켰습니다.
Prefect 3.6.29는 새 의존성의 보안 검사를 통과해도 최신 FastAPI에서
제거된 Body(min_items=...) 때문에 서버 초기화가 실패합니다.
3.9.0은 기존 별도 프로세스 회귀 테스트와 실제 서버 실행을 통과합니다.
구형 FastAPI·Starlette 제약을 제거했고, 최소·기본 Python 환경 모두 정상 설치 후
pip check와 pip-audit를 실행합니다.

Dagster 1.13.1은 동적 파티션 SQL 주입 문제
[GHSA-mjw2-v2hm-wj34](https://github.com/dagster-io/dagster/security/advisories/GHSA-mjw2-v2hm-wj34)를 수정합니다.
Kestra 2.0.7에서 DNS 이름을 이용한 HTTP 차단 목록 우회를 로컬 합성 HTTP 서버로
재현했습니다. 2.0.8에서는 두 SDK 모두 IP·DNS 요청이 실패하고 대상 서버에 요청이
도달하지 않았습니다.
[GHSA-f5h6-rg96-px74](https://github.com/kestra-io/kestra/security/advisories/GHSA-f5h6-rg96-px74)를 참고하세요.
CI는 2.0.8 공식 서버 파일의 SHA-256
fcc48c9812e6b5a0534f80b5bff9fddcc021280115434900da43870152f8f8f8을 확인합니다.
Python SDK 검사는 Java 의존성 검사가 아니며, 전체 Java SBOM 취약점 검사를
완료했다고 주장하지 않습니다.

Mage 최신 0.9.80은 PyJWT==2.6.0, aiohttp==3.10.0,
cryptography==42.0.4, Werkzeug==3.0.3 등 취약한 전이 의존성을 고정합니다.
정상 resolver는 mage-ai==0.9.80과 PyJWT>=2.15.0의 동시 설치를 거부하며,
상향할 최신 패치 배포판이 없습니다. 실제 스케줄러의 정상 데이터 완료·불량 데이터
실패는 확인했지만 보안 지원을 의미하지 않습니다. 인터프리터 분리는 Typer 충돌만
해결합니다. 보안 준수 환경으로 배포하거나 --no-deps로 의존성을 우회하지 마세요.
업스트림의 호환 가능한 패치 의존성 배포가 필요합니다.
기존 차단형 보안 게이트는 유지되며 Mage 호스트 보안은 미해결입니다.

## Version Line

`truthound-orchestration 3.x` supports `Truthound 3.x` only.

| Package Line | Supported Truthound Range | Status |
|--------------|---------------------------|--------|
| `truthound-orchestration 3.x` | `>=3.0,<4.0` | Active first-party line |
| `truthound-orchestration 1.x` | Pre-Truthound-3 lines | Legacy |

## How To Read This Contract

- "minimum supported" means a release-blocking host-plus-Python tuple still exercised in CI
- "primary supported" means the main documentation and default examples track that tuple
- compatibility is defined at the surface level, not only by package version names
- convenience aggregates such as `truthound-orchestration[all]` are useful for canaries but are not the primary release contract

## Platform Expectations

| Platform | Runtime Expectation |
|----------|---------------------|
| Airflow | `truthound-orchestration[airflow]` with `Truthound 3.x` |
| Dagster | `truthound-orchestration[dagster]` with `Truthound 3.x` |
| Prefect | `truthound-orchestration[prefect]` with `Truthound 3.x` |
| Mage | `truthound-orchestration[mage]` with a separate Core interpreter for Mage 0.9.80 (see below) |
| Kestra | `truthound-orchestration[kestra]` with `Truthound 3.x` |
| dbt | `truthound-orchestration[dbt]` with the bundled first-party `truthound_dbt` package |

Release-blocking guarantees apply to the declared audited surfaces; Mage host security remains unsupported. `truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.

## 보안 의존성 기준

현재 dbt Core 지원 범위는 **`>=1.11.14,<1.12.0`**이며 기본 버전은 1.11.16,
dbt-postgres는 1.10.0을 유지합니다. Python extra와 SQL 패키지 모두 이전 Core
버전을 거부합니다. 호환 Core도 취약 파서 설치를 허용할 수 있으므로
`sqlparse>=0.6.0,<0.7.0`을 함께 설치해야 합니다. 앞서 검증한 1.10.9/1.10.20은
과거 기능 검증 기록이며 현재 보안 지원 기준은 아닙니다.

[dbt 공식 보안 공지](https://github.com/dbt-labs/dbt/security/advisories/GHSA-pxwq-479r-hhwr)에
따르면 기존 파서 상한이 SQL 컴파일 경로의 서비스 거부 수정 버전 설치를 막습니다.
Core 1.11.14는 이 상한을 완화합니다. 1.10 계열은 수정 파서를 정상적으로 설치할
수 없으므로 `--no-deps`로 패키지 메타데이터를 우회하지 마세요.

Redshift 컴파일 검증은 `dbt-redshift>=1.11.1,<1.12.0`을 사용합니다.
1.11.0까지는 adapter 자체가 `sqlparse<0.6.0`으로 제한하므로 dbt Core만
올려서는 해결되지 않습니다. PostgreSQL·Snowflake·BigQuery·Databricks는
기존 1.10 adapter 검증 범위를 유지합니다.

OpenTelemetry API·SDK·두 OTLP exporter의 최소 버전은 **1.45.1**입니다.
[OpenTelemetry 공식 보안 공지](https://github.com/open-telemetry/opentelemetry-python/security/advisories/GHSA-xcq7-p3pf-p88h)의
비정상 `tracestate` 헤더 파싱 비용 문제를 해결합니다. dbt 호스트의 전이 의존성도
API를 사용하므로 dbt extra에 같은 최소 버전을 적용했습니다. HTTP gzip 전송은
문자열 대신 SDK의 압축 enum을 전달하도록 수정했습니다.

보안 CI는 선택한 extra 전체를 정상 설치하고 `pip check` 후 `pip-audit`를
실행합니다. 프로젝트 설치에 `--no-deps`를 사용하지 않으며 취약점 예외도 추가하지
않았습니다. Core 3.3.0과 dbt 1.11.14·1.11.16, OpenTelemetry 1.45.1의 독립 환경은
모두 의존성 검사 통과 및 알려진 취약점 0건을 확인했습니다. OpenTelemetry 감사는
기존 Protobuf 5 제약을 유지하며 dbt는 Protobuf 6을 사용합니다.

회귀 검증은 네이티브 dbt CTE 삽입, 패키지 보안 최소 버전, 비정상 헤더 파싱,
공개 exporter factory와 provider의 실제 HTTP·gRPC gzip 추적·메트릭 전송을
포함합니다. dbt 두 버전 모두 PostgreSQL 실행과 기존 manifest coverage 기준을
검증합니다. 수정 소스와 빌드 산출물에 대한 결과이며 새 패키지 발행이나 다른
모든 호스트의 보안 감사 통과를 의미하지 않습니다.

## Core 3.3.0 검증

이번 수정이 포함된 소스에서 빌드한 패키지를 공개 배포된 `truthound==3.3.0`
wheel과 함께 검증했습니다. 초기 어댑터 검증은 Python 3.11.14 / macOS ARM64이며,
후속 실제 서버·최소 버전 검증은 아래 별도 표에 기록했습니다. 프로젝트는
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
| Mage | 배포 패키지 Transformer와 실제 Core 실행; 후속 실제 서버 결과는 아래 참조 |
| Kestra | Python SDK 1.3.0 및 script executor; 후속 실제 서버 결과는 아래 참조 |
| dbt | Core 1.10.20 / postgres adapter 1.10.0; 독립 PostgreSQL 14.20에서 compile·seed·run·test·매크로 실행 |
| 고급 Core 기능 | NumPy/SciPy/scikit-learn 환경에서 실제 KS 드리프트 및 Isolation Forest·Z-score·Ensemble 이상치 탐지 |

dbt 데이터 검증은 통과 12건, 불량 seed에 대한 예상 경고 7건, 오류 0건,
스킵 0건이었습니다. Manifest coverage는 기존 80% 기준에서 100%였습니다.
이는 PostgreSQL 실행 검증이며 Snowflake·BigQuery·Redshift·Databricks의
실제 연결 검증은 포함하지 않습니다. 후속 최소 버전 및 스케줄러·서버 실행 결과는 아래에 별도로 기록했습니다.
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

## 실제 서버 및 최소 호스트 검증

이 절은 이전 검증 이력입니다. Airflow 2.6.0·3.2.0은 더 이상 지원하지 않으며
현재 지원 범위는 아래 Airflow 보안 기준과 생성된 CI 표를 따릅니다.

macOS ARM64에서 공개 Core 3.3.0과 아래 조합의 실제 스케줄러·서버 실행을
확인했습니다. 각 서버에 합성 작업을 제출하여 정상 입력은 성공, null 입력은
실패 상태로 저장되는지 검증했습니다. 운영 배포가 아닌 격리 검증 환경입니다.

| 호스트 | 버전 | Python | 검증한 실행 |
| --- | --- | --- | --- |
| Airflow | 2.6.0 / 3.2.0 | 3.11.14 / 3.12.11 | 스케줄러, 웹/API 서버, 작업 프로세스, 성공·실패 XCom 내용 |
| Dagster | 1.5.0 / 1.12.18 | 3.11.14 / 3.12.11 | 웹서버, daemon, GraphQL 제출, 자식 실행 프로세스 |
| Prefect | 2.14.0 / 3.6.29 | 3.11.14 / 3.12.11 | API 서버, served flow, Runner의 COMPLETED / FAILED |
| Mage | 0.9.80 | 3.11.14 / 3.12.11; Core 인터프리터 분리 | API 스케줄과 블록 실행의 completed / failed |
| Kestra | 서버 2.0.7; Python SDK 1.3.0 | 3.11.14; 서버 Java 25 | standalone 서버와 process worker의 SUCCESS / FAILED |
| dbt | Core 1.10.9 / 1.10.20; postgres adapter 1.10.0 | 3.12.11 | PostgreSQL 14.20 compile·seed·run·test·매크로 |

dbt 두 버전 모두 통과 12건, 예상 경고 7건, 오류·스킵 0건, manifest coverage
100%였습니다. Airflow 최소·기본 버전의 패키지 테스트는 각각 273개, Prefect는
각각 169개, Dagster 최소·기본 버전은 각각 200개 통과했습니다. 생성된 매트릭스의 Kestra 버전은
Java 서버 최소 버전이 아닌 **Python SDK**입니다. 서버는 2.0.7만 검증했습니다.
Mage는 Python 3.11과 3.12에서 각각 정상적으로 의존성을 해결한 호스트·Core 환경으로
서버 실행을 통과했습니다. Python 3.12 실행은 최종 재빌드 wheel을 사용했습니다.

선택한 호스트와 오케스트레이션 wheel을 각각 독립 환경에 설치하여 재현합니다.

```bash
python scripts/ci/verify_host_server.py airflow --output-dir /tmp/airflow-verification
python scripts/ci/verify_host_server.py dagster --output-dir /tmp/dagster-verification
python scripts/ci/verify_host_server.py prefect --output-dir /tmp/prefect-verification
python scripts/ci/verify_mage_server.py --host-python /path/to/mage/bin/python \
  --core-python /path/to/core/bin/python --output-dir /tmp/mage-verification
```

출력 폴더는 새 경로여야 합니다. 실행기는 로컬 증거를 보존하고 자신이 시작한
프로세스를 종료합니다. DB·생성 설정·Mage 원본 서버 로그는 비공개로 보관하며,
CI에는 선별된 요약 및 테스트 로그만 업로드합니다.

설치 결함과 검증 조건은 다음과 같습니다.

- Airflow 2.6.0의 `constraints-3.11.txt`는 배포되지 않아 기존 URL이 404입니다.
  공개된 3.10 의존성 핀으로 Python 3.11을 검증합니다. upstream의 Python 3.11
  지원을 주장하는 것은 아닙니다. macOS에서 기본 fork 경로가 네이티브
  `setproctitle` 내부에서 멈추므로 검증 실행기는 Airflow의 custom task-runner
  확장으로 별도 인터프리터를 실행합니다. Linux는 기본 실행기를 유지합니다.
  Airflow 3.2.0은 공식 Python 3.12 제약으로 Pydantic 비호환 설치를 방지합니다.
- Prefect 2.14.0 서버에 필요한 `cachetools`를 Prefect extra에 추가했습니다.
  구형 Starlette 조합은 `fastapi<0.100`, Prefect 3.6.29는 새 버전에서 제거된
  `Body(min_items=...)` 때문에 `fastapi<0.130`을 사용합니다. 호스트별 제약은
  `ci/support-matrix.toml`에 기록했습니다.
- dbt-core 1.10.0과 dbt-postgres 1.10.0은 Protobuf 주 버전이 충돌합니다.
  초기 검증에서는 1.10.9가 설치 가능한 최소 버전이었으나, 현재는 위의 보안
  기준에 따라 1.11.14 이상으로 지원 범위를 올렸습니다.
- Mage 0.9.80의 `typer[all]==0.9.0`과 Core 3.3.0의 `typer>=0.12.0`은 충돌하여
  **같은 인터프리터 설치는 지원하지 않습니다**. 검증한 Mage 블록은 정상적으로
  의존성을 해결한 별도 Core 인터프리터에서 공식 Transformer를 실행하고 실패를
  Mage에 전달합니다. `--no-deps`로 충돌을 우회하지 마세요. 이 버전 조합에는
  프로세스 분리가 필요합니다.
- 다른 호스트 버전·클라우드 데이터웨어하우스 실연결·운영 인프라까지 인증하는
  결과는 아닙니다. 프로젝트는 계속 Alpha 단계입니다.

## Airflow 보안 기준

기본 호스트와 필수 보안 검사는 Airflow **3.3.2**, Truthound **3.3.0**을 사용합니다.
기존 검사는 Airflow 3.3.0에 고정되어 upstream advisory 15건이 발생했습니다.
3.3.2에 수정이 포함되어 있으며 취약점 제외 목록은 사용하지 않습니다.
[CVE-2026-82355](https://lists.apache.org/thread/3p7zpdvv40tn01m0xk5mt2rxg88mw8w1)는
캐시된 세션 쿠키의 사용자 정보가 명시적인 Bearer 토큰보다 우선하던 결함입니다.
실제 인증 코드의 회귀 테스트는 Bearer 우선순위, 잘못된 토큰 거부, 정상 쿠키 캐시,
신뢰할 수 없는 캐시 거부를 확인하며 3.3.0에서 실패하고 3.3.2에서 통과합니다.

Airflow의 최소·기본 지원 버전은 모두 **3.3.2**입니다. 최소 Python 조합은
**3.11**, 기본 Python 조합은 **3.12**이며 같은 보안 제약으로 정상 의존성 해석을 수행합니다.
루트 패키지의 `airflow`·`all` extra와 `truthound-airflow` provider 모두
`apache-airflow>=3.3.2`를 요구합니다. 2.6.0을 포함한 이전 버전의 기능 호환성
지원도 종료했습니다. 아래 이전 검증 기록은 현재 지원 범위를 뜻하지 않습니다.

필수 보안 검사는 전체 Airflow extra를 아래 보안 하한과 함께
정상 해석해 설치한 뒤 `pip check`, 실제 인증 회귀 테스트, 제외 항목 없는 `pip-audit`를
실행합니다. 호스트 호환성 CI는 동일한 보안 제약을 적용하고 실제 서버와
성공·실패 작업의 XCom 저장 결과를 검증합니다.


최소 버전 상향 이전의 2026-10-29 검증에서 각 호스트의 Airflow 어댑터 테스트 273개, 3.3.2의 실제
인증 회귀 테스트 4개와 2.6.0/Python 3.11·3.3.2/Python 3.12의 실제
스케줄러·웹/API 서버·작업 실행을 통과했습니다. Core 3.3.0으로 두 서버 모두
`good=success`, `bad=failed`와 일치하는 XCom을 저장했습니다. 3.3.2 전체
환경의 알려진 취약점은 0건입니다. 초기 동시 로컬 실행에서는 2.6.0 워커 종료와
3.3.2 실행 API 시간 초과가 발생했으며 실패 기록을 보존했습니다. 새 격리 환경의
재실행은 호스트 코드 변경이나 검증 조건 완화 없이 통과했습니다.

2.6 전용 `pytest<9.2`·`packaging<24` 제약은 제거했습니다. 테스트 도구 설치 후
`pip check`는 유지하며, 같은 Airflow 버전을 검증하는 두 작업의 산출물 이름에
검증 조합과 Python 버전을 포함해 업로드 충돌을 방지합니다. 버전 하한 거부,
Python별 실행 조합, 산출물 이름 구분을 회귀 테스트로 검증합니다.

### 전이 의존성 보안 하한

Airflow 3.3.2 공식 제약 스냅샷 검사에서 5개 패키지에 advisory 보고 9건(certifi
중복 보고 포함)이 발견됐습니다. 패키지 extra·provider 메타데이터·실행 CI·보안
검사가 다음 하한을 함께 적용하도록 수정했습니다.

| 의존성 | 최소 버전 |
| --- | --- |
| certifi | 2026.10.19 |
| dnspython | 2.9.0 |
| opentelemetry-api | 1.45.1 |
| PyJWT | 2.15.0 |
| urllib3 | 2.8.0 |
| cryptography | 46.0.5 (기존 보안 하한) |

인증서 신뢰, DNS·tracestate 파싱, JWT 검증, HTTP 프록시·스트리밍 결함을
수정한 버전입니다. 공식 [certifi 릴리스](https://github.com/certifi/python-certifi/releases/tag/2026.10.19),
[dnspython 릴리스](https://github.com/rthalley/dnspython/releases/tag/v2.9.0),
[PyJWT 변경 이력](https://pyjwt.readthedocs.io/en/stable/changelog.html#v2-15-0),
[urllib3 변경 이력](https://urllib3.readthedocs.io/en/stable/changelog.html#id1)과
위 OpenTelemetry 공지를 참고하세요. 이전 공식 제약 스냅샷은 이 하한과 충돌하므로
CI는 저장소의 보안 제약으로 정상 설치합니다. 취약점 제외나 `--no-deps`는 사용하지 않습니다.

현재 하한인 Airflow 3.3.2와 Core 3.3.0을 Python 3.11·3.12에서 검증했습니다.
각 환경에서 어댑터·헤더 회귀 테스트 274개, 실제 인증 회귀 테스트 4개와
`pip check`가 통과했으며 전체 설치 환경의 알려진 취약점은 0건입니다.
실제 스케줄러·API·작업 실행은 `good=success`, `bad=failed`와 일치하는 XCom을
저장했습니다. Python 3.12 서버는 빌드한 wheel을 site-packages에서 로드했습니다.
두 배포 패키지의 메타데이터와 실제 의존성 해석은 Airflow 3.3.1을 거부합니다.
wheel·sdist 검증, 비공개 파일 제외 검사, strict 문서 빌드와 양언어 링크 검증도
통과했습니다. 수정 소스·산출물의 검증이며 새 패키지를 발행한 것은 아닙니다.

Linux Python 3.11 러너에는 setuptools 79.0.1이 포함되어 있어 실행 의존성과
별도로 보안 검사에 보고됐습니다. CI는 호스트 설치 전에 `setuptools>=84.1.0`으로
부트스트랩 도구를 업데이트하며 해당 보고를 제외하지 않습니다.
[setuptools 공식 보안 공지](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w)를
참고하세요. 기존 환경의 부트스트랩 도구도 함께 업데이트해야 합니다.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
## Generated CI Support Matrix

Truthound release line: `>=3.0,<4.0`

These are independently selected workflow presets, not an aggregate dispatch. Release Gate verifies completed same-commit runs; it does not start platform jobs.

| Lane | Airflow | Prefect | Dagster | Mage | Kestra | dbt Compile | dbt Execute |
|------|---------|---------|---------|------|--------|-------------|-------------|
| PR | `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.12` | `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `2.0.0` on `Python 3.12` | `postgres` | No |
| Main | `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.12` | `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `2.0.0` on `Python 3.12` | `postgres` | No |
| Release | `3.3.2` on `Python 3.11`, `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.11`, `3.9.0` on `Python 3.12` | `1.13.1` on `Python 3.11`, `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `1.3.0` on `Python 3.11`, `2.0.0` on `Python 3.12` | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |
| Nightly | `3.3.2` on `Python 3.12` | `3.9.0` on `Python 3.12` | `1.13.27` on `Python 3.12` | Functional only; host security unsupported | `2.0.0` on `Python 3.12` | `postgres`, `snowflake`, `bigquery`, `redshift`, `databricks` | Yes (`postgres`) |

## Supported Host Version Anchors

| Platform | Minimum Supported | Primary Supported |
|----------|-------------------|-------------------|
| Airflow | `3.3.2` on `Python 3.11` | `3.3.2` on `Python 3.12` |
| Prefect | `3.9.0` on `Python 3.11` | `3.9.0` on `Python 3.12` |
| Dagster | `1.13.1` on `Python 3.11` | `1.13.27` on `Python 3.12` |
| Kestra SDK | `1.3.0` on `Python 3.11` | `2.0.0` on `Python 3.12` |
| Kestra Java server | `2.0.8` | `2.0.8` |
| dbt | `dbt-core>=1.11.14,<1.12.0` + `dbt-postgres>=1.10.0,<1.11.0` | `dbt-core 1.11.16` + `dbt-postgres 1.10.0` |

## Security Audit Surfaces

| Surface | Install Surface | Release Blocking |
|---------|-----------------|------------------|
| `base` | `truthound-orchestration` | Yes |
| `airflow` | `truthound-orchestration[airflow]` | Yes |
| `prefect` | `truthound-orchestration[prefect]` | Yes |
| `dagster` | `truthound-orchestration[dagster]` | Yes |
| `dbt` | `truthound-orchestration[dbt]` | Yes |
| `kestra` | `truthound-orchestration[kestra]` | Yes |
| `opentelemetry` | `truthound-orchestration[opentelemetry]` | Yes |
| `all` | `truthound-orchestration[all]` | Nightly canary only |

First-party release guarantees apply to declared audited surfaces; Mage host security is unsupported. `truthound-orchestration[all]` remains available as a convenience aggregate and nightly canary surface.
Minimum compatibility guarantees are validated as tested host-plus-Python tuples, not as host versions on every Python runtime.
Airflow runtime lanes and security audits resolve the complete extra with the same security constraints and pinned supported host.
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
