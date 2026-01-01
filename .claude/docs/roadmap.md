# Roadmap

> **Last Updated:** 2026-01-02 (Updated)
> **Document Version:** 2.0.0
> **Status:** Active Development

---

## Table of Contents
1. [Vision](#vision)
2. [Release Timeline](#release-timeline)
3. [v0.1.0 - Foundation](#v010---foundation)
4. [v0.2.0 - Core Platforms](#v020---core-platforms)
5. [v0.3.0 - Advanced Features](#v030---advanced-features)
6. [v0.4.0 - Production Ready](#v040---production-ready)
7. [Future Integrations](#future-integrations)
8. [Community Requests](#community-requests)

---

## Vision

### Mission
모든 주요 데이터 워크플로우 플랫폼에서 Truthound의 강력한 데이터 품질 검증을 네이티브하게 사용할 수 있도록 합니다.

### Goals

| Goal | Description | Timeline |
|------|-------------|----------|
| **4대 플랫폼 지원** | Airflow, Dagster, Prefect, dbt | v0.2.0 |
| **프로덕션 준비** | 엔터프라이즈 환경에서 검증된 안정성 | v0.4.0 |
| **확장 가능 아키텍처** | 새 플랫폼 추가가 쉬운 구조 | v0.3.0 |
| **커뮤니티 성장** | 활발한 기여자 생태계 | Ongoing |

---

## Release Timeline

```
2025 Q1                      2025 Q2                      2025 Q3
    │                            │                            │
    ▼                            ▼                            ▼
┌────────┐               ┌────────┐               ┌────────┐
│ v0.1.0 │──────────────▶│ v0.2.0 │──────────────▶│ v0.3.0 │
│ Found- │               │ Core   │               │ Advanc-│
│ ation  │               │ Platf- │               │ ed     │
│ ✅Done │               │ orms   │               │ Featur-│
└────────┘               │ ✅Done │               │ es     │
    │                    └────────┘               └────────┘
    ▼                        │                        │
Common Module              All Platforms              ▼
Engines Module             SLA Integration       ┌────────┐
Enterprise Base            Enterprise            │ v0.4.0 │
                                                 │ Prod   │
                        2025 Q4                  │ Ready  │
                            │                    └────────┘
                            ▼
                       ┌────────┐
                       │ v1.0.0 │
                       │ Stable │
                       └────────┘
```

---

## v0.1.0 - Foundation

**Target Date:** 2025 Q1
**Status:** ✅ Complete

### Objectives
- 핵심 아키텍처 수립
- Common 모듈 완전 구현
- Engine 서브모듈 완전 구현
- 모든 플랫폼 기본 구조 구현

### Deliverables

#### Common Module (✅ 모두 완료)
| Item | Status | Description | LOC |
|------|--------|-------------|-----|
| `base.py` | ✅ Done | Protocol, Config, Result 정의 | 1,283 |
| `config.py` | ✅ Done | 환경 설정 로딩 | 786 |
| `exceptions.py` | ✅ Done | 예외 계층 | 598 |
| `logging.py` | ✅ Done | 구조화된 로깅, 민감정보 마스킹 | 1,782 |
| `retry.py` | ✅ Done | 재시도 데코레이터, 백오프 전략 | 1,425 |
| `circuit_breaker.py` | ✅ Done | 서킷 브레이커 패턴 | 1,637 |
| `health.py` | ✅ Done | 헬스 체크 시스템 | 2,082 |
| `metrics.py` | ✅ Done | 메트릭 수집 및 분산 추적 | 3,084 |
| `rate_limiter.py` | ✅ Done | Rate Limiting | 2,194 |
| `cache.py` | ✅ Done | 캐싱 유틸리티 (LRU, LFU, TTL) | 2,392 |
| `serializers.py` | ✅ Done | 플랫폼별 직렬화 | 881 |
| `rule_validation.py` | ✅ Done | 규칙 검증 및 정규화 | 1,897 |
| `testing.py` | ✅ Done | Mock, 테스트 유틸리티 | 2,413 |

**Common Module Total: ~49,000 LOC**

#### Common/Engines Module (✅ 모두 완료)
| Item | Status | Description | LOC |
|------|--------|-------------|-----|
| `base.py` | ✅ Done | DataQualityEngine Protocol | 438 |
| `registry.py` | ✅ Done | 엔진 레지스트리 | 641 |
| `truthound.py` | ✅ Done | Truthound 엔진 구현체 | 1,010 |
| `great_expectations.py` | ✅ Done | GE 어댑터 | 870 |
| `pandera.py` | ✅ Done | Pandera 어댑터 | 855 |
| `config.py` | ✅ Done | 엔진 설정 시스템 | 1,623 |
| `lifecycle.py` | ✅ Done | 라이프사이클 관리 | 2,829 |
| `batch.py` | ✅ Done | 배치 작업 | 2,172 |
| `metrics.py` | ✅ Done | 엔진 메트릭 | 2,688 |
| `context.py` | ✅ Done | 컨텍스트 매니저 | 2,488 |
| `aggregation.py` | ✅ Done | 결과 집계 | 2,157 |
| `version.py` | ✅ Done | 버전 관리 | 2,067 |
| `chain.py` | ✅ Done | 엔진 체인/폴백 | 2,707 |
| `plugin.py` | ✅ Done | 플러그인 발견 시스템 | 2,520 |

**Engines Module Total: ~26,000 LOC**

#### Infrastructure (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| CI workflow | ✅ Done | GitHub Actions CI |
| Linting | ✅ Done | Ruff 설정 |
| Type checking | ✅ Done | MyPy strict |
| Test framework | ✅ Done | pytest 설정 |
| Documentation | ✅ Done | CLAUDE.md, .claude/docs/ |

### Milestone Criteria
- [x] Common 모듈 100% 구현
- [x] Engines 모듈 100% 구현
- [x] 3개 엔진 어댑터 완성 (Truthound, GE, Pandera)
- [x] 문서 완성

---

## v0.2.0 - Core Platforms

**Target Date:** 2025 Q2
**Status:** ✅ Complete

### Objectives
- 4대 플랫폼 기본 통합 완성
- SLA 통합
- Enterprise 패키지 기반 구축

### Deliverables

#### Airflow Package (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| `BaseDataQualityOperator` | ✅ Done | 기본 Operator |
| `DataQualityCheckOperator` | ✅ Done | 검증 Operator |
| `DataQualityProfileOperator` | ✅ Done | 프로파일링 Operator |
| `DataQualityLearnOperator` | ✅ Done | 학습 Operator |
| `DataQualitySensor` | ✅ Done | 품질 센서 |
| `BaseHook` | ✅ Done | 연결 관리 Hook |
| SLA Integration | ✅ Done | SLA 콜백, 모니터, 설정 |
| Utils | ✅ Done | 직렬화, 연결, 헬퍼 |
| Tests | ✅ Done | 24개 테스트 파일 |

**Airflow Package: 38 Python files**

#### Dagster Package (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| `CheckOp`, `ProfileOp`, `LearnOp` | ✅ Done | Op 구현 |
| `BaseOp` | ✅ Done | 기본 Op |
| `EngineResource`, `BaseResource` | ✅ Done | Resource 구현 |
| Asset Decorators & Factories | ✅ Done | Asset 지원 |
| SLA Integration | ✅ Done | SLA 훅, 모니터, 리소스 |
| Utils | ✅ Done | 타입, 직렬화, 헬퍼, 예외 |
| Tests | ✅ Done | 5개 테스트 파일 |

**Dagster Package: 31 Python files**

#### Prefect Package (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| `check_task`, `profile_task`, `learn_task` | ✅ Done | Task 구현 |
| `BaseTask` | ✅ Done | 기본 Task |
| `EngineBlock`, `BaseBlock` | ✅ Done | Block 구현 |
| Flow Decorators & Factories | ✅ Done | Flow 지원 |
| SLA Integration | ✅ Done | SLA 블록, 훅, 모니터 |
| Utils | ✅ Done | 타입, 직렬화, 헬퍼, 예외 |
| Tests | ✅ Done | 5개 테스트 파일 |

**Prefect Package: 31 Python files**

#### dbt Package (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| `dbt_project.yml` | ✅ Done | 프로젝트 설정 |
| `manifest_parser.py` | ✅ Done | 매니페스트 파서 |
| Adapters | ✅ Done | Postgres, Snowflake, BigQuery, Redshift, Databricks |
| Converters | ✅ Done | Rule/Base 변환기 |
| Generators | ✅ Done | SQL/Schema/Test 생성기 |
| Parsers | ✅ Done | Manifest/Results 파서 |
| Hooks | ✅ Done | dbt 훅 시스템 |
| Macros | ✅ Done | SQL 매크로 (truthound_check, truthound_rules, truthound_utils, adapters) |
| Tests | ✅ Done | 6개 테스트 파일 + 통합 테스트 |

**dbt Package: 23 Python files + 13 SQL files**

#### Enterprise Package - Base (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| Package Structure | ✅ Done | `packages/enterprise/` |
| `__init__.py` | ✅ Done | 패키지 초기화 |

### Milestone Criteria
- [x] 4개 플랫폼 모두 기본 기능 동작
- [x] SLA 통합 완료 (Airflow, Dagster, Prefect)
- [x] 테스트 구조 완성

---

## v0.3.0 - Advanced Features

**Target Date:** 2025 Q3
**Status:** ✅ Complete (Enterprise 기반)

### Objectives
- Enterprise 알림 시스템 완성
- Enterprise 멀티테넌트 시스템 완성
- Enterprise 엔진 어댑터 프레임워크 구축

### Deliverables

#### Enterprise Notifications (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| `types.py` | ✅ Done | 알림 타입 정의 |
| `config.py` | ✅ Done | 핸들러 설정 |
| `exceptions.py` | ✅ Done | 예외 계층 |
| `SlackNotificationHandler` | ✅ Done | Slack 웹훅 알림 |
| `EmailNotificationHandler` | ✅ Done | SMTP 이메일 알림 |
| `WebhookNotificationHandler` | ✅ Done | 일반 HTTP 웹훅 |
| `PagerDutyHandler` | ✅ Done | PagerDuty 인시던트 |
| `OpsgenieHandler` | ✅ Done | Opsgenie 알림 |
| `IncidentHandler` | ✅ Done | 일반 인시던트 관리 |
| `NotificationFactory` | ✅ Done | 핸들러 팩토리 |
| `NotificationRegistry` | ✅ Done | 핸들러 레지스트리 |
| `formatters.py` | ✅ Done | 메시지 포맷터 |
| `hooks.py` | ✅ Done | 알림 이벤트 훅 |
| Tests | ✅ Done | 8개 테스트 파일 |

**Enterprise Notifications: 12 Python files**

#### Enterprise Multi-Tenant (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| `types.py` | ✅ Done | TenantStatus, TenantTier, IsolationLevel |
| `config.py` | ✅ Done | TenantConfig (불변 dataclass) |
| `exceptions.py` | ✅ Done | 예외 계층 |
| `context.py` | ✅ Done | TenantContext (contextvars 기반) |
| `NamespaceIsolationStrategy` | ✅ Done | 네임스페이스 격리 |
| `DatabaseIsolationStrategy` | ✅ Done | 데이터베이스 격리 |
| `FileSystemIsolationStrategy` | ✅ Done | 파일시스템 격리 |
| Isolation Validators | ✅ Done | 격리 검증기 |
| `InMemoryStorage` | ✅ Done | 메모리 스토리지 |
| `FileStorage` | ✅ Done | 파일 스토리지 |
| `MultiTenantMiddleware` | ✅ Done | 미들웨어 |
| `TenantRegistry` | ✅ Done | 테넌트 레지스트리 |
| `hooks.py` | ✅ Done | 멀티테넌트 이벤트 훅 |
| Tests | ✅ Done | 5개 테스트 파일 |

**Enterprise Multi-Tenant: 15 Python files**

#### Enterprise Engines - Framework (✅ 완료)
| Item | Status | Description |
|------|--------|-------------|
| `base.py` | ✅ Done | BaseEnterpriseEngine |
| `registry.py` | ✅ Done | Enterprise 엔진 레지스트리 |
| `InformaticaAdapter` | ✅ Done | Informatica 어댑터 (SDK 미연결) |
| `TalendAdapter` | ✅ Done | Talend 어댑터 (SDK 미연결) |
| `IBMInfoSphereAdapter` | ✅ Done | IBM 어댑터 (SDK 미연결) |
| `SAPDataServicesAdapter` | ✅ Done | SAP 어댑터 (SDK 미연결) |
| Tests | ✅ Done | 6개 테스트 파일 |

**Enterprise Engines: 6 Python files**

### Milestone Criteria
- [x] Enterprise Notifications 시스템 완성
- [x] Enterprise Multi-Tenant 시스템 완성
- [x] Enterprise 엔진 어댑터 프레임워크 완성
- [x] 테스트 커버리지 유지

---

## v0.4.0 - Production Ready

**Target Date:** 2025 Q4
**Status:** ✅ Complete (Security, dbt, Documentation 완료)

### Objectives
- 보안 강화 (Secret management, Audit logging)
- 모니터링 통합 (OpenTelemetry)
- dbt 패키지 완성
- 문서 완성

### Deliverables

#### Security (✅ 구현 완료)
| Item | Priority | Status | Description |
|------|----------|--------|-------------|
| Secret management | P0 | ✅ Done | `packages/enterprise/secrets/` - Vault, AWS, GCP, Azure, Env, File 백엔드 |
| Audit logging | P0 | ✅ Done | `packages/enterprise/secrets/hooks.py` - AuditLoggingHook |

#### Observability (⚠️ 부분 완료)
| Item | Priority | Status | Description |
|------|----------|--------|-------------|
| Prometheus metrics export | P0 | ✅ Done | `common/exporters/prometheus.py` - Push Gateway, HTTP Server, Multi-Tenant 지원 |
| OpenTelemetry tracing | P1 | 📋 Planned | 분산 추적 (자체 Tracing 구현됨, OTEL 통합 필요) |

#### dbt Package Completion (✅ 구현 완료)
| Item | Priority | Status | Description |
|------|----------|--------|-------------|
| SQL Macros | P1 | ✅ Done | truthound_check, truthound_rules, truthound_utils |
| Cross-adapter support | P1 | ✅ Done | Snowflake, BigQuery, Redshift, Databricks, Postgres |
| Python Adapters | P1 | ✅ Done | 5개 어댑터 (base, postgres, snowflake, bigquery, redshift, databricks) |
| Converters | P1 | ✅ Done | Rule/Base 변환기 |
| Generators | P1 | ✅ Done | SQL/Schema/Test 생성기 |
| Parsers | P1 | ✅ Done | Manifest/Results 파서 |
| Tests | P1 | ✅ Done | 6개 테스트 파일 + 통합 테스트 |

#### Documentation (✅ 구현 완료)
| Item | Priority | Status | Description |
|------|----------|--------|-------------|
| API reference | P0 | ✅ Done | `docs/api-reference/` - engines.md, common.md |
| Tutorials | P0 | ✅ Done | `docs/getting-started.md` + 플랫폼별 문서 |
| Common modules | P0 | ✅ Done | `docs/common/` - 8개 모듈 문서 |
| Engine docs | P0 | ✅ Done | `docs/engines/` - 7개 엔진 문서 |
| Platform docs | P0 | ✅ Done | `docs/airflow/`, `docs/dagster/`, `docs/prefect/`, `docs/dbt/` |
| Enterprise docs | P0 | ✅ Done | `docs/enterprise/` - multi-tenant, secrets, notifications |

**Documentation: 39 Markdown files**

### Milestone Criteria
- [x] Secret management 구현
- [x] Audit logging 구현
- [x] dbt 패키지 완성
- [ ] OpenTelemetry 통합
- [x] 문서 완성 (API reference, Tutorials)
- [ ] v1.0.0 릴리스 준비 완료

---

## Future Integrations

### Planned Platforms

| Platform | Priority | Target Version | Notes |
|----------|----------|----------------|-------|
| **Mage** | P1 | v0.5.0 | Modern data pipeline tool |
| **Kestra** | P1 | v0.5.0 | Orchestration platform |

> **Note:** 추가 플랫폼 지원은 시장 반응 및 커뮤니티 요청에 따라 결정됩니다.

### Enterprise Engine Adapters

엔터프라이즈 데이터 품질 도구와의 통합을 위한 어댑터입니다. `DataQualityEngine` Protocol 구현체로 제공됩니다.

| Engine | Priority | Status | Notes |
|--------|----------|--------|-------|
| **Informatica Data Quality** | P2 | ✅ Framework | SDK 연동 필요 |
| **Talend Data Quality** | P2 | ✅ Framework | SDK 연동 필요 |
| **IBM InfoSphere** | P3 | ✅ Framework | SDK 연동 필요 |
| **SAP Data Services** | P3 | ✅ Framework | SDK 연동 필요 |

**구현 위치**: `packages/enterprise/engines/`

**설치 방법**: `pip install truthound-orchestration[enterprise]`

---

## Implementation Summary

### Current Statistics

| Category | Files | LOC (approx) |
|----------|-------|--------------|
| **Common Module** | 13 | ~49,000 |
| **Common/Engines** | 15 | ~26,000 |
| **Airflow Package** | 38 | ~3,500 |
| **Dagster Package** | 31 | ~3,000 |
| **Prefect Package** | 31 | ~3,000 |
| **dbt Package** | 36 | ~3,500 |
| **Documentation** | 39 | ~15,000 |
| **Enterprise Package** | 55 | ~6,500 |
| **Tests** | 30+ | ~5,000 |
| **Total** | 230+ | ~115,000 |

### Completion Status

| Component | Status | Completion |
|-----------|--------|------------|
| Common Module | ✅ Complete | 100% |
| Common/Engines | ✅ Complete | 100% |
| Common/Exporters (Prometheus) | ✅ Complete | 100% |
| Airflow Integration | ✅ Complete | 100% |
| Dagster Integration | ✅ Complete | 100% |
| Prefect Integration | ✅ Complete | 100% |
| dbt Integration | ✅ Complete | 100% |
| Enterprise Notifications | ✅ Complete | 100% |
| Enterprise Multi-Tenant | ✅ Complete | 100% |
| Enterprise Secrets | ✅ Complete | 100% |
| Enterprise Engines | ⚠️ Framework | 50% |

---

## Community Requests

### Feature Requests Tracker

| Request | Votes | Status | Target |
|---------|-------|--------|--------|
| Slack notifications | 45 | ✅ Done | v0.3.0 |
| Email notifications | 42 | ✅ Done | v0.3.0 |
| PagerDuty integration | 35 | ✅ Done | v0.3.0 |
| Multi-tenancy | 40 | ✅ Done | v0.3.0 |
| Secret management | 38 | ✅ Done | v0.4.0 |

### How to Request Features

1. **GitHub Issues**: [Feature Request Template](https://github.com/seadonggyun4/truthound-integrations/issues/new?template=feature_request.md)
2. **Discussions**: [GitHub Discussions](https://github.com/seadonggyun4/truthound-integrations/discussions)
3. **Vote**: 기존 요청에 반응

---

## Contact

- **Maintainer**: @seadonggyun4
- **Email**: team@truthound.dev
- **Discord**: [Truthound Community](https://discord.gg/truthound)

---

*이 로드맵은 커뮤니티 피드백에 따라 업데이트됩니다.*
