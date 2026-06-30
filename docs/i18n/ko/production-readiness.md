!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Production Readiness
---

# 운영 준비

This page is the operator-facing checklist for deciding whether a Truthound
오케스트레이션 integration is ready to move from local 검증 to shared
production 워크플로우s.

## Who This Is For

- platform engineers preparing a first rollout
- teams reviewing a host package before broad adoption
- operators building release gates, promotion checks, or runbooks

## When To Use It

Use this page when a project has already passed a local quickstart and now
needs to answer operational questions:

- is the compatibility tuple supported?
- is the selected host the right execution boundary?
- are secrets, retries, and alerting defined explicitly?
- do result payloads land in a host-native surface that downstream users can
  rely on?

## Prerequisites

- a chosen host package or shared-runtime entry point
- a supported tuple from [Compatibility](compatibility.md)
- a known source shape such as a local path, SQL query, dataframe, stream, or
  warehouse relation
- a decision about whether Truthound, Pandera, or Great Expectations is the
  execution engine

## Minimal Readiness Review

Before rollout, confirm all of the following:

| Layer | What Must Be True |
|-------|-------------------|
| Host | the adapter is installed from a supported host-plus-Python tuple |
| Runtime | `run_preflight(...)` passes with the same inputs used in production |
| Source | local, SQL, stream, and warehouse inputs are resolved intentionally |
| Engine | the engine name and optional fallback strategy are explicit |
| Results | the team knows where results land: XCom, metadata, artifacts, outputs, or dbt test status |
| Operations | retries, caching, rate limiting, notifications, and SLA handling are configured where needed |

If the rollout depends on 워크플로우 flows such as scheduled sync, release tagging, or rollback triggers, also confirm the 오케스트레이션 side is treating 워크플로우 as a pipeline execution layer only. Approval, release safety, and rollback safety should remain 워크플로우-owned. See [워크플로우 Pipelines].

## Production Pattern

The most reliable rollout sequence is:

1. Start with the host-native quickstart for one narrow 검증 task.
2. Keep the default Truthound engine unless you have a concrete reason to route
   to Pandera or Great Expectations.
3. Add shared runtime preflight checks before the first production release.
4. Move source credentials or connection details into the host's native secret
   surface.
5. Add result consumers that read the documented host-specific payload rather
   than parsing logs.
6. Add SLA, notification, or incident-routing hooks only after the core
   검증 behavior is stable.

## Readiness Checklist

- [ ] Host package is installed from a supported tuple
- [ ] Rules use the shared Truthound rule vocabulary consistently
- [ ] Source resolution behavior is documented for this 워크플로우
- [ ] Preflight compatibility is part of the promotion path
- [ ] Result payload consumers are documented and tested
- [ ] Secrets are not embedded in DAGs, flow code, YAML, or checked-in files
- [ ] Soft-fail versus hard-fail semantics are explicit
- [ ] Retry, cache, rate limiting, and timeout behavior are intentional
- [ ] Operators know where to look for alerts, metrics, and audit evidence

## Failure Modes And Troubleshooting

| Symptom | Likely Cause | First Page To Read |
|---------|--------------|--------------------|
| quality task fails before execution | preflight incompatibility or source mismatch | [Preflight and Compatibility](common/preflight-compatibility.md) |
| host sees a result but downstream consumers disagree | result payload contract not understood | [Result Serialization](common/result-serialization.md) |
| local onboarding worked but production SQL fails | missing host-native connection or secret configuration | host-specific install and operations pages |
| the same rule behaves differently across hosts | source shape or engine selection changed | [Choose a Platform](choose-a-platform.md) and [Engine Selection Guide](engines/selection-guide.md) |

## Related Pages

- [Choose a Platform](choose-a-platform.md)
- [워크플로우 Pipelines]
- [Shared Runtime](common/index.md)
- [Data Quality Engines](engines/index.md)
- [Enterprise Operations](enterprise/index.md)
- [Troubleshooting](troubleshooting.md)
