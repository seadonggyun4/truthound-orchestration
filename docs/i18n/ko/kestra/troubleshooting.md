!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Kestra Troubleshooting
---

# Kestra Troubleshooting

## The Script Task Runs But No Structured Result Appears

Check whether the flow is using the package's output handlers or whether the task only
prints raw console output. Structured downstream automation should rely on emitted
outputs, not on log scraping.

## A Generated Flow Compiles But Does Not Match Team Conventions

Treat the generator as a scaffold, not the only valid shape. Generate once, then adapt
namespace, retries, trigger policy, and variables to the team's actual operating model.

## A Script Works Locally But Not In Kestra

Validate:

- the runtime image includes the required package extras
- the target URI is reachable from the execution environment
- secrets and environment variables are available in the task context

## Related Pages

- [Scripts and Flow Templates](scripts-templates.md)
- [Shared Runtime Overview](../common/index.md)
