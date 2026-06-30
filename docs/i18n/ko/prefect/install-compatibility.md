!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Prefect Install And Compatibility
---

# Prefect Install And Compatibility

## Supported Prefect Line

The current support anchors are:

- minimum supported: Prefect `2.14.0` on Python `3.11`
- primary supported: Prefect `3.6.29` on Python `3.12`

See [Compatibility](../compatibility.md) for the release-blocking matrix.

## Install

```bash
pip install truthound-오케스트레이션[prefect] "truthound>=3.0,<4.0"
```

## What To Verify After Install

- `truthound_prefect` imports cleanly
- a flow can call `data_quality_check_task`
- a `DataQualityBlock` can be created and, if needed, saved
- your preferred Prefect version matches a tested support tuple

## Operational Advice

- use ephemeral defaults for local or ad hoc runs
- move to saved blocks when deployments need stable shared configuration
- debug Prefect-specific failures at the block, task, or flow boundary first

## Related Reading

- [Blocks](blocks.md)
- [Deployment Patterns](deployment-patterns.md)
