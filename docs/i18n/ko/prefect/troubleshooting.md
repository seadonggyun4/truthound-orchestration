!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Prefect Troubleshooting
---

# Prefect Troubleshooting

## Flow Works Locally But Deployment Fails

Cause:

- local ephemeral defaults do not match deployment-time environment or saved block expectations

Fix:

- decide whether the deployment should stay ephemeral or block-backed
- verify the saved block exists in the target environment

## Saved Block Behavior Is Inconsistent

Cause:

- different flows are mixing saved and in-memory blocks without making that explicit

Fix:

- standardize on one block strategy per deployment type
- document block names and ownership

## Task Retries Hide Real Configuration Errors

Cause:

- retries are masking a bad source or missing connection detail

Fix:

- validate the same path once with retries minimized
- inspect source resolution and preflight behavior
