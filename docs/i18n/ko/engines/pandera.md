!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Pandera Adapter
---

# Pandera Adapter

`PanderaAdapter` is the contract-first option for teams that already rely on Pandera
schemas and want to orchestrate them through the shared Truthound runtime surface.

## When To Use It

Use Pandera when:

- schema definitions already exist in Pandera and should remain canonical
- data typing and dataframe schema guarantees are more important than migrating to a
  fully Truthound-native rule store immediately

## Strengths

- strong dataframe schema semantics
- familiar 워크플로우 for teams already invested in Pandera
- fits well into gradual migrations

## Tradeoffs

- less Truthound-first than the default engine
- some advanced capability areas may not map one-to-one with the default engine

## Operational Guidance

- treat Pandera as a compatibility bridge when the team is already committed to it
- use the shared runtime and preflight layer the same way as the default engine
- document where Pandera remains canonical versus where Truthound-native rules take
  over

## Related Pages

- [Engines Overview](index.md)
- [Preflight and Compatibility](../common/preflight-compatibility.md)
