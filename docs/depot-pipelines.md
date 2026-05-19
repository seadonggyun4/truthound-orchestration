---
title: Depot Pipelines
---

# Depot Pipelines

Depot pipelines are the shared orchestration surface for branch validation, scheduled sync, release tagging, rollback triggers, and approval-aware execution inside `truthound-orchestration`.

The most important boundary is this:

- Truthound Orchestration is the **pipeline execution layer**
- Depot is **not** reimplemented as a business-state owner inside the adapters

That means the orchestration layer can submit operations, wait, poll, normalize results, and project host metadata, but it does not decide approval, release safety, rollback safety, or Depot business state on its own.

## What Depot Pipelines Own

Depot pipelines exist so host-native adapters can reuse one execution contract for:

- snapshot pull requests
- branch validation
- merge-after-approval submission
- release tag requests
- rollback-to-snapshot requests
- scheduled sync and scheduled validation flows

The shared layer keeps those operations compact, host-safe, and observable without forcing Airflow, Prefect, Dagster, dbt, Mage, or Kestra to invent their own result semantics.

## Responsibility Split

| Layer | Owns | Must Not Own |
|-------|------|--------------|
| Core | validation semantics, rule execution, artifact generation | Depot business state, host-specific projection |
| Depot | approval, release safety, rollback safety, business state, canonical operation state | host-native retries, XCom shaping, task metadata |
| Orchestration | pipeline execution, submit/poll/wait, compact result emission, host-native projection | approval decisions, rollback safety decisions, business-state persistence |

## Shared Architecture

Depot support is layered the same way across every host:

```text
host-native entrypoint
  -> shared Depot runtime request normalization
  -> Depot client submit/read/wait
  -> artifact ref attachment
  -> failure normalization and redaction
  -> compact operation or flow payload
  -> host-native metadata wrapper
```

The canonical implementation lives in:

- `common/depot/*` for models, failure taxonomy, idempotency, client, polling, serialization, observability
- `common.runtime` for runtime and flow envelopes
- `common.orchestration` for Depot operation and flow façades
- `common.serializers` for compact runtime and flow payload composition

## Supported Operation Surfaces

These are the shared operation types exposed by the Depot runtime layer:

| Operation | Purpose | Typical terminal states |
|-----------|---------|-------------------------|
| `pull_snapshot` | ask Depot to resolve or pull a snapshot reference | `succeeded`, `failed`, `waiting` |
| `validate_branch` | run branch validation through the shared Depot contract | `succeeded`, `failed`, `waiting` |
| `merge_after_approval` | submit merge intent that remains Depot-owned for approval semantics | `succeeded`, `failed`, `waiting` |
| `release_tag` | request a release tag through Depot-owned release policy | `succeeded`, `failed`, `waiting`, `no_op` |
| `rollback_to_snapshot` | request rollback to a validated snapshot or release tag | `succeeded`, `failed` |
| `scheduled_sync` | run a scheduled synchronization request | `succeeded`, `failed`, `no_op` |

## Supported Flow Surfaces

Flows are intentionally thin wrappers over the shared operation layer. They are not a second workflow engine.

Supported flow shapes:

- submit-only
- submit + wait
- `no_op` terminal flows
- `waiting` flows that propagate approval or external hold states unchanged
- failed terminal flows with compact failure summaries

Current shared flow entrypoints:

| Flow | Shared behavior |
|------|-----------------|
| `scheduled_sync` | submit scheduled sync and optionally wait |
| `scheduled_validation` | submit branch validation and optionally wait |
| `release_tag` | submit release tag request and optionally wait |
| `rollback` | submit rollback request and optionally wait |

## Result Semantics

Depot operation and flow payloads are compact by design.

`WAITING` means:

- the operation is still owned by Depot
- orchestration may poll and propagate the state
- adapters must not reinterpret it as success or failure

`NO_OP` means:

- the request reached a valid terminal state
- no mutation was required
- the payload is still a successful shared contract surface

`FAILED` means:

- the failure code and compact error message are part of the contract
- adapters should preserve the full shared payload
- business inference stays with the caller or Depot, not the adapter

## Failure And Observability Contract

All hosts share the same Depot failure taxonomy and redacted observability surface.

The shared layer guarantees:

- common `DepotFailureCode` values across runtime, flow, and adapter projections
- retryable vs non-retryable classification at the shared layer
- compact payloads only
- redacted links, metadata, and execution context for observability outputs

The shared observability surface keeps:

- operation IDs and flow types
- host run metadata
- artifact refs
- failure summaries

It intentionally avoids:

- raw snapshot bodies
- raw evidence blobs
- dataset payloads
- secret-bearing headers or tokens

## Platform Mapping

Every first-party adapter projects the same shared Depot contract into a native surface:

| Platform | Native entrypoint | Best fit |
|----------|-------------------|----------|
| Airflow | Depot operators and flow operators | DAG-driven validation, scheduled sync, release, rollback |
| Prefect | Depot block and Depot tasks | Python-first flows with optional persisted config |
| Dagster | Depot resource and Depot ops | metadata-rich graph execution and scheduled validation |
| dbt | `run-operation` Depot hooks/macros | SQL-first branch validation and release requests |
| Mage | Depot blocks | pipeline-local validation and scheduled sync happy paths |
| Kestra | Depot scripts and generated flow templates | YAML flow generation with shared payload outputs |

## When To Use Depot Vs Ordinary Validation

Use ordinary validation surfaces when you need:

- dataset checks
- profile or learn execution
- streaming validation
- engine-focused quality behavior without Depot coordination

Use Depot pipelines when you need:

- branch or snapshot-aware orchestration
- approval-aware merge or release requests
- rollback triggers owned by Depot policy
- scheduled sync or scheduled validation as first-class orchestration events

## Host-Native Happy Paths

Use one host-native example per platform, then return to this page for the shared contract details:

- [Airflow Operators](airflow/operators.md)
- [Prefect Tasks](prefect/tasks.md)
- [Dagster Ops](dagster/ops.md)
- [dbt Macros and Operations](dbt/macros.md)
- [Mage Recipes](mage/recipes.md)
- [Kestra Scripts and Flow Templates](kestra/scripts-templates.md)

## Operational Boundaries And Non-Goals

Depot pipelines are not intended to:

- replace the host scheduler
- persist Depot business state locally
- redefine Core validation semantics
- guess approval or rollback safety
- emit raw runtime payloads for downstream parsing

The shared goal is deterministic execution, deterministic status propagation, and consistent host-native projection.

## Cleanup And Documentation Hygiene

When documenting or testing Depot pipelines, do not keep temporary runtime artifacts as canonical repository content.

Cleanup targets include:

- package-level `.pytest_cache/`
- `__pycache__/` under package and test trees
- `.DS_Store` files
- generated flow experiments not used as canonical examples
- stale evidence-only output snippets
- temporary connector or debug scripts
- obsolete `dist-ci-check` references that no longer describe the current Depot/CI path

Keep only:

- repo-tracked example code that documents the public surface
- compact payload examples
- canonical docs pages and checked-in tests

Avoid committing:

- runtime logs
- snapshot bodies
- raw evidence payloads
- ad hoc render output
