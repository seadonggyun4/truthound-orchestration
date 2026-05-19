"""Dagster ops for Depot operations."""

from __future__ import annotations

from typing import Any

from common.depot.idempotency import build_idempotency_key
from common.depot.models import DepotOperationRequest, DepotOperationType
from common.runtime import DepotFlowRequest, normalize_runtime_context
from dagster import Out, op

from truthound_dagster.utils.serialization import (
    serialize_depot_flow_result,
    serialize_depot_result,
    to_dagster_depot_metadata,
)


def _execute_depot_op(
    context: Any,
    operation_type: DepotOperationType,
) -> dict[str, Any]:
    resource = context.resources.depot
    cfg = context.op_config
    request = DepotOperationRequest(
        operation_id=cfg.get("operation_id") or f"{operation_type.value}:{cfg['depot_id']}:{cfg['asset_id']}:{context.run_id}",
        operation_type=operation_type,
        depot_id=cfg["depot_id"],
        asset_id=cfg["asset_id"],
        branch_id=cfg.get("branch_id"),
        snapshot_id=cfg.get("snapshot_id"),
        target_branch_id=cfg.get("target_branch_id"),
        merge_request_id=cfg.get("merge_request_id"),
        release_tag=cfg.get("release_tag"),
        source_ref=cfg.get("source_ref"),
        requested_by=cfg.get("requested_by"),
        idempotency_key=cfg.get("idempotency_key")
        or build_idempotency_key(
            operation_type,
            cfg["depot_id"],
            cfg["asset_id"],
            branch_id=cfg.get("branch_id"),
            snapshot_id=cfg.get("snapshot_id"),
            merge_request_id=cfg.get("merge_request_id"),
            release_tag=cfg.get("release_tag"),
            source_ref=cfg.get("source_ref"),
            target_branch_id=cfg.get("target_branch_id"),
        ),
        metadata=cfg.get("metadata") or {},
    )
    result = resource.execute(
        request,
        host_execution={"run_id": context.run_id, "op_name": context.op.name},
        host_metadata={"op": context.op.name},
        wait=cfg.get("wait", False),
        metadata=cfg.get("metadata"),
    )
    payload = serialize_depot_result(result)
    context.add_output_metadata(to_dagster_depot_metadata(payload))
    return payload


def _execute_depot_flow_op(
    context: Any,
    flow_type: str,
) -> dict[str, Any]:
    resource = context.resources.depot
    cfg = context.op_config
    runtime_context = normalize_runtime_context(
        platform="dagster",
        host_execution={"run_id": context.run_id, "op_name": context.op.name},
        host_metadata={"op": context.op.name, "flow_type": flow_type},
    )
    request = DepotFlowRequest(
        flow_type=flow_type,
        depot_id=cfg["depot_id"],
        asset_id=cfg["asset_id"],
        branch_id=cfg.get("branch_id"),
        snapshot_id=cfg.get("snapshot_id"),
        release_tag=cfg.get("release_tag"),
        target_branch_id=cfg.get("target_branch_id"),
        requested_by=cfg.get("requested_by"),
        runtime_context=runtime_context,
        metadata=cfg.get("metadata") or {},
    )
    result = resource.execute_flow(
        request,
        wait=cfg.get("wait", False),
    )
    payload = serialize_depot_flow_result(result, runtime_context=runtime_context)
    context.add_output_metadata(to_dagster_depot_metadata(payload["final_result"]))
    return payload


_COMMON_CONFIG: dict[str, Any] = {
    "depot_id": str,
    "asset_id": str,
    "branch_id": str,
    "snapshot_id": str,
    "target_branch_id": str,
    "merge_request_id": str,
    "release_tag": str,
    "source_ref": str,
    "requested_by": str,
    "operation_id": str,
    "idempotency_key": str,
    "wait": bool,
}


@op(
    name="depot_pull_snapshot",
    required_resource_keys={"depot"},
    out=Out(dict),
    config_schema=_COMMON_CONFIG,
    tags={"kind": "depot", "operation": "pull_snapshot"},
)
def pull_snapshot_op(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _execute_depot_op(context, DepotOperationType.PULL_SNAPSHOT)


@op(
    name="depot_validate_branch",
    required_resource_keys={"depot"},
    out=Out(dict),
    config_schema=_COMMON_CONFIG,
    tags={"kind": "depot", "operation": "validate_branch"},
)
def validate_branch_op(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _execute_depot_op(context, DepotOperationType.VALIDATE_BRANCH)


@op(
    name="depot_merge_after_approval",
    required_resource_keys={"depot"},
    out=Out(dict),
    config_schema=_COMMON_CONFIG,
    tags={"kind": "depot", "operation": "merge_after_approval"},
)
def merge_after_approval_op(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _execute_depot_op(context, DepotOperationType.MERGE_AFTER_APPROVAL)


@op(
    name="depot_scheduled_sync",
    required_resource_keys={"depot"},
    out=Out(dict),
    config_schema=_COMMON_CONFIG,
    tags={"kind": "depot_flow", "flow": "scheduled_sync"},
)
def scheduled_sync_op(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _execute_depot_flow_op(context, "scheduled_sync")


@op(
    name="depot_scheduled_validation",
    required_resource_keys={"depot"},
    out=Out(dict),
    config_schema=_COMMON_CONFIG,
    tags={"kind": "depot_flow", "flow": "scheduled_validation"},
)
def scheduled_validation_op(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _execute_depot_flow_op(context, "scheduled_validation")


@op(
    name="depot_release_tag_flow",
    required_resource_keys={"depot"},
    out=Out(dict),
    config_schema=_COMMON_CONFIG,
    tags={"kind": "depot_flow", "flow": "release_tag"},
)
def release_tag_flow_op(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _execute_depot_flow_op(context, "release_tag")


@op(
    name="depot_rollback_flow",
    required_resource_keys={"depot"},
    out=Out(dict),
    config_schema=_COMMON_CONFIG,
    tags={"kind": "depot_flow", "flow": "rollback"},
)
def rollback_flow_op(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    return _execute_depot_flow_op(context, "rollback")


__all__ = [
    "pull_snapshot_op",
    "validate_branch_op",
    "merge_after_approval_op",
    "scheduled_sync_op",
    "scheduled_validation_op",
    "release_tag_flow_op",
    "rollback_flow_op",
]
