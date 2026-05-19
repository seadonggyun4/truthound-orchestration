"""Depot block entry points for Mage."""

from __future__ import annotations

from typing import Any

from common.depot.client import DepotClient, DepotClientConfig
from common.depot.idempotency import build_idempotency_key
from common.depot.models import DepotOperationRequest, DepotOperationType
from common.orchestration import DepotOperationClient, execute_depot_flow, execute_depot_operation
from common.runtime import DepotFlowRequest, normalize_runtime_context

from truthound_mage.blocks.base import BlockExecutionContext
from truthound_mage.utils.serialization import (
    serialize_depot_flow_result,
    serialize_depot_result,
)


def _run_depot_block(
    operation_type: DepotOperationType,
    *,
    depot_id: str,
    asset_id: str,
    base_url: str | None = None,
    api_token: str | None = None,
    branch_id: str | None = None,
    snapshot_id: str | None = None,
    client: DepotOperationClient | None = None,
    context: BlockExecutionContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if client is None and (not base_url or not api_token):
        raise ValueError("Mage Depot blocks require either a client or base_url/api_token")
    block_context = context or BlockExecutionContext()
    runtime_context = normalize_runtime_context(
        platform="mage",
        host_metadata={"block_type": "depot", "operation_type": operation_type.value},
        host_execution={
            "block_uuid": block_context.block_uuid,
            "pipeline_uuid": block_context.pipeline_uuid,
            "execution_mode": getattr(
                getattr(block_context, "execution_mode", None),
                "value",
                None,
            ),
        },
    )
    block_id = block_context.block_uuid or "mage"
    request = DepotOperationRequest(
        operation_id=f"{operation_type.value}:{depot_id}:{asset_id}:{block_id}",
        operation_type=operation_type,
        depot_id=depot_id,
        asset_id=asset_id,
        branch_id=branch_id,
        snapshot_id=snapshot_id,
        idempotency_key=build_idempotency_key(
            operation_type,
            depot_id,
            asset_id,
            branch_id=branch_id,
            snapshot_id=snapshot_id,
        ),
        metadata=metadata or {},
    )
    resolved_client = client or DepotClient(
        DepotClientConfig(base_url=base_url or "", api_token=api_token or "")
    )
    result = execute_depot_operation(
        request,
        runtime_context=runtime_context,
        client=resolved_client,
        wait=False,
        metadata=metadata,
    )
    return dict(serialize_depot_result(result, runtime_context=runtime_context))


def sync_asset(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_block(DepotOperationType.PULL_SNAPSHOT, **kwargs)


def validate_branch(**kwargs: Any) -> dict[str, Any]:
    return _run_depot_block(DepotOperationType.VALIDATE_BRANCH, **kwargs)


def scheduled_sync(
    *,
    depot_id: str,
    asset_id: str,
    base_url: str | None = None,
    api_token: str | None = None,
    branch_id: str | None = None,
    snapshot_id: str | None = None,
    client: DepotOperationClient | None = None,
    context: BlockExecutionContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if client is None and (not base_url or not api_token):
        raise ValueError("Mage Depot flow blocks require either a client or base_url/api_token")
    block_context = context or BlockExecutionContext()
    runtime_context = normalize_runtime_context(
        platform="mage",
        host_metadata={"block_type": "depot_flow", "flow_type": "scheduled_sync"},
        host_execution={
            "block_uuid": block_context.block_uuid,
            "pipeline_uuid": block_context.pipeline_uuid,
            "execution_mode": getattr(
                getattr(block_context, "execution_mode", None),
                "value",
                None,
            ),
        },
    )
    flow_request = DepotFlowRequest(
        flow_type="scheduled_sync",
        depot_id=depot_id,
        asset_id=asset_id,
        branch_id=branch_id,
        snapshot_id=snapshot_id,
        runtime_context=runtime_context,
        metadata=metadata or {},
    )
    resolved_client = client or DepotClient(
        DepotClientConfig(base_url=base_url or "", api_token=api_token or "")
    )
    result = execute_depot_flow(
        flow_request,
        client=resolved_client,
        wait=False,
    )
    return dict(serialize_depot_flow_result(result, runtime_context=runtime_context))


__all__ = ["sync_asset", "validate_branch", "scheduled_sync"]
