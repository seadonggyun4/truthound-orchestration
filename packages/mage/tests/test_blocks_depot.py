"""Tests for Mage Depot blocks."""

from __future__ import annotations

from common.depot.models import DepotOperationStatus
from common.depot.testing import FakeDepotRuntimeClient, build_operation_result

from truthound_mage.blocks.base import BlockExecutionContext


def test_validate_branch_returns_shared_payload() -> None:
    from truthound_mage.blocks.depot import validate_branch

    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result())

    payload = validate_branch(
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        client=client,
        context=BlockExecutionContext(block_uuid="block-1", pipeline_uuid="pipe-1"),
    )

    assert payload["payload_type"] == "depot_operation"
    assert payload["status"] == "succeeded"


def test_scheduled_sync_returns_flow_payload() -> None:
    from truthound_mage.blocks.depot import scheduled_sync

    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.NO_OP))

    payload = scheduled_sync(
        depot_id="depot-1",
        asset_id="asset-1",
        client=client,
        context=BlockExecutionContext(block_uuid="block-1", pipeline_uuid="pipe-1"),
    )

    assert payload["payload_type"] == "depot_flow"
    assert payload["status"] == "no_op"
