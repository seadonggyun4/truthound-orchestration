"""Tests for Prefect Depot integration."""

from __future__ import annotations

from common.depot.models import DepotOperationStatus, DepotOperationType
from common.depot.testing import FakeDepotRuntimeClient, build_operation_result


def test_validate_branch_task_returns_shared_payload() -> None:
    from truthound_prefect.tasks.depot import validate_branch_task  # type: ignore[import-untyped]

    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.SUCCEEDED))

    payload = validate_branch_task.fn(
        block=None,
        client=client,
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
    )

    assert payload["payload_type"] == "depot_operation"
    assert payload["status"] == "succeeded"


def test_depot_block_creates_client() -> None:
    from truthound_prefect.blocks.depot import DepotBlock  # type: ignore[import-untyped]

    block = DepotBlock(base_url="https://depot.example", api_token="token")
    client = block.create_client()

    assert client.config.base_url == "https://depot.example"


def test_scheduled_validation_task_returns_flow_payload() -> None:
    from truthound_prefect.tasks.depot import scheduled_validation_task

    client = FakeDepotRuntimeClient()
    client.queue_submit(
        build_operation_result(
            operation_type=DepotOperationType.VALIDATE_BRANCH,
            status=DepotOperationStatus.WAITING,
        )
    )

    payload = scheduled_validation_task.fn(
        block=None,
        client=client,
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        wait=False,
    )

    assert payload["payload_type"] == "depot_flow"
    assert payload["status"] == "waiting"
    assert payload["steps"][0]["operation_type"] == "validate_branch"
