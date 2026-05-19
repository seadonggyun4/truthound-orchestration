"""Tests for Kestra Depot helpers."""

from __future__ import annotations

from types import SimpleNamespace

from common.depot.testing import FakeDepotRuntimeClient, build_operation_result


def test_validate_branch_script_returns_shared_payload(monkeypatch) -> None:
    from truthound_kestra.scripts import depot as depot_module

    monkeypatch.setattr(
        depot_module,
        "get_execution_context",
        lambda: SimpleNamespace(execution_id="exec-1", flow_id="flow-1", namespace="default"),
    )
    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result())

    payload = depot_module.validate_branch_script(
        depot_id="depot-1",
        asset_id="asset-1",
        branch_id="main",
        client=client,
    )

    assert payload["payload_type"] == "depot_operation"
    assert payload["status"] == "succeeded"


def test_generate_depot_validate_flow_contains_script_entrypoint() -> None:
    from truthound_kestra.flows.depot import generate_depot_validate_flow

    content = generate_depot_validate_flow(
        flow_id="validate-users",
        namespace="company.data",
        depot_id="depot-1",
        asset_id="asset-1",
    )

    assert "validate_branch_script" in content


def test_generate_depot_scheduled_validate_flow_contains_schedule_trigger() -> None:
    from truthound_kestra.flows.depot import generate_depot_scheduled_validate_flow

    content = generate_depot_scheduled_validate_flow(
        flow_id="scheduled-validate-users",
        namespace="company.data",
        depot_id="depot-1",
        asset_id="asset-1",
        schedule="0 6 * * *",
    )

    assert "Schedule" in content
    assert "validate_branch_script" in content
