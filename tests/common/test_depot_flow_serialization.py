"""Tests for shared Depot flow serialization helpers."""

from __future__ import annotations

from pathlib import Path

from common.depot.models import DepotOperationStatus, DepotOperationType
from common.depot.testing import build_artifact_refs, build_operation_result
from common.runtime import DepotFlowResult, DepotFlowStatus, DepotFlowStepResult
from common.serializers import (
    compose_platform_flow_payload,
    deserialize_depot_flow_wire,
    serialize_depot_flow_wire,
)


def _build_flow_result() -> DepotFlowResult:
    final_result = build_operation_result(
        status=DepotOperationStatus.SUCCEEDED,
        metadata={"raw_rows": [{"id": 1}]},
    )
    return DepotFlowResult(
        flow_type="scheduled_sync",
        status=DepotFlowStatus.SUCCEEDED,
        depot_id="depot-1",
        asset_id="asset-1",
        final_result=final_result,
        steps=(
            DepotFlowStepResult(
                step_name="scheduled_sync",
                operation_type=DepotOperationType.SCHEDULED_SYNC,
                result=final_result,
            ),
        ),
        artifact_refs=build_artifact_refs(),
        metadata={"safe": "value"},
    )


def test_serialize_depot_flow_wire_round_trips() -> None:
    result = _build_flow_result()

    payload = serialize_depot_flow_wire(result)
    restored = deserialize_depot_flow_wire(payload)

    assert payload["payload_type"] == "depot_flow"
    assert restored.flow_type == result.flow_type
    assert restored.final_result.operation_id == result.final_result.operation_id


def test_compose_platform_flow_payload_is_json_safe() -> None:
    payload = compose_platform_flow_payload(
        runtime_context={"platform": "airflow", "project_root": str(Path("/tmp/project"))},
        flow_result=_build_flow_result(),
        host_wrapper_metadata={"attempts": [1, 2]},
    )

    assert payload["payload_type"] == "depot_flow"
    assert payload["metadata"]["safe"] == "value"
    assert payload["runtime_context"]["platform"] == "airflow"
    assert payload["host_wrapper_metadata"]["attempts"] == [1, 2]
