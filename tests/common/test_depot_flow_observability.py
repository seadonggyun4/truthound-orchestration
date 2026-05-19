"""Tests for Depot orchestration flow observability events."""

from __future__ import annotations

from common.depot.models import DepotOperationStatus
from common.depot.testing import FakeDepotRuntimeClient, build_operation_result
from common.orchestration import OpenLineageEmitter, run_scheduled_sync_flow
from common.runtime import DepotFlowRequest, normalize_runtime_context


def test_scheduled_sync_flow_emits_flow_events() -> None:
    client = FakeDepotRuntimeClient()
    client.queue_submit(build_operation_result(status=DepotOperationStatus.NO_OP))
    emitter = OpenLineageEmitter(namespace="truthound", job_name="depot-flow")

    result = run_scheduled_sync_flow(
        DepotFlowRequest(
            flow_type="scheduled_sync",
            depot_id="depot-1",
            asset_id="asset-1",
            runtime_context=normalize_runtime_context(
                platform="airflow",
                host_execution={"run_id": "run-123", "dag_id": "quality"},
            ),
        ),
        client=client,
        emitter=emitter,
        wait=False,
    )

    assert result.status.value == "no_op"
    assert len(emitter.emitted_events) >= 2
    truthound_facet = emitter.emitted_events[-1]["run"]["facets"]["truthound"]
    facet = truthound_facet["depot_flow"]
    assert facet["flow_type"] == "scheduled_sync"
    assert facet["status"] == "no_op"
    assert truthound_facet["trace_fields"]["flow_type"] == "scheduled_sync"
    assert truthound_facet["failure"] is None
