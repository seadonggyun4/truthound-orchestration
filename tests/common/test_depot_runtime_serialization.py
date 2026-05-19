"""Tests for Depot runtime serialization helpers."""

from __future__ import annotations

from pathlib import Path

from common.depot.testing import build_operation_result
from common.serializers import (
    compose_platform_runtime_payload,
    deserialize_depot_runtime_wire,
    serialize_runtime_wire,
)


def test_serialize_runtime_wire_supports_depot_results() -> None:
    result = build_operation_result()

    payload = serialize_runtime_wire(result)

    assert payload["payload_type"] == "depot_operation"
    restored = deserialize_depot_runtime_wire(payload)
    assert restored.operation_id == result.operation_id
    assert restored.status == result.status


def test_compose_platform_runtime_payload_stays_json_safe() -> None:
    result = build_operation_result(
        metadata={"raw_rows": [{"id": 1}], "safe": "value"},
    )

    payload = compose_platform_runtime_payload(
        runtime_context={"platform": "airflow", "project_root": str(Path("/tmp/project"))},
        depot_result=result,
        core_result_ref="core://results/override",
        host_wrapper_metadata={"attempts": [1, 2]},
    )

    assert payload["artifact_refs"]["core_result_ref"] == result.artifact_refs.core_result_ref
    assert payload["metadata"]["safe"] == "value"
    assert "raw_rows" not in payload["metadata"]
    assert payload["runtime_context"]["platform"] == "airflow"
