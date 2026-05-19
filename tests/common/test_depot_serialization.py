"""Tests for Depot result serialization helpers."""

from __future__ import annotations

from common.depot.serialization import (
    deserialize_operation_result_wire,
    serialize_operation_result_wire,
    to_platform_payload,
)
from common.depot.testing import build_operation_result


def test_operation_result_wire_round_trip_preserves_core_fields() -> None:
    result = build_operation_result()

    restored = deserialize_operation_result_wire(serialize_operation_result_wire(result))

    assert restored.operation_id == result.operation_id
    assert restored.status == result.status
    assert restored.artifact_refs.core_result_ref == result.artifact_refs.core_result_ref


def test_platform_payload_stays_compact() -> None:
    result = build_operation_result(metadata={"raw_dataset": [{"id": 1}]})

    payload = to_platform_payload(result)

    assert payload["operation_id"] == "op-123"
    assert "artifact_refs" in payload
    assert "platform_metadata" in payload
    assert payload["metadata"] == {}
    assert "sample_values" not in str(payload)
