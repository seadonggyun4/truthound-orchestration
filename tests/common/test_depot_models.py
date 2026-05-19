"""Tests for shared Depot operation models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from common.depot.models import DepotOperationRequest, DepotOperationResult, make_json_safe
from common.depot.testing import build_operation_request, build_operation_result


def test_operation_request_round_trips_nested_platform_metadata() -> None:
    request = build_operation_request(
        metadata={"path": Path("/tmp/example"), "attempt": 1},
    )

    restored = DepotOperationRequest.from_dict(request.to_dict())

    assert restored.to_dict() == request.to_dict()
    assert restored.metadata["path"] == "/tmp/example"
    assert restored.platform_metadata is not None
    assert restored.platform_metadata.host_execution["run_id"] == "run-123"


def test_operation_result_round_trips_datetimes_and_nested_refs() -> None:
    started_at = datetime(2026, 5, 19, 1, 0, tzinfo=UTC)
    completed_at = datetime(2026, 5, 19, 1, 1, tzinfo=UTC)
    result = build_operation_result(started_at=started_at, completed_at=completed_at)

    restored = DepotOperationResult.from_dict(result.to_dict())

    assert restored == result
    assert restored.started_at == started_at
    assert restored.completed_at == completed_at


def test_make_json_safe_normalizes_paths_sets_and_datetimes() -> None:
    payload = {
        "path": Path("/tmp/example"),
        "seen": {"b", "a"},
        "ts": datetime(2026, 5, 19, 0, 0, tzinfo=UTC),
    }

    safe = make_json_safe(payload)

    assert safe["path"] == "/tmp/example"
    assert safe["seen"] == ["a", "b"]
    assert safe["ts"] == "2026-05-19T00:00:00+00:00"
