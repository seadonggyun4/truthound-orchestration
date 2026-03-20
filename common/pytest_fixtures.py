"""Shared pytest fixtures for repository-wide orchestration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import pytest

import common.orchestration as orchestration_module


@dataclass
class OpenLineageCollector:
    """In-memory collector that intercepts OpenLineage HTTP payloads."""

    endpoint: str = "https://collector.truthound.test/api/v1/lineage"
    received: list[dict[str, Any]] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)


class _MockHTTPResponse:
    """Minimal context-managed HTTP response used by the collector fixture."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False

    def read(self) -> bytes:
        return b"ok"


@pytest.fixture
def openlineage_collector(monkeypatch: pytest.MonkeyPatch) -> OpenLineageCollector:
    """Capture OpenLineage emissions without binding a real socket."""

    collector = OpenLineageCollector()

    def _fake_urlopen(request: Any, timeout: float | None = None) -> _MockHTTPResponse:
        body = json.loads((request.data or b"{}").decode("utf-8"))
        collector.received.append(body)
        collector.requests.append(
            {
                "url": getattr(request, "full_url", collector.endpoint),
                "headers": dict(request.header_items()),
                "timeout": timeout,
            }
        )
        return _MockHTTPResponse()

    monkeypatch.setattr(orchestration_module, "urlopen", _fake_urlopen)
    return collector
