"""Tests for TruthoundEngine streaming support (TASK 1-3)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from common.base import CheckResult, CheckStatus
from common.engines.truthound import TruthoundEngine, TruthoundEngineConfig


# =============================================================================
# Fixtures
# =============================================================================


class FakeCheckReport:
    """Fake Truthound check report for testing."""

    def __init__(self, has_critical: bool = False, issues: list | None = None) -> None:
        self.has_critical = has_critical
        self._issues = issues or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": self._issues,
            "row_count": 10,
            "column_count": 3,
            "source": "test",
        }


@pytest.fixture
def engine() -> TruthoundEngine:
    """Create a TruthoundEngine with mocked truthound module."""
    e = TruthoundEngine.__new__(TruthoundEngine)
    e._config = TruthoundEngineConfig()
    e._truthound = MagicMock()
    e._truthound.__version__ = "1.0.0"
    e._version = "1.0.0"
    from common.engines.lifecycle import EngineStateTracker

    e._state_tracker = EngineStateTracker("truthound")
    e._lock = __import__("threading").RLock()
    e._schema_cache = {}
    return e


# =============================================================================
# check_stream Tests
# =============================================================================


class TestCheckStream:
    """Tests for check_stream method."""

    def test_basic_stream_processing(self, engine: TruthoundEngine) -> None:
        """Verify stream is processed in batches."""
        engine._truthound.check.return_value = FakeCheckReport()

        stream = [{"a": i, "b": i * 2} for i in range(25)]
        results = list(engine.check_stream(stream, batch_size=10))

        assert len(results) == 3  # 10 + 10 + 5
        assert all(isinstance(r, CheckResult) for r in results)
        assert engine._truthound.check.call_count == 3

    def test_empty_stream(self, engine: TruthoundEngine) -> None:
        """Empty stream yields no results."""
        results = list(engine.check_stream(iter([]), batch_size=10))
        assert len(results) == 0

    def test_exact_batch_size(self, engine: TruthoundEngine) -> None:
        """Stream exactly divisible by batch_size."""
        engine._truthound.check.return_value = FakeCheckReport()

        stream = [{"a": i} for i in range(20)]
        results = list(engine.check_stream(stream, batch_size=10))

        assert len(results) == 2

    def test_smaller_than_batch(self, engine: TruthoundEngine) -> None:
        """Stream smaller than batch_size yields one result."""
        engine._truthound.check.return_value = FakeCheckReport()

        stream = [{"a": i} for i in range(3)]
        results = list(engine.check_stream(stream, batch_size=10))

        assert len(results) == 1

    def test_auto_schema_from_first_batch(self, engine: TruthoundEngine) -> None:
        """auto_schema learns schema from first batch."""
        fake_schema = MagicMock()
        engine._truthound.learn.return_value = fake_schema
        engine._truthound.check.return_value = FakeCheckReport()

        stream = [{"a": i} for i in range(15)]
        results = list(engine.check_stream(stream, batch_size=10, auto_schema=True))

        assert len(results) == 2
        # learn() called once for first batch
        engine._truthound.learn.assert_called_once()
        # check() called with schema for both batches
        for call in engine._truthound.check.call_args_list:
            assert call[1].get("schema") is fake_schema

    def test_schema_passed_through(self, engine: TruthoundEngine) -> None:
        """Explicit schema is passed to all batches."""
        fake_schema = MagicMock()
        engine._truthound.check.return_value = FakeCheckReport()

        stream = [{"a": i} for i in range(15)]
        list(engine.check_stream(stream, batch_size=10, schema=fake_schema))

        for call in engine._truthound.check.call_args_list:
            assert call[1].get("schema") is fake_schema

    def test_generator_input(self, engine: TruthoundEngine) -> None:
        """Generator works as stream input."""
        engine._truthound.check.return_value = FakeCheckReport()

        def gen():
            for i in range(5):
                yield {"a": i}

        results = list(engine.check_stream(gen(), batch_size=3))
        assert len(results) == 2  # 3 + 2

    def test_error_in_batch_yields_error_result(self, engine: TruthoundEngine) -> None:
        """Error during check yields ERROR status, doesn't stop generator."""
        engine._truthound.check.side_effect = RuntimeError("check failed")

        stream = [{"a": i} for i in range(5)]
        results = list(engine.check_stream(stream, batch_size=5))

        assert len(results) == 1
        assert results[0].status == CheckStatus.ERROR
        assert "check failed" in results[0].metadata.get("error", "")

    def test_each_result_independent(self, engine: TruthoundEngine) -> None:
        """Each batch produces an independent result."""
        reports = [
            FakeCheckReport(has_critical=False),
            FakeCheckReport(
                has_critical=True,
                issues=[{"issue_type": "null", "severity": "critical", "count": 1}],
            ),
        ]
        engine._truthound.check.side_effect = reports

        stream = [{"a": i} for i in range(20)]
        results = list(engine.check_stream(stream, batch_size=10))

        assert results[0].status == CheckStatus.PASSED
        assert results[1].status == CheckStatus.FAILED
