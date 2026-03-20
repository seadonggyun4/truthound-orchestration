"""Tests for DataQualityStreamOperator."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from common.testing import MockDataQualityEngine


class TestDataQualityStreamOperator:
    """Tests for the Airflow streaming operator."""

    def test_initialization_requires_stream_or_factory(self) -> None:
        """Operator should reject missing streaming inputs."""
        from truthound_airflow.operators.stream import DataQualityStreamOperator

        with pytest.raises(ValueError, match="Must provide either stream or stream_factory"):
            DataQualityStreamOperator(task_id="stream_check")

    def test_execute_pushes_stream_summary(
        self,
        airflow_context: dict[str, object],
    ) -> None:
        """Streaming execution should push shared batch envelopes to XCom."""
        from truthound_airflow.operators.stream import DataQualityStreamOperator

        mock_engine = MockDataQualityEngine()
        mock_engine.configure_check(success=True, passed_count=1)

        compatible_report = SimpleNamespace(
            compatible=True,
            compatibility=SimpleNamespace(failures=[]),
        )

        with (
            patch("truthound_airflow.operators.stream.run_preflight", return_value=compatible_report),
            patch("truthound_airflow.operators.stream.create_engine", return_value=mock_engine),
        ):
            op = DataQualityStreamOperator(
                task_id="stream_check",
                stream=iter([{"id": 1}, {"id": 2}, {"id": 3}]),
                rules=[],
                batch_size=2,
            )

            result = op.execute(airflow_context)  # type: ignore[arg-type]

        assert result["summary"]["total_batches"] == 2
        assert result["summary"]["total_records"] == 3
        assert len(result["batches"]) == 2
        ti = airflow_context["ti"]
        assert isinstance(ti, MagicMock)
        ti.xcom_push.assert_called_once()

