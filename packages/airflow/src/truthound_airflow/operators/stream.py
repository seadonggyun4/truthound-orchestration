"""Streaming operator for Apache Airflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from airflow.exceptions import AirflowException
from airflow.models import BaseOperator

from common.engines import EngineCreationRequest, create_engine, normalize_runtime_context, run_preflight
from common.orchestration import StreamCheckpointState, StreamRequest, run_stream_check, summarize_stream


class DataQualityStreamOperator(BaseOperator):
    """Execute bounded-memory streaming checks and push batch summaries to XCom."""

    template_fields: Sequence[str] = ()
    ui_color: str = "#2D9C74"

    def __init__(
        self,
        *,
        stream: Any | None = None,
        stream_factory: Callable[[], Any] | None = None,
        rules: list[dict[str, Any]] | None = None,
        engine_name: str | None = None,
        observability: dict[str, Any] | None = None,
        fail_on_error: bool = True,
        batch_size: int = 1000,
        max_batches: int | None = None,
        checkpoint: dict[str, Any] | None = None,
        timeout_seconds: int = 300,
        xcom_push_key: str = "data_quality_stream",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if stream is None and stream_factory is None:
            raise ValueError("Must provide either stream or stream_factory")

        self.stream = stream
        self.stream_factory = stream_factory
        self.rules = rules
        self.engine_name = engine_name or "truthound"
        self.observability = observability
        self.fail_on_error = fail_on_error
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.checkpoint = checkpoint
        self.timeout_seconds = timeout_seconds
        self.xcom_push_key = xcom_push_key

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        runtime_context = normalize_runtime_context(
            platform="airflow",
            source_name="stream",
            host_metadata={
                "task_id": self.task_id,
                "dag_id": getattr(context.get("dag"), "dag_id", None),
                "operator": type(self).__name__,
            },
            host_execution={
                "dag_id": getattr(context.get("dag"), "dag_id", None),
                "task_id": self.task_id,
                "run_id": (
                    getattr(context.get("ti"), "run_id", None)
                    or getattr(context.get("dag_run"), "run_id", None)
                ),
                "try_number": getattr(context.get("ti"), "try_number", None),
            },
        )
        request = EngineCreationRequest(
            engine_name=self.engine_name,
            runtime_context=runtime_context,
            required_operations=("stream",),
            observability=self.observability,
        )
        preflight = run_preflight(request, observability=self.observability)
        if not preflight.compatible:
            failures = "; ".join(check.message for check in preflight.compatibility.failures)
            raise AirflowException(f"Truthound Airflow preflight failed: {failures}")

        engine = create_engine(request)
        stream_input = self.stream_factory() if self.stream_factory is not None else self.stream
        checkpoint = (
            StreamCheckpointState(**self.checkpoint) if self.checkpoint is not None else None
        )
        envelopes = list(
            run_stream_check(
                engine,
                StreamRequest(
                    stream=stream_input,
                    rules=self.rules,
                    batch_size=self.batch_size,
                    checkpoint=checkpoint,
                    max_batches=self.max_batches,
                    kwargs={"timeout": self.timeout_seconds},
                ),
                runtime_context=runtime_context,
                observability=self.observability,
            )
        )
        summary = summarize_stream(envelopes)
        payload = {
            "batches": [envelope.to_dict() for envelope in envelopes],
            "summary": summary.to_dict(),
            "_metadata": {
                "engine": engine.engine_name,
                "engine_version": engine.engine_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_id": self.task_id,
                "dag_id": getattr(context.get("dag"), "dag_id", None),
            },
        }
        context["ti"].xcom_push(key=self.xcom_push_key, value=payload)
        if self.fail_on_error and summary.failed_batches:
            raise AirflowException(
                f"Streaming quality check failed in {summary.failed_batches} batches"
            )
        return payload
