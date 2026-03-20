"""Streaming data quality script for Kestra workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.orchestration import StreamCheckpointState, StreamRequest, run_stream_check, summarize_stream
from truthound_kestra.scripts.base import (
    CheckScriptConfig,
    DataQualityEngineProtocol,
    build_runtime_context,
    extract_observability_config,
    get_engine,
)
from truthound_kestra.utils.helpers import (
    create_kestra_output,
    get_execution_context,
    kestra_outputs,
)
from truthound_kestra.utils.types import ExecutionContext


@dataclass
class StreamScriptExecutor:
    """Execute bounded-memory stream checks from Kestra script tasks."""

    config: CheckScriptConfig = field(default_factory=CheckScriptConfig)
    engine: DataQualityEngineProtocol | None = None

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = get_engine(
                self.config.engine_name,
                observability=extract_observability_config(self.config.metadata),
                runtime_context=build_runtime_context(
                    "engine_create",
                    script_name=type(self).__name__,
                    metadata=self.config.metadata,
                ),
            )

    def execute(
        self,
        stream: Any,
        rules: list[dict[str, Any]] | None = None,
        context: ExecutionContext | None = None,
        *,
        batch_size: int = 1000,
        checkpoint: dict[str, Any] | None = None,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert self.engine is not None
        if context is None:
            try:
                context = get_execution_context()
            except Exception:
                context = ExecutionContext(
                    execution_id="local",
                    flow_id="script",
                    namespace="default",
                )
        checkpoint_state = (
            StreamCheckpointState(**checkpoint) if checkpoint is not None else None
        )
        envelopes = list(
            run_stream_check(
                self.engine,
                StreamRequest(
                    stream=stream,
                    rules=(
                        list(rules)
                        if rules is not None
                        else (list(self.config.rules) if self.config.rules else None)
                    ),
                    batch_size=batch_size,
                    checkpoint=checkpoint_state,
                    max_batches=max_batches,
                    kwargs=kwargs,
                ),
                runtime_context=build_runtime_context(
                    "stream",
                    context=context,
                    script_name=type(self).__name__,
                    metadata=self.config.metadata,
                ),
                observability=extract_observability_config(self.config.metadata),
            )
        )
        summary = summarize_stream(envelopes)
        result = {
            "status": summary.final_status.lower(),
            "is_success": summary.failed_batches == 0,
            "summary": summary.to_dict(),
            "batches": [envelope.to_dict() for envelope in envelopes],
        }
        return create_kestra_output(result)


def stream_quality_script(
    *,
    stream: Any,
    rules: list[dict[str, Any]] | None = None,
    engine_name: str = "truthound",
    batch_size: int = 1000,
    checkpoint: dict[str, Any] | None = None,
    max_batches: int | None = None,
    context: ExecutionContext | None = None,
    output_to_kestra: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Entry point for streaming quality checks in Kestra Python tasks."""

    executor = StreamScriptExecutor(config=CheckScriptConfig(engine_name=engine_name))
    output = executor.execute(
        stream,
        rules=rules,
        context=context,
        batch_size=batch_size,
        checkpoint=checkpoint,
        max_batches=max_batches,
        **kwargs,
    )
    if output_to_kestra:
        kestra_outputs(output)
    return output


__all__ = ["StreamScriptExecutor", "stream_quality_script"]
