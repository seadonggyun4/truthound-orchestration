"""Deferrable quality gate trigger for Apache Airflow."""

from __future__ import annotations

import asyncio
from typing import Any

from airflow.triggers.base import BaseTrigger, TriggerEvent

from common.engines import EngineCreationRequest, create_engine, normalize_runtime_context
from common.orchestration import QualityGateConfig, evaluate_quality_gate


class DataQualityGateTrigger(BaseTrigger):
    """Poll a shared quality gate without occupying a worker slot."""

    def __init__(
        self,
        *,
        data_path: str | None = None,
        sql: str | None = None,
        connection_id: str | None = None,
        rules: list[dict[str, Any]] | None = None,
        engine_name: str = "truthound",
        observability: dict[str, Any] | None = None,
        min_pass_rate: float = 1.0,
        min_row_count: int | None = None,
        max_failure_count: int | None = None,
        timeout_seconds: int = 300,
        poke_interval: float = 60.0,
        task_id: str | None = None,
        dag_id: str | None = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.sql = sql
        self.connection_id = connection_id
        self.rules = rules
        self.engine_name = engine_name
        self.observability = observability
        self.min_pass_rate = min_pass_rate
        self.min_row_count = min_row_count
        self.max_failure_count = max_failure_count
        self.timeout_seconds = timeout_seconds
        self.poke_interval = poke_interval
        self.task_id = task_id
        self.dag_id = dag_id

    def serialize(self) -> tuple[str, dict[str, Any]]:
        return (
            "truthound_airflow.triggers.quality.DataQualityGateTrigger",
            {
                "data_path": self.data_path,
                "sql": self.sql,
                "connection_id": self.connection_id,
                "rules": self.rules,
                "engine_name": self.engine_name,
                "observability": self.observability,
                "min_pass_rate": self.min_pass_rate,
                "min_row_count": self.min_row_count,
                "max_failure_count": self.max_failure_count,
                "timeout_seconds": self.timeout_seconds,
                "poke_interval": self.poke_interval,
                "task_id": self.task_id,
                "dag_id": self.dag_id,
            },
        )

    async def run(self):  # type: ignore[override]
        from truthound_airflow.hooks.base import DataQualityHook

        runtime_context = normalize_runtime_context(
            platform="airflow",
            connection_id=self.connection_id if self.sql else None,
            source_name=self.data_path or self.sql or "stream",
            host_metadata={
                "task_id": self.task_id,
                "dag_id": self.dag_id,
                "operator": "DeferrableDataQualitySensor",
                "trigger": type(self).__name__,
            },
            host_execution={
                "dag_id": self.dag_id,
                "task_id": self.task_id,
            },
        )
        engine = create_engine(
            EngineCreationRequest(
                engine_name=self.engine_name,
                runtime_context=runtime_context,
                required_operations=("quality_gate",),
                observability=self.observability,
            )
        )
        gate_config = QualityGateConfig(
            min_pass_rate=self.min_pass_rate,
            min_row_count=self.min_row_count,
            max_failure_count=self.max_failure_count,
            timeout_seconds=float(self.timeout_seconds),
        )
        hook = DataQualityHook(connection_id=self.connection_id or "truthound_default")

        while True:
            try:
                data = await asyncio.to_thread(self._load_data, hook)
                decision = await asyncio.to_thread(
                    evaluate_quality_gate,
                    engine,
                    data,
                    rules=self.rules,
                    config=gate_config,
                    runtime_context=runtime_context,
                    observability=self.observability,
                )
                if decision.satisfied:
                    yield TriggerEvent({"status": "success", "decision": decision.to_dict()})
                    return
            except FileNotFoundError:
                pass
            except Exception as exc:  # pragma: no cover - defensive trigger path
                yield TriggerEvent({"status": "error", "message": str(exc)})
                return

            await asyncio.sleep(self.poke_interval)

    def _load_data(self, hook: Any) -> Any:
        if self.data_path:
            return hook.load_data(self.data_path)
        return hook.query(self.sql)
