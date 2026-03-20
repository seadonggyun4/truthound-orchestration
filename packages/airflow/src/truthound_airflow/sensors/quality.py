"""Data quality sensors for Apache Airflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from airflow.exceptions import AirflowException
from airflow.sensors.base import BaseSensorOperator

from common.engines import EngineCreationRequest, create_engine, normalize_runtime_context, run_preflight
from common.orchestration import QualityGateConfig, QualityGateDecision, evaluate_quality_gate
from truthound_airflow.triggers.quality import DataQualityGateTrigger

if TYPE_CHECKING:
    from airflow.utils.context import Context

    from common.engines.base import DataQualityEngine


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Configuration for a shared quality gate."""

    min_pass_rate: float = 1.0
    min_row_count: int | None = None
    max_failure_count: int | None = None
    check_data_exists: bool = True
    continue_on_error: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.min_pass_rate <= 1:
            raise ValueError("min_pass_rate must be between 0 and 1")
        if self.min_row_count is not None and self.min_row_count < 0:
            raise ValueError("min_row_count must be non-negative")


class DataQualitySensor(BaseSensorOperator):
    """Classic Airflow quality gate backed by the shared gate contract."""

    template_fields: Sequence[str] = ("rules", "data_path", "sql", "connection_id")
    ui_color: str = "#E67E22"
    deferrable: bool = False

    def __init__(
        self,
        *,
        rules: list[dict[str, Any]] | None = None,
        data_path: str | None = None,
        sql: str | None = None,
        stream: Any | None = None,
        connection_id: str = "truthound_default",
        observability: dict[str, Any] | None = None,
        min_pass_rate: float = 1.0,
        min_row_count: int | None = None,
        max_failure_count: int | None = None,
        check_data_exists: bool = True,
        engine: DataQualityEngine | None = None,
        engine_name: str | None = None,
        timeout_seconds: int = 300,
        **kwargs: Any,
    ) -> None:
        selected_sources = sum(source is not None for source in (data_path, sql, stream))
        if selected_sources > 1:
            raise ValueError("Cannot specify more than one of data_path, sql, or stream")
        if selected_sources == 0:
            raise ValueError("Must specify one of data_path, sql, or stream")
        if not 0 <= min_pass_rate <= 1:
            raise ValueError("min_pass_rate must be between 0 and 1")

        super().__init__(**kwargs)
        self.rules = rules
        self.data_path = data_path
        self.sql = sql
        self.stream = stream
        self.connection_id = connection_id
        self.observability = observability
        self.min_pass_rate = min_pass_rate
        self.min_row_count = min_row_count
        self.max_failure_count = max_failure_count
        self.check_data_exists = check_data_exists
        self.timeout_seconds = timeout_seconds
        self._engine = engine
        self._engine_name = engine_name
        self._runtime_context: Any | None = None

    @property
    def engine(self) -> DataQualityEngine:
        if self._engine is None:
            self._engine = create_engine(
                EngineCreationRequest(
                    engine_name=self._engine_name or "truthound",
                    runtime_context=self._runtime_context,
                    required_operations=("quality_gate",),
                    observability=self.observability,
                )
            )
        return self._engine

    def execute(self, context: Context) -> bool:  # type: ignore[override]
        self._runtime_context = self._build_runtime_context(context)
        preflight = self._run_preflight()
        if not preflight.compatible:
            failures = "; ".join(check.message for check in preflight.compatibility.failures)
            raise AirflowException(f"Truthound Airflow preflight failed: {failures}")
        return super().execute(context)

    def poke(self, context: Context) -> bool:
        self._runtime_context = self._build_runtime_context(context)
        decision = self._evaluate_decision()
        self.log.info("Quality gate decision: %s", decision.reason)
        return decision.satisfied

    def _evaluate_decision(self) -> QualityGateDecision:
        if self.stream is not None:
            data = self.stream
        else:
            from truthound_airflow.hooks.base import DataQualityHook

            hook = DataQualityHook(connection_id=self.connection_id)
            try:
                data = hook.load_data(self.data_path) if self.data_path else hook.query(self.sql)
            except FileNotFoundError:
                if self.check_data_exists:
                    return self._unsatisfied("data not found yet")
                raise
            except Exception as exc:
                return self._unsatisfied(str(exc), error_type=type(exc).__name__)

        return evaluate_quality_gate(
            self.engine,
            data,
            rules=self.rules,
            config=QualityGateConfig(
                min_pass_rate=self.min_pass_rate,
                min_row_count=self.min_row_count,
                max_failure_count=self.max_failure_count,
                continue_on_error=True,
                timeout_seconds=float(self.timeout_seconds),
            ),
            runtime_context=self._runtime_context,
            observability=self.observability,
        )

    def _build_runtime_context(self, context: Context | None = None) -> Any:
        dag = context.get("dag") if context else None
        dag_run = context.get("dag_run") if context else None
        task_instance = context.get("ti") if context else None
        task = context.get("task") if context else None
        task_id = (
            getattr(task_instance, "task_id", None)
            or getattr(task, "task_id", None)
            or self.task_id
        )
        dag_id = (
            getattr(task_instance, "dag_id", None)
            or getattr(dag, "dag_id", None)
        )
        return normalize_runtime_context(
            platform="airflow",
            connection_id=self.connection_id if self.sql else None,
            source_name=self.data_path or self.sql or "stream",
            host_metadata={
                "task_id": task_id,
                "dag_id": dag_id,
                "sensor": type(self).__name__,
            },
            host_execution={
                "task_id": task_id,
                "dag_id": dag_id,
                "run_id": (
                    getattr(task_instance, "run_id", None)
                    or getattr(dag_run, "run_id", None)
                ),
                "try_number": getattr(task_instance, "try_number", None),
            },
        )

    def _run_preflight(self) -> Any:
        return run_preflight(
            EngineCreationRequest(
                engine_name=self._engine_name or "truthound",
                runtime_context=self._runtime_context,
                required_operations=("quality_gate",),
                observability=self.observability,
            ),
            data_path=self.data_path,
            sql=self.sql,
            observability=self.observability,
        )

    @staticmethod
    def _unsatisfied(reason: str, **metadata: Any) -> QualityGateDecision:
        return QualityGateDecision(
            satisfied=False,
            reason=reason,
            pass_rate=0.0,
            row_count=0,
            metadata=metadata,
        )


class DeferrableDataQualitySensor(DataQualitySensor):
    """Primary waiting surface for Airflow quality gates."""

    deferrable: bool = True

    def execute(self, context: Context) -> bool:  # type: ignore[override]
        if self.stream is not None:
            return super().execute(context)

        self._runtime_context = self._build_runtime_context(context)
        preflight = self._run_preflight()
        if not preflight.compatible:
            failures = "; ".join(check.message for check in preflight.compatibility.failures)
            raise AirflowException(f"Truthound Airflow preflight failed: {failures}")

        self.defer(
            trigger=DataQualityGateTrigger(
                data_path=self.data_path,
                sql=self.sql,
                connection_id=self.connection_id,
                rules=self.rules,
                engine_name=self._engine_name or "truthound",
                observability=self.observability,
                min_pass_rate=self.min_pass_rate,
                min_row_count=self.min_row_count,
                max_failure_count=self.max_failure_count,
                timeout_seconds=self.timeout_seconds,
                poke_interval=float(getattr(self, "poke_interval", 60.0)),
                task_id=self.task_id,
                dag_id=getattr(context.get("dag"), "dag_id", None),
            ),
            method_name="execute_complete",
        )
        return False

    def execute_complete(self, context: Context, event: dict[str, Any] | None = None) -> bool:
        del context
        if not event:
            raise AirflowException("Deferrable quality gate returned no event")
        if event.get("status") == "error":
            raise AirflowException(event.get("message", "Unknown trigger failure"))
        return True


TruthoundSensor = DataQualitySensor
