"""Tests for shared runtime contracts and preflight behavior."""

from __future__ import annotations

from dataclasses import dataclass

from common.engines import (
    EngineCreationRequest,
    build_compatibility_report,
    normalize_runtime_context,
    resolve_data_source,
    run_preflight,
)
from common.orchestration import (
    OpenLineageEmitter,
    execute_operation,
    QualityGateConfig,
    StreamRequest,
    evaluate_quality_gate,
    prepare_check_invocation,
    run_stream_check,
)
from common.runtime import DataSourceKind, ObservabilityBackend, ObservabilityConfig
from common.testing import MockDataQualityEngine


@dataclass
class _FakeEngine:
    engine_name: str = "truthound"
    engine_version: str = "3.1.0"


def test_resolve_data_source_local_path_is_zero_config() -> None:
    source = resolve_data_source(data_path="data/sample.csv")

    assert source is not None
    assert source.kind == DataSourceKind.LOCAL_PATH
    assert source.requires_connection is False
    assert source.format_hint == "csv"


def test_resolve_data_source_sql_requires_connection() -> None:
    source = resolve_data_source(sql="SELECT * FROM users")

    assert source is not None
    assert source.kind == DataSourceKind.SQL
    assert source.requires_connection is True


def test_build_compatibility_report_flags_missing_connection_for_sql() -> None:
    runtime_context = normalize_runtime_context(platform="airflow")
    source = resolve_data_source(sql="SELECT * FROM users")

    report = build_compatibility_report(
        _FakeEngine(),
        runtime_context=runtime_context,
        resolved_source=source,
    )

    assert report.compatible is False
    assert any(check.name == "source_resolution" for check in report.failures)


def test_run_preflight_includes_truthound_safe_auto_check(monkeypatch) -> None:
    monkeypatch.setattr("common.engines.resolver.create_engine", lambda *args, **kwargs: _FakeEngine())
    runtime_context = normalize_runtime_context(platform="prefect")

    report = run_preflight(
        EngineCreationRequest(engine_name="truthound", runtime_context=runtime_context),
        data_path="dataset.parquet",
    )

    assert report.compatible is True
    assert report.resolved_source is not None
    assert report.resolved_source.kind == DataSourceKind.LOCAL_PATH
    assert any(check.name == "zero_config" for check in report.compatibility.checks)


def test_prepare_check_invocation_enables_truthound_auto_schema() -> None:
    rules, kwargs = prepare_check_invocation("truthound", None)

    assert rules is None
    assert kwargs["auto_schema"] is True


def test_evaluate_quality_gate_stream_uses_shared_stream_summary() -> None:
    engine = MockDataQualityEngine()
    engine.configure_check(success=True, passed_count=1)

    decision = evaluate_quality_gate(
        engine,
        iter([{"id": 1}, {"id": 2}, {"id": 3}]),
        rules=[],
        config=QualityGateConfig(min_pass_rate=1.0),
    )

    assert decision.satisfied is True
    assert decision.result is not None
    assert decision.result["total_batches"] >= 1


def test_run_stream_check_tracks_checkpointed_record_counts() -> None:
    engine = MockDataQualityEngine()
    engine.configure_check(success=True, passed_count=1)

    envelopes = list(
        run_stream_check(
            engine,
            StreamRequest(
                stream=iter([{"id": 1}, {"id": 2}, {"id": 3}]),
                rules=[],
                batch_size=2,
            ),
        )
    )

    assert [envelope.records_in_batch for envelope in envelopes] == [2, 1]
    assert [envelope.checkpoint.records_processed for envelope in envelopes] == [2, 3]


def test_build_compatibility_report_flags_invalid_openlineage_endpoint() -> None:
    engine = MockDataQualityEngine()

    report = build_compatibility_report(
        engine,
        observability=ObservabilityConfig(
            backend=ObservabilityBackend.OPENLINEAGE,
            endpoint="ftp://invalid",
        ),
    )

    assert report.compatible is False
    assert any(check.name == "observability" for check in report.failures)


def test_execute_operation_emits_openlineage_host_execution() -> None:
    engine = MockDataQualityEngine(name="truthound", version="3.1.0")
    engine.configure_check(success=True, passed_count=1)
    emitter = OpenLineageEmitter(namespace="truthound", job_name="runtime-contract")

    execute_operation(
        "check",
        engine,
        data=[{"id": 1}],
        rules=None,
        runtime_context=normalize_runtime_context(
            platform="airflow",
            host_execution={"dag_id": "quality", "task_id": "check_users", "run_id": "run-123"},
        ),
        observability=emitter,
    )

    assert len(emitter.emitted_events) == 2
    payload = emitter.emitted_events[-1]
    facet = payload["run"]["facets"]["truthound"]
    assert facet["host_execution"]["dag_id"] == "quality"
    assert facet["host_execution"]["task_id"] == "check_users"
    assert payload["run"]["runId"] == "run-123"


def test_execute_operation_posts_to_shared_openlineage_collector(
    openlineage_collector: Any,
) -> None:
    engine = MockDataQualityEngine(name="truthound", version="3.1.0")
    engine.configure_check(success=True, passed_count=1)

    execute_operation(
        "check",
        engine,
        data=[{"id": 1}],
        rules=None,
        runtime_context=normalize_runtime_context(
            platform="prefect",
            host_execution={"flow_run_id": "flow-1", "task_run_id": "task-1"},
        ),
        observability=ObservabilityConfig(
            backend=ObservabilityBackend.OPENLINEAGE,
            endpoint=openlineage_collector.endpoint,
            namespace="truthound",
            job_name="runtime-contract",
        ),
    )

    assert len(openlineage_collector.received) == 2
    payload = openlineage_collector.received[-1]
    facet = payload["run"]["facets"]["truthound"]
    assert facet["host_execution"]["flow_run_id"] == "flow-1"
    assert facet["host_execution"]["task_run_id"] == "task-1"
