"""Platform boundary regression tests.

These tests make sure platform packages stay on shared resolver/serializer
boundaries instead of directly importing the Truthound engine internals.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_prefect_engine_block_does_not_directly_import_truthound_engine() -> None:
    source = (
        ROOT
        / "packages/prefect/src/truthound_prefect/blocks/engine.py"
    ).read_text(encoding="utf-8")

    assert "TruthoundEngine" not in source
    assert "TruthoundEngineConfig" not in source
    assert "create_engine(" in source
    assert "run_preflight(" in source


def test_dagster_resource_does_not_directly_import_truthound_engine() -> None:
    source = (
        ROOT
        / "packages/dagster/src/truthound_dagster/resources/engine.py"
    ).read_text(encoding="utf-8")

    assert "TruthoundEngine" not in source
    assert "TruthoundEngineConfig" not in source
    assert "create_engine(" in source
    assert "run_preflight(" in source


def test_kestra_script_base_does_not_directly_import_truthound_engine() -> None:
    source = (
        ROOT
        / "packages/kestra/src/truthound_kestra/scripts/base.py"
    ).read_text(encoding="utf-8")

    assert "TruthoundEngine" not in source
    assert "TruthoundEngineConfig" not in source
    assert "create_engine(" in source
    assert "run_preflight(" in source


def test_airflow_base_operator_uses_shared_runtime_boundary() -> None:
    source = (
        ROOT
        / "packages/airflow/src/truthound_airflow/operators/base.py"
    ).read_text(encoding="utf-8")

    assert "TruthoundEngine" not in source
    assert "create_engine(" in source
    assert "run_preflight(" in source
    assert "resolve_data_source(" in source


def test_mage_transformer_uses_shared_runtime_boundary() -> None:
    source = (
        ROOT
        / "packages/mage/src/truthound_mage/blocks/transformer.py"
    ).read_text(encoding="utf-8")

    assert "TruthoundEngine" not in source
    assert "create_engine(" in source
    assert "run_preflight(" in source
