"""Regression tests for CI workflow action compatibility and helper scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PR_CHANGE_FILTER_PATH = ROOT / "scripts" / "ci" / "pr_change_filter.py"
DEPENDENCY_REVIEW_PATH = ROOT / "scripts" / "ci" / "dependency_review.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pr_change_filter = _load_module(PR_CHANGE_FILTER_PATH, "pr_change_filter")
dependency_review = _load_module(DEPENDENCY_REVIEW_PATH, "dependency_review")


def test_workflows_use_node24_capable_action_versions() -> None:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    )

    assert "actions/checkout@v4" not in workflow_text
    assert "actions/setup-python@v5" not in workflow_text
    assert "actions/upload-artifact@v4" not in workflow_text
    assert "actions/download-artifact@v4" not in workflow_text
    assert "dorny/paths-filter@v3" not in workflow_text
    assert "actions/dependency-review-action@v4" not in workflow_text
    assert "github/codeql-action/init@v3" not in workflow_text
    assert "github/codeql-action/analyze@v3" not in workflow_text


def test_pr_change_filter_classifies_docs_only_changes() -> None:
    outputs = pr_change_filter.classify_changed_files(["docs/compatibility.md"])

    assert outputs == {
        "docs": True,
        "all_platforms": False,
        "airflow": False,
        "prefect": False,
        "dagster": False,
        "mage": False,
        "kestra": False,
        "dbt": False,
        "docs_only": True,
    }


def test_pr_change_filter_classifies_platform_and_package_changes() -> None:
    outputs = pr_change_filter.classify_changed_files(
        [".github/workflows/security.yml", "packages/dbt/macros/truthound_check.sql"]
    )

    assert outputs["all_platforms"] is True
    assert outputs["dbt"] is True
    assert outputs["docs_only"] is False


def test_pr_change_filter_routes_mage_and_kestra_changes_independently() -> None:
    mage_outputs = pr_change_filter.classify_changed_files(
        ["packages/mage/src/truthound_mage/blocks/transformer.py"]
    )
    kestra_outputs = pr_change_filter.classify_changed_files(
        ["packages/kestra/src/truthound_kestra/scripts/check.py"]
    )

    assert mage_outputs["mage"] is True
    assert mage_outputs["kestra"] is False
    assert kestra_outputs["mage"] is False
    assert kestra_outputs["kestra"] is True


def test_mage_and_kestra_workflows_are_independent() -> None:
    workflows_dir = ROOT / ".github" / "workflows"

    assert (workflows_dir / "mage.yml").exists()
    assert (workflows_dir / "kestra.yml").exists()
    assert not (workflows_dir / "mage-kestra.yml").exists()


def test_pr_change_filter_collects_previous_filenames_for_renames() -> None:
    paths = pr_change_filter.collect_changed_paths(
        [
            {"filename": "packages/dbt/new_name.sql", "previous_filename": "docs/old_name.md"},
            {"filename": "README.md"},
        ]
    )

    assert paths == ["README.md", "docs/old_name.md", "packages/dbt/new_name.sql"]


def test_dependency_review_only_flags_vulnerable_added_packages() -> None:
    changes = [
        {
            "change_type": "removed",
            "name": "safe-old",
            "version": "1.0.0",
            "manifest": "pyproject.toml",
            "vulnerabilities": [{"advisory_ghsa_id": "GHSA-old", "severity": "high"}],
        },
        {
            "change_type": "added",
            "name": "safe-new",
            "version": "2.0.0",
            "manifest": "pyproject.toml",
            "vulnerabilities": [],
        },
        {
            "change_type": "added",
            "name": "risky-new",
            "version": "3.0.0",
            "manifest": "requirements.txt",
            "vulnerabilities": [{"advisory_ghsa_id": "GHSA-risk", "severity": "critical"}],
        },
    ]

    findings = dependency_review.find_vulnerable_additions(changes)

    assert findings == [changes[2]]
    assert "risky-new 3.0.0 via requirements.txt: GHSA-risk (critical)" in dependency_review.format_findings(findings)


def test_foundation_workflow_checks_version_surfaces() -> None:
    workflow_text = (ROOT / ".github" / "workflows" / "ci-foundation.yml").read_text(encoding="utf-8")

    assert "python scripts/ci/sync_version_surfaces.py check" in workflow_text
    assert "tests/common/test_version_surfaces.py" in workflow_text


def test_release_gate_passes_expected_tag_to_foundation() -> None:
    workflow_text = (ROOT / ".github" / "workflows" / "release-gate.yml").read_text(encoding="utf-8")

    assert "expected_tag:" in workflow_text
    assert "github.ref_name" in workflow_text


def test_publish_artifact_uses_orchestration_distribution_name() -> None:
    workflows_dir = ROOT / ".github" / "workflows"
    publish_text = (workflows_dir / "publish.yml").read_text(encoding="utf-8")
    foundation_text = (workflows_dir / "ci-foundation.yml").read_text(encoding="utf-8")
    release_text = (workflows_dir / "release-gate.yml").read_text(encoding="utf-8")

    combined = "\n".join([publish_text, foundation_text, release_text])

    assert "truthound-orchestration-dist" in combined
    assert "truthound-foundation" not in combined


def test_pr_ci_uses_smoke_suite_tier_for_all_reusable_workflows() -> None:
    workflow_text = (ROOT / ".github" / "workflows" / "ci-pr.yml").read_text(encoding="utf-8")

    assert workflow_text.count("suite_tier: smoke") == 7


def test_main_uses_change_routed_smoke_and_nightly_keeps_full_suite_tier() -> None:
    main_text = (ROOT / ".github" / "workflows" / "ci-main.yml").read_text(encoding="utf-8")
    nightly_text = (ROOT / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")

    assert "jobs:\n  changes:" in main_text
    assert "docs_only:" in main_text
    assert main_text.count("suite_tier: smoke") == 7
    assert "suite_tier: full" not in main_text
    assert nightly_text.count("suite_tier: full") == 7


def test_main_ci_skips_adapter_fanout_for_docs_only_changes() -> None:
    workflow_text = (ROOT / ".github" / "workflows" / "ci-main.yml").read_text(encoding="utf-8")

    assert "needs.changes.outputs.docs_only != 'true'" in workflow_text
    assert "needs.changes.outputs.all_platforms == 'true' || needs.changes.outputs.airflow == 'true'" in workflow_text
    assert "needs.changes.outputs.all_platforms == 'true' || needs.changes.outputs.prefect == 'true'" in workflow_text
    assert "needs.changes.outputs.all_platforms == 'true' || needs.changes.outputs.dagster == 'true'" in workflow_text
    assert "needs.changes.outputs.all_platforms == 'true' || needs.changes.outputs.mage == 'true'" in workflow_text
    assert "needs.changes.outputs.all_platforms == 'true' || needs.changes.outputs.kestra == 'true'" in workflow_text
    assert "needs.changes.outputs.all_platforms == 'true' || needs.changes.outputs.dbt == 'true'" in workflow_text


def test_reusable_workflows_expose_suite_tier_input() -> None:
    workflows_dir = ROOT / ".github" / "workflows"
    for workflow_name in (
        "shared-runtime.yml",
        "airflow.yml",
        "prefect.yml",
        "dagster.yml",
        "mage.yml",
        "kestra.yml",
        "dbt.yml",
    ):
        workflow_text = (workflows_dir / workflow_name).read_text(encoding="utf-8")
        assert "suite_tier:" in workflow_text
        assert "default: smoke" in workflow_text


def test_shared_runtime_workflow_uses_representative_smoke_and_full_contracts() -> None:
    workflow_text = (ROOT / ".github" / "workflows" / "shared-runtime.yml").read_text(encoding="utf-8")

    assert 'if [ "${SUITE_TIER}" = "full" ]; then' in workflow_text
    assert "tests/common/test_depot_models.py" in workflow_text
    assert "tests/common/test_depot_failures.py" in workflow_text
    assert "tests/common/test_depot_runtime_integration.py" in workflow_text
    assert "tests/common/test_depot_flow_runtime.py" in workflow_text
    assert "tests/common/test_depot_redaction.py" in workflow_text
    assert "tests/common/test_depot_runtime_observability.py" in workflow_text
    assert "tests/common/test_truthound_capabilities.py" in workflow_text
    assert "tests/common/test_truthound_streaming.py" in workflow_text
    assert "tests/common/test_truthound_v3_contract.py" in workflow_text
    assert "tests/opentelemetry/test_config.py" in workflow_text
    assert "python -m pytest -q \\" in workflow_text
    assert "tests/common \\" not in workflow_text


def test_reference_adapter_workflows_include_depot_smoke_files() -> None:
    workflows_dir = ROOT / ".github" / "workflows"

    airflow_text = (workflows_dir / "airflow.yml").read_text(encoding="utf-8")
    prefect_text = (workflows_dir / "prefect.yml").read_text(encoding="utf-8")
    dagster_text = (workflows_dir / "dagster.yml").read_text(encoding="utf-8")

    assert "packages/airflow/tests/test_operators/test_depot.py" in airflow_text
    assert "packages/prefect/tests/test_depot.py" in prefect_text
    assert "packages/dagster/tests/test_depot.py" in dagster_text


def test_mage_and_kestra_pr_workflows_do_not_rely_only_on_full_package_sweeps() -> None:
    workflows_dir = ROOT / ".github" / "workflows"

    mage_text = (workflows_dir / "mage.yml").read_text(encoding="utf-8")
    kestra_text = (workflows_dir / "kestra.yml").read_text(encoding="utf-8")

    assert 'if [ "${SUITE_TIER}" = "full" ]; then' in mage_text
    assert "packages/mage/tests/test_blocks_base.py" in mage_text
    assert "packages/mage/tests/test_blocks_depot.py" in mage_text
    assert "packages/mage/tests/test_utils.py" in mage_text

    assert 'if [ "${SUITE_TIER}" = "full" ]; then' in kestra_text
    assert "packages/kestra/tests/test_scripts.py" in kestra_text
    assert "packages/kestra/tests/test_outputs.py" in kestra_text
    assert "packages/kestra/tests/test_depot.py" in kestra_text


def test_dbt_workflow_keeps_compile_execution_split_and_smoke_python_contracts() -> None:
    workflow_text = (ROOT / ".github" / "workflows" / "dbt.yml").read_text(encoding="utf-8")

    assert "compile-matrix:" in workflow_text
    assert "postgres-execution:" in workflow_text
    assert "needs.matrix.outputs.run_execution == 'true'" in workflow_text
    assert "packages/dbt/tests/test_depot.py" in workflow_text
    assert "packages/dbt/tests/test_hooks.py" in workflow_text
    assert "packages/dbt/tests/test_macros.py" in workflow_text
    assert "tests/dbt/test_first_party_suite.py" in workflow_text


def test_security_workflow_splits_blocking_and_advisory_pip_audit_groups() -> None:
    workflow_text = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")

    assert "advisory_matrix:" in workflow_text
    assert "pip-audit-advisory:" in workflow_text
    assert "continue-on-error: true" in workflow_text
    assert "summary-blocking:" in workflow_text
    assert "summary-advisory:" in workflow_text
