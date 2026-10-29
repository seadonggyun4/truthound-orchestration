"""Release evidence must reject missing, stale, partial and modified results."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "release_evidence", ROOT / "scripts/ci/release_evidence.py"
)
assert spec and spec.loader
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def successful(workflow):
    title, required = evidence.REQUIRED[workflow]
    run = {
        "path": f".github/workflows/{workflow}",
        "head_sha": "current",
        "event": "workflow_dispatch",
        "display_title": title,
        "status": "completed",
        "conclusion": "success",
    }
    jobs = [
        {
            "name": f"{prefix}{i})" if prefix.endswith("(") else prefix,
            "status": "completed",
            "conclusion": "success",
        }
        for prefix, count in required.items()
        for i in range(count)
    ]
    return run, jobs


@pytest.mark.parametrize("workflow", evidence.REQUIRED)
def test_accepts_complete_independent_release(workflow):
    run, jobs = successful(workflow)
    evidence.validate_run(run, jobs, workflow, "current")


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "old"),
        ("display_title", "Prefect · release · smoke"),
        ("conclusion", "failure"),
        ("conclusion", "cancelled"),
        ("status", "in_progress"),
        ("event", "pull_request"),
        ("path", ".github/workflows/other.yml"),
    ],
)
def test_rejects_wrong_or_incomplete_run(field, value):
    run, jobs = successful("prefect.yml")
    run[field] = value
    with pytest.raises(ValueError):
        evidence.validate_run(run, jobs, "prefect.yml", "current")


@pytest.mark.parametrize("status", ["skipped", "failure", "cancelled", None])
def test_successful_summary_cannot_hide_unsuccessful_host(status):
    run, jobs = successful("prefect.yml")
    jobs[1]["conclusion"] = status
    with pytest.raises(ValueError):
        evidence.validate_run(run, jobs, "prefect.yml", "current")


def test_missing_minimum_host_is_rejected():
    run, jobs = successful("kestra.yml")
    jobs.pop(1)
    with pytest.raises(ValueError):
        evidence.validate_run(run, jobs, "kestra.yml", "current")


def test_expected_security_nonrelease_jobs_may_be_skipped():
    run, jobs = successful("security.yml")
    jobs.extend(
        {"name": name, "status": "completed", "conclusion": "skipped"}
        for name in evidence.OPTIONAL_SKIPS
    )
    evidence.validate_run(run, jobs, "security.yml", "current")


def test_unexpected_skipped_security_check_is_rejected():
    run, jobs = successful("security.yml")
    jobs.append({"name": "new-security-gate", "status": "completed", "conclusion": "skipped"})
    with pytest.raises(ValueError):
        evidence.validate_run(run, jobs, "security.yml", "current")


def test_requires_all_independent_workflows():
    with pytest.raises(ValueError):
        evidence.verify_runs({}, "current", "owner/repo")


def candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "version", lambda: "3.0.4")
    (tmp_path / "truthound_orchestration-3.0.4-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / "truthound_orchestration-3.0.4.tar.gz").write_bytes(b"sdist")
    (tmp_path / "release-manifest.json").write_text(
        json.dumps(evidence.manifest(tmp_path, "current"))
    )


def test_verified_artifacts_are_accepted(tmp_path, monkeypatch):
    candidate(tmp_path, monkeypatch)
    evidence.verify_artifacts(tmp_path, "current")


@pytest.mark.parametrize("mutation", ["bytes", "sha", "missing", "extra"])
def test_artifact_tampering_is_rejected(tmp_path, monkeypatch, mutation):
    candidate(tmp_path, monkeypatch)
    wheel = tmp_path / "truthound_orchestration-3.0.4-py3-none-any.whl"
    if mutation == "bytes":
        wheel.write_bytes(b"changed")
    elif mutation == "missing":
        wheel.unlink()
    elif mutation == "extra":
        (tmp_path / "truthound_orchestration-9.0.0.tar.gz").write_bytes(b"foreign")
    with pytest.raises(ValueError):
        evidence.verify_artifacts(tmp_path, "different" if mutation == "sha" else "current")


def test_automatic_workflows_never_fan_out_to_other_workflows():
    import yaml

    workflows = ROOT / ".github/workflows"
    for filename in ("ci-main.yml", "ci-pr.yml", "nightly.yml"):
        assert not (workflows / filename).exists()
    for name in ("airflow", "prefect", "dagster", "mage", "kestra", "dbt"):
        data = yaml.safe_load((workflows / f"{name}.yml").read_text())
        triggers = data.get("on", data.get(True))
        for event in ("push", "pull_request"):
            paths = triggers[event]["paths"]
            assert f"packages/{name}/**" in paths
            assert "common/**" not in paths
            assert "pyproject.toml" not in paths
        assert all("uses" not in job for job in data["jobs"].values())


def test_release_and_publish_only_consume_verified_artifacts():
    import yaml

    for filename in ("release-gate.yml", "publish.yml"):
        data = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
        assert set(data.get("on", data.get(True))) == {"workflow_dispatch"}
        assert all("uses" not in job for job in data["jobs"].values())
    text = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "verify-gate" in text and "verify-runs" in text and "verify-artifacts" in text
    assert "python -m build" not in text
    assert "verify_pypi_release.py" in text and "--no-cache-dir" in text


def test_normal_foundation_does_not_install_every_host():
    spec = importlib.util.spec_from_file_location(
        "matrix", ROOT / "scripts/ci/export_support_matrix.py"
    )
    assert spec and spec.loader
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    data = matrix.load_support_matrix()
    for lane in ("pr", "main"):
        rows = matrix.build_workflow_payload(data, "foundation", lane)["wheel_smoke_matrix"][
            "include"
        ]
        assert [row["label"] for row in rows] == ["base"]
    rows = matrix.build_workflow_payload(data, "foundation", "release")["wheel_smoke_matrix"][
        "include"
    ]
    assert [row["label"] for row in rows] == data["wheel_smoke"]["extras"]
