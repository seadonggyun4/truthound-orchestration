"""Install built wheels into isolated virtualenvs and verify representative extras."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import venv


PACKAGE_NAME = "truthound-orchestration"

IMPORT_MAP = {
    "base": ["common"],
    "airflow": ["common", "truthound_airflow"],
    "prefect": ["common", "truthound_prefect"],
    "dagster": ["common", "truthound_dagster"],
    "dbt": ["common", "truthound_dbt", "dbt"],
    "opentelemetry": ["common", "opentelemetry"],
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--label", choices=sorted(IMPORT_MAP), required=True)
    parser.add_argument("--extra", default="")
    return parser.parse_args(argv)


def locate_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.rglob("truthound_orchestration-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError(f"Expected exactly one wheel in {dist_dir}, found {len(wheels)}")
    return wheels[0]


def build_install_spec(wheel: Path, extra: str) -> str:
    wheel_uri = wheel.resolve().as_uri()
    if not extra:
        return f"{PACKAGE_NAME} @ {wheel_uri}"
    return f"{PACKAGE_NAME}[{extra}] @ {wheel_uri}"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    wheel = locate_wheel(args.dist_dir)
    with tempfile.TemporaryDirectory(prefix=f"truthound-wheel-smoke-{args.label}-") as tmpdir:
        venv_dir = Path(tmpdir) / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
        python = venv_dir / "bin" / "python"

        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", build_install_spec(wheel, args.extra)])

        imports = "; ".join(f"import {module}" for module in IMPORT_MAP[args.label])
        run([str(python), "-c", imports])
    return 0


if __name__ == "__main__":
    sys.exit(main())
