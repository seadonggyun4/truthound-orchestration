"""Lightweight documentation contract checks for Truthound Orchestration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [
    ROOT / "mkdocs.yml",
    ROOT / "docs/index.md",
    ROOT / "docs/getting-started.md",
    ROOT / "docs/architecture.md",
    ROOT / "docs/compatibility.md",
    ROOT / "docs/zero-config.md",
    ROOT / "docs/migration/3.0.md",
    ROOT / "docs/engines/index.md",
]


def _require(path: Path, needle: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        return f"{path.relative_to(ROOT)} is missing required text: {needle!r}"
    return None


def _require_any(path: Path, needles: tuple[str, ...], *, label: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    if any(needle in text for needle in needles):
        return None
    return f"{path.relative_to(ROOT)} is missing required text for {label!r}: {needles!r}"


def main() -> int:
    errors: list[str] = []

    for path in DOC_FILES:
        if not path.exists():
            errors.append(f"Missing required docs file: {path.relative_to(ROOT)}")

    if errors:
        print("\n".join(errors))
        return 1

    checks = [
        _require(ROOT / "README.md", "official orchestration integrations for Truthound 3.x"),
        _require(ROOT / "README.md", "Truthound 3.x Compatibility"),
        _require_any(
            ROOT / "docs/index.md",
            (
                "official first-party orchestration compatibility line",
                "official first-party orchestration line for Truthound 3.x",
            ),
            label="homepage compatibility positioning",
        ),
        _require(ROOT / "docs/zero-config.md", "safe_auto"),
        _require(ROOT / "docs/compatibility.md", "supports `Truthound 3.x` only"),
        _require(ROOT / "docs/compatibility.md", "BEGIN GENERATED SUPPORT MATRIX"),
    ]
    errors.extend(error for error in checks if error is not None)

    support_matrix = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci/export_support_matrix.py"),
            "sync-docs",
            "--path",
            str(ROOT / "docs/compatibility.md"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if support_matrix.returncode != 0:
        errors.append(support_matrix.stdout.strip() or support_matrix.stderr.strip())

    if errors:
        print("\n".join(errors))
        return 1

    print("documentation contracts verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
