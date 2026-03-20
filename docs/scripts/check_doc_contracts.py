"""Lightweight documentation contract checks for Truthound Orchestration."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DOC_FILES = [
    ROOT / "mkdocs.yml",
    ROOT / "docs/index.md",
    ROOT / "docs/getting-started.md",
    ROOT / "docs/architecture.md",
    ROOT / "docs/compatibility.md",
    ROOT / "docs/zero-config.md",
    ROOT / "docs/migration/3.0.md",
    ROOT / "docs/advanced-engines/index.md",
]


def _require(path: Path, needle: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        return f"{path.relative_to(ROOT)} is missing required text: {needle!r}"
    return None


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
        _require(ROOT / "docs/index.md", "official first-party orchestration compatibility line"),
        _require(ROOT / "docs/zero-config.md", "safe_auto"),
        _require(ROOT / "docs/compatibility.md", "supports `Truthound 3.x` only"),
    ]
    errors.extend(error for error in checks if error is not None)

    if errors:
        print("\n".join(errors))
        return 1

    print("documentation contracts verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
