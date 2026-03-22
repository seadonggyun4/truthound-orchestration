"""Classify pull request file changes for CI workflow routing."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from urllib import error, request


API_VERSION = "2026-03-10"
FILTERS: dict[str, tuple[str, ...]] = {
    "docs": (
        "docs/**",
        "README.md",
        "mkdocs.yml",
    ),
    "all_platforms": (
        ".github/workflows/**",
        "ci/**",
        "scripts/ci/**",
        "common/**",
        "tests/common/**",
        "pyproject.toml",
    ),
    "airflow": ("packages/airflow/**",),
    "prefect": ("packages/prefect/**",),
    "dagster": ("packages/dagster/**",),
    "mage": (
        "packages/mage/**",
        "tests/mage/**",
    ),
    "kestra": (
        "packages/kestra/**",
        "tests/kestra/**",
    ),
    "dbt": (
        "packages/dbt/**",
        "tests/dbt/**",
    ),
}


def collect_changed_paths(file_entries: Iterable[dict[str, object]]) -> list[str]:
    """Normalize changed paths from the pull request files API."""

    paths: set[str] = set()
    for entry in file_entries:
        filename = entry.get("filename")
        if isinstance(filename, str) and filename:
            paths.add(filename)
        previous_filename = entry.get("previous_filename")
        if isinstance(previous_filename, str) and previous_filename:
            paths.add(previous_filename)
    return sorted(paths)


def classify_changed_files(paths: Iterable[str]) -> dict[str, bool]:
    """Map changed paths onto CI routing filters."""

    changed_paths = tuple(paths)
    outputs = {
        name: any(
            fnmatchcase(changed_path, pattern)
            for pattern in patterns
            for changed_path in changed_paths
        )
        for name, patterns in FILTERS.items()
    }
    outputs["docs_only"] = outputs["docs"] and not any(
        outputs[name]
        for name in ("all_platforms", "airflow", "prefect", "dagster", "mage", "kestra", "dbt")
    )
    return outputs


def fetch_pull_request_files(
    repository: str,
    pull_request_number: int,
    token: str,
    api_url: str = "https://api.github.com",
) -> list[dict[str, object]]:
    """Fetch the pull request files list from the GitHub REST API."""

    all_entries: list[dict[str, object]] = []
    page = 1
    while True:
        url = (
            f"{api_url}/repos/{repository}/pulls/{pull_request_number}/files"
            f"?per_page=100&page={page}"
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        }
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req) as response:
                payload = json.load(response)
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise SystemExit(
                f"Failed to fetch pull request files from GitHub ({exc.code}): {message}"
            ) from exc
        if not isinstance(payload, list):
            raise SystemExit(f"Unexpected pull request files payload: {payload!r}")
        all_entries.extend(payload)
        if len(payload) < 100:
            break
        page += 1
    return all_entries


def write_outputs(outputs: dict[str, bool], github_output: Path) -> None:
    """Write workflow step outputs in GitHub Actions format."""

    with github_output.open("a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            handle.write(f"{key}={'true' if value else 'false'}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="Repository in owner/name form.")
    parser.add_argument("--pull-request-number", type=int, required=True, help="Pull request number.")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT"),
        help="Path to the GitHub Actions output file.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("GITHUB_API_URL", "https://api.github.com"),
        help="GitHub REST API base URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is required to query pull request files.")
    if not args.github_output:
        raise SystemExit("A GitHub output path is required.")

    file_entries = fetch_pull_request_files(
        repository=args.repository,
        pull_request_number=args.pull_request_number,
        token=token,
        api_url=args.api_url,
    )
    outputs = classify_changed_files(collect_changed_paths(file_entries))
    write_outputs(outputs, Path(args.github_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
