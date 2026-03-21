"""Fail pull requests that add vulnerable dependencies."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from urllib import error, parse, request


API_VERSION = "2026-03-10"


def normalize_vulnerabilities(change: dict[str, object]) -> list[dict[str, object]]:
    """Return a typed vulnerability list from a dependency review change item."""

    raw_vulnerabilities = change.get("vulnerabilities")
    if not isinstance(raw_vulnerabilities, list):
        return []

    vulnerabilities: list[dict[str, object]] = []
    for vulnerability in raw_vulnerabilities:
        if isinstance(vulnerability, dict):
            vulnerabilities.append(vulnerability)
    return vulnerabilities


def fetch_dependency_changes(
    owner: str,
    repo: str,
    base: str,
    head: str,
    token: str,
    api_url: str = "https://api.github.com",
) -> list[dict[str, object]]:
    """Query GitHub's dependency review REST endpoint."""

    basehead = parse.quote(f"{base}...{head}", safe="")
    url = f"{api_url}/repos/{owner}/{repo}/dependency-graph/compare/{basehead}"
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
            f"Failed to fetch dependency review data from GitHub ({exc.code}): {message}"
        ) from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Unexpected dependency review payload: {payload!r}")
    return payload


def find_vulnerable_additions(changes: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return added dependencies that introduce known vulnerabilities."""

    findings: list[dict[str, object]] = []
    for change in changes:
        if change.get("change_type") != "added":
            continue
        vulnerabilities = normalize_vulnerabilities(change)
        if not vulnerabilities:
            continue
        findings.append(change)
    return findings


def format_findings(findings: Iterable[dict[str, object]]) -> str:
    """Render a stable, readable failure summary."""

    lines = ["Dependency review found vulnerable added dependencies:"]
    for finding in findings:
        name = finding.get("name", "<unknown>")
        version = finding.get("version", "<unknown>")
        manifest = finding.get("manifest", "<unknown>")
        advisories = ", ".join(
            (
                f"{vuln.get('advisory_ghsa_id', 'UNKNOWN')} ({vuln.get('severity', 'unknown')})"
            )
            for vuln in normalize_vulnerabilities(finding)
        )
        lines.append(f"- {name} {version} via {manifest}: {advisories}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Repository owner.")
    parser.add_argument("--repo", required=True, help="Repository name.")
    parser.add_argument("--base", required=True, help="Base commit SHA.")
    parser.add_argument("--head", required=True, help="Head commit SHA.")
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
        raise SystemExit("GITHUB_TOKEN is required to query dependency review data.")

    changes = fetch_dependency_changes(
        owner=args.owner,
        repo=args.repo,
        base=args.base,
        head=args.head,
        token=token,
        api_url=args.api_url,
    )
    findings = find_vulnerable_additions(changes)
    if findings:
        raise SystemExit(format_findings(findings))
    print("Dependency review passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
