"""Verify public PyPI file hashes against the already verified release artifact."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads((args.dist_dir / "release-manifest.json").read_text())
    url = f"https://pypi.org/pypi/truthound-orchestration/{expected['version']}/json"
    for _ in range(30):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                published = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        else:
            hashes = {file["filename"]: file["digests"]["sha256"] for file in published["urls"]}
            if set(expected["files"]) <= hashes.keys():
                for name, digest in expected["files"].items():
                    if hashes[name] != digest:
                        raise ValueError(f"PyPI artifact mismatch: {name}")
                print(f"Verified PyPI {expected['version']} wheel and sdist SHA-256")
                return
        time.sleep(10)
    raise TimeoutError("Published files did not appear on PyPI")


if __name__ == "__main__":
    main()
