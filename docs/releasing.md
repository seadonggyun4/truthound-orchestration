# Releasing

Releases use independent verification runs, an evidence-only Release Gate, and
a separately dispatched Publish workflow. Pushing a tag never publishes or starts
all platforms.

## One-Time Setup

Configure the repository secret named PYPI_API_TOKEN with a token scoped to the
truthound-orchestration PyPI project. Never put it in workflow inputs or files.

## Release Flow

1. Update ci/version-surfaces.toml and run
   python scripts/ci/sync_version_surfaces.py write.
2. Commit and push the manifest and synchronized files.
3. Run ci-foundation.yml with lane=release.
4. Run shared-runtime.yml and each of airflow.yml, prefect.yml, dagster.yml,
   mage.yml, kestra.yml and dbt.yml independently with lane=release and
   suite_tier=full. Run security.yml with lane=release. These runs may be
   completed separately; none dispatches the others.
5. Dispatch release-gate.yml with verification_runs: a JSON object mapping the
   nine workflow filenames to their numeric run IDs. Every run must belong to
   this exact commit, be a full release run and finish successfully.
6. Dispatch publish.yml on that same commit with verification_run_id set to the
   successful Release Gate run ID.
7. Confirm Publish succeeds, including public PyPI SHA-256 comparison and a fresh
   installation from the public index with real Core 3.3 good/bad checks.
8. Optionally tag the already verified commit. Tags do not cause publication.

A new commit requires new verification runs. Smoke runs, missing minimum-version
jobs, skipped required checks, failed/cancelled runs and older commits are rejected.
Release coverage still includes every supported minimum/primary host tuple,
all six wheel extras, dbt compile/execution and blocking security surfaces.
Mage's known host dependency security limitation remains explicit in
[Compatibility](compatibility.md); a functional pass does not resolve it.

## Artifact Integrity

Foundation builds one wheel and one sdist and records their version, commit and
SHA-256 in release-manifest.json. Release Gate validates those exact files.
Publish rechecks all referenced runs and hashes, downloads the same artifacts,
and never rebuilds. PyPI hashes must match before fresh-install verification.
The wheel/sdist exclude internal agent instructions, local verification records
and private server configuration.

## Version Source Of Truth

Use ci/version-surfaces.toml for package versions and ci/support-matrix.toml for
host compatibility. Run the synchronization helper instead of editing version
literals individually.

## Retry Behavior

Publish uses twine --skip-existing to recover a partial upload, then compares
the public files against the verified hashes. Existing different bytes fail
verification; skipping an upload is never treated as proof of a correct release.
