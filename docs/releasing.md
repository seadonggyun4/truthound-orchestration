# Releasing

`truthound-orchestration` publishes from the GitHub `Release Gate` workflow and uploads the built `truthound-foundation` artifact to PyPI.

## One-Time Setup

Before the first automated release, configure the repository secret:

- Secret name: `PYPI_API_TOKEN`
- Scope: repository secret on `seadonggyun4/truthound-orchestration`
- Value: the PyPI API token for the `truthound-orchestration` project

The publish workflow fails fast with a clear error if this secret is missing.

## Release Flow

1. Update `ci/version-surfaces.toml` with the next release version.
2. Run `python scripts/ci/sync_version_surfaces.py write`.
3. Commit both the manifest and the generated version-bearing files.
4. Ensure `main` is green.
5. Create and push a version tag such as `v3.0.1`.
6. Let `Release Gate` run on the tag, or run it manually with `publish_release=true`.
7. Verify that the nested `Publish` workflow finishes successfully.

## Version Source Of Truth

Do not hand-edit release-version literals in:

- `pyproject.toml` files
- adapter `version.py` modules
- `packages/dbt/dbt_project.yml`
- generated release-version snippets wired into CI checks

Release-version changes come from `ci/version-surfaces.toml`. Host compatibility changes still belong in `ci/support-matrix.toml`.

The foundation lane runs `python scripts/ci/sync_version_surfaces.py check`, and tagged release gates also verify that the pushed tag matches the manifest.

## Retry Behavior

The publish step uses `twine upload --skip-existing`.

That means a rerun is safe when:

- a previous attempt already uploaded one or more `3.x.y` files
- the release gate needs to be re-run after a transient GitHub or PyPI failure

PyPI still forbids replacing an existing file with different contents, so version numbers remain immutable.
