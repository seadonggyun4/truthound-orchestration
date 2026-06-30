!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

# Releasing

`truthound-오케스트레이션` publishes from the GitHub `Release Gate` 워크플로우 and
uploads the built `truthound-오케스트레이션-dist` artifact to PyPI.

## One-Time Setup

Before the first automated release, configure the repository secret:

- Secret name: `PYPI_API_TOKEN`
- Scope: repository secret on `seadonggyun4/truthound-오케스트레이션`
- Value: the PyPI API token for the `truthound-오케스트레이션` project

The publish 워크플로우 fails fast with a clear error if this secret is missing.

## Release Flow

1. Update `ci/version-surfaces.toml` with the next release version.
2. Run `python scripts/ci/sync_version_surfaces.py write`.
3. Commit both the manifest and the generated version-bearing files.
4. Ensure `main` is green.
5. Create and push a version tag such as `v3.0.1`.
6. Let `Release Gate` run on the tag, or run it manually with `publish_release=true`.
7. Verify that the nested `Publish` 워크플로우 finishes successfully.

## Version Source Of Truth

Do not hand-edit release-version literals in:

- `pyproject.toml` files
- adapter `version.py` modules
- `packages/dbt/dbt_project.yml`
- generated release-version snippets wired into CI checks

Release-version changes come from `ci/version-surfaces.toml`. Host compatibility changes still belong in `ci/support-matrix.toml`.

The root release lane runs `python scripts/ci/sync_version_surfaces.py check`,
and tagged release gates also verify that the pushed tag matches the manifest.

## Retry Behavior

The publish step uses `twine upload --skip-existing`.

That means a rerun is safe when:

- a previous attempt already uploaded one or more `3.x.y` files
- the release gate needs to be re-run after a transient GitHub or PyPI failure

PyPI still forbids replacing an existing file with different contents, so version numbers remain immutable.
