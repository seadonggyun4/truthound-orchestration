# CI/CD Strategy

> **Last Updated:** 2024-12-30
> **Document Version:** 1.0.0
> **Status:** Implementation Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Workflow Structure](#workflow-structure)
3. [CI Workflow (ci.yml)](#ci-workflow)
4. [Release Workflows](#release-workflows)
5. [Tag Convention](#tag-convention)
6. [Change Detection](#change-detection)
7. [PyPI Deployment](#pypi-deployment)
8. [dbt Hub Deployment](#dbt-hub-deployment)
9. [Security & Secrets](#security--secrets)

---

## Overview

### CI/CD Philosophy

| Principle | Description |
|-----------|-------------|
| **Package Independence** | 각 패키지는 독립적으로 릴리스 |
| **Tag-Based Release** | Git 태그로 릴리스 트리거 |
| **Change Detection** | 변경된 패키지만 테스트/배포 |
| **Quality Gates** | Lint, Type Check, Test 통과 필수 |
| **Semantic Versioning** | SemVer 2.0.0 준수 |

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GitHub Workflows                                 │
│                                                                         │
│  ┌────────────────┐                                                     │
│  │  Push/PR       │                                                     │
│  │  to main       │                                                     │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                        ci.yml                                   │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │  Lint    │  │  Type    │  │  Test    │  │  Build   │       │    │
│  │  │  (Ruff)  │─▶│  Check   │─▶│ (pytest) │─▶│  Check   │       │    │
│  │  └──────────┘  │  (MyPy)  │  └──────────┘  └──────────┘       │    │
│  │                └──────────┘                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌────────────────┐                                                     │
│  │  Tag Push      │                                                     │
│  │  airflow-v*    │                                                     │
│  └───────┬────────┘                                                     │
│          │                                                              │
│          ▼                                                              │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   release-airflow.yml                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │
│  │  │ Validate │  │  Build   │  │  Test    │  │  Publish │       │    │
│  │  │   Tag    │─▶│  Wheel   │─▶│  (PyPI)  │─▶│   PyPI   │       │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow Structure

### Directory Layout

```
.github/
├── workflows/
│   ├── ci.yml                 # 전체 CI
│   ├── release-airflow.yml    # Airflow 패키지 릴리스
│   ├── release-dagster.yml    # Dagster 패키지 릴리스
│   ├── release-prefect.yml    # Prefect 패키지 릴리스
│   └── release-dbt.yml        # dbt 패키지 릴리스
├── actions/
│   ├── setup-python/
│   │   └── action.yml         # Python 환경 설정 재사용 액션
│   └── detect-changes/
│       └── action.yml         # 변경 감지 재사용 액션
└── CODEOWNERS
```

---

## CI Workflow

### ci.yml

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: "3.11"
  UV_VERSION: "0.4.0"

jobs:
  # =========================================================================
  # Change Detection
  # =========================================================================
  detect-changes:
    name: Detect Changes
    runs-on: ubuntu-latest
    outputs:
      airflow: ${{ steps.filter.outputs.airflow }}
      dagster: ${{ steps.filter.outputs.dagster }}
      prefect: ${{ steps.filter.outputs.prefect }}
      dbt: ${{ steps.filter.outputs.dbt }}
      common: ${{ steps.filter.outputs.common }}
      any_package: ${{ steps.filter.outputs.any_package }}
    steps:
      - uses: actions/checkout@v4

      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            airflow:
              - 'packages/airflow/**'
              - 'common/**'
            dagster:
              - 'packages/dagster/**'
              - 'common/**'
            prefect:
              - 'packages/prefect/**'
              - 'common/**'
            dbt:
              - 'packages/dbt/**'
            common:
              - 'common/**'
            any_package:
              - 'packages/**'
              - 'common/**'

  # =========================================================================
  # Lint
  # =========================================================================
  lint:
    name: Lint
    runs-on: ubuntu-latest
    needs: detect-changes
    if: needs.detect-changes.outputs.any_package == 'true'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: ${{ env.UV_VERSION }}

      - name: Install dependencies
        run: |
          uv pip install ruff

      - name: Run Ruff Linter
        run: |
          ruff check packages/ common/ --output-format=github

      - name: Run Ruff Formatter Check
        run: |
          ruff format packages/ common/ --check

  # =========================================================================
  # Type Check
  # =========================================================================
  type-check:
    name: Type Check
    runs-on: ubuntu-latest
    needs: detect-changes
    if: needs.detect-changes.outputs.any_package == 'true'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: ${{ env.UV_VERSION }}

      - name: Install dependencies
        run: |
          uv pip install mypy types-PyYAML
          uv pip install -e packages/airflow[dev]
          uv pip install -e packages/dagster[dev]
          uv pip install -e packages/prefect[dev]

      - name: Run MyPy on common
        if: needs.detect-changes.outputs.common == 'true'
        run: |
          mypy common/ --strict

      - name: Run MyPy on Airflow
        if: needs.detect-changes.outputs.airflow == 'true'
        run: |
          mypy packages/airflow/src/ --strict

      - name: Run MyPy on Dagster
        if: needs.detect-changes.outputs.dagster == 'true'
        run: |
          mypy packages/dagster/src/ --strict

      - name: Run MyPy on Prefect
        if: needs.detect-changes.outputs.prefect == 'true'
        run: |
          mypy packages/prefect/src/ --strict

  # =========================================================================
  # Test - Airflow
  # =========================================================================
  test-airflow:
    name: Test Airflow
    runs-on: ubuntu-latest
    needs: [detect-changes, lint, type-check]
    if: needs.detect-changes.outputs.airflow == 'true'
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
        airflow-version: ["2.6.0", "2.7.0", "2.8.0"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: ${{ env.UV_VERSION }}

      - name: Install dependencies
        run: |
          uv pip install apache-airflow==${{ matrix.airflow-version }}
          uv pip install -e packages/airflow[dev]
          uv pip install -e common/

      - name: Run tests
        run: |
          pytest packages/airflow/tests/ -v --cov=truthound_airflow --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          flags: airflow
          name: airflow-${{ matrix.python-version }}-${{ matrix.airflow-version }}

  # =========================================================================
  # Test - Dagster
  # =========================================================================
  test-dagster:
    name: Test Dagster
    runs-on: ubuntu-latest
    needs: [detect-changes, lint, type-check]
    if: needs.detect-changes.outputs.dagster == 'true'
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
        dagster-version: ["1.5.0", "1.6.0", "1.7.0"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: ${{ env.UV_VERSION }}

      - name: Install dependencies
        run: |
          uv pip install dagster==${{ matrix.dagster-version }}
          uv pip install -e packages/dagster[dev]
          uv pip install -e common/

      - name: Run tests
        run: |
          pytest packages/dagster/tests/ -v --cov=truthound_dagster --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          flags: dagster
          name: dagster-${{ matrix.python-version }}-${{ matrix.dagster-version }}

  # =========================================================================
  # Test - Prefect
  # =========================================================================
  test-prefect:
    name: Test Prefect
    runs-on: ubuntu-latest
    needs: [detect-changes, lint, type-check]
    if: needs.detect-changes.outputs.prefect == 'true'
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
        prefect-version: ["2.14.0", "2.16.0", "2.18.0"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: ${{ env.UV_VERSION }}

      - name: Install dependencies
        run: |
          uv pip install prefect==${{ matrix.prefect-version }}
          uv pip install -e packages/prefect[dev]
          uv pip install -e common/

      - name: Run tests
        run: |
          pytest packages/prefect/tests/ -v --cov=truthound_prefect --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
          flags: prefect
          name: prefect-${{ matrix.python-version }}-${{ matrix.prefect-version }}

  # =========================================================================
  # Test - dbt
  # =========================================================================
  test-dbt:
    name: Test dbt
    runs-on: ubuntu-latest
    needs: [detect-changes, lint]
    if: needs.detect-changes.outputs.dbt == 'true'
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    strategy:
      matrix:
        dbt-version: ["1.6.0", "1.7.0", "1.8.0"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dbt
        run: |
          pip install dbt-postgres==${{ matrix.dbt-version }}

      - name: Run dbt deps
        working-directory: packages/dbt/integration_tests
        run: |
          dbt deps

      - name: Run dbt seed
        working-directory: packages/dbt/integration_tests
        env:
          DBT_PROFILES_DIR: .
        run: |
          dbt seed

      - name: Run dbt run
        working-directory: packages/dbt/integration_tests
        env:
          DBT_PROFILES_DIR: .
        run: |
          dbt run

      - name: Run dbt test
        working-directory: packages/dbt/integration_tests
        env:
          DBT_PROFILES_DIR: .
        run: |
          dbt test

  # =========================================================================
  # Build Check
  # =========================================================================
  build-check:
    name: Build Check
    runs-on: ubuntu-latest
    needs: [test-airflow, test-dagster, test-prefect, test-dbt]
    if: always() && !cancelled()
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: ${{ env.UV_VERSION }}

      - name: Install build tools
        run: |
          uv pip install build twine

      - name: Build Airflow package
        run: |
          cd packages/airflow && python -m build

      - name: Build Dagster package
        run: |
          cd packages/dagster && python -m build

      - name: Build Prefect package
        run: |
          cd packages/prefect && python -m build

      - name: Check packages with twine
        run: |
          twine check packages/airflow/dist/*
          twine check packages/dagster/dist/*
          twine check packages/prefect/dist/*

  # =========================================================================
  # CI Summary
  # =========================================================================
  ci-success:
    name: CI Success
    runs-on: ubuntu-latest
    needs: [build-check]
    if: always()
    steps:
      - name: Check CI result
        run: |
          if [ "${{ needs.build-check.result }}" != "success" ]; then
            echo "CI failed"
            exit 1
          fi
          echo "CI passed successfully!"
```

---

## Release Workflows

### release-airflow.yml

```yaml
# .github/workflows/release-airflow.yml
name: Release Airflow Package

on:
  push:
    tags:
      - 'airflow-v*'

permissions:
  contents: write
  id-token: write  # PyPI trusted publishing

env:
  PYTHON_VERSION: "3.11"
  PACKAGE_DIR: packages/airflow

jobs:
  # =========================================================================
  # Validate Tag
  # =========================================================================
  validate:
    name: Validate Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - uses: actions/checkout@v4

      - name: Extract version from tag
        id: version
        run: |
          TAG=${GITHUB_REF#refs/tags/airflow-v}
          echo "version=$TAG" >> $GITHUB_OUTPUT
          echo "Releasing version: $TAG"

      - name: Validate version format
        run: |
          VERSION=${{ steps.version.outputs.version }}
          if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
            echo "Invalid version format: $VERSION"
            exit 1
          fi

      - name: Check version in pyproject.toml
        run: |
          PYPROJECT_VERSION=$(grep -oP 'version = "\K[^"]+' ${{ env.PACKAGE_DIR }}/pyproject.toml)
          if [ "$PYPROJECT_VERSION" != "${{ steps.version.outputs.version }}" ]; then
            echo "Version mismatch: tag=${{ steps.version.outputs.version }}, pyproject=$PYPROJECT_VERSION"
            exit 1
          fi

  # =========================================================================
  # Test
  # =========================================================================
  test:
    name: Test Package
    runs-on: ubuntu-latest
    needs: validate
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: |
          uv pip install -e ${{ env.PACKAGE_DIR }}[dev]
          uv pip install -e common/

      - name: Run tests
        run: |
          pytest ${{ env.PACKAGE_DIR }}/tests/ -v

  # =========================================================================
  # Build
  # =========================================================================
  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [validate, test]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install build tools
        run: |
          pip install build

      - name: Build package
        run: |
          cd ${{ env.PACKAGE_DIR }}
          python -m build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: ${{ env.PACKAGE_DIR }}/dist/

  # =========================================================================
  # Publish to Test PyPI
  # =========================================================================
  publish-test:
    name: Publish to Test PyPI
    runs-on: ubuntu-latest
    needs: build
    environment: test-pypi
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to Test PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  # =========================================================================
  # Publish to PyPI
  # =========================================================================
  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: [build, publish-test]
    environment: pypi
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1

  # =========================================================================
  # Create GitHub Release
  # =========================================================================
  release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: [validate, publish]
    steps:
      - uses: actions/checkout@v4

      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Generate changelog
        id: changelog
        run: |
          # 간단한 changelog 생성
          echo "## Changes in truthound-airflow v${{ needs.validate.outputs.version }}" > CHANGELOG.md
          echo "" >> CHANGELOG.md
          git log --oneline $(git describe --tags --abbrev=0 HEAD^)..HEAD -- ${{ env.PACKAGE_DIR }} >> CHANGELOG.md

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          name: truthound-airflow v${{ needs.validate.outputs.version }}
          body_path: CHANGELOG.md
          files: dist/*
          draft: false
          prerelease: ${{ contains(needs.validate.outputs.version, '-') }}
```

### release-dagster.yml

```yaml
# .github/workflows/release-dagster.yml
name: Release Dagster Package

on:
  push:
    tags:
      - 'dagster-v*'

permissions:
  contents: write
  id-token: write

env:
  PYTHON_VERSION: "3.11"
  PACKAGE_DIR: packages/dagster

jobs:
  validate:
    name: Validate Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
      - name: Extract version
        id: version
        run: |
          TAG=${GITHUB_REF#refs/tags/dagster-v}
          echo "version=$TAG" >> $GITHUB_OUTPUT

  test:
    name: Test Package
    runs-on: ubuntu-latest
    needs: validate
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: astral-sh/setup-uv@v4
      - run: |
          uv pip install -e ${{ env.PACKAGE_DIR }}[dev]
          pytest ${{ env.PACKAGE_DIR }}/tests/ -v

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [validate, test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: |
          pip install build
          cd ${{ env.PACKAGE_DIR }} && python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: ${{ env.PACKAGE_DIR }}/dist/

  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: build
    environment: pypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1

  release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: [validate, publish]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: softprops/action-gh-release@v1
        with:
          name: truthound-dagster v${{ needs.validate.outputs.version }}
          files: dist/*
```

### release-prefect.yml

```yaml
# .github/workflows/release-prefect.yml
name: Release Prefect Package

on:
  push:
    tags:
      - 'prefect-v*'

permissions:
  contents: write
  id-token: write

env:
  PYTHON_VERSION: "3.11"
  PACKAGE_DIR: packages/prefect

jobs:
  validate:
    name: Validate Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
      - name: Extract version
        id: version
        run: |
          TAG=${GITHUB_REF#refs/tags/prefect-v}
          echo "version=$TAG" >> $GITHUB_OUTPUT

  test:
    name: Test Package
    runs-on: ubuntu-latest
    needs: validate
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: astral-sh/setup-uv@v4
      - run: |
          uv pip install -e ${{ env.PACKAGE_DIR }}[dev]
          pytest ${{ env.PACKAGE_DIR }}/tests/ -v

  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [validate, test]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: |
          pip install build
          cd ${{ env.PACKAGE_DIR }} && python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: ${{ env.PACKAGE_DIR }}/dist/

  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: build
    environment: pypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1

  release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: [validate, publish]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: softprops/action-gh-release@v1
        with:
          name: truthound-prefect v${{ needs.validate.outputs.version }}
          files: dist/*
```

### release-dbt.yml

```yaml
# .github/workflows/release-dbt.yml
name: Release dbt Package

on:
  push:
    tags:
      - 'dbt-v*'

permissions:
  contents: write

env:
  PACKAGE_DIR: packages/dbt

jobs:
  validate:
    name: Validate Release
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    steps:
      - uses: actions/checkout@v4
      - name: Extract version
        id: version
        run: |
          TAG=${GITHUB_REF#refs/tags/dbt-v}
          echo "version=$TAG" >> $GITHUB_OUTPUT

      - name: Check version in dbt_project.yml
        run: |
          DBT_VERSION=$(grep -oP "version: '\K[^']+" ${{ env.PACKAGE_DIR }}/dbt_project.yml)
          if [ "$DBT_VERSION" != "${{ steps.version.outputs.version }}" ]; then
            echo "Version mismatch"
            exit 1
          fi

  test:
    name: Test Package
    runs-on: ubuntu-latest
    needs: validate
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install dbt-postgres
      - name: Run dbt tests
        working-directory: ${{ env.PACKAGE_DIR }}/integration_tests
        env:
          DBT_PROFILES_DIR: .
        run: |
          dbt deps
          dbt seed
          dbt run
          dbt test

  release:
    name: Create GitHub Release
    runs-on: ubuntu-latest
    needs: [validate, test]
    steps:
      - uses: actions/checkout@v4
      - name: Package dbt files
        run: |
          cd ${{ env.PACKAGE_DIR }}
          tar -czvf truthound-dbt-${{ needs.validate.outputs.version }}.tar.gz \
            dbt_project.yml macros/ tests/
      - uses: softprops/action-gh-release@v1
        with:
          name: truthound-dbt v${{ needs.validate.outputs.version }}
          body: |
            ## Installation

            ```yaml
            packages:
              - git: "https://github.com/seadonggyun4/truthound-integrations.git"
                subdirectory: "packages/dbt"
                revision: "dbt-v${{ needs.validate.outputs.version }}"
            ```
          files: |
            ${{ env.PACKAGE_DIR }}/truthound-dbt-${{ needs.validate.outputs.version }}.tar.gz
```

---

## Tag Convention

### Tag Format

| Package | Tag Pattern | Example |
|---------|-------------|---------|
| truthound-airflow | `airflow-v*` | `airflow-v0.1.0` |
| truthound-dagster | `dagster-v*` | `dagster-v0.2.1` |
| truthound-prefect | `prefect-v*` | `prefect-v0.1.5` |
| truthound-dbt | `dbt-v*` | `dbt-v0.1.0` |

### Pre-release Tags

```bash
# Alpha
airflow-v0.1.0-alpha.1

# Beta
airflow-v0.1.0-beta.1

# Release Candidate
airflow-v0.1.0-rc.1
```

### Creating a Release

```bash
# 1. 버전 업데이트 (pyproject.toml 또는 dbt_project.yml)
# 2. 커밋
git add .
git commit -m "chore: bump airflow version to 0.1.0"

# 3. 태그 생성 및 푸시
git tag airflow-v0.1.0
git push origin airflow-v0.1.0
```

---

## Change Detection

### Reusable Action

```yaml
# .github/actions/detect-changes/action.yml
name: Detect Package Changes
description: Detect which packages have changed

outputs:
  airflow:
    description: "Airflow package changed"
    value: ${{ steps.filter.outputs.airflow }}
  dagster:
    description: "Dagster package changed"
    value: ${{ steps.filter.outputs.dagster }}
  prefect:
    description: "Prefect package changed"
    value: ${{ steps.filter.outputs.prefect }}
  dbt:
    description: "dbt package changed"
    value: ${{ steps.filter.outputs.dbt }}

runs:
  using: "composite"
  steps:
    - uses: dorny/paths-filter@v3
      id: filter
      with:
        filters: |
          airflow:
            - 'packages/airflow/**'
            - 'common/**'
          dagster:
            - 'packages/dagster/**'
            - 'common/**'
          prefect:
            - 'packages/prefect/**'
            - 'common/**'
          dbt:
            - 'packages/dbt/**'
```

---

## PyPI Deployment

### Trusted Publishing Setup

1. PyPI 프로젝트에서 Trusted Publisher 설정
2. GitHub Actions 워크플로우 연결
3. `id-token: write` 권한 추가

### Environment Protection

```yaml
# GitHub Repository Settings -> Environments

# test-pypi
- Environment name: test-pypi
- Required reviewers: (optional)
- Deployment branches: main

# pypi
- Environment name: pypi
- Required reviewers: 1+ (recommended)
- Deployment branches: main
```

---

## dbt Hub Deployment

### Manual Registration

1. [hub.getdbt.com](https://hub.getdbt.com) 접속
2. GitHub 저장소 연결
3. `packages/dbt` 서브디렉토리 지정
4. 버전 자동 동기화 활성화

### packages.yml Usage

```yaml
# 사용자 프로젝트의 packages.yml
packages:
  - package: truthound/truthound
    version: ">=0.1.0"

  # 또는 Git 직접 참조
  - git: "https://github.com/seadonggyun4/truthound-integrations.git"
    subdirectory: "packages/dbt"
    revision: "dbt-v0.1.0"
```

---

## Security & Secrets

### Required Secrets

| Secret | Description | Usage |
|--------|-------------|-------|
| `PYPI_API_TOKEN` | PyPI API 토큰 (레거시) | 수동 배포 시 |
| `CODECOV_TOKEN` | Codecov 업로드 토큰 | 커버리지 리포트 |

### Trusted Publishing (권장)

```yaml
# Trusted Publishing 사용 시 토큰 불필요
- uses: pypa/gh-action-pypi-publish@release/v1
  # with:
  #   password: ${{ secrets.PYPI_API_TOKEN }}  # 불필요
```

### CODEOWNERS

```
# .github/CODEOWNERS
* @seadonggyun4
packages/airflow/ @seadonggyun4
packages/dagster/ @seadonggyun4
packages/prefect/ @seadonggyun4
packages/dbt/ @seadonggyun4
common/ @seadonggyun4
```

---

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [dbt Hub](https://hub.getdbt.com)
- [Semantic Versioning](https://semver.org/)

---

*이 문서는 Truthound Integrations의 CI/CD 전략을 정의합니다.*
