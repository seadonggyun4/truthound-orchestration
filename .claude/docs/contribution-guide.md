# Contribution Guide

> **Last Updated:** 2024-12-30
> **Document Version:** 1.0.0
> **Status:** Active

---

## Table of Contents
1. [Welcome](#welcome)
2. [Development Environment Setup](#development-environment-setup)
3. [Adding a New Integration](#adding-a-new-integration)
4. [Pull Request Process](#pull-request-process)
5. [Code Style](#code-style)
6. [Commit Convention](#commit-convention)
7. [Release Process](#release-process)

---

## Welcome

Truthound Integrations에 기여해 주셔서 감사합니다! 이 가이드는 프로젝트에 기여하는 방법을 설명합니다.

### Ways to Contribute

| Type | Description |
|------|-------------|
| 🐛 **Bug Reports** | 버그 발견 시 이슈 생성 |
| ✨ **Feature Requests** | 새 기능 제안 |
| 📖 **Documentation** | 문서 개선 |
| 🔧 **Code** | 버그 수정, 기능 구현 |
| 🧪 **Tests** | 테스트 추가/개선 |
| 🌐 **New Integrations** | 새 플랫폼 통합 추가 |

---

## Development Environment Setup

### Prerequisites

- Python 3.11+
- uv (권장) 또는 pip
- Git
- Docker (통합 테스트용)

### Initial Setup

```bash
# 1. 저장소 Fork 및 Clone
git clone https://github.com/YOUR_USERNAME/truthound-integrations.git
cd truthound-integrations

# 2. uv 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 가상환경 생성 및 의존성 설치
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 4. 개발 의존성 설치
uv pip install -e ".[dev]"
uv pip install -e packages/airflow[dev]
uv pip install -e packages/dagster[dev]
uv pip install -e packages/prefect[dev]

# 5. pre-commit 훅 설치
pre-commit install
```

### IDE Setup

#### VS Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.formatting.provider": "none",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": true,
            "source.organizeImports.ruff": true
        }
    },
    "python.analysis.typeCheckingMode": "strict",
    "ruff.lint.args": ["--config=ruff.toml"]
}
```

#### PyCharm

1. Settings → Project → Python Interpreter → `.venv/bin/python` 선택
2. Settings → Tools → External Tools → Ruff 추가
3. Settings → Editor → Inspections → Python → Type Checker → MyPy 활성화

### Verification

```bash
# 린트 확인
ruff check .

# 타입 체크
mypy packages/ common/

# 테스트 실행
pytest

# 모든 체크 실행
pre-commit run --all-files
```

---

## Adding a New Integration

새 워크플로우 플랫폼 통합을 추가하는 가이드입니다.

### Step 1: Package Structure

```bash
# 예: Mage 통합 추가
mkdir -p packages/mage/src/truthound_mage
mkdir -p packages/mage/tests

# 기본 파일 생성
touch packages/mage/pyproject.toml
touch packages/mage/README.md
touch packages/mage/src/truthound_mage/__init__.py
```

### Step 2: pyproject.toml

```toml
# packages/mage/pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "truthound-mage"
version = "0.1.0"
description = "Mage integration for Truthound"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
authors = [
    { name = "Your Name", email = "your@email.com" }
]

dependencies = [
    "mage-ai>=0.9.0",
    "truthound>=1.0.0",
    "polars>=0.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "mypy>=1.5.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/truthound_mage"]
```

### Step 3: Implement Core Components

```python
# packages/mage/src/truthound_mage/__init__.py
"""Truthound integration for Mage."""

from truthound_mage.blocks import TruthoundBlock
from truthound_mage.decorators import truthound_check, truthound_profile

__all__ = [
    "TruthoundBlock",
    "truthound_check",
    "truthound_profile",
]

__version__ = "0.1.0"
```

```python
# packages/mage/src/truthound_mage/blocks.py
from common.base import WorkflowIntegration, CheckConfig, CheckResult
import polars as pl


class MageTruthoundAdapter:
    """Mage용 Truthound 어댑터"""

    @property
    def platform_name(self) -> str:
        return "mage"

    @property
    def platform_version(self) -> str:
        return ">=0.9.0"

    def check(
        self,
        data: pl.DataFrame,
        config: CheckConfig,
    ) -> CheckResult:
        import truthound as th

        result = th.check(data, **config.to_truthound_kwargs())
        return CheckResult.from_truthound(result)


class TruthoundBlock:
    """Mage Block으로 사용할 Truthound 래퍼"""

    def __init__(self, **kwargs):
        self.adapter = MageTruthoundAdapter()
        self.config = kwargs

    def check(self, data: pl.DataFrame, rules: list) -> dict:
        config = CheckConfig(
            rules=tuple(rules),
            **self.config,
        )
        result = self.adapter.check(data, config)
        return result.to_dict()
```

### Step 4: Add Tests

```python
# packages/mage/tests/test_blocks.py
import pytest
import polars as pl
from truthound_mage import TruthoundBlock


class TestTruthoundBlock:

    @pytest.fixture
    def sample_data(self) -> pl.DataFrame:
        return pl.DataFrame({
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
        })

    def test_check_success(self, sample_data):
        block = TruthoundBlock()
        result = block.check(
            sample_data,
            rules=[{"column": "id", "type": "not_null"}],
        )
        assert result["is_success"]
```

### Step 5: Documentation

```markdown
<!-- .claude/docs/package-mage.md -->
# Package: truthound-mage

> **Last Updated:** 2024-12-30
> **Document Version:** 1.0.0
> **Package Version:** 0.1.0

## Overview

`truthound-mage`는 Mage AI용 Truthound 통합 패키지입니다.

## Installation

```bash
pip install truthound-mage
```

## Usage

```python
from truthound_mage import TruthoundBlock

block = TruthoundBlock()
result = block.check(df, rules=[...])
```
```

### Step 6: CI/CD

```yaml
# .github/workflows/release-mage.yml
name: Release Mage Package

on:
  push:
    tags:
      - 'mage-v*'

# ... (release-airflow.yml과 동일한 구조)
```

### Step 7: Update CLAUDE.md

```markdown
### truthound-mage
Mage AI용 공식 통합 패키지

| Component | Description |
|-----------|-------------|
| `TruthoundBlock` | Mage Block 구현 |
| `truthound_check` | 데코레이터 함수 |

```python
pip install truthound-mage
```
```

---

## Pull Request Process

### Before Creating a PR

1. **Issue 확인**: 관련 이슈가 있는지 확인
2. **Branch 생성**: `feature/`, `fix/`, `docs/` 접두사 사용
3. **테스트**: 모든 테스트 통과 확인
4. **린트**: `ruff check .` 통과
5. **타입 체크**: `mypy packages/` 통과

### PR Checklist

```markdown
## Description
<!-- 변경 사항 설명 -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Checklist
- [ ] I have read the CONTRIBUTING guide
- [ ] My code follows the code style of this project
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass
- [ ] I have updated the documentation accordingly
- [ ] I have updated the CHANGELOG (if applicable)

## Related Issues
<!-- Closes #123 -->
```

### PR Template

```markdown
<!-- .github/pull_request_template.md -->
## Summary

<!-- 이 PR이 해결하는 문제나 추가하는 기능 -->

## Changes

<!-- 주요 변경 사항 목록 -->
-
-
-

## Testing

<!-- 테스트 방법 -->
- [ ] 단위 테스트 추가/수정
- [ ] 통합 테스트 추가/수정
- [ ] 수동 테스트 완료

## Screenshots (if applicable)

## Notes for Reviewers

<!-- 리뷰어가 알아야 할 사항 -->
```

### Review Process

1. **자동 체크**: CI 통과 필수
2. **코드 리뷰**: 최소 1명의 승인 필요
3. **변경 요청**: 피드백 반영 후 재요청
4. **Merge**: Squash and merge 사용

---

## Code Style

### Ruff Configuration

```toml
# ruff.toml
line-length = 100
target-version = "py311"
src = ["packages/*/src", "common"]

[lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "ARG",  # flake8-unused-arguments
    "SIM",  # flake8-simplify
    "PTH",  # flake8-use-pathlib
    "RUF",  # Ruff-specific
]

ignore = [
    "E501",  # line-length (handled by formatter)
]

[lint.isort]
known-first-party = ["truthound_airflow", "truthound_dagster", "truthound_prefect", "common"]

[format]
quote-style = "double"
indent-style = "space"
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| **Package** | lowercase | `truthound_airflow` |
| **Module** | lowercase | `check_operator.py` |
| **Class** | PascalCase | `TruthoundCheckOperator` |
| **Function** | snake_case | `truthound_check()` |
| **Constant** | UPPER_SNAKE | `DEFAULT_TIMEOUT` |
| **Variable** | snake_case | `check_result` |

### Docstring Style

Google Style Docstrings 사용:

```python
def truthound_check(
    data: pl.DataFrame,
    rules: list[dict[str, Any]],
    *,
    fail_on_error: bool = True,
) -> CheckResult:
    """
    데이터 품질 검증 실행.

    이 함수는 Truthound를 사용하여 데이터 품질을 검증합니다.

    Parameters
    ----------
    data : pl.DataFrame
        검증할 데이터

    rules : list[dict[str, Any]]
        적용할 검증 규칙 목록.
        예: [{"column": "email", "type": "regex", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"}]

    fail_on_error : bool
        검증 실패 시 예외 발생 여부. 기본값: True

    Returns
    -------
    CheckResult
        검증 결과 객체

    Raises
    ------
    TruthoundCheckError
        fail_on_error=True이고 검증 실패 시

    Examples
    --------
    >>> result = truthound_check(
    ...     df,
    ...     rules=[{"column": "id", "type": "not_null"}],
    ... )
    >>> print(result.is_success)
    True

    Notes
    -----
    대용량 데이터의 경우 sample_size 파라미터 사용을 권장합니다.
    """
```

### Type Hints

모든 함수에 완전한 타입 힌트 필수:

```python
from typing import Any
import polars as pl

def process_data(
    data: pl.DataFrame | pl.LazyFrame,
    config: dict[str, Any],
    *,
    timeout: int = 300,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """처리 함수"""
    ...
```

---

## Commit Convention

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | 새 기능 |
| `fix` | 버그 수정 |
| `docs` | 문서 변경 |
| `style` | 코드 스타일 (포맷팅 등) |
| `refactor` | 리팩토링 |
| `test` | 테스트 추가/수정 |
| `chore` | 빌드, 도구 변경 |
| `perf` | 성능 개선 |
| `ci` | CI 설정 변경 |

### Scopes

| Scope | Description |
|-------|-------------|
| `airflow` | Airflow 패키지 |
| `dagster` | Dagster 패키지 |
| `prefect` | Prefect 패키지 |
| `dbt` | dbt 패키지 |
| `common` | 공통 모듈 |
| `ci` | CI/CD |
| `docs` | 문서 |

### Examples

```bash
# 기능 추가
feat(airflow): add TruthoundSensor for quality monitoring

# 버그 수정
fix(dagster): resolve resource initialization error

# 문서 업데이트
docs(prefect): add usage examples for TruthoundBlock

# 리팩토링
refactor(common): simplify CheckResult serialization

# CI 변경
ci: add Python 3.12 to test matrix
```

### Breaking Changes

```bash
feat(airflow)!: change TruthoundCheckOperator API

BREAKING CHANGE: The `rules` parameter now expects a list of dicts
instead of a single dict. Update your DAGs accordingly.

Before:
  TruthoundCheckOperator(rules={"column": "id", "type": "not_null"})

After:
  TruthoundCheckOperator(rules=[{"column": "id", "type": "not_null"}])
```

---

## Release Process

### Version Bumping

각 패키지는 독립적인 버전을 가집니다:

```bash
# 1. 버전 업데이트
# packages/airflow/pyproject.toml
version = "0.2.0"

# 2. CHANGELOG 업데이트
# packages/airflow/CHANGELOG.md

# 3. 커밋
git add packages/airflow/
git commit -m "chore(airflow): bump version to 0.2.0"

# 4. 태그 생성
git tag airflow-v0.2.0

# 5. 푸시
git push origin main
git push origin airflow-v0.2.0
```

### CHANGELOG Format

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2024-12-30

### Added
- New `TruthoundSensor` for quality monitoring
- Support for Airflow 2.8.0

### Changed
- Improved error messages in `TruthoundCheckOperator`

### Fixed
- Fixed XCom serialization issue with large results

### Deprecated
- `use_legacy_api` parameter (will be removed in 0.3.0)

## [0.1.0] - 2024-12-01

### Added
- Initial release
- `TruthoundCheckOperator`
- `TruthoundProfileOperator`
- `TruthoundHook`
```

### Pre-release

```bash
# Alpha
git tag airflow-v0.2.0-alpha.1

# Beta
git tag airflow-v0.2.0-beta.1

# Release Candidate
git tag airflow-v0.2.0-rc.1
```

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/seadonggyun4/truthound-integrations/issues)
- **Discussions**: [GitHub Discussions](https://github.com/seadonggyun4/truthound-integrations/discussions)
- **Email**: team@truthound.dev

---

*이 문서는 Truthound Integrations 기여 가이드입니다.*
