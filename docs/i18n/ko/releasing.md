# 배포

검증은 독립 워크플로에서 수행하고, Release Gate는 그 결과만 확인합니다.
Publish는 별도로 실행합니다. 태그를 푸시해도 전체 플랫폼 실행이나 배포가
자동으로 시작되지 않습니다.

## 최초 설정

저장소 secret PYPI_API_TOKEN에 truthound-orchestration PyPI 프로젝트로
범위를 제한한 토큰을 설정합니다. 토큰을 입력값이나 파일에 기록하지 마세요.

## 배포 순서

1. ci/version-surfaces.toml을 수정하고
   python scripts/ci/sync_version_surfaces.py write를 실행합니다.
2. 매니페스트와 동기화한 파일을 커밋·푸시합니다.
3. ci-foundation.yml을 lane=release로 실행합니다.
4. shared-runtime.yml 및 airflow.yml, prefect.yml, dagster.yml, mage.yml,
   kestra.yml, dbt.yml을 각각 lane=release, suite_tier=full로 실행합니다.
   security.yml도 lane=release로 실행합니다. 서로 다른 시점에 실행해도 되며
   어떤 워크플로도 다른 플랫폼을 일괄 실행하지 않습니다.
5. release-gate.yml의 verification_runs에 위 9개 워크플로 파일명과 숫자
   실행 ID의 JSON 객체를 전달합니다. 모두 동일 커밋의 전체 릴리스 검증이어야 합니다.
6. 동일 커밋에서 publish.yml을 실행하고, 성공한 Release Gate 실행 ID를
   verification_run_id에 전달합니다.
7. 공개 PyPI 파일의 SHA-256 일치와 새 환경 설치, 실제 Core 3.3 정상·불량
   데이터 검증까지 Publish가 성공했는지 확인합니다.
8. 필요하면 검증한 커밋에 태그를 붙입니다. 태그는 배포를 시작하지 않습니다.

커밋이 바뀌면 다시 검증해야 합니다. smoke 실행, 최소 버전 작업 누락, 필수 검사
건너뛰기, 실패·취소, 과거 커밋의 결과를 거부합니다. 지원하는 최소·기본 호스트,
6개 wheel extra, dbt 컴파일·실행, 차단형 보안 검사 범위는 유지합니다.
Mage 호스트의 보안 제한은 [호환성](compatibility.md)에 명시하며 기능 검증
통과와 구분합니다.

## 배포 파일 무결성

Foundation이 wheel과 sdist를 한 번 만들고 버전·커밋·SHA-256을
release-manifest.json에 기록합니다. Release Gate는 동일 파일을 확인합니다.
Publish는 참조한 검사 결과와 해시를 다시 확인하고 같은 파일을 다운로드하여
재빌드 없이 배포합니다. 공개 PyPI 해시 일치 후 새 설치를 검증합니다.
내부 에이전트 지침·로컬 검증 기록·비공개 서버 설정은 패키지에 포함하지 않습니다.

## 버전 관리

패키지 버전은 ci/version-surfaces.toml, 호스트 호환성은 ci/support-matrix.toml을
수정하고 동기화 도구를 사용합니다.

## 재시도

부분 업로드 복구를 위해 twine --skip-existing을 사용한 후 공개 파일의 해시를
비교합니다. 기존 파일의 내용이 다르면 실패하며, 업로드 생략을 성공 증거로
간주하지 않습니다.
