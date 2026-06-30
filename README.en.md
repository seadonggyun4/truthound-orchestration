<div align="center">
  <img width="500" alt="Truthound Orchestration Banner" src="assets/truthound-icon-banner.png" />
</div>

<h1 align="center">Truthound Orchestration — Data Quality Workflow</h1>

<p align="center">
  <strong>Official integration layer for running Truthound data quality checks in workflow orchestration environments</strong> <br/>
  <strong>Truthound 데이터 품질 검증을 워크플로우 오케스트레이션 환경에서 실행하기 위한 공식 통합 레이어</strong>
</p>

<p align="center">
  <em>Run Truthound quality checks where your pipelines already live.</em>
</p>

<p align="center">
  <a href="https://truthound.netlify.app/orchestration/"><img src="https://img.shields.io/badge/docs-truthound.netlify.app%2Forchestration-blue" alt="Documentation"></a>
  <a href="https://pypi.org/project/truthound-orchestration/"><img src="https://img.shields.io/pypi/v/truthound-orchestration.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="License"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code Style: Ruff"></a>
  <a href="https://pepy.tech/project/truthound-orchestration">
    <img src="https://img.shields.io/pepy/dt/truthound-orchestration?color=brightgreen" alt="Downloads">
  </a>
</p>

<!--
README HEADER FORMAT LOCK:
The exact header format above MUST be preserved.
Do not rewrite it as Markdown-only syntax, do not remove the centered HTML,
do not change the banner image block, title, bilingual subtitle, slogan, or
badge block, and do not reorder those elements.
This required format starts at:
  <div align="center">
and ends at the closing </p> of the badge block above.
-->

---

## Overview

**Truthound Orchestration** is the official first-party integration layer for running [Truthound](https://github.com/seadonggyun4/truthound) data quality checks across major workflow orchestration platforms such as Airflow, Dagster, Prefect, dbt, Mage, and Kestra.

Truthound Orchestration은 주요 워크플로우 플랫폼에서 Truthound 데이터 품질 검증을 실행하기 위한 공식 통합 레이어입니다.

**한국어 README**: [README.md](README.md) <br/>
**Documentation**: [truthound.netlify.app/orchestration](https://truthound.netlify.app/orchestration/)

---

## Motivation

Data quality checks need to run repeatedly inside real pipelines, including batch processing, model training, reporting, and operational data products. But every orchestration tool has different conventions for execution, result passing, retries, and alerts, so validation logic can easily become fragmented.

Truthound Orchestration provides a standard integration layer that preserves Truthound validation semantics and result contracts while exposing them through the native patterns of each workflow platform.

---

## Introduction

Truthound Orchestration is an open-source workflow integration project for Truthound 3.x. It connects data quality checks to existing pipelines through Airflow Operators, Dagster Resources/Ops, Prefect Blocks/Tasks, dbt macros, Mage blocks, and Kestra scripts/templates.

This project does not replace the Truthound validation kernel. Truthound owns data quality validation and result models, while Truthound Orchestration helps run, deliver, and observe those checks inside schedulers and workflow systems.

| Component | Repository | Role |
| --- | --- | --- |
| **Truthound** | [`truthound`](https://github.com/seadonggyun4/truthound) | Data quality validation kernel, `th.check()`, `ValidationRunResult`, reporters, checkpoints |
| **Truthound Orchestration** | [`truthound-orchestration`](https://github.com/seadonggyun4/truthound-orchestration) | Workflow integration layer for Airflow, Dagster, Prefect, dbt, Mage, Kestra, and similar environments |

---

## Impact

Truthound Orchestration lets teams treat data quality checks as part of operational workflows rather than standalone scripts. Teams can keep one Truthound result contract while using each platform's retry, alerting, artifact, metadata, and scheduling capabilities.

This makes it easier to place consistent data quality gates in repeated workflows such as ETL/ELT, analytics reporting, AI/ML pre-training validation, and operational data synchronization.

---

## Key Features

- **Official Truthound 3.x integration**: Built around the `truthound>=3.0,<4.0` result contract.
- **Platform-native execution**: Follows the idiomatic execution model of Airflow, Dagster, Prefect, dbt, Mage, and Kestra.
- **Single result semantics**: Preserves Truthound validation meaning across platforms.
- **Protocol-based architecture**: Exposes extension points for alternative and custom engines.
- **Quality workflow automation**: Connects validation, profiling, schema learning, drift detection, and anomaly detection to pipeline steps.
- **Operational utilities**: Provides serialization, logging, retry, circuit breaker, health check, and metrics helpers.

> Truthound Orchestration can also be used in AI/ML pipelines for pre-training input validation, feature table quality gates, and drift detection. Its core purpose is not to be an AI product, but to connect Truthound data quality workflows reliably to orchestration environments.

---

## Supported Platforms

| Platform | Package/module | Main role |
| --- | --- | --- |
| Apache Airflow | `truthound_airflow` | Data quality checks through Operators, Sensors, and Hooks |
| Dagster | `truthound_dagster` | Validation workflows through Resources, Assets, and Ops |
| Prefect | `truthound_prefect` | Quality pipelines through Blocks, Tasks, and Flows |
| dbt | `packages/dbt` | SQL-based validation through Generic Tests and Jinja macros |
| Mage AI | `packages/mage` | Transformers, Sensors, and Condition blocks |
| Kestra | `packages/kestra` | Python scripts, YAML flow templates, and output handlers |

---

## Quick Start

### Installation

```bash
# Core package + Truthound 3.x
pip install truthound-orchestration "truthound>=3.0,<4.0"
```

```bash
# Platform integrations + Truthound 3.x
pip install truthound-orchestration[airflow] "truthound>=3.0,<4.0"
pip install truthound-orchestration[dagster] "truthound>=3.0,<4.0"
pip install truthound-orchestration[prefect] "truthound>=3.0,<4.0"
```

```bash
# Convenience aggregate for local experiments or nightly canaries
pip install truthound-orchestration[all] "truthound>=3.0,<4.0"
```

### Python API

```python
from common.engines import TruthoundEngine
import polars as pl

engine = TruthoundEngine()
df = pl.read_csv("data.csv")

with engine:
    result = engine.check(df, auto_schema=True)
    print(f"Status: {result.status.name}")

    drift = engine.detect_drift(baseline_df, current_df, method="ks")
    print(f"Drifted: {drift.is_drifted}, Rate: {drift.drift_rate:.2%}")

    anomalies = engine.detect_anomalies(df, detector="isolation_forest")
    print(f"Anomalies: {anomalies.has_anomalies}, Rate: {anomalies.anomaly_rate:.2%}")
```

### Airflow Example

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from truthound_airflow import DataQualityCheckOperator

with DAG(
    dag_id="data_quality_pipeline",
    start_date=days_ago(1),
    schedule_interval="@daily",
) as dag:
    validate_data = DataQualityCheckOperator(
        task_id="validate_user_data",
        rules=[
            {"column": "user_id", "type": "not_null"},
            {"column": "user_id", "type": "unique"},
        ],
        data_path="s3://data-lake/users/{{ ds }}/data.parquet",
        fail_on_error=True,
    )
```

---

## Truthound 3.x Compatibility

`truthound-orchestration` `3.x` supports `Truthound 3.x` only.

- Supported Truthound versions: `>=3.0,<4.0`
- Unsupported Truthound versions: `1.x`, `2.x`
- This policy applies to the root package and its official platform extras.
- If you need an older Truthound engine line, use an older `truthound-orchestration` release line.

---

## Architecture

Truthound Orchestration is composed of a common Protocol layer and platform-specific adapter layers. Truthound is the default validation runtime, while advanced users can connect custom engines by implementing the Protocols.

```text
Workflow Platforms
Airflow / Dagster / Prefect / dbt / Mage / Kestra
        |
        v
Truthound Orchestration Common Layer
Protocols / Config / Serializers / Logging / Retry / Metrics
        |
        v
Truthound Engine
Validation / Profiling / Learn / Drift / Anomaly / Streaming
```

---

## Advanced Engine Support

Alternative and custom engines remain available for advanced use cases, but Truthound 3.x is the default compatibility path for the `3.x` release line.

```bash
# Great Expectations adapter
pip install truthound-orchestration[dagster] great-expectations

# Pandera adapter
pip install truthound-orchestration[airflow,prefect] pandera
```

Supported advanced engine options:

- [Great Expectations](https://greatexpectations.io/) adapter
- [Pandera](https://pandera.readthedocs.io/) adapter
- custom engines implementing the `DataQualityEngine` Protocol

---

## Documentation

- Main docs portal: [truthound.netlify.app/orchestration](https://truthound.netlify.app/orchestration/)
- Architecture: [docs/architecture.md](docs/architecture.md)
- Choose a platform: [docs/choose-a-platform.md](docs/choose-a-platform.md)
- Airflow docs: [docs/airflow/index.md](docs/airflow/index.md)
- Dagster docs: [docs/dagster/index.md](docs/dagster/index.md)
- Prefect docs: [docs/prefect/index.md](docs/prefect/index.md)
- dbt docs: [docs/dbt/index.md](docs/dbt/index.md)
- Mage docs: [docs/mage/index.md](docs/mage/index.md)
- Kestra docs: [docs/kestra/index.md](docs/kestra/index.md)
- Common module docs: [docs/common/index.md](docs/common/index.md)
- Release guide: [docs/releasing.md](docs/releasing.md)

---

## Development

```bash
git clone https://github.com/seadonggyun4/truthound-orchestration.git
cd truthound-orchestration

uv venv
source .venv/bin/activate
uv sync --all-extras
```

```bash
# lint
ruff check .

# type check
mypy common/

# tests
PYTHONPATH=. pytest --import-mode=importlib
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
