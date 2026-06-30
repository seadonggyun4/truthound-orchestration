!!! note "Truthound Orchestration 한국어 문서"
    이 페이지는 Truthound 문서의 한국어 미러입니다. 코드, 명령어, API 이름은 정확성을 위해 원문 표기를 유지하고, 설명은 데이터 품질 워크플로우 관점으로 제공합니다.

---
title: Kestra Namespaces and Secrets
---

# Kestra Namespaces and Secrets

Kestra namespaces are often the cleanest way to separate environments, teams, or tenants. Truthound's Kestra integration is designed to fit that model: namespace and secret concerns stay Kestra-native while 검증 behavior stays Truthound-native.

## Who This Is For

- platform engineers structuring Kestra namespaces
- teams separating dev, staging, and production 검증 flows
- operators deciding where secret values should come from

## When To Use It

Use this page when:

- generated flows should land in a specific namespace
- secrets and environment routing should align with namespace boundaries
- teams need a policy for multi-environment or multi-tenant deployment

## Prerequisites

- a Kestra namespace strategy
- access to the secret provider used by the Kestra installation
- familiarity with `FlowConfig(namespace=...)`

## Minimal Quickstart

Generate a flow into the intended namespace:

```python
from truthound_kestra import FlowConfig, generate_flow_yaml

config = FlowConfig(
    id="users_quality",
    namespace="production",
)
yaml_content = generate_flow_yaml(config)
```

Use Kestra variable accessors when runtime values need to be read safely:

```python
from truthound_kestra import get_kestra_variable

threshold = get_kestra_variable("MAX_FAILURES", 10)
```

## Production Pattern

Use this split of responsibility:

| Concern | Recommended Kestra Boundary |
|--------|------------------------------|
| environment and ownership routing | namespace |
| secret values | Kestra secret or variable management |
| 검증 rules | versioned flow or script configuration |
| output contract | Truthound output helpers |

## Failure Modes and Troubleshooting

| Symptom | Likely Cause | What To Do |
|--------|--------------|------------|
| production flow reads staging values | namespace or variable scoping is inconsistent | align namespace strategy and secret lookup policy |
| generated flows are hard to audit | namespace is implicit or defaulted too often | set it explicitly in `FlowConfig` |
| rules drift between teams | namespace-specific secrets also carry rule logic | keep rules in versioned configuration, not secret stores |

## Related Pages

- [Kestra Overview](index.md)
- [Scripts and Flow Templates](scripts-templates.md)
- [Enterprise Multi-Tenant](../enterprise/multi-tenant.md)
- [Enterprise Secrets](../enterprise/secrets.md)
