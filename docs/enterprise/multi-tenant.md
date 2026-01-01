---
title: Multi-Tenant
---

# Multi-Tenant Support

Multi-tenant support for tenant-isolated data quality validation.

## TenantContext

Tenant context management:

```python
from packages.enterprise.multi_tenant import TenantContext

# Use as context manager
with TenantContext(tenant_id="tenant_1"):
    result = engine.check(data, auto_schema=True)
    # All operations in this block are isolated to tenant_1

# Nested contexts
with TenantContext(tenant_id="tenant_1"):
    with TenantContext(tenant_id="tenant_2"):
        # Inner context takes precedence
        pass
```

## TenantRegistry

Tenant registration and management:

```python
from packages.enterprise.multi_tenant import TenantRegistry

registry = TenantRegistry()

# Register tenant
registry.register(
    tenant_id="tenant_1",
    name="Tenant One",
    config={"max_workers": 4},
)

# Get tenant
tenant = registry.get("tenant_1")
print(f"Name: {tenant.name}")

# List tenants
for tenant_id in registry.list_all():
    print(tenant_id)
```

## Isolation Strategies

### Shared

Shared resources, logical isolation:

```python
from packages.enterprise.multi_tenant.isolation import SharedIsolation

isolation = SharedIsolation()
# All tenants share the same engine instance
```

### Logical

Logical partitioning:

```python
from packages.enterprise.multi_tenant.isolation import LogicalIsolation

isolation = LogicalIsolation()
# Tenant separation at data level
```

### Physical

Physical separation:

```python
from packages.enterprise.multi_tenant.isolation import PhysicalIsolation

isolation = PhysicalIsolation()
# Separate resource instances per tenant
```

## Storage Backends

### Memory

Memory storage for development/testing:

```python
from packages.enterprise.multi_tenant.storage import MemoryStorage

storage = MemoryStorage()
```

### File

File-based storage:

```python
from packages.enterprise.multi_tenant.storage import FileStorage

storage = FileStorage(base_path="/data/tenants")
```

## HTTP Middleware

Automatic tenant context setup in web applications:

```python
from packages.enterprise.multi_tenant import TenantMiddleware

# FastAPI example
from fastapi import FastAPI

app = FastAPI()
app.add_middleware(TenantMiddleware)
```

## Tenant Hooks

Tenant lifecycle events:

```python
from packages.enterprise.multi_tenant import TenantHook

class MyTenantHook(TenantHook):
    def on_enter(self, tenant_id):
        print(f"Entering tenant: {tenant_id}")

    def on_exit(self, tenant_id):
        print(f"Exiting tenant: {tenant_id}")
```

## Quota Management

Per-tenant resource limits:

```python
from packages.enterprise.multi_tenant import TenantQuota

quota = TenantQuota(
    max_requests_per_hour=1000,
    max_data_size_mb=100,
    max_workers=4,
)

registry.register(
    tenant_id="tenant_1",
    name="Tenant One",
    quota=quota,
)
```

## Async Support

```python
from packages.enterprise.multi_tenant import AsyncTenantContext

async with AsyncTenantContext(tenant_id="tenant_1"):
    result = await async_engine.check(data)
```

## Thread Safety

TenantContext is thread-safe:

```python
from concurrent.futures import ThreadPoolExecutor
from packages.enterprise.multi_tenant import TenantContext

def process_tenant(tenant_id, data):
    with TenantContext(tenant_id=tenant_id):
        return engine.check(data, auto_schema=True)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(process_tenant, f"tenant_{i}", data)
        for i in range(10)
    ]
    results = [f.result() for f in futures]
```
