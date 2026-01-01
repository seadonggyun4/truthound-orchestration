---
title: Secret Management
---

# Secret Management

A secret management system supporting various backends.

## SecretProvider Protocol

Interface implemented by all backends:

```python
from typing import Protocol

class SecretProvider(Protocol):
    def get_secret(self, key: str) -> str: ...
    def set_secret(self, key: str, value: str) -> None: ...
    def delete_secret(self, key: str) -> None: ...
    def list_secrets(self, prefix: str = "") -> list[str]: ...
```

## SecretProviderRegistry

Secret provider registry:

```python
from packages.enterprise.secrets import SecretProviderRegistry

registry = SecretProviderRegistry()

# Get secret
secret = registry.get_secret("database/password")

# Set secret
registry.set_secret("api/key", "my-api-key")
```

## Supported Backends

### HashiCorp Vault

```python
from packages.enterprise.secrets.backends import VaultSecretProvider

provider = VaultSecretProvider(
    url="https://vault.example.com",
    token="my-vault-token",
    mount_point="secret",
)

secret = provider.get_secret("database/password")
```

### AWS Secrets Manager

```python
from packages.enterprise.secrets.backends import AWSSecretProvider

provider = AWSSecretProvider(
    region_name="us-east-1",
)

secret = provider.get_secret("prod/database/password")
```

### GCP Secret Manager

```python
from packages.enterprise.secrets.backends import GCPSecretProvider

provider = GCPSecretProvider(
    project_id="my-project",
)

secret = provider.get_secret("database-password")
```

### Azure Key Vault

```python
from packages.enterprise.secrets.backends import AzureSecretProvider

provider = AzureSecretProvider(
    vault_url="https://my-vault.vault.azure.net",
)

secret = provider.get_secret("database-password")
```

### Environment Variables

For development/testing:

```python
from packages.enterprise.secrets.backends import EnvSecretProvider

provider = EnvSecretProvider(prefix="APP_SECRET_")

# Reads from APP_SECRET_DATABASE_PASSWORD environment variable
secret = provider.get_secret("database_password")
```

### Encrypted File

```python
from packages.enterprise.secrets.backends import FileSecretProvider

provider = FileSecretProvider(
    file_path="/secrets/secrets.enc",
    encryption_key="my-encryption-key",
)

secret = provider.get_secret("database/password")
```

### Memory

For testing:

```python
from packages.enterprise.secrets.backends import MemorySecretProvider

provider = MemorySecretProvider()
provider.set_secret("test/key", "test-value")
```

## Caching

TTL-based caching:

```python
from packages.enterprise.secrets import SecretCache

cache = SecretCache(
    ttl_seconds=300,  # 5 minute TTL
)

# Get cached secret
secret = cache.get_or_fetch("database/password", provider.get_secret)
```

### Hierarchical Caching

```python
from packages.enterprise.secrets import HierarchicalSecretCache

cache = HierarchicalSecretCache(
    l1_ttl_seconds=60,   # L1 cache: 1 minute
    l2_ttl_seconds=300,  # L2 cache: 5 minutes
)
```

## Encryption

### Fernet

```python
from packages.enterprise.secrets.encryption import FernetEncryption

encryption = FernetEncryption(key="my-secret-key")
encrypted = encryption.encrypt("my-secret-value")
decrypted = encryption.decrypt(encrypted)
```

### AES-GCM

```python
from packages.enterprise.secrets.encryption import AESGCMEncryption

encryption = AESGCMEncryption(key="my-32-byte-secret-key-here!!")
encrypted = encryption.encrypt("my-secret-value")
```

### ChaCha20

```python
from packages.enterprise.secrets.encryption import ChaCha20Encryption

encryption = ChaCha20Encryption(key="my-32-byte-secret-key-here!!")
encrypted = encryption.encrypt("my-secret-value")
```

## Automatic Rotation

```python
from packages.enterprise.secrets import SecretRotation

rotation = SecretRotation(
    provider=provider,
    rotation_interval_days=30,
)

# Check if rotation is needed
if rotation.needs_rotation("database/password"):
    new_secret = generate_new_password()
    rotation.rotate("database/password", new_secret)
```

## Audit Logging

```python
from packages.enterprise.secrets import AuditLoggingHook

hook = AuditLoggingHook(
    log_reads=True,
    log_writes=True,
)

# Add hook to registry
registry.add_hook(hook)
```

## Multi-Tenant Isolation

```python
from packages.enterprise.secrets import TenantSecretProvider
from packages.enterprise.multi_tenant import TenantContext

provider = TenantSecretProvider(base_provider=vault_provider)

with TenantContext(tenant_id="tenant_1"):
    # Only tenant_1's secrets are accessible
    secret = provider.get_secret("database/password")
```

## Middleware

Caching, encryption, validation wrappers:

```python
from packages.enterprise.secrets.middleware import (
    CachingMiddleware,
    EncryptionMiddleware,
    ValidationMiddleware,
)

# Middleware chaining
provider = CachingMiddleware(
    EncryptionMiddleware(
        ValidationMiddleware(base_provider)
    ),
    ttl_seconds=300,
)
```
