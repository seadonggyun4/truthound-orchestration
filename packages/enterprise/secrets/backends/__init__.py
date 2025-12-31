"""Secret storage backends.

This package provides various backend implementations for secret storage.

Available backends:
    - InMemorySecretProvider: In-memory storage for testing.
    - EnvironmentSecretProvider: Environment variables.
    - FileSecretProvider: Encrypted file storage.
    - VaultSecretProvider: HashiCorp Vault.
    - AWSSecretsManagerProvider: AWS Secrets Manager.
    - GCPSecretManagerProvider: Google Cloud Secret Manager.
    - AzureKeyVaultProvider: Azure Key Vault.

Example:
    >>> from packages.enterprise.secrets.backends import InMemorySecretProvider
    >>> provider = InMemorySecretProvider()
    >>> provider.set("secret", "value")
"""

from .memory import InMemorySecretProvider

__all__ = [
    "InMemorySecretProvider",
    "EnvironmentSecretProvider",
    "FileSecretProvider",
    "VaultSecretProvider",
    "AWSSecretsManagerProvider",
    "AWSParameterStoreProvider",
    "GCPSecretManagerProvider",
    "AzureKeyVaultProvider",
]

# Lazy imports for optional backends
def __getattr__(name: str):
    if name == "EnvironmentSecretProvider":
        from .env import EnvironmentSecretProvider
        return EnvironmentSecretProvider
    elif name == "FileSecretProvider":
        from .file import FileSecretProvider
        return FileSecretProvider
    elif name == "VaultSecretProvider":
        from .vault import VaultSecretProvider
        return VaultSecretProvider
    elif name == "AWSSecretsManagerProvider":
        from .aws import AWSSecretsManagerProvider
        return AWSSecretsManagerProvider
    elif name == "AWSParameterStoreProvider":
        from .aws import AWSParameterStoreProvider
        return AWSParameterStoreProvider
    elif name == "GCPSecretManagerProvider":
        from .gcp import GCPSecretManagerProvider
        return GCPSecretManagerProvider
    elif name == "AzureKeyVaultProvider":
        from .azure import AzureKeyVaultProvider
        return AzureKeyVaultProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
