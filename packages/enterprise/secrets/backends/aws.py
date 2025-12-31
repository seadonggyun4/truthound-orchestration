"""AWS Secrets Manager and Parameter Store providers.

This module provides secret providers that integrate with AWS Secrets Manager
and AWS Systems Manager Parameter Store.

Requires: pip install boto3

Example:
    >>> from packages.enterprise.secrets.backends import AWSSecretsManagerProvider
    >>> from packages.enterprise.secrets import AWSSecretsManagerConfig
    >>>
    >>> config = AWSSecretsManagerConfig(
    ...     region_name="us-east-1",
    ...     kms_key_id="alias/my-key",
    ... )
    >>> provider = AWSSecretsManagerProvider(config)
    >>> secret = provider.get("database/password")
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..base import (
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
    SecretMetadata,
    SecretType,
    SecretValue,
)
from ..exceptions import (
    SecretAccessDeniedError,
    SecretAuthenticationError,
    SecretBackendError,
    SecretConnectionError,
    SecretNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..config import AWSSecretsManagerConfig


class AWSSecretsManagerProvider(HealthCheckable):
    """AWS Secrets Manager secret provider.

    Supports JSON and plain text secrets, versioning, and KMS encryption.

    Example:
        >>> provider = AWSSecretsManagerProvider(config)
        >>> provider.set("myapp/db", '{"user": "admin", "pass": "secret"}')
        >>> secret = provider.get("myapp/db")
    """

    def __init__(self, config: AWSSecretsManagerConfig) -> None:
        """Initialize the provider.

        Args:
            config: AWS Secrets Manager configuration.

        Raises:
            ImportError: If boto3 is not installed.
        """
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as e:
            raise ImportError(
                "boto3 package required for AWSSecretsManagerProvider. "
                "Install with: pip install boto3"
            ) from e

        self._config = config
        self._boto3 = boto3
        self._BotoCoreError = BotoCoreError
        self._ClientError = ClientError
        self._client = None

    def _get_client(self):
        """Get or create the Secrets Manager client.

        Returns:
            Secrets Manager client.
        """
        if self._client is not None:
            return self._client

        session_kwargs: dict[str, Any] = {}
        if self._config.region_name:
            session_kwargs["region_name"] = self._config.region_name
        if self._config.profile_name:
            session_kwargs["profile_name"] = self._config.profile_name

        session = self._boto3.Session(**session_kwargs)

        client_kwargs: dict[str, Any] = {}
        if self._config.endpoint_url:
            client_kwargs["endpoint_url"] = self._config.endpoint_url

        self._client = session.client("secretsmanager", **client_kwargs)
        return self._client

    def _secret_name(self, path: str) -> str:
        """Convert path to secret name with optional prefix.

        Args:
            path: Secret path.

        Returns:
            Full secret name.
        """
        if self._config.prefix:
            return f"{self._config.prefix}{path}"
        return path

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a secret from AWS Secrets Manager.

        Args:
            path: Secret path (name).
            version: Optional version ID or stage.

        Returns:
            Secret value or None.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            kwargs: dict[str, Any] = {"SecretId": secret_name}
            if version:
                # Check if it's a stage name or version ID
                if version in ("AWSCURRENT", "AWSPREVIOUS"):
                    kwargs["VersionStage"] = version
                else:
                    kwargs["VersionId"] = version

            response = client.get_secret_value(**kwargs)

            # Determine if it's a string or binary secret
            if "SecretString" in response:
                value = response["SecretString"]
                secret_type = SecretType.STRING

                # Try to parse as JSON
                try:
                    json.loads(value)
                    secret_type = SecretType.JSON
                except json.JSONDecodeError:
                    pass
            else:
                value = response["SecretBinary"]
                secret_type = SecretType.BINARY

            return SecretValue(
                value=value,
                version=response.get("VersionId", "unknown"),
                created_at=response.get("CreatedDate", datetime.now(timezone.utc)),
                expires_at=None,
                secret_type=secret_type,
                metadata={
                    "arn": response.get("ARN"),
                    "name": response.get("Name"),
                    "version_stages": response.get("VersionStages", []),
                },
            )

        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                return None
            if error_code == "AccessDeniedException":
                raise SecretAccessDeniedError(
                    f"Access denied to secret: {path}",
                    path=path,
                )
            if error_code in ("UnrecognizedClientException", "InvalidSignatureException"):
                raise SecretAuthenticationError(
                    f"Authentication failed: {e}",
                    backend="aws_secrets_manager",
                    auth_method="iam",
                )
            raise SecretBackendError(
                f"AWS Secrets Manager error: {e}",
                backend="aws_secrets_manager",
                path=path,
            )
        except self._BotoCoreError as e:
            raise SecretConnectionError(
                f"Failed to connect to AWS: {e}",
                backend="aws_secrets_manager",
                endpoint=self._config.endpoint_url or "aws",
            )

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Store a secret in AWS Secrets Manager.

        Args:
            path: Secret path (name).
            value: Secret value.
            secret_type: Type of secret.
            expires_at: Ignored (AWS handles rotation separately).
            metadata: Additional metadata (stored as description/tags).

        Returns:
            The stored secret.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            # Check if secret exists
            try:
                client.describe_secret(SecretId=secret_name)
                exists = True
            except self._ClientError as e:
                if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                    exists = False
                else:
                    raise

            kwargs: dict[str, Any] = {"SecretId": secret_name}

            if isinstance(value, bytes):
                kwargs["SecretBinary"] = value
            else:
                kwargs["SecretString"] = value

            if self._config.kms_key_id:
                kwargs["KmsKeyId"] = self._config.kms_key_id

            if exists:
                # Update existing secret
                response = client.put_secret_value(**kwargs)
            else:
                # Create new secret
                create_kwargs: dict[str, Any] = {
                    "Name": secret_name,
                }
                if isinstance(value, bytes):
                    create_kwargs["SecretBinary"] = value
                else:
                    create_kwargs["SecretString"] = value

                if self._config.kms_key_id:
                    create_kwargs["KmsKeyId"] = self._config.kms_key_id

                if metadata and "description" in metadata:
                    create_kwargs["Description"] = metadata["description"]

                if metadata and "tags" in metadata:
                    create_kwargs["Tags"] = [
                        {"Key": k, "Value": v}
                        for k, v in metadata["tags"].items()
                    ]

                response = client.create_secret(**create_kwargs)

            return SecretValue(
                value=value if isinstance(value, str) else value.decode("utf-8"),
                version=response.get("VersionId", "unknown"),
                created_at=datetime.now(timezone.utc),
                secret_type=secret_type,
                metadata={"arn": response.get("ARN")},
            )

        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                raise SecretAccessDeniedError(
                    f"Access denied to set secret: {path}",
                    path=path,
                )
            raise SecretBackendError(
                f"Failed to store secret: {e}",
                backend="aws_secrets_manager",
                path=path,
            )
        except self._BotoCoreError as e:
            raise SecretConnectionError(
                f"Failed to connect to AWS: {e}",
                backend="aws_secrets_manager",
            )

    def delete(self, path: str) -> bool:
        """Delete a secret from AWS Secrets Manager.

        Args:
            path: Secret path.

        Returns:
            True if deleted.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)

            # Use force delete without recovery
            client.delete_secret(
                SecretId=secret_name,
                ForceDeleteWithoutRecovery=True,
            )
            return True

        except self._ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return False
            raise SecretBackendError(
                f"Failed to delete secret: {e}",
                backend="aws_secrets_manager",
                path=path,
            )

    def exists(self, path: str) -> bool:
        """Check if a secret exists.

        Args:
            path: Secret path.

        Returns:
            True if exists.
        """
        try:
            client = self._get_client()
            secret_name = self._secret_name(path)
            client.describe_secret(SecretId=secret_name)
            return True
        except self._ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return False
            raise

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List secrets matching a prefix.

        Args:
            prefix: Path prefix.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of secret metadata.
        """
        try:
            client = self._get_client()
            full_prefix = self._secret_name(prefix)

            results: list[SecretMetadata] = []
            paginator = client.get_paginator("list_secrets")

            filters = []
            if full_prefix:
                filters.append({"Key": "name", "Values": [full_prefix]})

            page_iterator = paginator.paginate(
                Filters=filters if filters else [],
                SortOrder="asc",
            )

            for page in page_iterator:
                for secret in page.get("SecretList", []):
                    # Strip prefix from name for path
                    name = secret["Name"]
                    if self._config.prefix and name.startswith(self._config.prefix):
                        path = name[len(self._config.prefix):]
                    else:
                        path = name

                    results.append(
                        SecretMetadata(
                            path=path,
                            version="AWSCURRENT",
                            created_at=secret.get("CreatedDate"),
                            secret_type=SecretType.STRING,
                            metadata={
                                "arn": secret.get("ARN"),
                                "description": secret.get("Description"),
                                "last_accessed": secret.get("LastAccessedDate"),
                                "last_changed": secret.get("LastChangedDate"),
                            },
                        )
                    )

            # Apply offset and limit
            if offset:
                results = results[offset:]
            if limit:
                results = results[:limit]

            return results

        except self._ClientError as e:
            raise SecretBackendError(
                f"Failed to list secrets: {e}",
                backend="aws_secrets_manager",
            )

    def health_check(self) -> HealthCheckResult:
        """Check AWS Secrets Manager health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            client = self._get_client()
            # List secrets with minimal results as a health check
            client.list_secrets(MaxResults=1)
            duration = (time.perf_counter() - start) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="AWS Secrets Manager is healthy",
                details={
                    "region": self._config.region_name,
                    "endpoint": self._config.endpoint_url,
                },
                latency_ms=duration,
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )


class AWSParameterStoreProvider(HealthCheckable):
    """AWS Systems Manager Parameter Store secret provider.

    Supports SecureString, String, and StringList parameters.

    Example:
        >>> provider = AWSParameterStoreProvider(config)
        >>> provider.set("/myapp/db/password", "secret", secure=True)
        >>> secret = provider.get("/myapp/db/password")
    """

    def __init__(self, config: AWSSecretsManagerConfig) -> None:
        """Initialize the provider.

        Args:
            config: AWS configuration (reuses same config type).

        Raises:
            ImportError: If boto3 is not installed.
        """
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError as e:
            raise ImportError(
                "boto3 package required for AWSParameterStoreProvider. "
                "Install with: pip install boto3"
            ) from e

        self._config = config
        self._boto3 = boto3
        self._BotoCoreError = BotoCoreError
        self._ClientError = ClientError
        self._client = None

    def _get_client(self):
        """Get or create the SSM client.

        Returns:
            SSM client.
        """
        if self._client is not None:
            return self._client

        session_kwargs: dict[str, Any] = {}
        if self._config.region_name:
            session_kwargs["region_name"] = self._config.region_name
        if self._config.profile_name:
            session_kwargs["profile_name"] = self._config.profile_name

        session = self._boto3.Session(**session_kwargs)

        client_kwargs: dict[str, Any] = {}
        if self._config.endpoint_url:
            client_kwargs["endpoint_url"] = self._config.endpoint_url

        self._client = session.client("ssm", **client_kwargs)
        return self._client

    def _param_name(self, path: str) -> str:
        """Convert path to parameter name.

        Args:
            path: Secret path.

        Returns:
            Full parameter name.
        """
        # Ensure path starts with /
        if not path.startswith("/"):
            path = f"/{path}"

        if self._config.prefix:
            prefix = self._config.prefix
            if not prefix.startswith("/"):
                prefix = f"/{prefix}"
            if not prefix.endswith("/"):
                prefix = f"{prefix}"
            return f"{prefix}{path}"
        return path

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a parameter from Parameter Store.

        Args:
            path: Parameter path.
            version: Optional version number.

        Returns:
            Secret value or None.
        """
        try:
            client = self._get_client()
            param_name = self._param_name(path)

            if version:
                param_name = f"{param_name}:{version}"

            response = client.get_parameter(
                Name=param_name,
                WithDecryption=True,
            )

            param = response["Parameter"]
            value = param["Value"]

            # Determine type based on parameter type
            param_type = param.get("Type", "String")
            if param_type == "SecureString":
                secret_type = SecretType.STRING
            elif param_type == "StringList":
                secret_type = SecretType.JSON
            else:
                secret_type = SecretType.STRING

            return SecretValue(
                value=value,
                version=str(param.get("Version", 1)),
                created_at=param.get("LastModifiedDate", datetime.now(timezone.utc)),
                secret_type=secret_type,
                metadata={
                    "arn": param.get("ARN"),
                    "type": param_type,
                    "data_type": param.get("DataType"),
                },
            )

        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ParameterNotFound":
                return None
            if error_code == "AccessDeniedException":
                raise SecretAccessDeniedError(
                    f"Access denied to parameter: {path}",
                    path=path,
                )
            raise SecretBackendError(
                f"Parameter Store error: {e}",
                backend="aws_parameter_store",
                path=path,
            )
        except self._BotoCoreError as e:
            raise SecretConnectionError(
                f"Failed to connect to AWS: {e}",
                backend="aws_parameter_store",
            )

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
        secure: bool = True,
    ) -> SecretValue:
        """Store a parameter in Parameter Store.

        Args:
            path: Parameter path.
            value: Parameter value.
            secret_type: Type of secret.
            expires_at: Ignored.
            metadata: Additional metadata.
            secure: Use SecureString type (default True).

        Returns:
            The stored secret.
        """
        try:
            client = self._get_client()
            param_name = self._param_name(path)

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            kwargs: dict[str, Any] = {
                "Name": param_name,
                "Value": value,
                "Type": "SecureString" if secure else "String",
                "Overwrite": True,
            }

            if secure and self._config.kms_key_id:
                kwargs["KeyId"] = self._config.kms_key_id

            if metadata and "description" in metadata:
                kwargs["Description"] = metadata["description"]

            if metadata and "tags" in metadata:
                kwargs["Tags"] = [
                    {"Key": k, "Value": v}
                    for k, v in metadata["tags"].items()
                ]

            response = client.put_parameter(**kwargs)

            return SecretValue(
                value=value,
                version=str(response.get("Version", 1)),
                created_at=datetime.now(timezone.utc),
                secret_type=secret_type,
                metadata={"tier": response.get("Tier")},
            )

        except self._ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "AccessDeniedException":
                raise SecretAccessDeniedError(
                    f"Access denied to set parameter: {path}",
                    path=path,
                )
            raise SecretBackendError(
                f"Failed to store parameter: {e}",
                backend="aws_parameter_store",
                path=path,
            )

    def delete(self, path: str) -> bool:
        """Delete a parameter from Parameter Store.

        Args:
            path: Parameter path.

        Returns:
            True if deleted.
        """
        try:
            client = self._get_client()
            param_name = self._param_name(path)
            client.delete_parameter(Name=param_name)
            return True

        except self._ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ParameterNotFound":
                return False
            raise SecretBackendError(
                f"Failed to delete parameter: {e}",
                backend="aws_parameter_store",
                path=path,
            )

    def exists(self, path: str) -> bool:
        """Check if a parameter exists.

        Args:
            path: Parameter path.

        Returns:
            True if exists.
        """
        return self.get(path) is not None

    def list(
        self,
        prefix: str = "",
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[SecretMetadata]:
        """List parameters matching a path prefix.

        Args:
            prefix: Path prefix.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of secret metadata.
        """
        try:
            client = self._get_client()
            full_prefix = self._param_name(prefix) if prefix else self._config.prefix or "/"

            results: list[SecretMetadata] = []
            paginator = client.get_paginator("get_parameters_by_path")

            page_iterator = paginator.paginate(
                Path=full_prefix,
                Recursive=True,
                WithDecryption=False,  # Don't decrypt for listing
            )

            for page in page_iterator:
                for param in page.get("Parameters", []):
                    name = param["Name"]
                    # Strip prefix
                    if self._config.prefix:
                        prefix_to_strip = self._param_name("")
                        if name.startswith(prefix_to_strip):
                            path = name[len(prefix_to_strip):]
                        else:
                            path = name
                    else:
                        path = name

                    results.append(
                        SecretMetadata(
                            path=path.lstrip("/"),
                            version=str(param.get("Version", 1)),
                            created_at=param.get("LastModifiedDate"),
                            secret_type=SecretType.STRING,
                            metadata={
                                "arn": param.get("ARN"),
                                "type": param.get("Type"),
                            },
                        )
                    )

            # Apply offset and limit
            if offset:
                results = results[offset:]
            if limit:
                results = results[:limit]

            return results

        except self._ClientError as e:
            raise SecretBackendError(
                f"Failed to list parameters: {e}",
                backend="aws_parameter_store",
            )

    def health_check(self) -> HealthCheckResult:
        """Check Parameter Store health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            client = self._get_client()
            # Describe parameters as a health check
            client.describe_parameters(MaxResults=1)
            duration = (time.perf_counter() - start) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="AWS Parameter Store is healthy",
                details={
                    "region": self._config.region_name,
                    "endpoint": self._config.endpoint_url,
                },
                latency_ms=duration,
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
