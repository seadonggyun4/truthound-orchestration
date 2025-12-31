"""File-based secret provider with encryption.

This module provides a secret provider that stores secrets in
encrypted files on the filesystem.

Example:
    >>> from packages.enterprise.secrets.backends import FileSecretProvider
    >>>
    >>> provider = FileSecretProvider(
    ...     directory="/etc/secrets",
    ...     encryption_key="...",
    ... )
    >>> provider.set("db/password", "secret123")
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..base import (
    HealthCheckable,
    HealthCheckResult,
    HealthStatus,
    SecretMetadata,
    SecretProvider,
    SecretType,
    SecretValue,
)
from ..exceptions import SecretBackendError, SecretConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class FileSecretProvider(HealthCheckable):
    """File-based secret provider with optional encryption.

    Stores each secret as an encrypted JSON file in a directory.
    Supports versioning through file naming.

    Attributes:
        directory: Storage directory.

    Example:
        >>> provider = FileSecretProvider(
        ...     directory=".secrets",
        ...     encryption_key_path="/etc/secret-key",
        ... )
        >>> provider.set("database/password", "secret")
    """

    SECRET_EXTENSION = ".secret"
    VERSION_SEPARATOR = "@"
    CURRENT_LINK = ".current"

    def __init__(
        self,
        directory: str = ".secrets",
        encryption_key: str | bytes | None = None,
        encryption_key_path: str | None = None,
        file_permissions: int = 0o600,
        dir_permissions: int = 0o700,
        create_directory: bool = True,
    ) -> None:
        """Initialize the provider.

        Args:
            directory: Directory to store secrets.
            encryption_key: Encryption key.
            encryption_key_path: Path to encryption key file.
            file_permissions: Unix file permissions.
            dir_permissions: Unix directory permissions.
            create_directory: Create directory if missing.
        """
        self._directory = Path(directory)
        self._file_permissions = file_permissions
        self._dir_permissions = dir_permissions
        self._lock = threading.RLock()
        self._encryptor = None

        # Load encryption key
        if encryption_key_path:
            encryption_key = Path(encryption_key_path).read_bytes().strip()
        if encryption_key:
            self._setup_encryption(encryption_key)

        # Create directory
        if create_directory:
            self._ensure_directory()

    def _setup_encryption(self, key: str | bytes) -> None:
        """Set up encryption with provided key.

        Args:
            key: Encryption key.
        """
        try:
            from ..encryption import FernetEncryptor

            if isinstance(key, bytes):
                key = key.decode("utf-8")
            self._encryptor = FernetEncryptor(key)
        except ImportError:
            raise SecretConfigurationError(
                "cryptography package required for file encryption"
            )

    def _ensure_directory(self) -> None:
        """Create the storage directory if it doesn't exist."""
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            os.chmod(self._directory, self._dir_permissions)
        except OSError as e:
            raise SecretBackendError(
                f"Failed to create directory: {e}",
                backend="file",
            )

    def _path_to_filename(self, path: str, version: str | None = None) -> Path:
        """Convert secret path to filename.

        Args:
            path: Secret path.
            version: Optional version.

        Returns:
            File path.
        """
        # Sanitize path - replace / with __
        safe_path = path.replace("/", "__").replace("\\", "__")

        if version:
            filename = f"{safe_path}{self.VERSION_SEPARATOR}{version}{self.SECRET_EXTENSION}"
        else:
            filename = f"{safe_path}{self.SECRET_EXTENSION}"

        return self._directory / filename

    def _filename_to_path(self, filename: str) -> tuple[str, str | None]:
        """Convert filename to secret path.

        Args:
            filename: File name.

        Returns:
            Tuple of (path, version).
        """
        if not filename.endswith(self.SECRET_EXTENSION):
            return "", None

        name = filename[: -len(self.SECRET_EXTENSION)]

        if self.VERSION_SEPARATOR in name:
            path, version = name.rsplit(self.VERSION_SEPARATOR, 1)
        else:
            path, version = name, None

        # Restore path separators
        path = path.replace("__", "/")

        return path, version

    def _read_secret_file(self, file_path: Path) -> dict[str, Any] | None:
        """Read and decrypt a secret file.

        Args:
            file_path: Path to the file.

        Returns:
            Parsed secret data or None.
        """
        if not file_path.exists():
            return None

        try:
            data = file_path.read_bytes()

            if self._encryptor:
                data = self._encryptor.decrypt(data)

            return json.loads(data.decode("utf-8"))
        except Exception:
            return None

    def _write_secret_file(self, file_path: Path, data: dict[str, Any]) -> None:
        """Encrypt and write a secret file.

        Args:
            file_path: Path to the file.
            data: Data to write.
        """
        json_data = json.dumps(data).encode("utf-8")

        if self._encryptor:
            json_data = self._encryptor.encrypt(json_data)

        # Write atomically
        temp_path = file_path.with_suffix(".tmp")
        try:
            temp_path.write_bytes(json_data)
            os.chmod(temp_path, self._file_permissions)
            temp_path.rename(file_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def get(
        self,
        path: str,
        *,
        version: str | None = None,
    ) -> SecretValue | None:
        """Get a secret from file.

        Args:
            path: Secret path.
            version: Optional specific version.

        Returns:
            Secret value or None if not found.
        """
        with self._lock:
            if version:
                file_path = self._path_to_filename(path, version)
            else:
                # Find current version
                file_path = self._find_current_version(path)
                if file_path is None:
                    return None

            data = self._read_secret_file(file_path)
            if data is None:
                return None

            return SecretValue(
                value=data["value"],
                version=data.get("version", "1"),
                created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
                expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                secret_type=SecretType[data.get("secret_type", "STRING")],
                metadata=data.get("metadata", {}),
            )

    def _find_current_version(self, path: str) -> Path | None:
        """Find the file for the current version of a secret.

        Args:
            path: Secret path.

        Returns:
            File path or None.
        """
        pattern = self._path_to_filename(path).name.replace(
            self.SECRET_EXTENSION, f"{self.VERSION_SEPARATOR}*{self.SECRET_EXTENSION}"
        )

        # Also check unversioned file
        unversioned = self._path_to_filename(path)
        if unversioned.exists():
            return unversioned

        matching = list(self._directory.glob(pattern))
        if not matching:
            return None

        # Return highest version
        def version_key(p: Path) -> int:
            _, ver = self._filename_to_path(p.name)
            return int(ver) if ver and ver.isdigit() else 0

        return max(matching, key=version_key)

    def set(
        self,
        path: str,
        value: str | bytes,
        *,
        secret_type: SecretType = SecretType.STRING,
        expires_at: datetime | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SecretValue:
        """Store a secret to file.

        Args:
            path: Secret path.
            value: Secret value.
            secret_type: Type of secret.
            expires_at: Expiration time.
            metadata: Additional metadata.

        Returns:
            The stored secret.
        """
        with self._lock:
            self._ensure_directory()

            # Find current version to determine next
            current = self.get(path)
            if current:
                new_version = str(int(current.version) + 1)
            else:
                new_version = "1"

            now = datetime.now(timezone.utc)

            if isinstance(value, bytes):
                value = value.decode("utf-8")

            data = {
                "value": value,
                "version": new_version,
                "created_at": now.isoformat(),
                "secret_type": secret_type.name,
                "metadata": dict(metadata) if metadata else {},
            }
            if expires_at:
                data["expires_at"] = expires_at.isoformat()

            file_path = self._path_to_filename(path, new_version)
            self._write_secret_file(file_path, data)

            return SecretValue(
                value=value,
                version=new_version,
                created_at=now,
                expires_at=expires_at,
                secret_type=secret_type,
                metadata=metadata or {},
            )

    def delete(self, path: str) -> bool:
        """Delete all versions of a secret.

        Args:
            path: Secret path.

        Returns:
            True if any files were deleted.
        """
        with self._lock:
            safe_path = path.replace("/", "__").replace("\\", "__")
            pattern = f"{safe_path}*{self.SECRET_EXTENSION}"

            deleted = False
            for file_path in self._directory.glob(pattern):
                file_path.unlink()
                deleted = True

            return deleted

    def exists(self, path: str) -> bool:
        """Check if a secret exists.

        Args:
            path: Secret path.

        Returns:
            True if exists.
        """
        with self._lock:
            return self._find_current_version(path) is not None

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
        with self._lock:
            secrets: dict[str, SecretMetadata] = {}

            for file_path in self._directory.glob(f"*{self.SECRET_EXTENSION}"):
                path, version = self._filename_to_path(file_path.name)

                if not path or not path.startswith(prefix):
                    continue

                # Only keep highest version per path
                if path not in secrets:
                    data = self._read_secret_file(file_path)
                    if data:
                        secrets[path] = SecretMetadata(
                            path=path,
                            version=data.get("version", "1"),
                            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
                            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                            secret_type=SecretType[data.get("secret_type", "STRING")],
                        )
                else:
                    # Compare versions
                    data = self._read_secret_file(file_path)
                    if data:
                        new_version = data.get("version", "1")
                        if int(new_version) > int(secrets[path].version):
                            secrets[path] = SecretMetadata(
                                path=path,
                                version=new_version,
                                created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else None,
                                expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                                secret_type=SecretType[data.get("secret_type", "STRING")],
                            )

            results = sorted(secrets.values(), key=lambda m: m.path)

            if offset:
                results = results[offset:]
            if limit:
                results = results[:limit]

            return results

    def health_check(self) -> HealthCheckResult:
        """Check provider health.

        Returns:
            Health check result.
        """
        start = time.perf_counter()
        try:
            # Check directory access
            if not self._directory.exists():
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Directory does not exist: {self._directory}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            if not os.access(self._directory, os.R_OK | os.W_OK):
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    message="Limited directory access",
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            # Count secrets
            count = len(list(self._directory.glob(f"*{self.SECRET_EXTENSION}")))

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="File provider is healthy",
                details={
                    "directory": str(self._directory),
                    "secret_files": count,
                    "encrypted": self._encryptor is not None,
                },
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
