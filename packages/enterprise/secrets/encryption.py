"""Client-side encryption utilities for secret management.

This module provides encryption and decryption utilities for
client-side secret encryption before storage.

Example:
    >>> from packages.enterprise.secrets import (
    ...     FernetEncryptor,
    ...     generate_encryption_key,
    ... )
    >>>
    >>> key = generate_encryption_key()
    >>> encryptor = FernetEncryptor(key)
    >>> encrypted = encryptor.encrypt(b"secret data")
    >>> decrypted = encryptor.decrypt(encrypted)
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .exceptions import SecretDecryptError, SecretEncryptError

if TYPE_CHECKING:
    pass


@runtime_checkable
class SecretEncryptor(Protocol):
    """Protocol for secret encryptors.

    Implementations provide encryption and decryption for secrets.
    """

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data.

        Args:
            data: Data to encrypt.

        Returns:
            Encrypted data.
        """
        ...

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data.

        Args:
            data: Data to decrypt.

        Returns:
            Decrypted data.
        """
        ...


class BaseEncryptor(ABC):
    """Base class for encryptors.

    Provides common functionality for encryption implementations.
    """

    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        ...

    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        ...

    def encrypt_string(self, data: str, encoding: str = "utf-8") -> bytes:
        """Encrypt a string.

        Args:
            data: String to encrypt.
            encoding: String encoding.

        Returns:
            Encrypted bytes.
        """
        return self.encrypt(data.encode(encoding))

    def decrypt_string(self, data: bytes, encoding: str = "utf-8") -> str:
        """Decrypt to a string.

        Args:
            data: Encrypted data.
            encoding: String encoding.

        Returns:
            Decrypted string.
        """
        return self.decrypt(data).decode(encoding)


class FernetEncryptor(BaseEncryptor):
    """Fernet encryption (AES-128-CBC with HMAC).

    Uses the cryptography library's Fernet implementation.
    Provides authenticated encryption.

    Example:
        >>> key = generate_fernet_key()
        >>> encryptor = FernetEncryptor(key)
        >>> encrypted = encryptor.encrypt(b"secret")
        >>> decrypted = encryptor.decrypt(encrypted)
    """

    def __init__(self, key: str | bytes) -> None:
        """Initialize with encryption key.

        Args:
            key: Fernet key (32 bytes, base64-encoded).
        """
        try:
            from cryptography.fernet import Fernet
        except ImportError as e:
            raise ImportError(
                "cryptography package required for FernetEncryptor. "
                "Install with: pip install cryptography"
            ) from e

        if isinstance(key, str):
            key = key.encode("utf-8")
        self._fernet = Fernet(key)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using Fernet.

        Args:
            data: Data to encrypt.

        Returns:
            Encrypted data.

        Raises:
            SecretEncryptError: If encryption fails.
        """
        try:
            return self._fernet.encrypt(data)
        except Exception as e:
            raise SecretEncryptError(
                f"Fernet encryption failed: {e}",
                algorithm="Fernet",
                cause=e,
            ) from e

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt Fernet-encrypted data.

        Args:
            data: Encrypted data.

        Returns:
            Decrypted data.

        Raises:
            SecretDecryptError: If decryption fails.
        """
        try:
            return self._fernet.decrypt(data)
        except Exception as e:
            raise SecretDecryptError(
                f"Fernet decryption failed: {e}",
                algorithm="Fernet",
                cause=e,
            ) from e


class AESGCMEncryptor(BaseEncryptor):
    """AES-256-GCM encryption.

    Provides authenticated encryption with associated data (AEAD).

    Example:
        >>> key = generate_aes_key()
        >>> encryptor = AESGCMEncryptor(key)
        >>> encrypted = encryptor.encrypt(b"secret")
    """

    NONCE_SIZE = 12  # 96 bits
    TAG_SIZE = 16  # 128 bits

    def __init__(self, key: bytes) -> None:
        """Initialize with encryption key.

        Args:
            key: AES key (32 bytes for AES-256).
        """
        if len(key) != 32:
            raise ValueError("AES-256 requires a 32-byte key")
        self._key = key

    def encrypt(self, data: bytes, associated_data: bytes | None = None) -> bytes:
        """Encrypt data using AES-256-GCM.

        Args:
            data: Data to encrypt.
            associated_data: Optional associated data for authentication.

        Returns:
            Encrypted data (nonce + ciphertext + tag).

        Raises:
            SecretEncryptError: If encryption fails.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as e:
            raise ImportError(
                "cryptography package required for AESGCMEncryptor. "
                "Install with: pip install cryptography"
            ) from e

        try:
            nonce = os.urandom(self.NONCE_SIZE)
            aesgcm = AESGCM(self._key)
            ciphertext = aesgcm.encrypt(nonce, data, associated_data)
            # Return nonce + ciphertext (tag is appended by cryptography)
            return nonce + ciphertext
        except Exception as e:
            raise SecretEncryptError(
                f"AES-GCM encryption failed: {e}",
                algorithm="AES-256-GCM",
                cause=e,
            ) from e

    def decrypt(self, data: bytes, associated_data: bytes | None = None) -> bytes:
        """Decrypt AES-256-GCM encrypted data.

        Args:
            data: Encrypted data (nonce + ciphertext + tag).
            associated_data: Optional associated data for authentication.

        Returns:
            Decrypted data.

        Raises:
            SecretDecryptError: If decryption fails.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as e:
            raise ImportError(
                "cryptography package required for AESGCMEncryptor. "
                "Install with: pip install cryptography"
            ) from e

        try:
            if len(data) < self.NONCE_SIZE:
                raise ValueError("Data too short")

            nonce = data[: self.NONCE_SIZE]
            ciphertext = data[self.NONCE_SIZE :]
            aesgcm = AESGCM(self._key)
            return aesgcm.decrypt(nonce, ciphertext, associated_data)
        except Exception as e:
            raise SecretDecryptError(
                f"AES-GCM decryption failed: {e}",
                algorithm="AES-256-GCM",
                cause=e,
            ) from e


class ChaCha20Poly1305Encryptor(BaseEncryptor):
    """ChaCha20-Poly1305 encryption.

    Provides authenticated encryption, often faster than AES-GCM
    on systems without hardware AES support.

    Example:
        >>> key = generate_chacha_key()
        >>> encryptor = ChaCha20Poly1305Encryptor(key)
        >>> encrypted = encryptor.encrypt(b"secret")
    """

    NONCE_SIZE = 12  # 96 bits

    def __init__(self, key: bytes) -> None:
        """Initialize with encryption key.

        Args:
            key: ChaCha20 key (32 bytes).
        """
        if len(key) != 32:
            raise ValueError("ChaCha20-Poly1305 requires a 32-byte key")
        self._key = key

    def encrypt(self, data: bytes, associated_data: bytes | None = None) -> bytes:
        """Encrypt using ChaCha20-Poly1305.

        Args:
            data: Data to encrypt.
            associated_data: Optional associated data.

        Returns:
            Encrypted data (nonce + ciphertext + tag).
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError as e:
            raise ImportError(
                "cryptography package required for ChaCha20Poly1305Encryptor. "
                "Install with: pip install cryptography"
            ) from e

        try:
            nonce = os.urandom(self.NONCE_SIZE)
            cipher = ChaCha20Poly1305(self._key)
            ciphertext = cipher.encrypt(nonce, data, associated_data)
            return nonce + ciphertext
        except Exception as e:
            raise SecretEncryptError(
                f"ChaCha20-Poly1305 encryption failed: {e}",
                algorithm="ChaCha20-Poly1305",
                cause=e,
            ) from e

    def decrypt(self, data: bytes, associated_data: bytes | None = None) -> bytes:
        """Decrypt ChaCha20-Poly1305 data.

        Args:
            data: Encrypted data.
            associated_data: Optional associated data.

        Returns:
            Decrypted data.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        except ImportError as e:
            raise ImportError(
                "cryptography package required for ChaCha20Poly1305Encryptor. "
                "Install with: pip install cryptography"
            ) from e

        try:
            if len(data) < self.NONCE_SIZE:
                raise ValueError("Data too short")

            nonce = data[: self.NONCE_SIZE]
            ciphertext = data[self.NONCE_SIZE :]
            cipher = ChaCha20Poly1305(self._key)
            return cipher.decrypt(nonce, ciphertext, associated_data)
        except Exception as e:
            raise SecretDecryptError(
                f"ChaCha20-Poly1305 decryption failed: {e}",
                algorithm="ChaCha20-Poly1305",
                cause=e,
            ) from e


class PasswordDerivedEncryptor(BaseEncryptor):
    """Encryptor that derives key from password using PBKDF2.

    Useful when a human-memorable password is preferred over
    a random key.

    Example:
        >>> encryptor = PasswordDerivedEncryptor("my-password")
        >>> encrypted = encryptor.encrypt(b"secret")
    """

    SALT_SIZE = 16
    ITERATIONS = 600_000  # OWASP recommended minimum for PBKDF2-SHA256

    def __init__(
        self,
        password: str,
        salt: bytes | None = None,
        iterations: int = ITERATIONS,
    ) -> None:
        """Initialize with password.

        Args:
            password: Password to derive key from.
            salt: Optional salt (generated if not provided).
            iterations: PBKDF2 iterations.
        """
        self._password = password
        self._salt = salt or os.urandom(self.SALT_SIZE)
        self._iterations = iterations
        self._key = self._derive_key()
        self._encryptor = AESGCMEncryptor(self._key)

    def _derive_key(self) -> bytes:
        """Derive encryption key from password.

        Returns:
            32-byte AES key.
        """
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        except ImportError as e:
            raise ImportError(
                "cryptography package required for PasswordDerivedEncryptor. "
                "Install with: pip install cryptography"
            ) from e

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._salt,
            iterations=self._iterations,
        )
        return kdf.derive(self._password.encode("utf-8"))

    @property
    def salt(self) -> bytes:
        """Get the salt used for key derivation."""
        return self._salt

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt with password-derived key.

        Returns salt + encrypted data.
        """
        encrypted = self._encryptor.encrypt(data)
        return self._salt + encrypted

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt with password-derived key.

        Expects salt + encrypted data format.
        """
        if len(data) < self.SALT_SIZE:
            raise SecretDecryptError("Data too short", algorithm="PBKDF2+AES-GCM")

        salt = data[: self.SALT_SIZE]
        if salt != self._salt:
            # Re-derive key with provided salt
            self._salt = salt
            self._key = self._derive_key()
            self._encryptor = AESGCMEncryptor(self._key)

        return self._encryptor.decrypt(data[self.SALT_SIZE :])


# =============================================================================
# Key Generation Utilities
# =============================================================================


def generate_fernet_key() -> str:
    """Generate a new Fernet key.

    Returns:
        Base64-encoded Fernet key.
    """
    try:
        from cryptography.fernet import Fernet

        return Fernet.generate_key().decode("ascii")
    except ImportError:
        # Fallback: generate URL-safe base64 key
        return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def generate_aes_key(size: int = 32) -> bytes:
    """Generate a random AES key.

    Args:
        size: Key size in bytes (16, 24, or 32).

    Returns:
        Random key bytes.
    """
    if size not in (16, 24, 32):
        raise ValueError("AES key size must be 16, 24, or 32 bytes")
    return os.urandom(size)


def generate_chacha_key() -> bytes:
    """Generate a ChaCha20 key.

    Returns:
        32-byte random key.
    """
    return os.urandom(32)


def generate_encryption_key() -> str:
    """Generate a generic encryption key.

    Returns:
        Base64-encoded 32-byte key.
    """
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")


def derive_key_from_password(
    password: str,
    salt: bytes | None = None,
    iterations: int = 600_000,
) -> tuple[bytes, bytes]:
    """Derive an encryption key from a password.

    Args:
        password: Password to derive from.
        salt: Optional salt (generated if not provided).
        iterations: PBKDF2 iterations.

    Returns:
        Tuple of (key, salt).
    """
    if salt is None:
        salt = os.urandom(16)

    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        key = kdf.derive(password.encode("utf-8"))
    except ImportError:
        # Fallback to hashlib
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
            dklen=32,
        )

    return key, salt


def create_encryptor(
    algorithm: str,
    key: bytes | str | None = None,
    password: str | None = None,
) -> SecretEncryptor:
    """Factory function to create an encryptor.

    Args:
        algorithm: Algorithm name (fernet, aes-gcm, chacha20).
        key: Encryption key.
        password: Password for key derivation.

    Returns:
        Encryptor instance.

    Raises:
        ValueError: If algorithm is unknown or key/password missing.
    """
    algorithm = algorithm.lower().replace("-", "_").replace(" ", "_")

    if password:
        return PasswordDerivedEncryptor(password)

    if key is None:
        raise ValueError("Either key or password must be provided")

    if algorithm in ("fernet",):
        return FernetEncryptor(key)
    elif algorithm in ("aes_gcm", "aes_256_gcm", "aesgcm"):
        if isinstance(key, str):
            key = base64.urlsafe_b64decode(key)
        return AESGCMEncryptor(key)
    elif algorithm in ("chacha20_poly1305", "chacha20", "chacha"):
        if isinstance(key, str):
            key = base64.urlsafe_b64decode(key)
        return ChaCha20Poly1305Encryptor(key)
    else:
        raise ValueError(f"Unknown encryption algorithm: {algorithm}")
