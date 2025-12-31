"""Secret rotation management.

This module provides secret rotation scheduling and execution,
including built-in secret generators and rotation strategies.

Example:
    >>> from packages.enterprise.secrets import (
    ...     SecretRotationManager,
    ...     RotationSchedule,
    ...     RotationStrategy,
    ... )
    >>>
    >>> manager = SecretRotationManager(provider)
    >>> manager.schedule(
    ...     path="api/key",
    ...     schedule=RotationSchedule(interval_days=30),
    ... )
    >>> manager.start()
"""

from __future__ import annotations

import secrets
import string
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from .base import SecretType, SecretValue
from .exceptions import RotationGeneratorError, RotationScheduleError, SecretRotationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .base import SecretProvider


class RotationStrategy(Enum):
    """Rotation trigger strategies.

    Attributes:
        TIME_BASED: Rotate on a fixed schedule.
        EVENT_BASED: Rotate on specific events.
        VERSION_BASED: Rotate after N uses.
        MANUAL: Manual rotation only.
    """

    TIME_BASED = auto()
    EVENT_BASED = auto()
    VERSION_BASED = auto()
    MANUAL = auto()


@dataclass(frozen=True, slots=True)
class RotationConfig:
    """Configuration for secret rotation.

    Attributes:
        strategy: Rotation trigger strategy.
        interval_days: Days between rotations (TIME_BASED).
        max_versions: Maximum versions to keep.
        notify_before_days: Days before expiry to notify.
        auto_cleanup: Whether to auto-delete old versions.
    """

    strategy: RotationStrategy = RotationStrategy.TIME_BASED
    interval_days: int = 30
    max_versions: int = 5
    notify_before_days: int = 7
    auto_cleanup: bool = True


@dataclass
class RotationSchedule:
    """Schedule for rotating a secret.

    Attributes:
        path: Secret path to rotate.
        interval_days: Days between rotations.
        generator: Generator name or function.
        last_rotated: Last rotation timestamp.
        next_rotation: Next scheduled rotation.
        enabled: Whether rotation is enabled.
        config: Additional rotation configuration.
    """

    path: str
    interval_days: int = 30
    generator: str = "password"
    last_rotated: datetime | None = None
    next_rotation: datetime | None = None
    enabled: bool = True
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate next rotation if not set."""
        if self.next_rotation is None:
            if self.last_rotated:
                self.next_rotation = self.last_rotated + timedelta(days=self.interval_days)
            else:
                self.next_rotation = datetime.now(timezone.utc) + timedelta(days=self.interval_days)


@dataclass
class RotationResult:
    """Result of a rotation operation.

    Attributes:
        path: Secret path that was rotated.
        success: Whether rotation succeeded.
        old_version: Previous version.
        new_version: New version.
        rotated_at: When rotation occurred.
        error: Error message if failed.
    """

    path: str
    success: bool
    old_version: str | None = None
    new_version: str | None = None
    rotated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None


@runtime_checkable
class SecretGenerator(Protocol):
    """Protocol for secret value generators."""

    def generate(self, **kwargs: Any) -> str | bytes:
        """Generate a new secret value.

        Args:
            **kwargs: Generator-specific options.

        Returns:
            Generated secret value.
        """
        ...


class PasswordGenerator:
    """Generate random passwords.

    Example:
        >>> gen = PasswordGenerator(length=32, include_special=True)
        >>> password = gen.generate()
    """

    DEFAULT_LENGTH = 32
    DEFAULT_SPECIAL = "!@#$%^&*"

    def __init__(
        self,
        length: int = DEFAULT_LENGTH,
        include_uppercase: bool = True,
        include_lowercase: bool = True,
        include_digits: bool = True,
        include_special: bool = True,
        special_chars: str = DEFAULT_SPECIAL,
    ) -> None:
        """Initialize the generator.

        Args:
            length: Password length.
            include_uppercase: Include uppercase letters.
            include_lowercase: Include lowercase letters.
            include_digits: Include digits.
            include_special: Include special characters.
            special_chars: Special characters to use.
        """
        self._length = length
        chars = ""
        if include_uppercase:
            chars += string.ascii_uppercase
        if include_lowercase:
            chars += string.ascii_lowercase
        if include_digits:
            chars += string.digits
        if include_special:
            chars += special_chars
        if not chars:
            chars = string.ascii_letters + string.digits
        self._chars = chars

    def generate(self, length: int | None = None, **kwargs: Any) -> str:
        """Generate a random password.

        Args:
            length: Optional override for length.
            **kwargs: Ignored.

        Returns:
            Random password.
        """
        length = length or self._length
        return "".join(secrets.choice(self._chars) for _ in range(length))


class UUIDGenerator:
    """Generate UUID-based secrets.

    Example:
        >>> gen = UUIDGenerator()
        >>> api_key = gen.generate()  # Returns UUID4 string
    """

    def __init__(self, version: int = 4, uppercase: bool = False) -> None:
        """Initialize the generator.

        Args:
            version: UUID version (4 or 7 if Python 3.12+).
            uppercase: Whether to uppercase the result.
        """
        self._version = version
        self._uppercase = uppercase

    def generate(self, **kwargs: Any) -> str:
        """Generate a UUID.

        Returns:
            UUID string.
        """
        result = str(uuid.uuid4())
        if self._uppercase:
            result = result.upper()
        return result


class APIKeyGenerator:
    """Generate API key style secrets.

    Format: prefix_randomstring

    Example:
        >>> gen = APIKeyGenerator(prefix="sk")
        >>> key = gen.generate()  # Returns "sk_abc123..."
    """

    def __init__(
        self,
        prefix: str = "key",
        length: int = 32,
        separator: str = "_",
    ) -> None:
        """Initialize the generator.

        Args:
            prefix: Key prefix.
            length: Random part length.
            separator: Separator between prefix and random.
        """
        self._prefix = prefix
        self._length = length
        self._separator = separator

    def generate(self, **kwargs: Any) -> str:
        """Generate an API key.

        Returns:
            API key string.
        """
        random_part = secrets.token_urlsafe(self._length)[:self._length]
        return f"{self._prefix}{self._separator}{random_part}"


class TokenGenerator:
    """Generate cryptographic tokens.

    Example:
        >>> gen = TokenGenerator(bytes_length=32)
        >>> token = gen.generate()  # Returns hex or base64 token
    """

    def __init__(
        self,
        bytes_length: int = 32,
        encoding: str = "hex",
    ) -> None:
        """Initialize the generator.

        Args:
            bytes_length: Number of random bytes.
            encoding: Output encoding (hex, base64, urlsafe).
        """
        self._bytes_length = bytes_length
        self._encoding = encoding

    def generate(self, **kwargs: Any) -> str:
        """Generate a token.

        Returns:
            Encoded token string.
        """
        data = secrets.token_bytes(self._bytes_length)
        if self._encoding == "hex":
            return data.hex()
        elif self._encoding == "base64":
            import base64

            return base64.b64encode(data).decode("ascii")
        else:  # urlsafe
            return secrets.token_urlsafe(self._bytes_length)


# Built-in generators registry
BUILTIN_GENERATORS: dict[str, SecretGenerator] = {
    "password": PasswordGenerator(),
    "uuid": UUIDGenerator(),
    "api_key": APIKeyGenerator(),
    "token": TokenGenerator(),
    "strong_password": PasswordGenerator(length=64, include_special=True),
}


class SecretRotationManager:
    """Manages secret rotation schedules and execution.

    Supports automatic background rotation and manual triggers.

    Example:
        >>> manager = SecretRotationManager(provider)
        >>> manager.register_generator("custom", my_generator)
        >>> manager.schedule(
        ...     path="db/password",
        ...     interval_days=30,
        ...     generator="password",
        ... )
        >>> manager.start()  # Start background rotation
    """

    def __init__(
        self,
        provider: SecretProvider,
        check_interval_seconds: float = 3600.0,
    ) -> None:
        """Initialize the rotation manager.

        Args:
            provider: Secret provider to manage.
            check_interval_seconds: Interval for checking schedules.
        """
        self._provider = provider
        self._check_interval = check_interval_seconds
        self._schedules: dict[str, RotationSchedule] = {}
        self._generators: dict[str, SecretGenerator] = dict(BUILTIN_GENERATORS)
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def register_generator(
        self,
        name: str,
        generator: SecretGenerator | Callable[..., str | bytes],
    ) -> None:
        """Register a custom generator.

        Args:
            name: Generator name.
            generator: Generator instance or callable.
        """
        with self._lock:
            if callable(generator) and not isinstance(generator, SecretGenerator):
                # Wrap callable in a generator

                class CallableGenerator:
                    def __init__(self, func: Callable[..., str | bytes]) -> None:
                        self._func = func

                    def generate(self, **kwargs: Any) -> str | bytes:
                        return self._func(**kwargs)

                generator = CallableGenerator(generator)
            self._generators[name] = generator

    def schedule(
        self,
        path: str,
        interval_days: int = 30,
        generator: str = "password",
        enabled: bool = True,
        config: Mapping[str, Any] | None = None,
    ) -> RotationSchedule:
        """Schedule a secret for rotation.

        Args:
            path: Secret path.
            interval_days: Days between rotations.
            generator: Generator name.
            enabled: Whether to enable the schedule.
            config: Additional configuration.

        Returns:
            The created schedule.

        Raises:
            RotationScheduleError: If generator not found.
        """
        if generator not in self._generators:
            raise RotationScheduleError(
                f"Unknown generator: {generator}",
                path=path,
            )

        schedule = RotationSchedule(
            path=path,
            interval_days=interval_days,
            generator=generator,
            enabled=enabled,
            config=config or {},
        )

        with self._lock:
            self._schedules[path] = schedule

        return schedule

    def unschedule(self, path: str) -> bool:
        """Remove a rotation schedule.

        Args:
            path: Secret path.

        Returns:
            True if removed.
        """
        with self._lock:
            return self._schedules.pop(path, None) is not None

    def get_schedule(self, path: str) -> RotationSchedule | None:
        """Get a rotation schedule.

        Args:
            path: Secret path.

        Returns:
            Schedule if found.
        """
        with self._lock:
            return self._schedules.get(path)

    def list_schedules(self) -> list[RotationSchedule]:
        """List all schedules.

        Returns:
            List of schedules.
        """
        with self._lock:
            return list(self._schedules.values())

    def rotate(self, path: str, force: bool = False) -> RotationResult:
        """Rotate a secret immediately.

        Args:
            path: Secret path to rotate.
            force: Force rotation even if not due.

        Returns:
            Rotation result.
        """
        schedule = self.get_schedule(path)

        # Get generator
        generator_name = schedule.generator if schedule else "password"
        generator = self._generators.get(generator_name)
        if generator is None:
            return RotationResult(
                path=path,
                success=False,
                error=f"Unknown generator: {generator_name}",
            )

        try:
            # Get current secret for version info
            current = self._provider.get(path)
            old_version = current.version if current else None

            # Generate new value
            gen_config = dict(schedule.config) if schedule else {}
            new_value = generator.generate(**gen_config)

            # Store new secret
            result = self._provider.set(
                path,
                new_value,
                secret_type=current.secret_type if current else SecretType.STRING,
            )

            # Update schedule
            if schedule:
                schedule.last_rotated = datetime.now(timezone.utc)
                schedule.next_rotation = schedule.last_rotated + timedelta(days=schedule.interval_days)

            return RotationResult(
                path=path,
                success=True,
                old_version=old_version,
                new_version=result.version,
            )

        except Exception as e:
            return RotationResult(
                path=path,
                success=False,
                error=str(e),
            )

    def rotate_due(self) -> list[RotationResult]:
        """Rotate all secrets that are due.

        Returns:
            List of rotation results.
        """
        results = []
        now = datetime.now(timezone.utc)

        with self._lock:
            due_schedules = [
                s
                for s in self._schedules.values()
                if s.enabled and s.next_rotation and s.next_rotation <= now
            ]

        for schedule in due_schedules:
            result = self.rotate(schedule.path)
            results.append(result)

        return results

    def start(self) -> None:
        """Start background rotation thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._rotation_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop background rotation thread.

        Args:
            timeout: Seconds to wait for thread to stop.
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _rotation_loop(self) -> None:
        """Background loop that checks for due rotations."""
        while self._running and not self._stop_event.is_set():
            try:
                self.rotate_due()
            except Exception:
                pass  # Log error but continue

            self._stop_event.wait(timeout=self._check_interval)

    @property
    def is_running(self) -> bool:
        """Check if background rotation is running."""
        return self._running

    def __enter__(self) -> SecretRotationManager:
        """Start on context entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop on context exit."""
        self.stop()
