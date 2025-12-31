"""Truthound Orchestration Enterprise Module.

This module provides enterprise-grade features for the Truthound Orchestration framework,
including multi-channel notifications, advanced monitoring, secret management, and
enterprise integrations.

Submodules:
    - notifications: Multi-channel notification system (Slack, Email, Webhook, etc.)
    - multi_tenant: Multi-tenant isolation and context management
    - secrets: Secret management with multiple backends (Vault, AWS, GCP, Azure)

Installation:
    pip install truthound-orchestration[enterprise]

Quick Start - Notifications:
    >>> from packages.enterprise.notifications import (
    ...     SlackNotificationHandler,
    ...     WebhookNotificationHandler,
    ...     NotificationRegistry,
    ...     NotificationConfig,
    ... )
    >>>
    >>> # Create handlers
    >>> slack = SlackNotificationHandler(webhook_url="https://hooks.slack.com/...")
    >>> webhook = WebhookNotificationHandler(url="https://api.example.com/notify")
    >>>
    >>> # Register handlers
    >>> registry = NotificationRegistry()
    >>> registry.register("slack", slack)
    >>> registry.register("webhook", webhook)
    >>>
    >>> # Send notifications
    >>> await registry.notify_all("SLA violation detected!", level=NotificationLevel.CRITICAL)

Quick Start - Secrets:
    >>> from packages.enterprise.secrets import (
    ...     get_secret_registry,
    ...     get_secret,
    ...     set_secret,
    ... )
    >>> from packages.enterprise.secrets.backends import InMemorySecretProvider
    >>>
    >>> # Register a provider
    >>> registry = get_secret_registry()
    >>> registry.register("memory", InMemorySecretProvider())
    >>>
    >>> # Use convenience functions
    >>> set_secret("database/password", "secret123")
    >>> secret = get_secret("database/password")
    >>> print(secret.value)
"""

__version__ = "0.1.0"

# Lazy imports for submodules
_submodule_cache: dict = {}

def __getattr__(name: str):
    """Lazy import for submodules."""
    if name in _submodule_cache:
        return _submodule_cache[name]

    if name == "secrets":
        import importlib
        module = importlib.import_module(".secrets", __name__)
        _submodule_cache[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
